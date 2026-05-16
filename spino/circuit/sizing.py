"""Gradient-based 5T OTA sizing via the Implicit Function Theorem (IFT).

Design vector θ = (W_diff, W_mirror, W_tail, L_common, V_bias).

Gradients flow through the KCL Newton-Raphson transient solver via IFT:

    dv*/dθ = -J_v⁻¹ @ J_θ

where J_v = ∂F/∂v at the converged state (returned by OtaTransientSolver
with ``return_final_jac=True``) and J_θ = ∂F/∂θ computed via 5-point
centred finite differences.  Each FD step rebuilds the 5-device set from
BSIM4 queries and evaluates the KCL residual at the fixed converged state.

Power is tracked but not connected to the IFT gradient (see ``extract_metrics``).
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=too-many-instance-attributes

from __future__ import annotations

import json
import logging
import subprocess
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from spino.circuit.composition_io import (
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    load_ota_5t_devices,
    _read_curated_physics,  # internal — used for BSIM re-query in FD loop
)
from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.ota_composition import (
    OtaDcSolution,
    OtaDcSolver,
    OtaTransientSolver,
)
from spino.mosfet.device_strategy import DeviceStrategy

__all__ = [
    "OtaSizingProblem",
    "compose_ota_differentiable",
    "extract_metrics",
    "loss_fn",
    "run_adam",
    "spice_validate",
]

logger = logging.getLogger(__name__)

_DEFAULT_NFET_L = 0.40
_DEFAULT_PFET_L = 0.40
_DEFAULT_VDD = 1.8
_DEFAULT_VCM = 0.9
_DEFAULT_VBIAS = 1.2
_DEFAULT_STEP_AMP = 0.05
_DEFAULT_RISE_TIME = 5e-9
_DEFAULT_T_STEP_START = 100e-9
_DEFAULT_T_END = 500e-9
_DEFAULT_T_STEP = 1e-9
_DEFAULT_C_LOAD = 1e-12
_TIKHONOV_LAMBDA = 1e-6


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------


@dataclass
class OtaSizingProblem:
    """Complete specification for the 5T OTA gradient sizing problem.

    :param nfet_checkpoint: NFET FNO checkpoint path.
    :param pfet_checkpoint: PFET FNO checkpoint path.
    :param nfet_dataset: NFET HDF5 dataset for normalization statistics.
    :param pfet_dataset: PFET HDF5 dataset for normalization statistics.
    :param w_diff_bounds: (min, max) W_diff in µm.
    :param w_mirror_bounds: (min, max) W_mirror in µm.
    :param w_tail_bounds: (min, max) W_tail in µm.
    :param l_bounds: (min, max) L_common in µm (applied to all 5 transistors).
    :param vbias_bounds: (min, max) V_bias in V.
    :param slew_rate_min_v_per_us: Slew-rate spec lower bound (V/µs).
    :param power_max_uw: Static power upper bound (µW).
    :param slew_weight: Loss weight for slew-rate constraint.
    :param power_weight: Loss weight for power constraint (no IFT gradient).
    :param vdd: Supply voltage (V).
    :param vcm: Common-mode input voltage (V).
    :param c_load_f: Output load capacitance (F).
    :param t_end: Simulation window (s).
    :param t_step: Max timestep (s).
    :param step_amp: Differential step half-amplitude (V).
    :param t_step_start: Step onset time (s).
    :param torch_device: Torch device string for FNO inference.
    :param pdk_root: Optional PDK root override.
    :param fd_eps_frac: Fractional epsilon for FD Jacobian (fraction of |θ_i|).
    """

    nfet_checkpoint: Path = field(default_factory=lambda: DEFAULT_NFET_CHECKPOINT)
    pfet_checkpoint: Path = field(default_factory=lambda: DEFAULT_PFET_CHECKPOINT)
    nfet_dataset: Path = field(default_factory=lambda: DEFAULT_NFET_DATASET)
    pfet_dataset: Path = field(default_factory=lambda: DEFAULT_PFET_DATASET)
    w_diff_bounds: tuple[float, float] = (0.5, 8.0)
    w_mirror_bounds: tuple[float, float] = (0.5, 8.0)
    w_tail_bounds: tuple[float, float] = (0.5, 4.0)
    l_bounds: tuple[float, float] = (0.18, 0.50)
    vbias_bounds: tuple[float, float] = (0.5, 1.6)
    slew_rate_min_v_per_us: float = 30.0
    power_max_uw: float = 200.0
    slew_weight: float = 10.0
    power_weight: float = 1.0
    vdd: float = _DEFAULT_VDD
    vcm: float = _DEFAULT_VCM
    c_load_f: float = _DEFAULT_C_LOAD
    t_end: float = _DEFAULT_T_END
    t_step: float = _DEFAULT_T_STEP
    step_amp: float = _DEFAULT_STEP_AMP
    t_step_start: float = _DEFAULT_T_STEP_START
    torch_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pdk_root: str | None = None
    fd_eps_frac: float = 0.01

    @property
    def lower_bounds(self) -> Tensor:
        """Return θ lower-bound vector (5,)."""
        return torch.tensor(
            [
                self.w_diff_bounds[0],
                self.w_mirror_bounds[0],
                self.w_tail_bounds[0],
                self.l_bounds[0],
                self.vbias_bounds[0],
            ],
            dtype=torch.float32,
        )

    @property
    def upper_bounds(self) -> Tensor:
        """Return θ upper-bound vector (5,)."""
        return torch.tensor(
            [
                self.w_diff_bounds[1],
                self.w_mirror_bounds[1],
                self.w_tail_bounds[1],
                self.l_bounds[1],
                self.vbias_bounds[1],
            ],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# Device construction helpers
# ---------------------------------------------------------------------------


def _build_input_trajectories(
    time_s: np.ndarray,
    *,
    vcm_v: float,
    t_step_start: float,
    rise_time_s: float,
    step_amp_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the differential step stimulus on a numpy time grid."""
    t_rise_end = t_step_start + rise_time_s
    vinp = np.where(
        time_s < t_step_start,
        vcm_v,
        np.where(
            time_s < t_rise_end,
            vcm_v + step_amp_v * (time_s - t_step_start) / rise_time_s,
            vcm_v + step_amp_v,
        ),
    )
    vinn = np.where(
        time_s < t_step_start,
        vcm_v,
        np.where(
            time_s < t_rise_end,
            vcm_v - step_amp_v * (time_s - t_step_start) / rise_time_s,
            vcm_v - step_amp_v,
        ),
    )
    return vinp.astype(np.float32), vinn.astype(np.float32)


def _device_with_physics(
    device: FnoMosfetDevice,
    physics_raw: Tensor,
) -> FnoMosfetDevice:
    """Return a new FnoMosfetDevice sharing model weights but with updated physics."""
    dev_loc = device.v_mean.device
    return FnoMosfetDevice(
        model=device.model,
        v_mean=device.v_mean,
        v_std=device.v_std,
        p_mean=device.p_mean,
        p_std=device.p_std,
        physics_raw=physics_raw.to(dev_loc),
        label=device.label,
    ).to(dev_loc)


def _rebuild_devices(
    theta_vals: tuple[float, float, float, float, float],
    base_devices: tuple[FnoMosfetDevice, ...],
    nfet_strategy: DeviceStrategy,
    pfet_strategy: DeviceStrategy,
) -> tuple[FnoMosfetDevice, ...]:
    """Rebuild all 5 OTA devices from (W_diff, W_mirror, W_tail, L, V_bias).

    Re-queries the BSIM4 model card for each perturbed (W, L) pair.  The FNO
    model weights are shared (not copied) with ``base_devices``.

    :param theta_vals: Tuple (w_diff, w_mirror, w_tail, l, vbias) as Python floats.
    :param base_devices: Baseline (M1…M5) devices carrying model and norm stats.
    :param nfet_strategy: NFET DeviceStrategy for BSIM queries.
    :param pfet_strategy: PFET DeviceStrategy for BSIM queries.
    :return: Tuple (M1, M2, M3, M4, M5) with updated physics_raw.
    """
    w_diff, w_mirror, w_tail, l_um, _ = theta_vals
    m1, m2, m3, m4, m5 = base_devices
    phys_diff = _read_curated_physics(nfet_strategy, w_diff, l_um)
    phys_mirror = _read_curated_physics(pfet_strategy, w_mirror, l_um)
    phys_tail = _read_curated_physics(nfet_strategy, w_tail, l_um)
    return (
        _device_with_physics(m1, phys_diff),
        _device_with_physics(m2, phys_diff),
        _device_with_physics(m3, phys_mirror),
        _device_with_physics(m4, phys_mirror),
        _device_with_physics(m5, phys_tail),
    )


# ---------------------------------------------------------------------------
# IFT backward helpers
# ---------------------------------------------------------------------------


def _eval_tran_residual(
    v_flat: Tensor,
    devices: tuple[FnoMosfetDevice, ...],
    v_dc: Tensor,
    tg: Tensor,
    vinp_t: Tensor,
    vinn_t: Tensor,
    vbias: float,
    vdd: float,
    c_load: float,
) -> Tensor:
    """Evaluate the transient KCL residual at ``v_flat`` with given devices.

    Creates a throw-away solver instance to access the private residual method.
    Only called in the FD loop — not performance-critical.
    """
    m1, m2, m3, m4, m5 = devices
    solver = OtaTransientSolver(m1, m2, m3, m4, m5, vdd=vdd, vbias_v=vbias, c_load_f=c_load)
    dt_vec = tg[1:] - tg[:-1]
    return solver._full_residual_flat(v_flat, vinp_t, vinn_t, v_dc, dt_vec)  # pylint: disable=protected-access


def _jtheta_fd(
    v_flat_star: Tensor,
    theta: Tensor,
    base_devices: tuple[FnoMosfetDevice, ...],
    nfet_strategy: DeviceStrategy,
    pfet_strategy: DeviceStrategy,
    tg: Tensor,
    vinp_t: Tensor,
    vinn_t: Tensor,
    problem: OtaSizingProblem,
) -> Tensor:
    """Compute the (3T, 5) FD Jacobian ∂F/∂θ at the converged state.

    For each θ_i, perturbs by ±eps, re-runs the DC solve (to update the
    initial condition), and evaluates the transient residual at v_flat_star.

    :return: J_θ of shape ``(3T, 5)`` on CPU.
    """
    n_state = v_flat_star.shape[0]
    j_theta = torch.zeros(n_state, 5, dtype=torch.float32)

    for i in range(5):
        eps_i = max(abs(float(theta[i])) * problem.fd_eps_frac, 1e-4)
        res_plus: Tensor | None = None
        res_minus: Tensor | None = None

        for sign in (+1, -1):
            t_vals = theta.detach().tolist()
            t_vals[i] += sign * eps_i
            w_diff, w_mirror, w_tail, l_um, vbias = t_vals
            # Clamp to avoid querying outside PDK validity range
            w_diff = float(np.clip(w_diff, 0.1, 20.0))
            w_mirror = float(np.clip(w_mirror, 0.1, 20.0))
            w_tail = float(np.clip(w_tail, 0.1, 20.0))
            l_um = float(np.clip(l_um, 0.18, 1.0))

            devices_pert = _rebuild_devices(
                (w_diff, w_mirror, w_tail, l_um, vbias),
                base_devices,
                nfet_strategy,
                pfet_strategy,
            )
            m1, m2, m3, m4, m5 = devices_pert
            # Re-run DC solve to update the initial condition for this theta
            dc_pert = OtaDcSolver(
                m1,
                m2,
                m3,
                m4,
                m5,
                vdd=problem.vdd,
                vcm_v=problem.vcm,
                vbias_v=vbias,
            ).solve()
            v_dc_pert = torch.tensor(
                [dc_pert.v_tail_v, dc_pert.v_left_v, dc_pert.v_out_v],
                dtype=torch.float32,
                device=v_flat_star.device,
            )
            with torch.no_grad():
                res = (
                    _eval_tran_residual(
                        v_flat_star.clone().requires_grad_(True),
                        devices_pert,
                        v_dc_pert,
                        tg,
                        vinp_t,
                        vinn_t,
                        vbias,
                        problem.vdd,
                        problem.c_load_f,
                    )
                    .detach()
                    .cpu()
                )
            if sign == +1:
                res_plus = res
            else:
                res_minus = res

        assert res_plus is not None and res_minus is not None
        j_theta[:, i] = (res_plus - res_minus) / (2.0 * eps_i)

    return j_theta


def _make_ift_function(
    base_devices: tuple[FnoMosfetDevice, ...],
    nfet_strategy: DeviceStrategy,
    pfet_strategy: DeviceStrategy,
    problem: OtaSizingProblem,
    tg: Tensor,
    vinp_t: Tensor,
    vinn_t: Tensor,
) -> type:
    """Factory that returns a ``torch.autograd.Function`` class closed over solver state.

    The returned class implements the IFT backward pass for the OTA transient solve.
    """

    class _OtaTransientIFT(torch.autograd.Function):  # pylint: disable=abstract-method
        """IFT-based custom autograd for the OTA transient Newton solve."""

        @staticmethod
        def forward(  # type: ignore[override]  # pylint: disable=arguments-differ
            ctx: torch.autograd.function.FunctionCtx,
            theta: Tensor,
            v_dc: Tensor,
        ) -> Tensor:
            """Run Newton to convergence; save J_v and v_flat_star for backward."""
            theta_vals = tuple(float(x) for x in theta.detach().tolist())
            w_diff, w_mirror, w_tail, l_um, vbias = theta_vals
            devices = _rebuild_devices(
                (w_diff, w_mirror, w_tail, l_um, vbias),
                base_devices,
                nfet_strategy,
                pfet_strategy,
            )
            m1, m2, m3, m4, m5 = devices
            solver = OtaTransientSolver(
                m1,
                m2,
                m3,
                m4,
                m5,
                vdd=problem.vdd,
                vbias_v=vbias,
                c_load_f=problem.c_load_f,
            )
            result, j_v, v_flat_star = solver.solve(tg, vinp_t, vinn_t, v_dc, return_final_jac=True)
            # Store non-tensor state as ctx attributes
            ctx.problem = problem
            ctx.base_devices = base_devices
            ctx.nfet_strategy = nfet_strategy
            ctx.pfet_strategy = pfet_strategy
            ctx.tg = tg
            ctx.vinp_t = vinp_t
            ctx.vinn_t = vinn_t
            ctx.nr_converged = result.report.converged
            ctx.save_for_backward(theta.detach(), j_v, v_flat_star)
            return result.v_out_v.detach()

        @staticmethod
        def backward(  # type: ignore[override]  # pylint: disable=arguments-differ
            ctx: torch.autograd.function.FunctionCtx,
            grad_v_out: Tensor,
        ) -> tuple[Tensor | None, Tensor | None]:
            """IFT: dv*/dθ = -J_v⁻¹ @ J_θ, then chain-rule to grad_theta."""
            theta, j_v, v_flat_star = ctx.saved_tensors  # type: ignore[misc]
            t_len = ctx.tg.shape[0]

            # Compute FD Jacobian ∂F/∂θ at converged state
            j_theta = _jtheta_fd(
                v_flat_star,
                theta,
                ctx.base_devices,
                ctx.nfet_strategy,
                ctx.pfet_strategy,
                ctx.tg,
                ctx.vinp_t,
                ctx.vinn_t,
                ctx.problem,
            ).to(j_v.device)

            # IFT solve: (3T, 3T) \ (3T, 5) with Tikhonov regularisation
            eye = torch.eye(j_v.shape[0], dtype=j_v.dtype, device=j_v.device)
            j_v_reg = j_v + _TIKHONOV_LAMBDA * eye
            try:
                dv_dtheta = -torch.linalg.solve(j_v_reg, j_theta)  # pylint: disable=not-callable  # (3T, 5)
            except torch.linalg.LinAlgError:
                logger.warning("IFT solve failed (singular J_v); returning zero gradient.")
                return torch.zeros(5, device=theta.device), None

            # v_out is rows [2T : 3T] in the flattened state
            dv_out_dtheta = dv_dtheta[2 * t_len : 3 * t_len, :]  # (T, 5)
            grad_out = grad_v_out.to(dv_out_dtheta.device)
            grad_theta = (grad_out.unsqueeze(-1) * dv_out_dtheta).sum(0)  # (5,)
            return grad_theta.to(theta.device), None  # None for v_dc

    return _OtaTransientIFT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compose_ota_differentiable(
    theta: Tensor,
    problem: OtaSizingProblem,
    base_devices: tuple[FnoMosfetDevice, ...],
    nfet_strategy: DeviceStrategy,
    pfet_strategy: DeviceStrategy,
    tg: Tensor,
    vinp_t: Tensor,
    vinn_t: Tensor,
) -> tuple[Tensor, OtaDcSolution]:
    """Return V_out(t) with IFT gradient connected to θ.

    :param theta: Design vector ``(5,)`` with ``requires_grad=True``.
        Layout: ``[W_diff, W_mirror, W_tail, L_common, V_bias]`` (µm, µm, µm, µm, V).
    :param problem: Sizing problem configuration.
    :param base_devices: Pre-loaded (M1…M5) baseline devices (model weights shared).
    :param nfet_strategy: NFET DeviceStrategy for BSIM re-queries.
    :param pfet_strategy: PFET DeviceStrategy for BSIM re-queries.
    :param tg: Time grid ``(T,)`` in seconds.
    :param vinp_t: Vinp trajectory ``(T,)`` (V).
    :param vinn_t: Vinn trajectory ``(T,)`` (V).
    :return: Tuple ``(v_out, dc_sol)`` — V_out trajectory tensor with IFT backward
        wired to θ, and the DC operating-point solution.
    """
    theta_vals = tuple(float(x) for x in theta.detach().tolist())
    w_diff, w_mirror, w_tail, l_um, vbias = theta_vals
    devices = _rebuild_devices(
        (w_diff, w_mirror, w_tail, l_um, vbias),
        base_devices,
        nfet_strategy,
        pfet_strategy,
    )
    m1, m2, m3, m4, m5 = devices
    dc_sol = OtaDcSolver(
        m1,
        m2,
        m3,
        m4,
        m5,
        vdd=problem.vdd,
        vcm_v=problem.vcm,
        vbias_v=vbias,
    ).solve()
    v_dc = torch.tensor(
        [dc_sol.v_tail_v, dc_sol.v_left_v, dc_sol.v_out_v],
        dtype=torch.float32,
    )
    ift_cls = _make_ift_function(base_devices, nfet_strategy, pfet_strategy, problem, tg, vinp_t, vinn_t)
    v_out = ift_cls.apply(theta, v_dc)
    return v_out, dc_sol


def extract_metrics(
    v_out: Tensor,
    tg: Tensor,
    dc_sol: OtaDcSolution,
    problem: OtaSizingProblem,
) -> dict[str, Tensor | float]:
    """Compute OTA performance metrics from the transient trajectory.

    ``slew_rate`` and ``swing`` are differentiable through ``v_out`` (IFT
    gradient available).  ``power_uw`` is evaluated at the DC operating point
    and is NOT connected to the IFT gradient — it is monitored but not
    gradient-optimised.

    :param v_out: ``(T,)`` output voltage trajectory (differentiable via IFT).
    :param tg: ``(T,)`` time grid in seconds.
    :param dc_sol: DC operating-point solution (detached floats).
    :param problem: Sizing configuration.
    :return: Dict with keys ``slew_rate_v_per_us``, ``swing_v``, ``power_uw``.
    """
    dt = float((tg[1] - tg[0]).item()) if tg.shape[0] > 1 else problem.t_step
    post_mask = tg >= problem.t_step_start
    v_post = v_out[post_mask]

    # Differentiable slew rate: peak |dV/dt| after step onset
    if v_out.shape[0] > 1:
        dv_dt = torch.diff(v_out) / dt  # (T-1,)
        slew_rate = torch.max(torch.abs(dv_dt)) * 1e-6  # V/µs
    else:
        slew_rate = torch.tensor(0.0)

    # Differentiable output swing: max - min in post-step window
    if v_post.numel() > 1:
        swing = torch.max(v_post) - torch.min(v_post)
    else:
        swing = torch.tensor(0.0)

    # Non-differentiable static power: I_tail from KCL at DC op (float)
    i_tail_a = abs(dc_sol.v_tail_v / max(abs(dc_sol.v_tail_v), 1e-12)) * 1e-4
    power_uw = float(i_tail_a * problem.vdd * 1e6)  # rough estimate; replaced by SPICE

    return {
        "slew_rate_v_per_us": slew_rate,
        "swing_v": swing,
        "power_uw": power_uw,
    }


def loss_fn(
    metrics: dict[str, Tensor | float],
    problem: OtaSizingProblem,
) -> Tensor:
    """Weighted relu-barrier loss over spec constraints.

    Only ``slew_rate`` and ``swing`` contribute gradients (IFT-differentiable).
    ``power`` is included as a constant penalty (no gradient to θ).

    :param metrics: Output of :func:`extract_metrics`.
    :param problem: Sizing configuration with weights and spec targets.
    :return: Scalar loss tensor.
    """
    slew = metrics["slew_rate_v_per_us"]
    assert isinstance(slew, Tensor)

    # Slew rate: penalise if below target
    loss_slew = problem.slew_weight * F.relu(torch.tensor(problem.slew_rate_min_v_per_us) - slew)

    # Power: constant penalty (no gradient through IFT)
    power_val = float(metrics["power_uw"])
    loss_power = problem.power_weight * max(0.0, power_val - problem.power_max_uw)

    return loss_slew + loss_power


# ---------------------------------------------------------------------------
# Adam optimisation loop
# ---------------------------------------------------------------------------


def _load_base_state(
    problem: OtaSizingProblem,
    theta_init: Tensor,
) -> tuple[
    tuple[FnoMosfetDevice, ...],
    DeviceStrategy,
    DeviceStrategy,
    Tensor,
    Tensor,
    Tensor,
]:
    """Load model weights, norm stats, and build time grid.  Called once."""
    logger.info("Loading base OTA devices…")
    w_diff, w_mirror, w_tail, l_um, _ = theta_init.tolist()
    base_devices = load_ota_5t_devices(
        diff_w_um=w_diff,
        diff_l_um=l_um,
        mirror_w_um=w_mirror,
        mirror_l_um=l_um,
        tail_w_um=w_tail,
        tail_l_um=l_um,
        nfet_checkpoint=problem.nfet_checkpoint,
        pfet_checkpoint=problem.pfet_checkpoint,
        nfet_dataset=problem.nfet_dataset,
        pfet_dataset=problem.pfet_dataset,
        map_location=problem.torch_device,
    )
    nfet_strategy = DeviceStrategy.create("sky130_nmos", pdk_root=problem.pdk_root)
    pfet_strategy = DeviceStrategy.create("sky130_pmos", pdk_root=problem.pdk_root)

    time_np = np.arange(0.0, problem.t_end, problem.t_step, dtype=np.float32)
    vinp_np, vinn_np = _build_input_trajectories(
        time_np,
        vcm_v=problem.vcm,
        t_step_start=problem.t_step_start,
        rise_time_s=_DEFAULT_RISE_TIME,
        step_amp_v=problem.step_amp,
    )
    tg = torch.tensor(time_np, dtype=torch.float32, device=problem.torch_device)
    vinp_t = torch.tensor(vinp_np, dtype=torch.float32, device=problem.torch_device)
    vinn_t = torch.tensor(vinn_np, dtype=torch.float32, device=problem.torch_device)

    return base_devices, nfet_strategy, pfet_strategy, tg, vinp_t, vinn_t


def run_adam(
    problem: OtaSizingProblem,
    *,
    theta_init: Tensor,
    n_iters: int = 200,
    lr: float = 1e-3,
    output_dir: Path,
) -> Tensor:
    """Run Adam gradient-based OTA sizing with IFT gradients.

    :param problem: Sizing problem specification.
    :param theta_init: Initial design vector ``(5,)``  [W_diff, W_mirror, W_tail, L, V_bias].
    :param n_iters: Maximum Adam iterations.
    :param lr: Adam learning rate.
    :param output_dir: Directory for trajectory JSON and final-θ log.
    :return: Final θ tensor.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t = _load_base_state(problem, theta_init)

    theta = theta_init.clone().float().requires_grad_(True)
    lower = problem.lower_bounds.to(theta.device)
    upper = problem.upper_bounds.to(theta.device)

    optimizer = torch.optim.Adam([theta], lr=lr)
    trajectory: list[dict] = []

    for step in range(n_iters):
        optimizer.zero_grad()
        t0 = time_module.perf_counter()

        try:
            v_out, dc_sol = compose_ota_differentiable(
                theta,
                problem,
                base_devices,
                nfet_strat,
                pfet_strat,
                tg,
                vinp_t,
                vinn_t,
            )
        except RuntimeError as exc:
            logger.warning("Step %d: Newton diverged (%s); halving lr and skipping.", step, exc)
            for pg in optimizer.param_groups:
                pg["lr"] /= 2.0
            continue

        metrics = extract_metrics(v_out, tg, dc_sol, problem)
        loss = loss_fn(metrics, problem)

        if not loss.isfinite():
            logger.warning("Step %d: non-finite loss; skipping backward.", step)
            continue

        loss.backward()
        optimizer.step()

        # Project theta back into bounds
        with torch.no_grad():
            theta.clamp_(lower, upper)

        elapsed = time_module.perf_counter() - t0
        row = {
            "step": step,
            "loss": float(loss),
            "slew_rate_v_per_us": float(metrics["slew_rate_v_per_us"]),
            "power_uw": float(metrics["power_uw"]),
            "theta": theta.detach().tolist(),
            "wall_s": round(elapsed, 3),
        }
        trajectory.append(row)
        logger.info(
            "Step %3d | loss=%.4f | slew=%.1f V/µs | power=%.0f µW | θ=%s",
            step,
            row["loss"],
            row["slew_rate_v_per_us"],
            row["power_uw"],
            [f"{v:.3f}" for v in row["theta"]],
        )

    (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    final_theta = theta.detach()
    (output_dir / "theta_final.json").write_text(
        json.dumps(
            {"theta": final_theta.tolist(), "layout": ["W_diff_um", "W_mirror_um", "W_tail_um", "L_um", "V_bias_v"]}
        ),
        encoding="utf-8",
    )
    logger.info("Adam complete. Final θ: %s", final_theta.tolist())
    return final_theta


# ---------------------------------------------------------------------------
# SPICE validation
# ---------------------------------------------------------------------------


def spice_validate(
    theta: Tensor,
    problem: OtaSizingProblem,
    output_dir: Path,
) -> dict:
    """Run NGSpice characterise_ota at the final θ for ground-truth validation.

    Calls the ``spino.circuit.characterize_ota`` CLI via subprocess so that
    Click's isolation is preserved.

    :param theta: Final design vector ``(5,)`` [W_diff, W_mirror, W_tail, L, V_bias].
    :param problem: Sizing problem configuration.
    :param output_dir: Directory for SPICE validation artefacts.
    :return: Parsed ``summary.json`` dict.
    """
    vals = theta.tolist()
    w_diff, w_mirror, w_tail, l_um, vbias = vals
    out = output_dir / "spice_validation"
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "spino.circuit.characterize_ota",
        "--output-dir",
        str(out),
        "--diff-w",
        str(w_diff),
        "--mirror-w",
        str(w_mirror),
        "--tail-w",
        str(w_tail),
        "--nfet-l",
        str(l_um),
        "--pfet-l",
        str(l_um),
        "--tail-l",
        str(l_um),
        "--vbias",
        str(vbias),
    ]
    if problem.pdk_root:
        cmd += ["--pdk-root", problem.pdk_root]

    logger.info("Running SPICE validation: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("characterize_ota failed:\n%s", result.stderr)
        return {"error": result.stderr}

    summary_path = out / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--mode", type=click.Choice(["adam-fno"]), default="adam-fno", show_default=True)
@click.option(
    "--theta-init",
    type=str,
    default="3.0,3.0,1.0,0.40,0.9",
    show_default=True,
    help="Comma-separated initial θ: W_diff,W_mirror,W_tail,L_um,V_bias",
)
@click.option("--n-iters", type=int, default=200, show_default=True, help="Adam iterations.")
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Adam learning rate.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/sizing/adam_ota_5var"),
    show_default=True,
)
@click.option("--slew-min", type=float, default=30.0, show_default=True, help="Slew-rate spec (V/µs).")
@click.option("--power-max", type=float, default=200.0, show_default=True, help="Power spec (µW).")
@click.option("--device", type=str, default=None, help="Torch device (default: cuda if available).")
@click.option(
    "--nfet-checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Override NFET checkpoint path.",
)
@click.option(
    "--pfet-checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Override PFET checkpoint path.",
)
@click.option("--validate-spice", is_flag=True, default=False, help="Run SPICE validation at final θ.")
@click.option("--pdk-root", type=str, default=None, help="Override PDK root.")
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments,unused-argument
    mode: str,
    theta_init: str,
    n_iters: int,
    lr: float,
    output_dir: Path,
    slew_min: float,
    power_max: float,
    device: str | None,
    nfet_checkpoint: Path | None,
    pfet_checkpoint: Path | None,
    validate_spice: bool,
    pdk_root: str | None,
) -> None:
    """Gradient-based 5T OTA sizing via IFT through the KCL Newton solver."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    theta_vals = [float(v) for v in theta_init.split(",")]
    if len(theta_vals) != 5:
        raise click.BadParameter("theta-init must have exactly 5 comma-separated values.")

    torch_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    problem = OtaSizingProblem(
        nfet_checkpoint=nfet_checkpoint or DEFAULT_NFET_CHECKPOINT,
        pfet_checkpoint=pfet_checkpoint or DEFAULT_PFET_CHECKPOINT,
        slew_rate_min_v_per_us=slew_min,
        power_max_uw=power_max,
        torch_device=torch_dev,
        pdk_root=pdk_root,
    )

    theta_t = torch.tensor(theta_vals, dtype=torch.float32)
    final_theta = run_adam(
        problem,
        theta_init=theta_t,
        n_iters=n_iters,
        lr=lr,
        output_dir=output_dir,
    )

    if validate_spice:
        summary = spice_validate(final_theta, problem, output_dir)
        logger.info("SPICE validation summary: %s", summary)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
