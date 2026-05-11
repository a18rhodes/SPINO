"""
Newton–Raphson solvers for FNO-composed 5T OTA analyses.

Two solvers cover the three internal KCL unknowns ``(n_tail, n_left, n_out)``:

* :class:`OtaDcSolver` — 3-vector Newton–Raphson at fixed common-mode inputs
  ``(Vcm, Vbias)``. Each device is probed at constant voltage for ``t_probe``
  timesteps and reduced to a scalar drain current via the post-trim mean.

* :class:`OtaTransientSolver` — whole-window implicit Newton–Raphson on the
  unknown trajectories ``V_tail(t), V_left(t), V_out(t) ∈ R^T``. State is
  flattened to ``R^{3T}``; residuals stack one initial-condition row per node
  followed by KCL rows at ``t = 1 … T-1``. A lumped ``c_load_f`` at
  ``n_out`` is the only capacitive term (``C_tail = C_left = 0``).

KCL convention (positive = current flows into the node):

    R_tail = I_M1 + I_M2 − I_M5
    R_left = I_M3 − I_M1
    R_out  = I_M4 − I_M2 − C_load · dV_out/dt

Current signs follow the FNO training convention: positive output always
indicates the device is conducting.  PFET M3/M4 source current *into* their
drain nodes (n_left, n_out); NFET M1/M2 sink current *from* their drain nodes
(n_left, n_out) and source an equal current *into* n_tail; NFET M5 sinks
current *from* n_tail.
"""

# Solver classes carry many NR tuning knobs by design.
# pylint: disable=too-many-arguments,too-many-instance-attributes,too-few-public-methods
# pylint: disable=too-many-locals,too-many-positional-arguments

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres
from torch.autograd.functional import jacobian
from torch.autograd.functional import jvp as torch_jvp

from spino.circuit.composition import ConvergenceReport, _backtrack, _cap_alpha, _inf_norm
from spino.circuit.devices import FnoMosfetDevice
from spino.mosfet.evaluate import DEFAULT_TRIM_EVAL

__all__ = [
    "OtaDcSolution",
    "OtaDcSolver",
    "OtaTransientSolution",
    "OtaTransientSolver",
    "build_ota_diffpair_probe",
    "build_ota_mirror_probe",
    "build_ota_tail_probe",
]

logger = logging.getLogger(__name__)

_DEFAULT_VDD = 1.8
_DEFAULT_VBIAS = 1.2
_DEFAULT_VCM = 0.9
_DEFAULT_T_PROBE = 256
_DEFAULT_DC_TOL_A = 1e-9
_DEFAULT_TRAN_TOL_A = 1e-7
_DEFAULT_DC_MAX_ITER = 50
_DEFAULT_TRAN_MAX_ITER = 25
_DEFAULT_V_STEP_CAP = 0.2
_DEFAULT_ARMIJO_C = 1e-4
_DEFAULT_ALPHA_MIN = 1e-3
# Empirical initial guess from Phase 0 conditioning probe (2026-05-09).
_DEFAULT_DC_INIT: tuple[float, float, float] = (0.1, 0.55, 0.55)


# ---------------------------------------------------------------------------
# Public probe builders
# ---------------------------------------------------------------------------


def build_ota_diffpair_probe(vg_t: Tensor, vd_t: Tensor, vs_tail_t: Tensor) -> Tensor:
    """
    Assembles the ``(1, 4, T)`` input for M1 or M2 (NFET differential-pair).

    Bulk is tied to GND (zero). Channel order is ``(Vg, Vd, Vs, Vb)``
    matching :meth:`~spino.circuit.devices.FnoMosfetDevice.drain_current`.

    :param vg_t: Gate trajectory ``(T,)`` — ``Vinp`` for M1, ``Vinn`` for M2.
    :param vd_t: Drain trajectory ``(T,)`` — ``V_left`` for M1, ``V_out`` for M2.
    :param vs_tail_t: Source trajectory ``(T,)`` — the ``n_tail`` node.
    :return: Probe tensor ``(1, 4, T)``.
    """
    t = vg_t.shape[0]
    zeros = torch.zeros(1, 1, t, dtype=vg_t.dtype, device=vg_t.device)
    return torch.cat(
        [vg_t.reshape(1, 1, t), vd_t.reshape(1, 1, t), vs_tail_t.reshape(1, 1, t), zeros],
        dim=1,
    )


def build_ota_mirror_probe(vg_t: Tensor, vd_t: Tensor, vdd_value: float) -> Tensor:
    """
    Assembles the ``(1, 4, T)`` input for M3 or M4 (PFET current-mirror).

    Source and bulk are tied to ``VDD``. For M3 (diode-connected) pass
    ``vg_t = vd_t = V_left``; for M4 pass ``vg_t = V_left``, ``vd_t = V_out``.

    :param vg_t: Gate trajectory ``(T,)`` — ``V_left`` for both M3 and M4.
    :param vd_t: Drain trajectory ``(T,)`` — ``V_left`` for M3, ``V_out`` for M4.
    :param vdd_value: Supply voltage in volts.
    :return: Probe tensor ``(1, 4, T)``.
    """
    t = vg_t.shape[0]
    vdd_t = torch.full((1, 1, t), vdd_value, dtype=vg_t.dtype, device=vg_t.device)
    return torch.cat([vg_t.reshape(1, 1, t), vd_t.reshape(1, 1, t), vdd_t, vdd_t], dim=1)


def build_ota_tail_probe(vbias_value: float, vd_tail_t: Tensor) -> Tensor:
    """
    Assembles the ``(1, 4, T)`` input for M5 (NFET tail current source).

    Gate is the constant external bias ``Vbias``; source and bulk are GND.

    :param vbias_value: Constant gate bias voltage in volts.
    :param vd_tail_t: Drain trajectory ``(T,)`` — the ``n_tail`` node.
    :return: Probe tensor ``(1, 4, T)``.
    """
    t = vd_tail_t.shape[0]
    zeros = torch.zeros(1, 1, t, dtype=vd_tail_t.dtype, device=vd_tail_t.device)
    vbias_t = torch.full((1, 1, t), vbias_value, dtype=vd_tail_t.dtype, device=vd_tail_t.device)
    return torch.cat([vbias_t, vd_tail_t.reshape(1, 1, t), zeros, zeros], dim=1)


# ---------------------------------------------------------------------------
# Solution dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OtaDcSolution:
    """
    DC operating-point solution for the 5T OTA.

    :param v_tail_v: Solved ``n_tail`` voltage (V).
    :param v_left_v: Solved ``n_left`` voltage (V).
    :param v_out_v: Solved ``n_out`` voltage (V).
    :param report: Newton–Raphson convergence diagnostics.
    """

    v_tail_v: float
    v_left_v: float
    v_out_v: float
    report: ConvergenceReport


@dataclass(frozen=True, slots=True)
class OtaTransientSolution:
    """
    Transient trajectories for the three OTA internal nodes.

    :param time_s: ``(T,)`` time grid in seconds.
    :param v_tail_v: ``(T,)`` n_tail trajectory in volts.
    :param v_left_v: ``(T,)`` n_left trajectory in volts.
    :param v_out_v: ``(T,)`` n_out trajectory in volts.
    :param report: Newton–Raphson convergence diagnostics.
    """

    time_s: Tensor
    v_tail_v: Tensor
    v_left_v: Tensor
    v_out_v: Tensor
    report: ConvergenceReport


# ---------------------------------------------------------------------------
# DC solver
# ---------------------------------------------------------------------------


class OtaDcSolver:
    """
    Three-node vector Newton–Raphson for the 5T OTA DC operating point.

    Solves ``(R_tail, R_left, R_out) = 0`` with both differential inputs
    held at the common-mode voltage ``vcm_v``. Each device is probed at
    constant voltage over ``t_probe`` timesteps; the post-trim mean yields
    the scalar drain current entering the residual.

    Armijo backtracking and a per-component voltage cap keep the Newton steps
    within a safe region. The default initial guess ``[0.1, 0.55, 0.55]`` V
    was determined experimentally during the Phase 0 conditioning probe.
    """

    def __init__(
        self,
        m1: FnoMosfetDevice,
        m2: FnoMosfetDevice,
        m3: FnoMosfetDevice,
        m4: FnoMosfetDevice,
        m5: FnoMosfetDevice,
        *,
        vdd: float = _DEFAULT_VDD,
        vcm_v: float = _DEFAULT_VCM,
        vbias_v: float = _DEFAULT_VBIAS,
        t_probe: int = _DEFAULT_T_PROBE,
        trim: int = DEFAULT_TRIM_EVAL,
        max_iter: int = _DEFAULT_DC_MAX_ITER,
        residual_tol: float = _DEFAULT_DC_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
    ) -> None:
        if t_probe <= trim:
            raise ValueError(f"t_probe ({t_probe}) must exceed trim ({trim})")
        self.m1, self.m2, self.m3, self.m4, self.m5 = m1, m2, m3, m4, m5
        self.vdd = vdd
        self.vcm_v = vcm_v
        self.vbias_v = vbias_v
        self.t_probe = t_probe
        self.trim = trim
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self._device = m1.v_mean.device
        self._dtype = m1.v_mean.dtype

    def _dc_traj(self, x: float | Tensor) -> Tensor:
        """Expand a scalar (float or 0-D tensor) to ``(1, 1, t_probe)``."""
        if isinstance(x, float):
            return torch.full((1, 1, self.t_probe), x, dtype=self._dtype, device=self._device)
        return x.expand(self.t_probe).reshape(1, 1, self.t_probe)

    def _dc_probe(
        self,
        vg: float | Tensor,
        vd: float | Tensor,
        vs: float | Tensor,
        vb: float | Tensor,
    ) -> Tensor:
        """Stack four constant trajectories into a ``(1, 4, t_probe)`` probe."""
        return torch.cat(
            [self._dc_traj(vg), self._dc_traj(vd), self._dc_traj(vs), self._dc_traj(vb)],
            dim=1,
        )

    def _mean_id(self, mos: FnoMosfetDevice, probe: Tensor) -> Tensor:
        return mos.drain_current(probe)[0, 0, self.trim :].mean()

    def _residual(self, v: Tensor) -> Tensor:
        """KCL residual ``(3,)`` at node voltages ``v = [v_tail, v_left, v_out]``."""
        vt, vl, vo = v[0], v[1], v[2]
        i_m1 = self._mean_id(self.m1, self._dc_probe(self.vcm_v, vl, vt, 0.0))
        i_m2 = self._mean_id(self.m2, self._dc_probe(self.vcm_v, vo, vt, 0.0))
        i_m3 = self._mean_id(self.m3, self._dc_probe(vl, vl, self.vdd, self.vdd))
        i_m4 = self._mean_id(self.m4, self._dc_probe(vl, vo, self.vdd, self.vdd))
        i_m5 = self._mean_id(self.m5, self._dc_probe(self.vbias_v, vt, 0.0, 0.0))
        r_tail = i_m1 + i_m2 - i_m5
        r_left = i_m3 - i_m1
        r_out = i_m4 - i_m2
        return torch.stack([r_tail, r_left, r_out])

    def solve(self, v_init: Tensor | None = None) -> OtaDcSolution:
        """
        Runs the Newton–Raphson loop and returns the operating-point solution.

        :param v_init: Optional ``(3,)`` initial guess ``[v_tail, v_left, v_out]``
            in volts. Defaults to ``[0.1, 0.55, 0.55]`` V.
        :return: :class:`OtaDcSolution` with converged (or best-effort) voltages.
        """
        start = time_module.perf_counter()
        if v_init is None:
            v = torch.tensor(list(_DEFAULT_DC_INIT), dtype=self._dtype, device=self._device)
        else:
            v = v_init.to(device=self._device, dtype=self._dtype).clone()
        v = v.requires_grad_(True)

        with torch.no_grad():
            rn = _inf_norm(self._residual(v.detach()))
        history: list[float] = [rn]
        converged = rn <= self.residual_tol
        iters = 0

        while not converged and iters < self.max_iter:
            iters += 1
            jac = jacobian(self._residual, v, vectorize=True, create_graph=False)
            res = self._residual(v)
            direction = torch.linalg.solve(jac, -res.detach())

            alpha, _ = _backtrack(
                self._residual,
                v.detach(),
                direction,
                rn,
                norm_fn=_inf_norm,
                armijo_c=self.armijo_c,
                alpha_min=self.alpha_min,
            )
            alpha = _cap_alpha(alpha, direction, self.v_step_cap)
            v = (v.detach() + alpha * direction).clamp(0.0, self.vdd).requires_grad_(True)
            with torch.no_grad():
                rn = _inf_norm(self._residual(v.detach()))
            history.append(rn)
            converged = rn <= self.residual_tol

        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        v_sol = v.detach()
        logger.debug(
            "OtaDcSolver: %s in %d iters (rn=%.2e, wall=%.1f ms)",
            "converged" if converged else "did not converge",
            iters,
            rn,
            wall_ms,
        )
        return OtaDcSolution(
            v_tail_v=float(v_sol[0].item()),
            v_left_v=float(v_sol[1].item()),
            v_out_v=float(v_sol[2].item()),
            report=ConvergenceReport(
                converged=converged,
                iter_count=iters,
                residual_norm_history=tuple(history),
                final_residual_norm=rn,
                wall_ms=wall_ms,
            ),
        )


# ---------------------------------------------------------------------------
# Transient solver
# ---------------------------------------------------------------------------


class OtaTransientSolver:
    """
    Whole-window implicit Newton–Raphson on ``(V_tail, V_left, V_out)`` trajectories.

    State is flattened to ``(3T,)`` with layout
    ``[V_tail(0:T), V_left(T:2T), V_out(2T:3T)]``. The residual stacks an
    initial-condition row per node followed by KCL rows at ``t = 1 … T-1``.
    Two Newton-direction strategies are available via ``use_gmres``:

    * ``False`` (default): dense ``(3T, 3T)`` Jacobian via
      ``torch.autograd.functional.jacobian(vectorize=True)`` — uses vmap to run
      all 3T VJPs as one batched GPU operation, then ``torch.linalg.solve``.
      Fastest on GPU (~65 s for T=500 at CUDA).

    * ``True``: GMRES with scipy matvec via ``torch.autograd.functional.jvp``.
      Each GMRES iteration is one sequential JVP call + numpy↔torch round-trip.
      Faster on CPU (avoids 3T sequential backward passes) but ~7× slower on GPU
      because it cannot exploit vmap batching and pays scipy call overhead.

    Only ``n_out`` carries a lumped load capacitance ``c_load_f``.
    ``C_tail`` and ``C_left`` are zero in this implementation.
    """

    def __init__(
        self,
        m1: FnoMosfetDevice,
        m2: FnoMosfetDevice,
        m3: FnoMosfetDevice,
        m4: FnoMosfetDevice,
        m5: FnoMosfetDevice,
        *,
        vdd: float = _DEFAULT_VDD,
        vbias_v: float = _DEFAULT_VBIAS,
        c_load_f: float = 0.0,
        max_iter: int = _DEFAULT_TRAN_MAX_ITER,
        residual_tol: float = _DEFAULT_TRAN_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
        use_gmres: bool = False,
        gmres_max_iter: int = 50,
    ) -> None:
        self.m1, self.m2, self.m3, self.m4, self.m5 = m1, m2, m3, m4, m5
        self.vdd = vdd
        self.vbias_v = vbias_v
        self.c_load_f = c_load_f
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self.use_gmres = use_gmres
        self.gmres_max_iter = gmres_max_iter
        self._device = m1.v_mean.device
        self._dtype = m1.v_mean.dtype

    def _full_residual_flat(
        self,
        v_flat: Tensor,
        vinp_t: Tensor,
        vinn_t: Tensor,
        v_dc: Tensor,
        dt_vec: Tensor,
    ) -> Tensor:
        """
        Assembles the ``(3T,)`` residual for state ``v_flat = (3T,)``.

        Layout: ``[r_tail(0:T), r_left(T:2T), r_out(2T:3T)]`` where each
        block opens with an initial-condition row and follows with ``T-1``
        KCL rows.
        """
        t = vinp_t.shape[0]
        v_tail = v_flat[0:t]
        v_left = v_flat[t : 2 * t]
        v_out = v_flat[2 * t : 3 * t]

        zeros = torch.zeros(1, 1, t, dtype=v_flat.dtype, device=v_flat.device)
        vdd_t = torch.full((1, 1, t), self.vdd, dtype=v_flat.dtype, device=v_flat.device)
        vbias_t = torch.full((1, 1, t), self.vbias_v, dtype=v_flat.dtype, device=v_flat.device)

        probe_m1 = torch.cat([vinp_t.reshape(1, 1, t), v_left.reshape(1, 1, t), v_tail.reshape(1, 1, t), zeros], dim=1)
        probe_m2 = torch.cat([vinn_t.reshape(1, 1, t), v_out.reshape(1, 1, t), v_tail.reshape(1, 1, t), zeros], dim=1)
        probe_m3 = torch.cat([v_left.reshape(1, 1, t), v_left.reshape(1, 1, t), vdd_t, vdd_t], dim=1)
        probe_m4 = torch.cat([v_left.reshape(1, 1, t), v_out.reshape(1, 1, t), vdd_t, vdd_t], dim=1)
        probe_m5 = torch.cat([vbias_t, v_tail.reshape(1, 1, t), zeros, zeros], dim=1)

        i_m1 = self.m1.drain_current(probe_m1)[0, 0, :]
        i_m2 = self.m2.drain_current(probe_m2)[0, 0, :]
        i_m3 = self.m3.drain_current(probe_m3)[0, 0, :]
        i_m4 = self.m4.drain_current(probe_m4)[0, 0, :]
        i_m5 = self.m5.drain_current(probe_m5)[0, 0, :]

        kcl_tail = i_m1[1:] + i_m2[1:] - i_m5[1:]
        kcl_left = i_m3[1:] - i_m1[1:]
        kcl_out = i_m4[1:] - i_m2[1:]
        if self.c_load_f > 0.0:
            dv_out = v_out[1:] - v_out[:-1]
            kcl_out = kcl_out - self.c_load_f * dv_out / dt_vec

        r_tail = torch.cat([(v_tail[0:1] - v_dc[0]).reshape(1), kcl_tail])
        r_left = torch.cat([(v_left[0:1] - v_dc[1]).reshape(1), kcl_left])
        r_out = torch.cat([(v_out[0:1] - v_dc[2]).reshape(1), kcl_out])
        return torch.cat([r_tail, r_left, r_out])

    def _gmres_direction(
        self,
        rfn: "Callable[[Tensor], Tensor]",  # type: ignore[name-defined]
        v_flat: Tensor,
        res_vec: Tensor,
        rn: float,
    ) -> Tensor:
        """Newton direction via GMRES with JVP matvec (avoids dense Jacobian)."""
        n = v_flat.shape[0]
        dtype = v_flat.dtype
        dev = v_flat.device
        rhs = -res_vec.detach().cpu().to(torch.float64).numpy()

        def _matvec(v_np: np.ndarray) -> np.ndarray:
            v_t = torch.from_numpy(v_np.astype(np.float32)).to(device=dev, dtype=dtype)
            _, jv = torch_jvp(rfn, (v_flat,), (v_t,), create_graph=False, strict=False)
            return jv.detach().cpu().to(torch.float64).numpy()

        A = LinearOperator((n, n), matvec=_matvec, dtype=np.float64)
        # Inexact Newton tolerance: tighten as residual decreases.
        gmres_tol = min(0.5, float(rn) ** 0.5) * max(1e-10, self.residual_tol) / max(rn, 1e-30)
        gmres_tol = max(1e-10, min(0.5, gmres_tol))
        direction_np, info = scipy_gmres(A, rhs, rtol=gmres_tol, maxiter=self.gmres_max_iter, atol=0.0)
        if info != 0:
            # Fall back to one Jacobian+direct solve if GMRES failed.
            logger.debug("GMRES did not converge (info=%d); falling back to direct solve", info)
            jac = jacobian(rfn, v_flat, vectorize=True, create_graph=False)
            return torch.linalg.solve(jac, -res_vec.detach())
        return torch.from_numpy(direction_np.astype(np.float32)).to(device=dev, dtype=dtype)

    def solve(
        self,
        time_s: Tensor,
        vinp_t: Tensor,
        vinn_t: Tensor,
        v_dc: Tensor,
        v_flat_init: Tensor | None = None,
    ) -> OtaTransientSolution:
        """
        Runs the whole-window implicit NR and returns the transient solution.

        :param time_s: ``(T,)`` time grid in seconds (uniform or non-uniform).
        :param vinp_t: ``(T,)`` Vinp trajectory (V).
        :param vinn_t: ``(T,)`` Vinn trajectory (V).
        :param v_dc: ``(3,)`` DC operating point ``[v_tail, v_left, v_out]``
            used as the initial condition at ``t = 0``.
        :param v_flat_init: Optional ``(3T,)`` warm-start for the solver.
            Defaults to a constant trajectory at the DC values.
        :return: :class:`OtaTransientSolution` with the solved trajectories.
        """
        start = time_module.perf_counter()
        tg = time_s.to(device=self._device, dtype=self._dtype)
        vinp_grid = vinp_t.to(device=self._device, dtype=self._dtype)
        vinn_grid = vinn_t.to(device=self._device, dtype=self._dtype)
        v_dc_ = v_dc.to(device=self._device, dtype=self._dtype)
        dt_vec = tg[1:] - tg[:-1]
        t = tg.shape[0]
        if dt_vec.numel() == 0:
            raise ValueError("time_s must contain at least two samples")

        if v_flat_init is None:
            v_flat = v_dc_.unsqueeze(1).expand(3, t).reshape(-1).clone().requires_grad_(True)
        else:
            v_flat = v_flat_init.to(device=self._device, dtype=self._dtype).clone().requires_grad_(True)

        def rfn(x: Tensor) -> Tensor:
            return self._full_residual_flat(x, vinp_grid, vinn_grid, v_dc_, dt_vec)

        def rn_fn(x: Tensor) -> float:
            with torch.no_grad():
                return float(_inf_norm(rfn(x)))

        rn = rn_fn(v_flat.detach())
        hist: list[float] = [rn]
        converged = rn <= self.residual_tol
        iters = 0

        while not converged and iters < self.max_iter:
            iters += 1
            res_vec = rfn(v_flat)
            if self.use_gmres:
                direction = self._gmres_direction(rfn, v_flat, res_vec, rn)
            else:
                jac = jacobian(rfn, v_flat, vectorize=True, create_graph=False)
                direction = torch.linalg.solve(jac, -res_vec.detach())
            alpha, _ = _backtrack(
                rfn,
                v_flat.detach(),
                direction,
                rn,
                norm_fn=_inf_norm,
                armijo_c=self.armijo_c,
                alpha_min=self.alpha_min,
            )
            alpha = _cap_alpha(alpha, direction, self.v_step_cap)
            v_flat = (v_flat.detach() + alpha * direction).clamp(0.0, self.vdd).requires_grad_(True)
            rn = rn_fn(v_flat.detach())
            hist.append(rn)
            converged = rn <= self.residual_tol

        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        sol = v_flat.detach().reshape(3, t)
        logger.debug(
            "OtaTransientSolver: %s in %d iters (rn=%.2e, wall=%.1f ms)",
            "converged" if converged else "did not converge",
            iters,
            rn,
            wall_ms,
        )
        return OtaTransientSolution(
            time_s=tg.detach(),
            v_tail_v=sol[0],
            v_left_v=sol[1],
            v_out_v=sol[2],
            report=ConvergenceReport(
                converged=converged,
                iter_count=iters,
                residual_norm_history=tuple(hist),
                final_residual_norm=rn,
                wall_ms=wall_ms,
            ),
        )
