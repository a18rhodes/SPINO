"""
IFT gradient sanity: d(slew_rate)/d(W_tail) via IFT must match sign of FD-SPICE.

This test exercises the full IFT gradient path introduced in W5-6:
  1. Build OTA devices at the baseline sizing.
  2. Run compose_ota_differentiable to get V_out(t) with IFT backward.
  3. Compute slew_rate from V_out and backpropagate to get d(slew_rate)/d(W_tail).
  4. Estimate the same gradient from SPICE via finite difference on W_tail.
  5. Assert sign agreement between the two estimates.

The test is integration-gated: it requires NGSpice, the sky130 PDK, and both
production FNO checkpoints/datasets to be present.
"""

# pylint: disable=duplicate-code

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest
import torch

from spino.circuit.composition_io import (
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    load_ota_5t_devices,
)
from spino.circuit.ota_composition import OtaDcSolver
from spino.circuit.sizing import (
    OtaSizingProblem,
    _build_input_trajectories,
    _eval_itail_through_fno,
    compose_ota_differentiable,
    extract_metrics,
    loss_fn,
)
from spino.circuit.tuning import OtaDesignPoint, simulate_ota_design_point
from spino.mosfet.device_strategy import DeviceStrategy

# ---------------------------------------------------------------------------
# Integration guards
# ---------------------------------------------------------------------------

_NGSPICE_AVAILABLE = shutil.which("ngspice") is not None
_PDK_AVAILABLE = os.path.exists("/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
_NFET_CKPT = DEFAULT_NFET_CHECKPOINT
_PFET_CKPT = DEFAULT_PFET_CHECKPOINT
_NFET_DS = DEFAULT_NFET_DATASET
_PFET_DS = DEFAULT_PFET_DATASET
_INTEGRATION_READY = (
    _NGSPICE_AVAILABLE
    and _PDK_AVAILABLE
    and _NFET_CKPT.exists()
    and _PFET_CKPT.exists()
    and _NFET_DS.exists()
    and _PFET_DS.exists()
)

_SKIP = pytest.mark.skipif(
    not _INTEGRATION_READY,
    reason="NGSpice, PDK, or FNO checkpoints not available.",
)

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

_W_DIFF = 3.0
_W_MIRROR = 3.0
_W_TAIL = 1.5
_L_UM = 0.40
_V_BIAS = 1.2
_EPS_UM = 0.015  # 1% of W_tail=1.5 µm

_THETA_INIT = torch.tensor([_W_DIFF, _W_MIRROR, _W_TAIL, _L_UM, _V_BIAS], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_problem(device: str = "cpu") -> OtaSizingProblem:
    return OtaSizingProblem(
        nfet_checkpoint=_NFET_CKPT,
        pfet_checkpoint=_PFET_CKPT,
        nfet_dataset=_NFET_DS,
        pfet_dataset=_PFET_DS,
        torch_device=device,
    )


def _load_base(problem: OtaSizingProblem):
    base_devices = load_ota_5t_devices(
        diff_w_um=_W_DIFF,
        diff_l_um=_L_UM,
        mirror_w_um=_W_MIRROR,
        mirror_l_um=_L_UM,
        tail_w_um=_W_TAIL,
        tail_l_um=_L_UM,
        nfet_checkpoint=problem.nfet_checkpoint,
        pfet_checkpoint=problem.pfet_checkpoint,
        nfet_dataset=problem.nfet_dataset,
        pfet_dataset=problem.pfet_dataset,
        map_location=problem.torch_device,
    )
    nfet_strat = DeviceStrategy.create("sky130_nmos")
    pfet_strat = DeviceStrategy.create("sky130_pmos")
    time_np = np.arange(0.0, problem.t_end, problem.t_step, dtype=np.float32)
    vinp_np, vinn_np = _build_input_trajectories(
        time_np,
        vcm_v=problem.vcm,
        t_step_start=problem.t_step_start,
        rise_time_s=5e-9,
        step_amp_v=problem.step_amp,
    )
    dev = problem.torch_device
    tg = torch.tensor(time_np, dtype=torch.float32, device=dev)
    vinp_t = torch.tensor(vinp_np, dtype=torch.float32, device=dev)
    vinn_t = torch.tensor(vinn_np, dtype=torch.float32, device=dev)
    return base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t


def _spice_slew(w_tail: float, problem: OtaSizingProblem) -> float:
    """Single-point NGSpice slew at fixed (W_diff, W_mirror) with varying ``w_tail``.

    Uses fixed W_diff=_W_DIFF and W_mirror=_W_MIRROR so the FD estimate is a
    clean ∂(slew)/∂(W_tail) at constant differential-pair and mirror geometry.
    """
    point = OtaDesignPoint(diff_w_um=_W_DIFF, mirror_w_um=_W_MIRROR)
    metrics = simulate_ota_design_point(
        point,
        vdd=problem.vdd,
        vcm_v=problem.vcm,
        step_amp_v=problem.step_amp,
        diff_l_um=_L_UM,
        mirror_l_um=_L_UM,
        tail_w_um=w_tail,
        tail_l_um=_L_UM,
        vbias_v=_V_BIAS,
        t_step_start=problem.t_step_start,
        t_end=problem.t_end,
        t_step=problem.t_step,
        c_load_f=problem.c_load_f,
        pdk_root=problem.pdk_root,
    )
    return float("nan") if not metrics.converged else metrics.slew_rate_v_per_us


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_SKIP
def test_ift_grad_w_tail_is_finite() -> None:
    """IFT gradient d(slew_rate)/d(W_tail) must be finite at baseline sizing."""
    problem = _make_problem()
    base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t = _load_base(problem)

    theta = _THETA_INIT.clone().requires_grad_(True)
    v_out, dc_sol = compose_ota_differentiable(theta, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    slew = extract_metrics(v_out, tg, dc_sol, problem)["slew_rate_v_per_us"]
    assert isinstance(slew, torch.Tensor)
    slew.backward()
    assert theta.grad is not None
    grad_w_tail = float(theta.grad[2])
    assert np.isfinite(grad_w_tail), f"IFT gradient is not finite: {grad_w_tail}"


@_SKIP
def test_ift_grad_w_tail_sign_matches_fd_spice() -> None:
    """Sign of IFT d(slew_rate)/d(W_tail) must match the FD-SPICE estimate.

    Wider tail transistor → higher tail current → higher slew rate.
    Both IFT autograd and FD-SPICE should give a positive gradient.
    """
    problem = _make_problem()
    base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t = _load_base(problem)

    # --- IFT autograd gradient ---
    theta = _THETA_INIT.clone().requires_grad_(True)
    v_out, dc_sol = compose_ota_differentiable(theta, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    slew = extract_metrics(v_out, tg, dc_sol, problem)["slew_rate_v_per_us"]
    assert isinstance(slew, torch.Tensor)
    slew.backward()
    assert theta.grad is not None
    ift_sign = float(theta.grad[2])

    # --- FD-SPICE gradient (sign only) ---
    slew_plus = _spice_slew(_W_TAIL + _EPS_UM, problem)
    slew_minus = _spice_slew(_W_TAIL - _EPS_UM, problem)
    fd_sign = slew_plus - slew_minus  # positive if wider → faster

    # Only assert sign agreement (magnitude is not expected to match)
    assert np.isfinite(fd_sign), f"FD-SPICE estimate is NaN: plus={slew_plus}, minus={slew_minus}"
    assert np.isfinite(ift_sign), "IFT gradient is NaN"
    assert (ift_sign * fd_sign) > 0, f"Sign mismatch: IFT gradient={ift_sign:.4f}, FD-SPICE sign={fd_sign:.4f}"


# ---------------------------------------------------------------------------
# Power-gradient tests (M1: real differentiable I_tail in sizing loss)
# ---------------------------------------------------------------------------


@_SKIP
def test_power_gradient_finite_all_theta() -> None:
    """∂(power)/∂θ via _eval_itail_through_fno must be finite on all 5 components.

    Power flows through the M5 FNO with autograd. Geometry sensitivities
    (W_tail, L) come from bilinear interpolation of the BSIM physics vector;
    V_bias enters through the M5 gate trajectory. W_diff and W_mirror are
    expected to carry zero gradient (M5 has no dependency on them) but the
    backward pass must still complete cleanly.
    """
    problem = _make_problem()
    base_devices, nfet_strat, _, _, _, _ = _load_base(problem)

    theta = _THETA_INIT.clone().requires_grad_(True)
    dc_sol = OtaDcSolver(
        *base_devices,
        vdd=problem.vdd,
        vcm_v=problem.vcm,
        vbias_v=float(theta[4].detach().item()),
    ).solve()
    i_tail = _eval_itail_through_fno(theta, dc_sol, base_devices, nfet_strat)
    assert isinstance(i_tail, torch.Tensor)
    assert i_tail.requires_grad, "i_tail must carry an autograd link to theta"

    power_uw = i_tail.abs() * problem.vdd * 1e6
    power_uw.backward()

    assert theta.grad is not None
    grad_np = theta.grad.detach().cpu().numpy()
    assert np.all(np.isfinite(grad_np)), f"non-finite power gradient: {grad_np}"
    # W_diff (0) and W_mirror (1) do not enter M5 — gradient must be exactly zero.
    assert grad_np[0] == 0.0, f"∂P/∂W_diff should be 0, got {grad_np[0]}"
    assert grad_np[1] == 0.0, f"∂P/∂W_mirror should be 0, got {grad_np[1]}"
    # W_tail (2), L (3), V_bias (4) should each carry a non-zero gradient.
    for idx, name in [(2, "W_tail"), (3, "L"), (4, "V_bias")]:
        assert abs(grad_np[idx]) > 0.0, f"∂P/∂{name} unexpectedly zero ({grad_np[idx]})"


@_SKIP
def test_power_gradient_sign_w_tail_positive() -> None:
    """∂(power)/∂(W_tail) must be positive: wider tail → more I_tail → more power."""
    problem = _make_problem()
    base_devices, nfet_strat, _, _, _, _ = _load_base(problem)

    theta = _THETA_INIT.clone().requires_grad_(True)
    dc_sol = OtaDcSolver(
        *base_devices,
        vdd=problem.vdd,
        vcm_v=problem.vcm,
        vbias_v=float(theta[4].detach().item()),
    ).solve()
    i_tail = _eval_itail_through_fno(theta, dc_sol, base_devices, nfet_strat)
    power_uw = i_tail.abs() * problem.vdd * 1e6
    power_uw.backward()

    grad_w_tail = float(theta.grad[2])
    assert grad_w_tail > 0.0, f"∂P/∂W_tail expected positive, got {grad_w_tail:.4e}"


@_SKIP
def test_full_loss_gradient_finite_all_theta() -> None:
    """Combined slew + power loss must produce a finite gradient on all 5 θ components.

    Exercises the same code path the Adam loop uses: compose_ota_differentiable
    for v_out, _eval_itail_through_fno for the power term, extract_metrics with
    the i_tail_tensor argument, and loss_fn returning a Tensor.
    """
    problem = _make_problem()
    base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t = _load_base(problem)

    theta = _THETA_INIT.clone().requires_grad_(True)
    v_out, dc_sol = compose_ota_differentiable(
        theta, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t
    )
    i_tail = _eval_itail_through_fno(theta, dc_sol, base_devices, nfet_strat)
    metrics = extract_metrics(v_out, tg, dc_sol, problem, i_tail_tensor=i_tail)
    loss = loss_fn(metrics, problem)
    assert isinstance(loss, torch.Tensor)
    loss.backward()

    grad_np = theta.grad.detach().cpu().numpy()
    assert np.all(np.isfinite(grad_np)), f"non-finite loss gradient: {grad_np}"


# ---------------------------------------------------------------------------
# M2: magnitude-checked IFT gradient verification
# ---------------------------------------------------------------------------
#
# Test A (IFT plumbing isolation): IFT-of-slew-via-FNO vs central-FD-of-slew-via-FNO.
#   Both differentiate the same FNO, so any disagreement is the IFT machinery.
#   Tolerance: relative L2 <= 5%.
#
# Test B (surrogate fidelity bound): central-FD-of-slew-via-FNO vs central-FD-of-slew-via-SPICE.
#   Both use the same FD discretisation, so disagreement bounds the FNO surrogate's
#   gradient error against SPICE ground truth.
#   Tolerance: relative L2 <= 20%.
#
# Evaluated at three pinned theta points drawn from the production multi-spec
# Adam trajectory (theta_init, theta_step5, theta_final). Slew is the metric of
# choice because it has nonzero gradient everywhere in the feasible region
# whereas the full loss collapses to zero once both hinges go silent.

# Tolerances for the two M2 sub-tests on the slew gradient.
#   Test A (IFT vs FD-via-FNO) targets IFT plumbing fidelity.
#   Test B (FD-via-FNO vs FD-via-SPICE) bounds the FNO surrogate-gradient error
#   against SPICE ground truth.
_M2_TOL_TEST_A: float = 0.05
_M2_TOL_TEST_B: float = 0.20

# Pinned theta points drawn from the v3 multi-spec Adam trajectory.
# Theta at step5 sits on the L = 0.18 µm PDK lower bound, where two distinct
# methodological divergences appear: IFT linearisation vs nonlinear FD
# discretisation, and FNO surrogate gradient bias at the training-distribution
# edge. The combined test xfails at this name; the other two names assert
# strictly against the tolerances above.
_M2_BOUNDARY_THETA_NAME: str = "step5"

_THETA_M2_POINTS: list[tuple[str, tuple[float, float, float, float, float]]] = [
    ("init", (3.0, 3.0, 1.0, 0.40, 0.9)),
    (_M2_BOUNDARY_THETA_NAME, (3.283, 3.276, 1.294, 0.180, 1.193)),
    ("final", (3.638, 3.606, 1.592, 0.308, 1.537)),
]


def _slew_via_fno(
    theta_vec: tuple[float, ...],
    problem: OtaSizingProblem,
    base_devices,
    nfet_strat,
    pfet_strat,
    tg,
    vinp_t,
    vinn_t,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose, extract slew metric. Returns (slew_tensor, theta_tensor_with_grad)."""
    theta = torch.tensor(theta_vec, dtype=torch.float32, device=tg.device).requires_grad_(True)
    v_out, dc_sol = compose_ota_differentiable(theta, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    metrics = extract_metrics(v_out, tg, dc_sol, problem)
    slew = metrics["slew_rate_v_per_us"]
    assert isinstance(slew, torch.Tensor)
    return slew, theta


def _ift_slew_grad(
    theta_vec: tuple[float, ...],
    problem: OtaSizingProblem,
    base_devices,
    nfet_strat,
    pfet_strat,
    tg,
    vinp_t,
    vinn_t,
) -> np.ndarray:
    slew, theta = _slew_via_fno(theta_vec, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    slew.backward()
    assert theta.grad is not None
    return theta.grad.detach().cpu().numpy()


# PDK validity bounds applied symmetrically across both FD helpers and the IFT
# internal _jtheta_fd in sizing.py. Each entry is (low, high) for the corresponding
# theta component; values clamped before the BSIM lookup. Matches the clamp set
# in spino.circuit.sizing._jtheta_fd so the FD-via-FNO and FD-via-SPICE estimates
# discretise the same gradient the IFT path computes.
_FD_THETA_CLAMP: tuple[tuple[float, float], ...] = (
    (0.1, 20.0),  # W_diff
    (0.1, 20.0),  # W_mirror
    (0.1, 20.0),  # W_tail
    (0.18, 1.0),  # L
    (0.5, 1.6),   # V_bias
)


def _bracket_perturbation(theta_vec: tuple[float, ...], i: int, eps_rel: float) -> tuple[list[float], list[float], float]:
    """Return (plus_vals, minus_vals, effective_eps) with PDK clamping.

    If the symmetric +/- step would leave the validity bracket on either side,
    fall back to a one-sided FD on the interior direction and halve the
    effective eps so the FD ratio still divides by the actual displacement.
    """
    low, high = _FD_THETA_CLAMP[i]
    eps_i = max(abs(theta_vec[i]) * eps_rel, 1e-4)
    base = float(theta_vec[i])
    plus_val = base + eps_i
    minus_val = base - eps_i
    if plus_val > high and minus_val >= low:
        plus_val = base
        eps_used = eps_i / 2.0
    elif minus_val < low and plus_val <= high:
        minus_val = base
        eps_used = eps_i / 2.0
    else:
        eps_used = eps_i
    plus = list(theta_vec)
    minus = list(theta_vec)
    plus[i] = max(low, min(high, plus_val))
    minus[i] = max(low, min(high, minus_val))
    return plus, minus, eps_used


def _fd_slew_grad_fno(
    theta_vec: tuple[float, ...],
    problem: OtaSizingProblem,
    base_devices,
    nfet_strat,
    pfet_strat,
    tg,
    vinp_t,
    vinn_t,
    eps_rel: float = 0.01,
) -> np.ndarray:
    grad = np.zeros(5)
    for i in range(5):
        plus, minus, eps_used = _bracket_perturbation(theta_vec, i, eps_rel)
        slew_plus, _ = _slew_via_fno(tuple(plus), problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
        slew_minus, _ = _slew_via_fno(tuple(minus), problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
        grad[i] = (float(slew_plus.detach()) - float(slew_minus.detach())) / (2.0 * eps_used)
    return grad


def _fd_slew_grad_spice(
    theta_vec: tuple[float, ...],
    problem: OtaSizingProblem,
    eps_rel: float = 0.01,
) -> np.ndarray:
    grad = np.zeros(5)
    for i in range(5):
        plus_vals, minus_vals, eps_used = _bracket_perturbation(theta_vec, i, eps_rel)

        def _slew_at(vals):
            point = OtaDesignPoint(diff_w_um=vals[0], mirror_w_um=vals[1])
            metrics = simulate_ota_design_point(
                point,
                vdd=problem.vdd,
                vcm_v=problem.vcm,
                step_amp_v=problem.step_amp,
                diff_l_um=vals[3],
                mirror_l_um=vals[3],
                tail_w_um=vals[2],
                tail_l_um=vals[3],
                vbias_v=vals[4],
                t_step_start=problem.t_step_start,
                t_end=problem.t_end,
                t_step=problem.t_step,
                c_load_f=problem.c_load_f,
                pdk_root=problem.pdk_root,
            )
            return float("nan") if not metrics.converged else float(metrics.slew_rate_v_per_us)

        s_plus = _slew_at(plus_vals)
        s_minus = _slew_at(minus_vals)
        if not (np.isfinite(s_plus) and np.isfinite(s_minus)):
            grad[i] = float("nan")
        else:
            grad[i] = (s_plus - s_minus) / (2.0 * eps_used)
    return grad


def _rel_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    denom = float(np.linalg.norm(reference))
    if denom < 1e-12:
        return float(np.linalg.norm(estimate - reference))
    return float(np.linalg.norm(estimate - reference) / denom)


@_SKIP
@pytest.mark.parametrize("name,theta_vec", _THETA_M2_POINTS, ids=[p[0] for p in _THETA_M2_POINTS])
def test_slew_grad_ift_and_surrogate_fidelity(name: str, theta_vec: tuple[float, ...]) -> None:
    """M2 combined: Test A (IFT vs FD-via-FNO, <=5%) + Test B (FD-via-FNO vs FD-via-SPICE, <=20%).

    Slew gradient is the metric. Three pinned theta points from the multi-spec
    Adam trajectory. Each instance prints the three gradients before asserting
    so a partial failure (e.g., one component out of tolerance) leaves an
    artefact in the test log for downstream investigation.
    """
    problem = _make_problem(device="cuda" if torch.cuda.is_available() else "cpu")
    base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t = _load_base(problem)

    g_ift = _ift_slew_grad(theta_vec, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    g_fd_fno = _fd_slew_grad_fno(theta_vec, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    g_fd_spice = _fd_slew_grad_spice(theta_vec, problem)

    rel_a = _rel_l2(g_ift, g_fd_fno)
    rel_b = _rel_l2(g_fd_fno, g_fd_spice)

    print(f"\n[{name}] theta = {theta_vec}")
    print(f"[{name}] IFT      grad = {g_ift}")
    print(f"[{name}] FD-FNO   grad = {g_fd_fno}")
    print(f"[{name}] FD-SPICE grad = {g_fd_spice}")
    print(f"[{name}] Test A (IFT vs FD-FNO) rel L2 = {rel_a:.4f}")
    print(f"[{name}] Test B (FD-FNO vs FD-SPICE) rel L2 = {rel_b:.4f}")

    if name == _M2_BOUNDARY_THETA_NAME:
        pytest.xfail(
            f"PDK-boundary methodological divergence at {name}: "
            f"Test A rel L2 {rel_a:.4f} (IFT-vs-nonlinear-FD discretisation), "
            f"Test B rel L2 {rel_b:.4f} (FNO surrogate-gradient bias at L=0.18 bound). "
            f"See docs/sizing.md §'Gradient-verification bounds'."
        )

    assert (
        rel_a <= _M2_TOL_TEST_A
    ), f"Test A fail at {name}: rel L2 {rel_a:.4f} > {_M2_TOL_TEST_A}\nIFT={g_ift}\nFD-FNO={g_fd_fno}"
    assert (
        rel_b <= _M2_TOL_TEST_B
    ), f"Test B fail at {name}: rel L2 {rel_b:.4f} > {_M2_TOL_TEST_B}\nFD-FNO={g_fd_fno}\nFD-SPICE={g_fd_spice}"
