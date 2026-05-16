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
from spino.circuit.sizing import (
    OtaSizingProblem,
    _build_input_trajectories,
    compose_ota_differentiable,
    extract_metrics,
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


def _make_problem() -> OtaSizingProblem:
    return OtaSizingProblem(
        nfet_checkpoint=_NFET_CKPT,
        pfet_checkpoint=_PFET_CKPT,
        nfet_dataset=_NFET_DS,
        pfet_dataset=_PFET_DS,
        torch_device="cpu",
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
        map_location="cpu",
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
    tg = torch.tensor(time_np, dtype=torch.float32)
    vinp_t = torch.tensor(vinp_np, dtype=torch.float32)
    vinn_t = torch.tensor(vinn_np, dtype=torch.float32)
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
