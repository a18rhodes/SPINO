"""
Gradient sanity: FNO autograd dI_D/dw matches SPICE finite-difference.

Single device probe (M1, NFET diff pair) at the OTA DC operating point.
Perturb the channel width by ±1 % via physics_raw[w_idx]; compare the FNO
autograd gradient to (I_D(W+ε) - I_D(W-ε)) / (2ε) from isolated NGSpice.

This test verifies that the FNO encodes the correct sensitivity to device
width — the same gradient the W5-6 Adam sizing loop will rely on.
"""

from __future__ import annotations

import os
import shutil

import pytest
import torch

from spino.circuit.composition_io import (
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DeviceCheckpoint,
    load_fno_device,
)
from spino.circuit.simulation import run_operating_point
from spino.circuit.standalone_mosfet import (
    build_isolated_mosfet_circuit,
    isolated_mosfet_id_a,
)
from spino.constants import ARCSINH_SCALE_MA
from spino.mosfet.gen_data import ParameterSchema

# --------------------------------------------------------------------------- #
# Fixed OTA DC operating-point for M1 (diff-pair NFET), W=2.0 µm, L=0.40 µm #
# --------------------------------------------------------------------------- #
_VG = 0.9  # Vcm
_VD = 0.55  # V_left (approx, from Phase-3b OTA OP)
_VS = 0.1  # V_tail
_VB = 0.0
_W_UM = 2.0
_L_UM = 0.40
_EPS_UM = 0.02  # 1 % perturbation of W for FD
_REL_ERR_TOL = 0.20  # 20 % relative error gate
_T_PROBE = 64
_TRIM = 8
_MA_TO_A = 1.0e-3

_W_IDX = ParameterSchema.TRAINING_KEYS.index("w")

# --------------------------------------------------------------------------- #
# Integration guards                                                           #
# --------------------------------------------------------------------------- #
_NGSPICE_AVAILABLE = shutil.which("ngspice") is not None
_PDK_AVAILABLE = os.path.exists("/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
_NFET_CKPT = DEFAULT_NFET_CHECKPOINT
_NFET_DS = DEFAULT_NFET_DATASET
_INTEGRATION_READY = _NGSPICE_AVAILABLE and _PDK_AVAILABLE and _NFET_CKPT.exists() and _NFET_DS.exists()

_SKIP = pytest.mark.skipif(
    not _INTEGRATION_READY,
    reason="Requires NFET checkpoint, HDF5 dataset, NGSpice, and sky130 PDK",
)
_E2E = pytest.mark.e2e_spice


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _fno_mean_id(device, physics_leaf: torch.Tensor) -> torch.Tensor:
    """
    One FNO forward pass at the DC OP terminal voltages.

    ``physics_leaf`` is a leaf tensor so autograd can flow to it.
    Returns the post-trim mean drain current as a scalar tensor.
    """
    t = _T_PROBE
    probe = torch.zeros(1, 4, t, dtype=torch.float32)
    probe[0, 0, :] = _VG
    probe[0, 1, :] = _VD
    probe[0, 2, :] = _VS
    probe[0, 3, :] = _VB

    v_norm = (probe - device.v_mean) / device.v_std
    physics_norm = ((physics_leaf - device.p_mean) / device.p_std).unsqueeze(0)
    pred_log = device.model(v_norm, physics_norm)
    i_d = ARCSINH_SCALE_MA * torch.sinh(pred_log) * _MA_TO_A  # (1, 1, T)
    return i_d[0, 0, _TRIM:].mean()


def _spice_id(w_um: float) -> float:
    """Isolated NFET DC operating-point current (A) at width ``w_um`` µm."""
    circuit = build_isolated_mosfet_circuit(
        is_pfet=False,
        width_um=w_um,
        length_um=_L_UM,
        vg=_VG,
        vd=_VD,
        vs=_VS,
        vb=_VB,
    )
    op = run_operating_point(circuit)
    return isolated_mosfet_id_a(op)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


@_E2E
@_SKIP
def test_fno_autograd_di_dw_sign_matches_spice_fd() -> None:
    """FNO dI_D/dw and SPICE central FD must have the same sign."""
    spec = DeviceCheckpoint("sky130_nmos", _NFET_CKPT, _NFET_DS, _W_UM, _L_UM, "M1_grad_sanity")
    device = load_fno_device(spec, map_location="cpu")
    device.eval()

    physics_leaf = device.physics_raw.detach().clone().requires_grad_(True)
    i_mean = _fno_mean_id(device, physics_leaf)
    i_mean.backward()
    fno_grad = float(physics_leaf.grad[_W_IDX])

    id_plus = _spice_id(_W_UM + _EPS_UM)
    id_minus = _spice_id(_W_UM - _EPS_UM)
    spice_grad = (id_plus - id_minus) / (2.0 * _EPS_UM)

    assert fno_grad * spice_grad > 0, (
        f"Sign mismatch: FNO dI/dw={fno_grad:.3e} A/µm, "
        f"SPICE FD={spice_grad:.3e} A/µm  "
        f"(id_plus={id_plus:.3e} A, id_minus={id_minus:.3e} A)"
    )


@_E2E
@_SKIP
def test_fno_autograd_di_dw_relative_error_below_threshold() -> None:
    """Relative error between FNO autograd and SPICE FD must be < 20 %."""
    spec = DeviceCheckpoint("sky130_nmos", _NFET_CKPT, _NFET_DS, _W_UM, _L_UM, "M1_grad_sanity")
    device = load_fno_device(spec, map_location="cpu")
    device.eval()

    physics_leaf = device.physics_raw.detach().clone().requires_grad_(True)
    i_mean = _fno_mean_id(device, physics_leaf)
    i_mean.backward()
    fno_grad = float(physics_leaf.grad[_W_IDX])

    id_plus = _spice_id(_W_UM + _EPS_UM)
    id_minus = _spice_id(_W_UM - _EPS_UM)
    spice_grad = (id_plus - id_minus) / (2.0 * _EPS_UM)

    rel_err = abs(fno_grad - spice_grad) / (abs(spice_grad) + 1e-30)
    assert rel_err < _REL_ERR_TOL, (
        f"Relative error {rel_err:.1%} exceeds {_REL_ERR_TOL:.0%} gate  "
        f"(FNO={fno_grad:.3e} A/µm, SPICE FD={spice_grad:.3e} A/µm)"
    )
