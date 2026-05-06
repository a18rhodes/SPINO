"""
Unit tests for :mod:`spino.circuit.error_attribution`.

Covers MOSFET drain-current key discovery and Probe 2 KCL residual bundle layout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spino.circuit.composition import TransientSolver, transient_kcl_residual_waveform
from spino.circuit.error_attribution import mosfet_id_key_for_instance, probe2_kcl_residual_bundle


@pytest.mark.parametrize(
    ("instance", "kind"),
    [
        ("XM1", "nfet"),
        ("XM2", "pfet"),
    ],
)
def test_mosfet_id_key_discovery(instance: str, kind: str) -> None:
    """Drain-current keys follow NGSpice hierarchical naming."""
    variables = {
        "v(out)": np.array([0.0]),
        "i(@m.xm1.msky130_fd_pr__nfet_01v8[id])": np.array([1e-6]),
        "i(@m.xm2.msky130_fd_pr__pfet_01v8[id])": np.array([-1e-6]),
    }
    k = mosfet_id_key_for_instance(variables, instance)
    assert kind in k.lower() and "[id]" in k.lower()


def _flat_currents(_self, v_out: torch.Tensor, _vin_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant branch currents for residual shape tests."""
    i = torch.full_like(v_out, fill_value=1e-5)
    return i, i


def test_transient_kcl_residual_waveform_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """Residual vector ``(T,)`` matches trajectory length."""
    monkeypatch.setattr(TransientSolver, "_device_currents", _flat_currents)
    nf = torch.nn.Module()
    nf.register_buffer("v_mean", torch.zeros(4, 1))
    nf.register_buffer("v_std", torch.ones(4, 1))
    pf = torch.nn.Module()
    pf.register_buffer("v_mean", torch.zeros(4, 1))
    pf.register_buffer("v_std", torch.ones(4, 1))
    solver = TransientSolver(nf, pf, vdd=1.8, c_load_f=1e-12)  # type: ignore[arg-type]
    t = 6
    time_s = torch.linspace(0.0, 5e-9, t)
    vin_t = torch.full((t,), 0.85)
    v_out = torch.linspace(0.5, 0.55, t)
    dt = time_s[1:] - time_s[:-1]
    r = transient_kcl_residual_waveform(solver, v_out, vin_t, 0.52, dt)
    assert r.numel() == t


def test_probe2_bundle_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe 2 JSON bundle contains SPICE and FNO pinned blocks."""
    monkeypatch.setattr(TransientSolver, "_device_currents", _flat_currents)
    nf = torch.nn.Module()
    nf.register_buffer("v_mean", torch.zeros(4, 1))
    nf.register_buffer("v_std", torch.ones(4, 1))
    pf = torch.nn.Module()
    pf.register_buffer("v_mean", torch.zeros(4, 1))
    pf.register_buffer("v_std", torch.ones(4, 1))
    sol = TransientSolver(nf, pf, vdd=1.8, c_load_f=1e-12)  # type: ignore[arg-type]
    t = 7
    time_s = torch.linspace(0.0, 6e-9, t)
    vin = torch.full((t,), 0.85)
    vout = torch.full((t,), 0.6)
    out = probe2_kcl_residual_bundle(
        sol,
        v_out_dc_spice=0.6,
        v_out_spice_aligned=np.full(t, 0.6),
        v_out_fno=vout,
        vin_t=vin,
        time_s=time_s,
    )
    assert set(out.keys()) == {"pinned_spice_vout", "pinned_fno_vout"}
    for pin in ("pinned_spice_vout", "pinned_fno_vout"):
        blk = out[pin]
        assert set(blk.keys()) == {"ic_residual_v", "kcl_max_a", "kcl_rms_a", "waveform_a"}
        assert len(blk["waveform_a"]) == t - 1
