"""
Error attribution probes for FNO-composed CS amplifier vs NGSpice.

Persists probe vectors next to a composed ``summary.json`` run for offline review.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spino.circuit.composition import TransientSolution, TransientSolver, transient_kcl_residual_waveform
from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.simulation import TransientResult

_SPICE_IN = "v(in)"
_SPICE_OUT = "v(out)"


def _resample_transient_vout(
    spice_tran: TransientResult,
    fno_solution: TransientSolution,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligns SPICE ``v(out)`` to the FNO time grid."""
    fno_time = fno_solution.time_s.cpu().numpy()
    vout = np.interp(fno_time, spice_tran.time, spice_tran.variables[_SPICE_OUT])
    return vout, fno_time

logger = logging.getLogger(__name__)

__all__ = [
    "CS_AMP_NFET_TRAN_KEY",
    "CS_AMP_PFET_TRAN_KEY",
    "mosfet_id_key_for_instance",
    "probe1_iv_branch_errors",
    "probe2_kcl_residual_bundle",
    "write_attribution_manifest",
    "write_probe_outputs",
]


CS_AMP_NFET_TRAN_KEY = "i(@m.xm1.msky130_fd_pr__nfet_01v8[id])"
CS_AMP_PFET_TRAN_KEY = "i(@m.xm2.msky130_fd_pr__pfet_01v8[id])"


def mosfet_id_key_for_instance(tran_vars: dict[str, Any], instance: str) -> str:
    """
    Finds the NGSpice transient drain-current key for ``XM1`` / ``XM2``.

    :param tran_vars: ``TransientResult.variables`` mapping.
    :param instance: ``\"XM1\"`` or ``\"XM2\"`` (case-insensitive).
    :return: Matching variable name.
    """
    prefix = f"i(@m.{instance.lower()}."
    for name in tran_vars:
        lower = name.lower()
        if lower.startswith(prefix) and "[id]" in lower:
            return name
    raise KeyError(f"No drain current key for instance {instance} among transient variables")


def probe1_iv_branch_errors(
    nfet: FnoMosfetDevice,
    pfet: FnoMosfetDevice,
    *,
    vdd: float,
    spice_tran: TransientResult,
    fno_solution: TransientSolution,
) -> dict[str, Any]:
    """
    Operator-only IV comparison at SPICE node voltages on the FNO time grid.

    Compares FNO drain-current magnitudes to SPICE branch ``id`` using the same
    ``(Vg,Vd,Vs,Vb)`` probes as composition.

    :param nfet: NFET FNO wrapper.
    :param pfet: PFET FNO wrapper.
    :param vdd: Supply voltage (volts).
    :param spice_tran: NGSpice transient with branch currents.
    :param fno_solution: Converged FNO transient (defines the time base).
    :return: JSON-serialisable statistics and per-sample error arrays.
    """
    spice_vout, _ = _resample_transient_vout(spice_tran, fno_solution)
    fno_time = fno_solution.time_s.cpu().numpy()
    vin_spice = np.interp(fno_time, spice_tran.time, spice_tran.variables[_SPICE_IN])
    device = nfet.v_mean.device
    dtype = nfet.v_mean.dtype
    v_out_t = torch.tensor(spice_vout, device=device, dtype=dtype)
    vin_t = torch.tensor(vin_spice, device=device, dtype=dtype)
    with torch.no_grad():
        i_nfet_f = nfet.drain_current(_nfet_probe_traj(v_out_t, vin_t))[0, 0, :].cpu().numpy()
        i_pfet_f = pfet.drain_current(_pfet_probe_traj(v_out_t, vdd, device, dtype))[0, 0, :].cpu().numpy()
    k_nf = mosfet_id_key_for_instance(spice_tran.variables, "XM1")
    k_pf = mosfet_id_key_for_instance(spice_tran.variables, "XM2")
    i_nfet_s = np.interp(fno_time, spice_tran.time, spice_tran.variables[k_nf])
    i_pfet_s = np.interp(fno_time, spice_tran.time, spice_tran.variables[k_pf])
    err_nf = np.abs(i_nfet_f - np.abs(i_nfet_s))
    err_pf = np.abs(i_pfet_f - np.abs(i_pfet_s))
    vgs_nf = vin_spice - 0.0
    mask_weak = vgs_nf < 0.50
    return {
        "nfet_max_abs_error_a": float(np.max(err_nf)),
        "pfet_max_abs_error_a": float(np.max(err_pf)),
        "nfet_max_abs_error_weak_vgs_a": float(np.max(err_nf[mask_weak])) if np.any(mask_weak) else 0.0,
        "nfet_max_abs_error_strong_vgs_a": float(np.max(err_nf[~mask_weak])) if np.any(~mask_weak) else 0.0,
        "time_s": fno_time.tolist(),
        "err_nfet_a": err_nf.tolist(),
        "err_pfet_a": err_pf.tolist(),
    }


def _nfet_probe_traj(v_out: torch.Tensor, vin_t: torch.Tensor) -> torch.Tensor:
    tsteps = v_out.shape[0]
    z = torch.zeros((1, 1, tsteps), dtype=v_out.dtype, device=v_out.device)
    return torch.cat([vin_t.reshape(1, 1, tsteps), v_out.reshape(1, 1, tsteps), z, z], dim=1)


def _pfet_probe_traj(v_out: torch.Tensor, vdd: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tsteps = v_out.shape[0]
    vdd_t = torch.full((1, 1, tsteps), vdd, dtype=dtype, device=device)
    vo = v_out.reshape(1, 1, tsteps)
    return torch.cat([vo, vo, vdd_t, vdd_t], dim=1)


def probe2_kcl_residual_bundle(
    solver: TransientSolver,
    *,
    v_out_dc_spice: float,
    v_out_spice_aligned: np.ndarray,
    v_out_fno: torch.Tensor,
    vin_t: torch.Tensor,
    time_s: torch.Tensor,
) -> dict[str, Any]:
    """
    Pointwise KCL residual waveforms at pinned SPICE and FNO trajectories.

    :param solver: Transient solver sharing device geometry with the FNO run.
    :param v_out_dc_spice: SPICE-aligned IC for the residual definition (volts).
    :param v_out_spice_aligned: SPICE ``v(out)`` resampled to the FNO grid.
    :param v_out_fno: FNO ``v(out)`` tensor ``(T,)``.
    :param vin_t: Input trajectory tensor ``(T,)``.
    :param time_s: Time grid ``(T,)``.
    :return: Dict with per-pin blocks. Row ``0`` of the underlying whole-window
        residual is the IC clamp ``V_out[0] - v_out_dc`` in **volts**; rows ``1..T-1``
        are KCL residuals in **amperes**. Returned ``ic_residual_v`` is row ``0``;
        ``kcl_max_a``, ``kcl_rms_a``, and ``waveform_a`` use **only** the KCL tail
        (rows ``1..T-1``), all in amperes.
    """
    dt = time_s[1:] - time_s[:-1]
    device = v_out_fno.device
    dtype = v_out_fno.dtype
    v_sp = torch.tensor(v_out_spice_aligned, device=device, dtype=dtype)
    r_sp = transient_kcl_residual_waveform(solver, v_sp, vin_t, float(v_out_dc_spice), dt)
    r_fn = transient_kcl_residual_waveform(solver, v_out_fno, vin_t, float(v_out_dc_spice), dt)
    r_sp_np = r_sp.detach().cpu().numpy()
    r_fn_np = r_fn.detach().cpu().numpy()
    kcl_sp = r_sp_np[1:]
    kcl_fn = r_fn_np[1:]
    return {
        "pinned_spice_vout": {
            "ic_residual_v": float(r_sp_np[0]),
            "kcl_max_a": float(np.max(np.abs(kcl_sp))),
            "kcl_rms_a": float(np.sqrt(np.mean(kcl_sp**2))),
            "waveform_a": kcl_sp.tolist(),
        },
        "pinned_fno_vout": {
            "ic_residual_v": float(r_fn_np[0]),
            "kcl_max_a": float(np.max(np.abs(kcl_fn))),
            "kcl_rms_a": float(np.sqrt(np.mean(kcl_fn**2))),
            "waveform_a": kcl_fn.tolist(),
        },
    }


def write_probe_outputs(out_dir: Path, probe1: dict[str, Any], probe2: dict[str, Any]) -> None:
    """Writes ``probe1_iv_error.npz`` (dense arrays) and ``probe2_kcl_residual.json``."""
    att = out_dir / "attribution"
    att.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        att / "probe1_iv_error.npz",
        time_s=np.array(probe1["time_s"], dtype=np.float64),
        err_nfet_a=np.array(probe1["err_nfet_a"], dtype=np.float64),
        err_pfet_a=np.array(probe1["err_pfet_a"], dtype=np.float64),
    )
    meta = {k: v for k, v in probe1.items() if k not in ("time_s", "err_nfet_a", "err_pfet_a")}
    (att / "probe1_iv_error_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (att / "probe2_kcl_residual.json").write_text(json.dumps(probe2, indent=2), encoding="utf-8")


def write_attribution_manifest(
    out_dir: Path,
    baseline_summary: Path,
    *,
    probe3_hybrid: Path | None = None,
) -> None:
    """Writes ``attribution_manifest.json`` sidecar."""
    payload: dict[str, str] = {
        "baseline_summary": str(baseline_summary),
        "probe_1_iv_error": str((out_dir / "attribution" / "probe1_iv_error.npz").resolve()),
        "probe_2_kcl_residual": str((out_dir / "attribution" / "probe2_kcl_residual.json").resolve()),
    }
    if probe3_hybrid is not None:
        payload["probe_3_hybrid_summary"] = str(probe3_hybrid.resolve())
    (out_dir / "attribution_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
