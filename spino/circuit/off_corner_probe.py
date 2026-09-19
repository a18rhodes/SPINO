"""
Off-corner transferability probe for the 5T-OTA FNO composition.

Runs the FNO-composed transient at one production design point and compares
its output trajectory against two NGSpice references:

1. NGSpice at ``tt`` corner, 27 °C (the FNO's training corner).
2. NGSpice at ``ff`` corner, 125 °C (single off-corner probe).

The FNO is trained on ``tt`` BSIM parameters only and has no corner-awareness
in its conditioning. The expectation is that FNO output stays close to
SPICE-tt and degrades against SPICE-ff. The reported metrics (Pearson r,
max\\|ΔV\\|) at each corner provide a single-corner-trained, single off-corner
generalisation bound.

Usage::

    python -m spino.circuit.off_corner_probe \\
        --output-dir runs/off_corner/ota_ff_125c
"""

from __future__ import annotations

import json
import logging
import time as time_module
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from spino.circuit.chain_metrics import max_abs_delta_v, pearson_r
from spino.circuit.composition_io import load_ota_5t_devices
from spino.circuit.simulation import TransientResult, run_transient
from spino.circuit.sizing import (
    OtaSizingProblem,
    _build_input_trajectories,
    compose_ota_differentiable,
)
from spino.circuit.topologies import build_ota_5t
from spino.mosfet.device_strategy import DeviceStrategy
from spino.plot_styles import get_palette

logger = logging.getLogger(__name__)

_SPICE_OUTPUT_NODE = "v(n_out)"

# Production-sizing design point (from Phase 3b OTA showcase, L = 0.40 µm).
_W_DIFF = 8.0
_W_MIRROR = 8.0
_W_TAIL = 2.0
_L_UM = 0.40
_V_BIAS = 1.2

# Off-corner: FF, 125 °C. Reference: TT, 27 °C (NGSpice default).
_TT_LABEL = "tt @ 27 °C"
_FF_LABEL = "ff @ 125 °C"


@dataclass
class _CornerResult:
    """SPICE transient at one process/temperature corner."""

    label: str
    corner: str
    temperature_c: float | None
    v_out_aligned: np.ndarray
    wall_ms: float


def _spice_at_corner(  # pylint: disable=too-many-locals
    *,
    problem: OtaSizingProblem,
    time_np: np.ndarray,
    corner: str,
    temperature_c: float | None,
    label: str,
) -> _CornerResult:
    """Run a single SPICE transient at the given corner + temperature.

    Returns the V_out trajectory resampled onto ``time_np``.
    """
    vcm_v = problem.vcm
    t_step_start = problem.t_step_start
    rise_time_s = 5e-9
    step_amp_v = problem.step_amp
    t_end = problem.t_end
    vinp_pwl = (
        f"PWL(0 {vcm_v} "
        f"{t_step_start} {vcm_v} "
        f"{t_step_start + rise_time_s} {vcm_v + step_amp_v} "
        f"{t_end} {vcm_v + step_amp_v})"
    )
    vinn_pwl = (
        f"PWL(0 {vcm_v} "
        f"{t_step_start} {vcm_v} "
        f"{t_step_start + rise_time_s} {vcm_v - step_amp_v} "
        f"{t_end} {vcm_v - step_amp_v})"
    )
    circuit = build_ota_5t(
        diff_w_um=_W_DIFF,
        diff_l_um=_L_UM,
        mirror_w_um=_W_MIRROR,
        mirror_l_um=_L_UM,
        tail_w_um=_W_TAIL,
        tail_l_um=_L_UM,
        vdd=problem.vdd,
        vbias_v=_V_BIAS,
        vcm_v=vcm_v,
        vinp_tran=vinp_pwl,
        vinn_tran=vinn_pwl,
        c_load_f=problem.c_load_f,
        corner=corner,
        pdk_root=problem.pdk_root or "/app/sky130_volare",
    )
    t0 = time_module.perf_counter()
    tran: TransientResult | None = run_transient(
        circuit,
        t_step=problem.t_step,
        t_end=problem.t_end,
        temperature=temperature_c,
    )
    wall_ms = 1000.0 * (time_module.perf_counter() - t0)
    if tran is None:
        raise RuntimeError(f"NGSpice transient failed for corner={corner}, temp={temperature_c}")
    v_out_aligned = np.interp(time_np, tran.time, tran.variables[_SPICE_OUTPUT_NODE]).astype(np.float64)
    logger.info("SPICE %s done in %.0f ms", label, wall_ms)
    return _CornerResult(label, corner, temperature_c, v_out_aligned, wall_ms)


def _slew_rate_v_per_us(time_s: np.ndarray, v_out: np.ndarray, t_step_start: float) -> float:
    """Peak |dV/dt| after step onset, in V/µs."""
    mask = time_s >= t_step_start
    v_post = v_out[mask]
    t_post = time_s[mask]
    if v_post.size < 2:
        return float("nan")
    dv_dt = np.diff(v_post) / np.diff(t_post)
    return float(np.max(np.abs(dv_dt)) * 1e-6)


def _run_fno_at_design_point(  # pylint: disable=too-many-locals
    problem: OtaSizingProblem,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build FNO devices, run composition, return ``(time, v_out, wall_ms)``."""
    logger.info("Loading FNO devices at production sizing…")
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
    tg = torch.tensor(time_np, dtype=torch.float32, device=problem.torch_device)
    vinp_t = torch.tensor(vinp_np, dtype=torch.float32, device=problem.torch_device)
    vinn_t = torch.tensor(vinn_np, dtype=torch.float32, device=problem.torch_device)
    theta = torch.tensor(
        [_W_DIFF, _W_MIRROR, _W_TAIL, _L_UM, _V_BIAS],
        dtype=torch.float32,
        device=problem.torch_device,
    )
    t0 = time_module.perf_counter()
    v_out_t, _ = compose_ota_differentiable(theta, problem, base_devices, nfet_strat, pfet_strat, tg, vinp_t, vinn_t)
    wall_ms = 1000.0 * (time_module.perf_counter() - t0)
    logger.info("FNO composition done in %.0f ms", wall_ms)
    return time_np.astype(np.float64), v_out_t.detach().cpu().numpy().astype(np.float64), wall_ms


def _plot(  # pylint: disable=too-many-locals
    time_s: np.ndarray,
    v_fno: np.ndarray,
    spice_tt: _CornerResult,
    spice_ff: _CornerResult,
    out_path: Path,
) -> None:
    """Overlay V_out(t) for FNO and the two SPICE corners."""
    palette = get_palette(dark=False)
    time_us = time_s * 1e6
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(time_us, spice_tt.v_out_aligned, color=palette["gt"], lw=1.6, label=f"SPICE {spice_tt.label}")
    ax.plot(
        time_us,
        spice_ff.v_out_aligned,
        color=palette["pred_sweep"],
        lw=1.6,
        ls="--",
        label=f"SPICE {spice_ff.label}",
    )
    ax.plot(time_us, v_fno, color=palette["pred"], lw=1.4, ls=":", label="FNO composition (tt-trained)")
    ax.set_xlabel("time (µs)")
    ax.set_ylabel(r"$V_\mathrm{out}$ (V)")
    ax.set_title(r"5T OTA off-corner probe — $V_\mathrm{out}(t)$ at production sizing")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/off_corner/ota_ff_125c"),
    show_default=True,
)
@click.option("--device", type=str, default=None, help="Torch device (default: cuda if available).")
def main(output_dir: Path, device: str | None) -> None:
    """Run the off-corner transferability probe and write summary + figure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    problem = OtaSizingProblem(torch_device=torch_dev)

    time_np, v_fno, fno_wall_ms = _run_fno_at_design_point(problem)
    spice_tt = _spice_at_corner(
        problem=problem,
        time_np=time_np,
        corner="tt",
        temperature_c=None,
        label=_TT_LABEL,
    )
    spice_ff = _spice_at_corner(
        problem=problem,
        time_np=time_np,
        corner="ff",
        temperature_c=125.0,
        label=_FF_LABEL,
    )

    metrics = {
        "tt_vs_fno": {
            "pearson_r": pearson_r(spice_tt.v_out_aligned, v_fno),
            "max_abs_delta_v": max_abs_delta_v(spice_tt.v_out_aligned, v_fno),
            "spice_slew_v_per_us": _slew_rate_v_per_us(time_np, spice_tt.v_out_aligned, problem.t_step_start),
            "spice_wall_ms": spice_tt.wall_ms,
        },
        "ff_vs_fno": {
            "pearson_r": pearson_r(spice_ff.v_out_aligned, v_fno),
            "max_abs_delta_v": max_abs_delta_v(spice_ff.v_out_aligned, v_fno),
            "spice_slew_v_per_us": _slew_rate_v_per_us(time_np, spice_ff.v_out_aligned, problem.t_step_start),
            "spice_wall_ms": spice_ff.wall_ms,
        },
        "fno_slew_v_per_us": _slew_rate_v_per_us(time_np, v_fno, problem.t_step_start),
        "fno_wall_ms": fno_wall_ms,
        "design_point": {
            "w_diff_um": _W_DIFF,
            "w_mirror_um": _W_MIRROR,
            "w_tail_um": _W_TAIL,
            "l_um": _L_UM,
            "vbias_v": _V_BIAS,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot(time_np, v_fno, spice_tt, spice_ff, output_dir / "v_out_overlay.png")

    logger.info("Off-corner probe complete:")
    logger.info(
        "  TT @ 27°C  vs FNO: r=%.5f  max|ΔV|=%.2f mV",
        metrics["tt_vs_fno"]["pearson_r"],
        1e3 * metrics["tt_vs_fno"]["max_abs_delta_v"],
    )
    logger.info(
        "  FF @ 125°C vs FNO: r=%.5f  max|ΔV|=%.2f mV",
        metrics["ff_vs_fno"]["pearson_r"],
        1e3 * metrics["ff_vs_fno"]["max_abs_delta_v"],
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
