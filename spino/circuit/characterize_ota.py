"""
CLI entrypoint for 5T OTA sizing characterization (Phase 1).

Drives a 2D :math:`(W_{diff}, W_{mirror})` sweep against NGSpice at fixed
channel lengths, applies the pre-registered selection rule, emits summary JSON
and publication figures, and prints the selected design to stdout.

Invocation::

    python -m spino.circuit.characterize_ota \\
        --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40 \\
        --output-dir docs/assets/ota_5t_l040

The differential-pair and mirror widths are fixed to the pre-registered
:data:`_DEFAULT_DIFF_WIDTHS` and :data:`_DEFAULT_MIRROR_WIDTHS` grids.
Tail sizing (``--tail-w``, ``--vbias``) and optional PDK root are CLI inputs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np

from spino.circuit.plotting import (
    plot_slew_heatmap,
    plot_slew_time_heatmap,
    plot_step_response_ota,
)
from spino.circuit.simulation import run_transient
from spino.circuit.topologies import build_ota_5t
from spino.circuit.tuning import (
    OtaDesignPoint,
    OtaMetrics,
    OtaSelectionRule,
    OtaSweepResult,
    _ota_differential_step_pwl_strings,
    select_ota_design_point,
    sweep_ota_design_space,
)

__all__ = ["main"]

logger = logging.getLogger(__name__)

# Pre-registered grids (locked at Phase 0.5, commit c3b199f)
_DEFAULT_DIFF_WIDTHS: tuple[float, ...] = (0.5, 0.8, 1.3, 2.0, 3.2, 5.0, 8.0)
_DEFAULT_MIRROR_WIDTHS: tuple[float, ...] = (0.5, 0.8, 1.3, 2.0, 3.2, 5.0, 8.0)

_DEFAULT_VDD = 1.8
_DEFAULT_VCM = 0.9
_DEFAULT_STEP_AMP = 0.05  # ±50 mV differential (pre-registered)
_DEFAULT_RISE_TIME = 5e-9  # 5 ns (pre-registered)
_DEFAULT_NFET_L = 0.40  # diff-pair L
_DEFAULT_PFET_L = 0.40  # mirror L
_DEFAULT_TAIL_L = 0.40  # tail L
_DEFAULT_TAIL_W = 2.0  # tail W (µm)
_DEFAULT_VBIAS = 1.2  # tail gate bias (V)
_DEFAULT_T_STEP_START = 100e-9
_DEFAULT_T_END = 500e-9  # 500 ns: covers full slew at CL=1 pF + 100 ns headroom
_DEFAULT_T_STEP = 1e-9  # 1 ns: resolves ~27 ns 10-90% rise at CL=1 pF
_DEFAULT_C_LOAD = 1e-12  # 1 pF: defines slew metric (slew = I_tail / C_load)
_DEFAULT_SLEW_MIN = 5.0  # V/µs (pre-registered gate)
_DEFAULT_SLEW_TIME_MAX = 500.0  # ns (pre-registered gate)
_FIGURE_T_WINDOW_S = 300e-9  # plot zoom window after step onset


def _log_progress(idx: int, total: int, point: OtaDesignPoint, metrics: OtaMetrics) -> None:
    tag = "OK" if metrics.converged else "FAIL"
    logger.info(
        "[%3d/%d] %s Wdiff=%.2f Wmirror=%.2f  slew=%.2f V/µs  t_slew=%.0f ns  Idd=%.3e A",
        idx + 1,
        total,
        tag,
        point.diff_w_um,
        point.mirror_w_um,
        metrics.slew_rate_v_per_us,
        metrics.slew_time_ns,
        metrics.static_current_a,
    )


def _serialise_sweep(sweep: OtaSweepResult) -> list[dict]:
    return [{"point": asdict(point), "metrics": asdict(metrics)} for point, metrics in zip(sweep.points, sweep.metrics)]


def _write_summary(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    summary_path: Path,
    sweep: OtaSweepResult,
    selected_point: OtaDesignPoint,
    selected_metrics: OtaMetrics,
    rule: OtaSelectionRule,
    config: dict,
) -> None:
    summary = {
        "config": config,
        "selection_rule": asdict(rule),
        "selected": {
            "point": asdict(selected_point),
            "metrics": asdict(selected_metrics),
        },
        "sweep": _serialise_sweep(sweep),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _generate_figure(
    selected_point: OtaDesignPoint,
    selected_metrics: OtaMetrics,
    output_dir: Path,
    config: dict,
) -> None:
    """Re-runs the transient for the selected design and writes the step-response figure."""
    rise_time = config["rise_time_s"]
    t_step_start = config["t_step_start"]
    t_end = config["t_end"]
    vcm_v = config["vcm_v"]
    step_amp_v = config["step_amp_v"]
    vinp_pwl, vinn_pwl = _ota_differential_step_pwl_strings(
        vcm_v=vcm_v,
        t_step_start=t_step_start,
        rise_time_s=rise_time,
        t_end=t_end,
        step_amp_v=step_amp_v,
    )
    kwargs = {
        "diff_w_um": selected_point.diff_w_um,
        "diff_l_um": config["diff_l_um"],
        "mirror_w_um": selected_point.mirror_w_um,
        "mirror_l_um": config["mirror_l_um"],
        "tail_w_um": config["tail_w_um"],
        "tail_l_um": config["tail_l_um"],
        "vdd": config["vdd"],
        "vbias_v": config["vbias_v"],
        "vcm_v": vcm_v,
        "vinp_tran": vinp_pwl,
        "vinn_tran": vinn_pwl,
        "c_load_f": config["c_load_f"],
    }
    if config.get("pdk_root") is not None:
        kwargs["pdk_root"] = config["pdk_root"]
    circuit = build_ota_5t(**kwargs)
    tran = run_transient(circuit, t_step=config["t_step"], t_end=t_end)
    if tran is None:
        logger.warning("Step-response figure re-simulation failed for selected design; skipping plot.")
        return
    plot_step_response_ota(
        tran,
        output_dir / "step_response.png",
        design=selected_point,
        metrics=selected_metrics,
        t_step_start=t_step_start,
        t_window_s=_FIGURE_T_WINDOW_S,
    )


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docs/assets/ota_5t_l040"),
    show_default=True,
    help="Directory where figures and summary JSON are written.",
)
@click.option("--vdd", type=float, default=_DEFAULT_VDD, show_default=True, help="Supply voltage (V).")
@click.option(
    "--nfet-l",
    type=float,
    default=_DEFAULT_NFET_L,
    show_default=True,
    help="Differential-pair channel length (µm).",
)
@click.option(
    "--pfet-l",
    type=float,
    default=_DEFAULT_PFET_L,
    show_default=True,
    help="Current-mirror channel length (µm).",
)
@click.option(
    "--tail-l",
    type=float,
    default=_DEFAULT_TAIL_L,
    show_default=True,
    help="Tail current-source channel length (µm).",
)
@click.option(
    "--tail-w",
    type=float,
    default=_DEFAULT_TAIL_W,
    show_default=True,
    help="Tail current-source width (µm).",
)
@click.option(
    "--vbias",
    type=float,
    default=_DEFAULT_VBIAS,
    show_default=True,
    help="Tail gate bias voltage (V).",
)
@click.option("--vcm", type=float, default=_DEFAULT_VCM, show_default=True, help="Input common-mode voltage (V).")
@click.option(
    "--step-amp",
    type=float,
    default=_DEFAULT_STEP_AMP,
    show_default=True,
    help="Differential step half-amplitude: each input steps ±this value (V).",
)
@click.option(
    "--t-step-start",
    type=float,
    default=_DEFAULT_T_STEP_START,
    show_default=True,
    help="Step onset time (s).",
)
@click.option("--t-end", type=float, default=_DEFAULT_T_END, show_default=True, help="Simulation window (s).")
@click.option("--t-step", type=float, default=_DEFAULT_T_STEP, show_default=True, help="SPICE max timestep (s).")
@click.option(
    "--slew-min",
    type=float,
    default=_DEFAULT_SLEW_MIN,
    show_default=True,
    help="Selection rule: minimum slew rate (V/µs).",
)
@click.option(
    "--slew-time-max",
    type=float,
    default=_DEFAULT_SLEW_TIME_MAX,
    show_default=True,
    help="Selection rule: maximum 10–90 %% slew time (ns).",
)
@click.option(
    "--c-load",
    type=float,
    default=_DEFAULT_C_LOAD,
    show_default=True,
    help=(
        "Load capacitance at n_out (F). Defines the slew metric: "
        "slew_rate = I_tail / c_load. Must match the value used in compose_ota."
    ),
)
@click.option("--pdk-root", type=str, default=None, help="Override Sky130 PDK root.")
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    output_dir: Path,
    vdd: float,
    nfet_l: float,
    pfet_l: float,
    tail_l: float,
    tail_w: float,
    vbias: float,
    vcm: float,
    step_amp: float,
    t_step_start: float,
    t_end: float,
    t_step: float,
    slew_min: float,
    slew_time_max: float,
    c_load: float,
    pdk_root: str | None,
) -> None:
    """Runs the 5T OTA characterization sweep and writes artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config: dict = {
        "diff_widths_um": list(_DEFAULT_DIFF_WIDTHS),
        "mirror_widths_um": list(_DEFAULT_MIRROR_WIDTHS),
        "vdd": vdd,
        "diff_l_um": nfet_l,
        "mirror_l_um": pfet_l,
        "tail_l_um": tail_l,
        "tail_w_um": tail_w,
        "vbias_v": vbias,
        "vcm_v": vcm,
        "step_amp_v": step_amp,
        "rise_time_s": _DEFAULT_RISE_TIME,
        "t_step_start": t_step_start,
        "t_end": t_end,
        "t_step": t_step,
        "c_load_f": c_load,
        "pdk_root": pdk_root,
    }
    rule = OtaSelectionRule(slew_min_v_per_us=slew_min, slew_time_max_ns=slew_time_max)

    n_points = len(_DEFAULT_DIFF_WIDTHS) * len(_DEFAULT_MIRROR_WIDTHS)
    logger.info(
        "Sweeping %d × %d = %d OTA design points (nfet_l=%.2f, pfet_l=%.2f, tail_l=%.2f µm, CL=%.0f fF)...",
        len(_DEFAULT_DIFF_WIDTHS),
        len(_DEFAULT_MIRROR_WIDTHS),
        n_points,
        nfet_l,
        pfet_l,
        tail_l,
        c_load * 1e15,
    )

    sweep = sweep_ota_design_space(
        _DEFAULT_DIFF_WIDTHS,
        _DEFAULT_MIRROR_WIDTHS,
        vdd=vdd,
        vcm_v=vcm,
        step_amp_v=step_amp,
        rise_time_s=_DEFAULT_RISE_TIME,
        diff_l_um=nfet_l,
        mirror_l_um=pfet_l,
        tail_w_um=tail_w,
        tail_l_um=tail_l,
        vbias_v=vbias,
        t_step_start=t_step_start,
        t_end=t_end,
        t_step=t_step,
        c_load_f=c_load,
        pdk_root=pdk_root,
        progress=_log_progress,
    )

    converged_count = int(np.sum([m.converged for m in sweep.metrics]))
    logger.info("Sweep complete: %d/%d converged.", converged_count, n_points)

    selected_point, selected_metrics = select_ota_design_point(sweep, rule)
    logger.info(
        "Selected: Wdiff=%.2f µm, Wmirror=%.2f µm  |  "
        "slew=%.2f V/µs  t_slew=%.0f ns  swing=%.3f V  Idd=%.3e A  gain=%.1f V/V",
        selected_point.diff_w_um,
        selected_point.mirror_w_um,
        selected_metrics.slew_rate_v_per_us,
        selected_metrics.slew_time_ns,
        selected_metrics.peak_swing_v,
        selected_metrics.static_current_a,
        selected_metrics.dc_gain_v_per_v,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_slew_heatmap(sweep, output_dir / "slew_rate_heatmap.png", selected=selected_point)
    plot_slew_time_heatmap(sweep, output_dir / "settling_time_heatmap.png", selected=selected_point)
    _generate_figure(selected_point, selected_metrics, output_dir, config)
    _write_summary(output_dir / "summary.json", sweep, selected_point, selected_metrics, rule, config)
    logger.info("Wrote artefacts to %s", output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
