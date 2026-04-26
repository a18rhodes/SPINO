"""
CLI entrypoint for CS amplifier sizing characterization.

Drives a 2D :math:`(W_n, W_p)` sweep against NGSpice, applies the published
selection rule, regenerates the documentation figures, and emits a JSON
summary suitable for archival in the project repository.

Invocation::

    python -m spino.circuit.characterize \\
        --output-dir docs/assets/cs_amp \\
        --summary-path docs/assets/cs_amp/summary.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np

from spino.circuit.plotting import plot_gain_heatmap, plot_step_response, plot_vtc
from spino.circuit.simulation import run_dc_sweep, run_transient
from spino.circuit.topologies import build_cs_amp_active_load
from spino.circuit.tuning import (
    DesignPoint,
    Metrics,
    SelectionRule,
    SweepResult,
    extract_settling_time,
    select_design_point,
    sweep_design_space,
)

__all__ = ["main"]

logger = logging.getLogger(__name__)

_DEFAULT_NFET_WIDTHS = (0.36, 0.6, 1.0, 1.6, 2.5, 4.0, 6.0)
_DEFAULT_PFET_WIDTHS = (0.36, 0.7, 1.4, 2.5, 4.5, 8.0, 14.0)
_DEFAULT_VDD = 1.8
_DEFAULT_NFET_L = 0.18
_DEFAULT_PFET_L = 0.18
_DEFAULT_VTC_STEP = 0.01
_DEFAULT_VIN_STEP_AMP = 0.05
_DEFAULT_T_STEP_START = 100e-9
_DEFAULT_T_END = 5e-6
_DEFAULT_T_STEP = 10e-9
_FIGURE_C_LOAD_F = 10e-12
_FIGURE_T_WINDOW_S = 200e-9


def _log_progress(idx: int, total: int, point: DesignPoint, metrics: Metrics) -> None:
    """
    Emits a one-line status update for each completed design point.

    :param idx: Zero-based index of the current point.
    :param total: Total number of points in the sweep.
    :param point: The design point that just finished.
    :param metrics: Metrics returned for this point.
    """
    tag = "OK" if metrics.converged else "FAIL"
    logger.info(
        "[%3d/%d] %s W_n=%.3f W_p=%.3f gain=%.3f Vbias=%.3f I_dd=%.3e A",
        idx + 1,
        total,
        tag,
        point.nfet_w_um,
        point.pfet_w_um,
        metrics.peak_gain_v_per_v,
        metrics.vout_at_peak_gain_v,
        metrics.static_current_a,
    )


def _serialise_sweep(sweep: SweepResult) -> list[dict]:
    """
    Converts the sweep into a JSON-friendly list of records.

    :param sweep: Completed sweep result.
    :return: One dictionary per design point with all metric fields.
    """
    return [{"point": asdict(point), "metrics": asdict(metrics)} for point, metrics in zip(sweep.points, sweep.metrics)]


def _write_summary(
    summary_path: Path,
    sweep: SweepResult,
    selected_point: DesignPoint,
    selected_metrics: Metrics,
    figure_metrics: Metrics,
    rule: SelectionRule,
    config: dict,
) -> None:
    """
    Writes the full sweep + selection record to a JSON file.

    Two metrics blocks are emitted for the selected design: ``metrics`` are the
    no-load values from the ranking sweep, and ``figure_metrics`` are recomputed
    on the loaded transient that backs the published step-response figure.

    :param summary_path: Destination path for the summary JSON.
    :param sweep: Completed sweep result.
    :param selected_point: The chosen design point.
    :param selected_metrics: Sweep metrics for the chosen design (no load).
    :param figure_metrics: Metrics recomputed on the figure-only loaded transient.
    :param rule: Selection rule used.
    :param config: Sweep configuration dictionary.
    """
    summary = {
        "config": config,
        "selection_rule": asdict(rule),
        "selected": {
            "point": asdict(selected_point),
            "metrics": asdict(selected_metrics),
            "figure_metrics": asdict(figure_metrics),
        },
        "sweep": _serialise_sweep(sweep),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _generate_design_plots(
    selected_point: DesignPoint,
    selected_metrics: Metrics,
    output_dir: Path,
    config: dict,
) -> Metrics:
    """
    Re-runs SPICE on the selected design and writes the VTC and step plots.

    The transient is re-run with the configured load capacitance so the
    published step-response figure exhibits a meaningful settling time. The
    loaded settling time is returned as part of a fresh metrics record so the
    summary JSON can document both the no-load ranking metrics and the
    figure-only loaded metrics without ambiguity.

    Re-running here rather than caching the original traces keeps the plot
    artefacts trivially reproducible from the summary JSON alone.

    :param selected_point: The chosen design point.
    :param selected_metrics: No-load sweep metrics for the chosen design.
    :param output_dir: Directory where PNG figures are written.
    :param config: Sweep configuration dictionary.
    :return: Metrics recomputed on the loaded transient (only the settling time
        differs from the no-load sweep metrics).
    """
    base_kwargs = {
        "nfet_w": selected_point.nfet_w_um,
        "nfet_l": config["nfet_l_um"],
        "pfet_w": selected_point.pfet_w_um,
        "pfet_l": config["pfet_l_um"],
        "vdd": config["vdd"],
    }
    vtc_circuit = build_cs_amp_active_load(**base_kwargs, vin_dc=config["vdd"] / 2.0)
    vtc = run_dc_sweep(vtc_circuit, source_name="Vin", start=0.0, stop=config["vdd"], step=config["vtc_step_v"])
    if vtc is None:
        raise RuntimeError("VTC re-simulation failed for the selected design")
    plot_vtc(vtc, output_dir / "vtc.png", design=selected_point, metrics=selected_metrics)
    pwl = (
        f"PWL(0 {selected_metrics.vin_at_peak_gain_v} "
        f"{config['t_step_start'] * 0.5} {selected_metrics.vin_at_peak_gain_v} "
        f"{config['t_step_start']} {selected_metrics.vin_at_peak_gain_v + config['vin_step_amplitude']} "
        f"{config['t_end']} {selected_metrics.vin_at_peak_gain_v + config['vin_step_amplitude']})"
    )
    tran_circuit = build_cs_amp_active_load(
        **base_kwargs,
        vin_dc=selected_metrics.vin_at_peak_gain_v,
        vin_tran=pwl,
        c_load_f=config["figure_c_load_f"],
    )
    tran = run_transient(tran_circuit, t_step=config["t_step"], t_end=config["t_end"])
    if tran is None:
        raise RuntimeError("Transient re-simulation failed for the selected design")
    figure_settling = extract_settling_time(tran, t_step_start=config["t_step_start"])
    figure_metrics = Metrics(
        converged=True,
        peak_gain_v_per_v=selected_metrics.peak_gain_v_per_v,
        vin_at_peak_gain_v=selected_metrics.vin_at_peak_gain_v,
        vout_at_peak_gain_v=selected_metrics.vout_at_peak_gain_v,
        static_current_a=selected_metrics.static_current_a,
        settling_time_s=figure_settling,
    )
    plot_step_response(
        tran,
        output_dir / "step_response.png",
        design=selected_point,
        metrics=figure_metrics,
        t_step_start=config["t_step_start"],
        t_window_s=config["figure_t_window_s"],
    )
    return figure_metrics


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docs/assets/cs_amp"),
    show_default=True,
    help="Directory where figures and summary JSON are written.",
)
@click.option("--vout-min", type=float, default=0.6, show_default=True, help="Selection rule lower bound (V).")
@click.option("--vout-max", type=float, default=1.2, show_default=True, help="Selection rule upper bound (V).")
@click.option("--vdd", type=float, default=_DEFAULT_VDD, show_default=True, help="Supply voltage in volts.")
@click.option("--pdk-root", type=str, default=None, help="Override Sky130 PDK root.")
def main(output_dir: Path, vout_min: float, vout_max: float, vdd: float, pdk_root: str | None) -> None:
    """Runs the CS amplifier characterization sweep and writes artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = {
        "nfet_widths_um": list(_DEFAULT_NFET_WIDTHS),
        "pfet_widths_um": list(_DEFAULT_PFET_WIDTHS),
        "vdd": vdd,
        "nfet_l_um": _DEFAULT_NFET_L,
        "pfet_l_um": _DEFAULT_PFET_L,
        "vtc_step_v": _DEFAULT_VTC_STEP,
        "vin_step_amplitude": _DEFAULT_VIN_STEP_AMP,
        "t_step_start": _DEFAULT_T_STEP_START,
        "t_end": _DEFAULT_T_END,
        "t_step": _DEFAULT_T_STEP,
        "figure_c_load_f": _FIGURE_C_LOAD_F,
        "figure_t_window_s": _FIGURE_T_WINDOW_S,
    }
    rule = SelectionRule(vout_min_v=vout_min, vout_max_v=vout_max)
    logger.info(
        "Sweeping %d x %d = %d design points...",
        len(_DEFAULT_NFET_WIDTHS),
        len(_DEFAULT_PFET_WIDTHS),
        len(_DEFAULT_NFET_WIDTHS) * len(_DEFAULT_PFET_WIDTHS),
    )
    sweep = sweep_design_space(
        _DEFAULT_NFET_WIDTHS,
        _DEFAULT_PFET_WIDTHS,
        vdd=vdd,
        nfet_l_um=_DEFAULT_NFET_L,
        pfet_l_um=_DEFAULT_PFET_L,
        pdk_root=pdk_root,
        progress=_log_progress,
    )
    converged = int(np.sum([m.converged for m in sweep.metrics]))
    logger.info("Sweep complete: %d/%d converged.", converged, len(sweep.metrics))
    selected_point, selected_metrics = select_design_point(sweep, rule)
    logger.info(
        "Selected: W_n=%.3f µm, W_p=%.3f µm  |  peak |gain|=%.3f V/V, V_bias=%.3f V, I_dd=%.3e A, t_settle=%.3e s",
        selected_point.nfet_w_um,
        selected_point.pfet_w_um,
        selected_metrics.peak_gain_v_per_v,
        selected_metrics.vout_at_peak_gain_v,
        selected_metrics.static_current_a,
        selected_metrics.settling_time_s,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gain_heatmap(sweep, output_dir / "gain_heatmap.png", selected=selected_point)
    figure_metrics = _generate_design_plots(selected_point, selected_metrics, output_dir, config)
    logger.info(
        "Loaded settling time (C_load=%.0e F): %.3e s",
        config["figure_c_load_f"],
        figure_metrics.settling_time_s,
    )
    _write_summary(
        output_dir / "summary.json",
        sweep,
        selected_point,
        selected_metrics,
        figure_metrics,
        rule,
        config,
    )
    logger.info("Wrote %s", output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
