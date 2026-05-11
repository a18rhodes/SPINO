"""
CLI entrypoint for the FNO-composed 5T OTA validation harness.

Loads the five OTA device FNOs, runs :class:`OtaDcSolver` to find the
quiescent operating point, then runs :class:`OtaTransientSolver` and NGSpice
on the same large-signal differential-step stimulus. Outputs:

* ``step_response_overlay.png`` — FNO vs SPICE ``n_out`` transient overlay.
* ``convergence.png``           — DC and transient residual-norm histories.
* ``summary.json``              — per-metric record (pre-registered acceptance
  gates listed in ``docs/ota_composition.md``).

Invocation::

    python -m spino.circuit.compose_ota \\
        --diff-w 2.0 --mirror-w 2.0 \\
        --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40 \\
        --output-dir docs/assets/ota_5t_fno_l040
"""

from __future__ import annotations

# pylint: disable=wrong-import-position,too-many-arguments,too-many-locals,too-many-positional-arguments

import json
import logging
import time as time_module
from dataclasses import asdict
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spino.circuit.composition import ConvergenceReport  # noqa: E402
from spino.circuit.composition_io import (  # noqa: E402
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    load_ota_5t_devices,
)
from spino.circuit.ota_composition import OtaDcSolution, OtaDcSolver, OtaTransientSolver  # noqa: E402
from spino.circuit.simulation import TransientResult, run_transient  # noqa: E402
from spino.circuit.topologies import build_ota_5t  # noqa: E402
from spino.circuit.tuning import (  # noqa: E402
    _ota_differential_step_pwl_strings,
    extract_slew_rate,
    extract_slew_time,
)

__all__ = ["main"]

logger = logging.getLogger(__name__)

_DEFAULT_DIFF_W = 2.0
_DEFAULT_MIRROR_W = 2.0
_DEFAULT_TAIL_W = 2.0
_DEFAULT_NFET_L = 0.40
_DEFAULT_PFET_L = 0.40
_DEFAULT_TAIL_L = 0.40
_DEFAULT_VDD = 1.8
_DEFAULT_VCM = 0.9
_DEFAULT_STEP_AMP = 0.25
_DEFAULT_VBIAS = 1.2
_DEFAULT_RISE_TIME = 5e-9
_DEFAULT_T_STEP_START = 100e-9
_DEFAULT_T_END = 5e-6
_DEFAULT_T_STEP = 10e-9
_SPICE_OUTPUT_NODE = "v(n_out)"
_FNO_OUTPUT_NODE = "v(n_out)"


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def _build_input_trajectories(
    time_s: np.ndarray,
    *,
    vcm_v: float,
    t_step_start: float,
    rise_time_s: float,
    step_amp_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Samples the differential step stimulus on a numpy time grid.

    :param time_s: Time points in seconds ``(T,)``.
    :param vcm_v: Common-mode voltage (V).
    :param t_step_start: Step-onset time (s).
    :param rise_time_s: PWL rise time (s).
    :param step_amp_v: Half-amplitude of the differential step (V).
    :return: Tuple ``(vinp, vinn)`` sampled on ``time_s``.
    """
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


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_true - y_pred)))


def _relative_error(ref: float, pred: float) -> float:
    denom = abs(ref) if abs(ref) > 1e-30 else 1e-30
    return abs(pred - ref) / denom


# ---------------------------------------------------------------------------
# SPICE runner
# ---------------------------------------------------------------------------


def _run_spice_transient(
    *,
    diff_w_um: float,
    diff_l_um: float,
    mirror_w_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    vdd: float,
    vcm_v: float,
    vbias_v: float,
    step_amp_v: float,
    rise_time_s: float,
    t_step_start: float,
    t_end: float,
    t_step: float,
    pdk_root: str | None,
) -> tuple[TransientResult, float]:
    """
    Runs the NGSpice reference transient for the selected OTA design.

    :return: Tuple ``(transient_result, wall_ms)``.
    :raises RuntimeError: If NGSpice does not converge.
    """
    vinp_pwl, vinn_pwl = _ota_differential_step_pwl_strings(
        vcm_v=vcm_v,
        t_step_start=t_step_start,
        rise_time_s=rise_time_s,
        t_end=t_end,
        step_amp_v=step_amp_v,
    )
    kw = {
        "diff_w_um": diff_w_um,
        "diff_l_um": diff_l_um,
        "mirror_w_um": mirror_w_um,
        "mirror_l_um": mirror_l_um,
        "tail_w_um": tail_w_um,
        "tail_l_um": tail_l_um,
        "vdd": vdd,
        "vbias_v": vbias_v,
        "vcm_v": vcm_v,
        "vinp_tran": vinp_pwl,
        "vinn_tran": vinn_pwl,
    }
    if pdk_root is not None:
        kw["pdk_root"] = pdk_root
    circuit = build_ota_5t(**kw)
    t0 = time_module.perf_counter()
    tran = run_transient(circuit, t_step=t_step, t_end=t_end)
    wall_ms = 1000.0 * (time_module.perf_counter() - t0)
    if tran is None:
        raise RuntimeError("NGSpice .tran failed for OTA reference transient")
    return tran, wall_ms


# ---------------------------------------------------------------------------
# FNO runs
# ---------------------------------------------------------------------------


def _run_fno_dc(solver: OtaDcSolver) -> tuple[dict, OtaDcSolution]:
    """Runs the DC solver and returns a serialisable report dict plus the raw solution."""
    sol = solver.solve()
    report = {
        "v_tail_v": sol.v_tail_v,
        "v_left_v": sol.v_left_v,
        "v_out_v": sol.v_out_v,
        "report": _serialise_report(sol.report),
    }
    return report, sol


def _run_fno_transient(
    solver: OtaTransientSolver,
    v_dc: torch.Tensor,
    *,
    time_s: np.ndarray,
    vinp_np: np.ndarray,
    vinn_np: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Runs the transient solver; returns report dict, time array, and v_out array."""
    time_tensor = torch.from_numpy(time_s)
    vinp_tensor = torch.from_numpy(vinp_np)
    vinn_tensor = torch.from_numpy(vinn_np)
    sol = solver.solve(time_tensor, vinp_tensor, vinn_tensor, v_dc)
    time_np = sol.time_s.cpu().numpy()
    v_out_np = sol.v_out_v.cpu().numpy()
    report = {"report": _serialise_report(sol.report)}
    return report, time_np, v_out_np


# ---------------------------------------------------------------------------
# Metrics and figures
# ---------------------------------------------------------------------------


def _extract_ota_metrics(
    time_np: np.ndarray,
    v_out_np: np.ndarray,
    *,
    t_step_start: float,
) -> dict:
    """Extracts slew rate and 10-90% slew time from a v_out trajectory."""
    fake_tran = TransientResult(time=time_np, variables={_FNO_OUTPUT_NODE: v_out_np}, iter_count=None)
    slew_rate = extract_slew_rate(fake_tran, t_step_start=t_step_start, output_node=_FNO_OUTPUT_NODE)
    slew_time = extract_slew_time(fake_tran, t_step_start=t_step_start, output_node=_FNO_OUTPUT_NODE)
    return {
        "slew_rate_v_per_us": float(slew_rate),
        "slew_time_ns": float(slew_time),
    }


def _serialise_report(report: ConvergenceReport) -> dict:
    record = asdict(report)
    record["residual_norm_history"] = list(report.residual_norm_history)
    return record


def _plot_step_response_overlay(
    time_np: np.ndarray,
    fno_v_out: np.ndarray,
    spice_tran: TransientResult,
    output_path: Path,
    *,
    t_step_start: float,
    pearson_r: float,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(
        spice_tran.time * 1e6,
        spice_tran.variables[_SPICE_OUTPUT_NODE],
        color="#0066cc",
        linewidth=1.4,
        label="SPICE",
    )
    ax.plot(time_np * 1e6, fno_v_out, color="#cc6600", linewidth=1.2, linestyle="--", label="FNO")
    ax.axvline(t_step_start * 1e6, color="#444444", linestyle=":", alpha=0.6, label="Step onset")
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(f"OTA step response overlay (Pearson r={pearson_r:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_convergence(
    dc_history: tuple[float, ...],
    tran_history: tuple[float, ...],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if dc_history:
        ax.semilogy(range(len(dc_history)), dc_history, marker="o", label="DC (|R|, A)")
    if tran_history:
        ax.semilogy(range(len(tran_history)), tran_history, marker="s", label=r"Transient ($\|R\|_\infty$, A)")
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Residual norm (A)")
    ax.set_title("OTA Newton–Raphson convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docs/assets/ota_5t_fno_l040"),
    show_default=True,
    help="Directory for figures and summary.json.",
)
@click.option("--diff-w", type=float, default=_DEFAULT_DIFF_W, show_default=True, help="Diff-pair width (µm).")
@click.option("--mirror-w", type=float, default=_DEFAULT_MIRROR_W, show_default=True, help="Mirror width (µm).")
@click.option("--tail-w", type=float, default=_DEFAULT_TAIL_W, show_default=True, help="Tail width (µm).")
@click.option("--nfet-l", type=float, default=_DEFAULT_NFET_L, show_default=True, help="Diff-pair length (µm).")
@click.option("--pfet-l", type=float, default=_DEFAULT_PFET_L, show_default=True, help="Mirror length (µm).")
@click.option("--tail-l", type=float, default=_DEFAULT_TAIL_L, show_default=True, help="Tail length (µm).")
@click.option("--vdd", type=float, default=_DEFAULT_VDD, show_default=True, help="Supply voltage (V).")
@click.option("--vcm", type=float, default=_DEFAULT_VCM, show_default=True, help="Common-mode voltage (V).")
@click.option("--vbias", type=float, default=_DEFAULT_VBIAS, show_default=True, help="Tail gate bias (V).")
@click.option("--step-amp", type=float, default=_DEFAULT_STEP_AMP, show_default=True, help="Diff step amplitude (V).")
@click.option("--t-step-start", type=float, default=_DEFAULT_T_STEP_START, show_default=True, help="Step onset (s).")
@click.option("--t-end", type=float, default=_DEFAULT_T_END, show_default=True, help="Simulation window (s).")
@click.option("--t-step", type=float, default=_DEFAULT_T_STEP, show_default=True, help="Max timestep (s).")
@click.option("--device", type=str, default="cpu", show_default=True, help="Torch device (e.g. cuda).")
@click.option("--pdk-root", type=str, default=None, help="Override Sky130 PDK root.")
@click.option("--nfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_NFET_CHECKPOINT, show_default=True)
@click.option("--pfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_PFET_CHECKPOINT, show_default=True)
@click.option("--nfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_NFET_DATASET, show_default=True)
@click.option("--pfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_PFET_DATASET, show_default=True)
def main(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    output_dir: Path,
    diff_w: float,
    mirror_w: float,
    tail_w: float,
    nfet_l: float,
    pfet_l: float,
    tail_l: float,
    vdd: float,
    vcm: float,
    vbias: float,
    step_amp: float,
    t_step_start: float,
    t_end: float,
    t_step: float,
    device: str,
    pdk_root: str | None,
    nfet_checkpoint: Path,
    pfet_checkpoint: Path,
    nfet_dataset: Path,
    pfet_dataset: Path,
) -> None:
    """FNO-composed 5T OTA validation: DC OP → transient → SPICE comparison."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Loading OTA FNO devices (diff=%.2f/%.2f µm, mirror=%.2f/%.2f µm, tail=%.2f/%.2f µm)…",
        diff_w,
        nfet_l,
        mirror_w,
        pfet_l,
        tail_w,
        tail_l,
    )
    m1, m2, m3, m4, m5 = load_ota_5t_devices(
        diff_w_um=diff_w,
        diff_l_um=nfet_l,
        mirror_w_um=mirror_w,
        mirror_l_um=pfet_l,
        tail_w_um=tail_w,
        tail_l_um=tail_l,
        nfet_checkpoint=nfet_checkpoint,
        pfet_checkpoint=pfet_checkpoint,
        nfet_dataset=nfet_dataset,
        pfet_dataset=pfet_dataset,
        map_location=device,
    )

    logger.info("Running FNO DC solver…")
    dc_solver = OtaDcSolver(m1, m2, m3, m4, m5, vdd=vdd, vcm_v=vcm, vbias_v=vbias)
    dc_report, dc_sol = _run_fno_dc(dc_solver)
    v_dc = torch.tensor([dc_sol.v_tail_v, dc_sol.v_left_v, dc_sol.v_out_v], dtype=torch.float32)
    logger.info(
        "DC OP: v_tail=%.3f V  v_left=%.3f V  v_out=%.3f V  converged=%s",
        dc_sol.v_tail_v,
        dc_sol.v_left_v,
        dc_sol.v_out_v,
        dc_sol.report.converged,
    )

    time_np = np.arange(0.0, t_end, t_step, dtype=np.float32)
    vinp_np, vinn_np = _build_input_trajectories(
        time_np, vcm_v=vcm, t_step_start=t_step_start, rise_time_s=_DEFAULT_RISE_TIME, step_amp_v=step_amp
    )

    logger.info("Running FNO transient solver (T=%d timesteps)…", len(time_np))
    tran_solver = OtaTransientSolver(m1, m2, m3, m4, m5, vdd=vdd, vbias_v=vbias)
    fno_tran_report, fno_time_np, fno_v_out_np = _run_fno_transient(
        tran_solver, v_dc, time_s=time_np, vinp_np=vinp_np, vinn_np=vinn_np
    )
    fno_metrics = _extract_ota_metrics(fno_time_np, fno_v_out_np, t_step_start=t_step_start)
    logger.info(
        "FNO transient: converged=%s  slew=%.2f V/µs  t_slew=%.0f ns",
        fno_tran_report["report"]["converged"],
        fno_metrics["slew_rate_v_per_us"],
        fno_metrics["slew_time_ns"],
    )

    logger.info("Running NGSpice reference transient…")
    spice_tran, spice_wall_ms = _run_spice_transient(
        diff_w_um=diff_w,
        diff_l_um=nfet_l,
        mirror_w_um=mirror_w,
        mirror_l_um=pfet_l,
        tail_w_um=tail_w,
        tail_l_um=tail_l,
        vdd=vdd,
        vcm_v=vcm,
        vbias_v=vbias,
        step_amp_v=step_amp,
        rise_time_s=_DEFAULT_RISE_TIME,
        t_step_start=t_step_start,
        t_end=t_end,
        t_step=t_step,
        pdk_root=pdk_root,
    )
    spice_metrics = _extract_ota_metrics(
        spice_tran.time,
        spice_tran.variables[_SPICE_OUTPUT_NODE],
        t_step_start=t_step_start,
    )
    logger.info(
        "SPICE transient: slew=%.2f V/µs  t_slew=%.0f ns  wall=%.1f ms",
        spice_metrics["slew_rate_v_per_us"],
        spice_metrics["slew_time_ns"],
        spice_wall_ms,
    )

    spice_v_out_resampled = np.interp(fno_time_np, spice_tran.time, spice_tran.variables[_SPICE_OUTPUT_NODE])
    pearson = _pearson_r(spice_v_out_resampled, fno_v_out_np)
    max_err = _max_abs_error(spice_v_out_resampled, fno_v_out_np)
    slew_rel_err = _relative_error(spice_metrics["slew_rate_v_per_us"], fno_metrics["slew_rate_v_per_us"])
    slew_time_rel_err = _relative_error(spice_metrics["slew_time_ns"], fno_metrics["slew_time_ns"])
    logger.info(
        "Comparison: Pearson r=%.4f  max|ΔV|=%.1f mV  slew_rel_err=%.1f%%  slew_time_rel_err=%.1f%%",
        pearson,
        max_err * 1e3,
        slew_rel_err * 100,
        slew_time_rel_err * 100,
    )

    _plot_step_response_overlay(
        fno_time_np,
        fno_v_out_np,
        spice_tran,
        output_dir / "step_response_overlay.png",
        t_step_start=t_step_start,
        pearson_r=pearson,
    )
    _plot_convergence(
        dc_sol.report.residual_norm_history,
        fno_tran_report["report"]["residual_norm_history"],
        output_dir / "convergence.png",
    )

    config = {
        "topology": "ota_5t",
        "diff_w_um": diff_w,
        "diff_l_um": nfet_l,
        "mirror_w_um": mirror_w,
        "mirror_l_um": pfet_l,
        "tail_w_um": tail_w,
        "tail_l_um": tail_l,
        "vdd_v": vdd,
        "vcm_v": vcm,
        "vbias_v": vbias,
        "step_amp_v": step_amp,
        "t_step_start_s": t_step_start,
        "t_end_s": t_end,
        "t_step_s": t_step,
        "device": device,
    }
    summary = {
        "config": config,
        "dc_op": dc_report,
        "transient": {
            "fno": {**fno_tran_report, **fno_metrics},
            "spice": {
                "iter_count": spice_tran.iter_count,
                "wall_ms": spice_wall_ms,
                **spice_metrics,
            },
            "pearson_r": pearson,
            "max_abs_error_v": max_err,
            "slew_rate_rel_err": slew_rel_err,
            "slew_time_rel_err": slew_time_rel_err,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote artefacts to %s", output_dir.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
