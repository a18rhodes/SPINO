"""
CLI entrypoint for the FNO-composed CS amplifier validation harness.

Loads the production NFET/PFET FNO checkpoints, runs the DC operating point
solver and the whole-window implicit Newton-Raphson transient solver against
the NGSpice reference, and writes:

* ``vtc_overlay.png`` — VTC overlay (FNO vs SPICE).
* ``step_response_overlay.png`` — figure stimulus step response overlay.
* ``convergence.png`` — per-iteration residual norm decay for the DC and
  transient solves.
* ``summary.json`` — per-metric record consumed by ``docs/composition.md``.

Invocation::

    python -m spino.circuit.compose \\
        --output-dir docs/assets/cs_amp_fno
"""

from __future__ import annotations

# pylint: disable=wrong-import-position,too-many-arguments,too-many-locals,too-many-positional-arguments

import json
import logging
import time as time_module
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spino.circuit.composition import (  # noqa: E402
    ConvergenceReport,
    DcOperatingPointSolver,
    DcSolution,
    TransientSolution,
    TransientSolver,
)
from spino.circuit.composition_io import (  # noqa: E402
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    load_cs_amp_devices,
)
from spino.circuit.simulation import (  # noqa: E402
    OperatingPoint,
    TransientResult,
    run_dc_sweep,
    run_operating_point,
    run_transient,
)
from spino.circuit.topologies import build_cs_amp_active_load  # noqa: E402
from spino.circuit.tuning import extract_settling_time  # noqa: E402

__all__ = ["main"]

logger = logging.getLogger(__name__)

_DEFAULT_NFET_W = 1.6
_DEFAULT_NFET_L = 0.4
_DEFAULT_PFET_W = 2.5
_DEFAULT_PFET_L = 0.4
_DEFAULT_VDD = 1.8
_DEFAULT_VIN_BIAS = 0.81
_DEFAULT_VIN_OFF_BIAS = 1.0
_DEFAULT_VTC_STEP = 0.05
_DEFAULT_FIGURE_C_LOAD = 10e-12
_DEFAULT_T_STEP = 10e-9
_DEFAULT_T_STEP_START = 100e-9
_DEFAULT_T_END = 5.1e-6
_DEFAULT_VIN_STEP_AMP = 0.05
_OUTPUT_NODE = "v(out)"


@dataclass(frozen=True, slots=True)
class FnoMetrics:
    """
    Per-bias FNO solver outputs paired with the SPICE reference.

    :param vin_v: Input bias voltage in volts.
    :param fno_v_out_v: FNO-predicted ``V_out`` in volts.
    :param spice_v_out_v: SPICE ground-truth ``V_out`` in volts.
    :param fno_i_static_a: FNO bias current (mean of NFET/PFET currents).
    :param spice_i_static_a: SPICE bias current ``|I(VDD)|``.
    :param iter_count: NR iteration count to converge.
    :param wall_ms: Solver wall time in milliseconds.
    """

    vin_v: float
    fno_v_out_v: float
    spice_v_out_v: float
    fno_i_static_a: float
    spice_i_static_a: float
    iter_count: int
    wall_ms: float


def _format_pwl_step(vin_dc: float, t_step_start: float, vin_step_amp: float, t_end: float) -> str:
    """
    Builds the SPICE PWL string for the figure step stimulus.

    :param vin_dc: Pre-step bias voltage in volts.
    :param t_step_start: Step-onset time in seconds.
    :param vin_step_amp: Step amplitude in volts (positive ramps up).
    :param t_end: Final time of the analysis window in seconds.
    :return: SPICE-compatible PWL specifier.
    """
    return (
        f"PWL(0 {vin_dc} "
        f"{t_step_start * 0.5} {vin_dc} "
        f"{t_step_start} {vin_dc + vin_step_amp} "
        f"{t_end} {vin_dc + vin_step_amp})"
    )


def _build_vin_trajectory(time_s: np.ndarray, vin_dc: float, t_step_start: float, vin_step_amp: float) -> np.ndarray:
    """
    Sample the figure step stimulus on the FNO time grid.

    :param time_s: Time samples in seconds.
    :param vin_dc: Pre-step bias voltage in volts.
    :param t_step_start: Step-onset time in seconds.
    :param vin_step_amp: Step amplitude in volts.
    :return: ``V_in(t)`` samples on ``time_s``.
    """
    vin = np.full_like(time_s, fill_value=vin_dc, dtype=np.float64)
    half = t_step_start * 0.5
    ramp_mask = (time_s > half) & (time_s < t_step_start)
    vin[ramp_mask] = vin_dc + vin_step_amp * (time_s[ramp_mask] - half) / (t_step_start - half)
    vin[time_s >= t_step_start] = vin_dc + vin_step_amp
    return vin


def _spice_op(circuit_kwargs: dict, vin: float) -> tuple[OperatingPoint, float]:
    """
    Runs a SPICE operating point at a given input bias.

    :param circuit_kwargs: ``build_cs_amp_active_load`` keyword arguments.
    :param vin: Input bias voltage in volts.
    :return: Tuple ``(operating_point, wall_ms)``.
    """
    circuit = build_cs_amp_active_load(**{**circuit_kwargs, "vin_dc": vin})
    start = time_module.perf_counter()
    op = run_operating_point(circuit, capture_iters=True)
    wall_ms = 1000.0 * (time_module.perf_counter() - start)
    if op is None:
        raise RuntimeError(f"NGSpice .op failed at Vin={vin}")
    return op, wall_ms


def _spice_transient(
    circuit_kwargs: dict, vin_dc: float, vin_step_amp: float, t_step: float, t_end: float
) -> tuple[TransientResult, float]:
    """
    Runs a SPICE transient using the figure step stimulus.

    :param circuit_kwargs: Base ``build_cs_amp_active_load`` kwargs.
    :param vin_dc: Pre-step bias voltage in volts.
    :param vin_step_amp: Step amplitude in volts.
    :param t_step: Maximum SPICE timestep in seconds.
    :param t_end: Analysis end time in seconds.
    :return: Tuple ``(transient_result, wall_ms)``.
    """
    pwl = _format_pwl_step(vin_dc, _DEFAULT_T_STEP_START, vin_step_amp, t_end)
    circuit = build_cs_amp_active_load(
        **{**circuit_kwargs, "vin_dc": vin_dc, "vin_tran": pwl, "c_load_f": _DEFAULT_FIGURE_C_LOAD}
    )
    start = time_module.perf_counter()
    tran = run_transient(circuit, t_step=t_step, t_end=t_end, capture_iters=True)
    wall_ms = 1000.0 * (time_module.perf_counter() - start)
    if tran is None:
        raise RuntimeError(f"NGSpice .tran failed at Vin={vin_dc}")
    return tran, wall_ms


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination on aligned 1-D arrays.

    :param y_true: Reference signal.
    :param y_pred: Predicted signal.
    :return: ``R^2`` value (1.0 perfect, can go arbitrarily negative).
    """
    residual_ss = float(np.sum((y_true - y_pred) ** 2))
    total_ss = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-30
    return 1.0 - residual_ss / total_ss


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient on aligned 1-D arrays.

    :param y_true: Reference signal.
    :param y_pred: Predicted signal.
    :return: Pearson ``r`` value.
    """
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum absolute error on aligned 1-D arrays.

    :param y_true: Reference signal.
    :param y_pred: Predicted signal.
    :return: Maximum absolute error.
    """
    return float(np.max(np.abs(y_true - y_pred)))


def _mean_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error on aligned 1-D arrays.

    :param y_true: Reference signal.
    :param y_pred: Predicted signal.
    :return: Mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def _vtc_sweep(
    solver: DcOperatingPointSolver,
    vin_grid: np.ndarray,
    spice_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Runs the FNO and SPICE VTC sweeps.

    :param solver: Initialized DC operating point solver.
    :param vin_grid: Input voltage sweep points in volts.
    :param spice_kwargs: ``build_cs_amp_active_load`` keyword arguments.
    :return: Tuple ``(fno_vout, spice_vout, fno_wall_ms, spice_wall_ms)``.
    """
    fno_vout = np.zeros_like(vin_grid)
    fno_start = time_module.perf_counter()
    v_seed = solver.vdd / 2.0
    for idx, vin in enumerate(vin_grid):
        solution = solver.solve(vin=float(vin), v_out_init=v_seed)
        fno_vout[idx] = solution.v_out_v
        v_seed = solution.v_out_v
    fno_wall_ms = 1000.0 * (time_module.perf_counter() - fno_start)
    spice_circuit = build_cs_amp_active_load(**spice_kwargs)
    spice_start = time_module.perf_counter()
    spice_sweep = run_dc_sweep(
        spice_circuit,
        source_name="Vin",
        start=float(vin_grid[0]),
        stop=float(vin_grid[-1]),
        step=float(vin_grid[1] - vin_grid[0]),
    )
    spice_wall_ms = 1000.0 * (time_module.perf_counter() - spice_start)
    if spice_sweep is None:
        raise RuntimeError("NGSpice .dc sweep failed for the VTC overlay")
    spice_vout = np.interp(vin_grid, spice_sweep.sweep_values, spice_sweep.variables[_OUTPUT_NODE])
    return fno_vout, spice_vout, fno_wall_ms, spice_wall_ms


def _summarize_dc(
    bias_label: str,
    vin: float,
    fno_solution: DcSolution,
    spice_op: OperatingPoint,
    spice_wall_ms: float,
) -> tuple[FnoMetrics, dict]:
    """
    Reduces a DC OP comparison to a metrics dataclass and report dict.

    :param bias_label: Human-readable bias label (e.g. ``"nominal"``).
    :param vin: Input bias voltage in volts.
    :param fno_solution: FNO solver output.
    :param spice_op: SPICE operating point at the same bias.
    :param spice_wall_ms: SPICE wall-clock time in milliseconds.
    :return: Tuple of ``(metrics, report_dict)``.
    """
    spice_vout = float(spice_op.variables[_OUTPUT_NODE])
    spice_iddv = abs(float(spice_op.variables.get("i(vdd)", float("nan"))))
    fno_iddv = 0.5 * (fno_solution.i_pfet_a + fno_solution.i_nfet_a)
    metrics = FnoMetrics(
        vin_v=vin,
        fno_v_out_v=fno_solution.v_out_v,
        spice_v_out_v=spice_vout,
        fno_i_static_a=fno_iddv,
        spice_i_static_a=spice_iddv,
        iter_count=fno_solution.report.iter_count,
        wall_ms=fno_solution.report.wall_ms,
    )
    report = {
        "bias": bias_label,
        "vin_v": vin,
        "fno": asdict(fno_solution),
        "spice": {
            "v_out_v": spice_vout,
            "i_vdd_a": spice_iddv,
            "iter_count": spice_op.iter_count,
            "wall_ms": spice_wall_ms,
        },
        "rel_v_out_error_vdd": abs(fno_solution.v_out_v - spice_vout)
        / max(spice_op.variables.get("v(vdd)", 1.8), 1e-9),
    }
    return metrics, report


def _resample_transient(
    spice_tran: TransientResult,
    fno_solution: TransientSolution,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns the SPICE transient to the FNO time grid for direct comparison.

    :param spice_tran: NGSpice ``.tran`` result.
    :param fno_solution: FNO transient solver output.
    :return: Tuple of ``(spice_vout_resampled, fno_time_array)`` on the FNO grid.
    """
    fno_time = fno_solution.time_s.cpu().numpy()
    spice_vout_resampled = np.interp(fno_time, spice_tran.time, spice_tran.variables[_OUTPUT_NODE])
    return spice_vout_resampled, fno_time


def _plot_vtc_overlay(
    vin_grid: np.ndarray,
    fno_vout: np.ndarray,
    spice_vout: np.ndarray,
    output_path: Path,
) -> None:
    """
    Renders the FNO vs SPICE VTC overlay.

    :param vin_grid: Input voltage sweep points in volts.
    :param fno_vout: FNO ``V_out`` predictions.
    :param spice_vout: SPICE ``V_out`` reference (resampled onto ``vin_grid``).
    :param output_path: Destination file path.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(vin_grid, spice_vout, color="#0066cc", linewidth=1.6, label="SPICE")
    ax.plot(vin_grid, fno_vout, color="#cc6600", linewidth=1.4, linestyle="--", label="FNO")
    pearson = _pearson_r(spice_vout, fno_vout)
    ax.set_xlabel(r"$V_{in}$ (V)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(f"CS Amp VTC overlay (Pearson r={pearson:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_step_response(
    fno_solution: TransientSolution,
    spice_tran: TransientResult,
    output_path: Path,
) -> None:
    """
    Renders the figure step-response overlay (FNO vs SPICE).

    :param fno_solution: FNO transient solution.
    :param spice_tran: SPICE transient reference.
    :param output_path: Destination file path.
    """
    fno_time_us = fno_solution.time_s.cpu().numpy() * 1e6
    fno_vout = fno_solution.v_out_v.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(spice_tran.time * 1e6, spice_tran.variables[_OUTPUT_NODE], color="#0066cc", linewidth=1.4, label="SPICE")
    ax.plot(fno_time_us, fno_vout, color="#cc6600", linewidth=1.2, linestyle="--", label="FNO")
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title("CS Amp step response overlay")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _annotate_parity(ax, spice_vout: np.ndarray, fno_vout: np.ndarray) -> None:
    """
    Adds the identity line and metric box to a parity axis.

    :param ax: Matplotlib axis to annotate.
    :param spice_vout: SPICE reference ``V_out`` samples.
    :param fno_vout: FNO ``V_out`` samples.
    """
    lim_min = float(min(spice_vout.min(), fno_vout.min()))
    lim_max = float(max(spice_vout.max(), fno_vout.max()))
    pad = 0.03 * max(lim_max - lim_min, 1e-9)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#cc0000", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_aspect("equal", adjustable="box")
    metrics = "\n".join(
        [
            f"Pearson r = {_pearson_r(spice_vout, fno_vout):.5f}",
            f"MAE = {_mean_abs_error(spice_vout, fno_vout) * 1e3:.2f} mV",
            f"Max |ΔV| = {_max_abs_error(spice_vout, fno_vout) * 1e3:.2f} mV",
            f"N = {spice_vout.size}",
        ]
    )
    ax.text(
        0.03,
        0.97,
        metrics,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
    )
    ax.grid(True, alpha=0.3)


def _plot_diagnostic_parity(
    vtc_block: dict,
    fno_solution: TransientSolution,
    spice_tran: TransientResult,
    output_path: Path,
) -> None:
    """
    Renders a 2x2 SPICE-vs-FNO diagnostic figure for documentation.

    :param vtc_block: VTC sweep arrays and timings.
    :param fno_solution: FNO transient solution.
    :param spice_tran: SPICE transient reference.
    :param output_path: Destination file path.
    """
    spice_tran_vout, fno_time = _resample_transient(spice_tran, fno_solution)
    fno_tran_vout = fno_solution.v_out_v.cpu().numpy()
    fno_time_us = fno_time * 1e6
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
    ax = axes[0, 0]
    ax.plot(vtc_block["vin_grid"], vtc_block["spice_vout"], color="#0066cc", linewidth=1.6, label="SPICE")
    ax.plot(vtc_block["vin_grid"], vtc_block["fno_vout"], color="#cc6600", linewidth=1.4, linestyle="--", label="FNO")
    ax.set_xlabel(r"$V_{in}$ (V)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(f"VTC overlay | Pearson r={_pearson_r(vtc_block['spice_vout'], vtc_block['fno_vout']):.5f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax = axes[0, 1]
    ax.plot(spice_tran.time * 1e6, spice_tran.variables[_OUTPUT_NODE], color="#0066cc", linewidth=1.4, label="SPICE")
    ax.plot(fno_time_us, fno_tran_vout, color="#cc6600", linewidth=1.2, linestyle="--", label="FNO")
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(f"Step response overlay | Pearson r={_pearson_r(spice_tran_vout, fno_tran_vout):.5f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax = axes[1, 0]
    ax.scatter(
        vtc_block["spice_vout"],
        vtc_block["fno_vout"],
        color="#0066cc",
        s=22,
        alpha=0.65,
        edgecolors="#333333",
        linewidth=0.35,
    )
    _annotate_parity(ax, vtc_block["spice_vout"], vtc_block["fno_vout"])
    ax.set_xlabel(r"SPICE $V_{out}$ (V)")
    ax.set_ylabel(r"FNO $V_{out}$ (V)")
    ax.set_title("VTC parity")
    ax = axes[1, 1]
    ax.scatter(spice_tran_vout, fno_tran_vout, color="#0066cc", s=14, alpha=0.45, edgecolors="#333333", linewidth=0.25)
    _annotate_parity(ax, spice_tran_vout, fno_tran_vout)
    ax.set_xlabel(r"SPICE $V_{out}$ (V)")
    ax.set_ylabel(r"FNO $V_{out}$ (V)")
    ax.set_title("Transient parity")
    fig.suptitle("CS amplifier FNO composition diagnostics", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_convergence(
    dc_history: tuple[float, ...],
    transient_history: tuple[float, ...],
    output_path: Path,
) -> None:
    """
    Renders the residual-norm decay for both Newton solvers.

    :param dc_history: DC solver residual norm history.
    :param transient_history: Transient solver residual norm history.
    :param output_path: Destination file path.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if dc_history:
        ax.semilogy(range(len(dc_history)), dc_history, marker="o", label="DC (|R|, A)")
    if transient_history:
        ax.semilogy(
            range(len(transient_history)),
            transient_history,
            marker="s",
            label=r"Transient ($\|R\|_\infty$, A)",
        )
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("Residual norm (A)")
    ax.set_title("Newton-Raphson convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _measure_fno_speedup(
    nfet_device,
    pfet_device,
    *,
    vdd: float,
    vin_bias: float,
    v_out_dc: float,
) -> dict:
    """
    Times repeated FNO DC + transient solves for cold vs warm comparison.

    The first pass is reported as ``cold_solver_ms`` (model loaded, first NR
    solves). The second pass is ``warm_solver_ms`` (same tensors, no reload).

    :param nfet_device: Loaded NFET wrapper.
    :param pfet_device: Loaded PFET wrapper.
    :param vdd: Supply voltage in volts.
    :param vin_bias: Input bias for the DC and transient stimulus.
    :param v_out_dc: Initial output voltage for the transient (aligned reference).
    :return: Dictionary with millisecond timings and Newton iteration counts.
    """
    dc_solver = DcOperatingPointSolver(nfet_device, pfet_device, vdd=vdd)
    time_grid = np.arange(0.0, _DEFAULT_T_END, _DEFAULT_T_STEP)
    vin_t = _build_vin_trajectory(time_grid, vin_bias, _DEFAULT_T_STEP_START, _DEFAULT_VIN_STEP_AMP)
    time_tensor = torch.from_numpy(time_grid)
    vin_tensor = torch.from_numpy(vin_t)
    tran_solver = TransientSolver(nfet_device, pfet_device, vdd=vdd, c_load_f=_DEFAULT_FIGURE_C_LOAD)
    t0 = time_module.perf_counter()
    dc_a = dc_solver.solve(vin=vin_bias)
    tran_a = tran_solver.solve(time_tensor, vin_tensor, v_out_dc=v_out_dc)
    t1 = time_module.perf_counter()
    dc_b = dc_solver.solve(vin=vin_bias)
    tran_b = tran_solver.solve(time_tensor, vin_tensor, v_out_dc=v_out_dc)
    t2 = time_module.perf_counter()
    return {
        "cold_solver_ms": 1000.0 * (t1 - t0),
        "warm_solver_ms": 1000.0 * (t2 - t1),
        "cold_dc_iters": dc_a.report.iter_count,
        "cold_tran_iters": tran_a.report.iter_count,
        "warm_dc_iters": dc_b.report.iter_count,
        "warm_tran_iters": tran_b.report.iter_count,
    }


def _measure_spice_speedup(
    spice_kwargs: dict,
    *,
    vin_bias: float,
    vin_step_amp: float,
    t_step: float,
    t_end: float,
) -> dict:
    """
    Times back-to-back NGSpice ``.op`` + ``.tran`` runs for cold vs warm comparison.

    :param spice_kwargs: ``build_cs_amp_active_load`` keyword arguments.
    :param vin_bias: Pre-step input bias in volts.
    :param vin_step_amp: Step amplitude in volts.
    :param t_step: Maximum SPICE timestep in seconds.
    :param t_end: Transient end time in seconds.
    :return: Dictionary with millisecond timings and parsed iteration counts.
    """
    t0 = time_module.perf_counter()
    op_a, _ = _spice_op(spice_kwargs, vin_bias)
    tran_a, _ = _spice_transient(spice_kwargs, vin_bias, vin_step_amp, t_step, t_end)
    t1 = time_module.perf_counter()
    op_b, _ = _spice_op(spice_kwargs, vin_bias)
    tran_b, _ = _spice_transient(spice_kwargs, vin_bias, vin_step_amp, t_step, t_end)
    t2 = time_module.perf_counter()
    return {
        "cold_solver_ms": 1000.0 * (t1 - t0),
        "warm_solver_ms": 1000.0 * (t2 - t1),
        "cold_op_iters": op_a.iter_count,
        "cold_tran_iters": tran_a.iter_count,
        "warm_op_iters": op_b.iter_count,
        "warm_tran_iters": tran_b.iter_count,
    }


def _serialise_report(report: ConvergenceReport) -> dict:
    """
    Converts a :class:`ConvergenceReport` to a JSON-friendly dict.

    :param report: Report to serialise.
    :return: Dictionary with the same fields (tuples → lists).
    """
    record = asdict(report)
    record["residual_norm_history"] = list(report.residual_norm_history)
    return record


def _render_figures(
    output_dir: Path,
    nominal_solution: DcSolution,
    transient_solution: TransientSolution,
    spice_tran: TransientResult,
    vtc_block: dict,
) -> None:
    """
    Writes the three reference figures for the validation harness.

    :param output_dir: Destination directory; created if missing.
    :param nominal_solution: DC OP at the nominal bias (for convergence panel).
    :param transient_solution: FNO transient solution.
    :param spice_tran: SPICE transient reference.
    :param vtc_block: Dictionary with ``vin_grid``, ``fno_vout``, ``spice_vout``.
    """
    _plot_vtc_overlay(
        vtc_block["vin_grid"], vtc_block["fno_vout"], vtc_block["spice_vout"], output_dir / "vtc_overlay.png"
    )
    _plot_step_response(transient_solution, spice_tran, output_dir / "step_response_overlay.png")
    _plot_diagnostic_parity(vtc_block, transient_solution, spice_tran, output_dir / "diagnostic_parity.png")
    _plot_convergence(
        nominal_solution.report.residual_norm_history,
        transient_solution.report.residual_norm_history,
        output_dir / "convergence.png",
    )


def _build_summary(
    *,
    config: dict,
    nominal_metrics: FnoMetrics,
    nominal_report: dict,
    off_metrics: FnoMetrics,
    off_report: dict,
    transient_report: dict,
    fno_speed: dict,
    spice_speed: dict,
    vtc_block: dict,
) -> dict:
    """
    Assembles the JSON summary payload from the per-stage records.

    :param config: Run configuration metadata (topology, geometry, biases).
    :param nominal_metrics: DC OP metrics at the nominal bias.
    :param nominal_report: DC OP report dict at the nominal bias.
    :param off_metrics: DC OP metrics at the off bias.
    :param off_report: DC OP report dict at the off bias.
    :param transient_report: Transient comparison report.
    :param fno_speed: FNO cold/warm timing dict.
    :param spice_speed: SPICE cold/warm timing dict.
    :param vtc_block: VTC overlay arrays and timings.
    :return: JSON-serialisable dictionary written to ``summary.json``.
    """
    return {
        "config": config,
        "dc_op": {
            "nominal": {"metrics": asdict(nominal_metrics), "report": nominal_report},
            "off_bias": {"metrics": asdict(off_metrics), "report": off_report},
        },
        "transient": transient_report,
        "speedup": {
            "fno_ms": fno_speed,
            "spice_ms": spice_speed,
            "notes": (
                "cold_solver_ms is the first timed DC+transient pair after model load; "
                "warm_solver_ms is an immediate repeat. Compare to SPICE columns the same way."
            ),
        },
        "vtc": {
            "vin_v": vtc_block["vin_grid"].tolist(),
            "fno_v_out_v": vtc_block["fno_vout"].tolist(),
            "spice_v_out_v": vtc_block["spice_vout"].tolist(),
            "r_squared": _r_squared(vtc_block["spice_vout"], vtc_block["fno_vout"]),
            "pearson_r": _pearson_r(vtc_block["spice_vout"], vtc_block["fno_vout"]),
            "max_abs_error_v": _max_abs_error(vtc_block["spice_vout"], vtc_block["fno_vout"]),
            "fno_wall_ms": vtc_block["fno_wall_ms"],
            "spice_wall_ms": vtc_block["spice_wall_ms"],
        },
    }


def _run_dc_op(  # pylint: disable=too-many-arguments
    nfet_device,
    pfet_device,
    spice_kwargs: dict,
    bias_label: str,
    vin: float,
    vdd: float,
) -> tuple[FnoMetrics, dict, DcSolution]:
    """
    Runs FNO + SPICE DC OP at one bias and returns per-bias artefacts.

    :param nfet_device: Loaded NFET wrapper.
    :param pfet_device: Loaded PFET wrapper.
    :param spice_kwargs: SPICE ``build_cs_amp_active_load`` kwargs.
    :param bias_label: Human-readable bias label.
    :param vin: Input bias in volts.
    :param vdd: Supply voltage in volts.
    :return: ``(metrics, report_dict, dc_solution)``.
    """
    solver = DcOperatingPointSolver(nfet_device, pfet_device, vdd=vdd)
    fno_solution = solver.solve(vin=vin)
    spice_op, spice_wall_ms = _spice_op(spice_kwargs, vin)
    metrics, report = _summarize_dc(bias_label, vin, fno_solution, spice_op, spice_wall_ms)
    return metrics, report, fno_solution


def _build_transient(
    nfet_device,
    pfet_device,
    spice_kwargs: dict,
    v_out_dc: float,
    vdd: float,
    *,
    spice_v_out_dc: float | None = None,
) -> tuple[TransientSolution, TransientResult, dict]:
    """
    Runs the FNO + SPICE transient using the figure step stimulus.

    :param nfet_device: Loaded NFET wrapper.
    :param pfet_device: Loaded PFET wrapper.
    :param spice_kwargs: SPICE ``build_cs_amp_active_load`` kwargs.
    :param v_out_dc: Initial-condition voltage for the FNO transient (neural DC).
    :param vdd: Supply voltage in volts.
    :param spice_v_out_dc: Optional SPICE ``v(out)`` at ``.op`` for aligned IC
        when comparing waveforms; defaults to ``v_out_dc``.
    :return: ``(fno_solution, spice_tran, report_dict)``.
    """
    ic = float(spice_v_out_dc) if spice_v_out_dc is not None else float(v_out_dc)
    time_grid = np.arange(0.0, _DEFAULT_T_END, _DEFAULT_T_STEP)
    vin_t = _build_vin_trajectory(time_grid, _DEFAULT_VIN_BIAS, _DEFAULT_T_STEP_START, _DEFAULT_VIN_STEP_AMP)
    solver = TransientSolver(nfet_device, pfet_device, vdd=vdd, c_load_f=_DEFAULT_FIGURE_C_LOAD)
    time_tensor = torch.from_numpy(time_grid)
    vin_tensor = torch.from_numpy(vin_t)
    fno_solution = solver.solve(time_tensor, vin_tensor, v_out_dc=ic)
    spice_tran, spice_wall_ms = _spice_transient(
        spice_kwargs, _DEFAULT_VIN_BIAS, _DEFAULT_VIN_STEP_AMP, _DEFAULT_T_STEP, _DEFAULT_T_END
    )
    spice_vout_resampled, fno_time = _resample_transient(spice_tran, fno_solution)
    fno_vout = fno_solution.v_out_v.cpu().numpy()
    fno_settling = extract_settling_time(
        TransientResult(time=fno_time, variables={_OUTPUT_NODE: fno_vout}, iter_count=None),
        t_step_start=_DEFAULT_T_STEP_START,
    )
    spice_settling = extract_settling_time(spice_tran, t_step_start=_DEFAULT_T_STEP_START)
    report = {
        "fno": {
            "report": _serialise_report(fno_solution.report),
            "settling_time_s": float(fno_settling),
            "v_out_ic_v": ic,
        },
        "spice": {
            "iter_count": spice_tran.iter_count,
            "wall_ms": spice_wall_ms,
            "settling_time_s": float(spice_settling),
        },
        "transient_r2": _r_squared(spice_vout_resampled, fno_vout),
        "transient_pearson_r": float(np.corrcoef(spice_vout_resampled, fno_vout)[0, 1]),
        "transient_max_abs_error_v": float(np.max(np.abs(spice_vout_resampled - fno_vout))),
    }
    return fno_solution, spice_tran, report


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docs/assets/cs_amp_fno"),
    show_default=True,
)
@click.option("--nfet-w", type=float, default=_DEFAULT_NFET_W, show_default=True)
@click.option("--nfet-l", type=float, default=_DEFAULT_NFET_L, show_default=True)
@click.option("--pfet-w", type=float, default=_DEFAULT_PFET_W, show_default=True)
@click.option("--pfet-l", type=float, default=_DEFAULT_PFET_L, show_default=True)
@click.option("--vdd", type=float, default=_DEFAULT_VDD, show_default=True)
@click.option("--vin-bias", type=float, default=_DEFAULT_VIN_BIAS, show_default=True)
@click.option("--vin-off-bias", type=float, default=_DEFAULT_VIN_OFF_BIAS, show_default=True)
@click.option("--vtc-step", type=float, default=_DEFAULT_VTC_STEP, show_default=True)
@click.option("--device", type=str, default="cpu", show_default=True, help="Torch device for the FNOs (e.g. cuda).")
@click.option("--nfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_NFET_CHECKPOINT, show_default=True)
@click.option("--pfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_PFET_CHECKPOINT, show_default=True)
@click.option("--nfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_NFET_DATASET, show_default=True)
@click.option("--pfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_PFET_DATASET, show_default=True)
def main(  # pylint: disable=too-many-arguments,too-many-locals
    output_dir: Path,
    nfet_w: float,
    nfet_l: float,
    pfet_w: float,
    pfet_l: float,
    vdd: float,
    vin_bias: float,
    vin_off_bias: float,
    vtc_step: float,
    device: str,
    nfet_checkpoint: Path,
    pfet_checkpoint: Path,
    nfet_dataset: Path,
    pfet_dataset: Path,
) -> None:
    """Drives the FNO-composed validation pipeline: DC OP, transient, VTC, JSON summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    spice_kwargs = {"nfet_w": nfet_w, "nfet_l": nfet_l, "pfet_w": pfet_w, "pfet_l": pfet_l, "vdd": vdd}
    nfet_device, pfet_device = load_cs_amp_devices(
        nfet_w_um=nfet_w,
        nfet_l_um=nfet_l,
        pfet_w_um=pfet_w,
        pfet_l_um=pfet_l,
        nfet_checkpoint=nfet_checkpoint,
        pfet_checkpoint=pfet_checkpoint,
        nfet_dataset=nfet_dataset,
        pfet_dataset=pfet_dataset,
        map_location=device,
    )
    nominal_metrics, nominal_report, nominal_solution = _run_dc_op(
        nfet_device, pfet_device, spice_kwargs, "nominal", vin_bias, vdd
    )
    off_metrics, off_report, _ = _run_dc_op(nfet_device, pfet_device, spice_kwargs, "off_bias", vin_off_bias, vdd)
    spice_vout_ref = float(nominal_report["spice"]["v_out_v"])
    transient_solution, spice_tran, transient_report = _build_transient(
        nfet_device,
        pfet_device,
        spice_kwargs,
        v_out_dc=nominal_solution.v_out_v,
        vdd=vdd,
        spice_v_out_dc=spice_vout_ref,
    )
    fno_speed = _measure_fno_speedup(nfet_device, pfet_device, vdd=vdd, vin_bias=vin_bias, v_out_dc=spice_vout_ref)
    spice_speed = _measure_spice_speedup(
        spice_kwargs,
        vin_bias=vin_bias,
        vin_step_amp=_DEFAULT_VIN_STEP_AMP,
        t_step=_DEFAULT_T_STEP,
        t_end=_DEFAULT_T_END,
    )
    vin_grid = np.arange(0.0, vdd + 0.5 * vtc_step, vtc_step)
    fno_vtc, spice_vtc, fno_vtc_wall, spice_vtc_wall = _vtc_sweep(
        DcOperatingPointSolver(nfet_device, pfet_device, vdd=vdd), vin_grid, spice_kwargs
    )
    vtc_block = {
        "vin_grid": vin_grid,
        "fno_vout": fno_vtc,
        "spice_vout": spice_vtc,
        "fno_wall_ms": fno_vtc_wall,
        "spice_wall_ms": spice_vtc_wall,
    }
    _render_figures(output_dir, nominal_solution, transient_solution, spice_tran, vtc_block)
    config = {
        "topology": "cs_amp_active_load",
        "nfet_w_um": nfet_w,
        "nfet_l_um": nfet_l,
        "pfet_w_um": pfet_w,
        "pfet_l_um": pfet_l,
        "vdd_v": vdd,
        "vin_bias_v": vin_bias,
        "vin_off_bias_v": vin_off_bias,
        "c_load_f": _DEFAULT_FIGURE_C_LOAD,
        "t_step_s": _DEFAULT_T_STEP,
        "t_end_s": _DEFAULT_T_END,
        "device": device,
    }
    summary = _build_summary(
        config=config,
        nominal_metrics=nominal_metrics,
        nominal_report=nominal_report,
        off_metrics=off_metrics,
        off_report=off_report,
        transient_report=transient_report,
        fno_speed=fno_speed,
        spice_speed=spice_speed,
        vtc_block=vtc_block,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path.resolve())


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
