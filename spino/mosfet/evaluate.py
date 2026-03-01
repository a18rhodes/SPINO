"""
Post-training and inline evaluation for MOSFET FNO models.

Provides two evaluation modes:
1. Fast dataset-sampling for inline training monitoring
2. SPICE-based sweep generation for deep post-training validation

Plots I-V curves in physical units (Volts, Amps) to validate model accuracy
across operating regimes (subthreshold, linear, saturation).
"""

import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from spino.constants import ARCSINH_SCALE_MA
from spino.mosfet.gen_data import GEOMETRY_BINS, InfiniteSpiceMosfetDataset, ParameterSchema

# Default number of initial timesteps to discard from evaluation.
# Removes the SPICE .op-to-transient solver artifact (not physical device behavior).
# Purely heurstic based on visual inspection of SPICE vs FNO curves.
DEFAULT_TRIM_EVAL = 41

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

__all__ = [
    "evaluate_sample_iv_curves",
    "evaluate_spice_iv_sweeps",
    "evaluate_comprehensive",
    "calculate_r2",
    "calculate_subthreshold_r2",
    "calculate_male",
    "log_evaluation_summary",
    "DEFAULT_TRIM_EVAL",
]


def _apply_eval_trim(*arrays: np.ndarray, trim: int) -> tuple[np.ndarray, ...]:
    """
    Discards the first ``trim`` samples from each array.

    Used to exclude the SPICE .op-to-transient initialization artifact
    from both ground truth and prediction before computing metrics.
    This artifact is a numerical solver transient, not physical device behavior.

    :param arrays: One or more 1-D numpy arrays of equal length.
    :param trim: Number of leading samples to discard.
    :return: Tuple of trimmed arrays (single array unwrapped).
    """
    if trim <= 0:
        return arrays if len(arrays) > 1 else arrays[0]
    trimmed = tuple(a[trim:] for a in arrays)
    return trimmed if len(trimmed) > 1 else trimmed[0]


def calculate_r2(y_true, y_pred):
    """
    Calculates R² Score (Coefficient of Determination).

    :param y_true: Ground truth values.
    :param y_pred: Predicted values.
    :return: R² score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-12))


def calculate_subthreshold_r2(vg, y_true, y_pred, vg_threshold=0.5):
    """
    Calculates R² Score for the sub-threshold region (Vg < threshold).

    :param vg: Gate voltage array.
    :param y_true: Ground truth current values.
    :param y_pred: Predicted current values.
    :param vg_threshold: Voltage threshold defining sub-threshold region (default 0.5V).
    :return: Sub-threshold R² score, or None if insufficient data points.
    """
    mask = vg < vg_threshold
    if np.sum(mask) < 5:
        return None
    return calculate_r2(y_true[mask], y_pred[mask])


def calculate_male(y_true, y_pred):
    """
    Calculates Mean Absolute Logarithmic Error (MALE) in microamps.

    MALE = mean(|log10(|y_true| + epsilon) - log10(|y_pred| + epsilon)|) * 1000

    Useful for evaluating accuracy across multiple orders of magnitude,
    especially in subthreshold region where currents span nA to mA.

    :param y_true: Ground truth current values (mA).
    :param y_pred: Predicted current values (mA).
    :return: MALE in microamps.
    """
    epsilon = 1e-12
    log_true = np.log10(np.abs(y_true) + epsilon)
    log_pred = np.log10(np.abs(y_pred) + epsilon)
    male_ua = np.mean(np.abs(log_true - log_pred)) * 1000.0
    return male_ua


def _style_plot(ax, title, xlabel, ylabel):
    """Applies consistent dark-mode styling to plots."""
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _compute_l2_relative_error(y_true, y_pred):
    """
    Computes L2 relative error (scale-invariant metric).

    :param y_true: Ground truth array.
    :param y_pred: Predicted array.
    :return: Relative L2 error.
    """
    norm_diff = np.linalg.norm(y_pred - y_true)
    norm_ref = np.linalg.norm(y_true) + 1e-12
    return norm_diff / norm_ref


def _plot_error_dual(ax, x, y_true, y_pred, xlabel, title_prefix):
    """
    Plots absolute and relative error on dual y-axes.

    Left axis: Absolute error (mA) - useful for strong inversion
    Right axis: Relative error (%) - useful for subthreshold

    :param ax: Matplotlib axis.
    :param x: X-axis values (Vg or Vd).
    :param y_true: Ground truth current (mA).
    :param y_pred: Predicted current (mA).
    :param xlabel: X-axis label.
    :param title_prefix: Title prefix string.
    """
    abs_error = y_pred - y_true
    rel_error_pct = 100.0 * abs_error / (np.abs(y_true) + 1e-9)
    rel_error_pct = np.clip(rel_error_pct, -500, 500)
    ax.plot(x, abs_error * 1000.0, color="#ff6b6b", linewidth=1.5, label="Abs (uA)")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Absolute Error (uA)", fontsize=10, color="#ff6b6b")
    ax.tick_params(axis="y", labelcolor="#ff6b6b", labelsize=9)
    ax_rel = ax.twinx()
    ax_rel.plot(x, rel_error_pct, color="#4ecdc4", linewidth=1.2, linestyle="--", alpha=0.8, label="Rel (%)")
    ax_rel.set_ylabel("Relative Error (%)", fontsize=10, color="#4ecdc4")
    ax_rel.tick_params(axis="y", labelcolor="#4ecdc4", labelsize=9)
    ax_rel.set_ylim(-100, 100)
    mae_ua = np.mean(np.abs(abs_error)) * 1000.0
    mape = np.mean(np.abs(rel_error_pct[np.abs(y_true) > 0.001]))
    male_ua = calculate_male(y_true, y_pred)
    ax.set_title(
        f"{title_prefix}\nMAE={mae_ua:.2f}uA, MAPE={mape:.1f}%, MALE={male_ua:.2f}uA",
        fontsize=10,
        fontweight="bold",
        color="white",
    )
    ax.legend(loc="upper left", fontsize=8)
    ax_rel.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)


def _infer_and_denormalize_sample(
    model,
    dataset,
    sample_idx: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs model inference on a dataset sample and denormalizes all outputs.

    :param model: Trained MosfetFNO model.
    :param dataset: PreGeneratedMosfetDataset instance.
    :param sample_idx: Index of the sample to evaluate.
    :param device: Torch device for inference.
    :return: (current_true_ma, current_pred_ma, vg, vd, vs, physics_raw)
    """
    (voltages_norm, physics_norm), current_log = dataset[sample_idx]
    v_in = voltages_norm.unsqueeze(0).to(device)
    p_in = physics_norm.to(device)
    with torch.no_grad():
        pred_log = model(v_in, p_in).cpu().numpy().flatten()
    current_true_ma = ARCSINH_SCALE_MA * np.sinh(current_log.cpu().numpy().flatten())
    current_pred_ma = ARCSINH_SCALE_MA * np.sinh(pred_log)
    voltages_raw = v_in.cpu().numpy()[0]
    if dataset.normalize:
        voltages_raw = voltages_raw * dataset.voltages_std.numpy() + dataset.voltages_mean.numpy()
    physics_raw = p_in.cpu().numpy()[0]
    if dataset.normalize:
        physics_raw = physics_raw * dataset.physics_std.numpy() + dataset.physics_mean.numpy()
    return current_true_ma, current_pred_ma, voltages_raw[0], voltages_raw[1], voltages_raw[2], physics_raw


def _extract_geometry_label(physics_raw: np.ndarray) -> tuple[float, float, float]:
    """
    Extracts device geometry parameters (W, L, Vth0) from a denormalized physics vector.

    :param physics_raw: Denormalized physics parameter array.
    :return: (w_um, l_um, vth0) tuple.
    """
    try:
        w = physics_raw[ParameterSchema.TRAINING_KEYS.index("w")]
        l = physics_raw[ParameterSchema.TRAINING_KEYS.index("l")]
        vth0 = physics_raw[ParameterSchema.TRAINING_KEYS.index("vth0")]
    except (ValueError, IndexError):
        w, l, vth0 = 0.0, 0.0, 0.0
    return w, l, vth0


def _draw_transient_panel(ax, time_us, current_true_ma, current_pred_ma, w, l, vth0, mse, r2):
    """Draws the time-domain transient response panel."""
    _style_plot(
        ax,
        f"MOSFET Transient Response\nW={w:.2f}\u00b5m, L={l:.2f}\u00b5m, Vth0={vth0:.3f}V | MSE={mse:.2e}, R\u00b2={r2:.4f}",
        "Normalized Time",
        "Current (mA)",
    )
    ax.plot(time_us, current_true_ma, color="#ffffff", linewidth=2, alpha=0.7, label="Ground Truth")
    ax.plot(time_us, current_pred_ma, color="#00ffff", linestyle=":", linewidth=2, label="FNO Prediction")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)


def _draw_voltage_panel(ax, time_us, vg, vd, vs):
    """Draws the terminal voltages panel."""
    _style_plot(ax, "Terminal Voltages", "Normalized Time", "Voltage (V)")
    ax.plot(time_us, vg, color="#00ff00", label="Vg (Gate)", linewidth=1.5)
    ax.plot(time_us, vd, color="#ff00ff", label="Vd (Drain)", linewidth=1.5)
    ax.plot(time_us, vs, color="#ffff00", label="Vs (Source)", linewidth=1.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)


def _draw_parity_panel(ax, current_true_ma, current_pred_ma, r2):
    """Draws the true-vs-predicted parity scatter panel."""
    _style_plot(ax, f"Parity Plot: Current Prediction | R\u00b2={r2:.4f}", "True Id (mA)", "Predicted Id (mA)")
    ax.scatter(current_true_ma, current_pred_ma, c="#00ffff", s=15, alpha=0.6, edgecolors="white", linewidth=0.5)
    lim_min = min(current_true_ma.min(), current_pred_ma.min())
    lim_max = max(current_true_ma.max(), current_pred_ma.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1.5, alpha=0.7, label="Perfect (y=x)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(loc="upper left", fontsize=9)


def _draw_snapshot_panel(ax, vd, current_true_ma, current_pred_ma):
    """Draws the Id-vs-Vd scatter snapshot panel."""
    _style_plot(ax, "I-V Snapshot: Id vs Vd (Output)", "Vd (V)", "Id (mA)")
    ax.scatter(vd, current_true_ma, c="#ffffff", s=10, alpha=0.3, label="True")
    ax.scatter(vd, current_pred_ma, c="#00ffff", s=10, alpha=0.5, marker="x", label="Pred")
    ax.legend(loc="upper left", fontsize=9)


def evaluate_sample_iv_curves(model, dataset, device="cuda", sample_idx=None):
    """
    Evaluates model on a random dataset sample and plots I-V characteristics.

    Converts log-scale predictions back to linear current (Amperes) and generates:
    1. Time-domain transient response (V_terminals vs I_d)
    2. Transfer curve approximation (I_d vs V_g snapshot)
    3. Output curve approximation (I_d vs V_d snapshot)

    :param model: Trained MosfetFNO model.
    :param dataset: PreGeneratedMosfetDataset instance.
    :param device: Torch device for inference.
    :param sample_idx: Specific sample index, or None for random.
    :return: (figure, r2_score) tuple.
    """
    model.eval()
    plt.style.use("dark_background")
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(dataset))
    current_true_ma, current_pred_ma, vg, vd, vs, physics_raw = _infer_and_denormalize_sample(
        model, dataset, sample_idx, device
    )
    w, l, vth0 = _extract_geometry_label(physics_raw)
    mse = np.mean((current_true_ma - current_pred_ma) ** 2)
    r2 = calculate_r2(current_true_ma, current_pred_ma)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_us = np.linspace(0, 1, len(current_true_ma))
    _draw_transient_panel(axes[0, 0], time_us, current_true_ma, current_pred_ma, w, l, vth0, mse, r2)
    _draw_voltage_panel(axes[0, 1], time_us, vg, vd, vs)
    _draw_parity_panel(axes[1, 0], current_true_ma, current_pred_ma, r2)
    _draw_snapshot_panel(axes[1, 1], vd, current_true_ma, current_pred_ma)
    plt.tight_layout()
    return fig, r2


def _build_p_tensor(spice_dataset, dataset, w_um: float, l_um: float, device: str):
    """
    Builds and normalizes the physics parameter tensor for a given device geometry.

    Inspects BSIM parameters at (w_um, l_um), selects the curated parameter subset
    if required, and normalizes using dataset statistics.

    :param spice_dataset: InfiniteSpiceMosfetDataset used for BSIM parameter queries.
    :param dataset: PreGeneratedMosfetDataset providing normalization statistics.
    :param w_um: Device width in microns.
    :param l_um: Device length in microns.
    :param device: Torch device string.
    :return: Physics parameter tensor of shape (1, n_params) on the target device.
    """
    raw_params = spice_dataset.parser.inspect_model(spice_dataset.strategy.model_name, w=str(w_um), l=str(l_um))
    p_full = ParameterSchema.to_tensor(raw_params).squeeze()
    p_curated = p_full[ParameterSchema.TRAINING_INDICES] if dataset.use_curated_params else p_full
    if dataset.normalize:
        p_curated = (p_curated - dataset.physics_mean) / dataset.physics_std
    return p_curated.unsqueeze(0).to(device)


def evaluate_spice_iv_sweeps(
    model, dataset, device="cuda", w_um=1.0, l_um=0.18, t_steps=512, trim_eval=DEFAULT_TRIM_EVAL
):
    """
    Generates deterministic Id-Vg and Id-Vd sweeps using SPICE ground truth.

    This is the "gold standard" validation for post-training deep analysis.
    Runs actual SPICE simulations to generate clean transfer/output curves.

    WARNING: Each SPICE sim takes ~15s. Total runtime ~30-45s for 2-3 sweeps.
    Use this for FINAL validation, not during training loops.

    :param model: Trained MosfetFNO model.
    :param dataset: PreGeneratedMosfetDataset (used only for normalization stats).
    :param device: Torch device for inference.
    :param w_um: Device width in microns.
    :param l_um: Device length in microns.
    :param t_steps: Number of time steps for quasi-static sweep (after trim).
    :param trim_eval: Number of initial timesteps to discard from SPICE and FNO
        outputs before computing metrics. Removes the .op-to-transient solver artifact.
    :return: (figure, metrics_dict) tuple with timing metrics.
    """
    model.eval()
    plt.style.use("dark_background")
    logger.info("Running SPICE-based I-V sweep validation (this will take ~30-60s)...")
    raw_steps = t_steps + trim_eval
    spice_dataset = InfiniteSpiceMosfetDataset(strategy_name="sky130_nmos", t_steps=raw_steps, t_end=raw_steps * 1e-9)
    time_grid = np.linspace(0, spice_dataset.t_end, raw_steps)
    vs_gnd = np.zeros(raw_steps)
    vb_gnd = np.zeros(raw_steps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = {}
    timing_spice_ms: list[float] = []
    timing_fno_ms: list[float] = []
    p_tensor = _build_p_tensor(spice_dataset, dataset, w_um, l_um, device)
    vg_sweep = np.linspace(0, 1.8, raw_steps)
    vd_sat = np.full(raw_steps, 1.8)
    logger.info("Running Id-Vg transfer sweep (Vd=1.8V, Vg: 0->1.8V)...")
    id_spice_transfer, id_pred_transfer, spice_ms, fno_ms = _run_timed_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg_sweep, vd_sat, vs_gnd, vb_gnd, w_um, l_um, device
    )
    if id_spice_transfer is None:
        logger.error("SPICE simulation failed for Id-Vg sweep")
        return fig, {}
    timing_spice_ms.append(spice_ms)
    timing_fno_ms.append(fno_ms)
    vg_plot, id_spice_plot, id_pred_plot = _apply_eval_trim(vg_sweep, id_spice_transfer, id_pred_transfer, trim=trim_eval)
    r2_transfer = calculate_r2(id_spice_plot, id_pred_plot)
    r2_subth = calculate_subthreshold_r2(vg_plot, id_spice_plot, id_pred_plot)
    l2_transfer = _compute_l2_relative_error(id_spice_plot, id_pred_plot)
    metrics["r2_transfer"] = r2_transfer
    metrics["r2_transfer_subth"] = r2_subth if r2_subth is not None else 0.0
    metrics["l2_transfer"] = l2_transfer
    metrics["male_transfer"] = calculate_male(id_spice_plot, id_pred_plot)
    subth_str = f", SubTh-R²={r2_subth:.4f}" if r2_subth is not None else ""
    _style_plot(
        axes[0, 0],
        f"Id-Vg Transfer (Saturation)\nW={w_um}µm, L={l_um}µm | R²={r2_transfer:.4f}{subth_str}, L2={l2_transfer:.4f}\nVd=1.8V, Vs=0V, Vb=0V",
        "Vg (V)",
        "Id (mA)",
    )
    axes[0, 0].plot(vg_plot, id_spice_plot, color="#ffffff", linewidth=2.5, alpha=0.7, label="SPICE")
    axes[0, 0].plot(vg_plot, id_pred_plot, color="#00ffff", linestyle=":", linewidth=2, label="FNO")
    axes[0, 0].set_yscale("log")
    axes[0, 0].axvline(0.5, color="#ffaa00", linewidth=1, linestyle=":", alpha=0.5, label="SubTh (Vg<0.5V)")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    _plot_error_dual(axes[0, 1], vg_plot, id_spice_plot, id_pred_plot, "Vg (V)", "Transfer Error")
    vd_sweep = np.linspace(0, 1.8, raw_steps)
    vg_drive = np.full(raw_steps, 1.2)
    logger.info("Running Id-Vd output sweep (Vg=1.2V, Vd: 0→1.8V)...")
    id_spice_output, id_pred_output, spice_ms2, fno_ms2 = _run_timed_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg_drive, vd_sweep, vs_gnd, vb_gnd, w_um, l_um, device
    )
    if id_spice_output is None:
        logger.error("SPICE simulation failed for Id-Vd sweep")
    else:
        timing_spice_ms.append(spice_ms2)
        timing_fno_ms.append(fno_ms2)
        vd_plot, id_spice_plot2, id_pred_plot2 = _apply_eval_trim(vd_sweep, id_spice_output, id_pred_output, trim=trim_eval)
        r2_output = calculate_r2(id_spice_plot2, id_pred_plot2)
        l2_output = _compute_l2_relative_error(id_spice_plot2, id_pred_plot2)
        metrics["r2_output"] = r2_output
        metrics["l2_output"] = l2_output
        metrics["male_output"] = calculate_male(id_spice_plot2, id_pred_plot2)
        _style_plot(
            axes[1, 0],
            f"Id-Vd Output (Linear/Sat)\nVg=1.2V, Vs=0V, Vb=0V | R²={r2_output:.4f}, L2={l2_output:.4f}",
            "Vd (V)",
            "Id (mA)",
        )
        axes[1, 0].plot(vd_plot, id_spice_plot2, color="#ffffff", linewidth=2.5, alpha=0.7, label="SPICE")
        axes[1, 0].plot(vd_plot, id_pred_plot2, color="#ff00ff", linestyle=":", linewidth=2, label="FNO")
        axes[1, 0].legend(loc="upper left", fontsize=9)
        _plot_error_dual(axes[1, 1], vd_plot, id_spice_plot2, id_pred_plot2, "Vd (V)", "Output Error")
    plt.tight_layout()
    avg_spice_ms = sum(timing_spice_ms) / len(timing_spice_ms) if timing_spice_ms else 0.0
    avg_fno_ms = sum(timing_fno_ms) / len(timing_fno_ms) if timing_fno_ms else 0.0
    speedup = avg_spice_ms / avg_fno_ms if avg_fno_ms > 0 else 0.0
    metrics["timing_spice_avg_ms"] = avg_spice_ms
    metrics["timing_fno_avg_ms"] = avg_fno_ms
    metrics["timing_speedup_x"] = speedup
    logger.info("SPICE avg: %.1f ms | FNO avg: %.2f ms | Speedup: %.0fx", avg_spice_ms, avg_fno_ms, speedup)
    logger.info("SPICE-based sweep validation complete.")
    return fig, metrics


def evaluate_multi_geometry(
    model,
    dataset,
    output_dir: Path,
    geometries: Optional[list[tuple[float, float]]] = None,
    device: str = "cuda",
    t_steps: int = 512,
    trim_eval: int = DEFAULT_TRIM_EVAL,
) -> dict:
    """
    Runs SPICE-based I-V validation across multiple device geometries.

    Generates one figure per geometry and returns aggregated metrics.
    Useful for validating model generalization across W/L space.

    :param model: Trained MosfetFNO model.
    :param dataset: PreGeneratedMosfetDataset (for normalization stats).
    :param output_dir: Directory to save per-geometry plots.
    :param geometries: List of (W_um, L_um) tuples. Defaults to design corners.
    :param device: Torch device for inference.
    :param t_steps: Time steps for quasi-static sweep (after trim).
    :param trim_eval: Number of initial timesteps to discard (SPICE .op artifact).
    :return: Tuple of (metrics_dict, figures_dict) where figures_dict maps geometry keys to matplotlib figures.
    """
    if geometries is None:
        geometries = [
            (0.42, 0.15),  # Min size (near minimum L)
            (1.0, 0.18),  # Standard digital
            (2.0, 0.5),  # Analog mid-size
            (5.0, 1.0),  # Large analog
            (10.0, 2.0),  # Very large (low-speed analog)
        ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    all_figures = {}
    logger.info("Running multi-geometry validation across %d geometries...", len(geometries))
    for w_um, l_um in geometries:
        geom_key = f"W{w_um:.2f}_L{l_um:.2f}"
        logger.info("Evaluating geometry: W=%.2f um, L=%.2f um", w_um, l_um)
        fig, metrics = evaluate_spice_iv_sweeps(
            model, dataset, device=device, w_um=w_um, l_um=l_um, t_steps=t_steps, trim_eval=trim_eval
        )
        fig_path = output_dir / f"iv_sweep_{geom_key}.png"
        fig.savefig(fig_path, dpi=150)
        all_figures[geom_key] = fig
        all_metrics[geom_key] = metrics
        logger.info("  Saved: %s", fig_path)
        if metrics:
            logger.info(
                "  R2_transfer=%.4f, R2_output=%.4f, Speedup=%.0fx",
                metrics.get("r2_transfer", 0),
                metrics.get("r2_output", 0),
                metrics.get("timing_speedup_x", 0),
            )
    logger.info("Multi-geometry validation complete. %d figures saved to: %s", len(geometries), output_dir)
    return all_metrics, all_figures


def evaluate_comprehensive(
    model,
    dataset,
    output_dir: Path,
    device: str = "cuda",
    t_steps: int = 512,
    trim_eval: int = DEFAULT_TRIM_EVAL,
) -> dict:
    """
    Comprehensive SPICE validation across geometry bins and waveform types.

    Tests 3 geometry sizes (tiny, medium, xlarge) with 3 waveform types each:
    - Ramp: Monotonic gate sweep (Id-Vg transfer curve)
    - Sweep: Monotonic drain sweep (Id-Vd output curve)
    - Random: PWL chaotic waveform (time-domain transient)

    Generates a 3x3 grid of plots per geometry and aggregates error metrics.

    :param model: Trained MosfetFNO model.
    :param dataset: PreGeneratedMosfetDataset (for normalization stats).
    :param output_dir: Directory to save comprehensive plots.
    :param device: Torch device for inference.
    :param t_steps: Time steps for quasi-static sweep (after trim).
    :param trim_eval: Number of initial timesteps to discard (SPICE .op artifact).
    :return: Tuple of (metrics_dict, figures_dict) where figures_dict maps geometry names to matplotlib figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_geometries = {
        "tiny": (GEOMETRY_BINS["tiny"].w_range[0] + 0.05, GEOMETRY_BINS["tiny"].l_range[0] + 0.02),
        "medium": (2.5, 0.75),
        "xlarge": (8.0, 1.75),
    }
    all_metrics = {}
    all_figures = {}
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE SPICE EVALUATION")
    logger.info("=" * 80)
    logger.info("Geometries: %s", list(test_geometries.keys()))
    logger.info("Waveforms: ramp (Id-Vg), sweep (Id-Vd), random (PWL transient)")
    logger.info("=" * 80)
    for geom_name, (w_um, l_um) in test_geometries.items():
        logger.info("Evaluating %s geometry: W=%.2f um, L=%.2f um", geom_name.upper(), w_um, l_um)
        geom_metrics, geom_fig = _evaluate_single_geometry_comprehensive(
            model, dataset, w_um, l_um, geom_name, output_dir, device, t_steps, trim_eval
        )
        all_metrics[geom_name] = geom_metrics
        all_figures[geom_name] = geom_fig
    _generate_summary_table(all_metrics, output_dir)
    logger.info("=" * 80)
    logger.info("Comprehensive evaluation complete. Results in: %s", output_dir)
    logger.info("=" * 80)
    return all_metrics, all_figures


def _evaluate_single_geometry_ramp(
    t_steps: int,
    model,
    dataset,
    spice_dataset,
    p_tensor,
    time_grid,
    vs_gnd,
    vb_gnd,
    w_um,
    l_um,
    device,
    metrics,
    axes,
    trim_eval: int = DEFAULT_TRIM_EVAL,
):
    raw_steps = len(time_grid)
    vg_sweep = np.linspace(0, 1.8, raw_steps)
    vd_sat = np.full(raw_steps, 1.8)
    id_spice_ramp, id_pred_ramp = _run_single_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg_sweep, vd_sat, vs_gnd, vb_gnd, w_um, l_um, device
    )
    if id_spice_ramp is not None:
        vg_sweep, id_spice_ramp, id_pred_ramp = _apply_eval_trim(vg_sweep, id_spice_ramp, id_pred_ramp, trim=trim_eval)
        r2_ramp = calculate_r2(id_spice_ramp, id_pred_ramp)
        r2_ramp_subth = calculate_subthreshold_r2(vg_sweep, id_spice_ramp, id_pred_ramp)
        metrics["ramp_r2"] = r2_ramp
        metrics["ramp_r2_subth"] = r2_ramp_subth if r2_ramp_subth is not None else 0.0
        metrics["ramp_mae_ua"] = np.mean(np.abs(id_pred_ramp - id_spice_ramp)) * 1000.0
        metrics["ramp_male_ua"] = calculate_male(id_spice_ramp, id_pred_ramp)
        subth_str = f", SubTh={r2_ramp_subth:.4f}" if r2_ramp_subth is not None else ""
        _style_plot(
            axes[0, 0], f"Ramp: Id-Vg | R2={r2_ramp:.4f}{subth_str}\nVd=1.8V, Vs=0V, Vb=0V", "Vg (V)", "Id (mA)"
        )
        axes[0, 0].plot(vg_sweep, id_spice_ramp, color="#ffffff", linewidth=2.5, alpha=0.7, label="SPICE")
        axes[0, 0].plot(vg_sweep, id_pred_ramp, color="#00ffff", linestyle=":", linewidth=2, label="FNO")
        axes[0, 0].axvline(0.5, color="#ffaa00", linewidth=1, linestyle=":", alpha=0.5, label="SubTh")
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend(loc="upper left", fontsize=9)
        _plot_error_dual(axes[0, 1], vg_sweep, id_spice_ramp, id_pred_ramp, "Vg (V)", "Ramp Error")
        axes[0, 2].scatter(id_spice_ramp, id_pred_ramp, c="#00ffff", s=10, alpha=0.5)
        lims = [min(id_spice_ramp.min(), id_pred_ramp.min()), max(id_spice_ramp.max(), id_pred_ramp.max())]
        axes[0, 2].plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        _style_plot(axes[0, 2], f"Ramp Parity | R2={r2_ramp:.4f}", "SPICE Id (mA)", "FNO Id (mA)")


def _evaluate_single_geometry_sweep(
    t_steps: int,
    model,
    dataset,
    spice_dataset,
    p_tensor,
    time_grid,
    vs_gnd,
    vb_gnd,
    w_um,
    l_um,
    device,
    metrics,
    axes,
    trim_eval: int = DEFAULT_TRIM_EVAL,
):
    raw_steps = len(time_grid)
    vd_sweep = np.linspace(0, 1.8, raw_steps)
    vg_drive = np.full(raw_steps, 1.2)
    id_spice_sweep, id_pred_sweep = _run_single_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg_drive, vd_sweep, vs_gnd, vb_gnd, w_um, l_um, device
    )
    if id_spice_sweep is not None:
        vd_sweep, id_spice_sweep, id_pred_sweep = _apply_eval_trim(
            vd_sweep, id_spice_sweep, id_pred_sweep, trim=trim_eval
        )
        r2_sweep = calculate_r2(id_spice_sweep, id_pred_sweep)
        metrics["sweep_r2"] = r2_sweep
        metrics["sweep_mae_ua"] = np.mean(np.abs(id_pred_sweep - id_spice_sweep)) * 1000.0
        metrics["sweep_male_ua"] = calculate_male(id_spice_sweep, id_pred_sweep)
        _style_plot(axes[1, 0], f"Sweep: Id-Vd | R2={r2_sweep:.4f}\nVg=1.2V, Vs=0V, Vb=0V", "Vd (V)", "Id (mA)")
        axes[1, 0].plot(vd_sweep, id_spice_sweep, color="#ffffff", linewidth=2.5, alpha=0.7, label="SPICE")
        axes[1, 0].plot(vd_sweep, id_pred_sweep, color="#ff00ff", linestyle=":", linewidth=2, label="FNO")
        axes[1, 0].legend(loc="upper left", fontsize=9)
        _plot_error_dual(axes[1, 1], vd_sweep, id_spice_sweep, id_pred_sweep, "Vd (V)", "Sweep Error")
        axes[1, 2].scatter(id_spice_sweep, id_pred_sweep, c="#ff00ff", s=10, alpha=0.5)
        lims = [min(id_spice_sweep.min(), id_pred_sweep.min()), max(id_spice_sweep.max(), id_pred_sweep.max())]
        axes[1, 2].plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        _style_plot(axes[1, 2], f"Sweep Parity | R2={r2_sweep:.4f}", "SPICE Id (mA)", "FNO Id (mA)")


def _evaluate_single_geometry_random(
    model,
    dataset,
    spice_dataset,
    p_tensor,
    time_grid,
    vs_gnd,
    vb_gnd,
    w_um,
    l_um,
    device,
    metrics,
    axes,
    geom_name,
    trim_eval: int = DEFAULT_TRIM_EVAL,
):
    np.random.seed(42 + hash(geom_name) % 1000)
    n_pts = np.random.randint(8, 15)
    pwl_times = np.sort(np.random.uniform(0, spice_dataset.t_end, n_pts))
    pwl_times = np.concatenate(([0], pwl_times, [spice_dataset.t_end]))
    vg_pwl_raw = np.random.uniform(0, 1.8, len(pwl_times))
    vd_pwl_raw = np.random.uniform(0, 1.8, len(pwl_times))
    vg_pwl = np.interp(time_grid, pwl_times, vg_pwl_raw)
    vd_pwl = np.interp(time_grid, pwl_times, vd_pwl_raw)
    id_spice_pwl, id_pred_pwl = _run_single_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg_pwl, vd_pwl, vs_gnd, vb_gnd, w_um, l_um, device
    )
    if id_spice_pwl is not None:
        time_grid_t, id_spice_pwl, id_pred_pwl = _apply_eval_trim(time_grid, id_spice_pwl, id_pred_pwl, trim=trim_eval)
        vg_pwl, vd_pwl = _apply_eval_trim(vg_pwl, vd_pwl, trim=trim_eval)
        r2_pwl = calculate_r2(id_spice_pwl, id_pred_pwl)
        metrics["random_r2"] = r2_pwl
        metrics["random_mae_ua"] = np.mean(np.abs(id_pred_pwl - id_spice_pwl)) * 1000.0
        metrics["random_male_ua"] = calculate_male(id_spice_pwl, id_pred_pwl)
        time_us = time_grid_t * 1e6
        _style_plot(
            axes[2, 0], f"Random: PWL Transient | R2={r2_pwl:.4f}\nVg/Vd=PWL, Vs=0V, Vb=0V", "Time (us)", "Id (mA)"
        )
        axes[2, 0].plot(time_us, id_spice_pwl, color="#ffffff", linewidth=2, alpha=0.7, label="SPICE")
        axes[2, 0].plot(time_us, id_pred_pwl, color="#00ff00", linestyle=":", linewidth=1.5, label="FNO")
        axes[2, 0].legend(loc="upper right", fontsize=9)
        ax_v = axes[2, 0].twinx()
        ax_v.plot(time_us, vg_pwl, color="#ffff00", linewidth=1, alpha=0.5, linestyle="--", label="Vg")
        ax_v.plot(time_us, vd_pwl, color="#ff6600", linewidth=1, alpha=0.5, linestyle=":", label="Vd")
        ax_v.set_ylabel("Voltage (V)", fontsize=9, color="#ffff00")
        ax_v.set_ylim(0, 2.0)
        ax_v.legend(loc="lower right", fontsize=8)
        _plot_error_dual(axes[2, 1], time_us, id_spice_pwl, id_pred_pwl, "Time (us)", "Random Error")
        axes[2, 2].scatter(id_spice_pwl, id_pred_pwl, c="#00ff00", s=10, alpha=0.5)
        lims = [min(id_spice_pwl.min(), id_pred_pwl.min()), max(id_spice_pwl.max(), id_pred_pwl.max())]
        axes[2, 2].plot(lims, lims, "r--", linewidth=1, alpha=0.7)
        _style_plot(axes[2, 2], f"Random Parity | R2={r2_pwl:.4f}", "SPICE Id (mA)", "FNO Id (mA)")


def _evaluate_single_geometry_comprehensive(
    model,
    dataset,
    w_um: float,
    l_um: float,
    geom_name: str,
    output_dir: Path,
    device: str,
    t_steps: int,
    trim_eval: int = DEFAULT_TRIM_EVAL,
) -> dict:
    """
    Evaluates a single geometry with all three waveform types.

    :param model: Trained model.
    :param dataset: Dataset for normalization.
    :param w_um: Device width.
    :param l_um: Device length.
    :param geom_name: Geometry bin name for labeling.
    :param output_dir: Output directory.
    :param device: Torch device.
    :param t_steps: Time steps (after trim).
    :param trim_eval: Number of initial timesteps to discard (SPICE .op artifact).
    :return: Metrics dict for this geometry.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    raw_steps = t_steps + trim_eval
    spice_dataset = InfiniteSpiceMosfetDataset(strategy_name="sky130_nmos", t_steps=raw_steps, t_end=raw_steps * 1e-9)
    time_grid = np.linspace(0, spice_dataset.t_end, raw_steps)
    vs_gnd = np.zeros(raw_steps)
    vb_gnd = np.zeros(raw_steps)
    p_tensor = _build_p_tensor(spice_dataset, dataset, w_um, l_um, device)
    metrics = {}
    logger.info("  [1/3] Ramp (Id-Vg transfer)...")
    _evaluate_single_geometry_ramp(
        t_steps=t_steps,
        model=model,
        dataset=dataset,
        spice_dataset=spice_dataset,
        p_tensor=p_tensor,
        time_grid=time_grid,
        vs_gnd=vs_gnd,
        vb_gnd=vb_gnd,
        w_um=w_um,
        l_um=l_um,
        device=device,
        metrics=metrics,
        axes=axes,
        trim_eval=trim_eval,
    )
    logger.info("  [2/3] Sweep (Id-Vd output)...")
    _evaluate_single_geometry_sweep(
        t_steps=t_steps,
        model=model,
        dataset=dataset,
        spice_dataset=spice_dataset,
        p_tensor=p_tensor,
        time_grid=time_grid,
        vs_gnd=vs_gnd,
        vb_gnd=vb_gnd,
        w_um=w_um,
        l_um=l_um,
        device=device,
        metrics=metrics,
        axes=axes,
        trim_eval=trim_eval,
    )
    logger.info("  [3/3] Random (PWL transient)...")
    _evaluate_single_geometry_random(
        model=model,
        dataset=dataset,
        spice_dataset=spice_dataset,
        p_tensor=p_tensor,
        time_grid=time_grid,
        vs_gnd=vs_gnd,
        vb_gnd=vb_gnd,
        w_um=w_um,
        l_um=l_um,
        device=device,
        metrics=metrics,
        axes=axes,
        geom_name=geom_name,
        trim_eval=trim_eval,
    )
    fig.suptitle(
        f"Comprehensive Evaluation: {geom_name.upper()} (W={w_um:.2f}um, L={l_um:.2f}um)",
        fontsize=14,
        fontweight="bold",
        color="white",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = output_dir / f"comprehensive_{geom_name}.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("  Saved: %s", fig_path)
    return metrics, fig


def _run_timed_sweep(
    model, dataset, spice_dataset, p_tensor, time_grid, vg, vd, vs, vb, w_um, l_um, device
) -> tuple[np.ndarray | None, np.ndarray | None, float, float]:
    """
    Runs a single SPICE + FNO comparison for given voltage waveforms, tracking wall-clock time.

    :return: (id_spice_ma, id_pred_ma, spice_elapsed_ms, fno_elapsed_ms), or
        (None, None, spice_elapsed_ms, 0.0) on SPICE failure.
    """
    pwl_g = spice_dataset._build_pwl_string(time_grid, vg)
    pwl_d = spice_dataset._build_pwl_string(time_grid, vd)
    pwl_s = spice_dataset._build_pwl_string(time_grid, vs)
    pwl_b = spice_dataset._build_pwl_string(time_grid, vb)
    netlist = spice_dataset._build_netlist(w_um, l_um, pwl_g, pwl_d, pwl_s, pwl_b)
    t0_spice = time.perf_counter()
    results = spice_dataset._run_transient_simulation(netlist)
    spice_ms = (time.perf_counter() - t0_spice) * 1000.0
    if results is None:
        return None, None, spice_ms, 0.0
    current_raw = spice_dataset._extract_drain_current(results)
    if current_raw is None:
        return None, None, spice_ms, 0.0
    id_spice_a = spice_dataset._interpolate_current(current_raw, results.get("time"))
    if id_spice_a is None:
        return None, None, spice_ms, 0.0
    id_spice_ma = id_spice_a * 1000.0
    v_stack = np.stack([vg, vd, vs, vb])
    v_tensor = torch.tensor(v_stack, dtype=torch.float32).unsqueeze(0).to(device)
    if dataset.normalize:
        v_tensor = (v_tensor - dataset.voltages_mean.to(device)) / dataset.voltages_std.to(device)
    t0_fno = time.perf_counter()
    with torch.no_grad():
        pred_log = model(v_tensor, p_tensor).cpu().numpy().flatten()
    fno_ms = (time.perf_counter() - t0_fno) * 1000.0
    return id_spice_ma, ARCSINH_SCALE_MA * np.sinh(pred_log), spice_ms, fno_ms


def _run_single_sweep(
    model, dataset, spice_dataset, p_tensor, time_grid, vg, vd, vs, vb, w_um, l_um, device
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Runs a single SPICE + FNO comparison for given voltage waveforms.

    :return: (id_spice_ma, id_pred_ma) tuple or (None, None) on failure.
    """
    id_spice_ma, id_pred_ma, _, _ = _run_timed_sweep(
        model, dataset, spice_dataset, p_tensor, time_grid, vg, vd, vs, vb, w_um, l_um, device
    )
    return id_spice_ma, id_pred_ma


def _generate_summary_table(all_metrics: dict, output_dir: Path):
    """
    Generates a summary table of all metrics as text and console output.

    :param all_metrics: Nested dict of metrics by geometry and waveform.
    :param output_dir: Output directory for summary files.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print()
    table = Table(title="Comprehensive SPICE Evaluation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Geometry", style="cyan", width=10)
    table.add_column("Ramp R²", justify="right", style="green")
    table.add_column("Ramp SubTh-R²", justify="right", style="yellow")
    table.add_column("Sweep R²", justify="right", style="green")
    table.add_column("Random R²", justify="right", style="green")
    table.add_column("Ramp MAE", justify="right", style="blue")
    table.add_column("Ramp MALE", justify="right", style="magenta")
    for geom_name, metrics in all_metrics.items():
        table.add_row(
            geom_name,
            f"{metrics.get('ramp_r2', 0):.4f}",
            f"{metrics.get('ramp_r2_subth', 0):.4f}",
            f"{metrics.get('sweep_r2', 0):.4f}",
            f"{metrics.get('random_r2', 0):.4f}",
            f"{metrics.get('ramp_mae_ua', 0):.2f}uA",
            f"{metrics.get('ramp_male_ua', 0):.2f}uA",
        )
    console.print(table)
    output = buffer.getvalue()
    for line in output.splitlines():
        logger.info(line)
    summary_path = output_dir / "summary_metrics.txt"
    with open(summary_path, "w") as f:
        f.write("COMPREHENSIVE SPICE EVALUATION SUMMARY\n")
        f.write("=" * 130 + "\n\n")
        f.write(
            f"{'Geometry':<10} | {'Ramp R2':<10} | {'Ramp SubTh-R2':<14} | {'Sweep R2':<10} | {'Random R2':<10} | "
            f"{'Ramp MAE':<12} | {'Ramp MALE':<12} | {'Sweep MAE':<12} | {'Sweep MALE':<12} | {'Random MAE':<12} | {'Random MALE':<12}\n"
        )
        f.write("-" * 160 + "\n")
        for geom_name, metrics in all_metrics.items():
            f.write(
                f"{geom_name:<10} | {metrics.get('ramp_r2', 0):>10.4f} | {metrics.get('ramp_r2_subth', 0):>14.4f} | "
                f"{metrics.get('sweep_r2', 0):>10.4f} | {metrics.get('random_r2', 0):>10.4f} | "
                f"{metrics.get('ramp_mae_ua', 0):>10.2f}uA | {metrics.get('ramp_male_ua', 0):>10.2f}uA | "
                f"{metrics.get('sweep_mae_ua', 0):>10.2f}uA | {metrics.get('sweep_male_ua', 0):>10.2f}uA | "
                f"{metrics.get('random_mae_ua', 0):>10.2f}uA | {metrics.get('random_male_ua', 0):>10.2f}uA\n"
            )
    logger.info("Summary saved: %s", summary_path)


def log_evaluation_summary(r2_fast, metrics_spice, comprehensive_metrics):
    """
    Logs standardized evaluation summary to console using Rich tables.

    Displays all key metrics from the full evaluation suite:
    - Fast dataset R²
    - SPICE transfer/output curves with sub-threshold R²
    - Comprehensive multi-geometry results

    :param r2_fast: R² score from fast dataset evaluation.
    :param metrics_spice: Dictionary of SPICE validation metrics.
    :param comprehensive_metrics: Nested dict of comprehensive evaluation metrics.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=100)
    console.print()
    console.rule("[bold cyan]EVALUATION SUMMARY[/bold cyan]", style="cyan")
    console.print()
    main_table = Table(title="Core Metrics", show_header=True, header_style="bold magenta")
    main_table.add_column("Metric", style="cyan", width=30)
    main_table.add_column("Value", justify="right", style="green")
    main_table.add_row("Fast Dataset R²", f"{r2_fast:.4f}")
    main_table.add_row("SPICE Transfer R²", f"{metrics_spice.get('r2_transfer', 0):.4f}")
    main_table.add_row("SPICE Transfer SubTh-R²", f"{metrics_spice.get('r2_transfer_subth', 0):.4f}")
    main_table.add_row("SPICE Transfer MALE", f"{metrics_spice.get('male_transfer', 0):.2f}uA")
    main_table.add_row("SPICE Output R²", f"{metrics_spice.get('r2_output', 0):.4f}")
    main_table.add_row("SPICE Output MALE", f"{metrics_spice.get('male_output', 0):.2f}uA")
    main_table.add_row("Speedup", f"{metrics_spice.get('timing_speedup_x', 0):.0f}x")
    console.print(main_table)
    console.print()
    comp_table = Table(title="Comprehensive Multi-Geometry Results", show_header=True, header_style="bold magenta")
    comp_table.add_column("Geometry", style="cyan", width=12)
    comp_table.add_column("Ramp R²", justify="right", style="green")
    comp_table.add_column("Ramp SubTh-R²", justify="right", style="yellow")
    comp_table.add_column("Sweep R²", justify="right", style="green")
    comp_table.add_column("Random R²", justify="right", style="green")
    for geom_name, geom_metrics in comprehensive_metrics.items():
        comp_table.add_row(
            geom_name,
            f"{geom_metrics.get('ramp_r2', 0):.4f}",
            f"{geom_metrics.get('ramp_r2_subth', 0):.4f}",
            f"{geom_metrics.get('sweep_r2', 0):.4f}",
            f"{geom_metrics.get('random_r2', 0):.4f}",
        )
    console.print(comp_table)
    console.print()
    console.rule(style="cyan")
    output = buffer.getvalue()
    for line in output.splitlines():
        logger.info(line)
