"""
Generates light-mode documentation plots for MOSFET operators.

Produces the same plot types as the training-time dark-mode evaluation
(sample I-V, core SPICE sweeps, comprehensive 3x3 grids) but rendered
with a white-background palette suitable for documentation.

Usage:
    python -m scripts.generate_doc_plots --device-type pfet
    python -m scripts.generate_doc_plots --device-type nfet
"""
import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Must precede pyplot import; backend requires no display
import matplotlib.pyplot as plt
import numpy as np
import torch

from spino.constants import ARCSINH_SCALE_MA
from spino.mosfet.device_strategy import DeviceStrategy
from spino.mosfet.evaluate import (
    DEFAULT_TRIM_EVAL,
    _apply_eval_trim,
    _build_p_tensor,
    _extract_geometry_label,
    _infer_and_denormalize_sample,
    _run_single_sweep,
    _run_timed_sweep,
    calculate_male,
    calculate_r2,
    calculate_subthreshold_r2,
)
from spino.mosfet.gen_data import (
    GEOMETRY_BINS,
    InfiniteSpiceMosfetDataset,
    ParameterSchema,
    PreGeneratedMosfetDataset,
)
from spino.mosfet.model import MosfetVCFiLMFNO

__all__ = ["DEVICE_CONFIGS", "generate_comprehensive", "generate_core_sweeps", "generate_sample_iv"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

SPICE_COLOR = "#333333"
FNO_TRANSFER_COLOR = "#0077bb"
FNO_SWEEP_COLOR = "#bb3388"
FNO_RANDOM_COLOR = "#228833"
SUBTH_LINE_COLOR = "#cc6600"
ERROR_ABS_COLOR = "#cc3333"
ERROR_REL_COLOR = "#118888"
VG_COLOR = "#228833"
VD_COLOR = "#bb3388"
VS_COLOR = "#cc6600"
PARITY_COLOR = "#0077bb"
PARITY_SWEEP_COLOR = "#bb3388"
PARITY_RANDOM_COLOR = "#228833"
PERFECT_LINE_COLOR = "#cc3333"

DEVICE_CONFIGS = {
    "pfet": {
        "model_path": "spino/models/mosfet/pfet/mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt",
        "dataset_path": "datasets/sky130_pmos_48k_sweep_aug.h5",
        "strategy_name": "sky130_pmos",
        "output_dir": "docs/assets/mosfet/pfet",
        "trim_startup": 41,
        "modes": 256,
    },
    "nfet": {
        "model_path": "spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt",
        "dataset_path": "datasets/sky130_nmos_61k_plus_shortch_supp8k.h5",
        "strategy_name": "sky130_nmos",
        "output_dir": "docs/assets/mosfet/nfet",
        "trim_startup": 41,
        "modes": 256,
    },
}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", color="black")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_pred - y_true) / (np.linalg.norm(y_true) + 1e-12)


def _plot_error(ax, x, y_true, y_pred, xlabel, title_prefix):
    abs_error = y_pred - y_true
    rel_error_pct = np.clip(100.0 * abs_error / (np.abs(y_true) + 1e-9), -500, 500)
    ax.plot(x, abs_error * 1000.0, color=ERROR_ABS_COLOR, linewidth=1.5, label="Abs (uA)")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Absolute Error (uA)", fontsize=10, color=ERROR_ABS_COLOR)
    ax.tick_params(axis="y", labelcolor=ERROR_ABS_COLOR, labelsize=9)
    ax_rel = ax.twinx()
    ax_rel.plot(x, rel_error_pct, color=ERROR_REL_COLOR, linewidth=1.2, linestyle="--", alpha=0.8, label="Rel (%)")
    ax_rel.set_ylabel("Relative Error (%)", fontsize=10, color=ERROR_REL_COLOR)
    ax_rel.tick_params(axis="y", labelcolor=ERROR_REL_COLOR, labelsize=9)
    ax_rel.set_ylim(-100, 100)
    mae_ua = np.mean(np.abs(abs_error)) * 1000.0
    valid_mask = np.abs(y_true) > 0.001
    mape = np.mean(np.abs(rel_error_pct[valid_mask])) if valid_mask.any() else 0.0
    male_ua = calculate_male(y_true, y_pred)
    ax.set_title(f"{title_prefix}\nMAE={mae_ua:.2f}uA, MAPE={mape:.1f}%, MALE={male_ua:.2f}uA", fontsize=10, fontweight="bold", color="black")
    ax.legend(loc="upper left", fontsize=8)
    ax_rel.legend(loc="upper right", fontsize=8)


def _load_model_and_dataset(cfg, device):
    root = Path(__file__).resolve().parent.parent
    logger.info("Loading dataset: %s", cfg["dataset_path"])
    dataset = PreGeneratedMosfetDataset(
        hdf5_path=str(root / cfg["dataset_path"]),
        normalize=True,
        use_curated_params=True,
        trim_startup=cfg["trim_startup"],
    )
    logger.info("Dataset loaded: %d samples", len(dataset))
    model = MosfetVCFiLMFNO(
        input_param_dim=ParameterSchema.input_dim(),
        embedding_dim=16,
        modes=cfg["modes"],
        width=64,
    ).to(device)
    ckpt_path = root / cfg["model_path"]
    logger.info("Loading checkpoint: %s", ckpt_path)
    state_dict = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded (%d parameters).", sum(p.numel() for p in model.parameters()))
    return model, dataset


def generate_sample_iv(
    model: MosfetVCFiLMFNO,
    dataset: PreGeneratedMosfetDataset,
    output_dir: Path,
    device: str = "cuda",
) -> None:
    """
    Renders a 4-panel sample I-V quality plot for a single random waveform.

    Panels: transient response overlay, terminal voltages, parity scatter, and
    I-V snapshot (Id vs Vd). Saved to ``output_dir/sample_iv.png``.

    :param model: Trained MOSFET surrogate model.
    :param dataset: Pre-generated dataset supplying normalization statistics.
    :param output_dir: Directory where the output PNG is written.
    :param device: Torch device identifier (``"cuda"`` or ``"cpu"``).
    """
    logger.info("Generating sample I-V plot...")
    np.random.seed(42)
    sample_idx = np.random.randint(0, len(dataset))
    current_true_ma, current_pred_ma, vg, vd, vs, physics_raw = _infer_and_denormalize_sample(model, dataset, sample_idx, device)
    w_um, l_um, vth0 = _extract_geometry_label(physics_raw)
    mse = np.mean((current_true_ma - current_pred_ma) ** 2)
    r2 = calculate_r2(current_true_ma, current_pred_ma)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    time_us = np.linspace(0, 1, len(current_true_ma))
    ax = axes[0, 0]
    _style(ax, f"MOSFET Transient Response\nW={w_um:.2f}\u00b5m, L={l_um:.2f}\u00b5m, Vth0={vth0:.3f}V | MSE={mse:.2e}, R\u00b2={r2:.4f}", "Normalized Time", "Current (mA)")
    ax.plot(time_us, current_true_ma, color=SPICE_COLOR, linewidth=2, alpha=0.7, label="Ground Truth")
    ax.plot(time_us, current_pred_ma, color=FNO_TRANSFER_COLOR, linestyle=":", linewidth=2, label="FNO Prediction")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[0, 1]
    _style(ax, "Terminal Voltages", "Normalized Time", "Voltage (V)")
    ax.plot(time_us, vg, color=VG_COLOR, label="Vg (Gate)", linewidth=1.5)
    ax.plot(time_us, vd, color=VD_COLOR, label="Vd (Drain)", linewidth=1.5)
    ax.plot(time_us, vs, color=VS_COLOR, label="Vs (Source)", linewidth=1.5, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1, 0]
    _style(ax, f"Parity Plot: Current Prediction | R\u00b2={r2:.4f}", "True Id (mA)", "Predicted Id (mA)")
    ax.scatter(current_true_ma, current_pred_ma, c=PARITY_COLOR, s=15, alpha=0.6, edgecolors="gray", linewidth=0.5)
    lim_min, lim_max = min(current_true_ma.min(), current_pred_ma.min()), max(current_true_ma.max(), current_pred_ma.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color=PERFECT_LINE_COLOR, linestyle="--", linewidth=1.5, alpha=0.7, label="Perfect (y=x)")
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1, 1]
    _style(ax, "I-V Snapshot: Id vs Vd (Output)", "Vd (V)", "Id (mA)")
    ax.scatter(vd, current_true_ma, c=SPICE_COLOR, s=10, alpha=0.3, label="True")
    ax.scatter(vd, current_pred_ma, c=FNO_TRANSFER_COLOR, s=10, alpha=0.5, marker="x", label="Pred")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out_path = output_dir / "sample_iv.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s (R2=%.4f)", out_path, r2)


def generate_core_sweeps(
    model: MosfetVCFiLMFNO,
    dataset: PreGeneratedMosfetDataset,
    output_dir: Path,
    device: str = "cuda",
    strategy_name: str = "sky130_nmos",
    w_um: float = 1.0,
    l_um: float = 0.18,
) -> None:
    """
    Renders SPICE-vs-FNO transfer and output sweep plots for a single geometry.

    Produces a 2x2 panel figure (transfer curve, transfer error, output curve,
    output error) and saves it to ``output_dir/core_iv_sweeps.png``. Logs
    SPICE and FNO wall-clock times plus the speedup ratio.

    :param model: Trained MOSFET surrogate model.
    :param dataset: Pre-generated dataset supplying normalization statistics.
    :param output_dir: Directory where the output PNG is written.
    :param device: Torch device identifier.
    :param strategy_name: Device strategy key (e.g. ``"sky130_nmos"``).
    :param w_um: Gate width in micrometres.
    :param l_um: Gate length in micrometres.
    """
    logger.info("Generating core I-V sweeps (W=%.1f, L=%.2f)...", w_um, l_um)
    t_steps = 512
    trim_eval = DEFAULT_TRIM_EVAL
    raw_steps = t_steps + trim_eval
    ec = DeviceStrategy.create(strategy_name).eval_config
    spice_dataset = InfiniteSpiceMosfetDataset(strategy_name=ec.strategy_name, t_steps=raw_steps, t_end=raw_steps * 1e-9)
    time_grid = np.linspace(0, spice_dataset.t_end, raw_steps)
    vs_bias = np.full(raw_steps, ec.vs_bias)
    vb_bias = np.full(raw_steps, ec.vb_bias)
    p_tensor = _build_p_tensor(spice_dataset, dataset, w_um, l_um, device)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    vg_sweep = np.linspace(ec.transfer_vg_start, ec.transfer_vg_stop, raw_steps)
    vd_sat = np.full(raw_steps, ec.transfer_vd_bias)
    logger.info("  Running transfer sweep (Vd=%.1fV)...", ec.transfer_vd_bias)
    id_spice_t, id_pred_t, spice_ms, fno_ms = _run_timed_sweep(model, dataset, spice_dataset, p_tensor, time_grid, vg_sweep, vd_sat, vs_bias, vb_bias, w_um, l_um, device)
    if id_spice_t is None:
        logger.error("  SPICE failed for transfer sweep.")
        plt.close(fig)
        return
    vg_plot, id_spice_t, id_pred_t = _apply_eval_trim(vg_sweep, id_spice_t, id_pred_t, trim=trim_eval)
    r2_transfer = calculate_r2(id_spice_t, id_pred_t)
    r2_subth = calculate_subthreshold_r2(vg_plot, id_spice_t, id_pred_t, vg_threshold=ec.subth_vg_threshold, below=ec.subth_below)
    l2_transfer = _l2_relative_error(id_spice_t, id_pred_t)
    subth_str = f", SubTh-R\u00b2={r2_subth:.4f}" if r2_subth is not None else ""
    _style(axes[0, 0], f"Id-Vg Transfer (Saturation)\nW={w_um}\u00b5m, L={l_um}\u00b5m | R\u00b2={r2_transfer:.4f}{subth_str}, L2={l2_transfer:.4f}\nVd={ec.transfer_vd_bias:.1f}V, Vs={ec.vs_bias:.1f}V, Vb={ec.vb_bias:.1f}V", "Vg (V)", "|Id| (mA)")
    axes[0, 0].plot(vg_plot, np.abs(id_spice_t), color=SPICE_COLOR, linewidth=2.5, alpha=0.7, label="SPICE")
    axes[0, 0].plot(vg_plot, np.abs(id_pred_t), color=FNO_TRANSFER_COLOR, linestyle=":", linewidth=2, label="FNO")
    axes[0, 0].set_yscale("log")
    subth_label = f"SubTh (Vg<{ec.subth_vg_threshold:.1f}V)" if ec.subth_below else f"SubTh (Vg>{ec.subth_vg_threshold:.1f}V)"
    axes[0, 0].axvline(ec.subth_vg_threshold, color=SUBTH_LINE_COLOR, linewidth=1, linestyle=":", alpha=0.5, label=subth_label)
    axes[0, 0].legend(loc="upper left", fontsize=9)
    _plot_error(axes[0, 1], vg_plot, id_spice_t, id_pred_t, "Vg (V)", "Transfer Error")
    vd_sweep = np.linspace(ec.output_vd_start, ec.output_vd_stop, raw_steps)
    vg_drive = np.full(raw_steps, ec.output_vg_drive)
    logger.info("  Running output sweep (Vg=%.1fV)...", ec.output_vg_drive)
    id_spice_o, id_pred_o, _, _ = _run_timed_sweep(model, dataset, spice_dataset, p_tensor, time_grid, vg_drive, vd_sweep, vs_bias, vb_bias, w_um, l_um, device)
    if id_spice_o is None:
        logger.error("  SPICE failed for output sweep.")
    else:
        vd_plot, id_spice_o, id_pred_o = _apply_eval_trim(vd_sweep, id_spice_o, id_pred_o, trim=trim_eval)
        r2_output = calculate_r2(id_spice_o, id_pred_o)
        l2_output = _l2_relative_error(id_spice_o, id_pred_o)
        _style(axes[1, 0], f"Id-Vd Output (Linear/Sat)\nVg={ec.output_vg_drive:.1f}V, Vs={ec.vs_bias:.1f}V, Vb={ec.vb_bias:.1f}V | R\u00b2={r2_output:.4f}, L2={l2_output:.4f}", "Vd (V)", "Id (mA)")
        axes[1, 0].plot(vd_plot, id_spice_o, color=SPICE_COLOR, linewidth=2.5, alpha=0.7, label="SPICE")
        axes[1, 0].plot(vd_plot, id_pred_o, color=FNO_SWEEP_COLOR, linestyle=":", linewidth=2, label="FNO")
        axes[1, 0].legend(loc="upper left", fontsize=9)
        _plot_error(axes[1, 1], vd_plot, id_spice_o, id_pred_o, "Vd (V)", "Output Error")
    plt.tight_layout()
    out_path = output_dir / "core_iv_sweeps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s (Transfer R2=%.4f, Output R2=%.4f)", out_path, r2_transfer, r2_output if id_spice_o is not None else 0.0)
    logger.info("  SPICE: %.1f ms | FNO: %.2f ms | Speedup: %.0fx", spice_ms, fno_ms, spice_ms / fno_ms if fno_ms > 0 else 0)


def _generate_comprehensive_single(model, dataset, w_um, l_um, geom_name, output_dir, device, strategy_name):
    logger.info("  Evaluating %s (W=%.2f, L=%.2f)...", geom_name.upper(), w_um, l_um)
    t_steps = 512
    trim_eval = DEFAULT_TRIM_EVAL
    raw_steps = t_steps + trim_eval
    ec = DeviceStrategy.create(strategy_name).eval_config
    spice_dataset = InfiniteSpiceMosfetDataset(strategy_name=ec.strategy_name, t_steps=raw_steps, t_end=raw_steps * 1e-9)
    time_grid = np.linspace(0, spice_dataset.t_end, raw_steps)
    vs_bias = np.full(raw_steps, ec.vs_bias)
    vb_bias = np.full(raw_steps, ec.vb_bias)
    p_tensor = _build_p_tensor(spice_dataset, dataset, w_um, l_um, device)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    metrics = {}
    vg_sweep = np.linspace(ec.transfer_vg_start, ec.transfer_vg_stop, raw_steps)
    vd_sat = np.full(raw_steps, ec.transfer_vd_bias)
    id_spice_ramp, id_pred_ramp = _run_single_sweep(model, dataset, spice_dataset, p_tensor, time_grid, vg_sweep, vd_sat, vs_bias, vb_bias, w_um, l_um, device)
    if id_spice_ramp is not None:
        vg_t, id_spice_ramp, id_pred_ramp = _apply_eval_trim(vg_sweep, id_spice_ramp, id_pred_ramp, trim=trim_eval)
        r2_ramp = calculate_r2(id_spice_ramp, id_pred_ramp)
        r2_ramp_subth = calculate_subthreshold_r2(vg_t, id_spice_ramp, id_pred_ramp, vg_threshold=ec.subth_vg_threshold, below=ec.subth_below)
        metrics["ramp_r2"] = r2_ramp
        metrics["ramp_r2_subth"] = r2_ramp_subth if r2_ramp_subth is not None else 0.0
        subth_str = f", SubTh={r2_ramp_subth:.4f}" if r2_ramp_subth is not None else ""
        _style(axes[0, 0], f"Ramp: Id-Vg | R2={r2_ramp:.4f}{subth_str}\nVd={ec.transfer_vd_bias:.1f}V", "Vg (V)", "|Id| (mA)")
        axes[0, 0].plot(vg_t, np.abs(id_spice_ramp), color=SPICE_COLOR, linewidth=2.5, alpha=0.7, label="SPICE")
        axes[0, 0].plot(vg_t, np.abs(id_pred_ramp), color=FNO_TRANSFER_COLOR, linestyle=":", linewidth=2, label="FNO")
        subth_label = f"SubTh (Vg<{ec.subth_vg_threshold:.1f}V)" if ec.subth_below else f"SubTh (Vg>{ec.subth_vg_threshold:.1f}V)"
        axes[0, 0].axvline(ec.subth_vg_threshold, color=SUBTH_LINE_COLOR, linewidth=1, linestyle=":", alpha=0.5, label=subth_label)
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend(loc="upper left", fontsize=9)
        _plot_error(axes[0, 1], vg_t, id_spice_ramp, id_pred_ramp, "Vg (V)", "Ramp Error")
        axes[0, 2].scatter(id_spice_ramp, id_pred_ramp, c=PARITY_COLOR, s=10, alpha=0.5)
        lims = [min(id_spice_ramp.min(), id_pred_ramp.min()), max(id_spice_ramp.max(), id_pred_ramp.max())]
        axes[0, 2].plot(lims, lims, color=PERFECT_LINE_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        _style(axes[0, 2], f"Ramp Parity | R2={r2_ramp:.4f}", "SPICE Id (mA)", "FNO Id (mA)")
    vd_sweep = np.linspace(ec.output_vd_start, ec.output_vd_stop, raw_steps)
    vg_drive = np.full(raw_steps, ec.output_vg_drive)
    id_spice_sweep, id_pred_sweep = _run_single_sweep(model, dataset, spice_dataset, p_tensor, time_grid, vg_drive, vd_sweep, vs_bias, vb_bias, w_um, l_um, device)
    if id_spice_sweep is not None:
        vd_t, id_spice_sweep, id_pred_sweep = _apply_eval_trim(vd_sweep, id_spice_sweep, id_pred_sweep, trim=trim_eval)
        r2_sweep = calculate_r2(id_spice_sweep, id_pred_sweep)
        metrics["sweep_r2"] = r2_sweep
        _style(axes[1, 0], f"Sweep: Id-Vd | R2={r2_sweep:.4f}\nVg={ec.output_vg_drive:.1f}V", "Vd (V)", "Id (mA)")
        axes[1, 0].plot(vd_t, id_spice_sweep, color=SPICE_COLOR, linewidth=2.5, alpha=0.7, label="SPICE")
        axes[1, 0].plot(vd_t, id_pred_sweep, color=FNO_SWEEP_COLOR, linestyle=":", linewidth=2, label="FNO")
        axes[1, 0].legend(loc="upper left", fontsize=9)
        _plot_error(axes[1, 1], vd_t, id_spice_sweep, id_pred_sweep, "Vd (V)", "Sweep Error")
        axes[1, 2].scatter(id_spice_sweep, id_pred_sweep, c=PARITY_SWEEP_COLOR, s=10, alpha=0.5)
        lims = [min(id_spice_sweep.min(), id_pred_sweep.min()), max(id_spice_sweep.max(), id_pred_sweep.max())]
        axes[1, 2].plot(lims, lims, color=PERFECT_LINE_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        _style(axes[1, 2], f"Sweep Parity | R2={r2_sweep:.4f}", "SPICE Id (mA)", "FNO Id (mA)")
    np.random.seed(42 + sum(ord(ch) for ch in geom_name))
    n_pts = np.random.randint(8, 15)
    pwl_times = np.sort(np.random.uniform(0, spice_dataset.t_end, n_pts))
    pwl_times = np.concatenate(([0], pwl_times, [spice_dataset.t_end]))
    vg_pwl_raw = np.random.uniform(*ec.random_vg_range, len(pwl_times))
    vd_pwl_raw = np.random.uniform(*ec.random_vd_range, len(pwl_times))
    vg_pwl = np.interp(time_grid, pwl_times, vg_pwl_raw)
    vd_pwl = np.interp(time_grid, pwl_times, vd_pwl_raw)
    id_spice_pwl, id_pred_pwl = _run_single_sweep(model, dataset, spice_dataset, p_tensor, time_grid, vg_pwl, vd_pwl, vs_bias, vb_bias, w_um, l_um, device)
    if id_spice_pwl is not None:
        time_t, id_spice_pwl, id_pred_pwl = _apply_eval_trim(time_grid, id_spice_pwl, id_pred_pwl, trim=trim_eval)
        vg_pwl, vd_pwl = _apply_eval_trim(vg_pwl, vd_pwl, trim=trim_eval)
        r2_pwl = calculate_r2(id_spice_pwl, id_pred_pwl)
        metrics["random_r2"] = r2_pwl
        time_us = time_t * 1e6
        _style(axes[2, 0], f"Random: PWL Transient | R2={r2_pwl:.4f}\nVg/Vd=PWL, Vs={ec.vs_bias:.1f}V", "Time (us)", "Id (mA)")
        axes[2, 0].plot(time_us, id_spice_pwl, color=SPICE_COLOR, linewidth=2, alpha=0.7, label="SPICE")
        axes[2, 0].plot(time_us, id_pred_pwl, color=FNO_RANDOM_COLOR, linestyle=":", linewidth=1.5, label="FNO")
        axes[2, 0].legend(loc="upper right", fontsize=9)
        ax_v = axes[2, 0].twinx()
        ax_v.plot(time_us, vg_pwl, color=VG_COLOR, linewidth=1, alpha=0.4, linestyle="--", label="Vg")
        ax_v.plot(time_us, vd_pwl, color=VD_COLOR, linewidth=1, alpha=0.4, linestyle=":", label="Vd")
        ax_v.set_ylabel("Voltage (V)", fontsize=9, color=VD_COLOR)
        ax_v.set_ylim(0, 2.0)
        ax_v.legend(loc="lower right", fontsize=8)
        _plot_error(axes[2, 1], time_us, id_spice_pwl, id_pred_pwl, "Time (us)", "Random Error")
        axes[2, 2].scatter(id_spice_pwl, id_pred_pwl, c=PARITY_RANDOM_COLOR, s=10, alpha=0.5)
        lims = [min(id_spice_pwl.min(), id_pred_pwl.min()), max(id_spice_pwl.max(), id_pred_pwl.max())]
        axes[2, 2].plot(lims, lims, color=PERFECT_LINE_COLOR, linestyle="--", linewidth=1, alpha=0.7)
        _style(axes[2, 2], f"Random Parity | R2={r2_pwl:.4f}", "SPICE Id (mA)", "FNO Id (mA)")
    fig.suptitle(f"Comprehensive: {geom_name.upper()} (W={w_um:.2f}um, L={l_um:.2f}um)", fontsize=14, fontweight="bold", color="black")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    comp_dir = output_dir / "comprehensive"
    comp_dir.mkdir(parents=True, exist_ok=True)
    out_path = comp_dir / f"comprehensive_{geom_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved: %s", out_path)
    return metrics


def generate_comprehensive(
    model: MosfetVCFiLMFNO,
    dataset: PreGeneratedMosfetDataset,
    output_dir: Path,
    device: str = "cuda",
    strategy_name: str = "sky130_nmos",
) -> None:
    """
    Renders 3x3 comprehensive evaluation grids for tiny, medium, and xlarge geometries.

    Each grid contains ramp (transfer), sweep (output), and random (PWL transient)
    rows, each with a curve overlay, error subplot, and parity scatter column.
    Saved to ``output_dir/comprehensive/<geom>.png``.

    :param model: Trained MOSFET surrogate model.
    :param dataset: Pre-generated dataset supplying normalization statistics.
    :param output_dir: Directory root for output PNGs.
    :param device: Torch device identifier.
    :param strategy_name: Device strategy key.
    """
    logger.info("Generating comprehensive plots...")
    test_geometries = {
        "tiny": (GEOMETRY_BINS["tiny"].w_range[0] + 0.05, GEOMETRY_BINS["tiny"].l_range[0] + 0.02),
        "medium": (2.5, 0.75),
        "xlarge": (8.0, 1.75),
    }
    for geom_name, (w_um, l_um) in test_geometries.items():
        metrics = _generate_comprehensive_single(model, dataset, w_um, l_um, geom_name, output_dir, device, strategy_name)
        if metrics:
            logger.info("    %s: Ramp=%.4f, Sweep=%.4f, Random=%.4f", geom_name, metrics.get("ramp_r2", 0), metrics.get("sweep_r2", 0), metrics.get("random_r2", 0))


def main() -> None:
    """
    CLI entry point for generating documentation plots.

    Parses ``--device-type`` (nfet or pfet) and ``--compute-device``, loads
    the corresponding checkpoint and dataset from ``DEVICE_CONFIGS``, then
    runs all three plot generators in sequence.
    """
    parser = argparse.ArgumentParser(description="Generate light-mode MOSFET doc plots.")
    parser.add_argument("--device-type", choices=list(DEVICE_CONFIGS.keys()), required=True, help="MOSFET type to generate plots for.")
    parser.add_argument(
        "--compute-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cuda/cpu). Defaults to cuda if available.",
    )
    args = parser.parse_args()
    cfg = DEVICE_CONFIGS[args.device_type]
    root = Path(__file__).resolve().parent.parent
    output_dir = root / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model, dataset = _load_model_and_dataset(cfg, args.compute_device)
    generate_sample_iv(model, dataset, output_dir, device=args.compute_device)
    generate_core_sweeps(model, dataset, output_dir, device=args.compute_device, strategy_name=cfg["strategy_name"])
    generate_comprehensive(model, dataset, output_dir, device=args.compute_device, strategy_name=cfg["strategy_name"])
    logger.info("All plots saved to: %s", output_dir)


if __name__ == "__main__":
    main()
