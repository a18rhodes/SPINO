"""
Standalone evaluation CLI for the diode neural operator.

Loads a trained DiodeFNO checkpoint and runs three evaluation suites:

- **Rectifier benchmark:** Standard R=1kΩ, C=10nF circuit vs. SPICE ground truth.
- **Resolution invariance:** Same circuit at 1024, 2048, and 4096 grid points.
- **Variable T_end:** Same circuit at 100µs, 1ms, and 10ms simulation windows.

Figures are saved to ``spino/figures/diode/<run_name>/``.
Metrics are printed to stdout and written to ``<figures_dir>/metrics.txt``.
"""

import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from spino.config import PathConfig
from spino.diode.evaluate import evaluate_adversarial, evaluate_rectifier, evaluate_resolution_invariance, evaluate_variable_t_end
from spino.diode.gen_data import PreGeneratedDiodeDataset
from spino.diode.model import DiodeFNO

__all__ = ["run_evaluation"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_FIGURES_BASE = PathConfig("diode").figure_dir


def _load_model(checkpoint_path: Path, device: str) -> DiodeFNO:
    """
    Loads a DiodeFNO from a checkpoint file.

    Supports both raw state_dict and the dict-wrapped format written by train.py:
    ``{"state_dict": ..., "model_config": {...}}``.

    :param checkpoint_path: Path to the ``.pt`` checkpoint.
    :param device: Torch device string.
    :return: Loaded DiodeFNO in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("model_config", {})
        model = DiodeFNO(
            in_channels=cfg.get("in_channels", 6),
            n_modes=(cfg.get("modes", 256),),
            hidden_channels=cfg.get("width", 64),
        )
        state = {k: v for k, v in ckpt["state_dict"].items() if not k.startswith("_")}
        model.load_state_dict(state)
    else:
        model = DiodeFNO()
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def _write_metrics(out_path: Path, sections: list[tuple[str, dict]]) -> None:
    """
    Writes a plain-text metrics report.

    :param out_path: Destination file path.
    :param sections: List of (section_label, metrics_dict) tuples.
    """
    lines = []
    for label, metrics in sections:
        lines.append(f"[{label}]")
        for k, v in metrics.items():
            lines.append(f"  {k}: {v:.6g}")
        lines.append("")
    out_path.write_text("\n".join(lines))
    logger.info("Metrics written to: %s", out_path)


@click.command()
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
@click.option("--device", default="cuda", show_default=True, help="Torch device.")
@click.option("--dark/--light", default=True, show_default=True, help="Plot colour scheme.")
@click.option("--out-dir", default=None, type=click.Path(path_type=Path), help="Override output directory.")
@click.option("--dataset-path", default=None, type=click.Path(exists=True, path_type=Path), help="HDF5 dataset for adversarial eval.")
def run_evaluation(checkpoint_path: Path, device: str, dark: bool, out_dir: Path | None, dataset_path: Path | None) -> None:
    """
    Runs the full diode evaluation suite on CHECKPOINT_PATH.

    CHECKPOINT_PATH: Path to the .pt model checkpoint (e.g. spino/models/diode/diode_dimless_v2_VokyITJR.pt).
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable, falling back to CPU.")
        device = "cpu"
    run_name = checkpoint_path.stem
    figures_dir = out_dir or (_FIGURES_BASE / run_name)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading checkpoint: %s", checkpoint_path)
    model = _load_model(checkpoint_path, device)
    logger.info("Model loaded. Device: %s", device)
    all_metrics: list[tuple[str, dict]] = []
    logger.info("--- Rectifier Benchmark ---")
    fig_rect, metrics_rect = evaluate_rectifier(model, dimensionless=True, device=device, dark=dark)
    rect_path = figures_dir / "rectifier.png"
    fig_rect.savefig(rect_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", rect_path)
    all_metrics.append(("rectifier R=1k C=10nF lambda=0.01", metrics_rect))
    logger.info("--- Resolution Invariance (1024 / 2048 / 4096) ---")
    fig_res, r2_res = evaluate_resolution_invariance(model, dimensionless=True, device=device, dark=dark)
    res_path = figures_dir / "resolution_invariance.png"
    fig_res.savefig(res_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", res_path)
    for t_steps, r2 in r2_res.items():
        all_metrics.append((f"resolution T={t_steps}", {"r2": r2}))
    logger.info("--- Variable T_end (100us / 1ms / 10ms) ---")
    fig_te, metrics_te = evaluate_variable_t_end(model, dimensionless=True, device=device, dark=dark)
    te_path = figures_dir / "variable_t_end.png"
    fig_te.savefig(te_path, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", te_path)
    for t_end, m in metrics_te.items():
        t_end_label = f"{t_end * 1e6:.0f}us" if t_end < 1e-3 else f"{t_end * 1e3:.0f}ms"
        all_metrics.append((f"variable_t_end T={t_end_label}", m))
    _write_metrics(figures_dir / "metrics.txt", all_metrics)
    if dataset_path is not None:
        logger.info("--- Adversarial Sample ---")
        dataset = PreGeneratedDiodeDataset(str(dataset_path))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        fig_adv, metrics_adv = evaluate_adversarial(model, loader, dimensionless=True, device=device, dark=dark)
        if fig_adv is not None:
            adv_path = figures_dir / "adversarial.png"
            fig_adv.savefig(adv_path, dpi=150, bbox_inches="tight")
            logger.info("Saved: %s", adv_path)
            all_metrics.append(("adversarial", metrics_adv))
            _write_metrics(figures_dir / "metrics.txt", all_metrics)
    logger.info("=== Summary ===")
    logger.info("Rectifier   : R2=%.4f, MAE=%.2fmV", metrics_rect["r2"], metrics_rect["mae_mv"])
    logger.info("Resolution  : %s", {k: f"{v:.4f}" for k, v in r2_res.items()})
    logger.info("Variable T  : %s", {(f"{k * 1e6:.0f}us" if k < 1e-3 else f"{k * 1e3:.0f}ms"): f"R2={v['r2']:.4f}" for k, v in metrics_te.items()})


if __name__ == "__main__":
    run_evaluation()
