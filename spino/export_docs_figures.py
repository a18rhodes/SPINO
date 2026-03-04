"""
Generates publication-quality (light-background) evaluation figures for all three devices.

This is the single authoritative entry point for producing figures destined for
documentation. All evaluation functions are called with dark=False so the resulting
PNGs use white backgrounds suitable for markdown and PDF rendering.

Run once per release or after retraining any device model.
"""

import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader

from spino.constants import MODELS_ROOT
from spino.diode.evaluate import evaluate_adversarial, evaluate_rectifier
from spino.diode.gen_data import InfiniteSpiceDiodeDataset
from spino.diode.model import get_model as _get_diode_model
from spino.mosfet.evaluate import evaluate_comprehensive, evaluate_sample_iv_curves, evaluate_spice_iv_sweeps
from spino.mosfet.gen_data import ParameterSchema, PreGeneratedMosfetDataset
from spino.mosfet.model import MosfetVCFiLMFNO
from spino.rc.evaluate import evaluate_adversarial_spectrum, evaluate_ic_spectrum, evaluate_ood_physics
from spino.rc.model import get_model as _get_rc_model

__all__ = ["export_mosfet_figures", "export_rc_figures", "export_diode_figures"]

# Production MOSFET checkpoint.
# Run ID: wtmjf8yn (W&B). Exp 19b full fine-tune on sky130_nmos_61k_plus_shortch_supp8k.h5.
# See CURRENT_STATUS.md -- 2026-03-03 entry for full metrics.
# This is a pre-config-embedding checkpoint; architecture params are hardcoded below.
_MOSFET_CHECKPOINT = MODELS_ROOT / "mosfet" / "mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt"
_DEFAULT_MOSFET_DATASET = Path("/app/datasets/sky130_nmos_61k_plus_shortch_supp8k.h5")
# Hardcoded fallback for legacy checkpoints that predate embedded config saving.
# Matches the architecture trained in Exp 16b and fine-tuned in Exp 19b.
_MOSFET_FALLBACK_CFG: dict = {"modes": 256, "width": 64, "embedding_dim": 16}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _latest_checkpoint(model_dir: Path) -> Path:
    """
    Returns the most recently modified .pt file in model_dir.

    :param model_dir: Directory containing checkpoint files.
    :return: Path to the latest checkpoint.
    :raises FileNotFoundError: If no .pt files exist.
    """
    candidates = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in {model_dir}")
    return candidates[-1]


def export_mosfet_figures(output_dir: Path, device: str, dataset_path: Path | None = None) -> None:
    """
    Loads the production MOSFET checkpoint and writes light-background figures.

    Generates three figure sets:
    - sample_iv.png: random dataset sample I-V curves
    - core_iv_sweeps.png: SPICE-validated transfer + output sweeps at W=1.0µm, L=0.18µm
    - comprehensive/comprehensive_{tiny,medium,xlarge}.png: full 3x3 waveform grid per geometry

    :param output_dir: Destination directory for all MOSFET documentation figures.
    :param device: Torch device string (e.g. "cuda" or "cpu").
    :param dataset_path: Path to the MOSFET HDF5 dataset; defaults to the production dataset.
    """
    resolved_dataset = dataset_path or _DEFAULT_MOSFET_DATASET
    if not resolved_dataset.exists():
        raise FileNotFoundError(f"MOSFET dataset not found: {resolved_dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading MOSFET checkpoint: %s", _MOSFET_CHECKPOINT)
    raw = torch.load(_MOSFET_CHECKPOINT, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        cfg = raw.get("model_config", _MOSFET_FALLBACK_CFG)
        logger.info("Embedded config found: %s", cfg)
    else:
        state_dict = raw
        cfg = _MOSFET_FALLBACK_CFG
        logger.warning("Legacy checkpoint — using hardcoded fallback config: %s", cfg)
    model = MosfetVCFiLMFNO(
        input_param_dim=ParameterSchema.input_dim(),
        embedding_dim=cfg["embedding_dim"],
        modes=cfg["modes"],
        width=cfg["width"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("MOSFET model loaded. Generating documentation figures...")
    with PreGeneratedMosfetDataset(str(resolved_dataset)) as dataset:
        fig_sample, _ = evaluate_sample_iv_curves(model, dataset, device=device, dark=False)
        fig_sample.savefig(output_dir / "sample_iv.png", dpi=150, bbox_inches="tight")
        fig_sample.clf()
        logger.info("Saved: %s/sample_iv.png", output_dir)
        fig_core, _ = evaluate_spice_iv_sweeps(model, dataset, device=device, w_um=1.0, l_um=0.18, dark=False)
        fig_core.savefig(output_dir / "core_iv_sweeps.png", dpi=150, bbox_inches="tight")
        fig_core.clf()
        logger.info("Saved: %s/core_iv_sweeps.png", output_dir)
        comprehensive_dir = output_dir / "comprehensive"
        comprehensive_dir.mkdir(exist_ok=True)
        evaluate_comprehensive(model, dataset, output_dir=comprehensive_dir, device=device, dark=False, save_summary=False)
    logger.info("MOSFET figures complete.")


def export_rc_figures(output_dir: Path, device: str) -> None:
    """
    Loads the latest RC checkpoint and writes light-background figures.

    Generates:
    - ic_spectrum.png: IC spectrum sweep across tau corners
    - adversarial.png: adversarial / OOD stress test
    - ood_physics.png: OOD chirp and sawtooth physics test

    :param output_dir: Destination directory for RC documentation figures.
    :param device: Torch device string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _latest_checkpoint(MODELS_ROOT / "simple_rc")
    logger.info("Loading RC checkpoint: %s", ckpt)
    model = _get_rc_model()
    raw_ckpt = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(raw_ckpt["state_dict"] if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt else raw_ckpt)
    model.to(device).eval()
    logger.info("RC model loaded. Generating documentation figures...")
    fig_ic, _ = evaluate_ic_spectrum(model, device=device, dark=False)
    fig_ic.savefig(output_dir / "ic_spectrum.png", dpi=150, bbox_inches="tight")
    fig_ic.clf()
    logger.info("Saved: %s/ic_spectrum.png", output_dir)
    fig_adv, _ = evaluate_adversarial_spectrum(model, device=device, dark=False)
    fig_adv.savefig(output_dir / "adversarial.png", dpi=150, bbox_inches="tight")
    fig_adv.clf()
    logger.info("Saved: %s/adversarial.png", output_dir)
    fig_ood, _ = evaluate_ood_physics(model, device=device, dark=False)
    fig_ood.savefig(output_dir / "ood_physics.png", dpi=150, bbox_inches="tight")
    fig_ood.clf()
    logger.info("RC figures complete.")


def export_diode_figures(output_dir: Path, device: str) -> None:
    """
    Loads the latest diode checkpoint and writes light-background figures.

    Generates:
    - rectifier.png: SPICE-validated standard rectifier comparison
    - adversarial.png: random adversarial sample from the infinite SPICE generator

    :param output_dir: Destination directory for diode documentation figures.
    :param device: Torch device string.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = _latest_checkpoint(MODELS_ROOT / "diode")
    logger.info("Loading diode checkpoint: %s", ckpt)
    model = _get_diode_model()
    raw_ckpt = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(raw_ckpt["state_dict"] if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt else raw_ckpt)
    model.to(device).eval()
    logger.info("Diode model loaded. Generating documentation figures...")
    fig_rect, _ = evaluate_rectifier(model, device=device, dark=False)
    fig_rect.savefig(output_dir / "rectifier.png", dpi=150, bbox_inches="tight")
    fig_rect.clf()
    logger.info("Saved: %s/rectifier.png", output_dir)
    adv_loader = DataLoader(InfiniteSpiceDiodeDataset(t_steps=1024), batch_size=8)
    fig_adv, _ = evaluate_adversarial(model, adv_loader, device=device, dark=False)
    fig_adv.savefig(output_dir / "adversarial.png", dpi=150, bbox_inches="tight")
    fig_adv.clf()
    logger.info("Diode figures complete.")


@click.command()
@click.option(
    "--docs-assets",
    default="docs/assets",
    show_default=True,
    type=click.Path(),
    help="Root directory for documentation assets.",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    show_default=True,
    help="Torch device to use for inference.",
)
@click.option("--mosfet/--no-mosfet", default=True, help="Export MOSFET figures.")
@click.option("--rc/--no-rc", default=True, help="Export RC circuit figures.")
@click.option("--diode/--no-diode", default=True, help="Export diode figures.")
@click.option(
    "--dataset-path",
    default=None,
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to MOSFET HDF5 dataset. Defaults to the production dataset path.",
)
def main(docs_assets: str, device: str, mosfet: bool, rc: bool, diode: bool, dataset_path: Path | None) -> None:
    """
    Exports light-background evaluation figures for documentation.

    Writes PNGs to docs/assets/{mosfet,simple_rc,diode}/ ready for embedding
    in markdown documentation files.
    """
    root = Path(docs_assets)
    if mosfet:
        export_mosfet_figures(root / "mosfet", device, dataset_path=dataset_path)
    if rc:
        export_rc_figures(root / "simple_rc", device)
    if diode:
        export_diode_figures(root / "diode", device)
    logger.info("All requested figures exported to: %s", root)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
