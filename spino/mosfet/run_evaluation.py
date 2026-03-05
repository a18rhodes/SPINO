#!/usr/bin/env python3
"""
Standalone evaluation script for trained MOSFET FNO models.

Runs the exact same evaluation suite as training's final evaluation:
1. Fast dataset-based I-V curve sampling
2. SPICE-based single-geometry I-V sweeps
3. Comprehensive multi-geometry (3x3) validation

Usage:
    python -m spino.mosfet.run_evaluation \
        --model-path /app/spino/models/mosfet/mosfet_spice_supervised_xyz.pt \
        --dataset-path /app/datasets/sky130_nmos_25k.h5
"""

import logging
import sys
from pathlib import Path

import click
import torch
from torch.utils.tensorboard import SummaryWriter

from spino.config import PathConfig
from spino.mosfet.gen_data import ParameterSchema, PreGeneratedMosfetDataset
from spino.mosfet.model import MosfetFNO
from spino.mosfet.train import run_final_evaluations
from spino.mosfet.evaluate import DEFAULT_TRIM_EVAL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to trained .pt model file.",
)
@click.option(
    "--dataset-path",
    default="/app/datasets/sky130_nmos_25k.h5",
    type=click.Path(exists=True),
    help="Path to HDF5 dataset (for normalization stats).",
)
@click.option(
    "--log-tensorboard/--no-tensorboard",
    default=True,
    help="Log results to TensorBoard.",
)
@click.option("--modes", default=128, help="Number of Fourier modes (must match trained model).")
@click.option("--width", default=64, help="FNO hidden width (must match trained model).")
@click.option("--embedding-dim", default=16, help="Physics embedding dimension (must match trained model).")
@click.option(
    "--tensorboard-suffix",
    default="",
    help="Suffix to append to tensorboard log dir (e.g. for hyperparameter variants).",
)
@click.option(
    "--trim-eval", default=DEFAULT_TRIM_EVAL, help="Timesteps to discard from eval start (SPICE .op artifact)."
)
def run_evaluation(
    model_path: str,
    dataset_path: str,
    log_tensorboard: bool,
    modes: int,
    width: int,
    embedding_dim: int,
    tensorboard_suffix: str,
    trim_eval: int,
):
    """
    Runs standardized post-training evaluation suite.

    Uses the same evaluation pipeline as training's final evaluation:
    - Fast dataset I-V curve sampling
    - SPICE-based I-V sweeps (single geometry)
    - Comprehensive multi-geometry validation (3 geometries x 3 waveforms)

    :param model_path: Path to trained .pt model file.
    :param dataset_path: Path to HDF5 dataset.
    :param log_tensorboard: Whether to log to TensorBoard.
    :param modes: Number of Fourier modes.
    :param width: FNO hidden width.
    :param embedding_dim: Physics embedding dimension.
    :param trim_eval: Number of initial timesteps to discard from SPICE .op artifact.
    """
    model_path = Path(model_path)
    run_name = model_path.stem + tensorboard_suffix
    path_config = PathConfig("mosfet")
    logger.info("Loading model: %s", model_path)
    input_param_dim = ParameterSchema.input_dim()
    model = MosfetFNO(
        input_param_dim=input_param_dim,
        embedding_dim=embedding_dim,
        modes=modes,
        width=width,
    ).cuda()
    state_dict = torch.load(model_path, weights_only=False)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Loading dataset for normalization stats: %s", dataset_path)
    with PreGeneratedMosfetDataset(dataset_path) as dataset:
        writer = None
        if log_tensorboard:
            tb_log_dir = path_config.run_dir / run_name
            writer = SummaryWriter(log_dir=tb_log_dir)
            logger.info("TensorBoard log dir: %s", tb_log_dir)
        logger.info("=" * 80)
        logger.info("Running standardized evaluation suite (same as training final eval)...")
        logger.info("=" * 80)
        run_final_evaluations(
            model=model,
            dataset=dataset,
            path_config=path_config,
            run_name=run_name,
            writer=writer,
            n_epochs=0,
            trim_eval=trim_eval,
        )
        if writer:
            writer.close()
        figure_dir = path_config.figure_dir / "training" / run_name
        logger.info("Figures saved to: %s", figure_dir)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run_evaluation()
