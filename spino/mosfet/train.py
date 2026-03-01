"""
Training script for the MOSFET Neural Operator (MosfetFNO).

This script orchestrates the training loop, including:
1. Pre-generated dataset loading via PreGeneratedMosfetDataset.
2. Hyperparameter & Strategy Configuration logging to TensorBoard.
3. Model Optimization (AdamW + Scheduler).
4. Checkpointing.
5. Early Stopping (when warm_restart_count=1).

Usage:
    python -m spino.mosfet.train --dataset /app/datasets/sky130_nmos_25k.h5 --batch_size 64
"""

import json
import logging
import sys
from collections import deque

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spino.config import PathConfig
from spino.loss import (
    ArcSinhMSELoss,
    LpLoss,
    LpLossWithFloor,
    Log10Loss,
    QuarticWeightedLoss,
    RegionAdaptiveLoss,
    SubthresholdWeightedLoss,
)
from spino.mosfet.evaluate import (
    evaluate_comprehensive,
    evaluate_sample_iv_curves,
    evaluate_spice_iv_sweeps,
    log_evaluation_summary,
    DEFAULT_TRIM_EVAL,
)
from spino.mosfet.gen_data import ParameterSchema, PreGeneratedMosfetDataset
from spino.mosfet.model import MosfetFNO, MosfetFiLMFNO, MosfetVCFiLMFNO
from spino.utils import generate_unique_id, timeit

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

plt.style.use("dark_background")


def _initialize_training_components(
    input_param_dim,
    embedding_dim,
    modes,
    width,
    learning_rate,
    weight_decay,
    n_epochs,
    warm_restart_count,
    loss_type="lp",
    loss_scale_ma=0.01,
    loss_exponent=2.0,
    denom_floor=10.0,
    model_type="concat",
    vg_mean=None,
    vg_std=None,
):
    """
    Initializes model, optimizer, scheduler, and loss function.

    :param input_param_dim: Number of input physics parameters.
    :param embedding_dim: Physics parameter embedding dimension.
    :param modes: Number of Fourier modes for FNO.
    :param width: Hidden channel width for FNO.
    :param learning_rate: Initial learning rate.
    :param weight_decay: AdamW weight decay coefficient.
    :param n_epochs: Total number of training epochs.
    :param warm_restart_count: Number of cosine annealing restarts.
    :param loss_type: Loss function type ('lp', 'mse', 'lp_floor', 'weighted', 'log10', 'quartic', 'region_adaptive').
    :param loss_scale_ma: Scale parameter for SubthresholdWeightedLoss (mA).
    :param loss_exponent: Exponent for SubthresholdWeightedLoss weighting curve.
    :param denom_floor: Minimum denominator for 'lp_floor' loss (arcsinh units).
    :param model_type: Architecture type ('concat', 'film', or 'vcfilm').
    :return: Tuple of (model, optimizer, scheduler, loss_fn).
    """
    if model_type == "vcfilm":
        model = MosfetVCFiLMFNO(
            input_param_dim=input_param_dim, embedding_dim=embedding_dim, modes=modes, width=width
        ).cuda()
        logger.info("Initialized MosfetVCFiLMFNO (Voltage-Conditioned FiLM architecture)")
    elif model_type == "film":
        model = MosfetFiLMFNO(
            input_param_dim=input_param_dim, embedding_dim=embedding_dim, modes=modes, width=width
        ).cuda()
        logger.info("Initialized MosfetFiLMFNO (FiLM architecture)")
    else:
        model = MosfetFNO(input_param_dim=input_param_dim, embedding_dim=embedding_dim, modes=modes, width=width).cuda()
        logger.info("Initialized MosfetFNO (Concat architecture)")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=n_epochs // warm_restart_count, eta_min=1e-6
    )
    if loss_type == "mse":
        loss_fn = ArcSinhMSELoss().cuda()
        logger.info("Using ArcSinhMSELoss (plain MSE in arcsinh space, no denominator)")
    elif loss_type == "lp_floor":
        loss_fn = LpLossWithFloor(p=2, floor=denom_floor).cuda()
        logger.info("Using LpLossWithFloor: floor=%.2f", denom_floor)
    elif loss_type == "weighted":
        loss_fn = SubthresholdWeightedLoss(scale_mA=loss_scale_ma, exponent=loss_exponent).cuda()
        logger.info("Using SubthresholdWeightedLoss: scale_mA=%.4f, exponent=%.2f", loss_scale_ma, loss_exponent)
    elif loss_type == "log10":
        loss_fn = Log10Loss().cuda()
        logger.info("Using Log10Loss")
    elif loss_type == "quartic":
        loss_fn = QuarticWeightedLoss().cuda()
        logger.info("Using QuarticWeightedLoss")
    elif loss_type == "region_adaptive":
        if vg_mean is None or vg_std is None:
            logger.warning("vg_mean/vg_std not provided, using defaults from 40K dataset")
            vg_mean, vg_std = 0.755, 0.476
        loss_fn = RegionAdaptiveLoss(
            subth_weight=loss_scale_ma, sat_weight=loss_exponent, vg_mean=vg_mean, vg_std=vg_std
        ).cuda()
        logger.info(
            "Using RegionAdaptiveLoss: subth_weight=%.2f, sat_weight=%.2f, vg_norm_threshold=%.3f",
            loss_scale_ma,
            loss_exponent,
            loss_fn.vth_threshold_normalized,
        )
    else:
        loss_fn = LpLoss(d=1, p=2, reduction="mean").cuda()
        logger.info("Using LpLoss (relative L2 norm)")
    return model, optimizer, scheduler, loss_fn


def _initialize_early_stopping(warm_restart_count, early_stop_patience, early_stop_threshold):
    """
    Initializes early stopping state if warm_restart_count is 1.

    :param warm_restart_count: Number of cosine annealing restarts.
    :param early_stop_patience: Epochs to wait before stopping.
    :param early_stop_threshold: Minimum loss change rate threshold.
    :return: Tuple of (enabled, loss_history, patience_counter).
    """
    enabled = warm_restart_count == 1
    if enabled:
        loss_history = deque(maxlen=early_stop_patience + 1)
        patience_counter = 0
        logger.info("Early stopping enabled: patience=%d, threshold=%.2e", early_stop_patience, early_stop_threshold)
        return enabled, loss_history, patience_counter
    return enabled, None, 0


_BACKBONE_PREFIXES = ("lifting.", "fno_blocks.", "projection.")
_TRAINABLE_PREFIXES = ("embedding.", "vcfilm_layers.")


def _freeze_backbone(model):
    """
    Freezes spectral operator backbone, keeping only conditioning layers trainable.

    Frozen: lifting, fno_blocks (SpectralConv + skips + channel MLP), projection.
    Trainable: embedding (DeviceEmbedding MLP), vcfilm_layers (8 VCFiLM modules).

    :param model: MosfetVCFiLMFNO model instance.
    :return: Tuple of (frozen_count, trainable_count) parameter counts.
    """
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in _BACKBONE_PREFIXES):
            param.requires_grad = False
            frozen_count += param.numel()
        elif any(name.startswith(prefix) for prefix in _TRAINABLE_PREFIXES):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
            logger.warning("Unexpected parameter '%s' frozen by default", name)
    logger.info(
        "Backbone frozen: %d params frozen, %d params trainable (%.1f%%)",
        frozen_count,
        trainable_count,
        100.0 * trainable_count / (frozen_count + trainable_count),
    )
    return frozen_count, trainable_count


def _train_epoch(model, loader, optimizer, scheduler, loss_fn):
    """
    Executes one full training epoch.

    :param model: The neural network model.
    :param loader: DataLoader for training batches.
    :param optimizer: Optimizer instance.
    :param scheduler: Learning rate scheduler.
    :param loss_fn: Loss function.
    :return: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    batch_count = 0
    for (voltages, physics), current_target in loader:
        voltages = voltages.cuda()
        physics = physics.cuda().squeeze(1)
        target = current_target.cuda()
        optimizer.zero_grad()
        pred = model(voltages, physics)
        if isinstance(loss_fn, RegionAdaptiveLoss):
            loss = loss_fn(pred, target, voltages)
        else:
            loss = loss_fn(pred, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    scheduler.step()
    torch.cuda.empty_cache()
    return total_loss / batch_count


def _check_early_stopping(
    loss_history, patience_counter, avg_loss, early_stop_patience, early_stop_threshold, writer, epoch
):
    """
    Checks if early stopping should be triggered.

    :param loss_history: Deque of recent loss values.
    :param patience_counter: Current patience counter value.
    :param avg_loss: Average loss for current epoch.
    :param early_stop_patience: Patience threshold.
    :param early_stop_threshold: Loss change rate threshold.
    :param writer: TensorBoard writer.
    :param epoch: Current epoch number.
    :return: Tuple of (should_stop, updated_patience_counter, loss_change_rate).
    """
    loss_history.append(avg_loss)
    if len(loss_history) <= early_stop_patience:
        return False, patience_counter, None
    loss_change_rate = abs(loss_history[0] - loss_history[-1]) / early_stop_patience
    writer.add_scalar("EarlyStopping/loss_change_rate", loss_change_rate, epoch)
    if loss_change_rate < early_stop_threshold:
        patience_counter += 1
        logger.info(
            "Early stopping: change rate %.2e < threshold %.2e (patience: %d/%d)",
            loss_change_rate,
            early_stop_threshold,
            patience_counter,
            early_stop_patience,
        )
    else:
        patience_counter = 0
    should_stop = patience_counter >= early_stop_patience
    return should_stop, patience_counter, loss_change_rate


def _run_periodic_evaluation(model, dataset, path_config, run_name, writer, epoch):
    """
    Runs inline I-V curve evaluation at specific epochs.

    :param model: The neural network model.
    :param dataset: Dataset instance.
    :param path_config: Path configuration object.
    :param run_name: Unique run identifier.
    :param writer: TensorBoard writer.
    :param epoch: Current epoch number.
    """
    logger.info("Running inline I-V evaluation at epoch %d...", epoch + 1)
    fig_iv, r2_iv = evaluate_sample_iv_curves(model, dataset, device="cuda")
    training_fig_dir = path_config.figure_dir / "training" / run_name
    training_fig_dir.mkdir(parents=True, exist_ok=True)
    fig_iv.savefig(training_fig_dir / f"iv_epoch_{epoch+1}.png")
    writer.add_figure("Validation/IV_Curves", fig_iv, epoch)
    writer.add_scalar("Validation/R2", r2_iv, epoch)
    plt.close(fig_iv)


def run_final_evaluations(model, dataset, path_config, run_name, writer, n_epochs, trim_eval=DEFAULT_TRIM_EVAL):
    """
    Executes all final evaluation procedures after training.

    :param model: The trained neural network model.
    :param dataset: Dataset instance.
    :param path_config: Path configuration object.
    :param run_name: Unique run identifier.
    :param writer: TensorBoard writer.
    :param n_epochs: Total number of epochs (for logging).
    :param trim_eval: Number of initial timesteps to discard from eval (SPICE .op artifact).
    :return: Tuple of (r2_fast, metrics_spice, comprehensive_metrics).
    """
    training_fig_dir = path_config.figure_dir / "training" / run_name
    training_fig_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running final fast dataset evaluation...")
    final_fig_fast, final_r2_fast = evaluate_sample_iv_curves(model, dataset, device="cuda")
    final_fig_fast.savefig(training_fig_dir / "iv_final_fast.png")
    writer.add_figure("Validation/Final_Dataset_IV", final_fig_fast, n_epochs)
    plt.close(final_fig_fast)
    logger.info("Running final SPICE-based I-V sweep validation...")
    final_fig_spice, final_metrics_spice = evaluate_spice_iv_sweeps(
        model, dataset, device="cuda", w_um=1.0, l_um=0.18, t_steps=512, trim_eval=trim_eval
    )
    final_fig_spice.savefig(training_fig_dir / "iv_final_spice.png")
    writer.add_figure("Validation/Final_SPICE_Sweeps", final_fig_spice, n_epochs)
    for metric_name, metric_value in final_metrics_spice.items():
        writer.add_scalar(f"Validation/final_{metric_name}", metric_value, n_epochs)
    plt.close(final_fig_spice)
    logger.info("Running comprehensive multi-geometry SPICE validation...")
    comprehensive_dir = training_fig_dir / "comprehensive"
    comprehensive_metrics, comprehensive_figures = evaluate_comprehensive(
        model, dataset, comprehensive_dir, device="cuda", t_steps=512, trim_eval=trim_eval
    )
    for geom_name, geom_metrics in comprehensive_metrics.items():
        for metric_name, metric_value in geom_metrics.items():
            writer.add_scalar(f"Validation/comprehensive_{geom_name}_{metric_name}", metric_value, n_epochs)
    for geom_name, fig in comprehensive_figures.items():
        writer.add_figure(f"Comprehensive/{geom_name}", fig, n_epochs)
        plt.close(fig)
    log_evaluation_summary(final_r2_fast, final_metrics_spice, comprehensive_metrics)
    return final_r2_fast, final_metrics_spice, comprehensive_metrics


def _log_hyperparameters(writer, params, avg_loss, r2_fast, metrics_spice):
    """
    Logs hyperparameters and final metrics to TensorBoard.

    :param writer: TensorBoard writer.
    :param params: Dictionary of all hyperparameters.
    :param avg_loss: Final training loss.
    :param r2_fast: R² from fast dataset evaluation.
    :param metrics_spice: Dictionary of SPICE validation metrics.
    """
    hparam_metrics = {
        "hparam/loss": avg_loss,
        "hparam/r2_fast": r2_fast,
        "hparam/r2_transfer": metrics_spice.get("r2_transfer", 0),
        "hparam/r2_output": metrics_spice.get("r2_output", 0),
    }
    writer.add_hparams(hparam_dict=params, metric_dict=hparam_metrics)


def run_mosfet_training(
    dataset_path,
    experiment_name,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    warm_restart_count,
    loss_type,
    modes,
    width,
    embedding_dim,
    early_stop_patience,
    early_stop_threshold,
    loss_scale_ma=0.01,
    loss_exponent=2.0,
    denom_floor=10.0,
    checkpoint_path=None,
    model_type="concat",
    trim_startup=0,
    freeze_backbone=False,
    geometry_filter=None,
):
    """
    Runs training loop using pre-generated HDF5 dataset.

    :param dataset_path: Path to HDF5 dataset file.
    :param experiment_name: Base name for run identification.
    :param n_epochs: Number of training epochs.
    :param batch_size: Training batch size.
    :param learning_rate: Initial learning rate.
    :param weight_decay: AdamW weight decay coefficient.
    :param warm_restart_count: Number of cosine annealing restarts.
    :param loss_type: Loss function type ('lp', 'mse', 'lp_floor', 'weighted', 'log10', 'quartic', 'region_adaptive').
    :param modes: Number of Fourier modes for FNO.
    :param width: Hidden channel width for FNO.
    :param embedding_dim: Physics parameter embedding dimension.
    :param early_stop_patience: Epochs to wait for improvement before stopping.
    :param early_stop_threshold: Minimum loss change rate to consider as improvement.
    :param loss_scale_ma: For 'weighted': scale_mA. For 'region_adaptive': subth_weight.
    :param loss_exponent: For 'weighted': exponent. For 'region_adaptive': sat_weight.
    :param denom_floor: Minimum denominator for 'lp_floor' loss (arcsinh units).
    :param checkpoint_path: Optional path to model checkpoint for initialization before training.
    :param model_type: Architecture type ('concat', 'film', or 'vcfilm').
    :param trim_startup: Number of initial timesteps to discard from training data.
    :param freeze_backbone: If True, freeze spectral operator and train only FiLM conditioning.
        Requires checkpoint_path and model_type='vcfilm'.
    :param geometry_filter: Optional geometry bin name (e.g. "xlarge") to restrict training samples.
        Normalization stats come from the full dataset for checkpoint compatibility.
    """
    params = locals().copy()
    if freeze_backbone and not checkpoint_path:
        raise ValueError("--freeze-backbone requires --checkpoint-path (cannot freeze a randomly initialized model)")
    if freeze_backbone and model_type != "vcfilm":
        raise ValueError("--freeze-backbone is only supported for --model-type vcfilm")
    path_config = PathConfig("mosfet")
    unique_id = generate_unique_id(json.dumps(params, sort_keys=True))
    run_name = f"{experiment_name}_{unique_id}"
    writer = SummaryWriter(log_dir=path_config.run_dir / run_name)
    writer.add_text("hyperparameters", json.dumps(params, indent=2))
    logger.info("Starting Run: %s", run_name)
    logger.info("Loading pre-generated dataset: %s", dataset_path)
    with PreGeneratedMosfetDataset(dataset_path, trim_startup=trim_startup, geometry_filter=geometry_filter) as dataset:
        logger.info("Dataset size: %d samples", len(dataset))
        input_param_dim = ParameterSchema.input_dim()
        logger.info("Using %d curated physics parameters for training", input_param_dim)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        vg_mean = dataset.voltages_mean[0].item() if hasattr(dataset, "voltages_mean") else None
        vg_std = dataset.voltages_std[0].item() if hasattr(dataset, "voltages_std") else None
        model, optimizer, scheduler, loss_fn = _initialize_training_components(
            input_param_dim=input_param_dim,
            embedding_dim=embedding_dim,
            modes=modes,
            width=width,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            warm_restart_count=warm_restart_count,
            loss_type=loss_type,
            loss_scale_ma=loss_scale_ma,
            loss_exponent=loss_exponent,
            denom_floor=denom_floor,
            vg_mean=vg_mean,
            vg_std=vg_std,
            model_type=model_type,
        )
        if checkpoint_path:
            logger.info("Loading checkpoint from: %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            if hasattr(state_dict, "pop"):
                state_dict.pop("_metadata", None)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                if isinstance(state_dict, dict) and "model" in state_dict and hasattr(state_dict["model"], "pop"):
                    nested_state_dict = state_dict["model"]
                    nested_state_dict.pop("_metadata", None)
                    model.load_state_dict(nested_state_dict)
                else:
                    raise
            logger.info("Checkpoint loaded successfully")
        if freeze_backbone:
            _freeze_backbone(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, n_epochs // warm_restart_count), eta_min=1e-6
            )
            logger.info("Optimizer rebuilt with %d trainable parameter groups", len(trainable_params))
        early_stopping_enabled, loss_history, patience_counter = _initialize_early_stopping(
            warm_restart_count, early_stop_patience, early_stop_threshold
        )
        logger.info("Starting Training Loop...")
        final_epoch = n_epochs - 1
        with timeit("Training Loop") as lap_epoch:
            for epoch in range(n_epochs):
                try:
                    avg_loss = _train_epoch(model, loader, optimizer, scheduler, loss_fn)
                    writer.add_scalar("Loss/train", avg_loss, epoch)
                    writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)
                    if early_stopping_enabled:
                        should_stop, patience_counter, loss_change_rate = _check_early_stopping(
                            loss_history,
                            patience_counter,
                            avg_loss,
                            early_stop_patience,
                            early_stop_threshold,
                            writer,
                            epoch,
                        )
                        if should_stop:
                            logger.info(
                                "EARLY STOPPING TRIGGERED at epoch %d/%d. Loss flatlined (change rate: %.2e < threshold: %.2e).",
                                epoch + 1,
                                n_epochs,
                                loss_change_rate,
                                early_stop_threshold,
                            )
                            final_epoch = epoch
                            break
                    if epoch % 10 == 0 or epoch == n_epochs - 1:
                        logger.info("Epoch %03d/%d | Loss (Lp): %.6f", epoch, n_epochs, avg_loss)
                        lap_epoch()
                    if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
                        _run_periodic_evaluation(model, dataset, path_config, run_name, writer, epoch)
                except KeyboardInterrupt:
                    logger.warning("Training interrupted by user at epoch %d. Saving checkpoint...", epoch)
                    final_epoch = epoch
                    break
                lap_epoch(alt_msg=f"Epoch {epoch+1} completed")
        logger.info("Training Complete.")
        torch.save(model.state_dict(), path_config.model_dir / f"{run_name}.pt")
        final_r2_fast, final_metrics_spice, comprehensive_metrics = run_final_evaluations(
            model, dataset, path_config, run_name, writer, n_epochs
        )
        _log_hyperparameters(writer, params, avg_loss, final_r2_fast, final_metrics_spice)
        writer.close()
        return final_r2_fast, final_metrics_spice, comprehensive_metrics, final_epoch + 1


@click.command()
@click.option("--dataset-path", default="/app/datasets/sky130_nmos_25k.h5", help="Path to HDF5 dataset file.")
@click.option("--experiment-name", default="mosfet_spice_supervised", help="Base name for run identification.")
@click.option("--n-epochs", default=50, help="Number of training epochs.")
@click.option("--batch-size", default=64, help="Training batch size.")
@click.option("--learning-rate", default=1e-3, help="Initial learning rate.")
@click.option("--weight-decay", default=1e-5, help="AdamW weight decay coefficient.")
@click.option("--warm-restart-count", default=1, help="Number of cosine annealing restarts.")
@click.option(
    "--loss-type",
    default="lp",
    type=click.Choice(["lp", "mse", "lp_floor", "weighted", "log10", "quartic", "region_adaptive"]),
    help="Loss function type. 'mse': plain MSE in arcsinh space (no denominator). 'lp_floor': relative L2 with clamped denominator.",
)
@click.option("--denom-floor", default=10.0, help="For 'lp_floor': minimum denominator value in arcsinh units.")
@click.option("--modes", default=128, help="Number of Fourier modes for FNO.")
@click.option("--width", default=64, help="Hidden channel width for FNO.")
@click.option("--embedding-dim", default=16, help="Physics parameter embedding dimension.")
@click.option("--early-stop-patience", default=10, help="Epochs to wait for improvement before stopping.")
@click.option("--early-stop-threshold", default=1e-5, help="Minimum loss change rate to consider as improvement.")
@click.option("--loss-scale-ma", default=0.01, help="For 'weighted': scale_mA. For 'region_adaptive': subth_weight.")
@click.option("--loss-exponent", default=2.0, help="For 'weighted': exponent. For 'region_adaptive': sat_weight.")
@click.option(
    "--checkpoint-path", default=None, help="Optional model checkpoint path for initialization before training."
)
@click.option(
    "--model-type", default="concat", type=click.Choice(["concat", "film", "vcfilm"]), help="Architecture type."
)
@click.option("--trim-startup", default=0, help="Timesteps to trim from start of each sample (removes .op blip).")
@click.option(
    "--freeze-backbone",
    is_flag=True,
    default=False,
    help="Freeze spectral operator, train only FiLM conditioning. Requires --checkpoint-path and --model-type vcfilm.",
)
@click.option(
    "--geometry-filter",
    default=None,
    type=click.Choice(["tiny", "small", "medium", "large", "xlarge"]),
    help="Restrict training to samples from a single geometry bin.",
)
def main(
    dataset_path,
    experiment_name,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    warm_restart_count,
    loss_type,
    modes,
    width,
    embedding_dim,
    early_stop_patience,
    early_stop_threshold,
    loss_scale_ma,
    loss_exponent,
    denom_floor,
    checkpoint_path,
    model_type,
    trim_startup,
    freeze_backbone,
    geometry_filter,
):
    run_mosfet_training(
        dataset_path=dataset_path,
        experiment_name=experiment_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warm_restart_count=warm_restart_count,
        loss_type=loss_type,
        modes=modes,
        width=width,
        embedding_dim=embedding_dim,
        early_stop_patience=early_stop_patience,
        early_stop_threshold=early_stop_threshold,
        loss_scale_ma=loss_scale_ma,
        loss_exponent=loss_exponent,
        denom_floor=denom_floor,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        trim_startup=trim_startup,
        freeze_backbone=freeze_backbone,
        geometry_filter=geometry_filter,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter  # click mutates.
    main()
