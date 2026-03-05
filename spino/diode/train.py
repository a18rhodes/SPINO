"""
Training script for the diode neural operator.

Supports two modes:

1. **Dimensionless (default):** Pre-generated HDF5 dataset with variable T_end,
   MSE + Sobolev loss with fade-in schedule. Resolution-invariant formulation.

2. **Legacy:** On-the-fly SPICE generation with fixed 1ms window, plain MSE loss.
   Retained for backward compatibility with existing checkpoints.

Usage::

    # Dimensionless training (recommended)
    python -m spino.diode.train --dataset-path /app/datasets/diode_10k.h5

    # Legacy training
    python -m spino.diode.train --legacy --n-epochs 80
"""

import json
import logging
import sys
from collections import deque

import click
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spino.archive import backup_artifacts
from spino.config import PathConfig
from spino.diode.evaluate import evaluate_adversarial, evaluate_rectifier
from spino.diode.gen_data import DimensionlessDiodeDataset, InfiniteSpiceDiodeDataset, PreGeneratedDiodeDataset
from spino.diode.model import get_model
from spino.loss import GenericDimensionlessPhysicsLoss
from spino.utils import generate_unique_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

plt.style.use("dark_background")


def _noop_residual(V_mid, dV_dt, inputs):
    """Placeholder physics residual that returns zero (Sobolev-only mode)."""
    return torch.zeros_like(dV_dt)


def _build_loss_fn(target_sobolev_weight: float) -> GenericDimensionlessPhysicsLoss:
    """
    Constructs the MSE + Sobolev loss with physics_weight=0.

    The Sobolev term penalizes derivative mismatch between prediction and
    ground truth. No diode-specific physics residual is used due to
    exponential instability in the Shockley equation.

    :param target_sobolev_weight: Peak Sobolev weight after warmup.
    :return: Configured loss function.
    """
    return GenericDimensionlessPhysicsLoss(
        physics_residual_fn=_noop_residual,
        sobolev_weight=target_sobolev_weight,
        physics_weight=0.0,
    )


def _compute_alpha(epoch: int, dead_zone: int, warmup: int, total_epochs: int, fine_tune: int) -> float:
    """
    Computes the fade-in multiplier for auxiliary loss terms.

    Schedule: dead_zone (alpha=0) -> warmup (linear 0->1) -> full (alpha=1)
    -> polish (alpha=0).

    :param epoch: Current epoch index.
    :param dead_zone: Epochs of pure MSE before any auxiliary terms.
    :param warmup: Epochs to linearly ramp auxiliary weights.
    :param total_epochs: Total number of training epochs.
    :param fine_tune: Epochs of pure MSE at the end (polishing).
    :return: Multiplier in [0, 1].
    """
    total_warmup = dead_zone + warmup
    if epoch < dead_zone:
        return 0.0
    if epoch < total_warmup:
        return (epoch - dead_zone) / warmup
    if epoch >= (total_epochs - fine_tune):
        return 0.0
    return 1.0


def _train_epoch_hdf5(model, loader, optimizer, loss_fn):
    """
    Executes one training epoch on pre-generated HDF5 data.

    :param model: DiodeFNO model.
    :param loader: DataLoader over PreGeneratedDiodeDataset.
    :param optimizer: Optimizer instance.
    :param loss_fn: GenericDimensionlessPhysicsLoss.
    :return: Tuple of (avg_total, avg_data, avg_sobolev).
    """
    model.train()
    total_loss = torch.tensor(0.0, device="cuda")
    total_data = torch.tensor(0.0, device="cuda")
    total_sobolev = torch.tensor(0.0, device="cuda")
    batch_count = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = model(x)
        loss, l_data, l_sobolev, _l_physics = loss_fn(pred, y, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.detach()
        total_data += l_data.detach()
        total_sobolev += l_sobolev.detach()
        batch_count += 1
    n = max(batch_count, 1)
    return total_loss.item() / n, total_data.item() / n, total_sobolev.item() / n


def _train_epoch_online(model, loader_iter, optimizer, loss_fn, steps_per_epoch, loader):
    """
    Executes one training epoch on an infinite SPICE-backed loader.

    :param model: DiodeFNO model.
    :param loader_iter: Current iterator over the DataLoader.
    :param optimizer: Optimizer instance.
    :param loss_fn: Loss function (MSE or GenericDimensionlessPhysicsLoss).
    :param steps_per_epoch: Gradient steps per epoch.
    :param loader: The DataLoader itself (for re-iteration if exhausted).
    :return: Tuple of (avg_loss, updated_loader_iter).
    """
    model.train()
    total_loss = 0.0
    for _ in range(steps_per_epoch):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, y = next(loader_iter)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        pred = model(x)
        if isinstance(loss_fn, GenericDimensionlessPhysicsLoss):
            loss, _, _, _ = loss_fn(pred, y, x)
        else:
            loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / steps_per_epoch, loader_iter


def run_dimensionless_training(
    dataset_path: str,
    experiment_name: str = "diode_dimensionless",
    n_epochs: int = 250,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    target_sobolev_weight: float = 1e-2,
    dead_zone_epochs: int = 50,
    warmup_epochs: int = 100,
    fine_tune_epochs: int = 50,
    early_stop_patience: int = 20,
    early_stop_threshold: float = 1e-6,
    modes: int = 256,
    width: int = 64,
    checkpoint_path: str | None = None,
):
    """
    Trains the dimensionless diode FNO on a pre-generated HDF5 dataset.

    :param dataset_path: Path to HDF5 file from generate_dataset.
    :param experiment_name: Base name for run identification.
    :param n_epochs: Total training epochs.
    :param batch_size: Training batch size.
    :param learning_rate: Initial learning rate for AdamW.
    :param weight_decay: AdamW weight decay.
    :param target_sobolev_weight: Peak Sobolev weight after warmup.
    :param dead_zone_epochs: Pure-MSE warmup epochs (no Sobolev).
    :param warmup_epochs: Linear ramp epochs for Sobolev weight.
    :param fine_tune_epochs: Pure-MSE polishing epochs at end.
    :param early_stop_patience: Epochs to wait for improvement.
    :param early_stop_threshold: Minimum loss change rate.
    :param modes: Fourier modes for DiodeFNO.
    :param width: Hidden channel width for DiodeFNO.
    :param checkpoint_path: Optional path to resume from.
    """
    params = {k: v for k, v in locals().items() if k != "checkpoint_path" or v is not None}
    path_config = PathConfig("diode")
    unique_id = generate_unique_id(json.dumps(params, sort_keys=True))
    run_name = f"{experiment_name}_{unique_id}"
    writer = SummaryWriter(log_dir=path_config.run_dir / run_name)
    writer.add_text("hyperparameters", json.dumps(params, indent=2))
    logger.info("Starting Run: %s", run_name)
    with PreGeneratedDiodeDataset(dataset_path) as dataset:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        from spino.diode.model import DiodeFNO
        model = DiodeFNO(in_channels=6, n_modes=(modes,), hidden_channels=width).cuda()
        if checkpoint_path:
            logger.info("Loading checkpoint: %s", checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            model.load_state_dict(state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, fine_tune_epochs), eta_min=1e-6
        )
        loss_fn = _build_loss_fn(target_sobolev_weight)
        loss_history = deque(maxlen=early_stop_patience + 1)
        patience_counter = 0
        avg_loss = float("nan")
        logger.info("Starting training: %d epochs, batch_size=%d", n_epochs, batch_size)
        for epoch in range(n_epochs):
            alpha = _compute_alpha(epoch, dead_zone_epochs, warmup_epochs, n_epochs, fine_tune_epochs)
            loss_fn.sobolev_weight = target_sobolev_weight * alpha
            avg_loss, avg_data, avg_sob = _train_epoch_hdf5(model, loader, optimizer, loss_fn)
            scheduler.step()
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Loss/data", avg_data, epoch)
            writer.add_scalar("Loss/sobolev", avg_sob, epoch)
            writer.add_scalar("Params/alpha", alpha, epoch)
            writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)
            if epoch % 10 == 0 or epoch == n_epochs - 1:
                logger.info(
                    "Epoch %03d/%d | Total: %.6f | Data: %.6f | Sobolev: %.6f | alpha: %.2f",
                    epoch, n_epochs, avg_loss, avg_data, avg_sob, alpha,
                )
            loss_history.append(avg_loss)
            if len(loss_history) > early_stop_patience:
                change_rate = abs(loss_history[0] - loss_history[-1]) / early_stop_patience
                if change_rate < early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.info("Early stopping at epoch %d (change rate %.2e)", epoch, change_rate)
                        break
                else:
                    patience_counter = 0
            if (epoch + 1) % 25 == 0:
                fig_rect, m_val = evaluate_rectifier(model, dimensionless=True)
                training_fig_dir = path_config.figure_dir / "training" / run_name
                training_fig_dir.mkdir(parents=True, exist_ok=True)
                fig_rect.savefig(training_fig_dir / f"rectifier_epoch_{epoch+1}.png")
                writer.add_figure("Validation/Rectifier", fig_rect, epoch)
                writer.add_scalar("Validation/R2_Rectifier", m_val["r2"], epoch)
                writer.add_scalar("Validation/MSE_Rectifier", m_val["mse"], epoch)
                plt.close(fig_rect)
        final_fig_dir = path_config.figure_dir / "training" / run_name
        final_fig_dir.mkdir(parents=True, exist_ok=True)
        fig_rect_final, m_rect = evaluate_rectifier(model, dimensionless=True)
        fig_rect_final.savefig(final_fig_dir / "rectifier_final.png")
        writer.add_figure("Validation/Rectifier_Final", fig_rect_final, n_epochs)
        plt.close(fig_rect_final)
        fig_adv_final, m_adv = evaluate_adversarial(model, loader, dimensionless=True)
        if fig_adv_final:
            fig_adv_final.savefig(final_fig_dir / "adversarial_final.png")
            writer.add_figure("Validation/Adversarial_Final", fig_adv_final, n_epochs)
            plt.close(fig_adv_final)
        logger.info(
            "Final Eval | Rectifier: R2=%.4f, MSE=%.2e, RMSE=%.4f, MAE=%.2fmV",
            m_rect["r2"], m_rect["mse"], m_rect["rmse"], m_rect["mae_mv"],
        )
        if m_adv:
            logger.info(
                "Final Eval | Adversarial: R2=%.4f, MSE=%.2e, RMSE=%.4f, MAE=%.2fmV",
                m_adv["r2"], m_adv["mse"], m_adv["rmse"], m_adv["mae_mv"],
            )
    logger.info("Training complete. Saving checkpoint: %s", run_name)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": {"modes": modes, "width": width, "in_channels": 6}},
        path_config.model_dir / f"{run_name}.pt",
    )
    hparam_metrics = {
        "hparam/loss": avg_loss,
        "hparam/r2_rectifier": m_rect["r2"],
        "hparam/mse_rectifier": m_rect["mse"],
        "hparam/mae_mv_rectifier": m_rect["mae_mv"],
    }
    if m_adv:
        hparam_metrics["hparam/r2_adversarial"] = m_adv["r2"]
    writer.add_hparams(hparam_dict=params, metric_dict=hparam_metrics)
    writer.close()
    backup_artifacts(run_name)
    return model


def run_legacy_training(
    experiment_name: str = "diode_spice_supervised",
    n_epochs: int = 50,
    steps_per_epoch: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
):
    """
    Legacy training with on-the-fly SPICE and fixed 1ms window.

    Retained for backward compatibility with 5-channel checkpoints.

    :param experiment_name: Base name for run identification.
    :param n_epochs: Total training epochs.
    :param steps_per_epoch: Gradient steps per epoch.
    :param batch_size: Training batch size.
    :param learning_rate: Initial learning rate.
    :param weight_decay: AdamW weight decay.
    """
    params = locals().copy()
    path_config = PathConfig("diode")
    unique_id = generate_unique_id(json.dumps(params, sort_keys=True))
    run_name = f"{experiment_name}_{unique_id}"
    writer = SummaryWriter(log_dir=path_config.run_dir / run_name)
    writer.add_text("hyperparameters", json.dumps(params, indent=2))
    logger.info("Starting Legacy Run: %s", run_name)
    train_loader = DataLoader(
        InfiniteSpiceDiodeDataset(t_steps=1024),
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=True,
    )
    model = get_model(dimensionless=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs, eta_min=1e-6)
    mse_loss = torch.nn.MSELoss()
    loader_iter = iter(train_loader)
    avg_loss = float("nan")
    for epoch in range(n_epochs):
        avg_loss, loader_iter = _train_epoch_online(model, loader_iter, optimizer, mse_loss, steps_per_epoch, train_loader)
        scheduler.step()
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)
        if epoch % 10 == 0:
            logger.info("Epoch %03d/%d | Loss (MSE): %.6f", epoch, n_epochs, avg_loss)
        if (epoch + 1) % 10 == 0:
            fig_rect, _ = evaluate_rectifier(model, dimensionless=False)
            fig_rect.savefig(path_config.figure_dir / f"{run_name}_rectifier_epoch_{epoch+1}.png")
            writer.add_figure("Validation/Rectifier", fig_rect, epoch)
            plt.close(fig_rect)
    logger.info("Training Complete.")
    torch.save(model.state_dict(), path_config.model_dir / f"{run_name}.pt")
    fig_final, m_rect_legacy = evaluate_rectifier(model, dimensionless=False)
    fig_final.savefig(path_config.figure_dir / f"{run_name}_rectifier_final.png")
    plt.close(fig_final)
    fig_adv_final, m_adv_legacy = evaluate_adversarial(model, train_loader, dimensionless=False)
    if fig_adv_final:
        fig_adv_final.savefig(path_config.figure_dir / f"{run_name}_adversarial_final.png")
        plt.close(fig_adv_final)
    writer.add_hparams(
        hparam_dict=params,
        metric_dict={
            "hparam/loss": avg_loss,
            "hparam/r2_rectifier": m_rect_legacy["r2"],
            "hparam/r2_adversarial": m_adv_legacy["r2"] if m_adv_legacy else 0.0,
        },
    )
    writer.close()
    backup_artifacts(run_name)


@click.command()
@click.option("--dataset-path", default=None, type=click.Path(exists=True), help="HDF5 dataset for dimensionless training.")
@click.option("--experiment-name", default="diode_dimensionless", help="Base name for run identification.")
@click.option("--n-epochs", default=250, show_default=True, help="Number of training epochs.")
@click.option("--batch-size", default=64, show_default=True, help="Training batch size.")
@click.option("--learning-rate", default=1e-3, show_default=True, help="Initial learning rate.")
@click.option("--weight-decay", default=1e-5, show_default=True, help="AdamW weight decay.")
@click.option("--sobolev-weight", default=1e-2, show_default=True, help="Peak Sobolev (derivative matching) weight.")
@click.option("--dead-zone-epochs", default=50, show_default=True, help="Pure-MSE epochs before Sobolev ramp.")
@click.option("--warmup-epochs", default=100, show_default=True, help="Linear ramp epochs for Sobolev weight.")
@click.option("--fine-tune-epochs", default=50, show_default=True, help="Pure-MSE polishing epochs at end.")
@click.option("--early-stop-patience", default=20, show_default=True, help="Epochs to wait for improvement.")
@click.option("--early-stop-threshold", default=1e-6, show_default=True, help="Minimum loss change rate.")
@click.option("--modes", default=256, show_default=True, help="Fourier modes for FNO.")
@click.option("--width", default=64, show_default=True, help="Hidden channel width for FNO.")
@click.option("--checkpoint-path", default=None, type=click.Path(exists=True), help="Resume from checkpoint.")
@click.option("--legacy", is_flag=True, default=False, help="Use legacy 5-channel fixed-grid training.")
@click.option("--legacy-epochs", default=80, show_default=True, help="Epochs for legacy training mode.")
def main(
    dataset_path,
    experiment_name,
    n_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    sobolev_weight,
    dead_zone_epochs,
    warmup_epochs,
    fine_tune_epochs,
    early_stop_patience,
    early_stop_threshold,
    modes,
    width,
    checkpoint_path,
    legacy,
    legacy_epochs,
):
    """Train the diode FNO (dimensionless by default, --legacy for fixed-grid)."""
    if legacy:
        run_legacy_training(
            experiment_name="diode_legacy",
            n_epochs=legacy_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        if not dataset_path:
            raise click.UsageError("--dataset-path is required for dimensionless training. Use --legacy for on-the-fly mode.")
        run_dimensionless_training(
            dataset_path=dataset_path,
            experiment_name=experiment_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            target_sobolev_weight=sobolev_weight,
            dead_zone_epochs=dead_zone_epochs,
            warmup_epochs=warmup_epochs,
            fine_tune_epochs=fine_tune_epochs,
            early_stop_patience=early_stop_patience,
            early_stop_threshold=early_stop_threshold,
            modes=modes,
            width=width,
            checkpoint_path=checkpoint_path,
        )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
