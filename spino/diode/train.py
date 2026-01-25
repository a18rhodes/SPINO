# %% [markdown]
### Module defining the training routine for the Diode circuit neural operator model.

# %%
import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from spino.archive import backup_artifacts
from spino.diode.evaluate import evaluate_adversarial, evaluate_rectifier
from spino.diode.model import get_model
from spino.diode.gen_data import InfiniteSpiceDiodeDataset
from spino.config import PathConfig
from spino.utils import generate_unique_id

# Set style
plt.style.use("dark_background")


# %% [markdown]
### Data Loader Setup
# This function sets up the data loader for the InfiniteSpiceDiodeDataset.
# %%
def setup_data_loader(batch_size: int) -> DataLoader:
    """
    Sets up the data loader for the InfiniteSpiceDiodeDataset.
    """
    return DataLoader(
        InfiniteSpiceDiodeDataset(t_steps=1024),
        batch_size=batch_size,
        num_workers=8,
        persistent_workers=True,  # Parallelize SPICE generation
    )


# %% [markdown]
### Training Routine for Diode Circuit Neural Operator
# This function sets up the data loader, model, optimizer, and runs the training loop,
# periodically evaluating the model on a standard rectifier test case.
def run_diode_training(
    experiment_name="diode_spice_supervised",
    n_epochs=50,
    steps_per_epoch=100,
    batch_size=64,
    learning_rate=1e-3,
    weight_decay=1e-5,
    warm_restart_count=1,
):
    # Capture parameters for logging
    params = locals().copy()
    path_config = PathConfig("diode")
    unique_id = generate_unique_id(json.dumps(params, sort_keys=True))
    run_name = f"{experiment_name}_{unique_id}"
    writer = SummaryWriter(log_dir=path_config.run_dir / run_name)
    writer.add_text("hyperparameters", json.dumps(params, indent=2))
    print(f"Starting Experiment: {run_name}")

    train_loader = setup_data_loader(batch_size)

    # 2. Setup Model (The Consumer)
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=n_epochs // warm_restart_count, eta_min=1e-6
    )
    mse_loss = nn.MSELoss()

    # 3. Training Loop
    iter_loader = iter(train_loader)

    loss_history = []
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            # Fetch Data (CPU -> GPU)
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                x, y = next(iter_loader)
            x, y = x.cuda(), y.cuda()
            # Forward
            optimizer.zero_grad()
            pred = model(x)
            # Loss (Supervised)
            loss = mse_loss(pred, y)
            # Backward
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            total_loss += loss_val
            global_step += 1
            # Optional: Log step loss
            # writer.add_scalar("Loss/step", loss_val, global_step)
        scheduler.step()
        avg_loss = total_loss / steps_per_epoch
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1:03d}/{n_epochs} | Loss (MSE): {avg_loss:.6f}")
        # Log Epoch Metrics
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)
        # Periodic Eval & Image Logging
        # Periodic Eval
        if (epoch + 1) % 10 == 0:
            # 1. Standard Rectifier Test
            fig_rect, _ = evaluate_rectifier(model)
            fig_rect.savefig(path_config.figure_dir / f"{run_name}_rectifier_epoch_{epoch+1}.png")
            writer.add_figure("Validation/Rectifier", fig_rect, epoch)
            plt.close(fig_rect)

            # 2. Random Sample Test (Generalization)
            fig_rand, _ = evaluate_adversarial(model, train_loader)
            if fig_rand:
                fig_rand.savefig(path_config.figure_dir / f"{run_name}_random_epoch_{epoch+1}.png")
                writer.add_figure("Validation/Random_Sample", fig_rand, epoch)
                plt.close(fig_rand)

    print("Training Complete.")
    torch.save(model.state_dict(), path_config.model_dir / f"{run_name}.pt")

    # Final Standard Eval
    fig_final, final_r2_rect = evaluate_rectifier(model)
    fig_final.savefig(path_config.figure_dir / f"{run_name}_rectifier_final.png")
    plt.close(fig_final)
    # Final Adversarial Eval
    fig_adv_final, final_r2_adv = evaluate_adversarial(model, train_loader)
    if fig_adv_final:
        fig_adv_final.savefig(path_config.figure_dir / f"{run_name}_adversarial_final.png")
        plt.close(fig_adv_final)

    writer.add_hparams(
        hparam_dict=params,
        metric_dict={
            "hparam/loss": avg_loss,
            "hparam/r2_rectifier": final_r2_rect,
            "hparam/r2_adversarial": final_r2_adv,
        },
    )
    writer.close()
    backup_artifacts(run_name)


if __name__ == "__main__":
    run_diode_training(n_epochs=80)
