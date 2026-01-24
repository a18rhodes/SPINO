# %% [markdown]
### Module defining the training routine for the RC circuit neural operator model.

# %%
import base64
import json

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from spino.archive import backup_artifacts
from spino.config import PathConfig
from spino.loss import GenericDimensionlessPhysicsLoss, rc_physics_residual
from spino.rc.evaluate import (
    evaluate_adversarial_spectrum,
    evaluate_ic_spectrum,
    evaluate_ood_physics,
)
from spino.rc.gen_data import (
    generate_dimensionless_data,
    generate_dimensionless_data_with_white_noise,
    generate_dimensionless_data_with_white_noise_and_chirp,
    generate_dimensionless_data_with_white_noise_and_chirp_log_uniform,
)
from spino.rc.model import get_model

plt.style.use("dark_background")


# %% [markdown]
### Experiment runner
# This function encapsulates the entire experiment lifecycle: data generation, model training, evaluation, and


# %%
def run_experiment(
    data_generator,
    experiment_name,
    n_samples: int = 10000,
    t_steps: int = 2048,
    epochs: int = 500,
    dead_zone_epochs: int = 50,
    warmup_epochs: int = 150,
    fine_tune_epochs: int = 100,
    target_sobolev_weight: float = 1e-2,
    target_physics_weight: float = 1e-4,
    batch_size: int = 64,
    starting_lr: int = 1e-3,
    adam_weight_decay: int = 1e-5,
):
    params = locals().copy()
    params["data_generator"] = data_generator.__name__
    path_config = PathConfig("simple_rc")
    # Generate Unique ID
    unique_id = base64.b64encode(json.dumps(params, sort_keys=True).encode("utf-8")).decode("utf-8")[:8]
    run_name = f"{experiment_name}_{unique_id}"
    writer = SummaryWriter(log_dir=path_config.run_dir / run_name)
    writer.add_text("hyperparameters", json.dumps(params, indent=2))

    print(f"Starting Experiment: {run_name}")

    # 2. Data
    print(f"Generating {n_samples} samples with {params['data_generator']}...")
    train_x, train_y = data_generator(n_samples=n_samples, t_steps=t_steps)
    dataset = TensorDataset(train_x.cpu(), train_y.cpu())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Model & Optimization
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=starting_lr, weight_decay=adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=fine_tune_epochs, T_mult=1, eta_min=1e-6
    )
    loss_fn = GenericDimensionlessPhysicsLoss(
        sobolev_weight=target_sobolev_weight,
        physics_weight=target_physics_weight,
        physics_residual_fn=rc_physics_residual,
    )

    # 4. Training Loop
    print(f"Starting Training...")
    total_warmup_epochs = dead_zone_epochs + warmup_epochs
    avg_loss = avg_data = avg_sob = avg_phys = 0.0

    for epoch in range(epochs):
        # Fade-In Logic
        if epoch < dead_zone_epochs:
            alpha = 0.0
        elif epoch < total_warmup_epochs:
            alpha = (epoch - dead_zone_epochs) / warmup_epochs
        elif epoch < (epochs - fine_tune_epochs):
            alpha = 1.0
        else:
            alpha = 0.0  # Polishing

        loss_fn.sobolev_weight = target_sobolev_weight * alpha
        loss_fn.physics_weight = target_physics_weight * alpha

        model.train()

        total_loss = torch.tensor(0.0, device="cuda")
        total_data = torch.tensor(0.0, device="cuda")
        total_sobolev = torch.tensor(0.0, device="cuda")
        total_physics = torch.tensor(0.0, device="cuda")

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            pred = model(x)
            loss, l_data, l_sobolev, l_physics = loss_fn(pred, y, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach()
            total_data += l_data.detach()
            total_sobolev += l_sobolev.detach()
            total_physics += l_physics.detach()

        scheduler.step()

        if epoch % 10 == 0:
            avg_loss = total_loss.item() / len(train_loader)
            avg_data = total_data.item() / len(train_loader)
            avg_sob = total_sobolev.item() / len(train_loader)
            avg_phys = total_physics.item() / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("DataLoss/train", avg_data, epoch)
            writer.add_scalar("PhysicsLoss/train", avg_phys, epoch)
            writer.add_scalar("Sobolev/train", avg_sob, epoch)
            writer.add_scalar("Params/alpha", alpha, epoch)
            writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)

    # 5. Save & Evaluate
    torch.save(model.state_dict(), path_config.model_dir / f"{run_name}.pt")
    all_metrics = {"hparam/loss": avg_loss}  # Start with final loss
    evaluations = [
        ("Spectrum", evaluate_ic_spectrum),
        ("Adversarial", evaluate_adversarial_spectrum),
        ("OOD", evaluate_ood_physics),
    ]
    for evaluation_name, evaluation_fn in evaluations:
        print(f"Running Eval: {evaluation_name}...")
        figures_dir = path_config.figure_dir / evaluation_name
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure, r2_dict = evaluation_fn(model)
        figure.savefig(figures_dir / f"{run_name}.png")
        writer.add_figure(f"Evaluation/{evaluation_name}", figure)
        plt.close(figure)
        for k, v in r2_dict.items():
            clean_key = k.replace(" ", "_").replace(":", "").replace("=", "")
            all_metrics[f"hparam/r2_{evaluation_name}_{clean_key}"] = v
    writer.add_hparams(hparam_dict=params, metric_dict=all_metrics)
    writer.close()
    backup_artifacts(run_name)
    print("Run complete.")
    return model


# %% [markdown]
### Main Execution
# This block runs multiple experiments with different data generators.
# We need to bootstrap multiple experiments to compare performance across data generation strategies
# because we did not capture the training experiments with TensorBoard initially.

# %%
if __name__ == "__main__":
    for experiment in [
        ("simple_rc/Dimensionless", generate_dimensionless_data),
        ("simple_rc/Dimensionless_With_Gaussian_Noise", generate_dimensionless_data_with_white_noise),
        (
            "simple_rc/Dimensionless_With_Gaussian_Noise_And_Chirp",
            generate_dimensionless_data_with_white_noise_and_chirp,
        ),
        (
            "simple_rc/Dimensionless_With_Gaussian_Noise_And_Chirp_Log_Uniform",
            generate_dimensionless_data_with_white_noise_and_chirp_log_uniform,
        ),
    ]:
        name, generator = experiment
        run_experiment(experiment_name=name, data_generator=generator)
