"""S11: Adam loop with autograd through the trained Uhlmann-style surrogate.

Loads the surrogate trained by ``s11_train_uhlmann_surrogate.py`` and runs
the same Adam loss used by ``spino.circuit.sizing`` (slew ReLU hinge +
power ReLU hinge) but with gradients flowing through the learned MLP from
θ to (slew, power, swing). This is the prior-art comparison point against
the FNO/IFT route: same loss, same hyperparameters, same θ_init, gradient
mechanism differs.

After Adam terminates, runs an NGSpice re-validation at the converged θ
so the SPICE-truth metrics are directly comparable to the FNO/IFT and
FD-SPICE final designs.

Outputs (under ``--output-dir``):

* ``trajectory.json`` — per-step (loss, slew, power, swing, theta).
* ``theta_final.json`` — final θ.
* ``spice_validation/summary.json`` — SPICE re-validation at θ_final.
* ``loss_and_slew.png``, ``theta_trajectory.png``, ``fno_vs_spice.png``
  rendered via :mod:`spino.circuit.plot_sizing_trajectory` for parity
  with the FNO/IFT plot set.

Usage::

    python -m scripts.s11_uhlmann_adam \\
        --surrogate runs/s11_uhlmann/surrogate/uhlmann_surrogate.pt \\
        --theta-init "3.0,3.0,1.0,0.40,0.40,0.40,0.9" \\
        --output-dir runs/s11_uhlmann/adam
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path

import click
import torch
import torch.nn.functional as F

from spino.circuit.sizing import OtaSizingProblem, spice_validate
from scripts.s11_train_uhlmann_surrogate import UhlmannSurrogate

logger = logging.getLogger(__name__)


def _load_surrogate(path: Path, device: torch.device) -> tuple[UhlmannSurrogate, dict]:
    """Reconstruct the surrogate + normalisation buffers from the saved checkpoint."""
    blob = torch.load(path, map_location=device, weights_only=False)
    arch = blob["arch"]
    model = UhlmannSurrogate(
        in_dim=arch["in_dim"],
        out_dim=arch["out_dim"],
        hidden_dim=arch["hidden_dim"],
        n_hidden=arch["n_hidden"],
    ).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    stats = {
        "theta_mean": blob["theta_mean"].to(device),
        "theta_std": blob["theta_std"].to(device),
        "target_mean": blob["target_mean"].to(device),
        "target_std": blob["target_std"].to(device),
    }
    return model, stats


def _surrogate_forward(model: UhlmannSurrogate, stats: dict, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (slew, power, swing) with autograd link to ``theta``."""
    theta_norm = (theta - stats["theta_mean"]) / stats["theta_std"]
    out_norm = model(theta_norm.unsqueeze(0))
    out = out_norm.squeeze(0) * stats["target_std"] + stats["target_mean"]
    return out[0], out[1], out[2]


def _loss_fn(slew: torch.Tensor, power: torch.Tensor, problem: OtaSizingProblem) -> torch.Tensor:
    loss_slew = problem.slew_weight * F.relu(torch.tensor(problem.slew_rate_min_v_per_us, device=slew.device) - slew)
    loss_power = problem.power_weight * F.relu(power - torch.tensor(problem.power_max_uw, device=power.device))
    return loss_slew + loss_power


@click.command()
@click.option("--surrogate", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--theta-init", type=str, default="3.0,3.0,1.0,0.40,0.40,0.40,0.9", show_default=True)
@click.option("--n-iters", type=int, default=50, show_default=True)
@click.option("--lr", type=float, default=5e-2, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s11_uhlmann/adam"),
    show_default=True,
)
@click.option("--slew-min", type=float, default=30.0, show_default=True)
@click.option("--power-max", type=float, default=200.0, show_default=True)
@click.option("--validate-spice", is_flag=True, default=False)
@click.option("--device", type=str, default=None)
def main(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    surrogate: Path,
    theta_init: str,
    n_iters: int,
    lr: float,
    output_dir: Path,
    slew_min: float,
    power_max: float,
    validate_spice: bool,
    device: str | None,
) -> None:
    """Adam loop with autograd-through-Uhlmann-MLP gradients."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    problem = OtaSizingProblem(slew_rate_min_v_per_us=slew_min, power_max_uw=power_max)
    model, stats = _load_surrogate(surrogate, torch_device)

    theta_vals = [float(v) for v in theta_init.split(",")]
    if len(theta_vals) != 7:
        raise click.BadParameter("theta-init must have 7 components")
    theta = torch.tensor(theta_vals, dtype=torch.float32, device=torch_device, requires_grad=True)
    lower = problem.lower_bounds.to(torch_device)
    upper = problem.upper_bounds.to(torch_device)
    optimizer = torch.optim.Adam([theta], lr=lr)

    trajectory: list[dict] = []
    for step in range(n_iters):
        optimizer.zero_grad()
        t0 = time.perf_counter()
        slew, power, swing = _surrogate_forward(model, stats, theta)
        loss = _loss_fn(slew, power, problem)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            theta.clamp_(lower, upper)
        wall = time.perf_counter() - t0
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "slew_rate_v_per_us": float(slew.detach()),
            "power_uw": float(power.detach()),
            "swing_v": float(swing.detach()),
            "theta": theta.detach().tolist(),
            "wall_s": round(wall, 4),
        }
        trajectory.append(row)
        logger.info(
            "Step %3d | loss=%.4f | slew=%.1f V/µs | power=%.0f µW | swing=%.3f V | θ=%s",
            step, row["loss"], row["slew_rate_v_per_us"], row["power_uw"], row["swing_v"],
            [f"{v:.3f}" for v in row["theta"]],
        )

    (output_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
    final_theta = theta.detach()
    (output_dir / "theta_final.json").write_text(
        json.dumps(
            {
                "theta": final_theta.cpu().tolist(),
                "layout": ["W_diff_um", "W_mirror_um", "W_tail_um", "L_diff_um", "L_mirror_um", "L_tail_um", "V_bias_v"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if validate_spice:
        summary = spice_validate(final_theta, problem, output_dir)
        logger.info("SPICE validation summary: %s", summary)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
