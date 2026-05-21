"""S11: train the Uhlmann-style performance surrogate on the LHS SPICE set.

Reads ``samples.json`` produced by ``scripts.s11_collect_spice_training_set``
and fits a small MLP from the 7-component design vector
$(W_\\mathrm{diff}, W_\\mathrm{mirror}, W_\\mathrm{tail}, L_\\mathrm{diff},
L_\\mathrm{mirror}, L_\\mathrm{tail}, V_\\mathrm{bias})$ to the SPICE
performance metrics (slew rate in V/µs, static power in µW, peak swing in
V). The surrogate is the "Uhlmann route": gradients are routed through a
learned abstraction of circuit behaviour rather than through device-level
operators inside the residual.

Outputs (under ``--output-dir``):

* ``uhlmann_surrogate.pt`` — trained model state dict + normalisation stats.
* ``train_history.json`` — per-epoch losses on train/test splits.
* ``test_metrics.json`` — held-out test R² per output plus the held-out
  *gradient* R² (∂slew_pred/∂θ vs central-FD-SPICE ∂slew_SPICE/∂θ) on a
  small set of points; this matters because predicting performance well
  does not imply predicting performance *gradients* well, and the latter
  is what an Uhlmann Adam loop actually consumes.

Usage::

    python -m scripts.s11_train_uhlmann_surrogate \\
        --samples runs/s11_uhlmann/training_set/samples.json \\
        --output-dir runs/s11_uhlmann/surrogate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import torch
from torch import nn

from spino.circuit.sizing import OtaSizingProblem
from spino.circuit.tuning import OtaDesignPoint, simulate_ota_design_point

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _NormStats:
    """Per-feature mean / std used to z-score the inputs and outputs."""

    theta_mean: torch.Tensor
    theta_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor


class UhlmannSurrogate(nn.Module):
    """Small MLP from theta -> (slew, power, swing).

    The architecture is deliberately conservative: the prior art (Uhlmann
    et al., MLCAD'23) trains a performance-metric NN surrogate and
    backpropagates through it for gm/ID sizing; replicating the same
    differentiable-abstraction shape on the SPINO OTA needs roughly the
    same capacity.
    """

    def __init__(self, in_dim: int = 7, out_dim: int = 3, hidden_dim: int = 128, n_hidden: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, theta_norm: torch.Tensor) -> torch.Tensor:
        return self.net(theta_norm)


def _load_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta, targets) for converged samples only."""
    data = json.loads(path.read_text(encoding="utf-8"))
    thetas: list[list[float]] = []
    targets: list[list[float]] = []
    for rec in data:
        if not rec.get("converged"):
            continue
        t = rec["theta"]
        thetas.append(
            [
                t["w_diff_um"],
                t["w_mirror_um"],
                t["w_tail_um"],
                t["l_diff_um"],
                t["l_mirror_um"],
                t["l_tail_um"],
                t["vbias_v"],
            ]
        )
        targets.append([rec["slew_v_per_us"], rec["power_uw"], rec["peak_swing_v"]])
    return np.asarray(thetas, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _r2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Coefficient of determination, per-column or flattened."""
    ss_res = float(np.sum((pred - ref) ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def _train(
    theta_train: np.ndarray,
    target_train: np.ndarray,
    theta_test: np.ndarray,
    target_test: np.ndarray,
    hidden_dim: int,
    n_hidden: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[UhlmannSurrogate, _NormStats, list[dict]]:
    """Fit the surrogate with z-scored inputs/outputs, MSE loss."""
    theta_mean = torch.tensor(theta_train.mean(axis=0), dtype=torch.float32, device=device)
    theta_std = torch.tensor(theta_train.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    target_mean = torch.tensor(target_train.mean(axis=0), dtype=torch.float32, device=device)
    target_std = torch.tensor(target_train.std(axis=0) + 1e-6, dtype=torch.float32, device=device)
    stats = _NormStats(theta_mean, theta_std, target_mean, target_std)

    model = UhlmannSurrogate(in_dim=theta_train.shape[1], out_dim=target_train.shape[1], hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    th_tr = (torch.tensor(theta_train, device=device) - theta_mean) / theta_std
    tg_tr = (torch.tensor(target_train, device=device) - target_mean) / target_std
    th_te = (torch.tensor(theta_test, device=device) - theta_mean) / theta_std
    tg_te = (torch.tensor(target_test, device=device) - target_mean) / target_std

    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        pred = model(th_tr)
        loss_tr = nn.functional.mse_loss(pred, tg_tr)
        opt.zero_grad()
        loss_tr.backward()
        opt.step()
        sched.step()
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                te_pred = model(th_te)
                loss_te = nn.functional.mse_loss(te_pred, tg_te)
            history.append({"epoch": epoch + 1, "loss_train": float(loss_tr), "loss_test": float(loss_te)})
            logger.info("epoch %d | train MSE %.4f | test MSE %.4f", epoch + 1, loss_tr.item(), loss_te.item())
    return model, stats, history


def _gradient_r2(
    model: UhlmannSurrogate,
    stats: _NormStats,
    held_out_theta: np.ndarray,
    problem: OtaSizingProblem,
    device: torch.device,
    eps_rel: float = 0.01,
) -> dict:
    """Compare ∂slew_pred/∂θ (surrogate autograd) to central-FD-SPICE on slew."""
    n_held, n_theta = held_out_theta.shape
    grad_surr = np.zeros((n_held, n_theta), dtype=np.float64)
    grad_spice = np.full((n_held, n_theta), np.nan, dtype=np.float64)
    target_mean = stats.target_mean.cpu().numpy()
    target_std = stats.target_std.cpu().numpy()

    model.eval()
    for i, theta_vec in enumerate(held_out_theta):
        th_t = torch.tensor(theta_vec, dtype=torch.float32, device=device, requires_grad=True)
        th_norm = (th_t - stats.theta_mean) / stats.theta_std
        pred_norm = model(th_norm)
        slew_pred = pred_norm[0] * stats.target_std[0] + stats.target_mean[0]
        grad_surr_i = torch.autograd.grad(slew_pred, th_t)[0].detach().cpu().numpy()
        grad_surr[i, :] = grad_surr_i

        for j in range(n_theta):
            eps = max(abs(theta_vec[j]) * eps_rel, 1e-4)
            plus = theta_vec.copy()
            plus[j] += eps
            minus = theta_vec.copy()
            minus[j] -= eps
            try:
                m_p = simulate_ota_design_point(
                    OtaDesignPoint(diff_w_um=float(plus[0]), mirror_w_um=float(plus[1])),
                    vdd=problem.vdd, vcm_v=problem.vcm, step_amp_v=problem.step_amp,
                    diff_l_um=float(plus[3]), mirror_l_um=float(plus[4]),
                    tail_w_um=float(plus[2]), tail_l_um=float(plus[5]),
                    vbias_v=float(plus[6]),
                    t_step_start=problem.t_step_start, t_end=problem.t_end,
                    t_step=problem.t_step, c_load_f=problem.c_load_f,
                    pdk_root=problem.pdk_root,
                )
                m_m = simulate_ota_design_point(
                    OtaDesignPoint(diff_w_um=float(minus[0]), mirror_w_um=float(minus[1])),
                    vdd=problem.vdd, vcm_v=problem.vcm, step_amp_v=problem.step_amp,
                    diff_l_um=float(minus[3]), mirror_l_um=float(minus[4]),
                    tail_w_um=float(minus[2]), tail_l_um=float(minus[5]),
                    vbias_v=float(minus[6]),
                    t_step_start=problem.t_step_start, t_end=problem.t_end,
                    t_step=problem.t_step, c_load_f=problem.c_load_f,
                    pdk_root=problem.pdk_root,
                )
                if m_p.converged and m_m.converged:
                    grad_spice[i, j] = (m_p.slew_rate_v_per_us - m_m.slew_rate_v_per_us) / (2.0 * eps)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("FD-SPICE failed at theta=%s component %d: %s", theta_vec.tolist(), j, exc)
        logger.info("held-out gradient point %d / %d done", i + 1, n_held)

    mask = np.all(np.isfinite(grad_spice), axis=1)
    if mask.sum() == 0:
        return {"n_used": 0, "per_component_r2": [float("nan")] * n_theta, "overall_r2": float("nan")}
    surr = grad_surr[mask]
    spi = grad_spice[mask]
    per_comp = [float(_r2(surr[:, k], spi[:, k])) for k in range(n_theta)]
    overall = float(_r2(surr.flatten(), spi.flatten()))
    return {
        "n_used": int(mask.sum()),
        "per_component_r2": per_comp,
        "overall_r2": overall,
        "grad_surrogate": surr.tolist(),
        "grad_spice": spi.tolist(),
    }


@click.command()
@click.option("--samples", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s11_uhlmann/surrogate"),
    show_default=True,
)
@click.option("--epochs", type=int, default=2000, show_default=True)
@click.option("--lr", type=float, default=3e-3, show_default=True)
@click.option("--hidden-dim", type=int, default=128, show_default=True)
@click.option("--n-hidden", type=int, default=3, show_default=True)
@click.option("--test-frac", type=float, default=0.15, show_default=True)
@click.option("--n-grad-eval", type=int, default=15, show_default=True, help="Held-out points for gradient R² check.")
@click.option("--skip-gradient-check", is_flag=True, default=False, help="Skip the slow SPICE gradient-R² check (only the MLP fit + test R² are reported).")
@click.option("--seed", type=int, default=0, show_default=True, help="RNG seed for train/test split + MLP init.")
@click.option("--device", type=str, default=None)
def main(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    samples: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    hidden_dim: int,
    n_hidden: int,
    test_frac: float,
    n_grad_eval: int,
    skip_gradient_check: bool,
    seed: int,
    device: str | None,
) -> None:
    """Train the Uhlmann-style θ→(slew, power, swing) surrogate."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    theta_all, target_all = _load_samples(samples)
    n_total = theta_all.shape[0]
    logger.info("Loaded %d converged samples from %s", n_total, samples)

    rng = np.random.default_rng(123 + seed)
    idx = rng.permutation(n_total)
    n_test = max(1, int(round(n_total * test_frac)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    theta_train, target_train = theta_all[train_idx], target_all[train_idx]
    theta_test, target_test = theta_all[test_idx], target_all[test_idx]
    logger.info("Split: %d train / %d test", theta_train.shape[0], theta_test.shape[0])

    model, stats, history = _train(
        theta_train, target_train, theta_test, target_test,
        hidden_dim=hidden_dim, n_hidden=n_hidden, epochs=epochs, lr=lr, device=torch_device,
    )

    model.eval()
    with torch.no_grad():
        th_te = (torch.tensor(theta_test, device=torch_device) - stats.theta_mean) / stats.theta_std
        pred_te = model(th_te) * stats.target_std + stats.target_mean
        pred_te = pred_te.cpu().numpy()
    test_r2 = {
        "slew": float(_r2(pred_te[:, 0], target_test[:, 0])),
        "power": float(_r2(pred_te[:, 1], target_test[:, 1])),
        "swing": float(_r2(pred_te[:, 2], target_test[:, 2])),
    }
    logger.info("Test R²: slew %.4f, power %.4f, swing %.4f", test_r2["slew"], test_r2["power"], test_r2["swing"])

    if skip_gradient_check:
        grad_check = {"skipped": True}
        logger.info("Skipping SPICE gradient R² check (--skip-gradient-check).")
    else:
        grad_held = theta_test[: min(n_grad_eval, theta_test.shape[0])]
        problem = OtaSizingProblem()
        grad_check = _gradient_r2(model, stats, grad_held, problem, torch_device)
        logger.info("Gradient R² (overall): %.4f on %d held-out points", grad_check["overall_r2"], grad_check["n_used"])

    torch.save(
        {
            "state_dict": model.state_dict(),
            "theta_mean": stats.theta_mean.cpu(),
            "theta_std": stats.theta_std.cpu(),
            "target_mean": stats.target_mean.cpu(),
            "target_std": stats.target_std.cpu(),
            "arch": {"in_dim": 7, "out_dim": 3, "hidden_dim": hidden_dim, "n_hidden": n_hidden},
        },
        output_dir / "uhlmann_surrogate.pt",
    )
    (output_dir / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(
        json.dumps(
            {
                "test_r2": test_r2,
                "gradient_r2": grad_check,
                "n_train": int(theta_train.shape[0]),
                "n_test": int(theta_test.shape[0]),
                "arch": {"hidden_dim": hidden_dim, "n_hidden": n_hidden, "epochs": epochs, "lr": lr},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
