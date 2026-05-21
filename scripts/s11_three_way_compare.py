"""S11: render the three-way Adam comparison overlay (FNO/IFT vs FD-SPICE vs Uhlmann).

All three runs use the same loss, hyperparameters, and theta_init. The
overlay shows loss + slew + power vs Adam step plus the 7-panel theta
trajectory, with the three trajectories on shared axes.

Inputs:

* ``--fno-dir``: directory containing the FNO/IFT trajectory.json
  (typically ``runs/sizing/adam_v4_nt7/`` or its tracked equivalent).
* ``--fd-dir``: directory with the forward-FD-SPICE baseline trajectory
  (``runs/sizing/fd_spice_nt7/``).
* ``--uhlmann-dir``: directory with the Uhlmann-surrogate Adam trajectory
  produced by ``s11_uhlmann_adam.py``.

Outputs (under ``--out-dir``):

* ``three_way_loss_slew_power.png``
* ``three_way_theta.png``

Usage::

    python -m scripts.s11_three_way_compare \\
        --fno-dir docs/assets/sizing/v4_nt7 \\
        --fd-dir docs/assets/sizing/v4_nt7/fd_spice \\
        --uhlmann-dir runs/s11_uhlmann/adam \\
        --out-dir docs/assets/sizing/v4_nt7/three_way
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_THETA_LABELS = (
    r"$W_\mathrm{diff}$ (µm)",
    r"$W_\mathrm{mirror}$ (µm)",
    r"$W_\mathrm{tail}$ (µm)",
    r"$L_\mathrm{diff}$ (µm)",
    r"$L_\mathrm{mirror}$ (µm)",
    r"$L_\mathrm{tail}$ (µm)",
    r"$V_\mathrm{bias}$ (V)",
)


@dataclass(frozen=True, slots=True)
class _Traj:
    label: str
    color: str
    steps: np.ndarray
    loss: np.ndarray
    slew: np.ndarray
    power: np.ndarray
    theta: np.ndarray


def _load(name: str, color: str, run_dir: Path) -> _Traj:
    rows = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    return _Traj(
        label=name,
        color=color,
        steps=np.array([r["step"] for r in rows]),
        loss=np.array([r["loss"] for r in rows]),
        slew=np.array([r["slew_rate_v_per_us"] for r in rows]),
        power=np.array([r["power_uw"] for r in rows]),
        theta=np.array([r["theta"] for r in rows]),
    )


def _plot_loss_slew_power(trajs: list[_Traj], out_path: Path, slew_min: float, power_max: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    for t in trajs:
        axes[0].plot(t.steps, t.loss, label=t.label, color=t.color, lw=1.5)
        axes[1].plot(t.steps, t.slew, label=t.label, color=t.color, lw=1.5)
        axes[2].plot(t.steps, t.power, label=t.label, color=t.color, lw=1.5)
    axes[0].set(xlabel="Adam step", ylabel="loss", title="loss")
    axes[0].set_yscale("symlog", linthresh=0.1)
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=9)
    axes[1].axhline(slew_min, color="red", ls=":", alpha=0.5, label=f"spec = {slew_min} V/µs")
    axes[1].set(xlabel="Adam step", ylabel="slew (V/µs)", title="slew rate")
    axes[1].grid(True, ls=":", alpha=0.4)
    axes[1].legend(fontsize=9)
    axes[2].axhline(power_max, color="red", ls=":", alpha=0.5, label=f"cap = {power_max} µW")
    axes[2].set(xlabel="Adam step", ylabel="power (µW)", title="static power")
    axes[2].grid(True, ls=":", alpha=0.4)
    axes[2].legend(fontsize=9)
    fig.suptitle("S11 three-way Adam comparison (FNO/IFT vs FD-SPICE vs Uhlmann surrogate)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_theta(trajs: list[_Traj], out_path: Path) -> None:
    n_theta = trajs[0].theta.shape[1]
    fig, axes = plt.subplots(1, n_theta, figsize=(2.6 * n_theta, 3.4), constrained_layout=True, sharex=True)
    for ax, label, idx in zip(axes, _THETA_LABELS, range(n_theta)):
        for t in trajs:
            ax.plot(t.steps, t.theta[:, idx], color=t.color, lw=1.5, label=t.label)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Adam step")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("S11 three-way θ trajectory")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option("--fno-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--fd-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--uhlmann-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("docs/assets/sizing/v4_nt7/three_way"),
    show_default=True,
)
@click.option("--slew-min", type=float, default=30.0, show_default=True)
@click.option("--power-max", type=float, default=200.0, show_default=True)
def main(fno_dir: Path, fd_dir: Path, uhlmann_dir: Path, out_dir: Path, slew_min: float, power_max: float) -> None:
    """Render the three-way Adam comparison overlay."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    trajs = [
        _load("FNO/IFT (this work)", "#2266aa", fno_dir),
        _load("FD-SPICE (oracle)", "#cc6633", fd_dir),
        _load("Uhlmann surrogate", "#338855", uhlmann_dir),
    ]
    _plot_loss_slew_power(trajs, out_dir / "three_way_loss_slew_power.png", slew_min, power_max)
    _plot_theta(trajs, out_dir / "three_way_theta.png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
