"""
Side-by-side comparison plots for FNO/IFT Adam vs FD-SPICE Adam.

Reads ``trajectory.json`` from two sizing run directories (one ``adam-fno`` and
one ``fd-spice``) and produces two figures used in the W7 writeup:

1. ``comparison_loss_slew.png`` — overlaid loss and slew vs Adam step.
2. ``comparison_theta.png`` — overlaid 5-panel θ trajectory.

The figures land in the FNO run directory by default.

Usage:

    python -m spino.circuit.plot_sizing_comparison \\
        --fno-dir runs/sizing/adam_full_lr5e-2 \\
        --fd-dir runs/sizing/fd_spice_lr5e-2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from spino.circuit.sizing import OtaSizingProblem
from spino.plot_styles import get_palette

logger = logging.getLogger(__name__)

_THETA_LABELS = (
    r"$W_\mathrm{diff}$ (µm)",
    r"$W_\mathrm{mirror}$ (µm)",
    r"$W_\mathrm{tail}$ (µm)",
    r"$L$ (µm)",
    r"$V_\mathrm{bias}$ (V)",
)


@dataclass
class _Trajectory:
    """Trajectory rows loaded from a sizing run."""

    label: str
    steps: np.ndarray
    loss: np.ndarray
    slew: np.ndarray
    power_uw: np.ndarray
    theta: np.ndarray
    sims_total: np.ndarray | None


def _load(run_dir: Path, label: str) -> _Trajectory:
    """Load trajectory.json into a :class:`_Trajectory`."""
    rows = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    slew = np.array([r["slew_rate_v_per_us"] for r in rows])
    power_uw = np.array([r["power_uw"] for r in rows])
    theta = np.array([r["theta"] for r in rows])
    sims_total = np.array([r["sims_total"] for r in rows]) if "sims_total" in rows[0] else None
    return _Trajectory(label, steps, loss, slew, power_uw, theta, sims_total)


def _plot_loss_slew(  # pylint: disable=too-many-locals
    traj_fno: _Trajectory, traj_fd: _Trajectory, problem: OtaSizingProblem, out_path: Path
) -> None:
    """Side-by-side overlay of loss (log) and slew vs Adam step."""
    palette = get_palette(dark=False)
    fig, (ax_loss, ax_slew) = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)

    ax_loss.semilogy(
        traj_fno.steps, np.maximum(traj_fno.loss, 1e-4), color=palette["pred"], lw=1.8, label=traj_fno.label
    )
    ax_loss.semilogy(
        traj_fd.steps, np.maximum(traj_fd.loss, 1e-4), color=palette["pred_sweep"], lw=1.8, label=traj_fd.label, ls="--"
    )
    ax_loss.set_xlabel("Adam step")
    ax_loss.set_ylabel("loss (log)")
    ax_loss.legend(loc="upper right", fontsize=10)
    ax_loss.grid(True, which="both", ls=":", alpha=0.4)
    ax_loss.set_title("Loss vs Adam step")

    ax_slew.plot(traj_fno.steps, traj_fno.slew, color=palette["pred"], lw=1.8, label=traj_fno.label)
    ax_slew.plot(traj_fd.steps, traj_fd.slew, color=palette["pred_sweep"], lw=1.8, label=traj_fd.label, ls="--")
    ax_slew.axhline(
        problem.slew_rate_min_v_per_us,
        color="red",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label=f"spec ({problem.slew_rate_min_v_per_us:.0f} V/µs)",
    )
    ax_slew.set_xlabel("Adam step")
    ax_slew.set_ylabel("slew rate (V/µs)")
    ax_slew.legend(loc="lower right", fontsize=10)
    ax_slew.grid(True, ls=":", alpha=0.4)
    ax_slew.set_title("Slew rate vs Adam step")

    fig.suptitle("FNO/IFT vs FD-SPICE Adam — convergence comparison", y=1.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_theta(  # pylint: disable=too-many-locals
    traj_fno: _Trajectory, traj_fd: _Trajectory, problem: OtaSizingProblem, out_path: Path
) -> None:
    """Overlay 5-panel θ trajectory."""
    bounds = [
        problem.w_diff_bounds,
        problem.w_mirror_bounds,
        problem.w_tail_bounds,
        problem.l_bounds,
        problem.vbias_bounds,
    ]
    palette = get_palette(dark=False)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), constrained_layout=True, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(traj_fno.steps, traj_fno.theta[:, i], color=palette["pred"], lw=1.6, label=traj_fno.label)
        ax.plot(traj_fd.steps, traj_fd.theta[:, i], color=palette["pred_sweep"], lw=1.6, label=traj_fd.label, ls="--")
        lo, hi = bounds[i]
        ax.axhline(lo, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(hi, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(_THETA_LABELS[i])
        ax.set_xlabel("Adam step")
        ax.grid(True, ls=":", alpha=0.4)
        if i == 0:
            ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("FNO/IFT vs FD-SPICE Adam — θ trajectory (red: bounds)", y=1.08)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option(
    "--fno-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Sizing run directory from --mode adam-fno.",
)
@click.option(
    "--fd-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Sizing run directory from --mode fd-spice.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Where to write comparison plots (default: ``fno-dir``).",
)
def main(fno_dir: Path, fd_dir: Path, out_dir: Path | None) -> None:
    """Render FNO-vs-FD-SPICE Adam comparison plots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    target = out_dir if out_dir is not None else fno_dir
    target.mkdir(parents=True, exist_ok=True)

    traj_fno = _load(fno_dir, "FNO Adam (IFT)")
    traj_fd = _load(fd_dir, "FD-SPICE Adam")
    problem = OtaSizingProblem()

    _plot_loss_slew(traj_fno, traj_fd, problem, target / "comparison_loss_slew.png")
    _plot_theta(traj_fno, traj_fd, problem, target / "comparison_theta.png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
