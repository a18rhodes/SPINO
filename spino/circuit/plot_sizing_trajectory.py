"""
Sizing-trajectory plotting for the W5-6 Adam OTA sizing experiment.

Reads ``trajectory.json`` (and optionally ``spice_validation/summary.json``)
from a sizing run directory and writes three figures used in the documentation
and paper:

1. ``loss_and_slew.png`` — loss (log) and slew rate vs step, with spec line.
2. ``theta_trajectory.png`` — 5-panel θ component trajectory with bound lines.
3. ``fno_vs_spice.png`` — bar chart comparing FNO-predicted to SPICE-validated
   metrics at the converged θ.

Usage:

    python -m spino.circuit.plot_sizing_trajectory \\
        --run-dir runs/sizing/adam_full_lr5e-2
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
    r"$L_\mathrm{diff}$ (µm)",
    r"$L_\mathrm{mirror}$ (µm)",
    r"$L_\mathrm{tail}$ (µm)",
    r"$V_\mathrm{bias}$ (V)",
)


@dataclass
class _Trajectory:
    """Loaded trajectory + optional SPICE validation summary."""

    steps: np.ndarray
    loss: np.ndarray
    slew: np.ndarray
    power_uw: np.ndarray
    theta: np.ndarray  # (n_steps, 5)
    spice_summary: dict | None


def _load(run_dir: Path) -> _Trajectory:
    """Load trajectory.json and spice_validation/summary.json (if present)."""
    traj_path = run_dir / "trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory.json missing in {run_dir}")
    rows = json.loads(traj_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"trajectory.json is empty in {run_dir}")

    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    slew = np.array([r["slew_rate_v_per_us"] for r in rows])
    power_uw = np.array([r["power_uw"] for r in rows])
    theta = np.array([r["theta"] for r in rows])

    spice_path = run_dir / "spice_validation" / "summary.json"
    spice_summary = None
    if spice_path.exists():
        spice_summary = json.loads(spice_path.read_text(encoding="utf-8"))

    return _Trajectory(steps, loss, slew, power_uw, theta, spice_summary)


def _plot_loss_and_slew(traj: _Trajectory, problem: OtaSizingProblem, out_path: Path) -> None:
    """Twin-axis loss (log) + slew (linear) vs step with spec line."""
    palette = get_palette(dark=False)
    fig, ax_loss = plt.subplots(figsize=(7.5, 4.0), constrained_layout=True)

    ax_loss.semilogy(traj.steps, np.maximum(traj.loss, 1e-4), color=palette["pred"], lw=1.8, label="loss")
    ax_loss.set_xlabel("Adam step")
    ax_loss.set_ylabel("loss (log)", color=palette["pred"])
    ax_loss.tick_params(axis="y", labelcolor=palette["pred"])
    ax_loss.grid(True, which="both", ls=":", alpha=0.4)

    ax_slew = ax_loss.twinx()
    ax_slew.plot(traj.steps, traj.slew, color=palette["gt"], lw=1.8, label="slew rate")
    ax_slew.axhline(
        problem.slew_rate_min_v_per_us,
        color="red",
        ls="--",
        lw=1.0,
        alpha=0.7,
        label=f"spec ({problem.slew_rate_min_v_per_us:.0f} V/µs)",
    )
    ax_slew.set_ylabel("slew rate (V/µs)", color=palette["gt"])
    ax_slew.tick_params(axis="y", labelcolor=palette["gt"])

    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_slew, labels_slew = ax_slew.get_legend_handles_labels()
    ax_slew.legend(lines_loss + lines_slew, labels_loss + labels_slew, loc="center right", fontsize=9)

    ax_loss.set_title("Adam sizing trajectory — loss and slew rate")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_theta_trajectory(traj: _Trajectory, problem: OtaSizingProblem, out_path: Path) -> None:
    """7-panel θ trajectory with bound lines."""
    bounds = [
        problem.w_diff_bounds,
        problem.w_mirror_bounds,
        problem.w_tail_bounds,
        problem.l_diff_bounds,
        problem.l_mirror_bounds,
        problem.l_tail_bounds,
        problem.vbias_bounds,
    ]
    palette = get_palette(dark=False)
    n_panels = len(bounds)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.6 * n_panels, 3.2), constrained_layout=True, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(traj.steps, traj.theta[:, i], color=palette["pred"], lw=1.6)
        lo, hi = bounds[i]
        ax.axhline(lo, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(hi, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(_THETA_LABELS[i])
        ax.set_xlabel("Adam step")
        ax.grid(True, ls=":", alpha=0.4)
    fig.suptitle("Adam sizing trajectory — design parameters (red: bounds)", y=1.06)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_fno_vs_spice(  # pylint: disable=too-many-locals
    traj: _Trajectory, problem: OtaSizingProblem, out_path: Path
) -> None:
    """Bar chart of FNO-predicted vs SPICE-validated metrics at θ_final."""
    if traj.spice_summary is None:
        logger.warning("No spice_validation summary; skipping FNO-vs-SPICE plot.")
        return

    fno_slew = float(traj.slew[-1])
    fno_power = float(traj.power_uw[-1])
    spice_metrics = traj.spice_summary["metrics"]
    spice_slew = float(spice_metrics["slew_rate_v_per_us"])
    spice_power = float(spice_metrics["static_current_a"]) * problem.vdd * 1e6  # µW

    labels = ["slew (V/µs)", "power (µW)"]
    fno_vals = [fno_slew, fno_power]
    spice_vals = [spice_slew, spice_power]
    specs = [problem.slew_rate_min_v_per_us, problem.power_max_uw]

    palette = get_palette(dark=False)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    bars_fno = ax.bar(x - width / 2, fno_vals, width, label="FNO predicted", color=palette["pred"])
    bars_spice = ax.bar(
        x + width / 2, spice_vals, width, label="SPICE validated", color=palette["gt"], edgecolor="black"
    )

    for rect, val in zip(bars_fno, fno_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for rect, val in zip(bars_spice, spice_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    for i, spec in enumerate(specs):
        ax.hlines(spec, x[i] - 0.5, x[i] + 0.5, color="red", ls="--", lw=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value")
    ax.set_title(r"FNO vs SPICE at $\theta_\mathrm{final}$  (red dashed = spec)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="Sizing run directory containing trajectory.json.",
)
def main(run_dir: Path) -> None:
    """Generate sizing-trajectory figures from a completed Adam run."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    traj = _load(run_dir)
    problem = OtaSizingProblem()

    _plot_loss_and_slew(traj, problem, run_dir / "loss_and_slew.png")
    _plot_theta_trajectory(traj, problem, run_dir / "theta_trajectory.png")
    _plot_fno_vs_spice(traj, problem, run_dir / "fno_vs_spice.png")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
