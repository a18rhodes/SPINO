"""S11 seed-variance test for the Uhlmann route.

Trains the same surrogate architecture on the same SPICE training set at
multiple RNG seeds (different MLP weight inits + train/test splits), runs
Uhlmann Adam at multiple theta_init perturbations per surrogate, and
records the SPICE-validated converged design per run. The aggregate tests
whether the L_mirror -> upper-bound landing reported in the headline S11
result is robust to surrogate-fit and starting-point variance or whether
it is a single-fit accident.

For each ``(surrogate_seed, theta_init_label)`` combination the driver
emits:

* The trained surrogate checkpoint (gitignored).
* The Uhlmann Adam trajectory.
* The SPICE re-validation summary at theta_final.

Aggregate output ``runs/s11_uhlmann/seed_variance/aggregate.json`` records
the converged design and SPICE-truth metrics per run; the doc reports the
distribution and a bar chart.

Usage::

    python -m scripts.s11_seed_variance --output-dir runs/s11_uhlmann/seed_variance
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SAMPLES = Path("runs/s11_uhlmann/training_set/samples.json")
_THETA_INIT_LABELS: dict[str, str] = {
    "canonical": "3.0,3.0,1.0,0.40,0.40,0.40,0.9",
    "under_slew_low_w": "2.0,2.0,0.8,0.40,0.40,0.40,0.8",
    "over_slew_high_l": "4.0,4.0,1.5,0.50,0.50,0.50,1.1",
}


@dataclass(frozen=True, slots=True)
class _RunRecord:
    surrogate_seed: int
    theta_init_label: str
    theta_init: list[float]
    theta_final: list[float]
    surrogate_test_r2_slew: float
    spice_slew_v_per_us: float
    spice_power_uw: float
    spice_dc_gain_v_per_v: float
    spice_peak_swing_v: float
    spice_converged: bool


def _train(samples: Path, output_dir: Path, seed: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.s11_train_uhlmann_surrogate",
        "--samples",
        str(samples),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--skip-gradient-check",
        "--epochs",
        "3000",
        "--lr",
        "3e-3",
    ]
    subprocess.run(cmd, check=True, env={**__import__("os").environ, "PYTHONPATH": "/app"})


def _run_adam(surrogate_pt: Path, theta_init: str, output_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "scripts.s11_uhlmann_adam",
        "--surrogate",
        str(surrogate_pt),
        "--theta-init",
        theta_init,
        "--n-iters",
        "50",
        "--lr",
        "5e-2",
        "--validate-spice",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, env={**__import__("os").environ, "PYTHONPATH": "/app"})


def _read_record(seed: int, label: str, theta_init: list[float], surr_dir: Path, adam_dir: Path) -> _RunRecord:
    test_metrics = json.loads((surr_dir / "test_metrics.json").read_text(encoding="utf-8"))
    spice_summary = json.loads((adam_dir / "spice_validation" / "summary.json").read_text(encoding="utf-8"))
    theta_final = json.loads((adam_dir / "theta_final.json").read_text(encoding="utf-8"))["theta"]
    m = spice_summary["metrics"]
    return _RunRecord(
        surrogate_seed=seed,
        theta_init_label=label,
        theta_init=theta_init,
        theta_final=theta_final,
        surrogate_test_r2_slew=float(test_metrics["test_r2"]["slew"]),
        spice_slew_v_per_us=float(m["slew_rate_v_per_us"]),
        spice_power_uw=float(m["static_current_a"]) * 1.8 * 1e6,
        spice_dc_gain_v_per_v=float(m["dc_gain_v_per_v"]),
        spice_peak_swing_v=float(m["peak_swing_v"]),
        spice_converged=bool(spice_summary["converged"]),
    )


def _plot(records: list[_RunRecord], output_dir: Path) -> None:
    """Bar chart of final L_mirror across runs, grouped by surrogate seed."""
    labels = [f"s{r.surrogate_seed}\n{r.theta_init_label[:8]}" for r in records]
    l_mirror = [r.theta_final[4] for r in records]  # index 4 = L_mirror in 7-vector
    gain = [r.spice_dc_gain_v_per_v for r in records]
    slew = [r.spice_slew_v_per_us for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    axes[0].bar(range(len(records)), l_mirror, color="#338855")
    axes[0].axhline(0.18, color="red", ls=":", alpha=0.5, label="lower bound 0.18 µm")
    axes[0].axhline(0.50, color="red", ls=":", alpha=0.5, label="upper bound 0.50 µm")
    axes[0].set(ylabel=r"final $L_\mathrm{mirror}$ (µm)", title="final L_mirror across runs")
    axes[0].set_xticks(range(len(records)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.3)

    axes[1].bar(range(len(records)), gain, color="#cc6633")
    axes[1].set(ylabel="SPICE DC gain (V/V)", title="DC gain at θ_final")
    axes[1].set_xticks(range(len(records)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].grid(True, ls=":", alpha=0.3)

    axes[2].bar(range(len(records)), slew, color="#2266aa")
    axes[2].axhline(30, color="red", ls=":", alpha=0.5, label="spec = 30 V/µs")
    axes[2].set(ylabel="SPICE slew (V/µs)", title="slew at θ_final")
    axes[2].set_xticks(range(len(records)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, ls=":", alpha=0.3)

    fig.suptitle("S11 Uhlmann seed-variance test — final L_mirror, DC gain, slew rate")
    fig.savefig(output_dir / "seed_variance.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", output_dir / "seed_variance.png")


@click.command()
@click.option("--samples", type=click.Path(exists=True, path_type=Path), default=_DEFAULT_SAMPLES, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s11_uhlmann/seed_variance"),
    show_default=True,
)
@click.option("--surrogate-seeds", type=str, default="0,7,13,17,42", show_default=True, help="Comma-separated surrogate RNG seeds.")
def main(samples: Path, output_dir: Path, surrogate_seeds: str) -> None:
    """Run the multi-seed Uhlmann variance test and write aggregate JSON + plots."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in surrogate_seeds.split(",")]
    logger.info("Surrogate seeds: %s; theta_init labels: %s", seeds, list(_THETA_INIT_LABELS))

    records: list[_RunRecord] = []
    for seed in seeds:
        surr_dir = output_dir / f"surrogate_s{seed}"
        surr_pt = surr_dir / "uhlmann_surrogate.pt"
        if not surr_pt.exists():
            logger.info("=== train surrogate seed=%d ===", seed)
            _train(samples, surr_dir, seed)
        else:
            logger.info("Surrogate seed=%d already exists at %s; reusing", seed, surr_pt)
        for label, theta_init_str in _THETA_INIT_LABELS.items():
            adam_dir = output_dir / f"surrogate_s{seed}" / f"adam_{label}"
            if not (adam_dir / "spice_validation" / "summary.json").exists():
                logger.info("=== Adam seed=%d label=%s ===", seed, label)
                _run_adam(surr_pt, theta_init_str, adam_dir)
            else:
                logger.info("Adam result already exists for seed=%d, label=%s; reusing", seed, label)
            theta_init = [float(v) for v in theta_init_str.split(",")]
            records.append(_read_record(seed, label, theta_init, surr_dir, adam_dir))

    aggregate = {
        "surrogate_seeds": seeds,
        "theta_init_labels": _THETA_INIT_LABELS,
        "n_runs": len(records),
        "runs": [dataclasses.asdict(r) for r in records],
    }
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    logger.info("Wrote %s with %d records", output_dir / "aggregate.json", len(records))

    l_mirror = np.array([r.theta_final[4] for r in records])
    gain = np.array([r.spice_dc_gain_v_per_v for r in records])
    slew = np.array([r.spice_slew_v_per_us for r in records])
    power = np.array([r.spice_power_uw for r in records])
    logger.info("L_mirror across runs: min=%.3f, mean=%.3f, max=%.3f, std=%.3f", l_mirror.min(), l_mirror.mean(), l_mirror.max(), l_mirror.std())
    logger.info("DC gain across runs:   min=%.2f, mean=%.2f, max=%.2f, std=%.2f", gain.min(), gain.mean(), gain.max(), gain.std())
    logger.info("Slew across runs:      min=%.2f, mean=%.2f, max=%.2f", slew.min(), slew.mean(), slew.max())
    logger.info("Power across runs:     min=%.2f, mean=%.2f, max=%.2f", power.min(), power.mean(), power.max())

    _plot(records, output_dir)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
