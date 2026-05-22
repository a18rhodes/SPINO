"""S8: aggregate NFET MLP multi-seed ablation results.

Parses the training-log Fast Dataset R² / SPICE Transfer R² / Transfer
SubTh-R² / Output R² from one log file per (capacity, seed) and reports
mean ± std across seeds per capacity. Adds the existing single-seed
baselines (mosfet_mlp_baseline_XpV7KFHL, mosfet_mlp_h128_OZyiUsFA) as
``seed = 0`` rows using the canonical numbers published in
``docs/results.md`` § MLP ablation. The two new seeds per capacity come
from the just-finished ``mosfet_nmos_mlp_h{64,128}_s{1,2}`` runs.

Output: ``runs/s8_mlp_seed_variance/aggregate.json`` and a small bar
chart.

Usage::

    python -m scripts.s8_aggregate_mlp_ablation \\
        --logs-dir runs/training_logs \\
        --output-dir runs/s8_mlp_seed_variance
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_FAST_RE = re.compile(r"Fast Dataset R²[^\d\n-]*(-?\d+\.\d+)")
_TRANSFER_RE = re.compile(r"SPICE Transfer R²[^\d\n-]*(-?\d+\.\d+)")
_SUBTH_RE = re.compile(r"SPICE Transfer SubTh-R²[^\d\n-]*(-?\d+\.\d+)")
_OUTPUT_RE = re.compile(r"SPICE Output R²[^\d\n-]*(-?\d+\.\d+)")


@dataclass(frozen=True, slots=True)
class _Row:
    """One (capacity, seed) row of the aggregation table."""

    capacity: str
    seed: int
    fast_r2: float
    transfer_r2: float
    transfer_subth_r2: float
    output_r2: float


def _parse_log(path: Path) -> dict[str, float] | None:
    """Pull the four post-training R² metrics from a train log file."""
    if not path.exists():
        logger.warning("Log file missing: %s", path)
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = _ANSI_RE.sub("", raw)
    fast = _FAST_RE.search(text)
    trans = _TRANSFER_RE.search(text)
    subth = _SUBTH_RE.search(text)
    out = _OUTPUT_RE.search(text)
    if not all([fast, trans, subth, out]):
        logger.warning("Could not parse all four R² values from %s", path)
        return None
    return {
        "fast_r2": float(fast.group(1)),
        "transfer_r2": float(trans.group(1)),
        "transfer_subth_r2": float(subth.group(1)),
        "output_r2": float(out.group(1)),
    }


@click.command()
@click.option(
    "--logs-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("runs/training_logs"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s8_mlp_seed_variance"),
    show_default=True,
)
def main(logs_dir: Path, output_dir: Path) -> None:
    """Aggregate Fast / Transfer / SubTh / Output R² across MLP seeds."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Existing single-seed baselines (canonical numbers, docs/results.md MLP ablation).
    rows: list[_Row] = [
        _Row("h64", 0, fast_r2=-4.42, transfer_r2=0.9990, transfer_subth_r2=0.9856, output_r2=0.9456),
        _Row("h128", 0, fast_r2=-5.43, transfer_r2=0.9989, transfer_subth_r2=0.9631, output_r2=0.9763),
    ]
    # New seeds parsed from this session's logs.
    new_logs = {
        ("h64", 1): "nmos_mlp_h64_s1.log",
        ("h64", 2): "nmos_mlp_h64_s2.log",
        ("h128", 1): "nmos_mlp_h128_s1.log",
        ("h128", 2): "nmos_mlp_h128_s2.log",
    }
    for (cap, seed), fname in new_logs.items():
        parsed = _parse_log(logs_dir / fname)
        if parsed is None:
            logger.error("Skipping (cap=%s, seed=%d) — could not parse %s", cap, seed, fname)
            continue
        rows.append(_Row(cap, seed, **parsed))

    aggregate: dict = {"per_run": [r.__dict__ if hasattr(r, "__dict__") else _row_dict(r) for r in rows]}
    # Manual dict per Row because slots=True kills __dict__.
    aggregate["per_run"] = [
        {
            "capacity": r.capacity,
            "seed": r.seed,
            "fast_r2": r.fast_r2,
            "transfer_r2": r.transfer_r2,
            "transfer_subth_r2": r.transfer_subth_r2,
            "output_r2": r.output_r2,
        }
        for r in rows
    ]

    # Mean ± std across seeds per capacity.
    by_cap: dict[str, list[_Row]] = {}
    for r in rows:
        by_cap.setdefault(r.capacity, []).append(r)
    summary: dict = {}
    for cap, run_list in by_cap.items():
        arr_fast = np.array([r.fast_r2 for r in run_list])
        arr_trans = np.array([r.transfer_r2 for r in run_list])
        arr_subth = np.array([r.transfer_subth_r2 for r in run_list])
        arr_out = np.array([r.output_r2 for r in run_list])
        summary[cap] = {
            "n_seeds": len(run_list),
            "seeds": [r.seed for r in run_list],
            "fast_r2": {"mean": float(arr_fast.mean()), "std": float(arr_fast.std(ddof=0)), "values": arr_fast.tolist()},
            "transfer_r2": {"mean": float(arr_trans.mean()), "std": float(arr_trans.std(ddof=0)), "values": arr_trans.tolist()},
            "transfer_subth_r2": {"mean": float(arr_subth.mean()), "std": float(arr_subth.std(ddof=0)), "values": arr_subth.tolist()},
            "output_r2": {"mean": float(arr_out.mean()), "std": float(arr_out.std(ddof=0)), "values": arr_out.tolist()},
        }
    aggregate["summary"] = summary
    (output_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_dir / "aggregate.json")

    # Bar chart: Fast Dataset R² per capacity with std bars.
    caps = list(summary)
    means = [summary[c]["fast_r2"]["mean"] for c in caps]
    stds = [summary[c]["fast_r2"]["std"] for c in caps]
    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.bar(range(len(caps)), means, yerr=stds, capsize=6, color="#cc6633")
    ax.set_xticks(range(len(caps)))
    ax.set_xticklabels(caps)
    ax.set_ylabel("Fast Dataset R²")
    ax.set_title("NFET MLP Fast Dataset R²: mean ± std across seeds")
    ax.grid(True, ls=":", alpha=0.3, axis="y")
    fig.savefig(output_dir / "fast_r2_per_capacity.png", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", output_dir / "fast_r2_per_capacity.png")

    for cap, s in summary.items():
        logger.info(
            "%s (n=%d, seeds=%s): Fast %.3f ± %.3f  Transfer %.4f ± %.4f  SubTh %.4f ± %.4f  Output %.4f ± %.4f",
            cap, s["n_seeds"], s["seeds"],
            s["fast_r2"]["mean"], s["fast_r2"]["std"],
            s["transfer_r2"]["mean"], s["transfer_r2"]["std"],
            s["transfer_subth_r2"]["mean"], s["transfer_subth_r2"]["std"],
            s["output_r2"]["mean"], s["output_r2"]["std"],
        )


def _row_dict(r: _Row) -> dict:
    """Manual asdict because slots=True suppresses __dict__."""
    return {
        "capacity": r.capacity,
        "seed": r.seed,
        "fast_r2": r.fast_r2,
        "transfer_r2": r.transfer_r2,
        "transfer_subth_r2": r.transfer_subth_r2,
        "output_r2": r.output_r2,
    }


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
