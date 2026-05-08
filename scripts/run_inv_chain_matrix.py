"""
Run inverter-chain composition sweeps across stage counts.

Cold/warm timings are reported **inside** each ``compose_chain`` process in
``summary.json`` under ``speedup`` (same semantics as ``spino.circuit.compose``:
first paired solve is cold, immediate repeat is warm). Optional ``--repeats``
starts extra processes for session-level repeatability only — it is not the
definition of warm versus cold.

This wrapper avoids Poetry and uses the current Python runtime.
"""

# pylint: disable=too-many-instance-attributes,too-many-arguments

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One ``compose_chain`` subprocess result (cold/warm from that summary)."""

    stage_count: int
    repeat_idx: int
    output_dir: Path
    wall_total_s: float
    spice_cold_ms: float
    spice_warm_ms: float
    fno_cold_ms: float
    fno_warm_ms: float
    speedup_cold_spice_over_fno: float
    speedup_warm_spice_over_fno: float
    final_max_abs_delta_v: float
    final_pearson_r: float


def _run_checked(cmd: list[str]) -> None:
    """Runs command and raises on non-zero status."""
    subprocess.run(cmd, check=True)


def _prepare_synth_caps(python_bin: str, caps_dir: Path) -> None:
    """Generates synthetic partition-cap tables in local staging."""
    cmd = [
        python_bin,
        "-m",
        "spino.circuit.extract_partition_caps",
        "synthetic",
        "--output-dir",
        str(caps_dir),
    ]
    _run_checked(cmd)


def _compose_once(
    python_bin: str,
    *,
    output_dir: Path,
    stage_count: int,
    device: str,
    nfet_cap_npz: Path,
    pfet_cap_npz: Path,
    extra_args: list[str],
) -> None:
    """Executes one compose-chain run."""
    cmd = [
        python_bin,
        "-m",
        "spino.circuit.compose_chain",
        "--output-dir",
        str(output_dir),
        "--stages",
        str(stage_count),
        "--device",
        device,
        "--nfet-cap-npz",
        str(nfet_cap_npz),
        "--pfet-cap-npz",
        str(pfet_cap_npz),
        *extra_args,
    ]
    _run_checked(cmd)


def _load_summary(path: Path) -> dict[str, Any]:
    """Reads compose-chain summary JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def _record_from_summary(stage_count: int, repeat_idx: int, output_dir: Path, wall_total_s: float) -> RunRecord:
    """Builds immutable run record from one summary.json."""
    summary = _load_summary(output_dir / "summary.json")
    sp = summary["speedup"]["spice_ms"]
    fn = summary["speedup"]["fno_ms"]
    sc, sw = float(sp["cold_solver_ms"]), float(sp["warm_solver_ms"])
    fc, fw = float(fn["cold_solver_ms"]), float(fn["warm_solver_ms"])
    final_metrics = summary["metrics"]["final_output"]
    return RunRecord(
        stage_count=stage_count,
        repeat_idx=repeat_idx,
        output_dir=output_dir,
        wall_total_s=wall_total_s,
        spice_cold_ms=sc,
        spice_warm_ms=sw,
        fno_cold_ms=fc,
        fno_warm_ms=fw,
        speedup_cold_spice_over_fno=sc / max(fc, 1e-30),
        speedup_warm_spice_over_fno=sw / max(fw, 1e-30),
        final_max_abs_delta_v=float(final_metrics["max_abs_delta_v"]),
        final_pearson_r=float(final_metrics["pearson_r"]),
    )


def _stats(vals: list[float]) -> dict[str, float]:
    """Min/median/mean/max for a non-empty list."""
    return {
        "min": min(vals),
        "median": statistics.median(vals),
        "mean": statistics.fmean(vals),
        "max": max(vals),
    }


def _aggregate(records: list[RunRecord]) -> dict[str, Any]:
    """Aggregates records by stage count."""
    by_stage: dict[int, list[RunRecord]] = {}
    for r in records:
        by_stage.setdefault(r.stage_count, []).append(r)
    stage_rows: list[dict[str, Any]] = []
    for stage in sorted(by_stage):
        rows = by_stage[stage]
        spice_cold = [r.spice_cold_ms for r in rows]
        spice_warm = [r.spice_warm_ms for r in rows]
        fno_cold = [r.fno_cold_ms for r in rows]
        fno_warm = [r.fno_warm_ms for r in rows]
        speed_cold = [r.speedup_cold_spice_over_fno for r in rows]
        speed_warm = [r.speedup_warm_spice_over_fno for r in rows]
        err_vals = [r.final_max_abs_delta_v for r in rows]
        corr_vals = [r.final_pearson_r for r in rows]
        stage_rows.append(
            {
                "stages": stage,
                "n_process_repeats": len(rows),
                "timing_ms_is_within_process_solver_pairs": True,
                "runtime_ms": {
                    "spice_cold_solver": _stats(spice_cold),
                    "spice_warm_solver": _stats(spice_warm),
                    "fno_cold_solver": _stats(fno_cold),
                    "fno_warm_solver": _stats(fno_warm),
                },
                "speedup_spice_over_fno": {
                    "using_cold_ms": _stats(speed_cold),
                    "using_warm_ms": _stats(speed_warm),
                },
                "final_output_metrics": {
                    "max_abs_delta_v": _stats(err_vals),
                    "pearson_r": _stats(corr_vals),
                },
                "runs": [
                    {
                        "repeat_idx": r.repeat_idx,
                        "output_dir": str(r.output_dir),
                        "wall_total_s": r.wall_total_s,
                        "spice_cold_ms": r.spice_cold_ms,
                        "spice_warm_ms": r.spice_warm_ms,
                        "fno_cold_ms": r.fno_cold_ms,
                        "fno_warm_ms": r.fno_warm_ms,
                        "speedup_cold_spice_over_fno": r.speedup_cold_spice_over_fno,
                        "speedup_warm_spice_over_fno": r.speedup_warm_spice_over_fno,
                        "final_max_abs_delta_v": r.final_max_abs_delta_v,
                        "final_pearson_r": r.final_pearson_r,
                    }
                    for r in sorted(rows, key=lambda x: x.repeat_idx)
                ],
            }
        )
    return {"stages": stage_rows}


def _parse_args() -> argparse.Namespace:
    """Parses CLI options."""
    p = argparse.ArgumentParser(description="Run inverter-chain compose sweeps.")
    p.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter to use.")
    p.add_argument("--device", type=str, choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--stages", type=int, nargs="+", default=[1, 2, 4], help="Stage counts to run.")
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Separate process launches per stage (session variance). Cold/warm is inside each run's summary.",
    )
    p.add_argument(
        "--caps-dir",
        type=Path,
        default=Path("runs/inv_chain/spice_caps"),
        help="Directory holding partition-cap tables.",
    )
    p.add_argument(
        "--prepare-synthetic-caps",
        action="store_true",
        help="Generate synthetic cap tables before running compose sweeps.",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/inv_chain/matrix"),
        help="Root directory for per-run outputs and aggregate summary.",
    )
    p.add_argument(
        "--compose-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra raw args forwarded to compose_chain (repeatable). Example: "
            '--compose-extra-arg=--t-end --compose-extra-arg=2e-6'
        ),
    )
    return p.parse_args()


def main() -> None:
    """Runs the requested stage/repeat matrix and writes aggregate JSON."""
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    for n in args.stages:
        if n < 1:
            raise ValueError(f"Invalid stage count {n}; stages must be >= 1")
    caps_dir = args.caps_dir
    nfet_cap = caps_dir / "partition_caps_nfet_synthetic.npz"
    pfet_cap = caps_dir / "partition_caps_pfet_synthetic.npz"
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.prepare_synthetic_caps:
        _prepare_synth_caps(args.python_bin, caps_dir)
    if not nfet_cap.exists() or not pfet_cap.exists():
        raise FileNotFoundError(
            "Cap tables missing. Use --prepare-synthetic-caps or pass existing caps in "
            f"{caps_dir}."
        )
    records: list[RunRecord] = []
    for stage_count in args.stages:
        for repeat_idx in range(args.repeats):
            out_dir = args.out_root / f"n{stage_count}" / f"rep{repeat_idx:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            _compose_once(
                args.python_bin,
                output_dir=out_dir,
                stage_count=stage_count,
                device=args.device,
                nfet_cap_npz=nfet_cap,
                pfet_cap_npz=pfet_cap,
                extra_args=args.compose_extra_arg,
            )
            wall = time.perf_counter() - t0
            records.append(_record_from_summary(stage_count, repeat_idx, out_dir, wall))
    summary = {
        "config": {
            "python_bin": args.python_bin,
            "device": args.device,
            "stages": args.stages,
            "process_repeats_per_stage": args.repeats,
            "caps_dir": str(caps_dir),
            "out_root": str(args.out_root),
            "compose_extra_arg": args.compose_extra_arg,
            "cold_warm_definition": (
                "Per compose_chain summary: speedup.spice_ms.fno_ms use cold_solver_ms = first paired "
                "analysis wall time, warm_solver_ms = immediate second identical pair — "
                "same as spino.circuit.compose CS amp harness."
            ),
        },
        **_aggregate(records),
    }
    agg_path = args.out_root / "aggregate_summary.json"
    agg_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote aggregate summary: {agg_path}")


if __name__ == "__main__":
    main()
