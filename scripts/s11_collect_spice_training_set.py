"""S11: collect SPICE training set for the Uhlmann-style performance surrogate.

Samples ``n`` design vectors via Latin Hypercube over the OtaSizingProblem
bounding box (n_theta = 7) and evaluates each one with
``simulate_ota_design_point``. The output table is the input to the
performance-surrogate MLP that approximates θ → (slew, power, swing) and
provides an autograd-differentiable gradient through the learned
performance abstraction, matching the Uhlmann et al. prior-art design
point.

Output: ``runs/s11_uhlmann/training_set/samples.json`` with one record per
sample containing the 7-vector θ, the SPICE-converged (slew_v_per_us,
power_uw, peak_swing_v, dc_gain_v_per_v, slew_time_ns) and a ``converged``
flag. Non-converged samples are kept (with NaN metrics) so the surrogate
can learn the boundary.

Usage::

    python -m scripts.s11_collect_spice_training_set \\
        --n-samples 1000 --seed 42 \\
        --output-dir runs/s11_uhlmann/training_set
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import click
import numpy as np
from scipy.stats import qmc

from spino.circuit.sizing import OtaSizingProblem
from spino.circuit.tuning import OtaDesignPoint, simulate_ota_design_point

logger = logging.getLogger(__name__)


def _lhs_samples(n: int, problem: OtaSizingProblem, seed: int) -> np.ndarray:
    """Draw ``n`` Latin-Hypercube samples from the 7-dimensional bounding box."""
    lo = problem.lower_bounds.numpy()
    hi = problem.upper_bounds.numpy()
    sampler = qmc.LatinHypercube(d=lo.size, seed=seed)
    unit = sampler.random(n=n)
    return qmc.scale(unit, lo, hi)


def _evaluate(theta_vec: np.ndarray, problem: OtaSizingProblem) -> dict:
    """Run one SPICE evaluation. Returns a record (NaN metrics if non-converged)."""
    w_diff, w_mirror, w_tail, l_diff, l_mirror, l_tail, vbias = (float(v) for v in theta_vec)
    record = {
        "theta": {
            "w_diff_um": w_diff,
            "w_mirror_um": w_mirror,
            "w_tail_um": w_tail,
            "l_diff_um": l_diff,
            "l_mirror_um": l_mirror,
            "l_tail_um": l_tail,
            "vbias_v": vbias,
        },
    }
    try:
        metrics = simulate_ota_design_point(
            OtaDesignPoint(diff_w_um=w_diff, mirror_w_um=w_mirror),
            vdd=problem.vdd,
            vcm_v=problem.vcm,
            step_amp_v=problem.step_amp,
            diff_l_um=l_diff,
            mirror_l_um=l_mirror,
            tail_w_um=w_tail,
            tail_l_um=l_tail,
            vbias_v=vbias,
            t_step_start=problem.t_step_start,
            t_end=problem.t_end,
            t_step=problem.t_step,
            c_load_f=problem.c_load_f,
            pdk_root=problem.pdk_root,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SPICE failure at θ=%s: %s", theta_vec.tolist(), exc)
        record.update({"converged": False, "error": repr(exc)})
        return record
    if not metrics.converged:
        record.update({"converged": False})
        return record
    md = asdict(metrics)
    record.update(
        {
            "converged": True,
            "slew_v_per_us": float(md["slew_rate_v_per_us"]),
            "power_uw": float(md["static_current_a"]) * problem.vdd * 1e6,
            "peak_swing_v": float(md["peak_swing_v"]),
            "dc_gain_v_per_v": float(md["dc_gain_v_per_v"]),
            "slew_time_ns": float(md["slew_time_ns"]),
            "static_current_a": float(md["static_current_a"]),
            "quiescent_n_out_v": float(md["quiescent_n_out_v"]),
        }
    )
    return record


@click.command()
@click.option("--n-samples", type=int, default=1000, show_default=True, help="Total LHS sample count.")
@click.option("--seed", type=int, default=42, show_default=True, help="LHS RNG seed.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s11_uhlmann/training_set"),
    show_default=True,
)
@click.option("--log-every", type=int, default=10, show_default=True, help="Status log cadence.")
def main(n_samples: int, seed: int, output_dir: Path, log_every: int) -> None:
    """Generate the S11 surrogate training set via LHS over the OTA bounds."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = OtaSizingProblem()
    theta_grid = _lhs_samples(n_samples, problem, seed)
    samples: list[dict] = []
    t0 = time.perf_counter()
    for idx, theta_vec in enumerate(theta_grid):
        rec = _evaluate(theta_vec, problem)
        rec["index"] = idx
        samples.append(rec)
        if (idx + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            n_conv = sum(1 for s in samples if s.get("converged"))
            logger.info(
                "%d / %d done (%.1f s/sample; %d converged so far; ~%.0f min remaining)",
                idx + 1,
                n_samples,
                1.0 / rate,
                n_conv,
                (n_samples - idx - 1) / max(rate, 1e-6) / 60,
            )
            (output_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (output_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    n_conv = sum(1 for s in samples if s.get("converged"))
    logger.info(
        "Collected %d samples (%d converged, %.1f %%) in %.1f min",
        n_samples,
        n_conv,
        100.0 * n_conv / n_samples,
        (time.perf_counter() - t0) / 60,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
