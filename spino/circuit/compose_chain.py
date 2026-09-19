"""
CLI for FNO-composed inverter-chain validation (SPICE vs composed NR).

Invocation::

    python -m spino.circuit.compose_chain \\
        --output-dir runs/inv_chain/fno
"""

from __future__ import annotations

# pylint: disable=wrong-import-position,too-many-arguments,too-many-locals,too-many-positional-arguments

import json
import math
import logging
import time as time_module
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spino.circuit.chain_composition import ChainDcSolver, ChainTransientSolver  # noqa: E402
from spino.circuit.chain_metrics import crossing_time_s, max_abs_delta_v, pearson_r  # noqa: E402
from spino.circuit.composition_io import (  # noqa: E402
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_PFET_CHECKPOINT,
    load_inverter_chain_devices,
)
from spino.circuit.partition_caps import load_torch_partition_caps  # noqa: E402
from spino.circuit.simulation import TransientResult, run_operating_point, run_transient  # noqa: E402
from spino.circuit.topologies import build_inverter_chain  # noqa: E402

_SPEEDUP_NOTES = (
    "cold_solver_ms is the first timed DC+transient pair after model load; "
    "warm_solver_ms is an immediate repeat. Compare to SPICE columns the same way."
)

logger = logging.getLogger(__name__)

_DEFAULT_VDD = 1.8
_DEFAULT_NFET_W = 0.82
_DEFAULT_NFET_L = 0.18
_DEFAULT_PFET_W = 2.05
_DEFAULT_PFET_L = 0.18
_DEFAULT_T_STEP = 5e-9
_DEFAULT_T_END = 1.02e-6
_DEFAULT_STEP_START = 100e-9
_DEFAULT_OUTPUT_DIR = Path("runs/inv_chain/fno")
_DEFAULT_CAPS_NFET = Path("runs/inv_chain/spice_caps/partition_caps_nfet_synthetic.npz")
_DEFAULT_CAPS_PFET = Path("runs/inv_chain/spice_caps/partition_caps_pfet_synthetic.npz")
_VCROSS_RATIO = 0.5

__all__ = ["main"]


def _spice_node_output(stage_idx: int) -> str:
    return f"v(n{stage_idx})"


def _pwl_rising_edge(vdd: float, t_rise_mid: float, t_end: float, t_step_floor: float) -> str:
    half = max(t_rise_mid * 0.5, t_step_floor)
    return f"PWL(0 0 {half} 0 {t_rise_mid} {vdd} {t_end} {vdd})"


def _delay_cross_delta_s(delay_fno_s: float, delay_spice_s: float) -> float:
    """FNO minus SPICE crossing-time delay; nan if either input is nan (no crossing etc.)."""
    if math.isnan(delay_fno_s) or math.isnan(delay_spice_s):
        return float("nan")
    return delay_fno_s - delay_spice_s


def _vin_waveform_numpy(time_s: np.ndarray, vdd: float, t_rise_mid: float, t_step_floor: float) -> np.ndarray:
    vin = np.zeros_like(time_s, dtype=np.float64)
    half = max(t_rise_mid * 0.5, t_step_floor)
    vin[time_s >= t_rise_mid] = vdd
    ramp_mask = (time_s > half) & (time_s < t_rise_mid)
    vin[ramp_mask] = vdd * (time_s[ramp_mask] - half) / (t_rise_mid - half)
    return vin


def _summarize_run(
    *,
    time_s: np.ndarray,
    spice_tran: TransientResult,
    fno_v: np.ndarray,
    vdd: float,
    n_stages: int,
) -> dict:
    v_cross = _VCROSS_RATIO * vdd
    per_node: dict[str, dict[str, float]] = {}
    st = spice_tran.time.astype(np.float64)
    spice_final = np.interp(time_s, st, spice_tran.variables[_spice_node_output(n_stages)].astype(np.float64))
    fno_final = fno_v[n_stages - 1]
    for k in range(n_stages):
        key = _spice_node_output(k + 1)
        sv = np.interp(time_s, st, spice_tran.variables[key].astype(np.float64))
        fv = fno_v[k]
        dsc = float(crossing_time_s(time_s, sv, v_cross) or float("nan"))
        dfn = float(crossing_time_s(time_s, fv, v_cross) or float("nan"))
        per_node[key] = {
            "max_abs_delta_v": max_abs_delta_v(sv, fv),
            "pearson_r": pearson_r(sv, fv),
            "delay_cross_spice_s": dsc,
            "delay_cross_fno_s": dfn,
            "delay_cross_delta_s": _delay_cross_delta_s(dfn, dsc),
        }
    dsc_f = float(crossing_time_s(time_s, spice_final, v_cross) or float("nan"))
    dfn_f = float(crossing_time_s(time_s, fno_final, v_cross) or float("nan"))
    return {
        "v_cross_v": v_cross,
        "per_node": per_node,
        "final_output": {
            "max_abs_delta_v": max_abs_delta_v(spice_final, fno_final),
            "pearson_r": pearson_r(spice_final, fno_final),
            "delay_cross_spice_s": dsc_f,
            "delay_cross_fno_s": dfn_f,
            "delay_cross_delta_s": _delay_cross_delta_s(dfn_f, dsc_f),
        },
    }


def _plot_final_overlay(
    time_s: np.ndarray,
    spice_tran: TransientResult,
    fno_final: np.ndarray,
    stages: int,
    path: Path,
) -> None:
    key = _spice_node_output(stages)
    st = spice_tran.time.astype(np.float64)
    spice_y = np.interp(time_s, st, spice_tran.variables[key].astype(np.float64))
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(time_s * 1e6, spice_y, label="SPICE", color="#0066cc")
    ax.plot(time_s * 1e6, fno_final, label="FNO", color="#cc6600", linestyle="--")
    ax.set_xlabel("t (us)")
    ax.set_ylabel("V")
    ax.set_title(f"Final output n{stages}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--output-dir", type=click.Path(path_type=Path), default=str(_DEFAULT_OUTPUT_DIR), show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu", show_default=True)
@click.option("--stages", type=int, default=1, show_default=True)
@click.option("--vdd", type=float, default=_DEFAULT_VDD, show_default=True)
@click.option("--nfet-w", type=float, default=_DEFAULT_NFET_W, show_default=True)
@click.option("--nfet-l", type=float, default=_DEFAULT_NFET_L, show_default=True)
@click.option("--pfet-w", type=float, default=_DEFAULT_PFET_W, show_default=True)
@click.option("--pfet-l", type=float, default=_DEFAULT_PFET_L, show_default=True)
@click.option("--t-step", type=float, default=_DEFAULT_T_STEP, show_default=True)
@click.option("--t-end", type=float, default=_DEFAULT_T_END, show_default=True)
@click.option("--t-step-rise", type=float, default=_DEFAULT_STEP_START, show_default=True)
@click.option("--nfet-cap-npz", type=click.Path(path_type=Path), default=str(_DEFAULT_CAPS_NFET), show_default=True)
@click.option("--pfet-cap-npz", type=click.Path(path_type=Path), default=str(_DEFAULT_CAPS_PFET), show_default=True)
@click.option("--nfet-checkpoint", type=click.Path(path_type=Path), default=str(DEFAULT_NFET_CHECKPOINT))
@click.option("--pfet-checkpoint", type=click.Path(path_type=Path), default=str(DEFAULT_PFET_CHECKPOINT))
@click.option("--c-load-f", type=float, default=0.0, show_default=True)
@click.option(
    "--device-class",
    type=click.Choice(["fno", "mlp"]),
    default="fno",
    show_default=True,
    help="Composition backbone. 'mlp' loads MosfetMLP-backed wrappers (per-timestep, off-diagonal-free).",
)
@click.option(
    "--mlp-hidden-dim",
    type=int,
    default=64,
    show_default=True,
    help="MosfetMLP hidden_dim when --device-class mlp. Production checkpoints: 64 (h64), 128 (h128).",
)
def main(  # pylint: disable=too-many-statements
    *,
    output_dir: Path,
    device: str,
    stages: int,
    vdd: float,
    nfet_w: float,
    nfet_l: float,
    pfet_w: float,
    pfet_l: float,
    t_step: float,
    t_end: float,
    t_step_rise: float,
    nfet_cap_npz: Path,
    pfet_cap_npz: Path,
    nfet_checkpoint: Path,
    pfet_checkpoint: Path,
    c_load_f: float,
    device_class: str,
    mlp_hidden_dim: int,
) -> None:
    """
    Run SPICE transient + composed inverter-chain NR; write plots and summary.json.
    """
    if stages < 1:
        raise click.BadParameter("--stages must be >= 1")
    logging.basicConfig(level=logging.INFO)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    map_location = torch.device("cuda" if use_cuda else "cpu")
    pwl = _pwl_rising_edge(vdd, t_step_rise, t_end, t_step)
    ck_spice = build_inverter_chain(
        n_stages=stages,
        nfet_w=nfet_w,
        nfet_l=nfet_l,
        pfet_w=pfet_w,
        pfet_l=pfet_l,
        vdd=vdd,
        vin_dc=0.0,
        vin_tran=pwl,
        c_load_f=c_load_f,
    )
    tsp0 = time_module.perf_counter()
    spice_op_a = run_operating_point(ck_spice, capture_iters=True)
    spice_tran_a = run_transient(ck_spice, t_step=t_step, t_end=t_end, capture_iters=True)
    tsp1 = time_module.perf_counter()
    spice_op_b = run_operating_point(ck_spice, capture_iters=True)
    _spice_tran_b = run_transient(ck_spice, t_step=t_step, t_end=t_end, capture_iters=True)
    tsp2 = time_module.perf_counter()
    if spice_op_a is None or spice_op_b is None:
        raise RuntimeError("NGSpice operating point failed for inverter chain baseline")
    if spice_tran_a is None or _spice_tran_b is None:
        raise RuntimeError("SPICE transient failed for inverter chain reference")
    spice_tran = spice_tran_a
    spice_speed_ms = {
        "cold_solver_ms": 1000.0 * (tsp1 - tsp0),
        "warm_solver_ms": 1000.0 * (tsp2 - tsp1),
        "cold_op_iters": spice_op_a.iter_count,
        "cold_tran_iters": spice_tran_a.iter_count,
        "warm_op_iters": spice_op_b.iter_count,
        "warm_tran_iters": _spice_tran_b.iter_count,
    }
    if device_class == "mlp":
        from spino.circuit.composition_mlp_adapter import (  # pylint: disable=import-outside-toplevel
            MlpArchitecture,
            load_inverter_chain_mlp_devices,
        )

        nfets, pfets = load_inverter_chain_mlp_devices(
            n_stages=stages,
            nfet_w_um=nfet_w,
            nfet_l_um=nfet_l,
            pfet_w_um=pfet_w,
            pfet_l_um=pfet_l,
            nfet_checkpoint=nfet_checkpoint,
            pfet_checkpoint=pfet_checkpoint,
            architecture=MlpArchitecture(hidden_dim=mlp_hidden_dim),
            map_location=map_location,
        )
        logger.info("Loaded MLP-backed chain devices (hidden_dim=%d)", mlp_hidden_dim)
    else:
        nfets, pfets = load_inverter_chain_devices(
            n_stages=stages,
            nfet_w_um=nfet_w,
            nfet_l_um=nfet_l,
            pfet_w_um=pfet_w,
            pfet_l_um=pfet_l,
            nfet_checkpoint=nfet_checkpoint,
            pfet_checkpoint=pfet_checkpoint,
            map_location=map_location,
        )
    nfet_lut = load_torch_partition_caps(nfet_cap_npz, is_pfet=False, map_location=map_location)
    pfet_lut = load_torch_partition_caps(pfet_cap_npz, is_pfet=True, map_location=map_location)
    dc_solver = ChainDcSolver(nfets, pfets, vdd=vdd)
    tran_solver = ChainTransientSolver(nfets, pfets, nfet_lut, pfet_lut, vdd=vdd, c_load_f=c_load_f)
    time_grid = np.arange(0.0, t_end, t_step)
    if time_grid.size < 2:
        raise click.BadParameter("--t-end and --t-step must yield >= 2 samples")
    vin_np = _vin_waveform_numpy(time_grid, vdd, t_step_rise, t_step)
    vin_t = torch.from_numpy(vin_np).to(map_location, dtype=torch.float32)
    time_tensor = torch.from_numpy(time_grid.astype(np.float64)).to(map_location, dtype=torch.float32)
    tn0 = time_module.perf_counter()
    dc_a = dc_solver.solve(vin=0.0)
    v_dc_a = dc_a.v_out_v.detach().cpu().numpy()
    v_dc_vec_a = torch.from_numpy(v_dc_a).to(map_location, dtype=torch.float32)
    tr_a = tran_solver.solve(time_tensor, vin_t, v_dc_vec_a)
    tn1 = time_module.perf_counter()
    dc_b = dc_solver.solve(vin=0.0)
    v_dc_b = dc_b.v_out_v.detach().cpu().numpy()
    v_dc_vec_b = torch.from_numpy(v_dc_b).to(map_location, dtype=torch.float32)
    tr_b = tran_solver.solve(time_tensor, vin_t, v_dc_vec_b)
    tn2 = time_module.perf_counter()
    tr_sol = tr_a
    fno_speed_ms = {
        "cold_solver_ms": 1000.0 * (tn1 - tn0),
        "warm_solver_ms": 1000.0 * (tn2 - tn1),
        "cold_dc_iters": dc_a.reports[0].iter_count if dc_a.reports else 0,
        "cold_tran_iters": tr_a.report.iter_count,
        "warm_dc_iters": dc_b.reports[0].iter_count if dc_b.reports else 0,
        "warm_tran_iters": tr_b.report.iter_count,
    }
    fno_v = tr_sol.v_nodes_v.detach().cpu().numpy()
    summary = {
        "topology": "inverter_chain",
        "stages": stages,
        "vdd_v": vdd,
        "metrics": _summarize_run(
            time_s=time_grid,
            spice_tran=spice_tran,
            fno_v=fno_v,
            vdd=vdd,
            n_stages=stages,
        ),
        "speedup": {
            "spice_ms": spice_speed_ms,
            "fno_ms": fno_speed_ms,
            "notes": _SPEEDUP_NOTES,
        },
        "solver": {
            "cold_pass": {
                "dc_converged": dc_a.reports[0].converged if dc_a.reports else False,
                "dc_iters": dc_a.reports[0].iter_count if dc_a.reports else 0,
                "transient_converged": tr_a.report.converged,
                "transient_iters": tr_a.report.iter_count,
            },
            "warm_pass": {
                "dc_converged": dc_b.reports[0].converged if dc_b.reports else False,
                "dc_iters": dc_b.reports[0].iter_count if dc_b.reports else 0,
                "transient_converged": tr_b.report.converged,
                "transient_iters": tr_b.report.iter_count,
            },
        },
        "stimulus": {"pwl": pwl, "t_step_s": t_step, "t_end_s": t_end, "t_rise_mid_s": t_step_rise},
        "partition_caps": {"nfet_npz": str(nfet_cap_npz), "pfet_npz": str(pfet_cap_npz)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_final_overlay(time_grid, spice_tran, fno_v[-1], stages, output_dir / "final_output_overlay.png")
    logger.info("compose_chain finished: %s", output_dir)


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=missing-kwoa,no-value-for-parameter
