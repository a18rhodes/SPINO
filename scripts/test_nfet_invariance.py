"""
NFET Invariance Characterization (Phase 0, Pre-PFET).

Empirically validates two hypotheses about the existing Exp 19b MOSFET operator:

1. **Time-scale invariance:** The MOSFET I-V mapping is quasi-static (algebraic,
   not ODE-governed). The FNO should produce equivalent predictions regardless of
   T_end when the normalized waveform shape is held constant.

2. **Resolution invariance:** The FNO's learned spectral filters are insensitive
   to grid size for bandwidth-limited PWL signals. Downsampling to 1024 or
   upsampling to 4096 should not degrade R^2.

Methodology:
  - Test A (time-scale): Fix a deterministic PWL waveform shape, run SPICE at
    T_end in {100ns, 500ns, 1us, 2us, 5us}, feed each to the model.
  - Test B (resolution): Fix T_end=1us, generate SPICE at 2048 steps, resample
    the voltage/current to {512, 1024, 2048, 4096} grids, run the model.
  - Both tests run at 3 geometries: core (1.0/0.18), tiny (0.47/0.17),
    xlarge (8.0/1.75).

Pass criteria:
  - Time-scale: delta R^2 < 0.01 across all T_end values.
  - Resolution: delta R^2 < 0.001 across all step counts.

Results feed directly into the PFET architecture decision: if both pass, lambda
is unnecessary for the MOSFET operator class and the PFET uses in_channels=4.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from spino.constants import ARCSINH_SCALE_MA
from spino.mosfet.gen_data import (
    InfiniteSpiceMosfetDataset,
    ParameterSchema,
    PreGeneratedMosfetDataset,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL_PATH = "/app/spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt"
DATASET_PATH = "/app/datasets/sky130_nmos_61k_plus_shortch_supp8k.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRIM = 41
SEED = 42
N_TRANSITIONS = 10


@dataclass
class Geometry:
    """Device geometry for evaluation."""

    name: str
    w_um: float
    l_um: float


GEOMETRIES = [
    Geometry("core", 1.0, 0.18),
    Geometry("tiny", 0.47, 0.17),
    Geometry("xlarge", 8.0, 1.75),
]

T_END_VALUES = [100e-9, 500e-9, 1e-6, 2e-6, 5e-6]
RESOLUTION_VALUES = [512, 1024, 2048, 4096]
REFERENCE_T_END = 1e-6
REFERENCE_STEPS = 2048


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 0.0
    return 1.0 - ss_res / ss_tot


def make_deterministic_pwl_fractions(n_transitions: int = N_TRANSITIONS, seed: int = SEED) -> dict[str, np.ndarray]:
    """
    Generates a deterministic PWL waveform as fractional breakpoints.

    Breakpoints are defined as fractions of T_end and voltages in [0, 1.8V] so the
    same *shape* can be instantiated at any T_end.

    :param n_transitions: Number of internal transition points.
    :param seed: RNG seed for reproducibility.
    :return: Mapping with keys ``fracs``, ``vg_vals``, and ``vd_vals`` as numpy arrays.
    """
    rng = np.random.default_rng(seed=seed)
    fracs = np.sort(rng.uniform(0.05, 0.95, n_transitions))
    fracs = np.concatenate([[0.0], fracs, [1.0]])
    vg_vals = np.concatenate([[0.0], rng.uniform(0.0, 1.8, n_transitions), [0.9]])
    vd_vals = np.concatenate([[0.9], rng.uniform(0.0, 1.8, n_transitions), [0.9]])
    return {"fracs": fracs, "vg_vals": vg_vals, "vd_vals": vd_vals}


def instantiate_pwl(pwl: dict[str, np.ndarray], t_end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales fractional PWL breakpoints to a specific T_end.

    :param pwl: Fractional PWL breakpoints as returned by ``make_deterministic_pwl_fractions``.
    :param t_end: Target simulation window in seconds.
    :return: (times, vg, vd) arrays scaled to the given T_end.
    """
    times = pwl["fracs"] * t_end
    return times, pwl["vg_vals"], pwl["vd_vals"]


def _simulate_transient(
    local_ds: InfiniteSpiceMosfetDataset,
    geom: Geometry,
    pwl: dict[str, np.ndarray],
    t_end: float,
    raw_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    times, vg_vals, vd_vals = instantiate_pwl(pwl, t_end)
    time_grid = np.linspace(0, t_end, raw_steps)
    vg = np.interp(time_grid, times, vg_vals)
    vd = np.interp(time_grid, times, vd_vals)
    vs = np.zeros(raw_steps)
    vb = np.zeros(raw_steps)
    pwl_g = local_ds._build_pwl_string(time_grid, vg)
    pwl_d = local_ds._build_pwl_string(time_grid, vd)
    pwl_s = local_ds._build_pwl_string(time_grid, vs)
    pwl_b = local_ds._build_pwl_string(time_grid, vb)
    netlist = local_ds._build_netlist(geom.w_um, geom.l_um, pwl_g, pwl_d, pwl_s, pwl_b)
    results = local_ds._run_transient_simulation(netlist)
    if results is None:
        return None
    current_raw = local_ds._extract_drain_current(results)
    if current_raw is None:
        return None
    id_a = local_ds._interpolate_current(current_raw, results.get("time"))
    if id_a is None:
        return None
    return id_a[TRIM:] * 1000.0, vg[TRIM:], vd[TRIM:], vs[TRIM:], vb[TRIM:]


def _infer_current(
    model: torch.nn.Module,
    dataset: PreGeneratedMosfetDataset,
    local_ds: InfiniteSpiceMosfetDataset,
    geom: Geometry,
    vg: np.ndarray,
    vd: np.ndarray,
    vs: np.ndarray,
    vb: np.ndarray,
) -> np.ndarray:
    raw_params = local_ds.parser.inspect_model(local_ds.strategy.model_name, w=str(geom.w_um), l=str(geom.l_um))
    p_full = ParameterSchema.to_tensor(raw_params).squeeze()
    p_curated = p_full[ParameterSchema.TRAINING_INDICES] if dataset.use_curated_params else p_full
    if dataset.normalize:
        p_curated = (p_curated - dataset.physics_mean) / dataset.physics_std
    p_tensor = p_curated.unsqueeze(0).to(DEVICE)
    v_stack = np.stack([vg, vd, vs, vb])
    v_tensor = torch.tensor(v_stack, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    if dataset.normalize:
        v_tensor = (v_tensor - dataset.voltages_mean.to(DEVICE)) / dataset.voltages_std.to(DEVICE)
    with torch.no_grad():
        pred_arcsinh = model(v_tensor, p_tensor).cpu().numpy().flatten()
    return np.sinh(pred_arcsinh) * ARCSINH_SCALE_MA


def run_spice_and_predict(
    model: torch.nn.Module,
    dataset: PreGeneratedMosfetDataset,
    pwl: dict[str, np.ndarray],
    geom: Geometry,
    t_end: float,
    t_steps: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Runs a SPICE transient simulation and FNO inference for a single configuration.

    :param model: Trained VCFiLM model.
    :param dataset: Pre-generated dataset for normalization statistics.
    :param pwl: Fractional PWL waveform dict from ``make_deterministic_pwl_fractions``.
    :param geom: Target device geometry.
    :param t_end: Simulation window in seconds.
    :param t_steps: Number of time steps (excluding startup trim).
    :return: (spice_current_ma, predicted_current_ma) or (None, None) on simulation failure.
    """
    raw_steps = t_steps + TRIM
    local_ds = InfiniteSpiceMosfetDataset(strategy_name="sky130_nmos", t_steps=raw_steps, t_end=t_end)
    transient = _simulate_transient(local_ds, geom, pwl, t_end, raw_steps)
    if transient is None:
        return None, None
    id_spice_ma, vg, vd, vs, vb = transient
    return id_spice_ma, _infer_current(model, dataset, local_ds, geom, vg, vd, vs, vb)


def test_time_scale_invariance(
    model: torch.nn.Module,
    dataset: PreGeneratedMosfetDataset,
    console: Console,
) -> dict:
    """
    Test A: Time-scale invariance across variable T_end.

    :param model: Trained VCFiLM model.
    :param dataset: Pre-generated dataset for normalization.
    :param console: Rich console for output.
    :return: Nested dict of results: {geometry: {t_end: r2}}.
    """
    console.print("\n[bold cyan]Test A: Time-Scale Invariance[/bold cyan]")
    console.print("  Fixed waveform shape, variable T_end. Model trained at T_end=1us.")
    console.print(f"  T_end values: {[f'{t*1e6:.1f}us' for t in T_END_VALUES]}")
    console.print(f"  Geometries: {[g.name for g in GEOMETRIES]}\n")
    pwl = make_deterministic_pwl_fractions()
    results = {}
    table = Table(title="Time-Scale Invariance Results", show_header=True)
    table.add_column("Geometry", style="cyan")
    for t_end in T_END_VALUES:
        table.add_column(f"{t_end*1e6:.1f}us", justify="right")
    table.add_column("Delta R2", justify="right", style="bold")
    for geom in GEOMETRIES:
        geom_results = {}
        row = [geom.name]
        for t_end in T_END_VALUES:
            id_spice, id_pred = run_spice_and_predict(
                model, dataset, pwl, geom, t_end, REFERENCE_STEPS
            )
            if id_spice is None:
                geom_results[t_end] = None
                row.append("[red]FAIL[/red]")
                continue
            r2 = r_squared(id_spice, id_pred)
            geom_results[t_end] = r2
            color = "green" if r2 > 0.99 else "yellow" if r2 > 0.95 else "red"
            row.append(f"[{color}]{r2:.6f}[/{color}]")
        valid_r2s = [v for v in geom_results.values() if v is not None]
        delta = max(valid_r2s) - min(valid_r2s) if len(valid_r2s) > 1 else 0.0
        pass_fail = "[green]PASS[/green]" if delta < 0.01 else "[red]FAIL[/red]"
        row.append(f"{delta:.6f} {pass_fail}")
        table.add_row(*row)
        results[geom.name] = geom_results
    console.print(table)
    return results


def test_resolution_invariance(
    model: torch.nn.Module,
    dataset: PreGeneratedMosfetDataset,
    console: Console,
) -> dict:
    """
    Test B: Resolution invariance across variable step counts.

    Generates SPICE ground truth at 4096 steps (highest resolution), then
    resamples both voltage inputs and SPICE current to each target grid size.
    The FNO runs on the resampled voltages and its output is compared against
    the resampled SPICE current.

    :param model: Trained VCFiLM model.
    :param dataset: Pre-generated dataset for normalization.
    :param console: Rich console for output.
    :return: Nested dict of results: {geometry: {steps: r2}}.
    """
    console.print("\n[bold cyan]Test B: Resolution Invariance[/bold cyan]")
    console.print("  Fixed T_end=1us, variable step count. Model trained at 2048 steps.")
    console.print(f"  Step counts: {RESOLUTION_VALUES}")
    console.print(f"  Geometries: {[g.name for g in GEOMETRIES]}\n")
    pwl = make_deterministic_pwl_fractions()
    results = {}
    hi_res_steps = max(RESOLUTION_VALUES)
    table = Table(title="Resolution Invariance Results", show_header=True)
    table.add_column("Geometry", style="cyan")
    for steps in RESOLUTION_VALUES:
        table.add_column(f"{steps} pts", justify="right")
    table.add_column("Delta R2", justify="right", style="bold")
    for geom in GEOMETRIES:
        console.print(f"  Generating hi-res SPICE for {geom.name} at {hi_res_steps} steps...")
        raw_steps_hi = hi_res_steps + TRIM
        local_ds = InfiniteSpiceMosfetDataset(
            strategy_name="sky130_nmos", t_steps=raw_steps_hi, t_end=REFERENCE_T_END
        )
        times_pwl, vg_vals, vd_vals = instantiate_pwl(pwl, REFERENCE_T_END)
        time_grid_hi = np.linspace(0, REFERENCE_T_END, raw_steps_hi)
        vg_hi = np.interp(time_grid_hi, times_pwl, vg_vals)
        vd_hi = np.interp(time_grid_hi, times_pwl, vd_vals)
        vs_hi = np.zeros(raw_steps_hi)
        vb_hi = np.zeros(raw_steps_hi)
        pwl_g = local_ds._build_pwl_string(time_grid_hi, vg_hi)
        pwl_d = local_ds._build_pwl_string(time_grid_hi, vd_hi)
        pwl_s = local_ds._build_pwl_string(time_grid_hi, vs_hi)
        pwl_b = local_ds._build_pwl_string(time_grid_hi, vb_hi)
        netlist = local_ds._build_netlist(geom.w_um, geom.l_um, pwl_g, pwl_d, pwl_s, pwl_b)
        results_spice = local_ds._run_transient_simulation(netlist)
        if results_spice is None:
            console.print(f"  [red]SPICE FAILED for {geom.name}[/red]")
            results[geom.name] = {s: None for s in RESOLUTION_VALUES}
            table.add_row(geom.name, *["[red]FAIL[/red]"] * (len(RESOLUTION_VALUES) + 1))
            continue
        current_raw = local_ds._extract_drain_current(results_spice)
        if current_raw is None:
            console.print(f"  [red]Current extraction failed for {geom.name}[/red]")
            results[geom.name] = {s: None for s in RESOLUTION_VALUES}
            table.add_row(geom.name, *["[red]FAIL[/red]"] * (len(RESOLUTION_VALUES) + 1))
            continue
        id_hi_a = local_ds._interpolate_current(current_raw, results_spice.get("time"))
        if id_hi_a is None:
            console.print(f"  [red]Interpolation failed for {geom.name}[/red]")
            results[geom.name] = {s: None for s in RESOLUTION_VALUES}
            table.add_row(geom.name, *["[red]FAIL[/red]"] * (len(RESOLUTION_VALUES) + 1))
            continue
        vg_hi_trimmed = vg_hi[TRIM:]
        vd_hi_trimmed = vd_hi[TRIM:]
        vs_hi_trimmed = vs_hi[TRIM:]
        vb_hi_trimmed = vb_hi[TRIM:]
        id_hi_ma_trimmed = id_hi_a[TRIM:] * 1000.0
        geom_results = {}
        row = [geom.name]
        for target_steps in RESOLUTION_VALUES:
            hi_indices = np.linspace(0, len(vg_hi_trimmed) - 1, target_steps)
            vg_rs = np.interp(hi_indices, np.arange(len(vg_hi_trimmed)), vg_hi_trimmed)
            vd_rs = np.interp(hi_indices, np.arange(len(vd_hi_trimmed)), vd_hi_trimmed)
            vs_rs = np.interp(hi_indices, np.arange(len(vs_hi_trimmed)), vs_hi_trimmed)
            vb_rs = np.interp(hi_indices, np.arange(len(vb_hi_trimmed)), vb_hi_trimmed)
            id_rs = np.interp(hi_indices, np.arange(len(id_hi_ma_trimmed)), id_hi_ma_trimmed)
            pred_ma = _infer_current(model, dataset, local_ds, geom, vg_rs, vd_rs, vs_rs, vb_rs)
            r2 = r_squared(id_rs, pred_ma)
            geom_results[target_steps] = r2
            color = "green" if r2 > 0.99 else "yellow" if r2 > 0.95 else "red"
            row.append(f"[{color}]{r2:.6f}[/{color}]")
        valid_r2s = [v for v in geom_results.values() if v is not None]
        delta = max(valid_r2s) - min(valid_r2s) if len(valid_r2s) > 1 else 0.0
        pass_fail = "[green]PASS[/green]" if delta < 0.001 else "[red]FAIL[/red]"
        row.append(f"{delta:.6f} {pass_fail}")
        table.add_row(*row)
        results[geom.name] = geom_results
    console.print(table)
    return results


def load_model() -> torch.nn.Module:
    """Loads the Exp 19b VCFiLM checkpoint."""
    from spino.mosfet.model import MosfetVCFiLMFNO
    model = MosfetVCFiLMFNO(input_param_dim=29, embedding_dim=16, modes=256, width=64)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def main():
    """Runs both invariance tests and prints summary."""
    console = Console()
    console.print("[bold]NFET Invariance Characterization (Phase 0)[/bold]")
    console.print(f"Model: {MODEL_PATH}")
    console.print(f"Dataset: {DATASET_PATH}")
    console.print(f"Device: {DEVICE}\n")
    console.print("Loading model...")
    model = load_model()
    console.print("Loading dataset for normalization stats...")
    dataset = PreGeneratedMosfetDataset(
        hdf5_path=DATASET_PATH,
        normalize=True,
        use_curated_params=True,
        trim_startup=TRIM,
    )
    console.print(f"  Samples: {len(dataset)}, V_mean shape: {dataset.voltages_mean.shape}\n")
    ts_results = test_time_scale_invariance(model, dataset, console)
    rs_results = test_resolution_invariance(model, dataset, console)
    console.print("\n[bold]Summary[/bold]")
    ts_pass = True
    for geom_name, geom_data in ts_results.items():
        valid = [v for v in geom_data.values() if v is not None]
        if len(valid) < 2:
            continue
        delta = max(valid) - min(valid)
        if delta >= 0.01:
            ts_pass = False
            console.print(f"  [red]Time-scale FAIL: {geom_name} delta={delta:.6f}[/red]")
    if ts_pass:
        console.print("  [green]Time-scale invariance: PASS (all delta R2 < 0.01)[/green]")
    rs_pass = True
    for geom_name, geom_data in rs_results.items():
        valid = [v for v in geom_data.values() if v is not None]
        if len(valid) < 2:
            continue
        delta = max(valid) - min(valid)
        if delta >= 0.001:
            rs_pass = False
            console.print(f"  [red]Resolution FAIL: {geom_name} delta={delta:.6f}[/red]")
    if rs_pass:
        console.print("  [green]Resolution invariance: PASS (all delta R2 < 0.001)[/green]")
    console.print("\n[bold]Decision Gate[/bold]")
    if ts_pass and rs_pass:
        console.print("  [green]Both PASS -> PFET uses in_channels=4, no lambda.[/green]")
        console.print("  The MOSFET I-V is quasi-static. Lambda carries no information.")
    elif not ts_pass and rs_pass:
        console.print("  [yellow]Time-scale FAIL -> Lambda may be needed for PFET.[/yellow]")
        console.print("  Displacement currents or transient effects are non-negligible.")
    elif ts_pass and not rs_pass:
        console.print("  [yellow]Resolution FAIL -> Canonical-grid resampling at inference.[/yellow]")
        console.print("  Spectral filters are grid-coupled. Multi-resolution training or resampling needed.")
    else:
        console.print("  [red]Both FAIL -> Full dimensionless formulation needed.[/red]")
    dataset.close()


if __name__ == "__main__":
    main()
