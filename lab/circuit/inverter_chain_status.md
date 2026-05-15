# Inverter chain: lab status

Markdown lab log for the **partitioned capacitor + multi-node inverter chain**
composition path (`chain_composition`, `compose_chain`, matrix harness). Jupyter
artifacts are intentionally not tracked here.

## Scope

Implemented: CMOS inverter primitives, two-node partitioned-cap table loading
(default synthetic under `runs/inv_chain/spice_caps/`), netlist emission for chains,
`DcOperatingPointSolver` / `TransientSolver` over partitioned nodes (`chain_metrics`
for delay / crossing stats), CLI `compose_chain`, matrix driver
`scripts/run_inv_chain_matrix.py`, alternating-rail DC initial guess via
`default_chain_dc_voltage_guess`, regression tests and `docs/inverter_chain.md`.
PRBS / random digital streams remain **explicitly deferred** (`docs/future_work.md`).

## Artifact layout

- Synthetic default caps (`vgs`, `vdd`, topology id): staged under repository
  `runs/inv_chain/spice_caps/`.
- Matrix outputs: `runs/inv_chain/matrix/n<n>/rep<ID>/`
  (`summary.json`, overlays, checkpoints as produced by the script).
- Canonical doc repro paths mirror `runs/` prefixes (`docs/results.md`).

## Summary JSON contract (parity with compose)

Cold / warm timings follow the CS-amp-aligned pattern: `cold.*_ms`,
`timing.spice_dc_ms`, `timing.fno_dc_ms`, `timing.spice_tran_ms`, `timing.fno_tran_ms`,
and `speedup.spice_ms` / `speedup.fno_ms` with the **same structured `notes`** as
`spino.circuit.compose`. There is **no** misleading top-level wall-clock blob.
Per-internal-node and aggregate entries include **`delay_cross_delta_s`**
`(t_FNO − t_SPICE)` at configured thresholds (see `ChainMetrics.compute` /
`compose_chain._write_summary`). Use these fields before inferring temporal match from
Pearson alone.

## Final experimental status

Latest production-style matrix summaries under `runs/inv_chain/matrix/n{1,2,4}/rep00/`
show a documented regime boundary, not acceptance-quality agreement. Warm-pass
numbers from those runs:

| N | DC converged | DC iters | Transient converged | Transient iters | max\|ΔV\| (V) | Pearson r | Delay (FNO − SPICE) |
|---|---|---|---|---|---|---|---|
| 1 | ✓ | 0 | ✗ | 25 (cap) | 1.800 | NaN | NaN |
| 2 | ✓ | 3 | ✗ | 25 (cap) | 1.777 | NaN | NaN |
| 4 | ✓ | 5 | ✗ | 25 (cap) | 1.778 | NaN | NaN |

The max\|ΔV\| values are ≈ VDD (1.8 V) — i.e., the FNO trajectory is stuck near its
initial condition and never traverses the switching event. Pearson r and delay
metrics return NaN as a downstream consequence (zero-variance comparison window).
These outputs **characterise failure at the digital switching boundary**; they are
not model-quality evidence.

### Resolution attempts that did not close the gap

1. **DC tolerance floor 5e-7 A.** Aligned the convergence tolerance with the FNO's
   physical residual floor at the VSD ≈ 0 rail corner; this stabilised the DC
   solve (0–5 iters across N=1,2,4) but had no effect on the transient.
2. **Damping policy changes.** Multiple Armijo and per-step voltage-cap variants
   tried; Newton kept producing oscillating, non-physical steps near saturation
   plateaus.
3. **Monotone-backtracking correctness fix.** A code bug had been silently masking
   *every* Newton step. Fix restored real step behaviour; the underlying
   convergence pathology then became visible (no progress, not no steps).
4. **Iteration cap 15 → 25.** Confirms the solver is not converging on a longer
   horizon — residual oscillates rather than damping.

The limitation is structural, not a missing solver knob. See root-cause section
below.

## Root cause

The FNO is a temporal operator: spectral convolutions couple all `T` timesteps.
When the chain solver asks autograd for a whole-window Newton Jacobian, the
resulting `(N * T) x (N * T)` matrix contains off-diagonal sensitivities
`d(I[t]) / d(V[t'])`. A quasi-static MOSFET has zero off-diagonal conductive
sensitivity by definition, so those terms are architecture artifacts.

The CS amplifier converges because its analog gain region has a large physical
conductive diagonal, roughly in the mA/V range, which dominates the FNO's
off-diagonal artifacts. The digital inverter chain spends the hard part of the
solve on saturation plateaus where the physical conductive diagonal collapses
toward the uA/V range. At that point, the spurious temporal couplings dominate
`linalg.solve(J, -R)`, and Newton produces oscillating, non-physical steps.

The training corpus reinforces the boundary. `sky130_nmos_61k_plus_shortch_supp8k.h5`
is heavily biased toward dynamic analog-style waveforms where gate and drain vary
together. It does not provide quasi-static fixed-gate drain sweeps or digital step
families rich enough to teach clean conductive Jacobians at fixed bias.

## Status decision

The inverter-chain implementation remains in tree because it is a useful
negative result. It maps the current formulation's validity boundary for digital
switching circuits. It is not an active TODO to be resolved by more damping,
looser tolerances, or another matrix rerun.

## References

- `docs/inverter_chain.md` — user-facing topology and CLI.
- `lab/circuit/circuit_next_steps.md` — composition roadmap with digital convergence removed
  from active work.
- `docs/future_work.md` — exploratory digital paths if the project revisits this regime.
