# Circuit Composition: Current Status

## Overview

This document tracks progress on the circuit composition layer -- the SPICE harness and the
neural-composition machinery that turn the trained device operators (RC, Diode, NFET, PFET)
into multi-device simulators. This is Phase 3 of the SPINO roadmap.

**Status: Phase 3a (SPICE CS amp ground truth) complete. Phase 3b (neural composition: KCL +
NR over production NFET/PFET FNOs, `compose` CLI, `docs/composition.md`, `docs/pipeline.md`)
is implemented and validated against the 3a reference with the metrics documented there.
Code review pass (2026-04-26): residual primitive deduplicated, Armijo damping naming
made precise across composition.py / tests / docs, `compose.py` `main` decomposed into
`_render_figures` and `_build_summary`, dead `spice_kwargs` arg removed from FNO speedup
helper, `pylint == 10.00/10` on the four files (`composition.py`, `compose.py`,
`devices.py`, `composition_io.py`).**

**Experiment 1 (2026-04-27, CPU baseline):**

- **DC accuracy at nominal bias:** `Vout_FNO = 0.5923 V`, `Vout_SPICE = 0.6089 V`,
  `|ΔV|/VDD = 0.92%` (passes the 5% gate).
- **Transient agreement (10 pF, 50 mV step):** Pearson `r = 0.99748`,
  `max |ΔV| = 25.79 mV` (passes current acceptance gates), but with a visible
  systematic negative offset and ~30 ns FNO settling vs 25 ns SPICE.
- **VTC caveat:** full rail-to-rail VTC still deviates at low `Vin`
  (`pearson_r = 0.9921`, `max_abs_error_v = 0.3829 V`), consistent with
  weak-inversion / near-off-region mismatch and rail clipping effects.
- **Runtime:** composition stack is slower than SPICE on CPU
  (`FNO warm ≈ 25.85 s` vs `SPICE warm ≈ 13.02 s` for the timed DC+transient pair).
  Device-level FNO speedups do not yet translate into composition-level speedups.

**Experiment 2 (2026-04-27, CUDA path, same code + same stimulus):**

- **DC accuracy at nominal bias:** `Vout_FNO = 0.5924 V`, `Vout_SPICE = 0.6089 V`,
  `|ΔV|/VDD = 0.92%` (no material change vs Experiment 1).
- **Transient agreement:** Pearson `r = 0.99748`, `max |ΔV| = 25.78 mV`
  (essentially unchanged vs Experiment 1).
- **VTC caveat persists:** low-`Vin` rail mismatch remains
  (`pearson_r = 0.9929`, `max_abs_error_v = 0.3797 V`).
- **Runtime:** major wall-time improvement from GPU execution:
  `FNO warm ≈ 2.74 s` vs `SPICE warm ≈ 13.42 s` (~4.9x faster).

**Interpretation:** GPU resolves the immediate composition runtime bottleneck without
improving composition fidelity. The dominant remaining accuracy issue is model fidelity
in weak-inversion / near-off regions, not NR convergence failure.

**Error attribution (roadmap):** decomposition of end-to-end disagreement into
operator-at-fixed-`V`, KCL residual, and solver/numeric probes is tracked under
**Error attribution (composition roadmap)** in `circuit_next_steps.md`, not in project-wide
future-work docs.

---

## Module Map

| File                | Purpose                                                                 |
|---------------------|-------------------------------------------------------------------------|
| `__init__.py`       | Public API surface (`MosfetInstance`, `VoltageSource`, `Circuit`, ...). |
| `netlist.py`        | Dataclass primitives + SPICE rendering. `SpiceDevice` Protocol.         |
| `simulation.py`     | Thin runners over `ngspice`: `.op`, `.dc`, `.tran`. Result dataclasses. |
| `topologies.py`     | Factory functions for validation circuits. CS amp implemented.          |
| `devices.py`        | `FnoMosfetDevice`: normalization + differentiable drain current.        |
| `composition_io.py` | Load checkpoints + HDF5 stats + BSIM physics into device wrappers.      |
| `composition.py`    | `DcOperatingPointSolver`, `TransientSolver`, shared NR damping.         |
| `compose.py`        | Composition CLI: DC, transient, VTC overlay, `summary.json`, speedup.   |

The intent of this layer is to be **boring**: declarative primitives with no hidden state,
straightforward rendering, no abstractions beyond what the four target circuits (CS amp,
inverter, inverter chain, and one diode-resistor sanity check) require.

## Design Decisions (and rejections)

- **Plain dataclasses over an ABC hierarchy.** An earlier draft used `DeviceMaster` and
  `DeviceInstance` ABCs with abstract properties. The interaction between `@dataclass` and
  inherited abstract properties produced surprising `TypeError` failures on field ordering,
  and the abstraction served no concrete purpose for the four target topologies. Replaced
  with `@dataclass(frozen=True, slots=True)` on each device, port validation in
  `__post_init__`, and a `SpiceDevice` `Protocol` for structural typing.
- **Frozen and slotted.** Devices are immutable hashable values; cheaper to keep them so
  than to chase aliasing bugs later in the Newton-Raphson solver where the same instance is
  evaluated thousands of times per simulation.
- **Tuple, not list, for `Circuit.devices`.** Same rationale: hashable circuit instances
  let the composition layer cache Jacobian sparsity patterns by circuit identity.
- **No symbolic netlist intermediate.** The `to_spice()` method renders directly to text.
  No abstract MNA matrix, no symbolic graph. The neural composition path builds KCL
  directly from device terminals without going through SPICE syntax.
- **One residual primitive per solver, not two.** Earlier drafts kept separate
  `_residual_with_grad` and `_residual_no_grad` methods on the DC solver. Their bodies
  were identical; the only difference was the `@torch.no_grad()` decorator. Collapsed
  to a single `_residual` and let callers control the autograd context. Same applies to
  the post-trim mean-current evaluation, now exposed as `_device_currents`.
- **"Armijo" not "Wolfe-style" in damping docstrings.** The line search applies the
  Armijo sufficient-decrease half of the Wolfe conditions to a residual *norm*; the
  curvature half is omitted because we do not have a smooth scalar gradient. Naming
  was made consistent across `composition.py`, the tests, and the public docs.

## Test Coverage

| Test file                   | Scope                                                            |
|-----------------------------|------------------------------------------------------------------|
| `test_netlist.py`           | Dataclass invariants, port validation, immutability, rendering.  |
| `test_simulation.py`        | `.op`, `.dc`, `.tran` runners against ngspice on a passive RC.    |
| `test_cs_amp_e2e.py`        | Full pipeline: factory -> deck -> ngspice -> bounded results.    |
| `test_composition.py`       | NR solvers, damping, NGSpice iter parse, production integration. |
| `test_fno_device_wrapper.py`| `FnoMosfetDevice` shapes, norm, autograd, polarity.              |
| `test_composition_io.py`    | Loader failure modes and optional CPU map smoke test.            |

Coverage on `netlist.py` and `__init__.py` is 100%. `simulation.py` is at 79% and
`topologies.py` at 95%; the gaps are error-handling branches that the upcoming sizing
harness will exercise naturally.

## Known Limitations

- **Single-port voltage sources only.** No PWL stimulus support yet. Step responses are
  composed manually from two `.tran` runs. Adequate for Phase 3a; will be revisited in 3b
  if the FNO composition layer needs richer drive waveforms than the sizing study used.
- **No `.ac` analysis.** Bandwidth measurements in 3a are derived from settling-time
  bounds rather than direct AC sweep. This is intentional: the FNO operators are
  trained on time-domain signals, so AC-domain validation has no analogous neural target.
- **PDK path is implicit.** `simulation.py` resolves the sky130 model card via the same
  environment hook used by `spino.spice`. No isolation between SPICE backend and harness yet.

## Open Questions (deferred to Phase 3c / 3d)

- Second internal node (inverter): refactor shared solver assembly only when required.
- Gate-charge / `dQ_G/dt` at driven gates (inverter chain): see `lab/circuit/circuit_next_steps.md` Phase 3d.

## Lab backlog (docs — do not edit from this file)

- **`lab/circuit/inverter_chain_status.md`:** Markdown lab log for the
  inverter chain composition path (staging under `runs/inv_chain/`, summary JSON
  fields, known non-convergence on current matrix runs, triage order). Not a Jupyter
  notebook.
- **`docs/cs_amp.md`:** Known **misstatement** in the L-rationale. **Next:** a second
  3a pass using the same `characterize` **(W_n, W_p) search** with `--nfet-l` /
  `--pfet-l` set from a chosen `GEOMETRY_BINS` L band (e.g. `small`: 0.30–0.50 µm,
  **0.40 µm** as a concrete length), new asset tree, then the dual-amp `cs_amp.md` /
  `composition.md` doc pass. The harness exposes L on
  `python -m spino.circuit.characterize` for that. Existing 0.18 µm 3a/3b results
  remain the cross-bin control.
