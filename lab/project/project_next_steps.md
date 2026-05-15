# SPINO Project: Next Steps (Roadmap)

## Strategic Context

The RC dimensionless formulation has been validated as the correct approach. The diode
experiment (Phase 1) confirmed that it scales to nonlinear systems: the same lambda-based
stiffness encoding that makes RC resolution-invariant and time-scale-invariant works for
the Shockley equation. Two of four target devices (RC and Diode) are now production-ready
with the unified dimensionless representation.

**Thesis:** Dimensionless PINOs enable composable circuit simulation via Neural Newton-Raphson.
Phase 1 removed the scientific risk from this thesis. Phase 2 (PFET) removed the practical
risk by adding the second nonlinear device needed for both analog (CS amp, 5T OTA) and
digital (CMOS inverter) composition tests. Phase 3 validates the composition machinery on
both classes — single-node analog (CS amp), multi-node analog (5T OTA), and digital
inverter chains — and characterizes how surrogate error compounds with circuit depth.

---

## Roadmap (Hybrid Path C)

### Phase 1: Dimensionless Diode -- COMPLETE (2026-03-05)

**Goal:** Validate that the dimensionless formulation scales to nonlinear systems.

1. [x] Code modernization: CLI, HDF5 pipeline, structured logging
2. [x] Dimensionless conversion: variable T_end, lambda = RC/T_end, normalised I/V
3. [x] Loss upgrade: MSE + Sobolev with fade-in schedule
4. [x] Validation: resolution invariance (1024/2048/4096), variable T_end test

**Results (D2 checkpoint `diode_dimless_v2_VokyITJR`):**
- Standard rectifier (lambda=0.01): R2=0.9994, MAE=45 mV
- Adversarial (random params): R2=0.9999, MAE=3 mV
- Resolution invariance: delta R2 < 0.0001 at 1024/2048/4096
- Variable T_end: R2 >= 0.997 across {100us, 1ms, 10ms}

### Phase 2: PFET -- COMPLETE (2026-03-22)

**Goal:** Train PFET using the proven VCFiLM architecture.

**Result:** Exp06 production model (`mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt`)
- Core: Transfer R2=0.9965, Output R2=0.9656, SubTh-R2=0.9523
- 44K training set (40K PWL + 2K output sweeps + 2K transfer sweeps)
- Speedup: ~522x vs SPICE
- Polarity bug found and fixed (Exp03+, 2026-03-11)
- Sweep-augmented dataset was the breakthrough (Exp06: Sweep R2 0.819 -> 0.994)

**Steps (all complete):**

1. [x] Phase 0: Invariance characterization (scripts/test_nfet_invariance.py)
2. [x] Refactor shared code: strategy-aware evaluate.py, train.py, gen_data.py
3. [x] Validate PMOS BSIM4 parameters (29 TRAINING_KEYS present, non-zero variance)
4. [x] Add variable T_end support to data generation pipeline
5. [x] Generate PFET dataset (44K sweep-augmented)
6. [x] Training: 300 epochs, LpLoss, batch_size=64 (Exps 01-06)
7. [x] Comprehensive evaluation (3 geom x 3 waveform)
8. [x] Create docs/pfet.md, update all status docs

See lab/mosfet/mosfet_status.md for full PFET experiment history (Exps 01-06b).

### Phase 3: Composition & Simulation (Current Focus)

**Goal:** Compose NFET + PFET operators into multi-device circuits, validate against ngspice
ground truth, and quantify how surrogate accuracy compounds across stages of depth. The phase
is split into four subphases so the SPICE foundation and the neural-composition machinery are
each validated against an independent reference before the cross-cutting depth study.

**Validation targets:**

| Subphase | Topology | Class | Primary metrics                                  |
|----------|----------|-------|--------------------------------------------------|
| 3a       | CS amp (NFET driver + PFET active load) | Analog | DC bias, peak gain, gain-bandwidth, settling |
| 3b       | CS amp (FNO composition of 3a)          | Analog | KCL residual, NR iters/step, R² vs 3a reference |
| 3a-OTA   | 5T OTA SPICE characterization (7×7 sweep, L ∈ {0.40, 0.50} µm) | Analog | slew rate, slew time, I_tail, DC gain, feasibility |
| 3b-OTA   | 5T OTA FNO composition (multi-node KCL) | Analog | Pearson r, max\|ΔV\|, slew metrics, NR iters/step |
| 3c       | CMOS inverter (negative result)          | Digital | Regime boundary — FNO NR does not converge       |
| 3d       | Inverter chain N ∈ {1,2,4} (negative result) | Digital | Regime boundary — FNO NR does not converge  |

Both MOSFET operators are quasi-static (algebraic I-V) and accept arbitrary time grids without
modification, so no mixed-formulation issues arise in the composition layer. The RC/diode
lambda formulation handles ODE dynamics independently if such devices appear later.

#### Phase 3a: CS Amp SPICE Characterization (foundational)

**Status:** Complete. See `docs/cs_amp.md`, `docs/assets/cs_amp/`, and `python -m spino.circuit.characterize`.

```
VDD --[PFET]-- V_out --[NFET]-- GND        (PFET = active load, NFET = driver)
```

1. [x] Dataclass-driven netlist + ngspice runners (`spino/circuit/netlist.py`, `simulation.py`).
2. [x] CS amp factory `build_cs_amp_active_load` (`spino/circuit/topologies.py`).
3. [x] Unit + e2e tests for SPICE harness (`tests/circuit/`).
4. [x] 2D sizing sweep over (W_n, W_p) at fixed L = 0.18 µm; metrics: peak |gain|, V_out at
       bias, settling time, static current. Selection rule documented a priori.
5. [x] Plots: VTC, step response at chosen point, gain heatmap.
6. [x] Document sizing methodology + results in `docs/cs_amp.md` and link from main README.

**Acceptance:** Documented SPICE reference with inverting VTC, peak |gain| > 1 V/V in the
selected band, output bias within the pre-registered feasible window. Sizing point and
traces archived for 3b.

#### Phase 3b: CS Amp Neural Composition

**Goal:** Replace ngspice device evaluations with FNO operators (NFET Exp 19b, PFET Exp 06)
inside a Newton-Raphson loop and reproduce the 3a reference behavior.

1. [x] Composition engine: circuit graph, KCL residual at internal nodes via differentiable
       FNO device evaluations.
2. [x] Jacobian assembly via PyTorch autograd through the FNO stack.
3. [x] Newton-Raphson solver: whole-window transient NR, convergence/damping, DC OP solver.
4. [x] Linear capacitor support (`C_load = 10 pF` figure transient): one `C/dt` term in
       the KCL residual and one in the Jacobian. Do NOT route the Phase 0 RC operator here;
       it is a learning-phase artefact, not a composition primitive.
5. [x] Validate FNO-composed CS amp vs the 3a reference: DC OP within 5%; transient agreement
       via Pearson correlation, peak voltage error, and documented `R²` caveats (see
       `docs/composition.md`).
6. [x] Speedup measurement vs full ngspice `.tran` for the same stimulus (`compose` JSON).
7. [x] Document in `docs/composition.md`.

**Experiment 1 (2026-04-27, CPU baseline):**
composition fidelity gates pass at the selected bias/stimulus, but composition
runtime is still slower than SPICE on the current dense-Jacobian CPU path
(`warm FNO ≈ 25.9 s` vs `warm SPICE ≈ 13.0 s` for timed DC+transient).
Treat composition-speed optimization as an immediate pre-write-up task.

**Experiment 2 (2026-04-27, CUDA path, same code and stimulus):**
composition fidelity gates remain effectively unchanged, while runtime improves to
`warm FNO ≈ 2.74 s` vs `warm SPICE ≈ 13.42 s` (composition now faster on this GPU).
This removes the immediate speed blocker for 3b on GPU but does not resolve
low-`Vin` VTC mismatch / transient equilibrium offset.

**Acceptance:** DC operating point within 5% of 3a SPICE reference; transient Pearson
`r > 0.997` and `max |ΔV| ≤ 30 mV` on the figure stimulus with SPICE-aligned IC; NR
outer loops under 10 iterations at tested biases; cold/warm speedup JSON; methodology
and metric caveats in `docs/composition.md`.

**Scope clarifications recorded during 3a (do not relitigate without cause):**

- Drain-side intrinsic capacitive contributions are inside the MOSFET FNOs. Training labels
  are `vd#branch` from full BSIM4 transients, which by KCL include the displacement currents
  through Cgd and Cdb. The operators reproduce these effects in the slew-rate envelope they
  were trained on (variable T_end mode samples up to ~1e8 V/s).
- Gate-side intrinsic capacitances (Cgs, Cgb, Cgd-from-gate) are NOT inside the operators.
  Irrelevant for the CS amp because the gate is voltage-forced. Relevant for inverter chains
  (Phase 3d), which are now a documented negative result; see `lab/circuit/circuit_next_steps.md`.

**3b performance hardening tasks (required before final write-up claims):**

1. [ ] Run `compose` with CUDA end-to-end (`--device cuda`) and verify all composition tensors
       and wrappers remain device-consistent (no CPU fallback in transient solve path).
2. [ ] Add profiler breakdown for one transient solve:
       FNO forwards vs Jacobian assembly vs linear solve vs line-search probes.
3. [ ] Prototype matrix-free transient Newton/Krylov (JVP/VJP + GMRES) to avoid dense `T×T`
       Jacobian materialization.
4. [ ] Re-benchmark cold/warm composition vs SPICE on the same hardware and update docs with
       measured numbers and caveats.
5. [ ] Prioritize MOSFET model fidelity improvement in weak-inversion / near-off regions
       (NFET/PFET) as the next accuracy lever for composition; NR convergence is currently
       healthy and not the first-order limiter.
6. [ ] **Geometry / docs (circuit track):** `docs/cs_amp.md` still misstates the L
       rationale; fix is **deferred** until a second 3a run exists. Re-run the same
       `python -m spino.circuit.characterize` (W_n, W_p) **search** with
       `--nfet-l` / `--pfet-l` set from a `GEOMETRY_BINS` choice (see circuit
       `NEXT_STEPS` backlog), a fresh `--output-dir`, then the dual-amp doc pass.
       **Do not** edit `docs/cs_amp.md` / `docs/composition.md` before that. **Preserve**
       existing 3a/3b cross-bin (0.18 µm) artifacts.

#### Phase 3a-OTA: 5T OTA SPICE Characterization (COMPLETE)

**Status:** Complete. 7×7 (W_diff, W_mirror) sweep at L ∈ {0.40, 0.50} µm with
M5 (tail) fixed at W = 2 µm, Vbias = 1.2 V. Stimulus: ±50 mV differential step
at Vcm = 0.9 V, C_load = 1 pF. Selection rule (locked pre-sweep): slew ≥ 5 V/µs
AND slew_time ≤ 500 ns, rank by descending slew. Selected sizing
W_diff = W_mirror = 8 µm at both L. Reference traces and selection-rule JSON
archived to `docs/assets/ota_5t_l040/` and `docs/assets/ota_5t_l050/`.
Methodology and figures in `docs/ota_5t.md`.

#### Phase 3b-OTA: 5T OTA FNO Composition (COMPLETE, documented gate failure)

**Status:** Composition machinery validated multi-node without structural changes
to the Newton solver. Pre-registered gates met on Pearson r and slew rate but
**max\|ΔV\| gate (≤ 30 mV) fails** at both L (68.7 mV at L = 0.40, 68.9 mV at
L = 0.50). Probe 1 attribution localised the gap to M4 PFET in the Vsd → 0
triode regime — a training-data coverage gap, not a solver pathology.

1. [x] Multi-node KCL residual (n_tail, n_left, n_out) with floating M1/M2
       source terminal and PFET current mirror.
2. [x] DC OP and whole-window transient Newton solvers (`OtaDcSolver`,
       `OtaTransientSolver`).
3. [x] JVP-GMRES Krylov path (`use_gmres=True`, default) for the transient
       linear solve — replaces dense Jacobian assembly.
4. [x] CLI `compose_ota` with `summary.json` parity vs CS-amp harness.
5. [x] Probe 1 attribution (`ota_attribution` CLI): per-device IV error at
       SPICE node voltages.
6. [x] PFET triode-boundary fine-tune (2026-05-15): targeted 2K-sample
       augmentation (Vsg ∈ [1.0, 1.6] V, Vsd ∈ [0.0, 0.3] V), frozen-backbone
       fine-tune. M4 peak \|ΔI\| 15.4 → 12.0 µA (−22 %); composition max\|ΔV\|
       68.7 → 61.0 mV (−11 %). Gate still fails by 31 mV. Production PFET
       checkpoint unchanged; fine-tune archived as a documented partial-closure.
       Full report in `lab/circuit/ota_status.md`; documented in
       `docs/results.md` § PFET triode-boundary fine-tune.

**Outstanding (if the gate must close):**

- Larger triode augmentation set (5–10K samples) — more coverage density.
- Full-backbone fine-tune at very tight LR with seed sweep — avoid the
  xlarge sweep regression seen in the frozen-backbone pass.
- Geometry-stratified retrain weighted by the OTA's actual operating bin.
- Deferred to future work; production checkpoint and OTA composition
  artefacts ship as-is with the partial-closure result documented.

**Validation reference:** `docs/assets/ota_5t_fno_l040/`,
`docs/assets/ota_5t_fno_l050/`, `docs/assets/ota_5t_fno_l040_exp07/`.

#### Phase 3c/3d: CMOS Inverter and Inverter Chain — CLOSED (negative result)

**Outcome:** Executed. SPICE converges for N = 1, 2, 4 inverter chains. The
whole-window FNO-composed Newton solver does not converge to acceptance-quality
digital transients. Acceptance criteria below were not met and cannot be met with
the current formulation.

**Root cause** (see `lab/circuit/circuit_next_steps.md` and
`lab/circuit/inverter_chain_status.md`): FNO spectral temporal
convolutions produce spurious autograd off-diagonal Jacobian terms. In digital
saturation plateaus the physical conductive diagonal collapses and those artifacts
dominate Newton steps. This is a formulation boundary, not a solver-tuning defect.

**Acceptance criteria (not met):**
- 3c: V_M within 50 mV of SPICE; propagation delay within 10%; transient R² > 0.99.
- 3d: Empirical error-growth scaling law with confidence intervals.

**What is in the repository:** Topology factories, chain solvers, partition-cap
infrastructure, CLI (`compose_chain`), matrix harness, tests, and
`docs/inverter_chain.md` are retained as a regime-boundary artifact. The
`summary.json` files under `runs/inv_chain/matrix/` document the negative result.
Do not treat these as pending milestones.

Exploratory directions for future digital work (IFT/DEQ, SPICE-as-host, data
augmentation, reparameterization) are listed in `lab/circuit/circuit_next_steps.md`
and `docs/future_work.md`.

### Phase 4: NFET Dimensionless Refactor -- CANCELLED

Phase 0 invariance characterization proved this is unnecessary. The MOSFET I-V is
algebraic; lambda carries no information. The NFET operator is already time-scale and
resolution invariant without any refactoring.

---

### Documentation push (2026-05): in progress

8-week push to consolidate results into a coherent write-up. Plan tracked in
the author's local task/plan store (not in repo).

**Completed:**

- **MLP architecture ablation (W1, 2026-05-14).** Per-timestep `MosfetMLP` trained on
  the same 61K NFET dataset at h64/h128 capacities. Fast Dataset R²: MLP −4.42 / −5.43
  vs FNO 0.99. Gap widens with capacity, ruling out underfitting. Documented in
  `docs/results.md` § MLP ablation. Branch `paper/mlp-ablation`
  (commits c25557e, 488497f) — branch name predates the rename of the push.
- **Gradient sanity test (W2, 2026-05-14).** `tests/circuit/test_gradient_sanity.py`
  verifies FNO autograd ∂I_D/∂w matches SPICE central FD at OTA M1 DC OP. Same
  gradient the W5–6 Adam sizing loop will rely on.
- **PFET triode-boundary fine-tune (W3, 2026-05-15).** 2K-sample augmentation,
  46K stratified set, frozen-backbone fine-tune of production PFET. M4 peak |ΔI|
  −22 %, OTA max|ΔV| 68.7 → 61.0 mV (−11 %). Gate still fails. M3 regressed +8 %,
  xlarge sweep R² regressed. Production checkpoint unchanged.
  Full report: `lab/circuit/ota_status.md` § PFET triode fine-tune.
  Documented in `docs/results.md` § PFET triode-boundary fine-tune.

**In flight (W5–6, weeks of 2026-05-19 / 2026-05-26):**

- Adam-based 5T OTA sizing loop on (W_diff, W_mirror, W_tail, L_common, V_bias).
- FD-SPICE gradient-descent baseline for the comparison.

**Queued (W7–W8):**

- Per-iteration evaluation-cost comparison, SPICE validation at converged θ.
- Write-up consolidation: figure set, headline numbers, reproduction commands.

---

## Timeline (Revised 2026-04-26)

| Phase    | Duration | Deliverable                                                          | Status     |
|----------|----------|----------------------------------------------------------------------|------------|
| 1        | 2 weeks  | Dimensionless diode, validated                                       | COMPLETE   |
| 2        | 3 weeks  | PFET trained and evaluated                                           | COMPLETE   |
| 3a       | 1 week   | CS amp SPICE harness + sizing characterization                       | COMPLETE   |
| 3b       | 2 weeks  | CS amp neural composition (KCL + NR over FNO operators)              | COMPLETE   |
| 3a-OTA   | 1 week   | 5T OTA SPICE characterization (7×7 sweep at L ∈ {0.40, 0.50} µm)     | COMPLETE   |
| 3b-OTA   | 2 weeks  | 5T OTA FNO composition (multi-node) + Probe 1 attribution + triode fine-tune | COMPLETE (gate open) |
| 3c       | --       | CMOS inverter — negative result, regime boundary                     | CLOSED     |
| 3d       | --       | Inverter chain — negative result, regime boundary                    | CLOSED     |

Phase 4 (NFET refactor) cancelled -- invariance proven.

---

## What NOT To Do

- Do NOT add a lambda channel to MOSFET operators. Phase 0 proved it carries zero information.
- Do NOT attempt physics residual for diode or PFET as a first pass. Exponential instability.
  MSE + Sobolev is sufficient (proven by diode D2).
- Do NOT over-engineer the composition engine. Start with the simplest circuit (CS amp)
  and the simplest solver (vanilla Newton-Raphson). Add complexity only when validated.
- Do NOT resurrect Phase 3c/3d as active milestones. The inverter-chain result is a
  documented negative result and regime boundary. See `lab/circuit/circuit_next_steps.md`
  for any future digital strategy options.
- Do NOT hand-derive Jacobians. Use PyTorch autograd through the FNO forward pass.
- Do NOT mix FNO-predicted and SPICE-computed currents in the same composition. All
  devices in a composed circuit must use their trained FNO operators for consistency.
