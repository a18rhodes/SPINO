# Circuit Composition: Next Steps

## Strategic Context

This file is the working roadmap for the circuit composition layer. The top-level
`lab/project/project_next_steps.md` carries the project-wide phasing; this file expands Phase 3 with the
operational detail that does not belong in the project-wide journal.

The original four-subphase split (3a/3b/3c/3d) separated SPICE foundations from
neural-composition machinery. The CS amplifier phases now carry the validated
system-level contribution; the digital phases remain documented as a boundary
result rather than active acceptance work.

---

## Phase 3a: CS Amp SPICE Characterization (COMPLETE)

**Goal:** Establish a SPICE-only reference CS amplifier with a defended sizing point and
an archived set of reference traces (DC OP, VTC, step response). This reference is the
ground truth for Phase 3b.

### Sizing methodology (a priori)

- **Fixed:** L = 0.18 µm for both devices, V_DD = 1.8 V, intrinsic load only (no explicit
  C_load). Bias for OP / transient is the V_in at which the VTC derivative is maximum
  (auto-bias for each corner).
- **Swept:** W_n and W_p on 7×7 grids (see `docs/cs_amp.md` and `characterize.py`). 49 ngspice
  design points, three analyses each. Sub-hour wall time on a typical workstation.
- **Selection rule (committed before running):** maximise peak |dV_out/dV_in| over the
  feasible region {0.6 V <= V_out_bias <= 1.2 V}. Tie-break on lowest static current.
- **Why the constraint:** the linear region of a sky130 CS amp at V_DD = 1.8 V sits roughly
  in [0.4, 1.4] V. The 0.6-1.2 V band keeps the bias point clear of the rails and avoids
  pre-pinch-off ambiguity that would distort the gain estimate.

### Plots produced

1. **Gain heatmap** over (W_n, W_p) -- the sweep result, with the selected sizing marked.
2. **VTC** at the selected sizing, showing the linear region and inverting behaviour.
3. **Transient step response** at the selected sizing, showing settling.

All saved under `docs/assets/cs_amp/` and linked from `docs/cs_amp.md`.

### Tasks

1. [x] SPICE harness committed (`netlist.py`, `simulation.py`, `topologies.py`).
2. [x] CS amp factory `build_cs_amp_active_load`.
3. [x] Unit + e2e tests.
4. [x] `tuning.py`: sweep driver + metric extraction + selection logic.
5. [x] `characterize.py` CLI + `plotting.py`.
6. [x] Update `build_cs_amp_active_load` defaults to the selected sizing.
7. [x] `docs/cs_amp.md`: methodology, sweep results, selected design point, plots.
8. [x] Link `docs/cs_amp.md` from main `README.md`.
9. [x] Final QA: `black`, `pylint`, `pytest --cov` (run before commit).

**Exit criterion:** `docs/cs_amp.md` is review-quality (methodology, results, figures,
discussion); the chosen sizing is reproducible from a single CLI invocation; the reference
traces are deterministic.

---

## Phase 3b: CS Amp Neural Composition (COMPLETE)

Delivered: KCL at `out`, scalar NR DC OP, whole-window implicit transient NR with analytical
`C_load/dt`, production checkpoint loaders, `python -m spino.circuit.compose`, and
`docs/composition.md`. Validation vs 3a uses DC error vs `VDD`, Pearson correlation and
peak voltage error on the figure transient (plain `R²` is reported but is not the gate
metric when plateaus dominate variance; see the doc), settling-time band, NR iteration
budgets, and cold/warm runtime JSON.

**Resolved implementation choices (were open in the 3a notebook):**

- **Damping:** residual-norm backtracking + step cap + rail clip (shared DC and transient).
- **Batched FNO eval:** one forward per device per outer NR iteration; CS amp stacks devices
  only conceptually (separate FNO modules).
- **Runtime protocol:** `summary.json` records cold vs warm solver milliseconds for FNO and
  NGSpice (`.option acct` iteration counts on the SPICE side).

### Composition strategy (locked in 3a, do not relitigate without cause)

- **Linear passives are analytical, not learned.** The `C_load = 10 pF` in the figure
  transient is the only non-MOSFET element in the 3a/3b loop. It enters the NR system
  as a single Jacobian term `C_load / dt` and a residual `C_load * (V^{n+1} - V^n) / dt`
  at the output node. Do not insert the Phase 0 RC operator here; it would only inject
  learned error into a closed-form physics term.
- **Drain-side intrinsic capacitive effects are inside the MOSFET FNOs.** Training labels
  are `vd#branch` from full BSIM4 transients, which includes the displacement currents
  through Cgd and Cdb. The operators reproduce these effects within the slew-rate
  envelope they were trained on (`variable_t_end` mode, up to ~1e8 V/s). At the loaded
  CS amp design, peak slew is ~1.3e7 V/s -- in distribution.
- **Gate-side intrinsic capacitances (Cgs, Cgb, Cgd-from-gate) are out of scope for
  the CS amp.** The gate is voltage-forced by an ideal source; there is no KCL
  obligation at the gate node. Driven gates matter in inverter chains, which are
  now documented as a digital regime boundary rather than an active milestone.

### Open questions

3b functional scope is complete, but two write-up-critical follow-ups remain:

1. **Runtime portability:** CPU and GPU behavior must both be documented as
   engineering context. Runtime is not the claim; differentiability and analog
   composability are.
2. **Bias-region fidelity:** full rail-to-rail VTC shows a low-`Vin` mismatch
   and the transient retains a systematic negative offset despite high Pearson `r`.

### 3b runtime hardening (engineering context)

- **GPU path end-to-end:** run `compose` on CUDA (`--device cuda`) and keep all
  transient tensors (`time`, `vin`, `vout`) and wrapper buffers on the same GPU
  device to remove host/device shuttling.
- **Profiler split:** instrument one transient solve into
  (a) FNO forwards, (b) Jacobian assembly, (c) linear solve, (d) line-search probes;
  record percentages so optimization targets are evidence-based.
- **Matrix-free Newton/Krylov prototype:** replace dense `T×T` Jacobian materialization
  with JVP/VJP-based Krylov (GMRES) for the transient outer loop; keep the analytical
  capacitor contribution exact.
- **Autograd/Jacobian engineering:** evaluate checkpointing and lower precision
  (`float32` baseline, optional mixed precision on GPU) while monitoring convergence
  stability and waveform error gates.
- **Acceptance discipline:** any runtime optimization must preserve
  (DC `|ΔV|/VDD`, transient Pearson `r`, max `|ΔV|`, NR iteration budget).

### 3b fidelity hardening (higher priority than runtime work)

- **Primary target:** improve MOSFET operator fidelity in weak-inversion / near-off
  regions (especially low-`Vin` rail behavior visible in VTC parity), since
  runtime changes do not address the observed accuracy gap.
- **Residual diagnostics:** add per-bias KCL residual-vs-`Vout` root plots comparing
  FNO-composed and SPICE-composed currents to localize equilibrium shifts.
- **Data/fit focus:** prioritize NFET/PFET subthreshold and transitional-regime
  recalibration before major solver rewrites; current NR convergence is already
  healthy and not the first-order accuracy limiter.

### Error attribution (composition roadmap; tracked here)

End-to-end FNO vs SPICE metrics mix IV mismatch, KCL coupling, and numerics.
Progress is recorded in this file and `circuit_status.md`; scope is **attribution
probes**, not statistical confidence bands on headline tables.

- **Operator at fixed voltages:** compare FNO vs SPICE branch currents using
  terminal voltages taken from the SPICE reference (same `W`, `L`, same bias).
- **KCL residual at pinned output:** evaluate the scalar KCL residual at
  converged SPICE `v(out)` and at converged FNO `v(out)` to separate equilibrium
  shift from raw IV error.
- **Solver sensitivity (secondary):** bounded sweeps of NR tolerance / transient
  time step to bound how much of the gap is discretization or solver tolerance
  vs the surrogate.

### Channel length strategy (evaluation; do not shortcut)

**What is invalid:** Running `python -m spino.circuit.compose` with different
`--nfet-l` / `--pfet-l` while keeping the Phase 3a **W_n, W_p** corner and the
**auto-bias protocol** tied to the L = 0.18 µm characterization is not a fair
ablation. The operating point, rail proximity, and which branch of the VTC
matters all move when L changes. Any “L sweep” numbers produced that way are
**not** evidence for or against min-length vs longer-channel composition; treat
them as garbage for decision-making.

**What would be valid:** Pick a candidate geometry policy, then defend it with
data: (a) stay at L = 0.18 µm and push **cross-bin** fidelity (wide W at short L)
plus weak-inversion / transitional **V_g** in the MOSFET track; **or** (b) adopt a
**longer L** after a **full Phase 3a** repeat (same sweep protocol and selection
rule at the new L), new ngspice reference traces, then 3b composition against
**that** reference. Goal for (b): find a channel length where the **selected**
(W_n, W_p) from a honest sweep still meets product goals **and** the operators sit
in a less stressed region of (W, L) space—**if** that is the chosen headline
story. Only apples-to-apples 3a+3b pairs count; naive `compose` L tweaks without a
new 3a sweep do not.

**Cross-binning is the honest stressor:** The locked 3a sizing uses **large**
widths at **L = 0.18 µm**. Stratified training bins separate **short L** (`tiny`
length band includes 0.15–0.30 µm) from **narrow** `tiny` widths (0.42–0.60 µm);
the CS amp is **not** a `tiny×tiny` cell. The current 3b parity plots and gates
are therefore **good science**: they prove the composition stack under NR and
document **where** surrogates still miss SPICE in that cross-bin regime. **Do not
throw away** those artifacts, JSON metrics, or the lab narrative around Experiment
1/2 when exploring a better bin or longer L. They remain the record that
composition meets the stated gates under **wide × short-L** stress while
**cross-bin I-V** error is still the accuracy frontier. Future work: device-side
**wide × short-L** coverage (data / conditioning / curriculum) and/or a second 3a
reference after bin selection (see the section **Backlog: dual CS-amp references and cs_amp.md fix** above).

**Context from the MOSFET track:** NFET/PFET evaluation and lab history already
flag **weaker surrogate metrics in tiny / short-channel geometry bins** (PDK
minimum-adjacent L is worse than mid-L for the same training recipe). Bare FNO
accuracy there was a known pain point; composition inherits that I-V error—it
does not cancel it.

**Path options (to decide explicitly, not by accidental CLI flags):**

1. **Short-channel / nominal L (current 3a default):** Keep the demonstrated
   sizing point and invest in device-side fidelity (subthreshold, transitional
   Vg) where the CS amp VTC already shows stress. Aligns with “analog uses
   aggressive L” reality but keeps the hardest I-V problem in scope.
2. **Longer-L showcase:** After a full 3a redo at larger L, composition may look
   better on paper because the operators sit in a better-covered region of
   (W, L) space—but the circuit is no longer the same design problem (gain,
   bandwidth, area trade-offs shift). That is a legitimate **framing choice**
   if declared honestly.

When the public docs are next revised (not in this lab notebook), any headline
that standardizes on a “cleaner” (W, L) for the composed amp should still
acknowledge the cross-bin / short-L stress track explicitly so the story is not
read as replacing evidence.

---

## Backlog: dual CS-amp references and cs_amp.md fix

**Scope:** The misstatement lives in **`docs/cs_amp.md` only** (e.g. rationale
for L = 0.18 µm that can be read as “training mass” or unqualified “core geometry”
without the stratified-bin facts). **Do not** use this backlog to edit
`docs/nfet.md` or `docs/pfet.md` unless a separate doc pass is opened. **Do not**
edit `docs/cs_amp.md` or `docs/composition.md` until the steps below are executed;
this section is the working checklist only.

1. **Pick L** from the MOSFET stratification, not from `compose` flags alone:
   `spino/mosfet/gen_data.py::GEOMETRY_BINS` gives per-bin `(W, L)` ranges. For
   an in-bin length away from the short-L stress band, the **`small` bin** has
   `L in [0.30, 0.50] µm` (e.g. **0.40 µm** is a defensible in-band value). The
   existing **(W_n, W_p) search** is
   `python -m spino.circuit.characterize` (same 7×7 width grids and
   `tuning.select_design_point` as today). Pass **`--nfet-l` and `--pfet-l`**
   to that command so the sweep and selection run at the new L. Do **not** point
   `compose` at a new L without a matching 3a `summary.json` at that L.
2. **Second 3a run:** same characterize sweep, new L, new `--output-dir` (e.g.
   `docs/assets/cs_amp_l040/`) so the old `docs/assets/cs_amp/` tree stays the
   cross-bin / 0.18 µm control. **Do not** discard the original artifacts.
3. **Then** update **`docs/cs_amp.md` once** to document **both** SPICE
   characterizations (labels, tables, which sweep produced which, links to
   `summary.json` / figures). Two amplifiers, two defensible reference traces.
4. **Only after step 3:** update **`docs/composition.md`** to cross-reference the
   correct 3a ground truth for each `compose` experiment (stress vs “showcase”),
   so FNO numbers are not compared to the wrong SPICE baseline.

`lab/project/project_next_steps.md` Phase 3b item 6 points here for coordination with the
project-wide roadmap.

---

## Phase 3-OTA: 5T OTA multi-node analog composition (COMPLETE, gate open)

OTA is the multi-node analog composition target — three internal KCL nodes
(n_tail, n_left, n_out), differential input pair with floating source terminal,
PFET current mirror, NFET tail current source. SPICE characterisation and FNO
composition are both complete; the L = 0.40 µm max\|ΔV\| gate fails on a
documented PFET training-data coverage gap rather than a solver pathology.

Full lab record: `lab/circuit/ota_status.md`.

### Sub-phase status

- **3a-OTA (SPICE characterisation):** COMPLETE. 7×7 (W_diff, W_mirror) sweep
  at L ∈ {0.40, 0.50} µm, M5 fixed at W = 2 µm, Vbias = 1.2 V. Selection rule
  locked pre-sweep: slew ≥ 5 V/µs AND slew_time ≤ 500 ns, rank by descending
  slew. Selected sizing W_diff = W_mirror = 8 µm at both L. Methodology and
  figures in `docs/ota_5t.md`. Artefacts under `docs/assets/ota_5t_l040/`
  and `docs/assets/ota_5t_l050/`.

- **3b-OTA (FNO composition):** COMPLETE. Multi-node KCL residual with
  backward-Euler load and JVP-GMRES transient linear solve. CLI:
  `compose_ota`. Pearson r and slew-rate gates pass at both L; max\|ΔV\|
  gate fails at 68.7 mV (L = 0.40) and 68.9 mV (L = 0.50) vs the 30 mV
  criterion. Slew-time also fails at L = 0.40.

- **Probe 1 attribution:** COMPLETE. M4 PFET dominates (peak \|ΔI\| = 15.4 µA
  at L = 0.40); other devices < 6 µA. Root cause: M4 Vsd → 0 triode regime
  where PFET training-data density is low.

- **PFET triode-boundary fine-tune (2026-05-15):** COMPLETE — partial closure.
  2K-sample augmentation (Vsg ∈ [1.0, 1.6] V, Vsd ∈ [0.0, 0.3] V), merged into
  46K dataset, frozen-backbone fine-tune (FiLM-conditioning layers only,
  50 epochs, LR = 1e-4). M4 peak \|ΔI\| 15.4 → 12.0 µA (−22 %); composition
  max\|ΔV\| 68.7 → 61.0 mV (−11 %). Gate still fails by 31 mV. M3 regressed
  +8 %; xlarge sweep R² regressed (FiLM-only adaptation over-specialised to
  the triode regime). **Production PFET checkpoint unchanged.** Documented
  in `docs/results.md` § PFET triode-boundary fine-tune.

### Open items (queued)

1. **Larger triode augmentation set (5–10K samples).** Frozen-backbone limit
   was reached on the 2K pass; more coverage density may move the needle
   without the M3 regression seen in FiLM-only training.
2. **Full-backbone fine-tune with seed sweep.** Exp06b (LR = 5e-5 full
   fine-tune on the prior dataset) regressed xlarge Random R²; a multi-seed
   pass at tighter LR may avoid that.
3. **Geometry-stratified retrain weighted by the OTA's actual operating bin.**
   Targets the W = 8 µm regime specifically rather than the full geometry
   distribution.
4. **Adam-based 5T OTA sizing loop** on (W_diff, W_mirror, W_tail, L_common,
   V_bias) — exercises the differentiable composition path. Tracked in
   `lab/project/project_next_steps.md` "Documentation push" / W5–W6.
5. **FD-SPICE gradient-descent baseline** for the sizing-loop comparison.

### Newton/runtime status (OTA)

- JVP-GMRES Krylov path is the default for the transient linear solve;
  replaces dense `T × T` Jacobian materialisation. No outstanding solver
  pathology — convergence is healthy (NR iters ≤ 12 at tested biases).
- Warm CUDA composition runtime: ~63 s (L = 0.40), ~68 s (L = 0.50) vs
  SPICE ~6.3 s / ~6.5 s. Runtime, not fidelity, is the next lever if the
  OTA gate is closed (matrix-free transient Newton + dense Jacobian removal
  per the CS-amp 3b hardening backlog).

---

## Two-stage Miller op-amp (next analog topology, queued)

Next analog composition target after the 5T OTA. Same KCL + Newton machinery
should scale (compensated two-pole amplifier with explicit Cc Miller capacitor
between stages). No active work; queued behind 3b-OTA gate closure decisions
and the sizing-loop demonstration.

---

## Digital switching circuits: documented regime boundary

The CMOS inverter and inverter-chain work is no longer active convergence work.
The implemented `compose_chain` path, chain solvers, topology factories,
partition-cap grids, tests, and documentation remain in the repository because
they characterize a useful negative result.

SPICE converges for the 1-, 2-, and 4-stage inverter-chain references. The
whole-window FNO-composed Newton formulation does not converge to
acceptance-quality digital transients. The cause is structural:

- The FNO is a temporal operator, so autograd produces a dense
  `(N * T) x (N * T)` Jacobian with off-diagonal terms `d(I[t]) / d(V[t'])`.
- A quasi-static MOSFET should have zero conductive sensitivity across distinct
  timesteps.
- In analog gain regions, large physical conductive diagonals dominate those
  artifacts.
- In digital saturation plateaus, the physical `dI/dVds` diagonal collapses and
  the spurious temporal coupling dominates `linalg.solve(J, -R)`.

Do not spend more roadmap time on damping, tolerance, or iteration-budget tuning
for the existing digital formulation. The result is a validity boundary, not an
unfinished Phase 3 milestone.

### Future work for digital

These are exploratory directions only:

1. **IFT / DEQ-style differentiation:** let SPICE solve the circuit, then use
   the implicit-function theorem to compute parameter gradients at the converged
   point.
2. **SPICE as host solver, FNO as device oracle:** keep SPICE's MNA and Newton
   machinery, with the FNO providing `(Vg, Vd, Vs, Vb) -> (I, gm, gds)` for
   devices whose physics SPICE cannot model.
3. **Quasi-static-rich retraining:** add fixed-gate drain sweeps,
   fixed-drain gate sweeps, and digital step waveforms to teach clean
   conductive Jacobians at fixed bias.
4. **Digital-state reparameterization:** solve for transition times and slopes
   rather than every voltage at every timestep.
5. **Sequential per-timestep Newton with quasi-static FNO:** remove
   whole-window temporal coupling, accepting the loss of the batched FNO solve.

---

## What NOT To Do (in this layer)

- Do not introduce a generic device registry. Four target circuits is not enough complexity
  to justify it. If a fifth topology lands later, refactor then.
- Do not invent a new netlist intermediate language. SPICE text is the contract.
- Do not optimise the SPICE harness unless it is needed for measurement hygiene.
  The contribution is differentiability and analog composability, not wall time.
- Do not resurrect the inverter-chain depth study as an active milestone without
  choosing a new digital strategy from the exploratory options above.
