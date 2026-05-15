# SPINO Project: Current Status

## Overview

SPINO (SPIce Neural Operator) uses Fourier Neural Operators to learn continuous operator
mappings from terminal waveforms + device parameters to node current/voltage responses,
replacing inner-loop SPICE device evaluation. The end goal is composable circuit simulation
via Neural Newton-Raphson coupling of modular FNO device blocks.

---

## Device Status Summary

| Device  | Status       | Dimensionless | Architecture     | Composable | Notes                                      |
|---------|-------------|---------------|------------------|------------|-------------------------------------------|
| RC      | Production  | Yes           | Vanilla FNO 2ch  | Yes        | Gold standard formulation                  |
| Diode   | Production  | Yes           | Vanilla FNO 6ch  | Pending    | D2: R2=0.9994 rectifier, R2≥0.997 all windows |
| NFET    | Production  | Quasi-static  | VCFiLM FNO 2.3M  | Pending    | Exp 19b, R2>0.99 across geoms, ~1300x speedup |
| PFET    | Production  | Quasi-static  | VCFiLM FNO 2.3M  | Pending    | Exp06, R2=0.9965 transfer, ~522x speedup   |

### RC (Complete)
- Dimensionless formulation: t_hat = t/T_end, lambda = tau/T_end
- 4 data generator variants (pulse, noise, chirp, log-uniform)
- Physics-informed loss: MSE + Sobolev + residual with fade-in
- Resolution invariant: validated at 1024, 2048, 4096 steps
- Time-scale invariant: tau from 100fs to 10s
- ~131K parameters

### Diode (Production, Dimensionless)
- **Dimensionless formulation validated:** lambda=RC/T_end, 6-channel input, V_hat denorm
- D2 checkpoint: R2=0.9994 standard rectifier, R2=0.9999 adversarial
- Resolution invariance confirmed: delta R2 < 0.0001 at 1024/2048/4096 grid points
- Time-scale invariance confirmed: R2 >= 0.997 across T_end in {100us, 1ms, 10ms}
- T_end clamped to [10ns, 10ms] to ensure SPICE-safe sim steps across full parameter space
- MSE + Sobolev loss with fade-in schedule (physics residual skipped: exponential instability)
- See lab/diode/diode_status.md for full experiment log

### NFET (Production, Quasi-Static Invariant)
- VCFiLM architecture: per-timestep voltage-conditioned FiLM modulation
- 29 curated BSIM4 parameters via DeviceEmbedding MLP
- Exp 19b: xlarge SubTh-R2 0.9113, Core Output R2 0.9960
- Pre-generated HDF5 pipeline (61K+ samples), click CLI
- Full evaluation suite (3x3 comprehensive, SPICE sweeps)
- **Invariance characterization (2026-03-06):** Time-scale invariant (delta R2 < 0.001 across 100ns-5us), resolution invariant (delta R2 < 0.000002 across 512-4096 steps). Lambda is unnecessary -- MOSFET I-V is algebraic, not ODE-governed.
- **MLP ablation (2026-05-14):** See below.
- See lab/mosfet/mosfet_status.md for experiment history

### NFET MLP Ablation (Architecture Defense)

Per-timestep `MosfetMLP` baseline trained on same 61K dataset as Exp 19b (same loss, optimizer, epochs).
Ablation addresses: does FNO temporal mixing provide anything beyond what a quasi-static pointwise MLP can learn?

**Result (fixed-seed 64-sample average, 2026-05-14):**

| Metric | MLP h64 (32K) | MLP h128 (58K) | FNO Exp19b (1.28M) |
|---|---|---|---|
| Fast Dataset R² | -4.42 | -5.43 | **0.9879** |
| Transfer R² | 0.9990 | 0.9989 | **0.9995** |
| Transfer SubTh-R² | **0.9856** | 0.9631 | 0.9861 |
| Output R² | 0.9456 | 0.9763 | **0.9960** |
| Speedup vs SPICE | 863x | 6501x | 473x |

**Interpretation:** MLP matches FNO on controlled I-V sweeps (monotonic ramp inputs) but fails
catastrophically on arbitrary PWL dataset waveforms (Fast Dataset R² ≈ −4 to −5). The gap
worsens with more capacity (h128 < h64 on dataset R²), ruling out a simple capacity explanation.
FNO temporal mixing acts as a **waveform-shape regularizer**: spectral convolutions average
information across the full input trajectory, improving generalization to diverse waveform types
seen in the training distribution. This is not physically required by quasi-static MOSFET physics
but is empirically necessary for distribution-robust generalization.

**Headline claim:** "The MLP surrogate matches FNO on controlled sweeps but fails to generalize to
arbitrary PWL waveforms (dataset R² −4.4 vs 0.99), demonstrating that FNO temporal mixing
provides waveform-shape regularization rather than physically necessary temporal dependencies."

Figures: `docs/assets/mosfet/nfet/mlp_ablation/`

### PFET (Production, Quasi-Static Invariant)
- **Exp06 is the production model** (`mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt`)
- VCFiLM architecture: same as NFET, in_channels=4, input_param_dim=29, no lambda
- 44K training set (40K PWL + 2K output sweeps + 2K transfer sweeps)
- Core: Transfer R2=0.9965, Output R2=0.9656, SubTh-R2=0.9523, MALE=60uA
- Speedup: ~522x vs SPICE (warm GPU)
- Known limitations: arcsinh compression reduces sweep signal span (PMOS Ids ~10x lower than NMOS), medium/xlarge SubTh-R2 degenerate due to compression
- Polarity bug found and fixed (Exp03+, 2026-03-11)
- **Exp07 triode-boundary fine-tune (2026-05-15):** Frozen-backbone FiLM-only adaptation on
  46K dataset (+2K Vsd→0 augmentation). M4 OTA peak |ΔI| reduced 22 % (15.4 → 12.0 uA);
  OTA max|ΔV| at L=0.40 reduced 11 % (68.7 → 61.0 mV). Gate (30 mV) still fails. M3
  non-triode PFET regressed slightly. Production model unchanged (Exp06). See
  `docs/results.md` and `docs/assets/ota_5t_fno_l040_exp07/`.
- See lab/mosfet/mosfet_status.md for experiment history (Exps 01-07)

---

## Composition / Newton-Raphson -- Phase 3 (Current Focus)

**Status (2026-05-07):** Phases 3a and 3b complete and validated. Phase 3c/3d
(digital) was executed, converged on a documented negative result, and is closed.

### Dimensional Consistency -- Resolved

Phase 0 invariance characterization (2026-03-06) demonstrated that MOSFET operators
are inherently time-scale and resolution invariant without dimensionless formulation.
The MOSFET I-V is algebraic (quasi-static), unlike RC/diode which are ODE-governed.

All device operators ready for composition:
- RC: dimensionless t_hat in [0,1], lambda encodes stiffness -- **production**
- Diode: dimensionless t_hat in [0,1], lambda encodes stiffness -- **production**
- NFET: quasi-static, empirically invariant -- **production** (Exp 19b)
- PFET: quasi-static, same architecture as NFET -- **production** (Exp 06)

Composition is simpler than originally expected: MOSFET operators accept arbitrary
time grids without modification, so the shared time domain is just the circuit's
simulation grid. The RC/diode lambda formulation handles their ODE dynamics independently.

### Phase 3a: SPICE CS amp reference (COMPLETE)

A pre-registered (W_n, W_p) sweep on the sky130 CS amp at L = 0.18 µm with
diode-connected PMOS load. Selection rule (max peak |gain| inside the 0.6-1.2 V
output-bias band, tie-break on lowest static current) was committed before the
sweep ran. Selected sizing W_n = 6.0 µm, W_p = 4.5 µm; reference traces (DC OP,
VTC, step response into 10 pF) archived under `docs/assets/cs_amp/`. Methodology
and figures in `docs/cs_amp.md`.

### Phase 3b: FNO-composed CS amp (COMPLETE)

KCL at the single internal node, scalar Newton-Raphson for DC OP, whole-window
implicit Newton-Raphson for transient with backward-Euler analytical 10 pF
load, autograd Jacobian through the trained NFET (Exp 19b) and PFET (Exp 06)
operators. Armijo backtracking damping, per-step voltage cap, rail clip.
Cold/warm wall-time and NR iteration counts emitted to `summary.json`.
Validation against the 3a SPICE reference: DC `|ΔV|/VDD ≤ 5%`, transient
Pearson `r > 0.997` and `max |ΔV| ≤ 30 mV`, NR outer loops `< 10`. Methodology
and metric caveats in `docs/composition.md`; end-to-end pipeline walkthrough
with mermaid diagrams in `docs/pipeline.md`.

Experiment 1 (2026-04-27, CPU baseline) confirms those gates at the selected bias/stimulus
(`|ΔV|/VDD = 0.92%`, transient Pearson `r = 0.99748`, max `|ΔV| = 25.79 mV`,
DC NR iters = 4, transient NR iters = 3), but also exposes two practical limits:

- Full rail-to-rail VTC has a low-`Vin` mismatch (`pearson_r = 0.9921`,
  `max_abs_error_v = 0.3829 V`) tied to weak-inversion / near-off behavior.
- Composition runtime is currently slower than SPICE on CPU
  (warm FNO ~25.9 s vs warm SPICE ~13.0 s for the timed DC+transient pair).
  Device-level speedups therefore do not yet imply composition-level speedup.

Experiment 2 (2026-04-27, CUDA path, same code and stimulus) reproduces essentially
the same composition accuracy (`|ΔV|/VDD = 0.92%`, transient Pearson `r = 0.99748`,
max `|ΔV| = 25.78 mV`, low-`Vin` VTC mismatch still present) while improving runtime
to warm FNO ~2.74 s vs warm SPICE ~13.42 s (~4.9x faster).

**Current takeaway:** speed claim is now supportable on GPU for this benchmark, but
accuracy limitations remain and are more likely tied to MOSFET operator fidelity in
weak-inversion / near-off regions than to NR solver convergence.

### Phase 3b-OTA: FNO-composed 5T OTA (COMPLETE)

Multi-node analog composition scale-up: three internal KCL nodes (n_tail, n_left,
n_out), differential input pair with floating source terminal, PFET current mirror,
NFET tail current source. Same KCL + autograd-Jacobian Newton machinery as the
CS amp; no structural changes to the solver. SPICE reference is a 7×7
(W_diff, W_mirror) sweep at L ∈ {0.40, 0.50} µm with selection rule locked
pre-sweep. Stimulus: ±50 mV differential step at Vcm = 0.9 V with C_load = 1 pF.

Composition outcomes (warm CUDA, production sizing W_diff = W_mirror = 8 µm,
W_tail = 2 µm):

| Gate | Criterion | L = 0.40 µm | L = 0.50 µm |
|---|---|---|---|
| Pearson r | ≥ 0.99 | **PASS** (0.9997) | **PASS** (0.9997) |
| max\|ΔV\| | ≤ 30 mV | **FAIL** (68.7 mV) | **FAIL** (68.9 mV) |
| Slew-rate rel. error | ≤ 10% | **PASS** (1.0%) | **PASS** (5.0%) |
| Slew-time rel. error | ≤ 10% | **FAIL** (16.3%) | **PASS** (8.5%) |
| NR iters (DC / transient) | ≤ 25 | **PASS** (5 / 11) | **PASS** (6 / 12) |

Pre-registered gate criteria; max\|ΔV\| and slew-time failures attributed via
Probe 1 attribution to **M4 PFET** in the Vsd → 0 triode regime — a training-data
coverage gap. PFET triode-boundary fine-tune (2026-05-15, see below) narrowed M4
peak \|ΔI\| by 22 % and composition max\|ΔV\| by 11 %; the gate at L = 0.40
remains open (61 mV vs 30 mV). The production PFET checkpoint is unchanged;
the fine-tune is documented as a partial-closure result. Full lab record in
`lab/circuit/ota_status.md`; documented in `docs/results.md` § 5T OTA.

### Phase 3c/3d: Digital circuits — documented regime boundary (CLOSED)

Phases 3c (single CMOS inverter) and 3d (inverter chain, N ∈ {1, 2, 4}) were
executed. SPICE converges for all chain depths. The FNO-composed whole-window
Newton solver does not converge to acceptance-quality digital transients.

Root cause is structural: the FNO's spectral temporal convolutions produce
off-diagonal autograd Jacobian terms that are spurious for a quasi-static MOSFET.
In digital saturation plateaus the physical `dI/dVds` diagonal collapses below
µA/V and those artifacts dominate Newton steps. DC convergence was stabilised by
tolerance-floor alignment; transient non-convergence persisted through all solver
tuning attempted. See `lab/circuit/inverter_chain_status.md` and
`lab/circuit/circuit_next_steps.md` for the full root-cause record and exploratory
future-work options. These phases are not a pending milestone.

The **scientific contribution remains the analog path**: differentiable CS
amplifier composition with SPICE-parity DC and transient, enabling gradient-based
device sizing and parameter sweeps without re-running SPICE. The digital result
is a responsible disclosure that maps where this FNO-composed formulation stops
being valid.

Before final write-up packaging, run a **3b performance hardening** pass:
GPU end-to-end composition path, runtime profiler split (forward/Jacobian/solve),
and matrix-free transient Newton/Krylov prototype to remove dense Jacobian cost.

---

## Infrastructure

- **Data:** PySpice/NGSPICE backend for ground truth generation
- **Training:** PyTorch + neuralop library, TensorBoard logging
- **Evaluation:** SPICE-validated I-V sweeps, comprehensive multi-geometry grids
- **Docs:** Documentation figures via export_docs_figures.py (light background)
- **CI:** black, pylint, pytest, mypy configured in pyproject.toml
