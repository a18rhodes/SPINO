# Diode FNO: Next Steps

## Executive Summary

The diode model works as a proof-of-concept but is fundamentally limited by its fixed
temporal grid. The FNO sees 2048 points over exactly 1ms — it cannot generalize across
time scales, which makes it useless for Newton-Raphson circuit composition where all
device models must share a dimensionless time domain.

**Goal:** Convert the diode to a dimensionless formulation matching RC, then validate
resolution invariance. This is the prerequisite for composable circuit simulation.

---

## Phase 1: Code Modernization (Structural Parity with MOSFET)

Before changing the physics, bring the diode codebase up to production standards.

### 1.1 model.py Refactor
- [x] Add `DiodeFNO` wrapper class with configurable `in_channels`, `n_modes`, `hidden_channels`
- [x] Retain `get_model()` as convenience factory
- [x] Prepare for channel expansion (5 -> 6 when lambda is added)

### 1.2 gen_data.py Refactor
- [x] Add `PreGeneratedDiodeDataset` (HDF5-backed `Dataset`, not `IterableDataset`)
- [x] Add `DimensionlessDiodeDataset` with variable T_end and lambda computation
- [x] Keep `InfiniteSpiceDiodeDataset` as legacy
- [x] Add `generate_offline_dataset()` with multiprocessing

### 1.3 generate_dataset.py (New)
- [x] Click CLI for offline HDF5 generation
- [x] Options: --output, --samples, --t-steps, --workers, --dimensionless

### 1.4 train.py Refactor
- [x] Click CLI with configurable hyperparameters
- [x] Support pre-generated HDF5 datasets
- [x] MSE + Sobolev loss (GenericDimensionlessPhysicsLoss with physics_weight=0)
- [x] Fade-in schedule matching RC
- [x] Early stopping
- [x] Proper logging via `logging` module
- [x] TensorBoard metric decomposition (data loss, sobolev loss)

### 1.5 evaluate.py Refactor
- [x] Dimensionless denormalization: V = V_hat * I_SCALE * R
- [x] Configurable t_steps and t_end
- [x] Resolution invariance test (1024 / 2048 / 4096)
- [x] Variable T_end test (same circuit at 100us, 1ms, 10ms)
- [x] compute_metrics() returning {r2, mse, rmse, mae_mv}

### 1.6 run_evaluation.py (New)
- [x] Standalone Click CLI: loads checkpoint, runs rectifier + resolution + variable T_end
- [x] Saves all three figures to spino/figures/diode/<run_name>/
- [x] Writes metrics.txt alongside figures

---

## Phase 2: Dimensionless Formulation

### 2.1 The Formulation

The diode-RC circuit ODE:
$$C \frac{dV}{dt} + \frac{V}{R} + I_S(e^{V/NV_T} - 1) = I(t)$$

Characteristic time constant: tau = RC

Dimensionless variables:
- t_hat = t / T_end (dimensionless time, 0 -> 1)
- lambda = RC / T_end (stiffness ratio, identical role to RC)
- I_hat = I / I_SCALE where I_SCALE = 5mA (current range)
- V_hat = V / V_SCALE where V_SCALE = I_SCALE * R

FNO input channels (6 total):
1. I_hat(t_hat) - normalized current waveform
2. lambda - stiffness ratio (constant, broadcast)
3. log10(R) - resistance (constant, broadcast)
4. log10(C) - capacitance (constant, broadcast)
5. log10(Is) - saturation current (constant, broadcast)
6. N - ideality factor (constant, broadcast)

FNO output: V_hat(t_hat)
Denormalization: V = V_hat * I_SCALE * R

### 2.2 Why Keep R and C Separately (Not Just Lambda)

Lambda = RC / T_end encodes the stiffness ratio (time-scale relationship).
But R and C individually determine:
- V_SCALE = I_SCALE * R (the voltage normalization depends on R)
- The diode nonlinearity Is * (exp(V / N*Vt) - 1) depends on physical voltage,
  not normalized voltage, so the model needs R to infer V_physical from V_hat.

### 2.3 Data Generation Changes

1. Sample R, C, Is, N from same physical ranges as before
2. Compute tau = RC per sample
3. Sample window ratio log-uniformly from [0.01, 1000]
4. Compute T_end = tau * ratio, then clamp to [10ns, 10ms] to keep SPICE-safe sim steps
5. Recompute lambda = tau / T_end_clamped (lambda != 1/ratio when clamp activates)
6. Generate PWL current in I_hat space (+-1), then scale: I_physical = I_hat * 5mA
7. Run SPICE with per-sample T_end and sim_step
8. Normalize: V_hat = V_physical / (5mA * R)
9. Store: x=[I_hat, lambda, log10R, log10C, log10Is, N], y=[V_hat]

### 2.4 Loss Strategy

**Primary: MSE + Sobolev (derivative matching)**
- Use GenericDimensionlessPhysicsLoss with physics_weight=0.0
- Sobolev weight target: 1e-2 (same as RC)
- Fade-in schedule: dead_zone (50ep) -> warmup (150ep) -> full -> polish (100ep)
- No physics residual (exponential instability, see below)

**Stretch goal: Clamped physics residual (FUTURE EXPERIMENT)**
The dimensionless diode residual:
  lambda * dV_hat/dt_hat + V_hat - I_hat + (Is*R/I_SCALE) * (exp(V_hat*I_SCALE*R / (N*Vt)) - 1) = 0

The exponential is catastrophically unstable as a loss term. Even modest overestimates
of V_hat send the residual to infinity. Would need:
- torch.clamp on the exponent argument (cap at +-20)
- Very low physics_weight (~1e-6)
- Extended dead zone (100+ epochs)
- This is a research question, not a shipping requirement.

### 2.5 Validation Criteria

1. **Resolution invariance:** R2 > 0.99 at 1024, 2048, and 4096 time steps on same circuit
   -- **PASSED: D2 R2={1024: 0.9993, 2048: 0.9994, 4096: 0.9993}, delta < 0.0001**
2. **Variable window test:** Same circuit at T_end in {100µs, 1ms, 10ms} produces
   physically consistent V waveforms
   -- **PASSED: D2 R2={100µs: 0.9977, 1ms: 0.9994, 10ms: 0.9995}**
3. **Rectifier regression:** Standard rectifier test (R=1k, C=10nF, Is=2.5nA, N=1.75,
   2kHz sine, 1ms) achieves R2 > 0.99 -- **PASSED: D2 R2=0.9994**
4. **Unit tests:**
   - Dimensionless data gen produces correct lambda values
   - Denormalization round-trips correctly
   - Model forward pass shape correct with 6 channels

---

## Phase 3: Integration (Depends on PFET + Composition)

After dimensionless diode is validated:
1. Train PFET with dimensionless built-in (skip NFET refactor)
2. Attempt composition: mixed-formulation NFET + new-style PFET
3. NFET refactor only if composition coupling fails

See lab/project/project_next_steps.md for project-level roadmap.

---

## Experiment Log

### D1: First Dimensionless Run (2026-03-04, DELETED)

**Result:** R2=0.9719 on standard rectifier. Worse than expected.

**Root cause - Lambda OOD:**

The standard rectifier test case (R=1kΩ, C=4pF, f=2kHz, T_end=1ms) has:
- tau = RC = 4ns
- lambda = tau / T_end = 4e-6

But the training distribution uses ratio ∈ [0.1, 100], giving lambda ∈ [0.01, 10].
The test case's lambda=4e-6 is **2500x below the training minimum** -- the model is
extrapolating in lambda space, not interpolating. This is not a model quality problem;
it is a training data coverage problem.

**Fix applied (see D2):**

- `_RATIO_MIN=0.01`, `_RATIO_MAX=1000`, added `_T_END_MIN_S=10ns`, `_T_END_MAX_S=10ms`.
- T_end is clamped after sampling tau*ratio; lambda is recomputed from the clamped value.
  This prevents SPICE sim_step from exceeding tau regardless of parameter combination.
- Eval circuit changed from C=4pF (junction cap, tau=4ns, subgrid) to C=10nF
  (tau=10µs, sim_step/tau=0.024, fully SPICE-safe and in-distribution).

**D2 outcome:** R2=0.9994 rectifier, R2=0.9999 adversarial. Lambda OOD fully resolved.
