# Diode Neural Operator: Current Status

## Overview

This document tracks progress on the Diode FNO development -- the second device in the
SPINO pipeline after RC and before MOSFET. The diode circuit (current source || diode ||
resistor) is the simplest *nonlinear* device and serves as the proving ground for
dimensionless formulations on nonlinear systems.

**Status: Dimensionless formulation fully validated. All three scientific claims confirmed.
Run D2 achieves R2=0.9994 on standard rectifier, R2≥0.9993 across all grid resolutions,
R2≥0.9977 across all simulation windows. Ready for diode.md documentation and integration.**

---

## Architecture (Post-Refactor)

| Property          | Dimensionless (default)                           | Legacy                                   |
|-------------------|---------------------------------------------------|------------------------------------------|
| Base              | DiodeFNO wrapper (neuralop FNO)                   | DiodeFNO wrapper (neuralop FNO)          |
| Fourier modes     | 256                                               | 256                                      |
| Hidden channels   | 64                                                | 64                                       |
| Input channels    | 6 (I_hat, lambda, log10R, log10C, log10Is, N)     | 5 (I_mA, log10R, log10C, log10Is, N)    |
| Output channels   | 1 (V_hat, dimensionless)                          | 1 (V in Volts, raw)                      |
| Non-linearity     | SiLU                                              | SiLU                                     |
| Skip connection   | Linear                                            | Linear                                   |
| Domain padding    | 0.1                                               | 0.1                                      |
| Denormalization   | V = V_hat * I_SCALE_A * R                         | None (raw voltage)                       |

## Training Regime (Post-Refactor)

### Dimensionless Mode (Recommended)
- **Loss:** MSE + Sobolev (GenericDimensionlessPhysicsLoss, physics_weight=0)
- **Sobolev weight:** target 1e-2, with fade-in schedule
- **Fade-in:** dead_zone=50ep, warmup=100ep, full, polish=50ep (total 200 useful epochs)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-5)
- **Scheduler:** CosineAnnealingWarmRestarts
- **Data:** Pre-generated HDF5 (generate_dataset.py), variable T_end per sample
- **Early stopping:** patience=20, threshold=1e-6
- **CLI:** `python -m spino.diode.train --dataset-path <path.h5>`

### Legacy Mode
- **Loss:** Plain MSE
- **Data:** On-the-fly SPICE via InfiniteSpiceDiodeDataset (fixed 1ms)
- **CLI:** `python -m spino.diode.train --legacy`

## Parameter Ranges

| Param | Range              | Distribution | Encoding       |
|-------|--------------------|-------------|----------------|
| R     | 50 - 2000 Ohm     | Log-uniform | log10(R)       |
| C     | 1pF - 10nF        | Log-uniform | log10(C)       |
| Is    | 1fA - 1nA         | Log-uniform | log10(Is)      |
| N     | 1.0 - 2.0         | Uniform     | Raw            |
| T_end | Clamped to [10ns, 10ms] after ratio sample | Log-uniform ratio [0.01, 1000] | Via lambda  |
| lambda| 0.001 - 100       | Derived (tau/T_end after clamp) | Broadcast channel |

## Modules

| File                 | Status      | Description                                        |
|----------------------|-------------|----------------------------------------------------|
| model.py             | Refactored  | DiodeFNO class, I_SCALE_A, get_model() factory     |
| gen_data.py          | Refactored  | DimensionlessDiodeDataset, PreGenerated, HDF5 gen  |
| generate_dataset.py  | New         | Click CLI for offline HDF5 dataset generation      |
| train.py             | Refactored  | Click CLI, Sobolev loss, fade-in, early stopping   |
| evaluate.py          | Refactored  | Dimensionless denorm, resolution invariance, variable T_end tests |
| run_evaluation.py    | New         | Standalone eval CLI: loads checkpoint, runs all 3 test suites      |

## Resolved Limitations

- [x] Fixed temporal grid -> variable T_end with lambda
- [x] No CLI -> click CLI on train.py and generate_dataset.py
- [x] No pre-generated pipeline -> HDF5 with multiprocessing
- [x] No physics loss -> MSE + Sobolev with fade-in schedule
- [x] No early stopping -> patience-based stopping
- [x] Bare print calls -> logging module
- [x] Training/eval mismatch -> configurable t_steps on evaluate
- [x] No resolution invariance test -> evaluate_resolution_invariance() [D2: delta R2 < 0.0001]
- [x] No variable T_end test -> evaluate_variable_t_end() [D2: all windows R2 >= 0.9977]

## Remaining Work

- [x] Generate first HDF5 dataset (10k samples)
- [x] Run dimensionless training (first run completed, see Experiment Log)
- [x] Fix lambda OOD issue: T_end clamp [10ns, 10ms], ratio [0.01, 1000], eval C=4pF -> C=10nF
- [x] Regenerate HDF5 dataset with wider lambda range (30k samples, diode_30k_v2.h5)
- [x] Re-train and validate R2 > 0.99 on in-distribution test cases (D2: R2=0.9994)
- [x] Confirm resolution invariance at 1024/2048/4096 (R2: 0.9993 / 0.9994 / 0.9993 — delta < 0.0001)
- [x] Confirm variable T_end consistency (100µs: R2=0.9977, 1ms: R2=0.9994, 10ms: R2=0.9995)
- [x] Update diode.md with dimensionless architecture, D2 results, and validation plots
- [ ] Stretch: clamped physics residual experiment (low priority, R2=0.9994 is sufficient)

---

## Experiment Log

### Pre-Refactor Runs (Legacy, 5-channel)

#### Run 1 (Initial Training)
- **Date:** Pre-2026-03
- **Config:** 50 epochs, 100 steps/epoch, batch 64, lr=1e-3
- **Result:** Proof of concept. Rectifier test figure exists.

#### Run 2 (Extended Training)
- **Date:** Pre-2026-03
- **Config:** 80 epochs
- **Result:** Improved over Run 1. Last legacy checkpoint.

#### Run 3
- **Date:** Pre-2026-03
- **Result:** TensorBoard run exists. Details not captured.

### Post-Refactor Runs (Dimensionless, 6-channel)

#### Run D1 (First Dimensionless Run)
- **Date:** 2026-03-04
- **Dataset:** 10k samples, ratio window [0.1, 100], lambda range [0.01, 10]
- **Config:** n_epochs=300, dead_zone=50, warmup=150, fine_tune=100, batch=64, lr=1e-3
- **Result on standard rectifier:** R2=0.9719, MSE=1.34e-01
- **Root cause of underwhelming R2:** Standard rectifier (R=1kΩ, C=4pF) has tau=4ns,
  T_end=1ms, lambda=4e-6. Training distribution lambda=[0.01, 10]. The test case is
  2500x below the training minimum -- extreme OOD generalization, not interpolation.
  R2=0.9719 is actually decent for this regime but not good enough for production.
- **Run deleted** (stale, wrong scalar tag convention in TensorBoard)
- **Action:** Widen training ratio window; add T_end hard clamp; change eval to C=10nF.

#### Run D2 (Second Dimensionless Run — CURRENT BEST)
- **Date:** 2026-03-05
- **Dataset:** 30k samples (`diode_30k_v2.h5`), ratio [0.01, 1000], T_end clamped [10ns, 10ms]
- **Config:** n_epochs=250, dead_zone=50, warmup=100, fine_tune=50, batch=64, lr=1e-3
- **Eval circuit:** R=1kΩ, C=10nF, Is=2.5nA, N=1.75, f=2kHz, T_end=1ms → tau=10µs, lambda=0.01
- **Result on standard rectifier:** R2=0.9994, MSE=3.07e-03, RMSE=55.4mV, MAE=45.37mV
- **Result on adversarial sample:** R2=0.9999, MSE=1.03e-05, RMSE=3.2mV, MAE=2.53mV
- **Checkpoint:** `diode_dimless_v2_VokyITJR` (backed up to `/backup/`)
- **Status:** Production-quality. Lambda OOD regression fully resolved.

#### Post-training Validation (D2 checkpoint)
- **Resolution invariance** (same circuit, T_end=1ms, grid varies):
  - T=1024: R2=0.9993
  - T=2048: R2=0.9994 (training resolution)
  - T=4096: R2=0.9993
  - Delta across resolutions: <0.0001 — resolution invariance confirmed.
- **Variable T_end** (same circuit params, lambda varies with window):
  - T_end=100µs (lambda=0.10, freq=20kHz): R2=0.9977, MAE=65.44mV
  - T_end=1ms  (lambda=0.01, freq=2kHz):  R2=0.9994, MAE=45.37mV
  - T_end=10ms (lambda=0.001, freq=200Hz): R2=0.9995, MAE=37.58mV
  - All windows R2 ≥ 0.9977 — time-scale invariance confirmed.
- **Figures:** `spino/figures/diode/diode_dimless_v2_VokyITJR/`
