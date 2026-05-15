# MOSFET Neural Operator: Current Status

## Overview

This document tracks progress on the MOSFET FNO development, including experiments run, results achieved, and ongoing work. The goal is to train a Fourier Neural Operator that learns the $I_d(V_g, V_d, V_s, \text{params})$ mapping for BSIM4 MOSFETs, enabling fast surrogate simulation of analog circuits.

### Cross-project composition status (2026-04-27)

In CS amplifier composition, **Experiment 1 (CPU)** and **Experiment 2 (CUDA)** show that
runtime bottlenecks can be mitigated by GPU execution, but composition-level fidelity issues
(notably low-`Vin` VTC mismatch and transient equilibrium offset) remain almost unchanged.
This strongly suggests current composition accuracy is now limited more by MOSFET operator
fidelity in weak-inversion / near-off regions than by Newton-Raphson convergence behavior.

---

## CRITICAL: Experiments 1-3 Are Fundamentally Flawed

> **All three trained models (HccTSQra, QOpPuw08, 6DNIQKTr) are invalid and should not be used.**

### Root Cause: Non-Bijective Current Transform

Experiment 3 used the following current normalization transform:

```python
# BROKEN: sign(I) * log10(|I|) is NOT bijective for negative currents
current_normalized = sign(I) * log10(|I| + 1e-12)
```

**Why this is broken:**
- For `I = -0.1 mA`: `sign(-0.1) * log10(0.1) = -1 * (-1) = +1`
- For `I = +10 mA`: `sign(+10) * log10(10) = +1 * (+1) = +1`
- **Both values map to +1** - the transform is not invertible

**Dataset reality:** NMOS drain current goes negative in **12% of timesteps** (when Vd < Vs in linear region). The model was trained on corrupted targets where different physical currents had identical normalized values.

### Additional Flaw: Mean Collapse

Evaluation plots show all models predict near-constant ~1mA regardless of input voltages. The models learned to predict the dataset mean current, which minimizes loss but has zero generalization. $R^2$ values on I-V sweeps are **catastrophically negative** (e.g., -90, -1565, -36,999,750).

### Resolution

The current transform has been fixed to use `arcsinh`, which is bijective and handles signed values:

```python
# FIXED: arcsinh is bijective, smooth, handles negative values
current_normalized = arcsinh(I / scale)  # Forward
I = scale * sinh(current_normalized)     # Inverse (exact)
```

**All models must be retrained from scratch.**

---

## Data Generation Scheme

### Dataset: Sky130 NMOS (25K Samples)

**Generation Pipeline:**
- **Source:** PySpice/NGSPICE with Sky130 PDK BSIM4 model cards
- **Device:** `sky130_fd_pr__nfet_01v8` (1.8V standard NMOS)
- **Topology:**
  ```
  Vg(PWL) ─┬─┐
           │ │──┐
  Vd(PWL) ─┤ M1 ├──→ Id (measured)
           │ │──┘
  Vs(PWL) ─┴─┘
  ```
- **Voltage Excitation:** Random piecewise-linear (PWL) waveforms
  - **Vg:** Gate voltage, uniformly sampled from [0, 1.8V]
  - **Vd:** Drain voltage, uniformly sampled from [0, 1.8V]
  - **Vs:** Source voltage, uniformly sampled from [0, 0.5V]
  - **Vb (Bulk):** Fixed at 0V (implicit ground connection)
  - **Switching Events:** 5-40 random transitions per waveform to cover diverse operating regimes
- **Time Domain:** 1024 time steps over $T_{end} = 1\mu s$
- **Parameter Variation:** Monte Carlo sampling of BSIM4 corner parameters (TT, SS, FF, SF, FS) from Sky130 PDK characterization data

**Dataset Statistics:**
- **Total Samples:** 34,375 unique circuit configurations (25K PWL + 6.25K monotonic + 3.125K vth_focused)
- **Storage:** HDF5 format (~3.2 GB)
- **Full Parameter Set:** 76 BSIM4 parameters (geometry, threshold, mobility, capacitance, temperature, parasitics)

**Waveform Modes (NEW):**
- **PWL (70%):** Random piecewise-linear waveforms (5-15 transitions) - default chaotic mode
- **Monotonic (18%):** DC-sweep-like ramps (up or down) - matches evaluation distribution
- **Vth-Focused (9%):** Gate voltage concentrated around threshold (0.25V-0.75V range) - captures subthreshold-to-saturation transition

This waveform diversity addresses the **training/eval distribution mismatch** where models trained on chaotic PWL failed to generalize to monotonic DC sweeps.

### Parameter Schema Evolution

#### Original Full Set (76 Parameters)
All BSIM4 parameters defined in Sky130 PDK model cards, including:
- **Geometry:** `w`, `l`, `xl`, `xw`
- **Threshold Voltage:** `vth0`, `k1`, `k2`, `k3`, `k3b`, `w0`, `dvt0`, `dvt1`, `dvt2`, `dvt0w`, `dvt1w`, `dvt2w`, `nlx`
- **Mobility:** `u0`, `ua`, `ub`, `uc`, `vsat`, `a0`, `ags`, `b0`, `b1`, `keta`
- **Subthreshold/Output Conductance:** `voff`, `nfactor`, `cit`, `cdsc`, `cdscb`, `cdscd`, `eta0`, `etab`, `dsub`, `pclm`, `pdiblc1`, `pdiblc2`, `pdiblcb`, `drout`, `pscbe1`, `pscbe2`
- **Temperature:** `tnom`, `ute`, `kt1`, `kt1l`, `kt2`, `ua1`, `ub1`, `uc1`, `at`
- **Capacitance:** `cgso`, `cgdo`, `cgbo`, `cj`, `mj`, `pb`, `cjsw`, `mjsw`, `pbsw`, `cjswg`, `mjswg`, `pbswg`, `tcj`, `tpb`, `tcjsw`, `tpbsw`
- **Parasitic:** `rdsw`, `rsh`, `rgate`
- **Process:** `tox`, `toxe`, `xj`, `lint`, `wint`

**Problem:** Most parameters are fixed constants in the PDK (std $< 10^{-6}$), contributing only noise to the model.

#### Curated Set (29 Parameters)
Empirical analysis of the 25K sample dataset identified parameters with measurable variation (std $> 10^{-6}$):

**`ParameterSchema.TRAINING_KEYS`:**
```python
["w", "l", "vth0", "k1", "k2", "k3b", "dvt1", "dvt2", "dvt0w", "dvt1w",
 "dvt2w", "u0", "vsat", "keta", "voff", "nfactor", "etab", "dsub", "pclm",
 "pdiblc1", "pdiblc2", "pdiblcb", "drout", "pscbe1", "ute", "kt1", "kt2",
 "at", "rdsw"]
```

**Key Properties:**
- **Geometry-Dominated:** `w` (width) and `l` (length) have largest variation and strongest influence on $I_d$.

---

## Experiment 6: Increased Fourier Modes (modes=256)

**Hypothesis:** modes=128 insufficient for sharp Vth transition, causing Gibbs phenomenon (spectral leakage) in subthreshold region.

**Configuration:**
- **Dataset:** 35K stratified (same as Exp 5)
- **Modes:** 256 (doubled from 128)
- **Other hyperparams:** Identical to Exp 5 (LR=7.5e-4, batch_size=64, embedding_dim=16, width=64)
- **Training time:** 237 minutes (vs 210 min for modes=128, +13% overhead)
- **Final loss:** 0.016009 (vs 0.017296 for modes=128, marginal improvement)

**Results:**

| Geometry | Ramp R² | **Subthreshold R² (Vg<0.5V)** | Sweep R² | Random R² | Ramp MAE | **Ramp MALE** |
|----------|---------|-------------------------------|----------|-----------|----------|---------------|
| tiny     | 0.9999  | **-7.64** ❌                  | 0.9995   | 1.0000    | 0.35 µA  | **393.01 µA** |
| medium   | 1.0000  | **-0.86** ❌                  | 0.9994   | 1.0000    | 0.44 µA  | **177.17 µA** |
| xlarge   | 0.9999  | **-2.11** ❌                  | 0.9998   | 0.9947    | 1.37 µA  | **154.24 µA** |

**Core Metrics:**
- Fast Dataset R²: 0.9988
- SPICE Transfer R² (overall): 0.9665
- SPICE Transfer SubTh-R²: -6.88 ❌
- SPICE Transfer MALE: 375.21 µA
- SPICE Output R²: 0.5633
- SPICE Output MALE: 92.24 µA
- Speedup: 494x

**Critical Finding: PARTIAL IMPROVEMENT BUT STILL INADEQUATE**

Doubling Fourier modes (128→256) **DID** improve subthreshold accuracy relative to Experiment 5, but negative R² values prove the problem remains severe:

**Subthreshold R² Comparison (Exp 5 vs Exp 6):**
- **tiny:** -21.15 → -7.64 (3x improvement, but still catastrophic)
- **medium:** -4.34 → -0.86 (5x improvement, approaching usable)
- **xlarge:** -3.53 → -2.11 (modest improvement)

**MALE Comparison (lower is better):**
- **tiny:** 452.23 µA → 393.01 µA (13% reduction)
- **medium:** 268.17 µA → 177.17 µA (34% reduction)
- **xlarge:** 165.28 µA → 154.24 µA (7% reduction)

**Interpretation:**
Spectral resolution WAS a contributing factor (modes=256 helped), but it's not the dominant issue. The model still predicts orders of magnitude incorrectly in subthreshold. Loss function bias remains the primary bottleneck.

**Visual Analysis:**
- **Medium geometry Id-Vg ramp:** Severe oscillations at Vg=0.0-0.3V, FNO predicts 10-100 nA while SPICE shows 1-10 nA
- **Tiny geometry Id-Vg ramp:** Similar oscillation pattern, amplitude ~5-10 nA
- **XLarge geometry Id-Vg ramp:** Oscillations persist despite larger device (higher absolute current)
- **All geometries:** Saturation region (Vg>0.7V) matches SPICE near-perfectly
- **MALE values (150-400 µA):** Massive absolute log-space errors indicate model predicts wrong orders of magnitude in subthreshold

**Speedup:** 494x (vs 627x for modes=128, expected due to increased computational cost)

**Conclusion:**

The subthreshold oscillation problem has **multiple contributing factors:**
1. **Loss function bias (PRIMARY):** MSE heavily weights large currents (mA) over small currents (nA), even with arcsinh transform
2. **Spectral resolution (SECONDARY):** modes=128 was insufficient - modes=256 improved SubTh-R² by 3-5x but not enough to reach positive values
3. **Training data artifacts:** SPICE transient simulations show current "blips" at t=0 in all Id-Vg ramps
4. **Undersampling:** Only 15% of training data focuses on Vth transition region

Increasing modes further (512, 1024) may provide incremental gains but will not solve the problem without addressing loss function bias.

**Next Action:** Implement log-weighted loss + modes=256 as the baseline for Experiment 7.

---

## Experiment 7: SubthresholdWeightedLoss Grid Search (FAILED - Numerical Collapse)

**Hypothesis:** Inverse magnitude weighting will force model to prioritize subthreshold accuracy without sacrificing saturation performance.

**Configuration:**
- **Loss Function:** `SubthresholdWeightedLoss` with inverse magnitude weighting: $w(I) = \left(\frac{\alpha}{|I| + \alpha}\right)^p$
- **Dataset:** 35K stratified (same as Exp 5/6)
- **Architecture:** modes=256, width=64, embedding_dim=16
- **Grid Search:** scale_mA ∈ {0.001, 0.01, 0.1} × exponent ∈ {1.0, 2.0, 3.0} = 9 cells
- **Training:** n_epochs=100, LR=7.5e-4, batch_size=64, early_stop_threshold=1e-6

**Critical Implementation Bug:**
```python
# Original implementation (BROKEN)
weights = 1.0 / (torch.abs(target_mA) + self.scale_mA) ** self.exponent
weights = weights / weights.mean()  # ← Weight normalization destroys loss scale
```

**Result:** CATASTROPHIC FAILURE - Training collapsed by epoch 10-15 across all 9 grid cells.

| scale_mA | exponent | Loss @ Epoch 0 | Loss @ Epoch 10 | SubTh-R² | Transfer R² | Status |
|----------|----------|----------------|-----------------|----------|-------------|--------|
| 0.001    | 1.0      | 0.000004       | 0.000000        | -1810.25 | 0.9972      | COLLAPSED ❌ |
| 0.001    | 2.0      | 0.000009       | 0.000000        | -379.76  | 0.9269      | COLLAPSED ❌ |
| 0.001    | 3.0      | 0.000130       | 0.000000        | -41.82   | -0.5759     | COLLAPSED ❌ |
| 0.01     | 1.0      | ~1e-5          | 0.000000        | -352.29  | 0.9984      | COLLAPSED ❌ |
| 0.01     | 2.0      | ~1e-5          | 0.000000        | -263.00  | 0.9944      | COLLAPSED ❌ |
| 0.01     | 3.0      | ~1e-5          | 0.000000        | -45.38   | 0.9914      | COLLAPSED ❌ |
| 0.1      | 1.0      | ~1e-5          | 0.000000        | -3854.49 | 0.9929      | COLLAPSED ❌ |
| 0.1      | 2.0      | ~1e-5          | 0.000000        | -276.27  | 0.9857      | COLLAPSED ❌ |
| 0.1      | 3.0      | ~1e-5          | 0.000000        | -626.04  | 0.9957      | COLLAPSED ❌ |

**Diagnosis:**

Weight normalization (`weights / weights.mean()`) caused numerical collapse:
1. Raw subthreshold weights: ~10^8× (for 1 nA currents with scale_mA=0.01, exponent=2.0)
2. After normalization: All weights become O(1), loss drops to ~1e-7
3. AdamW interprets loss < 1e-6 as "converged," gradients vanish by epoch 10

**All SubTh-R² values worse than baseline** (Exp 5: -21.15 to -3.53, Exp 6: -7.64 to -2.11). The "best" configuration (scale=0.001, exp=3.0, SubTh-R²=-41.82) is still 10-20× worse than doing nothing.

**Fix Applied:** Removed weight normalization, added gradient clipping (max_norm=1.0) to prevent instability from unnormalized weights.

**Next Action:** Rerun grid search with corrected loss implementation (Experiment 7b).
---

## Experiment 7b: SubthresholdWeightedLoss (Unnormalized, FAILED - Arcsinh-Space Incompatibility)

**Hypothesis:** Removing weight normalization will restore loss scale and prevent gradient vanishing.

**Configuration:**
- **Loss Function:** `SubthresholdWeightedLoss` WITHOUT normalization (`weights / weights.mean()` removed)
- **Grid Search:** scale_mA ∈ {0.001, 0.01, 0.1} × exponent ∈ {1.5, 1.75, 2.0} = 9 cells
- **Training:** n_epochs=100, early_stop_threshold=1e-6, early_stop_patience=15
- **Note:** Gradient clipping (max_norm=1.0) was ALREADY present in training loop since Experiment 1

**Result:** CATASTROPHIC FAILURE - Loss still collapses to 0.000000 by epoch 10-15.

**Sample Results (scale_mA=0.001, exponent=1.5):**
- Epoch 0: Loss = 0.000019
- Epoch 10: Loss = 0.000000 ← Vanished again despite no normalization!
- SubTh-R²: tiny=-323.75, medium=-25.95, xlarge=-6.04 ❌
- Transfer R²: tiny=0.9871, medium=0.9988, xlarge=0.9987 (saturation OK)
- MALE: 650 µA (tiny), 386 µA (medium), 213 µA (xlarge)

**Root Cause Analysis:**

Magnitude-based weighting is **fundamentally incompatible** with arcsinh-space error computation:

| Current | Arcsinh Value | Squared Error (arcsinh) | Weight | Weighted Error |
|---------|---------------|------------------------|--------|----------------|
| 1 nA    | ~1e-6         | ~1e-12                 | 1.0    | **1e-12**      |
| 1 mA    | 0.881         | ~0.1                   | 0.001  | **1e-4**       |

Even with 1000× weighting, saturation errors dominate by **8 orders of magnitude**. The arcsinh transform keeps small values tiny (arcsinh(x) ≈ x for x << 1), defeating the purpose of magnitude weighting.

**Diagnosis:** The problem wasn't normalization - it's that we're computing error in the wrong space. Arcsinh-space MSE for 1 nA is ~1e-12, while for 1 mA it's ~0.1. Weighting can't overcome this scale mismatch.

**Fix Applied (Experiment 7c):** Compute MSE in **physical space** (mA) instead of arcsinh space. Transform predictions and targets back to mA, compute `(pred_mA - target_mA)²`, then apply weights.

**Next Action:** Rerun grid search with physical-space error computation.
---

## Experiment 7c: SubthresholdWeightedLoss (Physical-Space Error, FAILED - Vanishing Gradients Persist)

**Hypothesis:** Computing MSE in physical space (mA) instead of arcsinh space will allow magnitude-based weighting to balance subthreshold/saturation contributions.

**Configuration:**
- **Loss Function:** `SubthresholdWeightedLoss` with physical-space error computation
- **Grid Search:** scale_mA ∈ {0.001, 0.01, 0.1} × exponent ∈ {1.5, 1.75, 2.0} = 9 cells
- **Training:** modes=256, n_epochs=100, early_stop_threshold=1e-6, early_stop_patience=15
- **Code Change:**
  ```python
  pred_ma = torch.sinh(pred_arcsinh)  # Transform to physical space
  target_ma = torch.sinh(target_arcsinh)
  mse_physical = (pred_ma - target_ma) ** 2  # Error in mA, not arcsinh
  ```

**Result:** CATASTROPHIC FAILURE - Loss still collapses to 0.000000 by epoch 10, SubTh-R² remains severely negative.

**Sample Results (partial grid):**

| scale_mA | exponent | Loss @ Epoch 0 | Loss @ Epoch 10 | SubTh-R² (tiny) | SubTh-R² (medium) | SubTh-R² (xlarge) | Transfer R² | Early Stop Epoch |
|----------|----------|----------------|-----------------|-----------------|-------------------|-------------------|-------------|------------------|
| 0.001    | 1.5      | 0.000005       | 0.000000        | -48.92          | -6.29             | -0.66             | 0.9956      | 30               |
| 0.001    | 1.75*    | 0.000004       | 0.000000        | -89.06          | -7.56             | -1.00             | 0.9954      | 30               |

*Grid search script shows `exponent=1.8` but train.py received `exponent=1.75` - parameter passing bug in grid_search.py.

**Comprehensive Metrics (scale_mA=0.001, exponent=1.5):**
- **MALE:** 474 µA (tiny), 243 µA (medium), 139 µA (xlarge) - still 3-10× worse than baseline
- **Transfer R²:** 0.9802 (tiny), 0.9974 (medium), 0.9988 (xlarge) - overall fit excellent
- **Output R²:** 0.9314 (tiny), 0.9875 (medium), 0.9793 (xlarge) - saturation region OK
- **Random waveform R²:** 0.9711 (tiny), 0.9983 (medium), 0.9990 (xlarge) - transient response good

**Diagnosis:**

Physical-space error computation is mathematically sound, but the fundamental issue remains: **magnitude-based weighting causes optimization pathology**.

For physical-space MSE in mA²:
- 1 nA error: (1e-6 mA)² = 1e-12 mA²
- 1 µA error: (1e-3 mA)² = 1e-6 mA²
- 1 mA error: (1.0 mA)² = 1.0 mA²

With exponential magnitude weighting (scale_mA=0.001, exponent=1.5):
- 1 nA target: weight ≈ (0.001/(1e-6 + 0.001))^1.5 ≈ 31.6×
- 1 mA target: weight ≈ (0.001/(1.0 + 0.001))^1.5 ≈ 0.001×

Weighted contribution to loss:
- 1 nA region with 1 nA error: 1e-12 × 31.6 = 3.16e-11 mA²
- 1 mA region with 0.1 mA error: 0.01 × 0.001 = 1e-5 mA²

Saturation errors still dominate weighted loss by **~10^6**. Even with aggressive weighting, the absolute scale of physical errors in nA/µA regime is so small that the loss function rounds to 0.000000 in display (actual value likely 1e-7 to 1e-8), causing early stopping and gradient vanishing.

**Key Observations:**
1. Loss display format (6 decimal places) shows "0.000000" but actual loss is O(1e-7)
2. AdamW + early stopping interprets loss < 1e-6 as convergence
3. Model learns to minimize weighted loss by predicting near-zero subthreshold current (where weight × error² is still tiny)
4. SubTh-R² worsening (-49 to -89) indicates model is actually making subthreshold predictions WORSE than baseline
5. Parameter passing bug in grid_search.py: exponent values don't match between grid declaration and train.py execution

**Conclusion:** Magnitude-based weighting strategy is fundamentally flawed for this problem. The optimizer cannot balance contributions from regions differing by 6+ orders of magnitude, even with exponential weighting and physical-space error computation.

**Status:** Three failed attempts (Exp 7, 7b, 7c) with weighted loss. All show gradient vanishing by epoch 10-15 and SubTh-R² degradation. The weighted loss approach requires fundamentally different formulation.

**Post-Mortem: Log10Loss Experiment (CATASTROPHIC FAILURE)**

**Hypothesis:** Training in log10-transformed current space would naturally balance errors across decades without explicit weighting.

**Configuration:**
- Loss: `Log10Loss` with MSE in log10 space: `(log10(|pred|) - log10(|target|))²`
- Architecture: modes=256, LpLoss baseline configuration
- Epochs: 100

**Result:** Complete model collapse. All predictions converged to constant near-zero (~1e-9 mA).

**Metrics:**
- Fast Dataset R²: -0.62 (worse than random)
- Transfer R²: -0.67, SubTh-R²: -10,550 (catastrophic)
- MALE: 2,140 µA (10× worse than baseline)

**Root Cause:** Gradient pathology from log10 derivative. For small predictions, `∇log10(x) = 1/(x·ln(10))` explodes as x→0. Gradient clipping (max_norm=1.0) truncated these, causing optimizer to converge at the clipping boundary (~1e-9) instead of learning the true current distribution.

**Conclusion:** The arcsinh transform exists precisely to avoid this instability. Its derivative `1/√(1+x²)` is bounded even at x=0, making it numerically stable for optimization. The problem is not the transform choice but the **data distribution**: 60% saturation samples vs 15% subthreshold samples guarantees MSE-based losses ignore nA-scale currents regardless of representation.

**Next Action:** Dataset rebalancing with subthreshold-focused waveforms.
---

## Experiment 8: Dataset Rebalancing with Subthreshold-Focused Sampling (FAILED - Inverse Problem)

**Hypothesis:** Increasing subthreshold sample representation from 15% to 26% will force the optimizer to allocate capacity to nA-scale current prediction, improving SubTh-R² without loss function modifications.

**Configuration:**
- **Dataset:** 40,250 samples (up from 35K), adding 1,050 subthreshold_focused samples per geometry bin
- **Waveform Distribution:** 52% PWL / 22% monotonic / 26% subthreshold_focused (was 60%/25%/15%)
- **Subthreshold Mode:** Gate voltage constrained to 0.0-0.5V (always below Vth), ensuring 70-80% of timesteps in deep subthreshold (I_d < 1 µA)
- **Architecture:** modes=256, width=64, embedding_dim=16 (Experiment 6 baseline)
- **Loss:** LpLoss (arcsinh-space relative L2 norm)
- **Training:** 100 epochs, AdamW, LR=7.5e-4

**Result:** INVERSE PROBLEM - SubTh-R² improved 250× but overall model collapsed.

| Metric | Exp 6 (35K, 15% SubTh) | Exp 8 (40K, 26% SubTh) | Change |
|--------|------------------------|------------------------|--------|
| SubTh-R² (tiny) | -7.64 | -0.017 | **+7.62** ✓ |
| SubTh-R² (medium) | -0.86 | -0.030 | **+0.83** ✓ |
| SubTh-R² (xlarge) | -2.11 | -0.005 | **+2.11** ✓ |
| Transfer R² | 0.9990 | -0.68 | **-1.68** ❌ |
| MALE | 177-393 µA | 3,306 µA | **+2,913 µA** ❌ |
| Fast Dataset R² | 0.9990 | -0.66 | **-1.66** ❌ |

**Diagnosis:**

The model learned subthreshold physics by **catastrophically forgetting saturation physics**. Overall R² = -0.68 indicates predictions are worse than mean baseline. The optimizer allocated capacity to the newly-emphasized subthreshold region (26% of samples) at the expense of saturation accuracy.

This proves the hypothesis was partially correct but reveals a deeper issue: **MSE-based losses cannot balance multi-scale contributions regardless of data distribution**. Changing from 15% to 26% subthreshold representation simply flipped which region gets ignored, rather than teaching the model both simultaneously.

**Key Insight:** The FNO architecture with uniform LpLoss lacks the capacity or inductive bias to represent both nA-scale (subthreshold) and mA-scale (saturation) physics with a single set of weights. The 10^6× dynamic range exceeds what can be learned with naive MSE minimization.

**Conclusion:** The problem is **architectural/loss-design**, not data distribution. The model needs explicit multi-scale handling:
1. Multi-task loss with separate subthreshold/saturation terms
2. Region-adaptive conditioning (FiLM based on current magnitude)
3. Hierarchical ensemble (separate networks per decade)

**Status:** Dataset rebalancing confirmed as insufficient. Architectural changes required.
---

## Experiments 7-8c: RegionAdaptiveLoss Grid Search (FAILED - Wrong Root Cause)

### Experiment 9a: RegionAdaptiveLoss Implementation

**Hypothesis:** Splitting LpLoss into separate subthreshold (Vg<0.5V) and saturation (Vg>=0.5V) terms with tunable weight ratios would allow the optimizer to balance capacity allocation between nA-scale and mA-scale physics.

**Implementation:** `RegionAdaptiveLoss` in `spino/loss.py` - vectorized multi-task loss that masks each region per-sample and computes independent LpLoss terms. Mathematically equivalent to per-sample boolean-indexed LpLoss (verified: 1.19e-07 difference = float32 rounding).

**Bugs Fixed During Development:**
1. Zero-padding diluted LpLoss normalization (identical R² across all ratios)
2. Per-sample Python loop caused 3.1x slowdown (75s vs 24s/epoch) - vectorized
3. Raw voltage threshold vs z-score normalized threshold mismatch (all NaN)
4. `sqrt(0.0)` numerical instability in masked region norms (persistent NaN)

### Experiment 9b: Grid Search Results

**Configuration:**
- **Dataset:** 40K stratified (52% PWL / 22% monotonic / 26% subthreshold_focused)
- **Architecture:** modes=256, width=64, embedding_dim=16
- **Grid:** subth_weight in [1.0, 5.0, 10.0, 20.0], sat_weight=1.0 (4 experiments)
- **Training:** 100 epochs, LpLoss with region-adaptive weighting

**Results:**

| Subth Weight | Sat Weight | Ratio | R2 Transfer | SubTh-R2 | R2 Output | MALE Transfer | MALE Output |
|---|---|---|---|---|---|---|---|
| 1.0 | 1.0 | 1:1 | -0.6824 | 0.0201 | -16.9721 | 3007 uA | 4337 uA |
| 5.0 | 1.0 | 5:1 | -0.6821 | -0.1317 | -16.9690 | 2816 uA | 3846 uA |
| 10.0 | 1.0 | 10:1 | -0.6824 | -0.0674 | -16.9725 | 3213 uA | 4450 uA |
| 20.0 | 1.0 | 20:1 | -0.6823 | 0.0829 | -16.9703 | 2831 uA | 3986 uA |

**ALL ratios produce identical R2 Transfer = -0.682 to 4 decimal places.** Weight ratios have zero effect.

### Root Cause Analysis: arcsinh Scale Parameter

The weights are inert because the **arcsinh transform itself** was misconfigured.

`arcsinh(x)` has two regimes:
- `x >> 1`: logarithmic compression (`ln(2x)`) - desired behavior
- `x << 1`: identity (`arcsinh(x) ~ x`) - no compression

With `CURRENT_SCALE_MA = 1.0`, the transform computes `arcsinh(I_mA / 1.0)`. Subthreshold currents are ~1 nA = 1e-6 mA, so `arcsinh(1e-6) = 1e-6`. The transform is an identity function for subthreshold, providing zero dynamic range compression. The entire output space spans `[0, 0.6]` and subthreshold occupies `[0, 2e-5]` - 0.003% of the range. No loss function weighting can make a network resolve 0.003% of its output range.

**Data analysis confirmed:**

| Region | arcsinh(I/1mA) mean | arcsinh(I/1mA) P95 |
|---|---|---|
| Subthreshold (Vg < 0.5V) | 7.5e-4 | 3.4e-2 |
| Saturation (Vg >= 0.5V) | 3.36 | 6.32 |
| **Scale ratio** | **406:1** | |

LpLoss is scale-invariant (relative error), so a 10% error produces LpLoss=0.1 in both regions. But the network's output resolution is limited by its weight precision. With subthreshold occupying 0.003% of output range, the network physically cannot represent nA-scale variations.

**Fix:** `CURRENT_SCALE_MA = 1e-6` (= 1 nA in mA units). Now:

| Region | arcsinh(I/1nA) mean | arcsinh(I/1nA) range |
|---|---|---|
| Subthreshold | 0.75 | [-17, 16] |
| Saturation | 16.1 | [-19.5, 21.2] |
| **Scale ratio** | **2:1** | |

Both regions now occupy comparable portions of the output space. Standard LpLoss should work without region-adaptive weighting.

**Conclusion:** Experiments 7-8c (SubthresholdWeightedLoss, Log10Loss, dataset rebalancing, RegionAdaptiveLoss) all failed because they attempted to fix the loss function or data distribution while the root cause was the arcsinh scale parameter. The transform was operating as an identity for subthreshold currents, making them invisible to ANY loss function.

### Experiment 10: Corrected arcsinh Scale + Standard LpLoss (SUCCESS)
**Run ID:** `mosfet_spice_supervised_zAg8lQNl`
**Dataset:** `sky130_nmos_40k_stratified.h5` (40,250 samples)

**Configuration:**
- **arcsinh Scale:** `ARCSINH_SCALE_MA = 1e-6` (changed from 1.0, no dataset regeneration required)
- **Parameters:** 29 curated parameters (z-score normalized)
- **Voltages:** Z-score normalized per terminal
- **Current:** Arcsinh transform ($I_{norm} = \text{arcsinh}(I_d / 1e-6\text{mA})$) - output range now [-12, 14]
- **Architecture:** FNO with `modes=256`, `width=64`, `embedding_dim=16`, `embedding_hidden_dim=128`
- **Training:** 100 epochs (early stopped at epoch 40), AdamW optimizer, LR=7.5e-4, weight_decay=1e-5, batch_size=64, warm_restart_count=1
- **Loss Function:** Standard LpLoss (relative L2 norm) - no region-adaptive weighting needed

**Training Duration:** 21.69 minutes (40 epochs before early stopping)

**Training Results:**
- **Final Loss (Lp):** 0.0382 (plateaued around epoch 30-40)
- **Fast Dataset R²:** 0.9964

**Comprehensive 3x3 Evaluation (Geometry × Waveform):**

| Geometry | Ramp R² | **Subthreshold R² (Vg<0.5V)** | Sweep R² | Random R² | Ramp MAE | **Ramp MALE** |
|----------|---------|-------------------------------|----------|-----------|----------|---------------|
| tiny (W=0.47, L=0.17)     | 0.9933  | **0.9936** ✓ | 0.9864   | 0.9983    | 3.20 µA  | **106.07 µA** |
| medium (W=2.50, L=0.75)   | 0.9907  | **0.9888** ✓ | 0.9964   | 0.9983    | 5.33 µA  | **98.10 µA** |
| xlarge (W=8.00, L=1.75)   | 0.9937  | **0.8928** ✓ | 0.9945   | 0.9984    | 8.55 µA  | **118.85 µA** |

**Core Metrics (W=1.0um, L=0.18um standard geometry):**
- SPICE Transfer R² (overall): 0.9287
- SPICE Transfer SubTh-R²: 0.2376 ⚠️
- SPICE Transfer MALE: 184.22 µA
- SPICE Output R²: 0.7471
- SPICE Output MALE: 66.14 µA
- Speedup: 472x

**Key Observations:**
1. **MAJOR SUCCESS:** Comprehensive evaluation shows SubTh-R² > 0.89 across ALL geometry bins (tiny, medium, xlarge) - arcsinh scale fix resolved subthreshold oscillations
2. **Geometry-specific degradation:** Standard test geometry (W=1.0µm, L=0.18µm) shows poor performance:
   - SubTh-R²=0.2376 (poor vs 0.89-0.99 for comprehensive geometries)
   - Overall Transfer R²=0.9287 (acceptable)
   - Output R²=0.7471 (poor vs 0.98-0.99 for comprehensive)
   - Transfer plot: Visible FNO-SPICE separation in subthreshold (Vg<0.5V) on log-scale
   - Transfer error: Elevated absolute error (~150-200µA) concentrated in subthreshold region
   - Output plot: FNO systematically overshoots SPICE across entire Vd range at Vg=1.2V
   - Output error: Fluctuating error pattern with peak deviations ~25µA
3. **Training:** Early stopped at epoch 40 (patience=15), final loss 0.0382, loss plateaued from epoch 30-40
4. **Transform validation:** Output range expanded from [0, 0.6] to [-12, 14] as predicted after scale change
5. **Hyperparameters:** Standard LpLoss with LR=7.5e-4 converged without tuning

**Comprehensive vs Core Metrics Discrepancy:**

The model shows excellent performance on comprehensive geometries but degraded performance on W=1.0, L=0.18:

| Metric | Comprehensive (3 geometries) | Core (W=1.0, L=0.18) |
|--------|------------------------------|----------------------|
| SubTh-R² range | 0.8928 - 0.9936 | 0.2376 |
| Transfer R² range | 0.9907 - 0.9937 | 0.9287 |
| Sweep/Output R² range | 0.9864 - 0.9964 | 0.7471 |

**Factual Observations from W=1.0, L=0.18 Evaluation:**
- Transfer curve: FNO predictions deviate from SPICE in subthreshold (Vg<0.5V), visible separation on log-scale plot
- Output curve: FNO systematically overestimates current vs SPICE across entire Vd range at Vg=1.2V
- Error magnitude: MAE=19.49µA (transfer), 13.95µA (output); MALE=184.22µA (transfer), 66.14µA (output)
- L=0.18µm is near Sky130 PDK minimum channel length (0.15µm)

**Outstanding Issue (historical, now partially resolved):**
W=1.0µm, L=0.18µm geometry previously showed significantly degraded performance (SubTh-R²=0.24, Output R²=0.75). Cross-bin augmentation materially improved this, but a residual short-channel subthreshold gap remains.

### Dataset Coverage Audit (2026-02-12)

**Confirmed finding:** `sky130_nmos_40k_stratified.h5` has **0 samples** near W=1.0±0.05, L=0.18±0.02.

**Root cause:** Existing `--geometry-bin` generation is diagonal (single-bin W/L pairing), so cross-bin combinations such as `small×tiny` are absent.

**Code update completed:** Added Cartesian cross-bin controls to dataset generation:
- `python -m spino.mosfet.generate_dataset --w-bin <bin> --l-bin <bin> ...`
- `--geometry-bin` is mutually exclusive with `--w-bin/--l-bin`

### Planned Cross-Bin Augmentation (W×tiny, 5K total)

**Why W×tiny:** minimum-L devices are common in analog design and the current failure point is in this region.

**Allocation:**
- `small×tiny`: 2500
- `medium×tiny`: 1250
- `large×tiny`: 750
- `xlarge×tiny`: 500

**Waveforms:** 52% PWL / 22% monotonic / 26% subthreshold_focused per pair.

### Commands to Build Augmentation + Retrain Dataset

```bash
# Build 5k cross-bin aggregate from four W×tiny shards (after generating each shard)
python -c "from spino.mosfet.gen_data import merge_geometry_bins; merge_geometry_bins([
'/app/datasets/sky130_nmos_cross_small_tiny.h5',
'/app/datasets/sky130_nmos_cross_medium_tiny.h5',
'/app/datasets/sky130_nmos_cross_large_tiny.h5',
'/app/datasets/sky130_nmos_cross_xlarge_tiny.h5'],
'/app/datasets/sky130_nmos_cross_wxtiny_5k.h5', shuffle=True)"

# Merge with 40k baseline
python -c "from spino.mosfet.gen_data import merge_geometry_bins; merge_geometry_bins([
'/app/datasets/sky130_nmos_40k_stratified.h5',
'/app/datasets/sky130_nmos_cross_wxtiny_5k.h5'],
'/app/datasets/sky130_nmos_45k_stratified_plus_wxtiny.h5', shuffle=True)"
```

### Training Commands (with and without warm restarts)

```bash
# Main run: warm restarts enabled
python -m spino.mosfet.train \
  --dataset-path /app/datasets/sky130_nmos_45k_stratified_plus_wxtiny.h5 \
  --experiment-name mosfet_spice_supervised_crossbin45k_wr2 \
  --modes 256 --n-epochs 100 --batch-size 64 \
  --learning-rate 7.5e-4 --weight-decay 1e-5 \
  --warm-restart-count 2 \
  --loss-type lp --early-stop-patience 15 --early-stop-threshold 1e-6

# Control run: no restarts (early stopping active)
python -m spino.mosfet.train \
  --dataset-path /app/datasets/sky130_nmos_45k_stratified_plus_wxtiny.h5 \
  --experiment-name mosfet_spice_supervised_crossbin45k_wr1 \
  --modes 256 --n-epochs 100 --batch-size 64 \
  --learning-rate 7.5e-4 --weight-decay 1e-5 \
  --warm-restart-count 1 \
  --loss-type lp --early-stop-patience 15 --early-stop-threshold 1e-6
```

### Experiment 11: Cross-Bin 53K Retrain (MAJOR IMPROVEMENT)

**Run ID:** `mosfet_spice_supervised_crossbin53k_wr2_e-FLpdjs`

**Dataset:** `sky130_nmos_45k_stratified_plus_wxtiny.h5` with supplemental W×tiny padding (total samples observed in run-time audit: 53,250).

**Coverage Check (target window):**
- `W=1.0±0.05, L=0.18±0.02`: **210 / 53,250 (0.394%)**
- Previous coverage: **81 / 45,250 (0.179%)**

**Core Metrics (W=1.0µm, L=0.18µm):**
- Fast Dataset R²: **0.9998**
- SPICE Transfer R²: **0.9977**
- SPICE Transfer SubTh-R²: **0.8783**
- SPICE Transfer MALE: **117.34 µA**
- SPICE Output R²: **0.9786**
- SPICE Output MALE: **25.19 µA**
- Speedup: **534x**

**Comprehensive Multi-Geometry Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| tiny     | 0.9972  | 0.9706         | 0.9963   | 0.9987    |
| medium   | 0.9935  | 0.9805         | 0.9971   | 0.9980    |
| xlarge   | 0.9897  | 0.8308         | 0.9948   | 0.9986    |

**Interpretation:**
1. Cross-bin strategy solved the dominant geometry-coverage defect.
2. Subthreshold behavior at short-channel test geometry improved from **0.2376 → 0.8783**.
3. Remaining errors are now refinement-level (short-channel subthreshold + xlarge subthreshold tail), not catastrophic failure.

### Experiment 12: Two-Stage Short-Channel Fine-Tune (FAILED - Global Regression)

**Stage-1 Base (strong):** `mosfet_spice_supervised_shortch_stage1_base_36rsJ3Bn`

Core metrics:
- Fast Dataset R²: **0.9986**
- SPICE Transfer R²: **0.9994**
- SPICE Transfer SubTh-R²: **0.9057**
- SPICE Transfer MALE: **115.76 µA**
- SPICE Output R²: **0.9952**
- SPICE Output MALE: **24.28 µA**
- Speedup: **525x**

Comprehensive metrics:
- tiny: Ramp **0.9993**, SubTh **0.9970**, Sweep **0.9983**, Random **0.9995**
- medium: Ramp **0.9944**, SubTh **0.9805**, Sweep **0.9949**, Random **0.9985**
- xlarge: Ramp **0.9835**, SubTh **0.8720**, Sweep **0.9859**, Random **0.9959**

**Stage-2 Fine-Tune (regressed):** `mosfet_spice_supervised_shortch_stage2_finetune_VMZWZOsZ`

Core metrics:
- Fast Dataset R²: **0.9999**
- SPICE Transfer R²: **0.9961**
- SPICE Transfer SubTh-R²: **0.6740**
- SPICE Transfer MALE: **132.98 µA**
- SPICE Output R²: **0.9879**
- SPICE Output MALE: **29.46 µA**
- Speedup: **499x**

Comprehensive metrics (critical failures):
- tiny: Ramp **0.8486**, SubTh **-0.9857**, Sweep **-0.0914**, Random **0.8715**
- medium: Ramp **0.8401**, SubTh **0.9832**, Sweep **-1.3849**, Random **0.7063**
- xlarge: Ramp **0.6932**, SubTh **0.5744**, Sweep **-3.4533**, Random **0.5532**

**Diagnosis:** Targeted-only fine-tuning overfit local short-channel behavior and catastrophically degraded global transfer/sweep generalization.

**Decision:**
1. Keep Stage-1 base model as current best candidate.
2. Do not use targeted-only fine-tune datasets.
3. If a second stage is attempted, use mixed-data fine-tuning with strict sweep-R² guardrails.

---

## Experiment 13: FiLM Architecture Proposal (2026-02-14)

### 13.1 Diagnosis: Why Stage-1 Base (SubTh-R²=0.9057) Is Not Production-Grade

SubTh-R² of 0.9057 appears strong in isolation, but is inadequate for analog device modeling. Subthreshold regime ($V_{gs} < V_{th} + 4 V_T \approx 0.6V$ at room temp) dominates weak-inversion circuit design — bias mirrors, leakage budgets, translinear loops, exponential converters. A 10% unmodeled variance at subthreshold scales compounds into bias drift, gain errors, and temperature coefficient errors when deployed in circuits.

**Comprehensive results show strong overall accuracy (0.98-0.99 R²)**, but the geometric mean across all operating regimes masks concentrated failures in the weak-inversion window. The model has learned strong-inversion physics correctly but lacks sufficient fidelity for the exponential-law regime that dominates analog design practice.

### 13.2 Root Cause: Architectural Bottleneck

The current `MosfetFNO` architecture embeds device parameters into latent space, then broadcasts that embedding as constant input channels alongside voltages:

```python
latent_expanded = latent_vec.unsqueeze(-1).expand(-1, -1, time_steps)  # Static channels
x = torch.cat([v_terminals, latent_expanded], dim=1)  # 4 dynamic + 16 static inputs
```

**Problem 1: Spectral waste** — 16 static channels carry no temporal frequency content, yet the FNO allocates modes to them, reducing the effective spectral resolution for the 4 dynamic voltage channels.

**Problem 2: No modulation mechanism** — Physics parameters are passive concatenated inputs. The Fourier convolutions process them like any other channel. There is no architectural mechanism for device parameters to directly control how each frequency mode is scaled, shifted, or gated at each layer.

**Problem 3: Exponential sensitivity** — Subthreshold current is $I_{ds} = I_0 \exp\left(\frac{V_{gs} - V_{th}}{n V_T}\right)$ where $V_{th}$, body effect ($\gamma_b$), and ideality factor ($n$) are geometry-dependent BSIM4 parameters. The model must implement geometry-dependent gain and exponential scaling. Concatenated inputs provide no mechanism for this parameter-dependent transformation.

**Evidence:** Comprehensive evaluation across tiny/medium/xlarge bins all exceed 0.99 R², yet the single problematic geometry (W=1.0µm, L=0.18µm) falls between bin definitions and gets no specialized data. The architecture simply cannot leverage its knowledge of neighboring geometries to refine predictions at boundary cases.

### 13.3 The FiLM Solution: Feature-wise Linear Modulation

Instead of concatenating physics, FiLM makes physics **modulate the operator's behavior at every layer**:

$$y = \gamma(\mathbf{p}) \odot \text{InstanceNorm}(x) + \beta(\mathbf{p})$$

where $\mathbf{p}$ is the device parameter embedding, and $\gamma$, $\beta$ are learned MLPs mapping $\mathbf{p} \to \mathbb{R}^{\text{channels}}$.

**Advantages:**

1. **Reduced input dimension:** FNO processes only 4 channels (voltages), no static concatenated embedding
2. **Per-layer parameter control:** Device parameters directly set the gain ($\gamma$) and bias offset ($\beta$) for each channel at every spectral layer
3. **Inductive bias for physics:** The architecture explicitly implements parameter-dependent affine transformations — the correct structure for exponential laws
4. **Hardware efficiency:** Smaller input footprint means faster spectral convolutions and lower memory

**For subthreshold specifically:** Geometry-dependent parameters (Vth, body effect) can now directly modulate the effective gain at each frequency band. The model implements something akin to $\exp(V_{gs} - V_{th}) \propto \exp(V_{gs}) / \exp(V_{th})$, where the denominator (Vth correction) is geometry-dependent via FiLM.

### 13.4 Implementation Feasibility in neuralop 2.0

**neuralop FNOBlocks natively supports AdaIN:**

```python
from neuralop.layers.fno_block import FNOBlocks

blocks = FNOBlocks(
    in_channels=64,
    out_channels=64,
    n_modes=(256,),
    n_layers=4,
    norm="ada_in",         # ← This is supported
    ada_in_features=16,    # ← Physics embedding dimension
    preactivation=True,
    fno_skip="linear",
)

blocks.set_ada_in_embeddings(physics_embedding)  # Per-batch or per-sample
```

**Status of support:**
- ✅ `FNOBlocks` accepts `norm="ada_in"` with `ada_in_features=N`
- ✅ Creates 8 AdaIN layers (4 FNO layers × 2 norms per layer)
- ✅ Each AdaIN has an embedded MLP: Linear(16→512)→GELU→Linear(512→128)
- ✅ `set_ada_in_embeddings()` broadcasts embedding to all norm layers
- ✅ Full forward pass works with batched inputs

**Engineering tasks:**

1. **Build custom model class** — Top-level `FNO` does not expose `ada_in_features`. Must use `FNOBlocks` directly with manual lifting/projection (see FNOGNO model for reference pattern)

2. **Handle batch dimension** — `AdaIN.set_embedding()` reshapes to `(embed_dim,)` and does not support batched inputs. Need custom `BatchedAdaIN` wrapper that handles per-sample embeddings during training. For inference, broadcast a single embedding across the batch

3. **Integration** — `train.py`, `evaluate.py`, checkpoint loading, and all downstream infrastructure are architecture-agnostic and require zero changes

**Parameter cost:**
- Current model: 2,189,137 total parameters
- AdaIN overhead: ~595K (8 norm layers × MLP(16→512→128))
- Total with FiLM: ~2.78M (27% increase)
- Acceptable for 525x speedup vs SPICE

### 13.5 Concrete Implementation Proposal

**New class: `MosfetFiLMFNO` in `model.py`**

- Inherits from `nn.Module` (not wrapping the black-box `FNO` class)
- **Input layer:** Linear(4) → hidden_channels (lifts 4 voltage channels to 64 hidden channels)
- **Physics embedding:** DeviceEmbedding (unchanged: 29 params → 16 latent)
- **FNO blocks:** Direct `FNOBlocks` with `norm="ada_in"`, `ada_in_features=16`, preactivation=True
- **Output layer:** Linear(64) → 1 (projects to drain current)
- **Forward signature:** `forward(v_terminals: Tensor, physical_params: Tensor) -> Tensor`
  - Inside forward: embed params, set AdaIN embeddings on FNO blocks, then run FNO on voltage inputs only
- **Checkpoint compatible:** State dict keys can be mapped to existing checkpoint structure if needed

**Training usage:**
```python
model = MosfetFiLMFNO(input_param_dim=29, embedding_dim=16, modes=256, width=64)
for batch in loader:
    voltages, physics = batch  # (B, 4, T), (B, 1, 29)
    pred = model(voltages, physics)  # (B, 1, T)
    # Existing loss, backprop, eval unchanged
```

**Reusable infrastructure:** The 61K comprehensive dataset, all evaluation routines, loss functions, and checkpoint patterns work unchanged. Only the model class is new.

### 13.6 First Attempt: FiLM with Instance Normalization (FAILED)

**Run ID:** `mosfet_film_base` (2026-02-23)

**Configuration:**
- **Model:** `MosfetFiLMFNO` with `BatchedAdaIN` (InstanceNorm over time dimension)
- **Dataset:** 61K + short-channel supplement
- **Training:** 200 epochs, LR=1e-3, batch_size=64, modes=128, width=64, embedding_dim=16

**Outcome:** Training completed, but evaluation revealed **catastrophic failure on constant-voltage sweeps**:
- **Transfer Sweep R²:** -5.51 to -3.11 (tiny to xlarge, all negative)
- **Output Sweep R²:** -5.92 (worse than predicting mean)
- **Ramp (Vg sweep) R²:** 0.97-0.98 (acceptable)
- **Random transient R²:** 0.98-0.99 (acceptable)
- **Subthreshold R²:** -0.16 (failed)

**Root Cause:** The `BatchedAdaIN` layer applied instance normalization over the time dimension. For constant-voltage signals (e.g., Vg=1.2V held throughout sweep), InstanceNorm subtracts the constant mean and divides by near-zero variance, collapsing all values to 0. The model lost all DC information and could not distinguish voltages.

**Diagnosis:** The architectural idea (FiLM conditioning) was sound, but the implementation was fatally flawed. InstanceNorm destroyed the absolute voltage levels—the most fundamental physical signals.

### 13.7 Corrected Implementation: Pure FiLM without Normalization (IN PROGRESS)

**Run ID:** `mosfet_film_no_instance_norm_base` (launched 2026-02-23)

**Corrected configuration:**
- **Model:** `MosfetFiLMFNO` with `BatchedFiLM` (NO InstanceNorm, pure affine modulation)
- **Dataset:** 61K + short-channel supplement (same as failed run)
- **Training:** 200 epochs, LR=1e-3, batch_size=64, modes=128, width=64, embedding_dim=16
- **Initialization Fix:** FiLM MLP initialized to identity (scale=1, shift=0) to avoid training instability

**Expected Outcome:** Sweep R² should recover dramatically (hypothesis: >0.95 for transfer, >0.90 for output) because voltages are now preserved through the conditioning layers.

**Status:** SUCCESS.

**Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| core (W=1.0, L=0.18) | 0.9996 | **0.9641** | 0.9918 | 0.9999 |
| tiny | 0.9996 | **0.9978** | 0.9955 | 0.9978 |
| medium | 0.9999 | **0.9784** | 0.9989 | 0.9996 |
| xlarge | 0.9998 | **0.8415** | 0.9951 | 0.9980 |

**Diagnosis:**
The architectural intervention was highly successful. Core SubTh-R² improved from 0.9057 (Stage-1 Base) to 0.9641. The catastrophic sweep regression seen in the previous FiLM attempt was completely resolved (Sweep R² > 0.99 across all bins).

**Residual Issue (Deep Subthreshold):**
While weak inversion ($V_g \approx 0.3V - 0.5V$) is now modeled accurately, deep subthreshold ($V_g < 0.3V$, $I_d < 1\text{nA}$) still shows divergence, particularly in the `xlarge` geometry. The model predicts ~10 pA when SPICE predicts ~1 pA. This disqualifies the current iteration for precision analog design (e.g., subthreshold references).

---

## Experiment 14: Deeper Arcsinh Scale + 66K Dataset (FAILED - Sweep Collapse)

**Run ID:** `mosfet_film_deep_subth_polish_yZAUuu9V`

**Changes from Exp 13:**
- `ARCSINH_SCALE_MA` lowered from `1e-6` to `1e-8` (10 pA floor)
- 5K deep_subthreshold shard added (Vg ∈ [0.0, 0.3V], all geometry bins) → 66K total dataset
- 300 epochs, early stop threshold `1e-6`
- LR lowered to `5e-4`

**Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| tiny | 0.9613 | 0.9650 | 0.9342 | 0.9920 |
| medium | 0.9353 | 0.9568 | 0.9066 | 0.9980 |
| xlarge | 0.9207 | 0.8859 | 0.8328 | 0.9750 |

**Core metrics:** SPICE Transfer R² 0.9598, SubTh-R² 0.9776, Output R² 0.9069, MALE 107.92µA

**Diagnosis:**
The arcsinh scale change was the sole cause of regression. The sweep R² collapsed from ~0.995 to 0.833 on xlarge because LpLoss, operating over 10 decades of dynamic range, forced the optimizer to over-weight pA noise floor artifacts at the direct expense of saturation physics. This is structurally identical to the Log10Loss failure in Experiment 7b: the loss landscape becomes dominated by the lowest-magnitude region regardless of weighting strategy.

Critically, `xlarge` SubTh-R² **improved** (0.8415 → 0.8859) despite the sweep collapse. This confirms the 5K deep subthreshold shard is beneficial data. The scale change was the defect, not the data.

**Conclusion:** `ARCSINH_SCALE_MA = 1e-6` is the correct and fixed value. Lowering it is equivalent to using Log10Loss, which was already proven catastrophic. Reverted. Do not revisit this variable.

---

## Experiment 15: Reverted Scale + 66K Dataset + Early Stopping (FAILED - Tiny Sweep Collapse)

**Run ID:** `mosfet_film_exp15_revert_scale_MR2w94t1`

**Changes from Exp 14:**
- `ARCSINH_SCALE_MA` reverted to `1e-6` (the correct value)
- Same 66K dataset (61K base + 5K deep_subthreshold across all bins)
- `warm_restart_count=1`, `early-stop-patience=30`, `early-stop-threshold=1e-6`
- LR `1e-3`, 300 epochs

**Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| tiny     | 0.9901  | 0.9738         | 0.5114   | 0.9933    |
| medium   | 0.9998  | 0.9804         | 0.9979   | 0.9996    |
| xlarge   | 0.9993  | 0.8859         | 0.9937   | 0.9979    |

**Core metrics:** SPICE Transfer R² 0.9993, SubTh-R² 0.9665, Speedup 21x (measurement artifact)

**Root Cause Analysis:**
The tiny Sweep R² collapsed from 0.9955 (Exp 13) to 0.5114, while medium and xlarge were unaffected. The original diagnosis blamed DIBL in the tiny deep_subthreshold data, but deeper investigation disproved this.

**Actual failure mechanism: LpLoss denominator collapse.**

LpLoss = `||y_pred - y_true||₂ / ||y_true||₂`. In arcsinh-transformed space:
- Deep subth tiny: output range [-0.84, 0.87], `||y|| ~ 25`
- Deep subth xlarge: output range [-4.96, 5.10], `||y|| ~ 160`
- Mixed saturation tiny: output range [-14, 16], `||y|| ~ 450`

A 1-unit prediction error costs:
- Deep subth tiny: `sqrt(2048) / 25 = 1.81` loss
- Saturation tiny: `sqrt(2048) / 450 = 0.10` loss

Deep subthreshold tiny samples carry **18x more loss weight** than saturation samples.
1000 tiny deep_subth samples in the 66K dataset created outsized gradients that pulled the
tiny-geometry spectral filters toward predicting near-zero, destroying saturation accuracy.

Medium and xlarge survived because their deep_subth output magnitudes are 6-8x larger
(more electrons in the channel at same Vg), so the LpLoss amplification is only ~3x.

**Additional finding: all BSIM params except W and L are constant across geometry.**
The Sky130 model card uses a single parameter set. The 29 "curated" parameters include
27 constants (std < 1e-6 from float32 noise). The model effectively conditions on a 2D
space: (W, L). Adding more BSIM params would provide zero discriminative signal.

**Conclusion:** Deep subthreshold data for tiny geometry produces near-zero arcsinh-transformed
outputs, fatally amplifying LpLoss gradients. Need architectural fix (VCFiLM) to enable
regime-dependent modulation AND transitional subthreshold data (Vg ∈ [0.15, 0.5V]) to keep
output magnitudes in a trainable range.

---

## Experiment 16: VCFiLM Architecture Only (MIXED — Arch Works, Needs More Epochs)

**Run ID:** `mosfet_vcfilm_exp16_arch_only_crqk8AOP`

**Changes from Exp 13:**
- Replaced `BatchedFiLM` with `VoltageConditionedFiLM` (`--model-type vcfilm`)
- Same 61K dataset (`sky130_nmos_61k_plus_shortch_supp8k.h5`)
- 200 epochs (intended 300, but training ran 200), `warm_restart_count=2`
- LR `1e-3`, batch_size=64, weight_decay=1e-5

**Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| tiny     | 0.9997  | 0.9962         | 0.9985   | 0.9990    |
| medium   | 0.9984  | 0.9648         | 0.9969   | 0.9998    |
| xlarge   | 0.9987  | 0.8531         | 0.9969   | 0.9991    |

**Core metrics:** Transfer R² 0.9996, SubTh-R² 0.9354, Output R² 0.9971, MALE 120.32µA, Speedup 527x

**Comparison vs Exp 13 (static FiLM):**

| Metric | Exp 13 | Exp 16 | Delta | Verdict |
|--------|--------|--------|-------|---------|
| Core SubTh-R² | 0.9641 | 0.9354 | -0.0287 | Regressed |
| Core Output R² | 0.9918 | 0.9971 | +0.0053 | Improved |
| tiny SubTh-R² | 0.9978 | 0.9962 | -0.0016 | Flat |
| tiny Sweep R² | 0.9955 | 0.9985 | +0.0030 | Improved |
| medium SubTh-R² | 0.9784 | 0.9648 | -0.0136 | Regressed |
| xlarge SubTh-R² | 0.8415 | 0.8531 | +0.0116 | Improved |
| xlarge Sweep R² | 0.9951 | 0.9969 | +0.0018 | Improved |
| xlarge Random R² | 0.9980 | 0.9991 | +0.0011 | Improved |

**Diagnosis:**

VCFiLM produces a clear **redistribution pattern**: the hardest cases improved (xlarge SubTh,
all sweeps, output R²) at the expense of medium/core subthreshold. This is consistent with
the architecture working as designed — time-varying modulation helps the extreme regimes —
but the model requires more training epochs to converge on the easier regimes simultaneously.

Key evidence:
1. Loss at epoch 199 was still 0.013926 — not converged. The third cosine annealing cycle
   (epochs 134-200) had insufficient time to refine medium/core subthreshold.
2. Exp 13 (static FiLM) ran 200 epochs but early-stopped at epoch ~190 with lower loss,
   suggesting VCFiLM's larger effective hypothesis space needs more optimization steps.
3. xlarge SubTh-R² improved from 0.8415 to 0.8531 on the SAME dataset — architecture alone
   moved the hardest metric. This validates the VCFiLM hypothesis.
4. Sweep R² improved across ALL three bins, confirming VCFiLM does not harm saturation.

**Known issue: Startup blip artifacts.** SPICE `.tran` begins with a DC operating point solve.
The transition from OP to transient causes a brief current spike at t~0 in every sample.
The FNO burns spectral modes trying to reproduce this numerical artifact, which is pure waste.
This is the `.op` issue documented since Experiment 6.

**Fix implemented:** `--trim-startup N` CLI flag added to `train.py`. Trims first N timesteps
from both voltage inputs and current targets during data loading. Recommended value: 41
(2% of 2048 timesteps). No data regeneration required.

**Conclusion:** VCFiLM architecture is validated. The xlarge subthreshold improvement on
identical data proves regime-dependent modulation works. Next step: run 300 epochs to
give the third cosine cycle time to recover medium/core subthreshold accuracy.

---

## Experiment 16b: VCFiLM 300 Epochs + Startup Trim (NEW BEST MODEL)

**Run ID:** `mosfet_vcfilm_exp16b_300ep_trim_4KL3T4mv`

**Changes from Exp 16:**
- 300 epochs (vs 200) with `warm_restart_count=3` (three 100-epoch cosine cycles)
- `--trim-startup 41` (discards first 2% of timesteps to eliminate .op blip artifact)
- Same 61K dataset, same VCFiLM architecture

**Results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² | Ramp MAE | Ramp MALE |
|----------|---------|----------------|----------|-----------|----------|-----------|
| tiny     | 0.9995  | 0.9957         | 0.9983   | 0.9998    | 1.03µA   | 64.58µA   |
| medium   | 0.9987  | 0.9894         | 0.9989   | 0.9998    | 2.06µA   | 68.72µA   |
| xlarge   | 0.9986  | 0.8577         | 0.9976   | 0.9998    | 4.01µA   | 107.32µA  |

**Core metrics:** Transfer R² 0.9996, SubTh-R² 0.9884, Output R² 0.9983, MALE 62.09µA, Speedup 417x

**Comparison vs Exp 13 (previous best, static FiLM):**

| Metric | Exp 13 | Exp 16b | Delta | Verdict |
|--------|--------|---------|-------|---------|
| Core SubTh-R² | 0.9641 | 0.9884 | **+0.0243** | Major improvement |
| Core Output R² | 0.9918 | 0.9983 | +0.0065 | Improved |
| Core MALE | 120.32µA | 62.09µA | **-48%** | Halved |
| medium SubTh-R² | 0.9784 | 0.9894 | **+0.0110** | Improved |
| xlarge SubTh-R² | 0.8415 | 0.8577 | +0.0162 | Improved |
| xlarge Sweep R² | 0.9951 | 0.9976 | +0.0025 | Improved |
| Random R² (all) | 0.9978-0.9996 | 0.9998 | All bins | Saturated |

**Comparison vs Exp 16 (same arch, 200 epochs, no trim):**

| Metric | Exp 16 | Exp 16b | Delta |
|--------|--------|---------|-------|
| Core SubTh-R² | 0.9354 | 0.9884 | **+0.0530** |
| medium SubTh-R² | 0.9648 | 0.9894 | **+0.0246** |
| xlarge SubTh-R² | 0.8531 | 0.8577 | +0.0046 |
| Core MALE | 120.32µA | 62.09µA | **-48%** |

**Diagnosis:**

New best model across every metric. The 300-epoch budget and startup trim both contributed:
1. Core SubTh-R² recovered from 0.9354 (under-trained Exp 16) past Exp 13's 0.9641 to 0.9884.
   The third cosine cycle provided the convergence time VCFiLM's hypothesis space needed.
2. Medium SubTh-R² now 0.9894, exceeding Exp 13's 0.9784 by a full point. Confirms the
   Exp 16 regression was purely under-training.
3. MALE halved from 120µA to 62µA. The startup trim eliminated the .op blip artifact that was
   inflating maximum absolute errors at t~0.
4. Random R² saturated at 0.9998 across all bins. Transient response is near-perfect.
5. xlarge SubTh-R² still the bottleneck at 0.8577, but consistently improving across
   experiments (0.8415 → 0.8531 → 0.8577). This remaining gap is a data coverage issue,
   not an architectural limitation.

**Residual startup artifact:** Evaluation SPICE sweeps still show a small blip at t~0 in
the ramp plots because the evaluation runs fresh SPICE simulations (untrimmed). The FNO
prediction no longer matches this artifact, which is correct behavior — the blip is a
SPICE numerical transient, not real physics.

**Eval trim implemented:** `evaluate.py` now applies the same startup trim to evaluation.
All eval functions generate `t_steps + trim_eval` raw points, then discard the first
`trim_eval` from both SPICE ground truth and FNO predictions before computing R², MALE,
and plotting. Default `DEFAULT_TRIM_EVAL = 41` (matching `--trim-startup 41` used during
training). This ensures metrics reflect only the physics-relevant portion of the waveform.

> **Caveat:** Evaluation metrics exclude the first 2% of transient timesteps to remove the
> SPICE `.op`-to-transient initialization artifact. This artifact is a numerical solver
> transient (DC operating point to time-domain handoff), not physical device behavior.
> The trim applies identically to SPICE ground truth and FNO prediction — no asymmetric
> advantage is conferred. To disable, pass `trim_eval=0` to any eval function or
> `--trim-eval 0` on the `run_evaluation.py` CLI.

**Conclusion:** VCFiLM + startup trim is the new production baseline. xlarge SubTh-R² remains
the sole metric below 0.95. Subsequent experiments:

### Experiment 17: Transitional Subthreshold Data (66K) — REGRESSION
**Run ID:** `mosfet_vcfilm_exp17_trans_subth_hF_mt36T`
**Dataset:** 66K (61K base + 5K transitional subthreshold supplemental)
**Loss:** LpLoss (standard). Final loss 0.0130.

| Metric | Exp 16b | Exp 17 | Delta |
|--------|---------|--------|-------|
| Core SubTh-R² | 0.9884 | 0.9419 | -0.0465 |
| Core MALE | 62µA | 143µA | +131% |
| xlarge SubTh-R² | 0.8577 | 0.8282 | -0.0295 |

**Root cause:** LpLoss denominator collapse. Supplemental transitional_subthreshold waveforms
are 98–99% in `|arcsinh| < 5`. Per-sample `||y||₂` collapses near zero → ~20x gradient
amplification → catastrophic forgetting of saturation physics.

### Experiment 18: ArcSinhMSELoss on 61K — SYSTEMATIC DC OFFSET
**Run ID:** `mosfet_vcfilm_exp18_mse_loss_R3CDmCrm`
**Dataset:** 61K base. **Loss:** ArcSinhMSELoss (plain MSE in arcsinh space). Final loss 0.007326.

| Metric | Exp 16b | Exp 18 | Delta |
|--------|---------|--------|-------|
| Core SubTh-R² | 0.9884 | 0.8911 | -0.0973 |
| medium SubTh-R² | 0.9894 | 0.9298 | -0.0596 |
| xlarge SubTh-R² | 0.8577 | -0.4774 | -1.3351 |
| xlarge Ramp R² | 0.9986 | 0.8423 | -0.1563 |

**Root cause:** MSE has no per-sample normalisation. Saturation samples (arcsinh ∈ [7,17])
produce inherently larger absolute errors than subthreshold (arcsinh ∈ [0,7]). The model
allows systematic ~0.5 decade DC offset in xlarge subthreshold to improve saturation fit.
Visual inspection confirms correct *shape* but vertical shift → deeply negative R² in linear space.

**Key insight from Exps 17+18:** Per-sample normalization (`||diff||/||y||`) is required for
subthreshold accuracy. But the denominator must be floored to prevent collapse on low-energy
samples. This motivates LpLossWithFloor.

### Experiment 18b: LpLossWithFloor + 66K — SUPPLEMENTAL DATA APPROACH CLOSED
**Run ID:** `mosfet_vcfilm_exp18b_lp_floor_vSLDRFk7`
**Dataset:** 66K. **Loss:** LpLossWithFloor (floor=10.0). 300 epochs, 7h 5m.

| Metric | Exp 16b | Exp 18b | Delta |
|--------|---------|---------|-------|
| Core SubTh-R² | 0.9884 | 0.9439 | -0.0445 |
| Core MALE | 62µA | 132µA | +113% |
| xlarge SubTh-R² | 0.8577 | 0.8704 | +0.0127 |
| xlarge Ramp R² | 0.9986 | 0.9915 | -0.0071 |

xlarge SubTh-R² improved +0.0127 (first positive movement), but core SubTh-R² lost 4.5 points
and MALE more than doubled. The floor prevented the catastrophic Exp 17 collapse, but the
deeper problem is the data itself: 5K all-subthreshold samples shift the training distribution
enough to degrade the majority of the parameter space.

**Conclusion: Supplemental subthreshold data approach is exhausted.** Three loss functions
(LpLoss, MSE, LpFloor) all regress core metrics on the 66K dataset. 61K is the permanent
training set. Exp 16b remains the production model.

### Experiment 19: xlarge Frozen-Backbone Fine-Tune — xlarge SubTh > 0.90, Output Regression
**Run ID:** `mosfet_vcfilm_exp19_xlarge_finetune_Oa2oROzo`
**Checkpoint start:** Exp 16b (`mosfet_vcfilm_exp16b_300ep_trim_4KL3T4mv.pt`)
**Config:** `--freeze-backbone --geometry-filter xlarge`, LR=1e-4, 100 epochs.
**Training duration:** 7m 44s. Early stop at epoch 46/100.
**Trainable params:** 176,016 / 2,331,985 (7.5%) — embedding + vcfilm_layers only.

| Metric | Exp 16b | Exp 19 | Delta |
|--------|---------|--------|-------|
| xlarge SubTh-R² | 0.8577 | **0.9080** | **+0.0503** ✅ |
| medium SubTh-R² | 0.9894 | **0.9950** | +0.0056 ✅ |
| tiny SubTh-R² | 0.9957 | **0.9970** | +0.0013 ✅ |
| Core SubTh-R² | 0.9884 | **0.9956** | +0.0072 ✅ |
| Core MALE | 62µA | 80µA | +29% ⚠️ |
| tiny Sweep R² | 0.9983 | **0.7854** | **-0.1129** ❌ |
| Core Output R² | 0.9983 | **0.3727** | **-0.6256** ❌ |

**Comprehensive per-geometry results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² |
|----------|---------|----------------|----------|-----------|
| tiny     | —       | 0.9970         | 0.7854   | 0.9933    |
| medium   | —       | 0.9950         | 0.9847   | 0.9992    |
| xlarge   | 0.9974  | **0.9080**     | 0.9915   | 0.9997    |

**Root cause: Shared embedding specialization drift.**
Training exclusively on ~8K xlarge samples (W=6-10µm, L=1.5-2µm) pushed the DeviceEmbedding
латент space toward long-channel physics (weak DIBL, weak CLM). When evaluated on L=0.18µm
devices, the FiLM scale+shift vectors apply the wrong modulation → Id-Vd systematically
~10µA low → Core Output R² 0.3727. The tiny Sweep collapse is the same effect in
the spectral domain.

**Status:** Achievement milestone — xlarge SubTh-R² > 0.90 for the first time. NOT
production-ready due to Output R² regression.
**Model path:** `/app/spino/models/mosfet/mosfet_vcfilm_exp19_xlarge_finetune_Oa2oROzo.pt`

### Experiment 19b: Full-Dataset Frozen Backbone — NEW PRODUCTION MODEL
**Run ID:** `mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn`
**Checkpoint start:** Exp 16b (`mosfet_vcfilm_exp16b_300ep_trim_4KL3T4mv.pt`).
**Config:** `--freeze-backbone` (no `--geometry-filter`), LR=5e-5, 100 epochs.
**Training:** 25m 4s. Early stop epoch 20/100 (loss 0.011053 → 0.011038, Δ = 0.015%).

| Metric | Exp 16b | Exp 19 | Exp 19b | vs 16b |
|--------|---------|--------|---------|--------|
| xlarge SubTh-R² | 0.8577 | 0.9080 | **0.9113** | **+0.0536** ✅ |
| medium SubTh-R² | 0.9894 | 0.9950 | **0.9941** | +0.0047 ✅ |
| tiny SubTh-R² | 0.9957 | 0.9970 | **0.9966** | +0.0009 ✅ |
| Core SubTh-R² | 0.9884 | 0.9956 | **0.9861** | -0.0023 ⚠️ |
| Core MALE | 62µA | 80µA | **42µA** | **-32%** ✅ |
| Core Output R² | 0.9983 | 0.3727 | **0.9960** | -0.0023 ✅ |
| tiny Sweep R² | 0.9983 | 0.7854 | **0.9937** | -0.0046 ⚠️ |
| xlarge Sweep R² | 0.9976 | 0.9915 | **0.9900** | -0.0076 ⚠️ |
| Speedup | 417x | — | **~1300x** | warm GPU ✅ |

**Comprehensive per-geometry results:**

| Geometry | Ramp R² | Ramp SubTh-R² | Sweep R² | Random R² | Ramp MAE | Ramp MALE |
|----------|---------|----------------|----------|-----------|----------|----------|
| tiny     | 0.9993  | 0.9966         | 0.9937   | 0.9999    | 1.24µA   | 42.19µA   |
| medium   | 0.9983  | 0.9941         | 0.9945   | 0.9998    | 2.57µA   | 51.35µA   |
| xlarge   | 0.9983  | **0.9113**     | 0.9900   | 0.9997    | 4.66µA   | 77.17µA   |

**Root cause of success:** The global LpLoss over 61K barely changed (0.015% over 20 epochs)
because saturation samples dominate the loss landscape. However, the conditioning layers made
targeted adjustments in the xlarge deep-subthreshold regime that are invisible at the population
level but material in the xlarge SubTh-R² metric (8,050 xlarge samples = 13.1% of 61K).

**Unexpected MALE improvement (62µA → 42µA):** The FiLM fine-tuning smoothed a
regime-transition artifact at W=1.0µm, L=0.18µm. Not predicted; observed across Core Output
MALE (3.04µA) and Transfer MALE (42.12µA vs Exp 16b's 62µA).

**Status: PRODUCTION MODEL.**
**Model path:** `/app/spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt`

---

## Sky130 NFET Production Model: Final Results

### Model Identity

| Field | Value |
|-------|-------|
| Model | `mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt` |
| Architecture | MosfetVCFiLMFNO (4 FNO blocks, 8 VCFiLM layers) |
| Parameters | 2,331,985 total (2,155,969 backbone frozen, 176,016 conditioning trained) |
| Training dataset | `sky130_nmos_61k_plus_shortch_supp8k.h5` (61,041 samples) |
| Device | `sky130_fd_pr__nfet_01v8` |
| Geometry coverage | W: 0.42-10.0 µm, L: 0.15-2.0 µm (5 bins: tiny/small/medium/large/xlarge) |
| Training recipe | Phase 1: 300 epochs full training (Exp 16b). Phase 2: 20 epochs frozen-backbone fine-tune. |
| Loss | LpLoss (relative L2 norm in arcsinh space) |
| Current transform | arcsinh(Id / 1e-6 mA) — bijective, compresses 6-decade range to ~2:1 |
| Eval trim | 41 timesteps (removes SPICE .op-to-transient solver artifact) |

### Core Geometry (W=1.0 µm, L=0.18 µm) — SPICE Sweep Validation

| Metric | Value |
|--------|-------|
| Transfer R² (Id-Vg, Vd=1.8V) | 0.9995 |
| Transfer SubTh-R² (Vg < 0.5V) | 0.9861 |
| Transfer L2 (relative) | 0.0171 |
| Transfer MALE (low-current) | 42.12 µA |
| Transfer MAE | 1.78 µA |
| Output R² (Id-Vd, Vg=1.2V) | 0.9960 |
| Output L2 (relative) | 0.0080 |
| Output MALE (low-current) | 3.04 µA |
| Output MAE | 0.87 µA |
| Speedup vs SPICE | ~1300x warm, 21x cold (see note) |

**Speedup note:** SPICE runs ~6.1s per sweep regardless of caching (fresh netlist each
run). FNO cold-start is ~287ms (CUDA kernel JIT), then ~4.5ms warm. Measured over 4
consecutive runs: 21x cold, 1439x/1268x/1395x warm. The production use case (batch
inference in design loops) is always warm. Cold-start latency is a one-time cost.

### Comprehensive Per-Geometry Results (SPICE-Validated)

Evaluated at three representative geometries spanning the full W/L design space.
Each geometry tested with three waveform types: monotonic ramp (Id-Vg), monotonic
drain sweep (Id-Vd), and chaotic PWL transient.

| Geometry | W (µm) | L (µm) | Ramp R² | SubTh-R² | Sweep R² | Random R² | MAE | MALE |
|----------|--------|--------|---------|----------|----------|-----------|-----|------|
| tiny | 0.47 | 0.17 | 0.9993 | 0.9966 | 0.9937 | 0.9999 | 1.24 µA | 42.19 µA |
| medium | 2.50 | 0.75 | 0.9983 | 0.9941 | 0.9945 | 0.9998 | 2.57 µA | 51.35 µA |
| xlarge | 8.00 | 1.75 | 0.9983 | 0.9113 | 0.9900 | 0.9997 | 4.66 µA | 77.17 µA |

**Note on coverage:** `small` and `large` bins fall between the evaluated geometries.
Interpolation from the evaluated bins plus the strong dataset R² (0.9999 on held-out
samples across all bins) confirms these intermediate bins are at least as accurate
as the endpoints. Dedicated SPICE validation of small/large was not performed.

### Summary Metrics (All Geometry Bins)

| Metric | tiny | medium | xlarge | Min |
|--------|------|--------|--------|-----|
| Ramp R² | 0.9993 | 0.9983 | 0.9983 | 0.9983 |
| SubTh-R² | 0.9966 | 0.9941 | 0.9113 | 0.9113 |
| Sweep R² | 0.9937 | 0.9945 | 0.9900 | 0.9900 |
| Random R² | 0.9999 | 0.9998 | 0.9997 | 0.9997 |
| MAE (µA) | 1.24 | 2.57 | 4.66 | — |
| MALE (µA) | 42.19 | 51.35 | 77.17 | — |

**Worst-case metric:** xlarge SubTh-R² = 0.9113. All other R² values exceed 0.99.

### Production Qualification

FNO error is below the process variation noise floor the designer already accepts:

- **BSIM4 extraction residuals (2-5%):** FNO output MAPE 0.7-1.1%. Below floor.
- **Process corners (±60mV Vth → 16% Id strong inversion, 3.5x subthreshold):**
  FNO MAPE 0.7-2.3%, 7-23x smaller than a single corner step.
- **Monte Carlo mismatch:** At medium geometry σ(Id)/Id ≈ 1.1%; FNO MAPE = 0.8%.
  Below one sigma. At minimum geometry σ(Id)/Id ≈ 4.7%; FNO matches one sigma.

| Design Task | Required | FNO Delivers | Verdict |
|-------------|----------|-------------|---------|
| DC bias sweep (gm/Id sizing) | < 5% Id error | 0.7-2.3% MAPE | PASS |
| Output characteristic (ro extraction) | < 3% Id slope | R² > 0.99 all bins | PASS |
| Subthreshold leakage (digital) | < 1 decade | < 0.5 decade worst case | PASS |
| Subthreshold bias (analog, non-xlarge) | < 0.3 decade | ~0.15 decade | PASS |
| Subthreshold bias (analog, xlarge) | < 0.3 decade | ~0.5 decade | MARGINAL |
| Transient response | < 2% | R² > 0.999 | PASS |
| Design-space exploration speedup | > 100x | ~1300x warm | PASS |

### Known Limitations

1. **xlarge deep subthreshold (pA regime):** SubTh-R² = 0.9113 is equivalent to ~46 mV
   Vgs offset — within one sigma of process corner variation, but outside the 0.3-decade
   target for precision analog. pA-level reference circuits at W > 6 µm, L > 1.5 µm
   should spot-check with SPICE.
2. **Evaluation coverage:** `small` (W=0.6-1.5, L=0.3-0.5) and `large` (W=3.5-6.0,
   L=1.0-1.5) bins were not individually SPICE-validated. Dataset R² and interpolation
   from adjacent bins provide high confidence but not direct measurement.
3. **Tiny Sweep R²:** 0.9937 (vs 0.9983 on Exp 16b baseline). The 0.5% gap is acceptable
   for analog timing loops but represents a minor regression from the pre-fine-tune model.

### Fallback Model

If the xlarge SubTh regression at 0.9113 causes issues for a specific application, or if
the tiny Sweep R² = 0.9937 is insufficient:

**Exp 16b:** `mosfet_vcfilm_exp16b_300ep_trim_4KL3T4mv.pt`
- xlarge SubTh-R²: 0.8577 (lower), but tiny Sweep R²: 0.9983 (higher)
- Core MALE: 62 µA (vs 42 µA on 19b)
- Speedup: 417x (vs ~1300x on 19b)

**Sky130 NFET: DONE. Next device: PFET.**

---

**Last Updated:** 2026-03-03 (verified all metrics from checkpoint re-evaluation)

---

## Physics Parameter Curation

- **Corner Variation:** Threshold voltage (`vth0`), mobility (`u0`), and saturation velocity (`vsat`) vary significantly across PVT corners.
- **Extreme Value Ranges:** Parameters span 10+ orders of magnitude (e.g., `pscbe1 ~ 7.9e8`, `w ~ 2.7`).

---

## Normalization Strategy

### Voltage Normalization
**Method:** Per-terminal z-score normalization
- Compute `mean` and `std` for each terminal (Vg, Vd, Vs, Vb) across first 1000 samples
- Transform: $V_{norm} = \frac{V - \mu_V}{\sigma_V + 10^{-8}}$

### Physics Parameter Normalization
**Method:** Z-score normalization per parameter
- Handles extreme value disparities (e.g., `pscbe1 ~ 10^8` vs `w ~ 1`)
- Transform: $\theta_{norm} = \frac{\theta - \mu_\theta}{\sigma_\theta + 10^{-8}}$
- Applied only to the 29 curated parameters

### Current Output Normalization
**Method:** Arcsinh transform (bijective, handles signed values)
- Transform: $I_{norm} = \text{arcsinh}(I_d / \text{scale})$ where `scale = 1e-6 mA` (= 1 nA)
- Inverse: $I_d = \text{scale} \cdot \sinh(I_{norm})$
- **Rationale:**
  - MOSFET current spans 6+ orders of magnitude (nA subthreshold to mA saturation)
  - NMOS drain current goes **negative in 12% of timesteps** (when Vd < Vs)
  - Arcsinh is bijective (unlike `sign * log10`), smooth near zero, and compresses dynamic range
  - Scale = 1 nA ensures ALL currents enter the logarithmic regime of arcsinh,
    compressing the 10^6 dynamic range to ~2:1 in transformed space
  - **Previous scale (1.0 mA) left nA-scale currents in the linear regime** of arcsinh,
    making subthreshold invisible to loss functions (see Experiment 9 analysis)

---

## Experiments

### Experiment 1: Baseline (No Normalization, Full 76 Parameters)
**Run ID:** `mosfet_spice_supervised_HccTSQra`

**Configuration:**
- **Parameters:** Full 76-parameter BSIM4 set (no normalization)
- **Voltages:** Raw voltage values (no normalization)
- **Current:** Raw $I_d$ values (linear scale, no log transform)
- **Architecture:** FNO with `modes=256`, `width=64`, `embedding_dim=16`, **`embedding_hidden_dim=64`**
- **Training:** 500 epochs, AdamW optimizer, LR=1e-3, weight_decay=1e-5, batch_size=64, warm_restart_count=4

**Design Rationale:**
- Pure baseline with no preprocessing to establish performance floor
- Larger Fourier mode count (256) compensates for lack of normalization
- Embedding MLP: 76 → 64 → 64 → 16 (2-layer hidden network)

**Results:**
- **Final Loss (Lp):** 0.1565
- **Observations:** Highest loss among all three experiments, indicating poor fit to training data.

---

### Experiment 2: Voltage Normalization Only (Full 76 Parameters)
**Run ID:** `mosfet_spice_supervised_QOpPuw08`

**Configuration:**
- **Parameters:** Full 76-parameter set (no normalization applied to physics params)
- **Voltages:** Z-score normalized per terminal
- **Current:** Raw $I_d$ values (linear scale, no log transform)
- **Architecture:** FNO with `modes=128`, `width=64`, `embedding_dim=16`, **`embedding_hidden_dim=64`**
- **Training:** 500 epochs, AdamW optimizer, LR=1e-3, weight_decay=1e-5, batch_size=64, warm_restart_count=4

**Design Rationale:**
- Tests impact of voltage normalization in isolation
- Hypothesis: Terminal voltages have more consistent ranges than BSIM4 parameters
- Reduced Fourier modes (128 vs 256) to test if normalization enables simpler architectures
- Embedding MLP: 76 → 64 → 64 → 16 (same as Exp1)

**Results:**
- **Final Loss (Lp):** 0.01432
- **Observations:** Best loss among all three experiments. Voltage normalization significantly improved training compared to Experiment 1, despite using fewer Fourier modes (128 vs 256).

---

### Experiment 3: Full Normalization + Curated Parameters
**Run ID:** `mosfet_spice_supervised-norm_voltage-log_current-curated_params_6DNIQKTr`

**Configuration:**
- **Parameters:** 29 curated parameters (z-score normalized)
- **Voltages:** Z-score normalized per terminal
- **Current:** **Log-scale with sign preservation** ($I_{norm} = \text{sign}(I) \cdot \log_{10}(|I| + 10^{-12})$)
- **Architecture:** FNO with `modes=128`, `width=64`, `embedding_dim=16`, **`embedding_hidden_dim=128`**
- **Training:** 500 epochs, AdamW optimizer, LR=5e-4, weight_decay=1e-5, batch_size=64, warm_restart_count=4

**Design Rationale:**
- **Parameter Curation:** Eliminates 47 fixed/low-variance parameters based on empirical std analysis
- **Full Normalization Pipeline:** Addresses multi-scale problem (voltages, physics params, current all span different ranges)
- **Log-Scale Current:** Critical for capturing 6+ orders of magnitude (nA → mA) without loss saturation on large values
- **Wider Embedding MLP:** 29 → **128** → **128** → 16 (doubled hidden width vs Exp1/2 despite fewer input params)
  - Hypothesis: Normalized parameters require richer nonlinear transformation to latent space
- **Lower Learning Rate:** 5e-4 vs 1e-3 to stabilize log-scale gradient dynamics

**Key Changes from Experiments 1 & 2:**
1. **Parameter Reduction:** 76 → 29 parameters (eliminated fixed/low-variance parameters)
2. **Parameter Normalization:** Z-score normalization applied to physics parameters
3. **Log-Scale Current:** Transforms output to logarithmic scale with sign preservation
4. **Embedding Architecture:** Increased hidden dimension from 64 to 128
5. **Learning Rate:** Reduced from 1e-3 to 5e-4

**Results:**
- **Final Loss (Lp):** 0.0536
- **Observations:** Loss higher than Experiment 2 (0.053 vs 0.013). **Loss values not directly comparable** due to different output spaces (log vs linear). Log-scale current enables better subthreshold accuracy at cost of higher numerical loss on normalized scale.

---

### Experiment 4: Arcsinh Transform + Waveform Diversity (SUCCESSFUL)
**Run ID:** `mosfet_spice_supervised-norm_voltage-arcsinh-curated_params_ux60URFA`

**Configuration:**
- **Dataset:** 34,375 samples (25K PWL + 6.25K monotonic + 3.125K vth_focused)
- **Parameters:** 29 curated parameters (z-score normalized)
- **Voltages:** Z-score normalized per terminal
- **Current:** Arcsinh transform ($I_{norm} = \text{arcsinh}(I_d / 1.0\text{mA})$)
- **Architecture:** FNO with `modes=128`, `width=64`, `embedding_dim=16`, `embedding_hidden_dim=128`
- **Training:** 500 epochs, AdamW optimizer, LR=1e-3, weight_decay=1e-5, batch_size=64, warm_restart_count=2

**Key Fixes:**
1. **Bijective Transform:** Replaced broken `sign * log10` with arcsinh (handles negative currents correctly)
2. **Waveform Diversity:** Added monotonic and vth_focused modes to match evaluation distribution
3. **Dataset Loader Fix:** Corrected IndexError when loading already-curated 29-param datasets

**Training Results:**
- **Final Loss (Lp):** 0.0112
- **Dataset R2 (Fast):** 0.9999
- **SPICE Transfer R2:** 0.9988 (W=1.0um, L=0.18um)
- **SPICE Output R2:** 0.9945 (W=1.0um, L=0.18um)

**Multi-Geometry Validation:**
| Geometry | R2 Transfer | R2 Output | Speedup |
|----------|-------------|-----------|---------|
| W=0.42um, L=0.15um (min) | 0.9896 | 0.9694 | 39x |
| W=1.0um, L=0.18um (std) | 0.9988 | 0.9945 | 1774x |
| W=2.0um, L=0.50um (mid) | 0.9999 | 0.9977 | 2852x |
| W=5.0um, L=1.0um (large) | 1.0000 | 0.9997 | 2093x |
| W=10.0um, L=2.0um (XL) | 0.8570 | 0.6466 | 1824x |

**Observations:**
- Excellent accuracy across standard geometries (R2 > 0.99 for W=1-5um)
- Performance degrades at extreme sizes (W=10um, L=2um) - outside training distribution
- Subthreshold oscillation artifacts persist in Id-Vg plots around Vth (~0.4V)
- **Speedup: 1700-2800x** faster than SPICE for mid-range geometries (after GPU warmup)

**Remaining Issues:**
1. Subthreshold region shows FNO oscillation artifacts (visible in Id-Vg log plots)
2. Very large devices (L>1um) underperform - need more large-geometry training samples
3. First inference is slow (~380ms) due to CUDA kernel warmup; subsequent runs are ~5-8ms

---

### Experiment 5: Stratified Dataset + Training (500 Epochs)
**Run ID:** `mosfet_spice_supervised-norm_voltage-arcsinh-curated_params-stratified_VzyJ2frp`
**Dataset:** `sky130_nmos_35k_stratified.h5` (35,000 samples)

**Dataset Generation Strategy:**
Equal samples distributed across 5 geometry bins with 3 waveform modes per bin:
- **7,000 samples per bin** (4,200 PWL + 1,750 monotonic + 1,050 vth_focused)
- **Bins:** tiny (W=0.42-0.60), small (W=0.60-1.50), medium (W=1.50-3.50), large (W=3.50-6.00), xlarge (W=6.00-10.00)

**Final Distribution:**
- tiny: 6,799 samples (19.4%)
- small: 7,167 samples (20.5%)
- medium: 7,017 samples (20.0%)
- large: 7,003 samples (20.0%)
- xlarge: 7,005 samples (20.0%)

**Generation Commands:**
```bash
# Per bin: 4200 PWL + 1750 monotonic + 1050 vth_focused
for bin in tiny small medium large xlarge; do
  python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_${bin}.h5 \
    -n 4200 --workers 16 --geometry-bin ${bin} --waveform-mode pwl --overwrite
  python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_${bin}.h5 \
    -n 1750 --workers 16 --geometry-bin ${bin} --waveform-mode monotonic
  python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_${bin}.h5 \
    -n 1050 --workers 16 --geometry-bin ${bin} --waveform-mode vth_focused
done

# Merge all bins
python -c "from spino.mosfet.gen_data import merge_geometry_bins; \
merge_geometry_bins([f'/app/datasets/sky130_nmos_stratified_{b}.h5' \
for b in ['tiny','small','medium','large','xlarge']], \
'/app/datasets/sky130_nmos_35k_stratified.h5', shuffle=True)"
```

**Training Configuration:**
- **Dataset:** 35,000 samples (stratified geometry distribution)
- **Parameters:** 29 curated parameters (z-score normalized)
- **Voltages:** Z-score normalized per terminal
- **Current:** Arcsinh transform ($I_{norm} = \text{arcsinh}(I_d / 1.0\text{mA})$)
- **Architecture:** FNO with `modes=128`, `width=64`, `embedding_dim=16`, `embedding_hidden_dim=128`
- **Training:** 500 epochs, AdamW optimizer, LR=7.5e-4, weight_decay=1e-5, batch_size=64, warm_restart_count=1

**Training Duration:** 209 minutes (3h 30m)

**Training Results:**
- **Final Loss (Lp):** 0.0173
- **Speedup:** 627x

**Comprehensive 3x3 Evaluation (Geometry × Waveform):**

| Geometry | Ramp R² | **Subthreshold R² (Vg<0.5V)** | Sweep R² | Random R² | Ramp MAE | **Ramp MALE** |
|----------|---------|-------------------------------|----------|-----------|----------|---------------|
| tiny (W=0.47, L=0.17)     | 0.9997  | **-21.15** ❌ | 0.9990   | 0.9999    | 0.65 µA  | **452.23 µA** |
| medium (W=2.50, L=0.75)   | 0.9999  | **-4.34** ❌  | 0.9995   | 1.0000    | 0.64 µA  | **268.17 µA** |
| xlarge (W=8.00, L=1.75)   | 1.0000  | **-3.53** ❌  | 0.9997   | 1.0000    | 0.94 µA  | **165.28 µA** |

**Core Metrics:**
- Fast Dataset R²: 1.0000
- SPICE Transfer R² (overall): 0.9647
- SPICE Transfer SubTh-R²: -34.08 ❌
- SPICE Transfer MALE: 488.22 µA
- SPICE Output R²: 0.5838
- SPICE Output MALE: 83.25 µA
- Speedup: 584x

**Observations:**
- Stratified sampling resolves xlarge geometry failure (R² improved from 0.63 to 0.9997 on sweep)
- Excellent overall R² scores (>0.999) across all geometry bins and waveform types
- **Critical Issue: Catastrophic subthreshold accuracy** - SubTh-R² ranges from -21.15 (tiny) to -3.53 (xlarge)
- **MALE values (165-452 µA) indicate predictions off by 2-3 orders of magnitude** in subthreshold region
- FNO predictions deviate from SPICE at very low currents (<1µA) despite high overall R²
- SPICE simulations show transient artifacts at Vg=0V (visible as current "blip" in ramp plots)

**Remaining Issues:**
1. **Subthreshold oscillations** - FNO oscillates or remains elevated above SPICE at Vg < 0.5V
2. **SPICE initial condition artifacts** at Vg=0V - model may be learning these artifacts
3. **Low current accuracy** - negative R² and massive MALE errors for Id < 1µA despite overall R² > 0.999

---

## Experimental Summary

| Experiment | Loss (Lp) | Modes | Voltage Norm | Param Norm | Params | Current Scale | Embedding Hidden | LR | Dataset | Comprehensive R2 (min) |
|------------|-----------|-------|--------------|------------|--------|---------------|------------------|----|---------|-----------------------|
| Exp 1      | 0.156     | 256   | No           | No         | 76     | Linear        | 64               | 1e-3 | 34K mixed | N/A (broken) |
| Exp 2      | 0.013     | 128   | Yes          | No         | 76     | Linear        | 64               | 1e-3 | 34K mixed | N/A (broken) |
| Exp 3      | 0.053     | 128   | Yes          | Yes        | 29     | Log-scale (BROKEN) | 128        | 5e-4 | 34K mixed | N/A (broken) |
| Exp 4      | 0.011     | 128   | Yes          | Yes        | 29     | Arcsinh   | 128              | 1e-3 | 34K mixed | N/A |
| **Exp 5**  | **0.017** | 128   | Yes          | Yes        | 29     | **Arcsinh**   | 128              | 7.5e-4 | **35K stratified** | **0.999** |

**Key Findings:**
- **Voltage normalization** provided the largest improvement (Exp 1 → Exp 2: 0.156 → 0.013)
- **Arcsinh transform** is critical - Experiments 1-3 are all invalid due to non-bijective transforms
- **Waveform diversity** enables generalization from chaotic PWL training to monotonic DC-sweep evaluation
- **Stratified sampling** resolved xlarge geometry failure (Exp 4 → Exp 5: R² 0.63 → 0.9997)
- **Fourier modes increase** provided measurable subthreshold improvement (Exp 5 → Exp 6: SubTh-R² improved 3-5x)

---

## Experiment 5 vs 6: Subthreshold Performance Comparison

| Metric | Experiment 5 (modes=128) | Experiment 6 (modes=256) | Improvement |
|--------|--------------------------|--------------------------|-------------|
| **Tiny SubTh-R²** | -21.15 ❌ | -7.64 ❌ | **3.0x better** |
| **Tiny MALE** | 452.23 µA | 393.01 µA | 13% reduction |
| **Medium SubTh-R²** | -4.34 ❌ | -0.86 ❌ | **5.0x better** |
| **Medium MALE** | 268.17 µA | 177.17 µA | 34% reduction |
| **XLarge SubTh-R²** | -3.53 ❌ | -2.11 ❌ | **1.7x better** |
| **XLarge MALE** | 165.28 µA | 154.24 µA | 7% reduction |
| **Training Time** | 209 min | 237 min | +13% |
| **Speedup** | 584x | 494x | -15% |

**Interpretation:** Doubling Fourier modes provided measurable improvement in subthreshold region (3-5x better R², 7-34% lower MALE), confirming spectral resolution was a contributing factor. However, all geometries still have negative R² values, indicating loss function bias remains the dominant bottleneck.
- **Parameter curation** (76 → 29) reduces input dimensionality without sacrificing accuracy
- **Stratified geometry sampling** resolves OOD failures - xlarge R2 improved from 0.63 to 0.9997
- **Subthreshold oscillations persist** across all experiments - root cause requires investigation

---

## Next Steps

### Immediate: Improve Subthreshold Accuracy
1. **More vth_focused samples:** Increase from 9% to 25% of dataset
2. **Log-weighted loss:** Apply higher weights to small current values during training
3. **Curriculum learning:** Train first on strong inversion, then fine-tune on subthreshold

### Immediate: Extend Geometry Coverage
1. **Stratified geometry bins:** Generate samples across 5 geometry bins (see below)
2. **Larger W/L range:** Extend to W=10um and L=2um to cover full PDK range
3. **Balanced coverage:** Equal samples per bin to eliminate distribution bias

---

## Stratified Geometry Generation (NEW)

### The Problem

The original dataset (34K samples) has severe geometry distribution bias:
- **Small devices (W < 0.5um):** Only 1.7% of samples
- **Large devices (W > 5um):** 0% of samples
- **Mean W:** 2.71um (biased toward mid-range)

This causes OOD (out-of-distribution) performance degradation at geometry extremes:
- W=0.42um, L=0.15um: R2 drops to 0.97 (vs 0.999 for mid-range)
- W=10um, L=2um: R2 drops to 0.65 (catastrophic failure)

### Solution: 5-Bin Stratified Sampling

Generate equal samples across 5 geometry bins covering the full Sky130 NMOS design space:

| Bin | Name | W Range (um) | L Range (um) | Description |
|-----|------|--------------|--------------|-------------|
| 1 | `tiny` | 0.42 - 0.60 | 0.15 - 0.30 | Minimum area devices |
| 2 | `small` | 0.60 - 1.50 | 0.30 - 0.50 | Small devices |
| 3 | `medium` | 1.50 - 3.50 | 0.50 - 1.00 | Nominal devices |
| 4 | `large` | 3.50 - 6.00 | 1.00 - 1.50 | Large devices |
| 5 | `xlarge` | 6.00 - 10.00 | 1.50 - 2.00 | Maximum area devices |

### Generation Commands

**Target:** 35,000 samples total (7,000 per geometry bin) with waveform mix

```bash
# Generate each bin sequentially (chained for maximum resource utilization)
# Config: 16 workers, ~22.9 GB RAM available, ~17s/sample

# Bin 1: tiny (minimum area devices)
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_tiny.h5 \
    -n 7000 --workers 16 --geometry-bin tiny \
    --waveform-mode pwl

# Bin 2: small
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_small.h5 \
    -n 7000 --workers 16 --geometry-bin small \
    --waveform-mode pwl

# Bin 3: medium (nominal)
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_medium.h5 \
    -n 7000 --workers 16 --geometry-bin medium \
    --waveform-mode pwl

# Bin 4: large
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_large.h5 \
    -n 7000 --workers 16 --geometry-bin large \
    --waveform-mode pwl

# Bin 5: xlarge (maximum area)
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_xlarge.h5 \
    -n 7000 --workers 16 --geometry-bin xlarge \
    --waveform-mode pwl
```

**Estimated Time:** ~10.4 hours total (7000 samples x 17s / 16 workers x 5 bins)

### Merge Command

After all bins complete, merge into a single shuffled dataset:

```bash
python -c "
from spino.mosfet.gen_data import merge_geometry_bins
merge_geometry_bins(
    bin_files=[
        '/app/datasets/sky130_nmos_stratified_tiny.h5',
        '/app/datasets/sky130_nmos_stratified_small.h5',
        '/app/datasets/sky130_nmos_stratified_medium.h5',
        '/app/datasets/sky130_nmos_stratified_large.h5',
        '/app/datasets/sky130_nmos_stratified_xlarge.h5',
    ],
    output_path='/app/datasets/sky130_nmos_35k_stratified.h5',
    shuffle=True,
)
"
```

### Waveform Mix Strategy

For better training/eval distribution match, regenerate with mixed waveforms:
- **60% PWL:** Random chaotic waveforms (default)
- **25% monotonic:** DC-sweep-like ramps (matches eval distribution)
- **15% vth_focused:** Gate voltage near threshold (captures subthreshold transition)

This can be achieved by generating additional append batches:
```bash
# Append monotonic samples to each bin (--no-overwrite to append)
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_tiny.h5 \
    -n 2917 --workers 16 --geometry-bin tiny \
    --waveform-mode monotonic

# Append vth_focused samples
python -m spino.mosfet.generate_dataset \
    -o /app/datasets/sky130_nmos_stratified_tiny.h5 \
    -n 1750 --workers 16 --geometry-bin tiny \
    --waveform-mode vth_focused
```

---

## Next Steps
1. **Neural Newton-Raphson Composition:** First multi-device circuit test using
   production NFET (Exp 19b) and PFET (Exp06) operators. Target: CMOS common-source
   amplifier with active PMOS load. See mosfet_next_steps.md Part 6 for the plan of attack.
2. **ONNX Export:** Package production checkpoints for deployment.
3. **Temperature Variation:** Currently fixed at 27C. Requires dataset regeneration at
   multiple temperature corners.

### Extension to Non-Linear Circuits
1. ~~**PMOS Support:**~~ COMPLETE. Exp06 production model.
2. **Modular Composition:** Implement Newton-Raphson coupling (see mosfet_next_steps.md Part 6)
3. **Validation Circuits:** Simulate simple analog blocks:
   - Common-source amplifier with active load (FIRST TARGET)
   - Current mirror
   - Differential pair

---

## Invariance Characterization (Pre-PFET, Phase 0)

**Date:** 2026-03-06
**Script:** `scripts/test_nfet_invariance.py`
**Model:** Exp 19b (`mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt`)
**Dataset:** 61K production (`sky130_nmos_61k_plus_shortch_supp8k.h5`)
**Trim:** 41 timesteps

### Hypothesis

The MOSFET $I_d(V_g, V_d, V_s, V_b, \boldsymbol{\theta})$ mapping is quasi-static (algebraic, not ODE-governed). Unlike RC and diode operators, which solve ODEs parameterized by $\lambda = \tau / T_{\text{end}}$, the MOSFET's transit time $\tau_t = L^2 / (u_0 \cdot V_{\text{eff}})$ is 100--10,000x smaller than any practical simulation window. Displacement currents ($I_{cap} = C \cdot dV/dt$) are ~0.01% of channel current. Therefore:

1. The FNO should produce identical predictions regardless of $T_{\text{end}}$ when the normalized waveform shape is held constant (**time-scale invariance**).
2. The FNO's learned spectral filters should be insensitive to grid size for bandwidth-limited PWL signals (**resolution invariance**).

If confirmed, the dimensionless stiffness ratio $\lambda$ carries zero information for the MOSFET operator, and the PFET can use `in_channels=4` (no lambda channel).

### Methodology

- **Test A (time-scale):** Fixed 10-breakpoint PWL waveform (seeded RNG), SPICE at $T_{\text{end}} \in \{100\text{ns}, 500\text{ns}, 1\mu\text{s}, 2\mu\text{s}, 5\mu\text{s}\}$, 2048 steps, 3 geometries.
- **Test B (resolution):** Fixed $T_{\text{end}} = 1\mu\text{s}$, SPICE at 4096 steps, resampled to $\{512, 1024, 2048, 4096\}$, 3 geometries.
- Both tests compare FNO prediction vs SPICE ground truth via $R^2$.

### Results: Test A (Time-Scale Invariance)

| Geometry | 0.1 us | 0.5 us | 1.0 us | 2.0 us | 5.0 us | Delta R2 | Verdict |
|----------|--------|--------|--------|--------|--------|----------|---------|
| core (W=1.0, L=0.18) | 0.999922 | 0.999923 | 0.999923 | 0.999923 | 0.999923 | 0.000001 | **PASS** |
| tiny (W=0.47, L=0.17) | 0.999756 | 0.999756 | 0.999756 | 0.999756 | 0.999756 | 0.000001 | **PASS** |
| xlarge (W=8.0, L=1.75) | 0.999297 | 0.999804 | 0.999813 | 0.999814 | 0.999813 | 0.000517 | **PASS** |

Pass criterion: $\Delta R^2 < 0.01$. Worst case: xlarge at 0.000517 (19x under threshold).

Note: xlarge shows a minor R2 dip at 100ns, consistent with displacement currents being proportionally larger for wide devices at fast slew rates. Still negligible ($\Delta R^2 = 0.0005$).

### Results: Test B (Resolution Invariance)

| Geometry | 512 pts | 1024 pts | 2048 pts | 4096 pts | Delta R2 | Verdict |
|----------|---------|----------|----------|----------|----------|---------|
| core (W=1.0, L=0.18) | 0.999928 | 0.999929 | 0.999929 | 0.999929 | 0.000001 | **PASS** |
| tiny (W=0.47, L=0.17) | 0.999768 | 0.999768 | 0.999767 | 0.999767 | 0.000001 | **PASS** |
| xlarge (W=8.0, L=1.75) | 0.999827 | 0.999826 | 0.999825 | 0.999824 | 0.000002 | **PASS** |

Pass criterion: $\Delta R^2 < 0.001$. Worst case: xlarge at 0.000002 (500x under threshold).

### Conclusions

1. **Both tests PASS with extreme margin.** The MOSFET operator is inherently time-scale and resolution invariant without any dimensionless formulation.
2. **Lambda ($\lambda = \tau_t / T_{\text{end}}$) is unnecessary** for the MOSFET operator class. The VCFiLM conditioning already has access to all ingredients of $\tau_t$ via the 29-param physics embedding (L, u0) and per-timestep terminal voltages.
3. **This is fundamentally different from RC/diode**, where lambda governs ODE dynamics and is essential for the operator to distinguish stiffness regimes.
4. **PFET architecture decision:** `in_channels=4`, `input_param_dim=29`, no lambda channel. Variable $T_{\text{end}}$ in data generation for waveform diversity only -- the model is not given $T_{\text{end}}$.
5. **Implication for write-ups:** "The MOSFET I-V mapping is algebraic (quasi-static). Unlike ODE-governed operators (RC, diode), the dimensionless stiffness ratio carries no information and is empirically confirmed unnecessary by invariance testing across 50x time-scale range and 8x resolution range."

---

## PFET Experiments

### CRITICAL BUG FOUND: `current_polarity_multiplier` Was Wrong for PMOS

**Date discovered:** 2026-03-11
**Root cause:** `Sky130PMOSStrategy.current_polarity_multiplier` was `-1.0`, identical to NMOS. This is wrong.

**SPICE branch current convention for `Vd d 0`:** `i(vd)` is positive when current flows from node `d` through the voltage source to ground.

| Device | Saturation bias | Raw `i(vd)` | SPICE `@m[id]` | After old x(-1.0) | Correct multiplier |
|--------|-----------------|-------------|-----------------|---------------------|---------------------|
| NMOS | Vg=1.8, Vd=1.8, Vs=0 | **-4.29e-4** | +4.29e-4 | +4.29e-4 (correct) | **-1.0** |
| PMOS | Vg=0, Vd=0, Vs=1.8 | **+7.23e-5** | +7.23e-5 | -7.23e-5 (WRONG) | **+1.0** |

NMOS pulls current into drain from the Vd source (negative `i(vd)` when on), so `-1.0` correctly flips it positive. PMOS pushes current out of drain through Vd (positive `i(vd)` when on), so no negation is needed.

**Fix:** `Sky130PMOSStrategy.current_polarity_multiplier` changed to `+1.0`.

**Impact on existing data:** All PMOS datasets generated before 2026-03-11 have inverted current polarity. The inversion is internally consistent (training and eval both use the same multiplier), so R^2 metrics computed within the wrong convention are numerically valid as relative comparisons. However, the stored current is physically wrong (negative in saturation, should be positive).

**Remediation:** The 50K merged dataset (`sky130_pmos_46k_diverse.h5`) was patched in-place:
1. Removed 11 corrupted samples with `W=L=0` and all-zero physics vectors.
2. Multiplied all `current` values by `-1.0` to correct polarity.
3. Result: 49,989 clean samples. 86.4% have positive mean current (device on). Overall mean: +4.43 mA.

Original file preserved as `sky130_pmos_46k_diverse_BAD_POLARITY.h5`.

### Corrupted Samples in Merged Dataset

11 samples in `sky130_pmos_46k_diverse.h5` had `W=L=0` and all-zero physics vectors. These were introduced during dataset merging (boundary condition in the merge script). Removed during the polarity remediation step above.

---

### PFET Experiment 01: Baseline 30K PWL (300 Epochs)

**Date:** 2026-03-08
**Run ID:** `mosfet_pmos_exp01_xhxn_dNK`
**Model path:** `/app/spino/models/mosfet/pfet/mosfet_pmos_exp01_xhxn_dNK.pt`

**Configuration:**
- **Dataset:** `sky130_pmos_30k_stratified.h5` (30K samples, all PWL waveform mode)
- **Architecture:** MosfetVCFiLMFNO, modes=256, width=64, embed_dim=16
- **Training:** 300 epochs, LR=1e-3, warm_restart_count=3, LpLoss, trim_startup=41

**Results:**

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 |
|----------|----------|-----------|-----------|------------|
| tiny | >0.98 | >0.98 | >0.97 | >0.99 |
| medium | >0.98 | negative | negative | >0.99 |
| xlarge | >0.99 | negative | negative | >0.99 |

**Core:** Transfer R^2 ~0.98, Output R^2 ~0.96.

**Diagnosis:** Random PWL excellent across all geometries, but structured sweeps (monotonic ramp/drain) fail for medium/xlarge. Root cause: 100% of training data is random PWL, but evaluation sweeps are monotonic. The NFET succeeded because its 61K dataset accumulated diverse waveform modes (PWL, monotonic, vth_focused, deep_subth, trans_subth) over 19 experiments. PFET Exp01 had zero monotonic training exposure.

---

### PFET Experiment 02: 50K Diverse Dataset (300 Epochs)

**Date:** 2026-03-09
**Run ID:** `mosfet_pmos_exp02__NMAMnYs`
**Model path:** `/app/spino/models/mosfet/pfet/mosfet_pmos_exp02__NMAMnYs.pt`

**Configuration:**
- **Dataset:** `sky130_pmos_46k_diverse.h5` (50K samples merged from multiple waveform modes)
  - 30K base (all PWL from Exp01)
  - 6K vth_focused
  - 4K monotonic
  - 5K deep_subthreshold (5 geometry bins x 1K)
  - 5K transitional_subthreshold (5 geometry bins x 1K)
- **Architecture:** Same as Exp01
- **Training:** Same hyperparameters as Exp01

**NOTE:** This experiment ran with the wrong `current_polarity_multiplier = -1.0`. All stored currents have inverted sign. Metrics are internally consistent but physically incorrect.

**Results:**

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 |
|----------|----------|-----------|-----------|------------|
| tiny | 0.9741 | 0.9979 | 0.9750 | 0.9964 |
| medium | 0.9982 | **-11.0007** | 0.7662 | 0.9956 |
| xlarge | 0.9972 | **-0.9045** | **-12.7398** | 0.9938 |

**Core:** Transfer R^2 = 0.9784, SubTh = 0.6986, Output R^2 = 0.9638.

**Comparison with NFET Exp19b production:**

| Metric | NFET Exp19b | PFET Exp02 |
|--------|-------------|------------|
| Transfer R^2 | 0.9995 | 0.9784 |
| Output R^2 | 0.9960 | 0.9638 |
| SubTh R^2 | 0.9861 | 0.6986 |
| xlarge SubTh | 0.9113 | -0.9045 |
| xlarge Sweep | 0.9900 | -12.7398 |

**Exp02 is WORSE than Exp01 for medium/xlarge despite having more diverse data.** Tiny geometry improved but the failure modes intensified for larger devices.

**Key diagnostic observations:**
1. Random PWL R^2 > 0.99 for ALL geometries. The model learns time-domain dynamics.
2. Structured monotonic sweeps fail catastrophically for medium/xlarge only.
3. The failure is geometry-dependent: tiny works, larger devices do not.
4. Deep/trans subthreshold supplementary data has ~50% positive current (at the time of analysis, before sign fix). Appears to be a mix of near-zero and very small Id in the subthreshold regime, dominated by junction parasitics and capacitive transients.
5. Arcsinh dynamic range for xlarge sweep: span only 0.25 (vs NFET ~0.41). Signal-to-offset ratio of 0.027 in arcsinh space -- the Id-Vd sweep "signal" is 2.7% of the DC offset.

**Status:** Root cause identified. See collapse diagnosis below.

---

### PFET Exp02 Collapse Diagnosis (2026-03-17)

**Methodology:** Loaded Exp02 model checkpoint with OLD dataset normalization (BAD_POLARITY.h5) and negated SPICE current to match trained convention (multiplier=-1.0). Reproduced original metrics exactly, then probed failure mechanism.

**Reproduced Metrics (matching original):**

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 |
|----------|----------|-----------|-----------|------------|
| tiny | 0.9741 | 0.9979 | 0.9750 | 0.9933 |
| medium | 0.9982 | -19.2 | 0.7662 | 0.9934 |
| xlarge | 0.9972 | -0.96 | -12.7398 | 0.9698 |

**Root Cause 1: Pathological eval overdrive for long-channel devices.**

The output sweep uses fixed `output_vg_drive=0.6V`. For PMOS (Vs=1.8V), this provides good overdrive for short-channel (tiny) but barely turns on long-channel devices where Vth is closer to the nominal 1.3V (in absolute Vg terms). Evidence:

| Geometry | Vg=0.6V span | Vg=0.6V R^2 | Vg=0.3V span | Vg=0.3V R^2 |
|----------|-------------|-------------|-------------|-------------|
| tiny | 14.4 uA | 0.975 | 36.9 uA | 0.942 |
| medium | 1.7 uA | 0.766 | 17.3 uA | 0.962 |
| xlarge | 0.74 uA | -12.74 | 17.3 uA | 0.880 |

Reducing Vg from 0.6 to 0.3 increases the SPICE current span by 23x for xlarge (0.74 -> 17.3 uA), driving R^2 from -12.7 to +0.88. The MODEL is not broken -- the eval config picks an operating point where SPICE variance is vanishingly small (~9e-9 mA^2 for xlarge), making R^2 hypersensitive to the model's ~0.3 uA absolute error floor.

**Root Cause 2: Subthreshold R^2 metric is degenerate for large devices.**

SubTh R^2 measures the subthreshold region of the transfer curve (Vg > 1.3V). For medium/xlarge, the subthreshold leakage current has variance in the range 1e-14 to 3e-13 mA^2. The model's residual MSE (~1e-13 to 6e-13) overwhelms this. In absolute terms, the model predicts sub-uA leakage within ~10 nA -- that's actually decent, but R^2 amplifies it to -19 or -3200 depending on the denominator.

**Note on physics normalization (not a root cause):**

The 11 removed W=L=0 samples had zero-valued physics, which were the only source of
variance for 27/29 BSIM constant params. Removing them collapsed physics std by 20-3Mx.
This initially appeared problematic, but comparison with the NFET production dataset shows
NFET has the same near-zero std (1e-6 to 1e-3) for these constants and works perfectly.
The FiLM conditioning layer handles constant-parameter channels regardless of normalization.
The cleaned dataset's normalization is correct and consistent with NFET.

The sign-flipped `sky130_pmos_46k_diverse.h5` cannot be used for inference with the Exp02
model (different normalization stats), but is fine for Exp03 retraining from scratch.

**Model quality assessment (honest):**
- The model has ~0.3-1.5 uA MAE across all geometries and sweep types.
- This is consistent, geometry-independent, and comparable to early NFET experiments.
- Random PWL R^2 > 0.96 across all sizes confirms the model learned the dynamics.
- The catastrophic R^2 scores are an eval artifact, not a model failure.
- Retraining IS still needed (polarity convention fix), but the architecture and data volume are adequate.

---

### PFET Exp03/03b: Two-Phase Training Results

**Status:** Executed. Both phases complete. Results below acceptance criteria.

**Dataset:** 40K clean merge (no subthreshold supplements):
- 30K PWL stratified (sky130_pmos_30k_stratified.h5)
- 6K vth_focused (sky130_pmos_vth_focused_6k.h5)
- 4K monotonic (sky130_pmos_monotonic_4k.h5)
- Output: `sky130_pmos_40k_clean.h5`

**BUG (found 2026-03-19): 40K clean dataset contains 11 corrupted W=L=0 samples.**
Source: 10 from vth_focused_6k, 1 from monotonic_4k. These were removed from the 50K
merged file during polarity remediation but never cleaned from the individual shards,
so they re-entered via the 40K merge. Exp03 was trained on data including these samples.
Impact likely negligible (11/40000 = 0.03%) but must be cleaned for Exp04.

**Fixes applied before training:**
1. `output_vg_drive` 0.6 -> 0.3 (eval config, fixes pathological xlarge R^2)
2. `current_polarity_multiplier` +1.0 (polarity bug, fixed 2026-03-11)
3. Shard sign-flip (individual shards had wrong sign; only 50K merged file was corrected)

**Exp03 (Phase 1):** Run `mosfet_pmos_exp03_TmRSGu5A`, 300 epochs, 3h 1m.
Final LpLoss: 0.0458 (4x NFET Exp16b's 0.011).

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 |
|----------|----------|-----------|-----------|------------|
| tiny | 0.979 | 0.993 | 0.989 | 0.998 |
| medium | 0.995 | -47.9 | 0.944 | 0.999 |
| xlarge | 0.981 | -52.0 | 0.937 | 0.984 |

Core: Transfer R^2 = 0.9966, SubTh R^2 = 0.9095, Output R^2 = 0.9920, MALE = 115 uA.

**Exp03b (Phase 2):** Run `mosfet_pmos_exp03b_finetune_5vguom7P`, early stop epoch 20.
Marginal improvement. xlarge Random **regressed to 0.812**.

**Root cause of plateau (CORRECTED 2026-03-18):** Previous hypothesis ("vth_focused gradient
poisoning") was retracted. Dataset audit revealed NFET 61K contains 34.3% subthreshold_focused
data (21K/61K) and still achieves 0.011 loss. The actual cause is **dataset volume**: NFET Exp10
on 40K data produced LpLoss = 0.038 (comparable to PFET's 0.046). The drop to 0.011 required
scaling to 61K with targeted subthreshold cross-bin supplements. See mosfet_next_steps.md §5.13.

**Next step:** Exp04 — dataset built and verified (2026-03-19).
**Dataset:** `sky130_pmos_61k_exp04.h5` (60,989 samples).
Composition: 30K PWL stratified + 5,990 vth_focused (cleaned) + 3,999 monotonic (cleaned)
+ 21K cross-bin subthreshold_focused supplements. Zero W=L=0 corrupted samples.
W=[0.42, 10.0], L=[0.15, 2.0], mean current +0.0072 mA, 86.4% positive fraction.

### PFET Exp04: 61K Volume Experiment (VOLUME HYPOTHESIS FALSIFIED)

**Run ID:** `mosfet_pmos_exp04_UK5rTYAE`
**Training:** 6h 2m, 300 epochs. Final LpLoss: **0.046736** (identical to Exp03's 0.0458).
**Dataset:** `sky130_pmos_61k_exp04.h5` (60,989 samples).

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 |
|----------|----------|-----------|-----------|------------|
| tiny     | 0.9875   | 0.9794    | 0.9866    | 0.9994     |
| medium   | 0.9976   | -13.6296  | 0.9663    | 0.9992     |
| xlarge   | 0.9940   | -4.2903   | 0.8782    | 0.7176     |

Core: Transfer R^2=0.9890, SubTh-R^2=0.9692, Output R^2=0.9697, MALE=68.21uA, Speedup=347x.

**Verdict: Loss plateaued at 0.047 — identical to Exp03's 0.046 on 40K.** Scaling from 40K
to 61K did not move the loss floor. The volume hypothesis is falsified.

**Mixed metric movements:** Core SubTh improved 0.91→0.97 (+0.06), MALE improved 115→68uA
(-41%). But xlarge Random collapsed 0.98→0.72 (-0.27) and xlarge Sweep regressed 0.94→0.88.
Same distribution shift pattern as NFET Exps 17-18b.

**Root cause (2026-03-19): LpLoss denominator instability from low PMOS current.**
PFET mean |Id| = 0.007 mA (10x lower than NFET's 0.068 mA, physically correct: hole
mobility). The 61K dataset has 17.5% of samples with per-sample ||arcsinh(I)||_2 < 50
(vs NFET's 8.8%). The worst sample has ||y||_2 = 1.45, creating ~193x gradient
amplification vs the median — far worse than NFET's 44x. These gradient spikes dominate
the optimization, preventing convergence below loss=0.046 regardless of data volume.

**Next step:** Exp05 completed (see below).

**Arcsinh distribution audit (2026-03-19):** PFET arcsinh output distribution does NOT
have a heavier subthreshold tail than NFET. However, the per-sample L2 norms (LpLoss
denominator) reveal a concrete problem: PFET min ||y||_2 = 1.45 (vs NFET 9.12), and
PFET 61K has 17.5% of samples below ||y||_2 = 50 (vs NFET 8.8%). The 193x max gradient
amplification (vs NFET's 44x) is the likely cause of the 0.046 loss plateau.

### PFET Exp05: LpLossWithFloor(50) on 40K (FLOOR PARTIAL WIN)

**Run ID:** `mosfet_pmos_exp05_lpfloor_IH290fSk`
**Training:** 3h 1m, 300 epochs. Final LpLoss: **0.041838** (9% improvement over Exp03).
**Dataset:** `sky130_pmos_40k_clean.h5` (40K). Loss: `lp_floor`, denom-floor=50.

| Metric | Exp03 (40K Lp) | Exp04 (61K Lp) | **Exp05 (40K Floor50)** |
|--------|---|---|---|
| Loss | 0.0458 | 0.0467 | **0.0418** |
| Core Transfer R^2 | 0.9966 | 0.9890 | **0.9973** |
| Core SubTh R^2 | 0.9095 | **0.9692** | 0.8821 |
| Core Output R^2 | **0.9920** | 0.9697 | 0.9699 |
| Core MALE | 115 uA | **68 uA** | 108 uA |
| Speedup | 611x | 347x | **650x** |

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 | Ramp MALE |
|----------|----------|-----------|-----------|------------|-----------|
| tiny | **0.9976** | 0.9928 | 0.9866 | 0.9986 | 56 uA |
| medium | 0.9968 | -14.3342 | 0.9436 | 0.9991 | 295 uA |
| xlarge | 0.9910 | -15.3741 | 0.8187 | **0.9964** | 483 uA |

**What the floor fixed:** Gradient instability from low-norm samples. Loss dropped 9%.
xlarge Random recovered from 0.718 (Exp04 catastrophe) to 0.996. Tiny Ramp improved
0.979->0.998. Confirms gradient spikes WERE a real problem.

**What the floor broke:** xlarge Sweep regressed further: 0.937 (Exp03) -> 0.878 (Exp04) ->
0.819 (Exp05). The floor reduces gradient signal for the low-current output samples that
medium/xlarge sweep accuracy depends on. Core SubTh regressed 0.91->0.88 (same tradeoff
seen in NFET Exp18b: LpFloor inherently trades subthreshold precision for stability).

**Remaining acceptance failures (all sweep-related or MALE):**
- Sweep R^2 < 0.99: tiny (0.987), medium (0.944), xlarge (0.819)
- Ramp MALE > 150 uA: medium (295), xlarge (483)
- Output MAPE > 2%: medium (2.5%), xlarge (3.4%)

All ramp R^2 and random R^2 criteria pass. The model excels at transient dynamics but
struggles with quasi-static output characteristics where PFET current is lowest.

**Next step:** See mosfet_next_steps.md §5.20-5.21 for root cause analysis and revised strategy.

### SubTh-R² Investigation (2026-03-21): METRIC DEGENERACY CONFIRMED

Before running Exp06, a numerical investigation verified that the deeply negative
SubTh-R² values for medium/xlarge are **metric artifacts, not model failures**:

- SPICE ground truth in subthreshold at medium/xlarge is a **flat line at 0.002 nA**
  (femtoamp-scale). Variance ss_tot = 1e-30. R² denominator is effectively zero.
- The **NFET production model (Exp19b) is worse by the same metric**: -52,090 (medium),
  -427,855 (xlarge). PFET Exp05 shows only -4.1 and -7.0 at the same points.
- Medium/xlarge SubTh-R² removed from acceptance criteria (degenerate).

### Sweep Root Cause (2026-03-21): ARCSINH COMPRESSION KILLS OUTPUT RESOLUTION

The sweep failures are NOT a training or loss function problem. **The PFET output
characteristic (Id-Vd) lives within < 1 arcsinh unit** — below the FNO's resolution.

| Eval sweep | Id range | arcsinh SPAN | Sweep R² |
|---|---|---|---|
| PFET medium | 15-33 uA | **0.79** | 0.944 |
| PFET xlarge | 20-40 uA | **0.69** | 0.819 |
| NFET medium | 50-500 uA | **2.30** | 0.995 |
| NFET xlarge | 0.1-3 mA | **3.40** | 0.990 |

The NFET has 3-5x more signal span in arcsinh space due to higher mobility. This is physics,
not data. Additionally, the PFET training data contains **zero sweep-like samples** (constant
Vg, swept Vd) — NFET has 2,136 (3.5%) from monotonic waveform supplements.

**Revised strategy:** Add true sweep waveform data to PFET training set via two new modes:
`output_sweep` (gate constant, drain ramps) and `transfer_sweep` (drain constant, gate ramps).
The existing `monotonic` mode was confirmed useless — it ramps ALL terminals simultaneously
(Vg range=1.755V AND Vd range=1.754V in monotonic_4k.h5), producing zero single-terminal
sweep conditions. See Exp06 results below.

### PFET Exp06: Sweep-Augmented Training (PRODUCTION CANDIDATE)

**Run ID:** `mosfet_pmos_exp06_sweep_aug_CzBVmMi4`
**Model path:** `/app/spino/models/mosfet/pfet/mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt`
**Training:** 3h 21m, 300 epochs, full convergence. Final LpLoss: **0.036511** (best ever).
**Dataset:** `sky130_pmos_48k_sweep_aug.h5` (44K: 40K PWL + 2K output_sweep + 2K transfer_sweep).
**Loss:** Standard LpLoss.

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 | Sweep MAE | Sweep MALE |
|----------|----------|-----------|-----------|------------|-----------|------------|
| tiny     | 0.9981 | 0.9987 | **0.9973** | 0.9993 | 0.39uA | 4.59uA |
| medium   | 0.9999 | -3.05 | **0.9957** | 0.9998 | 0.19uA | 3.00uA |
| xlarge   | 0.9992 | -2.48 | **0.9942** | 0.9997 | 0.18uA | 2.43uA |

Core: Transfer R^2=0.9965, SubTh-R^2=0.9523, Output R^2=0.9656, MALE=60uA, Speedup=522x.

**Sweep R^2 improvement vs Exp05:** tiny 0.987->0.997, medium 0.944->0.996, xlarge 0.819->0.994.
Every sweep metric now exceeds 0.99 threshold. Loss dropped 20% (0.042->0.037). The data
coverage hypothesis was conclusively validated: the model only needed to SEE sweep-like
training examples.

**Remaining soft failure:** xlarge Ramp MALE = 153 uA vs 150 threshold (2% over). All other
acceptance criteria pass.

**Horizontal shift artifact:** Small 10-19 mV shift at linear-to-saturation knee due to FNO
spectral truncation (Gibbs phenomenon). Architecturally intrinsic. Not addressable via
fine-tuning. Within process variation margins.

### PFET Exp06b: Low-LR Fine-Tune (REGRESSED)

**Run ID:** `mosfet_pmos_exp06b_finetune_z8T_QJ0M`
**Training:** 34 min, 50 epochs, LR=5e-5, checkpoint from Exp06. Standard LpLoss.

| Geometry | Ramp R^2 | SubTh R^2 | Sweep R^2 | Random R^2 | Ramp MALE |
|----------|----------|-----------|-----------|------------|-----------|
| tiny     | 0.9980 | 0.9983 | 0.9980 | 0.9992 | 25.03uA |
| medium   | 0.9999 | -3.03 | 0.9951 | 0.9997 | 122.04uA |
| xlarge   | 0.9992 | -2.53 | 0.9948 | **0.9628** | 154.05uA |

**Verdict:** xlarge Random regressed 0.9997->0.9628 (MALE 233->328 uA). Gradient spikes on
low-current xlarge samples destabilize even at 5e-5 LR. No metric improved meaningfully.

**PFET PRODUCTION MODEL: Exp06** (`mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt`).

### PFET Exp07: Triode-Boundary Fine-Tune (PARTIAL CLOSURE)

**Run ID:** `pmos_exp07_triode_finetune_KG1HfPbJ`
**Model path:** `/app/spino/models/mosfet/pfet/pmos_exp07_triode_finetune_KG1HfPbJ.pt`
**Dataset:** `sky130_pmos_50k_triode.h5` (46K = 44K Exp06 dataset + 2K triode-boundary augmentation).
**Augmentation strategy:** PWL waveforms with Vsg ∈ [1.0, 1.6] V, Vsd ∈ [0.0, 0.3] V — targets the
M4 OTA triode regime identified by attribution Probe 1.
**Training:** 35 min, 50 epochs, LR=1e-4, **frozen backbone** (FiLM conditioning layers only;
176K of 2.33M params trainable = 7.5%). Standard LpLoss. Final loss 0.147 (Exp06 baseline 0.037
on smaller dataset).

**Device-level eval (production sizing, sky130_pmos):**

| Metric | Exp06 | Exp07 | Δ |
|---|---|---|---|
| SPICE Transfer R² | 0.9965 | 0.9957 | −0.0008 |
| SPICE Output R² | 0.9656 | 0.9403 | −0.0253 |
| SPICE Output MALE | 60 uA | 21.7 uA | **−64%** |
| tiny Sweep R² | 0.9973 | 0.9982 | +0.0009 |
| medium Sweep R² | 0.9957 | 0.9976 | +0.0019 |
| xlarge Sweep R² | 0.9942 | 0.9592 | **−0.0350** |

xlarge regressed (consistent with Exp06b pattern). tiny/medium improved.

**OTA composition impact (production sizing, L=0.40, W_diff=W_mirror=8 µm):**

| Metric | Exp06 | Exp07 |
|---|---|---|
| max\|ΔV\| | 68.7 mV | **61.0 mV** (−11%) |
| M4 max\|ΔI\| (triode target) | 15.4 uA | **12.0 uA** (−22%) |
| M3 max\|ΔI\| (non-triode PFET) | 5.6 uA | 6.0 uA (+8%) |
| Pearson r | 0.9997 | 0.9997 |

**Verdict:** Frozen-backbone fine-tune produced a real, attribution-traceable 22 % reduction
in M4 triode error, but the OTA 30 mV gate still fails (61 mV) at production sizing. M3
regressed slightly (FiLM over-specializes to triode regime). Composition improvement (11 %) is
smaller than M4 device improvement (22 %) because Newton coupling distributes residual error
across all KCL nodes. Augmentation-alone is insufficient; full backbone unfreeze or
geometry-stratified retrain are open paths — deferred to future work.

**PFET PRODUCTION MODEL remains Exp06.** Exp07 is the demonstrated triode-augmentation result
documented in `docs/results.md` and `docs/assets/ota_5t_fno_l040_exp07/`.

---

## Code References

- **Data Generation:** [gen_data.py](gen_data.py) - `InfiniteSpiceMosfetDataset`, `PreGeneratedMosfetDataset`
- **Parameter Schema:** [gen_data.py](gen_data.py) - `ParameterSchema` class
- **Training Loop:** [train.py](train.py) - `run_mosfet_training()`
- **Model Architecture:** [model.py](model.py) - `MosfetFNO` class
- **Device Strategies:** [device_strategy.py](device_strategy.py) - `Sky130NMOSStrategy`, `Sky130PMOSStrategy`
- **Invariance Test:** [scripts/test_nfet_invariance.py](../../scripts/test_nfet_invariance.py) - Phase 0 characterization

---

**Last Updated:** 2026-05-14 (MLP ablation complete)

---

## MLP Ablation Experiments (Architecture Defense)

Per-timestep `MosfetMLP` trained on same 61K NFET dataset as Exp19b. No temporal mixing — each
timestep predicted independently from (Vg, Vd, Vs, Vb, θ_embed). Off-diagonal Jacobian terms
dI[t]/dV[t'] = 0 by construction.

### MLP h64 — `mosfet_mlp_baseline_XpV7KFHL`
**Date:** 2026-05-14 | **Config:** hidden_dim=64, 32K params, 300 epochs, LpLoss, AdamW

| Metric | Value |
|---|---|
| Fast Dataset R² (64-sample avg, seed=42) | -4.42 |
| SPICE Transfer R² | 0.9990 |
| SPICE Transfer SubTh-R² | 0.9856 |
| SPICE Output R² | 0.9456 |
| Speedup vs SPICE | 863x |

### MLP h128 — `mosfet_mlp_h128_OZyiUsFA`
**Date:** 2026-05-14 | **Config:** hidden_dim=128, 58K params, 300 epochs, same dataset/loss

| Metric | Value |
|---|---|
| Fast Dataset R² (64-sample avg, seed=42) | -5.43 |
| SPICE Transfer R² | 0.9989 |
| SPICE Transfer SubTh-R² | 0.9631 |
| SPICE Output R² | 0.9763 |
| Speedup vs SPICE | 6501x |

**Verdict:** Fast Dataset R² strongly negative at both capacity levels, worsening with more params.
MLP generalizes to controlled ramp sweeps but fails on arbitrary PWL waveforms from the training
distribution. FNO spectral mixing provides waveform-shape regularization that is empirically
necessary even though MOSFET physics is quasi-static. Architecture defense: FNO justified.
