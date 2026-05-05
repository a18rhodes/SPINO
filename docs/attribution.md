# Error attribution: L=0.18 CUDA stress geometry

This note records the causal attribution for the **cross-bin stress** common-source
composition run (`L_n = L_p = 0.18 um`, CUDA, archived as `cs_amp_fno_exp2`). The
**`L = 0.40 um` showcase** run is in-bin and already matches SPICE closely; it is
not attributed here.

Method details for the composition stack remain in
[Neural composition: CS amplifier method](composition.md). Aggregate composition
metrics remain in [CS amplifier composition results](results.md).

## Method: four-probe isolation

Attribution proceeds as a fixed sequence so IV-surface error, KCL imbalance,
substitution experiments, and Newton diagnostics are not conflated.

1. **Probe 1 (transient):** FNO vs SPICE drain current along the converged transient
   trajectory (nominal strong-inversion `V_gs` window).
2. **Probe 2 (transient):** KCL residual waveforms with `V_out` pinned to the
   SPICE-converged value vs the FNO-converged value. Large imbalance at the SPICE
   pin with negligible imbalance at the FNO pin isolates wrong branch currents,
   not a broken global Newton loop.
3. **Probe 3 (transient):** Hybrid / full-SPICE substitution runs. With the
   current implicit NR + whole-window routing, transient substitution causal
   closure is **not executable** (see Open items).
4. **Probe 4 (transient):** Per-iteration Newton diagnostics (Jacobian diagonal
   ratio, linear solve residual, line-search `alpha`). Healthy numbers here rule
   out numerical pathology as the primary driver.

A parallel **VTC** track repeats Probe 1 on a DC sweep and performs a **region-wise
IV substitution** (SPICE IV cache in the bad `V_in` band, FNO elsewhere) with
scalar KCL root-finding. That isolates weak-inversion / near-off IV error from the
nominal-bias transient window.

Canonical numbers are in
[`attribution_result.json`](assets/cs_amp_fno_exp2/attribution/attribution_result.json).
Raw probe arrays, IV caches, and full compose-run trees live under
`runs/attribution/cs_amp_fno_exp2/` (gitignored; multi-hour rebuild if deleted).

## Transient attribution

Baseline composition metrics for this geometry match the published stress run
(`transient` Pearson `r`, max `|Delta V|`, `R^2`, etc.; see `results.md`).

**Conclusion:** The transient Newton solver is well-behaved: three iterations,
Jacobian diagonal ratio near `616` (well below a practical ill-conditioning
threshold at `1e6` used in the notebook), full Newton step (`alpha = 1.0`) every
iteration. The dominant effect is **FNO IV error at the nominal bias** (`V_gs`
roughly `0.85`–`0.90` V along the step): at the SPICE-pinned `V_out`, the KCL
residual peaks near **19.6 uA** vs **73 nA** at the FNO-pinned `V_out` (ratio
**268x**). The solver converges to a **wrong IV surface**, not a numerically
unstable solve.

![Transient IV error along trajectory](assets/cs_amp_fno_exp2/attribution/probe1_iv_error.png)

![KCL residual at pinned V_out](assets/cs_amp_fno_exp2/attribution/probe2_kcl_residual_waveform.png)

![Three-run transient comparison](assets/cs_amp_fno_exp2/attribution/probe3_three_run_comparison.png)

![Newton diagnostics](assets/cs_amp_fno_exp2/attribution/probe4_nr_diagnostics.png)

## VTC attribution

**Conclusion:** VTC error is concentrated where both devices operate in weak
inversion / near-off (`V_in < 0.5` V in this sweep). Relative IV error ratios
(bad region vs good region) reach roughly **4500x** (NFET) and **3700x** (PFET,
with per-point capping in the analysis). At `V_in ~ 0.25` V the PFET exhibits a
catastrophic near-off spike: FNO on the order of **13.7 A** vs SPICE **785 pA**.
A **brentq** KCL solve using the SPICE IV cache in the bad region (FNO retained
elsewhere) collapses mean `V_out` error in that band from **176 mV** to **4.4 mV**
(**97.5%** reduction) with **no** measurable effect in the good region
(**0.0%** collapse). That closes causality on weak-inversion IV mismatch with
strong regional specificity.

![VTC IV error vs Vin](assets/cs_amp_fno_exp2/attribution/vtc_probe1_iv_error.png)

![VTC substitution overlay](assets/cs_amp_fno_exp2/attribution/vtc_substitution.png)

![VTC attribution summary](assets/cs_amp_fno_exp2/attribution/vtc_attribution_summary.png)

## Open items

1. **Transient substitution:** The step stimulus stays in strong inversion; a
   narrow weak-inversion hybrid threshold never arms. A broad threshold routes
   the entire implicit window to the SPICE cache and drives NR divergence because
   the FNO Jacobian is inconsistent with cached branch currents. **Probe 2**
   remains the causal evidence for the transient; substitution is blocked on the
   current whole-window NR architecture.
2. **Residual ~4.4 mV** after VTC substitution in the bad band is unexplained
   (likely cache interpolation or a secondary residual). It is small relative to
   the original **176 mV** gap and does not weaken the attribution claim.

## Reproduction

1. Run [`spino/attribution.ipynb`](../spino/attribution.ipynb) top-to-bottom from
   the **repository root** so `DOCS_ROOT` and `RUNS_ROOT` resolve correctly.
2. IV cache generation uses [`scripts/cache_spice_iv.py`](../scripts/cache_spice_iv.py)
   (see notebook Stage 3). Cache `.npz` files under `runs/attribution/cs_amp_fno_exp2/`
   are expensive to rebuild; do not delete them unless the geometry or PDK changes.
3. Composition reproduction commands for this geometry are listed in
   [`docs/results.md`](results.md).
