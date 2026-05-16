# SPINO: Universal Parametric Neural Operators for Differentiable Analog Circuit Simulation

> **Disclaimer:** Experiments and results are verified by the author. All code and
> experiments are original work by the project author, with AI tooling assistance.

---

## Overview

**SPINO** (SPICE Neural Operator) studies whether Fourier Neural Operator
(FNO) device surrogates can be composed inside Newton-Raphson circuit solvers
while preserving end-to-end differentiability. The scientific contribution is
not a blanket replacement for SPICE. It is a differentiable analog-circuit
simulation path: MOSFET operators map terminal voltage waveforms and device
parameters to drain current, then autograd supplies Jacobians for KCL-based
composition.

Composed-circuit validation spans two analog topologies: the single-stage
common-source amplifier (one internal node) and the 5T OTA (three internal nodes).
Both converge reliably in the analog operating region where conductive Jacobians
keep the residual well conditioned. The OTA result demonstrates that the method
scales to multi-node analog circuits without structural changes to the Newton
formulation. This makes the approach most relevant for rapid device sizing,
parameter sweeps, and gradient-based topology search during analog design.

Four device operators have been trained and validated against NGSPICE ground truth:

| Operator | Conditioning | Peak R² | Documentation |
|---|---|---|---|
| Linear RC | Dimensionless $`\lambda`$ | 0.9999 | [RC Circuit](docs/rc.md) |
| Shockley Diode | Dimensionless $`\lambda`$ + direct injection | 0.9999 | [Diode](docs/diode.md) |
| sky130 NMOS | VCFiLM (29-param BSIM) | 0.9995 | [NFET](docs/nfet.md) |
| sky130 PMOS | VCFiLM (29-param BSIM) | 0.9999 | [PFET](docs/pfet.md) |

The work is documented in a set of notes:

- [Neural composition: CS amplifier method](docs/composition.md) — KCL assembly,
  Newton-Raphson solvers, autograd Jacobians, and damping policy.
- [Neural composition: 5T OTA method](docs/ota_composition.md) — multi-node KCL
  formulation, generalized source-terminal displacement, and pre-registered gates.
- [Analog composition results](docs/results.md) — SPICE baselines, composed-fidelity
  metrics, runtime context, attribution, and documented limitations.
- [Gradient-based OTA sizing](docs/sizing.md) — IFT through KCL Newton, Adam loop,
  SPICE-validated final θ.
- [Future work](docs/future_work.md) — global forward roadmap.

SPICE-only characterization methodology is documented in
[CS amplifier characterization](docs/cs_amp.md) and
[5T OTA characterization](docs/ota_5t.md).

The NMOS operator achieves transfer R² = 0.9995 and subthreshold R² = 0.9861 at core geometry
(W = 1.0 µm, L = 0.18 µm).
The PMOS operator uses the same VCFiLM-FNO architecture trained on a sweep-augmented dataset
(40 K random PWL + 4 K deterministic sweeps), achieving transfer R² = 0.9965 and sweep R² >
0.99 across all tested geometries.
The diode operator extends the RC dimensionless framework to the nonlinear Shockley equation,
achieving R² = 0.9994 on a standard rectifier and R² = 0.9999 on adversarial samples,
with validated resolution invariance ($`\Delta R^2 < 0.0001`$ at 1024/2048/4096
steps) and time-scale invariance (R² ≥ 0.997 across $`T_{end}`$ spanning 100 µs to 10 ms).
The RC operator demonstrates that a single trained FNO generalises across the full stiffness
ratio spectrum without per-circuit solver configuration.

### Architecture justification: MLP ablation

The FNO's spectral temporal mixing was not chosen by default. A per-timestep quasi-static MLP
baseline (`MosfetMLP`) was trained on the *same* 61 K NFET dataset using the same loss,
optimizer, and epoch budget, at two capacity levels (32 K and 58 K parameters). On controlled
ramp sweeps — the kind of inputs MOSFET compact models are traditionally validated against —
the MLP matches the FNO at Transfer R² ≥ 0.999. On random PWL waveforms drawn from the
training distribution (the same distribution that drives Newton-Raphson circuit composition),
the MLP collapses to Fast Dataset R² ≈ **−4 to −5**, while the FNO holds **0.99**. The gap
*widens* with MLP capacity (h64: −4.42, h128: −5.43), ruling out underfitting as the cause.
FNO temporal mixing is therefore acting as a waveform-shape regularizer rather than a
physically necessary modelling choice — a structural argument the capacity sweep makes
cleanly without requiring multi-seed variance characterisation. Full table, figures, and
discussion in [Analog composition results — MLP ablation](docs/results.md#mlp-ablation-architecture-defense).

---

## Motivation

SPICE (Simulation Program with Integrated Circuit Emphasis) [1] has been the standard for
circuit verification since 1973. At its core, SPICE solves a system of differential-algebraic
equations (DAEs) by iterating Newton-Raphson at each time step, evaluating compact device
models (BSIM3/4, EKV, etc.) to compute terminal currents and their Jacobians analytically.

This approach faces two scaling pressures in modern design:

1. **Device evaluation cost.** Each Newton iteration evaluates every device model in the
   circuit. For post-layout netlists with millions of parasitic elements, device evaluation
   dominates wall-clock time [2]. Projection-based model reduction [3] alleviates this for
   weakly nonlinear circuits but breaks down for strongly nonlinear device models.
2. **Convergence sensitivity.** Newton-Raphson convergence depends on the quality of initial
   guesses and Jacobian conditioning. Stiff circuits (e.g., switched-capacitor, RF mixers)
   frequently require time-step backtracking, multiplying the number of device evaluations [5].

Physics-informed neural networks (PINNs) [4] and their extensions [7, 8] have demonstrated
that neural surrogates can approximate PDE solutions with physical consistency guarantees.
However, PINNs solve individual problem instances. Neural operators [6] learn the solution
*operator* — mapping from input functions to output functions — enabling a single trained
model to generalize across parameter families without retraining. SPINO applies this operator-
learning paradigm specifically to circuit device models, preserving temporal dynamics
(capacitive charging, subthreshold transients) that are critical for transient simulation
fidelity.

---

## Method Overview

SPINO takes a progressive approach, validating the FNO surrogate concept on increasingly
complex devices:

### 1. Linear RC Circuit

A 1D FNO learns the voltage response $`V(t) = \mathcal{F}(I(t), \lambda)`$ of a first-order
RC circuit in dimensionless form, where $`\lambda = \tau / T_{end}`$ is the stiffness ratio.
Training on the non-dimensionalized ODE makes the operator invariant to physical time
scale: a 100 fs parasitic transient and a 10 s saturation drift are identical if their
stiffness ratios match. Spectral augmentation (white noise, chirp, dense switching) ensures
the operator learns integration rather than memorizing pulse shapes.

### 2. Shockley Diode

The diode introduces exponential nonlinearity via the Shockley equation. Following the same
dimensionless approach as the RC operator, the circuit ODE is reformulated in terms of
$`\hat{t} = t/T_{end}`$, $`\hat{I} = I/I_{scale}`$, $`\hat{V} = V/(I_{scale}R)`$, with the
stiffness ratio $`\lambda = RC/T_{end}`$ injected as a constant channel. This makes the
operator invariant to simulation window and grid resolution. Five circuit parameters
($`\lambda`$, $`R`$, $`C`$, $`I_S`$, $`N`$) are injected directly as constant-valued channels
alongside the normalised current waveform. Log-encoding of parameters spanning 15 orders of
magnitude ($`I_S`$) prevents gradient instability.

### 3. sky130 NMOS (VCFiLM-FNO)

The MOSFET operator scales to 29 BSIM4 parameters via **Variable-Conditioning Feature-wise
Linear Modulation** (VCFiLM). Rather than injecting parameters as channels (which fails
at high dimensionality), the physics vector is projected through a learned MLP embedding
and applied as per-layer FiLM scaling and shifting within the spectral blocks. A two-phase
training strategy (300 epochs full training + 20 epochs frozen-backbone fine-tuning) decouples
waveform representation from geometry conditioning, achieving R² > 0.99 across five geometry
bins while maintaining subthreshold accuracy at nanoampere-scale currents.

### 4. sky130 PMOS (VCFiLM-FNO)

The PMOS operator reuses the VCFiLM-FNO architecture with independently trained weights.
Training uses a sweep-augmented dataset (40 K random PWL + 2 K output sweeps + 2 K transfer
sweeps) in a single 300-epoch phase — the deterministic sweep samples provide sufficient
coverage of the I-V manifold without requiring a frozen-backbone fine-tuning phase. Bias
polarities are handled by the device strategy layer: PMOS operates with $`V_S = V_B = V_{DD}`$
and sweeps gate/drain downward.

---

## Summary of Results

### Accuracy

| Operator | Test Condition | R² | Relative Error |
|---|---|---|---|
| Linear RC | Corner frequency ($`\lambda = 1.0`$) | 0.9999 | < 0.1% |
| Linear RC | White noise (OOD) | 0.9884 | ~1% |
| Shockley diode | Standard rectifier ($`\lambda = 0.01`$) | 0.9994 | ~0.45% |
| Shockley diode | Adversarial (random params) | 0.9999 | < 0.1% |
| sky130 NMOS | Transfer (W=1 µm, L=0.18 µm) | 0.9995 | -- |
| sky130 NMOS | Output (W=1 µm, L=0.18 µm) | 0.9960 | -- |
| sky130 NMOS | Subthreshold (W=1 µm, L=0.18 µm) | 0.9861 | -- |
| sky130 PMOS | Transfer (W=1 µm, L=0.18 µm) | 0.9965 | -- |
| sky130 PMOS | Output (W=1 µm, L=0.18 µm) | 0.9656 | -- |
| sky130 PMOS | Subthreshold (W=1 µm, L=0.18 µm) | 0.9523 | -- |
| CS amp composition (L=0.18, CUDA) | Transient vs SPICE | 0.99748 (Pearson r) | Max \|ΔV\| = 25.78 mV |
| CS amp composition (L=0.40, CUDA) | Transient vs SPICE | 0.99981 (Pearson r) | Max \|ΔV\| = 2.392 mV |
| OTA composition (L=0.40, CUDA) | Transient vs SPICE | 0.9997 (Pearson r) | Max \|ΔV\| = 68.7 mV† |
| OTA composition (L=0.50, CUDA) | Transient vs SPICE | 0.9997 (Pearson r) | Max \|ΔV\| = 68.9 mV† |

†OTA max|ΔV| attributed to M4 PFET output mirror current error at Vds ≈ 0 (triode boundary);
PFET training data underrepresents this regime. Pre-registered gate is ≤ 30 mV; failure
is reported as a documented finding. A targeted triode-boundary fine-tune narrows the gap
(M4 peak |ΔI|: 15.4 → 12.0 µA; composition max|ΔV|: 68.7 → 61.0 mV) but does not close it;
the production checkpoint is unchanged. See
[Analog composition results — PFET triode-boundary fine-tune](docs/results.md#pfet-triode-boundary-fine-tune-partial-gate-closure).

### Runtime context

Runtime is measured and archived, but it is not the central claim. The useful
property of the composed MOSFET flow is differentiability through the circuit
residual, which is the missing primitive for gradient-based analog sizing and
topology search. Circuit-level wall time depends strongly on hardware, dense
autograd Jacobian assembly, and linear-solve cost.

---

## Known Limitations

### Digital switching circuits

The inverter-chain extension is a documented negative result and marks the
current regime boundary. The whole-window FNO formulation does not converge to
acceptance-quality digital trajectories for 1-, 2-, or 4-stage CMOS inverter
chains. The cause is structural rather than a solver-tuning defect: temporal
spectral convolutions produce off-diagonal autograd Jacobian terms
$`dI[t] / dV[t']`$ that have no quasi-static MOSFET interpretation. In analog
gain regions, large conductive diagonals dominate those artifacts. In digital
saturation plateaus, the physical $`dI/dV_{DS}`$ diagonal collapses and the
spurious temporal coupling dominates Newton steps.

The inverter-chain code, tests, and CLI remain in the repository as a
regime-boundary artifact. They characterize where this FNO-composed formulation
currently stops being valid; they are not an active convergence TODO.

### Temporal and Resolution Invariance (MOSFET)

The MOSFET $`I_D(V_G, V_D, V_S, V_B, \boldsymbol{\theta})`$ mapping is **quasi-static**
(algebraic, not ODE-governed). The device transit time
$`\tau_t = L^2 / (\mu_0 \cdot V_{\text{eff}})`$ is 100–10,000× smaller than any practical
simulation window, and displacement currents are ~0.01% of channel current. This means the
dimensionless stiffness ratio $`\lambda = \tau / T_{end}`$ that governs the RC and diode
operators carries **no information** for the MOSFET operator.

The VCFiLM conditioning pathway already has access to all the ingredients of $`\tau_t`$ — gate
length, mobility parameters, and threshold voltage are present in the 29-element BSIM vector,
while per-timestep terminal voltages provide instantaneous $`V_{\text{eff}}`$. The network can
reconstruct an effective time constant internally without an explicit $`\lambda`$ channel.

This was empirically validated on the production NFET FNO across three geometries
(core, tiny, xlarge):

| Test | Variable | Range | Worst $`\Delta R^2`$ | Criterion | Verdict |
|---|---|---|---|---|---|
| Time-scale | $`T_{end}`$ | 100 ns – 5 µs (50×) | 0.000517 | < 0.01 | **PASS** |
| Resolution | Step count | 512 – 4096 (8×) | 0.000002 | < 0.001 | **PASS** |

All $`R^2`$ values remained above 0.999 across the full test matrix. The production PFET FNO
was independently validated with the same methodology:

| Test | Variable | Range | Core/Tiny $`\Delta R^2`$ | xlarge $`\Delta R^2`$ | Criterion | Verdict |
|---|---|---|---|---|---|---|
| Time-scale | $`T_{end}`$ | 100 ns – 5 µs (50×) | ≤ 0.000031 | 0.030 ($`T_{end}`$ = 100 ns outlier) | < 0.01 | **PASS** (core/tiny), **FAIL** (xlarge) |
| Resolution | Step count | 512 – 4096 (8×) | ≤ 0.000056 | 0.0016 | < 0.001 | **PASS** (core/tiny), **FAIL** (xlarge) |

The PFET xlarge failure is isolated to $`T_{end} = 100`$ ns, where R² drops to 0.960 (vs
0.990 at $`\geq`$ 500 ns). The likely cause is parasitic gate capacitance in the large PMOS
device: at extreme short timescales, displacement currents become a non-trivial fraction of
drain current. For practical simulation windows ($`T_{end} \geq 500`$ ns), both operators are
invariant across all geometries.

**This is fundamentally different from the RC and diode operators**, where $`\lambda`$ governs
ODE dynamics and is essential for the operator to distinguish stiffness regimes. For the
MOSFET, the physics is algebraic, and the VCFiLM architecture exploits this automatically.

---

## Composition Method

The composition layer assembles device FNO outputs into multi-node KCL residuals
and solves DC and whole-window transient trajectories with damped Newton-Raphson.
Jacobians are computed through autograd, with Armijo backtracking and voltage-step
safeguards.

Two analog topologies are implemented:
- **CS amplifier** (single internal node): NFET + PFET in a single KCL residual.
  Method details in [Neural composition: CS amplifier method](docs/composition.md).
- **5T OTA** (three internal nodes — n_tail, n_left, n_out): differential pair with
  floating source terminal, PFET current mirror, tail current source.
  Method details in [Neural composition: 5T OTA method](docs/ota_composition.md).

## Composition Results

### CS amplifier

- SPICE characterization at `L=0.18` (cross-bin stress) and `L=0.40` (in-bin showcase).
- CUDA composition runs at both geometries.

Headline outcomes:
- Cross-bin stress (`L=0.18`): transient Pearson `r = 0.99748`, max `|ΔV| = 25.78 mV`.
- In-bin showcase (`L=0.40`): transient Pearson `r = 0.99981`, max `|ΔV| = 2.392 mV`.

The `L=0.18` stress gap is causally decomposed (transient IV surface error vs weak-inversion
VTC failure) in [Error attribution: L=0.18 CUDA stress geometry](docs/attribution.md).

### 5T OTA

- SPICE characterization sweep: 7×7 (W\_diff, W\_mirror) grid at L ∈ {0.40, 0.50} µm.
  Selected design: W\_diff = W\_mirror = 8 µm at both L.
- CUDA composition runs at both geometries.

Headline outcomes:
- `L=0.40`: transient Pearson `r = 0.9997`, slew rate within 1% of SPICE; max `|ΔV| = 68.7 mV`
  (pre-registered gate ≤ 30 mV, fails; attributed to M4 PFET triode-boundary gap).
- `L=0.50`: transient Pearson `r = 0.9997`, slew rate within 5% of SPICE; max `|ΔV| = 68.9 mV`
  (same root cause).
- Newton convergence in ≤ 12 iterations at both L (gate ≤ 25: pass).

Full tables, figures, attribution, and reproduction commands are in
[Analog composition results](docs/results.md).

## Future Work

The global forward roadmap is consolidated in [Future work](docs/future_work.md),
organized around:

1. Improved model accuracy in weak-inversion and near-off regimes. An initial
   targeted data-augmentation pass on the PFET triode boundary reduced M4 IV error
   by 22 % but did not fully close the OTA gate; remaining options (full-backbone
   fine-tune, geometry-stratified retrain, larger augmentation budget) are queued.
2. Multi-stage analog topologies — the two-stage Miller op-amp is the next queued target.
3. Runtime studies: GPU-native Krylov linear solvers (PyTorch Arnoldi, queued for a later iteration),
   cross-hardware reproducibility envelopes, and system-level error budgeting.

---

## Development

For installation, training, evaluation, and figure generation instructions, see the
[Development Guide](docs/DEVELOPMENT.md).

---

## References

The references below ground the motivation and method. For a survey of competing
approaches in neural device surrogates, differentiable circuit simulators, operator
learning, and analog sizing, see [docs/related_work.md](docs/related_work.md).

\[1\] L. W. Nagel and D. O. Pederson, "SPICE (Simulation Program with Integrated Circuit
Emphasis)," Memorandum No. ERL-M382, Electronics Research Laboratory, University of
California, Berkeley, April 1973.

\[2\] K. Kundert, "Introduction to RF Simulation and Its Application," *IEEE Journal of
Solid-State Circuits*, vol. 34, no. 9, pp. 1298–1319, September 1999.
DOI: 10.1109/4.782091

\[3\] J. R. Phillips, "Projection-based approaches for model reduction of weakly nonlinear,
time-varying systems," *IEEE Transactions on Computer-Aided Design of Integrated Circuits
and Systems*, vol. 22, no. 2, pp. 171–187, 2003. DOI: 10.1109/TCAD.2002.806605

\[4\] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-Informed Neural Networks:
A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear
Partial Differential Equations," *Journal of Computational Physics*, vol. 378,
pp. 686–707, 2019. DOI: 10.1016/j.jcp.2018.10.045

\[5\] E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
Differential-Algebraic Problems*, 2nd ed., Springer Series in Computational Mathematics,
vol. 14, Springer, 1996.

\[6\] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and
A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations,"
*International Conference on Learning Representations (ICLR)*, 2021. arXiv:2010.08895

\[7\] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang,
"Physics-Informed Machine Learning," *Nature Reviews Physics*, vol. 3, no. 6,
pp. 422–440, 2021. DOI: 10.1038/s42254-021-00314-5

\[8\] S. Wang, H. Wang, and P. Perdikaris, "Learning the Solution Operator of Parametric
Partial Differential Equations with Physics-Informed DeepONets," *Science Advances*,
vol. 7, no. 40, eabi8605, 2021. DOI: 10.1126/sciadv.abi8605
