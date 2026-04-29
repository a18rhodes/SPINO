# SPINO: Universal Parametric Neural Operators for Accelerated Circuit Simulation

> **Disclaimer:** Experiments and results are verified by the author. References and citations
> have *not* been independently verified. All code and experiments are original work by the
> project author, with AI tooling assistance.

---

## Abstract

The verification of modern integrated circuits is bottlenecked by SPICE's super-linear
scaling with circuit size ($`O(N^{1.2})`$ to $`O(N^2)`$). **SPINO** (SPICE Neural Operator)
applies Fourier Neural Operators (FNOs) to learn continuous operator mappings from terminal
voltage waveforms and device parameters to node current, replacing the inner-loop SPICE
device evaluation with a single differentiable forward pass.

Four device operators have been trained and validated against NGSPICE ground truth:

| Operator | Conditioning | Peak R² | Speedup | Documentation |
|---|---|---|---|---|
| Linear RC | Dimensionless $`\lambda`$ | 0.9999 | < 1× | [RC Circuit](docs/rc.md) |
| Shockley Diode | Dimensionless $`\lambda`$ + direct injection | 0.9999 | ~66× | [Diode](docs/diode.md) |
| sky130 NMOS | VCFiLM (29-param BSIM) | 0.9995 | ~1300× | [NFET](docs/nfet.md) |
| sky130 PMOS | VCFiLM (29-param BSIM) | 0.9999 | ~522× | [PFET](docs/pfet.md) |

A composed-circuit CS amplifier is the first system-level validation milestone.
The work is split into three paper-style documents:

- [Neural composition: CS amplifier method](docs/composition.md) — KCL assembly,
  Newton-Raphson solvers, autograd Jacobians, and damping policy.
- [CS amplifier composition results](docs/results.md) — SPICE baselines,
  composed-fidelity metrics, and runtime tables (CPU/CUDA context).
- [Future work](docs/future_work.md) — global roadmap beyond active lab trackers.

SPICE-only characterization methodology and selected design points are documented in
[CS amplifier characterization](docs/cs_amp.md).

The NMOS operator achieves transfer R² = 0.9995 and subthreshold R² = 0.9861 at core geometry
(W = 1.0 µm, L = 0.18 µm), with warm-inference throughput approximately 1300× that of NGSPICE.
The PMOS operator uses the same VCFiLM-FNO architecture trained on a sweep-augmented dataset
(40 K random PWL + 4 K deterministic sweeps), achieving transfer R² = 0.9965 and sweep R² >
0.99 across all tested geometries at ~522× NGSPICE speed.
The diode operator extends the RC dimensionless framework to the nonlinear Shockley equation,
achieving R² = 0.9994 on a standard rectifier and R² = 0.9999 on adversarial samples at ~66×
NGSpice speed — with validated resolution invariance ($`\Delta R^2 < 0.0001`$ at 1024/2048/4096
steps) and time-scale invariance (R² ≥ 0.997 across $`T_{end}`$ spanning 100 µs to 10 ms).
The RC operator demonstrates that a single trained FNO generalises across the full stiffness
ratio spectrum without per-circuit solver configuration.

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
magnitude ($`I_S`$) prevents gradient instability. The FNO replaces NGSpice's inner-loop
Newton–Raphson solver with a single forward pass, achieving ~66× speedup.

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

### Throughput

| Operator | Baseline | FNO | Speedup | Notes |
|---|---|---|---|---|
| Shockley diode | NGSPICE ~264 ms | ~4 ms | **~66×** | Single 2048-step transient |
| sky130 NMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~1300×** | Sustained throughput |
| sky130 NMOS (cold) | NGSPICE `.tran` | First call (incl. JIT) | **~21×** | One-time compilation cost |
| sky130 PMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~522×** | Sustained throughput |
| CS amp composition (L=0.18, CUDA warm) | NGSPICE `.op + .tran` | FNO-KCL warm solve | **~4.83×** | Cross-bin stress point |
| CS amp composition (L=0.40, CUDA warm) | NGSPICE `.op + .tran` | FNO-KCL warm solve | **~7.31×** | In-bin showcase point |
| CS amp composition (L=0.18, CPU warm) | NGSPICE `.op + .tran` | FNO-KCL warm solve | **~0.50×** | CPU bottleneck vs optimized NGSPICE |
| CS amp composition (L=0.40, CPU warm) | NGSPICE `.op + .tran` | FNO-KCL warm solve | **~0.62×** | Same bottleneck; slightly better ratio at showcase geometry |
| Linear RC | ODE loop < 1 ms | FNO ~96 ms | **< 1×** | Value is generalization, not speed |

---

## Known Limitations

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

This was empirically validated on the NFET Exp 19b production model across three geometries
(core, tiny, xlarge):

| Test | Variable | Range | Worst $`\Delta R^2`$ | Criterion | Verdict |
|---|---|---|---|---|---|
| Time-scale | $`T_{end}`$ | 100 ns – 5 µs (50×) | 0.000517 | < 0.01 | **PASS** |
| Resolution | Step count | 512 – 4096 (8×) | 0.000002 | < 0.001 | **PASS** |

All $`R^2`$ values remained above 0.999 across the full test matrix. The PFET Exp 06 model
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

The composition layer is now implemented for the CS amplifier. It constructs a
single-node KCL residual from NFET and PFET operator outputs and solves DC and
whole-window transient trajectories with damped Newton-Raphson. Jacobians are
computed directly through autograd, with Armijo backtracking plus voltage-step
safeguards.

Method details are in [Neural composition: CS amplifier method](docs/composition.md).

## Composition Results

The current CS amplifier suite includes:

- SPICE characterization at `L=0.18` and `L=0.40` with independently selected
  `(Wn, Wp, Vin*)` points.
- CUDA composition runs against both references.
- CPU composition at `L=0.18` (stress) and `L=0.40` (showcase) to make platform bottlenecks explicit.

Headline composition outcomes:

- Cross-bin stress point (`L=0.18`, tiny-length-band with wide-width pairing):
  transient Pearson `r = 0.99748`, max `|ΔV| = 25.78 mV`.
- In-bin showcase point (`L=0.40`, small-length-band selection):
  transient Pearson `r = 0.99981`, max `|ΔV| = 2.392 mV`.
- Warm runtime speedup (SPICE/FNO): `4.83x` at `L=0.18` (CUDA) and `7.31x`
  at `L=0.40` (CUDA); on CPU the same headline pair is `0.50x` and `0.62x`
  (NGSpice still wins on wall time).

Full tables, figures, and interpretation are in
[CS amplifier composition results](docs/results.md).

## Future Work

The global forward roadmap is consolidated in [Future work](docs/future_work.md),
organized around:

1. Better model accuracy in weak-inversion / low-bias regions.
2. Larger architecture evaluation (including differential-pair-scale analog and
   deeper digital topologies).
3. Cross-cutting investigations beyond daily status trackers (runtime
   decomposition, reproducibility envelopes, and system-level error budgeting).

---

## Development

For installation, training, evaluation, and figure generation instructions, see the
[Development Guide](docs/DEVELOPMENT.md).

---

## References

\[1\] L. W. Nagel and D. O. Pederson, "SPICE (Simulation Program with Integrated Circuit
Emphasis)," Memorandum No. ERL-M382, University of California, Berkeley, 1973.

\[2\] K. Kundert, "Introduction to RF Simulation and Its Application," *IEEE Journal of
Solid-State Circuits*, vol. 34, no. 9, 1999.

\[3\] J. Phillips, "Projection-based approaches for model reduction of weakly nonlinear,
time-varying systems," *IEEE Transactions on Computer-Aided Design*, vol. 22, no. 2, 2003.

\[4\] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks:
A deep learning framework for solving forward and inverse problems involving nonlinear
partial differential equations," *Journal of Computational Physics*, vol. 378, 2019.

\[5\] E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
Differential-Algebraic Problems*, Springer, 1996.

\[6\] Z. Li, N. Kovachki, K. Azizzadenesheli, et al., "Fourier Neural Operator for
Parametric Partial Differential Equations," *ICLR*, 2021.

\[7\] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, et al., "Physics-informed machine
learning," *Nature Reviews Physics*, vol. 3, 2021.

\[8\] S. Wang, H. Wang, and P. Perdikaris, "Learning the solution operator of parametric
partial differential equations with physics-informed DeepONets," *Science Advances*, vol. 7,
2021.
