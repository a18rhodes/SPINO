# SPINO: Universal Parametric Neural Operators for Accelerated Circuit Simulation

> **Disclaimer:** Experiments and results are verified by the author. References and citations
> have *not* been independently verified. All code and experiments are original work by the
> project author, with AI tooling assistance.

---

## Abstract

The verification of modern integrated circuits is bottlenecked by SPICE's super-linear
scaling with circuit size ($O(N^{1.2})$ to $O(N^{2})$). **SPINO** (SPICE Neural Operator)
applies Fourier Neural Operators (FNOs) to learn continuous operator mappings from terminal
voltage waveforms and device parameters to node current, replacing the inner-loop SPICE
device evaluation with a single differentiable forward pass.

Four device operators have been trained and validated against NGSPICE ground truth:

| Operator | Conditioning | Peak R² | Speedup | Documentation |
|---|---|---|---|---|
| Linear RC | Dimensionless $\lambda$ | 0.9999 | < 1× | [RC Circuit](docs/rc.md) |
| Shockley Diode | Dimensionless $\lambda$ + direct injection | 0.9999 | ~66× | [Diode](docs/diode.md) |
| sky130 NMOS | VCFiLM (29-param BSIM) | 0.9995 | ~1300× | [NFET](docs/nfet.md) |
| sky130 PMOS | VCFiLM (29-param BSIM) | 0.9999 | ~522× | [PFET](docs/pfet.md) |

The NMOS operator achieves transfer R² = 0.9995 and subthreshold R² = 0.9861 at core geometry
(W = 1.0 µm, L = 0.18 µm), with warm-inference throughput approximately 1300× that of NGSPICE.
The PMOS operator uses the same VCFiLM-FNO architecture trained on a sweep-augmented dataset
(40 K random PWL + 4 K deterministic sweeps), achieving transfer R² = 0.9965 and sweep R² >
0.99 across all tested geometries at ~522× NGSPICE speed.
The diode operator extends the RC dimensionless framework to the nonlinear Shockley equation,
achieving R² = 0.9994 on a standard rectifier and R² = 0.9999 on adversarial samples at ~66×
NGSpice speed — with validated resolution invariance ($\Delta R^2 < 0.0001$ at 1024/2048/4096
steps) and time-scale invariance (R² ≥ 0.997 across $T_{end}$ spanning 100 µs to 10 ms).
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

A 1D FNO learns the voltage response $V(t) = \mathcal{F}(I(t), \lambda)$ of a first-order
RC circuit in dimensionless form, where $\lambda = \tau / T_{end}$ is the stiffness ratio.
Training on the non-dimensionalized ODE makes the operator invariant to physical time
scale: a 100 fs parasitic transient and a 10 s saturation drift are identical if their
stiffness ratios match. Spectral augmentation (white noise, chirp, dense switching) ensures
the operator learns integration rather than memorizing pulse shapes.

### 2. Shockley Diode

The diode introduces exponential nonlinearity via the Shockley equation. Following the same
dimensionless approach as the RC operator, the circuit ODE is reformulated in terms of
$\hat{t} = t/T_{end}$, $\hat{I} = I/I_{scale}$, $\hat{V} = V/(I_{scale}R)$, with the
stiffness ratio $\lambda = RC/T_{end}$ injected as a constant channel. This makes the
operator invariant to simulation window and grid resolution. Five circuit parameters
($\lambda$, $R$, $C$, $I_S$, $N$) are injected directly as constant-valued channels
alongside the normalised current waveform. Log-encoding of parameters spanning 15 orders of
magnitude ($I_S$) prevents gradient instability. The FNO replaces NGSpice's inner-loop
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
polarities are handled by the device strategy layer: PMOS operates with $V_S = V_B = V_{DD}$
and sweeps gate/drain downward.

---

## Summary of Results

### Accuracy

| Operator | Test Condition | R² | Relative Error |
|---|---|---|---|
| Linear RC | Corner frequency ($\lambda = 1.0$) | 0.9999 | < 0.1% |
| Linear RC | White noise (OOD) | 0.9884 | ~1% |
| Shockley diode | Standard rectifier ($\lambda = 0.01$) | 0.9994 | ~0.45% |
| Shockley diode | Adversarial (random params) | 0.9999 | < 0.1% |
| sky130 NMOS | Transfer (W=1 µm, L=0.18 µm) | 0.9995 | -- |
| sky130 NMOS | Output (W=1 µm, L=0.18 µm) | 0.9960 | -- |
| sky130 NMOS | Subthreshold (W=1 µm, L=0.18 µm) | 0.9861 | -- |
| sky130 PMOS | Transfer (W=1 µm, L=0.18 µm) | 0.9965 | -- |
| sky130 PMOS | Output (W=1 µm, L=0.18 µm) | 0.9656 | -- |
| sky130 PMOS | Subthreshold (W=1 µm, L=0.18 µm) | 0.9523 | -- |

### Throughput

| Operator | Baseline | FNO | Speedup | Notes |
|---|---|---|---|---|
| Shockley diode | NGSPICE ~264 ms | ~4 ms | **~66×** | Single 2048-step transient |
| sky130 NMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~1300×** | Sustained throughput |
| sky130 NMOS (cold) | NGSPICE `.tran` | First call (incl. JIT) | **~21×** | One-time compilation cost |
| sky130 PMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~522×** | Sustained throughput |
| Linear RC | ODE loop < 1 ms | FNO ~96 ms | **< 1×** | Value is generalization, not speed |

---

## Known Limitations

### Temporal and Resolution Invariance (MOSFET)

The MOSFET $I_D(V_G, V_D, V_S, V_B, \boldsymbol{\theta})$ mapping is **quasi-static**
(algebraic, not ODE-governed). The device transit time
$\tau_t = L^2 / (\mu_0 \cdot V_{\text{eff}})$ is 100–10,000× smaller than any practical
simulation window, and displacement currents are ~0.01% of channel current. This means the
dimensionless stiffness ratio $\lambda = \tau / T_{end}$ that governs the RC and diode
operators carries **no information** for the MOSFET operator.

The VCFiLM conditioning pathway already has access to all the ingredients of $\tau_t$ — gate
length, mobility parameters, and threshold voltage are present in the 29-element BSIM vector,
while per-timestep terminal voltages provide instantaneous $V_{\text{eff}}$. The network can
reconstruct an effective time constant internally without an explicit $\lambda$ channel.

This was empirically validated on the NFET Exp 19b production model across three geometries
(core, tiny, xlarge):

| Test | Variable | Range | Worst $\Delta R^2$ | Criterion | Verdict |
|---|---|---|---|---|---|
| Time-scale | $T_{end}$ | 100 ns – 5 µs (50×) | 0.000517 | < 0.01 | **PASS** |
| Resolution | Step count | 512 – 4096 (8×) | 0.000002 | < 0.001 | **PASS** |

All $R^2$ values remained above 0.999 across the full test matrix. The MOSFET operator is
therefore usable at arbitrary `.tran` resolutions and simulation windows, matching the
flexibility of the [RC](docs/rc.md) and [Diode](docs/diode.md) operators — albeit through a
different mechanism (implicit reconstruction via device physics embedding rather than explicit
dimensionless parameterization).

**This is fundamentally different from the RC and diode operators**, where $\lambda$ governs
ODE dynamics and is essential for the operator to distinguish stiffness regimes. For the
MOSFET, the physics is algebraic, and the VCFiLM architecture exploits this automatically.

---

## Future Work: Neural Newton-Raphson Composition

The trained device operators are fully differentiable. The planned composition layer exploits
this property to assemble multi-device circuits without hand-derived conductance equations.

Given a circuit with node voltage vector $\mathbf{V}$, Kirchhoff's Current Law (KCL) requires:

$$\mathbf{R}(\mathbf{V}) = \sum_k \mathbf{I}_k(\mathbf{V}) = \mathbf{0}$$

At each Newton iteration, PyTorch autograd computes the exact Jacobian
$\frac{\partial \mathbf{I}}{\partial \mathbf{V}}$ through the FNO stack -- the same
information SPICE derives analytically from BSIM model cards, obtained here from backprop
at no additional implementation cost.

The Newton system scales with the number of **interface nodes** (typically $O(10)$), not
the internal complexity of each device block. A common-source amplifier (MOSFET + resistor +
capacitor) is the intended first integration test.

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
