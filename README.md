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

Three device operators have been trained and validated against NGSPICE ground truth:

| Operator | Conditioning | Peak R² | Speedup | Documentation |
|---|---|---|---|---|
| Linear RC | Dimensionless $\lambda$ | 0.9999 | < 1× | [RC Circuit](docs/rc.md) |
| Shockley Diode | Dimensionless $\lambda$ + direct injection | 0.9999 | ~66× | [Diode](docs/diode.md) |
| sky130 NMOS | VCFiLM (29-param BSIM) | 0.9995 | ~1300× | [NFET](docs/nfet.md) |

The NMOS operator achieves transfer R² = 0.9995 and subthreshold R² = 0.9861 at core geometry
(W = 1.0 µm, L = 0.18 µm), with warm-inference throughput approximately 1300× that of NGSPICE.
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

### Throughput

| Operator | Baseline | FNO | Speedup | Notes |
|---|---|---|---|---|
| Shockley diode | NGSPICE ~264 ms | ~4 ms | **~66×** | Single 2048-step transient |
| sky130 NMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~1300×** | Sustained throughput |
| sky130 NMOS (cold) | NGSPICE `.tran` | First call (incl. JIT) | **~21×** | One-time compilation cost |
| Linear RC | ODE loop < 1 ms | FNO ~96 ms | **< 1×** | Value is generalization, not speed |

---

## Known Limitations

### Fixed Temporal Resolution (MOSFET)

The MOSFET operator is trained on a **fixed discretization grid** — 512 steps over ~1 µs.
The FNO's spectral convolutions learn frequency-domain filters tied to this specific grid:
each Fourier mode corresponds to a physical frequency determined by the training-time
relationship between step count and simulation window.

Changing either the number of steps or the simulation window at inference breaks this
correspondence. In practice, this means:

- The operator cannot be used at arbitrary `.tran` resolutions the way a SPICE model can.
- Designers who need sub-nanosecond resolution to catch switching transients, or microsecond
  windows for settling behaviour, cannot simply adjust `tstep` and `tstop`.

**The RC and diode operators do not share this limitation.** Both use the dimensionless
formulation with $\lambda = \tau / T_{end}$ to factor out physical time scale entirely.
The diode operator was validated at grid resolutions of 1024, 2048, and 4096 steps with
$\Delta R^2 < 0.0001$, and across simulation windows from 100 µs to 10 ms with R² ≥ 0.997
(see [Diode](docs/diode.md)).

**Potential mitigations for MOSFET (future work):**

1. **Dimensionless reformulation.** Identify the dominant MOSFET time constant (transit
   time $\tau_t = L^2 / \mu V_{DS}$ or parasitic RC) and condition the VCFiLM FNO on
   $\lambda = \tau_t / T_{end}$ as an additional input channel. Requires dataset regeneration.
2. **Time-scale conditioning.** Inject $T_{end}$ and $\Delta t$ as additional parameters.
   The FNO learns to adapt its spectral filters to the declared resolution.
3. **Canonical-grid resampling.** Interpolate arbitrary-resolution inputs to the training
   grid, run the FNO, resample output back. Preserves the trained operator unchanged but
   cannot recover information below the training Nyquist frequency.

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
