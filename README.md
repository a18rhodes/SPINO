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
| Shockley Diode | Direct channel injection | 0.9996 | ~58× | [Diode](docs/diode.md) |
| sky130 NMOS | VCFiLM (29-param BSIM) | 0.9995 | ~1300× | [NFET](docs/nfet.md) |

The NMOS operator achieves transfer R² = 0.9995 and subthreshold R² = 0.9861 at core geometry
(W = 1.0 µm, L = 0.18 µm), with warm-inference throughput approximately 1300× that of NGSPICE.
The diode operator matches NGSPICE to within 0.38% relative error at ~58× the simulation speed.
The RC operator, while not faster than a scalar ODE loop on CPU, demonstrates that a single
trained FNO generalizes across the full stiffness ratio spectrum without per-circuit solver
configuration.

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

The diode introduces exponential nonlinearity via the Shockley equation. Four circuit
parameters ($R$, $C$, $I_S$, $N$) are injected directly as constant-valued channels
alongside the input current waveform. Log-encoding of parameters spanning 14 orders of
magnitude ($I_S$) prevents gradient instability. The FNO replaces NGSPICE's inner-loop
Newton-Raphson solver with a single forward pass, achieving ~58× speedup.

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
| Shockley diode | Adversarial (random params) | 0.9996 | ~0.4% |
| sky130 NMOS | Transfer (W=1 µm, L=0.18 µm) | 0.9995 | -- |
| sky130 NMOS | Output (W=1 µm, L=0.18 µm) | 0.9960 | -- |
| sky130 NMOS | Subthreshold (W=1 µm, L=0.18 µm) | 0.9861 | -- |

### Throughput

| Operator | Baseline | FNO | Speedup | Notes |
|---|---|---|---|---|
| Shockley diode | NGSPICE ~240 ms | ~4 ms | **~58×** | Single 2048-step transient |
| sky130 NMOS (warm) | NGSPICE `.tran` | JIT-compiled pass | **~1300×** | Sustained throughput |
| sky130 NMOS (cold) | NGSPICE `.tran` | First call (incl. JIT) | **~21×** | One-time compilation cost |
| Linear RC | ODE loop < 1 ms | FNO ~96 ms | **< 1×** | Value is generalization, not speed |

---

## Known Limitations

### Fixed Temporal Resolution (Diode, MOSFET)

The diode and MOSFET operators are trained on a **fixed discretization grid** — 2048 steps
over 1 ms for the diode, 512 steps over ~1 µs for the MOSFET. The FNO's spectral
convolutions learn frequency-domain filters tied to this specific grid: each Fourier mode
corresponds to a physical frequency determined by the training-time relationship between
step count and simulation window.

Changing either the number of steps or the simulation window at inference breaks this
correspondence. A 2048-step simulation spanning 100 ns contains fundamentally different
spectral content than one spanning 100 µs at the same step count — yet the FNO treats both
identically. In practice, this means:

- The operator cannot be used at arbitrary `.tran` resolutions the way a SPICE model can.
- Designers who need sub-nanosecond resolution to catch switching transients, or millisecond
  windows for settling behaviour, cannot simply adjust `tstep` and `tstop`.
- Waveform features that fall between the training grid's effective Nyquist limit are invisible
  to the surrogate.

**The RC operator does not share this limitation.** Its dimensionless formulation
(see [RC Circuit](docs/rc.md)) non-dimensionalizes time via $\lambda = \tau / T_{end}$,
factoring out the physical time scale entirely. The operator is invariant to simulation
window — a 100 fs parasitic and a 10 s drift are identical if their stiffness ratios match.

**Potential mitigations (future work):**

1. **Dimensionless reformulation.** Extend the RC approach to nonlinear devices by
   identifying characteristic time scales (e.g., $\tau_{RC}$ from parasitic capacitances,
   transit time $\tau_t$) and conditioning the FNO on a dimensionless stiffness parameter.
   This is the cleanest solution but requires careful identification of the dominant time
   constant for each device regime.
2. **Time-scale conditioning.** Inject $T_{end}$ and $\Delta t$ (or their logarithms) as
   additional conditioning parameters alongside the physics vector. The FNO would learn to
   adapt its spectral filters to the declared resolution. Requires retraining on diverse
   time windows.
3. **Canonical-grid resampling.** Interpolate arbitrary-resolution input waveforms onto the
   training grid, run the FNO, and resample the output back. This preserves the trained
   operator unchanged but introduces interpolation error and cannot recover information
   below the training grid's Nyquist frequency.
4. **Multi-scale training.** Train on a distribution of $T_{end}$ values spanning the
   target resolution range. Brute-force but straightforward; trades training cost for
   generality.

Until one of these mitigations is implemented, the diode and MOSFET operators are restricted
to inference at or near their training resolution. This is the single most significant
constraint for real EDA integration.

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
