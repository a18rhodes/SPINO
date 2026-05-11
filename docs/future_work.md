# Future work

This document captures forward-looking work across SPINO at a project-wide
scope. It is intended to support paper planning and cross-track prioritization.

## 1) Improve model accuracy where it still matters

### A. Weak-inversion and near-off fidelity

- Continue targeted data/model work for low-bias regions where VTC mismatch is
  still visible.
- Keep evaluations centered on error modes that impact composition behavior,
  not just standalone device fit metrics.

### B. Cross-bin robustness

- Expand deliberate cross-bin validation beyond the current CS amplifier stress
  case.
- Quantify how error shifts when width and length move across bin boundaries
  defined in `GEOMETRY_BINS`.

## 2) Evaluate larger and more relevant analog architectures

### A. Analog topologies

The 5T OTA (differential pair with PFET current-mirror load, tail current source,
three internal KCL nodes) has been validated. Both channel lengths pass Pearson r ≥ 0.999
vs NGSpice; the pre-registered max|ΔV| gate fails at both L due to a triode-boundary gap
in PFET training data — documented as a pre-registered finding in
[Analog composition results](results.md).

The root-cause fix (denser PFET training data near Vds ≈ 0) is a prerequisite for
publication-quality voltage parity at the next topology.

Next analog target: **two-stage Miller op-amp** (differential input pair + common-source
second stage + Miller compensation capacitor). Introduces a second high-gain stage,
frequency compensation node, and a new internal topology structure not covered by the OTA.

### B. Selection criteria for new topologies

- Choose designs that stress distinct solver/model failure modes.
- Require paired SPICE and composed evidence with comparable stimulus and IC
  protocols.

## 3) Future work for digital circuits

The inverter-chain result is a documented limitation of the current whole-window
FNO-composed Newton formulation. The paths below are exploratory directions, not
committed roadmap items.

### A. IFT / DEQ-style differentiation

Let SPICE solve the circuit, then use the implicit-function theorem to recover
parameter gradients with a linear solve at the converged point. This pivots the
claim from "FNO solves the circuit" to "differentiable device physics can expose
gradients around a robust converged solution."

### B. SPICE as host solver, FNO as device oracle

Invert the relationship: keep SPICE's MNA and Newton machinery, but call the FNO
as a `(Vg, Vd, Vs, Vb) -> (I, gm, gds)` oracle. This is only compelling for
novel devices where SPICE lacks adequate compact physics.

### C. Quasi-static-rich retraining

Regenerate training data with fixed-gate drain sweeps, fixed-drain gate sweeps,
and step-input digital transition waveforms. This would target the observed root
cause directly by teaching conductive Jacobians at fixed bias points. The cost is
dataset regeneration plus retraining.

### D. Digital-state reparameterization

Solve for transition times and slopes rather than every node voltage at every
timestep. A three-parameter-per-stage representation would reduce the dimension
of the Newton problem and exploit the natural structure of digital waveforms.

### E. Sequential per-timestep Newton with quasi-static FNO

Replace whole-window Newton with timestep-by-timestep Newton and evaluate the FNO
quasi-statically at each step. This eliminates temporal coupling artifacts, but
also gives up the batched whole-window formulation that made the CS amplifier
composition coherent.

## 4) Overarching themes

These items cut across device models, solvers, and benchmarks: they assume the
current validation stack exists, but they are not reducible to a single circuit
or a single training run.

### A. Runtime architecture studies

- GPU-native Krylov linear solver: scipy GMRES with per-JVP matvec was
  investigated for the OTA and found ~7× slower than `jacobian(vectorize=True)`
  on GPU (vmap batches all VJPs in one kernel; sequential scipy calls cannot
  compete). The correct next step is a pure-PyTorch Arnoldi loop with
  `torch.func.vmap` over the Krylov basis, keeping everything on-device.
  Estimated ~150 lines; queued post-publication.
- Cross-hardware reproducibility envelopes (CPU vendor, GPU generation, precision mode).

### B. Hardware portability and reproducibility

- Cross-hardware reproducibility envelopes (CPU vendor, GPU generation,
  precision mode).


