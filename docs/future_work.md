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

## 2) Evaluate larger and more relevant architectures

### A. Analog topologies

- Differential pair as the next analog architecture-level target.
- Follow with at least one load/feedback variant to test compositional
  stability under additional internal nodes.

### B. Digital topologies

- Extend from inverter to chain/depth studies with explicit error growth laws.
- Keep gate-charge modeling explicit in scope, assumptions, and error budget.

### C. Selection criteria for new topologies

- Choose designs that stress distinct solver/model failure modes.
- Require paired SPICE and composed evidence with comparable stimulus and IC
  protocols.

## 3) Overarching themes

These items cut across device models, solvers, and benchmarks: they assume the
current validation stack exists, but they are not reducible to a single circuit
or a single training run.

### A. Runtime architecture studies

- CPU bottleneck decomposition: operator forward, Jacobian assembly, linear
  solve, and line-search overhead.
- Matrix-free Newton/Krylov experiments for larger windows and deeper circuits.

### B. Hardware portability and reproducibility

- Cross-hardware reproducibility envelopes (CPU vendor, GPU generation,
  precision mode).

### C. Error budgeting at system level

- Decompose end-to-end composition error into device-fit, solver, and
  initialization components.
- Provide uncertainty ranges for key result tables, not just point estimates.

### D. Publication packaging

- Converge on a stable methods/results narrative with explicit scope limits.
- Keep stress-track and showcase-track evidence side-by-side to avoid
  overclaiming.

