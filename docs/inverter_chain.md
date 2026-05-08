# Neural composition: inverter chain

> **Status: known limitation / regime-boundary artifact.**
>
> This workflow is retained to document a negative result for digital switching
> circuits. SPICE converges for the inverter-chain references; the whole-window
> FNO-composed Newton solver does not converge to acceptance-quality digital
> trajectories. Outputs from this CLI are therefore diagnostic artifacts, not
> validated circuit-simulation results.

This note describes the second system-level composition target: an **N-stage**
Sky130 CMOS inverter chain driven at the primary input `nin`. It extends the
CS-amplifier KCL viewpoint to **multiple inverter outputs**
`n1, n2, ... nN` with internal feedback where stage `k+1` gates follow stage
`k` outputs.

Standalone device validation remains in [NMOS](nfet.md) and [PMOS](pfet.md).
CS amplifier method details remain in [composition](composition.md). The CS
amplifier is the validated analog composition result; this page documents the
digital workflow that maps the current formulation's boundary.

## Why this is a limitation

The FNO MOSFET surrogate is a temporal operator. When it is evaluated over a
time-varying voltage trajectory, autograd returns a whole-window Jacobian with
off-diagonal terms $`dI[t] / dV[t']`$. Those terms are artifacts of the spectral
temporal architecture, not quasi-static MOSFET physics.

In the CS amplifier's analog gain region, the physical conductive Jacobian
diagonal is large enough to dominate those artifacts and Newton-Raphson
converges. In digital inverter saturation plateaus, the physical $`dI/dV_{DS}`$
diagonal is small, so the spurious temporal couplings dominate the linear solve
and produce oscillating, non-physical Newton updates. Solver tolerances,
iteration budgets, and damping changes did not turn the inverter chain into an
acceptance-quality result.

## Topology and netlists

Use :func:`~spino.circuit.topologies.build_inverter_chain` (and
``build_cmos_inverter`` as the ``n_stages=1`` specialization). Devices are
`XNk` / `XPk` with drains on `nk` and gates on ``nin`` (stage 1) or ``n{k-1}``.

Optional linear load ``CL`` attaches to the **final** node only.

## Partition capacitances

Quasi-static **BSIM partition** capacitors ``cgs``, ``cgd``, ``cgb`` are sampled
on rectilinear grids in bias space:

- NMOS lookups use ``(vgs, vds)``.
- PMOS lookups use ``(vsg, vsd)`` with positive axes.

:class:`~spino.circuit.partition_caps.TorchPartitionCapGrid` performs bilinear
interpolation in PyTorch so Newton steps see smooth ``dC/dV``.

**Extraction paths**

1. **Synthetic** (uniform ``C_ox/3`` placeholder per cell): for wiring tests and
   CI without a PDK run::

       python -m spino.circuit.extract_partition_caps synthetic \\
           --output-dir runs/inv_chain/spice_caps

2. **NGSpice extract** (requires Sky130 + ``ngspice``): sweeps operating points
   and reads internal ``.save`` partition vectors. See the module docstring of
   :mod:`spino.circuit.extract_partition_caps`.

## Solvers

### DC: :class:`~spino.circuit.chain_composition.ChainDcSolver`

Vector damped Newton–Raphson on ``(V_n1, ..., V_nN)`` at fixed input ``V_in``.
Each row enforces mean ``I_{pfet,k} - I_{nfet,k} = 0`` over a trimmed probe
window (same policy as the single-node composition solver).

Unless ``v_init`` is passed explicitly, the NR start uses
:func:`~spino.circuit.chain_composition.default_chain_dc_voltage_guess`: an
alternating ``0`` / ``VDD`` rail pattern based on whether ``V_in`` sits below
or at/above ``VDD/2``, instead of every node at ``VDD/2``.

### Transient: :class:`~spino.circuit.chain_composition.ChainTransientSolver`

**Whole-window** implicit NR on a flattened state of shape ``N * T``. The
residual stacks, per stage and time sample after the initial condition row:

- conductive KCL using the FNO drain-current means on inverter probe
  trajectories;
- backward-Euler gate displacement current from partition caps on **internal**
  gates (stage outputs that drive the next inverter), plus optional ``C_L`` on
  the last node only.

Armijo backtracking and inf-norm stopping mirror the CS amplifier transient
solver.

## Metrics and CLI

:mod:`spino.circuit.chain_metrics` provides ``max_abs_delta_v``, Pearson ``r``,
and monotone **crossing-time** delay at a reference voltage (default
``0.5 * VDD`` on the rising edge). ``summary.json`` extends each node plus
``final_output`` with ``delay_cross_delta_s`` = ``delay_cross_fno_s`` −
``delay_cross_spice_s`` when both crossings resolve; otherwise the delta is nan.

The harness::

    python -m spino.circuit.compose_chain \\
        --output-dir runs/inv_chain/fno \\
        --stages 1

writes ``summary.json`` (per-node and final-output metrics; cold/warm timings only
under ``speedup.spice_ms`` and ``speedup.fno_ms``, with ``notes`` identical in
meaning to ``spino.circuit.compose`` summaries) and ``final_output_overlay.png``
under the local-only ``runs/`` tree. The ``speedup`` key is a historical JSON
schema name; the inverter-chain measurements are not speed evidence because the
FNO transient does not converge.

Use ``--nfet-cap-npz`` / ``--pfet-cap-npz`` to point at extracted tables;
defaults expect synthetic artefacts under ``runs/inv_chain/spice_caps/``.

Do not promote inverter-chain waveforms as validation figures unless a future
method changes the convergence story. The current artifacts belong in the
regime-boundary record.

The documented negative result is summarized in
[results: digital circuits known limitation](results.md#digital-circuits-known-limitation).
