# Neural composition: CS amplifier method

This note documents the **composition method** for the CS amplifier: how the
trained NFET/PFET operators are assembled into a circuit residual, how Newton
updates are computed, and how stability is enforced.

Quantitative outcomes, figure comparisons, and runtime tables are reported in
[CS amplifier composition results](results.md). SPICE-only characterization
inputs are defined in [Common-source amplifier: NGSpice ground truth](cs_amp.md).

## Replication

```text
python -m spino.circuit.compose --output-dir docs/assets/cs_amp_fno
```

Primary implementation files:

- `spino/circuit/compose.py`
- `spino/circuit/composition.py`
- `spino/circuit/composition_io.py`
- `spino/circuit/devices.py`

## Topology and residual

The composed circuit is a common-source NFET with a diode-connected PFET load.
The unknown is the single internal node `out`; `Vin` and `VDD` are forced.

Kirchhoff current law at `out`:

$$I_{D,\mathrm{pfet}}(V_{out}) - I_{D,\mathrm{nfet}}(V_{in},V_{out}) - C_{load}\,\frac{dV_{out}}{dt} = 0.$$

Terminal ordering for both operators is `(Vg, Vd, Vs, Vb)`:

- NFET: `(Vin, Vout, 0, 0)`
- PFET (diode-connected): `(Vout, Vout, VDD, VDD)`

The load capacitor is analytical. It is not learned and does not use the RC
operator.

## DC operating point solver

The DC solve is scalar Newton-Raphson on `Vout`.

1. Build constant-voltage probe windows for NFET and PFET.
2. Evaluate each operator and reduce to mean post-trim drain current.
3. Form residual `R(Vout)=Ipfet-Infet`.
4. Compute `dR/dVout` via `torch.autograd.grad`.
5. Solve Newton direction and apply damping policy.

The solver reports convergence status, residual history, iteration count, and
wall time in `summary.json`.

## Transient solver

The transient solve is whole-window implicit Newton-Raphson on the vector
`Vout(t) in R^T`.

For time index `n > 0` (backward Euler with analytical `Cload`):

$$R[n]=I_{D,\mathrm{pfet}}[n]-I_{D,\mathrm{nfet}}[n]-\frac{C_{load}}{\Delta t_{n-1}}(V_{out}[n]-V_{out}[n-1]).$$

Row `0` pins the initial condition to the selected DC value.

A dense `T x T` Jacobian is assembled with
`torch.autograd.functional.jacobian`. Dense form is non-trivial because the
operator is non-local in time due to Fourier mixing.

The OTA solver (`spino/circuit/ota_composition.py`, `OtaTransientSolver`)
exposes an opt-in JVP-GMRES Krylov inner solve via `use_gmres=True` that
avoids the `(3T x 3T)` dense assembly by replacing it with sequential
JVP matvecs and scipy GMRES. The default is `use_gmres=False`, i.e.
the dense path described in this section, because the GPU-batched dense
backward outperforms the sequential GMRES matvec for the OTA problem
sizes in this work. The Krylov path stays in the codebase as the route
that scales to larger circuits where the dense assembly stops fitting.

## Jacobian source and numerical policy

No hand-derived compact-model conductance equations are used in composition.
The Jacobian is obtained from autograd through the trained operator stack.

This is the key architectural property: the same differentiability used for
training is reused for Newton updates at composition time.

## Damping and safety constraints

Both DC and transient solvers apply the same update policy:

1. **Armijo backtracking** on residual norm.
2. **Step cap** at 0.2 V max component change.
3. **Rail clip** to `[0, VDD]`.

Armijo uses `c1=1e-4` and minimum step `alpha_min=1e-3`.

## Speed measurement protocol

`compose` records both stacks with the same structure:

- **Cold solver ms**: first timed DC + transient pair.
- **Warm solver ms**: immediate repeat.

SPICE iteration counts are parsed from `.option acct` logs. FNO iteration
counts are reported directly from the Newton solvers.

## Scope limits

- Operators output drain current only; gate-charge terms are out of scope here.
- The explicit capacitor remains analytical by design.
- Composition defaults and checkpoint paths are defined in
  `spino/circuit/compose.py` and `spino/circuit/composition_io.py`.
