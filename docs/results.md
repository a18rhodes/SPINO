# CS amplifier composition results

This note reports the current CS amplifier composition results against NGSpice,
using two SPICE-characterized amplifier references and their corresponding
composition runs.

Method details are documented in [Neural composition: CS amplifier method](composition.md).

## Experimental context

Two SPICE characterization points are now in evidence:

1. **Cross-bin stress point** (`L = 0.18 um`): selected from
   `docs/assets/cs_amp/summary.json`.
2. **In-bin showcase point** (`L = 0.40 um`): selected from
   `docs/assets/cs_amp_l040/summary.json`.

The cross-bin stress point combines widths inside the `large` width range with
length inside the `tiny` length range (`GEOMETRY_BINS` in
`spino/mosfet/gen_data.py`). The VCFiLM models at this cross-bin had the worst
performance. The `L=0.40 um` point shifts channel length into
the `small` length range and correspondingly selected widths.

## SPICE-only reference summary (3a)

| Reference | Wn (um) | Ln (um) | Wp (um) | Lp (um) | Vin* (V) | Vout* (V) | Peak \|gain\| (V/V) | Idd @ Vin* (A) | Loaded settling (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `docs/assets/cs_amp/summary.json` | 6.0 | 0.18 | 4.5 | 0.18 | 0.85 | 0.60894 | 2.0519 | 1.285e-4 | 2.50e-8 |
| `docs/assets/cs_amp_l040/summary.json` | 1.6 | 0.40 | 2.5 | 0.40 | 0.81 | 0.60072 | 1.4167 | 1.565e-5 | 1.95e-7 |

## Composition runs used for comparison

- CUDA, `L=0.18` stress run: `docs/assets/cs_amp_fno_exp2/summary.json`
- CUDA, `L=0.40` showcase run: `docs/assets/cs_amp_fno_l040_exp2/summary.json`
- CPU, `L=0.18` stress run: `docs/assets/cs_amp_fno/summary.json`
- CPU, `L=0.40` showcase run: `docs/assets/cs_amp_fno_l040/summary.json`

## Core fidelity metrics (CUDA)

| Metric | `L=0.18` stress (CUDA) | `L=0.40` showcase (CUDA) |
|---|---:|---:|
| DC nominal $\lvert \Delta V_\mathrm{out}\rvert / V_\mathrm{DD}$ | 0.9209% | 0.0492% |
| Transient Pearson $r$ | 0.99748 | 0.99981 |
| Transient max $\lvert\Delta V\rvert$ | 25.78 mV | 2.392 mV |
| Transient $R^2$ | -2.6595 | 0.9946 |
| Transient settling (FNO / SPICE) | 30 ns / 25 ns | 210 ns / 195 ns |
| NR iterations (DC / transient) | 5 / 3 | 5 / 2 |

## Runtime and speedup

### CUDA composition vs SPICE

| Geometry/run | FNO cold (ms) | FNO warm (ms) | SPICE cold (ms) | SPICE warm (ms) | Warm speedup (SPICE/FNO) |
|---|---:|---:|---:|---:|---:|
| `L=0.18` stress CUDA | 2873.77 | 2697.09 | 13335.72 | 13031.04 | 4.83x |
| `L=0.40` showcase CUDA | 1757.65 | 1733.18 | 12528.74 | 12677.87 | 7.31x |

### CPU composition vs SPICE

| Geometry/run | FNO cold (ms) | FNO warm (ms) | SPICE cold (ms) | SPICE warm (ms) | Warm speedup (SPICE/FNO) |
|---|---:|---:|---:|---:|---:|
| `L=0.18` stress CPU | 26091.39 | 25852.92 | 13077.73 | 13018.72 | 0.50x |
| `L=0.40` showcase CPU | 19410.45 | 20193.68 | 12621.42 | 12482.44 | 0.62x |

On this stack, NGSpice is heavily optimized on CPU and remains faster than the
dense-autograd composition path at both archived geometries, while CUDA
re-establishes the expected speed advantage.

## Figure set (showcase run)

![VTC overlay, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/vtc_overlay.png)

![Step response overlay, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/step_response_overlay.png)

![Diagnostic parity panels, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/diagnostic_parity.png)

![Newton convergence, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/convergence.png)

## Error attribution (`L=0.18` stress geometry)

The `L=0.18` CUDA stress run shows the largest composition gaps (see **Core fidelity
metrics** above). Those gaps were causally decomposed into two separable mechanisms
via a four-probe isolation sequence: a **nominal-bias transient IV-surface** error
and a **weak-inversion / near-off VTC** failure that the step stimulus does not
traverse. Full method, figures, reproduction, and artefact layout are in
[Error attribution: L=0.18 CUDA stress geometry](attribution.md).

| Domain | Probe | Quantity | Value |
|---|---|---|---:|
| Transient | KCL residual (probe 2) | KCL max `|i|` at pinned SPICE `V_out` | 19.6 uA |
| Transient | KCL residual (probe 2) | KCL max `|i|` at pinned FNO `V_out` | 73 nA |
| Transient | KCL residual (probe 2) | SPICE/FNO KCL-max ratio | 268x |
| Transient | NR diagnostics (probe 4) | Iterations / max Jacobian diag ratio / line-search `alpha` | 3 / 616 / 1.0 |
| VTC | IV error (probe 1) | NFET bad/good rel. error ratio | 4501x |
| VTC | IV error (probe 1) | PFET bad/good rel. error ratio (1000x cap on each point before mean) | 3710x |
| VTC | IV error (probe 1) | PFET FNO vs SPICE at `V_in` ~ 0.25 V (SPICE-converged pins) | 13.7 A vs 785 pA |
| VTC | Substitution | Mean `|Delta V_out|` in bad region (`V_in` < 0.5 V) | 176 mV -> 4.4 mV (97.5%) |
| VTC | Substitution | Max `|Delta V_out|` in bad region (`V_in` < 0.5 V) | 379.7 mV -> 18.7 mV |
| VTC | Substitution | Good region (`V_in` >= 0.5 V): hybrid == FNO by gate | means unchanged (0.0%) |

_PFET ratio row_: uncapped mean bad/mean good ~`3.0e10` (dominated by the `V_in ~ 0.30` V spike); median ratio ~`266x`; geometric-mean ratio ~`1157x`. Definitions in [attribution.md](attribution.md).

In short: the transient mismatch is driven by FNO IV error at `V_gs` ~ 0.85-0.90 V
with a healthy Newton loop; the VTC mismatch is driven by weak-inversion IV error
at `V_in` < 0.5 V. That split matches the cross-bin stress narrative below.

## Interpretation

The `L=0.40` showcase run materially improves both DC and transient agreement.
This is consistent with the geometry regime shift relative to training bins:

- The `L=0.18` stress case probes a cross-bin combination where length sits in
  the `tiny` length range while widths are far beyond the `tiny` width range.
- The `L=0.40` case moves length to the `small` length range, with a selected
  width pair that is less stressed for the current operators.

For the `L=0.18` stress geometry, the low-`Vin` VTC gap is not a vague "MOSFET-track
limit" footnote: it is one of two **causally separated** mechanisms (nominal-bias
transient IV error vs weak-inversion VTC failure) documented with probes and
figures in [Error attribution: L=0.18 CUDA stress geometry](attribution.md).

## Reproduction commands

```text
# SPICE references
python -m spino.circuit.characterize \
    --nfet-w 6.0 --nfet-l 0.18 --pfet-w 4.5 --pfet-l 0.18 \
    --vin-bias 0.85 \
    --output-dir docs/assets/cs_amp

python -m spino.circuit.characterize \
    --nfet-w 1.6 --nfet-l 0.4 --pfet-w 2.5 --pfet-l 0.4 \
    --vin-bias 0.81 \
    --output-dir docs/assets/cs_amp_l040

# Composition (CUDA)
python -m spino.circuit.compose \
    --device cuda \
    --nfet-w 6.0 --nfet-l 0.18 --pfet-w 4.5 --pfet-l 0.18 \
    --vin-bias 0.85 \
    --output-dir docs/assets/cs_amp_fno_exp2

python -m spino.circuit.compose \
    --device cuda \
    --nfet-w 1.6 --nfet-l 0.4 --pfet-w 2.5 --pfet-l 0.4 \
    --vin-bias 0.81 \
    --output-dir docs/assets/cs_amp_fno_l040_exp2

# Composition (CPU)
python -m spino.circuit.compose \
    --device cpu \
    --nfet-w 6.0 --nfet-l 0.18 --pfet-w 4.5 --pfet-l 0.18 \
    --vin-bias 0.85 \
    --output-dir docs/assets/cs_amp_fno

python -m spino.circuit.compose \
    --device cpu \
    --nfet-w 1.6 --nfet-l 0.4 --pfet-w 2.5 --pfet-l 0.4 \
    --vin-bias 0.81 \
    --output-dir docs/assets/cs_amp_fno_l040
```
