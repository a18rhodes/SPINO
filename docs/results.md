# Analog composition results

This note reports composed-circuit validation results against NGSpice for two
analog topologies: the single-stage CS amplifier and the 5T OTA.

Method details are documented in [Neural composition: CS amplifier method](composition.md)
and [Neural composition: 5T OTA method](ota_composition.md). Digital inverter chains are
included below only as a known limitation and regime-boundary result.

## CS amplifier

Method: [Neural composition: CS amplifier method](composition.md).
Characterization: [CS amplifier characterization](cs_amp.md).

### Characterization context

Two geometry points are validated:

1. **Cross-bin stress** (`L = 0.18 µm`): length in the `tiny` bin, widths in the
   `large` bin — the worst cross-bin combination for VCFiLM. Artefacts in
   `docs/assets/cs_amp/summary.json`.
2. **In-bin showcase** (`L = 0.40 µm`): length in the `small` bin with
   correspondingly selected widths. Artefacts in `docs/assets/cs_amp_l040/summary.json`.

### SPICE-only reference summary

| Reference | Wn (µm) | Ln (µm) | Wp (µm) | Lp (µm) | Vin* (V) | Vout* (V) | Peak \|gain\| (V/V) | Idd @ Vin* (A) | Loaded settling (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `docs/assets/cs_amp/summary.json` | 6.0 | 0.18 | 4.5 | 0.18 | 0.85 | 0.60894 | 2.0519 | 1.285e-4 | 2.50e-8 |
| `docs/assets/cs_amp_l040/summary.json` | 1.6 | 0.40 | 2.5 | 0.40 | 0.81 | 0.60072 | 1.4167 | 1.565e-5 | 1.95e-7 |

### Composition runs

- CUDA, `L=0.18` stress: `docs/assets/cs_amp_fno_exp2/summary.json`
- CUDA, `L=0.40` showcase: `docs/assets/cs_amp_fno_l040_exp2/summary.json`
- CPU, `L=0.18` stress: `docs/assets/cs_amp_fno/summary.json`
- CPU, `L=0.40` showcase: `docs/assets/cs_amp_fno_l040/summary.json`

### Core fidelity metrics (CUDA)

| Metric | `L=0.18` stress | `L=0.40` showcase |
|---|---:|---:|
| DC nominal $\lvert \Delta V_\mathrm{out}\rvert / V_\mathrm{DD}$ | 0.9209% | 0.0492% |
| Transient Pearson $r$ | 0.99748 | 0.99981 |
| Transient max $\lvert\Delta V\rvert$ | 25.78 mV | 2.392 mV |
| Transient $R^2$ | -2.6595 | 0.9946 |
| Transient settling (FNO / SPICE) | 30 ns / 25 ns | 210 ns / 195 ns |
| NR iterations (DC / transient) | 5 / 3 | 5 / 2 |

### Runtime context

| Geometry/run | FNO cold (ms) | FNO warm (ms) | SPICE cold (ms) | SPICE warm (ms) |
|---|---:|---:|---:|---:|
| `L=0.18` stress CUDA | 2873.77 | 2697.09 | 13335.72 | 13031.04 |
| `L=0.40` showcase CUDA | 1757.65 | 1733.18 | 12528.74 | 12677.87 |
| `L=0.18` stress CPU | 26091.39 | 25852.92 | 13077.73 | 13018.72 |
| `L=0.40` showcase CPU | 19410.45 | 20193.68 | 12621.42 | 12482.44 |

Runtime is reported for reproducibility context; the scientific claim is
differentiable composition in the analog regime, not wall-clock superiority over NGSpice.

### Figure set (L=0.40 showcase)

![VTC overlay, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/vtc_overlay.png)

![Step response overlay, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/step_response_overlay.png)

![Diagnostic parity panels, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/diagnostic_parity.png)

![Newton convergence, CUDA `L=0.40` showcase run](assets/cs_amp_fno_l040_exp2/convergence.png)

### Error attribution (`L=0.18` stress geometry)

The `L=0.18` CUDA stress run shows the largest composition gaps. Those gaps were
causally decomposed into two separable mechanisms via a four-probe isolation sequence:
a **nominal-bias transient IV-surface** error and a **weak-inversion / near-off VTC**
failure that the step stimulus does not traverse. Full method, figures, and artefact
layout are in [Error attribution: L=0.18 CUDA stress geometry](attribution.md).

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
at `V_in` < 0.5 V.

### Interpretation

The `L=0.40` showcase run materially improves both DC and transient agreement,
consistent with the geometry regime shift: `L=0.18` sits at the worst cross-bin
combination while `L=0.40` moves into a better-conditioned training region.

For the `L=0.18` stress geometry, the low-`Vin` VTC gap is one of two **causally
separated** mechanisms documented with probes and figures in
[Error attribution: L=0.18 CUDA stress geometry](attribution.md).

### Reproduction commands

```text
# SPICE characterization
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

---

## 5T OTA: multi-node analog composition

The 5T OTA is a single-stage operational transconductance amplifier with three
internal KCL nodes (n_tail, n_left, n_out), a differential input pair (M1/M2),
a PFET current-mirror load (M3/M4), and an NFET tail current source (M5). It is
the multi-node scaling demonstration for the differentiable composition method.
Full method documentation is in [Neural composition: 5T OTA method](ota_composition.md)
and SPICE characterization methodology is in [5T OTA characterization](ota_5t.md).

### SPICE-only reference summary (Phase 3a)

Designs selected from a 7×7 (W\_diff, W\_mirror) sweep at each L; M5 fixed at
W=2 µm, Vbias=1.2 V; stimulus ±50 mV differential step at Vcm=0.9 V, C\_load=1 pF.

| L (µm) | W\_diff (µm) | W\_mirror (µm) | Slew rate (V/µs) | Slew time (ns) | I\_tail (µA) | DC gain (V/V) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 8.0 | 8.0 | 48.4 | 21.5 | 78.8 | 48.7 |
| 0.50 | 8.0 | 8.0 | 40.5 | 25.8 | 66.0 | 64.0 |

Sweep health: 49/49 converged at both L; 48/49 feasible at L=0.40, 41/49 at L=0.50.
Artefacts: `docs/assets/ota_5t_l040/`, `docs/assets/ota_5t_l050/`.

### Phase 3b gate outcomes (FNO composition vs SPICE)

| Gate | Criterion | L=0.40 | L=0.50 |
|---|---|---|---|
| Pearson r | ≥ 0.99 | **PASS** (0.9997) | **PASS** (0.9997) |
| max\|ΔV\| | ≤ 30 mV | **FAIL** (68.7 mV) | **FAIL** (68.9 mV) |
| Slew rate relative error | ≤ 10% | **PASS** (1.0%) | **PASS** (5.0%) |
| Slew time relative error | ≤ 10% | **FAIL** (16.3%) | **PASS** (8.5%) |
| NR iterations (DC / transient) | ≤ 25 | **PASS** (5 / 11) | **PASS** (6 / 12) |

The max|ΔV| and slew-time failures are pre-registered results (gate criteria locked
before Phase 3b ran) and are fully attributed to a training-data gap, not solver
tuning. See Attribution below.

### Runtime context

| | FNO GPU (s) | SPICE (s) |
|---|---:|---:|
| L=0.40 (CUDA) | 63.5 | 6.3 |
| L=0.50 (CUDA) | 68.0 | 6.5 |

The FNO runtime is dominated by dense 3×400-element Jacobian assembly via
`jacobian(vectorize=True)` (vmap-batched VJPs). A GPU-native Krylov solver
using `torch.func.vmap` over the Arnoldi basis is the queued speedup path;
scipy GMRES was investigated and found ~7× slower on GPU due to sequential
JVP calls with Python/numpy overhead defeating GPU batching.

### Figure set (L=0.40 showcase)

![Step response overlay, L=0.40](assets/ota_5t_fno_l040/step_response_overlay.png)

![Diagnostic parity: all three internal nodes, L=0.40](assets/ota_5t_fno_l040/diagnostic_parity.png)

*Three-row parity panel: **n\_tail** (M5 drain, common source of diff pair), **n\_left** (M3
diode drain and M4 gate — the mirror control node), and **n\_out** (single-ended output,
M2/M4 drain junction). Left column: FNO vs SPICE time-domain overlay; right column:
scatter parity with identity line.*

![Newton convergence, L=0.40](assets/ota_5t_fno_l040/convergence.png)

![Attribution Probe 1: per-device |ΔI|, L=0.40](assets/ota_5t_fno_l040/attribution/probe1_iv_errors.png)

### Attribution (Probe 1 — IV branch errors at SPICE node voltages)

For each device, the FNO is evaluated at the SPICE node voltage trajectories and
the predicted drain current is compared to the SPICE branch current. A large error
at a specific device localizes the composition error source.

| Device | Role | max\|ΔI\| L=0.40 | max\|ΔI\| L=0.50 |
|---|---|---:|---:|
| M4 PFET mirror out | Dominant | **15.4 µA** | **9.2 µA** |
| M3 PFET mirror diode | Secondary | 5.6 µA | 2.2 µA |
| M2 NFET diff pair | Secondary | 4.8 µA | 3.1 µA |
| M5 NFET tail | Negligible | 2.1 µA | 0.5 µA |
| M1 NFET diff pair | Negligible | 2.8 µA | 1.3 µA |

Root cause: when n_out slews near VDD, M4 exits saturation (Vds → 0). PFET training
data underrepresents this triode-boundary regime, causing ~15 µA current overestimate
at L=0.40 → ~70 mV voltage offset at the output plateau. The slew-time failure at
L=0.40 is a downstream consequence: the 10–90% threshold shifts by ~3.5 ns.

Artefacts: `docs/assets/ota_5t_fno_l040/attribution/`, `docs/assets/ota_5t_fno_l050/attribution/`.

### Reproduction commands

```text
# SPICE characterization
python -m spino.circuit.characterize_ota \
    --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40 \
    --output-dir docs/assets/ota_5t_l040

python -m spino.circuit.characterize_ota \
    --nfet-l 0.50 --pfet-l 0.50 --tail-l 0.50 \
    --output-dir docs/assets/ota_5t_l050

# FNO composition (CUDA)
python -m spino.circuit.compose_ota \
    --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40 \
    --output-dir docs/assets/ota_5t_fno_l040

python -m spino.circuit.compose_ota \
    --nfet-l 0.50 --pfet-l 0.50 --tail-l 0.50 \
    --output-dir docs/assets/ota_5t_fno_l050

# Attribution (Probe 1)
python -m spino.circuit.ota_attribution \
    --run-dir docs/assets/ota_5t_fno_l040 \
    --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40

python -m spino.circuit.ota_attribution \
    --run-dir docs/assets/ota_5t_fno_l050 \
    --nfet-l 0.50 --pfet-l 0.50 --tail-l 0.50
```

---

## MLP ablation: architecture defense

To justify the FNO architecture choice, a per-timestep `MosfetMLP` baseline was trained on the
same 61K NFET dataset as the production Exp 19b VCFiLM-FNO. The MLP maps each timestep
independently: `I_D(t) = f(Vg(t), Vd(t), Vs(t), Vb(t), θ_embed)`. Off-diagonal Jacobian terms
`dI[t]/dV[t']` are exactly zero by construction — this is the structurally correct choice for a
quasi-static device.

Two capacity levels were evaluated (h64: 32K params, h128: 58K params) against FNO Exp19b (1.28M
params). Fast Dataset R² is averaged over 64 fixed-seed samples from the training distribution;
SPICE metrics use deterministic ramp sweeps.

| Metric | MLP h64 | MLP h128 | FNO Exp19b |
|---|---|---|---|
| Fast Dataset R² (64-sample avg) | -4.42 | -5.43 | **0.9879** |
| SPICE Transfer R² | 0.9990 | 0.9989 | **0.9995** |
| SPICE Transfer SubTh-R² | **0.9856** | 0.9631 | 0.9861 |
| SPICE Output R² | 0.9456 | 0.9763 | **0.9960** |
| Speedup vs SPICE | 863x | 6501x | 473x |

The MLP matches FNO on controlled ramp sweeps (Transfer R² ≥ 0.999) but fails catastrophically
on arbitrary PWL waveforms (Fast Dataset R² ≈ −4 to −5). The gap **worsens** with more capacity,
ruling out underfitting as the cause. FNO spectral mixing acts as a **waveform-shape regularizer**:
by aggregating information across the full input trajectory, it generalizes to the diverse waveform
types present in the training distribution. This effect is empirically necessary even though
MOSFET physics does not require temporal dependencies.

The ablation also confirms that the inverter-chain failure (see below) is attributable to spurious
FNO off-diagonal Jacobian terms, not to a deficiency in the FNO's expressive capacity. An MLP
surrogate would produce an exactly diagonal Jacobian — but MLP fails on the waveform distribution
that makes it a viable surrogate in the first place.

Figures: [`docs/assets/mosfet/nfet/mlp_ablation/`](assets/mosfet/nfet/mlp_ablation/)

---

## Digital circuits: known limitation

The inverter-chain composition path was evaluated as a digital extension using
1-, 2-, and 4-stage CMOS chains. SPICE converges for these circuits. The
FNO-composed transient solver does not converge to acceptance-quality outputs,
so its waveforms, delay metrics, and parity statistics are diagnostic only.

Measured transient timing is not evidence of a usable digital accelerator:
relative FNO/SPICE transient time was approximately `0.77x` at `N=1`,
`2.8x` at `N=2`, and `10.5x` at `N=4`, while the FNO transient failed to
converge. Those timings are therefore a negative result, not a performance
claim.

The root cause is structural. The MOSFET is quasi-static, so physical conductive
sensitivity should be local in time: $`dI[t] / dV[t'] = 0`$ for $`t \ne t'`$.
The FNO is a temporal operator with spectral convolutions across the whole
window, so autograd produces a dense $`(N \cdot T) \times (N \cdot T)`$
Jacobian containing off-diagonal temporal couplings. In the CS amplifier's
analog gain region, the physical conductive diagonal is large enough to dominate
those artifacts. In the inverter's digital saturation plateau, the physical
$`dI/dV_{DS}`$ term collapses, the spurious off-diagonal terms dominate
`linalg.solve(J, -R)`, and Newton steps oscillate into non-physical trajectories.

The training data reinforces this boundary: the current
`sky130_nmos_61k_plus_shortch_supp8k.h5` corpus is biased toward dynamic
analog-style waveforms with gate and drain varying together. It does not include
quasi-static fixed-gate drain sweeps or digital step families sufficient to teach
clean conductive Jacobians at fixed bias.

