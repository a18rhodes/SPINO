# Diode Operator

SPINO's diode component learns the nonlinear circuit voltage operator
$V(t) = \mathcal{F}(I(t),\, \log R,\, \log C,\, \log I_S,\, N)$
for a diode–resistor–capacitor circuit governed by the Shockley equation.

---

## Architecture

1D Fourier Neural Operator (Li et al., 2021) with direct channel injection:

| Hyperparameter | Value |
|---|---|
| Fourier modes | 256 |
| Hidden channels | 64 |
| Input channels | 5 (see below) |
| Output channels | 1 ($V(t)$) |
| Activation | SiLU |
| Domain padding | 0.1 |
| Skip connection | Linear |

**Input tensor** $X \in \mathbb{R}^{B \times 5 \times T}$:

| Channel | Content | Encoding |
|---|---|---|
| 0 | Input current $I(t)$ | mA |
| 1 | Saturation current $I_S$ | $\log_{10}(I_S)$ |
| 2 | Shunt resistance $R$ | $\log_{10}(R)$ |
| 3 | Junction capacitance $C$ | $\log_{10}(C)$ |
| 4 | Ideality factor $N$ | Raw |

Log-encoding of $I_S$, $R$, $C$ is essential: $I_S$ spans 14 orders of magnitude and direct
injection causes gradient instability.

---

## Problem Formulation

The diode injects exponential nonlinearity:

$$I_D = I_S \left( e^{V_D / (N V_T)} - 1 \right)$$

No closed-form solution exists; traditional SPICE iterates Newton-Raphson at each time step.
The FNO replaces the inner-loop solver with a single forward pass.

---

## Training

- **Data generator:** PySpice/NGSPICE backend with randomized parameters
- **Parameter ranges:**

| Parameter | Range | Distribution |
|---|---|---|
| $R$ | 50 Ω – 2 kΩ | Log-uniform |
| $C$ | 1 nF – 100 nF | Log-uniform |
| $I_S$ | $10^{-14}$ – $10^{-9}$ A | Log-uniform |
| $N$ | 1.0 – 2.0 | Uniform |
| Waveforms | Sinusoids, pulses, noise | Mixed |

- **Epochs:** 80, converged from initial loss 0.469 → $3.67 \times 10^{-4}$
- **Loss:** MSE on physical voltage

---

## Results

| Test | MSE | R² |
|---|---|---|
| Standard rectifier | $3.52 \times 10^{-3}$ | **0.9993** |
| Adversarial (random params) | $6.15 \times 10^{-5}$ | **0.9996** |

**Error context:**
- RMSE $\approx 19$ mV on a typical $\pm 5$ V signal swing → relative error $\approx 0.38\%$
- SPICE default tolerance: `RELTOL=1e-3` (0.1 %); the FNO achieves comparable accuracy

The I–V hysteresis loops in the adversarial test confirm the network correctly integrates
the $C \frac{dV}{dt}$ term — it learned dynamic capacitive behaviour, not a static diode curve.

### Simulation Speedup

| | NGSPICE | FNO | Speedup |
|---|---|---|---|
| Standard rectifier | ~240 ms | ~4 ms | **~58×** |

Measured as wall-clock time for a single 2048-step transient simulation (NGSPICE `.tran`)
vs. a single FNO forward pass on the same circuit parameters. Batch GPU inference would
increase throughput further.

---

## Figures

![Standard Rectifier](assets/diode/rectifier.png)
*Standard rectifier (R=1 kΩ, C=4 pF, $I_S$=2.5 nA, N=1.75): time-domain comparison (left)
and dynamic I–V characteristic (right).*

![Adversarial](assets/diode/adversarial.png)
*Adversarial test with randomized circuit parameters. The I–V hysteresis loop (right panel)
confirms the learned dynamic capacitive response.*

---

## Known Limitations

- **Fixed temporal resolution:** The operator is trained on 2048-step waveforms over a 1 ms
  window ($\Delta t \approx 0.49\,\mu\text{s}$). The FNO's Fourier modes are coupled to this
  grid — using a different step count or simulation window at inference changes the physical
  frequency each mode represents. Unlike the [RC operator](rc.md), which non-dimensionalizes
  time via $\lambda = \tau / T_{end}$, the diode operator has no mechanism to adapt to
  arbitrary `.tran` resolutions. This is the most significant constraint for practical EDA
  integration. See the [project-level discussion](../README.md#known-limitations) for
  mitigation strategies.

---

## Design Decisions

**Logarithmic parameter encoding:** Raw $I_S$ values (spanning $10^{-14}$ to $10^{-9}$)
caused gradient instability during training. $\log_{10}$ encoding before channel injection
normalizes the parameter range and stabilizes optimization.

**Output normalization:** MSE on raw voltage overweighted high-current forward-bias regions.
Normalizing outputs relative to signal swing ensures balanced gradient contribution across
the full I-V characteristic.

**Direct channel injection vs. MLP encoder:** With only four dominant parameters, direct
channel injection is sufficient and avoids the complexity of a learned embedding. This
approach does not scale to higher-dimensional parameter spaces (e.g., 29 BSIM4 parameters
for MOSFET) — the [NFET operator](nfet.md) uses VCFiLM conditioning instead.
