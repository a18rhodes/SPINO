# **Universal Parametric Neural Operators for Accelerated Circuit Simulation: A Physics-Informed Approach**

## **Abstract**

The verification of modern integrated circuits is increasingly bottlenecked by the computational cost of transient simulations in SPICE, particularly as designs scale into the post-Moore era. While "FastSPICE" techniques trade accuracy for speed, they struggle with the complex, stiff differential equations found in modern parasitic-heavy designs. This project explores a data-driven approach using Fourier Neural Operators (FNOs) to create a "Universal Component" simulator. Unlike traditional Physics-Informed Neural Networks (PINNs) which solve a single instance of a PDE, our architecture learns the continuous operator mapping from any time-varying current input $I(t)$ and component parameters to the voltage response $V(t)$.

By implementing a **Dimensionless Physics-Informed Neural Operator (PiNO)** architecture with variable-horizon training and spectral augmentation, we demonstrate a model capable of zero-shot generalization across a wide range of time constants. Initial results show strong fidelity with ground-truth physics ($R^2 > 0.999$ for calibrated corner cases) and promising generalization to out-of-distribution signals like white noise ($R^2 > 0.98$) and chirps ($R^2 > 0.97$). Future work aims to extend this approach to non-linear components and explore modular "Neural-SPICE" architectures where learned operators can be composed like building blocks.

## **1\. Introduction**

For over five decades, the Simulation Program with Integrated Circuit Emphasis (SPICE) has served as the bedrock of Electronic Design Automation (EDA). Founded on the direct numerical integration of Ordinary Differential Equations (ODEs) using Newton-Raphson iteration and sparse matrix solvers, SPICE provides the "gold standard" for circuit verification \[1\]. However, the computational complexity of SPICE scales super-linearly with circuit size ($O(N^{1.2})$ to $O(N^{2})$), creating a verification gap for modern System-on-Chip (SoC) designs that contain billions of parasitic elements \[2\].

Traditional acceleration techniques, collectively known as "FastSPICE," employ table-lookup models, model order reduction, and piecewise-linear approximations. While effective, these methods often degrade accuracy in stiff regimes, leading to expensive silicon re-spins \[3\].

Recently, Scientific Machine Learning (SciML) has emerged as a promising alternative. Early efforts utilized Physics-Informed Neural Networks (PINNs) \[4\]. However, standard PINNs suffer from a critical limitation: they are "instance-specific" solvers requiring retraining for new initial conditions.

This project explores **operator learning** as an alternative approach. We experiment with a Universal Parametric FNO that acts as a surrogate for an RC circuit—what we call **SPINO** (SPICE Neural Operator). Once trained, this model can simulate the voltage response for *any* combination of $R$ and $C$ and *any* input current waveform $I(t)$ in a single forward pass, completely invariant to the physical time scale.

## **2\. Literature Review and State of the Art**

### **2.1 The Limits of Numerical Integration**

Standard solvers (e.g., GEER, Trapezoidal) rely on discretizing time. For "stiff" systems—circuits with widely varying time constants ($\\tau$)—the solver must take infinitesimally small steps to maintain stability, causing simulation time to explode \[5\].

### **2.2 Physics-Informed Neural Networks (PINNs)**

Raissi et al. (2019) introduced PINNs, which embed physical laws into the loss function \[6\]. While successful in fluid dynamics, applying PINNs to circuits has been limited by the "retraining bottleneck" \[7\].

### **2.3 Neural Operators (The State of the Art)**

The Fourier Neural Operator (FNO), introduced by Li et al. (2020), operates in the frequency domain to learn resolution-independent mappings \[8\]. Unlike CNNs which depend on grid resolution, FNOs learn the continuous operator kernel.

* **The Gap:** Most existing FNO work focuses on spatial fields (2D/3D). Circuit simulation is a stiff 1D time-series problem where "fast" parasitics ($100\\text{fs}$) and "slow" bias drifts ($10\\text{s}$) must coexist, presenting unique scaling challenges not typically seen in fluid dynamics.

## **3\. Methodology: The Dimensionless PiNO Architecture**

We developed a 1D Fourier Neural Operator adapted for time-domain circuit simulation. A key aspect of our approach is the rigorous **non-dimensionalization** of the input space, allowing a single model to represent an infinite family of circuits.

### **3.1 Architecture Details**

The model processes a 2-channel input tensor $X \in \mathbb{R}^{B \times 2 \times T}$:

1. **Channel 0:** Normalized Current $\hat{I}(t)$. The input waveform scaled to unit magnitude.
2. **Channel 1:** Stiffness Ratio $\lambda$. A scalar field representing the ratio of the circuit's time constant to the simulation window ($\lambda = \tau / T_{end}$).

The architecture consists of:

* **Lifting Layer:** Projects inputs to $d_{model}=64$.
* **Fourier Blocks (4 Layers):**
  * **Spectral Convolution:** Filters the top 256 modes ($k_{max}=256$) to capture sharp transient corners.
  * **Domain Padding:** We utilize `domain_padding=0.1` to mitigate the Gibbs phenomenon caused by non-periodic boundary conditions in time-domain signals.
  * **SiLU Activation:** Used for its smooth derivative, stabilizing the physics-informed loss calculation.
* **Projection Layer:** Decodes the hidden state to dimensionless voltage $\hat{V}(t)$.

### **3.2 Addressing Generalization: The "Dimensionless" Shift**

Early iterations using raw physical units ($R$ in Ohms, $C$ in Farads) failed to generalize across orders of magnitude due to numerical range explosion. **Solution: Physics-Informed Non-Dimensionalization.** We reformulated the RC ODE:

$$C \frac{dV}{dt} + \frac{V}{R} = I(t)$$

Into its dimensionless form:

$$\lambda \frac{d\hat{V}}{d\hat{t}} + \hat{V} = \hat{I}(\hat{t})$$

By training the FNO to solve this dimensionless operator, the model becomes invariant to physical time. A 100-picosecond parasitic transient and a 10-second capacitor discharge become mathematically identical to the network if their stiffness ratio $\lambda$ is the same.

### **3.3 Spectral Augmentation Strategy**

To overcome the "Spectral Bias" of FNOs (the tendency to learn low-frequency functions and ignore high-frequency jitter), we implemented a multi-modal training distribution:

1. **Square Pulses (50%):** Standard EDA logic signals to learn step responses.
2. **Gaussian** White **Noise (25%):** Forces the model to learn the "Integration" operator for high-frequency chaotic inputs (Brownian motion).
3. **Super-Dense Switching (25%):** Rapid switching near the Nyquist limit to force amplitude preservation at high frequencies.

### **3.4 Hybrid Physics Loss**

We employ a composite loss function to ensure physical consistency:

$$\mathcal{L} = \mathcal{L}_{data} + \lambda_1 \mathcal{L}_{sobolev} + \lambda_2 \mathcal{L}_{physics}$$

1. **Data Loss (MSE):** Matches the shape of the waveform.
2. **Sobolev Loss:** Matches the derivative (slope), critical for stiff transients.
3. **Dimensionless Physics Residual:**
    $$\mathcal{L}_{physics} = \left\| \lambda \frac{d\hat{V}}{d\hat{t}} + \hat{V} - \hat{I} \right\|_1$$

## **4\. Current Status and Results**

The model was trained on 10,000 synthesized samples using the NVIDIA CUDA stack with a Cosine Annealing Warm Restarts scheduler.

### **4.1 Adversarial Stress Test**

To verify the "Universal" claim, we subjected the model to signals it had never seen before (Out-Of-Distribution).

| Test Case | Description | Result ($R^2$) | Analysis |
| ----- | ----- | ----- | ----- |
| **Corner Frequency** | $\lambda=1.0$. The transition zone where resistive and capacitive effects are equal. | **0.9999** | **Perfect.** The model correctly identifies the system dynamics in the hardest transition region. |
| **White Noise** | Gaussian noise input. Tests the model's ability to integrate chaos. | **0.9884** | **Excellent.** The model correctly filters noise into a random walk, proving it learned the integration operator and is not just memorizing pulse shapes. |
| **Resolution Blind** | Inference at 4096 steps (2x training resolution). | **0.9994** | **Passed.** Confirms the model learned the continuous operator and is resolution invariant. |
| **Chirp Signal** | Sine sweep $f(t) \propto t$. Tests frequency response (Bode plot). | **0.9710** | **Good.** The model correctly attenuates high frequencies, though minor phase lag exists near the grid limit. |
| **Sawtooth** | Linear ramp input. | **0.9998** | **Perfect.** The model correctly integrates the ramp into a quadratic curve ($t^2$). |

![Spectrum](resources/simple_rc/Spectrum.png)

![Adversarial](resources/simple_rc/Adversarial.png)

![OOD](resources/simple_rc/OOD.png)

### **4.2 Qualitative Analysis**

The shift to dimensionless inputs allowed the model to simulate a **100 femtosecond** parasitic transient (EDA scale) and a **10 second** saturation drift (Power scale) using the exact same set of weights. The approach successfully decouples the simulation window size from the grid resolution, enabling this wide dynamic range.

## **5\. Obstacles and Methodological Refinements**

### **5.1 The "Flat Line" Failure (Scale Variance)**

Initial models predicted flat lines for fast circuits.

* **Root Cause:** Fixed time windows ($20ms$) meant fast transients happened between grid points.
* **Resolution:** Variable-Horizon Training. We dynamicallly scaled the simulation window $T_{end}$ to match the sampled $\tau$, ensuring the physics was always "visible" to the FNO.

### **5.2 Spectral Bias on Noise**

Early iterations achieved $R^2 \approx 0.65$ on White Noise.

* **Root Cause:** The model overfit to the smooth "step" nature of square pulses and filtered out all high-frequency content.
* **Resolution:** Spectral Augmentation. Injecting noise and dense switching into the training set forced the FNO to respect high-frequency components, raising the score to $>0.98$.

## **6\. Future Work**

### **6.1 Training Data Generation**

**Current Implementation:**

For the linear RC circuit demonstration, training data is generated using a custom Forward Euler ODE solver implemented in [spino/solvers.py](spino/solvers.py). The data generation pipeline (see [spino/simple_rc.py](spino/simple_rc.py)) creates thousands of samples with randomized circuit parameters and diverse input waveforms:

```python
# Sample random R, C values and compute time constants
tau_vals = R_vals * C_vals
dt_vals = t_end_vals / t_steps

# Generate diverse current waveforms (square pulses, noise, chirps)
I_physical = I_tensor * 1e-3  # Scale to physical units

# Solve using custom ODE integrator
V_physical = solve_rc_ode(I_physical, R_vals, C_vals, dt_vals)
```

The custom solver is sufficient for linear RC circuits where the analytical solution is well-understood and Forward Euler provides adequate accuracy for training purposes.

**Future Work: PySpice Integration**

To extend beyond linear components and ensure industrial-grade accuracy, we plan to integrate **PySpice** (a Python wrapper for NGSPICE) for two purposes:

1. **Training Data for Non-Linear Components:** Generate ground truth for diodes, MOSFETs, and other non-linear devices where analytical solutions are intractable. PySpice netlists will programmatically define circuits:
   ```python
   circuit = Circuit('Diode_Train')
   circuit.R('1', 'in', 'out', resistance)
   circuit.Diode('1', 'out', circuit.gnd, model='1N4148')
   circuit.PulseVoltageSource('input', 'in', circuit.gnd, ...)

   simulator = circuit.simulator()
   analysis = simulator.transient(step_time=dt, end_time=T_end)
   V_out = np.array(analysis['out'])
   I_in = np.array(analysis['in'])
   ```

2. **Cross-Validation and Benchmarking:** The [spice_benchmark.py](spice_benchmark.py) module contains work-in-progress code to:
   - Generate test sets of circuits not seen during training
   - Compare FNO predictions against NGSPICE `.TRAN` analysis
   - Quantify generalization error and wall-clock speedup metrics

This validation will benchmark:
- **SPICE Path:** Time for `.TRAN` analysis (includes matrix factorization at each time step)
- **FNO Path:** Time for single forward pass (constant time, independent of $N_{steps}$)

Early estimates suggest potential speedups of $100\times$ or more for stiff, long-duration simulations, though rigorous empirical validation across diverse topologies is still needed. Using SPICE as the ground truth generator ensures the FNO learns to mimic real-world SPICE behavior rather than idealized physics, making it a true surrogate for production EDA workflows.

### **6.2 From Current-Mode to Voltage-Mode Simulation**

The current FNO architecture learns the mapping $I(t) \to V(t)$, which is natural for RC circuits where current is the "driving" input. However, most circuit simulation follows a **voltage-based** paradigm where voltages are the primary state variables and currents are derived quantities.

#### **6.2.1 Constitutive Relation: The I-V Operator**

To enable modular composition, we reframe the FNO as learning a **constitutive relation**—the device's I-V characteristic over time:

$$I_{terminal}(t) = \text{FNO}(V_{terminal}(t), \theta)$$

where $\theta$ represents device parameters (e.g., $R$, $C$, or MOSFET model card parameters).

**For passive components (RC):**
- **Input:** Terminal voltage waveform $V(t)$
- **Output:** Terminal current waveform $I(t)$
- **Physics:** The FNO implicitly learns $I = C\frac{dV}{dt} + \frac{V}{R}$

**For active components (OTA, VCO):**
- **Inputs:** Control voltages $V_{in}^+$, $V_{in}^-$, load voltage $V_{out}$
- **Outputs:** Output current $I_{out}$, potentially frequency/phase for oscillators

This "bidirectional port" formulation allows the FNO to act as a black-box component in a larger nodal analysis framework, where the system solver iteratively finds voltages that satisfy Kirchhoff's Current Law (KCL) at all nodes.

#### **6.2.2 Training for Bidirectional Coupling**

To train this formulation, we generate data where both $V(t)$ and $I(t)$ vary:
1. Use PySpice to simulate the component under various **voltage excitations** (not just current sources).
2. Record the resulting **current drawn** by the component.
3. Train the FNO to predict $I(t)$ given $V(t)$, with the physics loss ensuring $\frac{dV}{dt}$ and $\frac{dI}{dt}$ relationships are preserved.

This inverts the typical FNO usage but is critical for integration into voltage-based simulators.

### **6.3 Non-Linear Frontiers (Diodes & MOSFETs)**

We plan to extend the architecture to non-linear devices, starting with the **Shockley Diode model** and eventually standard MOSFET cards (BSIM). This introduces significant challenges:

* **Logarithmic Scaling:** Unlike RC circuits, diode currents scale exponentially with voltage ($I = I_S (e^{V/V_T} - 1)$). We will need to develop new dimensionless normalization schemes based on Thermal Voltage ($V_T \approx 26mV$) and Saturation Current ($I_S$) to handle this dynamic range.
* **Model Complexity:** SPICE model cards contain dozens of parameters. A hierarchical training approach may be needed, starting with "Parametric Diodes" (learning $I_S, C_{J0}$) before tackling full parameter sets.

### **6.4 Modular "Lego Block" Architecture: Neural Newton-Raphson Solver**

The ultimate vision is a **modular circuit simulator** where individual FNOs act as reusable component macromodels that can be interconnected to simulate complex systems like Phase-Locked Loops (PLLs), filters, or analog front-ends. This requires solving the **loading effect problem**: when Block A drives Block B, Block B draws current that affects Block A's voltage—they are bidirectionally coupled and cannot be executed sequentially.

#### **6.4.1 The Core Problem: Kirchhoff's Current Law (KCL) at Interface Nodes**

Consider two FNO blocks connected at a shared node:
- **FNO-A** (e.g., an OTA): Outputs current $I_{out,A}(V_{node})$
- **FNO-B** (e.g., an RC filter): Draws input current $I_{in,B}(V_{node})$

Physics demands that the sum of currents at the node is zero:

$$R(V_{node}) = I_{out,A}(V_{node}) + I_{in,B}(V_{node}) = 0$$

This is a **root-finding problem**: we must find the voltage waveform $V_{node}(t)$ that satisfies KCL at every time point.

#### **6.4.2 Solution: Newton-Raphson with Automatic Differentiation**

We solve this using the **Neural Newton-Raphson method**, which exploits the fact that FNOs are fully differentiable:

**Algorithm:**
1. **Initial Guess:** Start with an initial voltage trajectory $V_{node}^{(0)}(t)$ (e.g., zeros or previous timestep solution).

2. **Iterative Refinement:**
   ```
   while ||R(V_node)|| > tolerance:
       # Forward pass: Compute currents from all connected blocks
       I_A = FNO_A(V_node)
       I_B = FNO_B(V_node)
       R = I_A + I_B  # KCL residual

       # Compute Jacobian using PyTorch autograd (the "magic")
       J = torch.autograd.functional.jacobian(
           lambda V: FNO_A(V) + FNO_B(V),
           V_node
       )

       # Newton step: Solve J * ΔV = -R
       delta_V = torch.linalg.solve(J, -R)
       V_node = V_node + delta_V
   ```

3. **Result:** The converged $V_{node}(t)$ represents the physically consistent voltage that balances all connected blocks.

**Key Insight:** In traditional SPICE, computing the Jacobian $\frac{\partial I}{\partial V}$ (the admittance/conductance matrix) requires analytical derivatives of device models and is expensive for large circuits. Here, **PyTorch's autograd engine computes the exact Jacobian for free** by backpropagating through the neural network. This is the "existing back-prop" memory referenced in the planning documents.

#### **6.4.3 System-Level Architecture**

For a complete system (e.g., a PLL with OTA, VCO, and Loop Filter):

1. **Component Library:** Train individual FNOs for each block type:
   - `FNO_OTA(V_in+, V_in-, V_out) → I_out`
   - `FNO_Filter(V_in) → I_in`
   - `FNO_VCO(V_ctrl) → Phase(t)`

2. **Topology Definition:** Specify the netlist (which blocks connect to which nodes):
   ```python
   connections = {
       'node_1': [FNO_OTA.out, FNO_Filter.in],
       'node_2': [FNO_Filter.out, FNO_VCO.in]
   }
   ```

3. **Global Newton Solve:** At each "macro time step," solve the coupled system:
   - Construct the global residual vector $R = [R_{node1}, R_{node2}, ...]$ (sum of currents at each node)
   - Compute the block-sparse Jacobian matrix using autograd
   - Update all node voltages simultaneously via Newton iteration

4. **Time-Tunneling:** Unlike SPICE which takes thousands of tiny steps, the FNO predicts the **entire waveform** for the next time window (e.g., $t \to t + 1\mu s$) in a single forward pass, then we re-solve the coupling equations. This is the "Time-Tunneling" operator.

#### **6.4.4 Advantages Over Monolithic Training**

- **Modularity:** Train each component once, reuse in infinite circuit configurations.
- **Scalability:** The Newton system size scales with the number of **interface nodes** (typically 10-100), not the internal complexity of each block (which could contain thousands of parasitic elements).
- **Physical Consistency:** The iterative solve guarantees KCL/KVL are satisfied to machine precision, avoiding "energy creation" artifacts that plague end-to-end neural network approaches.

**Computational Cost Comparison:**
- **Traditional SPICE:** Inverts a $10{,}000 \times 10{,}000$ matrix (all internal nodes) at each time step.
- **Modular FNO:** Inverts a $3 \times 3$ matrix (interface nodes only) at each macro time step, with FNO inference being $O(1)$ regardless of internal block complexity.

This approach directly addresses the "lego block" vision: components can be trained independently using PySpice-generated ground truth, then snapped together into arbitrary topologies using the Neural Newton-Raphson framework.

## **7\. Conclusion**

We have demonstrated a **Universal Parametric FNO** for linear circuit simulation with promising initial results. By combining dimensionless physics inputs with rigorous spectral augmentation, we achieved a model that shows strong invariance to time scale, resolution, and signal type across a wide range of test cases. While significant work remains—particularly validation against SPICE benchmarks and extension to non-linear components—this approach suggests a potential path toward modular "Neural-SPICE" architectures that could accelerate simulation of complex parasitic networks.

## **8\. References**

\[1\] L. W. Nagel and D. O. Pederson, "SPICE," U.C. Berkeley, 1973\. \[2\] Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations," ICLR, 2021\. \[3\] G. E. Karniadakis et al., "Physics-informed machine learning," Nature Reviews Physics, 2021\.
