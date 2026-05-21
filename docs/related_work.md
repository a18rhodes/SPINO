# Related Work

## Neural Device Surrogates

Prior work on neural compact models trains networks to approximate the static
current-voltage (I-V) relationship $I_D(V_{GS}, V_{DS}, \boldsymbol{\theta})$ directly
from SPICE or measurement data. Tung and Hu train a neural network framework directly
on Berkeley Short-channel IGFET Model (BSIM) generated sweep data to reproduce currents,
charges, and variability across geometry bins, demonstrating SPICE integration and
symmetry compliance [1]. Wei et al. develop a compact artificial neural network (ANN)
model for a 0.18 µm process with Latin hypercube sampling, reporting fitting accuracy
superior to BSIM4 on transconductance and output resistance at reduced dataset size [2].
Qi et al. propose a knowledge-based hybrid that models geometry via analytical equations
and delegates the remainder to an ANN, achieving SPICE-compatible compactness for both
bulk MOSFETs and 2D material FETs [3]. SPINO differs from all three in what the network
must represent: those works learn a static voltage-to-current map suitable for DC/AC
analysis, whereas SPINO's Fourier Neural Operator (FNO) maps whole terminal voltage
trajectories together with a 29-parameter BSIM conditioning vector, $(V(t),
\boldsymbol{\theta}) \to I(t)$, over arbitrary random piecewise-linear (PWL) waveforms,
with the conditioning injected through a Variable-Conditioning Feature-wise Linear
Modulation (VCFiLM) pathway. The multilayer perceptron (MLP) ablation (Fast Dataset
R² ≈ −5 vs. FNO R² ≈ 0.988) demonstrates that per-timestep quasi-static models collapse
on this waveform distribution, and that scaling MLP capacity widens rather than narrows
the gap (h64: −4.42 → h128: −5.43). The requirement is not driven by MOSFET physics,
which is quasi-static at practical simulation windows (transit time and displacement
currents both negligible); instead, the cross-timestep aggregation acts as a
waveform-shape regularizer (as documented in docs/results.md).

## Differentiable Circuit Simulators

The classical analog-CAD approach to differentiating through circuit simulation is the
adjoint transient sensitivity method. Hu and Ye derive O(N) adjoint sensitivities for
performance metrics tied to many time points (signal-to-noise-and-distortion ratio,
total harmonic distortion, spurious-free dynamic range), enabling gradient-based tuning
of analog circuits via SPICE without finite-difference perturbations [4]. Sun et al.
(Soda-PTA) bring neural ordinary differential equations (ODEs) into this loop: they model
the pseudo-transient-analysis (PTA) solution curve as a Neural ODE and apply the classical
adjoint method to derive gradients of Newton-Raphson iteration with respect to PTA
hyperparameters, enabling gradient-driven SPICE-convergence acceleration inside an
out-of-the-box simulator [5]. Galetzka, Loukrezis, and De Gersem reformulate modified
nodal analysis (MNA) as a double-minimization problem that replaces explicit compact
models with measurement data, enabling circuit solutions where no closed-form device
model exists [6]. Wang and Achour's Shem framework applies the adjoint method to
differentiate through ODE-based models of analog computing systems, targeting
reconfigurable analog computing paradigms (oscillator-based, transmission-line) rather
than foundry CMOS transient decks, with a translation pass that makes noise, mismatch,
and digital logic differentiable [7]. SPINO differs from all four. The classical adjoint
(Hu & Ye) requires analytic device models with hand-derived sensitivity equations and
computes scalar performance gradients. Soda-PTA (Sun et al.) is the closest prior art on
the differentiation mechanism (Neural ODE plus adjoint through Newton-Raphson iteration);
its optimization target is PTA hyperparameters (algorithm tuning to accelerate
convergence), not device geometry, and the solution curve is a learned scalar or
algorithmic surrogate rather than a device-level operator inside the residual. Data-driven
MNA (Galetzka et al.) replaces device models with tables rather than differentiating
through them. Shem targets a different device class (reconfigurable analog computing primitives such as
oscillators and transmission-line PUFs). SPINO
composes parametric FNO device operators inside the Newton-Raphson Kirchhoff's current
law (KCL) residual itself and obtains Jacobians through PyTorch autograd; no hand-coded
sensitivity equations or analytic device models are required, producing an end-to-end
differentiable path from BSIM geometry parameters to circuit node voltages on standard
complementary metal-oxide-semiconductor (CMOS) topologies. The mechanisms differ in
gradient destination, not in autograd plumbing: Soda-PTA routes gradients to PTA
algorithm hyperparameters, the classical adjoint routes them to scalar performance
metrics via hand-derived device sensitivities, data-driven MNA routes them nowhere
(device models are replaced by tables, not differentiated through), and Shem routes
them through ODE-modelled reconfigurable primitives. SPINO routes them to BSIM
device geometry through the transient residual, so the optimisation acts on the
circuit's physical design vector rather than on solver-algorithm parameters or on a
learned scalar abstraction of circuit behaviour.

## Operator Learning Foundations

Li et al. introduce the FNO, which parameterizes the integral kernel of a neural operator
in Fourier space to learn mappings between function spaces for parametric partial
differential equations (PDEs) such as Navier-Stokes, Darcy flow, and Burgers, with
zero-shot super-resolution [8]. Kovachki et al. provide the theoretical foundation,
proving a universal approximation theorem for neural operators and establishing
discretization-invariance as a formal property [9]. Operator learning has begun migrating
into electronic design automation (EDA); Wang et al. (ARO) train an autoregressive
operator-learning model with multi-fidelity fusion and active sampling to predict
steady-state and transient temperature fields in 3D integrated circuits (3D-ICs) from
power traces [10]. SPINO applies operator learning at a different scale and domain.
ARO and other thermal operator surrogates target chip-scale PDE physics (heat diffusion);
SPINO targets device-scale quasi-static physics (MOSFET I-V) and composes the operators
inside a Newton solver rather than calling them as standalone PDE surrogates. MOSFET
physics is algebraic, not ODE-governed, so the "operator" SPINO learns is a waveform
regularizer rather than a PDE solution map. The FNO's spectral convolutions aggregate
cross-timestep information that a per-timestep MLP cannot access; this property, not
physical necessity, explains why the FNO generalizes to random PWL inputs while scaling
the MLP's capacity fails to close the gap.

## Analog Sizing and Optimization

Bayesian optimization (BO) remains the dominant derivative-free approach for analog
sizing. Chen et al. address the curse of dimensionality with a dropout variable-selection
strategy and gm/ID methodology, scaling BO to circuits with more than ten parameters [11].
Touloupas, Chouridis, and Sotiriadis propose local BO with separate Gaussian process (GP)
models per local region, improving constraint handling on large-scale analog circuits [12].
Budak et al. (DNN-Opt) replace the GP surrogate with a deep neural network (DNN) trained
in a reinforcement learning (RL) actor-critic loop, reporting sample efficiency
competitive with BO on industrial benchmark circuits [13]. Their follow-up extends the
surrogate to a Bayesian neural network with layout-aware predictions for practical
analog/mixed-signal (AMS) sizing [14]. Lyu et al. introduce a derivative-aware BO that
combines error-suppressed adjoint analysis with the GP surrogate, injecting classical
SPICE adjoint gradients into the BO acquisition function for analog sizing [15]. Uhlmann,
Moldenhauer, and Scheible train neural network (NN) surrogates directly on
SPICE-evaluated performance metrics (gain, bandwidth, slew rate) and differentiate through
those surrogates with autograd, enabling gradient-based gm/ID sizing with 3,400× speedup
over direct simulation [16]. Ghosh et al. take a different ML route for the same target
as SPINO's OTA case: a Transformer maps OTA specifications, encoded via a driving-point
signal flow graph, to circuit parameters that are then resolved to transistor sizes
through precomputed gm/ID lookup tables [17]. The first four approaches treat SPICE as
an opaque oracle.
Lyu et al. [15] inject gradient information but only at the performance-metric level,
via adjoint post-processing of a black-box SPICE call. Uhlmann et al. [16] introduce end-to-end differentiability but route it through a learned
performance abstraction. A like-for-like reimplementation of the Uhlmann route on the
same 5T OTA problem and the same Adam loss as SPINO (LHS-sampled 1000-point SPICE
training set, 4-layer MLP from θ to (slew, power, swing), autograd-through-MLP gradients)
is reported in `docs/sizing.md` § "Performance-surrogate baseline (Uhlmann route)";
the surrogate's per-component gradient R² ranges from 0.99 ($`V_\mathrm{bias}`$) to 0.14
($`W_\mathrm{tail}`$) on held-out points, and the converged design disagrees with the
FNO/IFT and FD-SPICE routes on $`L_\mathrm{mirror}`$ (Uhlmann pushes it to the 0.50 µm
upper bound; the others leave it at 0.18 µm). Surrogate per-iter cost is essentially
free (one MLP forward+backward, < 0.02 s) but pre-amortises a one-time 5 h SPICE
training-set collection. Ghosh et al. [17] use ML for the spec-to-size map directly,
bypassing the circuit-physics simulator entirely. SPINO targets a lower level, namely
FNO device operators inside KCL; gradients are w.r.t.
device geometry through the Newton-Raphson residual itself, not through an external
adjoint pass or a learned abstraction of circuit behavior. The Adam sizing loop
closing this gradient path is reported in `docs/sizing.md`: a 5-variable 5T-OTA
design vector optimised via the Implicit Function Theorem (IFT) through the KCL
Newton solver converges to a spec-feasible design whose FNO-predicted slew matches
SPICE within 0.35 % on re-simulation, at roughly $6\times$ lower per-iteration
circuit-simulation cost than a forward-FD-SPICE Adam baseline on the same problem.

---

## References

[1] C.-T. Tung and C. Hu, "Neural Network-Based BSIM Transistor Model Framework:
Currents, Charges, Variability, and Circuit Simulation," *IEEE Transactions on Electron
Devices*, vol. 70, pp. 2157–2160, 2023. DOI: 10.1109/TED.2023.3244901

[2] J. Wei, H. Wang, T. Zhao, Y.-L. Jiang, and J. Wan, "A New Compact MOSFET Model
Based on Artificial Neural Network with Unique Data Preprocessing and Sampling Techniques,"
*IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 42,
pp. 1250–1254, 2023. DOI: 10.1109/TCAD.2022.3193330

[3] G. Qi, X. Chen, G. Hu, et al., "Knowledge-Based Neural Network SPICE Modeling for
MOSFETs and Its Application on 2D Material Field-Effect Transistors," *Science China
Information Sciences*, vol. 66, 122405, 2023. DOI: 10.1007/s11432-021-3483-6

[4] W. Hu and Z. Ye, "Adjoint Transient Sensitivity Analysis for Objective Functions
Associated to Many Time Points," *Proceedings of the 57th ACM/IEEE Design Automation
Conference (DAC)*, 2020. DOI: 10.1109/DAC18072.2020.9218602

[5] J. Sun, X. Zha, C. Wang, X. Wu, D. Niu, W. W. Xing, and Z. Jin, "Pseudo Adjoint
Optimization: Harnessing the Solution Curve for SPICE Acceleration," *Proceedings of the
43rd IEEE/ACM International Conference on Computer-Aided Design (ICCAD)*, 2024.
DOI: 10.1145/3676536.3676789

[6] A. Galetzka, D. Loukrezis, and H. De Gersem, "Data-Driven Modified Nodal Analysis
Circuit Solver," *International Journal of Numerical Modelling: Electronic Networks,
Devices and Fields*, 2024. arXiv:2303.03401. DOI: 10.1002/jnm.3205

[7] Y.-N. Wang and S. Achour, "Shem: A Hardware-Aware Optimization Framework for Analog
Computing Systems," arXiv:2411.03557, 2024.

[8] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and
A. Anandkumar, "Fourier Neural Operator for Parametric Partial Differential Equations,"
*International Conference on Learning Representations (ICLR)*, 2021.
arXiv:2010.08895

[9] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, and
A. Anandkumar, "Neural Operator: Learning Maps Between Function Spaces with Applications
to PDEs," *Journal of Machine Learning Research*, vol. 24, no. 89, pp. 1–97, 2023.
arXiv:2108.08481

[10] M. Wang, Y. Cheng, W. Zeng, Z. Lu, V. F. Pavlidis, and W. W. Xing, "ARO:
Autoregressive Operator Learning for Transferable and Multi-fidelity 3D-IC Thermal
Analysis With Active Learning," *Proceedings of the 43rd IEEE/ACM International Conference
on Computer-Aided Design (ICCAD)*, 2024. DOI: 10.1145/3676536.3676713

[11] C. Chen, H. Wang, X. Song, F. Liang, K. Wu, and T. Tao, "High-Dimensional Bayesian
Optimization for Analog Integrated Circuit Sizing Based on Dropout and gm/ID Methodology,"
*IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems*, vol. 41,
no. 11, pp. 4808–4820, 2022. DOI: 10.1109/TCAD.2022.3147431

[12] K. Touloupas, N. Chouridis, and P. P. Sotiriadis, "Local Bayesian Optimization for
Analog Circuit Sizing," *Proceedings of the 58th ACM/IEEE Design Automation Conference
(DAC)*, 2021. DOI: 10.1109/DAC18074.2021.9586172

[13] A. F. Budak, D. Z. Pan, C. V. Kashyap, et al., "DNN-Opt: An RL Inspired Optimization
for Analog Circuit Sizing Using Deep Neural Networks," *Proceedings of the 58th
ACM/IEEE Design Automation Conference (DAC)*, 2021. arXiv:2110.00211.
DOI: 10.1109/DAC18074.2021.9586139

[14] A. F. Budak, K. Zhu, and D. Z. Pan, "Practical Layout-Aware Analog/Mixed-Signal
Design Automation with Bayesian Neural Networks," *Proceedings of the 42nd IEEE/ACM
International Conference on Computer-Aided Design (ICCAD)*, 2023. arXiv:2311.17073.

[15] R. Lyu, A. Zhao, Y. Meng, K. Zhu, Z. Bi, C. Yan, F. Yang, D. Zhou, and X. Zeng,
"Revisiting Sensitivity-Based Analog Sizing with Derivative-Aware Bayesian Optimization
and Error-Suppressed Adjoint Analysis," *Proceedings of the 43rd IEEE/ACM International
Conference on Computer-Aided Design (ICCAD)*, 2024. DOI: 10.1145/3676536.3676736

[16] Y. Uhlmann, T. Moldenhauer, and J. Scheible, "Differentiable Neural Network Surrogate
Models for gm/ID-based Analog IC Sizing Optimization," *Proceedings of the ACM/IEEE 5th
Workshop on Machine Learning for CAD (MLCAD)*, 2023.
DOI: 10.1109/MLCAD58807.2023.10299834

[17] S. Ghosh, E. Y. Gebru, C. V. Kashyap, R. Harjani, and S. S. Sapatnekar, "Accelerating
OTA Circuit Design: Transistor Sizing Based on a Transformer Model and Precomputed Lookup
Tables," *Proceedings of the Design, Automation and Test in Europe Conference (DATE)*,
2025. arXiv:2502.03605
