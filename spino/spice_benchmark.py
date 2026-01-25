import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# PySpice Imports
import PySpice.Logging.Logging as Logging
import torch
import torch.nn as nn
from neuralop.models import FNO
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

# Suppress annoying SPICE logs
Logging.setup_logging(logging_level="ERROR")


# ==============================================================================
# 1. MODEL DEFINITION (Must match training exactly)
# ==============================================================================
class FNO_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # Dimensionless Config
        self.model = FNO(
            n_modes=(256,),
            hidden_channels=64,
            in_channels=2,  # I_norm, Lambda
            out_channels=1,
            preactivation=True,
            fno_skip="linear",
            non_linearity=nn.functional.silu,
            domain_padding=0.1,
        )

    def forward(self, x):
        return self.model(x)


def load_model(path, device="cuda"):
    print(f"Loading model from {path}...")
    model = FNO_Wrapper().to(device)
    # Load state dict (handling potential DataParallel wrappers if they existed)
    state_dict = torch.load(path, map_location=device, weights_only=False)

    # Simple check to handle if state_dict was saved directly from model or wrapper
    if "model.fno_blocks.0.convs.0.weight" in state_dict:
        # Dictionary matches wrapper structure
        model.model.load_state_dict(state_dict, strict=False)
    else:
        # Dictionary might match inner FNO structure, try loading into model.model
        try:
            model.model.load_state_dict(state_dict)
        except:
            # Fallback: Maybe it was saved as the wrapper itself
            model.load_state_dict(state_dict)

    model.eval()
    return model


# ------------------------------------------------------------------------------
# ADDED: Analytical Solver for "Pure Math" Ground Truth
# ------------------------------------------------------------------------------
def solve_rc_analytical(I, R, C, dt):
    """
    Solves V' + V/RC = I/C using exact Forward Euler / Exponential Integrator.
    Input: I (numpy array), R, C, dt (scalars)
    Output: V (numpy array)
    """
    tau = R * C
    decay = np.exp(-dt / tau)
    gain = R * (1.0 - decay)

    V = np.zeros_like(I)
    v_curr = 0.0

    for t in range(len(I)):
        v_next = v_curr * decay + I[t] * gain
        V[t] = v_next
        v_curr = v_next

    return V


# ==============================================================================
# 2. SPICE SIMULATION ENGINE
# ==============================================================================
def run_spice_simulation(I_np, R_val, C_val, T_end, t_steps):
    """
    Runs a single NGSPICE simulation for an RC circuit with arbitrary current source.
    Returns: V_spice (numpy array) interpolated to match t_steps
    """
    circuit = Circuit("RC Benchmark")

    # 1. Define Components
    circuit.R("R1", "node_in", "node_out", R_val @ u_Ohm)
    circuit.C("C1", "node_out", circuit.gnd, C_val @ u_F)

    # 2. Define Arbitrary Current Source (PWL - Piecewise Linear)
    dt = T_end / t_steps
    times = np.linspace(0, T_end, t_steps)

    # Create string pairs "t1, v1, t2, v2..."
    pwl_pairs = list(zip(times, I_np))

    # Input Current Source flowing into node_in
    circuit.PieceWiseLinearCurrentSource(1, circuit.gnd, "node_in", values=pwl_pairs)

    # 3. Simulation
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    # Transient analysis
    analysis = simulator.transient(step_time=dt, end_time=T_end, use_initial_condition=True)

    # 4. Extract & Interpolate
    spice_time = np.array(analysis.time)
    spice_voltage = np.array(analysis.nodes["node_out"])

    # Linear interpolation to match our FNO grid
    v_interp = np.interp(times, spice_time, spice_voltage)

    return v_interp


# ==============================================================================
# 3. FNO INFERENCE ENGINE
# ==============================================================================
def run_fno_inference(model, I_np, R_val, C_val, T_end, t_steps, device="cuda"):
    # 1. Non-Dimensionalization Math
    tau = R_val * C_val
    Lambda = tau / T_end

    # Scale current to dimensionless input
    # In training we normalized I to [-1, 1].
    # For robust inference, we divide by the peak current of this specific sample
    I_max = np.max(np.abs(I_np)) + 1e-9
    I_norm = I_np / I_max

    # 2. Tensor Prep
    I_tens = torch.tensor(I_norm, dtype=torch.float32).to(device)
    Lambda_tens = torch.full_like(I_tens, Lambda)

    # Batch dim
    x_in = torch.stack([I_tens, Lambda_tens], dim=0).unsqueeze(0)

    # 3. Inference
    if device == "cuda":
        torch.cuda.synchronize()
    start_t = time.perf_counter()

    with torch.no_grad():
        v_pred_hat = model(x_in).cpu().numpy().flatten()

    if device == "cuda":
        torch.cuda.synchronize()
    end_t = time.perf_counter()

    # 4. De-Normalization
    # V = V_hat * (I_scale * R)
    v_pred = v_pred_hat * (I_max * R_val)

    return v_pred, (end_t - start_t)


# ==============================================================================
# 4. MAIN BENCHMARK LOOP
# ==============================================================================
def benchmark(model_path, n_trials=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

    # WARMUP
    print("Warming up FNO...")
    dummy_I = np.zeros(2048)
    run_fno_inference(model, dummy_I, 1000, 1e-6, 1e-3, 2048, device)

    print(f"\n{'='*80}")
    print(f"{'BENCHMARK: FNO vs NGSPICE vs MATH':^80}")
    print(f"{'='*80}")

    # Define Test Cases (Physical Units)
    test_cases = [
        {"R": 1000.0, "C": 1e-6, "T_end": 2e-3, "Label": "Standard RC (Tau=1ms)"},
        {"R": 100.0, "C": 1e-9, "T_end": 500e-9, "Label": "Fast Parasitic (Tau=100ns)"},
        {"R": 100e3, "C": 10e-6, "T_end": 5.0, "Label": "Slow Decay (Tau=1s)"},
    ]

    t_steps = 2048

    for case in test_cases:
        print(f"\nRunning Case: {case['Label']}")
        R, C, T_end = case["R"], case["C"], case["T_end"]
        dt = T_end / t_steps
        t_axis = np.linspace(0, T_end, t_steps)

        # Generate Random Pulse Input
        I_np = np.zeros(t_steps)
        n_switches = 10
        idx = np.sort(np.random.choice(np.arange(1, t_steps - 1), n_switches, replace=False))
        idx = np.concatenate(([0], idx, [t_steps]))
        for j in range(len(idx) - 1):
            val = np.random.uniform(-0.005, 0.005)  # +/- 5mA
            I_np[idx[j] : idx[j + 1]] = val

        # --- RUN SPICE ---
        try:
            start_spice = time.perf_counter()
            v_spice = run_spice_simulation(I_np, R, C, T_end, t_steps)
            time_spice = time.perf_counter() - start_spice
        except Exception as e:
            print(f"SPICE Error: {e}")
            v_spice = np.zeros_like(I_np)  # Fallback to avoid crash
            time_spice = 1.0

        # --- RUN MATH (Ground Truth) ---
        v_math = solve_rc_analytical(I_np, R, C, dt)

        # --- RUN FNO ---
        v_fno, time_fno = run_fno_inference(model, I_np, R, C, T_end, t_steps, device)

        # --- METRICS ---
        # Compare against SPICE (Engineering Reality)
        mse_spice = np.mean((v_spice - v_fno) ** 2)
        r2_spice = 1 - (np.sum((v_spice - v_fno) ** 2) / (np.sum((v_spice - np.mean(v_spice)) ** 2) + 1e-9))

        # Compare against Math (Physics Ideal)
        r2_math = 1 - (np.sum((v_math - v_fno) ** 2) / (np.sum((v_math - np.mean(v_math)) ** 2) + 1e-9))

        r2_spice_vs_math = 1 - (np.sum((v_math - v_spice) ** 2) / (np.sum((v_math - np.mean(v_math)) ** 2) + 1e-9))

        speedup = time_spice / time_fno

        print(f"  > R2 vs SPICE:         {r2_spice:.5f}")
        print(f"  > R2 vs MATH:          {r2_math:.5f}")
        print(f"  > R2 SPICE vs MATH:    {r2_math:.5f}")
        print(f"  > SPICE Time:          {time_spice*1000:.2f} ms")
        print(f"  > FNO Time:            {time_fno*1000:.2f} ms")
        print(f"  > SPEEDUP:             {speedup:.1f}x")

        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Voltage Axis
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Voltage (V)", color="black")
        (l1,) = ax1.plot(t_axis, v_spice, "k-", linewidth=2.5, alpha=0.5, label="NGSPICE")
        (l2,) = ax1.plot(t_axis, v_math, "g:", linewidth=2.0, label="Math (ODE)")
        (l3,) = ax1.plot(t_axis, v_fno, "r--", linewidth=1.5, label="FNO (Pred)")
        ax1.tick_params(axis="y", labelcolor="black")

        # Current Axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Current (A)", color="blue")
        (l4,) = ax2.step(t_axis, I_np, color="blue", alpha=0.15, where="post", label="Input I")
        ax2.tick_params(axis="y", labelcolor="blue")

        # Title & Legend
        plt.title(f"{case['Label']} | Speedup: {speedup:.1f}x | R2(Spice): {r2_spice:.4f}")
        lines = [l1, l2, l3, l4]
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Update this path to your actual saved model
    MODEL_PATH = Path(__file__).parent / Path("models", "simple_rc", "Dimensionless_With_Gaussian_Noise_And_Chirp_Log_Uniform_eyJhZGFt.pt")
    if os.path.exists(MODEL_PATH):
        benchmark(MODEL_PATH)
    else:
        print(f"Model file {MODEL_PATH} not found. Please train model first.")
