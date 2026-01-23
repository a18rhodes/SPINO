# %% [markdown]
### 1. Imports and Configuration
# Standard imports. Setting style to dark mode for visualization.

import base64
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from neuralop.models import FNO
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from spino.data import backup_artifacts
from spino.loss import GenericDimensionlessPhysicsLoss, rc_physics_residual
from spino.solvers import solve_rc_ode

plt.style.use("dark_background")


# %%
def get_model():
    return FNO(
        n_modes=(256,),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
        preactivation=True,
        fno_skip="linear",
        non_linearity=nn.functional.silu,
        domain_padding=0.1,
    ).cuda()


# %% [markdown]
### 4. Non-Dimensional Data Generator
# This is the core innovation. It converts physical R, C, T, I into dimensionless groups.
# Output Channels: [I_norm, Lambda]


# %%
def generate_dimensionless_data(n_samples=2000, t_steps=2048, device="cuda"):
    """
    Generates data normalized by physics principles.
    Model sees: I_norm (shape) and Lambda (stiffness ratio).
    Model predicts: V_norm (shape).
    """
    # 1. Sample Tau (Bathtub) - Physical units
    min_tau, max_tau = 1e-13, 10.0
    m = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    samples_0_to_1 = m.sample((n_samples,)).to(device)
    log_tau = samples_0_to_1 * (np.log(max_tau) - np.log(min_tau)) + np.log(min_tau)
    tau_vals = torch.exp(log_tau)

    # 2. Determine Window (T_end) relative to Tau
    # Ratio "Window / Tau" determines the shape complexity
    # Low ratio = Zoomed in (Transient)
    # High ratio = Zoomed out (Decay)
    # We sample this ratio Log-Uniformly from 0.1 (very zoomed in) to 100 (very zoomed out)
    ratio_mult = torch.exp(torch.rand(n_samples, 1, device=device) * (np.log(100.0) - np.log(0.1)) + np.log(0.1))
    t_end_vals = tau_vals * ratio_mult
    dt_vals = t_end_vals / t_steps

    # 3. Calculate Lambda (The Physics Input)
    # Lambda = Tau / T_end = 1 / ratio_mult
    # This is the ONLY parameter that changes the shape of the curve!
    lambda_vals = tau_vals / t_end_vals

    # 4. Physical R and C (Needed for Ground Truth generation only)
    min_R, max_R = 100.0, 50000.0
    log_R = torch.rand(n_samples, 1, device=device) * (np.log(max_R) - np.log(min_R)) + np.log(min_R)
    R_vals = torch.exp(log_R)
    C_vals = tau_vals / R_vals

    # 5. Generate Current I(t)
    avg_switches = torch.rand(n_samples, 1, device=device) * 40.0 + 5.0
    prob_switch = avg_switches / t_steps
    switch_mask = torch.rand(n_samples, t_steps, device=device) < prob_switch
    switch_mask[:, 0] = True
    segment_ids = torch.cumsum(switch_mask, dim=1)
    max_segments = int(segment_ids.max().item()) + 1
    # Current values +/- 1.0 (Unitless for now, we scale later)
    random_values = torch.rand(n_samples, max_segments, device=device) * 2.0 - 1.0
    I_tensor = torch.gather(random_values, 1, segment_ids)

    # 6. Solve Physics (Ground Truth V)
    # We use a trick: If we assume I is in Amps and R in Ohms, V is Volts.
    # But if we assume I is "Unitless", we can just assume R=1.0 for the solve?
    # No, let's solve properly then normalize.

    # Let's say physical I = I_tensor * 1mA
    I_physical = I_tensor * 1e-3
    V_physical = solve_rc_ode(I_physical, R_vals, C_vals, dt_vals)

    # 7. Normalize Outputs (V_hat)
    # V_norm = V_physical / (I_physical_max * R)
    # Since I_physical is approx +/- 1mA, max is 1mA.
    # V_hat = V_physical / (1e-3 * R)
    V_hat = V_physical / (1e-3 * R_vals)

    # 8. Stack Inputs [I_hat, Lambda]
    # I_hat is just I_tensor (already +/- 1)
    # Lambda is expanded to time dimension
    train_x = torch.stack([I_tensor, lambda_vals.expand(-1, t_steps)], dim=1)

    # Output V_hat
    return train_x, V_hat.unsqueeze(1)


# %% [markdown]
### 4. Non-Dimensional Data Generator
# UPDATED: Injects Gaussian Noise into 20% of samples to fix Spectral Bias.


# %%
def generate_dimensionless_data_with_white_noise(n_samples=2000, t_steps=2048, device="cuda"):
    """
    Generates data normalized by physics principles.
    Includes Spectral Augmentation (Noise Injection).
    """
    # 1. Sample Tau (Bathtub) - Physical units
    min_tau, max_tau = 1e-13, 10.0
    m = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    samples_0_to_1 = m.sample((n_samples,)).to(device)
    log_tau = samples_0_to_1 * (np.log(max_tau) - np.log(min_tau)) + np.log(min_tau)
    tau_vals = torch.exp(log_tau)

    # 2. Determine Window (T_end) relative to Tau
    ratio_mult = torch.exp(torch.rand(n_samples, 1, device=device) * (np.log(100.0) - np.log(0.1)) + np.log(0.1))
    t_end_vals = tau_vals * ratio_mult
    dt_vals = t_end_vals / t_steps

    # 3. Calculate Lambda
    lambda_vals = tau_vals / t_end_vals

    # 4. Physical R and C
    min_R, max_R = 100.0, 50000.0
    log_R = torch.rand(n_samples, 1, device=device) * (np.log(max_R) - np.log(min_R)) + np.log(min_R)
    R_vals = torch.exp(log_R)
    C_vals = tau_vals / R_vals

    # 5. Generate Current I(t)
    # A. Square Pulses (Standard EDA Signals)
    avg_switches = torch.rand(n_samples, 1, device=device) * 40.0 + 5.0
    prob_switch = avg_switches / t_steps
    switch_mask = torch.rand(n_samples, t_steps, device=device) < prob_switch
    switch_mask[:, 0] = True
    segment_ids = torch.cumsum(switch_mask, dim=1)
    max_segments = int(segment_ids.max().item()) + 1
    random_values = torch.rand(n_samples, max_segments, device=device) * 2.0 - 1.0
    I_tensor = torch.gather(random_values, 1, segment_ids)

    # B. Spectral Augmentation (Gaussian Noise)
    # Replace 20% of the batch with White Noise to force the model to learn high-freq integration
    n_noise = int(n_samples * 0.2)
    if n_noise > 0:
        # Gaussian noise (mean 0, std 0.5) roughly covers the [-1.5, 1.5] range
        noise = torch.randn(n_noise, t_steps, device=device) * 0.5
        # Overwrite the last n_noise samples
        I_tensor[-n_noise:] = noise

    # 6. Solve Physics (Ground Truth V)
    I_physical = I_tensor * 1e-3
    V_physical = solve_rc_ode(I_physical, R_vals, C_vals, dt_vals)

    # 7. Normalize Outputs (V_hat)
    V_hat = V_physical / (1e-3 * R_vals)

    # 8. Stack Inputs [I_hat, Lambda]
    train_x = torch.stack([I_tensor, lambda_vals.expand(-1, t_steps)], dim=1)

    return train_x, V_hat.unsqueeze(1)


# %% [markdown]
### 4. Non-Dimensional Data Generator
# UPDATED: Now includes 3 data types to fix Chirp failure.
# 1. Standard Pulses (60%)
# 2. Gaussian Noise (20%)
# 3. Dense High-Freq Switching (20%) - Fixes the chirp roll-off


# %%
def generate_dimensionless_data_with_white_noise_and_chirp(n_samples=2000, t_steps=2048, device="cuda"):
    """
    Generates data normalized by physics principles.
    Includes Spectral Augmentation (Noise + Dense Switching).
    """
    # 1. Sample Tau (Bathtub) - Physical units
    min_tau, max_tau = 1e-13, 10.0
    m = torch.distributions.Beta(torch.tensor([0.5]), torch.tensor([0.5]))
    samples_0_to_1 = m.sample((n_samples,)).to(device)
    log_tau = samples_0_to_1 * (np.log(max_tau) - np.log(min_tau)) + np.log(min_tau)
    tau_vals = torch.exp(log_tau)

    # 2. Determine Window (T_end) relative to Tau
    ratio_mult = torch.exp(torch.rand(n_samples, 1, device=device) * (np.log(100.0) - np.log(0.1)) + np.log(0.1))
    t_end_vals = tau_vals * ratio_mult
    dt_vals = t_end_vals / t_steps

    # 3. Calculate Lambda
    lambda_vals = tau_vals / t_end_vals

    # 4. Physical R and C
    min_R, max_R = 100.0, 50000.0
    log_R = torch.rand(n_samples, 1, device=device) * (np.log(max_R) - np.log(min_R)) + np.log(min_R)
    R_vals = torch.exp(log_R)
    C_vals = tau_vals / R_vals

    # 5. Generate Current I(t) - MIXED STRATEGY

    # A. Base: All zeros (placeholder)
    I_tensor = torch.zeros(n_samples, t_steps, device=device)

    # Split indices
    n_pulses = int(n_samples * 0.6)
    n_noise = int(n_samples * 0.2)
    n_dense = n_samples - n_pulses - n_noise  # Remaining 20%

    # --- TYPE A: Standard Pulses (60%) ---
    # Avg 5-40 switches
    if n_pulses > 0:
        avg_switches = torch.rand(n_pulses, 1, device=device) * 35.0 + 5.0
        prob_switch = avg_switches / t_steps
        mask = torch.rand(n_pulses, t_steps, device=device) < prob_switch
        mask[:, 0] = True
        segs = torch.cumsum(mask, dim=1)
        vals = torch.rand(n_pulses, int(segs.max().item()) + 1, device=device) * 2.0 - 1.0
        I_tensor[:n_pulses] = torch.gather(vals, 1, segs)

    # --- TYPE B: White Noise (20%) ---
    # High frequency random walk training
    if n_noise > 0:
        I_tensor[n_pulses : n_pulses + n_noise] = torch.randn(n_noise, t_steps, device=device) * 0.5

    # --- TYPE C: Dense Switching / High Freq (20%) ---
    # Avg 100-500 switches! This forces the model to learn high-frequency preservation.
    if n_dense > 0:
        avg_switches_dense = torch.rand(n_dense, 1, device=device) * 400.0 + 100.0
        prob_switch_dense = avg_switches_dense / t_steps
        mask_dense = torch.rand(n_dense, t_steps, device=device) < prob_switch_dense
        mask_dense[:, 0] = True
        segs_dense = torch.cumsum(mask_dense, dim=1)
        vals_dense = torch.rand(n_dense, int(segs_dense.max().item()) + 1, device=device) * 2.0 - 1.0
        I_tensor[-n_dense:] = torch.gather(vals_dense, 1, segs_dense)

    # 6. Solve Physics (Ground Truth V)
    I_physical = I_tensor * 1e-3
    V_physical = solve_rc_ode(I_physical, R_vals, C_vals, dt_vals)

    # 7. Normalize Outputs (V_hat)
    V_hat = V_physical / (1e-3 * R_vals)

    # 8. Stack Inputs [I_hat, Lambda]
    train_x = torch.stack([I_tensor, lambda_vals.expand(-1, t_steps)], dim=1)

    return train_x, V_hat.unsqueeze(1)


# %% [markdown]
### 4. Non-Dimensional Data Generator
# UPDATED: More aggressive "Super-Dense" switching to fix Chirp frequency response.
# 1. Standard Pulses (50%)
# 2. Gaussian Noise (25%)
# 3. Super-Dense Switching (25%) - Pushes frequency to the grid limit.
# CHANGED: Reverted Tau sampling to Log-Uniform (Bathtub is redundant with spectral aug).


# %%
def generate_dimensionless_data_with_white_noise_and_chirp_log_uniform(n_samples=2000, t_steps=2048, device="cuda"):
    """
    Generates data normalized by physics principles.
    Includes Aggressive Spectral Augmentation.
    """
    # 1. Sample Tau (Log-Uniform) - Physical units
    min_tau, max_tau = 1e-13, 10.0

    log_min = np.log(min_tau)
    log_max = np.log(max_tau)

    # Explicitly view as [N, 1] to guarantee correct broadcasting
    log_tau = torch.rand(n_samples, 1, device=device) * (log_max - log_min) + log_min
    tau_vals = torch.exp(log_tau)  # Shape: [n_samples, 1]

    # 2. Determine Window (T_end) relative to Tau
    ratio_mult = torch.exp(torch.rand(n_samples, 1, device=device) * (np.log(100.0) - np.log(0.1)) + np.log(0.1))

    # This multiplication was the source of the [5, 5] bug.
    # Now guaranteed to be [N, 1] * [N, 1] -> [N, 1]
    t_end_vals = tau_vals * ratio_mult
    dt_vals = t_end_vals / t_steps

    # 3. Calculate Lambda
    lambda_vals = tau_vals / t_end_vals

    # 4. Physical R and C
    min_R, max_R = 100.0, 50000.0
    log_R = torch.rand(n_samples, 1, device=device) * (np.log(max_R) - np.log(min_R)) + np.log(min_R)
    R_vals = torch.exp(log_R)
    C_vals = tau_vals / R_vals

    # 5. Generate Current I(t) - MIXED STRATEGY
    I_tensor = torch.zeros(n_samples, t_steps, device=device)

    # Split indices (50% Standard, 25% Noise, 25% Super-Dense)
    n_pulses = int(n_samples * 0.5)
    n_noise = int(n_samples * 0.25)
    n_dense = n_samples - n_pulses - n_noise

    # --- TYPE A: Standard Pulses (50%) ---
    if n_pulses > 0:
        avg_switches = torch.rand(n_pulses, 1, device=device) * 35.0 + 5.0
        prob_switch = avg_switches / t_steps
        mask = torch.rand(n_pulses, t_steps, device=device) < prob_switch
        mask[:, 0] = True
        segs = torch.cumsum(mask, dim=1)
        vals = torch.rand(n_pulses, int(segs.max().item()) + 1, device=device) * 2.0 - 1.0
        I_tensor[:n_pulses] = torch.gather(vals, 1, segs)

    # --- TYPE B: White Noise (25%) ---
    if n_noise > 0:
        I_tensor[n_pulses : n_pulses + n_noise] = torch.randn(n_noise, t_steps, device=device) * 0.5

    # --- TYPE C: Super-Dense Switching (25%) ---
    # We want switching every ~4-10 pixels to mimic high-freq chirps
    if n_dense > 0:
        avg_switches_dense = torch.rand(n_dense, 1, device=device) * 400.0 + 100.0
        prob_switch_dense = avg_switches_dense / t_steps
        mask_dense = torch.rand(n_dense, t_steps, device=device) < prob_switch_dense
        mask_dense[:, 0] = True
        segs_dense = torch.cumsum(mask_dense, dim=1)
        vals_dense = torch.rand(n_dense, int(segs_dense.max().item()) + 1, device=device) * 2.0 - 1.0
        I_tensor[-n_dense:] = torch.gather(vals_dense, 1, segs_dense)

    # 6. Solve Physics (Ground Truth V)
    I_physical = I_tensor * 1e-3
    V_physical = solve_rc_ode(I_physical, R_vals, C_vals, dt_vals)

    # 7. Normalize Outputs (V_hat)
    V_hat = V_physical / (1e-3 * R_vals)

    # 8. Stack Inputs [I_hat, Lambda]
    train_x = torch.stack([I_tensor, lambda_vals.expand(-1, t_steps)], dim=1)

    return train_x, V_hat.unsqueeze(1)


# %% [markdown]
### 5. Data Visualization
# Verify that identical Lambda produces identical shapes, regardless of Time Scale.


# %% [markdown]
### 7. Evaluation (De-Normalization)
# This function is CRITICAL. It maps the unitless predictions back to Volts.


# %%
def evaluate_ic_spectrum(model, device="cuda", display=False):
    model.eval()

    print("=" * 60)
    print("RUNNING STANDARD TEST")
    print("=" * 60)

    # Define Corners with physical units
    test_cases = [
        {"R": 1000.0, "C": 100e-15, "T_end": 1e-9, "Label": "Parasitic (Tau=100ps)"},
        {"R": 1000.0, "C": 20e-9, "T_end": 200e-6, "Label": "Fast (Tau=20us)"},
        {"R": 10000.0, "C": 100e-9, "T_end": 0.02, "Label": "Standard (Tau=1ms)"},
        {"R": 100000.0, "C": 10e-6, "T_end": 10.0, "Label": "Deep Saturation (Tau=1s)"},
    ]

    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(len(test_cases), 2, width_ratios=[2, 1])

    t_steps = 2048
    r2_data = {}

    for i, case in enumerate(test_cases):
        R_val = case["R"]
        C_val = case["C"]
        T_end = case["T_end"]
        title = case["Label"]

        # 1. Physics Calculations
        tau = R_val * C_val
        Lambda = tau / T_end
        dt = T_end / t_steps
        t_axis = np.linspace(0, T_end, t_steps)

        # 2. Generate Input (Unitless)
        n_switches = 20
        idx = np.sort(np.random.choice(np.arange(1, t_steps - 1), n_switches, replace=False))
        idx = np.concatenate(([0], idx, [t_steps]))
        I_np = np.zeros(t_steps)
        for j in range(len(idx) - 1):
            I_np[idx[j] : idx[j + 1]] = np.random.uniform(-1.0, 1.0)  # +/- 1 unit current

        # 3. Ground Truth (Physical)
        # Scale input to physical amps (e.g. 1mA range)
        I_physical = I_np * 1e-3

        v_true = np.zeros_like(I_np)
        decay = np.exp(-dt / tau)
        gain = R_val * (1.0 - decay)
        v_curr = 0.0
        for t in range(t_steps):
            v_next = v_curr * decay + I_physical[t] * gain
            v_true[t] = v_next
            v_curr = v_next

        # 4. Prepare Model Input [I_unitless, Lambda]
        I_tens = torch.tensor(I_np, dtype=torch.float32).to(device)
        Lambda_tens = torch.full_like(I_tens, Lambda)

        x_in = torch.stack([I_tens, Lambda_tens], dim=0).unsqueeze(0)

        # 5. Predict & De-Normalize
        with torch.no_grad():
            v_pred_hat = model(x_in).cpu().numpy().flatten()

        # De-Norm: V = V_hat * (I_scale * R)
        v_pred = v_pred_hat * (1e-3 * R_val)

        # 3. Metrics
        mse = np.mean((v_true - v_pred) ** 2)
        ss_res = np.sum((v_true - v_pred) ** 2)
        ss_tot = np.sum((v_true - np.mean(v_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-9))
        r2_data[title] = r2

        print(f"{title:<40} | MSE: {mse:.2e} | R2: {r2:.5f}")

        # Plot
        ax = plt.subplot(gs[i, 0])
        ax2 = ax.twinx()
        ax2.step(t_axis, I_physical * 1000, color="gray", alpha=0.3)  # mA
        ax.plot(t_axis, v_true, "w--", label="True")
        ax.plot(t_axis, v_pred, "r-", alpha=0.8, label=f"Pred R2={r2:.4f}")
        ax.set_title(f"{case['Label']} | Lambda={Lambda:.4f}")
        ax.legend()

        ax_p = plt.subplot(gs[i, 1])
        ax_p.scatter(v_true, v_pred, alpha=0.1)
        ax_p.plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], "r--")
        ax_p.set_title(f"MSE: {mse:.2e}")

    plt.tight_layout()
    if display:
        plt.show()
    return (fig, r2_data)


# %%
def evaluate_adversarial_spectrum(model, device="cuda", display=False):
    model.eval()
    print("=" * 60)
    print("RUNNING ADVERSARIAL STRESS TEST")
    print("=" * 60)

    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1])

    # ---------------------------------------------------------
    # TEST 1: The "Corner Frequency" (Lambda = 1.0)
    # ---------------------------------------------------------
    # The hardest regime: Tau equals the Window length.
    # The circuit is neither purely resistive nor purely capacitive.

    T_end_1 = 1e-3  # 1ms
    t_steps_1 = 2048
    R_1, C_1 = 1000.0, 1e-6  # Tau = 1ms -> Lambda = 1.0
    r2_data = {}

    # Input: Standard pulses
    I_np_1 = np.zeros(t_steps_1)
    n_switches = 15
    idx = np.sort(np.random.choice(np.arange(1, t_steps_1 - 1), n_switches, replace=False))
    idx = np.concatenate(([0], idx, [t_steps_1]))
    for j in range(len(idx) - 1):
        I_np_1[idx[j] : idx[j + 1]] = np.random.uniform(-1.0, 1.0)

    # ---------------------------------------------------------
    # TEST 2: The "White Noise" Input
    # ---------------------------------------------------------
    # Can the model integrate noise? (Brownian Motion check)

    T_end_2 = 1e-3
    t_steps_2 = 2048
    R_2, C_2 = 1000.0, 1e-6  # Lambda = 1.0

    # Input: Gaussian White Noise
    # We smooth it slightly so Forward Euler isn't totally invalid,
    # but it is substantially "rougher" than training data.
    I_np_2 = np.random.normal(0, 0.5, size=t_steps_2)

    # ---------------------------------------------------------
    # TEST 3: The "Blind" Resolution Test (4096 Steps)
    # ---------------------------------------------------------
    # FNOs should be resolution invariant. We double the resolution.

    T_end_3 = 1e-3
    t_steps_3 = 4096  # <--- DOUBLED RESOLUTION
    R_3, C_3 = 1000.0, 1e-6

    # Input: Standard pulses on fine grid
    I_np_3 = np.zeros(t_steps_3)
    idx_3 = np.sort(np.random.choice(np.arange(1, t_steps_3 - 1), 30, replace=False))
    idx_3 = np.concatenate(([0], idx_3, [t_steps_3]))
    for j in range(len(idx_3) - 1):
        I_np_3[idx_3[j] : idx_3[j + 1]] = np.random.uniform(-1.0, 1.0)

    # ---------------------------------------------------------
    # EXECUTION LOOP
    # ---------------------------------------------------------
    tests = [
        (I_np_1, R_1, C_1, T_end_1, t_steps_1, "Test 1: Lambda=1.0 (Corner Frequency)"),
        (I_np_2, R_2, C_2, T_end_2, t_steps_2, "Test 2: White Noise Input (Integrator Check)"),
        (I_np_3, R_3, C_3, T_end_3, t_steps_3, "Test 3: Double Resolution (4096 Steps)"),
    ]

    for i, (I_np, R_val, C_val, T_end, t_steps, title) in enumerate(tests):
        dt = T_end / t_steps
        t_axis = np.linspace(0, T_end, t_steps) * 1000  # ms

        # 1. Physics Ground Truth
        # Scale to physical amps (1mA range)
        I_physical = I_np * 1e-3

        I_torch = torch.tensor(I_physical, dtype=torch.float32).unsqueeze(0).to(device)
        R_torch = torch.tensor(R_val, dtype=torch.float32).unsqueeze(0).to(device)
        C_torch = torch.tensor(C_val, dtype=torch.float32).unsqueeze(0).to(device)
        dt_torch = torch.tensor(dt, dtype=torch.float32).to(device)

        with torch.no_grad():
            v_true = solve_rc_ode(I_torch, R_torch, C_torch, dt_torch).cpu().numpy().flatten()

        # 2. Model Prediction (Dimensionless)
        # Lambda = Tau / T_end
        tau = R_val * C_val
        Lambda = tau / T_end

        I_tens = torch.tensor(I_np, dtype=torch.float32).to(device)  # Normalized I
        Lambda_tens = torch.full_like(I_tens, Lambda)

        # Stack: [Batch, 2, Time]
        x_in = torch.stack([I_tens, Lambda_tens], dim=0).unsqueeze(0)

        with torch.no_grad():
            v_pred_hat = model(x_in).cpu().numpy().flatten()

        # De-Normalize: V = V_hat * (I_scale * R)
        v_pred = v_pred_hat * (1e-3 * R_val)

        # 3. Metrics
        mse = np.mean((v_true - v_pred) ** 2)
        ss_res = np.sum((v_true - v_pred) ** 2)
        ss_tot = np.sum((v_true - np.mean(v_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-9))
        r2_data[title] = r2

        print(f"{title:<40} | MSE: {mse:.2e} | R2: {r2:.5f}")

        # 4. Plotting
        ax_wave = plt.subplot(gs[i, 0])
        ax_curr = ax_wave.twinx()

        # Input Current
        if i == 1:  # Noise case
            ax_curr.plot(t_axis, I_physical * 1000, color="gray", alpha=0.3, linewidth=0.5)
        else:
            ax_curr.step(t_axis, I_physical * 1000, color="gray", alpha=0.3, where="post")

        ax_curr.set_ylabel("I (mA)", color="gray")

        # Voltages
        ax_wave.plot(t_axis, v_true, "w-", linewidth=2.5, alpha=0.6, label="Ground Truth")
        ax_wave.plot(t_axis, v_pred, "r--", linewidth=1.5, label="Prediction")

        ax_wave.set_title(f"{title}", loc="left", fontsize=12)
        ax_wave.set_ylabel("V (Volts)")
        if i == 2:
            ax_wave.set_xlabel("Time (ms)")
        ax_wave.legend(loc="upper right")
        ax_wave.grid(True, alpha=0.3)

        # Parity
        ax_parity = plt.subplot(gs[i, 1])
        ax_parity.scatter(v_true, v_pred, alpha=0.2, s=5)
        min_v, max_v = v_true.min(), v_true.max()
        if max_v - min_v < 1e-6:  # Handle flat line
            min_v -= 0.1
            max_v += 0.1
        ax_parity.plot([min_v, max_v], [min_v, max_v], "r--")
        ax_parity.set_title(f"R² = {r2:.5f}")
        ax_parity.set_xlabel("True")
        ax_parity.set_ylabel("Pred")

    plt.tight_layout()
    if display:
        plt.show()
    return (fig, r2_data)


# %% [markdown]
### 8. Final OOD Test Suite (Chirp & Sawtooth)
# Tests generalization to signals the model has NEVER seen (Out-Of-Distribution).
# Validates Frequency Response (Chirp) and Polynomial Integration (Sawtooth).


# %%
def evaluate_ood_physics(model, device="cuda", display=False):
    model.eval()
    print("=" * 60)
    print("RUNNING OOD PHYSICS SUITE (Chirp & Sawtooth)")
    print("=" * 60)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

    # Configuration for OOD signals
    # We choose a "Corner Frequency" setup where Tau approx equals T_end
    # This ensures the physics are active/interesting.
    R_val, C_val = 1000.0, 1e-6
    tau = R_val * C_val
    T_end = 2.0 * tau  # 2ms window

    t_steps = 2048
    dt = T_end / t_steps
    t_axis = np.linspace(0, T_end, t_steps)
    r2_data = {}

    # --- SIGNAL GENERATION ---

    # 1. Chirp (Sine Sweep)
    # Frequencies from DC to 5x cutoff frequency
    # I(t) = sin(phi(t)), instantaneous freq increases linearly
    k = 50.0
    I_chirp = np.sin(2 * np.pi * k * (t_axis / T_end) ** 2)

    # 2. Sawtooth
    # Linear ramps
    period = T_end / 4
    I_saw = 2 * (t_axis / period - np.floor(t_axis / period + 0.5))

    tests = [(I_chirp, "Test 1: Chirp (Frequency Response Check)"), (I_saw, "Test 2: Sawtooth (Integration Check)")]

    for i, (I_np, title) in enumerate(tests):
        # 1. Ground Truth (Physical)
        # Scale to 1mA amplitude
        I_physical = I_np * 1e-3

        # Solve ODE
        I_torch = torch.tensor(I_physical, dtype=torch.float32).unsqueeze(0).to(device)
        R_torch = torch.tensor(R_val, dtype=torch.float32).unsqueeze(0).to(device)
        C_torch = torch.tensor(C_val, dtype=torch.float32).unsqueeze(0).to(device)
        dt_vals = torch.tensor(dt, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            v_true = solve_rc_ode(I_torch, R_torch, C_torch, dt_vals).cpu().numpy().flatten()

        # 2. Model Prediction (Dimensionless)
        # Lambda = Tau / T_end
        Lambda = tau / T_end

        I_tens = torch.tensor(I_np, dtype=torch.float32).to(device)  # Unitless Input
        Lambda_tens = torch.full_like(I_tens, Lambda)

        x_in = torch.stack([I_tens, Lambda_tens], dim=0).unsqueeze(0)

        with torch.no_grad():
            v_pred_hat = model(x_in).cpu().numpy().flatten()

        # De-Normalize
        v_pred = v_pred_hat * (1e-3 * R_val)

        # 3. Metrics
        mse = np.mean((v_true - v_pred) ** 2)
        ss_res = np.sum((v_true - v_pred) ** 2)
        ss_tot = np.sum((v_true - np.mean(v_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-9))
        r2_data[title] = r2

        print(f"{title:<40} | MSE: {mse:.2e} | R2: {r2:.5f}")

        # 4. Plotting
        ax_wave = plt.subplot(gs[i, 0])
        ax_curr = ax_wave.twinx()

        # Plot Input I
        ax_curr.plot(t_axis * 1000, I_physical * 1000, color="gray", alpha=0.3, label="Input I")
        ax_curr.set_ylabel("I (mA)", color="gray")

        # Plot Voltages
        ax_wave.plot(t_axis * 1000, v_true, "w-", linewidth=2.5, label="True")
        ax_wave.plot(t_axis * 1000, v_pred, "r--", linewidth=1.5, label="Pred")

        ax_wave.set_title(f"{title}\nWindow: {T_end*1000:.1f}ms", loc="left")
        ax_wave.set_ylabel("V (Volts)")
        ax_wave.legend(loc="upper right")
        ax_wave.grid(True, alpha=0.3)
        if i == 1:
            ax_wave.set_xlabel("Time (ms)")

        # Parity
        ax_p = plt.subplot(gs[i, 1])
        ax_p.scatter(v_true, v_pred, alpha=0.1)
        min_v, max_v = v_true.min(), v_true.max()
        ax_p.plot([min_v, max_v], [min_v, max_v], "r--")
        ax_p.set_title(f"R² = {r2:.5f}")
        ax_p.set_xlabel("Ground Truth")
        ax_p.set_ylabel("Prediction")

    plt.tight_layout()
    if display:
        plt.show()
    return (fig, r2_data)


# %% [markdown]
### 6. Training Loop (Dimensionless)
# Model now takes 2 channels: I and Lambda.
# Loss is pure shape matching.
# UPDATED: Aggressive schedule based on rapid convergence.


# %%
def run_experiment(
    data_generator,
    experiment_name,
    n_samples: int = 10000,
    t_steps: int = 2048,
    epochs: int = 500,
    dead_zone_epochs: int = 50,
    warmup_epochs: int = 150,
    fine_tune_epochs: int = 100,
    target_sobolev_weight: float = 1e-2,
    target_physics_weight: float = 1e-4,
    batch_size: int = 64,
    starting_lr: int = 1e-3,
    adam_weight_decay: int = 1e-5,
):
    params = locals().copy()
    params["data_generator"] = data_generator.__name__
    # Generate Unique ID
    unique_id = base64.b64encode(json.dumps(params, sort_keys=True).encode("utf-8")).decode("utf-8")[:8]
    run_name = f"{experiment_name}_{unique_id}"
    # writer = SummaryWriter(log_dir=f"runs/{run_name}")
    # writer.add_text("hyperparameters", json.dumps(params, indent=2))

    print(f"Starting Experiment: {run_name}")

    # 2. Data
    print(f"Generating {n_samples} samples with {params['data_generator']}...")
    train_x, train_y = data_generator(n_samples=n_samples, t_steps=t_steps)
    dataset = TensorDataset(train_x.cpu(), train_y.cpu())
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Model & Optimization
    model = get_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=starting_lr, weight_decay=adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=fine_tune_epochs, T_mult=1, eta_min=1e-6)
    loss_fn = GenericDimensionlessPhysicsLoss(
        sobolev_weight=target_sobolev_weight, physics_weight=target_physics_weight, physics_residual_fn=rc_physics_residual
    )

    # 4. Training Loop
    print(f"Starting Training...")
    total_warmup_epochs = dead_zone_epochs + warmup_epochs

    for epoch in range(epochs):
        # Fade-In Logic
        if epoch < dead_zone_epochs:
            alpha = 0.0
        elif epoch < total_warmup_epochs:
            alpha = (epoch - dead_zone_epochs) / warmup_epochs
        elif epoch < (epochs - fine_tune_epochs):
            alpha = 1.0
        else:
            alpha = 0.0  # Polishing

        loss_fn.sobolev_weight = target_sobolev_weight * alpha
        loss_fn.physics_weight = target_physics_weight * alpha

        model.train()

        total_loss = torch.tensor(0.0, device="cuda")
        total_data = torch.tensor(0.0, device="cuda")
        total_sobolev = torch.tensor(0.0, device="cuda")
        total_physics = torch.tensor(0.0, device="cuda")

        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            pred = model(x)
            loss, l_data, l_sobolev, l_physics = loss_fn(pred, y, x)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.detach()
            total_data += l_data.detach()
            total_sobolev += l_sobolev.detach()
            total_physics += l_physics.detach()

        scheduler.step()

        if epoch % 10 == 0:
            avg_loss = total_loss.item() / len(train_loader)
            avg_data = total_data.item() / len(train_loader)
            avg_sob = total_sobolev.item() / len(train_loader)
            avg_phys = total_physics.item() / len(train_loader)
            # writer.add_scalar("Loss/train", avg_loss, epoch)
            # writer.add_scalar("DataLoss/train", avg_data, epoch)
            # writer.add_scalar("PhysicsLoss/train", avg_phys, epoch)
            # writer.add_scalar("Sobolev/train", avg_sob, epoch)
            # writer.add_scalar("Params/alpha", alpha, epoch)
            # writer.add_scalar("Params/lr", optimizer.param_groups[0]["lr"], epoch)

    # 5. Save & Evaluate
    # Path("models").mkdir(exist_ok=True)
    # torch.save(model.state_dict(), f"models/{run_name}.pt")

    # all_metrics = {"hparam/loss": avg_loss}  # Start with final loss

    # evaluations = [
    #     ("Spectrum", evaluate_ic_spectrum),
    #     ("Adversarial", evaluate_adversarial_spectrum),
    #     ("OOD", evaluate_ood_physics),
    # ]

    # for evaluation_name, evaluation_fn in evaluations:
    #     print(f"Running Eval: {evaluation_name}...")
    #     Path(f"figures/{evaluation_name}").mkdir(parents=True, exist_ok=True)
    #     figure, r2_dict = evaluation_fn(model)
    #     figure.savefig(f"figures/{evaluation_name}/{run_name}.png")
    #     writer.add_figure(f"Evaluation/{evaluation_name}", figure)
    #     plt.close(figure)
    #     for k, v in r2_dict.items():
    #         clean_key = k.replace(" ", "_").replace(":", "").replace("=", "")
    #         all_metrics[f"hparam/r2_{evaluation_name}_{clean_key}"] = v
    # writer.add_hparams(hparam_dict=params, metric_dict=all_metrics)
    # writer.close()
    # backup_artifacts(run_name)
    print("Run complete.")
    return model


# %%
if __name__ == "__main__":
    for experiment in [
        ("simple_rc/Dimensionless", generate_dimensionless_data),
        ("simple_rc/Dimensionless_With_Gaussian_Noise", generate_dimensionless_data_with_white_noise),
        ("simple_rc/Dimensionless_With_Gaussian_Noise_And_Chirp", generate_dimensionless_data_with_white_noise_and_chirp),
        (
            "simple_rc/Dimensionless_With_Gaussian_Noise_And_Chirp_Log_Uniform",
            generate_dimensionless_data_with_white_noise_and_chirp_log_uniform,
        ),
    ]:
        name, generator = experiment
        run_experiment(experiment_name=name, data_generator=generator)

# %%
