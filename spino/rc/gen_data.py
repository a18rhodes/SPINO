# %% [markdown]
### Data Generators
# This module contains functions to generate training data for the RC circuit model.
# This converts physical R, C, T, I into dimensionless groups.
# Output Channels: [I_norm, Lambda]
# Model sees: I_norm (shape) and Lambda (stiffness ratio).
# Model predicts: V_norm (shape).


# %%
import numpy as np
import torch

from spino.rc.solve import solve_rc_ode

# %% [markdown]
### Basic Non-Dimensional Data Generator
# Generates data normalized by physics principles.

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
### Non-Dimensional Data Generator with White Noise
# Extends basic generator by injecting Gaussian Noise into 20% of samples to fix Spectral Bias.


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
### Non-Dimensional Data Generator with White Noise and Chirp
# Extends basic generator by injecting Gaussian Noise and Dense High-Frequency Switching to fix Chirp failure.
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
### Non-Dimensional Data Generator
# Extends basic generator by injecting Gaussian Noise and Dense High-Frequency Switching to fix Chirp failure.
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
