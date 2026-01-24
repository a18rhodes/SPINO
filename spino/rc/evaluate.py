# %% [markdown]
### Evaluation Suite
# This module contains functions to evaluate the trained RC circuit model
# on various test cases, including standard, adversarial, and out-of-distribution scenarios.

# %%
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

from spino.rc.solve import solve_rc_ode

# %% [markdown]
### Evaluation (De-Normalization)
# Evaluation functions map the unitless predictions back to Volts.

# %% [markdown]
### Basic IC Spectrum Test
# Tests the model on a variety of RC time constants (Tau) and evaluates R² performance.

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

# %% [markdown]
### Adversarial Stress Test
# Tests the model under challenging scenarios, including corner frequency, white noise, and resolution changes.

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
### OOD Test Suite (Chirp & Sawtooth)
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
