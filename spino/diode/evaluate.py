import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
import torch
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_s

# Suppress SPICE output
Logging.setup_logging(logging_level="ERROR")


def calculate_r2(y_true, y_pred):
    """Calculates R2 Score (Coefficient of Determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def run_spice_rectifier(t_steps, t_end, freq, R_val, C_val, Is_val, N_val):
    """
    Runs a ground-truth SPICE simulation for the standard Rectifier test case.
    Returns: (time_axis, input_current, voltage_true)
    """
    # 1. Setup Time and Input
    sim_steps = t_steps * 2
    times = np.linspace(0, t_end, sim_steps)

    # Input Current: Sine wave +/- 5mA
    amps = 5e-3 * np.sin(2 * np.pi * freq * times)

    # 2. Build Circuit
    circuit = Circuit("Rectifier_Ground_Truth")
    circuit.model("D1", "D", IS=Is_val, N=N_val, CJO=C_val)

    source_pairs = list(zip(times, amps))
    circuit.PieceWiseLinearCurrentSource("1", "0", "1", values=source_pairs)
    circuit.Diode("1", "1", "0", model="D1")
    circuit.Resistor("1", "1", "0", R_val @ u_Ohm)

    # 3. Simulate
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=(t_end / sim_steps) @ u_s, end_time=t_end @ u_s)

        # 4. Interpolate to target grid
        t_spice = np.array(analysis.time)
        v_spice = np.array(analysis["1"])

        target_t = np.linspace(0, t_end, t_steps)
        v_interp = np.interp(target_t, t_spice, v_spice)
        i_interp = 5.0 * np.sin(2 * np.pi * freq * target_t)  # mA for plotting

        return target_t, i_interp, v_interp

    except Exception as e:
        print(f"SPICE Evaluation Failed: {e}")
        return None, None, None


def _style_plot(ax, title, xlabel, ylabel):
    """Helper to apply consistent styling matching RC plots."""
    ax.set_title(title, fontsize=11, fontweight="bold", color="white")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=9)


def evaluate_rectifier(model, device="cuda"):
    """
    Visual Validation: Comparison against ground truth SPICE for the standard rectifier.
    """
    model.eval()

    # Standard Test Parameters
    R_val, C_val, Is_val, N_val = 1000.0, 4e-12, 2.5e-9, 1.75
    t_steps, t_end, freq = 2048, 1e-3, 2000

    # 1. Run Ground Truth Simulation
    t, I_mA, v_true = run_spice_rectifier(t_steps, t_end, freq, R_val, C_val, Is_val, N_val)

    if v_true is None:
        return plt.figure()

    # 2. Prepare Model Input
    # Input format: [I(mA), logR, logC, logIs, N]
    ch0 = torch.tensor(I_mA, dtype=torch.float32).to(device)
    ch1 = torch.full_like(ch0, np.log10(R_val))
    ch2 = torch.full_like(ch0, np.log10(C_val))
    ch3 = torch.full_like(ch0, np.log10(Is_val))
    ch4 = torch.full_like(ch0, N_val)

    x_in = torch.stack([ch0, ch1, ch2, ch3, ch4]).unsqueeze(0)

    # 3. Predict
    with torch.no_grad():
        v_pred = model(x_in).cpu().numpy().flatten()

    # 4. Metrics
    mse = np.mean((v_true - v_pred) ** 2)
    r2 = calculate_r2(v_true, v_pred)

    # 5. Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Time Domain
    _style_plot(ax[0], f"Standard Rectifier Check\nMSE: {mse:.2e} | R2: {r2:.4f}", "Time (ms)", "Voltage (V)")
    ax[0].plot(t * 1000, I_mA, color="#00ff00", linestyle="--", alpha=0.4, label="Input Current (mA)")
    ax[0].plot(t * 1000, v_true, color="#ffffff", linewidth=2.5, alpha=0.6, label="Ground Truth (SPICE)")
    ax[0].plot(t * 1000, v_pred, color="#00ffff", linestyle=":", linewidth=2, label="Prediction (FNO)")
    ax[0].legend(loc="upper right", fontsize=9)

    # IV Curve
    _style_plot(ax[1], "Dynamic I-V Characteristic", "Voltage (V)", "Current (mA)")
    ax[1].plot(v_true, I_mA, color="#ffffff", marker="o", markersize=2, linestyle="None", alpha=0.3, label="True")
    ax[1].plot(v_pred, I_mA, color="#ff00ff", marker="x", markersize=2, linestyle="None", alpha=0.5, label="Pred")
    ax[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig, r2


def evaluate_adversarial(model, data_loader, device="cuda"):
    """
    Evaluates the model on a random "Adversarial" sample from the generator.
    These samples are effectively OOD/Adversarial because the parameters are random.
    """
    model.eval()

    # Fetch one batch
    try:
        x, y = next(iter(data_loader))
    except StopIteration:
        return None, None

    # Pick the first sample
    x_sample = x[0:1].to(device)
    y_true = y[0:1].cpu().numpy().flatten()

    # Predict
    with torch.no_grad():
        y_pred = model(x_sample).cpu().numpy().flatten()

    # Extract params for title
    # Channels: [I, logR, logC, logIs, N]
    I_mA = x[0, 0, :].cpu().numpy()
    params = x[0, :, 0].cpu().numpy()
    R = 10 ** params[1]
    C = 10 ** params[2]
    Is = 10 ** params[3]

    # Metrics
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = calculate_r2(y_true, y_pred)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    t_axis = np.linspace(0, 1, len(y_true))  # Normalized time

    # Time Domain
    title_str = (
        f"Adversarial Sample (Random Params)\nR={R:.0f}Ω, C={C:.1e}F, Is={Is:.1e}A\nMSE: {mse:.2e} | R2: {r2:.4f}"
    )
    _style_plot(ax[0], title_str, "Normalized Time", "Voltage (V)")

    ax[0].plot(t_axis, I_mA, color="#00ff00", linestyle="--", alpha=0.3, label="Input Current (mA)")
    ax[0].plot(t_axis, y_true, color="#ffffff", linewidth=2.5, alpha=0.6, label="Ground Truth")
    ax[0].plot(t_axis, y_pred, color="#00ffff", linestyle=":", linewidth=2, label="Prediction")
    ax[0].legend(loc="upper right", fontsize=9)

    # IV Curve (Hysteresis Check)
    _style_plot(ax[1], "Dynamic I-V Hysteresis Loop", "Voltage (V)", "Current (mA)")
    ax[1].plot(y_true, I_mA, color="#ffffff", marker="o", markersize=2, linestyle="None", alpha=0.3, label="True")
    ax[1].plot(y_pred, I_mA, color="#ff00ff", marker="x", markersize=2, linestyle="None", alpha=0.5, label="Pred")
    ax[1].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    return fig, r2
