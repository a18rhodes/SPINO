"""
Evaluation suite for the diode neural operator.

Supports both dimensionless (6-channel, variable T_end) and legacy
(5-channel, fixed 1ms window) modes. The dimensionless mode
denormalizes predictions via V = V_hat * I_SCALE_A * R.

Functions:

- ``evaluate_rectifier``: Standard rectifier vs. SPICE ground truth.
- ``evaluate_adversarial``: Random sample from a data loader.
- ``evaluate_resolution_invariance``: Same circuit at multiple grid sizes.
"""

import logging
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
import torch
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_s

from spino.diode.model import I_SCALE_A
from spino.plot_styles import coerce_palette, get_palette

Logging.setup_logging(logging_level="ERROR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

__all__ = [
    "calculate_r2",
    "compute_metrics",
    "evaluate_adversarial",
    "evaluate_rectifier",
    "evaluate_resolution_invariance",
    "evaluate_variable_t_end",
    "run_spice_rectifier",
]


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the coefficient of determination (R-squared).

    :param y_true: Ground truth array.
    :param y_pred: Predicted array.
    :return: R-squared score.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Computes standard evaluation metrics for voltage predictions.

    Mirrors the metric set used in the MOSFET evaluator (R², MSE, RMSE, MAE in mV).

    :param y_true: Ground truth voltage array (Volts).
    :param y_pred: Predicted voltage array (Volts).
    :return: Dict with keys: r2, mse, rmse, mae_mv.
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "r2": calculate_r2(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae_mv": float(np.mean(np.abs(y_true - y_pred))) * 1000.0,
    }


def run_spice_rectifier(
    t_steps: int,
    t_end: float,
    freq: float,
    R_val: float,
    C_val: float,
    Is_val: float,
    N_val: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Runs a SPICE transient simulation for the standard rectifier test case.

    :param t_steps: Number of output grid points.
    :param t_end: Simulation end time in seconds.
    :param freq: Input sine wave frequency in Hz.
    :param R_val: Resistance in Ohms.
    :param C_val: Capacitance in Farads (used as CJO).
    :param Is_val: Diode saturation current in Amps.
    :param N_val: Diode ideality factor.
    :return: Tuple of (time_axis, input_current_amps, voltage_true) or (None, None, None) on failure.
    """
    sim_steps = t_steps * 2
    times = np.linspace(0, t_end, sim_steps)
    amps = I_SCALE_A * np.sin(2 * np.pi * freq * times)
    circuit = Circuit("Rectifier_Ground_Truth")
    circuit.model("D1", "D", IS=Is_val, N=N_val, CJO=C_val)
    source_pairs = list(zip(times, amps))
    circuit.PieceWiseLinearCurrentSource("1", "0", "1", values=source_pairs)
    circuit.Diode("1", "1", "0", model="D1")
    circuit.Resistor("1", "1", "0", R_val @ u_Ohm)
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=(t_end / sim_steps) @ u_s, end_time=t_end @ u_s)
        t_spice = np.array(analysis.time)
        v_spice = np.array(analysis["1"])
        target_t = np.linspace(0, t_end, t_steps)
        v_interp = np.interp(target_t, t_spice, v_spice)
        i_interp = I_SCALE_A * np.sin(2 * np.pi * freq * target_t)
        return target_t, i_interp, v_interp
    except Exception as exc:
        logger.error("SPICE evaluation failed: %s", exc)
        return None, None, None


def _style_plot(ax, title: str, xlabel: str, ylabel: str, palette=None):
    """Applies consistent styling to an axis."""
    p = coerce_palette(palette)
    ax.set_title(title, fontsize=11, fontweight="bold", color=p["title"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.tick_params(axis="both", which="major", labelsize=9)


def _prepare_dimensionless_input(
    i_amps: np.ndarray,
    R_val: float,
    C_val: float,
    Is_val: float,
    N_val: float,
    t_end: float,
    device: str = "cuda",
) -> tuple[torch.Tensor, float]:
    """
    Constructs the 6-channel dimensionless input tensor.

    :param i_amps: Input current waveform in Amps.
    :param R_val: Resistance in Ohms.
    :param C_val: Capacitance in Farads.
    :param Is_val: Saturation current in Amps.
    :param N_val: Ideality factor.
    :param t_end: Simulation window in seconds.
    :param device: Torch device string.
    :return: Tuple of (x_in [1, 6, T], lambda_val).
    """
    tau = R_val * C_val
    lambda_val = tau / t_end
    i_hat = torch.tensor(i_amps / I_SCALE_A, dtype=torch.float32).to(device)
    ch_lambda = torch.full_like(i_hat, lambda_val)
    ch_logR = torch.full_like(i_hat, np.log10(R_val))
    ch_logC = torch.full_like(i_hat, np.log10(C_val))
    ch_logIs = torch.full_like(i_hat, np.log10(Is_val))
    ch_N = torch.full_like(i_hat, N_val)
    x_in = torch.stack([i_hat, ch_lambda, ch_logR, ch_logC, ch_logIs, ch_N]).unsqueeze(0)
    return x_in, lambda_val


def _prepare_legacy_input(
    i_amps: np.ndarray,
    R_val: float,
    C_val: float,
    Is_val: float,
    N_val: float,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Constructs the 5-channel legacy input tensor (current in mA).

    :param i_amps: Input current waveform in Amps.
    :param R_val: Resistance in Ohms.
    :param C_val: Capacitance in Farads.
    :param Is_val: Saturation current in Amps.
    :param N_val: Ideality factor.
    :param device: Torch device string.
    :return: x_in tensor [1, 5, T].
    """
    i_mA = i_amps / 1e-3
    ch0 = torch.tensor(i_mA, dtype=torch.float32).to(device)
    ch1 = torch.full_like(ch0, np.log10(R_val))
    ch2 = torch.full_like(ch0, np.log10(C_val))
    ch3 = torch.full_like(ch0, np.log10(Is_val))
    ch4 = torch.full_like(ch0, N_val)
    return torch.stack([ch0, ch1, ch2, ch3, ch4]).unsqueeze(0)


def evaluate_rectifier(
    model,
    dimensionless: bool = True,
    t_steps: int = 2048,
    t_end: float = 1e-3,
    device: str = "cuda",
    dark: bool = True,
) -> tuple[plt.Figure, dict[str, float]]:
    """
    Compares FNO prediction against SPICE ground truth for a standard rectifier.

    For dimensionless mode, the model output is denormalized via
    V = V_hat * I_SCALE_A * R.

    :param model: Trained DiodeFNO.
    :param dimensionless: If True, uses 6-channel input and denorms output.
    :param t_steps: Output grid resolution.
    :param t_end: Simulation window in seconds.
    :param device: Torch device string.
    :param dark: If True, uses dark background plotting style.
    :return: Tuple of (matplotlib Figure, metrics dict with keys r2, mse, rmse, mae_mv).
    """
    model.eval()
    plt.style.use("dark_background" if dark else "default")
    p = get_palette(dark)
    # C=10nF: tau=10us, T_end=1ms, lambda=0.01, sim_step=244ns << tau (SPICE-safe, real ceramic cap)
    # C=4pF (old) was the diode's own junction cap -- subgrid at 244ns, not a meaningful test
    R_val, C_val, Is_val, N_val = 1000.0, 10e-9, 2.5e-9, 1.75
    freq = 2000
    t0_spice = time.perf_counter()
    t, i_amps, v_true = run_spice_rectifier(t_steps, t_end, freq, R_val, C_val, Is_val, N_val)
    spice_ms = (time.perf_counter() - t0_spice) * 1000
    if v_true is None:
        return plt.figure(), {"r2": 0.0, "mse": float("nan"), "rmse": float("nan"), "mae_mv": float("nan")}
    if dimensionless:
        x_in, lambda_val = _prepare_dimensionless_input(i_amps, R_val, C_val, Is_val, N_val, t_end, device)
    else:
        x_in = _prepare_legacy_input(i_amps, R_val, C_val, Is_val, N_val, device)
    t0_fno = time.perf_counter()
    with torch.no_grad():
        v_pred_hat = model(x_in).cpu().numpy().flatten()
    fno_ms = (time.perf_counter() - t0_fno) * 1000
    v_pred = v_pred_hat * (I_SCALE_A * R_val) if dimensionless else v_pred_hat
    metrics = compute_metrics(v_true, v_pred)
    logger.info(
        "SPICE: %.0fms | FNO: %.2fms | Speedup: %.0fx | R2=%.4f, MSE=%.2e, MAE=%.2fmV",
        spice_ms, fno_ms, spice_ms / max(fno_ms, 1e-9), metrics["r2"], metrics["mse"], metrics["mae_mv"],
    )
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    i_mA = i_amps / 1e-3
    mode_str = f"Dimensionless (lambda={lambda_val:.4f})" if dimensionless else "Legacy (fixed 1ms)"
    _style_plot(
        ax[0],
        f"Standard Rectifier Check [{mode_str}]\nR2={metrics['r2']:.4f} | MSE={metrics['mse']:.2e} | MAE={metrics['mae_mv']:.2f}mV",
        "Time (ms)",
        "Voltage (V)",
        palette=p,
    )
    ax[0].plot(t * 1000, i_mA, color=p["vg"], linestyle="--", alpha=0.4, label="Input Current (mA)")
    ax[0].plot(t * 1000, v_true, color=p["gt"], linewidth=2.5, alpha=0.6, label="Ground Truth (SPICE)")
    ax[0].plot(t * 1000, v_pred, color=p["pred"], linestyle=":", linewidth=2, label="Prediction (FNO)")
    ax[0].legend(loc="upper right", fontsize=9)
    _style_plot(ax[1], "Dynamic I-V Characteristic", "Voltage (V)", "Current (mA)", palette=p)
    ax[1].plot(v_true, i_mA, color=p["gt"], marker="o", markersize=2, linestyle="None", alpha=0.3, label="True")
    ax[1].plot(v_pred, i_mA, color=p["pred_sweep"], marker="x", markersize=2, linestyle="None", alpha=0.5, label="Pred")
    ax[1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig, metrics


def evaluate_adversarial(
    model,
    data_loader,
    dimensionless: bool = True,
    device: str = "cuda",
    dark: bool = True,
) -> tuple[plt.Figure | None, dict[str, float] | None]:
    """
    Evaluates the model on a random sample drawn from a data loader.

    Supports both legacy (5-channel) and dimensionless (6-channel) loaders.
    In dimensionless mode, predictions are denormalized for display.

    :param model: Trained DiodeFNO.
    :param data_loader: DataLoader yielding (x, y) batches.
    :param dimensionless: If True, expects 6-channel data and denorms output.
    :param device: Torch device string.
    :param dark: If True, uses dark background plotting style.
    :return: Tuple of (matplotlib Figure, metrics dict) or (None, None) on failure.
    """
    model.eval()
    plt.style.use("dark_background" if dark else "default")
    p = get_palette(dark)
    try:
        x, y = next(iter(data_loader))
    except StopIteration:
        return None, None
    x_sample = x[0:1].to(device)
    y_hat_true = y[0].cpu().numpy().flatten()
    with torch.no_grad():
        y_hat_pred = model(x_sample).cpu().numpy().flatten()
    params = x[0, :, 0].cpu().numpy()
    if dimensionless:
        R = 10 ** params[2]
        C = 10 ** params[3]
        Is = 10 ** params[4]
        v_scale = I_SCALE_A * R
        y_true_phys = y_hat_true * v_scale
        y_pred_phys = y_hat_pred * v_scale
        i_hat = x[0, 0, :].cpu().numpy()
        i_label = "I_hat (dimensionless)"
    else:
        R = 10 ** params[1]
        C = 10 ** params[2]
        Is = 10 ** params[3]
        y_true_phys = y_hat_true
        y_pred_phys = y_hat_pred
        i_hat = x[0, 0, :].cpu().numpy()
        i_label = "Input Current (mA)"
    metrics = compute_metrics(y_true_phys, y_pred_phys)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    t_axis = np.linspace(0, 1, len(y_true_phys))
    title_str = f"Adversarial Sample (Random Params)\nR={R:.0f}Ohm, C={C:.1e}F, Is={Is:.1e}A\nR2={metrics['r2']:.4f} | MSE={metrics['mse']:.2e} | MAE={metrics['mae_mv']:.2f}mV"
    _style_plot(ax[0], title_str, "Normalized Time", "Voltage (V)", palette=p)
    ax[0].plot(t_axis, i_hat, color=p["vg"], linestyle="--", alpha=0.3, label=i_label)
    ax[0].plot(t_axis, y_true_phys, color=p["gt"], linewidth=2.5, alpha=0.6, label="Ground Truth")
    ax[0].plot(t_axis, y_pred_phys, color=p["pred"], linestyle=":", linewidth=2, label="Prediction")
    ax[0].legend(loc="upper right", fontsize=9)
    _style_plot(ax[1], "Dynamic I-V Hysteresis Loop", "Voltage (V)", "Current", palette=p)
    ax[1].plot(y_true_phys, i_hat, color=p["gt"], marker="o", markersize=2, linestyle="None", alpha=0.3, label="True")
    ax[1].plot(y_pred_phys, i_hat, color=p["pred_sweep"], marker="x", markersize=2, linestyle="None", alpha=0.5, label="Pred")
    ax[1].legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return fig, metrics


def evaluate_resolution_invariance(
    model,
    dimensionless: bool = True,
    resolutions: tuple[int, ...] = (1024, 2048, 4096),
    device: str = "cuda",
    dark: bool = True,
) -> tuple[plt.Figure, dict[int, float]]:
    """
    Tests resolution invariance by evaluating the same circuit at multiple grid sizes.

    FNOs are theoretically resolution-invariant. This test verifies that the
    prediction quality does not degrade significantly when the number of
    temporal grid points changes from the training resolution.

    :param model: Trained DiodeFNO.
    :param dimensionless: If True, uses dimensionless formulation.
    :param resolutions: Tuple of grid sizes to test.
    :param device: Torch device string.
    :param dark: If True, uses dark background plotting style.
    :return: Tuple of (matplotlib Figure, dict mapping resolution -> R2).
    """
    model.eval()
    plt.style.use("dark_background" if dark else "default")
    p = get_palette(dark)
    R_val, C_val, Is_val, N_val = 1000.0, 10e-9, 2.5e-9, 1.75
    t_end, freq = 1e-3, 2000
    r2_map = {}
    fig, axes = plt.subplots(1, len(resolutions), figsize=(7 * len(resolutions), 6))
    if len(resolutions) == 1:
        axes = [axes]
    for ax, t_steps in zip(axes, resolutions):
        t, i_amps, v_true = run_spice_rectifier(t_steps, t_end, freq, R_val, C_val, Is_val, N_val)
        if v_true is None:
            r2_map[t_steps] = 0.0
            continue
        if dimensionless:
            x_in, lam = _prepare_dimensionless_input(i_amps, R_val, C_val, Is_val, N_val, t_end, device)
        else:
            x_in = _prepare_legacy_input(i_amps, R_val, C_val, Is_val, N_val, device)
        with torch.no_grad():
            v_pred_hat = model(x_in).cpu().numpy().flatten()
        v_pred = v_pred_hat * (I_SCALE_A * R_val) if dimensionless else v_pred_hat
        r2 = calculate_r2(v_true, v_pred)
        mse = np.mean((v_true - v_pred) ** 2)
        r2_map[t_steps] = r2
        logger.info("Resolution %d: MSE=%.2e, R2=%.4f", t_steps, mse, r2)
        i_mA = i_amps / 1e-3
        _style_plot(ax, f"T={t_steps} | R2={r2:.4f} | MSE={mse:.2e}", "Time (ms)", "Voltage (V)", palette=p)
        ax.plot(t * 1000, v_true, color=p["gt"], linewidth=2.5, alpha=0.6, label="SPICE")
        ax.plot(t * 1000, v_pred, color=p["pred"], linestyle=":", linewidth=2, label="FNO")
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig, r2_map


def evaluate_variable_t_end(
    model,
    dimensionless: bool = True,
    windows: tuple[float, ...] = (100e-6, 1e-3, 10e-3),
    t_steps: int = 2048,
    device: str = "cuda",
    dark: bool = True,
) -> tuple[plt.Figure, dict[float, dict[str, float]]]:
    """
    Tests time-scale invariance by evaluating the same circuit at multiple simulation windows.

    The FNO's dimensionless formulation encodes all time-scale information in lambda=RC/T_end.
    This test verifies that prediction quality does not degrade as T_end varies, confirming that
    lambda is a sufficient stiffness descriptor and the model is not implicitly coupled to the
    training window distribution.

    All three windows use the same circuit parameters (R=1kΩ, C=10nF); lambda covers
    {0.1, 0.01, 0.001} across the three windows -- all within the training distribution.

    :param model: Trained DiodeFNO.
    :param dimensionless: If True, uses dimensionless formulation.
    :param windows: Simulation end times in seconds to test.
    :param t_steps: Grid resolution (same for all windows).
    :param device: Torch device string.
    :param dark: If True, uses dark background plotting style.
    :return: Tuple of (matplotlib Figure, dict mapping t_end -> metrics dict).
    """
    model.eval()
    plt.style.use("dark_background" if dark else "default")
    p = get_palette(dark)
    R_val, C_val, Is_val, N_val = 1000.0, 10e-9, 2.5e-9, 1.75
    # Keep 2 cycles visible regardless of window -- freq scales with T_end
    freq_map = {t_end: 2.0 / t_end for t_end in windows}
    tau = R_val * C_val
    metrics_map: dict[float, dict[str, float]] = {}
    fig, axes = plt.subplots(1, len(windows), figsize=(7 * len(windows), 6))
    if len(windows) == 1:
        axes = [axes]
    for ax, t_end in zip(axes, windows):
        freq = freq_map[t_end]
        lambda_val = tau / t_end
        t, i_amps, v_true = run_spice_rectifier(t_steps, t_end, freq, R_val, C_val, Is_val, N_val)
        if v_true is None:
            metrics_map[t_end] = {"r2": 0.0, "mse": float("nan"), "rmse": float("nan"), "mae_mv": float("nan")}
            continue
        if dimensionless:
            x_in, _ = _prepare_dimensionless_input(i_amps, R_val, C_val, Is_val, N_val, t_end, device)
        else:
            x_in = _prepare_legacy_input(i_amps, R_val, C_val, Is_val, N_val, device)
        with torch.no_grad():
            v_pred_hat = model(x_in).cpu().numpy().flatten()
        v_pred = v_pred_hat * (I_SCALE_A * R_val) if dimensionless else v_pred_hat
        metrics = compute_metrics(v_true, v_pred)
        metrics_map[t_end] = metrics
        t_end_label = f"{t_end * 1e6:.0f}us" if t_end < 1e-3 else f"{t_end * 1e3:.0f}ms"
        logger.info(
            "T_end=%s (lambda=%.4f): R2=%.4f, MSE=%.2e, MAE=%.2fmV",
            t_end_label, lambda_val, metrics["r2"], metrics["mse"], metrics["mae_mv"],
        )
        _style_plot(
            ax,
            f"T_end={t_end_label} | lambda={lambda_val:.4f}\nR2={metrics['r2']:.4f} | MAE={metrics['mae_mv']:.2f}mV",
            f"Time ({t_end_label})",
            "Voltage (V)",
            palette=p,
        )
        t_norm = t / t_end
        ax.plot(t_norm, v_true, color=p["gt"], linewidth=2.5, alpha=0.6, label="SPICE")
        ax.plot(t_norm, v_pred, color=p["pred"], linestyle=":", linewidth=2, label="FNO")
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig, metrics_map
