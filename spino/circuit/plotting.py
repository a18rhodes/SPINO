"""
Publication-quality plot helpers for the CS amp characterization.

All plots are rendered to file (no interactive display) and use a consistent
style suitable for inclusion in ``docs/cs_amp.md`` and the project paper.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402  # headless rendering before pyplot imports the backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from spino.circuit.simulation import DCSweepResult, TransientResult  # noqa: E402
from spino.circuit.tuning import DesignPoint, Metrics, SweepResult  # noqa: E402

__all__ = ["plot_gain_heatmap", "plot_step_response", "plot_vtc"]

_FIGSIZE_HEATMAP = (7.0, 5.5)
_FIGSIZE_TRACE = (6.5, 4.5)
_DPI = 150
_OUTPUT_NODE = "v(out)"


def _ensure_parent(path: str | Path) -> Path:
    """
    Resolves the parent directory and creates it if missing.

    :param path: Target file path.
    :return: Path object pointing at the file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_gain_heatmap(sweep: SweepResult, output_path: str | Path, *, selected: DesignPoint | None = None) -> Path:
    """
    Renders the peak-gain heatmap over :math:`(W_n, W_p)`.

    The colormap encodes :math:`|peak\\,gain|` in V/V; failed points appear
    as missing cells. When ``selected`` is provided, the chosen design is
    annotated with a red cross.

    :param sweep: Completed sweep result.
    :param output_path: Destination file path (PNG).
    :param selected: Optional design point to mark on the heatmap.
    :return: The destination path actually written.
    """
    target = _ensure_parent(output_path)
    n_axis, p_axis = sweep.axes()
    grid = sweep.gain_grid()
    fig, ax = plt.subplots(figsize=_FIGSIZE_HEATMAP)
    masked = np.ma.array(grid, mask=np.isnan(grid))
    mesh = ax.pcolormesh(n_axis, p_axis, masked.T, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$W_n$ ($\mu$m)")
    ax.set_ylabel(r"$W_p$ ($\mu$m)")
    ax.set_title("CS amplifier peak gain over device sizing")
    fig.colorbar(mesh, ax=ax, label=r"Peak $|$gain$|$ (V/V)")
    if selected is not None:
        ax.plot(
            selected.nfet_w_um,
            selected.pfet_w_um,
            marker="x",
            color="red",
            markersize=14,
            markeredgewidth=2.5,
            linestyle="None",
            label="Selected",
        )
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    return target


def plot_vtc(vtc: DCSweepResult, output_path: str | Path, *, design: DesignPoint, metrics: Metrics) -> Path:
    """
    Plots the voltage transfer characteristic with the peak-gain bias marker.

    :param vtc: DC sweep result for the chosen design.
    :param output_path: Destination file path (PNG).
    :param design: Sizing label for the title.
    :param metrics: Extracted metrics (used to annotate the bias point).
    :return: The destination path actually written.
    """
    target = _ensure_parent(output_path)
    vin = vtc.sweep_values
    vout = vtc.variables[_OUTPUT_NODE]
    fig, ax = plt.subplots(figsize=_FIGSIZE_TRACE)
    ax.plot(vin, vout, color="#0066cc", linewidth=1.6)
    ax.axvline(
        metrics.vin_at_peak_gain_v,
        color="#cc0000",
        linestyle="--",
        alpha=0.7,
        label=rf"$V_{{in,bias}} = {metrics.vin_at_peak_gain_v:.3f}$ V",
    )
    ax.axhline(metrics.vout_at_peak_gain_v, color="#cc0000", linestyle=":", alpha=0.5)
    ax.set_xlabel(r"$V_{in}$ (V)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(
        f"VTC: $W_n = {design.nfet_w_um:g}$ µm, $W_p = {design.pfet_w_um:g}$ µm "
        f"(peak |gain| = {metrics.peak_gain_v_per_v:.2f} V/V)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    return target


def plot_step_response(
    tran: TransientResult,
    output_path: str | Path,
    *,
    design: DesignPoint,
    metrics: Metrics,
    t_step_start: float,
    t_window_s: float | None = None,
) -> Path:
    """
    Plots the transient step response with a settling-time marker.

    The simulation window typically extends well beyond the step transient so
    that ``v_final`` can be estimated reliably for settling extraction. When
    ``t_window_s`` is provided, the plot zooms to ``[0, t_step_start + t_window_s]``
    so a fast settling transient is visible without re-running SPICE on a
    shorter horizon.

    :param tran: Transient result for the chosen design.
    :param output_path: Destination file path (PNG).
    :param design: Sizing label for the title.
    :param metrics: Extracted metrics (used to annotate the settling time).
    :param t_step_start: Time at which the step occurs (annotated as a vertical guide).
    :param t_window_s: Optional plot horizon measured from ``t_step_start``.
        When ``None`` the full simulated range is shown.
    :return: The destination path actually written.
    """
    target = _ensure_parent(output_path)
    time_us = tran.time * 1e6
    vout = tran.variables[_OUTPUT_NODE]
    settling_us = metrics.settling_time_s * 1e6 if np.isfinite(metrics.settling_time_s) else float("nan")
    fig, ax = plt.subplots(figsize=_FIGSIZE_TRACE)
    ax.plot(time_us, vout, color="#0066cc", linewidth=1.4)
    ax.axvline(t_step_start * 1e6, color="#444444", linestyle=":", alpha=0.7, label="Step onset")
    if np.isfinite(settling_us):
        ax.axvline(
            t_step_start * 1e6 + settling_us,
            color="#cc0000",
            linestyle="--",
            alpha=0.8,
            label=rf"Settling ($\pm$5%) = {settling_us:.3f} µs",
        )
    if t_window_s is not None:
        ax.set_xlim(0.0, (t_step_start + t_window_s) * 1e6)
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$V_{out}$ (V)")
    ax.set_title(f"Step response: $W_n = {design.nfet_w_um:g}$ µm, $W_p = {design.pfet_w_um:g}$ µm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(target, dpi=_DPI)
    plt.close(fig)
    return target
