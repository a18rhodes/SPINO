"""
Smoke tests for the CS amp plotting helpers.

Verifies that each plotter renders to a valid PNG file from synthetic input
without exercising the SPICE backend. The image content is not asserted; the
intent is to catch API drift and matplotlib breakage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spino.circuit.plotting import plot_gain_heatmap, plot_step_response, plot_vtc
from spino.circuit.simulation import DCSweepResult, TransientResult
from spino.circuit.tuning import DesignPoint, Metrics, SweepResult


def _metrics(gain: float = 8.0, vbias: float = 0.9, settling: float = 1e-7) -> Metrics:
    return Metrics(
        converged=True,
        peak_gain_v_per_v=gain,
        vin_at_peak_gain_v=0.7,
        vout_at_peak_gain_v=vbias,
        static_current_a=2e-6,
        settling_time_s=settling,
    )


@pytest.fixture
def sweep() -> SweepResult:
    n_axis = [0.5, 1.0, 2.0]
    p_axis = [1.0, 2.0]
    points: list[DesignPoint] = []
    metrics: list[Metrics] = []
    for wn in n_axis:
        for wp in p_axis:
            points.append(DesignPoint(nfet_w_um=wn, pfet_w_um=wp))
            metrics.append(_metrics(gain=wn * wp))
    return SweepResult(points=tuple(points), metrics=tuple(metrics))


@pytest.fixture
def vtc() -> DCSweepResult:
    vin = np.linspace(0.0, 1.8, 181)
    vout = 1.8 - 8.0 * np.clip(vin - 0.6, 0.0, 0.2) / 0.2
    return DCSweepResult(sweep_param="v-sweep", sweep_values=vin, variables={"v(out)": vout})


@pytest.fixture
def transient() -> TransientResult:
    time = np.linspace(0.0, 5e-6, 5001)
    tau = 5e-7
    v_initial, v_final = 0.9, 0.4
    vout = np.where(time < 1e-7, v_initial, v_final + (v_initial - v_final) * np.exp(-(time - 1e-7) / tau))
    return TransientResult(time=time, variables={"v(out)": vout})


def test_plot_gain_heatmap_writes_png(sweep, tmp_path):
    output = plot_gain_heatmap(sweep, tmp_path / "heatmap.png")
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_gain_heatmap_with_selection_writes_png(sweep, tmp_path):
    selected = DesignPoint(nfet_w_um=1.0, pfet_w_um=2.0)
    output = plot_gain_heatmap(sweep, tmp_path / "heatmap_sel.png", selected=selected)
    assert output.exists()


def test_plot_vtc_writes_png(vtc, tmp_path):
    design = DesignPoint(nfet_w_um=1.0, pfet_w_um=2.0)
    output = plot_vtc(vtc, tmp_path / "vtc.png", design=design, metrics=_metrics(gain=8.0, vbias=0.9))
    assert output.exists()


def test_plot_step_response_writes_png(transient, tmp_path):
    design = DesignPoint(nfet_w_um=1.0, pfet_w_um=2.0)
    output = plot_step_response(
        transient, tmp_path / "step.png", design=design, metrics=_metrics(settling=1e-6), t_step_start=1e-7
    )
    assert output.exists()


def test_plot_step_response_handles_nan_settling(transient, tmp_path):
    design = DesignPoint(nfet_w_um=1.0, pfet_w_um=2.0)
    output = plot_step_response(
        transient,
        tmp_path / "step_nan.png",
        design=design,
        metrics=_metrics(settling=float("nan")),
        t_step_start=1e-7,
    )
    assert output.exists()


def test_plotting_creates_missing_parents(sweep, tmp_path):
    nested = tmp_path / "a" / "b" / "c" / "heat.png"
    output = plot_gain_heatmap(sweep, nested)
    assert output.exists()
    assert output.parent == tmp_path / "a" / "b" / "c"
