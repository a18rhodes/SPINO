"""
Unit tests for the CS amp tuning harness.

Covers the pure-logic surface (metric extraction, selection rule, sweep result
reshaping) using synthetic NumPy arrays. The simulator-driven orchestration is
covered indirectly by ``test_cs_amp_e2e.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spino.circuit.simulation import DCSweepResult, TransientResult
from spino.circuit.tuning import (
    DesignPoint,
    Metrics,
    SelectionRule,
    SweepResult,
    extract_peak_gain,
    extract_settling_time,
    select_design_point,
)


def _make_vtc(vin: np.ndarray, vout: np.ndarray) -> DCSweepResult:
    """Builds a synthetic DCSweepResult for extractor tests."""
    return DCSweepResult(sweep_param="v-sweep", sweep_values=vin, variables={"v(out)": vout})


def _make_tran(time: np.ndarray, vout: np.ndarray) -> TransientResult:
    """Builds a synthetic TransientResult for extractor tests."""
    return TransientResult(time=time, variables={"v(out)": vout})


def _make_metrics(*, converged: bool, gain: float, vout_bias: float, current: float) -> Metrics:
    """Convenience constructor for selection-rule tests."""
    return Metrics(
        converged=converged,
        peak_gain_v_per_v=gain,
        vin_at_peak_gain_v=0.7,
        vout_at_peak_gain_v=vout_bias,
        static_current_a=current,
        settling_time_s=1e-7,
    )


class TestExtractPeakGain:
    def test_linear_vtc_gain_matches_slope(self):
        vin = np.linspace(0.0, 1.8, 181)
        vout = 1.8 - 5.0 * vin
        gain, vin_peak, vout_peak = extract_peak_gain(_make_vtc(vin, vout))
        assert gain == pytest.approx(5.0, rel=1e-3)
        assert 0.0 <= vin_peak <= 1.8
        assert math.isclose(vout_peak, 1.8 - 5.0 * vin_peak, rel_tol=1e-6)

    def test_picks_steepest_region(self):
        vin = np.linspace(0.0, 1.8, 1801)
        vout = np.where(vin < 0.6, 1.6, np.where(vin > 1.0, 0.2, 1.6 - 3.5 * (vin - 0.6)))
        gain, vin_peak, _ = extract_peak_gain(_make_vtc(vin, vout))
        assert gain == pytest.approx(3.5, rel=5e-2)
        assert 0.6 < vin_peak < 1.0


class TestExtractSettlingTime:
    def test_first_order_step_settles_within_window(self):
        time = np.linspace(0.0, 5e-6, 5001)
        v_initial, v_final = 0.9, 0.4
        tau = 2e-7
        vout = np.where(time < 1e-7, v_initial, v_final + (v_initial - v_final) * np.exp(-(time - 1e-7) / tau))
        settling = extract_settling_time(_make_tran(time, vout), t_step_start=1e-7)
        analytic = -tau * math.log(0.05)
        assert settling == pytest.approx(analytic, rel=0.1)

    def test_zero_step_returns_zero(self):
        time = np.linspace(0.0, 1e-6, 1001)
        vout = np.full_like(time, 0.9)
        settling = extract_settling_time(_make_tran(time, vout), t_step_start=1e-7)
        assert settling == 0.0

    def test_final_sample_outside_band_returns_nan(self):
        time = np.linspace(0.0, 1e-6, 1001)
        vout = np.full_like(time, 0.4)
        vout[:50] = 0.9
        vout[-1] = 0.9  # late spike pushes the asymptote estimate off
        settling = extract_settling_time(_make_tran(time, vout), t_step_start=1e-7)
        assert math.isnan(settling)

    def test_already_settled_returns_zero(self):
        time = np.linspace(0.0, 1e-6, 1001)
        vout = np.full_like(time, 0.5)
        vout[0] = 0.5 + 1e-12
        settling = extract_settling_time(_make_tran(time, vout), t_step_start=0.0)
        assert settling == 0.0


class TestMetrics:
    def test_failed_metrics_has_nan_fields(self):
        metrics = Metrics.failed()
        assert metrics.converged is False
        assert math.isnan(metrics.peak_gain_v_per_v)
        assert math.isnan(metrics.vin_at_peak_gain_v)
        assert math.isnan(metrics.vout_at_peak_gain_v)
        assert math.isnan(metrics.static_current_a)
        assert math.isnan(metrics.settling_time_s)

    def test_metrics_is_immutable(self):
        metrics = _make_metrics(converged=True, gain=10.0, vout_bias=0.9, current=1e-6)
        with pytest.raises((AttributeError, TypeError)):
            metrics.converged = False  # type: ignore[misc]


class TestSelectionRule:
    def test_admits_inside_band(self):
        rule = SelectionRule(vout_min_v=0.6, vout_max_v=1.2)
        assert rule.admits(_make_metrics(converged=True, gain=5.0, vout_bias=0.9, current=1e-6))

    def test_rejects_outside_band(self):
        rule = SelectionRule(vout_min_v=0.6, vout_max_v=1.2)
        assert not rule.admits(_make_metrics(converged=True, gain=5.0, vout_bias=0.3, current=1e-6))
        assert not rule.admits(_make_metrics(converged=True, gain=5.0, vout_bias=1.5, current=1e-6))

    def test_rejects_failed_metrics(self):
        rule = SelectionRule(vout_min_v=0.6, vout_max_v=1.2)
        assert not rule.admits(Metrics.failed())


class TestSelectDesignPoint:
    def _sweep(self, items):
        points = tuple(p for p, _ in items)
        metrics = tuple(m for _, m in items)
        return SweepResult(points=points, metrics=metrics)

    def test_picks_max_gain_in_band(self):
        sweep = self._sweep(
            [
                (DesignPoint(1.0, 2.0), _make_metrics(converged=True, gain=8.0, vout_bias=0.9, current=2e-6)),
                (DesignPoint(2.0, 4.0), _make_metrics(converged=True, gain=12.0, vout_bias=0.9, current=3e-6)),
                (DesignPoint(0.5, 1.0), _make_metrics(converged=True, gain=20.0, vout_bias=1.6, current=1e-6)),
            ]
        )
        chosen_point, chosen_metrics = select_design_point(sweep, SelectionRule())
        assert chosen_point == DesignPoint(2.0, 4.0)
        assert chosen_metrics.peak_gain_v_per_v == 12.0

    def test_breaks_ties_by_lower_static_current(self):
        sweep = self._sweep(
            [
                (DesignPoint(1.0, 2.0), _make_metrics(converged=True, gain=10.0, vout_bias=0.9, current=5e-6)),
                (DesignPoint(2.0, 4.0), _make_metrics(converged=True, gain=10.0, vout_bias=0.9, current=2e-6)),
            ]
        )
        chosen_point, _ = select_design_point(sweep, SelectionRule())
        assert chosen_point == DesignPoint(2.0, 4.0)

    def test_raises_when_no_design_is_feasible(self):
        sweep = self._sweep(
            [
                (DesignPoint(1.0, 2.0), _make_metrics(converged=True, gain=8.0, vout_bias=1.5, current=2e-6)),
                (DesignPoint(2.0, 4.0), Metrics.failed()),
            ]
        )
        with pytest.raises(ValueError):
            select_design_point(sweep, SelectionRule())


class TestSweepResultReshaping:
    def _sweep(self, n_axis, p_axis, gain_fn):
        points = []
        metrics = []
        for wn in n_axis:
            for wp in p_axis:
                points.append(DesignPoint(nfet_w_um=wn, pfet_w_um=wp))
                metrics.append(_make_metrics(converged=True, gain=gain_fn(wn, wp), vout_bias=0.9, current=1e-6))
        return SweepResult(points=tuple(points), metrics=tuple(metrics))

    def test_gain_grid_has_axis_aligned_shape(self):
        sweep = self._sweep([0.5, 1.0, 2.0], [1.0, 2.0], gain_fn=lambda n, p: n * p)
        grid = sweep.gain_grid()
        assert grid.shape == (3, 2)
        n_axis, p_axis = sweep.axes()
        np.testing.assert_array_equal(n_axis, [0.5, 1.0, 2.0])
        np.testing.assert_array_equal(p_axis, [1.0, 2.0])
        assert grid[1, 0] == pytest.approx(1.0)
        assert grid[2, 1] == pytest.approx(4.0)

    def test_failed_points_become_nan_cells(self):
        sweep = SweepResult(
            points=(DesignPoint(1.0, 1.0), DesignPoint(1.0, 2.0)),
            metrics=(
                _make_metrics(converged=True, gain=5.0, vout_bias=0.9, current=1e-6),
                Metrics.failed(),
            ),
        )
        grid = sweep.gain_grid()
        assert grid.shape == (1, 2)
        assert grid[0, 0] == pytest.approx(5.0)
        assert math.isnan(grid[0, 1])
