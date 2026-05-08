"""Unit tests for :mod:`spino.circuit.chain_metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from spino.circuit.chain_metrics import crossing_time_s, max_abs_delta_v, pearson_r


def test_max_abs_delta_v() -> None:
    """Peak absolute pair voltage difference."""
    a = np.array([0.0, 2.0, 1.0])
    b = np.array([0.1, 2.2, 1.0])
    assert max_abs_delta_v(a, b) == pytest.approx(0.2)


def test_pearson_r_perfect() -> None:
    """Collinear traces yield Pearson 1."""
    x = np.linspace(0.0, 1.0, 20)
    assert pearson_r(x, 2.0 * x + 0.1) == pytest.approx(1.0)


def test_crossing_time_s_rising() -> None:
    """Linear interpolation on a monotone rising segment."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([0.0, 0.3, 0.7, 1.0])
    tc = crossing_time_s(t, v, 0.5)
    assert tc is not None
    assert tc == pytest.approx(1.5)
