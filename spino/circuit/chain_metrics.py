"""
Invariant metrics for inverter-chain SPICE vs composed transients.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "crossing_time_s",
    "max_abs_delta_v",
    "pearson_r",
]

# Treat adjacent samples as vertically flat vs machine noise (not physical V resolution).
_CROSSING_SEGMENT_FLAT_DELTA_V_THRESH = 1e-9


def crossing_time_s(time_s: NDArray[np.float64], wave_v: NDArray[np.float64], v_cross: float) -> float | None:
    """
    First upward crossing time of ``wave_v`` through ``v_cross`` (linear interp).

    Adjacent pairs with negligible ``ΔV`` (below `_CROSSING_SEGMENT_FLAT_DELTA_V_THRESH`)
    skip division and return the right-hand sample time — avoids meaningless
    denominators relative to volts-scale quantization.

    :param time_s: Strictly increasing time stamps (s).
    :param wave_v: Voltage samples aligned with ``time_s``.
    :param v_cross: Crossing threshold (volts).
    :return: Time in seconds or ``None`` if no crossing detected.
    """
    if time_s.size < 2 or wave_v.size != time_s.size:
        return None
    for i in range(len(wave_v) - 1):
        a, b = float(wave_v[i]), float(wave_v[i + 1])
        if a < v_cross <= b or a > v_cross >= b:
            t0, t1 = float(time_s[i]), float(time_s[i + 1])
            if abs(b - a) < _CROSSING_SEGMENT_FLAT_DELTA_V_THRESH:
                # Degenerate interpolation: defer to successor sample time.
                return t1
            frac = (v_cross - a) / (b - a)
            return t0 + frac * (t1 - t0)
    return None


def pearson_r(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Pearson correlation coefficient.

    :param a: Reference samples.
    :param b: Candidate samples (same length).
    :return: Correlation or ``nan`` if undefined.
    """
    if a.size < 2 or a.size != b.size:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def max_abs_delta_v(ref: NDArray[np.float64], pred: NDArray[np.float64]) -> float:
    """
    Peak absolute voltage error after alignment on identical grids.

    :param ref: Reference ``V(t)``.
    :param pred: Predicted ``V(t)``.
    :return: ``max |ref - pred|`` in volts.
    """
    return float(np.max(np.abs(ref - pred)))
