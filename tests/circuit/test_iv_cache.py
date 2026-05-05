"""
Unit tests for :mod:`spino.circuit.iv_cache`.

Validates public threshold aliases used by relative/absolute IV routing.
"""

from __future__ import annotations

from spino.circuit.iv_cache import ABS_FLOOR_A, ABS_MODE_SWITCH_A, RelAbsThresholds


def test_iv_cache_threshold_aliases() -> None:
    """Public constants match defaults on RelAbsThresholds."""
    assert ABS_MODE_SWITCH_A == ABS_FLOOR_A == 10e-9
    r = RelAbsThresholds()
    assert r.rel_eps == ABS_MODE_SWITCH_A and r.abs_floor == ABS_FLOOR_A
