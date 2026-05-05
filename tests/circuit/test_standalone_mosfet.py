"""
Unit tests for :mod:`spino.circuit.standalone_mosfet`.

Covers drain-current key discovery from NGSpice operating-point output and
library path resolution for minimal isolated-device decks.
"""

from __future__ import annotations

import pytest

from spino.circuit.simulation import OperatingPoint
from spino.circuit.standalone_mosfet import _resolve_lib_path, drain_current_key_from_op


def test_drain_current_key_from_op_regex() -> None:
    """Drain key for instance XIV is discovered via the shared regex."""
    op = OperatingPoint(
        variables={
            "v(ng)": 0.5,
            "i(@m.xiv.msky130_fd_pr__nfet_01v8[id])": -1.2e-5,
        },
        iter_count=1,
    )
    k = drain_current_key_from_op(op)
    assert "xiv" in k.lower() and "[id]" in k.lower()


def test_resolve_lib_path_missing_pdk() -> None:
    """Missing sky130.lib.spice raises before any deck is built."""
    with pytest.raises(FileNotFoundError, match="PDK library not found"):
        _resolve_lib_path("/nonexistent_pdk_root_9f3a2b1c")
