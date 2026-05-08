"""
Unit tests for inverter-chain netlist construction.

SPICE execution is not required; the Sky130 library path is patched to a dummy
string because :func:`~spino.circuit.topologies.build_inverter_chain` validates
existence eagerly.
"""

from __future__ import annotations

from unittest.mock import patch

from spino.circuit.topologies import build_cmos_inverter, build_inverter_chain

_DUMMY_LIB = "/tmp/spino_dummy_sky130_lib_for_tests.spice"


@patch("spino.circuit.topologies._resolve_lib_path", return_value=_DUMMY_LIB)
class TestBuildInverterChain:
    """Deck string assertions for chained CMOS inverter stages."""

    def test_three_stages_naming_and_nets(self, _mock_lib) -> None:
        """XNk/XPk lines and net fan-out match three cascaded inverters."""
        ck = build_inverter_chain(n_stages=3, vin_dc=0.0)
        deck = ck.build_deck(".tran 5n 1u")
        assert "nin" in deck
        one = deck.replace("\n", " ")
        assert "XN1 n1 nin 0 0 sky130_fd_pr__nfet_01v8" in one
        assert "XP1 n1 nin vdd vdd sky130_fd_pr__pfet_01v8" in one
        assert "XN2 n2 n1 0 0 sky130_fd_pr__nfet_01v8" in one
        assert "XP2 n2 n1 vdd vdd sky130_fd_pr__pfet_01v8" in one
        assert "XN3 n3 n2 0 0 sky130_fd_pr__nfet_01v8" in one
        assert "XP3 n3 n2 vdd vdd sky130_fd_pr__pfet_01v8" in one
        assert "Vin nin 0 DC 0" in one

    def test_final_load_capacitor_optional(self, _mock_lib) -> None:
        """Optional CL attaches to the final output node only."""
        deck = build_inverter_chain(n_stages=2, c_load_f=5e-15).build_deck(".op")
        assert "CL n2 0 5e-15" in deck.replace("\n", " ")

    def test_cmos_inverter_delegates_to_chain(self, _mock_lib) -> None:
        """Single-stage factory matches ``build_inverter_chain(n_stages=1)``."""
        inv = build_cmos_inverter()
        chain1 = build_inverter_chain(n_stages=1)
        assert inv.build_deck(".op") == chain1.build_deck(".op")
