"""
Unit tests for 5T OTA netlist construction.

SPICE execution is not required; the Sky130 library path is patched to a dummy
string because :func:`~spino.circuit.topologies.build_ota_5t` validates
existence eagerly.
"""

from __future__ import annotations

from unittest.mock import patch

from spino.circuit.topologies import build_ota_5t

_DUMMY_LIB = "/tmp/spino_dummy_sky130_lib_for_tests.spice"


@patch("spino.circuit.topologies._resolve_lib_path", return_value=_DUMMY_LIB)
class TestBuildOta5t:
    """Deck string assertions for the 5T OTA netlist factory."""

    def test_diff_pair_nets_and_model(self, _mock_lib) -> None:
        """M1 and M2 are NFET with sources tied to n_tail."""
        ck = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
        )
        deck = ck.build_deck(".op").replace("\n", " ")
        assert "XM1 n_left vinp n_tail 0 sky130_fd_pr__nfet_01v8" in deck
        assert "XM2 n_out vinn n_tail 0 sky130_fd_pr__nfet_01v8" in deck

    def test_current_mirror_nets_and_model(self, _mock_lib) -> None:
        """M3 is diode-connected (gate=drain=n_left) and M4 copies n_left gate."""
        ck = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=3.2, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
        )
        deck = ck.build_deck(".op").replace("\n", " ")
        assert "XM3 n_left n_left vdd vdd sky130_fd_pr__pfet_01v8" in deck
        assert "XM4 n_out n_left vdd vdd sky130_fd_pr__pfet_01v8" in deck

    def test_tail_current_source_nets(self, _mock_lib) -> None:
        """M5 drain is n_tail; gate is the vbias node; source and bulk are GND."""
        ck = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=4.0, tail_l_um=0.5,
        )
        deck = ck.build_deck(".op").replace("\n", " ")
        assert "XM5 n_tail vbias 0 0 sky130_fd_pr__nfet_01v8" in deck

    def test_bias_and_input_sources(self, _mock_lib) -> None:
        """VDD, Vbias, Vinp and Vinn sources all appear in the deck."""
        ck = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
            vbias_v=1.2,
            vcm_v=0.9,
        )
        deck = ck.build_deck(".op").replace("\n", " ")
        assert "VDD vdd 0 DC 1.8" in deck
        assert "Vbias vbias 0 DC 1.2" in deck
        assert "Vinp vinp 0 DC 0.9" in deck
        assert "Vinn vinn 0 DC 0.9" in deck

    def test_load_capacitor_optional(self, _mock_lib) -> None:
        """CL attaches to n_out when c_load_f > 0."""
        deck_no_cl = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
        ).build_deck(".op")
        assert "CL" not in deck_no_cl

        deck_with_cl = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
            c_load_f=100e-15,
        ).build_deck(".op").replace("\n", " ")
        assert "CL n_out 0 1e-13" in deck_with_cl

    def test_tran_pwl_strings_appear_in_tran_deck(self, _mock_lib) -> None:
        """Transient stimulus strings propagate to ``.tran`` deck."""
        ck = build_ota_5t(
            diff_w_um=2.0, diff_l_um=0.4,
            mirror_w_um=2.0, mirror_l_um=0.4,
            tail_w_um=2.0, tail_l_um=0.4,
            vinp_tran="PWL(0 0.9 100n 0.9 105n 1.15)",
            vinn_tran="PWL(0 0.9 100n 0.9 105n 0.65)",
        )
        deck = ck.build_deck(".tran 10n 5u").replace("\n", " ")
        assert "PWL(0 0.9 100n 0.9 105n 1.15)" in deck
        assert "PWL(0 0.9 100n 0.9 105n 0.65)" in deck
