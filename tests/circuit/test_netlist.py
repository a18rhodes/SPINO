"""Unit tests for circuit netlist generation."""

import pytest

from spino.circuit.netlist import Capacitor, Circuit, MosfetInstance, VoltageSource

_DUMMY_LIB = "/fake/pdk/sky130.lib.spice"


class TestMosfetInstance:
    """Validates MOSFET instance representation and SPICE output."""

    def test_missing_port_raises(self):
        with pytest.raises(ValueError, match="missing port connections"):
            MosfetInstance(
                name="X1",
                model_name="nfet",
                width_um=1.0,
                length_um=0.18,
                nets={"drain": "d", "gate": "g"},
            )

    def test_to_spice_format(self):
        dev = MosfetInstance(
            name="XM1",
            model_name="sky130_fd_pr__nfet_01v8",
            width_um=1.0,
            length_um=0.18,
            nets={"drain": "out", "gate": "in", "source": "0", "bulk": "0"},
        )
        assert dev.to_spice() == "XM1 out in 0 0 sky130_fd_pr__nfet_01v8 w=1.0 l=0.18"

    def test_port_order_is_drain_gate_source_bulk(self):
        dev = MosfetInstance(
            name="XM2",
            model_name="sky130_fd_pr__pfet_01v8",
            width_um=2.0,
            length_um=0.5,
            nets={"bulk": "vdd", "source": "vdd", "gate": "out", "drain": "out"},
        )
        assert dev.to_spice() == "XM2 out out vdd vdd sky130_fd_pr__pfet_01v8 w=2.0 l=0.5"

    def test_ports_constant_is_canonical(self):
        assert MosfetInstance.PORTS == ("drain", "gate", "source", "bulk")

    def test_satisfies_spice_device_protocol(self):
        dev = MosfetInstance(
            name="X1",
            model_name="m",
            width_um=1.0,
            length_um=0.18,
            nets={"drain": "d", "gate": "g", "source": "s", "bulk": "b"},
        )
        assert isinstance(dev.name, str) and callable(dev.to_spice)

    def test_is_immutable(self):
        dev = MosfetInstance(
            name="X1",
            model_name="m",
            width_um=1.0,
            length_um=0.18,
            nets={"drain": "d", "gate": "g", "source": "s", "bulk": "b"},
        )
        with pytest.raises((AttributeError, TypeError)):
            dev.name = "X2"


class TestVoltageSource:
    """Validates voltage source SPICE formatting for DC and transient modes."""

    def test_dc_without_ac(self):
        src = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=1.8)
        assert src.to_spice_dc() == "VDD vdd 0 DC 1.8"

    def test_dc_with_ac(self):
        src = VoltageSource(name="Vin", positive_node="in", negative_node="0", dc_value=0.9, ac_value=0.01)
        assert src.to_spice_dc() == "Vin in 0 DC 0.9 AC 0.01"

    def test_tran_with_pwl(self):
        src = VoltageSource(
            name="Vin", positive_node="in", negative_node="0", dc_value=0.9, tran_value="PWL(0 0.9 1n 1.0)"
        )
        assert src.to_spice_tran() == "Vin in 0 PWL(0 0.9 1n 1.0)"

    def test_tran_falls_back_to_dc_when_no_stimulus(self):
        src = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=1.8)
        assert src.to_spice_tran() == "VDD vdd 0 DC 1.8"


class TestCapacitor:
    """Validates two-terminal linear capacitor SPICE formatting."""

    def test_to_spice_emits_three_tokens_plus_value(self):
        cap = Capacitor(name="CL", positive_node="out", negative_node="0", capacitance_f=1e-12)
        assert cap.to_spice() == "CL out 0 1e-12"

    def test_immutable(self):
        cap = Capacitor(name="CL", positive_node="out", negative_node="0", capacitance_f=1e-12)
        with pytest.raises((AttributeError, TypeError)):
            cap.capacitance_f = 2e-12


def _make_two_device_circuit() -> Circuit:
    """Builds a minimal two-device circuit for deck generation tests."""
    nfet = MosfetInstance(
        name="XM1",
        model_name="nfet",
        width_um=1.0,
        length_um=0.18,
        nets={"drain": "out", "gate": "in", "source": "0", "bulk": "0"},
    )
    pfet = MosfetInstance(
        name="XM2",
        model_name="pfet",
        width_um=2.0,
        length_um=0.18,
        nets={"drain": "out", "gate": "out", "source": "vdd", "bulk": "vdd"},
    )
    v_supply = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=1.8)
    v_input = VoltageSource(
        name="Vin", positive_node="in", negative_node="0", dc_value=0.9, tran_value="PWL(0 0.9 1n 1.0)"
    )
    return Circuit(
        name="Test Circuit",
        devices=(nfet, pfet),
        sources=(v_supply, v_input),
        lib_path=_DUMMY_LIB,
    )


class TestCircuit:
    """Validates SPICE deck generation for different analysis types."""

    def test_deck_starts_with_title(self):
        deck = _make_two_device_circuit().build_deck(".op")
        assert deck.startswith("* Test Circuit\n")

    def test_deck_ends_with_dotend(self):
        deck = _make_two_device_circuit().build_deck(".op")
        assert deck.strip().endswith(".end")

    def test_deck_contains_lib_directive(self):
        deck = _make_two_device_circuit().build_deck(".op")
        assert f".lib '{_DUMMY_LIB}' tt" in deck

    def test_op_deck_uses_dc_sources(self):
        deck = _make_two_device_circuit().build_deck(".op")
        assert "Vin in 0 DC 0.9" in deck
        assert "PWL" not in deck

    def test_tran_deck_uses_tran_sources(self):
        deck = _make_two_device_circuit().build_deck(".tran 1n 10u")
        assert "Vin in 0 PWL(0 0.9 1n 1.0)" in deck
        assert "VDD vdd 0 DC 1.8" in deck

    def test_deck_contains_both_devices(self):
        deck = _make_two_device_circuit().build_deck(".op")
        assert "XM1 out in 0 0 nfet w=1.0 l=0.18" in deck
        assert "XM2 out out vdd vdd pfet w=2.0 l=0.18" in deck

    def test_deck_includes_options(self):
        deck = _make_two_device_circuit().build_deck(".op", options=("savecurrents", "strict_errorhandling=0"))
        assert ".option savecurrents" in deck
        assert ".option strict_errorhandling=0" in deck

    def test_dc_sweep_deck_uses_dc_sources(self):
        deck = _make_two_device_circuit().build_deck(".dc Vin 0 1.8 0.01")
        assert "Vin in 0 DC 0.9" in deck
        assert "PWL" not in deck

    def test_custom_corner(self):
        dev = MosfetInstance(
            name="X1",
            model_name="n",
            width_um=1.0,
            length_um=0.18,
            nets={"drain": "d", "gate": "g", "source": "s", "bulk": "b"},
        )
        circuit = Circuit(name="Corner Test", devices=(dev,), sources=(), lib_path=_DUMMY_LIB, lib_corner="ss")
        deck = circuit.build_deck(".op")
        assert ".lib '/fake/pdk/sky130.lib.spice' ss" in deck
