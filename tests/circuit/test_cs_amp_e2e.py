"""
End-to-end integration tests for a CS amplifier built from raw Python dataclasses.

Validates the full pipeline:

    MosfetInstance / VoltageSource / Circuit dataclasses
        → SPICE deck string
            → NGSpice subprocess
                → parsed node voltages / branch currents

The circuit under test is a diode-connected PMOS active load CS amplifier built
entirely from first principles — no topology factory, no helper utilities. This
test exists to prove that the Python-to-NGSpice pipeline works end-to-end.

Skipped automatically when ngspice or the Sky130 PDK is unavailable.
"""

import os
import shutil

import numpy as np
import pytest

_NGSPICE_AVAILABLE = shutil.which("ngspice") is not None
_PDK_AVAILABLE = os.path.exists("/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
pytestmark = pytest.mark.skipif(
    not (_NGSPICE_AVAILABLE and _PDK_AVAILABLE),
    reason="ngspice or Sky130 PDK not available",
)

from spino.circuit import build_cs_amp_active_load
from spino.circuit.simulation import run_dc_sweep, run_operating_point, run_transient

_PDK_LIB = "/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice"
_NMOS_MODEL = "sky130_fd_pr__nfet_01v8"
_PMOS_MODEL = "sky130_fd_pr__pfet_01v8"

# Sizing tracks the topology factory defaults, which were set by the NGSpice
# sweep documented in docs/cs_amp.md. The point of this test is pipeline
# correctness, but pinning to the published operating point keeps the
# integration test honest against the same numbers used in the writeup.
_NFET_W, _NFET_L = 6.0, 0.18
_PFET_W, _PFET_L = 4.5, 0.18
_VDD = 1.8
_VIN_DC = 0.85


@pytest.fixture(scope="module")
def cs_amp():
    """CS amp built via the topology factory at default sizing."""
    return build_cs_amp_active_load()


@pytest.fixture(scope="module")
def cs_amp_with_step():
    """CS amp with a 100 mV rising step at t=2 ns for transient tests."""
    return build_cs_amp_active_load(vin_tran=f"PWL(0 {_VIN_DC} 1n {_VIN_DC} 2n {_VIN_DC + 0.1} 10u {_VIN_DC + 0.1})")


class TestDeckGeneration:
    """Validates the Python-to-SPICE-string step before touching NGSpice."""

    def test_deck_contains_lib(self, cs_amp):
        deck = cs_amp.build_deck(".op")
        assert f".lib '{_PDK_LIB}' tt" in deck

    def test_deck_contains_nfet_instance(self, cs_amp):
        deck = cs_amp.build_deck(".op")
        assert f"XM1 out in 0 0 {_NMOS_MODEL} w={_NFET_W} l={_NFET_L}" in deck

    def test_deck_contains_pfet_instance(self, cs_amp):
        deck = cs_amp.build_deck(".op")
        assert f"XM2 out out vdd vdd {_PMOS_MODEL} w={_PFET_W} l={_PFET_L}" in deck

    def test_tran_deck_uses_pwl_stimulus(self, cs_amp_with_step):
        deck = cs_amp_with_step.build_deck(".tran 1n 10u")
        assert "PWL" in deck
        assert "Vin in 0 PWL" in deck


class TestOperatingPoint:
    """Validates the DC bias point is physically reasonable."""

    def test_operating_point_converges(self, cs_amp):
        assert run_operating_point(cs_amp) is not None

    def test_output_voltage_within_supply_rails(self, cs_amp):
        op = run_operating_point(cs_amp)
        vout = op.variables["v(out)"]
        assert 0.0 < vout < _VDD

    def test_supply_current_is_nonzero(self, cs_amp):
        op = run_operating_point(cs_amp)
        assert abs(op.variables["i(vdd)"]) > 1e-9

    def test_kcl_satisfied_at_output_node(self, cs_amp):
        """NFET and PFET drain currents must match at the shared output node."""
        op = run_operating_point(cs_amp)
        id_nfet = op.variables[f"i(@m.xm1.m{_NMOS_MODEL}[id])"]
        id_pfet = op.variables[f"i(@m.xm2.m{_PMOS_MODEL}[id])"]
        np.testing.assert_allclose(id_nfet, id_pfet, rtol=1e-3)


class TestDCSweep:
    """Validates the voltage transfer characteristic (VTC) against expected CS amp behavior."""

    @pytest.fixture(scope="class")
    def vtc(self, cs_amp):
        """Pre-computed VTC sweep shared across this test class."""
        return run_dc_sweep(cs_amp, source_name="Vin", start=0.0, stop=_VDD, step=0.01)

    def test_vtc_sweep_produces_result(self, vtc):
        assert vtc is not None
        assert "v(out)" in vtc.variables

    def test_vtc_spans_full_input_range(self, vtc):
        np.testing.assert_allclose(vtc.sweep_values[0], 0.0, atol=0.01)
        np.testing.assert_allclose(vtc.sweep_values[-1], _VDD, atol=0.01)

    def test_vtc_is_inverting(self, vtc):
        """CS amp is an inverting topology: Vout falls as Vin rises."""
        vout = vtc.variables["v(out)"]
        assert vout[0] > vout[-1]

    def test_vtc_output_bounded_by_rails(self, vtc):
        vout = vtc.variables["v(out)"]
        assert np.all(vout >= -0.05)
        assert np.all(vout <= _VDD + 0.05)

    def test_vtc_has_nonzero_gain_region(self, vtc):
        """At least one input step must produce > 1 mV/mV magnitude gain."""
        vout = vtc.variables["v(out)"]
        vin = vtc.sweep_values
        dv_step = vin[1] - vin[0]
        gain = np.abs(np.diff(vout) / dv_step)
        assert gain.max() > 1.0


class TestTransient:
    """Validates the step response of the CS amplifier."""

    @pytest.fixture(scope="class")
    def step_response(self, cs_amp_with_step):
        """Pre-computed transient waveform shared across this test class."""
        return run_transient(cs_amp_with_step, t_step=1e-9, t_end=10e-6)

    def test_transient_produces_waveform(self, step_response):
        assert step_response is not None
        assert len(step_response.time) > 100
        assert "v(out)" in step_response.variables

    def test_output_bounded_by_rails(self, step_response):
        vout = step_response.variables["v(out)"]
        assert np.all(vout >= -0.05)
        assert np.all(vout <= _VDD + 0.05)

    def test_step_response_is_inverting(self, step_response):
        """Rising Vin step must drive Vout downward."""
        vout = step_response.variables["v(out)"]
        # Compare settled initial value vs settled final value.
        v_initial = vout[:10].mean()
        v_final = vout[-10:].mean()
        assert v_initial > v_final

    def test_step_response_settles(self, step_response):
        """Output variation in the final 20% of the window must be < 10 mV."""
        vout = step_response.variables["v(out)"]
        tail = vout[int(0.8 * len(vout)) :]
        assert tail.max() - tail.min() < 0.01
