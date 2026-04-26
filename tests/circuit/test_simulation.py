"""Integration tests for circuit simulation against real NGSpice."""

import os
import shutil

import numpy as np
import pytest

_NGSPICE_AVAILABLE = shutil.which("ngspice") is not None
_PDK_AVAILABLE = os.path.exists("/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
pytestmark = pytest.mark.skipif(
    not (_NGSPICE_AVAILABLE and _PDK_AVAILABLE), reason="ngspice or Sky130 PDK not available"
)

from spino.circuit import build_cs_amp_active_load, run_dc_sweep, run_operating_point, run_transient


@pytest.fixture()
def cs_amp():
    """Standard CS amp with diode-connected PMOS load at default sizing."""
    return build_cs_amp_active_load()


@pytest.fixture()
def cs_amp_with_step():
    """CS amp with 100mV step input for transient testing."""
    return build_cs_amp_active_load(vin_tran="PWL(0 0.9 1n 0.9 2n 1.0 10u 1.0)")


class TestOperatingPoint:
    """Validates DC operating point convergence and physical correctness."""

    def test_converges(self, cs_amp):
        op = run_operating_point(cs_amp)
        assert op is not None

    def test_output_voltage_within_rails(self, cs_amp):
        op = run_operating_point(cs_amp)
        assert 0.0 < op.variables["v(out)"] < 1.8

    def test_supply_current_flows(self, cs_amp):
        op = run_operating_point(cs_amp)
        assert "i(vdd)" in op.variables
        assert abs(op.variables["i(vdd)"]) > 1e-9

    def test_kcl_at_output_node(self, cs_amp):
        op = run_operating_point(cs_amp)
        id_nfet = op.variables["i(@m.xm1.msky130_fd_pr__nfet_01v8[id])"]
        id_pfet = op.variables["i(@m.xm2.msky130_fd_pr__pfet_01v8[id])"]
        np.testing.assert_allclose(id_nfet, id_pfet, rtol=1e-3)


class TestTransient:
    """Validates transient simulation output integrity."""

    def test_produces_waveform(self, cs_amp_with_step):
        result = run_transient(cs_amp_with_step, t_step=1e-9, t_end=10e-6)
        assert result is not None
        assert len(result.time) > 100
        assert "v(out)" in result.variables

    def test_output_bounded_by_rails(self, cs_amp_with_step):
        result = run_transient(cs_amp_with_step, t_step=1e-9, t_end=10e-6)
        vout = result.variables["v(out)"]
        assert np.all(vout >= -0.1)
        assert np.all(vout <= 1.9)

    def test_step_response_is_inverting(self, cs_amp_with_step):
        result = run_transient(cs_amp_with_step, t_step=1e-9, t_end=10e-6)
        vout = result.variables["v(out)"]
        assert vout[0] > vout[-1]


class TestDCSweep:
    """Validates DC sweep (voltage transfer characteristic) extraction."""

    def test_produces_vtc(self, cs_amp):
        result = run_dc_sweep(cs_amp, source_name="Vin", start=0.0, stop=1.8, step=0.01)
        assert result is not None
        assert len(result.sweep_values) == 181
        assert "v(out)" in result.variables

    def test_vtc_is_monotonically_decreasing(self, cs_amp):
        result = run_dc_sweep(cs_amp, source_name="Vin", start=0.0, stop=1.8, step=0.01)
        vout = result.variables["v(out)"]
        assert vout[0] > vout[-1]

    def test_vtc_output_within_rails(self, cs_amp):
        result = run_dc_sweep(cs_amp, source_name="Vin", start=0.0, stop=1.8, step=0.01)
        vout = result.variables["v(out)"]
        assert np.all(vout >= -0.1)
        assert np.all(vout <= 1.9)

    def test_sweep_values_span_input_range(self, cs_amp):
        result = run_dc_sweep(cs_amp, source_name="Vin", start=0.0, stop=1.8, step=0.01)
        np.testing.assert_allclose(result.sweep_values[0], 0.0, atol=0.01)
        np.testing.assert_allclose(result.sweep_values[-1], 1.8, atol=0.01)
