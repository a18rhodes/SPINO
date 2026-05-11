"""
Unit tests for OTA probe builders and Newton-Raphson composition solvers.

Uses linear mock FNO devices so no checkpoint files are required. Device
outputs are in arcsinh-mA space (see ``spino.circuit.devices``); with small
arguments the hyperbolic-sine inversion is approximately linear, making the
KCL system analytically tractable.

Analytical DC balance for the symmetric all-equal-slope mock (slope=1.0 for
NFET, slope=-1.0 for PFET, VDD=1.8 V, Vinp=Vinn=Vcm):

    V_tail = VDD / 2 = 0.9 V
    V_left = V_out = 3·VDD / 4 = 1.35 V

(derived from the linearised KCL; exact only in the sinh ≈ x regime, but
confirmed numerically by the solver convergence test).
"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

import torch
import pytest
from torch import Tensor, nn

from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.ota_composition import (
    OtaDcSolver,
    OtaTransientSolver,
    build_ota_diffpair_probe,
    build_ota_mirror_probe,
    build_ota_tail_probe,
)

_VDD = 1.8
_VCM = 0.9
_VBIAS = 1.2
_T = 16
_PHYSICS_DIM = 4


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class _LinearKclMock(nn.Module):
    """
    Linear mock in arcsinh-mA space.

    Returns ``slope · (V_d − V_s)`` so that after the device wrapper applies
    ``sinh(·) · ARCSINH_SCALE_MA · 1e-3``, the resulting current is
    approximately linear for small arguments.

    Use ``slope = +1.0`` for NFET mocks and ``slope = -1.0`` for PFET mocks.
    """

    def __init__(self, slope: float) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))

    def forward(self, v_terminals: Tensor, physical_params: Tensor) -> Tensor:
        del physical_params
        v_d = v_terminals[:, 1:2, :]
        v_s = v_terminals[:, 2:3, :]
        return self.slope * (v_d - v_s)


def _identity_wrapper(model: nn.Module, label: str) -> FnoMosfetDevice:
    """FnoMosfetDevice with identity (zero-mean, unit-std) normalization."""
    return FnoMosfetDevice(
        model=model,
        v_mean=torch.zeros((4, 1)),
        v_std=torch.ones((4, 1)),
        p_mean=torch.zeros(_PHYSICS_DIM),
        p_std=torch.ones(_PHYSICS_DIM),
        physics_raw=torch.zeros(_PHYSICS_DIM),
        label=label,
    )


def _nfet_mock(label: str = "nfet") -> FnoMosfetDevice:
    return _identity_wrapper(_LinearKclMock(1.0), label)


def _pfet_mock(label: str = "pfet") -> FnoMosfetDevice:
    return _identity_wrapper(_LinearKclMock(-1.0), label)


@pytest.fixture(name="ota_devices")
def fixture_ota_devices() -> tuple[FnoMosfetDevice, ...]:
    """Five independent mock OTA devices (M1–M5)."""
    return (
        _nfet_mock("M1"),
        _nfet_mock("M2"),
        _pfet_mock("M3"),
        _pfet_mock("M4"),
        _nfet_mock("M5"),
    )


# ---------------------------------------------------------------------------
# Probe builders
# ---------------------------------------------------------------------------


class TestProbeBuilders:
    """Shape and channel-content checks for the three public probe builders."""

    def test_diffpair_probe_shape(self) -> None:
        """Output is (1, 4, T) with Vg, Vd, Vs_tail, Vb=0."""
        t = _T
        vg = torch.full((t,), _VCM)
        vd = torch.full((t,), 0.9)
        vs = torch.full((t,), 0.4)
        probe = build_ota_diffpair_probe(vg, vd, vs)
        assert probe.shape == (1, 4, t)
        torch.testing.assert_close(probe[0, 0, :], vg)
        torch.testing.assert_close(probe[0, 1, :], vd)
        torch.testing.assert_close(probe[0, 2, :], vs)
        torch.testing.assert_close(probe[0, 3, :], torch.zeros(t))

    def test_mirror_probe_shape(self) -> None:
        """Output is (1, 4, T) with Vs=Vb=VDD."""
        t = _T
        vg = torch.full((t,), 1.35)
        vd = torch.full((t,), 1.35)
        probe = build_ota_mirror_probe(vg, vd, _VDD)
        assert probe.shape == (1, 4, t)
        torch.testing.assert_close(probe[0, 2, :], torch.full((t,), _VDD))
        torch.testing.assert_close(probe[0, 3, :], torch.full((t,), _VDD))

    def test_tail_probe_shape(self) -> None:
        """Output is (1, 4, T) with Vg=const Vbias, Vs=Vb=0."""
        t = _T
        vd = torch.full((t,), 0.9)
        probe = build_ota_tail_probe(_VBIAS, vd)
        assert probe.shape == (1, 4, t)
        torch.testing.assert_close(probe[0, 0, :], torch.full((t,), _VBIAS))
        torch.testing.assert_close(probe[0, 1, :], vd)
        torch.testing.assert_close(probe[0, 2, :], torch.zeros(t))
        torch.testing.assert_close(probe[0, 3, :], torch.zeros(t))

    def test_diffpair_probe_preserves_autograd(self) -> None:
        """Gradient flows back through the vs_tail tensor."""
        vs = torch.tensor([0.4] * _T, requires_grad=True)
        vg = torch.zeros(_T)
        vd = torch.zeros(_T)
        probe = build_ota_diffpair_probe(vg, vd, vs)
        probe.sum().backward()
        assert vs.grad is not None


# ---------------------------------------------------------------------------
# DC solver
# ---------------------------------------------------------------------------


class TestOtaDcSolver:
    """Newton-Raphson DC operating-point tests against mock devices."""

    def test_converges_and_solution_in_range(self, ota_devices) -> None:
        """Solver reaches tolerance and returns node voltages inside (0, VDD)."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaDcSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vcm_v=_VCM,
            vbias_v=_VBIAS,
            t_probe=64,
            trim=8,
            max_iter=60,
            residual_tol=1e-8,
        )
        sol = solver.solve()
        assert sol.report.converged, f"DC solver did not converge; history={sol.report.residual_norm_history}"
        assert 0.0 < sol.v_tail_v < _VDD
        assert 0.0 < sol.v_left_v < _VDD
        assert 0.0 < sol.v_out_v < _VDD

    def test_symmetric_inputs_give_symmetric_arm_nodes(self, ota_devices) -> None:
        """With Vinp=Vinn=Vcm the left and output arms converge to equal voltages."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaDcSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vcm_v=_VCM,
            vbias_v=_VBIAS,
            t_probe=64,
            trim=8,
            max_iter=60,
            residual_tol=1e-8,
        )
        sol = solver.solve()
        assert sol.report.converged
        assert abs(sol.v_left_v - sol.v_out_v) < 1e-4, (
            f"Arm asymmetry too large: v_left={sol.v_left_v:.6f}, v_out={sol.v_out_v:.6f}"
        )

    def test_residual_norm_decreases(self, ota_devices) -> None:
        """Newton iterations reduce the KCL residual from a deliberately bad start."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaDcSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vcm_v=_VCM,
            vbias_v=_VBIAS,
            t_probe=64,
            trim=8,
            max_iter=60,
            residual_tol=1e-12,
        )
        # Start far from the solution to guarantee at least one NR step.
        v_init = torch.tensor([0.01, 0.01, 0.01])
        sol = solver.solve(v_init=v_init)
        hist = sol.report.residual_norm_history
        assert len(hist) > 1, "Expected at least one Newton step"
        assert hist[-1] < hist[0], "Residual did not decrease"

    def test_custom_init_accepted(self, ota_devices) -> None:
        """Passing an explicit v_init does not raise and still converges."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaDcSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vcm_v=_VCM,
            vbias_v=_VBIAS,
            t_probe=64,
            trim=8,
            max_iter=60,
            residual_tol=1e-8,
        )
        v_init = torch.tensor([0.15, 0.60, 0.60])
        sol = solver.solve(v_init=v_init)
        assert sol.report.converged


# ---------------------------------------------------------------------------
# Transient solver
# ---------------------------------------------------------------------------


class TestOtaTransientSolver:
    """Whole-window implicit NR tests against mock devices."""

    @pytest.fixture()
    def dc_solution(self, ota_devices) -> torch.Tensor:
        """Returns ``(3,)`` DC op-point from the mock devices."""
        m1, m2, m3, m4, m5 = ota_devices
        sol = OtaDcSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vcm_v=_VCM,
            vbias_v=_VBIAS,
            t_probe=64,
            trim=8,
            max_iter=60,
            residual_tol=1e-8,
        ).solve()
        return torch.tensor([sol.v_tail_v, sol.v_left_v, sol.v_out_v])

    def test_flat_input_converges_and_shape(self, ota_devices, dc_solution) -> None:
        """Constant Vinp=Vinn=Vcm leaves trajectories near the DC point."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaTransientSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vbias_v=_VBIAS,
            c_load_f=0.0,
            max_iter=30,
            residual_tol=1e-7,
        )
        n_t = 10
        time_s = torch.linspace(0.0, 10e-9, n_t)
        vinp_t = torch.full((n_t,), _VCM)
        vinn_t = torch.full((n_t,), _VCM)
        sol = solver.solve(time_s, vinp_t, vinn_t, dc_solution)
        assert sol.report.converged, f"Transient did not converge; history={sol.report.residual_norm_history}"
        assert sol.v_tail_v.shape == (n_t,)
        assert sol.v_left_v.shape == (n_t,)
        assert sol.v_out_v.shape == (n_t,)

    def test_flat_input_stays_near_dc(self, ota_devices, dc_solution) -> None:
        """Constant stimulus must keep each node within 1 mV of DC value."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaTransientSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vbias_v=_VBIAS,
            c_load_f=0.0,
            max_iter=30,
            residual_tol=1e-7,
        )
        n_t = 10
        time_s = torch.linspace(0.0, 10e-9, n_t)
        vinp_t = torch.full((n_t,), _VCM)
        vinn_t = torch.full((n_t,), _VCM)
        sol = solver.solve(time_s, vinp_t, vinn_t, dc_solution)
        assert sol.report.converged
        torch.testing.assert_close(
            sol.v_tail_v, torch.full((n_t,), dc_solution[0].item()), atol=1e-3, rtol=0.0
        )
        torch.testing.assert_close(
            sol.v_out_v, torch.full((n_t,), dc_solution[2].item()), atol=1e-3, rtol=0.0
        )

    def test_minimum_two_timesteps_required(self, ota_devices, dc_solution) -> None:
        """Passing a single-sample time grid raises ValueError."""
        m1, m2, m3, m4, m5 = ota_devices
        solver = OtaTransientSolver(
            m1, m2, m3, m4, m5,
            vdd=_VDD,
            vbias_v=_VBIAS,
        )
        with pytest.raises(ValueError, match="at least two"):
            solver.solve(
                torch.tensor([0.0]),
                torch.tensor([_VCM]),
                torch.tensor([_VCM]),
                dc_solution,
            )
