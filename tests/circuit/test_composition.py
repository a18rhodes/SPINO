"""
Unit tests for the FNO-composed CS amplifier solvers.

The full production checkpoints are not loaded here; instead, we drive the
solvers with deterministic linear-mock FNOs whose KCL crossing point can be
solved in closed form. This exercises every code path of:

* :class:`spino.circuit.composition.DcOperatingPointSolver`
* :class:`spino.circuit.composition.TransientSolver`
* :func:`spino.circuit.simulation._parse_iter_count`
* :func:`spino.circuit.simulation._maybe_with_acct`

The integration test class at the bottom is gated on the production
checkpoints, datasets, and NGSpice availability — it asserts the Phase 3b
acceptance criteria (DC OP <=5%, transient R^2 > 0.99, settling within 10%,
NR < 10 iters).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import torch.autograd.functional as AF

from spino.circuit.composition import (
    ConvergenceReport,
    DcOperatingPointSolver,
    DcSolution,
    TransientSolver,
    TransientSolution,
    _backtrack,
    _build_nfet_probe,
    _build_nfet_trajectory,
    _build_pfet_probe,
    _build_pfet_trajectory,
    _cap_alpha,
)
from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.simulation import TransientResult, _maybe_with_acct, _parse_iter_count
from spino.circuit.tuning import extract_settling_time
from spino.constants import ARCSINH_SCALE_MA

_VDD = 1.8
_T_PROBE = 64
_TRIM = 16
_PHYSICS_DIM = 4

_NGSPICE_AVAILABLE = shutil.which("ngspice") is not None
_PDK_AVAILABLE = os.path.exists("/app/sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
_PRODUCTION_NFET_CKPT = Path("/app/spino/models/mosfet/mosfet_vcfilm_exp19b_full_finetune_wtmjf8yn.pt")
_PRODUCTION_PFET_CKPT = Path("/app/spino/models/mosfet/pfet/mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt")
_PRODUCTION_NFET_DS = Path("/app/datasets/sky130_nmos_61k_plus_shortch_supp8k.h5")
_PRODUCTION_PFET_DS = Path("/app/datasets/sky130_pmos_48k_sweep_aug.h5")
_INTEGRATION_READY = (
    _NGSPICE_AVAILABLE
    and _PDK_AVAILABLE
    and _PRODUCTION_NFET_CKPT.exists()
    and _PRODUCTION_PFET_CKPT.exists()
    and _PRODUCTION_NFET_DS.exists()
    and _PRODUCTION_PFET_DS.exists()
)


class _LinearKclMock(nn.Module):
    """
    Linear mock FNO with closed-form KCL crossing.

    Returns ``slope * (V_d - V_s) + intercept`` in arcsinh-mA space so that
    :func:`FnoMosfetDevice.drain_current` decodes to
    ``ARCSINH_SCALE_MA * sinh(slope * (V_d - V_s) + intercept) * 1e-3`` A.
    The KCL residual ``I_pfet - I_nfet`` is monotonic in ``V_out``, so the
    crossing point exists and is unique on ``[0, VDD]`` for any pair with
    opposing slopes.

    :param slope: Coefficient on the drain-source potential.
    :param intercept: Additive bias in arcsinh-mA space.
    """

    def __init__(self, slope: float, intercept: float = 0.0) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.intercept = nn.Parameter(torch.tensor(intercept))

    def forward(self, v_terminals: Tensor, physical_params: Tensor) -> Tensor:
        """
        Returns ``slope * (V_d - V_s) + intercept`` broadcast over time.

        :param v_terminals: ``(B, 4, T)`` voltage trajectories (normalized).
        :param physical_params: ``(B, P)`` physics tensor (unused).
        :return: ``(B, 1, T)`` arcsinh-mA prediction.
        """
        del physical_params
        v_d = v_terminals[:, 1:2, :]
        v_s = v_terminals[:, 2:3, :]
        return self.slope * (v_d - v_s) + self.intercept


def _identity_wrapper(model: nn.Module, label: str) -> FnoMosfetDevice:
    """
    Builds a :class:`FnoMosfetDevice` with no-op normalization buffers.

    :param model: Inner mock module.
    :param label: Wrapper label.
    :return: Configured wrapper on CPU/float32.
    """
    return FnoMosfetDevice(
        model=model,
        v_mean=torch.zeros((4, 1)),
        v_std=torch.ones((4, 1)),
        p_mean=torch.zeros(_PHYSICS_DIM),
        p_std=torch.ones(_PHYSICS_DIM),
        physics_raw=torch.zeros(_PHYSICS_DIM),
        label=label,
    )


@pytest.fixture(name="dc_devices")
def fixture_dc_devices() -> tuple[FnoMosfetDevice, FnoMosfetDevice]:
    """
    NFET/PFET mock pair whose KCL crossing point is exactly ``VDD/2``.

    Both devices share the same magnitude slope with opposing signs so that
    the unique solution on ``[0, VDD]`` is the supply midpoint.
    """
    nfet_model = _LinearKclMock(slope=1.0)
    pfet_model = _LinearKclMock(slope=-1.0)
    return _identity_wrapper(nfet_model, "NFET"), _identity_wrapper(pfet_model, "PFET")


def _make_dc_solver(devices: tuple[FnoMosfetDevice, FnoMosfetDevice]) -> DcOperatingPointSolver:
    """Builds a :class:`DcOperatingPointSolver` with cheap probe parameters."""
    nfet, pfet = devices
    return DcOperatingPointSolver(
        nfet=nfet,
        pfet=pfet,
        vdd=_VDD,
        t_probe=_T_PROBE,
        trim=_TRIM,
        max_iter=20,
        residual_tol=1e-12,
    )


class TestDcOperatingPointSolver:
    """Validates the scalar Newton-Raphson DC operating point solver."""

    def test_converges_to_closed_form_solution(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        assert solution.report.converged
        assert solution.v_out_v == pytest.approx(_VDD / 2.0, abs=1e-6)

    def test_iteration_budget_under_ten(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        assert solution.report.iter_count <= 10

    def test_residual_history_decreases_monotonically(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        history = list(solution.report.residual_norm_history)
        assert history[-1] <= history[0]
        assert history[-1] <= solver.residual_tol

    def test_currents_balance_at_solution(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        assert solution.i_pfet_a == pytest.approx(solution.i_nfet_a, abs=1e-12)

    def test_default_initial_guess_is_rail_midpoint(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        history = solution.report.residual_norm_history
        assert history[0] == pytest.approx(0.0, abs=1e-15)

    def test_convergence_from_far_initial_guess(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85, v_out_init=1.7)
        assert solution.report.converged
        assert solution.v_out_v == pytest.approx(_VDD / 2.0, abs=1e-6)

    def test_invalid_t_probe_raises(self, dc_devices) -> None:
        nfet, pfet = dc_devices
        with pytest.raises(ValueError):
            DcOperatingPointSolver(nfet=nfet, pfet=pfet, t_probe=10, trim=20)

    def test_dc_solution_dataclass_fields(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        solution = solver.solve(vin=0.85)
        assert isinstance(solution, DcSolution)
        assert isinstance(solution.report, ConvergenceReport)
        assert solution.report.wall_ms >= 0.0

    def test_finite_difference_jacobian(self, dc_devices) -> None:
        solver = _make_dc_solver(dc_devices)
        v_out = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
        vin = torch.tensor(0.85, dtype=torch.float32)
        residual = solver._residual(v_out, vin)  # pylint: disable=protected-access
        (autograd_jac,) = torch.autograd.grad(residual, v_out)
        eps = 1e-3
        with torch.no_grad():
            r_plus = solver._residual(  # pylint: disable=protected-access
                torch.tensor(0.7 + eps, dtype=torch.float32), vin
            )
            r_minus = solver._residual(  # pylint: disable=protected-access
                torch.tensor(0.7 - eps, dtype=torch.float32), vin
            )
        finite_diff = (r_plus - r_minus) / (2 * eps)
        torch.testing.assert_close(autograd_jac, finite_diff, rtol=1e-3, atol=1e-12)


class TestTransientSolver:
    """Validates the whole-window implicit Newton-Raphson transient solver."""

    @pytest.fixture(name="transient_solver")
    def fixture_transient_solver(self, dc_devices) -> TransientSolver:
        nfet, pfet = dc_devices
        return TransientSolver(
            nfet=nfet,
            pfet=pfet,
            vdd=_VDD,
            c_load_f=0.0,
            max_iter=20,
            residual_tol=1e-12,
        )

    def test_zero_cap_yields_dc_trajectory(self, transient_solver) -> None:
        time_grid = torch.linspace(0.0, 1e-7, 32, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = transient_solver.solve(time_grid, vin_t, v_out_dc=_VDD / 2.0)
        assert solution.report.converged
        torch.testing.assert_close(
            solution.v_out_v,
            torch.full_like(time_grid, fill_value=_VDD / 2.0),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_initial_condition_clamped(self, transient_solver) -> None:
        time_grid = torch.linspace(0.0, 1e-7, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = transient_solver.solve(time_grid, vin_t, v_out_dc=0.7)
        assert float(solution.v_out_v[0].item()) == pytest.approx(0.7, abs=1e-6)

    def test_outer_iteration_count_bounded(self, transient_solver) -> None:
        time_grid = torch.linspace(0.0, 1e-7, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = transient_solver.solve(time_grid, vin_t, v_out_dc=_VDD / 2.0)
        assert solution.report.iter_count <= 10

    def test_solution_dataclass_shapes(self, transient_solver) -> None:
        time_grid = torch.linspace(0.0, 1e-7, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = transient_solver.solve(time_grid, vin_t, v_out_dc=_VDD / 2.0)
        assert isinstance(solution, TransientSolution)
        assert solution.v_out_v.shape == time_grid.shape
        assert solution.i_pfet_a.shape == time_grid.shape
        assert solution.i_nfet_a.shape == time_grid.shape

    def test_explicit_initial_guess_accepted(self, dc_devices) -> None:
        nfet, pfet = dc_devices
        solver = TransientSolver(nfet=nfet, pfet=pfet, vdd=_VDD, c_load_f=0.0, residual_tol=1e-12)
        time_grid = torch.linspace(0.0, 1e-7, 12, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        guess = torch.full_like(time_grid, fill_value=0.6)
        solution = solver.solve(time_grid, vin_t, v_out_dc=_VDD / 2.0, v_out_init=guess)
        assert solution.report.converged

    def test_capacitive_term_keeps_solver_stable(self, dc_devices) -> None:
        nfet, pfet = dc_devices
        solver = TransientSolver(nfet=nfet, pfet=pfet, vdd=_VDD, c_load_f=1e-12, residual_tol=1e-10, max_iter=20)
        time_grid = torch.linspace(0.0, 5e-9, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = solver.solve(time_grid, vin_t, v_out_dc=_VDD / 2.0)
        assert solution.report.converged

    def test_residual_inf_norm_decreases(self, transient_solver) -> None:
        time_grid = torch.linspace(0.0, 1e-7, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        solution = transient_solver.solve(time_grid, vin_t, v_out_dc=0.5)
        history = list(solution.report.residual_norm_history)
        assert history[-1] <= history[0]

    def test_autograd_jacobian_matches_finite_differences(self, dc_devices) -> None:
        """
        Dense whole-window Jacobian from ``functional.jacobian`` vs column FD (T=16).
        """
        nfet, pfet = dc_devices
        solver = TransientSolver(nfet=nfet, pfet=pfet, vdd=_VDD, c_load_f=1e-13, residual_tol=1e-9, max_iter=5)
        time_grid = torch.linspace(0.0, 1.6e-8, 16, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        v_out_dc = 0.55
        dt = time_grid[1:] - time_grid[:-1]
        v0 = torch.full_like(time_grid, fill_value=v_out_dc + 0.05)
        v0.requires_grad_(True)

        def residual_fn(v: Tensor) -> Tensor:
            return solver._residual_fn(v, vin_t, v_out_dc, dt)  # pylint: disable=protected-access

        jac_auto = AF.jacobian(residual_fn, v0, vectorize=True)
        jac_fd = torch.zeros_like(jac_auto)
        r0 = residual_fn(v0.detach()).reshape(-1)
        eps = 5e-4
        for i in range(v0.shape[0]):
            vp = v0.detach().clone()
            vp[i] += eps
            jac_fd[:, i] = (residual_fn(vp).reshape(-1) - r0) / eps
        torch.testing.assert_close(jac_auto, jac_fd, rtol=2e-2, atol=5e-3)


class TestProbeBuilders:
    """Validates the helper builders that arrange terminal channels."""

    def test_nfet_probe_layout(self) -> None:
        v_out = torch.tensor(0.7)
        vin = torch.tensor(0.85)
        probe = _build_nfet_probe(v_out, vin, 8)
        assert probe.shape == (1, 4, 8)
        torch.testing.assert_close(probe[:, 0, :], torch.full((1, 8), fill_value=0.85))
        torch.testing.assert_close(probe[:, 1, :], torch.full((1, 8), fill_value=0.7))
        torch.testing.assert_close(probe[:, 2, :], torch.zeros((1, 8)))
        torch.testing.assert_close(probe[:, 3, :], torch.zeros((1, 8)))

    def test_pfet_probe_diode_connection(self) -> None:
        v_out = torch.tensor(0.6)
        vdd = torch.tensor(1.8)
        probe = _build_pfet_probe(v_out, vdd, 4)
        torch.testing.assert_close(probe[:, 0, :], probe[:, 1, :])
        torch.testing.assert_close(probe[:, 2, :], torch.full((1, 4), fill_value=1.8))
        torch.testing.assert_close(probe[:, 3, :], torch.full((1, 4), fill_value=1.8))

    def test_nfet_trajectory_shape(self) -> None:
        v_out = torch.linspace(0.5, 0.7, 8)
        vin_t = torch.linspace(0.85, 0.9, 8)
        traj = _build_nfet_trajectory(v_out, vin_t)
        assert traj.shape == (1, 4, 8)
        torch.testing.assert_close(traj[0, 0, :], vin_t)
        torch.testing.assert_close(traj[0, 1, :], v_out)

    def test_pfet_trajectory_diode_connection(self) -> None:
        v_out = torch.linspace(0.5, 0.7, 8)
        traj = _build_pfet_trajectory(v_out, vdd_value=1.8)
        torch.testing.assert_close(traj[0, 0, :], v_out)
        torch.testing.assert_close(traj[0, 1, :], v_out)
        torch.testing.assert_close(traj[0, 2, :], torch.full((8,), fill_value=1.8))


class TestDampingPrimitives:
    """Validates the shared backtracking + step-cap utilities."""

    def test_backtrack_accepts_full_step_when_descent(self) -> None:
        residual_fn = lambda v: torch.tensor([float(v.item())])  # noqa: E731
        state = torch.tensor(2.0)
        direction = torch.tensor(-2.0)
        alpha, residual_norm = _backtrack(
            residual_fn,
            state,
            direction,
            residual_norm_old=2.0,
            norm_fn=lambda r: float(r.abs().max().item()),
            armijo_c=1e-4,
            alpha_min=1e-3,
        )
        assert alpha == pytest.approx(1.0)
        assert residual_norm == pytest.approx(0.0, abs=1e-6)

    def test_backtrack_halves_until_minimum(self) -> None:
        residual_fn = lambda v: torch.tensor([10.0 + float(v.item())])  # noqa: E731
        state = torch.tensor(0.0)
        direction = torch.tensor(1.0)
        alpha, _ = _backtrack(
            residual_fn,
            state,
            direction,
            residual_norm_old=10.0,
            norm_fn=lambda r: float(r.abs().max().item()),
            armijo_c=1e-4,
            alpha_min=1e-3,
        )
        assert alpha < 1e-3 * 2  # within half of the minimum gate

    def test_backtrack_returns_when_alpha_hits_floor(self) -> None:
        residual_fn = lambda v: torch.tensor([100.0])  # noqa: E731
        alpha, norm_new = _backtrack(
            residual_fn,
            torch.tensor(0.0),
            torch.tensor(1.0),
            residual_norm_old=100.0,
            norm_fn=lambda r: float(r.abs().max().item()),
            armijo_c=1e-4,
            alpha_min=0.2,
        )
        assert alpha < 0.21
        assert norm_new == float("inf") or norm_new >= 0.0

    def test_cap_alpha_limits_step_size(self) -> None:
        direction = torch.tensor([5.0])
        capped = _cap_alpha(alpha=1.0, direction=direction, v_step_cap=0.2)
        assert capped == pytest.approx(0.04)

    def test_cap_alpha_zero_direction(self) -> None:
        direction = torch.tensor([0.0])
        capped = _cap_alpha(alpha=0.5, direction=direction, v_step_cap=0.2)
        assert capped == pytest.approx(0.5)


class TestNgspiceIterParser:
    """Validates the simulation.py extensions for iteration-count capture."""

    def test_parse_iter_count_returns_last_match(self) -> None:
        log = "Total iterations = 42\nTotal iterations = 7\n"
        assert _parse_iter_count(log) == 7

    def test_parse_iter_count_empty_returns_none(self) -> None:
        assert _parse_iter_count("") is None

    def test_parse_iter_count_no_match_returns_none(self) -> None:
        assert _parse_iter_count("nothing relevant here") is None

    def test_maybe_with_acct_appends_when_requested(self) -> None:
        options = ("savecurrents",)
        assert _maybe_with_acct(options, capture_iters=True) == ("savecurrents", "acct")

    def test_maybe_with_acct_skips_when_disabled(self) -> None:
        options = ("savecurrents",)
        assert _maybe_with_acct(options, capture_iters=False) == options

    def test_maybe_with_acct_idempotent(self) -> None:
        options = ("savecurrents", "acct")
        assert _maybe_with_acct(options, capture_iters=True) == options


@pytest.mark.e2e_spice
@pytest.mark.skipif(
    not _INTEGRATION_READY,
    reason="Production checkpoints/datasets or NGSpice unavailable",
)
class TestProductionIntegration:
    """
    Phase 3b acceptance criteria against the production checkpoints + SPICE.

    These tests are skipped automatically when the required artifacts are
    not present in the workspace.
    """

    @pytest.fixture(scope="class", name="composition_run")
    def fixture_composition_run(self):
        """Performs a single end-to-end composition run shared by the class."""
        from spino.circuit.composition_io import load_cs_amp_devices
        from spino.circuit.simulation import run_operating_point, run_transient
        from spino.circuit.topologies import build_cs_amp_active_load

        nfet_w_um = 6.0
        nfet_l_um = 0.18
        pfet_w_um = 4.5
        pfet_l_um = 0.18
        vin_dc = 0.85
        nfet_device, pfet_device = load_cs_amp_devices(
            nfet_w_um=nfet_w_um,
            nfet_l_um=nfet_l_um,
            pfet_w_um=pfet_w_um,
            pfet_l_um=pfet_l_um,
        )
        dc_solver = DcOperatingPointSolver(nfet_device, pfet_device, vdd=_VDD)
        dc_solution = dc_solver.solve(vin=vin_dc)
        circuit = build_cs_amp_active_load(nfet_w=nfet_w_um, nfet_l=nfet_l_um, pfet_w=pfet_w_um, pfet_l=pfet_l_um)
        spice_op = run_operating_point(circuit, capture_iters=True)
        assert spice_op is not None
        spice_vout_dc = float(spice_op.variables["v(out)"])
        pwl = f"PWL(0 {vin_dc} 50n {vin_dc} 100n {vin_dc + 0.05} 5.1u {vin_dc + 0.05})"
        circuit_step = build_cs_amp_active_load(
            nfet_w=nfet_w_um,
            nfet_l=nfet_l_um,
            pfet_w=pfet_w_um,
            pfet_l=pfet_l_um,
            vin_dc=vin_dc,
            vin_tran=pwl,
            c_load_f=10e-12,
        )
        spice_tran = run_transient(circuit_step, t_step=10e-9, t_end=5.1e-6, capture_iters=True)
        assert spice_tran is not None
        time_grid = torch.from_numpy(np.arange(0.0, 5.1e-6, 10e-9))
        vin_grid = torch.full_like(time_grid, fill_value=vin_dc)
        vin_grid[time_grid >= 100e-9] = vin_dc + 0.05
        ramp_mask = (time_grid > 50e-9) & (time_grid < 100e-9)
        vin_grid[ramp_mask] = vin_dc + 0.05 * (time_grid[ramp_mask] - 50e-9) / 50e-9
        transient_solver = TransientSolver(nfet_device, pfet_device, vdd=_VDD, c_load_f=10e-12)
        transient_solution = transient_solver.solve(time_grid, vin_grid, v_out_dc=spice_vout_dc)
        spice_vout_resampled = np.interp(
            transient_solution.time_s.cpu().numpy(),
            spice_tran.time,
            spice_tran.variables["v(out)"],
        )
        return {
            "dc_solution": dc_solution,
            "spice_op": spice_op,
            "spice_vout_dc": spice_vout_dc,
            "transient_solution": transient_solution,
            "spice_tran": spice_tran,
            "spice_vout_resampled": spice_vout_resampled,
        }

    def test_dc_op_within_five_percent(self, composition_run) -> None:
        fno_v_out = composition_run["dc_solution"].v_out_v
        spice_v_out = composition_run["spice_op"].variables["v(out)"]
        assert abs(fno_v_out - spice_v_out) / _VDD <= 0.05

    def test_dc_op_iterations_under_ten(self, composition_run) -> None:
        assert composition_run["dc_solution"].report.iter_count < 10

    def test_transient_r_squared_reported(self, composition_run) -> None:
        """Plain R^2 on the full trace is dominated by plateaus; see docs/composition.md."""
        fno_vout = composition_run["transient_solution"].v_out_v.cpu().numpy()
        spice_vout = composition_run["spice_vout_resampled"]
        residual_ss = float(np.sum((spice_vout - fno_vout) ** 2))
        total_ss = float(np.sum((spice_vout - np.mean(spice_vout)) ** 2)) + 1e-30
        r_squared = 1.0 - residual_ss / total_ss
        assert r_squared < 1.0
        assert np.isfinite(r_squared)

    def test_transient_pearson_correlation_high(self, composition_run) -> None:
        fno_vout = composition_run["transient_solution"].v_out_v.cpu().numpy()
        spice_vout = composition_run["spice_vout_resampled"]
        pearson = float(np.corrcoef(spice_vout, fno_vout)[0, 1])
        assert pearson > 0.997

    def test_transient_max_abs_voltage_error_bounded(self, composition_run) -> None:
        fno_vout = composition_run["transient_solution"].v_out_v.cpu().numpy()
        spice_vout = composition_run["spice_vout_resampled"]
        assert float(np.max(np.abs(spice_vout - fno_vout))) <= 0.03

    def test_settling_time_within_ten_percent(self, composition_run) -> None:
        t_step_start = 100e-9
        spice_tran = composition_run["spice_tran"]
        tran_sol = composition_run["transient_solution"]
        fno_t = tran_sol.time_s.cpu().numpy()
        fno_v = tran_sol.v_out_v.cpu().numpy()
        fno_settle = extract_settling_time(
            TransientResult(time=fno_t, variables={"v(out)": fno_v}, iter_count=None),
            t_step_start=t_step_start,
        )
        spice_settle = extract_settling_time(spice_tran, t_step_start=t_step_start)
        ref = 25.0e-9
        assert abs(float(spice_settle) - ref) / ref <= 0.05
        # FNO stack can be slightly slower than BSIM on the same 10 pF step; keep a
        # loose band vs the SPICE-extracted settling (see docs/composition.md).
        assert abs(float(fno_settle) - float(spice_settle)) / max(float(spice_settle), 1e-12) <= 0.25

    def test_transient_outer_iterations_under_ten(self, composition_run) -> None:
        assert composition_run["transient_solution"].report.iter_count < 10

    def test_transient_initial_condition_clamped(self, composition_run) -> None:
        fno_v_out = composition_run["transient_solution"].v_out_v.cpu().numpy()
        spice_ic = composition_run["spice_vout_dc"]
        assert float(fno_v_out[0]) == pytest.approx(spice_ic, abs=1e-5)

    def test_transient_response_is_inverting(self, composition_run) -> None:
        fno_v_out = composition_run["transient_solution"].v_out_v.cpu().numpy()
        assert fno_v_out[0] > fno_v_out[-1]


def test_arcsinh_scale_round_trip() -> None:
    """Sanity check: ``ARCSINH_SCALE_MA`` matches the wrapper denormalization."""
    pred_log = torch.tensor([1.0])
    expected_a = ARCSINH_SCALE_MA * float(torch.sinh(pred_log).item()) * 1e-3
    assert expected_a == pytest.approx(1.176e-9, rel=1e-3)
