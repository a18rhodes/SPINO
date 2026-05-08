"""
Unit tests for inverter-chain composition solvers and partition-cap grids.

Uses linear mock FNO devices (same construction idea as ``test_composition``)
so no checkpoint files are required.
"""

# pylint: disable=too-few-public-methods

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from spino.circuit.chain_composition import (
    ChainDcSolver,
    ChainTransientSolver,
    default_chain_dc_voltage_guess,
)
from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.partition_caps import TorchPartitionCapGrid, load_torch_partition_caps

_VDD = 1.8
_T_PROBE = 64
_TRIM = 16
_PHYSICS_DIM = 4


class _LinearKclMock(nn.Module):
    """
    Linear mock in arcsinh-mA space (see ``tests/circuit/test_composition.py``).
    """

    def __init__(self, slope: float, intercept: float = 0.0) -> None:
        """Stores linear coefficients in arcsinh-mA space."""
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.intercept = nn.Parameter(torch.tensor(intercept))

    def forward(self, v_terminals: Tensor, physical_params: Tensor) -> Tensor:
        """Returns ``slope * (V_d - V_s) + intercept`` in arcsinh-mA space."""
        del physical_params
        v_d = v_terminals[:, 1:2, :]
        v_s = v_terminals[:, 2:3, :]
        return self.slope * (v_d - v_s) + self.intercept


def _identity_wrapper(model: nn.Module, label: str) -> FnoMosfetDevice:
    """Builds a wrapper with identity normalization for deterministic tests."""
    return FnoMosfetDevice(
        model=model,
        v_mean=torch.zeros((4, 1)),
        v_std=torch.ones((4, 1)),
        p_mean=torch.zeros(_PHYSICS_DIM),
        p_std=torch.ones(_PHYSICS_DIM),
        physics_raw=torch.zeros(_PHYSICS_DIM),
        label=label,
    )


def _mock_chain_pair() -> tuple[FnoMosfetDevice, FnoMosfetDevice]:
    """One inverter worth of opposing-slope linear mocks."""
    return _identity_wrapper(_LinearKclMock(1.0), "XN"), _identity_wrapper(_LinearKclMock(-1.0), "XP")


def _tiny_cap_npz(path, *, nfet: bool) -> None:
    """Writes a minimal 2x2 partition-cap archive for unit tests."""
    v1 = np.array([0.0, 1.8], dtype=np.float64)
    v2 = np.array([0.0, 1.8], dtype=np.float64)
    z = np.ones((2, 2), dtype=np.float64) * 1e-20
    if nfet:
        np.savez(path, vgs=v1, vds=v2, cgs=z, cgd=z, cgb=z)
    else:
        np.savez(path, vsg=v1, vsd=v2, cgs=z, cgd=z, cgb=z)


@pytest.fixture(name="chain_devices_n2")
def fixture_chain_devices_n2() -> tuple[tuple[FnoMosfetDevice, ...], tuple[FnoMosfetDevice, ...]]:
    """Two NFET/PFET mock pairs for a two-stage chain."""
    n0, p0 = _mock_chain_pair()
    n1, p1 = _mock_chain_pair()
    return (n0, n1), (p0, p1)


class TestDefaultChainDcGuess:
    """Alternating rail guesses for vector DC NR."""

    def test_low_vin_starts_high_on_first_stage(self) -> None:
        """Vin below VDD/2 implies first output sits near the high rail."""
        v = default_chain_dc_voltage_guess(0.0, 3, vdd=1.8, dtype=torch.float32, device=torch.device("cpu"))
        torch.testing.assert_close(v, torch.tensor([1.8, 0.0, 1.8]))

    def test_high_vin_flips_first_stage(self) -> None:
        """Vin at/above VDD/2 implies first output sits near ground."""
        v = default_chain_dc_voltage_guess(1.8, 2, vdd=1.8, dtype=torch.float32, device=torch.device("cpu"))
        torch.testing.assert_close(v, torch.tensor([0.0, 1.8]))


class TestChainDcSolver:
    """Vector DC Newton coverage for N=2."""

    def test_two_stage_converges_and_shape(self, chain_devices_n2) -> None:
        """Mock chain reaches tolerance and returns ``(2,)`` voltages."""
        nfets, pfets = chain_devices_n2
        solver = ChainDcSolver(
            nfets,
            pfets,
            vdd=_VDD,
            t_probe=_T_PROBE,
            trim=_TRIM,
            max_iter=30,
            residual_tol=1e-10,
        )
        sol = solver.solve(vin=0.85)
        assert sol.reports[0].converged
        assert sol.v_out_v.shape == (2,)


class TestChainTransientSolver:
    """Whole-window transient with negligible partition caps."""

    def test_single_stage_flat_vin_matches_dc_manifold(self, tmp_path) -> None:
        """Constant ``Vin`` and tiny caps imply flat output at the DC root."""
        nfet, pfet = _mock_chain_pair()
        npz = tmp_path / "nf.npz"
        _tiny_cap_npz(npz, nfet=True)
        npzp = tmp_path / "pf.npz"
        _tiny_cap_npz(npzp, nfet=False)
        caps_n = load_torch_partition_caps(npz, is_pfet=False)
        caps_p = load_torch_partition_caps(npzp, is_pfet=True)
        solver = ChainTransientSolver(
            (nfet,),
            (pfet,),
            caps_n,
            caps_p,
            vdd=_VDD,
            c_load_f=0.0,
            max_iter=40,
            residual_tol=1e-9,
        )
        dc = ChainDcSolver(
            (nfet,),
            (pfet,),
            vdd=_VDD,
            t_probe=_T_PROBE,
            trim=_TRIM,
            residual_tol=1e-10,
        ).solve(vin=0.85)
        v_dc = dc.v_out_v.unsqueeze(0)
        time_grid = torch.linspace(0.0, 20e-9, 12, dtype=torch.float32)
        vin_t = torch.full_like(time_grid, fill_value=0.85)
        tr = solver.solve(time_grid, vin_t, v_dc[0])
        assert tr.report.converged
        torch.testing.assert_close(tr.v_nodes_v[0], torch.full_like(time_grid, v_dc[0, 0]), rtol=1e-4, atol=1e-4)


class TestTorchPartitionCapGrid:
    """Bilinear sampling smoke test."""

    def test_corners_match_grid(self) -> None:
        """Corner ``(0,0)`` returns the programmed ``cgs/cgd/cgb`` tuple."""
        v1 = torch.tensor([0.0, 2.0], dtype=torch.float32)
        v2 = torch.tensor([0.0, 4.0], dtype=torch.float32)
        cgs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        cgd = cgs * 0.1
        cgb = cgs * 0.2
        grid = TorchPartitionCapGrid(v1, v2, cgs, cgd, cgb)
        o1, o2, o3 = grid(torch.tensor(0.0), torch.tensor(0.0))
        assert o1 == pytest.approx(1.0)
        assert o2 == pytest.approx(0.1)
        assert o3 == pytest.approx(0.2)
