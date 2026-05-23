"""
Unit tests for :mod:`spino.circuit.hybrid_mosfet`.

Ensures autograd flows through the FNO branch only where the bad-region mask is false.

The wrapper under test is dormant scaffolding (see the module docstring of
``spino.circuit.hybrid_mosfet``). No committed pipeline calls it; the published
L=0.18 VTC attribution uses scalar ``brentq`` on the IV cache directly in
``spino/attribution.ipynb`` Stage 7. These tests guard the masking and autograd
contract so a future transient-substitution path can adopt it cleanly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from spino.circuit.hybrid_mosfet import HybridMosfetDevice


def test_hybrid_mixed_branch_grad_through_fno_only_where_good() -> None:
    """
    Mixed-mask path must not detach the full FNO output: gradients flow through
    ``torch.where`` only at timesteps where the bad-region mask is False.
    """

    class FakeFno(nn.Module):  # pylint: disable=abstract-method
        """Linear sensitivity to gate voltage for a deterministic nonzero grad."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("v_mean", torch.zeros(4, 1))
            self.register_buffer("v_std", torch.ones(4, 1))

        def drain_current(self, v_terminals: torch.Tensor) -> torch.Tensor:
            return v_terminals[:, 0:1, :] * 1e-3

    vg_leaf = torch.tensor([0.2, 0.2, 0.9, 0.9], requires_grad=True)
    z = torch.zeros(4, dtype=vg_leaf.dtype)
    probe = torch.stack([vg_leaf, z, z, z], dim=0).unsqueeze(0)
    vg_grid = np.linspace(0.0, 1.0, 12)
    vd_grid = np.linspace(0.0, 1.0, 12)
    ids = np.full((12, 12), 1e-4, dtype=np.float64)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp.name, vg=vg_grid, vd=vd_grid, ids=ids)
        npz_path = Path(tmp.name)
    try:
        hy = HybridMosfetDevice(
            FakeFno(),  # type: ignore[arg-type]
            npz_path,
            bad_region_fn=lambda vgs: vgs < 0.5,
            use_pfet_axes=False,
        )
        out = hy.drain_current(probe)
        assert out.shape == (1, 1, 4)
        out.sum().backward()
    finally:
        npz_path.unlink(missing_ok=True)
    g = vg_leaf.grad
    assert g is not None
    assert abs(g[0].item()) < 1e-12 and abs(g[1].item()) < 1e-12
    assert abs(g[2].item()) > 1e-12 and abs(g[3].item()) > 1e-12
