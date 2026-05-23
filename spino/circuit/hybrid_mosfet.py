"""
Hybrid FNO / tabulated SPICE-IV MOSFET wrapper for composition experiments.

When terminal voltages fall in a *bad* region, drain current is taken from a
pre-computed NGSpice IV cache (``npz``); otherwise the production FNO is used.
Gradients through the tabulated path are zero by design.

**Dormant code path.** No committed pipeline calls this wrapper. It was built
for the transient Probe 3 substitution but that probe is documented as "Not
executable" in
``docs/assets/cs_amp_fno_exp2/attribution/attribution_result.json``: with a
narrow Vgs threshold the bad mask never triggers along the transient trajectory,
and with a broad threshold NR diverges because the FNO Jacobian disagrees with
the SPICE-cache currents at every timestep. The published L=0.18 VTC
attribution substitution path uses scalar ``brentq`` on the IV cache directly
in ``spino/attribution.ipynb`` Stage 7 (the cell labelled "Stage 7 - Cell 1"),
not this wrapper. Kept as a reference scaffold for a future whole-window
transient substitution that would need a consistent Jacobian story (e.g.
SPICE-cache finite-difference Jacobians); that work is not in scope here.
``tests/circuit/test_hybrid_mosfet.py`` exercises the masking and autograd
contract only; it is **not** evidence that this class participates in any
published result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from torch import nn

from spino.circuit.devices import FnoMosfetDevice

__all__ = ["HybridMosfetDevice"]


class HybridMosfetDevice(nn.Module):
    """
    Routes ``drain_current`` to NGSpice IV lookup or the wrapped FNO.

    :param fno_device: Production :class:`FnoMosfetDevice`.
    :param cache_npz: Path to ``.npz`` with ``vg``, ``vd``, ``ids`` grids.
    :param bad_region_fn: Maps ``vgs`` or ``vsg`` numpy vectors to boolean masks.
    :param use_pfet_axes: When True, ``bad_region_fn`` receives ``vsg = Vs - Vg``.
    """

    def __init__(
        self,
        fno_device: FnoMosfetDevice,
        cache_npz: Path,
        *,
        bad_region_fn: Callable[[np.ndarray], np.ndarray],
        use_pfet_axes: bool = False,
    ) -> None:
        super().__init__()
        self.fno = fno_device
        self.bad_region_fn = bad_region_fn
        self.use_pfet_axes = use_pfet_axes
        self.v_mean = fno_device.v_mean
        self.v_std = fno_device.v_std
        data = np.load(Path(cache_npz), allow_pickle=False)
        vg = np.asarray(data["vg"], dtype=np.float64)
        vd = np.asarray(data["vd"], dtype=np.float64)
        ids = np.asarray(data["ids"], dtype=np.float64)
        self._interp = RegularGridInterpolator(
            (vg, vd),
            ids,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def drain_current(self, v_terminals: torch.Tensor) -> torch.Tensor:
        """
        Maps ``(B,4,T)`` terminal voltages to drain current ``(B,1,T)``.

        Channel order is ``Vg, Vd, Vs, Vb`` per training convention.
        """
        probe = v_terminals
        vg = probe[0, 0, :].detach().cpu().numpy()
        vd = probe[0, 1, :].detach().cpu().numpy()
        vs = probe[0, 2, :].detach().cpu().numpy()
        vgs = vg - vs
        vsg = vs - vg
        bad_mask = self.bad_region_fn(vsg) if self.use_pfet_axes else self.bad_region_fn(vgs)
        if not np.any(bad_mask):
            return self.fno.drain_current(v_terminals)
        if np.all(bad_mask):
            # All timesteps in bad region. Run FNO to keep autograd graph intact
            # for the NR Jacobian (approximate shape), then override all values
            # with SPICE truth via torch.where so current magnitudes are correct.
            ids_fno = self.fno.drain_current(v_terminals)  # (1, 1, T) with grad
            pts = np.stack([vg, vd], axis=-1)
            ids_spice_full = torch.tensor(
                self._interp(pts),
                dtype=v_terminals.dtype,
                device=v_terminals.device,
            ).reshape(1, 1, -1)
            mask_t = torch.ones(1, 1, v_terminals.shape[2],
                                dtype=torch.bool, device=v_terminals.device)
            return torch.where(mask_t, ids_spice_full, ids_fno)
        ids_fno = self.fno.drain_current(v_terminals)
        tsteps = int(v_terminals.shape[2])
        ids_spice_full = torch.zeros(
            1,
            1,
            tsteps,
            dtype=v_terminals.dtype,
            device=v_terminals.device,
        )
        pts_bad = np.stack([vg[bad_mask], vd[bad_mask]], axis=-1)
        ids_bad = self._interp(pts_bad)
        bm = torch.from_numpy(bad_mask).to(device=v_terminals.device)
        ids_spice_full[0, 0, bm] = torch.tensor(
            ids_bad,
            dtype=v_terminals.dtype,
            device=v_terminals.device,
        )
        mask_t = bm.reshape(1, 1, -1).to(dtype=torch.bool)
        return torch.where(mask_t, ids_spice_full, ids_fno)
