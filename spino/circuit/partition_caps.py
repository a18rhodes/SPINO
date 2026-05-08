"""
Quasi-static BSIM partition-capacitance grids for inverter-chain composition.

Loads :class:`numpy.ndarray` caches (``.npz``) and evaluates **bilinear**
interpolation in PyTorch so Newton–Raphson sees smooth ``dC/dV`` through
the interpolation support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "TorchPartitionCapGrid",
    "build_pfet_cap_bias_axes",
    "build_piecewise_nfet_vg_axis",
    "build_piecewise_pfet_vsg_axis",
    "build_uniform_axis",
    "check_cox_sum",
    "load_partition_cap_npz",
    "load_torch_partition_caps",
    "nfet_axes_from_iv_coords",
    "pfet_axes_from_iv_coords",
    "sky130_eps_oxide_f_per_m_approx",
]


def sky130_eps_oxide_f_per_m_approx() -> float:
    """
    Rough parallel-plate oxide capacitance per area for sanity checks only.

    Uses εr ≈ 3.9 and tox ≈ 4.148e-9 m (representative nfet 1v8 frontend).

    :return: Approximate Cox'' in F/m².
    """
    eps0 = 8.854e-12
    eps_si = 3.9
    tox = 4.148e-9
    return eps0 * eps_si / tox


def build_piecewise_nfet_vg_axis() -> np.ndarray:
    """
    Builds the NFET gate voltage axis per inverter-chain plan (µm-independent).

    :return: 1-D ``vgs`` samples in volts (~60 points).
    """
    a = build_uniform_axis(0.0, 0.25, 8)
    b = build_uniform_axis(0.25, 0.55, 32)
    c = build_uniform_axis(0.55, 1.8, 20)
    return np.unique(np.concatenate([a, b, c]))


def build_piecewise_pfet_vsg_axis(*, vdd: float = 1.8) -> np.ndarray:
    """
    PFET **VSG** axis with densification around |Vthp| (~0.42 V).

    Uses the same breakpoints as NFET mapped to ``VSG = VDD - Vg`` when the
    reference NFET sweep used ``Vg ∈ [0, 1.8]``:

    :param vdd: Positive supply defining ``VSG = VDD - Vg_nfet_mirror``.
    :return: 1-D ``VSG`` samples (non-negative).
    """
    vg_nfet = build_piecewise_nfet_vg_axis()
    vsg = np.clip(vdd - vg_nfet, 0.0, vdd)
    return np.unique(np.sort(vsg))


def build_uniform_axis(lo: float, hi: float, n: int) -> np.ndarray:
    """
    Inclusive-ish uniform spacing on ``[lo, hi]`` with ``n >= 2`` samples.

    :param lo: Low endpoint (V).
    :param hi: High endpoint (V).
    :param n: Sample count.
    :return: 1-D array.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    return np.linspace(lo, hi, n, dtype=np.float64)


def nfet_axes_from_iv_coords(width_um: float, vdd: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds ``(vgs_axis, vds_axis)`` for NFET capacitance tables.

    ``vds`` spans ``[0, VDD]`` with 30 uniform points.

    :param width_um: Unused placeholder for API symmetry with PFET helpers.
    :param vdd: Drain sweep limit.
    :return: Tuple ``(vgs, vds)``.
    """
    _ = width_um
    vgs = build_piecewise_nfet_vg_axis()
    vds = build_uniform_axis(0.0, vdd, 30)
    return vgs, vds


def pfet_axes_from_iv_coords(width_um: float, vdd: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds ``(vsg_axis, vsd_axis)`` for PFET partition caps (positive axes).

    :param width_um: Unused (symmetry).
    :param vdd: Supply voltage.
    :return: Tuple ``(vsg, vsd)`` each 1-D.
    """
    _ = width_um
    vsg = build_piecewise_pfet_vsg_axis(vdd=vdd)
    vsd = build_uniform_axis(0.0, vdd, 30)
    return vsg, vsd


def build_pfet_cap_bias_axes(vdd: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """
    Alias for :func:`pfet_axes_from_iv_coords` with default width.

    :param vdd: Supply voltage used for sweep limits.
    :return: ``(vsg, vsd)`` arrays.
    """
    return pfet_axes_from_iv_coords(1.0, vdd=vdd)


def load_partition_cap_npz(path: Path) -> dict[str, Any]:
    """
    Loads a partition-cap ``.npz`` written by the extraction script.

    Expected keys ``vgs``, ``vds``, ``cgs``, ``cgd``, ``cgb`` (or ``vsg``, ``vsd``
    for PMOS archives using the same layout).

    :param path: Path to ``.npz``.
    :return: Mapping of ndarray tensors plus optional scalar metadata.
    """
    raw = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {}
    for k in raw.files:
        out[k] = raw[k]
    return out


def check_cox_sum(
    *,
    cgs_f: float,
    cgd_f: float,
    cgb_f: float,
    width_m: float,
    length_m: float,
    cox_pp_f_m2: float | None = None,
    rtol: float = 0.05,
) -> bool:
    """
    Returns True when ``cgs+cgd+cgb ≈ Cox*W*L`` within ``rtol``.

    :param cgs_f: Partition cap ``C_gs`` (F).
    :param cgd_f: ``C_gd`` (F).
    :param cgb_f: ``C_gb`` (F).
    :param width_m: Channel width (m).
    :param length_m: Channel length (m).
    :param cox_pp_f_m2: Optional Cox'' override; defaults to sky130-ish estimate.
    :param rtol: Relative tolerance band (5% default).
    :return: Whether the check passes.
    """
    cox = sky130_eps_oxide_f_per_m_approx() if cox_pp_f_m2 is None else float(cox_pp_f_m2)
    c_ideal = cox * width_m * length_m
    total = float(cgs_f + cgd_f + cgb_f)
    if c_ideal <= 0.0:
        return False
    return abs(total - c_ideal) / c_ideal <= rtol


class TorchPartitionCapGrid(torch.nn.Module):
    """
    Bilinear interpolation of ``(cgs, cgd, cgb)`` on a rectilinear grid.

    Matches the IV-cache policy conceptually (regular grid); implementation is
    native torch for autograd through bias-dependent caps.

    :param v1: First-axis coordinates ``(Nv1,)`` increasing.
    :param v2: Second-axis coordinates ``(Nv2,)`` increasing.
    :param stacks: Stack ``[cgs,cgd,cgb]`` each ``(Nv1, Nv2)`` in farads.
    """

    def __init__(
        self,
        v1: Tensor,
        v2: Tensor,
        cgs: Tensor,
        cgd: Tensor,
        cgb: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("v1", v1.clone())
        self.register_buffer("v2", v2.clone())
        self.register_buffer("cgs_grid", cgs.clone())
        self.register_buffer("cgd_grid", cgd.clone())
        self.register_buffer("cgb_grid", cgb.clone())

    @classmethod
    def from_numpy(cls, v1: np.ndarray, v2: np.ndarray, cgs: np.ndarray, cgd: np.ndarray, cgb: np.ndarray) -> TorchPartitionCapGrid:
        """
        Builds from numpy grids on CPU float32 tensors.

        :param v1: First bias axis (e.g. Vgs or VSG).
        :param v2: Second axis (e.g. Vds or VSD).
        :param cgs: ``C_gs`` grid (F).
        :param cgd: ``C_gd`` grid (F).
        :param cgb: ``C_gb`` grid (F).
        :return: Module ready for ``.to(device)``.
        """
        return cls(
            v1=torch.from_numpy(np.asarray(v1, dtype=np.float32)),
            v2=torch.from_numpy(np.asarray(v2, dtype=np.float32)),
            cgs=torch.from_numpy(np.asarray(cgs, dtype=np.float32)),
            cgd=torch.from_numpy(np.asarray(cgd, dtype=np.float32)),
            cgb=torch.from_numpy(np.asarray(cgb, dtype=np.float32)),
        )

    def _bilinear_vals(self, g1: Tensor, g2: Tensor, grid2d: Tensor) -> Tensor:
        """
        Bilinear interpolation of ``grid2d`` at batches of `(g1, g2)` pairs.

        :param g1: Values on first axis, any shape ending with arbitrary broadcast.
        :param g2: Second axis samples, broadcastable with ``g1``.
        :return: Sampled values matching broadcast shape.
        """
        g1x, g2x = torch.broadcast_tensors(g1, g2)
        shape = g1x.shape
        g1f = g1x.reshape(-1)
        g2f = g2x.reshape(-1)
        g1f = torch.clamp(g1f, self.v1[0], self.v1[-1])
        g2f = torch.clamp(g2f, self.v2[0], self.v2[-1])
        i = torch.searchsorted(self.v1, g1f, right=True) - 1
        j = torch.searchsorted(self.v2, g2f, right=True) - 1
        i = torch.clamp(i, 0, self.v1.shape[0] - 2)
        j = torch.clamp(j, 0, self.v2.shape[0] - 2)
        x = self.v1[i]
        x1 = self.v1[i + 1]
        y = self.v2[j]
        y1 = self.v2[j + 1]
        tx = (g1f - x) / (x1 - x + 1e-24)
        ty = (g2f - y) / (y1 - y + 1e-24)
        c00 = grid2d[i, j]
        c01 = grid2d[i, j + 1]
        c10 = grid2d[i + 1, j]
        c11 = grid2d[i + 1, j + 1]
        out = (
            (1 - tx) * (1 - ty) * c00
            + (1 - tx) * ty * c01
            + tx * (1 - ty) * c10
            + tx * ty * c11
        )
        return out.reshape(shape)

    def forward(self, g1: Tensor, g2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns interpolated ``(cgs,cgd,cgb)`` in farads.

        :param g1: First-coordinate tensor (broadcastable batch).
        :param g2: Second-coordinate tensor.
        :return: Tuple of tensors matching broadcast shape.
        """
        cg_s = self._bilinear_vals(g1, g2, self.cgs_grid)
        cgd = self._bilinear_vals(g1, g2, self.cgd_grid)
        cgb = self._bilinear_vals(g1, g2, self.cgb_grid)
        return cg_s, cgd, cgb


def load_torch_partition_caps(path: Path, *, is_pfet: bool, map_location: torch.device | str = "cpu") -> TorchPartitionCapGrid:
    """
    Loads ``TorchPartitionCapGrid`` from an ``.npz`` on disk.

    :param path: ``.npz`` file.
    :param is_pfet: When True expects ``vsg``/``vsd`` keys; otherwise ``vgs``/``vds``.
    :param map_location: Torch device for grid tensors.
    :return: Interpolator module.
    """
    data = np.load(path, allow_pickle=True)
    if is_pfet:
        v1 = np.asarray(data["vsg"], dtype=np.float64)
        v2 = np.asarray(data["vsd"], dtype=np.float64)
    else:
        v1 = np.asarray(data["vgs"], dtype=np.float64)
        v2 = np.asarray(data["vds"], dtype=np.float64)
    grid = TorchPartitionCapGrid.from_numpy(
        v1,
        v2,
        np.asarray(data["cgs"], dtype=np.float64),
        np.asarray(data["cgd"], dtype=np.float64),
        np.asarray(data["cgb"], dtype=np.float64),
    )
    return grid.to(map_location)


def write_cap_metadata(sidecar: Path, meta: dict[str, Any]) -> None:
    """
    Writes JSON metadata next to an ``.npz`` artefact.

    :param sidecar: ``.json`` path.
    :param meta: JSON-serialisable dict.
    """
    sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
