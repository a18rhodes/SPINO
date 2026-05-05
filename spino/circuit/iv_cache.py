"""
IV surface caching and Step-1b validation for hybrid FNO/SPICE composition.

Caches use :mod:`spino.circuit.standalone_mosfet` (same PDK path and corner as
the composition harness) so lookup tables stay consistent with live ``.op``.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spino.circuit.simulation import run_operating_point
from spino.circuit.standalone_mosfet import (
    build_isolated_mosfet_circuit,
    isolated_mosfet_id_a,
)

logger = logging.getLogger(__name__)

_ABS_MODE_SWITCH_A = 10e-9
_ABS_FLOOR_A = 10e-9
ABS_MODE_SWITCH_A = _ABS_MODE_SWITCH_A
ABS_FLOOR_A = _ABS_FLOOR_A
_REL_FRAC_PASS_DEFAULT = 0.01

__all__ = [
    "ABS_MODE_SWITCH_A",
    "ABS_FLOOR_A",
    "RelAbsThresholds",
    "build_iv_cache_npz",
    "load_iv_cache_npz",
    "validate_iv_cache_against_fresh_op",
    "write_cache_metadata",
]


@dataclass(frozen=True, slots=True)
class RelAbsThresholds:
    """
    Pass/fail rules for cache spot-checks (see error-attribution plan).

    When ``|I_D| > rel_eps``, require relative error below ``rel_frac``.
    Otherwise require absolute error below ``abs_floor``.
    """

    rel_frac: float = _REL_FRAC_PASS_DEFAULT
    rel_eps: float = _ABS_MODE_SWITCH_A
    abs_floor: float = _ABS_FLOOR_A


def _even_vg_chunk_ranges(vg_count: int, max_chunks: int) -> list[tuple[int, int]]:
    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1")
    n_chunks = min(max_chunks, vg_count)
    if n_chunks <= 0:
        return []
    base = vg_count // n_chunks
    rem = vg_count % n_chunks
    ranges: list[tuple[int, int]] = []
    start = 0
    for c in range(n_chunks):
        end = start + base + (1 if c < rem else 0)
        ranges.append((start, end))
        start = end
    return ranges


def _simulate_iv_cache_vg_chunk(
    args: tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        bool,
        float,
        float,
        float,
        float,
        str,
        str,
    ],
) -> tuple[int, int, np.ndarray]:
    """
    Runs isolated-device ``.op`` for one contiguous block of Vg rows (all Vd).

    Lives at module scope so :class:`ProcessPoolExecutor` workers can import it.
    """
    i0, i1_excl, vg_block, vd_vals, is_pfet, width_um, length_um, vs, vb, pdk_root, corner = args
    n_rows = i1_excl - i0
    block = np.zeros((n_rows, int(vd_vals.shape[0])), dtype=np.float64)
    for ii, vg in enumerate(vg_block):
        for j, vd in enumerate(vd_vals):
            ck = build_isolated_mosfet_circuit(
                is_pfet=is_pfet,
                width_um=width_um,
                length_um=length_um,
                vg=float(vg),
                vd=float(vd),
                vs=float(vs),
                vb=float(vb),
                pdk_root=pdk_root,
                corner=corner,
            )
            op = run_operating_point(ck)
            if op is None:
                raise RuntimeError(f"NGSpice .op failed at vg={vg}, vd={vd}")
            block[ii, j] = isolated_mosfet_id_a(op)
    return i0, i1_excl, block


def build_iv_cache_npz(
    *,
    out_path: Path,
    is_pfet: bool,
    width_um: float,
    length_um: float,
    vg_vals: np.ndarray,
    vd_vals: np.ndarray,
    vs: float,
    vb: float,
    pdk_root: str,
    corner: str = "tt",
    max_workers: int | None = None,
) -> dict[str, Any]:
    """
    Fills a 2D I_D grid using isolated-device ``.op`` at each (Vg, Vd) point.

    :param out_path: Path to ``.npz`` output.
    :param is_pfet: PMOS when True, else NMOS.
    :param width_um: Channel width (um).
    :param length_um: Channel length (um).
    :param vg_vals: 1-D gate voltage samples (V).
    :param vd_vals: 1-D drain voltage samples (V).
    :param vs: Fixed source bias (V).
    :param vb: Fixed bulk bias (V).
    :param pdk_root: PDK root for library resolution.
    :param corner: Process corner.
    :param max_workers: When ``None`` or ``<= 1``, runs the grid sequentially in
        this process. Otherwise splits Vg rows across a
        :class:`~concurrent.futures.ProcessPoolExecutor` with at most this many
        worker processes (capped by the Vg dimension). Workers use the
        ``spawn`` start method so NGSpice runs in clean subprocesses.
    :return: Metadata dict written beside cache as JSON when requested.
    """
    ids = np.zeros((vg_vals.size, vd_vals.size), dtype=np.float64)
    workers = 1 if max_workers is None or max_workers <= 1 else int(max_workers)
    if workers == 1:
        for i, vg in enumerate(vg_vals):
            for j, vd in enumerate(vd_vals):
                ck = build_isolated_mosfet_circuit(
                    is_pfet=is_pfet,
                    width_um=width_um,
                    length_um=length_um,
                    vg=float(vg),
                    vd=float(vd),
                    vs=float(vs),
                    vb=float(vb),
                    pdk_root=pdk_root,
                    corner=corner,
                )
                op = run_operating_point(ck)
                if op is None:
                    raise RuntimeError(f"NGSpice .op failed at vg={vg}, vd={vd}")
                ids[i, j] = isolated_mosfet_id_a(op)
    else:
        vd_copy = np.asarray(vd_vals, dtype=np.float64)
        specs: list[
            tuple[
                int,
                int,
                np.ndarray,
                np.ndarray,
                bool,
                float,
                float,
                float,
                float,
                str,
                str,
            ]
        ] = []
        for i0, i1 in _even_vg_chunk_ranges(int(vg_vals.size), workers):
            specs.append(
                (
                    i0,
                    i1,
                    np.ascontiguousarray(vg_vals[i0:i1], dtype=np.float64),
                    vd_copy,
                    is_pfet,
                    float(width_um),
                    float(length_um),
                    float(vs),
                    float(vb),
                    pdk_root,
                    corner,
                )
            )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            for i0, i1_excl, block in pool.map(_simulate_iv_cache_vg_chunk, specs):
                ids[i0:i1_excl, :] = block
    meta = {
        "is_pfet": is_pfet,
        "width_um": width_um,
        "length_um": length_um,
        "vs_v": vs,
        "vb_v": vb,
        "vg_vals": vg_vals.tolist(),
        "vd_vals": vd_vals.tolist(),
        "pdk_root": pdk_root,
        "corner": corner,
    }
    np.savez_compressed(
        out_path,
        vg=vg_vals,
        vd=vd_vals,
        ids=ids,
        is_pfet=is_pfet,
        width_um=width_um,
        length_um=length_um,
        vs=np.float64(vs),
        vb=np.float64(vb),
    )
    return meta


def load_iv_cache_npz(path: Path) -> dict[str, np.ndarray]:
    """
    Loads a cache written by :func:`build_iv_cache_npz`.

    :param path: Path to ``.npz``.
    :return: Mapping with arrays ``vg``, ``vd``, ``ids``.
    """
    raw = np.load(path, allow_pickle=False)
    return {"vg": raw["vg"], "vd": raw["vd"], "ids": raw["ids"], "vs": raw["vs"], "vb": raw["vb"]}


def validate_iv_cache_against_fresh_op(
    cache_path: Path,
    *,
    is_pfet: bool,
    width_um: float,
    length_um: float,
    vs: float,
    vb: float,
    pdk_root: str,
    corner: str = "tt",
    spot_indices: list[tuple[int, int]] | None = None,
    thresholds: RelAbsThresholds | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Spot-checks cached I_D against fresh isolated-device ``.op`` runs.

    :param cache_path: Path to ``.npz`` cache.
    :param is_pfet: Device polarity used when building the cache.
    :param width_um: Channel width (um).
    :param length_um: Channel length (um).
    :param vs: Source bias used for the cache (must match file).
    :param vb: Bulk bias used for the cache (must match file).
    :param pdk_root: PDK root for fresh simulations.
    :param corner: Corner label.
    :param spot_indices: Optional ``(i,j)`` indices into ``vg x vd`` grid;
        default covers up to 10 scattered interior/near-edge points.
    :param thresholds: Relative/absolute acceptance rule.
    :return: Tuple ``(all_passed, per-point records)``.
    """
    thr = thresholds or RelAbsThresholds()
    data = np.load(cache_path, allow_pickle=False)
    vg_arr = data["vg"]
    vd_arr = data["vd"]
    ids_cache = data["ids"]
    if spot_indices is None:
        ni, nj = ids_cache.shape
        pairs: list[tuple[int, int]] = []
        for idx, (ig, jd) in enumerate(
            [
                (0, 0),
                (ni - 1, nj - 1),
                (ni // 2, nj // 2),
                (ni // 4, nj // 4),
                (3 * ni // 4, 3 * nj // 4),
                (1, nj // 2),
                (ni // 2, 1),
                (ni - 2, nj // 2),
            ]
        ):
            if idx >= 10:
                break
            if ig < ni and jd < nj:
                pairs.append((ig, jd))
        spot_indices = pairs[:10]
    records: list[dict[str, Any]] = []
    ok = True
    for ig, jd in spot_indices:
        vg = float(vg_arr[ig])
        vd = float(vd_arr[jd])
        ic = float(ids_cache[ig, jd])
        ck = build_isolated_mosfet_circuit(
            is_pfet=is_pfet,
            width_um=width_um,
            length_um=length_um,
            vg=vg,
            vd=vd,
            vs=vs,
            vb=vb,
            pdk_root=pdk_root,
            corner=corner,
        )
        op = run_operating_point(ck)
        if op is None:
            records.append({"i": ig, "j": jd, "pass": False, "reason": "op_failed"})
            ok = False
            continue
        iref = isolated_mosfet_id_a(op)
        abs_err = abs(ic - iref)
        ref_mag = max(abs(iref), 1e-30)
        if abs(iref) > thr.rel_eps:
            rel_err = abs_err / ref_mag
            passed = rel_err < thr.rel_frac
            metric = {"kind": "relative", "value": rel_err, "limit": thr.rel_frac}
        else:
            passed = abs_err < thr.abs_floor
            metric = {"kind": "absolute_a", "value": abs_err, "limit": thr.abs_floor}
        records.append(
            {
                "i": ig,
                "j": jd,
                "vg": vg,
                "vd": vd,
                "id_cache_a": ic,
                "id_op_a": iref,
                "pass": passed,
                "metric": metric,
            }
        )
        ok = ok and passed
    return ok, records


def write_cache_metadata(meta_path: Path, payload: dict[str, Any]) -> None:
    """Writes JSON metadata next to an ``.npz`` cache."""
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
