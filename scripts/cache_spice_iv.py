#!/usr/bin/env python3
"""
Build NGSpice IV caches (``npz``) for :class:`spino.circuit.hybrid_mosfet.HybridMosfetDevice`.

Example (NFET weak-inversion band, stress geometry)::

    python scripts/cache_spice_iv.py nfet --out /tmp/nfet_iv.npz --w-um 6 --l-um 0.18

Validation spot-checks reuse :func:`spino.circuit.iv_cache.validate_iv_cache_against_fresh_op`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

from spino.circuit.iv_cache import (
    RelAbsThresholds,
    build_iv_cache_npz,
    validate_iv_cache_against_fresh_op,
    write_cache_metadata,
)

logger = logging.getLogger(__name__)


@click.group()
def cli() -> None:
    """IV cache utilities for hybrid composition experiments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@cli.command("nfet")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--w-um", type=float, required=True)
@click.option("--l-um", type=float, required=True)
@click.option("--vg-min", type=float, default=0.0)
@click.option("--vg-max", type=float, default=0.6)
@click.option("--vg-steps", type=int, default=61)
@click.option("--vd-min", type=float, default=0.0)
@click.option("--vd-max", type=float, default=1.8)
@click.option("--vd-steps", type=int, default=61)
@click.option("--pdk-root", type=str, default="/app/sky130_volare")
@click.option("--corner", type=str, default="tt")
@click.option(
    "--workers",
    type=click.IntRange(1),
    default=4,
    show_default=True,
    help="Process count for NGSpice .op sweeps (1 runs sequentially in-process).",
)
@click.option("--validate/--no-validate", default=True)
def cache_nfet(
    out: Path,
    w_um: float,
    l_um: float,
    vg_min: float,
    vg_max: float,
    vg_steps: int,
    vd_min: float,
    vd_max: float,
    vd_steps: int,
    pdk_root: str,
    corner: str,
    workers: int,
    validate: bool,
) -> None:
    """NFET IV grid with ``Vs = Vb = 0`` (reference node)."""
    vg_vals = np.linspace(vg_min, vg_max, vg_steps)
    vd_vals = np.linspace(vd_min, vd_max, vd_steps)
    meta = build_iv_cache_npz(
        out_path=out,
        is_pfet=False,
        width_um=w_um,
        length_um=l_um,
        vg_vals=vg_vals,
        vd_vals=vd_vals,
        vs=0.0,
        vb=0.0,
        pdk_root=pdk_root,
        corner=corner,
        max_workers=workers,
    )
    write_cache_metadata(out.with_suffix(".meta.json"), meta)
    if validate:
        ok, records = validate_iv_cache_against_fresh_op(
            out,
            is_pfet=False,
            width_um=w_um,
            length_um=l_um,
            vs=0.0,
            vb=0.0,
            pdk_root=pdk_root,
            corner=corner,
            thresholds=RelAbsThresholds(),
        )
        (out.parent / (out.stem + "_validation.json")).write_text(json.dumps({"pass": ok, "records": records}, indent=2))
        if not ok:
            raise SystemExit("IV cache validation failed; see validation JSON")
    logger.info("Wrote %s", out.resolve())


@cli.command("pfet")
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--w-um", type=float, required=True)
@click.option("--l-um", type=float, required=True)
@click.option("--vs-v", type=float, default=1.8, help="Source/bulk bias (V); tie to VDD in CS amp.")
@click.option("--vg-min", type=float, default=0.0)
@click.option("--vg-max", type=float, default=1.8)
@click.option("--vg-steps", type=int, default=61)
@click.option("--vd-min", type=float, default=0.0)
@click.option("--vd-max", type=float, default=1.8)
@click.option("--vd-steps", type=int, default=61)
@click.option("--pdk-root", type=str, default="/app/sky130_volare")
@click.option("--corner", type=str, default="tt")
@click.option(
    "--workers",
    type=click.IntRange(1),
    default=4,
    show_default=True,
    help="Process count for NGSpice .op sweeps (1 runs sequentially in-process).",
)
@click.option("--validate/--no-validate", default=True)
def cache_pfet(
    out: Path,
    w_um: float,
    l_um: float,
    vs_v: float,
    vg_min: float,
    vg_max: float,
    vg_steps: int,
    vd_min: float,
    vd_max: float,
    vd_steps: int,
    pdk_root: str,
    corner: str,
    workers: int,
    validate: bool,
) -> None:
    """PFET IV grid with ``Vs = Vb = vs_v`` (reference CS-amp rail)."""
    vg_vals = np.linspace(vg_min, vg_max, vg_steps)
    vd_vals = np.linspace(vd_min, vd_max, vd_steps)
    meta = build_iv_cache_npz(
        out_path=out,
        is_pfet=True,
        width_um=w_um,
        length_um=l_um,
        vg_vals=vg_vals,
        vd_vals=vd_vals,
        vs=vs_v,
        vb=vs_v,
        pdk_root=pdk_root,
        corner=corner,
        max_workers=workers,
    )
    write_cache_metadata(out.with_suffix(".meta.json"), meta)
    if validate:
        ok, records = validate_iv_cache_against_fresh_op(
            out,
            is_pfet=True,
            width_um=w_um,
            length_um=l_um,
            vs=vs_v,
            vb=vs_v,
            pdk_root=pdk_root,
            corner=corner,
            thresholds=RelAbsThresholds(),
        )
        (out.parent / (out.stem + "_validation.json")).write_text(json.dumps({"pass": ok, "records": records}, indent=2))
        if not ok:
            raise SystemExit("IV cache validation failed; see validation JSON")
    logger.info("Wrote %s", out.resolve())


if __name__ == "__main__":  # pragma: no cover
    cli()
