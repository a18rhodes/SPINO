"""
Extract BSIM quasi-static partition-cap tables for inverter-chain composition.

When the Volare PDK and NGSpice are available, probes internal capacitances on
the isolated ``XIV`` device deck. Otherwise use ``--synthetic`` to mint
placeholder ``.npz`` files for reproducibility without the full PDK toolchain.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from spino.circuit.partition_caps import (
    build_pfet_cap_bias_axes,
    nfet_axes_from_iv_coords,
    pfet_axes_from_iv_coords,
    sky130_eps_oxide_f_per_m_approx,
    write_cap_metadata,
)
from spino.circuit.simulation import run_operating_point
from spino.circuit.standalone_mosfet import build_isolated_mosfet_circuit


def _partition_cap_save_directive() -> str:
    return ".save @m.xiv[cgs] @m.xiv[cgd] @m.xiv[cgb]"


def extract_nfet_grid(  # pylint: disable=too-many-arguments
    *,
    width_um: float,
    length_um: float,
    vdd: float,
    pdk_root: str,
    corner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vgs_axis, vds_axis = nfet_axes_from_iv_coords(width_um, vdd=vdd)
    cgs = np.zeros((vgs_axis.size, vds_axis.size), dtype=np.float64)
    cgd = np.zeros_like(cgs)
    cgb = np.zeros_like(cgs)
    for i, vg in enumerate(vgs_axis):
        for j, vd in enumerate(vds_axis):
            ck = build_isolated_mosfet_circuit(
                is_pfet=False,
                width_um=width_um,
                length_um=length_um,
                vg=float(vg),
                vd=float(vd),
                vs=0.0,
                vb=0.0,
                pdk_root=pdk_root,
                corner=corner,
            )
            op = run_operating_point(ck, directives_before_analysis=(_partition_cap_save_directive(),))
            if op is None:
                raise RuntimeError("NGSpice .op failed during NFET cap extraction")
            cgs[i, j], cgd[i, j], cgb[i, j] = _read_caps_from_op(op)
    return vgs_axis, vds_axis, cgs, cgd, cgb


def extract_pfet_grid(
    *,
    width_um: float,
    length_um: float,
    vdd: float,
    pdk_root: str,
    corner: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vsg_axis, vsd_axis = pfet_axes_from_iv_coords(width_um, vdd=vdd)
    cgs = np.zeros((vsg_axis.size, vsd_axis.size), dtype=np.float64)
    cgd = np.zeros_like(cgs)
    cgb = np.zeros_like(cgs)
    for i, vsg in enumerate(vsg_axis):
        for j, vsd in enumerate(vsd_axis):
            vg = float(vdd - vsg)
            vd = float(vdd - vsd)
            ck = build_isolated_mosfet_circuit(
                is_pfet=True,
                width_um=width_um,
                length_um=length_um,
                vg=vg,
                vd=vd,
                vs=float(vdd),
                vb=float(vdd),
                pdk_root=pdk_root,
                corner=corner,
            )
            op = run_operating_point(ck, directives_before_analysis=(_partition_cap_save_directive(),))
            if op is None:
                raise RuntimeError("NGSpice .op failed during PFET cap extraction")
            cgs[i, j], cgd[i, j], cgb[i, j] = _read_caps_from_op(op)
    return vsg_axis, vsd_axis, cgs, cgd, cgb


def _read_caps_from_op(op: object) -> tuple[float, float, float]:
    variables = getattr(op, "variables", {})

    def pick(sub: str) -> float:
        hits: list[tuple[str, float]] = []
        needle = f"[{sub}]"
        for name, val in variables.items():
            lower = name.lower()
            if needle in lower or lower.endswith(sub):
                hits.append((name, float(val)))
        if not hits:
            raise KeyError(f"Capacitance token {sub} missing in {list(variables)[:40]}")
        return hits[0][1]

    return pick("cgs"), pick("cgd"), pick("cgb")


def write_synthetic_tables(
    out_dir: Path,
    *,
    width_um_n: float = 0.82,
    length_um: float = 0.18,
    width_um_p: float = 2.05,
    vdd: float = 1.8,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vgn, vdn = nfet_axes_from_iv_coords(width_um_n, vdd=vdd)
    cox = sky130_eps_oxide_f_per_m_approx()
    # CI placeholder only: coarse split often uses C_gs ≈ 2 Cox/3, C_gd ≈ Cox/3 strong-
    # inversion folklore; synthetic mode does **not** model that faithfully — it lumps
    # each partition as Cox/3 for three equal placeholder columns rather than realism.
    c_tile = (cox * width_um_n * 1e-6 * length_um * 1e-6) / 3.0
    nf_cgs = np.full((vgn.size, vdn.size), c_tile, dtype=np.float64)
    np.savez_compressed(
        out_dir / "partition_caps_nfet_synthetic.npz",
        vgs=vgn,
        vds=vdn,
        cgs=nf_cgs,
        cgd=nf_cgs.copy(),
        cgb=nf_cgs.copy(),
        width_um=np.float64(width_um_n),
        length_um=np.float64(length_um),
        corner=np.array("tt"),
        vdd=np.float64(vdd),
        extraction_script=np.array(__name__),
        synthetic=np.bool_(True),
    )
    vsg_axis, vsd_axis = build_pfet_cap_bias_axes(vdd=vdd)
    c_tile_p = (cox * width_um_p * 1e-6 * length_um * 1e-6) / 3.0  # same Cox/3 split note as NMOS synthetic
    pf_cgs = np.full((vsg_axis.size, vsd_axis.size), c_tile_p, dtype=np.float64)
    np.savez_compressed(
        out_dir / "partition_caps_pfet_synthetic.npz",
        vsg=vsg_axis,
        vsd=vsd_axis,
        cgs=pf_cgs,
        cgd=pf_cgs.copy(),
        cgb=pf_cgs.copy(),
        width_um=np.float64(width_um_p),
        length_um=np.float64(length_um),
        corner=np.array("tt"),
        vdd=np.float64(vdd),
        extraction_script=np.array(__name__),
        synthetic=np.bool_(True),
    )
    meta = {
        "nfet_npz": "partition_caps_nfet_synthetic.npz",
        "pfet_npz": "partition_caps_pfet_synthetic.npz",
        "note": "Uniform third Cox/3 placeholders for CI/doc reproduction without live BSIM probing.",
        "vdd_v": vdd,
    }
    write_cap_metadata(out_dir / "partition_caps_manifest.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition-cap extraction / synthetic NPZ minting")
    sub = parser.add_subparsers(dest="cmd")
    synth = sub.add_parser("synthetic", help="Write placeholder .npz under runs/inv_chain/spice_caps")
    synth.add_argument("--output-dir", type=Path, default=Path("runs/inv_chain/spice_caps"))
    extr = sub.add_parser("extract", help="Run NGSpice DC grid (requires PDK)")
    extr.add_argument("--output-dir", type=Path, default=Path("runs/inv_chain/spice_caps"))
    extr.add_argument("--pdk-root", type=str, default="/app/sky130_volare")
    extr.add_argument("--corner", type=str, default="tt")
    extr.add_argument("--nfet-w", type=float, default=0.82)
    extr.add_argument("--nfet-l", type=float, default=0.18)
    extr.add_argument("--pfet-w", type=float, default=2.05)
    extr.add_argument("--pfet-l", type=float, default=0.18)
    extr.add_argument("--vdd", type=float, default=1.8)
    args = parser.parse_args()
    if args.cmd == "synthetic":
        write_synthetic_tables(args.output_dir)
        print(f"Wrote synthetic caps under {args.output_dir}", file=sys.stderr)
        return
    if args.cmd == "extract":
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        vg, vd, cgs, cgd, cgb = extract_nfet_grid(
            width_um=args.nfet_w,
            length_um=args.nfet_l,
            vdd=args.vdd,
            pdk_root=args.pdk_root,
            corner=args.corner,
        )
        np.savez_compressed(
            out / "partition_caps_nfet.npz",
            vgs=vg,
            vds=vd,
            cgs=cgs,
            cgd=cgd,
            cgb=cgb,
            width_um=np.float64(args.nfet_w),
            length_um=np.float64(args.nfet_l),
            corner=np.array(args.corner),
            vdd=np.float64(args.vdd),
            extraction_script=np.array("spino.circuit.extract_partition_caps extract"),
            synthetic=np.bool_(False),
        )
        vsg, vsd, cgsp, cgdp, cgbp = extract_pfet_grid(
            width_um=args.pfet_w,
            length_um=args.pfet_l,
            vdd=args.vdd,
            pdk_root=args.pdk_root,
            corner=args.corner,
        )
        np.savez_compressed(
            out / "partition_caps_pfet.npz",
            vsg=vsg,
            vsd=vsd,
            cgs=cgsp,
            cgd=cgdp,
            cgb=cgbp,
            width_um=np.float64(args.pfet_w),
            length_um=np.float64(args.pfet_l),
            corner=np.array(args.corner),
            vdd=np.float64(args.vdd),
            extraction_script=np.array("spino.circuit.extract_partition_caps extract"),
            synthetic=np.bool_(False),
        )
        write_cap_metadata(
            out / "partition_caps_manifest.json",
            {
                "nfet_npz": "partition_caps_nfet.npz",
                "pfet_npz": "partition_caps_pfet.npz",
                "vdd_v": args.vdd,
                "pdk_root": args.pdk_root,
                "corner": args.corner,
                "save_directive": _partition_cap_save_directive(),
            },
        )
        print(f"Extracted NFET/PFET caps to {out}", file=sys.stderr)
        return
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
