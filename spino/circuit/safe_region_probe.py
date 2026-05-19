"""
FNO safe-operating-region characterisation from the L = 0.18 µm IV caches.

Loads the SPICE-only IV cache (``runs/attribution/cs_amp_fno_exp2/*.npz``)
and evaluates the production FNO at the same ``(V_gs, V_ds)`` grid via the
:class:`FnoMosfetDevice` constant-voltage probe path. The output is a
ratio-error surface ``|I_FNO − I_SPICE| / max(|I_SPICE|, I_floor)`` per
device and a set of contour heat maps over the (V_gs, V_ds) plane with
τ ∈ {0.1, 0.3, 1.0} levels.

The "safe operating region" is the locus where the surrogate's local
prediction is faithful enough to ground truth that an optimiser using the
FNO is not exploiting model error. Documents the doc-side answer to
"what stops the gradient sizing optimiser from finding a θ where the FNO
lies."

Usage::

    python -m spino.circuit.safe_region_probe \\
        --output-dir runs/safe_region/cs_amp_l018
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from spino.circuit.composition_io import (
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    DeviceCheckpoint,
    load_fno_device,
)
from spino.circuit.devices import FnoMosfetDevice
from spino.plot_styles import get_palette

logger = logging.getLogger(__name__)

_DEFAULT_IV_CACHE_DIR = Path("/app/runs/attribution/cs_amp_fno_exp2")
_DEFAULT_OUTPUT_DIR = Path("runs/safe_region/cs_amp_l018")
_TAU_LEVELS: tuple[float, ...] = (0.1, 0.3, 1.0)
_I_FLOOR_A = 1.0e-9  # 1 nA floor suppresses near-off division noise.
_T_PROBE = 256


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    """One IV cache: SPICE ``ids`` on a (V_g, V_d) grid at fixed (V_s, V_b)."""

    label: str
    vg: np.ndarray
    vd: np.ndarray
    vs: float
    vb: float
    ids_spice: np.ndarray
    width_um: float
    length_um: float
    is_pfet: bool


def _load_cache(path: Path, label: str) -> _CacheEntry:
    """Load one IV cache .npz into a structured entry."""
    d = np.load(path)
    return _CacheEntry(
        label=label,
        vg=d["vg"].astype(np.float64),
        vd=d["vd"].astype(np.float64),
        vs=float(d["vs"]),
        vb=float(d["vb"]),
        ids_spice=d["ids"].astype(np.float64),
        width_um=float(d["width_um"]),
        length_um=float(d["length_um"]),
        is_pfet=bool(d["is_pfet"]),
    )


def _evaluate_fno_grid(  # pylint: disable=too-many-locals
    device: FnoMosfetDevice,
    entry: _CacheEntry,
    *,
    t_probe: int = _T_PROBE,
) -> np.ndarray:
    """Evaluate ``device`` at every (V_g, V_d) on ``entry``'s grid.

    Returns an array ``(len(vg), len(vd))`` of FNO drain currents in amperes.
    The PMOS polarity convention is baked into training labels, so the
    returned magnitude is positive when conducting for both device classes;
    the IV cache stores the same polarity, so the surfaces are directly
    comparable.
    """
    dev_loc = device.v_mean.device
    nvg, nvd = len(entry.vg), len(entry.vd)
    fno = np.zeros((nvg, nvd), dtype=np.float64)
    # Constant trajectories: (V_g, V_d, V_s, V_b) per probe.
    vs_traj = torch.full((1, 1, t_probe), entry.vs, dtype=torch.float32, device=dev_loc)
    vb_traj = torch.full((1, 1, t_probe), entry.vb, dtype=torch.float32, device=dev_loc)
    with torch.no_grad():
        for i, vg in enumerate(entry.vg):
            vg_traj = torch.full((1, 1, t_probe), float(vg), dtype=torch.float32, device=dev_loc)
            for j, vd in enumerate(entry.vd):
                vd_traj = torch.full((1, 1, t_probe), float(vd), dtype=torch.float32, device=dev_loc)
                probe = torch.cat([vg_traj, vd_traj, vs_traj, vb_traj], dim=1)
                i_d = device.drain_current(probe)
                fno[i, j] = float(i_d[0, 0, -64:].mean())
    return fno


def _ratio_error(fno: np.ndarray, spice: np.ndarray, i_floor: float = _I_FLOOR_A) -> np.ndarray:
    """Compute the per-bin ratio error |fno - spice| / max(|spice|, i_floor)."""
    denom = np.maximum(np.abs(spice), i_floor)
    return np.abs(fno - spice) / denom


def _safe_region_summary(
    entry: _CacheEntry,
    err_ratio: np.ndarray,
    tau_levels: Sequence[float] = _TAU_LEVELS,
) -> dict:
    """Compute coverage fractions and (V_gs, V_ds) bounding boxes per τ."""
    nvg, nvd = err_ratio.shape
    total = float(nvg * nvd)
    out: dict = {
        "device": entry.label,
        "width_um": entry.width_um,
        "length_um": entry.length_um,
        "vs_v": entry.vs,
        "vb_v": entry.vb,
        "vg_range_v": [float(entry.vg.min()), float(entry.vg.max())],
        "vd_range_v": [float(entry.vd.min()), float(entry.vd.max())],
        "grid_points": int(total),
        "tau_levels": [],
    }
    for tau in tau_levels:
        mask = err_ratio <= tau
        coverage = float(mask.sum() / total)
        if mask.any():
            vg_idx, vd_idx = np.where(mask)
            vg_box = [float(entry.vg[vg_idx.min()]), float(entry.vg[vg_idx.max()])]
            vd_box = [float(entry.vd[vd_idx.min()]), float(entry.vd[vd_idx.max()])]
        else:
            vg_box = [float("nan"), float("nan")]
            vd_box = [float("nan"), float("nan")]
        out["tau_levels"].append(
            {
                "tau": tau,
                "coverage_fraction": coverage,
                "vg_bounding_box_v": vg_box,
                "vd_bounding_box_v": vd_box,
            }
        )
    return out


def _plot_safe_region(  # pylint: disable=too-many-locals
    entry: _CacheEntry,
    err_ratio: np.ndarray,
    out_path: Path,
    tau_levels: Sequence[float] = _TAU_LEVELS,
) -> None:
    """2D pcolormesh of log10(err_ratio) with τ contours overlaid."""
    palette = get_palette(dark=False)
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    log_err = np.log10(np.maximum(err_ratio, 1e-3))
    vg, vd = np.meshgrid(entry.vg, entry.vd, indexing="ij")
    mesh = ax.pcolormesh(
        vg,
        vd,
        log_err,
        cmap="viridis",
        vmin=-2.0,
        vmax=1.0,
        shading="auto",
    )
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(r"$\log_{10}\,|I_\mathrm{FNO} - I_\mathrm{SPICE}| / \max(|I_\mathrm{SPICE}|, I_\mathrm{floor})$")
    cs = ax.contour(
        vg,
        vd,
        err_ratio,
        levels=list(tau_levels),
        colors=[palette["pred"], palette["pred_sweep"], palette["gt"]],
        linewidths=1.5,
    )
    ax.clabel(cs, fmt=lambda v: f"τ={v:.2g}", inline=True, fontsize=9)
    ax.set_xlabel(r"$V_g$ (V)")
    ax.set_ylabel(r"$V_d$ (V)")
    geom = f"W={entry.width_um:g} µm, L={entry.length_um:g} µm"
    bias = f"V_s={entry.vs:g}, V_b={entry.vb:g} V"
    ax.set_title(f"FNO safe-operating region — {entry.label} ({geom}; {bias})")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option(
    "--iv-cache-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=_DEFAULT_IV_CACHE_DIR,
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_OUTPUT_DIR,
    show_default=True,
)
@click.option("--device", type=str, default=None, help="Torch device (default: cuda if available).")
@click.option(
    "--nfet-checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Override NFET checkpoint path.",
)
@click.option(
    "--pfet-checkpoint",
    type=click.Path(path_type=Path),
    default=None,
    help="Override PFET checkpoint path.",
)
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    iv_cache_dir: Path,
    output_dir: Path,
    device: str | None,
    nfet_checkpoint: Path | None,
    pfet_checkpoint: Path | None,
) -> None:
    """Generate the FNO safe-operating-region heat maps and summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    entries = [
        ("nfet_core_L018", iv_cache_dir / "nfet_iv_018_full.npz", "sky130 NFET", False),
        ("pfet_core_L018", iv_cache_dir / "pfet_iv_018_full.npz", "sky130 PFET", True),
    ]
    nfet_ckpt = nfet_checkpoint if nfet_checkpoint is not None else DEFAULT_NFET_CHECKPOINT
    pfet_ckpt = pfet_checkpoint if pfet_checkpoint is not None else DEFAULT_PFET_CHECKPOINT
    summaries: list[dict] = []
    for key, npz_path, label, is_pfet in entries:
        cache = _load_cache(npz_path, label)
        logger.info("Loaded %s cache: %d x %d grid", key, len(cache.vg), len(cache.vd))
        spec = DeviceCheckpoint(
            strategy_name="sky130_pmos" if is_pfet else "sky130_nmos",
            checkpoint_path=pfet_ckpt if is_pfet else nfet_ckpt,
            dataset_path=DEFAULT_PFET_DATASET if is_pfet else DEFAULT_NFET_DATASET,
            width_um=cache.width_um,
            length_um=cache.length_um,
            label=label,
        )
        fno_device = load_fno_device(spec, map_location=torch_device)
        logger.info("Evaluating FNO at %d x %d grid on %s", len(cache.vg), len(cache.vd), torch_device)
        fno_grid = _evaluate_fno_grid(fno_device, cache)
        err = _ratio_error(fno_grid, cache.ids_spice)

        # Save raw arrays.
        np.savez_compressed(
            output_dir / f"{key}_grid.npz",
            vg=cache.vg,
            vd=cache.vd,
            ids_spice=cache.ids_spice,
            ids_fno=fno_grid,
            err_ratio=err,
            width_um=cache.width_um,
            length_um=cache.length_um,
            vs=cache.vs,
            vb=cache.vb,
            is_pfet=cache.is_pfet,
            i_floor_a=_I_FLOOR_A,
        )
        logger.info("Wrote %s", output_dir / f"{key}_grid.npz")

        _plot_safe_region(cache, err, output_dir / f"{key}_safe_region.png")
        summary = _safe_region_summary(cache, err)
        summary["key"] = key
        summaries.append(summary)

    full_summary = {
        "iv_cache_dir": str(iv_cache_dir),
        "i_floor_a": _I_FLOOR_A,
        "tau_levels": list(_TAU_LEVELS),
        "devices": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_dir / "summary.json")

    for s in summaries:
        logger.info("=== %s ===", s["device"])
        for entry in s["tau_levels"]:
            logger.info(
                "  τ = %.2g: coverage = %.1f %%, Vg bbox = %.2f-%.2f V, Vd bbox = %.2f-%.2f V",
                entry["tau"],
                100.0 * entry["coverage_fraction"],
                entry["vg_bounding_box_v"][0],
                entry["vg_bounding_box_v"][1],
                entry["vd_bounding_box_v"][0],
                entry["vd_bounding_box_v"][1],
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
