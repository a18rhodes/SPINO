"""S10: geometry-binned M4 PFET triode-boundary error heat maps.

For each (W_mirror, L) on a grid, run an OTA composition + Probe 1
attribution and record the per-device max |ΔI| (FNO vs SPICE drain
current at the SPICE node-voltage trajectories). Runs the same grid
twice: once with the production PFET checkpoint, once with the
triode-boundary fine-tune. The pairwise difference shows where the
fine-tune helps and where it regresses across the (W_mirror, L) plane.

Output: ``runs/s10_triode_grid/{production, triode_finetune}/<key>/``
per grid point with a probe1_summary.json, plus
``runs/s10_triode_grid/grid_summary.json`` and two heat maps
(``runs/s10_triode_grid/m4_dI_{production, triode_finetune}.png``)
plus a difference heat map.

Usage::

    python -m scripts.s10_triode_grid \\
        --output-dir runs/s10_triode_grid
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_W_MIRROR_GRID: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)
_DEFAULT_L_GRID: tuple[float, ...] = (0.18, 0.40, 0.50)

_DIFF_W_UM = 8.0
_TAIL_W_UM = 2.0
_VBIAS = 1.2

_PRODUCTION_PFET_CKPT = Path("/app/spino/models/mosfet/pfet/mosfet_pmos_exp06_sweep_aug_CzBVmMi4.pt")
_PRODUCTION_PFET_DS = Path("/app/datasets/sky130_pmos_48k_sweep_aug.h5")
_TRIODE_PFET_CKPT = Path("/app/spino/models/mosfet/pfet/pmos_exp07_triode_finetune_KG1HfPbJ.pt")
_TRIODE_PFET_DS = Path("/app/datasets/sky130_pmos_50k_triode.h5")


@dataclass(frozen=True, slots=True)
class _GridPoint:
    """One (W_mirror, L) sample on the S10 attribution grid."""

    w_mirror: float
    l_um: float

    @property
    def key(self) -> str:
        return f"wmirror{self.w_mirror:.0f}_l{int(self.l_um * 100):03d}"


def _run_one(
    point: _GridPoint,
    output_root: Path,
    pfet_label: str,
    pfet_checkpoint: Path,
    pfet_dataset: Path,
) -> Path:
    """Execute compose_ota + ota_attribution at ``point`` for one PFET checkpoint."""
    run_dir = output_root / pfet_label / point.key
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=== %s | %s | %s ===", pfet_label, point.key, run_dir)
    compose_cmd = [
        sys.executable,
        "-m",
        "spino.circuit.compose_ota",
        "--diff-w",
        str(_DIFF_W_UM),
        "--mirror-w",
        str(point.w_mirror),
        "--tail-w",
        str(_TAIL_W_UM),
        "--nfet-l",
        str(point.l_um),
        "--pfet-l",
        str(point.l_um),
        "--tail-l",
        str(point.l_um),
        "--vbias",
        str(_VBIAS),
        "--output-dir",
        str(run_dir),
        "--pfet-checkpoint",
        str(pfet_checkpoint),
        "--pfet-dataset",
        str(pfet_dataset),
    ]
    try:
        subprocess.run(compose_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("compose_ota failed at %s/%s: %s", pfet_label, point.key, exc)
        return run_dir
    attribute_cmd = [
        sys.executable,
        "-m",
        "spino.circuit.ota_attribution",
        "--run-dir",
        str(run_dir),
        "--mirror-w",
        str(point.w_mirror),
        "--nfet-l",
        str(point.l_um),
        "--pfet-l",
        str(point.l_um),
        "--tail-l",
        str(point.l_um),
        "--pfet-checkpoint",
        str(pfet_checkpoint),
        "--pfet-dataset",
        str(pfet_dataset),
    ]
    try:
        subprocess.run(attribute_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logger.warning("ota_attribution failed at %s/%s: %s", pfet_label, point.key, exc)
    return run_dir


def _load_max_di(run_dir: Path) -> dict[str, float]:
    """Read probe1_summary.json and return per-device max |ΔI| in amperes.

    Schema: device names are top-level keys; the per-device record has a
    ``max_abs_error_a`` field. Non-device metadata keys are skipped.
    """
    summary_path = run_dir / "attribution" / "probe1_summary.json"
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for dev, entry in data.items():
        if not isinstance(entry, dict) or "max_abs_error_a" not in entry:
            continue
        out[dev] = float(entry["max_abs_error_a"])
    return out


def _heat_map(
    w_grid: tuple[float, ...],
    l_grid: tuple[float, ...],
    surface: np.ndarray,
    title: str,
    cmap: str,
    out_path: Path,
    label: str = r"max $|\Delta I|$ (µA)",
    symmetric_zero: bool = False,
) -> None:
    """2D heat map of ``surface`` over (W_mirror, L) with cell-centred labels."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    if symmetric_zero:
        v = float(np.nanmax(np.abs(surface)))
        kwargs = {"vmin": -v, "vmax": v}
    else:
        kwargs = {}
    im = ax.imshow(surface.T, origin="lower", aspect="auto", cmap=cmap, **kwargs)
    ax.set_xticks(range(len(w_grid)))
    ax.set_xticklabels([f"{w:g}" for w in w_grid])
    ax.set_yticks(range(len(l_grid)))
    ax.set_yticklabels([f"{l:g}" for l in l_grid])
    ax.set_xlabel(r"$W_\mathrm{mirror}$ (µm)")
    ax.set_ylabel(r"$L$ (µm; applied to all 5 devices)")
    ax.set_title(title)
    for i, w in enumerate(w_grid):
        for j, l in enumerate(l_grid):
            v = surface[i, j]
            if np.isnan(v):
                txt = "n/a"
            else:
                txt = f"{v:.2f}"
            ax.text(i, j, txt, ha="center", va="center", color="white", fontsize=9)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


@click.command()
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs/s10_triode_grid"),
    show_default=True,
)
@click.option(
    "--w-mirror-grid",
    type=str,
    default=",".join(f"{w:g}" for w in _DEFAULT_W_MIRROR_GRID),
    show_default=True,
    help="Comma-separated W_mirror values (µm).",
)
@click.option(
    "--l-grid",
    type=str,
    default=",".join(f"{l:g}" for l in _DEFAULT_L_GRID),
    show_default=True,
    help="Comma-separated L values (µm). Applied identically to nfet/pfet/tail.",
)
def main(output_dir: Path, w_mirror_grid: str, l_grid: str) -> None:
    """Run the S10 grid and write heat maps + a grid_summary.json."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    w_grid_tup = tuple(float(v) for v in w_mirror_grid.split(","))
    l_grid_tup = tuple(float(v) for v in l_grid.split(","))
    points = [_GridPoint(w, l) for w in w_grid_tup for l in l_grid_tup]
    logger.info("S10 grid: %d points (W_mirror %s, L %s)", len(points), w_grid_tup, l_grid_tup)

    runs: dict[str, dict[str, dict[str, float]]] = {"production": {}, "triode_finetune": {}}
    for point in points:
        prod_run = _run_one(point, output_dir, "production", _PRODUCTION_PFET_CKPT, _PRODUCTION_PFET_DS)
        runs["production"][point.key] = _load_max_di(prod_run)
        ft_run = _run_one(point, output_dir, "triode_finetune", _TRIODE_PFET_CKPT, _TRIODE_PFET_DS)
        runs["triode_finetune"][point.key] = _load_max_di(ft_run)

    grid_summary: dict = {
        "w_mirror_grid_um": list(w_grid_tup),
        "l_grid_um": list(l_grid_tup),
        "diff_w_um": _DIFF_W_UM,
        "tail_w_um": _TAIL_W_UM,
        "vbias_v": _VBIAS,
        "runs": runs,
    }
    (output_dir / "grid_summary.json").write_text(json.dumps(grid_summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", output_dir / "grid_summary.json")

    nw, nl = len(w_grid_tup), len(l_grid_tup)
    m4_prod = np.full((nw, nl), np.nan)
    m4_ft = np.full((nw, nl), np.nan)
    for i, w in enumerate(w_grid_tup):
        for j, l in enumerate(l_grid_tup):
            key = _GridPoint(w, l).key
            m4_prod[i, j] = runs["production"].get(key, {}).get("M4_pfet_mirror_out", float("nan")) * 1e6
            m4_ft[i, j] = runs["triode_finetune"].get(key, {}).get("M4_pfet_mirror_out", float("nan")) * 1e6
    delta = m4_ft - m4_prod

    _heat_map(w_grid_tup, l_grid_tup, m4_prod, "M4 max |ΔI| (production PFET)", "viridis", output_dir / "m4_dI_production.png")
    _heat_map(w_grid_tup, l_grid_tup, m4_ft, "M4 max |ΔI| (triode fine-tune)", "viridis", output_dir / "m4_dI_triode_finetune.png")
    _heat_map(
        w_grid_tup,
        l_grid_tup,
        delta,
        "M4 max |ΔI| change (triode fine-tune − production); negative = improvement",
        "RdBu_r",
        output_dir / "m4_dI_delta.png",
        label=r"$\Delta$ max $|\Delta I|$ (µA)",
        symmetric_zero=True,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
