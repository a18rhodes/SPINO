"""
Error attribution probes for FNO-composed 5T OTA vs NGSpice.

Probe 1 — IV branch errors at SPICE node voltages:
    For each of M1–M5, build the (Vg,Vd,Vs,Vb) probe from the SPICE node
    trajectories, evaluate the corresponding FNO device, and compare the
    predicted drain current to the SPICE branch current ``i[id]``.

    A large error at a specific device and time window localises the source of
    the composition error.  The 68.8 mV max|ΔV_out| failure at both L values
    is hypothesised to originate from M4 (PFET output mirror) near VDD
    saturation; this probe tests that hypothesis.

Usage::

    python -m spino.circuit.ota_attribution \\
        --run-dir docs/assets/ota_5t_fno_l040 \\
        --nfet-l 0.40 --pfet-l 0.40 --tail-l 0.40

Outputs:

* ``<run-dir>/attribution/probe1_summary.json``  — scalar max/RMS errors per device
* ``<run-dir>/attribution/probe1_iv_errors.png``  — per-device |ΔI| vs time plot
* ``scratch/<run-dir-name>/attribution/probe1_iv_errors.npz``  — raw arrays (gitignored)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from spino.circuit.composition_io import (  # noqa: E402
    DEFAULT_NFET_CHECKPOINT,
    DEFAULT_NFET_DATASET,
    DEFAULT_PFET_CHECKPOINT,
    DEFAULT_PFET_DATASET,
    load_ota_5t_devices,
)

__all__ = ["run_probe1"]

logger = logging.getLogger(__name__)

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------


def _to_probe(vg: np.ndarray, vd: np.ndarray, vs: np.ndarray, vb: np.ndarray) -> torch.Tensor:
    """Returns (1, 4, T) probe tensor from four (T,) numpy arrays."""
    return torch.stack(
        [
            torch.from_numpy(vg).float(),
            torch.from_numpy(vd).float(),
            torch.from_numpy(vs).float(),
            torch.from_numpy(vb).float(),
        ],
        dim=0,
    ).unsqueeze(0)  # (1, 4, T)


# ---------------------------------------------------------------------------
# Probe 1: IV branch errors at SPICE node voltages
# ---------------------------------------------------------------------------


def run_probe1(
    run_dir: Path,
    *,
    diff_w_um: float,
    diff_l_um: float,
    mirror_w_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    nfet_checkpoint: Path = DEFAULT_NFET_CHECKPOINT,
    pfet_checkpoint: Path = DEFAULT_PFET_CHECKPOINT,
    nfet_dataset: Path = DEFAULT_NFET_DATASET,
    pfet_dataset: Path = DEFAULT_PFET_DATASET,
    device: str = _DEFAULT_DEVICE,
) -> dict:
    """
    Evaluates FNO branch currents at SPICE node voltages and computes |ΔI|.

    Reads ``spice_traces.npz`` from *run_dir*.  Writes results under
    ``run_dir/attribution/``.

    :return: Summary dict (also written to ``probe1_summary.json``).
    """
    # Traces are written to scratch/ by default (gitignored), not inside run_dir.
    trace_root = Path("scratch") / run_dir.name
    traces_path = trace_root / "spice_traces.npz"
    if not traces_path.exists():
        # Fall back to run_dir itself (e.g. if --trace-dir was set explicitly).
        traces_path = run_dir / "spice_traces.npz"
    if not traces_path.exists():
        raise FileNotFoundError(
            f"SPICE traces not found in scratch/{run_dir.name}/ or {run_dir}.  "
            "Re-run compose_ota to generate them."
        )

    t = np.load(traces_path)
    time_s: np.ndarray = t["time_s"]
    v_vinp: np.ndarray = t["v_vinp"]
    v_vinn: np.ndarray = t["v_vinn"]
    v_n_out: np.ndarray = t["v_n_out"]
    v_n_left: np.ndarray = t["v_n_left"]
    v_n_tail: np.ndarray = t["v_n_tail"]
    id_m1_spice: np.ndarray = t["id_m1"]
    id_m2_spice: np.ndarray = t["id_m2"]
    id_m3_spice: np.ndarray = t["id_m3"]
    id_m4_spice: np.ndarray = t["id_m4"]
    id_m5_spice: np.ndarray = t["id_m5"]

    vdd_v = 1.8
    vb_gnd = np.zeros_like(time_s)
    vb_vdd = np.full_like(time_s, vdd_v)
    vs_vdd = np.full_like(time_s, vdd_v)

    m1, m2, m3, m4, m5 = load_ota_5t_devices(
        diff_w_um=diff_w_um,
        diff_l_um=diff_l_um,
        mirror_w_um=mirror_w_um,
        mirror_l_um=mirror_l_um,
        tail_w_um=tail_w_um,
        tail_l_um=tail_l_um,
        nfet_checkpoint=nfet_checkpoint,
        pfet_checkpoint=pfet_checkpoint,
        nfet_dataset=nfet_dataset,
        pfet_dataset=pfet_dataset,
        map_location=device,
    )

    with torch.no_grad():
        # M1: NFET diff pair — Vg=Vinp, Vd=n_left, Vs=n_tail, Vb=GND
        id_m1_fno = m1.drain_current(_to_probe(v_vinp, v_n_left, v_n_tail, vb_gnd).to(device))[0, 0, :].cpu().numpy()
        # M2: NFET diff pair — Vg=Vinn, Vd=n_out, Vs=n_tail, Vb=GND
        id_m2_fno = m2.drain_current(_to_probe(v_vinn, v_n_out, v_n_tail, vb_gnd).to(device))[0, 0, :].cpu().numpy()
        # M3: PFET diode — Vg=n_left, Vd=n_left, Vs=VDD, Vb=VDD
        id_m3_fno = m3.drain_current(_to_probe(v_n_left, v_n_left, vs_vdd, vb_vdd).to(device))[0, 0, :].cpu().numpy()
        # M4: PFET output mirror — Vg=n_left, Vd=n_out, Vs=VDD, Vb=VDD
        id_m4_fno = m4.drain_current(_to_probe(v_n_left, v_n_out, vs_vdd, vb_vdd).to(device))[0, 0, :].cpu().numpy()
        # M5: NFET tail — Vg=Vbias (from v_vinp trace timing; read from traces), Vd=n_tail, Vs=GND, Vb=GND
        # Vbias is constant; read summary.json for the value
        summary_path = run_dir / "summary.json"
        vbias = 1.2
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            vbias = s.get("config", {}).get("vbias_v", vbias)
        v_vbias = np.full_like(time_s, vbias)
        id_m5_fno = m5.drain_current(_to_probe(v_vbias, v_n_tail, vb_gnd, vb_gnd).to(device))[0, 0, :].cpu().numpy()

    # SPICE PFET id convention: SPICE id for PFET is typically negative when
    # the device conducts (current flows source→drain, i.e., VDD→n_out).
    # The FNO is trained on |id| with positive sign for conducting PFET.
    # Take absolute value of SPICE PFET currents for magnitude comparison.
    devices = {
        "M1_nfet_diffpair": (id_m1_fno, id_m1_spice),
        "M2_nfet_diffpair": (id_m2_fno, id_m2_spice),
        "M3_pfet_mirror_diode": (id_m3_fno, np.abs(id_m3_spice)),
        "M4_pfet_mirror_out": (id_m4_fno, np.abs(id_m4_spice)),
        "M5_nfet_tail": (id_m5_fno, id_m5_spice),
    }

    summary: dict = {}
    errors: dict[str, np.ndarray] = {}
    for name, (fno_i, spice_i) in devices.items():
        err = np.abs(fno_i - spice_i)
        errors[name] = err
        summary[name] = {
            "max_abs_error_a": float(np.max(err)),
            "rms_error_a": float(np.sqrt(np.mean(err**2))),
            "max_fno_a": float(np.max(np.abs(fno_i))),
            "max_spice_a": float(np.max(np.abs(spice_i))),
        }
        logger.info(
            "%s: max|ΔI|=%.3e A  rms|ΔI|=%.3e A  max|I_FNO|=%.3e A  max|I_SPICE|=%.3e A",
            name,
            summary[name]["max_abs_error_a"],
            summary[name]["rms_error_a"],
            summary[name]["max_fno_a"],
            summary[name]["max_spice_a"],
        )

    att_dir = run_dir / "attribution"
    att_dir.mkdir(parents=True, exist_ok=True)

    # Raw arrays go to scratch/ (gitignored); figures and summary stay in att_dir.
    scratch_att = Path("scratch") / run_dir.name / "attribution"
    scratch_att.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scratch_att / "probe1_iv_errors.npz",
        time_s=time_s,
        v_n_out=v_n_out,
        **{f"err_{k}": v for k, v in errors.items()},
        **{f"fno_{k}": devices[k][0] for k in devices},
        **{f"spice_{k}": devices[k][1] for k in devices},
    )
    (att_dir / "probe1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_probe1(time_s, v_n_out, errors, att_dir / "probe1_iv_errors.png")

    return summary


def _plot_probe1(
    time_s: np.ndarray,
    v_n_out: np.ndarray,
    errors: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax_v, ax_i = axes
    ax_v.plot(time_s * 1e6, v_n_out, color="#0066cc", linewidth=1.2)
    ax_v.set_ylabel(r"$V_{n\_out}$ (V)")
    ax_v.set_title("SPICE n_out trajectory")
    ax_v.grid(True, alpha=0.3)

    colors = ["#cc0000", "#cc6600", "#007700", "#0000cc", "#770077"]
    for (name, err), color in zip(errors.items(), colors):
        ax_i.semilogy(time_s * 1e6, np.maximum(err, 1e-15), linewidth=1.0, label=name, color=color)
    ax_i.set_xlabel(r"$t$ ($\mu$s)")
    ax_i.set_ylabel(r"$|{ΔI}|$ (A)")
    ax_i.set_title("Per-device FNO vs SPICE |ΔI| at SPICE node voltages")
    ax_i.grid(True, which="both", alpha=0.3)
    ax_i.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--run-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory produced by compose_ota (contains spice_traces.npz and summary.json).",
)
@click.option("--diff-w", type=float, default=8.0, show_default=True)
@click.option("--mirror-w", type=float, default=8.0, show_default=True)
@click.option("--tail-w", type=float, default=2.0, show_default=True)
@click.option("--nfet-l", type=float, required=True, help="NFET channel length (µm).")
@click.option("--pfet-l", type=float, required=True, help="PFET channel length (µm).")
@click.option("--tail-l", type=float, required=True, help="Tail channel length (µm).")
@click.option("--device", type=str, default=_DEFAULT_DEVICE, show_default=True)
@click.option("--nfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_NFET_CHECKPOINT)
@click.option("--pfet-checkpoint", type=click.Path(path_type=Path), default=DEFAULT_PFET_CHECKPOINT)
@click.option("--nfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_NFET_DATASET)
@click.option("--pfet-dataset", type=click.Path(path_type=Path), default=DEFAULT_PFET_DATASET)
def main(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    run_dir: Path,
    diff_w: float,
    mirror_w: float,
    tail_w: float,
    nfet_l: float,
    pfet_l: float,
    tail_l: float,
    device: str,
    nfet_checkpoint: Path,
    pfet_checkpoint: Path,
    nfet_dataset: Path,
    pfet_dataset: Path,
) -> None:
    """OTA attribution: Probe 1 (IV branch errors at SPICE node voltages)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_probe1(
        run_dir,
        diff_w_um=diff_w,
        diff_l_um=nfet_l,
        mirror_w_um=mirror_w,
        mirror_l_um=pfet_l,
        tail_w_um=tail_w,
        tail_l_um=tail_l,
        nfet_checkpoint=nfet_checkpoint,
        pfet_checkpoint=pfet_checkpoint,
        nfet_dataset=nfet_dataset,
        pfet_dataset=pfet_dataset,
        device=device,
    )
    logger.info("Attribution complete — results in %s/attribution/", run_dir)


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
