"""CLI tool for generating offline MOSFET training datasets."""

import argparse
import logging
import sys
from pathlib import Path

import click

from spino.mosfet.gen_data import GEOMETRY_BINS, generate_offline_dataset, merge_geometry_bins

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-computed MOSFET dataset for fast training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", "-o", required=True, help="Output HDF5 file path")
    parser.add_argument("--samples", "-n", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--strategy", default="sky130_nmos", help="Device strategy (sky130_nmos, sky130_pmos)")
    parser.add_argument("--t-steps", type=int, default=2048, help="Time steps per sample")
    parser.add_argument("--t-end", type=float, default=1e-6, help="Simulation end time (seconds)")
    parser.add_argument("--workers", "-j", type=int, default=16, help="Number of parallel workers (16 recommended)")
    parser.add_argument("--gate-max", type=float, default=1.8, help="Maximum gate voltage")
    parser.add_argument("--drain-max", type=float, default=1.8, help="Maximum drain voltage")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing file (default: append)")
    parser.add_argument(
        "--waveform-mode",
        choices=[
            "pwl",
            "monotonic",
            "vth_focused",
            "subthreshold_focused",
            "deep_subthreshold",
            "transitional_subthreshold",
        ],
        default="pwl",
        help="Waveform generation mode: pwl (chaotic), monotonic (DC-sweep-like), vth_focused (gate near Vth)",
    )
    parser.add_argument(
        "--geometry-bin",
        choices=list(GEOMETRY_BINS.keys()),
        default=None,
        help="Geometry bin for stratified sampling (tiny/small/medium/large/xlarge). If not set, uses full uniform.",
    )
    parser.add_argument(
        "--w-bin",
        choices=list(GEOMETRY_BINS.keys()),
        default=None,
        help="Width bin for cross-bin geometry sampling. Requires --l-bin.",
    )
    parser.add_argument(
        "--l-bin",
        choices=list(GEOMETRY_BINS.keys()),
        default=None,
        help="Length bin for cross-bin geometry sampling. Requires --w-bin.",
    )
    args = parser.parse_args()
    if args.geometry_bin and (args.w_bin or args.l_bin):
        parser.error("--geometry-bin cannot be combined with --w-bin/--l-bin")
    if bool(args.w_bin) != bool(args.l_bin):
        parser.error("--w-bin and --l-bin must be provided together")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_config = {
        "gate_range": (0.0, args.gate_max),
        "drain_range": (0.0, args.drain_max),
    }
    if args.w_bin and args.l_bin:
        bin_info = f" [cross-bin W={args.w_bin}, L={args.l_bin}]"
    elif args.geometry_bin:
        bin_info = f" [bin={args.geometry_bin}]"
    else:
        bin_info = " [full uniform]"
    logger.info("=" * 80)
    logger.info("MOSFET Dataset Generation")
    logger.info("=" * 80)
    logger.info("Output: %s", output_path)
    logger.info("Samples: %d", args.samples)
    logger.info("Strategy: %s", args.strategy)
    logger.info("Waveform mode: %s", args.waveform_mode)
    logger.info(
        "Geometry: %s%s", args.geometry_bin or (f"{args.w_bin}x{args.l_bin}" if args.w_bin else "uniform"), bin_info
    )
    if args.geometry_bin:
        gbin = GEOMETRY_BINS[args.geometry_bin]
        logger.info("  W range: [%.2f, %.2f] um", *gbin.w_range)
        logger.info("  L range: [%.2f, %.2f] um", *gbin.l_range)
    if args.w_bin and args.l_bin:
        w_bin = GEOMETRY_BINS[args.w_bin]
        l_bin = GEOMETRY_BINS[args.l_bin]
        logger.info("  W sampled from %s: [%.2f, %.2f] um", args.w_bin, *w_bin.w_range)
        logger.info("  L sampled from %s: [%.2f, %.2f] um", args.l_bin, *l_bin.l_range)
    logger.info("Time steps: %d, End time: %.2e s", args.t_steps, args.t_end)
    logger.info("Workers: %d", args.workers)
    logger.info("Voltage config: %s", strategy_config)
    logger.info("=" * 80)
    logger.info("Estimated time: ~%.1f hours (assuming 17s/sample)", args.samples * 17 / 3600 / args.workers)
    logger.info("=" * 80)
    with click.progressbar(length=args.samples, label="Generating samples") as bar:

        def update_progress(*args_inner, **kwargs):
            bar.update(1)

        generate_offline_dataset(
            output_path=str(output_path),
            num_samples=args.samples,
            strategy_name=args.strategy,
            strategy_config=strategy_config,
            t_steps=args.t_steps,
            t_end=args.t_end,
            num_workers=args.workers,
            progress_callback=update_progress,
            overwrite=args.overwrite,
            waveform_mode=args.waveform_mode,
            geometry_bin=args.geometry_bin,
            w_bin=args.w_bin,
            l_bin=args.l_bin,
        )


def merge_main():
    """CLI entry point for merging geometry bin files."""
    parser = argparse.ArgumentParser(
        description="Merge geometry bin HDF5 files into a single stratified dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", "-o", required=True, help="Output merged HDF5 file path")
    parser.add_argument("--bins", "-b", nargs="+", required=True, help="Input geometry bin HDF5 files to merge")
    parser.add_argument("--no-shuffle", action="store_true", help="Skip shuffling after merge")
    args = parser.parse_args()
    logger.info("=" * 80)
    logger.info("MOSFET Dataset Merge")
    logger.info("=" * 80)
    logger.info("Output: %s", args.output)
    logger.info("Input bins: %d files", len(args.bins))
    for bin_file in args.bins:
        logger.info("  - %s", bin_file)
    logger.info("Shuffle: %s", not args.no_shuffle)
    logger.info("=" * 80)
    merge_geometry_bins(
        bin_files=args.bins,
        output_path=args.output,
        shuffle=not args.no_shuffle,
    )


if __name__ == "__main__":
    main()
