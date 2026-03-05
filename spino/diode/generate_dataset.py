"""
CLI tool for generating offline dimensionless diode training datasets.

Usage::

    python -m spino.diode.generate_dataset --output /app/datasets/diode_10k.h5 --samples 10000
    python -m spino.diode.generate_dataset -o /app/datasets/diode_5k.h5 -n 5000 --workers 16
"""

import logging
import sys

import click

from spino.diode.gen_data import generate_offline_dataset

__all__ = ["main"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--output", "-o", required=True, type=click.Path(), help="Output HDF5 file path.")
@click.option("--samples", "-n", type=int, default=10000, show_default=True, help="Number of samples to generate.")
@click.option("--t-steps", type=int, default=2048, show_default=True, help="Time steps per sample.")
@click.option("--workers", "-j", type=int, default=8, show_default=True, help="Number of parallel workers.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing file.")
def main(output: str, samples: int, t_steps: int, workers: int, overwrite: bool) -> None:
    """
    Generates a pre-computed dimensionless diode dataset in HDF5 format.

    Each sample uses a variable T_end derived from tau=RC and a log-uniform
    window ratio, producing resolution-invariant training data with stiffness
    ratio lambda as an explicit input channel.
    """
    logger.info("=" * 80)
    logger.info("Diode Dimensionless Dataset Generation")
    logger.info("=" * 80)
    logger.info("Output:    %s", output)
    logger.info("Samples:   %d", samples)
    logger.info("T-steps:   %d", t_steps)
    logger.info("Workers:   %d", workers)
    logger.info("Overwrite: %s", overwrite)
    generate_offline_dataset(
        output_path=output,
        num_samples=samples,
        t_steps=t_steps,
        num_workers=workers,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
