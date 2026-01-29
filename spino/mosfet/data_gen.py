import logging
import sys
from spino.mosfet.bsim_parser import BSIMParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
# Constants
PDK_ROOT = "/app/sky130_volare"


def debug_parser():
    """
    Test routine to verify BSIMParser correctly reads the Sky130 models via PySpice.
    """
    logger.info("Initializing BSIMParser (PySpice Engine)...")
    parser = BSIMParser(pdk_root=PDK_ROOT)

    target_model = "sky130_fd_pr__nfet_01v8"

    try:
        # The new parser exposes inspect_model() instead of load()
        params = parser.inspect_model(model_name=target_model)
    except Exception as e:
        logger.error("Parser failed: %s", e, exc_info=True)
        return

    if params:
        logger.info("\n--- Parameters for %s ---", target_model)

        # Print a few key BSIM parameters if they exist
        keys_to_check = ["vth0", "tox", "u0", "l", "w"]
        for k in keys_to_check:
            val = params.get(k, "Not Found")
            logger.info("  %s: %s", k, val)

        logger.info("  Total params extracted: %d", len(params))
    else:
        logger.warning("No parameters extracted for %s.", target_model)


if __name__ == "__main__":
    debug_parser()
