import logging
import subprocess
import tempfile
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class BSIMParser:
    """
    Wraps the NGSpice executable via subprocess to inspect PDK models.
    Uses the Official Volare PDK structure.
    """

    def __init__(self, pdk_root: str):
        self.pdk_root = Path(pdk_root)

        self.lib_path = self.pdk_root / "sky130A" / "libs.tech" / "ngspice" / "sky130.lib.spice"

        if not self.lib_path.exists():
            local_check = Path("sky130_volare/sky130A/libs.tech/ngspice/sky130.lib.spice")
            if local_check.exists():
                self.lib_path = local_check.absolute()

    def inspect_model(self, model_name: str = "sky130_fd_pr__nfet_01v8") -> dict[str, str]:
        """
        Creates a temporary SPICE deck, runs NGSpice, and parses the 'show' output
        to extract physics parameters.
        """
        if not self.lib_path.exists():
            raise FileNotFoundError(f"PDK not found: {self.lib_path}")

        logger.info("Inspecting model via NGSpice Subprocess: %s", model_name)

        spice_content = f"""
* BSIM Parameter Extraction Deck
.lib '{self.lib_path.absolute()}' tt

.option width=4096

X1 0 0 0 0 {model_name} w=1.0 l=0.15

.op
.print op v(0)

.control
  set width=4096
  op

  echo "__START_LISTING__"
  listing e
  echo "__END_LISTING__"

  quit
.endc
.end
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            spice_file = tmp_dir_path / "extract.spice"
            spice_file.write_text(spice_content, encoding="utf-8")

            try:
                result = subprocess.run(["ngspice", "-b", str(spice_file)], capture_output=True, text=True, check=False)

                output = result.stdout

                if "__START_LISTING__" not in output:
                    logger.error("NGSpice failed (code %d)", result.returncode)
                    logger.error("Stderr: %s", result.stderr)
                    logger.debug("Stdout: %s", output)
                    return {}

            except Exception as e:  # pylint: disable=broad-except
                logger.exception("Subprocess execution failed: %s", e)
                return {}

        params = {}
        listing_params = self._parse_listing(output, "__START_LISTING__", "__END_LISTING__")
        params.update(listing_params)

        required_physics = ["vth0", "tox", "toxe", "u0"]
        if not any(k in params for k in required_physics):
            logger.warning("Physics parameters (vth0, tox, u0) not found in listing.")
            logger.debug("NGSpice Output:\n%s", output)

        params = {k: v for k, v in params.items() if v}

        logger.info("Extracted %d parameters.", len(params))
        return params

    def _parse_listing(self, output: str, marker_start: str, marker_end: str) -> dict[str, str]:
        """
        Parses 'listing e' output.
        Looks for lines starting with '.model' and extracted instances 'm.x1...'.
        """
        pattern = re.compile(f"{marker_start}(.*?){marker_end}", re.DOTALL)
        match = pattern.search(output)
        section_params = {}

        if match:
            block = match.group(1)
            line_strip_re = re.compile(r"^\s*\d+\s*:\s*(.*)")

            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue

                m = line_strip_re.match(line)
                content = m.group(1) if m else line
                content_lower = content.lower()

                if content_lower.startswith(".model"):
                    pairs = re.findall(r"(\w+)\s*=\s*([^\s]+)", content)
                    for k, v in pairs:
                        section_params[k.lower()] = v

                elif content_lower.startswith("m.x1"):
                    pairs = re.findall(r"(\w+)\s*=\s*([^\s]+)", content)
                    for k, v in pairs:
                        section_params[k.lower()] = v

        return section_params
