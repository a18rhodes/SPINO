"""
Wraps NGSpice to extract raw BSIM parameters from PDK models.

This module provides a low-level interface to NGSpice, executing inspection
decks and returning the raw, unfiltered parameter dictionaries. It does strictly
no filtering or validation of the extracted keys.
"""

import logging
import re
from pathlib import Path

from spino.spice import OutputMode, run_ngspice

__all__ = ["BSIMParser"]

logger = logging.getLogger(__name__)

_LISTING_START_MARKER = "__START_LISTING__"
_LISTING_END_MARKER = "__END_LISTING__"
_LISTING_BLOCK_RE = re.compile(rf"{_LISTING_START_MARKER}(.*?){_LISTING_END_MARKER}", re.DOTALL)
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*:\s*(.*)")
_KEY_VALUE_PAIR_RE = re.compile(r"(\w+)\s*=\s*([^\s]+)")
_SKY130_LIB_RELATIVE_PATH = Path("sky130A/libs.tech/ngspice/sky130.lib.spice")
_LOCAL_FALLBACK_PATH = Path("sky130_volare") / _SKY130_LIB_RELATIVE_PATH

_SPICE_DECK_TEMPLATE = """\
* BSIM Parameter Extraction Deck
.lib '{lib_path}' tt
.option width=4096
* Instance to inspect
X1 d g s b {model_name} {instance_params}
* Dummy voltage sources to establish connectivity
Vd d 0 0
Vg g 0 0
Vs s 0 0
Vb b 0 0
.op
.control
  set width=4096
  op
  echo "{start_marker}"
  * 'listing e' expands the model and instance parameters
  listing e
  echo "{end_marker}"
  quit
.endc
.end
"""


class BSIMParser:
    """
    Extracts BSIM transistor parameters from Sky130 PDK via NGSpice.

    :param pdk_root: Filesystem path to the Volare PDK installation root.
    """

    def __init__(self, pdk_root: str) -> None:
        """
        Initializes the parser and resolves PDK library path.

        :param pdk_root: Root directory of the Sky130 PDK installation.
        """
        self.pdk_root = Path(pdk_root)
        self.lib_path = self.pdk_root / _SKY130_LIB_RELATIVE_PATH
        if not self.lib_path.exists():
            local_check = Path.cwd() / _LOCAL_FALLBACK_PATH
            if local_check.exists():
                self.lib_path = local_check.absolute()

    def inspect_model(self, model_name: str = "sky130_fd_pr__nfet_01v8", **kwargs) -> dict[str, str]:
        """
        Extracts BSIM parameters for specified transistor model and geometry.

        :param model_name: PDK model identifier (e.g., sky130_fd_pr__nfet_01v8).
        :param kwargs: Device instance parameters (e.g., w="1.0", l="0.15").
        :return: Dictionary mapping parameter names to string values.
        :raises FileNotFoundError: If PDK library file is not accessible.
        """
        if not self.lib_path.exists():
            raise FileNotFoundError(f"PDK library not found at: {self.lib_path}")
        if not kwargs:
            kwargs = {"w": "1.0", "l": "0.15"}

        instance_params_str = " ".join([f"{k}={v}" for k, v in kwargs.items()])

        spice_content = _SPICE_DECK_TEMPLATE.format(
            lib_path=self.lib_path.absolute(),
            model_name=model_name,
            instance_params=instance_params_str,
            start_marker=_LISTING_START_MARKER,
            end_marker=_LISTING_END_MARKER,
        )
        success, output = run_ngspice(
            spice_content,
            output_mode=OutputMode.STDOUT,
            spice_filename="extract.spice",
            success_marker=_LISTING_START_MARKER,
            timeout=30.0,
        )
        if not success or output is None:
            return {}
        return self._parse_listing(output)

    def _parse_listing(self, output: str) -> dict[str, str]:
        """
        Extracts parameter key-value pairs from NGSpice listing output.

        :param output: Raw NGSpice stdout containing listing markers.
        :return: Dictionary of lowercase parameter names to values.
        """
        match = _LISTING_BLOCK_RE.search(output)
        if not match:
            return {}
        block_content = match.group(1)
        params = {}
        for line in block_content.splitlines():
            clean_line = self._clean_line(line.strip())
            if not clean_line:
                continue
            pairs = self._extract_pairs(clean_line)
            params.update(pairs)
        return params

    def _clean_line(self, line: str) -> str:
        """
        Strips NGSpice line number prefixes from listing output.

        :param line: Single line from listing block.
        :return: Line content without numeric prefix.
        """
        if m := _LINE_NUMBER_PREFIX_RE.match(line):
            return m.group(1)
        return line

    def _extract_pairs(self, content: str) -> dict[str, str]:
        """
        Extracts all key=value parameter pairs from a line.

        :param content: Preprocessed listing line content.
        :return: Dictionary of lowercase keys to values.
        """
        return {k.lower(): v for k, v in _KEY_VALUE_PAIR_RE.findall(content)}
