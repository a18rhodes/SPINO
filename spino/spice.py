"""
Provides NGSpice execution harness and output parsing utilities.

Handles subprocess management, temp file lifecycle, and format-specific
parsing for both binary raw files and stdout-based marker extraction.
"""

import logging
import os
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["parse_ngspice_raw", "run_ngspice", "spice_temp_workspace", "OutputMode"]

logger = logging.getLogger(__name__)

_BINARY_MARKER = b"Binary:"
_HEADER_NUM_VARIABLES = "No. Variables:"
_HEADER_NUM_POINTS = "No. Points:"
_HEADER_VARIABLES_START = "Variables:"
_VARIABLE_NAME_INDEX = 1
_TIME_VARIABLE_NAME = "time"
_NGSPICE_DTYPE = np.float64
_NGSPICE_BIN = "ngspice"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class OutputMode(Enum):
    """NGSpice output capture mode."""

    STDOUT = auto()
    RAW_FILE = auto()


@contextmanager
def spice_temp_workspace():
    """
    Provides a temporary workspace directory for SPICE simulation files.

    :yield: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


def run_ngspice(
    deck_content: str,
    output_mode: OutputMode = OutputMode.RAW_FILE,
    spice_filename: str = "sim.spice",
    success_marker: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bool, str | dict | None]:
    """
    Executes NGSpice in batch mode with unified error handling.

    :param deck_content: Complete SPICE netlist string including .end directive.
    :param output_mode: STDOUT for text capture or RAW_FILE for binary output.
    :param spice_filename: Name for temporary SPICE file (aids debugging).
    :param success_marker: Required string in stdout for STDOUT mode validation.
    :param timeout: Maximum execution time in seconds before termination.
    :return: Tuple of (success, result) where result is stdout string, parsed dict, or None.
    """
    with spice_temp_workspace() as workspace:
        spice_file = workspace / spice_filename
        spice_file.write_text(deck_content, encoding="utf-8")
        if output_mode == OutputMode.STDOUT:
            return _run_stdout_mode(spice_file, success_marker, timeout)
        return _run_raw_file_mode(spice_file, workspace, timeout)


def _kill_process_group(process: subprocess.Popen) -> None:
    """
    Forcefully terminates process and all children via process group.

    :param process: Popen instance to terminate.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        pass


def _execute_ngspice(cmd: list[str], timeout: float) -> subprocess.CompletedProcess | None:
    """
    Executes NGSpice subprocess with unified exception handling.

    :param cmd: Command list for subprocess.run.
    :param timeout: Subprocess timeout in seconds.
    :return: CompletedProcess on success, None on failure.
    """
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        if process is not None:
            _kill_process_group(process)
        logger.exception("NGSpice subprocess timeout after %.1f seconds", timeout)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        if process is not None:
            _kill_process_group(process)
        logger.exception("NGSpice subprocess execution failed: %s", e)
        return None
    finally:
        if process is not None and process.poll() is None:
            _kill_process_group(process)


def _run_stdout_mode(spice_file: Path, success_marker: str | None, timeout: float) -> tuple[bool, str | None]:
    """
    Executes NGSpice capturing stdout for marker-based parsing.

    :param spice_file: Path to the SPICE deck.
    :param success_marker: Required string for output validation.
    :param timeout: Subprocess timeout in seconds.
    :return: Tuple of (success, stdout_content).
    """
    if (result := _execute_ngspice([_NGSPICE_BIN, "-b", str(spice_file)], timeout)) is None:
        return False, None
    if success_marker and success_marker not in result.stdout:
        logger.error("NGSpice output missing success marker. stderr: %s", result.stderr)
        return False, None
    return True, result.stdout


def _run_raw_file_mode(spice_file: Path, workspace: Path, timeout: float) -> tuple[bool, dict | None]:
    """
    Executes NGSpice writing binary raw output file and parses it.

    :param spice_file: Path to the SPICE deck.
    :param workspace: Temporary directory for output file.
    :param timeout: Subprocess timeout in seconds.
    :return: Tuple of (success, parsed_data).
    """
    raw_file = workspace / f"{spice_file.stem}.raw"
    if _execute_ngspice([_NGSPICE_BIN, "-b", "-r", str(raw_file), str(spice_file)], timeout) is None:
        return False, None
    if not raw_file.exists():
        logger.error("NGSpice did not produce expected raw file: %s", raw_file)
        return False, None
    return True, parse_ngspice_raw(raw_file)


def parse_ngspice_raw(file_path: Path) -> dict[str, NDArray[np.float64] | dict[str, NDArray[np.float64]]] | None:
    """
    Parses an NGSpice binary raw file into time-series data.

    :param file_path: Path to the .raw file generated by NGSpice.
    :return: Dictionary with 'time' array and 'nodes' dict of variable arrays, or None on error.
    """
    with open(file_path, "rb") as f:
        header_lines = _read_header(f)
        n_vars, n_points, var_names = _parse_metadata(header_lines)
        if not var_names:
            return None
        raw_bytes = f.read()
    if (data := _decode_binary_data(raw_bytes, n_vars, n_points)) is None:
        return None
    return _organize_results(data, var_names)


def _read_header(file_handle) -> list[str]:
    """
    Reads ASCII header lines until the binary data marker.

    :param file_handle: Open file handle positioned at start.
    :return: List of decoded header lines.
    """
    header_lines = []
    while True:
        if not (line := file_handle.readline()):
            break
        if line.strip() == _BINARY_MARKER:
            break
        header_lines.append(line.decode("utf-8", errors="ignore"))
    return header_lines


def _parse_metadata(header_lines: list[str]) -> tuple[int, int, list[str]]:
    """
    Extracts variable count, point count, and variable names from header.

    :param header_lines: Decoded header text lines.
    :return: Tuple of (num_vars, num_points, variable_names).
    """
    n_vars = 0
    n_points = 0
    var_names = []
    it = iter(header_lines)
    for line in it:
        if line.startswith(_HEADER_NUM_VARIABLES):
            n_vars = int(line.split(":")[-1].strip())
        elif line.startswith(_HEADER_NUM_POINTS):
            n_points = int(line.split(":")[-1].strip())
        elif line.startswith(_HEADER_VARIABLES_START):
            var_names = _extract_variable_names(it, n_vars)
    return n_vars, n_points, var_names


def _extract_variable_names(iterator, count: int) -> list[str]:
    """
    Extracts variable names from the Variables section.

    :param iterator: Iterator positioned after Variables: line.
    :param count: Number of variable definitions to read.
    :return: List of variable names.
    """
    var_names = []
    for _ in range(count):
        var_line = next(iterator)
        parts = var_line.split()
        var_names.append(parts[_VARIABLE_NAME_INDEX])
    return var_names


def _decode_binary_data(raw_bytes: bytes, n_vars: int, n_points: int) -> NDArray[np.float64] | None:
    """
    Decodes binary float64 data and reshapes into matrix form.

    :param raw_bytes: Raw binary data section.
    :param n_vars: Number of variables per time point.
    :param n_points: Number of time points.
    :return: Reshaped array (n_points, n_vars) or None on failure.
    """
    try:
        data = np.frombuffer(raw_bytes, dtype=_NGSPICE_DTYPE)
    except (ValueError, TypeError):
        return None
    if data.size != n_vars * n_points:
        return None
    return data.reshape((n_points, n_vars))


def _organize_results(
    data: NDArray[np.float64], var_names: list[str]
) -> dict[str, NDArray[np.float64] | dict[str, NDArray[np.float64]]]:
    """
    Organizes simulation data into time and node dictionaries.

    :param data: Reshaped data matrix (n_points, n_vars).
    :param var_names: Variable names corresponding to columns.
    :return: Dictionary with 'time' and 'nodes' keys.
    """
    result: dict[str, NDArray[np.float64] | dict[str, NDArray[np.float64]]] = {"time": None, "nodes": {}}
    for idx, name in enumerate(var_names):
        clean_name = name.lower()
        if clean_name == _TIME_VARIABLE_NAME:
            result["time"] = data[:, idx]
        else:
            result["nodes"][clean_name] = data[:, idx]
    return result
