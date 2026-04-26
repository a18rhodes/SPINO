"""
Multi-device circuit simulation via NGSpice.

Wraps the core spice.run_ngspice harness to execute circuit-level analyses
(DC operating point, transient, DC sweep) and return structured results.
Does not modify any single-device simulation infrastructure.
"""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spino.circuit.netlist import Circuit
from spino.spice import OutputMode, run_ngspice

__all__ = [
    "DCSweepResult",
    "OperatingPoint",
    "TransientResult",
    "run_dc_sweep",
    "run_operating_point",
    "run_transient",
]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 120.0
_OP_OPTIONS = ("savecurrents",)
_TRAN_OPTIONS = ("savecurrents", "strict_errorhandling=0")
_DC_OPTIONS = ("savecurrents",)


@dataclass(frozen=True)
class OperatingPoint:
    """
    DC operating point solution.

    All variables are scalar values representing the steady-state
    DC bias condition of the circuit.

    :param variables: Maps NGSpice variable name to its DC value (V or A).
    """

    variables: dict[str, float]


@dataclass(frozen=True)
class TransientResult:
    """
    Time-domain simulation result.

    Contains the full time-series output of a .tran analysis including
    all saved node voltages and branch currents.

    :param time: Simulation time points in seconds.
    :param variables: Maps NGSpice variable name to time-series array.
    """

    time: NDArray[np.float64]
    variables: dict[str, NDArray[np.float64]]


@dataclass(frozen=True)
class DCSweepResult:
    """
    DC sweep analysis result.

    Contains the swept parameter values and all dependent outputs.

    :param sweep_param: Name of the swept variable as reported by NGSpice.
    :param sweep_values: Array of swept parameter values.
    :param variables: Maps variable name to output array at each sweep point.
    """

    sweep_param: str
    sweep_values: NDArray[np.float64]
    variables: dict[str, NDArray[np.float64]]


def run_operating_point(circuit: Circuit, timeout: float = _DEFAULT_TIMEOUT) -> OperatingPoint | None:
    """
    Runs DC operating point analysis on the circuit.

    :param circuit: Circuit topology to simulate.
    :param timeout: NGSpice subprocess timeout in seconds.
    :return: Operating point with all node voltages and branch currents, or None on failure.
    """
    deck = circuit.build_deck(".op", options=_OP_OPTIONS)
    success, parsed = run_ngspice(
        deck, output_mode=OutputMode.RAW_FILE, spice_filename="circuit_op.spice", timeout=timeout
    )
    if not success or parsed is None:
        logger.error("Operating point analysis failed for circuit: %s", circuit.name)
        return None
    return OperatingPoint(variables={name: float(data[0]) for name, data in parsed["nodes"].items()})


def run_transient(
    circuit: Circuit,
    t_step: float,
    t_end: float,
    timeout: float = _DEFAULT_TIMEOUT,
) -> TransientResult | None:
    """
    Runs transient analysis on the circuit.

    :param circuit: Circuit topology to simulate.
    :param t_step: Maximum simulation timestep in seconds.
    :param t_end: Simulation end time in seconds.
    :param timeout: NGSpice subprocess timeout in seconds.
    :return: Time-series result with all node voltages and currents, or None on failure.
    """
    deck = circuit.build_deck(f".tran {t_step} {t_end}", options=_TRAN_OPTIONS)
    success, parsed = run_ngspice(
        deck, output_mode=OutputMode.RAW_FILE, spice_filename="circuit_tran.spice", timeout=timeout
    )
    if not success or parsed is None:
        logger.error("Transient analysis failed for circuit: %s", circuit.name)
        return None
    if parsed["time"] is None:
        logger.error("Transient result missing time vector for circuit: %s", circuit.name)
        return None
    return TransientResult(time=parsed["time"], variables=dict(parsed["nodes"]))


def run_dc_sweep(
    circuit: Circuit,
    source_name: str,
    start: float,
    stop: float,
    step: float,
    timeout: float = _DEFAULT_TIMEOUT,
) -> DCSweepResult | None:
    """
    Runs DC sweep analysis on the circuit.

    Sweeps the specified voltage source across the given range and records
    all node voltages and branch currents at each sweep point.

    :param circuit: Circuit topology to simulate.
    :param source_name: Name of the voltage source to sweep (e.g., "Vin").
    :param start: Sweep start voltage in volts.
    :param stop: Sweep stop voltage in volts.
    :param step: Sweep step size in volts.
    :param timeout: NGSpice subprocess timeout in seconds.
    :return: Sweep result with VTC data, or None on failure.
    """
    deck = circuit.build_deck(f".dc {source_name} {start} {stop} {step}", options=_DC_OPTIONS)
    success, parsed = run_ngspice(
        deck, output_mode=OutputMode.RAW_FILE, spice_filename="circuit_dc.spice", timeout=timeout
    )
    if not success or parsed is None:
        logger.error("DC sweep analysis failed for circuit: %s", circuit.name)
        return None
    nodes = dict(parsed["nodes"])
    if (sweep_key := _find_sweep_variable(nodes, source_name)) is not None:
        sweep_values = nodes.pop(sweep_key)
        return DCSweepResult(sweep_param=sweep_key, sweep_values=sweep_values, variables=nodes)
    if parsed["time"] is not None:
        return DCSweepResult(sweep_param=source_name.lower(), sweep_values=parsed["time"], variables=nodes)
    logger.error("Could not identify sweep variable in DC results for circuit: %s", circuit.name)
    return None


def _find_sweep_variable(nodes: dict[str, NDArray], source_name: str) -> str | None:
    """
    Identifies the sweep variable key in parsed DC analysis results.

    NGSpice labels the independent sweep variable differently across versions.
    This function checks known naming conventions in priority order.

    :param nodes: All parsed variable arrays from raw file.
    :param source_name: Name of the voltage source being swept.
    :return: Matching key from nodes dict, or None if not found.
    """
    source_lower = source_name.lower()
    for candidate in ("v-sweep", "v(v-sweep)", source_lower, f"v({source_lower})"):
        if candidate in nodes:
            return candidate
    return None
