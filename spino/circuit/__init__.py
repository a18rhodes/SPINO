"""
Multi-device circuit simulation and composition.

Provides circuit topology representation, SPICE netlist generation,
and NGSpice simulation runners for ground truth generation of
multi-device analog circuits. Built on top of the core spice.py
harness without modifying any single-device infrastructure.
"""

from spino.circuit.netlist import Capacitor, Circuit, MosfetInstance, SpiceDevice, VoltageSource
from spino.circuit.simulation import (
    DCSweepResult,
    OperatingPoint,
    TransientResult,
    run_dc_sweep,
    run_operating_point,
    run_transient,
)
from spino.circuit.topologies import build_cs_amp_active_load
from spino.circuit.tuning import (
    DesignPoint,
    Metrics,
    SelectionRule,
    SweepResult,
    extract_peak_gain,
    extract_settling_time,
    select_design_point,
    simulate_design_point,
    sweep_design_space,
)

__all__ = [
    "Capacitor",
    "Circuit",
    "DCSweepResult",
    "DesignPoint",
    "Metrics",
    "MosfetInstance",
    "OperatingPoint",
    "SelectionRule",
    "SpiceDevice",
    "SweepResult",
    "TransientResult",
    "VoltageSource",
    "build_cs_amp_active_load",
    "extract_peak_gain",
    "extract_settling_time",
    "run_dc_sweep",
    "run_operating_point",
    "run_transient",
    "select_design_point",
    "simulate_design_point",
    "sweep_design_space",
]
