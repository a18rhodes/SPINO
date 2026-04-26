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

__all__ = [
    "Capacitor",
    "Circuit",
    "DCSweepResult",
    "MosfetInstance",
    "OperatingPoint",
    "SpiceDevice",
    "TransientResult",
    "VoltageSource",
    "build_cs_amp_active_load",
    "run_dc_sweep",
    "run_operating_point",
    "run_transient",
]
