"""
Isolated single-MOSFET circuits for NGSpice DC operating-point IV extraction.

Used by IV cache generation and cache validation so probe coordinates match a
minimal device deck built with the same library path and corner as composition.
"""

from __future__ import annotations

import re
from pathlib import Path

from spino.circuit.netlist import Circuit, MosfetInstance, VoltageSource
from spino.circuit.simulation import OperatingPoint

__all__ = [
    "build_isolated_mosfet_circuit",
    "drain_current_key_from_op",
    "isolated_mosfet_id_a",
]

_STANDALONE_INSTANCE = "XIV"
_SKY130_LIB_RELATIVE = "sky130A/libs.tech/ngspice/sky130.lib.spice"
_SKY130_NMOS_MODEL = "sky130_fd_pr__nfet_01v8"
_SKY130_PMOS_MODEL = "sky130_fd_pr__pfet_01v8"
_DEFAULT_PDK_ROOT = "/app/sky130_volare"

_DRAIN_CURRENT_KEY_RE = re.compile(r"i\(@m\.xiv\..*\[id\].*", re.IGNORECASE)


def _resolve_lib_path(pdk_root: str) -> str:
    """Return absolute path to sky130.lib.spice."""
    lib_path = Path(pdk_root) / _SKY130_LIB_RELATIVE
    if not lib_path.exists():
        raise FileNotFoundError(f"PDK library not found: {lib_path}")
    return str(lib_path.absolute())


def drain_current_key_from_op(op: OperatingPoint) -> str:
    """
    Returns the NGSpice variable name for the standalone device drain current.

    :param op: Operating point from :func:`run_operating_point`.
    :return: Key into ``op.variables`` for ``I_d``.
    :raises KeyError: When no matching drain-current key exists.
    """
    for name in op.variables:
        if _DRAIN_CURRENT_KEY_RE.match(name):
            return name
    raise KeyError(
        f"No drain current key matching {_DRAIN_CURRENT_KEY_RE.pattern} in {list(op.variables)[:12]}..."
    )


def build_isolated_mosfet_circuit(
    *,
    is_pfet: bool,
    width_um: float,
    length_um: float,
    vg: float,
    vd: float,
    vs: float,
    vb: float,
    pdk_root: str = _DEFAULT_PDK_ROOT,
    corner: str = "tt",
) -> Circuit:
    """
    Builds a four-source isolated MOSFET with DC biases on ``g,d,s,b``.

    Instance name is fixed to ``XIV`` so drain-current keys from NGSpice map
    consistently across cache code.

    :param is_pfet: When True, use sky130 PMOS model; else NMOS.
    :param width_um: Channel width in microns.
    :param length_um: Channel length in microns.
    :param vg: Gate node voltage (positive nodes vs reference ``0``).
    :param vd: Drain node voltage.
    :param vs: Source node voltage.
    :param vb: Bulk node voltage.
    :param pdk_root: PDK root containing Volare layout.
    :param corner: SPICE corner label passed to ``.lib``.
    :return: Circuit suitable for ``run_operating_point``.
    """
    lib_path = _resolve_lib_path(pdk_root)
    model = _SKY130_PMOS_MODEL if is_pfet else _SKY130_NMOS_MODEL
    mos = MosfetInstance(
        name=_STANDALONE_INSTANCE,
        model_name=model,
        width_um=width_um,
        length_um=length_um,
        nets={"drain": "nd", "gate": "ng", "source": "ns", "bulk": "nb"},
    )
    sources = (
        VoltageSource(name="Vg", positive_node="ng", negative_node="0", dc_value=vg),
        VoltageSource(name="Vd", positive_node="nd", negative_node="0", dc_value=vd),
        VoltageSource(name="Vs", positive_node="ns", negative_node="0", dc_value=vs),
        VoltageSource(name="Vb", positive_node="nb", negative_node="0", dc_value=vb),
    )
    label = "PFET IV tile" if is_pfet else "NFET IV tile"
    return Circuit(
        name=f"{label} ({width_um}/{length_um})",
        devices=(mos,),
        sources=sources,
        lib_path=lib_path,
        lib_corner=corner,
    )


def isolated_mosfet_id_a(op: OperatingPoint) -> float:
    """
    Reads drain current magnitude (amperes) from an isolated-device OP result.

    :param op: Operating point for ``build_isolated_mosfet_circuit``.
    :return: Drain current in amperes (signed as reported by NGSpice).
    """
    key = drain_current_key_from_op(op)
    return float(op.variables[key])
