"""
Pre-defined circuit topologies for simulation and composition validation.

Factory functions that construct Circuit objects for standard analog building
blocks using Sky130 device models.
"""

from pathlib import Path

from spino.circuit.netlist import Capacitor, Circuit, MosfetInstance, SpiceDevice, VoltageSource

__all__ = ["build_cs_amp_active_load"]

_SKY130_LIB_RELATIVE = "sky130A/libs.tech/ngspice/sky130.lib.spice"
_SKY130_NMOS_MODEL = "sky130_fd_pr__nfet_01v8"
_SKY130_PMOS_MODEL = "sky130_fd_pr__pfet_01v8"
_DEFAULT_PDK_ROOT = "/app/sky130_volare"


def _resolve_lib_path(pdk_root: str = _DEFAULT_PDK_ROOT) -> str:
    """
    Resolves the Sky130 SPICE library path from PDK root.

    :param pdk_root: Root directory of the PDK installation.
    :return: Absolute path string to the SPICE library file.
    :raises FileNotFoundError: If the library file does not exist at the resolved path.
    """
    lib_path = Path(pdk_root) / _SKY130_LIB_RELATIVE
    if not lib_path.exists():
        raise FileNotFoundError(f"PDK library not found: {lib_path}")
    return str(lib_path.absolute())


def build_cs_amp_active_load(
    nfet_w: float = 1.0,
    nfet_l: float = 0.18,
    pfet_w: float = 2.0,
    pfet_l: float = 0.18,
    vdd: float = 1.8,
    vin_dc: float = 0.9,
    vin_tran: str = "",
    c_load_f: float = 0.0,
    pdk_root: str = _DEFAULT_PDK_ROOT,
    corner: str = "tt",
) -> Circuit:
    """
    Builds a common-source amplifier with diode-connected PMOS active load.

    Topology::

        VDD ---|
               |
            [PFET] XM2 (diode-connected: gate = drain = out)
               |
               +----> out
               |
            [NFET] XM1 (common-source: gate = in, drain = out)
               |
              GND

    :param nfet_w: NFET channel width in microns.
    :param nfet_l: NFET channel length in microns.
    :param pfet_w: PFET channel width in microns.
    :param pfet_l: PFET channel length in microns.
    :param vdd: Supply voltage in volts.
    :param vin_dc: Input DC bias voltage in volts.
    :param vin_tran: Transient input stimulus (e.g., "PWL(0 0.9 1n 1.0)").
    :param c_load_f: Optional explicit output load capacitance in farads. When
        zero (default) only intrinsic device capacitance loads the output.
        Affects transient analyses only; DC analyses are unchanged.
    :param pdk_root: Path to the Sky130 PDK installation.
    :param corner: Process corner (e.g., "tt", "ss", "ff").
    :return: Configured Circuit object ready for simulation.
    """
    lib_path = _resolve_lib_path(pdk_root)
    nfet = MosfetInstance(
        name="XM1",
        model_name=_SKY130_NMOS_MODEL,
        width_um=nfet_w,
        length_um=nfet_l,
        nets={"drain": "out", "gate": "in", "source": "0", "bulk": "0"},
    )
    pfet = MosfetInstance(
        name="XM2",
        model_name=_SKY130_PMOS_MODEL,
        width_um=pfet_w,
        length_um=pfet_l,
        nets={"drain": "out", "gate": "out", "source": "vdd", "bulk": "vdd"},
    )
    v_supply = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=vdd)
    v_input = VoltageSource(name="Vin", positive_node="in", negative_node="0", dc_value=vin_dc, tran_value=vin_tran)
    devices: tuple[SpiceDevice, ...] = (nfet, pfet)
    if c_load_f > 0.0:
        devices = devices + (Capacitor(name="CL", positive_node="out", negative_node="0", capacitance_f=c_load_f),)
    return Circuit(
        name=f"CS Amp Active Load (NFET {nfet_w}/{nfet_l}, PFET {pfet_w}/{pfet_l})",
        devices=devices,
        sources=(v_supply, v_input),
        lib_path=lib_path,
        lib_corner=corner,
    )
