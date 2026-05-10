"""
Pre-defined circuit topologies for simulation and composition validation.

Factory functions that construct Circuit objects for standard analog building
blocks using Sky130 device models.
"""

from pathlib import Path

from spino.circuit.netlist import (
    Capacitor,
    Circuit,
    MosfetInstance,
    SpiceDevice,
    VoltageSource,
)

__all__ = ["build_cmos_inverter", "build_cs_amp_active_load", "build_inverter_chain", "build_ota_5t"]

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


def build_cs_amp_active_load(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    nfet_w: float = 1.6,
    nfet_l: float = 0.4,
    pfet_w: float = 2.5,
    pfet_l: float = 0.4,
    vdd: float = 1.8,
    vin_dc: float = 0.81,
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

    Default widths, lengths, and input bias follow the 3a sweep in
    ``docs/assets/cs_amp_l040/summary.json`` (L = 0.4 um, tt corner) unless
    overridden.

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


def build_cmos_inverter(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    nfet_w: float = 0.82,
    nfet_l: float = 0.18,
    pfet_w: float = 2.05,
    pfet_l: float = 0.18,
    vdd: float = 1.8,
    vin_dc: float = 0.0,
    vin_tran: str = "",
    pdk_root: str = _DEFAULT_PDK_ROOT,
    corner: str = "tt",
) -> Circuit:
    """
    Builds a matched sky130 CMOS inverter (NFET + PFET).

    Output node ``n1``; input ``nin`` driven by ``Vin``.

    :param nfet_w: NFET width in microns.
    :param nfet_l: NFET length in microns.
    :param pfet_w: PFET width in microns.
    :param pfet_l: PFET length in microns.
    :param vdd: Supply voltage.
    :param vin_dc: DC bias on the input (when ``vin_tran`` empty).
    :param vin_tran: Transient stimulus for ``Vin``.
    :param pdk_root: PDK root path.
    :param corner: Process corner.
    :return: Single-stage inverter circuit.
    """
    return build_inverter_chain(
        n_stages=1,
        nfet_w=nfet_w,
        nfet_l=nfet_l,
        pfet_w=pfet_w,
        pfet_l=pfet_l,
        vdd=vdd,
        vin_dc=vin_dc,
        vin_tran=vin_tran,
        c_load_f=0.0,
        pdk_root=pdk_root,
        corner=corner,
    )


def build_inverter_chain(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    n_stages: int,
    nfet_w: float = 0.82,
    nfet_l: float = 0.18,
    pfet_w: float = 2.05,
    pfet_l: float = 0.18,
    vdd: float = 1.8,
    vin_dc: float = 0.0,
    vin_tran: str = "",
    c_load_f: float = 0.0,
    pdk_root: str = _DEFAULT_PDK_ROOT,
    corner: str = "tt",
) -> Circuit:
    """
    Builds a chain of ``n_stages`` matched CMOS inverters.

    Node ``nin`` is the primary input; ``n{k}`` is the output of stage ``k``.
    The final output is ``n{n_stages}``.

    :param n_stages: Number of inverters (>= 1).
    :param nfet_w: NFET width per stage (µm).
    :param nfet_l: NFET length (µm).
    :param pfet_w: PFET width per stage (µm).
    :param pfet_l: PFET length (µm).
    :param vdd: Supply voltage.
    :param vin_dc: Input DC voltage at ``nin``.
    :param vin_tran: Optional transient waveform for ``Vin``.
    :param c_load_f: Optional linear load on the **final** output to ground (F).
    :param pdk_root: Sky130 PDK root.
    :param corner: Spice corner label.
    :return: Completed :class:`~spino.circuit.netlist.Circuit`.
    """
    if n_stages < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    lib_path = _resolve_lib_path(pdk_root)
    devices_list: list[SpiceDevice] = []
    for k in range(1, n_stages + 1):
        gate_net = "nin" if k == 1 else f"n{k - 1}"
        drain_net = f"n{k}"
        devices_list.append(
            MosfetInstance(
                name=f"XN{k}",
                model_name=_SKY130_NMOS_MODEL,
                width_um=nfet_w,
                length_um=nfet_l,
                nets={"drain": drain_net, "gate": gate_net, "source": "0", "bulk": "0"},
            ),
        )
        devices_list.append(
            MosfetInstance(
                name=f"XP{k}",
                model_name=_SKY130_PMOS_MODEL,
                width_um=pfet_w,
                length_um=pfet_l,
                nets={"drain": drain_net, "gate": gate_net, "source": "vdd", "bulk": "vdd"},
            ),
        )
    if c_load_f > 0.0:
        devices_list.append(
            Capacitor(name="CL", positive_node=f"n{n_stages}", negative_node="0", capacitance_f=c_load_f)
        )
    v_supply = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=vdd)
    v_input = VoltageSource(name="Vin", positive_node="nin", negative_node="0", dc_value=vin_dc, tran_value=vin_tran)
    return Circuit(
        name=f"Inverter chain x{n_stages} (Wn={nfet_w} Ln={nfet_l} Wp={pfet_w} Lp={pfet_l})",
        devices=tuple(devices_list),
        sources=(v_supply, v_input),
        lib_path=lib_path,
        lib_corner=corner,
    )


def build_ota_5t(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    *,
    diff_w_um: float,
    diff_l_um: float,
    mirror_w_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    vdd: float = 1.8,
    vbias_v: float = 1.2,
    vcm_v: float = 0.9,
    vinp_tran: str = "",
    vinn_tran: str = "",
    c_load_f: float = 0.0,
    pdk_root: str = _DEFAULT_PDK_ROOT,
    corner: str = "tt",
) -> Circuit:
    """
    Builds a five-transistor OTA with NFET differential pair and PFET current-mirror load.

    Topology::

                         VDD (1.8 V)
                  ┌────┐    ┌────┐
                  │ M3 │    │ M4 │   (PFET mirror; M3 diode-connected)
                  └─┬──┘    └─┬──┘
           n_left ──┤        ├── n_out  ← single-ended output
                  ┌─┴──┐    ┌─┴──┐
       Vinp ──── G│ M1 │    │ M2 │G ──── Vinn
                  └─┬──┘    └─┬──┘
                    └────┬────┘
                         ├── n_tail
                       ┌─┴──┐
                Vbias─G│ M5 │       (NFET tail current source)
                       └─┴──┘
                        GND

    Internal nodes (KCL unknowns for FNO composition): ``n_tail``, ``n_left``, ``n_out``.

    :param diff_w_um: Differential-pair MOSFET width (M1, M2) in microns.
    :param diff_l_um: Differential-pair channel length in microns.
    :param mirror_w_um: Current-mirror MOSFET width (M3, M4) in microns.
    :param mirror_l_um: Current-mirror channel length in microns.
    :param tail_w_um: Tail current-source width (M5) in microns.
    :param tail_l_um: Tail current-source channel length in microns.
    :param vdd: Supply voltage in volts (default 1.8 V).
    :param vbias_v: DC gate bias for the tail current source (M5).
    :param vcm_v: Input common-mode DC voltage for both Vinp and Vinn.
    :param vinp_tran: Transient stimulus string for Vinp (e.g., "PWL(...)").
        When empty the source holds ``vcm_v`` throughout.
    :param vinn_tran: Transient stimulus string for Vinn.
    :param c_load_f: Optional linear load capacitance at ``n_out`` to ground (F).
    :param pdk_root: Sky130 PDK root directory.
    :param corner: Process corner (e.g., "tt").
    :return: Configured Circuit ready for SPICE simulation.
    """
    lib_path = _resolve_lib_path(pdk_root)
    m1 = MosfetInstance(
        name="XM1",
        model_name=_SKY130_NMOS_MODEL,
        width_um=diff_w_um,
        length_um=diff_l_um,
        nets={"drain": "n_left", "gate": "vinp", "source": "n_tail", "bulk": "0"},
    )
    m2 = MosfetInstance(
        name="XM2",
        model_name=_SKY130_NMOS_MODEL,
        width_um=diff_w_um,
        length_um=diff_l_um,
        nets={"drain": "n_out", "gate": "vinn", "source": "n_tail", "bulk": "0"},
    )
    m3 = MosfetInstance(
        name="XM3",
        model_name=_SKY130_PMOS_MODEL,
        width_um=mirror_w_um,
        length_um=mirror_l_um,
        nets={"drain": "n_left", "gate": "n_left", "source": "vdd", "bulk": "vdd"},
    )
    m4 = MosfetInstance(
        name="XM4",
        model_name=_SKY130_PMOS_MODEL,
        width_um=mirror_w_um,
        length_um=mirror_l_um,
        nets={"drain": "n_out", "gate": "n_left", "source": "vdd", "bulk": "vdd"},
    )
    m5 = MosfetInstance(
        name="XM5",
        model_name=_SKY130_NMOS_MODEL,
        width_um=tail_w_um,
        length_um=tail_l_um,
        nets={"drain": "n_tail", "gate": "vbias", "source": "0", "bulk": "0"},
    )
    v_supply = VoltageSource(name="VDD", positive_node="vdd", negative_node="0", dc_value=vdd)
    v_bias = VoltageSource(name="Vbias", positive_node="vbias", negative_node="0", dc_value=vbias_v)
    v_inp = VoltageSource(name="Vinp", positive_node="vinp", negative_node="0", dc_value=vcm_v, tran_value=vinp_tran)
    v_inn = VoltageSource(name="Vinn", positive_node="vinn", negative_node="0", dc_value=vcm_v, tran_value=vinn_tran)
    devices: tuple[SpiceDevice, ...] = (m1, m2, m3, m4, m5)
    if c_load_f > 0.0:
        devices = devices + (Capacitor(name="CL", positive_node="n_out", negative_node="0", capacitance_f=c_load_f),)
    return Circuit(
        name=(
            f"5T OTA (Wdiff={diff_w_um}/{diff_l_um} µm, Wmirror={mirror_w_um}/{mirror_l_um} µm,"
            f" Wtail={tail_w_um}/{tail_l_um} µm)"
        ),
        devices=devices,
        sources=(v_supply, v_bias, v_inp, v_inn),
        lib_path=lib_path,
        lib_corner=corner,
    )
