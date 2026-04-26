"""
Defines circuit topology dataclasses and SPICE netlist generation.

Provides immutable dataclasses for multi-device circuit descriptions and a deck
builder that emits valid NGSpice netlists for operating-point, transient, and
DC sweep analyses.
"""

from dataclasses import dataclass
from typing import ClassVar, Protocol

__all__ = ["Circuit", "MosfetInstance", "SpiceDevice", "VoltageSource"]


def _validate_required_net_keys(owner_name: str, nets: dict[str, str], required_keys: tuple[str, ...]) -> None:
    """
    Validates that ``nets`` contains all required port keys.

    :param owner_name: Device identifier used in error messages.
    :param nets: Mapping from port name to circuit node name.
    :param required_keys: Port keys that must be present in ``nets``.
    :raises ValueError: If one or more required keys are missing.
    """
    if missing := set(required_keys) - nets.keys():
        raise ValueError(f"Device {owner_name} missing port connections: {missing}")


class SpiceDevice(Protocol):
    """Structural type for any device that can render a SPICE instance line."""

    name: str

    def to_spice(self) -> str:
        """
        Renders the device as a single SPICE netlist line.

        :return: SPICE instance string (no trailing newline).
        """


@dataclass(frozen=True, slots=True)
class MosfetInstance:
    """
    Four-terminal MOSFET subcircuit instance.

    Port order ``(drain, gate, source, bulk)`` matches the Sky130 subcircuit
    convention and is fixed at the class level. ``nets`` must bind every port
    name in :data:`PORTS`.

    :param name: SPICE instance identifier (e.g., "XM1").
    :param model_name: PDK model identifier (e.g., "sky130_fd_pr__nfet_01v8").
    :param width_um: Channel width in microns.
    :param length_um: Channel length in microns.
    :param nets: Maps each port name to a circuit node name.
    """

    PORTS: ClassVar[tuple[str, ...]] = ("drain", "gate", "source", "bulk")
    name: str
    model_name: str
    width_um: float
    length_um: float
    nets: dict[str, str]

    def __post_init__(self) -> None:
        _validate_required_net_keys(owner_name=self.name, nets=self.nets, required_keys=self.PORTS)

    def to_spice(self) -> str:
        """
        Renders this MOSFET as a single SPICE subcircuit instance line.

        :return: SPICE instance string.
        """
        nodes = " ".join(self.nets[p] for p in self.PORTS)
        return f"{self.name} {nodes} {self.model_name} w={self.width_um} l={self.length_um}"


@dataclass(frozen=True, slots=True)
class VoltageSource:
    """
    Independent voltage source.

    :param name: SPICE source identifier (e.g., "VDD").
    :param positive_node: Positive terminal node.
    :param negative_node: Negative terminal node.
    :param dc_value: DC voltage in volts.
    :param ac_value: Small-signal AC magnitude in volts.
    :param tran_value: Transient stimulus string (e.g., "PWL(0 0 1n 1.8)").
        When non-empty, replaces ``dc_value`` for ``.tran`` analysis.
    """

    name: str
    positive_node: str
    negative_node: str
    dc_value: float = 0.0
    ac_value: float = 0.0
    tran_value: str = ""

    def to_spice_dc(self) -> str:
        """
        Renders the source for DC analyses (``.op``, ``.dc``, ``.ac``).

        :return: SPICE source line.
        """
        ac_part = f" AC {self.ac_value}" if self.ac_value != 0.0 else ""
        return f"{self.name} {self.positive_node} {self.negative_node} DC {self.dc_value}{ac_part}"

    def to_spice_tran(self) -> str:
        """
        Renders the source for transient analysis.

        Falls back to the DC form when no transient stimulus is set.

        :return: SPICE source line.
        """
        if self.tran_value:
            return f"{self.name} {self.positive_node} {self.negative_node} {self.tran_value}"
        return self.to_spice_dc()


@dataclass(frozen=True, slots=True)
class Circuit:
    """
    Multi-device circuit topology for SPICE simulation.

    Immutable representation of a complete circuit netlist. The analysis
    directive is supplied at simulation time via :meth:`build_deck`.

    :param name: Descriptive label for the circuit (deck title line).
    :param devices: Devices in the circuit (any object implementing
        :class:`SpiceDevice`).
    :param sources: Independent voltage sources providing bias and stimulus.
    :param lib_path: Absolute path to the PDK SPICE library file.
    :param lib_corner: Process corner for model selection ("tt", "ss", "ff", ...).
    """

    name: str
    devices: tuple[SpiceDevice, ...]
    sources: tuple[VoltageSource, ...]
    lib_path: str
    lib_corner: str = "tt"

    def build_deck(self, analysis: str, options: tuple[str, ...] = ()) -> str:
        """
        Builds a complete SPICE deck for the given analysis directive.

        Source rendering selects the transient form when ``analysis`` begins
        with ``.tran``, otherwise the DC form.

        :param analysis: SPICE analysis command (e.g., ".op", ".tran 1n 10u").
        :param options: ``.option`` directives appended after the analysis line.
        :return: Deck string terminated with ``.end``.
        """
        is_transient = ".tran" in analysis.lower()
        lines = [f"* {self.name}", f".lib '{self.lib_path}' {self.lib_corner}"]
        for src in self.sources:
            lines.append(src.to_spice_tran() if is_transient else src.to_spice_dc())
        for dev in self.devices:
            lines.append(dev.to_spice())
        lines.append(analysis.strip())
        for opt in options:
            lines.append(f".option {opt}")
        lines.append(".end")
        return "\n".join(lines) + "\n"
