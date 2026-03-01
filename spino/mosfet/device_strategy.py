"""
Defines device-specific strategies for MOSFET simulation and data generation.

Strategies encapsulate model names, PDK paths, voltage biasing schemes, and
polarity corrections for different transistor types (NMOS, PMOS) and process nodes.
"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

__all__ = ["DeviceStrategy", "Sky130NMOSStrategy", "Sky130PMOSStrategy"]


class DeviceStrategy(ABC):
    """
    Abstract base class for device-specific MOSFET simulation strategies.

    Concrete strategies are automatically registered via __init_subclass__ and
    can be instantiated by name using the create() factory method.
    """

    _registry: dict[str, type["DeviceStrategy"]] = {}

    def __init_subclass__(cls, strategy_name: str = None, **kwargs):
        """
        Registers concrete strategy classes by name during class definition.

        :param strategy_name: Unique identifier for registry lookup.
        """
        super().__init_subclass__(**kwargs)
        if strategy_name:
            cls._registry[strategy_name] = cls
            cls.strategy_name = strategy_name

    @classmethod
    def create(cls, name: str, **kwargs) -> "DeviceStrategy":
        """
        Factory method to instantiate strategies by registered name.

        :param name: Strategy identifier (e.g., "sky130_nmos").
        :param kwargs: Configuration parameters passed to strategy constructor.
        :return: Configured strategy instance.
        :raises ValueError: If strategy name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """
        Returns list of all registered strategy names.

        :return: List of available strategy identifiers.
        """
        return list(cls._registry.keys())

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the SPICE model name for this device type.

        :return: PDK model identifier (e.g., "sky130_fd_pr__nfet_01v8").
        """

    @property
    @abstractmethod
    def pdk_root(self) -> str:
        """
        Returns the filesystem path to the PDK root directory.

        :return: Absolute path to PDK installation.
        """

    @abstractmethod
    def sample_terminal_voltages(
        self, pwl_generator: Callable[[float, float], tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Generates device-appropriate voltage waveforms for all terminals.

        :param pwl_generator: Callback to generate PWL waveforms with (min, max) ranges.
        :return: Dictionary mapping terminal names to (times, voltages) tuples.
        """

    @property
    @abstractmethod
    def current_polarity_multiplier(self) -> float:
        """
        Returns multiplier to correct SPICE current sign convention.

        :return: Typically -1.0 for NMOS, may vary for PMOS.
        """


class Sky130NMOSStrategy(DeviceStrategy, strategy_name="sky130_nmos"):
    """
    Strategy for SkyWater 130nm NMOS transistors.

    Biasing: Gate and drain swing positive (0-1.8V), source near ground,
    bulk reverse-biased to substrate. Creates Vgs > 0 for N-channel conduction.
    """

    def __init__(
        self,
        gate_range: tuple[float, float] = (0.0, 1.8),
        drain_range: tuple[float, float] = (0.0, 1.8),
        source_range: tuple[float, float] = (0.0, 0.1),
        bulk_range: tuple[float, float] = (-0.5, 0.0),
    ):
        """
        Initializes NMOS strategy with configurable voltage ranges.

        :param gate_range: Gate voltage bounds (V).
        :param drain_range: Drain voltage bounds (V).
        :param source_range: Source voltage bounds (V).
        :param bulk_range: Bulk voltage bounds (V).
        """
        self.gate_range = gate_range
        self.drain_range = drain_range
        self.source_range = source_range
        self.bulk_range = bulk_range

    @property
    def model_name(self) -> str:
        """
        Returns Sky130 NMOS model identifier.

        :return: PDK model name for 1.8V NFET.
        """
        return "sky130_fd_pr__nfet_01v8"

    @property
    def pdk_root(self) -> str:
        """
        Returns Sky130 PDK installation path.

        :return: Absolute path to Volare-managed PDK.
        """
        return "/app/sky130_volare"

    def sample_terminal_voltages(
        self, pwl_generator: Callable[[float, float], tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Generates NMOS-specific voltage waveforms.

        :param pwl_generator: PWL waveform generator function.
        :return: Terminal voltages creating positive Vgs for conduction.
        """
        return {
            "gate": pwl_generator(*self.gate_range),
            "drain": pwl_generator(*self.drain_range),
            "source": pwl_generator(*self.source_range),
            "bulk": pwl_generator(*self.bulk_range),
        }

    @property
    def current_polarity_multiplier(self) -> float:
        """
        Returns NMOS current polarity correction factor.

        :return: -1.0 to convert SPICE branch current to drain current convention.
        """
        return -1.0


class Sky130PMOSStrategy(DeviceStrategy, strategy_name="sky130_pmos"):
    """
    Strategy for SkyWater 130nm PMOS transistors.

    Biasing: Source near Vdd (1.7-1.8V), gate/drain swing full range (0-1.8V),
    bulk tied near Vdd. Creates Vgs < 0 for P-channel conduction when Vg < Vs.
    """

    def __init__(
        self,
        gate_range: tuple[float, float] = (0.0, 1.8),
        drain_range: tuple[float, float] = (0.0, 1.8),
        source_range: tuple[float, float] = (1.7, 1.8),
        bulk_range: tuple[float, float] = (1.7, 1.8),
    ):
        """
        Initializes PMOS strategy with configurable voltage ranges.

        :param gate_range: Gate voltage bounds (V).
        :param drain_range: Drain voltage bounds (V).
        :param source_range: Source voltage bounds (V, kept near Vdd).
        :param bulk_range: Bulk voltage bounds (V, typically tied to Vdd).
        """
        self.gate_range = gate_range
        self.drain_range = drain_range
        self.source_range = source_range
        self.bulk_range = bulk_range

    @property
    def model_name(self) -> str:
        """
        Returns Sky130 PMOS model identifier.

        :return: PDK model name for 1.8V PFET.
        """
        return "sky130_fd_pr__pfet_01v8"

    @property
    def pdk_root(self) -> str:
        """
        Returns Sky130 PDK installation path.

        :return: Absolute path to Volare-managed PDK.
        """
        return "/app/sky130_volare"

    def sample_terminal_voltages(
        self, pwl_generator: Callable[[float, float], tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Generates PMOS-specific voltage waveforms.

        :param pwl_generator: PWL waveform generator function.
        :return: Terminal voltages creating negative Vgs for conduction.
        """
        return {
            "gate": pwl_generator(*self.gate_range),
            "drain": pwl_generator(*self.drain_range),
            "source": pwl_generator(*self.source_range),
            "bulk": pwl_generator(*self.bulk_range),
        }

    @property
    def current_polarity_multiplier(self) -> float:
        """
        Returns PMOS current polarity correction factor.

        :return: -1.0 (same as NMOS due to SPICE branch current convention).
        """
        return -1.0
