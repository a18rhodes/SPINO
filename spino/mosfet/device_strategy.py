"""
Defines device-specific strategies for MOSFET simulation and data generation.

Strategies encapsulate model names, PDK paths, voltage biasing schemes, polarity
corrections, and evaluation sweep configurations for different transistor types
(NMOS, PMOS) and process nodes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np

__all__ = ["DeviceStrategy", "EvalConfig", "Sky130NMOSStrategy", "Sky130PMOSStrategy", "WaveformConfig"]


@dataclass(frozen=True, slots=True)
class WaveformConfig:
    """
    Device-specific configuration for focused waveform generation.

    Encapsulates threshold voltage location and subthreshold region boundaries
    in absolute gate voltage space. These differ between NMOS and PMOS because
    the relationship between Vg and |Vgs| inverts with device polarity.

    For NMOS: subthreshold => small Vg (near 0V), far from Vs=0 threshold.
    For PMOS: subthreshold => large Vg (near VDD), far from Vs=VDD threshold.

    :param vth_nominal: Nominal threshold voltage in absolute Vg space (V).
        NMOS: ~0.50V (Vth0). PMOS: ~1.30V (VDD - |Vth0|).
    :param vth_spread: Random spread around Vth for process corner coverage (V).
    :param deep_subth_vg_range: Absolute Vg bounds for deep subthreshold waveforms (V).
    :param trans_subth_vg_range: Absolute Vg bounds for transitional subthreshold waveforms (V).
    """

    vth_nominal: float
    vth_spread: float
    deep_subth_vg_range: tuple[float, float]
    trans_subth_vg_range: tuple[float, float]


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """
    Device-specific configuration for SPICE-based I-V evaluation sweeps.

    Encapsulates all numerical constants that differ between NMOS and PMOS
    evaluation: sweep directions, bias points, subthreshold definitions,
    and random waveform bounds. Eliminates hardcoded assumptions about
    device polarity from the evaluation pipeline.

    :param strategy_name: Registry key for InfiniteSpiceMosfetDataset.
    :param vdd: Supply voltage (V). Used for plot range annotations.
    :param vs_bias: Fixed source bias for all evaluation sweeps (V).
    :param vb_bias: Fixed bulk bias for all evaluation sweeps (V).
    :param transfer_vg_start: Gate voltage at start of transfer sweep (V).
    :param transfer_vg_stop: Gate voltage at end of transfer sweep (V).
    :param transfer_vd_bias: Fixed drain voltage during transfer sweep (V).
    :param output_vd_start: Drain voltage at start of output sweep (V).
    :param output_vd_stop: Drain voltage at end of output sweep (V).
    :param output_vg_drive: Fixed gate voltage during output sweep (V).
    :param subth_vg_threshold: Gate voltage threshold defining subthreshold region (V).
    :param subth_below: If True, subthreshold is Vg < threshold (NMOS).
        If False, subthreshold is Vg > threshold (PMOS).
    :param random_vg_range: Min/max gate voltage for random PWL waveforms (V).
    :param random_vd_range: Min/max drain voltage for random PWL waveforms (V).
    """

    strategy_name: str
    vdd: float
    vs_bias: float
    vb_bias: float
    transfer_vg_start: float
    transfer_vg_stop: float
    transfer_vd_bias: float
    output_vd_start: float
    output_vd_stop: float
    output_vg_drive: float
    subth_vg_threshold: float
    subth_below: bool
    random_vg_range: tuple[float, float]
    random_vd_range: tuple[float, float]


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

    @property
    @abstractmethod
    def eval_config(self) -> EvalConfig:
        """
        Returns device-specific evaluation sweep configuration.

        :return: Frozen configuration for SPICE-based I-V validation sweeps.
        """

    @property
    @abstractmethod
    def waveform_config(self) -> WaveformConfig:
        """
        Returns device-specific focused waveform generation configuration.

        :return: Frozen configuration for threshold-aware waveform modes.
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

        SPICE branch convention: i(Vd) is positive when current flows out of node d
        through the voltage source to ground. For NMOS in saturation, conventional
        current flows drain-to-source (out of drain), so i(Vd) is negative when on.
        Negation required to produce the standard positive Id convention.

        :return: -1.0 (NMOS i(Vd) is negative when device conducts).
        """
        return -1.0

    @cached_property
    def eval_config(self) -> EvalConfig:
        """
        Returns NMOS evaluation sweep configuration.

        Transfer: Vg 0->1.8V (turning on), Vd=1.8V (saturation).
        Output: Vd 0->1.8V, Vg=1.2V (moderate overdrive).
        Subthreshold: Vg < 0.5V (below Vth ~0.5V).
        Source and bulk at ground.

        :return: NMOS-specific evaluation configuration.
        """
        return EvalConfig(
            strategy_name="sky130_nmos",
            vdd=1.8,
            vs_bias=0.0,
            vb_bias=0.0,
            transfer_vg_start=0.0,
            transfer_vg_stop=1.8,
            transfer_vd_bias=1.8,
            output_vd_start=0.0,
            output_vd_stop=1.8,
            output_vg_drive=1.2,
            subth_vg_threshold=0.5,
            subth_below=True,
            random_vg_range=(0.0, 1.8),
            random_vd_range=(0.0, 1.8),
        )

    @cached_property
    def waveform_config(self) -> WaveformConfig:
        """
        Returns NMOS waveform generation configuration.

        Subthreshold is the low-Vg region (Vg < Vth ~0.50V).
        Deep subthreshold: Vg in [0, 0.3V].
        Transitional: Vg in [0.05, 0.55V].

        :return: NMOS-specific waveform configuration.
        """
        return WaveformConfig(
            vth_nominal=0.50,
            vth_spread=0.25,
            deep_subth_vg_range=(0.0, 0.3),
            trans_subth_vg_range=(0.05, 0.55),
        )


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

        SPICE branch convention: i(Vd) is positive when current flows out of node d
        through the voltage source to ground. For PMOS in saturation, conventional
        current flows source-to-drain (into drain), so i(Vd) is already positive.
        No negation required (unlike NMOS where i(Vd) is negative when on).

        :return: 1.0 (PMOS i(Vd) is natively positive when device conducts).
        """
        return 1.0

    @cached_property
    def eval_config(self) -> EvalConfig:
        """
        Returns PMOS evaluation sweep configuration.

        Transfer: Vg 1.8->0V (increasing |Vgs| to turn on), Vd=0V (Vds=-1.8V, saturation).
        Output: Vd 1.8->0V (increasing |Vds|), Vg=0.3V (|Vgs|=1.5V, strong overdrive).
        Subthreshold: Vg > 1.3V (|Vgs| < |Vtp| ~0.5V).
        Source and bulk tied near Vdd.

        Vg=0.3V chosen over 0.6V because PMOS hole mobility (~0.4x NMOS) produces
        drastically lower current at the same |Vov|. At Vg=0.6V, xlarge Id-Vd span
        is only 0.74 uA (R^2 numerically degenerate). At Vg=0.3V, span is 17 uA.

        :return: PMOS-specific evaluation configuration.
        """
        return EvalConfig(
            strategy_name="sky130_pmos",
            vdd=1.8,
            vs_bias=1.8,
            vb_bias=1.8,
            transfer_vg_start=1.8,
            transfer_vg_stop=0.0,
            transfer_vd_bias=0.0,
            output_vd_start=1.8,
            output_vd_stop=0.0,
            output_vg_drive=0.3,
            subth_vg_threshold=1.3,
            subth_below=False,
            random_vg_range=(0.0, 1.8),
            random_vd_range=(0.0, 1.8),
        )

    @cached_property
    def waveform_config(self) -> WaveformConfig:
        """
        Returns PMOS waveform generation configuration.

        Subthreshold is the high-Vg region (Vg > VDD-|Vth| ~1.30V) where
        |Vgs| = Vs - Vg is small. Mirror-image of NMOS about VDD/2.
        Deep subthreshold: Vg in [1.5, 1.8V] (|Vgs| < 0.3V).
        Transitional: Vg in [1.25, 1.75V] (|Vgs| near |Vth|).

        :return: PMOS-specific waveform configuration.
        """
        return WaveformConfig(
            vth_nominal=1.30,
            vth_spread=0.25,
            deep_subth_vg_range=(1.5, 1.8),
            trans_subth_vg_range=(1.25, 1.75),
        )
