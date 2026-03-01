"""Handles data generation and tensor formatting for MOSFET training."""

import logging
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from spino.constants import ARCSINH_SCALE_MA
from spino.mosfet.bsim_parser import BSIMParser
from spino.mosfet.device_strategy import DeviceStrategy
from spino.mosfet.physics_cache import PhysicsCache
from spino.spice import OutputMode, run_ngspice

__all__ = [
    "ParameterSchema",
    "InfiniteSpiceMosfetDataset",
    "PreGeneratedMosfetDataset",
    "generate_offline_dataset",
    "merge_geometry_bins",
    "GeometryBin",
    "GEOMETRY_BINS",
]


class GeometryBin:
    """
    Defines a stratified geometry bin for balanced W/L coverage.

    Five bins cover the full Sky130 NMOS geometry space:
    - tiny: Minimum area devices (W < 0.6um, L < 0.3um)
    - small: Small devices (0.6 <= W < 1.5um, 0.3 <= L < 0.5um)
    - medium: Nominal devices (1.5 <= W < 3.5um, 0.5 <= L < 1.0um)
    - large: Large devices (3.5 <= W < 6.0um, 1.0 <= L < 1.5um)
    - xlarge: Maximum area devices (W >= 6.0um, L >= 1.5um)
    """

    def __init__(self, name: str, w_range: tuple[float, float], l_range: tuple[float, float]):
        """
        Initializes a geometry bin with W/L boundaries.

        :param name: Human-readable bin identifier.
        :param w_range: (w_min, w_max) in microns.
        :param l_range: (l_min, l_max) in microns.
        """
        self.name = name
        self.w_range = w_range
        self.l_range = l_range

    def sample(self) -> tuple[float, float]:
        """
        Samples random W/L within bin boundaries, snapped to 0.01um grid.

        :return: Tuple of (width_um, length_um).
        """
        w = np.random.uniform(*self.w_range)
        l = np.random.uniform(*self.l_range)
        return round(w, 2), round(l, 2)

    def __repr__(self) -> str:
        return f"GeometryBin({self.name}, W={self.w_range}, L={self.l_range})"


GEOMETRY_BINS: dict[str, GeometryBin] = {
    "tiny": GeometryBin("tiny", w_range=(0.42, 0.60), l_range=(0.15, 0.30)),
    "small": GeometryBin("small", w_range=(0.60, 1.50), l_range=(0.30, 0.50)),
    "medium": GeometryBin("medium", w_range=(1.50, 3.50), l_range=(0.50, 1.00)),
    "large": GeometryBin("large", w_range=(3.50, 6.00), l_range=(1.00, 1.50)),
    "xlarge": GeometryBin("xlarge", w_range=(6.00, 10.00), l_range=(1.50, 2.00)),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_GEOMETRY_PARAMS = (
    "w",
    "l",
    "xl",
    "xw",
)
_THRESHOLD_VOLTAGE_PARAMS = (
    "vth0",
    "k1",
    "k2",
    "k3",
    "k3b",
    "w0",
    "dvt0",
    "dvt1",
    "dvt2",
    "dvt0w",
    "dvt1w",
    "dvt2w",
    "nlx",
)
_MOBILITY_PARAMS = (
    "u0",
    "ua",
    "ub",
    "uc",
    "vsat",
    "a0",
    "ags",
    "b0",
    "b1",
    "keta",
)
_SUBTHRESHOLD_OUTPUT_CONDUCTANCE_PARAMS = (
    "voff",
    "nfactor",
    "cit",
    "cdsc",
    "cdscb",
    "cdscd",
    "eta0",
    "etab",
    "dsub",
    "pclm",
    "pdiblc1",
    "pdiblc2",
    "pdiblcb",
    "drout",
    "pscbe1",
    "pscbe2",
)
_TEMPERATURE_PARAMS = (
    "tnom",
    "ute",
    "kt1",
    "kt1l",
    "kt2",
    "ua1",
    "ub1",
    "uc1",
    "at",
)
_CAPACITANCE_PARAMS = (
    "cgso",
    "cgdo",
    "cgbo",
    "cj",
    "mj",
    "pb",
    "cjsw",
    "mjsw",
    "pbsw",
    "cjswg",
    "mjswg",
    "pbswg",
    "tcj",
    "tpb",
    "tcjsw",
    "tpbsw",
)
_PARASITIC_PARAMS = (
    "rdsw",
    "rsh",
    "rgate",
)
_PROCESS_PARAMS = (
    "tox",
    "toxe",
    "xj",
    "lint",
    "wint",
)
_SUPPORTED_KEYS = (
    _GEOMETRY_PARAMS
    + _THRESHOLD_VOLTAGE_PARAMS
    + _MOBILITY_PARAMS
    + _SUBTHRESHOLD_OUTPUT_CONDUCTANCE_PARAMS
    + _TEMPERATURE_PARAMS
    + _CAPACITANCE_PARAMS
    + _PARASITIC_PARAMS
    + _PROCESS_PARAMS
)


class ParameterSchema:
    """
    Defines the curated set of BSIM4 parameters for neural operator training.

    Curated list is based on empirical analysis of Sky130 NMOS dataset (25K samples):
    - Includes all 29 parameters that show measurable variation (std > 1e-6)
    - Each parameter is z-score normalized by its own mean/std, handling extreme values
    - Dominated by geometry (w, l), but includes BSIM4 parameters with corner variation

    Z-score normalization handles parameters with extreme absolute values:
    - pscbe1 ~ 7.914e+08 ± 7.867e+04 → normalized to N(0,1)
    - w ~ 2.7 ± 1.32 → normalized to N(0,1)
    Both become comparable after normalization, allowing model to learn from all variations.
    """

    SUPPORTED_KEYS = _SUPPORTED_KEYS
    TRAINING_KEYS = [
        "w",
        "l",
        "vth0",
        "k1",
        "k2",
        "k3b",
        "dvt1",
        "dvt2",
        "dvt0w",
        "dvt1w",
        "dvt2w",
        "u0",
        "vsat",
        "keta",
        "voff",
        "nfactor",
        "etab",
        "dsub",
        "pclm",
        "pdiblc1",
        "pdiblc2",
        "pdiblcb",
        "drout",
        "pscbe1",
        "ute",
        "kt1",
        "kt2",
        "at",
        "rdsw",
    ]
    TRAINING_INDICES = [
        0,
        1,
        4,
        5,
        6,
        8,
        11,
        12,
        13,
        14,
        15,
        17,
        21,
        26,
        27,
        28,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        44,
        45,
        47,
        51,
        68,
    ]

    @classmethod
    def to_tensor(cls, raw_params: dict[str, str], device: str = "cpu") -> torch.Tensor:
        """
        Converts raw BSIM parameter dictionary to fixed-schema tensor (legacy support).

        :param raw_params: Dictionary of parameter names to string values.
        :param device: PyTorch device for tensor allocation.
        :return: Tensor of shape (1, N) with zero-padded missing parameters.
        """
        vec = []
        for key in cls.SUPPORTED_KEYS:
            val_str = raw_params.get(key, "0.0")
            try:
                val = float(val_str)
            except ValueError:
                val = 0.0
            vec.append(val)
        return torch.tensor([vec], dtype=torch.float32, device=device)

    @classmethod
    def to_training_tensor(cls, params: dict[str, float]) -> torch.Tensor:
        """
        Extracts curated parameters from full BSIM4 parameter dict.

        Enforces consistent parameter ordering as defined in TRAINING_KEYS.
        Missing parameters are filled with 0.0 (should not occur in practice
        as all curated params are present in Sky130 PDK model cards).

        :param params: Dictionary mapping parameter names to values
        :return: Tensor of shape (29,) with curated parameters in fixed order
        """
        return torch.tensor([params.get(key, 0.0) for key in cls.TRAINING_KEYS], dtype=torch.float32)

    @classmethod
    def extract_from_full_tensor(cls, full_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extracts curated parameters from full 76-parameter tensor.

        Used by PreGeneratedMosfetDataset to load only varying parameters
        from existing dataset without regeneration.

        :param full_tensor: Full parameter tensor of shape (..., 76)
        :return: Curated tensor of shape (..., 29)
        """
        return full_tensor[..., cls.TRAINING_INDICES]

    @classmethod
    def input_dim(cls) -> int:
        """
        Returns the number of curated parameters used as neural operator input.

        :return: Curated parameter count (29 parameters with measurable variation)
        """
        return len(cls.TRAINING_KEYS)


class InfiniteSpiceMosfetDataset(IterableDataset):
    """
    Generates random MOSFET transient simulations on-the-fly.

    Topology:
              Vd(PWL) ----+
                          |
                      ||--+      |
          Vg(PWL) ----||------+  |  Id
                      ||--+   |  |
                          |   |  V
              VS(PWL) ----+   |
              VB(PWL) --------+

    Waveform Modes:
        - "pwl": Random piecewise-linear (default, chaotic)
        - "monotonic": Smooth DC-sweep-like ramps (up or down)
        - "vth_focused": Gate voltage concentrated near threshold (uses approximate vth calculated from BSIM params)

    """

    # Nominal Vth for Sky130 NMOS with generous spread to cover all corners
    VTH_NOMINAL = 0.50
    VTH_SPREAD = 0.25

    def __init__(
        self,
        strategy_name: str = "sky130_nmos",
        strategy_config: dict | None = None,
        t_steps: int = 1024,
        t_end: float = 1e-6,
        cache_dir: str = "/tmp/spino_physics_cache",
        waveform_mode: str = "pwl",
        geometry_bin: str | None = None,
        w_bin: str | None = None,
        l_bin: str | None = None,
    ):
        """
        Initializes the dataset generator with simulation parameters.

        :param strategy_name: Device strategy identifier (e.g., "sky130_nmos", "sky130_pmos").
        :param strategy_config: Optional dict of voltage range overrides for hyperparameter tuning.
        :param t_steps: Number of time points in output grid.
        :param t_end: Simulation end time in seconds.
        :param cache_dir: Directory for persistent physics parameter cache.
        :param waveform_mode: Waveform generation mode ("pwl", "monotonic", "vth_focused", "subthreshold_focused", "deep_subthreshold").
        :param geometry_bin: Optional geometry bin name for stratified sampling (tiny/small/medium/large/xlarge).
        :param w_bin: Optional width bin for cross-bin sampling (requires l_bin).
        :param l_bin: Optional length bin for cross-bin sampling (requires w_bin).
        """
        if strategy_config is None:
            strategy_config = {}
        self.strategy = DeviceStrategy.create(strategy_name, **strategy_config)
        self.cache = PhysicsCache()
        self.t_steps = t_steps
        self.t_end = t_end
        self.sim_step = t_end / (t_steps * 2.0)
        self.parser = BSIMParser(pdk_root=self.strategy.pdk_root)
        self.cache = PhysicsCache(cache_dir=cache_dir)
        self.waveform_mode = waveform_mode
        if geometry_bin and (w_bin or l_bin):
            raise ValueError("geometry_bin cannot be combined with w_bin/l_bin")
        if bool(w_bin) != bool(l_bin):
            raise ValueError("w_bin and l_bin must be provided together")
        self.geometry_bin = GEOMETRY_BINS.get(geometry_bin) if geometry_bin else None
        if geometry_bin and self.geometry_bin is None:
            raise ValueError(f"Unknown geometry bin: {geometry_bin}. Valid: {list(GEOMETRY_BINS.keys())}")
        self.w_bin = GEOMETRY_BINS.get(w_bin) if w_bin else None
        self.l_bin = GEOMETRY_BINS.get(l_bin) if l_bin else None
        if w_bin and self.w_bin is None:
            raise ValueError(f"Unknown w_bin: {w_bin}. Valid: {list(GEOMETRY_BINS.keys())}")
        if l_bin and self.l_bin is None:
            raise ValueError(f"Unknown l_bin: {l_bin}. Valid: {list(GEOMETRY_BINS.keys())}")

    def _get_physics_tensor(self, w: float, l: float) -> torch.Tensor:
        """
        Extracts and caches curated physics parameters for specific device geometry.

        Returns only the 24 curated BSIM4 parameters defined in ParameterSchema.TRAINING_KEYS.

        :param w: Transistor width in microns.
        :param l: Transistor length in microns.
        :return: Curated parameter tensor of shape (24,).
        """
        cached_params = self.cache.get(self.strategy.model_name, w, l)
        if cached_params is not None:
            full_tensor = cached_params.squeeze()
            params_dict = {key: full_tensor[i].item() for i, key in enumerate(ParameterSchema.SUPPORTED_KEYS)}
            return ParameterSchema.to_training_tensor(params_dict)
        raw_params = self.parser.inspect_model(self.strategy.model_name, w=str(w), l=str(l))
        if not raw_params:
            raw_params = {}
        full_tensor = ParameterSchema.to_tensor(raw_params).squeeze()
        params_dict = {key: full_tensor[i].item() for i, key in enumerate(ParameterSchema.SUPPORTED_KEYS)}
        self.cache.put(self.strategy.model_name, w, l, full_tensor.unsqueeze(0))
        return ParameterSchema.to_training_tensor(params_dict)

    def _generate_pwl_voltage(self, min_v: float = 0.0, max_v: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates randomized piecewise-linear voltage waveform.

        :param min_v: Minimum voltage boundary.
        :param max_v: Maximum voltage boundary.
        :return: Tuple of (time_points, voltage_values).
        """
        n_points = np.random.randint(5, 15)
        times = np.sort(np.random.uniform(0, self.t_end, n_points))
        times = np.concatenate(([0], times, [self.t_end]))
        volts = np.random.uniform(min_v, max_v, len(times))
        return times, volts

    def _generate_monotonic_ramp(self, min_v: float = 0.0, max_v: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates monotonic voltage ramp (DC-sweep-like waveform).

        Randomly chooses between rising (min->max) and falling (max->min) ramps.
        Adds small random offset to start/end voltages for variety.

        :param min_v: Minimum voltage boundary.
        :param max_v: Maximum voltage boundary.
        :return: Tuple of (time_points, voltage_values).
        """
        times = np.array([0.0, self.t_end])
        offset = np.random.uniform(-0.05, 0.05) * (max_v - min_v)
        v_start = np.clip(min_v + offset, min_v, max_v)
        v_end = np.clip(max_v + offset, min_v, max_v)
        if np.random.random() < 0.5:
            v_start, v_end = v_end, v_start
        volts = np.array([v_start, v_end])
        return times, volts

    def _generate_vth_focused_voltage(self, min_v: float = 0.0, max_v: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates gate voltage waveform concentrated around threshold voltage.

        Creates ramps centered near nominal Vth with generous spread to capture
        the subthreshold-to-saturation transition across process corners.

        :param min_v: Minimum voltage boundary.
        :param max_v: Maximum voltage boundary.
        :return: Tuple of (time_points, voltage_values).
        """
        vth_center = self.VTH_NOMINAL + np.random.uniform(-self.VTH_SPREAD, self.VTH_SPREAD)
        vth_window = 0.35
        v_low = max(min_v, vth_center - vth_window)
        v_high = min(max_v, vth_center + vth_window)
        times = np.array([0.0, self.t_end])
        if np.random.random() < 0.5:
            volts = np.array([v_low, v_high])
        else:
            volts = np.array([v_high, v_low])
        return times, volts

    def _generate_subthreshold_focused_voltage(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates gate voltage waveform concentrated around subthreshold region.

        :return: Tuple of (time_points, voltage_values).
        """
        vth_jitter = np.random.uniform(0, self.VTH_SPREAD)
        v_low = 0.0
        v_high = self.VTH_NOMINAL - vth_jitter
        end_time = np.random.uniform(0.5 * self.t_end, self.t_end)
        times = np.array([0.0, end_time, self.t_end])
        if np.random.random() < 0.5:
            volts = np.array([v_low, v_high, v_high])
        else:
            volts = np.array([v_high, v_low, v_low])
        return times, volts

    def _generate_deep_subthreshold_voltage(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates gate voltage waveform concentrated strictly in deep subthreshold region.

        :return: Tuple of (time_points, voltage_values).
        """
        v_low = 0.0
        v_high = np.random.uniform(0.1, 0.3)
        end_time = np.random.uniform(0.5 * self.t_end, self.t_end)
        times = np.array([0.0, end_time, self.t_end])
        if np.random.random() < 0.5:
            volts = np.array([v_low, v_high, v_high])
        else:
            volts = np.array([v_high, v_low, v_low])
        return times, volts

    def _generate_transitional_subthreshold_voltage(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates gate voltage in the transitional subthreshold region (0.15V-0.5V).

        Targets the gm/Id design sweet spot where currents are 1-100 nA: large
        enough that LpLoss denominators do not collapse, yet firmly in weak
        inversion where analog designers care about accuracy.

        :return: Tuple of (time_points, voltage_values).
        """
        v_low = np.random.uniform(0.05, 0.20)
        v_high = np.random.uniform(0.35, 0.55)
        end_time = np.random.uniform(0.5 * self.t_end, self.t_end)
        times = np.array([0.0, end_time, self.t_end])
        if np.random.random() < 0.5:
            volts = np.array([v_low, v_high, v_high])
        else:
            volts = np.array([v_high, v_low, v_low])
        return times, volts

    def _get_voltage_generator(self, terminal: str):
        """
        Returns the appropriate voltage generator based on waveform mode and terminal.

        :param terminal: Terminal name ("gate", "drain", "source", "bulk").
        :return: Voltage generation function.
        """
        if self.waveform_mode == "monotonic":
            return self._generate_monotonic_ramp
        elif self.waveform_mode == "vth_focused" and terminal == "gate":
            return self._generate_vth_focused_voltage
        elif self.waveform_mode == "subthreshold_focused" and terminal == "gate":
            return self._generate_subthreshold_focused_voltage
        elif self.waveform_mode == "deep_subthreshold" and terminal == "gate":
            return self._generate_deep_subthreshold_voltage
        elif self.waveform_mode == "transitional_subthreshold" and terminal == "gate":
            return self._generate_transitional_subthreshold_voltage
        return self._generate_pwl_voltage

    def _build_pwl_string(self, times: np.ndarray, volts: np.ndarray) -> str:
        """
        Constructs PWL string for NGSpice voltage sources.

        :param times: Time points array.
        :param volts: Voltage values array.
        :return: PWL string formatted as PWL(t1, v1 t2, v2 ...).
        """
        pairs = [f"{t}, {v}" for t, v in zip(times, volts)]
        return f"PWL({' '.join(pairs)})"

    def _build_netlist(
        self,
        w_um: float,
        l_um: float,
        pwl_g: str,
        pwl_d: str,
        pwl_s: str,
        pwl_b: str,
    ) -> str:
        """
        Generates MOSFET transient netlist as string.

        :param w_um: Width in microns (unitless for PDK scaling).
        :param l_um: Length in microns (unitless for PDK scaling).
        :param pwl_g: Gate voltage PWL string.
        :param pwl_d: Drain voltage PWL string.
        :param pwl_s: Source voltage PWL string.
        :param pwl_b: Bulk voltage PWL string.
        :return: Complete SPICE netlist body without .end directive.
        """
        return f"""\
* MOSFET Training Simulation
.lib '{self.parser.lib_path}' tt
X1 d g s b {self.strategy.model_name} w={w_um} l={l_um}
Vg g 0 {pwl_g}
Vd d 0 {pwl_d}
Vs s 0 {pwl_s}
Vb b 0 {pwl_b}
"""

    def _run_transient_simulation(self, netlist_body: str) -> dict | None:
        """
        Executes transient simulation and returns parsed results.

        :param netlist_body: SPICE netlist without .end directive.
        :return: Parsed simulation data or None on failure.
        """
        deck_content = (
            f"{netlist_body}"
            f".tran {self.sim_step} {self.t_end}\n"
            ".option strict_errorhandling=0\n"
            ".option savecurrents\n"
            ".end\n"
        )
        success, parsed_data = run_ngspice(
            deck_content,
            output_mode=OutputMode.RAW_FILE,
            spice_filename="mosfet_sim.spice",
            timeout=30.0,
        )
        return parsed_data if success else None

    def _sample_device_geometry(self) -> tuple[float, float]:
        """
        Samples random transistor geometry snapped to PDK grid.

        If geometry_bin is set, samples within that bin's W/L range.
        Otherwise uses full uniform sampling (legacy behavior).

        :return: Tuple of (width_um, length_um) rounded to 0.01um precision.
        """
        if self.w_bin is not None and self.l_bin is not None:
            width_raw = np.random.uniform(*self.w_bin.w_range)
            length_raw = np.random.uniform(*self.l_bin.l_range)
            return round(width_raw, 2), round(length_raw, 2)
        if self.geometry_bin is not None:
            return self.geometry_bin.sample()
        width_raw = np.random.uniform(0.42, 5.0)
        length_raw = np.random.uniform(0.15, 2.0)
        return round(width_raw, 2), round(length_raw, 2)

    def _sample_terminal_voltages(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Generates voltage waveforms for all transistor terminals based on waveform mode.

        :return: Dictionary mapping terminal names to (times, voltages) tuples.
        """
        if self.waveform_mode == "pwl":
            return self.strategy.sample_terminal_voltages(self._generate_pwl_voltage)
        elif self.waveform_mode == "monotonic":
            return self.strategy.sample_terminal_voltages(self._generate_monotonic_ramp)
        elif self.waveform_mode == "vth_focused":
            gate_range = self.strategy.gate_range
            drain_range = self.strategy.drain_range
            source_range = self.strategy.source_range
            bulk_range = self.strategy.bulk_range
            return {
                "gate": self._generate_vth_focused_voltage(*gate_range),
                "drain": self._generate_monotonic_ramp(*drain_range),
                "source": self._generate_monotonic_ramp(*source_range),
                "bulk": self._generate_monotonic_ramp(*bulk_range),
            }
        elif self.waveform_mode == "subthreshold_focused":
            drain_range = self.strategy.drain_range
            source_range = self.strategy.source_range
            bulk_range = self.strategy.bulk_range
            return {
                "gate": self._generate_subthreshold_focused_voltage(),
                "drain": self._generate_monotonic_ramp(*drain_range),
                "source": self._generate_monotonic_ramp(*source_range),
                "bulk": self._generate_monotonic_ramp(*bulk_range),
            }
        elif self.waveform_mode == "deep_subthreshold":
            drain_range = self.strategy.drain_range
            source_range = self.strategy.source_range
            bulk_range = self.strategy.bulk_range
            return {
                "gate": self._generate_deep_subthreshold_voltage(),
                "drain": self._generate_monotonic_ramp(*drain_range),
                "source": self._generate_monotonic_ramp(*source_range),
                "bulk": self._generate_monotonic_ramp(*bulk_range),
            }
        elif self.waveform_mode == "transitional_subthreshold":
            drain_range = self.strategy.drain_range
            source_range = self.strategy.source_range
            bulk_range = self.strategy.bulk_range
            return {
                "gate": self._generate_transitional_subthreshold_voltage(),
                "drain": self._generate_monotonic_ramp(*drain_range),
                "source": self._generate_monotonic_ramp(*source_range),
                "bulk": self._generate_monotonic_ramp(*bulk_range),
            }
        return self.strategy.sample_terminal_voltages(self._generate_pwl_voltage)

    def _interpolate_terminal_voltages(
        self, terminal_voltages: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """
        Resamples voltage waveforms onto uniform time grid.

        :param terminal_voltages: Dictionary of (times, volts) per terminal.
        :return: Dictionary of interpolated voltage arrays.
        """
        time_grid = np.linspace(0, self.t_end, self.t_steps)
        return {terminal: np.interp(time_grid, times, volts) for terminal, (times, volts) in terminal_voltages.items()}

    def _extract_drain_current(self, results: dict) -> np.ndarray | None:
        """
        Locates and extracts drain current from simulation results.

        :param results: Parsed NGSpice output dictionary.
        :return: Raw drain current array or None if not found.
        """
        current_key_candidates = ["vd#branch", "v.vd.branch", "vd", "i(vd)"]
        for key in current_key_candidates:
            if key in results["nodes"]:
                return results["nodes"][key] * self.strategy.current_polarity_multiplier
        return None

    def _interpolate_current(self, current_raw: np.ndarray, sim_time: np.ndarray | None) -> np.ndarray | None:
        """
        Resamples drain current onto uniform time grid.

        :param current_raw: Raw current data from simulation.
        :param sim_time: Simulation time points or None.
        :return: Interpolated current array or None if time data missing.
        """
        if sim_time is None:
            return None
        time_grid = np.linspace(0, self.t_end, self.t_steps)
        return np.interp(time_grid, sim_time, current_raw)

    def _assemble_tensors(
        self,
        interpolated_voltages: dict[str, np.ndarray],
        current_ma: np.ndarray,
        width_um: float,
        length_um: float,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Converts numpy data to PyTorch tensors for training.

        :param interpolated_voltages: Resampled terminal voltages.
        :param current_ma: Drain current in milliamps.
        :param width_um: Device width in microns.
        :param length_um: Device length in microns.
        :return: Tuple of ((voltage_tensor, physics_tensor), current_tensor).
        """
        voltage_stack = np.stack(
            [
                interpolated_voltages["gate"],
                interpolated_voltages["drain"],
                interpolated_voltages["source"],
                interpolated_voltages["bulk"],
            ]
        )
        voltages = torch.tensor(voltage_stack, dtype=torch.float32)
        physics = self._get_physics_tensor(width_um, length_um)
        current = torch.tensor(current_ma, dtype=torch.float32).unsqueeze(0)
        return (voltages, physics), current

    def generate_sample(self):
        """
        Generates a single training sample via SPICE simulation.

        :return: Tuple of ((voltages, physics), current) tensors.
        """
        while True:
            width_um, length_um = self._sample_device_geometry()
            terminal_voltages = self._sample_terminal_voltages()
            netlist = self._build_netlist(
                width_um,
                length_um,
                self._build_pwl_string(*terminal_voltages["gate"]),
                self._build_pwl_string(*terminal_voltages["drain"]),
                self._build_pwl_string(*terminal_voltages["source"]),
                self._build_pwl_string(*terminal_voltages["bulk"]),
            )
            if (results := self._run_transient_simulation(netlist)) is None:
                continue
            if (current_raw := self._extract_drain_current(results)) is None:
                continue
            if (current_interp := self._interpolate_current(current_raw, results.get("time"))) is None:
                continue
            interpolated_voltages = self._interpolate_terminal_voltages(terminal_voltages)
            current_ma = current_interp * 1000.0
            return self._assemble_tensors(interpolated_voltages, current_ma, width_um, length_um)

    def __iter__(self):
        """
        Yields infinite stream of training samples.

        :yield: Tuple of ((voltages, physics), current) tensors.
        """
        worker_info = get_worker_info()
        if worker_info is not None:
            # We are in a worker process.
            # Combine the base seed with the worker ID to ensure unique streams.
            # (2**32 - 1) is max for np.random.seed
            seed = (torch.initial_seed() + worker_info.id) % (2**32)
            np.random.seed(seed)
        while True:
            yield self.generate_sample()

    def __getitem__(self, index):
        """
        Random access not supported for iterable dataset.

        :param index: Unused index parameter.
        :raises NotImplementedError: Always raised.
        """
        raise NotImplementedError("IterableDataset does not support random access.")


class PreGeneratedMosfetDataset(Dataset):
    """
    Loads pre-generated MOSFET samples from HDF5 for fast training.

    Provides random access to cached simulation results, eliminating
    SPICE execution overhead during training loops. Applies normalization
    to voltages, currents, and physics parameters.
    """

    ARCSINH_SCALE_MA = ARCSINH_SCALE_MA

    def __init__(
        self,
        hdf5_path: str,
        normalize: bool = True,
        use_curated_params: bool = True,
        trim_startup: int = 0,
        geometry_filter: str | None = None,
    ):
        """
        Initializes dataset from HDF5 file.

        :param hdf5_path: Path to HDF5 file created by generate_offline_dataset.
        :param normalize: Whether to apply normalization to inputs/outputs.
        :param use_curated_params: Whether to use curated physics parameters or all.
        :param trim_startup: Number of initial timesteps to discard (removes .op blip artifact).
        :param geometry_filter: Optional geometry bin name (e.g. "xlarge") to restrict samples.
            Normalization stats are computed from the FULL dataset before filtering, preserving
            compatibility with pretrained checkpoints.
        """
        self.file = h5py.File(hdf5_path, "r")
        self.voltages = self.file["voltages"]
        self.physics = self.file["physics"]
        self.current = self.file["current"]
        self.num_samples = self.voltages.shape[0]
        self.normalize = normalize
        self.use_curated_params = use_curated_params
        self.trim_startup = trim_startup
        self._valid_indices = None
        if trim_startup > 0:
            logger.info(
                "Startup trim enabled: discarding first %d timesteps (%.1f%%) from each sample",
                trim_startup,
                100.0 * trim_startup / self.voltages.shape[2],
            )
        self._already_curated = self.physics.shape[1] == ParameterSchema.input_dim()
        if self._already_curated:
            logger.info(
                "Physics parameters already curated: %d params (skipping index extraction)",
                self.physics.shape[1],
            )
        else:
            logger.info(
                "Pruning physics parameters: %d (stored) → %d (curated)",
                self.physics.shape[1],
                ParameterSchema.input_dim(),
            )
        if normalize:
            self._compute_normalization_stats()
        if geometry_filter:
            self._apply_geometry_filter(geometry_filter)
        logger.info("Loaded pre-generated dataset: %d samples from %s", self.num_samples, hdf5_path)

    def _compute_normalization_stats(self):
        """
        Computes mean and std for voltages and physics parameters using first 1000 samples.

        Voltage normalization: Per-terminal z-score using mean/std across samples.
        Physics normalization: Z-score normalization for curated or all parameters.
        No dynamic filtering - uses static schema defined in ParameterSchema.TRAINING_KEYS.

        Note: Current normalization is applied dynamically in __getitem__ using log-scale.
        """
        logger.info("Computing normalization statistics...")
        voltages_data = self.voltages[:1000]
        physics_data = self.physics[:]
        if self.use_curated_params and not self._already_curated:
            physics_curated = physics_data[:, ParameterSchema.TRAINING_INDICES]
            logger.info(
                "Extracting %d curated physics parameters from %d stored",
                ParameterSchema.input_dim(),
                physics_data.shape[1],
            )
        else:
            physics_curated = physics_data
            logger.info("Using %d physics parameters for training", physics_data.shape[1])
        self.voltages_mean = torch.from_numpy(voltages_data.mean(axis=(0, 2))).float().unsqueeze(1)
        self.voltages_std = torch.from_numpy(voltages_data.std(axis=(0, 2)) + 1e-8).float().unsqueeze(1)
        self.physics_mean = torch.from_numpy(physics_curated.mean(axis=0)).float()
        self.physics_std = torch.from_numpy(physics_curated.std(axis=0) + 1e-8).float()
        logger.info(
            "Voltage normalization: mean=%s, std=%s",
            self.voltages_mean.squeeze().numpy(),
            self.voltages_std.squeeze().numpy(),
        )
        logger.info(
            "Physics normalization: mean range [%.3e, %.3e], std range [%.3e, %.3e]",
            self.physics_mean.min().item(),
            self.physics_mean.max().item(),
            self.physics_std.min().item(),
            self.physics_std.max().item(),
        )

    def _apply_geometry_filter(self, geometry_filter: str):
        """
        Restricts dataset to samples matching a named geometry bin.

        W is at physics column 0, L at column 1 in both curated and full arrays.
        Normalization stats are already computed from the full dataset before this
        call, preserving checkpoint compatibility.

        :param geometry_filter: Key into GEOMETRY_BINS (e.g. "xlarge").
        """
        if geometry_filter not in GEOMETRY_BINS:
            valid_names = ", ".join(sorted(GEOMETRY_BINS.keys()))
            raise ValueError(f"Unknown geometry bin '{geometry_filter}'. Valid: {valid_names}")
        geo_bin = GEOMETRY_BINS[geometry_filter]
        raw_physics = self.physics[:]
        w_vals = raw_physics[:, 0]
        l_vals = raw_physics[:, 1]
        mask = (
            (w_vals >= geo_bin.w_range[0])
            & (w_vals <= geo_bin.w_range[1])
            & (l_vals >= geo_bin.l_range[0])
            & (l_vals <= geo_bin.l_range[1])
        )
        total = len(mask)
        self._valid_indices = np.where(mask)[0]
        self.num_samples = len(self._valid_indices)
        logger.info(
            "Geometry filter '%s' (W=[%.2f,%.2f], L=[%.2f,%.2f]): %d / %d samples (%.1f%%)",
            geometry_filter,
            geo_bin.w_range[0],
            geo_bin.w_range[1],
            geo_bin.l_range[0],
            geo_bin.l_range[1],
            self.num_samples,
            total,
            100.0 * self.num_samples / total,
        )

    def __len__(self) -> int:
        """
        Returns total number of pre-generated samples.

        :return: Dataset size.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Retrieves sample by index with normalization applied.

        :param idx: Sample index.
        :return: Tuple of ((voltages, physics), current) tensors, all normalized.
        """
        raw_idx = self._valid_indices[idx] if self._valid_indices is not None else idx
        voltages = torch.from_numpy(self.voltages[raw_idx][:]).float()
        physics = torch.from_numpy(self.physics[raw_idx][:]).float()
        current = torch.from_numpy(self.current[raw_idx][:]).float()
        if self.trim_startup > 0:
            voltages = voltages[:, self.trim_startup :]
            current = current[..., self.trim_startup :]
        if self.use_curated_params and not self._already_curated:
            physics = physics[ParameterSchema.TRAINING_INDICES]
        if self.normalize:
            voltages = (voltages - self.voltages_mean) / self.voltages_std
            physics = ((physics - self.physics_mean) / self.physics_std).unsqueeze(0)
            # arcsinh transform: handles negative currents, bijective, smooth near zero
            current = torch.asinh(current / self.ARCSINH_SCALE_MA)
        else:
            physics = physics.unsqueeze(0)
        return (voltages, physics), current

    def close(self):
        """Closes HDF5 file handle."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        """Ensures file is closed on garbage collection."""
        self.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False


def debug_visualize(
    filename: str = "Mosfet_Sample_Debug.png",
    strategy_name: str = "sky130_nmos",
    strategy_config: dict | None = None,
):
    """
    Generates and visualizes a single training sample for debugging.

    :param filename: Output path for the diagnostic plot image.
    :param strategy_name: Device strategy to test (e.g., "sky130_nmos", "sky130_pmos").
    :param strategy_config: Optional voltage range overrides.
    """
    logger.info("Initializing Dataset with strategy: %s", strategy_name)
    dataset = InfiniteSpiceMosfetDataset(
        strategy_name=strategy_name,
        strategy_config=strategy_config,
        t_steps=1024,
        t_end=1e-6,
    )
    logger.info("Generating Sample...")
    (inputs, current_tensor) = next(iter(dataset))
    v_terminals, physics_params = inputs
    logger.info("Shapes verified:")
    logger.info("  Volts:   %s (Expected: 4, 1024)", v_terminals.shape)
    logger.info("  Physics: %s (Expected: 1, %s)", physics_params.shape, ParameterSchema.input_dim())
    logger.info("  Current: %s (Expected: 1, 1024)", current_tensor.shape)
    time_axis_us = np.linspace(0, dataset.t_end * 1e6, dataset.t_steps)
    gate_voltage = v_terminals[0].numpy()
    drain_voltage = v_terminals[1].numpy()
    source_voltage = v_terminals[2].numpy()
    drain_current_ma = current_tensor[0].numpy()
    device_params = _extract_device_parameters(physics_params)
    _create_diagnostic_plot(
        time_axis_us, gate_voltage, drain_voltage, source_voltage, drain_current_ma, device_params, filename
    )


def _extract_device_parameters(physics_params: torch.Tensor) -> dict[str, float]:
    """
    Extracts key device parameters from physics tensor for plot labeling.

    :param physics_params: Physics parameter tensor (curated 24-parameter schema).
    :return: Dictionary with width, length, and threshold voltage.
    """
    try:
        idx_w = ParameterSchema.TRAINING_KEYS.index("w")
        idx_l = ParameterSchema.TRAINING_KEYS.index("l")
        idx_vth = ParameterSchema.TRAINING_KEYS.index("vth0")
        return {
            "width": physics_params[0, idx_w].item(),
            "length": physics_params[0, idx_l].item(),
            "vth0": physics_params[0, idx_vth].item(),
        }
    except ValueError:
        logger.warning("Could not extract physics params for title (key mismatch).")
        return {"width": 0, "length": 0, "vth0": 0}


def _create_diagnostic_plot(
    time_us: np.ndarray,
    gate_v: np.ndarray,
    drain_v: np.ndarray,
    source_v: np.ndarray,
    drain_current_ma: np.ndarray,
    device_params: dict[str, float],
    filename: str,
):
    """
    Creates a two-panel diagnostic plot of voltage inputs and current response.

    :param time_us: Time axis in microseconds.
    :param gate_v: Gate voltage waveform.
    :param drain_v: Drain voltage waveform.
    :param source_v: Source voltage waveform.
    :param drain_current_ma: Drain current in milliamps.
    :param device_params: Device geometry and threshold voltage.
    :param filename: Output file path.
    """
    logger.info(
        "Sample Params: W=%su, L=%su, Vth0=%sV", device_params["width"], device_params["length"], device_params["vth0"]
    )
    plt.style.use("dark_background")
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(time_us, gate_v, label="Vg", color="lime")
    ax1.plot(time_us, drain_v, label="Vd", color="cyan")
    ax1.plot(time_us, source_v, label="Vs", color="yellow", linestyle="--")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"MOSFET Transient Inputs (W={device_params['width']}u, L={device_params['length']}u)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax2.plot(time_us, drain_current_ma, label="Id (Drain Current)", color="magenta")
    ax2.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Current (mA)")
    ax2.set_xlabel("Time (us)")
    ax2.set_title(f"Response (Vth0 = {device_params['vth0']:.3f} V)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info("Debug plot saved to %s", filename)
    plt.close()


def generate_offline_dataset(
    output_path: str,
    num_samples: int,
    strategy_name: str = "sky130_nmos",
    strategy_config: dict | None = None,
    t_steps: int = 2048,
    t_end: float = 1e-6,
    num_workers: int = 8,
    progress_callback=None,
    overwrite: bool = False,
    waveform_mode: str = "pwl",
    geometry_bin: str | None = None,
    w_bin: str | None = None,
    l_bin: str | None = None,
):
    """
    Generates pre-computed dataset using multiprocessing and saves to HDF5.

    :param output_path: Path to output HDF5 file.
    :param num_samples: Total number of samples to generate.
    :param strategy_name: Device strategy identifier.
    :param strategy_config: Optional voltage range overrides.
    :param t_steps: Number of time points per sample.
    :param t_end: Simulation end time in seconds.
    :param num_workers: Number of parallel workers (16 recommended for 22GB RAM).
    :param progress_callback: Optional callback(completed_count) for progress tracking.
    :param overwrite: If True, overwrite existing file. If False (default), append to existing file.
    :param waveform_mode: Waveform generation mode ("pwl", "monotonic", "vth_focused", "subthreshold_focused", "deep_subthreshold").
    :param geometry_bin: Optional geometry bin for stratified sampling (tiny/small/medium/large/xlarge).
    :param w_bin: Optional width bin for cross-bin sampling (requires l_bin).
    :param l_bin: Optional length bin for cross-bin sampling (requires w_bin).
    """
    if geometry_bin and (w_bin or l_bin):
        raise ValueError("geometry_bin cannot be combined with w_bin/l_bin")
    if bool(w_bin) != bool(l_bin):
        raise ValueError("w_bin and l_bin must be provided together")
    if geometry_bin and geometry_bin not in GEOMETRY_BINS:
        raise ValueError(f"Unknown geometry bin: {geometry_bin}. Valid: {list(GEOMETRY_BINS.keys())}")
    if w_bin and w_bin not in GEOMETRY_BINS:
        raise ValueError(f"Unknown w_bin: {w_bin}. Valid: {list(GEOMETRY_BINS.keys())}")
    if l_bin and l_bin not in GEOMETRY_BINS:
        raise ValueError(f"Unknown l_bin: {l_bin}. Valid: {list(GEOMETRY_BINS.keys())}")
    output_file = Path(output_path)
    existing_samples = 0
    if output_file.exists() and not overwrite:
        with h5py.File(output_path, "r") as f:
            existing_samples = f["voltages"].shape[0]
            if f["voltages"].chunks is None:
                raise ValueError(
                    f"Cannot append to {output_path}: datasets are not chunked. "
                    "Use --overwrite to recreate with chunked storage."
                )
        logger.info(
            "Appending to existing dataset: %d samples → %d samples", existing_samples, existing_samples + num_samples
        )
    elif output_file.exists() and overwrite:
        logger.info("Overwriting existing dataset: %s", output_path)
        output_file.unlink()
    samples_per_worker = num_samples // num_workers
    remainder = num_samples % num_workers
    logger.info(
        "Generating %d samples with %d workers (%d samples/worker)", num_samples, num_workers, samples_per_worker
    )
    temp_dir = Path(output_path).parent / "temp_generation"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def worker_generate(worker_id: int, num_samples_worker: int, temp_file: str, progress_queue):
        """Worker process that generates samples and writes to temp HDF5."""
        try:
            dataset = InfiniteSpiceMosfetDataset(
                strategy_name=strategy_name,
                strategy_config=strategy_config or {},
                t_steps=t_steps,
                t_end=t_end,
                waveform_mode=waveform_mode,
                geometry_bin=geometry_bin,
                w_bin=w_bin,
                l_bin=l_bin,
            )
            with h5py.File(temp_file, "w") as f:
                voltages_ds = f.create_dataset("voltages", shape=(num_samples_worker, 4, t_steps), dtype="float32")
                physics_ds = f.create_dataset(
                    "physics", shape=(num_samples_worker, ParameterSchema.input_dim()), dtype="float32"
                )
                current_ds = f.create_dataset("current", shape=(num_samples_worker, 1, t_steps), dtype="float32")
                for i in range(num_samples_worker):
                    (voltages, physics), current = dataset.generate_sample()
                    voltages_ds[i] = voltages.numpy()
                    physics_ds[i] = physics.numpy()
                    current_ds[i] = current.numpy()
                    progress_queue.put(1)
        except Exception as e:
            logger.exception("Worker %d: Fatal error during generation: %s", worker_id, e)
            progress_queue.put(-1)
            raise

    progress_queue = mp.Queue()
    processes = []
    temp_files = []
    for worker_id in range(num_workers):
        worker_samples = samples_per_worker + (1 if worker_id < remainder else 0)
        temp_file = str(temp_dir / f"worker_{worker_id}.h5")
        temp_files.append(temp_file)
        p = mp.Process(target=worker_generate, args=(worker_id, worker_samples, temp_file, progress_queue))
        p.start()
        processes.append(p)
    completed = 0
    errors = 0
    while any(p.is_alive() for p in processes):
        try:
            result = progress_queue.get(timeout=1.0)
            if result == -1:
                errors += 1
            else:
                completed += result
                if progress_callback:
                    progress_callback(completed)
        except:
            pass
    for p in processes:
        p.join()
    if errors > 0:
        raise RuntimeError(f"{errors} worker(s) failed during generation")
    logger.info("All workers completed. Merging into final dataset...")
    total_samples = existing_samples + num_samples
    mode = "a" if existing_samples > 0 else "w"
    with h5py.File(output_path, mode) as f_out:
        if existing_samples > 0:
            f_out["voltages"].resize((total_samples, 4, t_steps))
            f_out["physics"].resize((total_samples, ParameterSchema.input_dim()))
            f_out["current"].resize((total_samples, 1, t_steps))
        else:
            f_out.create_dataset(
                "voltages", shape=(num_samples, 4, t_steps), dtype="float32", maxshape=(None, 4, t_steps), chunks=True
            )
            f_out.create_dataset(
                "physics",
                shape=(num_samples, ParameterSchema.input_dim()),
                dtype="float32",
                maxshape=(None, ParameterSchema.input_dim()),
                chunks=True,
            )
            f_out.create_dataset(
                "current", shape=(num_samples, 1, t_steps), dtype="float32", maxshape=(None, 1, t_steps), chunks=True
            )
        offset = existing_samples
        for temp_file in temp_files:
            with h5py.File(temp_file, "r") as f_in:
                n = f_in["voltages"].shape[0]
                f_out["voltages"][offset : offset + n] = f_in["voltages"][:]
                f_out["physics"][offset : offset + n] = f_in["physics"][:]
                f_out["current"][offset : offset + n] = f_in["current"][:]
                offset += n
        f_out.attrs["strategy_name"] = strategy_name
        f_out.attrs["t_steps"] = t_steps
        f_out.attrs["t_end"] = t_end
        f_out.attrs["num_samples"] = total_samples
        f_out.attrs["waveform_mode"] = waveform_mode
        if geometry_bin:
            f_out.attrs["geometry_bin"] = geometry_bin
            f_out.attrs["geometry_mode"] = "single_bin"
        elif w_bin and l_bin:
            f_out.attrs["geometry_mode"] = "cross_bin"
            f_out.attrs["w_bin"] = w_bin
            f_out.attrs["l_bin"] = l_bin
        else:
            f_out.attrs["geometry_mode"] = "uniform"
    shutil.rmtree(temp_dir)
    logger.info("Dataset saved to %s (%d samples)", output_path, total_samples)


def merge_geometry_bins(
    bin_files: list[str],
    output_path: str,
    shuffle: bool = True,
):
    """
    Merges multiple geometry bin HDF5 files into a single stratified dataset.

    :param bin_files: List of paths to geometry bin HDF5 files.
    :param output_path: Path to output merged HDF5 file.
    :param shuffle: Whether to shuffle samples after merging (default True).
    """
    if not bin_files:
        raise ValueError("No bin files provided")
    total_samples = 0
    t_steps = None
    physics_dim = None
    for bin_file in bin_files:
        with h5py.File(bin_file, "r") as f:
            n = f["voltages"].shape[0]
            total_samples += n
            if t_steps is None:
                t_steps = f["voltages"].shape[2]
                physics_dim = f["physics"].shape[1]
            logger.info("  %s: %d samples (bin=%s)", bin_file, n, f.attrs.get("geometry_bin", "unknown"))
    logger.info("Merging %d files into %s (%d total samples)", len(bin_files), output_path, total_samples)
    with h5py.File(output_path, "w") as f_out:
        voltages_ds = f_out.create_dataset(
            "voltages", shape=(total_samples, 4, t_steps), dtype="float32", maxshape=(None, 4, t_steps), chunks=True
        )
        physics_ds = f_out.create_dataset(
            "physics", shape=(total_samples, physics_dim), dtype="float32", maxshape=(None, physics_dim), chunks=True
        )
        current_ds = f_out.create_dataset(
            "current", shape=(total_samples, 1, t_steps), dtype="float32", maxshape=(None, 1, t_steps), chunks=True
        )
        offset = 0
        for bin_file in bin_files:
            with h5py.File(bin_file, "r") as f_in:
                n = f_in["voltages"].shape[0]
                voltages_ds[offset : offset + n] = f_in["voltages"][:]
                physics_ds[offset : offset + n] = f_in["physics"][:]
                current_ds[offset : offset + n] = f_in["current"][:]
                offset += n
        if shuffle:
            logger.info("Shuffling %d samples...", total_samples)
            perm = np.random.permutation(total_samples)
            voltages_shuffled = voltages_ds[:][perm]
            physics_shuffled = physics_ds[:][perm]
            current_shuffled = current_ds[:][perm]
            voltages_ds[:] = voltages_shuffled
            physics_ds[:] = physics_shuffled
            current_ds[:] = current_shuffled
        f_out.attrs["num_samples"] = total_samples
        f_out.attrs["t_steps"] = t_steps
        f_out.attrs["merged_from"] = ",".join(bin_files)
    logger.info("Merged dataset saved to %s (%d samples)", output_path, total_samples)


if __name__ == "__main__":
    debug_visualize()
