"""
Data generation for diode circuit neural operator training.

Provides three dataset implementations:

- ``InfiniteSpiceDiodeDataset`` (legacy): on-the-fly SPICE generation with fixed
  1ms window and raw physical units.
- ``DimensionlessDiodeDataset``: on-the-fly SPICE generation with variable T_end,
  dimensionless time/current/voltage normalization, and stiffness ratio lambda.
- ``PreGeneratedDiodeDataset``: HDF5-backed random-access dataset for fast training
  from pre-computed dimensionless samples.
"""

import logging
import multiprocessing as mp
import queue
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
import torch
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_s
from torch.utils.data import Dataset, IterableDataset

from spino.diode.model import I_SCALE_A

Logging.setup_logging(logging_level="ERROR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

__all__ = [
    "DiodeParameters",
    "DimensionlessDiodeDataset",
    "InfiniteSpiceDiodeDataset",
    "PreGeneratedDiodeDataset",
    "generate_offline_dataset",
]


@dataclass
class DiodeParameters:
    """
    Data Transfer Object for diode circuit parameters.

    Encapsulates the physics definitions to keep method signatures clean.
    All physical units: R (Ohm), C (Farad), Is (Amp), N (dimensionless).
    """

    R_val: float
    C_val: float
    Is_val: float
    N_val: float

    @property
    def tau(self) -> float:
        """Characteristic RC time constant in seconds."""
        return self.R_val * self.C_val

    def to_log_tensor(self, length: int) -> torch.Tensor:
        """
        Converts parameters to the log-normalized tensor format for FNO input.

        :param length: Number of time steps to broadcast each parameter to.
        :return: Tensor of shape [4, length] with channels [logR, logC, logIs, N].
        """
        log_R = np.full(length, np.log10(self.R_val))
        log_C = np.full(length, np.log10(self.C_val))
        log_Is = np.full(length, np.log10(self.Is_val))
        val_N = np.full(length, self.N_val)
        return torch.tensor(np.stack([log_R, log_C, log_Is, val_N]), dtype=torch.float32)

    def to_dimensionless_tensor(self, length: int, lambda_val: float) -> torch.Tensor:
        """
        Converts parameters to dimensionless FNO input format (5 constant channels).

        Channel order: [lambda, log10(R), log10(C), log10(Is), N].
        Each value is broadcast to ``length`` time steps.

        :param length: Number of time steps.
        :param lambda_val: Stiffness ratio RC / T_end.
        :return: Tensor of shape [5, length].
        """
        return torch.tensor(
            np.stack([
                np.full(length, lambda_val),
                np.full(length, np.log10(self.R_val)),
                np.full(length, np.log10(self.C_val)),
                np.full(length, np.log10(self.Is_val)),
                np.full(length, self.N_val),
            ]),
            dtype=torch.float32,
        )


class InfiniteSpiceDiodeDataset(IterableDataset):
    """
    Legacy dataset: on-the-fly SPICE generation with fixed 1ms window.

    Produces 5-channel inputs in raw physical units. Retained for backward
    compatibility with existing checkpoints. New training should use
    ``DimensionlessDiodeDataset`` instead.
    """

    def __init__(self, t_steps: int = 2048, t_end: float = 1e-3):
        self.t_steps = t_steps
        self.t_end = t_end
        # Simulation runs slightly finer than target grid to capture transients before interpolation
        self.sim_step = self.t_end / (self.t_steps * 2.0)

    # =========================================================================
    # Phase 1: Parameter & Signal Generation
    # =========================================================================

    def _sample_random_parameters(self) -> DiodeParameters:
        """
        Generates random circuit parameters using Log-Uniform distributions.

        Ranges tailored to produce sensible voltage swings (+/- 10V) with +/- 5mA drive:
        - R: 50 Ohm to 2 kOhm (prevents huge negative voltage swings)
        - C: 1 pF to 10 nF (ensures transients are visible in 1ms window)
        - Is: 1 fA to 1 nA (standard diode leakage/saturation)
        """
        return DiodeParameters(
            R_val=np.power(10.0, np.random.uniform(1.7, 3.3)),  # ~50 Ohm - 2000 Ohm
            C_val=np.power(10.0, np.random.uniform(-12, -8)),  # 1 pF - 10 nF
            Is_val=np.power(10.0, np.random.uniform(-15, -9)),  # 1 fA - 1 nA
            N_val=np.random.uniform(1.0, 2.0),  # Ideality 1.0 - 2.0
        )

    def _generate_random_pwl_source(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates a random Piecewise Linear (PWL) current waveform."""
        n_points = np.random.randint(5, 20)

        # Random time points sorted within the window
        times = np.sort(np.random.uniform(0, self.t_end, n_points))
        # Ensure start and end points exist
        times = np.concatenate(([0], times, [self.t_end]))

        # Random amplitudes +/- 5mA
        # With R maxing at 2k, this gives +/- 10V swings max, which is reasonable.
        amps = np.random.uniform(-5e-3, 5e-3, len(times))

        return times, amps

    # =========================================================================
    # Phase 2: SPICE Construction & Simulation
    # =========================================================================

    def _build_circuit(self, params: DiodeParameters, times: np.ndarray, amps: np.ndarray) -> Circuit:
        """
        Constructs the PySpice Circuit object.
        Topology: Current Source || Diode || Parallel Resistor
        """
        circuit = Circuit("Diode_Training_Sim")

        # 1. Component Models
        circuit.model("D1", "D", IS=params.Is_val, N=params.N_val, CJO=params.C_val)

        # 2. Sources (PWL)
        # Zip times/amps into pairs for PySpice
        source_pairs = list(zip(times, amps))
        circuit.PieceWiseLinearCurrentSource("1", "0", "1", values=source_pairs)

        # 3. Netlist Topology
        circuit.Diode("1", "1", "0", model="D1")
        circuit.Resistor("1", "1", "0", params.R_val @ u_Ohm)  # Using explicit unit multiplier for safety

        return circuit

    def _run_simulation(self, circuit: Circuit) -> object | None:
        """
        Executes the transient analysis.
        Returns the analysis object or None if convergence fails.
        """
        try:
            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            # Use raw float for steps to avoid unit overhead in inner loop if possible,
            # but PySpice wrapper expects units.
            analysis = simulator.transient(step_time=self.sim_step @ u_s, end_time=self.t_end @ u_s)
            return analysis
        except Exception:
            # Convergence failures are common in random circuits.
            # We catch them and return None to trigger a retry.
            return None

    # =========================================================================
    # Phase 3: Post-Processing & Normalization
    # =========================================================================

    def _process_results(
        self, analysis, source_times: np.ndarray, source_amps: np.ndarray, params: DiodeParameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolates raw SPICE results to fixed grid and normalizes tensors.

        Returns:
            x_tensor: [5, t_steps] -> Channels: [I(mA), logR, logC, logIs, N]
            y_tensor: [1, t_steps] -> Channels: [V(Volts)]
        """
        # 1. Extract raw data
        t_spice = np.array(analysis.time)
        v_spice = np.array(analysis["1"])

        # 2. Define fixed target grid
        t_grid = np.linspace(0, self.t_end, self.t_steps)

        # 3. Interpolate Voltage (Ground Truth)
        v_interp = np.interp(t_grid, t_spice, v_spice)

        # 4. Interpolate Current (Input)
        # We must interpolate the source profile to match the exact grid seen by the network
        i_interp = np.interp(t_grid, source_times, source_amps)

        # 5. Build Input Tensor
        # Channel 0: Current scaled to mA
        ch0_current = torch.tensor(i_interp * 1000.0, dtype=torch.float32)

        # Channels 1-4: Physics Parameters (Log Scaled)
        # Expand scalar parameters to match time dimension
        param_channels = params.to_log_tensor(self.t_steps)

        # Stack: [Current] + [Params] -> [5, T]
        x_tensor = torch.cat([ch0_current.unsqueeze(0), param_channels], dim=0)

        # 6. Build Target Tensor
        y_tensor = torch.tensor(v_interp, dtype=torch.float32).unsqueeze(0)

        return x_tensor, y_tensor

    # =========================================================================
    # Public Interface
    # =========================================================================

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Public API: Generates a completely random training sample.
        """
        # Retry loop for convergence failures
        while True:
            # 1. Sample
            params = self._sample_random_parameters()
            times, amps = self._generate_random_pwl_source()

            # 2. Build
            circuit = self._build_circuit(params, times, amps)

            # 3. Run
            analysis = self._run_simulation(circuit)

            if analysis is not None:
                # 4. Process
                return self._process_results(analysis, times, amps, params)
            # Else: loop continues to try a new random circuit

    def generate_sensible_sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates a deterministic validation sample (approx 1N4148)."""
        params = DiodeParameters(R_val=1000.0, C_val=4e-12, Is_val=2.5e-9, N_val=1.75)
        t1, t2, t3 = self.t_end * 0.2, self.t_end * 0.5, self.t_end * 0.8
        times = np.array([0, t1, t1 + 1e-9, t2, t2 + 1e-9, t3, t3 + 1e-9, self.t_end])
        amps = np.array([0, 0, 5e-3, 5e-3, -2e-3, -2e-3, 0, 0])
        while True:
            circuit = self._build_circuit(params, times, amps)
            if (analysis := self._run_simulation(circuit)) is not None:
                return self._process_results(analysis, times, amps, params)
            logger.warning("Sensible simulation failed convergence. Retrying...")

    def __iter__(self):
        while True:
            yield self.generate_sample()

    def __getitem__(self, index):
        raise NotImplementedError("IterableDataset does not support random access.")


# ---------------------------------------------------------------------------
# Dimensionless Dataset (new)
# ---------------------------------------------------------------------------

def _sample_diode_parameters() -> DiodeParameters:
    """
    Samples random diode circuit parameters from log-uniform distributions.

    Ranges produce sensible voltage swings (+-10V) with +-5mA drive:
    - R:  50 Ohm to 2 kOhm
    - C:  1 pF to 10 nF
    - Is: 1 fA to 1 nA
    - N:  1.0 to 2.0

    :return: Randomly sampled DiodeParameters.
    """
    return DiodeParameters(
        R_val=np.power(10.0, np.random.uniform(1.7, 3.3)),
        C_val=np.power(10.0, np.random.uniform(-12, -8)),
        Is_val=np.power(10.0, np.random.uniform(-15, -9)),
        N_val=np.random.uniform(1.0, 2.0),
    )


def _build_diode_circuit(params: DiodeParameters, times: np.ndarray, amps: np.ndarray) -> Circuit:
    """
    Constructs the PySpice circuit: Current Source || Diode || Resistor.

    :param params: Circuit component values.
    :param times: PWL breakpoint times in seconds.
    :param amps: PWL breakpoint amplitudes in Amps.
    :return: Assembled PySpice Circuit.
    """
    circuit = Circuit("Diode_Training_Sim")
    circuit.model("D1", "D", IS=params.Is_val, N=params.N_val, CJO=params.C_val)
    source_pairs = list(zip(times, amps))
    circuit.PieceWiseLinearCurrentSource("1", "0", "1", values=source_pairs)
    circuit.Diode("1", "1", "0", model="D1")
    circuit.Resistor("1", "1", "0", params.R_val @ u_Ohm)
    return circuit


def _run_diode_simulation(circuit: Circuit, sim_step: float, t_end: float):
    """
    Executes transient analysis, returning analysis object or None on failure.

    :param circuit: Assembled PySpice circuit.
    :param sim_step: Simulation step size in seconds.
    :param t_end: Simulation end time in seconds.
    :return: Analysis object or None if convergence fails.
    """
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        return simulator.transient(step_time=sim_step @ u_s, end_time=t_end @ u_s)
    except Exception:
        return None


class DimensionlessDiodeDataset(IterableDataset):
    """
    On-the-fly SPICE dataset with dimensionless time formulation.

    Each sample has a randomly sampled T_end derived from tau = RC and a
    log-uniform window ratio in [0.1, 100]. Inputs are normalized to
    dimensionless units; the stiffness ratio lambda = RC / T_end is provided
    as an explicit FNO input channel.

    Produces 6-channel inputs: [I_hat, lambda, log10(R), log10(C), log10(Is), N].
    Output: 1-channel V_hat.

    Denormalization: V = V_hat * I_SCALE_A * R.
    """

    # Window-to-tau ratio range drives the initial draw, but T_end is always
    # clamped to [T_END_MIN_S, T_END_MAX_S] before simulation.  This decouples
    # the shape of the lambda distribution from SPICE-safety constraints.
    #
    #   ratio=0.01  -> lambda=100  (very stiff: tau >> T_end)
    #   ratio=1000  -> lambda=0.001 (relaxed: tau << T_end)
    _RATIO_MIN = 0.01
    _RATIO_MAX = 1000.0

    # Hard SPICE-safety bounds on the simulation window.
    #   T_END_MIN_S: sim_step = 10ns/4096 ≈ 2.4ps  (safe for NGSPICE, well above 1ps floor)
    #   T_END_MAX_S: keeps per-sample generation under ~1s wall time
    _T_END_MIN_S = 10e-9
    _T_END_MAX_S = 10e-3

    def __init__(self, t_steps: int = 2048):
        """
        :param t_steps: Number of time points in the output grid.
        """
        self.t_steps = t_steps

    def _sample_t_end(self, params: DiodeParameters) -> tuple[float, float]:
        """
        Derives a variable T_end and lambda from the sampled circuit parameters.

        Samples a window/tau ratio log-uniformly, computes T_end = tau * ratio,
        then clamps T_end to a SPICE-safe range.  Lambda is recomputed from the
        clamped T_end so that it always reflects the actual simulation window.

        Clamping rather than rejection-sampling avoids bias: extreme-tau circuits
        still appear in training but with T_end snapped to the nearest safe value.

        :param params: Sampled circuit parameters (uses tau = RC).
        :return: Tuple of (t_end, lambda_val).
        """
        ratio = np.exp(np.random.uniform(np.log(self._RATIO_MIN), np.log(self._RATIO_MAX)))
        t_end = np.clip(params.tau * ratio, self._T_END_MIN_S, self._T_END_MAX_S)
        lambda_val = params.tau / t_end
        return t_end, lambda_val

    def _generate_random_pwl_source(self, t_end: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a random PWL current in dimensionless amplitude space (+-1).

        Physical current is I_hat * I_SCALE_A. The SPICE simulation receives
        physical Amps; normalization happens in post-processing.

        :param t_end: Simulation window in seconds.
        :return: Tuple of (times_seconds, amps_amps) with physical units.
        """
        n_points = np.random.randint(5, 20)
        times = np.sort(np.random.uniform(0, t_end, n_points))
        times = np.concatenate(([0], times, [t_end]))
        i_hat = np.random.uniform(-1.0, 1.0, len(times))
        amps = i_hat * I_SCALE_A
        return times, amps

    def _process_dimensionless(
        self,
        analysis,
        source_times: np.ndarray,
        source_amps: np.ndarray,
        params: DiodeParameters,
        t_end: float,
        lambda_val: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolates SPICE output to a fixed grid and normalizes to dimensionless form.

        :param analysis: PySpice analysis result.
        :param source_times: PWL breakpoint times (seconds).
        :param source_amps: PWL breakpoint amplitudes (Amps).
        :param params: Circuit parameters.
        :param t_end: Simulation window (seconds).
        :param lambda_val: Stiffness ratio RC / T_end.
        :return: Tuple of (x [6, T], y [1, T]) in dimensionless units.
        """
        t_spice = np.array(analysis.time)
        v_spice = np.array(analysis["1"])
        t_grid = np.linspace(0, t_end, self.t_steps)
        v_interp = np.interp(t_grid, t_spice, v_spice)
        i_interp = np.interp(t_grid, source_times, source_amps)
        v_scale = I_SCALE_A * params.R_val
        i_hat = torch.tensor(i_interp / I_SCALE_A, dtype=torch.float32)
        v_hat = torch.tensor(v_interp / v_scale, dtype=torch.float32)
        param_channels = params.to_dimensionless_tensor(self.t_steps, lambda_val)
        x_tensor = torch.cat([i_hat.unsqueeze(0), param_channels], dim=0)
        y_tensor = v_hat.unsqueeze(0)
        return x_tensor, y_tensor

    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Generates a single dimensionless training sample.

        :return: Tuple of (x [6, T], y [1, T], metadata dict).
        """
        while True:
            params = _sample_diode_parameters()
            t_end, lambda_val = self._sample_t_end(params)
            sim_step = t_end / (self.t_steps * 2.0)
            times, amps = self._generate_random_pwl_source(t_end)
            circuit = _build_diode_circuit(params, times, amps)
            if (analysis := _run_diode_simulation(circuit, sim_step, t_end)) is not None:
                x, y = self._process_dimensionless(analysis, times, amps, params, t_end, lambda_val)
                metadata = {
                    "R": params.R_val,
                    "C": params.C_val,
                    "Is": params.Is_val,
                    "N": params.N_val,
                    "t_end": t_end,
                    "lambda": lambda_val,
                }
                return x, y, metadata

    def __iter__(self):
        while True:
            x, y, _meta = self.generate_sample()
            yield x, y

    def __getitem__(self, index):
        raise NotImplementedError("IterableDataset does not support random access.")


# ---------------------------------------------------------------------------
# Pre-Generated HDF5 Dataset
# ---------------------------------------------------------------------------


class PreGeneratedDiodeDataset(Dataset):
    """
    Loads pre-generated dimensionless diode samples from HDF5.

    HDF5 layout (written by ``generate_offline_dataset``):
    - ``inputs``  : float32 [N, 6, T]  (I_hat, lambda, log10R, log10C, log10Is, N)
    - ``targets`` : float32 [N, 1, T]  (V_hat)
    - ``metadata``: float32 [N, 6]     (R, C, Is, N, t_end, lambda)

    :param hdf5_path: Path to HDF5 file.
    """

    def __init__(self, hdf5_path: str):
        self.file = h5py.File(hdf5_path, "r")
        self.inputs = self.file["inputs"]
        self.targets = self.file["targets"]
        self.num_samples = self.inputs.shape[0]
        logger.info("Loaded pre-generated diode dataset: %d samples from %s", self.num_samples, hdf5_path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample by index.

        :param idx: Sample index.
        :return: Tuple of (x [6, T], y [1, T]) tensors.
        """
        x = torch.from_numpy(self.inputs[idx][:]).float()
        y = torch.from_numpy(self.targets[idx][:]).float()
        return x, y

    def close(self):
        """Closes the HDF5 file handle."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# Offline Dataset Generation (multiprocessing)
# ---------------------------------------------------------------------------


def generate_offline_dataset(
    output_path: str,
    num_samples: int,
    t_steps: int = 2048,
    num_workers: int = 8,
    overwrite: bool = False,
):
    """
    Generates a pre-computed dimensionless diode dataset using multiprocessing.

    Each worker runs ``DimensionlessDiodeDataset.generate_sample`` in a loop
    and writes results to a temporary HDF5 shard. Shards are merged into the
    final output file when all workers complete.

    :param output_path: Destination HDF5 file path.
    :param num_samples: Total number of samples to generate.
    :param t_steps: Time steps per sample.
    :param num_workers: Parallel worker count.
    :param overwrite: If True, overwrite existing file.
    """
    output_file = Path(output_path)
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {output_path}. Use --overwrite to replace.")
    if output_file.exists():
        output_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    in_channels = 6
    meta_cols = 6
    samples_per_worker = num_samples // num_workers
    remainder = num_samples % num_workers
    logger.info("Generating %d dimensionless diode samples with %d workers", num_samples, num_workers)
    temp_dir = output_file.parent / "temp_diode_gen"
    temp_dir.mkdir(parents=True, exist_ok=True)

    def _worker(worker_id: int, n_samples: int, temp_file: str, progress_queue):
        try:
            dataset = DimensionlessDiodeDataset(t_steps=t_steps)
            with h5py.File(temp_file, "w") as f:
                inputs_ds = f.create_dataset("inputs", shape=(n_samples, in_channels, t_steps), dtype="float32")
                targets_ds = f.create_dataset("targets", shape=(n_samples, 1, t_steps), dtype="float32")
                meta_ds = f.create_dataset("metadata", shape=(n_samples, meta_cols), dtype="float64")
                for i in range(n_samples):
                    x, y, meta = dataset.generate_sample()
                    inputs_ds[i] = x.numpy()
                    targets_ds[i] = y.numpy()
                    meta_ds[i] = [meta["R"], meta["C"], meta["Is"], meta["N"], meta["t_end"], meta["lambda"]]
                    progress_queue.put(1)
        except Exception as exc:
            logger.exception("Worker %d failed: %s", worker_id, exc)
            progress_queue.put(-1)
            raise

    progress_queue = mp.Queue()
    processes = []
    temp_files = []
    for wid in range(num_workers):
        worker_n = samples_per_worker + (1 if wid < remainder else 0)
        temp_file = str(temp_dir / f"worker_{wid}.h5")
        temp_files.append(temp_file)
        p = mp.Process(target=_worker, args=(wid, worker_n, temp_file, progress_queue))
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
                if completed % 500 == 0:
                    logger.info("Progress: %d / %d samples", completed, num_samples)
        except queue.Empty:
            pass
    for p in processes:
        p.join()
    if errors > 0:
        raise RuntimeError(f"{errors} worker(s) failed during generation")
    logger.info("Merging %d shards into %s ...", len(temp_files), output_path)
    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("inputs", shape=(num_samples, in_channels, t_steps), dtype="float32", chunks=True)
        f_out.create_dataset("targets", shape=(num_samples, 1, t_steps), dtype="float32", chunks=True)
        f_out.create_dataset("metadata", shape=(num_samples, meta_cols), dtype="float64", chunks=True)
        offset = 0
        for temp_file in temp_files:
            with h5py.File(temp_file, "r") as f_in:
                n = f_in["inputs"].shape[0]
                f_out["inputs"][offset : offset + n] = f_in["inputs"][:]
                f_out["targets"][offset : offset + n] = f_in["targets"][:]
                f_out["metadata"][offset : offset + n] = f_in["metadata"][:]
                offset += n
        f_out.attrs["num_samples"] = num_samples
        f_out.attrs["t_steps"] = t_steps
        f_out.attrs["formulation"] = "dimensionless"
        f_out.attrs["i_scale_a"] = I_SCALE_A
    shutil.rmtree(temp_dir)
    logger.info("Dataset saved: %s (%d samples, %d steps)", output_path, num_samples, t_steps)


# ---------------------------------------------------------------------------
# Debug Visualization
# ---------------------------------------------------------------------------


def visualize_generated_sample(filename: str = "Diode_Sample_Debug.png"):
    """
    Generates one legacy sample and plots transient + I-V curves for debugging.

    :param filename: Output image path.
    """
    dataset = InfiniteSpiceDiodeDataset(t_steps=2048)
    logger.info("Generating SPICE sample...")
    x, y = dataset.generate_sample()
    I_mA = x[0].numpy()
    R_val = np.power(10, x[1, 0].item())
    C_val = np.power(10, x[2, 0].item())
    Is_val = np.power(10, x[3, 0].item())
    N_val = x[4, 0].item()
    V_out = y[0].numpy()
    t_axis = np.linspace(0, dataset.t_end, dataset.t_steps) * 1000
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(t_axis, I_mA, "c", label="Input Current (mA)", alpha=0.7)
    ax[0].set_ylabel("Current (mA)", color="c")
    ax[0].tick_params(axis="y", labelcolor="c")
    ax0_twin = ax[0].twinx()
    ax0_twin.plot(t_axis, V_out, "y", label="Output Voltage (V)", linewidth=2)
    ax0_twin.set_ylabel("Voltage (V)", color="y")
    ax0_twin.tick_params(axis="y", labelcolor="y")
    ax[0].set_title(f"Transient Response\nR={R_val:.0f}, C={C_val:.1e}, Is={Is_val:.1e}, N={N_val:.2f}")
    ax[0].set_xlabel("Time (ms)")
    ax[1].plot(V_out, I_mA, "m.", markersize=2, alpha=0.5)
    ax[1].set_title("Dynamic I-V Hysteresis Loop")
    ax[1].set_xlabel("Voltage (V)")
    ax[1].set_ylabel("Current (mA)")
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    logger.info("Saved debug plot to %s", filename)
    plt.close()


if __name__ == "__main__":
    visualize_generated_sample()
