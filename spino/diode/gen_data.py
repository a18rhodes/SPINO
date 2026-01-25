# %% [markdown]
### Data Generation for Diode Circuit Simulation
# This module defines an infinite dataset that generates random Diode circuits
# and simulates them using NGSPICE (via PySpice/libngspice) on the CPU.

# %%
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import PySpice.Logging.Logging as Logging
import torch
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import u_Ohm, u_s
from torch.utils.data import IterableDataset

# Suppress SPICE console output for speed
Logging.setup_logging(logging_level="ERROR")


# %% [markdown]
### Dataclass to encapsulate Diode Parameters
# This helps keep method signatures clean.


# %%
@dataclass
class DiodeParameters:
    """
    Data Transfer Object (DTO) for Diode Circuit Parameters.
    Encapsulates the physics definitions to keep method signatures clean.
    """

    R_val: float
    C_val: float
    Is_val: float
    N_val: float

    def to_log_tensor(self, length: int) -> torch.Tensor:
        """
        Converts parameters to the log-normalized tensor format expected by the FNO.
        Shape: [4, length] corresponding to channels [logR, logC, logIs, N]
        """
        # Create vectors
        log_R = np.full(length, np.log10(self.R_val))
        log_C = np.full(length, np.log10(self.C_val))
        log_Is = np.full(length, np.log10(self.Is_val))
        val_N = np.full(length, self.N_val)

        return torch.tensor(np.stack([log_R, log_C, log_Is, val_N]), dtype=torch.float32)


# %% [markdown]
### Infinite Dataset Definition
# This dataset generates random diode circuits on-the-fly for training.
# It is designed to be used with multiple workers for parallel SPICE simulation.


# %%
class InfiniteSpiceDiodeDataset(IterableDataset):
    """
    An infinite dataset that generates random Diode circuits and simulates them
    using NGSPICE (via PySpice/libngspice) on the CPU.

    Architecture:
    - Producer-Consumer compatible (designed for num_workers > 0).
    - Modularized pipeline: Sample -> Build -> Simulate -> Tensorize.
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
        """
        Public API: Generates a deterministic validation sample (Approx 1N4148).
        """
        # 1. Define Fixed Parameters (1N4148-ish)
        params = DiodeParameters(R_val=1000.0, C_val=4e-12, Is_val=2.5e-9, N_val=1.75)

        # 2. Define Fixed Waveform (Pulse with Reverse Recovery)
        t1, t2, t3 = self.t_end * 0.2, self.t_end * 0.5, self.t_end * 0.8
        times = np.array([0, t1, t1 + 1e-9, t2, t2 + 1e-9, t3, t3 + 1e-9, self.t_end])
        amps = np.array([0, 0, 5e-3, 5e-3, -2e-3, -2e-3, 0, 0])

        # Retry loop (unlikely to fail for sensible params, but good practice)
        while True:
            circuit = self._build_circuit(params, times, amps)
            analysis = self._run_simulation(circuit)

            if analysis is not None:
                return self._process_results(analysis, times, amps, params)

            # If fixed parameters fail convergence, we are in trouble.
            # But we fallback to random just to keep the pipe flowing if this were real training.
            print("Warning: Sensible simulation failed convergence. Retrying...")

    def __iter__(self):
        while True:
            yield self.generate_sample()

    def __getitem__(self, index):
        raise NotImplementedError("IterableDataset does not support random access.")


# %% [markdown]
### Debugging Utility: Visualize Generated Sample
# This function generates one sample and plots the transient response and IV curve.


# %%
def visualize_generated_sample(filename="Diode_Sample_Debug.png"):
    """
    Debug/Verification utility.
    Generates one sample and plots the transient and IV curves.
    """
    dataset = InfiniteSpiceDiodeDataset(t_steps=2048)
    print("Generating SPICE sample...")
    x, y = dataset.generate_sample()

    # Unpack for plotting
    # x: [I, logR, logC, logIs, N]
    I_mA = x[0].numpy()

    # Recover physical params for title
    R_val = np.power(10, x[1, 0].item())
    C_val = np.power(10, x[2, 0].item())
    Is_val = np.power(10, x[3, 0].item())
    N_val = x[4, 0].item()

    V_out = y[0].numpy()
    t_axis = np.linspace(0, dataset.t_end, dataset.t_steps) * 1000  # ms

    # Plotting setup
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Transient Plot
    ax[0].plot(t_axis, I_mA, "c", label="Input Current (mA)", alpha=0.7)
    ax[0].set_ylabel("Current (mA)", color="c")
    ax[0].tick_params(axis="y", labelcolor="c")

    ax0_twin = ax[0].twinx()
    ax0_twin.plot(t_axis, V_out, "y", label="Output Voltage (V)", linewidth=2)
    ax0_twin.set_ylabel("Voltage (V)", color="y")
    ax0_twin.tick_params(axis="y", labelcolor="y")

    ax[0].set_title(f"Transient Response\nR={R_val:.0f}, C={C_val:.1e}, Is={Is_val:.1e}, N={N_val:.2f}")
    ax[0].set_xlabel("Time (ms)")

    # 2. IV Hysteresis Plot
    ax[1].plot(V_out, I_mA, "m.", markersize=2, alpha=0.5)
    ax[1].set_title("Dynamic I-V Hysteresis Loop")
    ax[1].set_xlabel("Voltage (V)")
    ax[1].set_ylabel("Current (mA)")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved debug plot to {filename}")
    plt.close()


# %%
if __name__ == "__main__":
    visualize_generated_sample()
