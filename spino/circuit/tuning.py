"""
CS amplifier sizing characterization harness.

Performs a 2D :math:`(W_n, W_p)` sweep at fixed channel length against NGSpice,
extracts the canonical small-signal performance metrics (peak gain, output bias,
settling time, static current) at the auto-bias point, and applies a published
selection rule. The output is the artefact set referenced from
``docs/cs_amp.md``.

The harness is split into pure metric extractors (testable without NGSpice) and
SPICE-driven orchestration (skipped under CI when the PDK or NGSpice are not
available).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from spino.circuit.netlist import Circuit
from spino.circuit.simulation import (
    DCSweepResult,
    OperatingPoint,
    TransientResult,
    run_dc_sweep,
    run_operating_point,
    run_transient,
)
from spino.circuit.topologies import build_cs_amp_active_load, build_ota_5t

# pylint: disable=too-many-lines


__all__ = [
    "DesignPoint",
    "Metrics",
    "OtaDesignPoint",
    "OtaMetrics",
    "OtaSelectionRule",
    "OtaSweepResult",
    "SelectionRule",
    "SweepResult",
    "extract_dc_gain",
    "extract_peak_gain",
    "extract_settling_time",
    "extract_slew_rate",
    "extract_slew_time",
    "select_design_point",
    "select_ota_design_point",
    "simulate_design_point",
    "simulate_ota_design_point",
    "sweep_design_space",
    "sweep_ota_design_space",
]

logger = logging.getLogger(__name__)

_OUTPUT_NODE = "v(out)"
_SUPPLY_CURRENT = "i(vdd)"


@dataclass(frozen=True, slots=True)
class DesignPoint:
    """
    A single ``(W_n, W_p)`` sizing in microns at fixed channel length.

    :param nfet_w_um: NFET channel width in microns.
    :param pfet_w_um: PFET channel width in microns.
    """

    nfet_w_um: float
    pfet_w_um: float


@dataclass(frozen=True, slots=True)
class Metrics:
    """
    Extracted small-signal performance metrics for a CS amplifier sizing.

    Failed simulations are encoded with ``converged=False`` and ``NaN`` field
    values so downstream code can filter without sentinels.

    :param converged: True when all three SPICE analyses (.op, .dc, .tran) succeeded.
    :param peak_gain_v_per_v: Maximum :math:`|dV_{out}/dV_{in}|` along the VTC.
    :param vin_at_peak_gain_v: VTC input voltage at which the peak gain occurs.
    :param vout_at_peak_gain_v: Output voltage at the peak-gain bias point.
    :param static_current_a: Magnitude of supply current at the peak-gain bias.
    :param settling_time_s: 5%-band settling time of the step response from
        ``t_step_start``; ``NaN`` if the response never settles within the
        simulation window.
    """

    converged: bool
    peak_gain_v_per_v: float
    vin_at_peak_gain_v: float
    vout_at_peak_gain_v: float
    static_current_a: float
    settling_time_s: float

    @classmethod
    def failed(cls) -> "Metrics":
        """
        Returns a sentinel ``Metrics`` for designs that did not converge.

        :return: A non-converged metrics instance with all numeric fields set to NaN.
        """
        nan = float("nan")
        return cls(False, nan, nan, nan, nan, nan)


@dataclass(frozen=True, slots=True)
class SelectionRule:
    """
    A-priori selection criterion for the CS amp design point.

    The rule must be defined before the sweep is run to avoid post-hoc
    cherry-picking. The default keeps the bias clear of the rails so the
    chosen design has a usable linear region.

    :param vout_min_v: Minimum acceptable output bias voltage in volts.
    :param vout_max_v: Maximum acceptable output bias voltage in volts.
    """

    vout_min_v: float = 0.6
    vout_max_v: float = 1.2

    def admits(self, metrics: Metrics) -> bool:
        """
        Tests whether a given metrics set is feasible under this rule.

        :param metrics: Extracted metrics for one design point.
        :return: True when the design converged and its bias falls inside the band.
        """
        if not metrics.converged:
            return False
        return self.vout_min_v <= metrics.vout_at_peak_gain_v <= self.vout_max_v


@dataclass(frozen=True, slots=True)
class SweepResult:
    """
    Full design-space sweep over :math:`W_n \\times W_p` at fixed channel length.

    :param points: Design points evaluated, in row-major (W_n, W_p) order.
    :param metrics: Metrics aligned 1:1 with ``points``.
    """

    points: tuple[DesignPoint, ...]
    metrics: tuple[Metrics, ...]

    def gain_grid(self) -> NDArray[np.float64]:
        """
        Reshapes the per-point peak-gain values into a 2D grid for plotting.

        Failed points are encoded as ``NaN`` so colormaps can mask them.

        :return: Array of shape ``(len(unique W_n), len(unique W_p))`` containing
            ``|peak gain|`` in V/V.
        """
        n_axis = sorted({p.nfet_w_um for p in self.points})
        p_axis = sorted({p.pfet_w_um for p in self.points})
        grid = np.full((len(n_axis), len(p_axis)), np.nan, dtype=np.float64)
        for point, metric in zip(self.points, self.metrics):
            i = n_axis.index(point.nfet_w_um)
            j = p_axis.index(point.pfet_w_um)
            grid[i, j] = abs(metric.peak_gain_v_per_v) if metric.converged else np.nan
        return grid

    def axes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the unique W_n and W_p axis values in ascending order.

        :return: Tuple of ``(nfet_w_axis, pfet_w_axis)`` arrays in microns.
        """
        n_axis = np.array(sorted({p.nfet_w_um for p in self.points}), dtype=np.float64)
        p_axis = np.array(sorted({p.pfet_w_um for p in self.points}), dtype=np.float64)
        return n_axis, p_axis


def extract_peak_gain(vtc: DCSweepResult) -> tuple[float, float, float]:
    """
    Extracts the peak small-signal gain from a VTC sweep.

    Uses a centred finite difference (``np.gradient``) so the derivative is
    well-defined at every interior sample. The location of the maximum
    :math:`|dV_{out}/dV_{in}|` defines the auto-bias point used downstream.

    :param vtc: NGSpice DC sweep result containing the ``v(out)`` trace and the
        swept :math:`V_{in}` values.
    :return: Tuple ``(|peak gain|, V_in_at_peak, V_out_at_peak)`` in V/V, V, V.
    """
    vin = vtc.sweep_values
    vout = vtc.variables[_OUTPUT_NODE]
    gain = np.gradient(vout, vin)
    idx = int(np.argmax(np.abs(gain)))
    return float(abs(gain[idx])), float(vin[idx]), float(vout[idx])


def extract_settling_time(tran: TransientResult, *, t_step_start: float, settle_band: float = 0.05) -> float:
    """
    Computes the band-settling time of a step response.

    The settling time is the smallest :math:`t_s \\geq t_{step\\_start}` such that
    :math:`|V_{out}(t) - V_{out}(\\infty)| < \\epsilon \\cdot |\\Delta V_{out}|` for
    all :math:`t \\geq t_s`. The asymptote :math:`V_{out}(\\infty)` is estimated
    from the last 1% of samples; the step magnitude is estimated from the
    difference between the final and initial means.

    :param tran: Transient result containing ``v(out)`` and the time vector.
    :param t_step_start: Time at which the input step occurs in seconds.
    :param settle_band: Tolerance band as a fraction of step magnitude.
    :return: Settling time in seconds; ``NaN`` if the response never settles
        within the simulated window.
    """
    time = tran.time
    vout = tran.variables[_OUTPUT_NODE]
    tail = max(1, len(vout) // 100)
    head = max(1, len(vout) // 100)
    v_final = float(vout[-tail:].mean())
    v_initial = float(vout[:head].mean())
    step_magnitude = abs(v_final - v_initial)
    if step_magnitude < 1e-9:
        return 0.0
    band = settle_band * step_magnitude
    in_band = np.abs(vout - v_final) < band
    if in_band.all():
        return 0.0
    last_outside = int(np.where(~in_band)[0].max())
    if last_outside + 1 >= len(time):
        return float("nan")
    return max(float(time[last_outside + 1]) - t_step_start, 0.0)


def _build(  # pylint: disable=too-many-arguments
    point: DesignPoint,
    *,
    vdd: float,
    vin_dc: float,
    vin_tran: str = "",
    nfet_l_um: float,
    pfet_l_um: float,
    pdk_root: str | None,
) -> Circuit:
    """
    Builds a CS amp ``Circuit`` for the given design point and bias.

    :param point: Sizing under test.
    :param vdd: Supply voltage in volts.
    :param vin_dc: DC input bias voltage in volts.
    :param vin_tran: Transient stimulus string for ``Vin``.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param pdk_root: Optional override for the PDK root path.
    :return: Circuit instance ready for SPICE simulation.
    """
    kwargs = {
        "nfet_w": point.nfet_w_um,
        "nfet_l": nfet_l_um,
        "pfet_w": point.pfet_w_um,
        "pfet_l": pfet_l_um,
        "vdd": vdd,
        "vin_dc": vin_dc,
        "vin_tran": vin_tran,
    }
    if pdk_root is not None:
        kwargs["pdk_root"] = pdk_root
    return build_cs_amp_active_load(**kwargs)


def _vtc(  # pylint: disable=too-many-arguments
    point: DesignPoint, *, vdd: float, nfet_l_um: float, pfet_l_um: float, pdk_root: str | None, step_v: float
) -> DCSweepResult | None:
    """
    Runs the VTC sweep for one design point.

    :param point: Sizing under test.
    :param vdd: Supply voltage in volts.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param pdk_root: Optional override for the PDK root path.
    :param step_v: VTC sweep step in volts.
    :return: DC sweep result, or ``None`` on simulator failure.
    """
    circuit = _build(point, vdd=vdd, vin_dc=vdd / 2.0, nfet_l_um=nfet_l_um, pfet_l_um=pfet_l_um, pdk_root=pdk_root)
    return run_dc_sweep(circuit, source_name="Vin", start=0.0, stop=vdd, step=step_v)


def _operating_point(  # pylint: disable=too-many-arguments
    point: DesignPoint,
    *,
    vdd: float,
    vin_dc: float,
    nfet_l_um: float,
    pfet_l_um: float,
    pdk_root: str | None,
) -> OperatingPoint | None:
    """
    Runs a DC operating point at a chosen bias for one design.

    :param point: Sizing under test.
    :param vdd: Supply voltage in volts.
    :param vin_dc: DC input bias voltage in volts.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param pdk_root: Optional override for the PDK root path.
    :return: Operating point, or ``None`` on simulator failure.
    """
    circuit = _build(point, vdd=vdd, vin_dc=vin_dc, nfet_l_um=nfet_l_um, pfet_l_um=pfet_l_um, pdk_root=pdk_root)
    return run_operating_point(circuit)


def _step_response(  # pylint: disable=too-many-arguments
    point: DesignPoint,
    *,
    vdd: float,
    vin_dc: float,
    vin_step_amplitude: float,
    nfet_l_um: float,
    pfet_l_um: float,
    pdk_root: str | None,
    t_step_start: float,
    t_end: float,
    t_step: float,
) -> TransientResult | None:
    """
    Runs the transient step response for one design at its bias.

    :param point: Sizing under test.
    :param vdd: Supply voltage in volts.
    :param vin_dc: DC input bias voltage in volts (pre-step level).
    :param vin_step_amplitude: Amplitude of the small-signal step in volts.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param pdk_root: Optional override for the PDK root path.
    :param t_step_start: Time at which the input step occurs in seconds.
    :param t_end: Total simulation duration in seconds.
    :param t_step: Maximum SPICE timestep in seconds.
    :return: Transient result, or ``None`` on simulator failure.
    """
    pwl = (
        f"PWL(0 {vin_dc} {t_step_start * 0.5} {vin_dc} "
        f"{t_step_start} {vin_dc + vin_step_amplitude} "
        f"{t_end} {vin_dc + vin_step_amplitude})"
    )
    circuit = _build(
        point,
        vdd=vdd,
        vin_dc=vin_dc,
        vin_tran=pwl,
        nfet_l_um=nfet_l_um,
        pfet_l_um=pfet_l_um,
        pdk_root=pdk_root,
    )
    return run_transient(circuit, t_step=t_step, t_end=t_end)


def simulate_design_point(  # pylint: disable=too-many-arguments,too-many-locals
    point: DesignPoint,
    *,
    vdd: float = 1.8,
    nfet_l_um: float = 0.18,
    pfet_l_um: float = 0.18,
    vtc_step_v: float = 0.01,
    vin_step_amplitude: float = 0.05,
    t_step_start: float = 100e-9,
    t_end: float = 5e-6,
    t_step: float = 10e-9,
    pdk_root: str | None = None,
) -> Metrics:
    """
    Runs the full CS amp characterization on a single ``(W_n, W_p)`` design.

    The sequence is: VTC sweep to locate the peak-gain bias, DC operating point
    at that bias for static current and bias verification, transient step
    around the bias for settling time. Any SPICE failure short-circuits to a
    ``Metrics.failed()`` sentinel.

    :param point: Sizing under test.
    :param vdd: Supply voltage in volts.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param vtc_step_v: Resolution of the VTC sweep in volts.
    :param vin_step_amplitude: Amplitude of the small-signal step in volts.
    :param t_step_start: Time at which the step occurs in seconds.
    :param t_end: Transient simulation duration in seconds.
    :param t_step: Maximum SPICE timestep in seconds.
    :param pdk_root: Optional override for the PDK root path.
    :return: Metrics for this design point; converged flag indicates validity.
    """
    if (
        vtc := _vtc(point, vdd=vdd, nfet_l_um=nfet_l_um, pfet_l_um=pfet_l_um, pdk_root=pdk_root, step_v=vtc_step_v)
    ) is None:
        logger.warning("VTC sweep failed for %s", point)
        return Metrics.failed()
    peak_gain, vin_peak, vout_peak = extract_peak_gain(vtc)
    if (
        op := _operating_point(
            point, vdd=vdd, vin_dc=vin_peak, nfet_l_um=nfet_l_um, pfet_l_um=pfet_l_um, pdk_root=pdk_root
        )
    ) is None:
        logger.warning("Operating point failed for %s at vin=%.4f V", point, vin_peak)
        return Metrics.failed()
    static_current = abs(float(op.variables[_SUPPLY_CURRENT]))
    if (
        tran := _step_response(
            point,
            vdd=vdd,
            vin_dc=vin_peak,
            vin_step_amplitude=vin_step_amplitude,
            nfet_l_um=nfet_l_um,
            pfet_l_um=pfet_l_um,
            pdk_root=pdk_root,
            t_step_start=t_step_start,
            t_end=t_end,
            t_step=t_step,
        )
    ) is None:
        logger.warning("Transient failed for %s at vin=%.4f V", point, vin_peak)
        return Metrics.failed()
    settling = extract_settling_time(tran, t_step_start=t_step_start)
    return Metrics(
        converged=True,
        peak_gain_v_per_v=peak_gain,
        vin_at_peak_gain_v=vin_peak,
        vout_at_peak_gain_v=vout_peak,
        static_current_a=static_current,
        settling_time_s=settling,
    )


def sweep_design_space(  # pylint: disable=too-many-arguments
    nfet_widths_um: tuple[float, ...],
    pfet_widths_um: tuple[float, ...],
    *,
    vdd: float = 1.8,
    nfet_l_um: float = 0.18,
    pfet_l_um: float = 0.18,
    pdk_root: str | None = None,
    progress: Callable[[int, int, DesignPoint, Metrics], None] | None = None,
) -> SweepResult:
    """
    Evaluates ``simulate_design_point`` over the Cartesian product of widths.

    :param nfet_widths_um: NFET channel widths to sweep in microns.
    :param pfet_widths_um: PFET channel widths to sweep in microns.
    :param vdd: Supply voltage in volts.
    :param nfet_l_um: NFET channel length in microns.
    :param pfet_l_um: PFET channel length in microns.
    :param pdk_root: Optional override for the PDK root path.
    :param progress: Optional callback invoked after each design with
        ``(index, total, point, metrics)``.
    :return: Sweep result with points in row-major order.
    """
    points = tuple(DesignPoint(nfet_w_um=wn, pfet_w_um=wp) for wn in nfet_widths_um for wp in pfet_widths_um)
    total = len(points)
    metrics: list[Metrics] = []
    for idx, point in enumerate(points):
        result = simulate_design_point(point, vdd=vdd, nfet_l_um=nfet_l_um, pfet_l_um=pfet_l_um, pdk_root=pdk_root)
        metrics.append(result)
        if progress is not None:
            progress(idx, total, point, result)
    return SweepResult(points=points, metrics=tuple(metrics))


def select_design_point(sweep: SweepResult, rule: SelectionRule) -> tuple[DesignPoint, Metrics]:
    """
    Applies the selection rule and returns the chosen design.

    The primary objective is maximum :math:`|peak\\_gain|`; ties are broken by
    minimum static current (most efficient operating point).

    :param sweep: Completed sweep result.
    :param rule: A-priori feasibility predicate.
    :return: Tuple ``(chosen point, chosen metrics)``.
    :raises ValueError: When no point in the sweep satisfies ``rule``.
    """
    feasible = [(p, m) for p, m in zip(sweep.points, sweep.metrics) if rule.admits(m)]
    if not feasible:
        raise ValueError("No design point in the sweep satisfies the selection rule.")
    feasible.sort(key=lambda pm: (-abs(pm[1].peak_gain_v_per_v), pm[1].static_current_a))
    return feasible[0]


# ---------------------------------------------------------------------------
# 5T OTA characterization harness
# ---------------------------------------------------------------------------

_OTA_OUTPUT_NODE = "v(n_out)"
_OTA_SUPPLY_CURRENT = "i(vdd)"


@dataclass(frozen=True, slots=True)
class OtaDesignPoint:
    """
    A single ``(W_diff, W_mirror)`` sizing in microns at fixed channel lengths.

    :param diff_w_um: Differential-pair MOSFET width (M1, M2) in microns.
    :param mirror_w_um: Current-mirror MOSFET width (M3, M4) in microns.
    """

    diff_w_um: float
    mirror_w_um: float


@dataclass(frozen=True, slots=True)
class OtaMetrics:
    """
    Extracted large-signal performance metrics for one OTA sizing.

    Failed simulations are encoded with ``converged=False`` and ``NaN`` fields.

    :param converged: True when the transient simulation succeeded.
    :param slew_rate_v_per_us: Peak :math:`|dV_{out}/dt|` after the step (V/µs).
    :param slew_time_ns: 10–90 % of output swing duration (ns).
    :param peak_swing_v: Total output voltage swing observed post-step (V).
    :param static_current_a: Quiescent supply current magnitude (A).
    :param dc_gain_v_per_v: Small-signal open-loop gain from DC sweep (V/V).
        Reported for information only; not used by the selection rule.
    :param quiescent_n_out_v: Quiescent output node voltage (V).
    """

    converged: bool
    slew_rate_v_per_us: float
    slew_time_ns: float
    peak_swing_v: float
    static_current_a: float
    dc_gain_v_per_v: float
    quiescent_n_out_v: float

    @classmethod
    def failed(cls) -> "OtaMetrics":
        """Returns a sentinel instance for designs that did not converge."""
        nan = float("nan")
        return cls(False, nan, nan, nan, nan, nan, nan)


@dataclass(frozen=True, slots=True)
class OtaSelectionRule:
    """
    Pre-registered selection rule for the 5T OTA sweep.

    Feasibility requires both conditions; ranking is by descending slew rate
    with ascending slew time as the tiebreaker.

    :param slew_min_v_per_us: Minimum acceptable slew rate (V/µs).
    :param slew_time_max_ns: Maximum acceptable 10–90 % slew duration (ns).
    """

    slew_min_v_per_us: float = 5.0
    slew_time_max_ns: float = 500.0

    def admits(self, metrics: OtaMetrics) -> bool:
        """
        Tests whether a metrics set is feasible under this rule.

        :param metrics: Extracted metrics for one design point.
        :return: True when the design converged and both slew criteria are met.
        """
        if not metrics.converged:
            return False
        return metrics.slew_rate_v_per_us >= self.slew_min_v_per_us and metrics.slew_time_ns <= self.slew_time_max_ns


@dataclass(frozen=True, slots=True)
class OtaSweepResult:
    """
    Full design-space sweep over :math:`W_{diff} \\times W_{mirror}`.

    :param points: Design points evaluated, in row-major order.
    :param metrics: Metrics aligned 1:1 with ``points``.
    """

    points: tuple[OtaDesignPoint, ...]
    metrics: tuple[OtaMetrics, ...]

    def slew_grid(self) -> NDArray[np.float64]:
        """
        Reshapes per-point slew rate into a 2D grid for plotting.

        :return: Array of shape ``(len(unique W_diff), len(unique W_mirror))``
            with slew rate in V/µs; ``NaN`` for failed points.
        """
        diff_axis = sorted({p.diff_w_um for p in self.points})
        mirror_axis = sorted({p.mirror_w_um for p in self.points})
        grid = np.full((len(diff_axis), len(mirror_axis)), np.nan, dtype=np.float64)
        for point, metric in zip(self.points, self.metrics):
            i = diff_axis.index(point.diff_w_um)
            j = mirror_axis.index(point.mirror_w_um)
            grid[i, j] = metric.slew_rate_v_per_us if metric.converged else np.nan
        return grid

    def slew_time_grid(self) -> NDArray[np.float64]:
        """
        Reshapes per-point slew time into a 2D grid for plotting.

        :return: Array of shape ``(len(unique W_diff), len(unique W_mirror))``
            with slew time in ns; ``NaN`` for failed points.
        """
        diff_axis = sorted({p.diff_w_um for p in self.points})
        mirror_axis = sorted({p.mirror_w_um for p in self.points})
        grid = np.full((len(diff_axis), len(mirror_axis)), np.nan, dtype=np.float64)
        for point, metric in zip(self.points, self.metrics):
            i = diff_axis.index(point.diff_w_um)
            j = mirror_axis.index(point.mirror_w_um)
            grid[i, j] = metric.slew_time_ns if metric.converged else np.nan
        return grid

    def axes(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns the unique W_diff and W_mirror axis values in ascending order.

        :return: Tuple of ``(diff_w_axis, mirror_w_axis)`` arrays in microns.
        """
        diff_axis = np.array(sorted({p.diff_w_um for p in self.points}), dtype=np.float64)
        mirror_axis = np.array(sorted({p.mirror_w_um for p in self.points}), dtype=np.float64)
        return diff_axis, mirror_axis


def extract_slew_rate(
    tran: TransientResult,
    *,
    t_step_start: float,
    output_node: str = _OTA_OUTPUT_NODE,
) -> float:
    """
    Computes the peak slew rate of the OTA output after a large-signal step.

    Uses a centred finite difference (``np.gradient``) on the post-step region
    to find the peak :math:`|dV_{out}/dt|`.

    :param tran: Transient simulation result.
    :param t_step_start: Time at which the step occurs (seconds).
    :param output_node: Variable name for the output node (default ``v(n_out)``).
    :return: Peak slew rate in V/µs; ``NaN`` if the post-step region is empty.
    """
    time = tran.time
    vout = tran.variables[output_node]
    mask = time >= t_step_start
    if not mask.any() or mask.sum() < 2:
        return float("nan")
    t_post = time[mask]
    v_post = vout[mask]
    dv_dt = np.gradient(v_post, t_post)  # V/s
    return float(np.max(np.abs(dv_dt))) * 1e-6  # V/µs


def extract_slew_time(  # pylint: disable=too-many-locals
    tran: TransientResult,
    *,
    t_step_start: float,
    output_node: str = _OTA_OUTPUT_NODE,
) -> float:
    """
    Computes the 10–90 % slew time of the OTA output after a large-signal step.

    The swing is measured between the mean of the first 5 % and the mean of
    the last 5 % of post-step samples. The 10 % and 90 % threshold crossings
    are located as the first samples that exceed the respective threshold in the
    direction of the step.

    :param tran: Transient simulation result.
    :param t_step_start: Time at which the step occurs (seconds).
    :param output_node: Variable name for the output node.
    :return: 10–90 % slew time in nanoseconds; ``NaN`` when the swing is
        negligible or the thresholds are not crossed.
    """
    time = tran.time
    vout = tran.variables[output_node]
    mask = time >= t_step_start
    if not mask.any() or mask.sum() < 4:
        return float("nan")
    t_post = time[mask]
    v_post = vout[mask]
    # v_post[0] is at t=t_step_start where the PWL is still at quiescent (ramp not yet started)
    v_initial = float(v_post[0])
    tail = max(1, len(v_post) // 20)
    v_final = float(v_post[-tail:].mean())
    swing = abs(v_final - v_initial)
    if swing < 1e-3:
        return float("nan")
    rising = v_final > v_initial
    direction = 1.0 if rising else -1.0
    v10 = v_initial + direction * 0.1 * swing
    v90 = v_initial + direction * 0.9 * swing
    if rising:
        above10 = v_post >= v10
        above90 = v_post >= v90
    else:
        above10 = v_post <= v10
        above90 = v_post <= v90
    if not above10.any() or not above90.any():
        return float("nan")
    idx10 = int(np.argmax(above10))
    idx90 = int(np.argmax(above90))
    if idx90 <= idx10:
        return float("nan")
    return max((float(t_post[idx90]) - float(t_post[idx10])) * 1e9, 0.0)


def extract_dc_gain(dc_result: DCSweepResult, *, output_node: str = _OTA_OUTPUT_NODE) -> float:
    """
    Extracts the peak small-signal open-loop gain from an OTA DC sweep.

    Sweeps Vinp with Vinn fixed at Vcm. The gain is computed as
    :math:`|dV_{out}/dV_{inp}|` via ``np.gradient``.

    :param dc_result: DC sweep result (Vinp as the swept source).
    :param output_node: Variable name for the output node.
    :return: Peak gain magnitude in V/V.
    """
    vinp = dc_result.sweep_values
    vout = dc_result.variables[output_node]
    gain = np.gradient(vout, vinp)
    return float(np.max(np.abs(gain)))


def _ota_differential_step_pwl_strings(
    *,
    vcm_v: float,
    t_step_start: float,
    rise_time_s: float,
    t_end: float,
    step_amp_v: float,
) -> tuple[str, str]:
    """
    Builds Vinp/Vinn piecewise-linear sources for the registered differential step.

    :param vcm_v: Input common-mode voltage.
    :param t_step_start: Step onset time (s).
    :param rise_time_s: Input rise duration (s).
    :param t_end: Simulation end time (s).
    :param step_amp_v: Differential half-amplitude (V).
    :return: ``(vinp_pwl, vinn_pwl)`` SPICE PWL strings.
    """
    t_rise_end = t_step_start + rise_time_s
    vinp_pwl = (
        f"PWL(0 {vcm_v} {t_step_start} {vcm_v} " f"{t_rise_end} {vcm_v + step_amp_v} {t_end} {vcm_v + step_amp_v})"
    )
    vinn_pwl = (
        f"PWL(0 {vcm_v} {t_step_start} {vcm_v} " f"{t_rise_end} {vcm_v - step_amp_v} {t_end} {vcm_v - step_amp_v})"
    )
    return vinp_pwl, vinn_pwl


def _ota_op(  # pylint: disable=too-many-arguments
    point: OtaDesignPoint,
    *,
    vdd: float,
    vcm_v: float,
    diff_l_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    vbias_v: float,
    pdk_root: str | None,
) -> OperatingPoint | None:
    """Runs a DC operating point for the OTA at the quiescent bias."""
    kwargs = {
        "diff_w_um": point.diff_w_um,
        "diff_l_um": diff_l_um,
        "mirror_w_um": point.mirror_w_um,
        "mirror_l_um": mirror_l_um,
        "tail_w_um": tail_w_um,
        "tail_l_um": tail_l_um,
        "vdd": vdd,
        "vbias_v": vbias_v,
        "vcm_v": vcm_v,
    }
    if pdk_root is not None:
        kwargs["pdk_root"] = pdk_root
    return run_operating_point(build_ota_5t(**kwargs))


def _ota_tran(  # pylint: disable=too-many-arguments,too-many-locals
    point: OtaDesignPoint,
    *,
    vdd: float,
    vcm_v: float,
    step_amp_v: float,
    rise_time_s: float,
    diff_l_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    vbias_v: float,
    t_step_start: float,
    t_end: float,
    t_step: float,
    c_load_f: float,
    pdk_root: str | None,
) -> TransientResult | None:
    """Runs the large-signal differential-step transient for one OTA design."""
    vinp_pwl, vinn_pwl = _ota_differential_step_pwl_strings(
        vcm_v=vcm_v,
        t_step_start=t_step_start,
        rise_time_s=rise_time_s,
        t_end=t_end,
        step_amp_v=step_amp_v,
    )
    kwargs = {
        "diff_w_um": point.diff_w_um,
        "diff_l_um": diff_l_um,
        "mirror_w_um": point.mirror_w_um,
        "mirror_l_um": mirror_l_um,
        "tail_w_um": tail_w_um,
        "tail_l_um": tail_l_um,
        "vdd": vdd,
        "vbias_v": vbias_v,
        "vcm_v": vcm_v,
        "vinp_tran": vinp_pwl,
        "vinn_tran": vinn_pwl,
        "c_load_f": c_load_f,
    }
    if pdk_root is not None:
        kwargs["pdk_root"] = pdk_root
    return run_transient(build_ota_5t(**kwargs), t_step=t_step, t_end=t_end)


def _ota_dc_gain_sweep(  # pylint: disable=too-many-arguments
    point: OtaDesignPoint,
    *,
    vdd: float,
    vcm_v: float,
    diff_l_um: float,
    mirror_l_um: float,
    tail_w_um: float,
    tail_l_um: float,
    vbias_v: float,
    dc_sweep_amp_v: float,
    dc_sweep_step_v: float,
    pdk_root: str | None,
) -> DCSweepResult | None:
    """Runs a small-signal DC sweep of Vinp (±dc_sweep_amp_v) with Vinn at Vcm."""
    kwargs = {
        "diff_w_um": point.diff_w_um,
        "diff_l_um": diff_l_um,
        "mirror_w_um": point.mirror_w_um,
        "mirror_l_um": mirror_l_um,
        "tail_w_um": tail_w_um,
        "tail_l_um": tail_l_um,
        "vdd": vdd,
        "vbias_v": vbias_v,
        "vcm_v": vcm_v,
    }
    if pdk_root is not None:
        kwargs["pdk_root"] = pdk_root
    circuit = build_ota_5t(**kwargs)
    return run_dc_sweep(
        circuit,
        source_name="Vinp",
        start=vcm_v - dc_sweep_amp_v,
        stop=vcm_v + dc_sweep_amp_v,
        step=dc_sweep_step_v,
    )


def simulate_ota_design_point(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    point: OtaDesignPoint,
    *,
    vdd: float = 1.8,
    vcm_v: float = 0.9,
    step_amp_v: float = 0.05,
    rise_time_s: float = 5e-9,
    diff_l_um: float = 0.40,
    mirror_l_um: float = 0.40,
    tail_w_um: float = 2.0,
    tail_l_um: float = 0.40,
    vbias_v: float = 1.2,
    t_step_start: float = 100e-9,
    t_end: float = 500e-9,
    t_step: float = 1e-9,
    c_load_f: float = 1e-12,
    dc_sweep_amp_v: float = 0.02,
    dc_sweep_step_v: float = 0.001,
    pdk_root: str | None = None,
) -> OtaMetrics:
    """
    Runs the full OTA characterization sequence for one ``(W_diff, W_mirror)`` design.

    Three SPICE analyses are executed in sequence:

    1. DC operating point — quiescent supply current and output bias.
    2. Large-signal differential-step transient — slew rate, slew time, peak swing.
    3. Small-signal Vinp DC sweep — open-loop gain (reported only, not gated).

    Any transient failure short-circuits to ``OtaMetrics.failed()``. DC OP and
    gain-sweep failures produce ``NaN`` for the affected fields while
    ``converged`` remains ``True``.

    :param point: Sizing under test.
    :param vdd: Supply voltage.
    :param vcm_v: Input common-mode voltage.
    :param step_amp_v: Differential step half-amplitude (each input steps ±this).
    :param rise_time_s: Input step rise time in seconds (pre-registered: 5 ns).
    :param diff_l_um: Differential-pair channel length.
    :param mirror_l_um: Mirror channel length.
    :param tail_w_um: Tail current-source width.
    :param tail_l_um: Tail current-source length.
    :param vbias_v: Tail gate bias voltage.
    :param t_step_start: Step onset time in seconds.
    :param t_end: Total simulation window in seconds.
    :param t_step: SPICE maximum timestep.
    :param c_load_f: Load capacitance at ``n_out`` in farads. Unlike the CS amp
        where ``c_load`` is only aesthetic, this value directly defines the slew
        metric: ``slew_rate = I_tail / c_load_f``. Must be nonzero for
        meaningful slew-rate and slew-time measurements.
    :param dc_sweep_amp_v: Half-range of the gain sweep (V).
    :param dc_sweep_step_v: DC sweep resolution (V).
    :param pdk_root: Optional PDK root override.
    :return: Extracted metrics; ``converged`` reflects transient success.
    """
    shared = {
        "vdd": vdd,
        "vcm_v": vcm_v,
        "diff_l_um": diff_l_um,
        "mirror_l_um": mirror_l_um,
        "tail_w_um": tail_w_um,
        "tail_l_um": tail_l_um,
        "vbias_v": vbias_v,
        "pdk_root": pdk_root,
    }

    op = _ota_op(point, **shared)
    if op is not None:
        static_current = abs(float(op.variables.get(_OTA_SUPPLY_CURRENT, float("nan"))))
        quiescent_n_out_v = float(op.variables.get(_OTA_OUTPUT_NODE, float("nan")))
    else:
        logger.warning("DC operating point failed for OTA %s", point)
        static_current = float("nan")
        quiescent_n_out_v = float("nan")

    tran = _ota_tran(
        point,
        step_amp_v=step_amp_v,
        rise_time_s=rise_time_s,
        t_step_start=t_step_start,
        t_end=t_end,
        t_step=t_step,
        c_load_f=c_load_f,
        **shared,
    )
    if tran is None:
        logger.warning("Transient failed for OTA %s", point)
        return OtaMetrics.failed()

    slew_rate = extract_slew_rate(tran, t_step_start=t_step_start)
    slew_time = extract_slew_time(tran, t_step_start=t_step_start)
    vout_post = tran.variables[_OTA_OUTPUT_NODE][tran.time >= t_step_start]
    head = max(1, len(vout_post) // 20)
    tail = max(1, len(vout_post) // 20)
    peak_swing = float(abs(vout_post[-tail:].mean() - vout_post[:head].mean())) if len(vout_post) >= 4 else float("nan")

    dc_sweep = _ota_dc_gain_sweep(
        point,
        dc_sweep_amp_v=dc_sweep_amp_v,
        dc_sweep_step_v=dc_sweep_step_v,
        **shared,
    )
    dc_gain = extract_dc_gain(dc_sweep) if dc_sweep is not None else float("nan")

    return OtaMetrics(
        converged=True,
        slew_rate_v_per_us=slew_rate,
        slew_time_ns=slew_time,
        peak_swing_v=peak_swing,
        static_current_a=static_current,
        dc_gain_v_per_v=dc_gain,
        quiescent_n_out_v=quiescent_n_out_v,
    )


def sweep_ota_design_space(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    diff_widths_um: tuple[float, ...],
    mirror_widths_um: tuple[float, ...],
    *,
    vdd: float = 1.8,
    vcm_v: float = 0.9,
    step_amp_v: float = 0.05,
    rise_time_s: float = 5e-9,
    diff_l_um: float = 0.40,
    mirror_l_um: float = 0.40,
    tail_w_um: float = 2.0,
    tail_l_um: float = 0.40,
    vbias_v: float = 1.2,
    t_step_start: float = 100e-9,
    t_end: float = 500e-9,
    t_step: float = 1e-9,
    c_load_f: float = 1e-12,
    pdk_root: str | None = None,
    progress: Callable[[int, int, OtaDesignPoint, OtaMetrics], None] | None = None,
) -> OtaSweepResult:
    """
    Evaluates ``simulate_ota_design_point`` over the Cartesian product of widths.

    :param diff_widths_um: Differential-pair widths to sweep.
    :param mirror_widths_um: Mirror widths to sweep.
    :param vdd: Supply voltage.
    :param vcm_v: Input common-mode voltage.
    :param step_amp_v: Differential step half-amplitude.
    :param rise_time_s: Input rise time.
    :param diff_l_um: Differential-pair channel length.
    :param mirror_l_um: Mirror channel length.
    :param tail_w_um: Tail width (fixed across the sweep).
    :param tail_l_um: Tail channel length.
    :param vbias_v: Tail gate bias.
    :param t_step_start: Step onset time.
    :param t_end: Simulation window.
    :param t_step: SPICE timestep.
    :param c_load_f: Load capacitance at ``n_out`` passed to every design point.
    :param pdk_root: Optional PDK root override.
    :param progress: Optional callback ``(index, total, point, metrics)``.
    :return: Completed sweep result.
    """
    points = tuple(OtaDesignPoint(diff_w_um=wd, mirror_w_um=wm) for wd in diff_widths_um for wm in mirror_widths_um)
    total = len(points)
    metrics_list: list[OtaMetrics] = []
    for idx, point in enumerate(points):
        result = simulate_ota_design_point(
            point,
            vdd=vdd,
            vcm_v=vcm_v,
            step_amp_v=step_amp_v,
            rise_time_s=rise_time_s,
            diff_l_um=diff_l_um,
            mirror_l_um=mirror_l_um,
            tail_w_um=tail_w_um,
            tail_l_um=tail_l_um,
            vbias_v=vbias_v,
            t_step_start=t_step_start,
            t_end=t_end,
            t_step=t_step,
            c_load_f=c_load_f,
            pdk_root=pdk_root,
        )
        metrics_list.append(result)
        if progress is not None:
            progress(idx, total, point, result)
    return OtaSweepResult(points=points, metrics=tuple(metrics_list))


def select_ota_design_point(sweep: OtaSweepResult, rule: OtaSelectionRule) -> tuple[OtaDesignPoint, OtaMetrics]:
    """
    Applies the OTA selection rule and returns the chosen design.

    Primary objective: maximum slew rate. Tiebreaker: minimum slew time.

    :param sweep: Completed OTA sweep result.
    :param rule: Pre-registered feasibility predicate.
    :return: Tuple ``(chosen point, chosen metrics)``.
    :raises ValueError: When no point satisfies ``rule``.
    """
    feasible = [(p, m) for p, m in zip(sweep.points, sweep.metrics) if rule.admits(m)]
    if not feasible:
        raise ValueError("No OTA design point in the sweep satisfies the selection rule.")
    feasible.sort(key=lambda pm: (-pm[1].slew_rate_v_per_us, pm[1].slew_time_ns))
    return feasible[0]
