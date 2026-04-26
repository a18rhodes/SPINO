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
from spino.circuit.topologies import build_cs_amp_active_load

__all__ = [
    "DesignPoint",
    "Metrics",
    "SelectionRule",
    "SweepResult",
    "extract_peak_gain",
    "extract_settling_time",
    "select_design_point",
    "simulate_design_point",
    "sweep_design_space",
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


def _build(
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


def _vtc(
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


def _operating_point(
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


def _step_response(
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


def simulate_design_point(
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


def sweep_design_space(
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
