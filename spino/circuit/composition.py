"""
Newton-Raphson solvers for FNO-composed CS amplifier analyses.

Two solvers share one residual primitive, one line-search primitive, and one
convergence dataclass:

* :class:`DcOperatingPointSolver` — scalar Newton-Raphson on ``V_out`` at a
  fixed input bias. Probes each FNO with a constant-voltage window and
  reduces the trajectory to a scalar drain current via the post-trim mean.
  Residual ``R = I_pfet(V_out) - I_nfet(V_out, V_in)``.
* :class:`TransientSolver` — whole-window implicit Newton-Raphson on the
  unknown trajectory ``V_out(t) ∈ R^T``. KCL residual per timestep is
  ``I_pfet[n] - I_nfet[n] - (C/dt)(V_out[n] - V_out[n-1])``; row 0 pins
  the initial condition.

Both solvers damp Newton steps with **Armijo backtracking** on the
residual norm: starting from ``alpha = 1`` the step is halved until the
new residual norm drops by at least a small fraction ``c1 * alpha`` of the
old norm, or until ``alpha`` falls below ``alpha_min``. A hard cap on the
per-component voltage change and a clip into ``[0, VDD]`` follow. Solver
diagnostics are returned via :class:`ConvergenceReport`.
"""

# Solver classes are configuration-heavy by design; keep NR parameters explicit.
# pylint: disable=too-many-arguments,too-many-instance-attributes,too-few-public-methods
# pylint: disable=too-many-locals,too-many-positional-arguments

from __future__ import annotations

import logging
import time as time_module
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

from spino.circuit.devices import FnoMosfetDevice
from spino.mosfet.evaluate import DEFAULT_TRIM_EVAL

__all__ = [
    "ConvergenceReport",
    "DcOperatingPointSolver",
    "DcSolution",
    "TransientSolver",
    "TransientSolution",
]

logger = logging.getLogger(__name__)

_DEFAULT_VDD = 1.8
_DEFAULT_VIN = 0.85
_DEFAULT_T_PROBE = 256
_DEFAULT_DC_TOL_A = 1e-9
_DEFAULT_TRAN_TOL_A = 1e-7
_DEFAULT_DC_MAX_ITER = 50
_DEFAULT_TRAN_MAX_ITER = 20
_DEFAULT_V_STEP_CAP = 0.2
_DEFAULT_ARMIJO_C = 1e-4
_DEFAULT_ALPHA_MIN = 1e-3
_JACOBIAN_FLOOR = 1e-15


def _abs_scalar_norm(residual: Tensor) -> float:
    """
    Reduces a 0-D residual tensor to its absolute value as a Python float.

    :param residual: Scalar residual tensor.
    :return: ``|residual|`` as a float.
    """
    return float(residual.abs().item())


def _inf_norm(residual: Tensor) -> float:
    """
    Reduces a vector residual to its supremum norm as a Python float.

    :param residual: 1-D residual tensor.
    :return: ``max_n |residual[n]|`` as a float.
    """
    return float(residual.abs().max().item())


def _sign_preserving_clamp(value: Tensor, floor: float) -> Tensor:
    """
    Bounds ``|value|`` from below while keeping the original sign.

    Plain :meth:`torch.Tensor.clamp_min` would flip the sign of negative
    inputs whose magnitude falls below the floor; that is fatal for
    Newton-Raphson, where the natural Jacobian sign is the one driving the
    descent direction. This helper guarantees ``|out| >= floor`` and
    ``sign(out) == sign(value)`` (with a positive floor for exact zeros).

    :param value: Input tensor.
    :param floor: Minimum absolute value tolerated.
    :return: Sign-preserving floored tensor.
    """
    sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
    magnitude = value.abs().clamp_min(floor)
    return sign * magnitude


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    """
    Per-solve diagnostic record.

    :param converged: True when the residual norm reached ``residual_tol``
        before exhausting ``max_iter``.
    :param iter_count: Number of outer Newton iterations executed.
    :param residual_norm_history: One residual norm per accepted Newton step,
        starting with the initial guess. Length is ``iter_count + 1``.
    :param final_residual_norm: Last residual norm in the history.
    :param wall_ms: Wall-clock time spent inside the solver, in
        milliseconds.
    """

    converged: bool
    iter_count: int
    residual_norm_history: tuple[float, ...] = field(default_factory=tuple)
    final_residual_norm: float = float("nan")
    wall_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class DcSolution:
    """
    Output of :meth:`DcOperatingPointSolver.solve`.

    :param v_out_v: Scalar output voltage in volts.
    :param i_pfet_a: PFET drain current at the operating point in amperes.
    :param i_nfet_a: NFET drain current at the operating point in amperes.
    :param report: Convergence diagnostics.
    """

    v_out_v: float
    i_pfet_a: float
    i_nfet_a: float
    report: ConvergenceReport


@dataclass(frozen=True, slots=True)
class TransientSolution:
    """
    Output of :meth:`TransientSolver.solve`.

    :param time_s: Time grid in seconds.
    :param v_out_v: Solved output trajectory in volts ``(T,)``.
    :param i_pfet_a: PFET drain current trajectory in amperes ``(T,)``.
    :param i_nfet_a: NFET drain current trajectory in amperes ``(T,)``.
    :param report: Convergence diagnostics.
    """

    time_s: Tensor
    v_out_v: Tensor
    i_pfet_a: Tensor
    i_nfet_a: Tensor
    report: ConvergenceReport


def _backtrack(
    residual_fn: Callable[[Tensor], Tensor],
    state: Tensor,
    direction: Tensor,
    residual_norm_old: float,
    norm_fn: Callable[[Tensor], float],
    *,
    armijo_c: float,
    alpha_min: float,
) -> tuple[float, float]:
    """
    Armijo backtracking line search on the residual norm.

    Starts from ``alpha = 1`` and halves it until the candidate state
    ``state + alpha * direction`` produces a residual whose norm satisfies
    the sufficient-decrease test
    ``||R_new|| <= (1 - armijo_c * alpha) * ||R_old||``,
    or ``alpha`` falls below ``alpha_min``. The test is the Armijo half of
    the classical Wolfe conditions, applied to the surrogate objective
    ``phi(alpha) = ||R(state + alpha * direction)||``.

    :param residual_fn: Callable producing the residual at a given state.
    :param state: Current state vector.
    :param direction: Newton direction (``-J^{-1} R``).
    :param residual_norm_old: Norm of the residual at ``state``.
    :param norm_fn: Reduction used to score residual vectors.
    :param armijo_c: Sufficient-decrease coefficient (default ``1e-4``).
    :param alpha_min: Minimum step the search will accept.
    :return: Tuple ``(alpha, residual_norm_new)``.
    """
    alpha = 1.0
    residual_norm_new = float("inf")
    while True:
        with torch.no_grad():
            candidate = state + alpha * direction
            residual_norm_new = norm_fn(residual_fn(candidate))
        if residual_norm_new <= (1.0 - armijo_c * alpha) * residual_norm_old:
            return alpha, residual_norm_new
        alpha *= 0.5
        if alpha < alpha_min:
            return alpha, residual_norm_new


def _cap_alpha(alpha: float, direction: Tensor, v_step_cap: float) -> float:
    """
    Reduces ``alpha`` further so that the largest voltage move is
    bounded by ``v_step_cap``.

    :param alpha: Step accepted by the line search.
    :param direction: Newton direction.
    :param v_step_cap: Maximum allowed change in any voltage component.
    :return: Capped step size.
    """
    max_dir = float(direction.abs().max().item())
    if max_dir < 1e-12:
        return alpha
    return min(alpha, v_step_cap / max_dir)


def _build_nfet_probe(v_out: Tensor, vin: Tensor, time_steps: int) -> Tensor:
    """
    Builds an NFET probe window for the DC scalar solver.

    Channel order matches training: ``Vg, Vd, Vs, Vb``. NFET XM1 sees the
    forced input on its gate and the unknown ``V_out`` on its drain; source
    and bulk are tied to ground.

    :param v_out: Scalar tensor; ``V_out`` candidate (with grad).
    :param vin: Scalar tensor; input bias voltage at this step.
    :param time_steps: Length of the probe window.
    :return: Voltage trajectory of shape ``(1, 4, T)``.
    """
    v_out_t = v_out.reshape(1, 1, 1).expand(1, 1, time_steps)
    vin_t = vin.reshape(1, 1, 1).expand(1, 1, time_steps)
    zero = torch.zeros_like(v_out_t)
    return torch.cat([vin_t, v_out_t, zero, zero], dim=1)


def _build_pfet_probe(v_out: Tensor, vdd: Tensor, time_steps: int) -> Tensor:
    """
    Builds a PFET probe window for the diode-connected active load.

    Gate and drain are tied to ``V_out`` (diode connection); source and
    bulk are tied to ``VDD``. Both gate and drain channels carry the same
    autograd-traced ``V_out`` so the Jacobian sums the two paths.

    :param v_out: Scalar tensor; ``V_out`` candidate (with grad).
    :param vdd: Scalar tensor with the supply voltage value.
    :param time_steps: Length of the probe window.
    :return: Voltage trajectory of shape ``(1, 4, T)``.
    """
    v_out_t = v_out.reshape(1, 1, 1).expand(1, 1, time_steps)
    vdd_t = vdd.reshape(1, 1, 1).expand(1, 1, time_steps)
    return torch.cat([v_out_t, v_out_t, vdd_t, vdd_t], dim=1)


def _build_nfet_trajectory(v_out: Tensor, vin_t: Tensor) -> Tensor:
    """
    Builds the NFET trajectory probe for the transient solver.

    :param v_out: ``V_out(t)`` of shape ``(T,)``.
    :param vin_t: Forced ``V_in(t)`` of shape ``(T,)``.
    :return: Trajectory tensor of shape ``(1, 4, T)``.
    """
    time_steps = v_out.shape[0]
    zero = torch.zeros((1, 1, time_steps), dtype=v_out.dtype, device=v_out.device)
    return torch.cat(
        [
            vin_t.reshape(1, 1, time_steps),
            v_out.reshape(1, 1, time_steps),
            zero,
            zero,
        ],
        dim=1,
    )


def _build_pfet_trajectory(v_out: Tensor, vdd_value: float) -> Tensor:
    """
    Builds the diode-connected PFET trajectory probe for the transient solver.

    :param v_out: ``V_out(t)`` of shape ``(T,)``.
    :param vdd_value: Supply voltage in volts.
    :return: Trajectory tensor of shape ``(1, 4, T)``.
    """
    time_steps = v_out.shape[0]
    vdd_t = torch.full(
        (1, 1, time_steps),
        fill_value=vdd_value,
        dtype=v_out.dtype,
        device=v_out.device,
    )
    return torch.cat(
        [
            v_out.reshape(1, 1, time_steps),
            v_out.reshape(1, 1, time_steps),
            vdd_t,
            vdd_t,
        ],
        dim=1,
    )


class DcOperatingPointSolver:
    """
    Scalar Newton-Raphson solver for the CS amplifier DC operating point.

    The unknown is the single internal node voltage ``V_out``. Each device
    is queried with a constant-V probe window and the trajectory is reduced
    to a scalar drain current via the post-trim time mean. The KCL residual
    is ``R = I_pfet(V_out) - I_nfet(V_out, V_in)``.

    :param nfet: Loaded NFET device wrapper.
    :param pfet: Loaded PFET device wrapper.
    :param vdd: Supply voltage in volts.
    :param t_probe: Length of the probe window in samples.
    :param trim: Number of leading samples discarded before the time mean
        (matches :data:`spino.mosfet.evaluate.DEFAULT_TRIM_EVAL`).
    :param max_iter: Iteration budget.
    :param residual_tol: Convergence tolerance on ``|R|`` in amperes.
    :param v_step_cap: Maximum per-step voltage change.
    :param armijo_c: Sufficient-decrease coefficient for the line search.
    :param alpha_min: Minimum line-search step before giving up.
    """

    def __init__(
        self,
        nfet: FnoMosfetDevice,
        pfet: FnoMosfetDevice,
        *,
        vdd: float = _DEFAULT_VDD,
        t_probe: int = _DEFAULT_T_PROBE,
        trim: int = DEFAULT_TRIM_EVAL,
        max_iter: int = _DEFAULT_DC_MAX_ITER,
        residual_tol: float = _DEFAULT_DC_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
    ) -> None:
        if t_probe <= trim:
            raise ValueError(f"t_probe ({t_probe}) must exceed trim ({trim}) for the post-trim mean to be defined")
        self.nfet = nfet
        self.pfet = pfet
        self.vdd = vdd
        self.t_probe = t_probe
        self.trim = trim
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self._device = nfet.v_mean.device
        self._dtype = nfet.v_mean.dtype

    def _device_currents(self, v_out: Tensor, vin: Tensor) -> tuple[Tensor, Tensor]:
        """
        Builds probe windows and decodes the post-trim mean drain currents.

        Runs in whatever autograd context the caller establishes — wrap with
        :func:`torch.no_grad` for line-search probes, leave un-wrapped to
        keep the graph attached for Jacobian extraction.

        :param v_out: Scalar ``V_out`` candidate.
        :param vin: Scalar input bias voltage.
        :return: Tuple ``(i_pfet, i_nfet)`` of scalar tensors.
        """
        nfet_probe = _build_nfet_probe(v_out, vin, self.t_probe)
        pfet_probe = _build_pfet_probe(v_out, self._scalar(self.vdd), self.t_probe)
        i_nfet = self.nfet.drain_current(nfet_probe)[..., self.trim :].mean()
        i_pfet = self.pfet.drain_current(pfet_probe)[..., self.trim :].mean()
        return i_pfet, i_nfet

    def _residual(self, v_out: Tensor, vin: Tensor) -> Tensor:
        """
        Computes the scalar KCL residual ``R = I_pfet - I_nfet``.

        :param v_out: Scalar ``V_out`` candidate.
        :param vin: Scalar input bias voltage.
        :return: Scalar residual tensor (gradient context is the caller's).
        """
        i_pfet, i_nfet = self._device_currents(v_out, vin)
        return i_pfet - i_nfet

    def _scalar(self, value: float) -> Tensor:
        """
        Helper: creates a 0-D tensor on the solver's device and dtype.

        :param value: Scalar value to wrap.
        :return: 0-D tensor.
        """
        return torch.tensor(value, dtype=self._dtype, device=self._device)

    def solve(self, vin: float = _DEFAULT_VIN, v_out_init: float | None = None) -> DcSolution:
        """
        Runs the damped scalar Newton-Raphson loop to convergence.

        :param vin: Forced input bias voltage in volts.
        :param v_out_init: Optional initial guess for ``V_out``. Defaults to
            the rail mid-point ``VDD / 2``.
        :return: :class:`DcSolution` with the converged operating point and
            convergence diagnostics.
        """
        start = time_module.perf_counter()
        v_out_initial = self.vdd / 2.0 if v_out_init is None else v_out_init
        v_out = self._scalar(v_out_initial).requires_grad_(True)
        vin_tensor = self._scalar(vin)
        with torch.no_grad():
            residual_norm = _abs_scalar_norm(self._residual(v_out.detach(), vin_tensor))
        history = [residual_norm]
        converged = residual_norm <= self.residual_tol
        iter_count = 0
        while not converged and iter_count < self.max_iter:
            iter_count += 1
            v_out, residual_norm = self._newton_step(v_out, vin_tensor, residual_norm)
            history.append(residual_norm)
            converged = residual_norm <= self.residual_tol
        with torch.no_grad():
            i_pfet, i_nfet = self._device_currents(v_out.detach(), vin_tensor)
        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        return DcSolution(
            v_out_v=float(v_out.detach().item()),
            i_pfet_a=float(i_pfet.item()),
            i_nfet_a=float(i_nfet.item()),
            report=ConvergenceReport(
                converged=converged,
                iter_count=iter_count,
                residual_norm_history=tuple(history),
                final_residual_norm=residual_norm,
                wall_ms=wall_ms,
            ),
        )

    def _newton_step(self, v_out: Tensor, vin: Tensor, residual_norm_old: float) -> tuple[Tensor, float]:
        """
        Executes one damped Newton step.

        :param v_out: Current ``V_out`` (with grad enabled).
        :param vin: Forced input bias voltage.
        :param residual_norm_old: Residual norm at ``v_out``.
        :return: Tuple ``(v_out_new_with_grad, residual_norm_new)``.
        """
        residual = self._residual(v_out, vin)
        (jacobian,) = torch.autograd.grad(residual, v_out, create_graph=False)
        direction = -residual.detach() / _sign_preserving_clamp(jacobian, _JACOBIAN_FLOOR)

        def line_search_residual(candidate: Tensor) -> Tensor:
            return self._residual(candidate, vin).reshape(())

        alpha, _ = _backtrack(
            line_search_residual,
            v_out.detach(),
            direction,
            residual_norm_old,
            norm_fn=_abs_scalar_norm,
            armijo_c=self.armijo_c,
            alpha_min=self.alpha_min,
        )
        alpha = _cap_alpha(alpha, direction, self.v_step_cap)
        v_out_new = (v_out.detach() + alpha * direction).clamp(0.0, self.vdd)
        with torch.no_grad():
            residual_norm_new = _abs_scalar_norm(self._residual(v_out_new, vin))
        return v_out_new.requires_grad_(True), residual_norm_new


class TransientSolver:
    """
    Whole-window implicit Newton-Raphson solver for ``V_out(t)``.

    Treats the entire time trajectory as a single unknown vector
    ``V_out ∈ R^T``. Each outer Newton iteration produces one FNO forward
    pass per device on the full trajectory; the Jacobian is assembled
    densely via :func:`torch.autograd.functional.jacobian`. The linear
    capacitive load contributes a banded backward-Euler ``C/dt`` operator
    added analytically. The initial-condition row pins ``V_out[0]`` to the
    DC operating point.

    :param nfet: Loaded NFET device wrapper.
    :param pfet: Loaded PFET device wrapper.
    :param vdd: Supply voltage in volts.
    :param c_load_f: Load capacitance in farads.
    :param max_iter: Outer Newton iteration budget.
    :param residual_tol: Convergence tolerance on ``||R||_inf`` in amperes.
    :param v_step_cap: Maximum per-step voltage change.
    :param armijo_c: Sufficient-decrease coefficient for the line search.
    :param alpha_min: Minimum line-search step before giving up.
    """

    def __init__(
        self,
        nfet: FnoMosfetDevice,
        pfet: FnoMosfetDevice,
        *,
        vdd: float = _DEFAULT_VDD,
        c_load_f: float = 10e-12,
        max_iter: int = _DEFAULT_TRAN_MAX_ITER,
        residual_tol: float = _DEFAULT_TRAN_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
    ) -> None:
        self.nfet = nfet
        self.pfet = pfet
        self.vdd = vdd
        self.c_load_f = c_load_f
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self._device = nfet.v_mean.device
        self._dtype = nfet.v_mean.dtype

    def _device_currents(self, v_out: Tensor, vin_t: Tensor) -> tuple[Tensor, Tensor]:
        """
        Evaluates the per-step PFET and NFET drain currents on the trajectory.

        :param v_out: ``V_out(t)`` of shape ``(T,)``.
        :param vin_t: Forced ``V_in(t)`` of shape ``(T,)``.
        :return: Tuple ``(i_pfet, i_nfet)`` each of shape ``(T,)``.
        """
        i_nfet = self.nfet.drain_current(_build_nfet_trajectory(v_out, vin_t))[0, 0, :]
        i_pfet = self.pfet.drain_current(_build_pfet_trajectory(v_out, self.vdd))[0, 0, :]
        return i_pfet, i_nfet

    def _residual_fn(
        self,
        v_out: Tensor,
        vin_t: Tensor,
        v_out_dc: float,
        dt: Tensor,
    ) -> Tensor:
        """
        Whole-window KCL residual including IC clamp and analytical cap term.

        Row 0 fixes the initial condition: ``R[0] = V_out[0] - V_out_DC``.
        Rows ``1..T-1`` enforce KCL with backward-Euler discretization of
        the load capacitor: ``R[n] = I_pfet[n] - I_nfet[n] - (C/dt[n-1]) * (V_out[n] - V_out[n-1])``.

        :param v_out: ``V_out(t)`` of shape ``(T,)``.
        :param vin_t: Forced ``V_in(t)`` of shape ``(T,)``.
        :param v_out_dc: Initial-condition voltage in volts.
        :param dt: Per-step time deltas of shape ``(T-1,)``.
        :return: Residual vector of shape ``(T,)``.
        """
        i_pfet, i_nfet = self._device_currents(v_out, vin_t)
        v_diff = v_out[1:] - v_out[:-1]
        cap_term = self.c_load_f * v_diff / dt
        kcl_residual = (i_pfet[1:] - i_nfet[1:]) - cap_term
        ic_residual = v_out[0:1] - v_out_dc
        return torch.cat([ic_residual, kcl_residual], dim=0)

    def _newton_step(
        self,
        v_out: Tensor,
        vin_t: Tensor,
        v_out_dc: float,
        dt: Tensor,
        residual_norm_old: float,
    ) -> tuple[Tensor, float]:
        """
        Executes one damped whole-window Newton step.

        :param v_out: Current ``V_out`` trajectory ``(T,)``.
        :param vin_t: Forced ``V_in(t)`` of shape ``(T,)``.
        :param v_out_dc: Initial-condition voltage in volts.
        :param dt: Per-step time deltas of shape ``(T-1,)``.
        :param residual_norm_old: ``||R||_inf`` at ``v_out``.
        :return: Tuple ``(v_out_new, residual_norm_new)``.
        """

        def residual_fn(x: Tensor) -> Tensor:
            return self._residual_fn(x, vin_t, v_out_dc, dt)

        with torch.no_grad():
            residual = residual_fn(v_out)
        jacobian = torch.autograd.functional.jacobian(residual_fn, v_out, vectorize=True)
        direction = torch.linalg.solve(jacobian, -residual)  # pylint: disable=not-callable
        alpha, _ = _backtrack(
            residual_fn,
            v_out,
            direction,
            residual_norm_old,
            norm_fn=_inf_norm,
            armijo_c=self.armijo_c,
            alpha_min=self.alpha_min,
        )
        alpha = _cap_alpha(alpha, direction, self.v_step_cap)
        v_out_new = (v_out + alpha * direction).clamp(0.0, self.vdd)
        with torch.no_grad():
            residual_norm_new = _inf_norm(residual_fn(v_out_new))
        return v_out_new, residual_norm_new

    def solve(
        self,
        time_s: Tensor,
        vin_t: Tensor,
        v_out_dc: float,
        v_out_init: Tensor | None = None,
    ) -> TransientSolution:
        """
        Runs whole-window implicit Newton-Raphson on the transient trajectory.

        :param time_s: Time grid of shape ``(T,)`` in seconds. Must be
            strictly increasing and uniform-friendly (per-step ``dt``).
        :param vin_t: Forced ``V_in(t)`` of shape ``(T,)``.
        :param v_out_dc: Initial-condition voltage from the DC OP solver.
        :param v_out_init: Optional initial guess for ``V_out(t)``.
            Defaults to a constant trajectory at ``v_out_dc``.
        :return: :class:`TransientSolution` with the trajectory, currents,
            and convergence diagnostics.
        """
        start = time_module.perf_counter()
        time_grid = time_s.to(device=self._device, dtype=self._dtype)
        vin_grid = vin_t.to(device=self._device, dtype=self._dtype)
        dt = time_grid[1:] - time_grid[:-1]
        v_out = self._initial_trajectory(time_grid, v_out_dc, v_out_init)
        residual_norm = self._initial_residual(v_out, vin_grid, v_out_dc, dt)
        history = [residual_norm]
        converged = residual_norm <= self.residual_tol
        iter_count = 0
        while not converged and iter_count < self.max_iter:
            iter_count += 1
            v_out, residual_norm = self._newton_step(v_out, vin_grid, v_out_dc, dt, residual_norm)
            history.append(residual_norm)
            converged = residual_norm <= self.residual_tol
        with torch.no_grad():
            i_pfet, i_nfet = self._device_currents(v_out, vin_grid)
        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        return TransientSolution(
            time_s=time_grid.detach(),
            v_out_v=v_out.detach(),
            i_pfet_a=i_pfet.detach(),
            i_nfet_a=i_nfet.detach(),
            report=ConvergenceReport(
                converged=converged,
                iter_count=iter_count,
                residual_norm_history=tuple(history),
                final_residual_norm=residual_norm,
                wall_ms=wall_ms,
            ),
        )

    def _initial_trajectory(
        self,
        time_grid: Tensor,
        v_out_dc: float,
        v_out_init: Tensor | None,
    ) -> Tensor:
        """
        Materializes the starting trajectory for the outer Newton loop.

        :param time_grid: Time grid ``(T,)``.
        :param v_out_dc: DC operating point voltage in volts.
        :param v_out_init: Optional pre-shaped guess.
        :return: Trajectory tensor on the solver device.
        """
        if v_out_init is None:
            return torch.full_like(time_grid, fill_value=v_out_dc)
        return v_out_init.to(device=self._device, dtype=self._dtype).clone()

    def _initial_residual(self, v_out: Tensor, vin_t: Tensor, v_out_dc: float, dt: Tensor) -> float:
        """
        Computes the seed residual norm for the convergence history.

        :param v_out: Initial trajectory.
        :param vin_t: Forced ``V_in(t)``.
        :param v_out_dc: Initial-condition voltage.
        :param dt: Per-step time deltas.
        :return: ``||R||_inf`` at the initial trajectory.
        """
        with torch.no_grad():
            return _inf_norm(self._residual_fn(v_out, vin_t, v_out_dc, dt))
