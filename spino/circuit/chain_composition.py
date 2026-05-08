"""
Newton–Raphson solvers for FNO-composed CMOS inverter chains.

Extends the CS-amp :mod:`spino.circuit.composition` policy: **backward Euler**
capacitive terms, whole-window implicit transient NR, Armijo damping shared
with ``TransientSolver``.
"""

from __future__ import annotations

import time as time_module
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from spino.circuit.composition import _backtrack, _cap_alpha, _inf_norm, ConvergenceReport
from spino.circuit.devices import FnoMosfetDevice
from spino.circuit.partition_caps import TorchPartitionCapGrid
from spino.mosfet.evaluate import DEFAULT_TRIM_EVAL

__all__ = [
    "ChainDcSolution",
    "ChainDcSolver",
    "ChainTransientSolution",
    "ChainTransientSolver",
    "build_inv_nfet_trajectory",
    "build_inv_pfet_trajectory",
    "default_chain_dc_voltage_guess",
]


_DEFAULT_VDD = 1.8
_DEFAULT_DC_TOL_A = 5e-7
_DEFAULT_TRAN_TOL_A = 5e-7
_DEFAULT_DC_MAX_ITER = 80
_DEFAULT_TRAN_MAX_ITER = 25
_DEFAULT_V_STEP_CAP = 0.2
_DEFAULT_ARMIJO_C = 1e-4
_DEFAULT_ALPHA_MIN = 1e-3
_DEFAULT_MONOTONE_SWITCH_RESIDUAL_A = 1e-6


def _backtrack_monotone(
    residual_fn,
    state: Tensor,
    direction: Tensor,
    residual_norm_old: float,
    norm_fn,
    *,
    alpha_min: float,
) -> tuple[float, float]:
    """
    Returns ``alpha=1`` unconditionally in the monotone (high-residual) regime.

    In the saturation plateau of a digital inverter the KCL residual is nearly
    flat over a ~1.8 V Vout range, so halving never achieves monotone decrease.
    Returning 0 there (the previous behaviour) silenced every Newton step.
    Instead we return alpha=1 and rely on ``_cap_alpha`` + the rail clamp in the
    caller to keep the actual voltage move bounded by ``v_step_cap``.
    """
    return 1.0, residual_norm_old


def build_inv_nfet_trajectory(v_gate: Tensor, v_drain: Tensor) -> Tensor:
    """
    Layout ``(Vg, Vd, Vs, Vb)`` for an inverter NFET (source/bulk at ground).

    :param v_gate: ``(T,)`` gate waveform.
    :param v_drain: ``(T,)`` drain waveform.
    :return: Tensor ``(1, 4, T)``.
    """
    t_steps = v_gate.shape[0]
    zero = torch.zeros((1, 1, t_steps), dtype=v_gate.dtype, device=v_gate.device)
    return torch.cat(
        [
            v_gate.reshape(1, 1, t_steps),
            v_drain.reshape(1, 1, t_steps),
            zero,
            zero,
        ],
        dim=1,
    )


def default_chain_dc_voltage_guess(
    vin_v: float,
    n_stages: int,
    *,
    vdd: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """
    Alternating rail initial guess ``(v_n1,...,v_nN)`` for chain DC NR.

    For ``Vin`` below roughly ``VDD/2``, the first inverter output favors the
    high rail; successive inverters alternate logically. Above ``VDD/2``, the
    polarity flips so the first inverter output favors the low rail. This avoids initializing
    every node at ``VDD/2``, which poorly matches switching stacks and slows or
    destabilizes Newton.

    :param vin_v: Input bias at ``nin`` (volts).
    :param n_stages: Number of stages ``N``.
    :param vdd: Supply voltage.
    :param dtype: Float dtype for returned tensor.
    :param device: Torch device placement.
    :return: Shape ``(N,)`` bounded in ``[0, VDD]``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    vin_high = vin_v >= 0.5 * float(vdd)
    vals: list[float] = []
    for k in range(n_stages):
        rail_high = (k % 2 == 1) if vin_high else (k % 2 == 0)
        vals.append(float(vdd) if rail_high else 0.0)
    return torch.tensor(vals, dtype=dtype, device=device)


def build_inv_pfet_trajectory(v_gate: Tensor, v_drain: Tensor, vdd_value: float) -> Tensor:
    """
    Layout ``(Vg, Vd, Vs, Vb)`` for an inverter PFET (source/bulk at ``VDD``).

    :param v_gate: ``(T,)``.
    :param v_drain: ``(T,)``.
    :param vdd_value: Supply voltage.
    :return: Tensor ``(1, 4, T)``.
    """
    t_steps = v_gate.shape[0]
    vdd_t = torch.full((1, 1, t_steps), fill_value=vdd_value, dtype=v_gate.dtype, device=v_gate.device)
    return torch.cat(
        [
            v_gate.reshape(1, 1, t_steps),
            v_drain.reshape(1, 1, t_steps),
            vdd_t,
            vdd_t,
        ],
        dim=1,
    )


@dataclass(frozen=True, slots=True)
class ChainDcSolution:
    """
    DC solution for an ``N``-stage inverter chain.

    :param v_out_v: ``(N,)`` output voltages per stage (volts).
    :param reports: Convergence reports (last entry is the aggregate solve).
    """

    v_out_v: Tensor
    reports: tuple[ConvergenceReport, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ChainTransientSolution:
    """
    Transient solution trajectories for monitored stage outputs.

    :param time_s: ``(T,)`` time samples (s).
    :param v_nodes_v: ``(N, T)`` stage-output voltages (V).
    :param report: Newton diagnostics.
    """

    time_s: Tensor
    v_nodes_v: Tensor
    report: ConvergenceReport


class ChainDcSolver:
    """
    Vector damped Newton–Raphson for **N** inverter output nodes at fixed input.

    Each stage ``k`` solves ``I_pfet,k - I_nfet,k = 0`` with gates driven by
    ``V_in`` (stage 1) or ``V_{k-1}`` (internal feedback from previous output).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        nfet_devices: tuple[FnoMosfetDevice, ...],
        pfet_devices: tuple[FnoMosfetDevice, ...],
        *,
        vdd: float = _DEFAULT_VDD,
        t_probe: int = 256,
        trim: int = DEFAULT_TRIM_EVAL,
        max_iter: int = _DEFAULT_DC_MAX_ITER,
        residual_tol: float = _DEFAULT_DC_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
    ) -> None:
        self.n_stages = len(nfet_devices)
        if len(pfet_devices) != self.n_stages:
            raise ValueError("nfet_devices and pfet_devices must have identical length")
        if t_probe <= trim:
            raise ValueError(f"t_probe ({t_probe}) must exceed trim ({trim})")
        self.nfets = nfet_devices
        self.pfets = pfet_devices
        self.vdd = vdd
        self.t_probe = t_probe
        self.trim = trim
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self._device = nfet_devices[0].v_mean.device
        self._dtype = nfet_devices[0].v_mean.dtype

    def _mean_id(self, mos: FnoMosfetDevice, probe_4t: Tensor) -> Tensor:
        return mos.drain_current(probe_4t)[..., self.trim :].mean()

    def _residual_vec(self, v_vec: Tensor, vin: Tensor) -> Tensor:
        outs: list[Tensor] = []
        for k in range(self.n_stages):
            v_gate = vin if k == 0 else v_vec[k - 1]
            v_d = v_vec[k]
            proben = build_inv_nfet_trajectory(v_gate.expand(self.t_probe), v_d.expand(self.t_probe))
            probep = build_inv_pfet_trajectory(v_gate.expand(self.t_probe), v_d.expand(self.t_probe), self.vdd)
            outs.append(self._mean_id(self.pfets[k], probep) - self._mean_id(self.nfets[k], proben))
        return torch.stack(outs, dim=0)

    def _newton_step(self, v: Tensor, vin_t: Tensor, residual_norm_old: float) -> tuple[Tensor, float]:
        residual = self._residual_vec(v, vin_t)
        jac = jacobian(lambda x: self._residual_vec(x, vin_t), v, vectorize=True, create_graph=False)
        direction = torch.linalg.solve(jac, -residual.detach())

        def residual_fn(candidate: Tensor) -> Tensor:
            return self._residual_vec(candidate, vin_t)

        alpha, _ = _backtrack(
            residual_fn,
            v.detach(),
            direction,
            residual_norm_old,
            norm_fn=_inf_norm,
            armijo_c=self.armijo_c,
            alpha_min=self.alpha_min,
        )
        alpha = _cap_alpha(alpha, direction, self.v_step_cap)
        v_new = (v.detach() + alpha * direction).clamp(0.0, self.vdd)
        with torch.no_grad():
            rn = _inf_norm(self._residual_vec(v_new, vin_t))
        return v_new.requires_grad_(True), rn

    def solve(self, vin: float, v_init: Tensor | None = None) -> ChainDcSolution:
        start = time_module.perf_counter()
        vin_t = torch.tensor(vin, dtype=self._dtype, device=self._device)
        if v_init is None:
            v = default_chain_dc_voltage_guess(
                vin,
                self.n_stages,
                vdd=self.vdd,
                dtype=self._dtype,
                device=self._device,
            )
        else:
            v = v_init.to(device=self._device, dtype=self._dtype).clone()
        v = v.requires_grad_(True)
        with torch.no_grad():
            rn = _inf_norm(self._residual_vec(v.detach(), vin_t))
        history: list[float] = [rn]
        converged = rn <= self.residual_tol
        iters = 0
        while not converged and iters < self.max_iter:
            iters += 1
            v, rn = self._newton_step(v, vin_t, rn)
            history.append(rn)
            converged = rn <= self.residual_tol
        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        return ChainDcSolution(
            v_out_v=v.detach(),
            reports=(
                ConvergenceReport(
                    converged=converged,
                    iter_count=iters,
                    residual_norm_history=tuple(history),
                    final_residual_norm=rn,
                    wall_ms=wall_ms,
                ),
            ),
        )


def _gate_displacement_be(
    caps: TorchPartitionCapGrid,
    *,
    vdd: float,
    vg_n: Tensor,
    vd_n: Tensor,
    vg_prev: Tensor,
    vd_prev: Tensor,
    dt_vec: Tensor,
    is_pfet: bool,
) -> Tensor:
    """
    Gate quasi-static displacement using BE increments and caps at implicit ``n``.

    With ``Vs,Vb`` fixed to rails: ``∂Vgs = ∂Vg``, ``∂Vgd = ∂(Vg−Vd)``, ``∂Vgb = ∂Vg``.
    NFET lookups use ``(Vgs,Vds) = (Vg,Vd)``; PFET use ``(VSG,VSD) = (Vdd−Vg,Vdd−Vd)``.
    """
    if is_pfet:
        cgs_i, cgd_i, cgb_i = caps(vdd - vg_n, vdd - vd_n)
    else:
        cgs_i, cgd_i, cgb_i = caps(vg_n, vd_n)
    dv_gs = vg_n - vg_prev
    dv_gd = (vg_n - vd_n) - (vg_prev - vd_prev)
    dv_gb = vg_n - vg_prev
    return (cgs_i * dv_gs + cgd_i * dv_gd + cgb_i * dv_gb) / dt_vec


class ChainTransientSolver:
    """
    Whole-window implicit NR on stacked stage-output voltages ``(N, T)``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        nfet_devices: tuple[FnoMosfetDevice, ...],
        pfet_devices: tuple[FnoMosfetDevice, ...],
        nfet_caps: TorchPartitionCapGrid,
        pfet_caps: TorchPartitionCapGrid,
        *,
        vdd: float = _DEFAULT_VDD,
        c_load_f: float = 0.0,
        max_iter: int = _DEFAULT_TRAN_MAX_ITER,
        residual_tol: float = _DEFAULT_TRAN_TOL_A,
        v_step_cap: float = _DEFAULT_V_STEP_CAP,
        armijo_c: float = _DEFAULT_ARMIJO_C,
        alpha_min: float = _DEFAULT_ALPHA_MIN,
        monotone_switch_residual: float = _DEFAULT_MONOTONE_SWITCH_RESIDUAL_A,
    ) -> None:
        n = len(nfet_devices)
        if len(pfet_devices) != n:
            raise ValueError("device tuple length mismatch")
        self.n_stages = n
        self.nfets = nfet_devices
        self.pfets = pfet_devices
        self.nfet_caps = nfet_caps
        self.pfet_caps = pfet_caps
        self.vdd = vdd
        self.c_load_f = c_load_f
        self.max_iter = max_iter
        self.residual_tol = residual_tol
        self.v_step_cap = v_step_cap
        self.armijo_c = armijo_c
        self.alpha_min = alpha_min
        self.monotone_switch_residual = monotone_switch_residual
        self._device = nfet_devices[0].v_mean.device
        self._dtype = nfet_devices[0].v_mean.dtype

    def _conductive_residual_row(
        self,
        k: int,
        v_flat: Tensor,
        vin_t: Tensor,
        n: int,
        cap_t: int,
    ) -> Tensor:
        vnode = v_flat.reshape(n, cap_t)
        v_gate_k = vin_t if k == 0 else vnode[k - 1]
        vd_k = vnode[k]
        inp = build_inv_nfet_trajectory(v_gate_k, vd_k)
        ipp = build_inv_pfet_trajectory(v_gate_k, vd_k, self.vdd)
        ip_f = self.pfets[k].drain_current(ipp)[0, 0, :]
        in_f = self.nfets[k].drain_current(inp)[0, 0, :]
        return ip_f - in_f

    def _displacement_at_node_k(
        self,
        k: int,
        vnode: Tensor,
        dt_vec: Tensor,
        cap_t: int,
    ) -> Tensor:
        """
        Quasi-static gate displacement allotted to inverter output-node KCL rows.

        When ``k < n-1``, ``(vg, vd) = (vnode[k], vnode[k+1])`` evaluates NFET and
        PFET partition-cap displacement for the inverter **one stage after**
        output row ``k``: gates track ``vnode[k]`` and drains ``vnode[k+1]``.

        Row ``k=0`` (first output ``n1``) includes gate charging for stage 2 driven
        from ``n1``, yet omits stage 1 gate displacement (those terminals are on
        ``nin`` driven by transient ``Vin``, not by KCL surplus on ``n1``).

        Rows ``k=1 .. n-2`` symmetrically accumulate the same downstream gate-
        current bookkeeping. Row ``k=n-1`` has no ``k < n-1`` branch except
        optional explicit ``CL`` displacement on the tail node.
        """
        n = vnode.shape[0]
        idx = slice(1, cap_t)
        dt_b = dt_vec
        ig_sum = torch.zeros(cap_t - 1, dtype=vnode.dtype, device=vnode.device)
        if k < n - 1:
            vg_now = vnode[k, idx]
            vg_prev_w = vnode[k, : cap_t - 1]
            vd_now = vnode[k + 1, idx]
            vd_prev_w = vnode[k + 1, : cap_t - 1]
            ig_n = _gate_displacement_be(
                self.nfet_caps,
                vdd=self.vdd,
                vg_n=vg_now,
                vd_n=vd_now,
                vg_prev=vg_prev_w,
                vd_prev=vd_prev_w,
                dt_vec=dt_b,
                is_pfet=False,
            )
            ig_p = _gate_displacement_be(
                self.pfet_caps,
                vdd=self.vdd,
                vg_n=vg_now,
                vd_n=vd_now,
                vg_prev=vg_prev_w,
                vd_prev=vd_prev_w,
                dt_vec=dt_b,
                is_pfet=True,
            )
            ig_sum = ig_sum + ig_n + ig_p
        if k == n - 1 and self.c_load_f > 0.0:
            dv = vnode[k, idx] - vnode[k, : cap_t - 1]
            ig_sum = ig_sum + self.c_load_f * dv / dt_b
        return ig_sum

    def _full_residual_flat(self, v_flat: Tensor, vin_t: Tensor, v_dc: Tensor, dt_vec: Tensor) -> Tensor:
        cap_t = vin_t.shape[0]
        n = self.n_stages
        vnode = v_flat.reshape(n, cap_t)
        conductive = torch.empty((n, cap_t), dtype=v_flat.dtype, device=v_flat.device)
        for kk in range(n):
            conductive[kk] = self._conductive_residual_row(kk, v_flat, vin_t, n, cap_t)
        out_chunks: list[Tensor] = []
        for kk in range(n):
            ic = vnode[kk, 0] - v_dc[kk]
            rhs = conductive[kk, 1:] - self._displacement_at_node_k(kk, vnode, dt_vec, cap_t)
            out_chunks.append(ic.reshape(1))
            out_chunks.append(rhs)
        return torch.cat(out_chunks, dim=0)

    def solve(
        self,
        time_s: Tensor,
        vin_t: Tensor,
        v_dc_vec: Tensor,
        v_flat_init: Tensor | None = None,
    ) -> ChainTransientSolution:
        start = time_module.perf_counter()
        tg = time_s.to(device=self._device, dtype=self._dtype)
        vin_grid = vin_t.to(device=self._device, dtype=self._dtype)
        v_dc_vec = v_dc_vec.to(device=self._device, dtype=self._dtype)
        dt_vec = tg[1:] - tg[:-1]
        cap_t = tg.shape[0]
        if dt_vec.numel() == 0:
            raise ValueError("time_s must contain at least two samples")
        n = self.n_stages
        if v_flat_init is None:
            v_flat = (v_dc_vec.unsqueeze(1).expand(n, cap_t)).reshape(-1).clone().requires_grad_(True)
        else:
            v_flat = v_flat_init.to(device=self._device, dtype=self._dtype).clone().requires_grad_(True)

        def res_norm_state(v_: Tensor) -> float:
            with torch.no_grad():
                return float(_inf_norm(self._full_residual_flat(v_, vin_grid, v_dc_vec, dt_vec)))

        rn = res_norm_state(v_flat.detach())
        hist = [rn]
        converged = rn <= self.residual_tol
        iters = 0
        while not converged and iters < self.max_iter:
            iters += 1

            def rfn_outer(x: Tensor) -> Tensor:
                return self._full_residual_flat(x, vin_grid, v_dc_vec, dt_vec)

            jac = jacobian(lambda z: rfn_outer(z), v_flat, vectorize=True, create_graph=False)
            res_vec = rfn_outer(v_flat)
            direction = torch.linalg.solve(jac, -res_vec.detach())
            if rn >= self.monotone_switch_residual:
                alpha, _ = _backtrack_monotone(
                    rfn_outer,
                    v_flat.detach(),
                    direction,
                    rn,
                    norm_fn=_inf_norm,
                    alpha_min=self.alpha_min,
                )
            else:
                alpha, _ = _backtrack(
                    rfn_outer,
                    v_flat.detach(),
                    direction,
                    rn,
                    norm_fn=_inf_norm,
                    armijo_c=self.armijo_c,
                    alpha_min=self.alpha_min,
                )
            alpha = _cap_alpha(alpha, direction, self.v_step_cap)
            v_flat = (v_flat.detach() + alpha * direction).clamp(0.0, self.vdd).requires_grad_(True)
            rn = res_norm_state(v_flat.detach())
            hist.append(rn)
            converged = rn <= self.residual_tol
        wall_ms = 1000.0 * (time_module.perf_counter() - start)
        sol_vec = v_flat.detach().reshape(n, cap_t)
        return ChainTransientSolution(
            time_s=tg.detach(),
            v_nodes_v=sol_vec,
            report=ConvergenceReport(
                converged=converged,
                iter_count=iters,
                residual_norm_history=tuple(hist),
                final_residual_norm=rn,
                wall_ms=wall_ms,
            ),
        )
