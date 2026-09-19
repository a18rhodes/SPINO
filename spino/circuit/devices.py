"""
Differentiable FNO device adapters for neural circuit composition.

Wraps a trained ``MosfetVCFiLMFNO`` together with the dataset normalization
statistics and the curated 29-parameter physics tensor for a single (model,
W, L) point. Exposes a single differentiable primitive,
:meth:`FnoMosfetDevice.drain_current`, that maps raw four-terminal voltage
trajectories to the device drain current in amperes.

The wrapper performs three jobs the bare ``MosfetVCFiLMFNO`` does not:

1. Normalizes ``v_terminals`` and the static physics vector using the
   dataset z-score statistics the checkpoint was trained against.
2. Decodes the arcsinh-mA output back to physical amperes.
3. Encodes the polarity convention: training labels were already multiplied
   by ``DeviceStrategy.current_polarity_multiplier`` during data generation,
   so the decoded output is the *magnitude* of conventional drain current,
   positive when the device is in the conducting regime, for both NMOS and
   PMOS. Sign assignment at the KCL node is the caller's responsibility.

All buffers are registered on the wrapper module so ``.to(device)`` /
``.cuda()`` move them in lockstep with the underlying FNO.
"""

from __future__ import annotations

import torch
from torch import nn

from spino.constants import ARCSINH_SCALE_MA

__all__ = ["FnoMosfetDevice"]

_MA_TO_A = 1.0e-3


class FnoMosfetDevice(nn.Module):
    """
    Differentiable per-device wrapper around a trained MOSFET FNO.

    :param model: Trained ``MosfetVCFiLMFNO`` (or compatible architecture
        whose ``forward(v_terminals, physics)`` returns the arcsinh-mA
        normalized current trajectory).
    :param v_mean: Per-terminal voltage mean of shape ``(4, 1)`` from the
        training dataset.
    :param v_std: Per-terminal voltage std of shape ``(4, 1)`` from the
        training dataset.
    :param p_mean: Per-parameter physics mean of shape ``(P,)``.
    :param p_std: Per-parameter physics std of shape ``(P,)``.
    :param physics_raw: Curated 29-parameter physics tensor ``(P,)`` for the
        device geometry under test (W, L, vth0, ...) in the order defined by
        :data:`spino.mosfet.gen_data.ParameterSchema.TRAINING_KEYS`.
    :param label: Human-readable identifier (e.g. ``"NFET XM1"``) used in
        repr and log messages.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        model: nn.Module,
        v_mean: torch.Tensor,
        v_std: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        physics_raw: torch.Tensor,
        label: str = "",
    ) -> None:
        super().__init__()
        self._validate_shapes(v_mean, v_std, p_mean, p_std, physics_raw)
        self.model = model
        self.label = label
        self.register_buffer("v_mean", v_mean.detach().to(torch.float32).clone())
        self.register_buffer("v_std", v_std.detach().to(torch.float32).clone())
        self.register_buffer("p_mean", p_mean.detach().to(torch.float32).clone())
        self.register_buffer("p_std", p_std.detach().to(torch.float32).clone())
        self.register_buffer("physics_raw", physics_raw.detach().to(torch.float32).clone())

    @staticmethod
    def _validate_shapes(
        v_mean: torch.Tensor,
        v_std: torch.Tensor,
        p_mean: torch.Tensor,
        p_std: torch.Tensor,
        physics_raw: torch.Tensor,
    ) -> None:
        """
        Asserts that the normalization tensors have the layouts the FNO
        expects.

        :raises ValueError: When any shape disagreement is detected.
        """
        if v_mean.shape != (4, 1) or v_std.shape != (4, 1):
            raise ValueError(f"v_mean/v_std must be (4, 1); got {tuple(v_mean.shape)}, {tuple(v_std.shape)}")
        if p_mean.ndim != 1 or p_std.shape != p_mean.shape:
            raise ValueError(f"p_mean/p_std must be 1D and same shape; got {tuple(p_mean.shape)}, {tuple(p_std.shape)}")
        if physics_raw.shape != p_mean.shape:
            raise ValueError(
                f"physics_raw must match p_mean shape; got {tuple(physics_raw.shape)} vs {tuple(p_mean.shape)}"
            )

    @property
    def physics_norm(self) -> torch.Tensor:
        """
        Returns the dataset-normalized curated physics vector with batch dim.

        :return: Tensor of shape ``(1, P)`` ready for batched FNO conditioning.
        """
        return ((self.physics_raw - self.p_mean) / self.p_std).unsqueeze(0)

    def drain_current(self, v_terminals: torch.Tensor) -> torch.Tensor:
        """
        Maps raw four-terminal voltage trajectories to drain current.

        :param v_terminals: Voltage trajectories ``(B, 4, T)`` in volts. The
            channel order is ``Vg, Vd, Vs, Vb`` to match the training
            convention enforced by :class:`spino.mosfet.gen_data.InfiniteSpiceMosfetDataset`.
        :return: Drain current ``I_d(t)`` of shape ``(B, 1, T)`` in amperes.
            Positive values indicate the device is in the conducting regime
            for both NMOS and PMOS (polarity convention baked into training
            labels).
        """
        if v_terminals.ndim != 3 or v_terminals.shape[1] != 4:
            raise ValueError(f"v_terminals must be (B, 4, T); got {tuple(v_terminals.shape)}")
        v_norm = (v_terminals - self.v_mean) / self.v_std
        physics_norm = self.physics_norm.expand(v_terminals.shape[0], -1)
        pred_log = self.model(v_norm, physics_norm)
        return ARCSINH_SCALE_MA * torch.sinh(pred_log) * _MA_TO_A

    def forward(self, v_terminals: torch.Tensor) -> torch.Tensor:
        """
        Convenience alias for :meth:`drain_current` so the wrapper can be
        composed inside larger ``nn.Module`` graphs and traced.

        :param v_terminals: Voltage trajectories ``(B, 4, T)`` in volts.
        :return: Drain current ``(B, 1, T)`` in amperes.
        """
        return self.drain_current(v_terminals)

    def extra_repr(self) -> str:
        """
        Adds the device label to the module ``repr`` for easier debugging.

        :return: Single-line summary string.
        """
        return f"label='{self.label}', physics_dim={self.physics_raw.numel()}"
