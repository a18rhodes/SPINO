"""
Unit tests for the differentiable FNO device wrapper.

The full production checkpoints are too heavy to load in CI, so these
tests rely on a deterministic linear mock model that exercises every
arithmetic path of :class:`spino.circuit.devices.FnoMosfetDevice`:

* shape contracts on ``v_terminals``
* z-score normalization of voltages and physics
* arcsinh-mA decoding to amperes
* differentiability with respect to ``v_terminals``
* polarity convention (training label sign embedded, no extra flip)
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from spino.circuit.devices import FnoMosfetDevice
from spino.constants import ARCSINH_SCALE_MA

_MA_TO_A = 1.0e-3


class _LinearMockFno(nn.Module):
    """
    Linear surrogate matching ``MosfetVCFiLMFNO``'s public signature.

    The mock returns ``slope * v_d + intercept`` in arcsinh-mA space so the
    decoded amperes have a closed-form expression for direct verification.

    :param slope: Coefficient applied to the drain channel (``v_d``).
    :param intercept: Additive bias in arcsinh-mA space.
    """

    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.intercept = nn.Parameter(torch.tensor(intercept))

    def forward(self, v_terminals: Tensor, physical_params: Tensor) -> Tensor:
        """
        Returns ``slope * v_d + intercept`` broadcast across time.

        :param v_terminals: ``(B, 4, T)`` voltage trajectories (normalized).
        :param physical_params: ``(B, P)`` physics tensor (unused).
        :return: ``(B, 1, T)`` arcsinh-mA prediction.
        """
        del physical_params
        v_d = v_terminals[:, 1:2, :]
        return self.slope * v_d + self.intercept


def _identity_normalization(physics_dim: int = 4) -> dict[str, Tensor]:
    """
    Builds a no-op normalization triple plus a zero physics vector.

    :param physics_dim: Number of curated physics parameters in the mock.
    :return: Mapping from field name to tensor.
    """
    return {
        "v_mean": torch.zeros((4, 1)),
        "v_std": torch.ones((4, 1)),
        "p_mean": torch.zeros(physics_dim),
        "p_std": torch.ones(physics_dim),
        "physics_raw": torch.zeros(physics_dim),
    }


def _build_wrapper(model: nn.Module, **overrides) -> FnoMosfetDevice:
    """
    Constructs a :class:`FnoMosfetDevice` with optional overrides.

    :param model: Inner mock module.
    :param overrides: Optional buffer overrides to test specific paths.
    :return: Configured wrapper.
    """
    spec = _identity_normalization()
    spec.update(overrides)
    return FnoMosfetDevice(model=model, label="MOCK", **spec)


class TestShapeContracts:
    """Validates input/output shape invariants of the wrapper."""

    def test_physics_shape_mismatch_raises(self) -> None:
        bad = torch.zeros(3)
        with pytest.raises(ValueError, match="physics_raw"):
            FnoMosfetDevice(
                model=_LinearMockFno(),
                v_mean=torch.zeros((4, 1)),
                v_std=torch.ones((4, 1)),
                p_mean=torch.zeros(4),
                p_std=torch.ones(4),
                physics_raw=bad,
                label="bad",
            )

    def test_p_std_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="p_mean/p_std"):
            FnoMosfetDevice(
                model=_LinearMockFno(),
                v_mean=torch.zeros((4, 1)),
                v_std=torch.ones((4, 1)),
                p_mean=torch.zeros(4),
                p_std=torch.ones(3),
                physics_raw=torch.zeros(4),
                label="bad",
            )

    def test_forward_alias_matches_drain_current(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.1, intercept=0.0))
        v = torch.zeros((1, 4, 4))
        v[:, 1, :] = 0.5
        torch.testing.assert_close(wrapper.forward(v), wrapper.drain_current(v))

    def test_output_shape_matches_input(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.0, intercept=0.0))
        v_terminals = torch.zeros((3, 4, 16))
        result = wrapper.drain_current(v_terminals)
        assert result.shape == (3, 1, 16)

    def test_v_mean_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="v_mean/v_std"):
            FnoMosfetDevice(
                model=_LinearMockFno(),
                v_mean=torch.zeros((3, 1)),
                v_std=torch.ones((4, 1)),
                p_mean=torch.zeros(4),
                p_std=torch.ones(4),
                physics_raw=torch.zeros(4),
                label="bad",
            )

    def test_invalid_terminal_count_raises(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno())
        with pytest.raises(ValueError):
            wrapper.drain_current(torch.zeros((1, 3, 16)))

    def test_invalid_dimensionality_raises(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno())
        with pytest.raises(ValueError):
            wrapper.drain_current(torch.zeros((4, 16)))


class TestNormalizationAndDecoding:
    """Validates z-score normalization and arcsinh-mA decoding paths."""

    def test_zero_input_decodes_to_zero(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.0, intercept=0.0))
        v_terminals = torch.full((1, 4, 8), 1.5)
        result = wrapper.drain_current(v_terminals)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_arcsinh_decoding_matches_closed_form(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=1.0, intercept=0.0))
        v_terminals = torch.zeros((1, 4, 8))
        v_terminals[:, 1, :] = 0.7
        result = wrapper.drain_current(v_terminals)
        expected_a = ARCSINH_SCALE_MA * math.sinh(0.7) * _MA_TO_A
        torch.testing.assert_close(result, torch.full_like(result, expected_a), rtol=1e-6, atol=1e-15)

    def test_voltage_normalization_subtracts_mean(self) -> None:
        wrapper = _build_wrapper(
            _LinearMockFno(slope=1.0, intercept=0.0),
            v_mean=torch.tensor([[0.0], [0.7], [0.0], [0.0]]),
        )
        v_terminals = torch.zeros((1, 4, 4))
        v_terminals[:, 1, :] = 0.7
        result = wrapper.drain_current(v_terminals)
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=1e-6, atol=1e-15)

    def test_voltage_normalization_divides_std(self) -> None:
        wrapper = _build_wrapper(
            _LinearMockFno(slope=1.0, intercept=0.0),
            v_std=torch.tensor([[1.0], [2.0], [1.0], [1.0]]),
        )
        v_terminals = torch.zeros((1, 4, 4))
        v_terminals[:, 1, :] = 1.0
        result = wrapper.drain_current(v_terminals)
        expected_a = ARCSINH_SCALE_MA * math.sinh(0.5) * _MA_TO_A
        torch.testing.assert_close(result, torch.full_like(result, expected_a), rtol=1e-6, atol=1e-15)

    def test_physics_norm_applied_with_batch_dim(self) -> None:
        wrapper = _build_wrapper(
            _LinearMockFno(),
            p_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
            p_std=torch.tensor([2.0, 2.0, 2.0, 2.0]),
            physics_raw=torch.tensor([3.0, 4.0, 5.0, 6.0]),
        )
        normalized = wrapper.physics_norm
        torch.testing.assert_close(
            normalized,
            torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        )


class TestAutogradFlow:
    """Validates differentiability with respect to ``v_terminals``."""

    def test_gradient_flows_to_drain_channel(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=1.0, intercept=0.0))
        v_terminals = torch.zeros((1, 4, 8), requires_grad=True)
        i_d = wrapper.drain_current(v_terminals)
        i_d.sum().backward()
        assert v_terminals.grad is not None
        drain_grad = v_terminals.grad[:, 1, :]
        non_drain_grad = torch.cat([v_terminals.grad[:, 0:1, :], v_terminals.grad[:, 2:, :]], dim=1)
        assert (drain_grad.abs() > 0).all()
        torch.testing.assert_close(non_drain_grad, torch.zeros_like(non_drain_grad))

    def test_finite_difference_matches_autograd(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.5, intercept=0.0))
        v_grad_input = torch.zeros((1, 4, 8))
        v_grad_input[:, 1, :] = 0.3
        v_grad_input.requires_grad_(True)
        i_d = wrapper.drain_current(v_grad_input).sum()
        i_d.backward()
        eps = 1e-3
        v_plus = v_grad_input.detach().clone()
        v_plus[:, 1, :] += eps
        v_minus = v_grad_input.detach().clone()
        v_minus[:, 1, :] -= eps
        with torch.no_grad():
            i_plus = wrapper.drain_current(v_plus).sum()
            i_minus = wrapper.drain_current(v_minus).sum()
        finite_diff = (i_plus - i_minus) / (2 * eps)
        autograd_total = float(v_grad_input.grad[:, 1, :].sum())
        torch.testing.assert_close(torch.tensor(autograd_total), finite_diff, rtol=1e-3, atol=1e-9)


class TestPolarityConvention:
    """Validates the polarity convention is honored without re-flipping."""

    def test_positive_pred_log_yields_positive_current(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.0, intercept=2.0))
        v_terminals = torch.zeros((1, 4, 4))
        result = wrapper.drain_current(v_terminals)
        assert (result > 0).all()

    def test_negative_pred_log_yields_negative_current(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno(slope=0.0, intercept=-2.0))
        v_terminals = torch.zeros((1, 4, 4))
        result = wrapper.drain_current(v_terminals)
        assert (result < 0).all()


class TestBufferRegistration:
    """Validates buffers track device/dtype changes via ``.to``."""

    def test_buffers_move_to_double_precision(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno()).to(torch.float64)
        assert wrapper.v_mean.dtype == torch.float64
        assert wrapper.physics_raw.dtype == torch.float64

    def test_extra_repr_mentions_label(self) -> None:
        wrapper = _build_wrapper(_LinearMockFno())
        assert "MOCK" in repr(wrapper)
