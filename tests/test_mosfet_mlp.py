"""Unit tests for MosfetMLP — per-timestep quasi-static MOSFET baseline."""

# pytest fixtures shadow outer names by design; suppress the pylint warning globally.
# pylint: disable=redefined-outer-name

import pytest
import torch

from spino.mosfet.model import MosfetMLP

BATCH = 4
TIME = 128
PARAM_DIM = 29
EMB_DIM = 16
HIDDEN = 64


@pytest.fixture
def model():
    """Small MosfetMLP suitable for fast CPU unit tests."""
    return MosfetMLP(
        input_param_dim=PARAM_DIM,
        embedding_dim=EMB_DIM,
        hidden_dim=HIDDEN,
        n_hidden_layers=2,
    )


@pytest.fixture
def inputs():
    """Random (v_terminals, physical_params) pair for the batch/time dimensions above."""
    v = torch.randn(BATCH, 4, TIME)
    p = torch.randn(BATCH, PARAM_DIM)
    return v, p


def test_output_shape(model, inputs):
    """Output tensor must have shape (B, 1, T)."""
    v, p = inputs
    out = model(v, p)
    assert out.shape == (BATCH, 1, TIME), f"Expected ({BATCH}, 1, {TIME}), got {out.shape}"


def test_gradient_flows(model, inputs):
    """Autograd must propagate through v_terminals and physical_params."""
    v, p = inputs
    v.requires_grad_(True)
    p.requires_grad_(True)
    out = model(v, p)
    loss = out.sum()
    loss.backward()
    assert v.grad is not None, "No gradient w.r.t. v_terminals"
    assert p.grad is not None, "No gradient w.r.t. physical_params"


def test_no_temporal_mixing(model, inputs):
    """Off-diagonal Jacobian dI[t] / dV[t'] must be exactly zero for t != t'."""
    v, p = inputs
    v = v[:1].detach().requires_grad_(True)  # single sample
    p_single = p[:1].detach()
    out = model(v, p_single)  # (1, 1, T)
    t_check = TIME // 2
    # Gradient of output at t_check w.r.t. all input timesteps
    grad = torch.autograd.grad(out[0, 0, t_check], v, retain_graph=False)[0]
    # grad shape: (1, 4, T); off-diagonal columns (t != t_check) must be zero
    off_diag = grad[0, :, :t_check].abs().max().item()
    off_diag_right = grad[0, :, t_check + 1 :].abs().max().item()
    assert off_diag == 0.0, f"Non-zero off-diagonal Jacobian before t_check: {off_diag}"
    assert off_diag_right == 0.0, f"Non-zero off-diagonal Jacobian after t_check: {off_diag_right}"


def test_batch_independence(model, inputs):
    """Each sample in a batch must be processed independently."""
    v, p = inputs
    out_batch = model(v, p)
    out_single = model(v[:1], p[:1])
    assert torch.allclose(out_batch[:1], out_single, atol=1e-6), "Batch item 0 differs from solo run"


def test_different_time_lengths(model):
    """Model must accept arbitrary time-axis lengths (quasi-static = no fixed grid)."""
    p = torch.randn(2, PARAM_DIM)
    for t_len in (64, 256, 512, 1024):
        v = torch.randn(2, 4, t_len)
        out = model(v, p)
        assert out.shape == (2, 1, t_len), f"Wrong shape at t_len={t_len}: {out.shape}"
