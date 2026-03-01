"""
Loss functions for neural operator training on analog circuit simulations.

All losses operate in arcsinh-transformed current space with
``ARCSINH_SCALE_MA = 1e-6``. At this scale, subthreshold currents map to
arcsinh ∈ [0, 7] and saturation currents to [7, 17], providing implicit
multi-scale balance without explicit weighting.
"""

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ArcSinhMSELoss",
    "GenericDimensionlessPhysicsLoss",
    "LpLoss",
    "LpLossWithFloor",
    "Log10Loss",
    "QuarticWeightedLoss",
    "RegionAdaptiveLoss",
    "SubthresholdWeightedLoss",
    "rc_physics_residual",
]


def rc_physics_residual(V_mid, dV_dt, inputs):
    """
    Defines the Dimensionless RC Equation: Lambda * dV/dt + V - I = 0
    inputs: [Batch, Channels, Time] (Already time-aligned)
            Ch0 = I_hat
            Ch1 = Lambda
    """
    I_hat = inputs[:, 0:1, :]
    Lambda = inputs[:, 1:2, :]
    return (Lambda * dV_dt) + V_mid - I_hat


class GenericDimensionlessPhysicsLoss(nn.Module):
    def __init__(self, physics_residual_fn: Callable, sobolev_weight=1.0, physics_weight=1e-4):
        super().__init__()
        self.physics_residual_fn = physics_residual_fn
        self.sobolev_weight = sobolev_weight
        self.physics_weight = physics_weight
        self.mse = nn.MSELoss()

    def forward(self, pred_v, target_v, inputs):
        """
        inputs: [Batch, Channels, Time]
        pred_v: [Batch, 1, Time]
        """
        # 1. Data Loss (MSE on shape)
        loss_data = self.mse(pred_v, target_v)

        loss_sobolev = torch.tensor(0.0, device=pred_v.device)
        loss_physics = torch.tensor(0.0, device=pred_v.device)

        if self.sobolev_weight > 0 or self.physics_weight > 0:
            # Grid step in normalized time (t runs 0 to 1)
            # We derive it dynamically from the shape to be safe
            t_steps = pred_v.shape[-1]
            dt_hat = 1.0 / float(t_steps)

            # dV_hat / dt_hat
            # Shape becomes [Batch, 1, Time-1]
            pred_delta = pred_v[:, :, 1:] - pred_v[:, :, :-1]
            dV_dt = pred_delta / dt_hat

            # 2. Sobolev Loss
            if self.sobolev_weight > 0:
                target_delta = target_v[:, :, 1:] - target_v[:, :, :-1]
                target_dV_dt = target_delta / dt_hat
                loss_sobolev = self.mse(dV_dt, target_dV_dt)

            # 3. Physics Loss
            if self.physics_weight > 0:
                # Align inputs to the derivative grid (0 to T-1)
                inputs_aligned = inputs[:, :, :-1]

                # Midpoint Voltage (Crank-Nicolson alignment)
                V_hat_mid = (pred_v[:, :, :-1] + pred_v[:, :, 1:]) / 2.0

                residual = self.physics_residual_fn(V_hat_mid, dV_dt, inputs_aligned)

                loss_physics = torch.mean(torch.abs(residual))

        total_loss = loss_data + (self.sobolev_weight * loss_sobolev) + (self.physics_weight * loss_physics)
        return total_loss, loss_data, loss_sobolev, loss_physics


class SubthresholdWeightedLoss(nn.Module):
    """
    MSE loss with exponential magnitude-based weighting for small currents.

    Addresses subthreshold oscillation problem where standard MSE heavily
    weights saturation region (mA scale) over subthreshold (nA scale).
    Applies inverse magnitude weighting: small currents get ~100x weight,
    large currents get ~1x weight.

    Weight formula: w = 1 / (|I_target| + scale)^exponent
    Loss is computed in arcsinh-transformed space but weights are based
    on physical current magnitude.
    """

    def __init__(self, scale_mA=0.01, exponent=2.0):
        """
        :param scale_mA: Current scale for weighting (smaller = more aggressive subthreshold focus).
        :param exponent: Exponent for weight gradient (higher = steeper weight curve).
        """
        super().__init__()
        self.register_buffer("scale_mA", torch.tensor(scale_mA, dtype=torch.float32))
        self.register_buffer("exponent", torch.tensor(exponent, dtype=torch.float32))

    def forward(self, pred_arcsinh, target_arcsinh):
        """
        Computes weighted MSE in PHYSICAL SPACE (mA), not arcsinh space.

        Critical: Arcsinh-space errors are incompatible with magnitude weighting.
        For subthreshold currents (1 nA), arcsinh ≈ linear, so squared errors
        are ~1e-12. Even with 1000x weighting, saturation errors (MSE ~0.1)
        dominate by 8 orders of magnitude.

        Solution: Transform back to physical space, compute error there,
        apply magnitude-based weighting, then average.

        :param pred_arcsinh: Predicted current in arcsinh-transformed space [Batch, 1, Time].
        :param target_arcsinh: Target current in arcsinh-transformed space [Batch, 1, Time].
        :return: Weighted MSE loss in physical space (mA).
        """
        pred_ma = torch.sinh(pred_arcsinh)
        target_ma = torch.sinh(target_arcsinh)
        weights = self.scale_mA / (torch.abs(target_ma) + self.scale_mA)
        weights = torch.pow(weights, self.exponent)
        mse_physical = (pred_ma - target_ma) ** 2
        return (weights * mse_physical).mean()


class LpLoss(nn.Module):
    """
    Calculates the relative Lp loss between the prediction and the target.
    L = ||x - y||_p / ||y||_p

    Used in Neural Operator training to measure function-space error,
    making the metric resolution-invariant.
    """

    def __init__(self, d=1, p=2, reduction="mean"):
        """
        :param d: Dimension of the data (1 for time-series).
        :param p: Lp norm (p=2 is Euclidean/L2 norm).
        :param reduction: 'mean', 'sum', or 'none'.
        """
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.register_buffer("epsilon", torch.tensor(1e-9, dtype=torch.float32))

    def forward(self, x, y):
        """
        :param x: Prediction tensor [Batch, ...].
        :param y: Target tensor [Batch, ...].
        """
        # Flatten all dimensions except batch to compute vector norm per sample
        num_examples = x.size(0)

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        # Avoid division by zero (though y_norms=0 is physically impossible for active circuits)
        loss = diff_norms / (y_norms + self.epsilon)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)

        return loss


class LpLossWithFloor(nn.Module):
    """
    Relative Lp loss with a hard minimum on the denominator.

    Identical to :class:`LpLoss` for samples with large signal energy, but
    clamps the per-sample denominator to ``floor`` to prevent catastrophic
    gradient amplification on deep-subthreshold waveforms where ``||y||_2``
    collapses near zero in arcsinh space.

    Calibration guidance: for a 2007-point series with arcsinh values in
    ``[0, 2]`` (weak subthreshold), ``||y||_2 ≈ sqrt(2007) * 1 ≈ 45``.  A
    floor of 10–20 protects that regime without touching saturation samples
    whose ``||y||_2`` typically exceeds 200.
    """

    def __init__(self, p: int = 2, floor: float = 10.0):
        """
        :param p: Lp norm order.
        :param floor: Minimum denominator value in arcsinh units.
        """
        super().__init__()
        self.p = p
        self.register_buffer("floor", torch.tensor(floor, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: Prediction tensor [Batch, ...].
        :param y: Target tensor [Batch, ...].
        :return: Scalar mean relative Lp loss with floor-clamped denominator.
        """
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        return torch.mean(diff_norms / torch.clamp(y_norms, min=self.floor))


class ArcSinhMSELoss(nn.Module):
    """
    Plain MSE loss applied directly in arcsinh-transformed current space.

    Eliminates the relative-norm denominator of :class:`LpLoss` entirely.
    This prevents the denominator-collapse failure mode where deep-subthreshold
    samples (``||y||_2 ≈ 0``) receive catastrophically amplified gradients.

    The arcsinh encoding with ``ARCSINH_SCALE_MA = 1e-6`` already compresses
    the current dynamic range: subthreshold maps to ``[0, 7]`` and saturation
    to ``[7, 17]``.  Standard MSE in this space is therefore implicitly
    multi-scale balanced without any explicit weighting or denominator term.

    The absence of resolution-invariant normalisation is intentional: training
    and inference always use the same fixed 2007-point time grid, so
    resolution invariance offers no benefit over this architecture.
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        :param x: Prediction tensor in arcsinh space [Batch, 1, Time].
        :param y: Target tensor in arcsinh space [Batch, 1, Time].
        :return: Scalar mean squared error.
        """
        return F.mse_loss(x, y)


class Log10Loss(nn.Module):
    """
    MSE loss in log10-transformed current space.

    Naturally balances errors across decades: a 1nA→10nA error has the
    same magnitude as a 1mA→10mA error (both are Δlog10=1.0).
    No explicit weighting required.
    """

    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float32))

    def forward(self, pred_arcsinh, target_arcsinh):
        pred_ma = torch.sinh(pred_arcsinh)
        target_ma = torch.sinh(target_arcsinh)
        pred_abs = pred_ma.abs()
        target_abs = target_ma.abs()
        pred_log = torch.log10(pred_abs.add(self.epsilon))
        target_log = torch.log10(target_abs.add(self.epsilon))
        diff = pred_log.sub(target_log)
        return diff.pow(2).mean()


class RegionAdaptiveLoss(nn.Module):
    """
    Multi-task loss with separate terms for subthreshold and saturation regions.

    Splits the loss computation into two components based on gate voltage:
    - Subthreshold region: Vg < vth_threshold (default 0.5V in raw space)
    - Saturation region: Vg >= vth_threshold

    CRITICAL: Voltages are z-score normalized, so threshold must be transformed.
    For 40K dataset: Vg_mean=0.755V, Vg_std=0.476V
    Raw threshold 0.5V → normalized threshold -0.536

    Each region's loss is computed independently using LpLoss, then combined
    with tunable weights to balance capacity allocation between nA-scale and
    mA-scale physics.
    """

    def __init__(self, subth_weight=1.0, sat_weight=1.0, vth_threshold=0.5, vg_mean=0.755, vg_std=0.476):
        """
        :param subth_weight: Weight for subthreshold region loss (higher = more focus on nA-scale).
        :param sat_weight: Weight for saturation region loss (higher = more focus on mA-scale).
        :param vth_threshold: Gate voltage threshold in RAW voltage space (volts).
        :param vg_mean: Mean gate voltage for normalization (from dataset stats).
        :param vg_std: Std gate voltage for normalization (from dataset stats).
        """
        super().__init__()
        self.subth_weight = subth_weight
        self.sat_weight = sat_weight
        self.vth_threshold_normalized = (vth_threshold - vg_mean) / vg_std
        self.lp_loss = LpLoss(d=1, p=2, reduction="mean")

    def forward(self, pred, target, voltages):
        """
        Computes region-weighted loss.

        Vectorized equivalent of per-sample LpLoss on boolean-indexed subsets.
        Identity: sqrt(sum((p-y)^2 * mask)) == ||diff[mask]||_2 because 0^2=0.

        :param pred: Predicted current [Batch, 1, Time].
        :param target: Target current [Batch, 1, Time].
        :param voltages: NORMALIZED input voltages [Batch, 4, Time] (Vg, Vd, Vs, Vb).
        :return: Weighted sum of subthreshold and saturation losses.
        """
        vg = voltages[:, 0:1, :]
        subth_mask = (vg < self.vth_threshold_normalized).float()
        sat_mask = 1.0 - subth_mask
        diff_sq = (pred - target) ** 2
        target_sq = target**2
        subth_diff_norm = torch.sqrt((diff_sq * subth_mask).sum(dim=(1, 2)) + self.lp_loss.epsilon)
        subth_y_norm = torch.sqrt((target_sq * subth_mask).sum(dim=(1, 2)) + self.lp_loss.epsilon)
        subth_lp = subth_diff_norm / (subth_y_norm + self.lp_loss.epsilon)
        subth_valid = subth_mask.sum(dim=(1, 2)) > 0
        if subth_valid.any():
            loss_subth = subth_lp[subth_valid].mean()
        else:
            loss_subth = torch.tensor(0.0, device=pred.device)
        sat_diff_norm = torch.sqrt((diff_sq * sat_mask).sum(dim=(1, 2)) + self.lp_loss.epsilon)
        sat_y_norm = torch.sqrt((target_sq * sat_mask).sum(dim=(1, 2)) + self.lp_loss.epsilon)
        sat_lp = sat_diff_norm / (sat_y_norm + self.lp_loss.epsilon)
        sat_valid = sat_mask.sum(dim=(1, 2)) > 0
        if sat_valid.any():
            loss_sat = sat_lp[sat_valid].mean()
        else:
            loss_sat = torch.tensor(0.0, device=pred.device)
        return self.subth_weight * loss_subth + self.sat_weight * loss_sat
