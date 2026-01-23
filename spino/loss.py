from typing import Callable

import torch
from torch import nn


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
