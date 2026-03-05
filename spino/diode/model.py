"""
Defines the FNO architecture for the diode circuit neural operator.

Two configurations are provided:

- **Legacy (5-channel):** I(mA), log10(R), log10(C), log10(Is), N.
  Fixed temporal grid, no dimensionless formulation.
- **Dimensionless (6-channel):** I_hat, lambda, log10(R), log10(C), log10(Is), N.
  Variable T_end, resolution-invariant.
"""

import torch.nn as nn
from neuralop.models import FNO

__all__ = ["DiodeFNO", "get_model", "I_SCALE_A"]

# Channel contract: dimensionless formulation
# Ch 0: I_hat(t_hat)  -- normalized current waveform (+-1)
# Ch 1: lambda        -- stiffness ratio RC/T_end (constant, broadcast)
# Ch 2: log10(R)      -- resistance (constant, broadcast)
# Ch 3: log10(C)      -- capacitance (constant, broadcast)
# Ch 4: log10(Is)     -- saturation current (constant, broadcast)
# Ch 5: N             -- ideality factor (constant, broadcast)
_DIMENSIONLESS_IN_CHANNELS = 6

# Channel contract: legacy formulation
# Ch 0: I(mA)         -- current in milliamps
# Ch 1: log10(R)      -- resistance (constant, broadcast)
# Ch 2: log10(C)      -- capacitance (constant, broadcast)
# Ch 3: log10(Is)     -- saturation current (constant, broadcast)
# Ch 4: N             -- ideality factor (constant, broadcast)
_LEGACY_IN_CHANNELS = 5

# Normalization constant for current (Amps).
# PWL source range is +-5mA, so I_hat = I / I_SCALE gives +-1.
I_SCALE_A = 5e-3


class DiodeFNO(nn.Module):
    """
    Fourier Neural Operator for diode circuit transient prediction.

    Wraps the neuralop FNO with a configurable channel contract to support
    both legacy (5-channel, fixed grid) and dimensionless (6-channel,
    variable T_end) formulations.

    :param in_channels: Number of input channels.
    :param n_modes: Tuple of Fourier mode counts per spatial dimension.
    :param hidden_channels: Width of the hidden FNO layers.
    :param domain_padding: Fractional padding applied before FFT (reduces boundary artifacts).
    """

    def __init__(
        self,
        in_channels: int = _DIMENSIONLESS_IN_CHANNELS,
        n_modes: tuple[int, ...] = (256,),
        hidden_channels: int = 64,
        domain_padding: float = 0.1,
    ):
        super().__init__()
        self.fno = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=1,
            preactivation=True,
            fno_skip="linear",
            non_linearity=nn.functional.silu,
            domain_padding=domain_padding,
        )

    def forward(self, x):
        """
        Forward pass through the FNO backbone.

        :param x: Input tensor of shape [B, in_channels, T].
        :return: Predicted output of shape [B, 1, T].
        """
        return self.fno(x)


def get_model(dimensionless: bool = True) -> DiodeFNO:
    """
    Factory for the default diode FNO configuration.

    :param dimensionless: If True, returns 6-channel model (with lambda).
        If False, returns legacy 5-channel model.
    :return: DiodeFNO instance on CUDA.
    """
    in_ch = _DIMENSIONLESS_IN_CHANNELS if dimensionless else _LEGACY_IN_CHANNELS
    return DiodeFNO(in_channels=in_ch).cuda()
