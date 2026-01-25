# %% [markdown]
### Model Definition
# This module defines the neural operator model used for learning the RC circuit dynamics.

# %%
import torch.nn as nn
from neuralop.models import FNO


# %%
def get_model():
    """
    Returns an FNO configured for the Diode task.
    In Channels: 5
        1. I(t) (mA)
        2. log10(R)
        3. log10(C)
        4. log10(Is)
        5. N
    Out Channels: 1 (Voltage)
    """
    return FNO(
        n_modes=(256,),
        hidden_channels=64,
        in_channels=5,
        out_channels=1,
        preactivation=True,
        fno_skip="linear",
        non_linearity=nn.functional.silu,
        domain_padding=0.1,
    ).cuda()
