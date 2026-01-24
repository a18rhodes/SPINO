# %% [markdown]
### Model Definition
# This module defines the neural operator model used for learning the RC circuit dynamics.

# %%
from neuralop.models import FNO
from torch import nn


# %%
def get_model():
    return FNO(
        n_modes=(256,),
        hidden_channels=64,
        in_channels=2,
        out_channels=1,
        preactivation=True,
        fno_skip="linear",
        non_linearity=nn.functional.silu,
        domain_padding=0.1,
    ).cuda()
