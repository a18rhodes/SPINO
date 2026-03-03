"""
Defines the Neural Operator architecture for MOSFET modeling.

This module implements the physics-aware encoder (DeviceEmbedding) and the
temporal operator (MosfetFNO) that maps device parameters and bias conditions
to transient current responses.
"""

import torch
import torch.nn as nn
from neuralop.models import FNO
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.spectral_convolution import SpectralConv

__all__ = ["DeviceEmbedding", "MosfetFNO", "BatchedFiLM", "MosfetFiLMFNO", "VoltageConditionedFiLM", "MosfetVCFiLMFNO"]


class BatchedFiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) with batched conditioning support.

    Unlike AdaIN, this does NOT normalize the input features over the spatial/time
    dimension. Normalizing over time destroys the DC component (mean voltage) and
    amplitude (voltage swing) of the waveforms, which are critical physical signals.
    """

    def __init__(self, embed_dim: int, in_channels: int, eps: float = 1e-5) -> None:
        """
        Initialize the Batched FiLM layer.

        :param embed_dim: Dimension of the conditioning embedding.
        :param in_channels: Number of channels in the feature map to normalize.
        :param eps: Small constant for numerical stability.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.eps = eps

        # MLP to generate scale (weight) and shift (bias) from embedding
        self.mlp = nn.Sequential(nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, in_channels * 2))

        # Initialize the final linear layer to output weight=1 and bias=0
        # This ensures the FiLM layer starts as an identity mapping
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        # We want the first half (scale) to be 1, and second half (shift) to be 0
        with torch.no_grad():
            self.mlp[-1].bias[:in_channels] = 1.0

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Apply feature-wise linear modulation.

        :param x: Feature map (Batch, Channels, Time).
        :param embedding: Conditioning vector (Batch, Embed_Dim).
        :return: Modulated feature map.
        """
        # Generate scale and shift: (Batch, 2 * Channels)
        style = self.mlp(embedding)

        # Split into weight and bias: (Batch, Channels)
        weight, bias = torch.split(style, self.in_channels, dim=1)

        # Reshape for broadcasting over spatial/temporal dimensions: (Batch, Channels, 1)
        weight = weight.unsqueeze(-1)
        bias = bias.unsqueeze(-1)

        # Apply pure FiLM modulation (NO InstanceNorm)
        # We MUST NOT normalize over the time dimension, because the DC component
        # (mean over time) and amplitude (variance over time) are the actual physical
        # voltage values. If we InstanceNorm a constant Vg=1.2V sweep, it becomes 0!
        return weight * x + bias


class VoltageConditionedFiLM(nn.Module):
    """
    Voltage-Conditioned Feature-wise Linear Modulation (VCFiLM).

    Unlike static BatchedFiLM which computes a SINGLE (scale, shift) from the
    physics embedding and applies it identically at every timestep, VCFiLM
    produces TIME-VARYING modulation by conditioning on both the physics
    embedding AND the instantaneous input voltages.

    This gives the model an explicit mechanism for regime-dependent spectral
    filtering: when Vg is in deep subthreshold, the FiLM applies different
    modulation than when Vg is in saturation, even for the same (W, L) device.
    The MLP operates pointwise over the time dimension via a shared-weight
    1D convolution pattern (Linear applied to channels-last).
    """

    def __init__(self, embed_dim: int, n_voltage_channels: int, in_channels: int) -> None:
        """
        Initialize the Voltage-Conditioned FiLM layer.

        :param embed_dim: Dimension of the physics conditioning embedding.
        :param n_voltage_channels: Number of input voltage channels (typically 4: Vg, Vd, Vs, Vb).
        :param in_channels: Number of channels in the feature map to modulate.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_voltage_channels = n_voltage_channels
        self.in_channels = in_channels
        combined_dim = embed_dim + n_voltage_channels
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Linear(128, in_channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        with torch.no_grad():
            self.mlp[-1].bias[:in_channels] = 1.0

    def forward(self, x: torch.Tensor, embedding: torch.Tensor, v_terminals: torch.Tensor) -> torch.Tensor:
        """
        Apply voltage-conditioned feature-wise linear modulation.

        :param x: Feature map (Batch, Channels, Time).
        :param embedding: Physics conditioning vector (Batch, Embed_Dim).
        :param v_terminals: Input voltages (Batch, 4, Time) — used for regime awareness.
        :return: Modulated feature map (Batch, Channels, Time).
        """
        time_steps = x.shape[-1]
        embed_expanded = embedding.unsqueeze(-1).expand(-1, -1, time_steps)
        combined = torch.cat([embed_expanded, v_terminals], dim=1)
        combined = combined.permute(0, 2, 1)
        style = self.mlp(combined)
        style = style.permute(0, 2, 1)
        weight, bias = torch.split(style, self.in_channels, dim=1)
        return weight * x + bias


class DeviceEmbedding(nn.Module):
    """
    Nonlinear encoder that compresses physical device parameters into a latent representation.

    This MLP acts as the bridge between the sparse, high-dimensional physical space
    (BSIM model cards) and the dense, low-dimensional manifold required by the FNO.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 32, hidden_dim: int = 128) -> None:
        """
        Initialize the embedding network.

        :param input_dim: Number of physical input parameters (from ParameterSchema).
        :param embedding_dim: Size of the output latent vector.
        :param hidden_dim: Hidden layer width in the MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project physical parameters to latent embedding.

        :param x: Batch of parameter vectors (Batch, Input_Dim).
        :return: Latent embeddings (Batch, Embedding_Dim).
        """
        return self.net(x)


class MosfetFNO(nn.Module):
    """
    Universal MOSFET Operator.

    Learns the mapping: (V_terminals(t), Params_physical) -> I_drain(t).
    This wraps the standard FNO implementation to handle the hybrid input
    (static physics + dynamic voltages).
    """

    def __init__(
        self,
        input_param_dim: int,
        embedding_dim: int = 32,
        modes: int = 128,
        width: int = 64,
        embedding_hidden_dim: int = 128,
    ) -> None:
        """
        Initialize the FNO architecture.

        :param input_param_dim: Dimension of the raw BSIM parameter vector.
        :param embedding_dim: Size of the device parameter embedding.
        :param modes: Number of Fourier modes to keep (frequency resolution).
        :param width: Channel width of the FNO layers.
        :param embedding_hidden_dim: Hidden dimension for the embedding MLP.
        """
        super().__init__()

        self.embedding = DeviceEmbedding(
            input_dim=input_param_dim, embedding_dim=embedding_dim, hidden_dim=embedding_hidden_dim
        )

        # Input Channels Strategy:
        # 4 Dynamic Channels: Vg(t), Vd(t), Vs(t), Vb(t)
        # N Static Channels: Latent Embedding (broadcasted over time)
        self.fno_in_channels = 4 + embedding_dim

        # Leverage the standard neuralop FNO implementation.
        # We use FNO1d (implied by n_modes tuple size) for time-series.
        self.fno = FNO(
            n_modes=(modes,),
            hidden_channels=width,
            in_channels=self.fno_in_channels,
            out_channels=1,  # Output: Id(t)
            preactivation=True,
            fno_skip="linear",
            non_linearity=nn.functional.silu,
            domain_padding=0.1,  # Critical for non-periodic transient signals
        )

    def forward(self, v_terminals: torch.Tensor, physical_params: torch.Tensor) -> torch.Tensor:
        """
        Predict transient drain current.

        :param v_terminals: Voltage waveforms (Batch, 4, Time).
                            Expected Order: Vg, Vd, Vs, Vb.
        :param physical_params: Raw physics vector (Batch, Param_Dim).
        :return: Drain current I_d(t) (Batch, 1, Time).
        """
        _, _, time_steps = v_terminals.shape

        # 1. Compress Physics: (Batch, P) -> (Batch, Emb)
        latent_vec = self.embedding(physical_params)

        # 2. Expand: (Batch, Emb) -> (Batch, Emb, Time)
        # We broadcast the static physics across the entire time window.
        latent_expanded = latent_vec.unsqueeze(-1).expand(-1, -1, time_steps)

        # 3. Concatenate: (Batch, 4, Time) + (Batch, Emb, Time) -> (Batch, 4+Emb, Time)
        x = torch.cat([v_terminals, latent_expanded], dim=1)

        # 4. Operator Inference
        return self.fno(x)


class MosfetFiLMFNO(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) MOSFET Operator.

    Learns the mapping: (V_terminals(t), Params_physical) -> I_drain(t).
    Unlike MosfetFNO which concatenates physics as static input channels,
    this architecture uses Feature-wise Linear Modulation (FiLM) to
    modulate the intermediate spectral feature maps based on the physics.
    """

    def __init__(
        self,
        input_param_dim: int,
        embedding_dim: int = 16,
        modes: int = 256,
        width: int = 64,
        embedding_hidden_dim: int = 128,
        n_layers: int = 4,
    ) -> None:
        """
        Initialize the FiLM FNO architecture.

        :param input_param_dim: Dimension of the raw BSIM parameter vector.
        :param embedding_dim: Size of the device parameter embedding.
        :param modes: Number of Fourier modes to keep (frequency resolution).
        :param width: Channel width of the FNO layers.
        :param embedding_hidden_dim: Hidden dimension for the embedding MLP.
        :param n_layers: Number of FNO blocks.
        """
        super().__init__()

        self.embedding = DeviceEmbedding(
            input_dim=input_param_dim, embedding_dim=embedding_dim, hidden_dim=embedding_hidden_dim
        )

        # Input Channels Strategy:
        # 4 Dynamic Channels: Vg(t), Vd(t), Vs(t), Vb(t)
        # Physics embedding is NOT concatenated. It is passed to AdaIN layers.
        self.fno_in_channels = 4
        self.width = width
        self.n_layers = n_layers

        # 1. Lifting Layer: Project 4 input channels to hidden width
        self.lifting = nn.Linear(self.fno_in_channels, self.width)

        # 2. FNO Blocks: We instantiate standard FNOBlocks but bypass its broken AdaIN
        self.fno_blocks = FNOBlocks(
            in_channels=self.width,
            out_channels=self.width,
            n_modes=(modes,),
            n_layers=self.n_layers,
            use_channel_mlp=True,
            preactivation=True,
            fno_skip="linear",
            non_linearity=nn.functional.silu,
            norm=None,  # Bypass neuralop's norm
            conv_module=SpectralConv,
        )

        # 3. Custom Batched FiLM Layers
        # FNOBlocks with channel_mlp uses 2 norm layers per block
        self.n_norms_per_block = 2
        self.film_layers = nn.ModuleList(
            [
                BatchedFiLM(embed_dim=embedding_dim, in_channels=self.width)
                for _ in range(self.n_layers * self.n_norms_per_block)
            ]
        )

        # 4. Projection Layer: Project hidden width to 1 output channel
        self.projection = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Domain padding for non-periodic signals
        self.domain_padding = 0.1

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad the input tensor along the time dimension."""
        pad_len = int(round(x.shape[-1] * self.domain_padding))
        x = nn.functional.pad(x, (0, pad_len), mode="replicate")
        return x, pad_len

    def _unpad(self, x: torch.Tensor, pad_len: int) -> torch.Tensor:
        """Remove padding from the output tensor."""
        return x[..., :-pad_len]

    def forward(self, v_terminals: torch.Tensor, physical_params: torch.Tensor) -> torch.Tensor:
        """
        Predict transient drain current using FiLM conditioning.

        :param v_terminals: Voltage waveforms (Batch, 4, Time).
        :param physical_params: Raw physics vector (Batch, Param_Dim).
        :return: Drain current I_d(t) (Batch, 1, Time).
        """
        # 1. Compress Physics: (Batch, P) -> (Batch, Emb)
        latent_vec = self.embedding(physical_params)

        # 2. Pad input for non-periodic boundaries
        x, pad_len = self._pad(v_terminals)

        # 3. Lifting: (Batch, 4, Time) -> (Batch, Width, Time)
        # Linear expects channels last, so we permute
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)

        # 4. FNO Blocks with interleaved BatchedFiLM
        # We manually execute the preactivation logic of FNOBlocks to inject our FiLM
        for i in range(self.n_layers):
            # --- Preactivation Block Logic ---
            # 4a. Non-linearity
            x = self.fno_blocks.non_linearity(x)

            # 4b. First FiLM (Norm 0)
            x = self.film_layers[self.n_norms_per_block * i](x, latent_vec)

            # 4c. FNO Skip Connection
            x_skip_fno = self.fno_blocks.fno_skips[i](x)
            x_skip_fno = self.fno_blocks.convs[i].transform(x_skip_fno)

            # 4d. Channel MLP Skip Connection
            x_skip_channel_mlp = self.fno_blocks.channel_mlp_skips[i](x)
            x_skip_channel_mlp = self.fno_blocks.convs[i].transform(x_skip_channel_mlp)

            # 4e. Spectral Convolution
            x_fno = self.fno_blocks.convs[i](x)

            # 4f. Add FNO Skip
            x = x_fno + x_skip_fno

            # 4g. Non-linearity (if not last layer)
            if i < (self.n_layers - 1):
                x = self.fno_blocks.non_linearity(x)

            # 4h. Second FiLM (Norm 1)
            x = self.film_layers[self.n_norms_per_block * i + 1](x, latent_vec)

            # 4i. Channel MLP
            x = self.fno_blocks.channel_mlp[i](x) + x_skip_channel_mlp

        # 5. Projection: (Batch, Width, Time) -> (Batch, 1, Time)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)

        # 6. Unpad
        x = self._unpad(x, pad_len)

        return x


class MosfetVCFiLMFNO(nn.Module):
    """
    Voltage-Conditioned FiLM MOSFET Operator.

    Extension of MosfetFiLMFNO where the FiLM layers receive both the static
    physics embedding AND the instantaneous input voltages at each timestep.
    This produces time-varying modulation that enables regime-dependent spectral
    filtering: the model can learn different behavior for subthreshold vs.
    saturation operating points of the same physical device.

    The key architectural difference from MosfetFiLMFNO: BatchedFiLM generates
    a single (scale, shift) per sample that is broadcast over time. VCFiLM
    generates a per-timestep (scale, shift) conditioned on (embedding, Vg(t),
    Vd(t), Vs(t), Vb(t)), allowing the spectral convolutions to be modulated
    differently in subthreshold vs. saturation regimes within the same sample.
    """

    def __init__(
        self,
        input_param_dim: int,
        embedding_dim: int = 16,
        modes: int = 256,
        width: int = 64,
        embedding_hidden_dim: int = 128,
        n_layers: int = 4,
    ) -> None:
        """
        Initialize the VCFiLM FNO architecture.

        :param input_param_dim: Dimension of the raw BSIM parameter vector.
        :param embedding_dim: Size of the device parameter embedding.
        :param modes: Number of Fourier modes to keep (frequency resolution).
        :param width: Channel width of the FNO layers.
        :param embedding_hidden_dim: Hidden dimension for the embedding MLP.
        :param n_layers: Number of FNO blocks.
        """
        super().__init__()
        self.embedding = DeviceEmbedding(
            input_dim=input_param_dim, embedding_dim=embedding_dim, hidden_dim=embedding_hidden_dim
        )
        self.fno_in_channels = 4
        self.n_voltage_channels = 4
        self.width = width
        self.n_layers = n_layers
        self.lifting = nn.Linear(self.fno_in_channels, self.width)
        self.fno_blocks = FNOBlocks(
            in_channels=self.width,
            out_channels=self.width,
            n_modes=(modes,),
            n_layers=self.n_layers,
            use_channel_mlp=True,
            preactivation=True,
            fno_skip="linear",
            non_linearity=nn.functional.silu,
            norm=None,
            conv_module=SpectralConv,
        )
        self.n_norms_per_block = 2
        self.vcfilm_layers = nn.ModuleList(
            [
                VoltageConditionedFiLM(
                    embed_dim=embedding_dim,
                    n_voltage_channels=self.n_voltage_channels,
                    in_channels=self.width,
                )
                for _ in range(self.n_layers * self.n_norms_per_block)
            ]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.width, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.domain_padding = 0.1

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad the input tensor along the time dimension."""
        pad_len = int(round(x.shape[-1] * self.domain_padding))
        x = nn.functional.pad(x, (0, pad_len), mode="replicate")
        return x, pad_len

    def _unpad(self, x: torch.Tensor, pad_len: int) -> torch.Tensor:
        """Remove padding from the output tensor."""
        return x[..., :-pad_len]

    def forward(self, v_terminals: torch.Tensor, physical_params: torch.Tensor) -> torch.Tensor:
        """
        Predict transient drain current using voltage-conditioned FiLM.

        :param v_terminals: Voltage waveforms (Batch, 4, Time).
        :param physical_params: Raw physics vector (Batch, Param_Dim).
        :return: Drain current I_d(t) (Batch, 1, Time).
        """
        latent_vec = self.embedding(physical_params)
        x, pad_len = self._pad(v_terminals)
        v_padded = x
        x = x.permute(0, 2, 1)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)
        for i in range(self.n_layers):
            x = self.fno_blocks.non_linearity(x)
            x = self.vcfilm_layers[self.n_norms_per_block * i](x, latent_vec, v_padded)
            x_skip_fno = self.fno_blocks.fno_skips[i](x)
            x_skip_fno = self.fno_blocks.convs[i].transform(x_skip_fno)
            x_skip_channel_mlp = self.fno_blocks.channel_mlp_skips[i](x)
            x_skip_channel_mlp = self.fno_blocks.convs[i].transform(x_skip_channel_mlp)
            x_fno = self.fno_blocks.convs[i](x)
            x = x_fno + x_skip_fno
            if i < (self.n_layers - 1):
                x = self.fno_blocks.non_linearity(x)
            x = self.vcfilm_layers[self.n_norms_per_block * i + 1](x, latent_vec, v_padded)
            x = self.fno_blocks.channel_mlp[i](x) + x_skip_channel_mlp
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self._unpad(x, pad_len)
        return x
