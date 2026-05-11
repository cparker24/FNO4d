"""MLX FNO model — mirrors neuralop/models/fno.py (FNO class).

Supports arbitrary spatial dimensionality (1D–4D+) via n_modes tuple length.
Dense weights only (no Tucker/CP/TT factorization).
"""

import mlx.core as mx
import mlx.nn as nn

from neuralop_mlx.layers.embeddings import GridEmbeddingND
from neuralop_mlx.layers.channel_mlp import ChannelMLP
from neuralop_mlx.layers.fno_block import FNOBlocks
from neuralop_mlx.utils import count_model_params


class FNO(nn.Module):
    """Fourier Neural Operator (N-dimensional, real-data path).

    Architecture:
        GridEmbeddingND → Lifting MLP → n_layers × FNOBlock → Projection MLP

    Parameters
    ----------
    n_modes : tuple/list of int
        Fourier modes per spatial dimension. len(n_modes) determines n_dim.
    in_channels : int
    out_channels : int
    hidden_channels : int
        Width of the FNO layers.
    n_layers : int, default 4
    lifting_channel_ratio : float, default 2
        Lifting MLP hidden width = hidden_channels * lifting_channel_ratio.
    projection_channel_ratio : float, default 2
        Projection MLP hidden width = out_channels * projection_channel_ratio.
    positional_embedding : str or None
        "grid" to append coordinate grids, None to skip.
    non_linearity : callable, default nn.gelu
    fno_skip : str, default "linear"
    channel_mlp_skip : str, default "soft-gating"
    use_channel_mlp : bool, default True
    channel_mlp_expansion : float, default 0.5
    """

    def __init__(
        self,
        n_modes,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: float = 2,
        projection_channel_ratio: float = 2,
        positional_embedding: str = "grid",
        non_linearity=None,
        fno_skip: str = "linear",
        channel_mlp_skip: str = "soft-gating",
        use_channel_mlp: bool = True,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = list(n_modes)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity or nn.gelu

        # Optional positional embedding (adds n_dim channels)
        if positional_embedding == "grid":
            self.positional_embedding = GridEmbeddingND(
                in_channels=in_channels,
                dim=self.n_dim,
                grid_boundaries=[[0, 1]] * self.n_dim,
            )
            lift_in = in_channels + self.n_dim
        else:
            self.positional_embedding = None
            lift_in = in_channels

        # Lifting MLP
        lift_hidden = round(hidden_channels * lifting_channel_ratio)
        self.lifting = ChannelMLP(
            in_channels=lift_in,
            out_channels=hidden_channels,
            hidden_channels=lift_hidden,
            n_layers=2,
        )

        # FNO blocks
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            n_layers=n_layers,
            non_linearity=self.non_linearity,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_dropout=channel_mlp_dropout,
        )

        # Projection MLP
        proj_hidden = round(out_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=proj_hidden,
            n_layers=2,
        )

    def __call__(self, x, **kwargs):
        """Forward pass.

        Parameters
        ----------
        x : mx.array, shape (batch, in_channels, d0, d1, ..., d_{N-1})

        Returns
        -------
        mx.array, shape (batch, out_channels, d0, d1, ..., d_{N-1})
        """
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        for i in range(self.n_layers):
            x = self.fno_blocks(x, index=i)

        x = self.projection(x)
        return x
