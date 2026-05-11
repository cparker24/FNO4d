import mlx.core as mx
import mlx.nn as nn

from neuralop_mlx.layers.spectral_convolution import SpectralConvND
from neuralop_mlx.layers.channel_mlp import ChannelMLP
from neuralop_mlx.layers.skip_connections import build_skip


class FNOBlocks(nn.Module):
    """Sequential FNO layers sharing the same channel width.

    Each layer applies: SpectralConv + FNO skip, then ChannelMLP + MLP skip.
    Mirrors the post-activation form of neuralop/layers/fno_block.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes,
        n_layers: int = 4,
        non_linearity=None,
        fno_skip: str = "linear",
        channel_mlp_skip: str = "soft-gating",
        use_channel_mlp: bool = True,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.non_linearity = non_linearity or nn.gelu

        self.convs = [
            SpectralConvND(in_channels, out_channels, list(n_modes))
            for _ in range(n_layers)
        ]

        self.fno_skips = [
            build_skip(fno_skip, in_channels, out_channels)
            for _ in range(n_layers)
        ]

        if use_channel_mlp:
            hidden = max(1, round(out_channels * channel_mlp_expansion))
            self.channel_mlps = [
                ChannelMLP(
                    out_channels,
                    out_channels,
                    hidden_channels=hidden,
                    n_layers=2,
                    dropout=channel_mlp_dropout,
                )
                for _ in range(n_layers)
            ]
            self.mlp_skips = [
                build_skip(channel_mlp_skip, out_channels, out_channels)
                for _ in range(n_layers)
            ]
        else:
            self.channel_mlps = None
            self.mlp_skips = None

    def __call__(self, x, index: int = 0):
        # Mirrors PyTorch forward_with_postactivation:
        # - Both skip connections are computed from the block's INPUT x.
        # - gelu is applied after fno+skip and after channel_mlp, but only
        #   for non-final layers. The last layer returns without activation.
        x_in = x

        x_fno = self.convs[index](x_in)
        if self.fno_skips[index] is not None:
            x_fno = x_fno + self.fno_skips[index](x_in)

        if index < self.n_layers - 1:
            x = self.non_linearity(x_fno)
        else:
            x = x_fno

        if self.channel_mlps is not None:
            x_mlp = self.channel_mlps[index](x)
            if self.mlp_skips[index] is not None:
                x_mlp = x_mlp + self.mlp_skips[index](x_in)  # skip from original input
            if index < self.n_layers - 1:
                x = self.non_linearity(x_mlp)
            else:
                x = x_mlp

        return x
