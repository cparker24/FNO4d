import math
import mlx.core as mx
import mlx.nn as nn


class ChannelMLP(nn.Module):
    """Point-wise MLP applied along the channel axis.

    Equivalent to a stack of Conv1d(kernel_size=1) layers in PyTorch.
    Handles any spatial dimensionality by flattening the spatial dims first.

    Input/output shape: (batch, channels, d1, d2, ..., dN)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        hidden_channels: int = None,
        n_layers: int = 2,
        non_linearity=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity or nn.gelu

        dims = [in_channels] + [hidden_channels] * (n_layers - 1) + [out_channels]
        self.fcs = [nn.Linear(dims[i], dims[i + 1], bias=True) for i in range(n_layers)]

        if dropout > 0:
            self.dropouts = [nn.Dropout(p=dropout) for _ in range(n_layers)]
        else:
            self.dropouts = None

        self.out_channels = out_channels

    def __call__(self, x):
        shape = x.shape  # (b, c, d1, ..., dN)
        if x.ndim > 3:
            # Flatten spatial dims: (b, c, -1)
            x = x.reshape(shape[0], shape[1], -1)
            reshaped = True
        else:
            reshaped = False

        # Transpose to (b, spatial, channels) for Linear
        x = x.transpose(0, 2, 1)

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
                if self.dropouts is not None:
                    x = self.dropouts[i](x)

        # Transpose back: (b, channels, spatial)
        x = x.transpose(0, 2, 1)

        if reshaped:
            x = x.reshape(shape[0], self.out_channels, *shape[2:])

        return x
