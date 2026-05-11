import mlx.core as mx
import mlx.nn as nn


class LinearSkip(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def __call__(self, x):
        shape = x.shape
        x_flat = x.reshape(shape[0], shape[1], -1).transpose(0, 2, 1)
        out = self.linear(x_flat).transpose(0, 2, 1)
        return out.reshape(shape[0], -1, *shape[2:])


class SoftGatingSkip(nn.Module):
    """Per-channel learnable gate: out = alpha[c] * x.

    Matches PyTorch SoftGating: weight shape (channels,), broadcast over spatial dims.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert in_channels == out_channels, "SoftGatingSkip requires in_channels == out_channels"
        self.in_channels = in_channels
        self.alpha = mx.ones([in_channels])

    def __call__(self, x):
        # x: (batch, channels, *spatial) — reshape alpha to broadcast
        gate = self.alpha.reshape(1, self.in_channels, *([1] * (x.ndim - 2)))
        return gate * x


class IdentitySkip(nn.Module):
    def __call__(self, x):
        return x


def build_skip(skip_type: str, in_channels: int, out_channels: int):
    if skip_type == "linear":
        return LinearSkip(in_channels, out_channels)
    elif skip_type == "soft-gating":
        return SoftGatingSkip(in_channels, out_channels)
    elif skip_type == "identity":
        return IdentitySkip()
    elif skip_type is None:
        return None
    else:
        raise ValueError(f"Unknown skip type: {skip_type!r}")
