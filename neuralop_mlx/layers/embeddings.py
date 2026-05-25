import math

import mlx.core as mx
import mlx.nn as nn


class GridEmbeddingND(nn.Module):
    """Append ND coordinate grids to a tensor along the channel axis.

    Matches the PyTorch GridEmbeddingND in neuralop/layers/embeddings.py.
    Input shape: (batch, in_channels, d1, d2, ..., dN)
    Output shape: (batch, in_channels + dim, d1, d2, ..., dN)
    """

    def __init__(self, in_channels: int, dim: int = 2, grid_boundaries=None):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        if grid_boundaries is None:
            grid_boundaries = [[0, 1]] * dim
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self):
        return self.in_channels + self.dim

    def _build_grid(self, spatial_dims, device):
        axes = []
        for res, (start, stop) in zip(spatial_dims, self.grid_boundaries):
            # Matches torch.linspace(start, stop, res+1)[:-1]
            pts = mx.linspace(start, stop, res + 1)[:-1]
            axes.append(pts)
        grids = mx.meshgrid(*axes, indexing="ij")
        # Each grid: (d1, d2, ..., dN) → (1, 1, d1, ..., dN)
        result = []
        for g in grids:
            for _ in range(2):
                g = g[None]
            result.append(g)
        return result

    def __call__(self, data):
        spatial_dims = data.shape[2:]
        if self._grid is None or self._res != spatial_dims:
            self._grid = self._build_grid(spatial_dims, device=None)
            self._res = spatial_dims
        batch = data.shape[0]
        grids = [mx.broadcast_to(g, (batch, 1) + spatial_dims) for g in self._grid]
        return mx.concatenate([data] + grids, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# τ / η sinusoidal coordinate embeddings for Milne-coordinate FNO surrogates
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_tau_embedding(tau: mx.array, tau_max: float) -> mx.array:
    """2-channel sinusoidal embedding of the proper-time axis.

    Parameters
    ----------
    tau     : (T,)  physical proper-time values [fm/c]
    tau_max : float reference scale [fm/c]

    Returns
    -------
    (T, 2)  columns are [sin(π·τ/τ_max), cos(π·τ/τ_max)]
    """
    t = tau / tau_max
    return mx.stack([mx.sin(math.pi * t), mx.cos(math.pi * t)], axis=-1)


def sinusoidal_eta_embedding(eta: mx.array, eta_max: float) -> mx.array:
    """2-channel sinusoidal embedding of the spacetime-rapidity axis.

    sin is odd (antisymmetric in η) and cos is even (symmetric), so together
    they distinguish forward from backward rapidity.

    Parameters
    ----------
    eta     : (Z,)  spacetime-rapidity values η_s
    eta_max : float |η| range used during training

    Returns
    -------
    (Z, 2)  columns are [sin(π·η/η_max), cos(π·η/η_max)]
    """
    e = eta / eta_max
    return mx.stack([mx.sin(math.pi * e), mx.cos(math.pi * e)], axis=-1)


class TauBias(nn.Module):
    """Per-FNO-block learned τ-embedding injection.

    Projects (T, 2) → (1, C, 1, 1, 1, T) and adds it to (B, C, X, Y, Z, T).

    Parameters
    ----------
    channels : int  hidden width of the FNO block
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(2, channels, bias=False)

    def __call__(self, x: mx.array, tau_emb: mx.array) -> mx.array:
        """
        x       : (B, C, X, Y, Z, T)
        tau_emb : (T, 2)
        """
        T = tau_emb.shape[0]
        C = x.shape[1]
        bias = self.proj(tau_emb)             # (T, C)
        bias = bias.T                         # (C, T)
        bias = bias.reshape(1, C, 1, 1, 1, T)
        return x + bias


class EtaBias(nn.Module):
    """Per-FNO-block learned η-embedding injection.

    Projects (Z, 2) → (1, C, 1, 1, Z, 1) and adds it to (B, C, X, Y, Z, T).

    Parameters
    ----------
    channels : int  hidden width of the FNO block
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(2, channels, bias=False)

    def __call__(self, x: mx.array, eta_emb: mx.array) -> mx.array:
        """
        x       : (B, C, X, Y, Z, T)
        eta_emb : (Z, 2)
        """
        Z = eta_emb.shape[0]
        C = x.shape[1]
        bias = self.proj(eta_emb)             # (Z, C)
        bias = bias.T                         # (C, Z)
        bias = bias.reshape(1, C, 1, 1, Z, 1)
        return x + bias
