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
