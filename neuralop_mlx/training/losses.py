"""LpLoss and H1Loss — MLX ports of neuralop/losses/data_losses.py.

Supports spatial dimensionality d=1..4 (same constraint as PyTorch H1Loss).
"""

import math
import mlx.core as mx


class LpLoss:
    """Relative Lp norm loss.

    Matches neuralop LpLoss with reduction="sum".

    Parameters
    ----------
    d : int
        Number of spatial dimensions to include in the norm (last d dims).
    p : int or float, default 2
    reduction : str, "sum" or "mean"
    eps : float, small constant to avoid division by zero
    """

    def __init__(self, d: int = 1, p: float = 2, reduction: str = "sum", eps: float = 1e-8):
        self.d = d
        self.p = p
        self.reduction = reduction
        self.eps = eps

    def _reduce(self, x):
        if self.reduction == "sum":
            return mx.sum(x)
        return mx.mean(x)

    def abs(self, x, y):
        """Absolute Lp norm."""
        diff = (x - y).reshape(x.shape[0], x.shape[1], -1)
        norm = mx.sum(mx.abs(diff) ** self.p, axis=-1) ** (1.0 / self.p)
        return self._reduce(norm)

    def rel(self, x, y):
        """Relative Lp norm: ||x-y||_p / (||y||_p + eps)."""
        diff = (x - y).reshape(x.shape[0], x.shape[1], -1)
        y_flat = y.reshape(y.shape[0], y.shape[1], -1)
        diff_norm = mx.sum(mx.abs(diff) ** self.p, axis=-1) ** (1.0 / self.p)
        y_norm = mx.sum(mx.abs(y_flat) ** self.p, axis=-1) ** (1.0 / self.p)
        return self._reduce(diff_norm / (y_norm + self.eps))

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)


class H1Loss:
    """Relative H1 Sobolev norm loss (function + all first-order derivatives).

    Matches neuralop H1Loss. Supports d=1,2,3,4.

    Derivatives are computed via finite differences (periodic by default).
    """

    def __init__(
        self,
        d: int = 1,
        reduction: str = "sum",
        eps: float = 1e-8,
        periodic_in_x: bool = True,
        periodic_in_y: bool = True,
        periodic_in_z: bool = True,
        periodic_in_t: bool = True,
    ):
        assert 1 <= d <= 4, "H1Loss supports d=1..4"
        self.d = d
        self.reduction = reduction
        self.eps = eps
        self.periodic = [periodic_in_x, periodic_in_y, periodic_in_z, periodic_in_t][:d]

    def _reduce(self, x):
        if self.reduction == "sum":
            return mx.sum(x)
        return mx.mean(x)

    def _finite_diff(self, x, axis: int, periodic: bool):
        """First-order finite difference along axis."""
        if periodic:
            return mx.roll(x, -1, axis=axis) - x
        # Non-periodic: central diff interior, forward/backward at boundaries
        slc_fwd = [slice(None)] * x.ndim
        slc_bwd = [slice(None)] * x.ndim
        slc_fwd[axis] = slice(1, None)
        slc_bwd[axis] = slice(None, -1)
        interior = x[tuple(slc_fwd)] - x[tuple(slc_bwd)]
        # Pad to restore size (replicate boundary)
        slc_first = [slice(None)] * x.ndim
        slc_last = [slice(None)] * x.ndim
        slc_first[axis] = slice(None, 1)
        slc_last[axis] = slice(-1, None)
        interior = mx.concatenate(
            [interior[tuple(slc_first)], interior, interior[tuple(slc_last)]], axis=axis
        )
        return interior

    def _norm_sq_flat(self, z):
        """||z||^2 flattened over last d spatial dims."""
        b, c = z.shape[0], z.shape[1]
        flat = z.reshape(b, c, -1)
        return mx.sum(flat ** 2, axis=-1)

    def abs(self, x, y):
        """Absolute H1 norm."""
        # Spatial axes: last d dims, e.g. (-d, ..., -1) = axes starting at -(d+2)+2
        spatial_axes = list(range(-self.d, 0))
        # Adjust for (batch, channel, spatial...) layout: actual axes are 2..2+d-1
        axes = [2 + self.d + a for a in spatial_axes]  # = [2, 3, ..., 2+d-1]

        norm_sq = self._norm_sq_flat(x - y)
        for i, ax in enumerate(axes):
            dx = self._finite_diff(x, ax, self.periodic[i])
            dy = self._finite_diff(y, ax, self.periodic[i])
            norm_sq = norm_sq + self._norm_sq_flat(dx - dy)

        return self._reduce(mx.sqrt(norm_sq + 1e-16))

    def rel(self, x, y):
        """Relative H1 norm: ||x-y||_H1 / (||y||_H1 + eps)."""
        axes = list(range(2, 2 + self.d))

        diff_sq = self._norm_sq_flat(x - y)
        y_sq = self._norm_sq_flat(y)
        for i, ax in enumerate(axes):
            dx = self._finite_diff(x, ax, self.periodic[i])
            dy = self._finite_diff(y, ax, self.periodic[i])
            diff_sq = diff_sq + self._norm_sq_flat(dx - dy)
            y_sq = y_sq + self._norm_sq_flat(dy)

        diff_norm = mx.sqrt(diff_sq + 1e-16)
        y_norm = mx.sqrt(y_sq + 1e-16)
        return self._reduce(diff_norm / (y_norm + self.eps))

    def __call__(self, x, y, **kwargs):
        return self.rel(x, y)
