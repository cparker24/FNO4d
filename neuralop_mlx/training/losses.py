"""LpLoss, H1Loss, CombinedTemporalLoss — MLX loss functions for FNO training.

LpLoss / H1Loss are ports of neuralop/losses/data_losses.py (d=1..4).
CombinedTemporalLoss is an MLX port of loc_libs/LossFn.py for 3+1D tensors.
"""

import math
from typing import Dict, Literal, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Presets for CombinedTemporalLoss
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, Tuple[float, float]] = {
    'l2':         (1.0, 0.0),
    'h1':         (1.0, 1.0),
    'grad_focus': (0.5, 2.0),
    'smooth':     (1.0, 0.1),
    'boundary':   (0.0, 1.0),
    'custom':     (None, None),
}

Mode = Literal['relative', 'absolute']


class CombinedTemporalLoss:
    """Combined temporally-weighted L² + spatial-gradient loss for 3+1D FNO.

    MLX port of loc_libs/LossFn.CombinedTemporalLoss. Tensor layout: (B, C, X, Y, Z, T).

    Loss
    ----
        L = α · L_value  +  β · L_grad

    where each component is computed per τ-slice and weighted-summed over T:

        Relative:   L_value(t) = Σ |û_t - u_t|²  /  Σ |u_t|²
        Absolute:   L_value(t) = mean |û_t - u_t|²

        Relative:   L_grad(t)  = Σ |∇(û_t-u_t)|²  /  Σ |∇u_t|²
        Absolute:   L_grad(t)  = mean |∇(û_t-u_t)|²

    Parameters
    ----------
    alpha, beta : float
        Weights on the L² and gradient components respectively.
    preset : str
        Named (α, β) pair — 'l2', 'h1', 'grad_focus', 'smooth', 'boundary', 'custom'.
    mode : {'relative', 'absolute'}
        Applied to both components; overridden per-component by l2_mode/grad_mode.
    l2_mode, grad_mode : {'relative', 'absolute'} or None
        Per-component mode overrides.
    tau_weights : mx.array of shape (T,), optional
        L1-normalised temporal quadrature weights. Uniform 1/T by default.
    eps : float
        Denominator stabiliser for relative mode.
    reduction : {'mean', 'sum'}
        Aggregation over the batch dimension.
    """

    def __init__(
        self,
        alpha:       float                  = 1.0,
        beta:        float                  = 1.0,
        preset:      str                    = 'h1',
        mode:        Mode                   = 'relative',
        l2_mode:     Optional[Mode]         = None,
        grad_mode:   Optional[Mode]         = None,
        tau_weights: Optional[mx.array]     = None,
        eps:         float                  = 1e-8,
        reduction:   str                    = 'mean',
    ):
        if preset not in _PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(_PRESETS)}.")
        if preset != 'custom':
            alpha, beta = _PRESETS[preset]

        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.l2_mode   = l2_mode   if l2_mode   is not None else mode
        self.grad_mode = grad_mode if grad_mode is not None else mode

        for name, m in (('l2_mode', self.l2_mode), ('grad_mode', self.grad_mode)):
            if m not in ('relative', 'absolute'):
                raise ValueError(f"{name} must be 'relative' or 'absolute', got '{m}'.")

        self.tau_weights = tau_weights
        self.eps         = float(eps)
        self.reduction   = reduction
        self.preset      = preset

    # ------------------------------------------------------------------
    # Temporal weights
    # ------------------------------------------------------------------

    def _weights(self, T: int) -> mx.array:
        if self.tau_weights is not None:
            return self.tau_weights
        return mx.ones((T,)) / T

    # ------------------------------------------------------------------
    # Spatial norms — (B, C, X, Y, Z) → (B, C)
    # ------------------------------------------------------------------

    @staticmethod
    def _sq_sum(f: mx.array) -> mx.array:
        return mx.sum(f ** 2, axis=(-3, -2, -1))

    @staticmethod
    def _sq_mean(f: mx.array) -> mx.array:
        return mx.mean(f ** 2, axis=(-3, -2, -1))

    @staticmethod
    def _grad_sq_sum(f: mx.array) -> mx.array:
        gx = mx.roll(f, -1, axis=-3) - mx.roll(f, 1, axis=-3)
        gy = mx.roll(f, -1, axis=-2) - mx.roll(f, 1, axis=-2)
        gz = mx.roll(f, -1, axis=-1) - mx.roll(f, 1, axis=-1)
        return mx.sum(gx ** 2 + gy ** 2 + gz ** 2, axis=(-3, -2, -1))

    @staticmethod
    def _grad_sq_mean(f: mx.array) -> mx.array:
        gx = mx.roll(f, -1, axis=-3) - mx.roll(f, 1, axis=-3)
        gy = mx.roll(f, -1, axis=-2) - mx.roll(f, 1, axis=-2)
        gz = mx.roll(f, -1, axis=-1) - mx.roll(f, 1, axis=-1)
        return mx.mean(gx ** 2 + gy ** 2 + gz ** 2, axis=(-3, -2, -1))

    # ------------------------------------------------------------------
    # Per-slice component losses — (B, C, X, Y, Z) → (B,)
    # ------------------------------------------------------------------

    def _l2_slice(self, diff: mx.array, target: mx.array) -> mx.array:
        if self.l2_mode == 'relative':
            per_bc = self._sq_sum(diff) / (self._sq_sum(target) + self.eps)
        else:
            per_bc = self._sq_mean(diff)
        return mx.mean(per_bc, axis=-1)

    def _grad_slice(self, diff: mx.array, target: mx.array) -> mx.array:
        if self.grad_mode == 'relative':
            per_bc = self._grad_sq_sum(diff) / (self._grad_sq_sum(target) + self.eps)
        else:
            per_bc = self._grad_sq_mean(diff)
        return mx.mean(per_bc, axis=-1)

    # ------------------------------------------------------------------
    # Batch reduction
    # ------------------------------------------------------------------

    def _reduce(self, x: mx.array) -> mx.array:
        return mx.mean(x) if self.reduction == 'mean' else mx.sum(x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_dict(self, pred: mx.array, target: mx.array) -> Dict[str, mx.array]:
        """Full forward pass returning all loss components.

        Parameters
        ----------
        pred, target : mx.array, shape (B, C, X, Y, Z, T)

        Returns
        -------
        dict with keys 'total', 'l2', 'grad', 'h1'
        """
        assert pred.shape == target.shape, \
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        assert pred.ndim == 6, \
            f"Expected (B, C, X, Y, Z, T), got ndim={pred.ndim}"

        T    = pred.shape[-1]
        w    = self._weights(T)
        diff = pred - target

        acc_l2   = mx.array(0.0)
        acc_grad = mx.array(0.0)

        for t in range(T):
            d_t = diff[..., t]
            y_t = target[..., t]
            if self.alpha != 0.0:
                acc_l2   = acc_l2   + w[t] * self._reduce(self._l2_slice(d_t, y_t))
            if self.beta != 0.0:
                acc_grad = acc_grad + w[t] * self._reduce(self._grad_slice(d_t, y_t))

        return {
            'total': self.alpha * acc_l2 + self.beta * acc_grad,
            'l2':    acc_l2,
            'grad':  acc_grad,
            'h1':    acc_l2 + acc_grad,
        }

    def __call__(self, pred: mx.array, target: mx.array) -> mx.array:
        return self.forward_dict(pred, target)['total']

    def as_eval_dict(self) -> Dict[str, object]:
        """Per-component callables for logging 'l2', 'grad', 'h1' per epoch."""
        return {
            k: (lambda key: lambda p, y: self.forward_dict(p, y)[key])(k)
            for k in ('l2', 'grad', 'h1')
        }

    def __repr__(self) -> str:
        return (
            f"CombinedTemporalLoss("
            f"preset='{self.preset}', "
            f"α={self.alpha}, β={self.beta}, "
            f"l2_mode='{self.l2_mode}', "
            f"grad_mode='{self.grad_mode}')"
        )
