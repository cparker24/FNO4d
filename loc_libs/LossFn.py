"""
LossFn.py
================
Combined temporally-weighted L² + spatial-gradient loss for 3+1D FNO.

Tensor layout: (B, C, X, Y, Z, T)

Loss
----
    L = α · L_value  +  β · L_grad

where each component is computed per τ-slice and weighted-summed over T:

    Relative:   L_value(t) = Σ_{x,y,η} |û_t - u_t|²  /  Σ |u_t|²
    Absolute:   L_value(t) = mean_{x,y,η} |û_t - u_t|²          (MSE per cell)

    Relative:   L_grad(t)  = Σ |∇(û_t-u_t)|²  /  Σ |∇u_t|²
    Absolute:   L_grad(t)  = mean_{x,y,η} |∇(û_t-u_t)|²

Grid spacings cancel in the relative gradient norm, so they are not needed.
For absolute mode the mean-over-cells normalisation also makes h irrelevant.
→ No grid metadata required in either mode.

Modes
-----
    mode='relative'   scale-invariant; recommended after field normalisation
    mode='absolute'   scale-sensitive; useful for raw error magnitude
    mode='mixed'      set l2_mode / grad_mode independently for each component

Presets  (α, β)
---------------
    'l2'          (1.0, 0.0)   pure L²
    'h1'          (1.0, 1.0)   balanced  ← default
    'grad_focus'  (0.5, 2.0)   emphasise spatial structure → better v₂/v₃
    'smooth'      (1.0, 0.1)   mild gradient regularisation
    'boundary'    (0.0, 1.0)   pure gradient (MG-TFNO patch stitching)
    'custom'                    set α, β directly

Temporal weights
----------------
Uniform by default (appropriate after Bjorken normalisation).
Pass a pre-built, L1-normalised  tau_weights  tensor of shape (T,) for
non-uniform schemes (e.g. freeze-out emphasis).

Usage
-----
    from LossFn import CombinedTemporalLoss

    # Relative H¹  (default, after normalisation)
    loss_fn = CombinedTemporalLoss(preset='h1', mode='relative')

    # Absolute H¹  (raw error magnitude diagnostics)
    loss_fn = CombinedTemporalLoss(preset='h1', mode='absolute')

    # Per-component: relative L², absolute gradient
    loss_fn = CombinedTemporalLoss(
        preset    = 'custom',
        alpha     = 1.0,   beta = 0.5,
        l2_mode   = 'relative',
        grad_mode = 'absolute',
    )

    # Non-uniform tau weights (e.g. freeze-out emphasis)
    w = torch.softmax(torch.linspace(0, 3, ntau), dim=0)   # shape (T,)
    loss_fn = CombinedTemporalLoss(preset='h1', mode='relative', tau_weights=w)

    # Drop-in replacement in test_train.py
    train_loss  = loss_fn
    eval_losses = loss_fn.as_eval_dict()   # logs 'l2', 'grad', 'h1' per epoch
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Presets: (alpha, beta)
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

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CombinedTemporalLoss:
    """Combined temporally-weighted L² + spatial-gradient loss.

    Parameters
    ----------
    alpha : float
        Weight on the per-slice L² (field-value) component.
    beta : float
        Weight on the per-slice spatial-gradient component.
    preset : str
        Named (α, β) pair.  Overrides alpha/beta unless preset='custom'.
    mode : {'relative', 'absolute'}
        Loss mode applied to both components.
        Overridden per-component by l2_mode / grad_mode when set.
    l2_mode : {'relative', 'absolute'} or None
        Per-component mode for the L² term.  Falls back to `mode` if None.
    grad_mode : {'relative', 'absolute'} or None
        Per-component mode for the gradient term.  Falls back to `mode` if None.
    tau_weights : torch.Tensor of shape (T,), optional
        L1-normalised temporal quadrature weights.
        If None, uniform weights 1/T are used.
    eps : float
        Denominator stabiliser (relative mode only).
    reduction : {'mean', 'sum'}
        How to aggregate over the batch dimension.
    """

    def __init__(
        self,
        alpha:       float                    = 1.0,
        beta:        float                    = 1.0,
        preset:      str                      = 'h1',
        mode:        Mode                     = 'relative',
        l2_mode:     Optional[Mode]           = None,
        grad_mode:   Optional[Mode]           = None,
        tau_weights: Optional[torch.Tensor]   = None,
        eps:         float                    = 1e-8,
        reduction:   str                      = 'mean',
    ):
        if preset not in _PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Choose from: {list(_PRESETS)}."
            )
        if preset != 'custom':
            alpha, beta = _PRESETS[preset]

        self.alpha = float(alpha)
        self.beta  = float(beta)

        # Per-component modes fall back to global mode
        self.l2_mode   = l2_mode   if l2_mode   is not None else mode
        self.grad_mode = grad_mode if grad_mode is not None else mode

        for name, m in (('l2_mode', self.l2_mode), ('grad_mode', self.grad_mode)):
            if m not in ('relative', 'absolute'):
                raise ValueError(
                    f"{name} must be 'relative' or 'absolute', got '{m}'."
                )

        self.tau_weights = tau_weights
        self.eps         = float(eps)
        self.reduction   = reduction
        self.preset      = preset

    # ------------------------------------------------------------------
    # Temporal weights
    # ------------------------------------------------------------------

    def _weights(self, T: int, device: torch.device) -> torch.Tensor:
        if self.tau_weights is not None:
            return self.tau_weights.to(device)
        return torch.full((T,), 1.0 / T, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Spatial norms — (B, C, X, Y, Z) → (B, C)
    # ------------------------------------------------------------------

    @staticmethod
    def _sq_sum(f: torch.Tensor) -> torch.Tensor:
        """Σ_{x,y,η} f²  — for relative norms (h cancels)."""
        return (f ** 2).sum(dim=(-3, -2, -1))

    @staticmethod
    def _sq_mean(f: torch.Tensor) -> torch.Tensor:
        """mean_{x,y,η} f²  — for absolute norms (resolution-invariant)."""
        return (f ** 2).mean(dim=(-3, -2, -1))

    @staticmethod
    def _grad_sq_sum(f: torch.Tensor) -> torch.Tensor:
        """Σ |∇f|² with h=1 finite differences (h cancels in relative norm)."""
        gx = torch.roll(f, -1, -3) - torch.roll(f, 1, -3)
        gy = torch.roll(f, -1, -2) - torch.roll(f, 1, -2)
        gz = torch.roll(f, -1, -1) - torch.roll(f, 1, -1)
        return ((gx ** 2) + (gy ** 2) + (gz ** 2)).sum(dim=(-3, -2, -1))

    @staticmethod
    def _grad_sq_mean(f: torch.Tensor) -> torch.Tensor:
        """mean |∇f|² with h=1 finite differences (resolution-invariant)."""
        gx = torch.roll(f, -1, -3) - torch.roll(f, 1, -3)
        gy = torch.roll(f, -1, -2) - torch.roll(f, 1, -2)
        gz = torch.roll(f, -1, -1) - torch.roll(f, 1, -1)
        return ((gx ** 2) + (gy ** 2) + (gz ** 2)).mean(dim=(-3, -2, -1))

    # ------------------------------------------------------------------
    # Per-slice component loss — (B, C, X, Y, Z) → scalar
    # ------------------------------------------------------------------

    def _l2_slice(self, diff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L² component at one τ slice.  Returns (B,)."""
        if self.l2_mode == 'relative':
            per_bc = self._sq_sum(diff) / (self._sq_sum(target) + self.eps)
        else:
            per_bc = self._sq_mean(diff)
        return per_bc.mean(dim=-1)      # mean over channels → (B,)

    def _grad_slice(self, diff: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Gradient component at one τ slice.  Returns (B,)."""
        if self.grad_mode == 'relative':
            per_bc = self._grad_sq_sum(diff) / (self._grad_sq_sum(target) + self.eps)
        else:
            per_bc = self._grad_sq_mean(diff)
        return per_bc.mean(dim=-1)      # (B,)

    # ------------------------------------------------------------------
    # Batch reduction
    # ------------------------------------------------------------------

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        """(B,) → scalar."""
        return x.mean() if self.reduction == 'mean' else x.sum()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_dict(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass returning all loss components.

        Parameters
        ----------
        pred, target : torch.Tensor, shape (B, C, X, Y, Z, T)

        Returns
        -------
        dict
            'total'  α·L_value + β·L_grad   ← use for .backward()
            'l2'     temporally-weighted L² component
            'grad'   temporally-weighted gradient component
            'h1'     l2 + grad  (unscaled reference, mode-consistent)
        """
        assert pred.shape == target.shape, \
            f"Shape mismatch: pred {pred.shape} vs target {target.shape}"
        assert pred.ndim == 6, \
            f"Expected (B, C, X, Y, Z, T), got ndim={pred.ndim}"

        T    = pred.shape[-1]
        w    = self._weights(T, pred.device)
        diff = pred - target

        acc_l2   = pred.new_zeros(())
        acc_grad = pred.new_zeros(())

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

    def __call__(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar total loss — drop-in for neuralop Trainer."""
        return self.forward_dict(pred, target)['total']

    # ------------------------------------------------------------------
    # Eval dict for neuralop Trainer
    # ------------------------------------------------------------------

    def as_eval_dict(self) -> Dict[str, object]:
        """Per-component callables for Trainer's eval_losses argument.

        Example
        -------
            eval_losses = loss_fn.as_eval_dict()
            # Trainer logs 'l2', 'grad', 'h1' at each eval interval.
        """
        return {
            k: (lambda key: lambda p, y: self.forward_dict(p, y)[key])(k)
            for k in ('l2', 'grad', 'h1')
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CombinedTemporalLoss("
            f"preset='{self.preset}', "
            f"α={self.alpha}, β={self.beta}, "
            f"l2_mode='{self.l2_mode}', "
            f"grad_mode='{self.grad_mode}')"
        )

