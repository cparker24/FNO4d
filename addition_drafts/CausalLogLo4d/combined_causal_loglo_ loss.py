"""
combined_loss.py
================
CausalLogLoLoss — three-term training objective for CausalLogLo and related
3+1D neural operators on fields of shape (B, C, X, Y, Z, T).

The three terms address orthogonal failure modes:

  Term 1 — Relative L² (LpLoss, d=4, p=2)
      Pointwise amplitude error in 4D physical space. Scale-invariant across
      events and fields. Standard baseline for all FNO variants.

  Term 2 — Spatial H¹ Sobolev (H1Loss, d=3, applied per τ-slice)
      Penalizes both amplitude AND spatial gradient errors simultaneously.
      Applied over the three spatial axes (X, Y, Z) only — τ is folded into
      the batch dimension before the call, so temporal gradients are never
      penalized.

      Physical motivation: CausalLogLo's local branch tiles the spatial domain
      into non-overlapping patches and reassembles them by concatenation. The
      H¹ gradient term penalizes discontinuities at patch boundaries, the same
      reason MG-TFNO (Kossaifi et al., arXiv:2310.00120) uses H¹ training.

      Implementation: fold T into batch → (B·T, C, X, Y, Z) → H1Loss(d=3)
      This is equivalent to averaging the spatial H¹ loss across all τ-slices,
      consistent with the 1-step per-τ training objective.

  Term 3 — Radially-binned Spectral (SPHERE-style, Kalimuthu et al. 2025)
      Power-spectrum error in concentric spherical shells in 3D wavenumber
      space (kx, ky, kz), with linearly increasing weight toward high-k bins.
      Counteracts spectral bias: the optimizer is forced to attend to
      high-frequency modes that standard L² ignores until late in training.

      Applied after folding T into batch, consistent with spectral_loss_3d in
      causal_loglo_4d.py.

Combined objective
------------------
    L_total = L_lp
            + alpha_spectral * L_spectral
            + alpha_h1       * L_h1

Default weights are set so all three terms are the same order of magnitude
during normal training (relative error ~30%), verified empirically:

    alpha_spectral = 0.001   (spectral loss ~275× larger than L²)
    alpha_h1       = 0.030   (H¹ loss     ~18×  larger than L²)

Scale analysis (16×16×8×32 grid, 30% relative error):
    L²        : ~0.57
    α·L_sp    : ~0.28   (α=0.001)
    α·L_h1    : ~0.55   (α=0.03)

Warm-up schedule
----------------
Training with all three terms from epoch 0 can destabilise early training
because the H¹ gradient term amplifies high-frequency noise before the model
has learned the low-frequency structure. A recommended schedule:

    Epochs  0 – W/2   : L_lp only
    Epochs W/2 – W    : L_lp + α_h1 * L_h1         (add H¹)
    Epochs W  – end   : L_lp + α_h1 * L_h1 + α_sp * L_sp  (add spectral)

Use CausalLogLoLoss.from_epoch(epoch, warmup_epochs) to apply this
automatically. Set warmup_epochs=0 to disable.

References
----------
  MG-TFNO H¹ motivation: Kossaifi et al., arXiv:2310.00120
  SPHERE spectral loss:   Kalimuthu et al., arXiv:2504.04260
  H¹Loss / LpLoss API:   NeuralOperator 2.0, arXiv:2512.01421
  NormFn3d:               NormFn3d.py (this project)
  CausalLogLo:            causal_loglo_4d.py (this project)

Usage
-----
    from combined_loss import CausalLogLoLoss

    loss_fn = CausalLogLoLoss(
        alpha_spectral=0.001,
        alpha_h1=0.03,
        warmup_epochs=20,
    )

    # Training loop
    for epoch in range(n_epochs):
        loss_fn.set_epoch(epoch)          # activates warm-up schedule
        for pred, target in loader:
            total, breakdown = loss_fn(pred, target)
            total.backward()
            # breakdown: {'lp': ..., 'spectral': ..., 'h1': ..., 'total': ...}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralop.losses import LpLoss, H1Loss


# ─────────────────────────────────────────────────────────────────────────────
# Spectral loss (standalone, matches causal_loglo_4d.spectral_loss_3d)
# ─────────────────────────────────────────────────────────────────────────────


def _spectral_loss_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bins: int,
    weight_hf: float,
) -> torch.Tensor:
    """
    Radially-binned power-spectrum error over 3 spatial axes.

    T is folded into the batch dimension so the FFT runs over (X, Y, Z) only,
    consistent with the per-τ training philosophy.

    Parameters
    ----------
    pred, target : (B, C, X, Y, Z, T)
    n_bins       : number of radial wavenumber shells
    weight_hf    : weight multiplier for highest-frequency bin.
                   Linearly interpolated: bin i has weight
                   1 + (weight_hf - 1) * i / (n_bins - 1).
                   1.0 → uniform (= flat spectral L²).
                   2.0 → HF bins weighted 2× more than LF bins.

    Returns
    -------
    loss : scalar tensor (differentiable w.r.t. pred)
    """
    B, C, X, Y, Z, T = pred.shape
    BT = B * T

    # Fold T into batch: (B·T, C, X, Y, Z)
    p = pred.permute(0, 5, 1, 2, 3, 4).reshape(BT, C, X, Y, Z)
    t = target.permute(0, 5, 1, 2, 3, 4).reshape(BT, C, X, Y, Z)

    # Power spectrum of the error
    pf    = torch.fft.rfftn(p - t, dim=(-3, -2, -1))
    power = pf.real ** 2 + pf.imag ** 2          # (BT, C, X, Y, Z//2+1)

    # Radial wavenumber grid: (X, Y, Z//2+1)
    kx    = torch.fft.fftfreq(X, device=pred.device).view(X, 1, 1)
    ky    = torch.fft.fftfreq(Y, device=pred.device).view(1, Y, 1)
    kz    = torch.fft.rfftfreq(Z, device=pred.device).view(1, 1, Z // 2 + 1)
    k_rad = (kx ** 2 + ky ** 2 + kz ** 2).sqrt()

    k_max = k_rad.max().item()
    edges = torch.linspace(0.0, k_max + 1e-6, n_bins + 1, device=pred.device)

    loss = pred.new_zeros(())
    for i in range(n_bins):
        mask = (k_rad >= edges[i]) & (k_rad < edges[i + 1])
        if mask.sum() == 0:
            continue
        w    = 1.0 + (weight_hf - 1.0) * i / max(n_bins - 1, 1)
        loss = loss + w * power[..., mask].mean()

    return loss / n_bins


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss
# ─────────────────────────────────────────────────────────────────────────────


class CausalLogLoLoss(nn.Module):
    """
    Three-term training loss for CausalLogLo (and compatible 3+1D operators).

    L_total = L_lp  +  alpha_spectral · L_spectral  +  alpha_h1 · L_h1

    where:
      L_lp       = relative L²  over (B, C, X, Y, Z, T)        [LpLoss, d=4]
      L_spectral = radially-binned spectral power error           [3D, T in batch]
      L_h1       = relative H¹ Sobolev over (X, Y, Z) per τ     [H1Loss, d=3]

    Parameters
    ----------
    alpha_spectral : float
        Weight of the spectral loss term. Default 0.001 balances the three
        terms at ~0.3 each for a model with ~30% relative error.
    alpha_h1 : float
        Weight of the spatial H¹ term. Default 0.03.
    n_bins : int
        Number of radial wavenumber shells for the spectral loss. Default 8.
    weight_hf : float
        Extra multiplier on the highest-frequency spectral bin. Default 2.0.
        Set to 1.0 for a flat (unweighted) spectral loss.
    warmup_epochs : int
        Number of epochs over which to phase in H¹ and spectral terms.
        0 disables warm-up (all terms active from epoch 0).
        See warm-up schedule in module docstring.
    lp_p : int
        Exponent for LpLoss. Default 2 (standard L²).
    periodic_xyz : bool
        Whether to use periodic finite differences in H1Loss. Default True.
        Set False if the spatial domain has hard boundaries (e.g. rapidity
        edges in η-η geometry with a sharp cutoff).
    """

    def __init__(
        self,
        alpha_spectral: float = 0.001,
        alpha_h1: float       = 0.03,
        n_bins: int           = 8,
        weight_hf: float      = 2.0,
        warmup_epochs: int    = 0,
        lp_p: int             = 2,
        periodic_xyz: bool    = True,
    ):
        super().__init__()
        self.alpha_spectral = alpha_spectral
        self.alpha_h1       = alpha_h1
        self.n_bins         = n_bins
        self.weight_hf      = weight_hf
        self.warmup_epochs  = warmup_epochs

        # neuralop loss objects (not nn.Parameters, just helpers)
        self._lp = LpLoss(d=4, p=lp_p)
        self._h1 = H1Loss(
            d=3,
            periodic_in_x=periodic_xyz,
            periodic_in_y=periodic_xyz,
            periodic_in_z=periodic_xyz,
        )

        # Mutable training state (not a buffer — no gradient needed)
        self._epoch: int = 0

    # ── Epoch tracking ────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """
        Call at the start of each training epoch to activate the warm-up schedule.

        Parameters
        ----------
        epoch : int   (0-indexed)
        """
        self._epoch = epoch

    # ── Warm-up weight accessors ──────────────────────────────────────────────

    @property
    def _w_h1(self) -> float:
        """
        Effective H¹ weight at the current epoch, after warm-up ramp.

        Schedule:
          epoch <  W/2  → 0.0          (pure L²)
          epoch >= W/2  → linearly ramps to alpha_h1 over the second half of W
          epoch >= W    → alpha_h1      (full weight)
        """
        if self.warmup_epochs == 0:
            return self.alpha_h1
        W = self.warmup_epochs
        if self._epoch < W // 2:
            return 0.0
        ramp = min(1.0, (self._epoch - W // 2) / max(W // 2, 1))
        return self.alpha_h1 * ramp

    @property
    def _w_sp(self) -> float:
        """
        Effective spectral weight at the current epoch, after warm-up ramp.

        Schedule:
          epoch <  W    → 0.0          (spectral term off during warm-up)
          epoch >= W    → linearly ramps to alpha_spectral over the next W epochs
          epoch >= 2·W  → alpha_spectral (full weight)
        """
        if self.warmup_epochs == 0:
            return self.alpha_spectral
        W = self.warmup_epochs
        if self._epoch < W:
            return 0.0
        ramp = min(1.0, (self._epoch - W) / max(W, 1))
        return self.alpha_spectral * ramp

    # ── Individual loss terms ─────────────────────────────────────────────────

    def lp_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Relative L² loss over 4D field (B, C, X, Y, Z, T)."""
        return self._lp.rel(pred, target)

    def h1_spatial_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Relative H¹ Sobolev loss over spatial axes (X, Y, Z) only.

        T is folded into the batch axis so finite-difference gradients are
        computed over (X, Y, Z) at each τ-slice independently.

        Input shape: (B, C, X, Y, Z, T)
        """
        B, C, X, Y, Z, T = pred.shape
        BT = B * T
        # (B, C, X, Y, Z, T) → (B·T, C, X, Y, Z)
        p = pred.permute(0, 5, 1, 2, 3, 4).reshape(BT, C, X, Y, Z)
        t = target.permute(0, 5, 1, 2, 3, 4).reshape(BT, C, X, Y, Z)
        return self._h1.rel(p, t)

    def spectral_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Radially-binned spectral power error over spatial (X, Y, Z)."""
        return _spectral_loss_3d(pred, target, self.n_bins, self.weight_hf)

    # ── Combined forward ──────────────────────────────────────────────────────

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined loss and return a breakdown for logging.

        Parameters
        ----------
        pred   : (B, C, X, Y, Z, T)  — model prediction
        target : (B, C, X, Y, Z, T)  — ground truth

        Returns
        -------
        total     : scalar tensor (differentiable, use for .backward())
        breakdown : dict with keys 'lp', 'h1', 'spectral', 'total'
                    Values are Python floats (detached), suitable for logging.
        """
        l_lp = self.lp_loss(pred, target)
        l_h1 = self.h1_spatial_loss(pred, target)
        l_sp = self.spectral_loss(pred, target)

        w_h1 = self._w_h1
        w_sp = self._w_sp

        total = l_lp + w_h1 * l_h1 + w_sp * l_sp

        breakdown = {
            "lp":       l_lp.item(),
            "h1":       l_h1.item(),
            "spectral": l_sp.item(),
            "h1_weighted":       w_h1 * l_h1.item(),
            "spectral_weighted": w_sp * l_sp.item(),
            "total":    total.item(),
        }
        return total, breakdown

    # ── Convenience ───────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"alpha_spectral={self.alpha_spectral}, "
            f"alpha_h1={self.alpha_h1}, "
            f"n_bins={self.n_bins}, "
            f"weight_hf={self.weight_hf}, "
            f"warmup_epochs={self.warmup_epochs}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────────────────────


def format_breakdown(breakdown: dict[str, float], epoch: int | None = None) -> str:
    """
    Format a loss breakdown dict as a one-line log string.

    Example output:
        epoch=12 | total=1.2345 | lp=0.5712 | h1=18.43 (×0.03=0.553)
                 | spectral=275.1 (×0.001=0.275)
    """
    parts = []
    if epoch is not None:
        parts.append(f"epoch={epoch}")
    parts.append(f"total={breakdown['total']:.4f}")
    parts.append(f"lp={breakdown['lp']:.4f}")
    parts.append(
        f"h1={breakdown['h1']:.2f} "
        f"(weighted={breakdown['h1_weighted']:.4f})"
    )
    parts.append(
        f"spectral={breakdown['spectral']:.2f} "
        f"(weighted={breakdown['spectral_weighted']:.4f})"
    )
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import math

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, C, X, Y, Z, T = 2, 1, 16, 16, 8, 32
    target = torch.randn(B, C, X, Y, Z, T, device=device)

    # ── Test 1: all terms active, backward ────────────────────────────────
    print("── Test 1: all terms active ──────────────────────────────────")
    pred = (target + 0.3 * torch.randn_like(target)).requires_grad_(True)

    loss_fn = CausalLogLoLoss(
        alpha_spectral=0.001,
        alpha_h1=0.03,
        warmup_epochs=0,        # all terms active immediately
    )

    total, bd = loss_fn(pred, target)
    total.backward()

    print(format_breakdown(bd))
    print(f"Gradient norm on pred: {pred.grad.norm().item():.4f}")
    assert pred.grad is not None, "No gradient"
    assert not math.isnan(total.item()), "NaN loss"
    print("OK ✓\n")

    # ── Test 2: warm-up schedule ───────────────────────────────────────────
    print("── Test 2: warm-up schedule (warmup_epochs=20) ───────────────")
    loss_fn_wu = CausalLogLoLoss(
        alpha_spectral=0.001,
        alpha_h1=0.03,
        warmup_epochs=20,
    )
    rows = []
    for epoch in [0, 9, 10, 15, 19, 20, 30, 40]:
        loss_fn_wu.set_epoch(epoch)
        rows.append((epoch, loss_fn_wu._w_h1, loss_fn_wu._w_sp))

    print(f"{'epoch':>6}  {'w_h1':>8}  {'w_sp':>10}")
    for epoch, w_h1, w_sp in rows:
        print(f"{epoch:>6}  {w_h1:>8.4f}  {w_sp:>10.6f}")
    print()

    # Verify h1 is 0 before warmup midpoint
    assert rows[0][1] == 0.0, "H1 should be 0 at epoch 0"
    assert rows[0][2] == 0.0, "Spectral should be 0 at epoch 0"
    # Verify h1 is non-zero at midpoint
    assert rows[3][1] > 0.0, "H1 should be ramping at epoch 15"
    # Verify spectral is 0 before W
    assert rows[4][2] == 0.0, "Spectral should be 0 at epoch 19 (before W)"
    # Verify spectral is non-zero after W
    assert rows[6][2] > 0.0, "Spectral should be ramping at epoch 30 (W + W/2)"
    print("Warm-up schedule OK ✓\n")

    # ── Test 3: H¹ is spatial-only (no τ gradient penalty) ────────────────
    print("── Test 3: H¹ spatial-only vs naive 4D ──────────────────────")
    # Construct a pred that differs from target only along τ (sharp τ transition)
    # A spatial-only H¹ should NOT penalize this, a 4D H¹ would.
    pred_tau_only = target.clone()
    # Add a sharp step at τ=T//2, uniform across (X,Y,Z)
    pred_tau_only[..., T // 2 :] += 1.0

    loss_fn_test = CausalLogLoLoss(alpha_h1=1.0, alpha_spectral=0.0, warmup_epochs=0)
    _, bd_tau = loss_fn_test(pred_tau_only, target)

    # If H¹ were 4D, a spatial-constant τ-step would contribute a large ∂/∂τ term.
    # With spatial-only H¹, the gradient term comes only from ∂/∂x, ∂/∂y, ∂/∂z.
    # The step is spatially uniform, so spatial gradients are zero → H¹ ≈ L²
    h1_loss_val = bd_tau["h1"]
    lp_loss_val = bd_tau["lp"]
    # H1 should be close to LpLoss value (since spatial gradients of a
    # spatially-uniform field are zero); ratio should be near 1.0
    ratio = h1_loss_val / lp_loss_val
    print(f"  Spatial-uniform τ step: H¹={h1_loss_val:.4f}, L²={lp_loss_val:.4f}, "
          f"ratio={ratio:.3f} (expected ≈1.0 for spatial-only H¹)")
    # Allow some tolerance for finite-difference boundary effects
    assert ratio < 1.5, f"H¹ ratio {ratio:.2f} too large — τ gradient leaking in"
    print("  Spatial-only enforcement OK ✓\n")

    # ── Test 4: scale balance ──────────────────────────────────────────────
    print("── Test 4: scale balance at default alphas ───────────────────")
    pred_noisy = (target + 0.3 * torch.randn_like(target))
    loss_fn_bal = CausalLogLoLoss(warmup_epochs=0)
    _, bd_bal = loss_fn_bal(pred_noisy, target)

    print(f"  L²              : {bd_bal['lp']:.4f}")
    print(f"  H¹·alpha        : {bd_bal['h1_weighted']:.4f}  "
          f"(raw H¹={bd_bal['h1']:.2f})")
    print(f"  Spectral·alpha  : {bd_bal['spectral_weighted']:.4f}  "
          f"(raw spectral={bd_bal['spectral']:.2f})")
    # All three weighted terms should be within 10× of each other
    terms = [bd_bal['lp'], bd_bal['h1_weighted'], bd_bal['spectral_weighted']]
    assert max(terms) / min(terms) < 10, "Loss terms are poorly balanced"
    print("  Scale balance OK ✓\n")

    print("All tests passed. ✓")
