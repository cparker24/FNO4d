"""
loglo_fno_4d.py
===============
Extension of LOGLO-FNO to 4D fields (3 spatial + 1 temporal dimension).

This file extends loglo_fno_2d.py from 2D spatial data to 4D fields of shape
(B, C, X, Y, Z, T), while keeping the paper's architecture as intact as possible.

Reference: Kalimuthu, Holzmüller, Niepert — arXiv:2504.04260 (TMLR 2025)

Design Choices for the 4D Extension
-------------------------------------

Global spectral branch — 4D SpectralConv
    The global branch uses a single SpectralConv with a 4-tuple n_modes
    (mx, my, mz, mt), learning correlations across all four axes jointly.
    This is the natural generalisation of the paper's global branch and
    matches the TFNO baseline in mg_tfno_4d_fsdp.py.

Local spectral branch — 3D spatial patches, T preserved
    The paper tiles the domain into spatial patches and applies SpectralConv
    within each patch to capture fine-grained local features. Following the
    paper's intent (local spatial structure, not temporal), patches cover
    only (X, Y, Z); the full T axis is kept on each patch. T is folded into
    the batch dimension before the 3D patch-level SpectralConv, giving the
    local branch shape (B·M·T, C, Px, Py, Pz).

    Local modes retain ALL coefficients within each patch:
      n_modes_local = (Px//2, Py//2, Pz//2+1)
    The +1 on the rfft axis (Pz) includes the Nyquist frequency (§3.3 and
    the 2D convention n_modes_local = (Px, Py//2+1)).

HFP module — AvgPool3d + trilinear upsample (paper §3.1)
    The paper extracts high-frequency features via spatial blurring:
      H = X − Interpolate(AvgPool(X))
    Extended to 4D by folding T into the batch axis so that AvgPool3d blurs
    over (X, Y, Z) at each τ-slice independently:
      T folded → AvgPool3d(k, s) → trilinear upsample to (X, Y, Z) → T unfolded
    The paper uses kernel=4, stride=4 (Appendix L.7).

HFP recurrent stream — z_prime (paper architecture)
    The HFP features are lifted to channels with shared lifting weights and
    flow as a recurrent state z_prime through all Fourier layers. At each
    layer, z_prime is updated by ChannelMLP(z_prime) and that output is added
    to the layer sum (Eq. 2). This preserves the paper's design exactly.

No causal constraint, no τ-embedding
    Unlike causal_loglo_4d.py, this model treats T as a standard spectral
    dimension. It makes no causality assumptions and does not inject proper-time
    embeddings. It is appropriate as a non-causal reference baseline or for
    problems where the full temporal context is available (e.g. spatiotemporal
    field reconstruction rather than initial-value propagation).

Last-layer GELU skipped (paper footnote 1)
    GELU non-linearity is applied on all Fourier layers except the last.

No outer residual (paper architecture)
    Each layer's output replaces z directly (no skip around the full block).
    The SoftGating skip Wc ⊙ z + bc provides a learned channel-wise skip
    connection within each branch.

Tensor layout: (B, C, X, Y, Z, T) throughout.

Consistency with loglo_fno_2d.py
----------------------------------
  loglo_fno_2d.py   → 2D spatial,  global SpectralConv2D, local SpectralConv2D
  loglo_fno_4d.py   → 4D field,    global SpectralConv4D, local SpectralConv3D

All other architectural choices (HFP, recurrence, SoftGating, ChannelMLP,
lifting, projection, loss) are direct extensions of the 2D version.

Differences from causal_loglo_4d.py
--------------------------------------
  causal_loglo_4d.py        |  loglo_fno_4d.py (this file)
  ──────────────────────────|────────────────────────────────────────────
  No spectral T mode        |  4D SpectralConv with mt temporal modes
  Causal Conv1d on T        |  None — T is a spectral dimension
  τ-embedding (sin/cos)     |  None
  HFP: spectral lowpass     |  HFP: AvgPool3d + trilinear upsample (paper)
  HFP: fresh per block      |  HFP: recurrent z_prime (paper)
  Outer residual h = h+blk  |  None — paper architecture
  Tucker factorization      |  Dense weights (paper default)

Usage
-----
    from loglo_fno_4d import LOGLOFNO4D, LOGLOLoss4D

    model = LOGLOFNO4D(
        in_channels=4,
        out_channels=4,
        hidden_channels=48,
        n_modes_global=(8, 8, 4, 8),   # (mx, my, mz, mt)
        patch_size=(16, 16, 8),        # spatial patch (X, Y, Z)
        n_layers=4,
    ).cuda()

    x = torch.randn(2, 4, 64, 64, 32, 64, device='cuda')  # (B, C, X, Y, Z, T)
    y = model(x)                                            # same shape

References
----------
  arXiv:2504.04260  — LOGLO-FNO paper (Kalimuthu et al., TMLR 2025)
  loglo_fno_2d.py   — 2D reference implementation (this project)
  causal_loglo_4d.py — causal 4D variant with τ-embedding (this project)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for operating on (B, C, X, Y, Z, T) via T-fold-into-batch
# ─────────────────────────────────────────────────────────────────────────────


def _fold_T(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """(B, C, X, Y, Z, T) → (B·T, C, X, Y, Z).  Returns tensor, B, T."""
    B, C, X, Y, Z, T = x.shape
    return x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z), B, T


def _unfold_T(x_bt: torch.Tensor, B: int, T: int) -> torch.Tensor:
    """(B·T, C, X, Y, Z) → (B, C, X, Y, Z, T)."""
    _, C, X, Y, Z = x_bt.shape
    return x_bt.view(B, T, C, X, Y, Z).permute(0, 2, 3, 4, 5, 1)


def _apply_3d_op_on_4d(op: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a 3D nn.Module to (B, C, X, Y, Z, T) by folding T into batch."""
    x_bt, B, T = _fold_T(x)
    return _unfold_T(op(x_bt), B, T)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules (4D-aware versions of the 2D originals)
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelMLP4D(nn.Module):
    """
    Two-layer pointwise MLP over channels for (B, C, X, Y, Z, T).

    T is folded into the batch axis so standard Conv3d 1×1 ops apply.
    Implements Wc2 ∗ σ[Wc1 ∗ x + bc1] + bc2 (Eq. 1–2).
    """

    def __init__(self, channels: int):
        super().__init__()
        self._op = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_3d_op_on_4d(self._op, x)


class _SoftGating4D(nn.Module):
    """
    Learnable channel-wise affine: Wc ⊙ x + bc  (Eq. 1–2, skip connections).

    Scale and bias broadcast over all spatial and temporal positions.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, channels, 1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class _Conv1x1_4D(nn.Module):
    """
    Pointwise 1×1×1×1 convolution for (B, C, X, Y, Z, T).

    Implemented as Conv3d(kernel=1) with T folded into batch — equivalent
    to a channel-mixing linear transform at every spatiotemporal point.
    Corresponds to the W ∗ Z + b linear skip in Eq. 1–2.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _apply_3d_op_on_4d(self._conv, x)


class _HFPExtractor4D(nn.Module):
    """
    High-Frequency Propagation extractor for 4D fields (§3.1, Appendix E).

    Extends the paper's 2D AvgPool2d + bilinear upsample to 4D by applying
    AvgPool3d over the spatial axes (X, Y, Z) with T folded into the batch:

      1. Fold T into batch → (B·T, C, X, Y, Z)
      2. x_smooth = Interpolate(AvgPool3d(x; kernel, stride), size=(X,Y,Z), trilinear)
      3. H = x − x_smooth
      4. Unfold T → (B, C, X, Y, Z, T)

    Pool kernel and stride must be ≤ min(X, Y, Z); default 4 matches Appendix L.7.
    """

    def __init__(self, pool_kernel: int = 4, pool_stride: int = 4):
        super().__init__()
        self.pool = nn.AvgPool3d(
            kernel_size=pool_kernel,
            stride=pool_stride,
            count_include_pad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, X, Y, Z, T = x.shape
        x_bt, _, _ = _fold_T(x)               # (B·T, C, X, Y, Z)
        x_smooth = F.interpolate(
            self.pool(x_bt),
            size=(X, Y, Z),
            mode='trilinear',
            align_corners=False,
        )
        h_bt = x_bt - x_smooth                 # HF residual
        return _unfold_T(h_bt, B, T)           # (B, C, X, Y, Z, T)


# ─────────────────────────────────────────────────────────────────────────────
# Patch operations (spatial patches, T preserved)
# ─────────────────────────────────────────────────────────────────────────────


def _split_patches_4d(
    x: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Split (B, C, X, Y, Z, T) into (B·M·T, C, Px, Py, Pz) non-overlapping
    spatial patches with T folded into batch.

    Patch count M = (X/Px) × (Y/Py) × (Z/Pz).
    T is preserved on every patch before being folded with the batch.
    Requires X%Px==0, Y%Py==0, Z%Pz==0.
    """
    B, C, X, Y, Z, T = x.shape
    Px, Py, Pz = patch_size
    Mx, My, Mz = X // Px, Y // Py, Z // Pz

    # (B, C, X, Y, Z, T)
    # → view   (B, C, Mx, Px, My, Py, Mz, Pz, T)
    # → perm   (B, Mx, My, Mz, C, Px, Py, Pz, T)  indices: 0,2,4,6,1,3,5,7,8
    # → reshape (B·M, C, Px, Py, Pz, T)
    patches_with_T = (
        x.view(B, C, Mx, Px, My, Py, Mz, Pz, T)
        .permute(0, 2, 4, 6, 1, 3, 5, 7, 8)
        .reshape(B * Mx * My * Mz, C, Px, Py, Pz, T)
    )
    # Fold T into batch for 3D SpectralConv
    BM = B * Mx * My * Mz
    return patches_with_T.reshape(BM * T, C, Px, Py, Pz)


def _merge_patches_4d(
    patches_bt: torch.Tensor,
    B: int,
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    T: int,
) -> torch.Tensor:
    """
    Inverse of _split_patches_4d.
    patches_bt : (B·M·T, C, Px, Py, Pz) → (B, C, X, Y, Z, T)
    """
    X, Y, Z = spatial_shape
    Px, Py, Pz = patch_size
    Mx, My, Mz = X // Px, Y // Py, Z // Pz
    M  = Mx * My * Mz
    C  = patches_bt.shape[1]

    # Unfold T from batch → (B·M, C, Px, Py, Pz, T)
    patches_with_T = patches_bt.view(B * M, C, Px, Py, Pz, T)

    # (B·M, C, Px, Py, Pz, T)
    # → view   (B, Mx, My, Mz, C, Px, Py, Pz, T)
    # → perm   (B, C, Mx, Px, My, Py, Mz, Pz, T)  indices: 0,4,1,5,2,6,3,7,8
    # → view   (B, C, X, Y, Z, T)
    return (
        patches_with_T
        .view(B, Mx, My, Mz, C, Px, Py, Pz, T)
        .permute(0, 4, 1, 5, 2, 6, 3, 7, 8)
        .contiguous()
        .view(B, C, X, Y, Z, T)
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOGLO Fourier layer  (Eq. 2 — 4D extension)
# ─────────────────────────────────────────────────────────────────────────────


class LOGLOFourierLayer4D(nn.Module):
    """
    One LOGLO Fourier layer for 4D fields, extending Eq. 2 of the paper.

    Three parallel branches operating on x ∈ (B, C, X, Y, Z, T):

      Global branch  — SpectralConv4D (mx, my, mz, mt modes) + Conv1x1_4D
      Local branch   — 3D spatial patches, SpectralConv3D (all modes), T in batch
      HFP branch     — ChannelMLP(z_prime), z_prime passed from previous layer

    Parameters
    ----------
    channels : int
        Hidden channel width C.
    n_modes_global : (mx, my, mz, mt)
        4-tuple of spectral modes for the global branch.
    patch_size : (Px, Py, Pz)
        Spatial patch for the local branch. T is not patched.
        n_modes_local is set automatically to (Px//2, Py//2, Pz//2+1).
    apply_activation : bool
        If False, the GELU on the branch sum is skipped (paper footnote 1:
        last Fourier layer has no activation).
    """

    def __init__(
        self,
        channels: int,
        n_modes_global: tuple[int, int, int, int],
        patch_size: tuple[int, int, int],
    ):
        super().__init__()
        C  = channels
        Px, Py, Pz = patch_size
        # Local branch keeps ALL modes within each spatial patch (§3.3).
        # rfft on the last dim (Pz) → Pz//2+1 unique coefficients (incl. Nyquist).
        n_modes_local = (Px // 2, Py // 2, Pz // 2 + 1)

        self.patch_size = patch_size

        # ── Global branch  (Υg = σ[Kg(Z) + Wg∗Z + bg]) ──────────────────────
        self.global_spectral = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_global,            # 4-tuple: full (X,Y,Z,T) spectrum
            factorization=None,
            implementation='reconstructed',
        )
        self.global_skip = _Conv1x1_4D(C, C)  # Wg ∗ Z + bg
        self.global_mlp  = _ChannelMLP4D(C)   # Wgc2 ∗ σ[Wgc1 ∗ Υg + ·] + ·
        self.global_gate = _SoftGating4D(C)   # Wgc ⊙ Z + bgc

        # ── Local branch  (Υl = σ[Kl(Ẑ) + Wl∗Ẑ + bl]) ───────────────────────
        # SpectralConv3D operating on (B·M·T, C, Px, Py, Pz).
        self.local_spectral = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_local,             # 3-tuple: all modes within patch
            factorization=None,
            implementation='reconstructed',
        )
        self.local_skip = nn.Conv3d(C, C, kernel_size=1)  # Wl ∗ Ẑ + bl
        self.local_mlp  = _ChannelMLP4D(C)               # Wlc2 ∗ σ[Wlc1 ∗ · + ·] + ·
        self.local_gate = _SoftGating4D(C)               # Wlc ⊙ Z + blc

        # ── HFP branch  (Whfc2 ∗ σ[Whfc1 ∗ Z' + ·] + ·) ────────────────────
        self.hfp_mlp = _ChannelMLP4D(C)

    def forward(
        self,
        z: torch.Tensor,              # (B, C, X, Y, Z, T) — main stream
        z_prime: torch.Tensor,        # (B, C, X, Y, Z, T) — HFP recurrent state
        apply_activation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (z_new, z_prime_new).

        z_new       : updated main stream  (B, C, X, Y, Z, T)
        z_prime_new : updated HFP state    (B, C, X, Y, Z, T)
                      passed to the next layer as z_prime
        """
        B, C, X, Y, Z, T = z.shape

        # ── Global branch ─────────────────────────────────────────────────────
        upsilon_g = F.gelu(self.global_spectral(z) + self.global_skip(z))
        y_global  = self.global_mlp(upsilon_g) + self.global_gate(z)

        # ── Local branch ───────────────────────────────────────────────────────
        # Patch (X,Y,Z), fold T into batch → SpectralConv3D → reassemble.
        patches   = _split_patches_4d(z, self.patch_size)   # (B·M·T, C, Px, Py, Pz)
        upsilon_l = F.gelu(
            self.local_spectral(patches) + self.local_skip(patches)
        )
        y_local   = _merge_patches_4d(
            upsilon_l, B, (X, Y, Z), self.patch_size, T
        )                                                     # (B, C, X, Y, Z, T)
        y_local   = self.local_mlp(y_local) + self.local_gate(z)

        # ── HFP branch ─────────────────────────────────────────────────────────
        # z_prime flows as a recurrent state; updated by ChannelMLP each layer.
        y_hfp = self.hfp_mlp(z_prime)                        # (B, C, X, Y, Z, T)

        # ── Sum all branches (Eq. 2) ───────────────────────────────────────────
        z_new = y_global + y_local + y_hfp
        if apply_activation:
            z_new = F.gelu(z_new)

        # z_prime for the next layer = this layer's HFP ChannelMLP output
        return z_new, y_hfp


# ─────────────────────────────────────────────────────────────────────────────
# Full LOGLO-FNO 4D model
# ─────────────────────────────────────────────────────────────────────────────


class LOGLOFNO4D(nn.Module):
    """
    LOGLO-FNO extended to 4D fields (B, C, X, Y, Z, T).

    Direct generalisation of LOGLOFNO2D (loglo_fno_2d.py) to 4D, keeping the
    paper's architecture: global 4D SpectralConv, local 3D spatial patches,
    HFP with AvgPool3d + trilinear upsample and a recurrent z_prime stream.

    Parameters
    ----------
    in_channels : int
        Number of physical input fields.
    out_channels : int
        Number of predicted output fields.
    hidden_channels : int
        Latent width C.
    n_modes_global : (mx, my, mz, mt)
        Global spectral modes along (X, Y, Z, T). The number of temporal modes
        mt controls how many temporal Fourier modes the global branch sees.
        Typical: mx=my≈grid//8, mz≈grid_z//8, mt≈T//8.
    patch_size : (Px, Py, Pz)
        Spatial patch for the local branch. T is NOT patched.
        Grid must satisfy X%Px==0, Y%Py==0, Z%Pz==0.
        n_modes_local = (Px//2, Py//2, Pz//2+1) is set automatically.
    n_layers : int
        Number of LOGLO Fourier layers.
    hfp_pool_kernel : int
        AvgPool3d kernel in HFP extractor. Must be ≤ min(X,Y,Z). Default 4.
    hfp_pool_stride : int
        AvgPool3d stride in HFP extractor. Default 4.
    hfp_noise_alpha : float
        Scale factor α for HF-adaptive Gaussian noise injected during training
        (Appendix E, Eq. 11). Set 0.0 to disable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_global: tuple[int, int, int, int],
        patch_size: tuple[int, int, int],
        n_layers: int = 4,
        hfp_pool_kernel: int = 4,
        hfp_pool_stride: int = 4,
        hfp_noise_alpha: float = 0.025,
    ):
        super().__init__()
        C = hidden_channels
        self.hfp_noise_alpha = hfp_noise_alpha
        self.n_layers = n_layers

        # ── Lifting (P) — shared for main stream and HFP stream ──────────────
        # The paper lifts both the input and HFP features using shared weights.
        # Implemented as Conv3d(C_in, C, 1) with T folded into batch.
        self._lift_conv = nn.Conv3d(in_channels, C, kernel_size=1)

        # ── HFP extractor ──────────────────────────────────────────────────────
        self.hfp = _HFPExtractor4D(
            pool_kernel=hfp_pool_kernel,
            pool_stride=hfp_pool_stride,
        )

        # ── LOGLO Fourier layers ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            LOGLOFourierLayer4D(C, n_modes_global, patch_size)
            for _ in range(n_layers)
        ])

        # ── Projection (Q) — two-layer MLP ────────────────────────────────────
        self.proj = nn.Sequential(
            _Conv1x1_4D(C, C),
            nn.GELU(),
            _Conv1x1_4D(C, out_channels),
        )

    def _lift(self, x: torch.Tensor) -> torch.Tensor:
        """Lift (B, in_channels, X, Y, Z, T) → (B, C, X, Y, Z, T)."""
        return _apply_3d_op_on_4d(self._lift_conv, x)

    def forward(
        self,
        x: torch.Tensor,
        training_noise: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, in_channels, X, Y, Z, T)
        training_noise : bool
            If True, inject HF-adaptive Gaussian noise into the input during
            training (Appendix E). Should only be True during training.

        Returns
        -------
        (B, out_channels, X, Y, Z, T)
        """
        # ── HFP feature extraction before lifting ──────────────────────────────
        x_hf = self.hfp(x)    # H = X − smooth(X),  shape (B, in_C, X, Y, Z, T)

        # Optional HF-adaptive noise injection (Appendix E, Eq. 11)
        # N_dynamic = μ + α·σ·N(0,1), added to input during training.
        if training_noise and self.training and self.hfp_noise_alpha > 0.0:
            mu    = x_hf.mean(dim=(1, 2, 3, 4, 5), keepdim=True)
            std   = x_hf.std(dim=(1, 2, 3, 4, 5), keepdim=True) + 1e-8
            noise = mu + self.hfp_noise_alpha * std * torch.randn_like(x)
            x     = x + noise

        # ── Lifting (shared P for both streams) ────────────────────────────────
        z       = self._lift(x)     # (B, C, X, Y, Z, T)
        z_prime = self._lift(x_hf)  # shared weights, HFP stream

        # ── LOGLO Fourier layers ───────────────────────────────────────────────
        for i, layer in enumerate(self.layers):
            is_last = (i == self.n_layers - 1)
            z, z_prime = layer(z, z_prime, apply_activation=not is_last)

        # ── Projection ─────────────────────────────────────────────────────────
        return self.proj(z)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-aware loss  (§3.2, Appendix D.2 — 4D extension)
# ─────────────────────────────────────────────────────────────────────────────


def radially_binned_spectral_loss_4d(
    pred: torch.Tensor,
    target: torch.Tensor,
    i_low: int = 4,
    i_high: int = 12,
    penalise_bands: tuple[bool, bool, bool] = (False, True, True),
) -> torch.Tensor:
    """
    Frequency-aware loss (§3.2, Appendix D.2) extended to 4D fields.

    Computes radially-binned spectral energy errors over the 3 SPATIAL axes
    (X, Y, Z). The T axis is folded into the batch dimension so each τ-slice
    contributes independently — preserving the per-step training interpretation.

    Parameters
    ----------
    pred, target : (B, C, X, Y, Z, T)
    i_low  : radial bin separating low from mid frequencies. Paper: 4 (64² grid).
             Scale proportionally: e.g. 128³ grid → i_low=8.
    i_high : radial bin separating mid from high frequencies. Paper: 12.
    penalise_bands : (bool, bool, bool) for (low, mid, high).
                     Paper default: (False, True, True) — only mid+high.

    Returns
    -------
    loss : scalar tensor
    """
    B, C, X, Y, Z, T = pred.shape
    # Fold T into batch → treat each τ-slice as an independent sample
    p = pred.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
    t = target.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
    nb = p.shape[0]

    # FFT over spatial axes; take positive-octant only
    err_fft_sq = torch.abs(
        torch.fft.fftn(p - t, dim=[-3, -2, -1])
    ) ** 2                                                 # (nb, C, X, Y, Z)
    err_half = err_fft_sq[..., :X // 2, :Y // 2, :Z // 2]  # positive octant

    # Radial distance grid (integer bins, 3D)
    xi = torch.arange(X // 2, device=pred.device)
    yi = torch.arange(Y // 2, device=pred.device)
    zi = torch.arange(Z // 2, device=pred.device)
    Xg, Yg, Zg = torch.meshgrid(xi, yi, zi, indexing='ij')
    radii = (Xg.float() ** 2 + Yg.float() ** 2 + Zg.float() ** 2).sqrt().floor().long()
    max_radius = int(radii.max().item())

    radii_flat      = radii.flatten()                         # (X//2 * Y//2 * Z//2)
    err_flat        = err_half.reshape(nb, C, -1)             # (nb, C, spatial_flat)

    # Accumulate per radial bin
    err_F = torch.zeros(nb, C, max_radius + 1, device=pred.device)
    valid = radii_flat <= max_radius
    err_F.index_add_(2, radii_flat[valid], err_flat[:, :, valid])

    # Normalise + mean over batch
    nrm   = X * Y * Z
    err_F = (err_F.mean(dim=0) / nrm).sqrt()                 # (C, max_radius+1)

    def _band_mean(start: int, end: int) -> torch.Tensor:
        end = min(end, err_F.shape[-1])
        if start >= end:
            return err_F.new_zeros(())
        return err_F[:, start:end].mean()

    err_low  = _band_mean(0, i_low)
    err_mid  = _band_mean(i_low, i_high)
    err_high = _band_mean(i_high, max_radius + 1)

    loss = pred.new_zeros(())
    for penalise, band in zip(penalise_bands, [err_low, err_mid, err_high]):
        if penalise:
            loss = loss + band

    return loss


class LOGLOLoss4D(nn.Module):
    """
    Combined training loss for LOGLOFNO4D: MSE + λ·L_freq  (Eq. 5).

    L_freq is computed over 3D spatial radial bins at each τ-step
    (T folded into batch). Only mid- and high-frequency bands are penalised
    by default, matching the paper's training objective.

    Parameters
    ----------
    lam : float
        Weight λ on the frequency loss. Tune down (0.1–0.5) if the frequency
        loss dominates early training; the MSE should converge first.
    i_low, i_high : int
        Radial bin boundaries for low/mid/high classification.
        Scale with grid size: i_low=4, i_high=12 for 64³ spatial grid.
        For 128³: i_low=8, i_high=24. For 32³: i_low=2, i_high=6.
    """

    def __init__(
        self,
        lam: float = 1.0,
        i_low: int = 4,
        i_high: int = 12,
    ):
        super().__init__()
        self.lam    = lam
        self.i_low  = i_low
        self.i_high = i_high

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_mse  = F.mse_loss(pred, target)
        l_freq = radially_binned_spectral_loss_4d(
            pred, target,
            i_low=self.i_low,
            i_high=self.i_high,
            penalise_bands=(False, True, True),
        )
        total = l_mse + self.lam * l_freq
        return total, {
            'mse':   l_mse.item(),
            'freq':  l_freq.item(),
            'total': total.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    import math
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    # Grid: X=Y=16, Z=8, T=16 (CPU-safe smoke test)
    # Patch: 8×8×4 → (16/8)×(16/8)×(8/4) = 2×2×2 = 8 spatial patches
    # Global modes: (4, 4, 2, 4) — (mx, my, mz, mt)
    # Local modes (auto): (4, 4, 3) = (Px//2, Py//2, Pz//2+1) for patch 8×8×4
    X, Y, Z, T = 16, 16, 8, 16
    patch_size  = (8, 8, 4)
    Px, Py, Pz  = patch_size

    model = LOGLOFNO4D(
        in_channels=2,
        out_channels=2,
        hidden_channels=12,
        n_modes_global=(4, 4, 2, 4),
        patch_size=patch_size,
        n_layers=4,
        hfp_pool_kernel=4,
        hfp_pool_stride=4,
        hfp_noise_alpha=0.025,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters())
    n_complex = sum(p.numel() for p in model.parameters() if p.is_complex())
    print(f'Total parameters : {n_params:,}')
    print(f'Complex params   : {n_complex:,}  (SpectralConv weights)')
    print(f'Real params      : {n_params - n_complex:,}')
    print(f'Local modes (Px//2, Py//2, Pz//2+1): '
          f'({Px//2}, {Py//2}, {Pz//2+1})')
    print(f'  → keeps ALL spatial modes within each {Px}×{Py}×{Pz} patch')
    print()

    # ── Forward pass ──────────────────────────────────────────────────────────
    x = torch.randn(2, 2, X, Y, Z, T, device=device)
    y = model(x, training_noise=True)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y.shape)}')
    assert y.shape == x.shape, 'Shape mismatch'
    print('Shape OK ✓\n')

    # ── Loss ──────────────────────────────────────────────────────────────────
    target  = torch.randn_like(y)
    loss_fn = LOGLOLoss4D(lam=1.0, i_low=2, i_high=6)  # scaled for 16³ grid
    total, bd = loss_fn(y, target)
    print(f"MSE  : {bd['mse']:.4f}")
    print(f"Freq : {bd['freq']:.4f}")
    print(f"Total: {bd['total']:.4f}")

    # ── Backward ──────────────────────────────────────────────────────────────
    total.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f'\nGradients: {n_grads}/{n_total} parameter tensors ✓')
    assert not math.isnan(total.item()), 'NaN loss'

    # ── Patch/merge round-trip ─────────────────────────────────────────────────
    print('\n── Patch round-trip ──────────────────────────────────────────────')
    x_test = torch.randn(2, 12, X, Y, Z, T)
    Mx, My, Mz = X//Px, Y//Py, Z//Pz
    M = Mx * My * Mz
    patches = _split_patches_4d(x_test, patch_size)
    print(f'  Split: {tuple(x_test.shape)} → {tuple(patches.shape)}')
    assert patches.shape == (2 * M * T, 12, Px, Py, Pz)
    x_rec = _merge_patches_4d(patches, 2, (X, Y, Z), patch_size, T)
    print(f'  Merge: {tuple(patches.shape)} → {tuple(x_rec.shape)}')
    assert torch.allclose(x_test, x_rec), 'Patch round-trip failed'
    print('  Round-trip OK ✓')

    # ── HFP extractor shape ────────────────────────────────────────────────────
    print('\n── HFP extractor ─────────────────────────────────────────────────')
    hfp = _HFPExtractor4D(pool_kernel=4, pool_stride=4)
    xh  = hfp(x)
    print(f'  Input HF residual: {tuple(xh.shape)}')
    assert xh.shape == x.shape, 'HFP shape mismatch'
    print('  HFP shape OK ✓')

    # ── Consistency check vs LOGLOFNO2D structure ──────────────────────────────
    print('\n── Architecture consistency with loglo_fno_2d.py ─────────────────')
    print('  Global branch : SpectralConv4D  (X,Y,Z,T modes)       ← extends 2D SpectralConv2D')
    print('  Local branch  : SpectralConv3D  (Px,Py,Pz, T in batch) ← extends 2D SpectralConv2D')
    print('  HFP           : AvgPool3d + trilinear upsample          ← extends 2D AvgPool2d + bilinear')
    print('  HFP state     : recurrent z_prime                        ← identical to 2D')
    print('  SoftGating    : (1,C,1,1,1,1) scale+bias                ← extends (1,C,1,1)')
    print('  ChannelMLP    : Conv3d(1) with T in batch               ← extends Conv2d(1)')
    print('  Last-layer GELU skipped                                  ← identical to 2D')
    print()

    print('All checks passed. ✓')
