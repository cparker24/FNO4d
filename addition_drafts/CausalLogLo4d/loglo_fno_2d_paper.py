"""
loglo_fno_2d.py
===============
Reference implementation of LOGLO-FNO for 2D spatial data.

Faithfully implements the architecture described in:
  Kalimuthu, Holzmüller, Niepert — "LOGLO-FNO: Efficient Learning of Local and
  Global Features in Fourier Neural Operators", TMLR 2025, arXiv:2504.04260

All equation, section, and appendix references below point to that paper.

Architecture (Fig. 1, Eq. 2)
-----------------------------
The LOGLO Fourier layer modifies the standard FNO layer (Eq. 1) with three
parallel branches:

  Υg := σ[ Kg(Z) + Wg ∗ Z + bg ]                                 (global pre-act)
  Υl := σ[ Kl(Ẑ) + Wl ∗ Ẑ + bl ]                               (local pre-act)

  L(Z, Ẑ, Z') :=
      Wgc2 ∗ σ[Wgc1 ∗ Υg + bgc1] + bgc2   (global ChannelMLP)
    + Wgc ⊙ Z + bgc                         (global SoftGating skip)
    + Wlc2 ∗ σ[Wlc1 ∗ Υl + blc1] + blc2   (local ChannelMLP)
    + Wlc ⊙ Ẑ + blc                         (local SoftGating skip)
    + Whfc2 ∗ σ[Whfc1 ∗ Z' + bhfc1] + bhfc2 (HFP ChannelMLP)

Where:
  Z  — main feature stream (B, C, Nx, Ny), full spatial resolution
  Ẑ  — local branch input: derived from Z by non-overlapping patch decomposition
  Z' — HFP (High-Frequency Propagation) recurrent stream (B, C, Nx, Ny)
  Kg — global SpectralConv2D (truncated modes, learnable kernel)
  Kl — local SpectralConv2D (ALL modes within each patch)
  ∗  — 1×1 convolution
  ⊙  — element-wise multiplication (channel-wise learnable scale)
  σ  — GELU (skipped on the last Fourier layer)

HFP module (§3.1, Appendix E)
------------------------------
Extracts high-frequency features via average pooling + bilinear upsampling:
  XS = AvgPool2d(X; kernel_size=4, stride=4)
  XU = Interpolate(XS; size=(Nx, Ny), mode='bilinear')
  H  = X − XU      ← high-frequency residual

H is lifted with a shared lifting layer P. The HFP stream flows as a
recurrent state Z' through all LOGLO layers (updated by ChannelMLP at each layer).

Local spectral convolution (§3.1, Appendix F.1)
------------------------------------------------
Domain D is partitioned into M = (Nx/Px) × (Ny/Py) non-overlapping patches.
SpectralConv2D retains ALL Fourier modes within each patch (no truncation).
For patch size Px × Py, the mode count is:
  n_modes_local = (Px, Py // 2 + 1)     [Px modes × Nyquist along y]

In the paper's Kolmogorov Flow experiment: patch 16×16 → modes (16, 9).

Frequency-aware loss (§3.2, Appendix D.2)
------------------------------------------
Radially-binned spectral energy errors classified into low/mid/high bands.
Only mid- and high-frequency errors are penalised in the training objective:
  L_total = MSE + λ · (L_mid + L_high)
With iLow=4, iHigh=12 as band boundaries (radial bin indices in the paper's
2D Kolmogorov Flow experiments, grid 64×64).

Adaptive Gaussian noise (Appendix E)
-------------------------------------
During teacher-forcing training, HF-adaptive Gaussian noise is injected into
the input to improve robustness of autoregressive rollout:
  N_dynamic = μ + α·σ·N(0,1),  α = 0.025
where μ, σ are per-sample statistics of the HF features extracted by the HFP module.

Hyperparameters (Appendix L.7, Table 14)
-----------------------------------------
  2D Kolmogorov Flow (1-step, Re=5000):
    n_modes_global = 40
    hidden_channels = 65
    patch_size = (16, 16) → n_modes_local = (16, 9)
    n_layers = 4
    hfp_pool_kernel = 4, hfp_pool_stride = 4
    lr = 4.4e-3, StepLR(step=33, gamma=0.5), 136 epochs
    batch_size = 256
    loss = MSE + (mid+high) freq

  2D Diffusion-Reaction (autoregressive):
    n_modes_global = 10, hidden_channels = 24, patch_size = (8, 8)
    lr = 8e-4, StepLR(step=100, gamma=0.5), 500 epochs

  TRL3D (1-step, 3D extension):
    n_modes_global = 18, hidden_channels = 48, global modes 3D
    patch_size = (16, 16, 16) [asymmetric: (16, 16, 17) in practice]

Tensor layout: (B, C, Nx, Ny)  [2D; time handled autoregressively outside]

References
----------
  arXiv:2504.04260   — LOGLO-FNO paper (Kalimuthu et al., TMLR 2025)
  arXiv:2010.08895   — Original FNO (Li et al., ICLR 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv


# ─────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelMLP2D(nn.Module):
    """
    Two-layer pointwise MLP over channels, operating on (B, C, Nx, Ny) via
    1×1 Conv2d. Corresponds to Wc2 ∗ σ[Wc1 ∗ x + bc1] + bc2 in Eq. 1–2.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SoftGating2D(nn.Module):
    """
    Learnable channel-wise affine: Wc ⊙ x + bc  (Eq. 1–2, skip connections).
    ⊙ is element-wise multiplication by a per-channel scalar (broadcast over Nx, Ny).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


class _HFPExtractor(nn.Module):
    """
    High-Frequency Propagation extractor (§3.1, Appendix E).

    Extracts high-frequency spatial features by subtracting a blurred
    (average-pooled then bilinearly upsampled) version of the input:
      H = X − Interpolate(AvgPool2d(X; k, s); size=(Nx, Ny))

    The degree of blurring (and hence the frequency cutoff) is controlled
    by pool_kernel and pool_stride: larger kernel → more aggressive blurring
    → more high-frequency content in H. Paper uses kernel=4, stride=4.
    """

    def __init__(self, pool_kernel: int = 4, pool_stride: int = 4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Nx, Ny = x.shape[-2], x.shape[-1]
        x_smooth = F.interpolate(
            self.pool(x),
            size=(Nx, Ny),
            mode='bilinear',
            align_corners=False,
        )
        return x - x_smooth    # high-frequency residual H


def _split_patches_2d(
    x: torch.Tensor,
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """
    Split (B, C, Nx, Ny) into (B·M, C, Px, Py) non-overlapping patches.

    Patch count M = (Nx/Px) × (Ny/Py). Requires Nx % Px == 0, Ny % Py == 0.
    """
    B, C, Nx, Ny = x.shape
    Px, Py = patch_size
    Mx, My = Nx // Px, Ny // Py

    # (B, C, Nx, Ny)
    # → view   (B, C, Mx, Px, My, Py)
    # → perm   (B, Mx, My, C, Px, Py)   indices 0,2,4,1,3,5
    # → reshape (B·Mx·My, C, Px, Py)
    return (
        x.view(B, C, Mx, Px, My, Py)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(B * Mx * My, C, Px, Py)
    )


def _merge_patches_2d(
    patches: torch.Tensor,
    B: int,
    spatial_shape: tuple[int, int],
    patch_size: tuple[int, int],
) -> torch.Tensor:
    """
    Inverse of _split_patches_2d.
    patches : (B·M, C, Px, Py)  →  (B, C, Nx, Ny)
    """
    Nx, Ny = spatial_shape
    Px, Py = patch_size
    Mx, My = Nx // Px, Ny // Py
    C = patches.shape[1]

    # (B·Mx·My, C, Px, Py)
    # → view   (B, Mx, My, C, Px, Py)
    # → perm   (B, C, Mx, Px, My, Py)   indices 0,3,1,4,2,5
    # → view   (B, C, Nx, Ny)
    return (
        patches
        .view(B, Mx, My, C, Px, Py)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(B, C, Nx, Ny)
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOGLO Fourier layer  (Eq. 2)
# ─────────────────────────────────────────────────────────────────────────────


class LOGLOFourierLayer(nn.Module):
    """
    One LOGLO Fourier layer implementing Eq. 2 of Kalimuthu et al. 2025.

    Takes three inputs:
      z       — main feature stream  (B, C, Nx, Ny)
      z_prime — HFP recurrent stream (B, C, Nx, Ny)
      apply_activation — False for the last layer (§2, footnote 1)

    Returns:
      z_new       — updated main stream
      z_prime_new — updated HFP stream (ChannelMLP output, passed to next layer)

    Parameters
    ----------
    channels : int
        Hidden channel width C.
    n_modes_global : (int, int)
        (mx, my) — Fourier modes for global SpectralConv (truncated).
    patch_size : (int, int)
        (Px, Py) — spatial patch for the local branch.
        n_modes_local is set automatically to (Px, Py // 2 + 1).
    """

    def __init__(
        self,
        channels: int,
        n_modes_global: tuple[int, int],
        patch_size: tuple[int, int],
    ):
        super().__init__()
        C = channels
        Px, Py = patch_size
        # Local branch retains ALL modes within each patch (§3.3):
        # Px modes along x, Py//2 + 1 modes along y (Nyquist limit after rfft).
        n_modes_local = (Px, Py // 2 + 1)

        self.patch_size = patch_size

        # ── Global branch (Kg, Wg, bg; Wgc1, Wgc2) ──────────────────────────
        self.global_spectral = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_global,
            factorization=None,           # paper uses dense (unfactorized) weights
            implementation='reconstructed',
        )
        self.global_skip_conv = nn.Conv2d(C, C, kernel_size=1)  # Wg ∗ Z + bg
        self.global_mlp       = _ChannelMLP2D(C)                # Wgc2 ∗ σ[Wgc1 ∗ · + ·] + ·
        self.global_gate      = _SoftGating2D(C)                # Wgc ⊙ Z + bgc

        # ── Local branch (Kl, Wl, bl; Wlc1, Wlc2) ───────────────────────────
        self.local_spectral  = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_local,
            factorization=None,
            implementation='reconstructed',
        )
        self.local_skip_conv = nn.Conv2d(C, C, kernel_size=1)   # Wl ∗ Ẑ + bl
        self.local_mlp       = _ChannelMLP2D(C)                 # Wlc2 ∗ σ[Wlc1 ∗ · + ·] + ·
        self.local_gate      = _SoftGating2D(C)                 # Wlc ⊙ Ẑ + blc

        # ── HFP branch (Whfc1, Whfc2) ─────────────────────────────────────────
        self.hfp_mlp = _ChannelMLP2D(C)  # Whfc2 ∗ σ[Whfc1 ∗ Z' + ·] + ·

    def forward(
        self,
        z: torch.Tensor,
        z_prime: torch.Tensor,
        apply_activation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, Nx, Ny = z.shape

        # ── Global branch ─────────────────────────────────────────────────────
        upsilon_g = F.gelu(self.global_spectral(z) + self.global_skip_conv(z))
        y_global  = self.global_mlp(upsilon_g) + self.global_gate(z)

        # ── Local branch (patch-wise spectral conv) ────────────────────────────
        # Ẑ is derived from Z at each layer by patch decomposition.
        patches   = _split_patches_2d(z, self.patch_size)    # (B·M, C, Px, Py)
        upsilon_l = F.gelu(
            self.local_spectral(patches) + self.local_skip_conv(patches)
        )
        y_local_patches = upsilon_l                           # (B·M, C, Px, Py)
        y_local = _merge_patches_2d(
            y_local_patches, B, (Nx, Ny), self.patch_size
        )                                                      # (B, C, Nx, Ny)
        y_local = self.local_mlp(y_local) + self.local_gate(z)

        # ── HFP branch ────────────────────────────────────────────────────────
        # Z' is the recurrent HFP state; updated by ChannelMLP each layer.
        y_hfp = self.hfp_mlp(z_prime)

        # ── Combine all branches (Eq. 2) ──────────────────────────────────────
        z_new = y_global + y_local + y_hfp

        if apply_activation:
            z_new = F.gelu(z_new)

        # HFP state for next layer = current HFP branch output (Appendix E)
        z_prime_new = y_hfp

        return z_new, z_prime_new


# ─────────────────────────────────────────────────────────────────────────────
# Full LOGLO-FNO model
# ─────────────────────────────────────────────────────────────────────────────


class LOGLOFNO2D(nn.Module):
    """
    Standard LOGLO-FNO for 2D spatial data (arXiv:2504.04260).

    Intended as a reference implementation faithful to the paper. It handles
    2D spatial fields of shape (B, C, Nx, Ny); the time dimension is managed
    autoregressively outside this module (1-step or AR rollout).

    Parameters
    ----------
    in_channels : int
        Number of input field channels (e.g. 1 for vorticity in Kolmogorov flow).
    out_channels : int
        Number of predicted output channels.
    hidden_channels : int
        Latent width C.  Paper uses 65 (Kolmogorov), 24 (Diffusion-Reaction).
    n_modes_global : (int, int)
        (mx, my) global spectral modes. Paper uses 40 (Kolmogorov), 10 (DR).
    patch_size : (int, int)
        (Px, Py) local branch patch. Paper uses (16, 16) (Kol.), (8, 8) (DR).
        Grid must be divisible: Nx % Px == 0, Ny % Py == 0.
    n_layers : int
        Number of LOGLO Fourier layers. Standard FNO uses 4.
    hfp_pool_kernel : int
        AvgPool2d kernel size in HFP extractor. Paper uses 4 (Appendix L.7).
    hfp_pool_stride : int
        AvgPool2d stride in HFP extractor. Paper uses 4 (Appendix L.7).
    hfp_noise_alpha : float
        Scale factor α for adaptive Gaussian noise injected during training.
        Paper uses 0.025 (Appendix E). Set 0.0 to disable.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_global: tuple[int, int],
        patch_size: tuple[int, int],
        n_layers: int = 4,
        hfp_pool_kernel: int = 4,
        hfp_pool_stride: int = 4,
        hfp_noise_alpha: float = 0.025,
    ):
        super().__init__()
        C = hidden_channels
        self.hfp_noise_alpha = hfp_noise_alpha

        # ── Lifting (P) ───────────────────────────────────────────────────────
        # Shared for both main features and HFP stream (Appendix E:
        # "lifted using P with shared parameters").
        # In practice we use one Conv2d with shared weights for both.
        self.lift = nn.Conv2d(in_channels, C, kernel_size=1)

        # ── HFP extractor ──────────────────────────────────────────────────────
        self.hfp = _HFPExtractor(pool_kernel=hfp_pool_kernel,
                                  pool_stride=hfp_pool_stride)

        # ── Trunk: LOGLO Fourier layers ───────────────────────────────────────
        self.layers = nn.ModuleList([
            LOGLOFourierLayer(C, n_modes_global, patch_size)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers

        # ── Projection (Q) ────────────────────────────────────────────────────
        # Paper uses a two-layer MLP for projection.
        self.proj = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(C, out_channels, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        training_noise: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, in_channels, Nx, Ny)
        training_noise : bool
            If True, inject HF-adaptive Gaussian noise into the input during
            training (Appendix E). Should only be True during training.

        Returns
        -------
        (B, out_channels, Nx, Ny)
        """
        # ── Extract HFP features before lifting ────────────────────────────────
        x_hf = self.hfp(x)   # H = X − smooth(X), shape (B, in_channels, Nx, Ny)

        # Optional: HF-adaptive Gaussian noise injection (Appendix E)
        # N_dynamic = μ + α·σ·N(0,1), added to input during training.
        if training_noise and self.training and self.hfp_noise_alpha > 0.0:
            # Per-sample mean and std of HF features (over spatial + channel dims)
            mu  = x_hf.mean(dim=(1, 2, 3), keepdim=True)
            std = x_hf.std(dim=(1, 2, 3), keepdim=True) + 1e-8
            noise = mu + self.hfp_noise_alpha * std * torch.randn_like(x)
            x = x + noise

        # ── Lifting ────────────────────────────────────────────────────────────
        z       = self.lift(x)     # (B, C, Nx, Ny)
        z_prime = self.lift(x_hf)  # shared P: same weights for HFP stream

        # ── LOGLO Fourier layers ───────────────────────────────────────────────
        for i, layer in enumerate(self.layers):
            is_last = (i == self.n_layers - 1)
            z, z_prime = layer(z, z_prime, apply_activation=not is_last)

        # ── Projection ─────────────────────────────────────────────────────────
        return self.proj(z)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-aware loss  (§3.2, Appendix D.2)
# ─────────────────────────────────────────────────────────────────────────────


def radially_binned_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    i_low: int = 4,
    i_high: int = 12,
    penalise_bands: tuple[bool, bool, bool] = (False, True, True),
) -> torch.Tensor:
    """
    Frequency-aware loss from §3.2 and Appendix D.2.

    Computes radially-binned spectral energy errors over the 2D spatial axes,
    then classifies them into low / mid / high frequency bands.

    The paper's training objective penalises only mid- and high-frequency errors:
      L_freq = L_mid + L_high   (Eq. 5, λ weighted by the caller)

    Parameters
    ----------
    pred, target : (B, C, Nx, Ny, T) or (B, C, Nx, Ny)
        If 5D, the T axis is treated as an extra batch dimension (per-step loss).
    i_low : int
        Radial bin index separating low from mid frequencies.  Paper: 4.
    i_high : int
        Radial bin index separating mid from high frequencies. Paper: 12.
    penalise_bands : (bool, bool, bool)
        Which of (low, mid, high) bands to include in the returned loss.
        Paper default: (False, True, True).

    Returns
    -------
    loss : scalar tensor
    """
    # Accept either (B, C, Nx, Ny) or (B, C, Nx, Ny, T)
    if pred.dim() == 5:
        B, C, Nx, Ny, T = pred.shape
        pred   = pred.permute(0, 4, 1, 2, 3).reshape(B * T, C, Nx, Ny)
        target = target.permute(0, 4, 1, 2, 3).reshape(B * T, C, Nx, Ny)
    elif pred.dim() == 4:
        B, C, Nx, Ny = pred.shape
    else:
        raise ValueError(f"pred must be 4D or 5D, got {pred.dim()}D")

    nb = pred.shape[0]

    # Step 1: FFT over spatial axes; take positive-quadrant only (Appendix D.2)
    err_fft_sq = torch.abs(
        torch.fft.fftn(pred - target, dim=[-2, -1])
    ) ** 2                                                # (nb, nc, nx, ny)
    # Restrict to first quadrant (DC + positive freqs)
    err_fft_sq_h = err_fft_sq[..., :Nx // 2, :Ny // 2]  # (nb, nc, nx//2, ny//2)

    # Step 2: Radial distance grid (integer bins)
    x_idx = torch.arange(Nx // 2, device=pred.device)
    y_idx = torch.arange(Ny // 2, device=pred.device)
    X, Y  = torch.meshgrid(x_idx, y_idx, indexing='ij')
    radii = (X.float() ** 2 + Y.float() ** 2).sqrt().floor().long()  # (Nx//2, Ny//2)
    max_radius = int(radii.max().item())

    # Flatten spatial → (nb, nc, Nx//2 * Ny//2)
    radii_flat       = radii.flatten()
    err_fft_sq_flat  = err_fft_sq_h.reshape(nb, -1, Nx // 2 * Ny // 2)

    # Step 3: Accumulate spectral error per radial bin → (nb, nc, max_radius+1)
    err_F = torch.zeros(
        nb, err_fft_sq_flat.shape[1], max_radius + 1, device=pred.device
    )
    valid = radii_flat <= max_radius
    err_F.index_add_(2, radii_flat[valid], err_fft_sq_flat[:, :, valid])

    # Step 4: Normalise + mean over batch → (nc, max_radius+1)
    nrm   = (Nx * Ny)
    err_F = (err_F.mean(dim=0) / nrm).sqrt()   # (nc, max_radius+1)

    # Step 5: Band-classify
    def _band_mean(start, end):
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


class LOGLOLoss(nn.Module):
    """
    Combined training loss: MSE + λ · L_freq  (Eq. 5).

    The paper penalises only mid- and high-frequency bands (§4).

    Parameters
    ----------
    lam : float
        Weight λ of the frequency loss. Default 1.0 (equal weighting with MSE).
        Tune down (0.1–0.5) if frequency loss dominates early training.
    i_low, i_high : int
        Radial bin boundaries. Paper uses 4 and 12 for 64×64 Kolmogorov grid.
        Scale proportionally for other resolutions: e.g. 128×128 → i_low=8, i_high=24.
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
        """
        Returns (total_loss, breakdown_dict).

        breakdown keys: 'mse', 'freq', 'total'
        """
        l_mse  = F.mse_loss(pred, target)
        l_freq = radially_binned_spectral_loss(
            pred, target,
            i_low=self.i_low,
            i_high=self.i_high,
            penalise_bands=(False, True, True),   # mid + high only
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

    # ── Paper's Kolmogorov Flow configuration (reduced for CPU test) ──────────
    # Full config: Nx=Ny=64, hidden=65, modes=40, patch=16×16
    # Reduced: Nx=Ny=32, hidden=16, modes=8 (same ratios)
    Nx, Ny     = 32, 32
    patch_size = (16, 16)     # M = (32/16)² = 4 patches
    model = LOGLOFNO2D(
        in_channels=1,
        out_channels=1,
        hidden_channels=16,
        n_modes_global=(8, 8),
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
    print()

    # ── Forward pass ──────────────────────────────────────────────────────────
    x = torch.randn(2, 1, Nx, Ny, device=device)
    y = model(x, training_noise=True)
    print(f'Input  : {tuple(x.shape)}')
    print(f'Output : {tuple(y.shape)}')
    assert y.shape == x.shape, 'Shape mismatch'
    print('Shape OK ✓\n')

    # ── Loss ──────────────────────────────────────────────────────────────────
    target  = torch.randn_like(y)
    loss_fn = LOGLOLoss(lam=1.0, i_low=4, i_high=12)
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

    # ── Patch sizes ───────────────────────────────────────────────────────────
    print()
    print('Local modes (Px, Py//2+1):',
          (patch_size[0], patch_size[1] // 2 + 1),
          '← paper: (16, 9) for patch 16×16')

    # ── Autoregressive rollout example (2 steps) ──────────────────────────────
    print('\n── Autoregressive rollout (2 steps) ───────────────────────────')
    model.eval()
    with torch.no_grad():
        state = x[:1]   # (1, 1, Nx, Ny)
        for step in range(2):
            state = model(state, training_noise=False)
            print(f'  Step {step+1}: {tuple(state.shape)}')

    print('\nAll checks passed. ✓')
