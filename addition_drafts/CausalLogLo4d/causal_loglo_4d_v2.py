"""
causal_loglo_4d.py
==================
CausalLogLo: LOGLO-FNO + Causal temporal factorization + τ-channel embedding
for 3+1D PDE surrogate modeling (e.g. viscous relativistic hydrodynamics).

Combines three orthogonal ideas, each addressing a distinct failure mode of
standard 4D FNO:

  (1) LOGLO local spectral branch  [arXiv:2504.04260]
        Global FNO misses high-frequency / local spatial features due to
        spectral bias and explicit mode truncation. A parallel branch applies
        spectral convolutions on non-overlapping spatial patches, retaining
        all modes within each patch at lower FLOP cost than the global branch.

  (2) Causal temporal factorization  [causal_fno_4d.py]
        A joint 4D FFT assumes periodicity and acausal receptive field along τ.
        Instead, T is folded into the batch axis so all spatial FFTs are 3D
        (no τ in the spectral domain), and a dilated causal Conv1d handles
        temporal mixing with a structurally enforced one-sided receptive field.

  (3) τ-channel embedding
        After NormFn3d normalization the dominant amplitude cue that would let
        the network self-locate in τ is removed. A sinusoidal τ-embedding
        (sin/cos of τ/τ_max) is injected at the input lifting stage and as a
        learned per-block additive bias, giving each spatial convolution
        explicit knowledge of the current proper-time regime without any
        architectural overhead.

Architecture per CausalLogLoBlock:
  ┌─────────────────────────────────────────────────────────┐
  │  x  (B, C, X, Y, Z, T)                                 │
  │  │                                                      │
  │  ├─ τ-bias injection  (learned proj of sin/cos(τ))      │
  │  │                                                      │
  │  ├─ fold T into batch → (B·T, C, X, Y, Z)              │
  │  │        │                                             │
  │  │        ├─ GLOBAL branch: SpectralConv3D (trunc modes)│
  │  │        │   + Conv3d_1x1 skip                         │
  │  │        │   + GELU + ChannelMLP                       │
  │  │        │   + SoftGating(x_bt)                        │
  │  │        │                                             │
  │  │        ├─ LOCAL branch: patches → SpectralConv3D     │
  │  │        │   (all modes per patch) → reassemble        │
  │  │        │   + Conv3d_1x1 skip                         │
  │  │        │   + GELU + ChannelMLP                       │
  │  │        │   + SoftGating(x_bt)                        │
  │  │        │                                             │
  │  │        └─ HF stream: (x - lowpass(x)) → ChannelMLP  │
  │  │                                                      │
  │  ├─ sum branches  →  GELU  →  unfold T                  │
  │  │                                                      │
  │  └─ causal dilated Conv1d along T  (left-pad only)      │
  │                                                          │
  │  + outer residual connection                             │
  └─────────────────────────────────────────────────────────┘

Tensor layout throughout: (B, C, X, Y, Z, T).

FSDP compatibility:
  Complex parameters live only inside SpectralConv instances.
  Use the existing wrap_tfno_with_fsdp() helper from mg_tfno_4d_fsdp.py
  (ignored_modules for complex params + all_reduce gradient hooks).
  All other parameters (Conv3d, Conv1d, Linear) are real → normal FSDP sharding.

Gradient checkpointing:
  Pass use_checkpoint=True to CausalLogLo.__init__.
  Each block's forward is checkpointed: activations are recomputed on the
  backward pass, reducing peak memory by ~60-70% at the cost of one extra
  forward per block.

Usage
-----
    from causal_loglo_4d import CausalLogLo, verify_causality

    model = CausalLogLo(
        in_channels=4,
        out_channels=1,
        hidden_channels=32,
        n_modes_global=(8, 8, 4),      # (mx, my, mz) truncated global modes
        patch_size=(16, 16, 8),        # local branch spatial patch
        n_layers=7,
        t_kernel_size=2,               # causal Conv1d kernel
        tau_max=12.0,                  # fm/c, used for sin/cos normalisation
        use_checkpoint=True,
    ).cuda()

    x   = torch.randn(1, 4, 64, 64, 32, 128, device='cuda')
    tau = torch.linspace(0.4, 12.0, 128, device='cuda')   # (T,) [fm/c]
    y   = model(x, tau)                                    # (1, 1, 64, 64, 32, 128)

    verify_causality(model, shape=(1, 4, 8, 8, 4, 16), tau_max=12.0)

References
----------
  LOGLO-FNO : Kalimuthu, Holzmüller, Niepert — TMLR 2025, arXiv:2504.04260
  Causal FNO: causal_fno_4d.py  (this project)
  NormFn3d  : NormFn3d.py       (this project)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from neuralop.layers.spectral_convolution import SpectralConv


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelMLP(nn.Module):
    """
    Two-layer pointwise MLP over the channel axis, operating on
    (B·T, C, X, Y, Z) via 1×1×1 convolutions.

    This is the "ChannelMLP" in the LOGLO Fourier layer equation.
    """

    def __init__(self, channels: int, expansion: int = 1):
        super().__init__()
        hidden = max(channels, channels * expansion)
        self.net = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BT, C, X, Y, Z)
        return self.net(x)


class _SoftGating(nn.Module):
    """
    Learnable channel-wise affine (scale + bias) applied to (BT, C, X, Y, Z).

    Corresponds to the "SoftGating" in the LOGLO Fourier layer equation:
    a lightweight gating that lets each channel independently scale the skip.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.bias  = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class _TauBias(nn.Module):
    """
    Per-block τ-embedding injection.

    Projects the 2-channel sinusoidal embedding (sin, cos) to a bias tensor
    of shape (1, C, 1, 1, 1, T) that is broadcast-added to the block input.

    This is the *per-block* component of the τ-embedding: it keeps the
    positional signal fresh across layers, preventing it from decaying
    after many channel-mixing operations.

    Parameters
    ----------
    hidden_channels : int
        Width of the hidden representation C.
    """

    def __init__(self, hidden_channels: int):
        super().__init__()
        # bias=False: the τ-embedding itself already carries absolute position.
        self.proj = nn.Linear(2, hidden_channels, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, tau_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        x       : (B, C, X, Y, Z, T)
        tau_emb : (T, 2)  — output of sinusoidal_tau_embedding()
        returns : (B, C, X, Y, Z, T)  — x + per-τ channel bias
        """
        # (T, 2) → (T, C) → (1, C, 1, 1, 1, T)
        bias = self.proj(tau_emb).T.contiguous().view(1, -1, 1, 1, 1, tau_emb.shape[0])
        return x + bias


def sinusoidal_tau_embedding(tau: torch.Tensor, tau_max: float) -> torch.Tensor:
    """
    Compute a 2-channel sinusoidal embedding of the proper-time axis.

    The embedding is sin(π·τ/τ_max) and cos(π·τ/τ_max), giving a unique
    pair of values for every τ ∈ [0, τ_max] that varies smoothly with time
    and is bounded in [-1, 1].

    Parameters
    ----------
    tau     : (T,)  physical proper-time values [fm/c]
    tau_max : float reference scale [fm/c]

    Returns
    -------
    emb : (T, 2)
    """
    t = tau / tau_max
    return torch.stack([torch.sin(math.pi * t), torch.cos(math.pi * t)], dim=-1)


def _spectral_lowpass(
    x: torch.Tensor,
    n_modes: tuple[int, int, int],
) -> torch.Tensor:
    """
    Remove all Fourier modes above the given cutoffs from x.

    Used to split x into a low-frequency part (retained by the global
    spectral branch) and a high-frequency residual (fed to the HF stream).

    Parameters
    ----------
    x       : (BT, C, X, Y, Z)  real tensor
    n_modes : (mx, my, mz)  global spectral mode cutoffs

    Returns
    -------
    x_low : same shape as x, high modes zeroed.
    """
    mx, my, mz = n_modes
    X, Y, Z = x.shape[-3], x.shape[-2], x.shape[-1]

    xf = torch.fft.rfftn(x, dim=(-3, -2, -1))
    # rfft output shape: (..., X, Y, Z//2+1)

    # Clamp cutoffs to valid ranges
    mz = min(mz, xf.shape[-1])          # rfft dim: only positive freqs
    my = min(my, Y // 2 + 1)            # full FFT dim: keep symmetric pair
    mx = min(mx, X // 2 + 1)

    mask = torch.zeros_like(xf, dtype=torch.bool)
    # Low-freq corner (4 quadrants in X×Y, one-sided in Z)
    mask[...,  :mx,  :my, :mz] = True
    mask[..., -mx:,  :my, :mz] = True
    mask[...,  :mx, -my:, :mz] = True
    mask[..., -mx:, -my:, :mz] = True

    return torch.fft.irfftn(xf * mask, s=(X, Y, Z), dim=(-3, -2, -1))


def _apply_local_spectral(
    conv: SpectralConv,
    skip_conv: nn.Conv3d,
    x_bt: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Apply a SpectralConv on non-overlapping spatial patches, then reassemble.

    Within each patch all Fourier modes are retained (no truncation), which
    allows the local branch to resolve high-frequency spatial features that
    the global branch's truncated-mode convolution misses.

    Parameters
    ----------
    conv       : SpectralConv3D  operating on (N, C, Px, Py, Pz)
    skip_conv  : Conv3d 1×1 skip, same shape convention
    x_bt       : (BT, C, X, Y, Z)  — T already folded into batch
    patch_size : (Px, Py, Pz)

    Returns
    -------
    y : (BT, C, X, Y, Z)  — branch output before gating/MLP
    """
    BT, C, X, Y, Z = x_bt.shape
    Px, Py, Pz = patch_size
    Mx, My, Mz = X // Px, Y // Py, Z // Pz
    M = Mx * My * Mz

    # ── Split into patches ────────────────────────────────────────────────
    # (BT, C, X, Y, Z)
    # → view  (BT, C, Mx, Px, My, Py, Mz, Pz)
    # → perm  (BT, Mx, My, Mz, C, Px, Py, Pz)   indices: 0,2,4,6,1,3,5,7
    # → reshape  (BT·M, C, Px, Py, Pz)
    patches = (
        x_bt
        .view(BT, C, Mx, Px, My, Py, Mz, Pz)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(BT * M, C, Px, Py, Pz)
    )

    # ── Spectral conv + 1×1 skip inside each patch ────────────────────────
    y_patches = F.gelu(conv(patches) + skip_conv(patches))

    # ── Reassemble patches ────────────────────────────────────────────────
    # (BT·M, C, Px, Py, Pz)
    # → view   (BT, Mx, My, Mz, C, Px, Py, Pz)
    # → perm   (BT, C, Mx, Px, My, Py, Mz, Pz)   indices: 0,4,1,5,2,6,3,7
    # → view   (BT, C, X, Y, Z)
    y = (
        y_patches
        .view(BT, Mx, My, Mz, C, Px, Py, Pz)
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .contiguous()
        .view(BT, C, X, Y, Z)
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main block
# ─────────────────────────────────────────────────────────────────────────────


class CausalLogLoBlock(nn.Module):
    """
    One CausalLogLo block.

    Implements the combined LOGLO Fourier layer (global + local + HF branches)
    with causal dilated Conv1d for temporal mixing and sinusoidal τ-bias
    injection for proper-time conditioning.

    Parameters
    ----------
    channels : int
        Hidden channel width C (input = output = C).
    n_modes_global : (mx, my, mz)
        Spectral modes retained in the global branch (truncated).
    patch_size : (Px, Py, Pz)
        Spatial patch dimensions for the local branch.
        Must divide the grid evenly: X % Px == 0, etc.
    t_kernel_size : int
        Kernel size K for the causal dilated Conv1d.
    t_dilation : int
        Dilation d for the causal Conv1d.
        Causal temporal receptive field of this block: 1 + (K-1)·d.
    tau_max : float
        Reference proper time [fm/c] for the sinusoidal τ-embedding.
    factorization : str
        Tucker / CP / TT or None — forwarded to SpectralConv.
    rank : float
        Compression rank fraction for SpectralConv (e.g. 0.05).
    """

    def __init__(
        self,
        channels: int,
        n_modes_global: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        t_kernel_size: int = 2,
        t_dilation: int = 1,
        tau_max: float = 12.0,
        factorization: str = "tucker",
        rank: float = 0.05,
    ):
        super().__init__()
        C = channels
        Px, Py, Pz = patch_size
        # Local branch retains ALL Fourier modes within each patch.
        # SpectralConv applies rfftn along the LAST spatial dim (Z here), so:
        #   X, Y axes (full FFT): keep Px//2 positive + Px//2 negative = Px modes
        #   Z axis   (rfft):      keep Pz//2+1 unique coefficients (DC→Nyquist)
        # The +1 on the rfft dimension includes the Nyquist frequency, matching the
        # paper's 2D convention n_modes_local = (Px, Py//2+1) for patch (Px, Py).
        n_modes_local = (Px // 2, Py // 2, Pz // 2 + 1)

        self.n_modes_global = n_modes_global
        self.patch_size = patch_size
        self.t_kernel_size = t_kernel_size
        self.t_dilation = t_dilation

        # ── τ-bias (per-block injection) ──────────────────────────────────
        self.tau_bias = _TauBias(C)

        # ── Global branch ─────────────────────────────────────────────────
        self.global_conv = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_global,
            factorization=factorization, rank=rank,
            implementation="factorized",
        )
        self.global_skip  = nn.Conv3d(C, C, kernel_size=1)
        self.global_mlp   = _ChannelMLP(C)
        self.global_gate  = _SoftGating(C)

        # ── Local branch ──────────────────────────────────────────────────
        self.local_conv = SpectralConv(
            in_channels=C, out_channels=C,
            n_modes=n_modes_local,
            factorization=factorization, rank=rank,
            implementation="factorized",
        )
        self.local_skip  = nn.Conv3d(C, C, kernel_size=1)
        self.local_mlp   = _ChannelMLP(C)
        self.local_gate  = _SoftGating(C)

        # ── HF propagation stream ─────────────────────────────────────────
        # Lifts the high-frequency residual (x - lowpass(x)) to channels.
        self.hf_mlp = _ChannelMLP(C)

        # ── Causal temporal Conv1d ─────────────────────────────────────────
        # padding=0: causality enforced by explicit left-only pad in forward().
        self.temporal_conv = nn.Conv1d(
            in_channels=C, out_channels=C,
            kernel_size=t_kernel_size,
            dilation=t_dilation,
            padding=0,
            bias=True,
        )

    def _spatial_forward(
        self, x_bt: torch.Tensor, apply_activation: bool = True
    ) -> torch.Tensor:
        """
        Compute the three LOGLO branches on (BT, C, X, Y, Z).

        apply_activation : bool
            If False, the final GELU on the branch sum is skipped.
            The paper (arXiv:2504.04260, §2 footnote 1) skips the GELU
            non-linearity on the last Fourier layer.

        Separated from forward() so gradient checkpointing can wrap it.
        """
        # Global branch
        y_glob = F.gelu(self.global_conv(x_bt) + self.global_skip(x_bt))
        y_glob = self.global_mlp(y_glob) + self.global_gate(x_bt)

        # Local branch (patch-wise spectral conv, all modes retained)
        y_loc = _apply_local_spectral(
            self.local_conv, self.local_skip, x_bt, self.patch_size
        )
        y_loc = self.local_mlp(y_loc) + self.local_gate(x_bt)

        # HF stream: inject high-frequency residual not captured by global branch
        x_hf = x_bt - _spectral_lowpass(x_bt, self.n_modes_global)
        y_hf = self.hf_mlp(x_hf)

        out = y_glob + y_loc + y_hf
        return F.gelu(out) if apply_activation else out

    def forward(
        self,
        x: torch.Tensor,           # (B, C, X, Y, Z, T)
        tau_emb: torch.Tensor,     # (T, 2)
        use_checkpoint: bool = False,
        apply_activation: bool = True,
    ) -> torch.Tensor:
        """
        Returns (B, C, X, Y, Z, T) — outer residual applied by the caller.

        apply_activation : bool
            Forwarded to _spatial_forward. Set False for the last block
            (paper footnote 1: GELU skipped on last Fourier layer).
        """
        B, C, X, Y, Z, T = x.shape

        # ── (1) τ-bias injection ──────────────────────────────────────────
        x = self.tau_bias(x, tau_emb)

        # ── (2) Fold T into batch ─────────────────────────────────────────
        # (B, C, X, Y, Z, T) → (B·T, C, X, Y, Z)
        x_bt = x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)

        # ── (3) LOGLO spatial branches ────────────────────────────────────
        if use_checkpoint:
            # torch.utils.checkpoint cannot pass keyword-only args cleanly;
            # wrap in a lambda to forward apply_activation into the function.
            y_bt = grad_ckpt(
                lambda inp: self._spatial_forward(inp, apply_activation),
                x_bt, use_reentrant=False,
            )
        else:
            y_bt = self._spatial_forward(x_bt, apply_activation)

        # ── (4) Unfold T ──────────────────────────────────────────────────
        # (B·T, C, X, Y, Z) → (B, C, X, Y, Z, T)
        y = y_bt.view(B, T, C, X, Y, Z).permute(0, 2, 3, 4, 5, 1)

        # ── (5) Causal dilated Conv1d along T ─────────────────────────────
        # Flatten spatial into batch: (B·X·Y·Z, C, T)
        y_t = y.permute(0, 2, 3, 4, 1, 5).reshape(B * X * Y * Z, C, T)
        # Left-only causal padding: output at τ sees only inputs at τ' ≤ τ
        pad_left = (self.t_kernel_size - 1) * self.t_dilation
        y_t = F.pad(y_t, (pad_left, 0))
        y_t = self.temporal_conv(y_t)                          # length stays T
        # (B·X·Y·Z, C, T) → (B, C, X, Y, Z, T)
        y = y_t.reshape(B, X, Y, Z, C, T).permute(0, 4, 1, 2, 3, 5)

        return y


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────


class CausalLogLo(nn.Module):
    """
    Full 3+1D CausalLogLo surrogate model.

    Drop-in replacement for CausalFNO4D / TFNO with three architectural
    enhancements: LOGLO local spectral branch, causal temporal factorization,
    and sinusoidal τ-channel embedding.

    Parameters
    ----------
    in_channels : int
        Number of physical input fields (e.g. 4 for ed, vx, vy, vz).
    out_channels : int
        Number of predicted output fields.
    hidden_channels : int
        Latent width C throughout the network.
    n_modes_global : (mx, my, mz)
        Global spectral branch mode cutoffs.
    patch_size : (Px, Py, Pz)
        Local branch spatial patch.  Grid dimensions must be divisible by patch.
    n_layers : int
        Number of CausalLogLoBlocks stacked in the trunk.
    t_kernel_size : int
        Conv1d kernel size K (default 2 for minimal receptive field per layer).
    dilation_schedule : 'double' | 'constant'
        'double'   — dilations 1, 2, 4, …, 2^(n_layers-1).
                     Causal receptive field: 2^n_layers - 1 τ-steps.
        'constant' — dilations all 1.
                     Causal receptive field: n_layers * (K-1) + 1 τ-steps.
    tau_max : float
        Reference proper time [fm/c] for the sinusoidal embedding.
        Typically the final τ of the hydrodynamic evolution (e.g. 12.0 fm/c).
    factorization : str
        Tensor factorization for SpectralConv weights.
    rank : float
        Tucker/CP/TT compression rank fraction.
    use_checkpoint : bool
        If True, each block's spatial forward is gradient-checkpointed, trading
        ~60% peak activation memory for one extra forward pass per block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_global: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        n_layers: int = 7,
        t_kernel_size: int = 2,
        dilation_schedule: str = "double",
        tau_max: float = 12.0,
        factorization: str = "tucker",
        rank: float = 0.05,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        C = hidden_channels
        self.tau_max = tau_max
        self.use_checkpoint = use_checkpoint

        if dilation_schedule == "double":
            dilations = [2 ** i for i in range(n_layers)]
        elif dilation_schedule == "constant":
            dilations = [1] * n_layers
        else:
            raise ValueError(
                f"dilation_schedule must be 'double' or 'constant', "
                f"got {dilation_schedule!r}"
            )
        # Causal receptive field summary
        self.temporal_receptive_field = 1 + sum(
            (t_kernel_size - 1) * d for d in dilations
        )

        # ── Lifting ───────────────────────────────────────────────────────
        # Accepts in_channels + 2 because the τ-embedding (sin, cos) is
        # concatenated to the physical channels at the input.
        # This gives the network τ-awareness from the very first spectral op.
        self.lift = nn.Conv3d(in_channels + 2, C, kernel_size=1)

        # ── Trunk ─────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            CausalLogLoBlock(
                channels=C,
                n_modes_global=n_modes_global,
                patch_size=patch_size,
                t_kernel_size=t_kernel_size,
                t_dilation=d,
                tau_max=tau_max,
                factorization=factorization,
                rank=rank,
            )
            for d in dilations
        ])

        # ── Projection ────────────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Conv3d(C, C, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(C, out_channels, kernel_size=1),
        )

    def _lift_with_tau(
        self, x: torch.Tensor, tau_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate τ-embedding to input channels and apply lifting MLP.

        x       : (B, C_in, X, Y, Z, T)
        tau_emb : (T, 2)
        returns : (B, C, X, Y, Z, T)
        """
        B, C_in, X, Y, Z, T = x.shape

        # τ-embedding: (T, 2) → (B, 2, X, Y, Z, T)
        emb = tau_emb.T.contiguous().view(1, 2, 1, 1, 1, T)
        emb = emb.expand(B, 2, X, Y, Z, T)

        # Concatenate along channel dim → (B, C_in+2, X, Y, Z, T)
        x = torch.cat([x, emb], dim=1)

        # Fold T into batch for lifting Conv3d
        # (B, C_in+2, X, Y, Z, T) → (B·T, C_in+2, X, Y, Z)
        BT = B * T
        x_bt = x.permute(0, 5, 1, 2, 3, 4).reshape(BT, C_in + 2, X, Y, Z)
        y_bt = self.lift(x_bt)    # (B·T, C, X, Y, Z)

        # Unfold T: (B, C, X, Y, Z, T)
        return y_bt.view(B, T, -1, X, Y, Z).permute(0, 2, 3, 4, 5, 1)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply projection MLP to (B, C, X, Y, Z, T), returning (B, C_out, X, Y, Z, T).
        """
        B, C, X, Y, Z, T = x.shape
        x_bt = x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
        y_bt = self.proj(x_bt)
        C_out = y_bt.shape[1]
        return y_bt.view(B, T, C_out, X, Y, Z).permute(0, 2, 3, 4, 5, 1)

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x   : (B, in_channels, X, Y, Z, T)
        tau : (T,) physical proper-time values [fm/c], optional.
              If None, a uniform grid [0, tau_max] with T steps is assumed.

        Returns
        -------
        y : (B, out_channels, X, Y, Z, T)
        """
        T = x.shape[-1]
        if tau is None:
            tau = torch.linspace(0.0, self.tau_max, T, device=x.device, dtype=x.dtype)

        # Sinusoidal embedding: (T, 2)
        tau_emb = sinusoidal_tau_embedding(tau, self.tau_max).to(dtype=x.dtype)

        # Lifting (includes τ embedding at input)
        h = self._lift_with_tau(x, tau_emb)      # (B, C, X, Y, Z, T)

        # Trunk with outer residual connections.
        # apply_activation=False on the last block: paper footnote 1 skips
        # the GELU non-linearity on the last Fourier layer.
        for i, blk in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            h = h + blk(
                h, tau_emb,
                use_checkpoint=self.use_checkpoint,
                apply_activation=not is_last,
            )

        # Projection
        return self._project(h)                   # (B, out_channels, X, Y, Z, T)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral loss (LOGLO radially-binned variant)
# ─────────────────────────────────────────────────────────────────────────────


def spectral_loss_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_bins: int = 8,
    weight_hf: float = 2.0,
) -> torch.Tensor:
    """
    Radially-binned spectral loss over the 3 spatial dimensions.

    Computes the power-spectrum error in concentric spherical shells in
    (kx, ky, kz) wavenumber space, assigning higher weight to high-frequency
    bins to counteract spectral bias (LOGLO contribution).

    The loss is averaged over time steps and batch, so it is consistent with
    a standard per-step L2 training objective.

    Parameters
    ----------
    pred, target : (B, C, X, Y, Z, T)
    n_bins       : number of radial frequency bins
    weight_hf    : extra weight multiplier applied linearly from bin 0 to n_bins.
                   1.0 = uniform weights; 2.0 = HF bins weighted 2× more.

    Returns
    -------
    loss : scalar tensor
    """
    B, C, X, Y, Z, T = pred.shape
    # Fold T into batch for spectral computation
    p = pred.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
    t = target.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)

    pf = torch.fft.rfftn(p - t, dim=(-3, -2, -1))
    power = (pf.real ** 2 + pf.imag ** 2)   # (BT, C, X, Y, Z//2+1)

    # Build radial wavenumber grid
    kx = torch.fft.fftfreq(X, device=pred.device).view(X, 1, 1)
    ky = torch.fft.fftfreq(Y, device=pred.device).view(1, Y, 1)
    kz = torch.fft.rfftfreq(Z, device=pred.device).view(1, 1, Z // 2 + 1)
    k_rad = torch.sqrt(kx ** 2 + ky ** 2 + kz ** 2)  # (X, Y, Z//2+1)

    k_max = k_rad.max().item()
    bin_edges = torch.linspace(0.0, k_max + 1e-6, n_bins + 1, device=pred.device)

    loss = torch.tensor(0.0, device=pred.device)
    for i in range(n_bins):
        mask = (k_rad >= bin_edges[i]) & (k_rad < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        # Linear weight: low-freq bins weight 1.0, high-freq bins weight weight_hf
        w = 1.0 + (weight_hf - 1.0) * (i / max(n_bins - 1, 1))
        loss = loss + w * power[..., mask].mean()

    return loss / n_bins


# ─────────────────────────────────────────────────────────────────────────────
# Causality verification
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_causality(
    model: nn.Module,
    shape: tuple[int, int, int, int, int, int] = (1, 4, 8, 8, 4, 16),
    tau_max: float = 12.0,
    perturbation_step: int | None = None,
    tol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    Verify that the model's output at τ is unaffected by input changes at τ' > τ.

    Perturbs a single τ-slice of the input and checks that the output at all
    earlier τ steps is unchanged to within numerical tolerance.

    Parameters
    ----------
    model            : CausalLogLo (or any (B,C,X,Y,Z,T) → (B,C',X,Y,Z,T) model)
    shape            : (B, C, X, Y, Z, T) test input shape
    tau_max          : float, [fm/c]
    perturbation_step: τ index at which to perturb input (default T // 2)
    tol              : tolerance for the "must be zero" check
    verbose          : print diagnostics

    Returns
    -------
    bool : True if causal, False otherwise.
    """
    model.eval()
    device = next(model.parameters()).device
    B, C, X, Y, Z, T = shape
    if perturbation_step is None:
        perturbation_step = T // 2

    tau = torch.linspace(0.0, tau_max, T, device=device)
    x = torch.randn(B, C, X, Y, Z, T, device=device)

    y_base = model(x, tau)

    x_pert = x.clone()
    x_pert[..., perturbation_step] += 1.0
    y_pert = model(x_pert, tau)

    diff = (y_pert - y_base).abs().mean(dim=(0, 1, 2, 3, 4))   # (T,)
    max_before = diff[:perturbation_step].max().item() if perturbation_step > 0 else 0.0
    max_after  = diff[perturbation_step:].max().item()
    is_causal  = max_before <= tol

    if verbose:
        print(f"Causality check  (perturbed input at τ = {perturbation_step}/{T})")
        print(f"  max |Δy| at τ <  {perturbation_step}: {max_before:.3e}  "
              f"(must be ≤ {tol:.0e})")
        print(f"  max |Δy| at τ >= {perturbation_step}: {max_after:.3e}  "
              f"(should be > 0)")
        print(f"  → causal: {is_causal}")
    return is_causal


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build model ───────────────────────────────────────────────────────
    # Smoke-test grid: 8×8×4×16  (tiny, CPU-safe)
    # Patch: 4×4×4 → 2×2×1 = 4 patches
    # Global modes: (2, 2, 2)
    # n_layers=5, doubling dilations → receptive field 31 τ-steps
    model = CausalLogLo(
        in_channels=4,
        out_channels=1,
        hidden_channels=8,
        n_modes_global=(2, 2, 2),
        patch_size=(4, 4, 4),
        n_layers=5,
        t_kernel_size=2,
        dilation_schedule="double",
        tau_max=12.0,
        factorization="tucker",
        rank=0.5,           # higher rank fraction for tiny test model
        use_checkpoint=False,
    ).to(device)

    n_params  = sum(p.numel() for p in model.parameters())
    n_complex = sum(p.numel() for p in model.parameters() if p.is_complex())
    print(f"Total parameters : {n_params:,}")
    print(f"Complex params   : {n_complex:,}  (SpectralConv Tucker weights)")
    print(f"Real params      : {n_params - n_complex:,}")
    print(f"Causal receptive field: {model.temporal_receptive_field} τ-steps\n")

    # ── Forward pass shape check ──────────────────────────────────────────
    # Grid must be divisible by patch: X%Px==0, Y%Py==0, Z%Pz==0
    x   = torch.randn(1, 4, 8, 8, 4, 16, device=device)
    tau = torch.linspace(0.4, 12.0, 16, device=device)
    y   = model(x, tau)
    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    assert x.shape[2:] == y.shape[2:], "Spatial+T dims must be preserved"
    assert y.shape[1] == 1,            "out_channels=1 check"
    print("Shape OK ✓\n")

    # ── Spectral loss ─────────────────────────────────────────────────────
    y_target = torch.randn_like(y)
    loss_l2  = F.mse_loss(y, y_target)
    loss_sp  = spectral_loss_3d(y, y_target, n_bins=4, weight_hf=2.0)
    print(f"MSE loss     : {loss_l2.item():.4f}")
    print(f"Spectral loss: {loss_sp.item():.4f}\n")

    # ── Backward pass check ───────────────────────────────────────────────
    (loss_l2 + loss_sp).backward()
    grad_norms = {
        n: p.grad.norm().item()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"Gradients flow through {len(grad_norms)} / "
          f"{sum(1 for _ in model.parameters())} parameter tensors ✓\n")

    # ── Causality check ───────────────────────────────────────────────────
    ok = verify_causality(
        model, shape=(1, 4, 8, 8, 4, 16), tau_max=12.0
    )
    assert ok, "Causality check FAILED"
    print("\nAll checks passed. ✓")
