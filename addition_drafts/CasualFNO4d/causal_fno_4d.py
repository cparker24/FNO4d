"""
causal_fno_4d.py
================

A 4D Fourier Neural Operator with structurally causal time axis.

The standard FNO applies a single FFT-based spectral convolution jointly over
all four axes (X, Y, Z, T). That implicitly assumes periodicity and global
two-sided receptive field along T, neither of which fits an initial-value
problem like viscous relativistic hydrodynamics, where:

  - the time axis has a single inflow boundary (initial condition at tau_0),
  - causality forbids information flow from later tau to earlier tau,
  - statistics along T are highly non-stationary (early-time gradients are
    much larger than late-time gradients), so a single set of spectral weights
    along T has to compromise between regimes.

This module replaces the joint 4D spectral convolution with:

  - a 3D spectral convolution over (X, Y, Z), applied at each tau independently,
    keeping the global spatial receptive field and the Tucker factorization, and
  - an explicit dilated causal Conv1d along T, applied at each spatial point.

The causality is enforced structurally: the temporal Conv1d uses padding=0 with
an explicit LEFT-only zero pad of (K-1)*d samples, so the output at tau depends
only on inputs at tau' <= tau. No training or regularization is required to make
the model causal.

With kernel size K=2 and a doubling dilation schedule (1, 2, 4, ...),
L layers give a causal receptive field of (2^L - 1) timesteps:

    L=4  :  receptive field 15 tau-steps
    L=6  :  receptive field 63 tau-steps
    L=7  :  receptive field 127 tau-steps   <-- covers T=128
    L=8  :  receptive field 255 tau-steps   <-- covers T=150 with margin

Tensor layout throughout: (B, C, X, Y, Z, T).

Usage
-----
    from causal_fno_4d import CausalFNO4D

    model = CausalFNO4D(
        in_channels=8,
        out_channels=1,
        hidden_channels=32,
        n_modes_spatial=(8, 8, 8),    # NOTE: 3-tuple, not 4-tuple
        n_layers=7,
        t_kernel_size=2,
        rank=0.05,
    ).cuda()

    x = torch.randn(1, 8, 64, 64, 32, 128, device="cuda")
    y = model(x)                       # (1, 1, 64, 64, 32, 128)

To swap into mg_tfno_4d_fsdp.py, replace the TFNO(...) construction with
CausalFNO4D(...) and drop the T entry from the n_modes tuple. The patching,
multigrid prefiltering, and FSDP wrapping with `ignored_modules` for complex
Tucker parameters all continue to work unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise channel MLP (lift / projection)
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelMLP(nn.Module):
    """
    Pointwise MLP over the channel axis, layout-preserving for (B, C, X, Y, Z, T).

    Equivalent to a 1x1x1x1 convolution implemented as nn.Linear over channels.
    Faster than Conv4d for pure channel mixing because there is no spatial
    kernel to slide.
    """

    def __init__(self, in_ch: int, out_ch: int, n_hidden: int = 0):
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(n_hidden):
            layers += [nn.Linear(in_ch, in_ch), nn.GELU()]
        layers.append(nn.Linear(in_ch, out_ch))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, X, Y, Z, T) -> (B, X, Y, Z, T, C) -> Linear -> (B, C', X, Y, Z, T)
        return self.mlp(x.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Causal FNO block
# ─────────────────────────────────────────────────────────────────────────────


class CausalTimeFNOBlock(nn.Module):
    """
    One FNO block with global spatial receptive field and causal temporal
    receptive field.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths.
    n_modes_spatial : tuple of int, length 3
        (m_x, m_y, m_z) -- spectral modes for the spatial FFT.
    t_kernel_size : int
        Kernel size of the causal temporal convolution (K).
    t_dilation : int
        Dilation of the causal temporal convolution (d). The temporal
        receptive field of this block alone is 1 + (K-1)*d timesteps.
    factorization : str
        Tensor factorization for the SpectralConv weights ("tucker", "cp",
        "tt", or None). Tucker matches the existing MG-TFNO setup.
    rank : float
        Compression rank for the factorization (e.g. 0.05 = 5% of full rank).

    Input / output shape
    --------------------
    (B, C, X, Y, Z, T)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_spatial: tuple[int, int, int],
        t_kernel_size: int = 2,
        t_dilation: int = 1,
        factorization: str = "tucker",
        rank: float = 0.05,
    ):
        super().__init__()
        if len(n_modes_spatial) != 3:
            raise ValueError(
                f"n_modes_spatial must be a 3-tuple (m_x, m_y, m_z); "
                f"got {n_modes_spatial!r}. The temporal axis is handled "
                f"by a Conv1d, not by a spectral mode."
            )
        self.t_kernel_size = t_kernel_size
        self.t_dilation = t_dilation

        # 3D spectral convolution applied per-timestep.
        # The Tucker factorization (and its complex parameters) live here.
        self.spatial_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes_spatial,
            factorization=factorization,
            rank=rank,
            implementation="factorized",
        )
        # Standard FNO linear "W" skip in physical space.
        self.spatial_skip = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        # Causal 1D temporal convolution applied per-spatial-point.
        # padding=0 here; the causal LEFT pad is applied explicitly in forward().
        self.temporal_conv = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=t_kernel_size,
            dilation=t_dilation,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, X, Y, Z, T = x.shape

        # ── 1. Spatial spectral convolution at every tau ─────────────────────
        # Fold T into the batch axis so SpectralConv sees a 3D tensor:
        # (B, C, X, Y, Z, T)  ->  (B*T, C, X, Y, Z)
        x_s = x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, -1, X, Y, Z)
        y_s = self.spatial_conv(x_s) + self.spatial_skip(x_s)
        y_s = F.gelu(y_s)
        C_out = y_s.shape[1]
        # (B*T, C', X, Y, Z)  ->  (B, C', X, Y, Z, T)
        y_s = y_s.reshape(B, T, C_out, X, Y, Z).permute(0, 2, 3, 4, 5, 1)

        # ── 2. Causal 1D temporal convolution at every spatial point ─────────
        # (B, C', X, Y, Z, T)  ->  (B*X*Y*Z, C', T)
        y_t = y_s.permute(0, 2, 3, 4, 1, 5).reshape(B * X * Y * Z, C_out, T)
        # Left-only pad of (K-1)*d zeros: this is what enforces causality
        # AND preserves the length T at the output.
        pad_left = (self.t_kernel_size - 1) * self.t_dilation
        y_t = F.pad(y_t, (pad_left, 0))      # pad LEFT only along last dim
        y_t = self.temporal_conv(y_t)        # output length == T
        # (B*X*Y*Z, C', T)  ->  (B, C', X, Y, Z, T)
        return y_t.reshape(B, X, Y, Z, C_out, T).permute(0, 4, 1, 2, 3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────


class CausalFNO4D(nn.Module):
    """
    4D Fourier Neural Operator with structurally causal time axis.

    Drop-in replacement for TFNO from neuralop.models, with two differences:

      1. n_modes_spatial is a 3-tuple over (X, Y, Z) only; the T axis is
         handled by dilated causal Conv1d layers, not by spectral modes.
      2. The temporal receptive field is controlled by n_layers, t_kernel_size,
         and dilation_schedule rather than by an n_modes entry for T.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths of the model input / output.
    hidden_channels : int
        Width of the latent representation between lift and projection.
    n_modes_spatial : tuple of int, length 3
        Spectral modes used along (X, Y, Z) in every block.
    n_layers : int
        Number of CausalTimeFNOBlocks stacked in the trunk.
    t_kernel_size : int
        Kernel size of the causal temporal convolution in every block (K).
    dilation_schedule : str
        "double"   -- dilations are 1, 2, 4, 8, ..., 2^(n_layers - 1).
                       Receptive field along T is 2^n_layers - 1 timesteps.
        "constant" -- dilations are 1 everywhere.
                       Receptive field along T is n_layers * (K - 1) + 1.
    factorization : str
        Forwarded to SpectralConv ("tucker", "cp", "tt", or None).
    rank : float
        Compression rank for the SpectralConv factorization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes_spatial: tuple[int, int, int],
        n_layers: int = 7,
        t_kernel_size: int = 2,
        dilation_schedule: str = "double",
        factorization: str = "tucker",
        rank: float = 0.05,
    ):
        super().__init__()
        if dilation_schedule == "double":
            dilations = [2 ** layer for layer in range(n_layers)]
        elif dilation_schedule == "constant":
            dilations = [1] * n_layers
        else:
            raise ValueError(
                f"dilation_schedule must be 'double' or 'constant', "
                f"got {dilation_schedule!r}"
            )

        self.lift = _ChannelMLP(in_channels, hidden_channels)
        self.proj = _ChannelMLP(hidden_channels, out_channels, n_hidden=1)
        self.blocks = nn.ModuleList(
            [
                CausalTimeFNOBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_modes_spatial=n_modes_spatial,
                    t_kernel_size=t_kernel_size,
                    t_dilation=d,
                    factorization=factorization,
                    rank=rank,
                )
                for d in dilations
            ]
        )

        # For reporting / introspection
        self.temporal_receptive_field = 1 + sum((t_kernel_size - 1) * d for d in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x) + x                  # outer residual
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Causality verification
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_causality(
    model: nn.Module,
    shape: tuple[int, int, int, int, int, int] = (1, 4, 8, 8, 8, 32),
    perturbation_step: int | None = None,
    tol: float = 1e-6,
    verbose: bool = True,
) -> bool:
    """
    Confirm that the model's output at tau is unaffected by changes in the
    input at any tau' > tau.

    The check perturbs the input at a single timestep and verifies that the
    change in output is exactly zero (to numerical tolerance) for every
    earlier timestep. Because causality is enforced by left-only padding in
    Conv1d, the violation should be at machine precision, not just small.

    Parameters
    ----------
    model : nn.Module
        Any model with input/output shape (B, C, X, Y, Z, T).
    shape : tuple
        Test input shape (kept small by default for speed).
    perturbation_step : int, optional
        The tau at which to perturb the input. Defaults to T // 2.
    tol : float
        Tolerance for the "should be zero" check.
    verbose : bool
        Print the per-timestep change.

    Returns
    -------
    bool : True if causal, False otherwise.
    """
    model.eval()
    B, C, X, Y, Z, T = shape
    if perturbation_step is None:
        perturbation_step = T // 2

    x = torch.randn(B, C, X, Y, Z, T, device=next(model.parameters()).device)
    y_baseline = model(x)

    x_perturbed = x.clone()
    x_perturbed[..., perturbation_step] += 1.0
    y_perturbed = model(x_perturbed)

    diff = (y_perturbed - y_baseline).abs().mean(dim=(0, 1, 2, 3, 4))   # length T
    max_before = diff[:perturbation_step].max().item() if perturbation_step > 0 else 0.0
    max_after = diff[perturbation_step:].max().item()
    is_causal = max_before <= tol

    if verbose:
        print(f"Causality check  (perturbed input at tau = {perturbation_step}/{T})")
        print(f"  max |delta y| at tau <  {perturbation_step}: {max_before:.3e}   "
              f"(must be 0 within {tol:.0e})")
        print(f"  max |delta y| at tau >= {perturbation_step}: {max_after:.3e}   "
              f"(should be > 0)")
        print(f"  -> causal: {is_causal}")
    return is_causal


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Small CPU smoke test: build model, check shape, verify causality.
    torch.manual_seed(0)
    model = CausalFNO4D(
        in_channels=4,
        out_channels=1,
        hidden_channels=16,
        n_modes_spatial=(4, 4, 4),
        n_layers=5,
        t_kernel_size=2,
        dilation_schedule="double",
        rank=0.1,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters       : {n_params:,}")
    print(f"Temporal receptive field: {model.temporal_receptive_field} tau-steps")
    print()

    x = torch.randn(1, 4, 8, 8, 8, 32)
    y = model(x)
    print(f"Input shape  : {tuple(x.shape)}")
    print(f"Output shape : {tuple(y.shape)}")
    print(f"Shape OK     : {x.shape[2:] == y.shape[2:]}")
    print()

    ok = verify_causality(model, shape=(1, 4, 8, 8, 8, 32))
    assert ok, "Causality check FAILED"
    print()
    print("All checks passed.")
