"""
causal_fno_4d.py
================

4D Fourier Neural Operator with structurally causal time axis, plus an
optional τ/η positional-embedding variant for Milne-coordinate hydrodynamics.

The standard FNO applies a single FFT-based spectral convolution jointly over
all four axes (X, Y, Z, T).  That implicitly assumes periodicity and a global
two-sided receptive field along T, neither of which fits an initial-value
problem like viscous relativistic hydrodynamics, where:

  - the time axis has a single inflow boundary (initial condition at τ₀),
  - causality forbids information flow from later τ to earlier τ,
  - statistics along T are highly non-stationary (early-time gradients are
    much larger than late-time gradients), so a single set of spectral weights
    along T has to compromise between regimes.

This module replaces the joint 4D spectral convolution with:

  - a 3D spectral convolution over (X, Y, Z), applied at each τ independently,
    keeping the global spatial receptive field and the Tucker factorization, and
  - an explicit dilated causal Conv1d along T, applied at each spatial point.

Causality is enforced structurally: the temporal Conv1d uses padding=0 with an
explicit LEFT-only zero pad of (K-1)·d samples, so the output at τ depends only
on inputs at τ' ≤ τ.  No training or regularisation is required.

With kernel size K=2 and a doubling dilation schedule (1, 2, 4, …),
L layers give a causal receptive field of 2^L timesteps:

    L=4  :  receptive field  16 τ-steps
    L=5  :  receptive field  32 τ-steps
    L=6  :  receptive field  64 τ-steps
    L=7  :  receptive field 128 τ-steps   <-- covers T=128
    L=8  :  receptive field 256 τ-steps   <-- covers T=150 with margin

Tensor layout throughout: (B, C, X, Y, Z, T).

Two classes are provided:

  CausalFNO4D          — base model (no coordinate embeddings).
  CausalFNO4DTauEta    — adds sinusoidal τ/η positional embeddings using the
                         same two-part injection as FNOTauEta: 4 sinusoidal
                         input channels + per-block TauBias/EtaBias.

Usage
-----
    from neuralop.models import CausalFNO4D, CausalFNO4DTauEta

    # Base model
    model = CausalFNO4D(
        in_channels=4, out_channels=4, hidden_channels=32,
        n_modes_spatial=(8, 8, 8),    # 3-tuple, not 4-tuple
        n_layers=7, t_kernel_size=2, rank=0.05,
    ).cuda()

    # With τ/η embedding
    model = CausalFNO4DTauEta(
        in_channels=4, out_channels=4, hidden_channels=32,
        n_modes_spatial=(8, 8, 8),
        n_layers=7, t_kernel_size=2, rank=0.05,
        tau_max=12.4, eta_max=4.0,
    ).cuda()

    x   = torch.randn(1, 4, 64, 64, 32, 128, device="cuda")
    tau = torch.linspace(0.4, 12.4, 128, device="cuda")   # (T,) [fm/c]
    eta = torch.linspace(-4.0, 4.0, 32,  device="cuda")   # (Z,)

    y = model(x, tau, eta)   # (1, 4, 64, 64, 32, 128)

To swap into train_torch.py set model_type: causal_fno4d or
model_type: causal_fno4d_tau_eta in the YAML config.  The patching, multigrid
prefiltering, and FSDP wrapping all continue to work without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from ..layers.embeddings import (
    TauBias,
    EtaBias,
    sinusoidal_tau_embedding,
    sinusoidal_eta_embedding,
)
from .base_model import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise channel MLP (lift / projection)
# ─────────────────────────────────────────────────────────────────────────────


class _ChannelMLP(nn.Module):
    """Pointwise MLP over the channel axis, layout-preserving for (B, C, X, Y, Z, T).

    Implemented as nn.Linear over the channel axis (equivalent to a 1×1×1×1
    convolution but faster for pure channel mixing with no spatial kernel).
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
    """One FNO block with global spatial receptive field and causal temporal
    receptive field.

    Parameters
    ----------
    in_channels, out_channels : int
        Channel widths.
    n_modes_spatial : tuple of int, length 3
        (m_x, m_y, m_z) — spectral modes for the 3D spatial FFT.
    t_kernel_size : int
        Kernel size of the causal temporal convolution (K).
    t_dilation : int
        Dilation of the causal temporal convolution (d).  The temporal
        receptive field of this single block is 1 + (K-1)·d timesteps.
    factorization : str or None
        Tensor factorisation for the SpectralConv weights ("tucker", "cp",
        "tt", or None).  Tucker matches the existing MG-TFNO setup.
    rank : float
        Compression rank for the factorisation (e.g. 0.05 = 5 % of full rank).

    Input / output shape: (B, C, X, Y, Z, T)
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
                f"got {n_modes_spatial!r}.  The temporal axis is handled "
                f"by a Conv1d, not by a spectral mode."
            )
        self.t_kernel_size = t_kernel_size
        self.t_dilation = t_dilation

        # 3D spectral convolution applied per-timestep (Tucker factorisation
        # and complex weights live here — FSDP discovery is unchanged).
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

        # ── 1. Spatial spectral convolution at every τ ───────────────────────
        # Fold T into the batch axis so SpectralConv sees a 3D tensor:
        # (B, C, X, Y, Z, T)  ->  (B*T, C, X, Y, Z)
        x_s = x.permute(0, 5, 1, 2, 3, 4).reshape(B * T, -1, X, Y, Z)
        y_s = self.spatial_conv(x_s) + self.spatial_skip(x_s)
        y_s = F.gelu(y_s)
        C_out = y_s.shape[1]
        # Store as (B, X, Y, Z, C', T) so the temporal reshape below is a
        # zero-copy view (avoids a duplicate contiguous allocation mid-block).
        y_s = y_s.reshape(B, T, C_out, X, Y, Z).permute(0, 3, 4, 5, 2, 1).contiguous()

        # ── 2. Causal 1D temporal convolution at every spatial point ─────────
        # (B, X, Y, Z, C', T)  ->  (B*X*Y*Z, C', T)  — zero-copy view
        y_t = y_s.reshape(B * X * Y * Z, C_out, T)
        # Left-only pad of (K-1)·d zeros: enforces causality and preserves
        # output length T.
        pad_left = (self.t_kernel_size - 1) * self.t_dilation
        y_t = F.pad(y_t, (pad_left, 0))
        y_t = self.temporal_conv(y_t)   # output length == T
        # (B*X*Y*Z, C', T)  ->  (B, C', X, Y, Z, T)
        # No activation here — GELU lives at the start of the next block's
        # spatial stage (pre-activation residual pattern).
        return y_t.reshape(B, X, Y, Z, C_out, T).permute(0, 4, 1, 2, 3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Full model — base (no coordinate embeddings)
# ─────────────────────────────────────────────────────────────────────────────


class CausalFNO4D(BaseModel, name="CausalFNO4D"):
    """4D Fourier Neural Operator with structurally causal time axis.

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
        Spectral modes along (X, Y, Z) in every block.
    n_layers : int
        Number of CausalTimeFNOBlocks stacked in the trunk.  With K=2 and
        dilation_schedule="double" the temporal receptive field is 2^n_layers.
    t_kernel_size : int
        Kernel size of the causal temporal convolution in every block (K).
    dilation_schedule : str
        "double"   — dilations 1, 2, 4, …, 2^(n_layers-1).  RF = 2^n_layers.
        "constant" — all dilations = 1.  RF = n_layers·(K-1) + 1.
    factorization : str or None
        Forwarded to SpectralConv ("tucker", "cp", "tt", or None).
    rank : float
        Compression rank for the SpectralConv factorisation.
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

        self.temporal_receptive_field = 1 + sum(
            (t_kernel_size - 1) * d for d in dilations
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x) + x   # outer residual
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model — with τ/η positional embeddings
# ─────────────────────────────────────────────────────────────────────────────


class CausalFNO4DTauEta(BaseModel, name="CausalFNO4DTauEta"):
    """CausalFNO4D extended with sinusoidal τ/η positional embeddings.

    Follows the same two-part injection pattern as FNOTauEta:

      1. INPUT CONCATENATION: 4 sinusoidal channels (2 τ + 2 η) are appended
         to the physical input before the lifting MLP, so the first projection
         already sees where it is in (τ, η) space.

      2. PER-BLOCK ADDITIVE BIAS: a small Linear(2→C) refreshes the positional
         signal after every block, preventing it from decaying with depth.

    Parameters
    ----------
    in_channels, out_channels : int
        Physical channel widths (τ/η channels are added internally).
    hidden_channels : int
        FNO hidden width.
    n_modes_spatial : tuple of int, length 3
        Spectral modes along (X, Y, Z).
    n_layers : int
        Number of CausalTimeFNOBlocks.
    t_kernel_size : int
        Causal Conv1d kernel size (K).
    dilation_schedule : str
        "double" or "constant" — see CausalFNO4D.
    factorization : str or None
        Tucker / CP / TT factorisation for SpectralConv weights.
    rank : float
        Compression rank.
    tau_max : float
        Reference proper time [fm/c] for sinusoidal embedding normalisation.
        Also used as the upper bound when tau is not supplied at inference.
    eta_max : float
        |η| range for sinusoidal embedding normalisation.
        Also used as the symmetric bound when eta is not supplied at inference.
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
        tau_max: float = 12.4,
        eta_max: float = 4.0,
    ):
        super().__init__()
        self.tau_max = tau_max
        self.eta_max = eta_max
        self.n_layers = n_layers

        # The inner CausalFNO4D receives 4 extra input channels (2 τ + 2 η).
        self.model = CausalFNO4D(
            in_channels=in_channels + 4,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_modes_spatial=n_modes_spatial,
            n_layers=n_layers,
            t_kernel_size=t_kernel_size,
            dilation_schedule=dilation_schedule,
            factorization=factorization,
            rank=rank,
        )

        self.tau_biases = nn.ModuleList(
            [TauBias(hidden_channels) for _ in range(n_layers)]
        )
        self.eta_biases = nn.ModuleList(
            [EtaBias(hidden_channels) for _ in range(n_layers)]
        )

    @property
    def temporal_receptive_field(self) -> int:
        return self.model.temporal_receptive_field

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x   : (B, C_in, X, Y, Z, T)
        tau : (T,) physical proper-time values [fm/c].
              Defaults to a uniform grid [0, tau_max] when None.
        eta : (Z,) spacetime-rapidity values.
              Defaults to a uniform grid [-eta_max, +eta_max] when None.

        Returns
        -------
        (B, C_out, X, Y, Z, T)
        """
        B, C_in, X, Y, Z, T = x.shape
        device, dtype = x.device, x.dtype

        if tau is None:
            tau = torch.linspace(0.0, self.tau_max, T, device=device, dtype=dtype)
        if eta is None:
            eta = torch.linspace(-self.eta_max, self.eta_max, Z, device=device, dtype=dtype)

        tau_emb = sinusoidal_tau_embedding(tau, self.tau_max).to(dtype=dtype)  # (T, 2)
        eta_emb = sinusoidal_eta_embedding(eta, self.eta_max).to(dtype=dtype)  # (Z, 2)

        # Broadcast embedding channels across the spatial grid.
        tau_ch = tau_emb.T.view(1, 2, 1, 1, 1, T).expand(B, 2, X, Y, Z, T)
        eta_ch = eta_emb.T.view(1, 2, 1, 1, Z, 1).expand(B, 2, X, Y, Z, T)
        x_aug = torch.cat([x, tau_ch, eta_ch], dim=1)   # (B, C_in+4, X, Y, Z, T)

        # Replicate CausalFNO4D.forward() with per-block τ/η bias injection.
        x_aug = self.model.lift(x_aug)
        for i, blk in enumerate(self.model.blocks):
            x_aug = blk(x_aug) + x_aug              # outer residual
            x_aug = self.tau_biases[i](x_aug, tau_emb)
            x_aug = self.eta_biases[i](x_aug, eta_emb)
        return self.model.proj(x_aug)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: coordinate tensors from NormFn3d grid metadata
# ─────────────────────────────────────────────────────────────────────────────


def make_tau_tensor(
    tau_min: float,
    dtau: float,
    n_tau: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Physical τ values (n_tau,) matching NormFn3d grid metadata."""
    return torch.arange(n_tau, device=device, dtype=dtype) * dtau + tau_min


def make_eta_tensor(
    eta_min: float,
    deta: float,
    n_eta: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Physical η values (n_eta,) matching NormFn3d grid metadata."""
    return torch.arange(n_eta, device=device, dtype=dtype) * deta + eta_min


# ─────────────────────────────────────────────────────────────────────────────
# Causality verification utility
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def verify_causality(
    model: nn.Module,
    shape: tuple[int, int, int, int, int, int] = (1, 4, 8, 8, 8, 32),
    perturbation_step: int | None = None,
    tol: float = 1e-6,
    verbose: bool = True,
) -> bool:
    """Confirm that the model's output at τ is unaffected by changes in the
    input at any τ' > τ.

    The check perturbs the input at a single timestep and verifies that the
    change in output is exactly zero (to numerical tolerance) for every earlier
    timestep.  Because causality is enforced by left-only padding in Conv1d,
    the violation should be at machine precision, not just small.

    Parameters
    ----------
    model : nn.Module
        Any model with input/output shape (B, C, X, Y, Z, T).
    shape : tuple
        Test input shape (kept small by default for speed).
    perturbation_step : int, optional
        The τ at which to perturb the input.  Defaults to T // 2.
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

    diff = (y_perturbed - y_baseline).abs().mean(dim=(0, 1, 2, 3, 4))   # (T,)
    max_before = diff[:perturbation_step].max().item() if perturbation_step > 0 else 0.0
    max_after = diff[perturbation_step:].max().item()
    is_causal = max_before <= tol

    if verbose:
        print(f"Causality check  (perturbed input at τ = {perturbation_step}/{T})")
        print(f"  max |Δy| at τ <  {perturbation_step}: {max_before:.3e}   "
              f"(must be 0 within {tol:.0e})")
        print(f"  max |Δy| at τ >= {perturbation_step}: {max_after:.3e}   "
              f"(should be > 0)")
        print(f"  -> causal: {is_causal}")
    return is_causal


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    torch.manual_seed(0)

    # ── Base model ────────────────────────────────────────────────────────────
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
    print(f"CausalFNO4D parameters        : {n_params:,}")
    print(f"Temporal receptive field       : {model.temporal_receptive_field} τ-steps")

    x = torch.randn(1, 4, 8, 8, 8, 32)
    y = model(x)
    assert x.shape[2:] == y.shape[2:], "shape mismatch"
    print(f"Input / output shapes OK       : {tuple(x.shape)} -> {tuple(y.shape)}")

    ok = verify_causality(model, shape=(1, 4, 8, 8, 8, 32))
    assert ok, "Causality check FAILED"
    print()

    # ── RF sanity check ───────────────────────────────────────────────────────
    for L in [4, 5, 6, 7, 8]:
        m = CausalFNO4D(1, 1, 4, (2, 2, 2), n_layers=L, t_kernel_size=2,
                        rank=1.0, factorization=None)
        assert m.temporal_receptive_field == 2 ** L, (
            f"L={L}: expected RF={2**L}, got {m.temporal_receptive_field}"
        )
    print("RF values correct for L=4..8")

    # ── TauEta model ──────────────────────────────────────────────────────────
    model_te = CausalFNO4DTauEta(
        in_channels=4,
        out_channels=1,
        hidden_channels=16,
        n_modes_spatial=(4, 4, 4),
        n_layers=5,
        t_kernel_size=2,
        rank=0.1,
        tau_max=12.4,
        eta_max=4.0,
    )
    tau = torch.linspace(0.4, 12.4, 32)
    eta = torch.linspace(-4.0, 4.0, 8)
    x_te = torch.randn(1, 4, 8, 8, 8, 32)
    y_te = model_te(x_te, tau, eta)
    assert y_te.shape == (1, 1, 8, 8, 8, 32), f"unexpected shape: {y_te.shape}"
    print(f"CausalFNO4DTauEta forward OK   : {tuple(y_te.shape)}")

    print()
    print("All checks passed.")
