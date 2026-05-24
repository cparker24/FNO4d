"""
fno_tau_eta_embedding.py
========================
Standard FNO (neuralop.models.FNO) extended with sinusoidal τ- and η-embeddings
for 3+1D relativistic hydrodynamics surrogate modelling in Milne coordinates.

Tensor layout throughout: (B, C, X, Y, Z, T)
  X, Y  — transverse plane [fm]
  Z     — spacetime rapidity η_s  (axis 4)
  T     — proper time τ           (axis 5)

Why both embeddings are needed
-------------------------------
After NormFn3d normalisation (any of etau75_*, cosh_etau75_*, …) the dominant
amplitude cue the network could use to self-locate in τ or η is removed.
Without an explicit positional signal the FNO's spectral weights are applied
identically at every (η, τ) slice — even though the physics (cooling rate,
longitudinal gradients, flow velocities) is qualitatively different across both
axes.

Embedding design
----------------
Both embeddings follow the same two-part pattern used in causal_loglo_4d_v2.py:

  1. INPUT CONCATENATION (at lifting)
     Sinusoidal coordinates are concatenated as extra input channels so the
     very first linear projection already sees where it is in (τ, η) space.

  2. PER-BLOCK ADDITIVE BIAS (at every FNO layer)
     A small learned Linear projection of (sin, cos) adds a per-(τ,η) channel
     bias to the hidden representation before each spectral convolution.
     This refreshes the positional signal at every depth, preventing it from
     decaying after many channel-mixing operations.

Usage
-----
    model = FNOWithEmbedding(
        in_channels=4,          # e.g. ed, vx, vy, vz
        out_channels=4,
        hidden_channels=32,
        n_modes=(8, 8, 4, 6),   # (mx, my, mz_eta, mt_tau) — all 4 axes
        n_layers=6,
        tau_max=12.4,           # fm/c  (final proper time of your evolution)
        eta_max=4.0,            # |η| range
    ).cuda()

    x   = torch.randn(1, 4, 64, 64, 32, 128, device='cuda')
    tau = torch.linspace(0.4, 12.4, 128, device='cuda')   # (T,)  [fm/c]
    eta = torch.linspace(-4.0, 4.0, 32,  device='cuda')   # (Z,)  [1]

    y   = model(x, tau, eta)   # (1, 4, 64, 64, 32, 128)

Dependencies
------------
  pip install neuraloperator torch
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from neuralop.models import FNO


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sinusoidal embedding functions
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_tau_embedding(tau: torch.Tensor, tau_max: float) -> torch.Tensor:
    """
    2-channel sinusoidal embedding of the proper-time axis.

    Parameters
    ----------
    tau     : (T,)  physical proper-time values [fm/c]
    tau_max : float reference scale [fm/c]

    Returns
    -------
    emb : (T, 2)   [sin(π·τ/τ_max), cos(π·τ/τ_max)]
    """
    t = tau / tau_max
    return torch.stack([torch.sin(math.pi * t),
                        torch.cos(math.pi * t)], dim=-1)


def sinusoidal_eta_embedding(eta: torch.Tensor, eta_max: float) -> torch.Tensor:
    """
    2-channel sinusoidal embedding of the spacetime-rapidity axis.

    The argument is η / η_max, mapping the symmetric interval [-η_max, η_max]
    to [-1, 1] before the sinusoidal transform.  sin is odd (antisymmetric in
    η) and cos is even (symmetric), so together they give the network both the
    magnitude and the sign of the rapidity position.

    Parameters
    ----------
    eta     : (Z,)  physical η_s values (typically −4 … +4)
    eta_max : float |η| range used during training

    Returns
    -------
    emb : (Z, 2)   [sin(π·η/η_max), cos(π·η/η_max)]
    """
    e = eta / eta_max
    return torch.stack([torch.sin(math.pi * e),
                        torch.cos(math.pi * e)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-block bias modules
# ─────────────────────────────────────────────────────────────────────────────

class _TauBias(nn.Module):
    """
    Learned per-block τ-embedding injection.

    Projects (T, 2) → (1, C, 1, 1, 1, T) and adds it to (B, C, X, Y, Z, T).
    Mirrors the _TauBias in causal_loglo_4d_v2.py exactly.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(2, channels, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, tau_emb: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, C, X, Y, Z, T)
        tau_emb : (T, 2)
        """
        T = tau_emb.shape[0]
        bias = self.proj(tau_emb)            # (T, C)
        bias = bias.T.contiguous()           # (C, T)
        bias = bias.view(1, -1, 1, 1, 1, T) # (1, C, 1, 1, 1, T)
        return x + bias


class _EtaBias(nn.Module):
    """
    Learned per-block η-embedding injection.

    Projects (Z, 2) → (1, C, 1, 1, Z, 1) and adds it to (B, C, X, Y, Z, T).

    The η-bias is constant along (X, Y, T) — it encodes rapidity position,
    not transverse or temporal position.  This is analogous to _TauBias but
    acting on the Z (η_s) axis instead of the T (τ) axis.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Linear(2, channels, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x: torch.Tensor, eta_emb: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, C, X, Y, Z, T)
        eta_emb : (Z, 2)
        """
        Z = eta_emb.shape[0]
        bias = self.proj(eta_emb)            # (Z, C)
        bias = bias.T.contiguous()           # (C, Z)
        bias = bias.view(1, -1, 1, 1, Z, 1) # (1, C, 1, 1, Z, 1)
        return x + bias


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FNO wrapper with τ- and η-embeddings
# ─────────────────────────────────────────────────────────────────────────────

class FNOWithEmbedding(nn.Module):
    """
    Standard neuralop FNO extended with sinusoidal τ- and η-embeddings.

    Architecture
    ------------
    INPUT (B, C_in, X, Y, Z, T)
      │
      ├─ concatenate τ-emb (1, 2, 1, 1, 1, T) broadcast → (B, C_in+2, X, Y, Z, T)
      ├─ concatenate η-emb (1, 2, 1, 1, Z, 1) broadcast → (B, C_in+4, X, Y, Z, T)
      │
      └─ FNO trunk  (neuralop.FNO, sees C_in+4 input channels)
             │
             ├─ FNO layer 0  →  + τ-bias  + η-bias
             ├─ FNO layer 1  →  + τ-bias  + η-bias
             │   …
             └─ FNO layer N  →  + τ-bias  + η-bias
                  │
                  └─ linear projection → (B, C_out, X, Y, Z, T)

    Notes
    -----
    - The neuralop FNO expects a flat (B, C, *spatial) input.  Here spatial
      includes the η and τ axes, so the FNO sees a 4D spatial grid
      (X, Y, Z, T) = (X, Y, η, τ) and performs a 4D spectral convolution.

    - The per-block τ/η bias is injected by hooking into each FNO layer's
      output via forward hooks.  This is the cleanest way to extend an
      off-the-shelf neuralop FNO without forking its internals.

    - n_modes must therefore be a 4-tuple: (mx, my, mη, mτ).

    Parameters
    ----------
    in_channels     : int    physical input channels (e.g. 4 for ed, vx, vy, vz)
    out_channels    : int    physical output channels
    hidden_channels : int    FNO hidden width
    n_modes         : tuple  (mx, my, mz_eta, mt_tau) Fourier mode cutoffs
    n_layers        : int    number of FNO layers
    tau_max         : float  reference proper time [fm/c] for sinusoidal embedding
    eta_max         : float  |η| range for sinusoidal embedding
    **fno_kwargs    : passed verbatim to neuralop.models.FNO
                      (e.g. factorization='tucker', rank=0.05)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: tuple[int, int, int, int],
        n_layers: int = 6,
        tau_max: float = 12.4,
        eta_max: float = 4.0,
        **fno_kwargs,
    ):
        super().__init__()

        self.tau_max = tau_max
        self.eta_max = eta_max

        # The FNO receives C_in + 4 channels: original + 2 τ-channels + 2 η-channels.
        fno_in = in_channels + 4

        self.fno = FNO(
            n_modes=n_modes,
            in_channels=fno_in,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **fno_kwargs,
        )

        # One _TauBias and one _EtaBias per FNO layer.
        C = hidden_channels
        self.tau_biases = nn.ModuleList([_TauBias(C) for _ in range(n_layers)])
        self.eta_biases = nn.ModuleList([_EtaBias(C) for _ in range(n_layers)])

        # Internal state set during forward so hooks can read it.
        self._tau_emb: torch.Tensor | None = None
        self._eta_emb: torch.Tensor | None = None
        self._hook_handles = []

    # ── Hook registration ─────────────────────────────────────────────────────

    def _register_hooks(self) -> None:
        """
        Attach one post-forward hook to every FNO Fourier layer.

        The hook adds the τ-bias and η-bias to the layer output before it
        is passed to the next layer.  Hooks are removed after each forward
        call to avoid double-application.
        """
        # neuralop stores Fourier layers in fno.fno_blocks (a ModuleList)
        fno_layers = list(self.fno.fno_blocks.convs.children()) \
            if hasattr(self.fno.fno_blocks, "convs") \
            else list(self.fno.fno_blocks.children())

        # Fallback: walk children until we find the list of spectral layers
        if not fno_layers:
            fno_layers = [
                m for m in self.fno.modules()
                if m.__class__.__name__ in ("FNOBlocks", "SpectralConv")
            ]

        for i, layer in enumerate(fno_layers):
            if i >= len(self.tau_biases):
                break

            def make_hook(idx):
                def hook(module, inp, out):
                    # out may be a tensor or a tuple; handle both.
                    if isinstance(out, (tuple, list)):
                        t = out[0]
                        t = self.tau_biases[idx](t, self._tau_emb)
                        t = self.eta_biases[idx](t, self._eta_emb)
                        return (t,) + out[1:]
                    else:
                        out = self.tau_biases[idx](out, self._tau_emb)
                        out = self.eta_biases[idx](out, self._eta_emb)
                        return out
                return hook

            h = layer.register_forward_hook(make_hook(i))
            self._hook_handles.append(h)

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor | None = None,
        eta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, C_in, X, Y, Z, T)
        tau : (T,) proper-time values [fm/c].
              If None, uniform grid [0, tau_max] is assumed.
        eta : (Z,) spacetime-rapidity values.
              If None, uniform grid [-eta_max, +eta_max] is assumed.

        Returns
        -------
        y : (B, C_out, X, Y, Z, T)
        """
        B, C_in, X, Y, Z, T = x.shape
        device, dtype = x.device, x.dtype

        # ── Build coordinate tensors ──────────────────────────────────────────
        if tau is None:
            tau = torch.linspace(0.0, self.tau_max, T, device=device, dtype=dtype)
        if eta is None:
            eta = torch.linspace(-self.eta_max, self.eta_max, Z, device=device, dtype=dtype)

        # ── Sinusoidal embeddings ─────────────────────────────────────────────
        tau_emb = sinusoidal_tau_embedding(tau, self.tau_max).to(dtype=dtype)  # (T, 2)
        eta_emb = sinusoidal_eta_embedding(eta, self.eta_max).to(dtype=dtype)  # (Z, 2)

        # Store for hooks
        self._tau_emb = tau_emb
        self._eta_emb = eta_emb

        # ── Concatenate embeddings as extra input channels ────────────────────
        #
        # τ-embedding: (T, 2) → broadcast to (B, 2, X, Y, Z, T)
        tau_ch = tau_emb.T.view(1, 2, 1, 1, 1, T).expand(B, 2, X, Y, Z, T)
        #
        # η-embedding: (Z, 2) → broadcast to (B, 2, X, Y, Z, T)
        eta_ch = eta_emb.T.view(1, 2, 1, 1, Z, 1).expand(B, 2, X, Y, Z, T)
        #
        # Concatenate along channel axis → (B, C_in+4, X, Y, Z, T)
        x_aug = torch.cat([x, tau_ch, eta_ch], dim=1)

        # ── Register per-block hooks, run FNO, clean up ───────────────────────
        self._register_hooks()
        try:
            y = self.fno(x_aug)
        finally:
            self._remove_hooks()
            self._tau_emb = None
            self._eta_emb = None

        return y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Coordinate helpers (call these to build tau / eta tensors from metadata)
# ─────────────────────────────────────────────────────────────────────────────

def make_tau_tensor(
    tau_min: float,
    dtau: float,
    n_tau: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return physical τ values (n_tau,) for the grid used in NormFn3d."""
    return torch.arange(n_tau, device=device, dtype=dtype) * dtau + tau_min


def make_eta_tensor(
    eta_min: float,
    deta: float,
    n_eta: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return physical η values (n_eta,) for the grid used in NormFn3d."""
    return torch.arange(n_eta, device=device, dtype=dtype) * deta + eta_min


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Smoke-test grid: 8×8×4×16  (tiny, CPU-safe)
    B, C_in, C_out = 1, 4, 4
    X, Y, Z, T     = 8, 8, 4, 16

    # Example NormFn3d metadata (match your actual grid)
    TAU_MIN, DTAU = 0.4, 0.076        # fm/c  (0.4 + 128*0.076 ~ 10.1 fm/c)
    ETA_MIN, DETA = -3.5, 7.0 / (Z-1) # η_s symmetric around 0

    tau = make_tau_tensor(TAU_MIN, DTAU, T, device=device)
    eta = make_eta_tensor(ETA_MIN, DETA, Z, device=device)

    model = FNOWithEmbedding(
        in_channels=C_in,
        out_channels=C_out,
        hidden_channels=16,
        n_modes=(2, 2, 2, 4),          # (mx, my, mη, mτ)
        n_layers=4,
        tau_max=tau[-1].item(),
        eta_max=abs(float(eta[-1])),
        factorization="tucker",
        rank=0.5,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}\n")

    x = torch.randn(B, C_in, X, Y, Z, T, device=device)
    y = model(x, tau, eta)

    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (B, C_out, X, Y, Z, T), "Shape mismatch!"
    print("Shape OK ✓\n")

    # Verify embedding values are physically reasonable
    tau_emb = sinusoidal_tau_embedding(tau, tau[-1].item())
    eta_emb = sinusoidal_eta_embedding(eta, abs(float(eta[-1])))
    print(f"τ-embedding shape: {tuple(tau_emb.shape)}, "
          f"range [{tau_emb.min():.2f}, {tau_emb.max():.2f}]")
    print(f"η-embedding shape: {tuple(eta_emb.shape)}, "
          f"range [{eta_emb.min():.2f}, {eta_emb.max():.2f}]")
    print("\nAll checks passed ✓")
