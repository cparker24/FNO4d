"""FNOTauEta — FNO extended with sinusoidal τ/η positional embeddings.

Designed for 3+1D relativistic hydrodynamics surrogates in Milne coordinates
where NormFn3d normalisation removes the dominant amplitude cues along the
proper-time (τ) and spacetime-rapidity (η) axes.

Tensor layout throughout: (B, C, X, Y, Z, T)
  X, Y  — transverse plane
  Z     — spacetime rapidity η_s  (axis 4)
  T     — proper time τ           (axis 5)

Two-point embedding injection
------------------------------
1. INPUT CONCATENATION: 4 sinusoidal channels (2 τ + 2 η) are appended to
   the input before the lifting MLP, so the first projection already sees
   where it is in (τ, η) space.

2. PER-BLOCK ADDITIVE BIAS: a small Linear(2→C) refreshes the positional
   signal at every FNO layer, preventing it from decaying with depth.
"""

from typing import List, Tuple, Union

import torch
import torch.nn as nn

from .base_model import BaseModel
from .fno import FNO
from ..layers.embeddings import (
    TauBias,
    EtaBias,
    sinusoidal_tau_embedding,
    sinusoidal_eta_embedding,
)


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


class FNOTauEta(BaseModel, name="FNOTauEta"):
    """FNO with sinusoidal τ/η positional embeddings for Milne-coordinate hydro.

    Parameters
    ----------
    in_channels : int
        Physical input channels (e.g. 4 for ed, vx, vy, vz).
    out_channels : int
        Physical output channels.
    hidden_channels : int
        FNO hidden width.
    n_modes : tuple of 4 ints
        Fourier mode cutoffs (mx, my, m_eta, m_tau).
    n_layers : int, default 6
        Number of FNO layers.
    tau_max : float, default 12.4
        Reference proper time [fm/c] for sinusoidal embedding.
    eta_max : float, default 4.0
        |η| range for sinusoidal embedding.
    tau_min, dtau : float, optional
        Physical τ grid origin and spacing (NormFn3d metadata). When set, a τ
        grid is built from them via make_tau_tensor() whenever forward() is
        called without an explicit tau.
    eta_min, deta : float, optional
        Physical η grid origin and spacing. Used the same way for eta.
    **fno_kwargs
        Forwarded to the underlying FNO (e.g. factorization, rank,
        domain_padding, positional_embedding, norm, …).

    Notes
    -----
    tau_max, eta_max and the grid params (tau_min, dtau, eta_min, deta) are
    constructor kwargs and are therefore captured in BaseModel._init_kwargs, so
    they round-trip through state_dict()['_metadata'] automatically — no extra
    work needed when saving or resuming a checkpoint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes: Tuple[int, int, int, int],
        n_layers: int = 6,
        tau_max: float = 12.4,
        eta_max: float = 4.0,
        tau_min: float = None,
        dtau: float = None,
        eta_min: float = None,
        deta: float = None,
        **fno_kwargs,
    ):
        super().__init__()

        self.tau_max = tau_max
        self.eta_max = eta_max
        self.tau_min = tau_min
        self.dtau = dtau
        self.eta_min = eta_min
        self.deta = deta
        self.n_layers = n_layers

        # Underlying FNO receives 4 extra channels (2 τ + 2 η).
        self.fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels + 4,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **fno_kwargs,
        )

        self.tau_biases = nn.ModuleList(
            [TauBias(hidden_channels) for _ in range(n_layers)]
        )
        self.eta_biases = nn.ModuleList(
            [EtaBias(hidden_channels) for _ in range(n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        tau: torch.Tensor = None,
        eta: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x   : (B, C_in, X, Y, Z, T)
        tau : (T,) physical proper-time values [fm/c]. When None, the physical
              grid is built from the stored tau_min/dtau; if those were not set
              at construction a ValueError is raised (no silent fallback).
        eta : (Z,) spacetime-rapidity values. Built from eta_min/deta when None,
              with the same error if unavailable.

        Returns
        -------
        (B, C_out, X, Y, Z, T)
        """
        B, C_in, X, Y, Z, T = x.shape
        device, dtype = x.device, x.dtype

        if tau is None:
            if self.tau_min is None or self.dtau is None:
                raise ValueError(
                    "FNOTauEta.forward: no tau grid available. Pass tau=... "
                    "explicitly, or construct the model with tau_min/dtau so the "
                    "physical grid can be built."
                )
            tau = make_tau_tensor(self.tau_min, self.dtau, T, device=device, dtype=dtype)
        if eta is None:
            if self.eta_min is None or self.deta is None:
                raise ValueError(
                    "FNOTauEta.forward: no eta grid available. Pass eta=... "
                    "explicitly, or construct the model with eta_min/deta so the "
                    "physical grid can be built."
                )
            eta = make_eta_tensor(self.eta_min, self.deta, Z, device=device, dtype=dtype)

        tau_emb = sinusoidal_tau_embedding(tau, self.tau_max).to(dtype=dtype)  # (T, 2)
        eta_emb = sinusoidal_eta_embedding(eta, self.eta_max).to(dtype=dtype)  # (Z, 2)

        # Broadcast embedding channels across the spatial grid.
        tau_ch = tau_emb.T.view(1, 2, 1, 1, 1, T).expand(B, 2, X, Y, Z, T)
        eta_ch = eta_emb.T.view(1, 2, 1, 1, Z, 1).expand(B, 2, X, Y, Z, T)
        x_aug = torch.cat([x, tau_ch, eta_ch], dim=1)  # (B, C_in+4, X, Y, Z, T)

        # Replicate FNO.forward() with per-block bias injection.
        fno = self.fno

        if fno.positional_embedding is not None:
            x_aug = fno.positional_embedding(x_aug)

        x_aug = fno.lifting(x_aug)

        if fno.domain_padding is not None:
            x_aug = fno.domain_padding.pad(x_aug)

        for i in range(self.n_layers):
            x_aug = fno.fno_blocks(x_aug, i)
            x_aug = self.tau_biases[i](x_aug, tau_emb)
            x_aug = self.eta_biases[i](x_aug, eta_emb)

        if fno.domain_padding is not None:
            x_aug = fno.domain_padding.unpad(x_aug)

        return fno.projection(x_aug)
