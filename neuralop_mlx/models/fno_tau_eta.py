"""FNOTauEta (MLX) — FNO extended with sinusoidal τ/η positional embeddings.

Mirrors neuraloperator-4d-2.0.0/neuralop/models/fno_tau_eta.py in the MLX
backend. See that file for full design notes.

Tensor layout: (B, C, X, Y, Z, T)
  Z — spacetime rapidity η_s  (axis 4)
  T — proper time τ           (axis 5)
"""

import math

import mlx.core as mx
import mlx.nn as nn

from neuralop_mlx.models.fno import FNO
from neuralop_mlx.layers.embeddings import (
    TauBias,
    EtaBias,
    sinusoidal_tau_embedding,
    sinusoidal_eta_embedding,
)


def make_tau_tensor(
    tau_min: float,
    dtau: float,
    n_tau: int,
) -> mx.array:
    """Physical τ values (n_tau,) matching NormFn3d grid metadata."""
    return mx.arange(n_tau, dtype=mx.float32) * dtau + tau_min


def make_eta_tensor(
    eta_min: float,
    deta: float,
    n_eta: int,
) -> mx.array:
    """Physical η values (n_eta,) matching NormFn3d grid metadata."""
    return mx.arange(n_eta, dtype=mx.float32) * deta + eta_min


class FNOTauEta(nn.Module):
    """FNO with sinusoidal τ/η positional embeddings for Milne-coordinate hydro.

    Parameters
    ----------
    in_channels : int
        Physical input channels (e.g. 4 for ed, vx, vy, vz).
    out_channels : int
        Physical output channels.
    hidden_channels : int
        FNO hidden width.
    n_modes : tuple/list of 4 ints
        Fourier mode cutoffs (mx, my, m_eta, m_tau).
    n_layers : int, default 6
        Number of FNO layers.
    tau_max : float, default 12.4
        Reference proper time [fm/c].
    eta_max : float, default 4.0
        |η| range.
    tau_min, dtau : float, optional
        Physical τ grid origin and spacing (NormFn3d metadata). When set, a τ
        grid is built from them via make_tau_tensor() whenever the model is
        called without an explicit tau.
    eta_min, deta : float, optional
        Physical η grid origin and spacing. Used the same way for eta.
    **fno_kwargs
        Forwarded to the underlying FNO.

    Notes
    -----
    tau_max and eta_max are stored as plain attributes.  The MLX trainer's
    checkpoint() method saves them as rank-0 tensors (keys "_tau_max" /
    "_eta_max") alongside the model weights so they are available at load time.
    The grid params (tau_min, dtau, eta_min, deta) are persisted via the
    embedding_grid.json sidecar written by the training script.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_modes,
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

        self._fno = FNO(
            n_modes=n_modes,
            in_channels=in_channels + 4,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            **fno_kwargs,
        )

        self.tau_biases = [TauBias(hidden_channels) for _ in range(n_layers)]
        self.eta_biases = [EtaBias(hidden_channels) for _ in range(n_layers)]

    def __call__(
        self,
        x: mx.array,
        tau: mx.array = None,
        eta: mx.array = None,
    ) -> mx.array:
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

        if tau is None:
            if self.tau_min is None or self.dtau is None:
                raise ValueError(
                    "FNOTauEta: no tau grid available. Pass tau=... explicitly, "
                    "or construct the model with tau_min/dtau so the physical "
                    "grid can be built."
                )
            tau = make_tau_tensor(self.tau_min, self.dtau, T)
        if eta is None:
            if self.eta_min is None or self.deta is None:
                raise ValueError(
                    "FNOTauEta: no eta grid available. Pass eta=... explicitly, "
                    "or construct the model with eta_min/deta so the physical "
                    "grid can be built."
                )
            eta = make_eta_tensor(self.eta_min, self.deta, Z)

        tau_emb = sinusoidal_tau_embedding(tau, self.tau_max)  # (T, 2)
        eta_emb = sinusoidal_eta_embedding(eta, self.eta_max)  # (Z, 2)

        # Broadcast sinusoidal channels across the full spatial grid.
        tau_ch = mx.broadcast_to(
            tau_emb.T.reshape(1, 2, 1, 1, 1, T), (B, 2, X, Y, Z, T)
        )
        eta_ch = mx.broadcast_to(
            eta_emb.T.reshape(1, 2, 1, 1, Z, 1), (B, 2, X, Y, Z, T)
        )
        x_aug = mx.concatenate([x, tau_ch, eta_ch], axis=1)  # (B, C_in+4, …)

        # Replicate FNO.__call__ with per-block bias injection.
        fno = self._fno

        if fno.positional_embedding is not None:
            x_aug = fno.positional_embedding(x_aug)

        x_aug = fno.lifting(x_aug)

        for i in range(self.n_layers):
            x_aug = fno.fno_blocks(x_aug, index=i)
            x_aug = self.tau_biases[i](x_aug, tau_emb)
            x_aug = self.eta_biases[i](x_aug, eta_emb)

        return fno.projection(x_aug)
