"""
model_eval.py

Comprehensive evaluation suite for FNO surrogates of heavy-ion hydrodynamics.

Data convention
---------------
    pred, true : ndarray (N, C, nx, ny, neta, T)
    N     — test events
    C     — channels  (0=ε, 1=vx, 2=vy, 3=vη)
    nx,ny — transverse spatial grid
    neta  — pseudo-rapidity bins
    T     — output τ steps  (initial condition *not* included)

Most metric methods return shape (C, neta, T) unless stated otherwise.
Averaging over events (N) is always performed internally.

Metrics implemented
-------------------
Pointwise  : nRMSE, RMSE, MaxError, bRMSE, cRMSE, vRMSE
Spectral   : fRMSE(low/mid/high), MELR, WLR,
             spectral_error_map(), energy_spectrum()
Temporal   : ar_rollout()  (static helper for 1-step AR evaluation)
Summary    : summary() → flat dict of scalars

Plots
-----
    plot_nrmse_vs_tau(), plot_frmse_vs_tau(), plot_melr_wlr_vs_tau(),
    plot_spectral_error_map(), plot_energy_spectrum(),
    plot_dashboard()  ← 2×3 composite figure

Known failure modes
-------------------
1. Sparse-field blow-up (nRMSE, vRMSE, fRMSE >> 1)
   Cause   : fields with many freeze-out / vacuum cells have near-zero
             mean(true²) and var(true), making all normalized metrics diverge.
   Fix     : pass active_threshold (GeV/fm³) or active_mask to ModelEval.
             Only cells where true[channel_0] > threshold are included in
             the normalized averages and in the FFTs.

2. cRMSE blow-up with vacuum cells (cRMSE >> 1 even for channel 0)
   Cause   : the volume integral is summed over ALL cells including vacuum /
             freeze-out ones.  If the model predicts non-zero ε in vacuum
             cells the numerator accumulates without a matching increase in
             the denominator, driving cRMSE → 10¹¹.
   Fix     : pass active_threshold to ModelEval; cRMSE now respects the
             active_mask and sums only over active cells (same masking as
             nRMSE and vRMSE).

3. cRMSE undefined for symmetric channels (vx, vy, vη)
   Cause   : velocity fields are antisymmetric → their volume integral ≈ 0
             → denominator collapses to eps → cRMSE → 10¹³.
   Fix     : the denominator is now max(|signed_sum|, 0.01 * L1_sum).
             cRMSE is still most meaningful for channel 0 (ε); treat
             velocity-channel cRMSE values as approximate only.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
#  Module-level helpers                                                         #
# --------------------------------------------------------------------------- #

def _build_kmap(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """Radial wavenumber map for rfft2 output (nx, ny//2+1)."""
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.rfftfreq(ny) * ny
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    kmap = np.sqrt(KX ** 2 + KY ** 2)           # (nx, ny//2+1)
    k_max = int(np.floor(kmap.max()))
    shells = np.arange(0, k_max + 1, dtype=float)
    return kmap, shells


def _boundary_mask(nx: int, ny: int) -> np.ndarray:
    mask = np.zeros((nx, ny), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    return mask


# --------------------------------------------------------------------------- #
#  ModelEval                                                                    #
# --------------------------------------------------------------------------- #

class ModelEval:
    """
    Full evaluation of a neural surrogate model for 3+1D hydrodynamics.

    Parameters
    ----------
    pred : ndarray (N, C, nx, ny, neta, T)   — model output (physical units)
    true : ndarray (N, C, nx, ny, neta, T)   — ground truth (physical units)
    tau_coords : 1-D array, length T          — τ values for the T output steps (fm/c)
    eta_coords : 1-D array, length neta       — η values
    channel_names : list[str], length C
    eps : small constant to avoid division by zero
    active_threshold : float or None
        If given, only cells where true[:, 0] > active_threshold (channel 0,
        energy density in GeV/fm³) are included in normalized metrics (nRMSE,
        vRMSE) and in the FFTs used by spectral metrics.  Absolute metrics
        (RMSE, MaxError, bRMSE) are unaffected.
        Typical value: 0.1–0.5 GeV/fm³  (just above the QCD freeze-out energy
        density ~0.3 GeV/fm³ at T_c ≈ 155 MeV).
    active_mask : ndarray or None
        Explicit boolean mask, shape (N, nx, ny, neta, T) or
        (N, C, nx, ny, neta, T).  Takes precedence over active_threshold.
    """

    _DEFAULT_CHANNELS = ['ε', 'vx', 'vy', 'vη']

    def __init__(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        tau_coords: Optional[np.ndarray] = None,
        eta_coords: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None,
        eps: float = 1e-10,
        active_threshold: Optional[float] = None,
        active_mask: Optional[np.ndarray] = None,
    ):
        assert pred.shape == true.shape, "pred and true must have identical shapes"
        assert pred.ndim == 6, "Expected shape (N, C, nx, ny, neta, T)"
        self.pred = np.asarray(pred, dtype=np.float64)
        self.true = np.asarray(true, dtype=np.float64)
        self.N, self.C, self.nx, self.ny, self.neta, self.T = pred.shape
        self.eps = eps

        self.tau_coords = (
            np.asarray(tau_coords, dtype=float) if tau_coords is not None
            else np.arange(self.T, dtype=float)
        )
        self.eta_coords = (
            np.asarray(eta_coords, dtype=float) if eta_coords is not None
            else np.arange(self.neta, dtype=float)
        )
        self.channel_names = (channel_names or self._DEFAULT_CHANNELS)[: self.C]

        self._kmap, self._k_shells = _build_kmap(self.nx, self.ny)

        # Active-cell mask — derived from threshold or supplied explicitly.
        # Shape after construction: (N, C, nx, ny, neta, T) bool.
        if active_mask is not None:
            m = np.asarray(active_mask, dtype=bool)
            if m.ndim == 5:                        # (N, nx, ny, neta, T)
                m = m[:, np.newaxis]               # → (N, 1, nx, ny, neta, T)
            self._active_mask: Optional[np.ndarray] = np.broadcast_to(
                m, self.true.shape
            ).copy()
        elif active_threshold is not None:
            # Drive mask from channel-0 (ε) of the ground truth field
            m = self.true[:, :1] > active_threshold   # (N, 1, nx, ny, neta, T)
            self._active_mask = np.broadcast_to(m, self.true.shape).copy()
        else:
            self._active_mask = None

        if self._active_mask is not None:
            frac = self._active_mask[:, 0].mean()
            print(f'ModelEval: active_mask set — {frac:.1%} of cells are active')

        # lazy caches
        self._F_pred: Optional[np.ndarray] = None
        self._F_true: Optional[np.ndarray] = None
        self._E_pred: Optional[np.ndarray] = None  # (N, C, neta, T, n_shells)
        self._E_true: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ffts(self) -> Tuple[np.ndarray, np.ndarray]:
        """rfft2 over (nx, ny) axes → shape (N, C, nx, ny//2+1, neta, T).

        When an active_mask is set, inactive cells are zeroed before the FFT
        so that freeze-out / vacuum regions do not inflate spectral energy.
        """
        if self._F_pred is None:
            if self._active_mask is not None:
                p = self.pred * self._active_mask
                t = self.true * self._active_mask
            else:
                p, t = self.pred, self.true
            self._F_pred = np.fft.rfft2(p, axes=(2, 3))
            self._F_true = np.fft.rfft2(t, axes=(2, 3))
        return self._F_pred, self._F_true

    def _radial_energies(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Radially-binned spectral power for every (N, C, neta, T) slice.

        Returns E_pred, E_true each shape (N, C, neta, T, n_shells).
        Parseval-corrected so that E.sum(axis=-1) ≈ spatial variance.
        """
        if self._E_pred is not None:
            return self._E_pred, self._E_true

        F_pred, F_true = self._ffts()
        Pp = 0.5 * np.abs(F_pred) ** 2   # (N, C, nx, ny//2+1, neta, T)
        Pt = 0.5 * np.abs(F_true) ** 2

        # Parseval correction: double the non-DC, non-Nyquist ky components
        # rfft2 with axes=(2,3): axis 3 is the real-FFT axis (ky), axis 2 is full FFT (kx)
        if self.ny % 2 == 0:
            Pp[:, :, :, 1:-1, :, :] *= 2   # ky = 1 … ny//2-1
            Pt[:, :, :, 1:-1, :, :] *= 2
        else:
            Pp[:, :, :, 1:, :, :] *= 2     # ky = 1 … ny//2
            Pt[:, :, :, 1:, :, :] *= 2

        n_shells = len(self._k_shells)
        E_p = np.zeros((self.N, self.C, self.neta, self.T, n_shells))
        E_t = np.zeros((self.N, self.C, self.neta, self.T, n_shells))

        for ik, k in enumerate(self._k_shells):
            mask_k = (self._kmap >= k - 0.5) & (self._kmap < k + 0.5)
            if not mask_k.any():
                continue
            # Advanced indexing with 2-D bool mask on axes (2,3)
            # result shape: (N, C, n_pix, neta, T)
            E_p[:, :, :, :, ik] = Pp[:, :, mask_k, :, :].sum(axis=2)
            E_t[:, :, :, :, ik] = Pt[:, :, mask_k, :, :].sum(axis=2)

        self._E_pred = E_p
        self._E_true = E_t
        return E_p, E_t

    def _resolve_itau(self, itau: int) -> int:
        return self.T + itau if itau < 0 else itau

    # ------------------------------------------------------------------ #
    #  Pointwise metrics  →  (C, neta, T)  [cRMSE → (C, T)]             #
    # ------------------------------------------------------------------ #

    def nrmse(self) -> np.ndarray:
        """
        nRMSE per (channel, η, τ): per-sample sqrt(MSE_xy) / RMS_xy(true),
        averaged over events.  Shape (C, neta, T).

        When active_mask is set, the mean is taken only over active cells,
        preventing freeze-out / vacuum zeros from collapsing the denominator.
        """
        if self._active_mask is not None:
            mf = self._active_mask.astype(np.float64)        # (N, C, nx, ny, neta, T)
            n_act = np.maximum(mf.sum(axis=(2, 3)), 1)       # (N, C, neta, T)
            diff_sq = ((self.pred - self.true) ** 2 * mf).sum(axis=(2, 3)) / n_act
            true_sq = (self.true ** 2 * mf).sum(axis=(2, 3)) / n_act
        else:
            diff_sq = np.mean((self.pred - self.true) ** 2, axis=(2, 3))
            true_sq = np.mean(self.true ** 2, axis=(2, 3))
        per_sample = np.sqrt(diff_sq / np.maximum(true_sq, self.eps))
        return per_sample.mean(axis=0)                                 # (C, neta, T)

    def rmse(self) -> np.ndarray:
        """Absolute RMSE over events and (x, y).  Shape (C, neta, T)."""
        return np.sqrt(np.mean((self.pred - self.true) ** 2, axis=(0, 2, 3)))

    def max_error(self) -> np.ndarray:
        """Max |pred − true| over events and (x, y).  Shape (C, neta, T)."""
        return np.max(np.abs(self.pred - self.true), axis=(0, 2, 3))

    def brmse(self) -> np.ndarray:
        """
        Boundary RMSE: RMSE restricted to the (x, y) domain boundary.
        Proxy for freeze-out surface accuracy.  Shape (C, neta, T).
        """
        flat_mask = _boundary_mask(self.nx, self.ny).reshape(-1)
        p = self.pred.reshape(self.N, self.C, -1, self.neta, self.T)
        t = self.true.reshape(self.N, self.C, -1, self.neta, self.T)
        diff_sq = np.mean((p[:, :, flat_mask] - t[:, :, flat_mask]) ** 2, axis=(0, 2))
        return np.sqrt(diff_sq)                                        # (C, neta, T)

    def crmse(self) -> np.ndarray:
        """
        Normalized RMSE of volume-integrated fields (proxy for conservation-law
        violations).  Shape (C, T)  — neta is collapsed by the volume integral.

        Denominator: max(|signed_sum|, 0.01 * L1_sum)
        For energy density (channel 0, always positive) this equals |int_t|,
        identical to the naive formula.
        For velocity channels (antisymmetric → signed_sum ≈ 0), the L1 floor
        replaces the near-zero denominator with 1% of the total velocity
        magnitude, giving a physically meaningful bound instead of diverging
        to 10¹³.  Velocity cRMSE values remain approximate; channel 0 is the
        primary conservation diagnostic.

        When active_mask is set the integral is restricted to active cells only,
        preventing vacuum / freeze-out cells from contributing zero-denominator
        or large-prediction-error terms that inflate the ratio.
        """
        if self._active_mask is not None:
            mf        = self._active_mask.astype(np.float64)
            int_p     = (self.pred * mf).sum(axis=(2, 3, 4))       # (N, C, T)
            int_t     = (self.true * mf).sum(axis=(2, 3, 4))
            int_abs_t = (np.abs(self.true) * mf).sum(axis=(2, 3, 4))
        else:
            int_p     = self.pred.sum(axis=(2, 3, 4))               # (N, C, T)
            int_t     = self.true.sum(axis=(2, 3, 4))
            int_abs_t = np.abs(self.true).sum(axis=(2, 3, 4))
        denom = np.maximum(np.abs(int_t), 0.01 * int_abs_t) + self.eps
        rel   = np.abs(int_p - int_t) / denom
        return rel.mean(axis=0)                                        # (C, T)

    def vrmse(self) -> np.ndarray:
        """
        Variance-normalized RMSE: per-sample sqrt(MSE_xy / var_xy(true)).
        Shape (C, neta, T).

        When active_mask is set, mean and variance are computed only over
        active cells, preventing near-zero variance from freeze-out regions.
        """
        if self._active_mask is not None:
            mf    = self._active_mask.astype(np.float64)
            n_act = np.maximum(mf.sum(axis=(2, 3)), 1)
            mse   = ((self.pred - self.true) ** 2 * mf).sum(axis=(2, 3)) / n_act
            mu    = (self.true * mf).sum(axis=(2, 3)) / n_act
            mu2   = (self.true ** 2 * mf).sum(axis=(2, 3)) / n_act
            var   = mu2 - mu ** 2
        else:
            mse = np.mean((self.pred - self.true) ** 2, axis=(2, 3))
            var = np.var(self.true, axis=(2, 3))
        per_sample = np.sqrt(mse / np.maximum(var, self.eps))
        return per_sample.mean(axis=0)

    # ------------------------------------------------------------------ #
    #  Spectral metrics                                                    #
    # ------------------------------------------------------------------ #

    def frmse(self) -> Dict[str, np.ndarray]:
        """
        Frequency-band normalized RMSE in the low / mid / high thirds of the
        radial wavenumber range.  Each entry has shape (C, neta, T).

        fRMSE_band = sqrt( Σ_{k∈band} |F_pred−F_true|² / Σ_{k∈band} |F_true|² )
        averaged over events.
        """
        F_pred, F_true = self._ffts()
        diff_F = F_pred - F_true

        k_max = float(self._kmap.max())
        bands = {
            'low':  (0.0,          k_max / 3.0),
            'mid':  (k_max / 3.0,  2.0 * k_max / 3.0),
            'high': (2.0 * k_max / 3.0, k_max + 1.0),
        }

        result: Dict[str, np.ndarray] = {}
        for name, (klo, khi) in bands.items():
            band_mask = (self._kmap >= klo) & (self._kmap < khi)
            # (N, C, n_pix, neta, T)
            d = diff_F[:, :, band_mask, :, :]
            t = F_true[:, :, band_mask, :, :]
            num = (np.abs(d) ** 2).mean(axis=0).sum(axis=1)   # (C, neta, T)
            den = (np.abs(t) ** 2).mean(axis=0).sum(axis=1)
            result[name] = np.sqrt(num / np.maximum(den, self.eps))
        return result

    def melr(self) -> np.ndarray:
        """
        Mean Energy Log Ratio: wavenumber-weighted mean of log(E_pred/E_true)
        over radial shells.  Shape (C, neta, T), averaged over events.

        MELR > 0 → over-prediction of spectral energy (too sharp)
        MELR < 0 → under-prediction (over-smoothed, typical for FNO)
        """
        E_p, E_t = self._radial_energies()        # (N, C, neta, T, n_shells)
        K = self._k_shells                         # (n_shells,)

        valid = (E_t > self.eps) & (E_p > self.eps) & (K > 0)
        ratio = np.where(valid, E_p / np.maximum(E_t, self.eps), 1.0)
        log_r = np.where(valid, np.log(ratio), 0.0)
        w     = np.where(valid, K, 0.0)

        w_sum  = w.sum(axis=-1)
        melr_n = (w * log_r).sum(axis=-1) / np.maximum(w_sum, self.eps)
        return melr_n.mean(axis=0)                 # (C, neta, T)

    def wlr(self) -> np.ndarray:
        """
        Wasserstein Log Ratio: W₁ distance between the normalized spectral energy
        distributions of pred and true.  Shape (C, neta, T), averaged over events.
        """
        E_p, E_t = self._radial_energies()

        sp = E_p.sum(axis=-1, keepdims=True)
        st = E_t.sum(axis=-1, keepdims=True)
        p_norm = E_p / np.maximum(sp, self.eps)
        q_norm = E_t / np.maximum(st, self.eps)

        wlr_n = np.abs(
            np.cumsum(p_norm, axis=-1) - np.cumsum(q_norm, axis=-1)
        ).sum(axis=-1)
        return wlr_n.mean(axis=0)                  # (C, neta, T)

    def spectral_error_map(
        self, channel: int = 0, ieta: int = 0, itau: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2-D log₁₀(|F_pred − F_true|²) map in Fourier space, averaged over events.

        Returns
        -------
        kx : 1-D array (nx,), fftshifted  [cycles per pixel]
        ky : 1-D array (ny//2+1,)
        log_err : 2-D array (nx, ny//2+1)
        """
        itau = self._resolve_itau(itau)
        F_pred, F_true = self._ffts()
        diff = F_pred[:, channel, :, :, ieta, itau] - F_true[:, channel, :, :, ieta, itau]
        err  = np.mean(np.abs(diff) ** 2, axis=0)              # (nx, ny//2+1)
        log_err = np.log10(np.fft.fftshift(err, axes=0) + self.eps)
        kx = np.fft.fftshift(np.fft.fftfreq(self.nx) * self.nx)
        ky = np.fft.rfftfreq(self.ny) * self.ny
        return kx, ky, log_err

    def energy_spectrum(
        self, channel: int = 0, ieta: int = 0, itau: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Radially-averaged 1-D power spectrum E(K) vs wavenumber K.

        Returns (K_shells, E_pred_mean, E_true_mean).
        """
        itau = self._resolve_itau(itau)
        E_p, E_t = self._radial_energies()
        return (
            self._k_shells,
            E_p[:, channel, ieta, itau, :].mean(axis=0),
            E_t[:, channel, ieta, itau, :].mean(axis=0),
        )

    # ------------------------------------------------------------------ #
    #  Autoregressive rollout helper                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def ar_rollout(
        model_fn: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
    ) -> np.ndarray:
        """
        Autoregressive multi-step rollout for a 1-step trained model.

        model_fn receives a numpy array of shape (1, C, nx, ny, neta, 1)
        (initial condition, τ-dim = 1) and must return (1, C, nx, ny, neta, 1).

        Parameters
        ----------
        model_fn : callable(x) → x_next
        data     : ndarray (N, C, nx, ny, neta, T+1)
                   Full data including the initial condition at τ-index 0.

        Returns
        -------
        ar_pred : ndarray (N, C, nx, ny, neta, T)
        """
        N, C, nx, ny, neta, full_T = data.shape
        T = full_T - 1
        ar_pred = np.zeros((N, C, nx, ny, neta, T), dtype=np.float32)
        for n in range(N):
            x_cur = data[n, :, :, :, :, :1].astype(np.float32)   # (C, nx, ny, neta, 1)
            for t in range(T):
                x_in  = x_cur[np.newaxis]                          # (1, C, nx, ny, neta, 1)
                x_next = model_fn(x_in)[0, :, :, :, :, :1]        # (C, nx, ny, neta, 1)
                ar_pred[n, :, :, :, :, t] = x_next[:, :, :, :, 0]
                x_cur = x_next
        return ar_pred

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def active_frac(self) -> float:
        """Fraction of (x, y, η, τ) cells that are active. 1.0 when no mask set."""
        if self._active_mask is None:
            return 1.0
        return float(self._active_mask[:, 0].mean())

    def summary(self) -> Dict[str, float]:
        """
        Scalar summary: mean over all channels, η, τ.

        cRMSE_e0 is reported separately for channel 0 (energy density only),
        which is the only channel where the signed volume integral is physically
        meaningful as a conservation proxy.
        cRMSE_all is the mean over all channels using the L1-floored denominator.
        When active_threshold is set, cRMSE is computed over active cells only
        (consistent with nRMSE and vRMSE masking).
        active_frac shows what fraction of cells were included in normalized
        metrics (1.0 when no active_mask / active_threshold was set).
        """
        frmse_v = self.frmse()
        crmse_v = self.crmse()   # (C, T)
        return {
            'nRMSE':      float(self.nrmse().mean()),
            'RMSE':       float(self.rmse().mean()),
            'MaxError':   float(self.max_error().mean()),
            'bRMSE':      float(self.brmse().mean()),
            'cRMSE_e0':   float(crmse_v[0].mean()),          # ε channel only
            'cRMSE_all':  float(crmse_v.mean()),             # all channels
            'vRMSE':      float(self.vrmse().mean()),
            'fRMSE_low':  float(frmse_v['low'].mean()),
            'fRMSE_mid':  float(frmse_v['mid'].mean()),
            'fRMSE_high': float(frmse_v['high'].mean()),
            'MELR':       float(self.melr().mean()),
            'WLR':        float(self.wlr().mean()),
            'active_frac': self.active_frac(),
        }

    # ------------------------------------------------------------------ #
    #  Individual plots                                                    #
    # ------------------------------------------------------------------ #

    def plot_nrmse_vs_tau(
        self,
        channel: int = 0,
        fig=None,
        ax=None,
        title: Optional[str] = None,
    ):
        nrmse_v = self.nrmse()
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        for ie in range(self.neta):
            lbl = f'η={self.eta_coords[ie]:.2f}' if self.neta <= 12 else None
            ax.plot(self.tau_coords, nrmse_v[channel, ie], alpha=0.8, label=lbl)
        ax.set_xlabel('τ (fm/c)')
        ax.set_ylabel('nRMSE')
        ax.set_title(title or f'nRMSE vs τ — {self.channel_names[channel]}')
        if self.neta <= 12:
            ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def plot_frmse_vs_tau(
        self,
        channel: int = 0,
        ieta: int = 0,
        fig=None,
        ax=None,
    ):
        frmse_v = self.frmse()
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.tau_coords, frmse_v['low'][channel, ieta],
                color='steelblue', label='fRMSE (low)')
        ax.plot(self.tau_coords, frmse_v['mid'][channel, ieta],
                color='darkorange', label='fRMSE (mid)')
        ax.plot(self.tau_coords, frmse_v['high'][channel, ieta],
                color='crimson', label='fRMSE (high)')
        ax.set_xlabel('τ (fm/c)')
        ax.set_ylabel('fRMSE (normalized)')
        ax.set_title(
            f'Spectral RMSE — {self.channel_names[channel]}, '
            f'η={self.eta_coords[ieta]:.2f}'
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def plot_melr_wlr_vs_tau(
        self,
        channel: int = 0,
        fig=None,
        ax=None,
    ):
        melr_v = self.melr()
        wlr_v  = self.wlr()
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.tau_coords, melr_v[channel].mean(axis=0),
                color='purple', label='MELR (η-mean)')
        ax.axhline(0, color='purple', lw=0.6, ls=':')
        ax2 = ax.twinx()
        ax2.plot(self.tau_coords, wlr_v[channel].mean(axis=0),
                 color='teal', linestyle='--', label='WLR (η-mean)')
        ax.set_xlabel('τ (fm/c)')
        ax.set_ylabel('MELR', color='purple')
        ax2.set_ylabel('WLR', color='teal')
        ax.set_title(f'Spectral Energy Metrics — {self.channel_names[channel]}')
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(lines, labels, fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig, ax

    def plot_spectral_error_map(
        self,
        channel: int = 0,
        ieta: int = 0,
        itau: int = -1,
        fig=None,
        ax=None,
    ):
        kx, ky, log_err = self.spectral_error_map(channel, ieta, itau)
        itau_abs = self._resolve_itau(itau)
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(ky, kx, log_err, cmap='inferno', shading='auto')
        plt.colorbar(im, ax=ax, label='log₁₀(|ΔF|²)')
        ax.set_xlabel('k_y')
        ax.set_ylabel('k_x')
        ax.set_title(
            f'Spectral error map — {self.channel_names[channel]}, '
            f'η={self.eta_coords[ieta]:.2f}, '
            f'τ={self.tau_coords[itau_abs]:.3f} fm/c'
        )
        fig.tight_layout()
        return fig, ax

    def plot_energy_spectrum(
        self,
        channel: int = 0,
        ieta: int = 0,
        tau_indices: Optional[List[int]] = None,
        fig=None,
        ax=None,
    ):
        if tau_indices is None:
            tau_indices = [0, self.T // 2, -1]
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(7, 4))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_indices)))
        for color, it in zip(colors, tau_indices):
            it_abs = self._resolve_itau(it)
            K, E_p, E_t = self.energy_spectrum(channel, ieta, it)
            valid = (K > 0) & (E_t > 0) & (E_p > 0)
            tau_str = f'{self.tau_coords[it_abs]:.3f}'
            ax.loglog(K[valid], E_t[valid], color=color, label=f'true τ={tau_str}')
            ax.loglog(K[valid], E_p[valid], color=color, linestyle='--',
                      alpha=0.7, label=f'pred τ={tau_str}')
        ax.set_xlabel('Wavenumber K')
        ax.set_ylabel('E(K)')
        ax.set_title(
            f'Energy spectrum — {self.channel_names[channel]}, '
            f'η={self.eta_coords[ieta]:.2f}'
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, which='both')
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------ #
    #  Dashboard                                                           #
    # ------------------------------------------------------------------ #

    def plot_dashboard(
        self,
        channel: int = 0,
        ieta: int = 0,
        tau_indices: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (19, 11),
    ) -> plt.Figure:
        """
        2×3 composite figure:
          row 0: nRMSE vs τ | fRMSE vs τ | MELR & WLR vs τ
          row 1: spectral error map (last τ) | energy spectrum | summary table
        """
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

        self.plot_nrmse_vs_tau(channel, fig=fig, ax=fig.add_subplot(gs[0, 0]))
        self.plot_frmse_vs_tau(channel, ieta, fig=fig, ax=fig.add_subplot(gs[0, 1]))
        self.plot_melr_wlr_vs_tau(channel, fig=fig, ax=fig.add_subplot(gs[0, 2]))
        self.plot_spectral_error_map(
            channel, ieta, itau=-1, fig=fig, ax=fig.add_subplot(gs[1, 0])
        )
        self.plot_energy_spectrum(
            channel, ieta, tau_indices=tau_indices,
            fig=fig, ax=fig.add_subplot(gs[1, 1])
        )

        ax_tbl = fig.add_subplot(gs[1, 2])
        ax_tbl.axis('off')
        s = self.summary()
        tbl = ax_tbl.table(
            cellText=[[k, f'{v:.4g}'] for k, v in s.items()],
            colLabels=['Metric', 'Value (mean)'],
            loc='center',
            cellLoc='left',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        ax_tbl.set_title('Summary (mean over C, η, τ)', pad=8, fontsize=10)

        fig.suptitle(
            f'FNO Evaluation — {self.channel_names[channel]}, '
            f'η={self.eta_coords[ieta]:.2f} | '
            f'N={self.N} events',
            fontsize=13,
        )
        return fig
