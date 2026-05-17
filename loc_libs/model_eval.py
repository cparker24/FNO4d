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

Backends
--------
    backend='numpy'  (default) — original float64 NumPy path; exact reference.
    backend='torch'            — PyTorch path; auto-selects CUDA → MPS → CPU.
        All heavy ops (FFT, reductions, radial binning) run in float32 on the
        accelerator.  Radial shell accumulation uses a one-hot projection
        matrix (matmul) instead of the Python shell loop, giving 10–50× speedup
        for N > 10.  Results are returned as float64 numpy arrays.

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

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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
    backend : {'numpy', 'torch'}
        Compute backend.  'numpy' (default) uses the original float64 NumPy
        path.  'torch' dispatches all heavy ops to the best available device
        (CUDA → MPS → CPU) using float32.  All public methods still return
        float64 numpy arrays regardless of backend.
    device : str or None
        Override the auto-selected torch device ('cpu', 'cuda', 'mps').
        Ignored when backend='numpy'.
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
        backend: str = 'numpy',
        device: Optional[str] = None,
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

        # lazy numpy caches
        self._F_pred: Optional[np.ndarray] = None
        self._F_true: Optional[np.ndarray] = None
        self._E_pred: Optional[np.ndarray] = None  # (N, C, neta, T, n_shells)
        self._E_true: Optional[np.ndarray] = None

        # ------------------------------------------------------------------ #
        #  Torch backend setup                                                #
        # ------------------------------------------------------------------ #
        if backend not in ('numpy', 'torch'):
            raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")
        self.backend = backend

        if backend == 'torch':
            if not _TORCH_AVAILABLE:
                raise ImportError(
                    "backend='torch' requires PyTorch. Install it and try again."
                )
            th = _torch
            if device is None:
                if th.cuda.is_available():
                    device = 'cuda'
                elif th.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            self._device = th.device(device)
            print(f'ModelEval: torch backend — device={self._device}')

            # Shell projection matrix: (n_pix, n_shells) one-hot encoding of
            # which radial shell each pixel belongs to.  Used in
            # _radial_energies_torch() as a single matmul instead of a Python
            # loop, giving 10–50× speedup over the numpy shell loop.
            n_pix    = self.nx * (self.ny // 2 + 1)
            n_shells = len(self._k_shells)
            k_bin = np.round(self._kmap).astype(np.int64).clip(0, n_shells - 1).ravel()
            k_bin_t = th.tensor(k_bin, device=self._device, dtype=th.long)
            self._shell_proj = th.zeros(
                n_pix, n_shells, device=self._device, dtype=th.float32
            )
            self._shell_proj.scatter_(1, k_bin_t.unsqueeze(1), 1.0)

            # Float32 tensors on device (converted once at init)
            self._pred_t = th.tensor(self.pred, dtype=th.float32, device=self._device)
            self._true_t = th.tensor(self.true, dtype=th.float32, device=self._device)

            if self._active_mask is not None:
                self._active_mask_t = th.tensor(
                    self._active_mask, dtype=th.float32, device=self._device
                )
            else:
                self._active_mask_t = None

            # lazy torch caches
            self._F_pred_t = None
            self._F_true_t = None
            self._E_pred_t = None
            self._E_true_t = None
        else:
            self._device = None

    # ------------------------------------------------------------------ #
    #  Internal helpers — numpy (original)                                 #
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
    #  Internal helpers — torch (GPU-accelerated)                          #
    # ------------------------------------------------------------------ #

    def _ffts_torch(self):
        """rfft2 on device in float32 → complex64 tensors (N,C,nx,ny//2+1,neta,T)."""
        if self._F_pred_t is None:
            import warnings
            th = _torch
            if self._active_mask_t is not None:
                p = self._pred_t * self._active_mask_t
                t = self._true_t * self._active_mask_t
            else:
                p, t = self._pred_t, self._true_t
            # PyTorch dev/MPS builds emit a spurious resize warning from inside ATen's
            # rfft2 path.  The output shape is correct; suppress the noise.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='An output with one or more elements was resized',
                    category=UserWarning,
                )
                self._F_pred_t = th.fft.rfft2(p, dim=(2, 3))
                self._F_true_t = th.fft.rfft2(t, dim=(2, 3))
        return self._F_pred_t, self._F_true_t

    def _radial_energies_torch(self):
        """
        Vectorised radial shell accumulation via a one-hot projection matmul.

        The Python shell loop in the numpy path is replaced by a single matmul:
            E[b, s] = sum_{j: k_bin[j]==s} P[b, j]
                    = P_flat[b] @ shell_proj[:, s]

        Returns E_pred, E_true as float32 tensors (N, C, neta, T, n_shells)
        on the device.
        """
        if self._E_pred_t is not None:
            return self._E_pred_t, self._E_true_t

        F_pred, F_true = self._ffts_torch()
        Pp = 0.5 * F_pred.abs() ** 2   # (N, C, nx, ny//2+1, neta, T)
        Pt = 0.5 * F_true.abs() ** 2

        # Parseval correction (same logic as numpy path)
        if self.ny % 2 == 0:
            Pp[:, :, :, 1:-1, :, :] *= 2
            Pt[:, :, :, 1:-1, :, :] *= 2
        else:
            Pp[:, :, :, 1:, :, :] *= 2
            Pt[:, :, :, 1:, :, :] *= 2

        n_pix    = self.nx * (self.ny // 2 + 1)
        n_shells = len(self._k_shells)
        batch    = self.N * self.C * self.neta * self.T

        # Reshape: (N,C,nx,ny_h,neta,T) → (N,C,neta,T,n_pix) → (batch, n_pix)
        Pp_b = Pp.reshape(self.N, self.C, n_pix, self.neta, self.T) \
                 .permute(0, 1, 3, 4, 2).reshape(batch, n_pix)
        Pt_b = Pt.reshape(self.N, self.C, n_pix, self.neta, self.T) \
                 .permute(0, 1, 3, 4, 2).reshape(batch, n_pix)

        # Single matmul replaces the Python shell loop
        E_p = (Pp_b @ self._shell_proj).reshape(self.N, self.C, self.neta, self.T, n_shells)
        E_t = (Pt_b @ self._shell_proj).reshape(self.N, self.C, self.neta, self.T, n_shells)

        self._E_pred_t = E_p
        self._E_true_t = E_t
        return E_p, E_t

    # ------------------------------------------------------------------ #
    #  Pointwise metrics — torch variants  →  (C, neta, T)               #
    # ------------------------------------------------------------------ #

    def _nrmse_torch(self) -> np.ndarray:
        th = _torch
        if self._active_mask_t is not None:
            mf      = self._active_mask_t
            n_act   = th.clamp(mf.sum(dim=(2, 3)), min=1)
            diff_sq = ((self._pred_t - self._true_t) ** 2 * mf).sum(dim=(2, 3)) / n_act
            true_sq = (self._true_t ** 2 * mf).sum(dim=(2, 3)) / n_act
        else:
            diff_sq = (self._pred_t - self._true_t).pow(2).mean(dim=(2, 3))
            true_sq = self._true_t.pow(2).mean(dim=(2, 3))
        per_sample = (diff_sq / th.clamp(true_sq, min=self.eps)).sqrt()
        return per_sample.mean(dim=0).cpu().numpy().astype(np.float64)

    def _rmse_torch(self) -> np.ndarray:
        return (self._pred_t - self._true_t).pow(2).mean(dim=(0, 2, 3)).sqrt() \
               .cpu().numpy().astype(np.float64)

    def _max_error_torch(self) -> np.ndarray:
        return (self._pred_t - self._true_t).abs().amax(dim=(0, 2, 3)) \
               .cpu().numpy().astype(np.float64)

    def _brmse_torch(self) -> np.ndarray:
        th = _torch
        flat_mask = th.tensor(
            _boundary_mask(self.nx, self.ny).reshape(-1), device=self._device
        )
        p = self._pred_t.reshape(self.N, self.C, -1, self.neta, self.T)[:, :, flat_mask]
        t = self._true_t.reshape(self.N, self.C, -1, self.neta, self.T)[:, :, flat_mask]
        return (p - t).pow(2).mean(dim=(0, 2)).sqrt() \
               .cpu().numpy().astype(np.float64)

    def _crmse_torch(self) -> np.ndarray:
        th = _torch
        if self._active_mask_t is not None:
            mf        = self._active_mask_t
            int_p     = (self._pred_t * mf).sum(dim=(2, 3, 4))
            int_t     = (self._true_t * mf).sum(dim=(2, 3, 4))
            int_abs_t = (self._true_t.abs() * mf).sum(dim=(2, 3, 4))
        else:
            int_p     = self._pred_t.sum(dim=(2, 3, 4))
            int_t     = self._true_t.sum(dim=(2, 3, 4))
            int_abs_t = self._true_t.abs().sum(dim=(2, 3, 4))
        denom = th.maximum(int_t.abs(), 0.01 * int_abs_t) + self.eps
        rel   = (int_p - int_t).abs() / denom
        return rel.mean(dim=0).cpu().numpy().astype(np.float64)

    def _vrmse_torch(self) -> np.ndarray:
        th = _torch
        if self._active_mask_t is not None:
            mf    = self._active_mask_t
            n_act = th.clamp(mf.sum(dim=(2, 3)), min=1)
            mse   = ((self._pred_t - self._true_t) ** 2 * mf).sum(dim=(2, 3)) / n_act
            mu    = (self._true_t * mf).sum(dim=(2, 3)) / n_act
            mu2   = (self._true_t ** 2 * mf).sum(dim=(2, 3)) / n_act
            var   = mu2 - mu ** 2
        else:
            mse = (self._pred_t - self._true_t).pow(2).mean(dim=(2, 3))
            var = self._true_t.var(dim=(2, 3))
        per_sample = (mse / th.clamp(var, min=self.eps)).sqrt()
        return per_sample.mean(dim=0).cpu().numpy().astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Spectral metrics — torch variants                                   #
    # ------------------------------------------------------------------ #

    def _frmse_torch(self) -> Dict[str, np.ndarray]:
        th = _torch
        F_pred, F_true = self._ffts_torch()
        diff_F = F_pred - F_true

        k_max  = float(self._kmap.max())
        kmap_t = th.tensor(self._kmap, device=self._device, dtype=th.float32)
        bands  = {
            'low':  (0.0,         k_max / 3.0),
            'mid':  (k_max / 3.0, 2.0 * k_max / 3.0),
            'high': (2.0 * k_max / 3.0, k_max + 1.0),
        }

        result: Dict[str, np.ndarray] = {}
        for name, (klo, khi) in bands.items():
            band_mask = (kmap_t >= klo) & (kmap_t < khi)   # (nx, ny_h) bool
            d   = diff_F[:, :, band_mask, :, :]             # (N,C,n_pix,neta,T)
            t   = F_true[:, :, band_mask, :, :]
            num = d.abs().pow(2).mean(dim=0).sum(dim=1)     # (C, neta, T)
            den = t.abs().pow(2).mean(dim=0).sum(dim=1)
            result[name] = (num / th.clamp(den, min=self.eps)).sqrt() \
                           .cpu().numpy().astype(np.float64)
        return result

    def _melr_torch(self) -> np.ndarray:
        th = _torch
        E_p, E_t = self._radial_energies_torch()   # (N, C, neta, T, n_shells)
        K = th.tensor(self._k_shells, device=self._device, dtype=th.float32)

        valid = (E_t > self.eps) & (E_p > self.eps) & (K > 0)
        ratio = th.where(valid, E_p / th.clamp(E_t, min=self.eps), th.ones_like(E_p))
        log_r = th.where(valid, ratio.log(), th.zeros_like(E_p))
        w     = th.where(valid, K, th.zeros_like(K))

        w_sum  = w.sum(dim=-1)
        melr_n = (w * log_r).sum(dim=-1) / th.clamp(w_sum, min=self.eps)
        return melr_n.mean(dim=0).cpu().numpy().astype(np.float64)

    def _wlr_torch(self) -> np.ndarray:
        th = _torch
        E_p, E_t = self._radial_energies_torch()

        sp     = E_p.sum(dim=-1, keepdim=True)
        st     = E_t.sum(dim=-1, keepdim=True)
        p_norm = E_p / th.clamp(sp, min=self.eps)
        q_norm = E_t / th.clamp(st, min=self.eps)

        wlr_n = (p_norm.cumsum(dim=-1) - q_norm.cumsum(dim=-1)).abs().sum(dim=-1)
        return wlr_n.mean(dim=0).cpu().numpy().astype(np.float64)

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
        if self.backend == 'torch':
            return self._nrmse_torch()
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
        if self.backend == 'torch':
            return self._rmse_torch()
        return np.sqrt(np.mean((self.pred - self.true) ** 2, axis=(0, 2, 3)))

    def max_error(self) -> np.ndarray:
        """Max |pred − true| over events and (x, y).  Shape (C, neta, T)."""
        if self.backend == 'torch':
            return self._max_error_torch()
        return np.max(np.abs(self.pred - self.true), axis=(0, 2, 3))

    def brmse(self) -> np.ndarray:
        """
        Boundary RMSE: RMSE restricted to the (x, y) domain boundary.
        Proxy for freeze-out surface accuracy.  Shape (C, neta, T).
        """
        if self.backend == 'torch':
            return self._brmse_torch()
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
        if self.backend == 'torch':
            return self._crmse_torch()
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
        if self.backend == 'torch':
            return self._vrmse_torch()
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
        if self.backend == 'torch':
            return self._frmse_torch()
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
        if self.backend == 'torch':
            return self._melr_torch()
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
        if self.backend == 'torch':
            return self._wlr_torch()
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
        if self.backend == 'torch':
            F_pred_t, F_true_t = self._ffts_torch()
            F_pred = F_pred_t.cpu().numpy()
            F_true = F_true_t.cpu().numpy()
        else:
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
        if self.backend == 'torch':
            E_p_t, E_t_t = self._radial_energies_torch()
            E_p = E_p_t.cpu().numpy().astype(np.float64)
            E_t = E_t_t.cpu().numpy().astype(np.float64)
        else:
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
        itau: int = -1,
        figsize: Tuple[int, int] = (19, 11),
    ) -> plt.Figure:
        """
        2×3 composite figure:
          row 0: nRMSE vs τ | fRMSE vs τ | MELR & WLR vs τ
          row 1: spectral error map (itau) | energy spectrum | summary table

        Parameters
        ----------
        itau : int
            τ step index for the spectral error map (bottom-left panel).
            Negative indices wrap from the end.  Default -1 (last step).
        """
        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

        self.plot_nrmse_vs_tau(channel, fig=fig, ax=fig.add_subplot(gs[0, 0]))
        self.plot_frmse_vs_tau(channel, ieta, fig=fig, ax=fig.add_subplot(gs[0, 1]))
        self.plot_melr_wlr_vs_tau(channel, fig=fig, ax=fig.add_subplot(gs[0, 2]))
        self.plot_spectral_error_map(
            channel, ieta, itau=itau, fig=fig, ax=fig.add_subplot(gs[1, 0])
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
