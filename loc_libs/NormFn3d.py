"""
NormFn3d.py

Normalisation helpers for 3-D hydro arrays with shape
    (n_events, n_features, nx, ny, n_eta, n_tau)
where feature 0 = energy density (ed) and features 1+ = vx, vy, vz
(passed through unchanged by every function here).

Four normalisation families:

  etau75_3d             — ed → ed^0.75 * τ
                          (tau only; no η correction)
                          Helper: —

  etau75_eta_3d         — ed → (ed^0.75 * τ) / ref[ieta, itau]
                          (tau + data-driven per-(η,τ)-bin absolute rescaling)
                          Helper: compute_ref_etau75_eta  — build ref from training data

  cosh_etau75_3d        — ed → (ed · cosh(η)^α)^0.75 · τ
                          (tau + analytic cosh(η) correction, no training data needed)
                          Helper: —

  etau75_eta_shape_3d   — ed → (ed^0.75 * τ) / shape_ref[ieta, itau]
                          (tau + shape-only η correction; shape_ref ∈ (0,1] so the
                          Bjorken tau evolution is preserved at the peak η bin while
                          the relative η dynamic range is equalised across bins)
                          Helper: compute_ref_etau75_eta_shape  — build shape_ref from training data

All families supply a matching inverse (inv_*).

Choosing a family
-----------------
  • etau75_3d              — simplest baseline; η variation fully visible to the network.
  • etau75_eta_3d          — full absolute rescaling; every (η,τ) cell arrives O(1),
                             but the Bjorken tau evolution is absorbed into the normalisation.
  • cosh_etau75_3d         — analytic η correction; needs no training data, assumes a
                             cosh(η) longitudinal profile.
  • etau75_eta_shape_3d    — shape-only η correction; the network still sees the Bjorken
                             tau evolution while the η dynamic range is equalised.

Broadcasting convention
-----------------------
tau       is reshaped to (1, 1, 1, 1, nT)    so it broadcasts against data[:, 0]
eta       is reshaped to (1, 1, 1, n_eta, 1) so it broadcasts against data[:, 0]
ref / shape_ref  are indexed as [..., np.newaxis, np.newaxis, np.newaxis, :, :] = (1,1,1,neta,ntau)
"""

import numpy as np


class NormFn3d:

    # =========================================================================
    # 1. etau75_3d — tau scaling only
    # =========================================================================

    @staticmethod
    def etau75_3d(data: np.ndarray,
                  tau_min: float,
                  dtau: float,
                  ) -> np.ndarray:
        """ed → ed^0.75 * τ  (tau-only normalisation).

        Motivated by ideal Bjorken flow where e(τ) ~ τ^{-4/3}, so
        (e · τ^{4/3})^{3/4} = e^{3/4} · τ is approximately constant in time.
        No correction is applied along the η axis.

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]

        Returns
        -------
        out : same shape as *data*; feature 0 = ed^0.75 · τ, others unchanged.
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)
        out = data.copy()
        out[:, 0] = data[:, 0] ** 0.75 * tau
        return out

    @staticmethod
    def inv_etau75_3d(data: np.ndarray,
                      tau_min: float,
                      dtau: float,
                      eps: float = 1e-6,
                      ) -> np.ndarray:
        """Inverse of etau75_3d.  ed = (ed_norm / τ)^{4/3}.

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)  — normalised
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]
        eps     : small additive floor on τ to guard against division by zero

        Returns
        -------
        out : same shape as *data* with physical energy density restored.
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)
        out = data.copy()
        out[:, 0] = (data[:, 0] / (tau + eps)) ** (1.0 / 0.75)
        return out

    # =========================================================================
    # 2. etau75_eta_3d — tau scaling + data-driven per-(η,τ) rescaling
    # =========================================================================

    @staticmethod
    def compute_ref_etau75_eta(train_data: np.ndarray,
                               tau_min: float,
                               dtau: float,
                               eps: float = 1e-6,
                               ) -> np.ndarray:
        """Build the (n_eta, n_tau) reference map from the *training split only*.

        For every (η bin, τ bin) the reference is the mean, over training
        events, of the spatial maximum of ed^0.75 · τ:

            ref[ieta, itau] = mean_events( max_{x,y}( ed^0.75 · τ ) )

        Workflow
        --------
        1.  Call this function once on your training events and save the result:
                ref = NormFn3d.compute_ref_etau75_eta(train_arr, tau_min, dtau)
                np.save('ref_etau75_eta.npy', ref)

        2.  Normalise every split with the *same* ref:
                train_norm, _ = NormFn3d.etau75_eta_3d(train_arr, tau_min, dtau, ref=ref)
                val_norm,   _ = NormFn3d.etau75_eta_3d(val_arr,   tau_min, dtau, ref=ref)

        3.  At inference time, load ref and call inv_etau75_eta_3d to recover
            physical units.

        Parameters
        ----------
        train_data : (n_train, n_features, nx, ny, n_eta, n_tau)
        tau_min    : proper-time value at tau bin 0  [fm/c]
        dtau       : proper-time step size            [fm/c]
        eps        : floor applied to ref to prevent zero-ref bins
                     (can arise at large |η| or early τ with few active cells)

        Returns
        -------
        ref : (n_eta, n_tau) normalisation map
        """
        nT  = train_data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)

        # ed^0.75 * τ  →  (n_train, nx, ny, n_eta, n_tau)
        ed_scaled = train_data[:, 0] ** 0.75 * tau

        # max over (nx, ny)  →  (n_train, n_eta, n_tau)
        # mean over events   →  (n_eta,   n_tau)
        ref = ed_scaled.max(axis=(1, 2)).mean(axis=0)
        return np.maximum(ref, eps)

    @staticmethod
    def etau75_eta_3d(data: np.ndarray,
                      tau_min: float,
                      dtau: float,
                      ref: np.ndarray = None,
                      eps: float = 1e-6,
                      ) -> tuple:
        """ed → (ed^0.75 · τ) / ref[ieta, itau]  (tau + data-driven eta).

        Makes every (η, τ) slice arrive at the network with an O(1) typical
        scale while preserving the transverse x-y pattern the FNO must learn.

        If *ref* is None it is computed from *data* on the fly — convenient for
        quick inspection, but not suitable for training/inference workflows
        where the same ref must be shared across all data splits.  In that
        case, call compute_ref_etau75_eta on the training split and pass the
        result here for every split.

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]
        ref     : (n_eta, n_tau) from compute_ref_etau75_eta, or None
        eps     : small floor on ref to guard against division by zero

        Returns
        -------
        out : normalised array, same shape as *data*
        ref : (n_eta, n_tau) — save alongside the model for inversion
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)

        out = data.copy()
        out[:, 0] = data[:, 0] ** 0.75 * tau            # (n_events, nx, ny, n_eta, n_tau)

        if ref is None:
            ref = out[:, 0].max(axis=(1, 2)).mean(axis=0)   # (n_eta, n_tau)
            ref = np.maximum(ref, eps)

        out[:, 0] /= ref[np.newaxis, np.newaxis, np.newaxis, :, :]
        return out, ref

    @staticmethod
    def inv_etau75_eta_3d(data: np.ndarray,
                          ref: np.ndarray,
                          tau_min: float,
                          dtau: float,
                          eps: float = 1e-6,
                          ) -> np.ndarray:
        """Inverse of etau75_eta_3d.

        ed = ( ed_norm · ref[ieta, itau] / τ )^{4/3}

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)  — normalised
        ref     : (n_eta, n_tau) saved during (or returned by) the forward pass
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]
        eps     : small additive floor on τ to guard against division by zero

        Returns
        -------
        out : same shape as *data* with physical energy density restored.
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)

        out = data.copy()
        ed_rescaled = data[:, 0] * ref[np.newaxis, np.newaxis, np.newaxis, :, :]
        out[:, 0]   = (ed_rescaled / (tau + eps)) ** (1.0 / 0.75)
        return out

    # =========================================================================
    # 3. cosh_etau75_3d — tau scaling + analytic cosh(η) correction
    # =========================================================================

    @staticmethod
    def cosh_etau75_3d(data: np.ndarray,
                       tau_min: float,
                       dtau: float,
                       eta_min: float,
                       deta: float,
                       alpha: float = 1.0,
                       ) -> np.ndarray:
        """ed → (ed · cosh(η)^α)^0.75 · τ  (tau + analytic eta correction).

        Physical motivation
        -------------------
        In Milne coordinates (τ, x, y, η_s) the longitudinal energy-density
        profile decreases away from mid-rapidity roughly as ~ 1/cosh(η_s)
        due to the finite beam rapidity and the frame-transformation Jacobian.
        Multiplying ed by cosh(η)^α before the Bjorken rescaling compensates
        this suppression analytically, requiring no training data.

        Recommended choices for α
        -------------------------
        α = 1.0 (default)
            Removes the leading-order cosh fall-off.  Use when you want the
            normalised field to be approximately flat across η bins.

        α = 4/3
            Absorbs a longitudinal boost factor into the coordinate
            transformation, making the pre-factor (e · cosh(η))^{3/4} · τ
            directly analogous to the Bjorken constant of motion.

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]
        eta_min : spatial rapidity value at eta bin 0
        deta    : spatial rapidity step size
        alpha   : exponent on cosh(η); physically motivated choices are 1 or 4/3

        Returns
        -------
        out : same shape as *data*;
              feature 0 = (ed · cosh(η)^α)^0.75 · τ, others unchanged.
        """
        n_eta = data.shape[4]
        nT    = data.shape[5]
        eta = (np.arange(n_eta) * deta  + eta_min).reshape(1, 1, 1, n_eta, 1)
        tau = (np.arange(nT)    * dtau  + tau_min).reshape(1, 1, 1, 1,    nT)

        out = data.copy()
        out[:, 0] = (data[:, 0] * np.cosh(eta) ** alpha) ** 0.75 * tau
        return out

    @staticmethod
    def inv_cosh_etau75_3d(data: np.ndarray,
                           tau_min: float,
                           dtau: float,
                           eta_min: float,
                           deta: float,
                           alpha: float = 1.0,
                           eps: float = 1e-6,
                           ) -> np.ndarray:
        """Inverse of cosh_etau75_3d.

        ed = ( ed_norm / τ )^{4/3} / cosh(η)^α

        *alpha* must match the value used in the forward pass.

        Parameters
        ----------
        data    : (n_events, n_features, nx, ny, n_eta, n_tau)  — normalised
        tau_min : proper-time value at tau bin 0  [fm/c]
        dtau    : proper-time step size            [fm/c]
        eta_min : spatial rapidity value at eta bin 0
        deta    : spatial rapidity step size
        alpha   : same exponent used during cosh_etau75_3d
        eps     : small additive floor on τ to guard against division by zero

        Returns
        -------
        out : same shape as *data* with physical energy density restored.
        """
        n_eta = data.shape[4]
        nT    = data.shape[5]
        eta = (np.arange(n_eta) * deta  + eta_min).reshape(1, 1, 1, n_eta, 1)
        tau = (np.arange(nT)    * dtau  + tau_min).reshape(1, 1, 1, 1,    nT)

        out = data.copy()
        out[:, 0] = (data[:, 0] / (tau + eps)) ** (1.0 / 0.75) / np.cosh(eta) ** alpha
        return out

    # =========================================================================
    # 4. etau75_eta_shape_3d — tau scaling + shape-only η correction
    # =========================================================================

    @staticmethod
    def compute_ref_etau75_eta_shape(train_data: np.ndarray,
                                     tau_min: float,
                                     dtau: float,
                                     eps: float = 1e-6,
                                     ) -> np.ndarray:
        """Build the (n_eta, n_tau) shape-only reference from the *training split only*.

        The shape reference is the η-profile of ed^0.75·τ, normalised at each
        τ step so its peak across η bins equals 1:

            raw_ref[ieta, itau]   = mean_events( max_{x,y}( ed^0.75 · τ ) )
            shape_ref[ieta, itau] = raw_ref[ieta, itau] / max_over_η( raw_ref[:, itau] )

        shape_ref ∈ (0, 1] by construction:
          • 1 at whichever η bin has the highest mean-max energy at that τ step
            (typically near midrapidity, η ≈ 0)
          • smaller values at other η bins, encoding the relative η shape

        Unlike compute_ref_etau75_eta, this reference carries no absolute
        scale information — it only describes how the η profile varies
        relative to the peak at each τ.  The Bjorken tau evolution
        ed^0.75·τ is therefore fully preserved at the peak η bin.

        Workflow
        --------
        1.  Compute once from training data and save:
                shape_ref = NormFn3d.compute_ref_etau75_eta_shape(train_arr, tau_min, dtau)
                np.save('shape_ref_etau75_eta.npy', shape_ref)

        2.  Apply the same shape_ref to all splits:
                train_norm, _ = NormFn3d.etau75_eta_shape_3d(train_arr, tau_min, dtau, shape_ref)
                val_norm,   _ = NormFn3d.etau75_eta_shape_3d(val_arr,   tau_min, dtau, shape_ref)

        3.  Invert with the same shape_ref at inference time.

        Parameters
        ----------
        train_data : (n_train, n_features, nx, ny, n_eta, n_tau)
        tau_min    : proper-time value at tau bin 0  [fm/c]
        dtau       : proper-time step size            [fm/c]
        eps        : floor applied after normalisation to prevent zero-ref bins

        Returns
        -------
        shape_ref : (n_eta, n_tau), values in (0, 1]
        """
        raw_ref   = NormFn3d.compute_ref_etau75_eta(train_data, tau_min, dtau, eps)
        ref_peak  = raw_ref.max(axis=0, keepdims=True)           # (1, n_tau)
        shape_ref = raw_ref / np.maximum(ref_peak, eps)          # (n_eta, n_tau) ∈ (0, 1]
        return np.maximum(shape_ref, eps)

    @staticmethod
    def etau75_eta_shape_3d(data: np.ndarray,
                            tau_min: float,
                            dtau: float,
                            shape_ref: np.ndarray = None,
                            eps: float = 1e-6,
                            ) -> tuple:
        """ed → (ed^0.75 · τ) / shape_ref[ieta, itau]  (tau + shape-only η correction).

        Comparison with etau75_eta_3d
        ------------------------------
        etau75_eta_3d divides by ref[ieta, itau], which carries the full
        absolute scale at each (η, τ) cell.  The output is O(1) everywhere
        but the Bjorken tau evolution is absorbed into the normalisation.

        etau75_eta_shape_3d divides by shape_ref[ieta, itau] ∈ (0, 1], which
        encodes only the *relative* η profile shape at each τ.  Because
        shape_ref = 1 at the peak η bin, the absolute ed^0.75·τ scale is
        preserved there; all other η bins are amplified to match that same
        scale.  The Bjorken tau evolution therefore remains visible in the
        normalised output, while the η dynamic range across bins is equalised.

        Use this when
        -------------
        • the FNO should still "see" the Bjorken tau evolution of the energy
          (learn time dynamics, not just the spatial pattern at fixed time)
        • you only want to remove the variation of the η shape with τ
          (e.g. narrowing / widening of the rapidity plateau over time)

        Parameters
        ----------
        data      : (n_events, n_features, nx, ny, n_eta, n_tau)
        tau_min   : proper-time value at tau bin 0  [fm/c]
        dtau      : proper-time step size            [fm/c]
        shape_ref : (n_eta, n_tau) from compute_ref_etau75_eta_shape, or None.
                    If None, shape_ref is computed from *data* — fine for
                    inspection, not suitable for training workflows.
        eps       : small floor on shape_ref to prevent division by zero

        Returns
        -------
        out       : normalised array, same shape as *data*
        shape_ref : (n_eta, n_tau) ∈ (0, 1] — save alongside the model for inversion
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)

        out = data.copy()
        out[:, 0] = data[:, 0] ** 0.75 * tau

        if shape_ref is None:
            raw_ref   = out[:, 0].max(axis=(1, 2)).mean(axis=0)     # (n_eta, n_tau)
            raw_ref   = np.maximum(raw_ref, eps)
            ref_peak  = raw_ref.max(axis=0, keepdims=True)           # (1, n_tau)
            shape_ref = np.maximum(raw_ref / np.maximum(ref_peak, eps), eps)

        out[:, 0] /= shape_ref[np.newaxis, np.newaxis, np.newaxis, :, :]
        return out, shape_ref

    @staticmethod
    def inv_etau75_eta_shape_3d(data: np.ndarray,
                                shape_ref: np.ndarray,
                                tau_min: float,
                                dtau: float,
                                eps: float = 1e-6,
                                ) -> np.ndarray:
        """Inverse of etau75_eta_shape_3d.

        ed = ( ed_norm · shape_ref[ieta, itau] / τ )^{4/3}

        Parameters
        ----------
        data      : (n_events, n_features, nx, ny, n_eta, n_tau)  — normalised
        shape_ref : (n_eta, n_tau) ∈ (0, 1] saved during the forward pass
        tau_min   : proper-time value at tau bin 0  [fm/c]
        dtau      : proper-time step size            [fm/c]
        eps       : small additive floor on τ to guard against division by zero

        Returns
        -------
        out : same shape as *data* with physical energy density restored.
        """
        nT  = data.shape[-1]
        tau = (np.arange(nT) * dtau + tau_min).reshape(1, 1, 1, 1, nT)

        out = data.copy()
        ed_rescaled = data[:, 0] * shape_ref[np.newaxis, np.newaxis, np.newaxis, :, :]
        out[:, 0]   = (ed_rescaled / (tau + eps)) ** (1.0 / 0.75)
        return out
