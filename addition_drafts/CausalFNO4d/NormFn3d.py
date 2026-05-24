"""
NormFn3d.py

Normalisation helpers for 3-D hydro arrays with shape
    (n_events, n_features, nx, ny, n_eta, n_tau)
where feature 0 = energy density (ed) and features 1+ = vx, vy, vz.
The Bjorken families below operate only on feature 0; velocity features
pass through unchanged.  The z-score family at the end operates on every
feature.

Bjorken-style ed rescaling families
-----------------------------------

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

All Bjorken families supply a matching inverse (inv_*).

Choosing a Bjorken family
-------------------------
  • etau75_3d              — simplest baseline; η variation fully visible to the network.
  • etau75_eta_3d          — full absolute rescaling; every (η,τ) cell arrives O(1),
                             but the Bjorken tau evolution is absorbed into the normalisation.
  • cosh_etau75_3d         — analytic η correction; needs no training data, assumes a
                             cosh(η) longitudinal profile.
  • etau75_eta_shape_3d    — shape-only η correction; the network still sees the Bjorken
                             tau evolution while the η dynamic range is equalised.

Per-(feature, τ) z-score family
-------------------------------

  apply_zscore_3d       — for every (feature c, τ bin t):  data → (data - μ[c,t]) / σ[c,t]
                          Helper: compute_zscore_stats  — build (μ, σ) from training data
                          Inverse: inv_zscore_3d

τ embedding helper
------------------

  tau_sinusoidal_embedding — produce an (n_emb, n_tau) array of sinusoidal positional
                             encodings of τ.  Broadcast and concatenate as extra channels
                             so the network has explicit access to absolute τ after the
                             z-score step has removed it from the data.

Recommended composition for CausalFNO4D
---------------------------------------
CausalFNO4D uses time-translation-invariant Conv1d filters along τ, so it benefits
from inputs that are roughly stationary in τ.  A three-step pipeline:

    1. Bjorken rescaling on ed   (any family above, e.g. etau75_eta_shape_3d)
    2. Per-(feature, τ) z-score  (apply_zscore_3d using stats from training data)
    3. τ sinusoidal embedding    (tau_sinusoidal_embedding, concatenated as channels)

Step 1 removes the leading τ trend in ed; step 2 cleans up any residual non-stationarity
and brings velocity features into the same numerical range; step 3 restores absolute-τ
information as a side channel after step 2 has whitened it out of the data.

Loss in physical units
----------------------
Always invert steps 2 and 1 before computing the data-fit loss, so the optimiser
targets physically meaningful errors rather than normalised ones.  PINO-style PDE
residual losses (∂_μ T^{μν} = 0) MUST be evaluated in the original physical frame —
the τ-dependent rescaling introduces spurious source terms in the rescaled PDE.

Broadcasting convention
-----------------------
tau         is reshaped to (1, 1, 1, 1, nT)             so it broadcasts against data[:, 0]
eta         is reshaped to (1, 1, 1, n_eta, 1)          so it broadcasts against data[:, 0]
ref / shape_ref     are reshaped to (1, 1, 1, n_eta, n_tau)
μ / σ (zscore)      are reshaped to (1, n_features, 1, 1, 1, n_tau)
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

    # =========================================================================
    # 5. Per-(feature, τ) z-score — apply AFTER any Bjorken family above
    # =========================================================================

    @staticmethod
    def compute_zscore_stats(data: np.ndarray,
                             eps: float = 1e-6,
                             ) -> tuple:
        """Compute per-(feature, τ) mean and std from already-Bjorken-rescaled data.

        Motivation
        ----------
        The Bjorken families above flatten the leading τ trend of the energy
        density but leave two sources of non-stationarity in the input that the
        network still has to absorb:

          • Velocity features (vx, vy, vz) pass through the Bjorken step
            unchanged and may have very different magnitudes from the
            Bjorken-rescaled ed, and their own τ-dependent statistics.
          • Even after Bjorken rescaling, ed has a residual mild τ trend from
            transverse expansion, viscosity, and freeze-out, plus a strongly
            τ-dependent spatial dynamic range.

        For an FNO architecture with τ-translation-invariant convolutions along
        the time axis (e.g. CausalFNO4D), feeding non-stationary inputs forces
        the filters to act as gain control on top of the dynamics.  A simple
        per-(feature, τ) z-score removes both effects.

        Workflow
        --------
        1.  Apply a Bjorken family to the training data:
                train_bjorken = NormFn3d.etau75_eta_shape_3d(train_arr, ...)[0]
        2.  Compute statistics from the Bjorken-rescaled training data:
                mu, sigma = NormFn3d.compute_zscore_stats(train_bjorken)
                np.savez('zscore_stats.npz', mu=mu, sigma=sigma)
        3.  Apply the SAME (mu, sigma) to every split:
                train_norm = NormFn3d.apply_zscore_3d(train_bjorken, mu, sigma)
                val_norm   = NormFn3d.apply_zscore_3d(val_bjorken,   mu, sigma)
        4.  At inference, invert z-score first, then the Bjorken family.

        Parameters
        ----------
        data : (n_events, n_features, nx, ny, n_eta, n_tau)
               Already Bjorken-rescaled training data.
        eps  : floor on σ to prevent division by zero for features that are
               identically zero at some τ bin (e.g. velocity at τ_0 if the
               initial condition is at rest).

        Returns
        -------
        mu    : (n_features, n_tau)  — per-(feature, τ) mean
        sigma : (n_features, n_tau)  — per-(feature, τ) std, floored at *eps*
        """
        # Average over events and the three spatial axes; keep feature and τ.
        mu    = data.mean(axis=(0, 2, 3, 4))                 # (n_features, n_tau)
        sigma = data.std (axis=(0, 2, 3, 4))                 # (n_features, n_tau)
        return mu, np.maximum(sigma, eps)

    @staticmethod
    def apply_zscore_3d(data: np.ndarray,
                        mu: np.ndarray,
                        sigma: np.ndarray,
                        ) -> np.ndarray:
        """Apply per-(feature, τ) z-score:  data → (data − μ) / σ.

        Parameters
        ----------
        data  : (n_events, n_features, nx, ny, n_eta, n_tau)
        mu    : (n_features, n_tau)  from compute_zscore_stats
        sigma : (n_features, n_tau)  from compute_zscore_stats

        Returns
        -------
        out : same shape as *data*, with every (feature, τ) slice having
              approximately zero mean and unit variance across the training set.
        """
        mu_b    = mu   [np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :]
        sigma_b = sigma[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :]
        return (data - mu_b) / sigma_b

    @staticmethod
    def inv_zscore_3d(data: np.ndarray,
                      mu: np.ndarray,
                      sigma: np.ndarray,
                      ) -> np.ndarray:
        """Inverse of apply_zscore_3d:  data → data · σ + μ.

        Apply this BEFORE inverting the Bjorken family at inference time.

        Parameters
        ----------
        data  : (n_events, n_features, nx, ny, n_eta, n_tau)  — z-scored
        mu    : (n_features, n_tau)  used in the forward pass
        sigma : (n_features, n_tau)  used in the forward pass

        Returns
        -------
        out : same shape as *data*, in Bjorken-rescaled (but un-zscored) units.
        """
        mu_b    = mu   [np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :]
        sigma_b = sigma[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :]
        return data * sigma_b + mu_b

    # =========================================================================
    # 6. τ sinusoidal embedding — optional extra input channels
    # =========================================================================

    @staticmethod
    def tau_sinusoidal_embedding(n_tau: int,
                                 tau_min: float,
                                 dtau: float,
                                 n_emb: int = 8,
                                 max_period: float = 20.0,
                                 ) -> np.ndarray:
        """Build a sinusoidal positional encoding of τ, shape (n_emb, n_tau).

        Motivation
        ----------
        After per-(feature, τ) z-score (apply_zscore_3d), every τ slice of the
        input is approximately zero-mean and unit-variance — useful for the
        network's optimisation, but it also removes any information about
        *which* τ this slice is.  A τ-translation-invariant CausalFNO4D cannot
        distinguish early-time from late-time regimes once the data has been
        whitened.

        The remedy is to feed τ back in as an explicit auxiliary input.  A
        sinusoidal embedding with multiple frequencies provides a rich
        positional signature for the spectral convolutions to lock onto, much
        like positional encodings in transformers and conditioning embeddings
        in diffusion models.

        Construction
        ------------
        For k = 0, 1, …, n_emb/2 - 1:

            emb[2k,   t] = sin( τ_t / period_k )
            emb[2k+1, t] = cos( τ_t / period_k )

        with period_k = max_period · (1 / max_period)^(2k / n_emb), so the
        smallest period is 1 fm/c and the largest is *max_period* fm/c.
        This covers timescales from the fastest hydro evolution (τ ~ 1 fm/c)
        to the full simulation duration.

        Usage as extra input channels
        -----------------------------
        emb = NormFn3d.tau_sinusoidal_embedding(n_tau, tau_min, dtau, n_emb=8)
        emb_broadcast = np.broadcast_to(
            emb[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :],
            (n_events, n_emb, nx, ny, n_eta, n_tau),
        )
        data_with_tau = np.concatenate([data_norm, emb_broadcast], axis=1)
        # data_with_tau now has shape (n_events, n_features + n_emb, nx, ny, n_eta, n_tau)
        # Set CausalFNO4D's in_channels accordingly.

        Parameters
        ----------
        n_tau      : number of τ bins to encode.
        tau_min    : proper-time at τ bin 0  [fm/c].
        dtau       : proper-time step size    [fm/c].
        n_emb      : embedding dimension; must be even.  8 is a sensible default.
        max_period : longest period to encode, in fm/c.  Set to the full
                     simulation duration or slightly larger.

        Returns
        -------
        emb : (n_emb, n_tau) array of sinusoidal embeddings, dtype float64.
        """
        if n_emb % 2 != 0:
            raise ValueError(f"n_emb must be even, got {n_emb}")

        tau = np.arange(n_tau) * dtau + tau_min                              # (n_tau,)
        # Geometric series of periods from 1.0 fm/c up to max_period:
        k = np.arange(n_emb // 2)
        periods = max_period ** (2.0 * k / n_emb)                            # (n_emb/2,)

        # arg has shape (n_emb/2, n_tau)
        arg = tau[np.newaxis, :] / periods[:, np.newaxis]

        emb = np.empty((n_emb, n_tau), dtype=np.float64)
        emb[0::2] = np.sin(arg)
        emb[1::2] = np.cos(arg)
        return emb

    @staticmethod
    def broadcast_tau_embedding(emb: np.ndarray,
                                n_events: int,
                                nx: int,
                                ny: int,
                                n_eta: int,
                                ) -> np.ndarray:
        """Broadcast a (n_emb, n_tau) embedding to (n_events, n_emb, nx, ny, n_eta, n_tau).

        Convenience for concatenation as extra input channels.  Uses
        np.broadcast_to, which does not allocate new memory — the result is a
        read-only view.  If you need a writable copy, wrap the result in
        np.ascontiguousarray.

        Parameters
        ----------
        emb      : (n_emb, n_tau) from tau_sinusoidal_embedding.
        n_events : batch dimension to broadcast over.
        nx, ny, n_eta : spatial dimensions to broadcast over.

        Returns
        -------
        out : (n_events, n_emb, nx, ny, n_eta, n_tau) view into *emb*.
        """
        n_emb, n_tau = emb.shape
        return np.broadcast_to(
            emb[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, :],
            (n_events, n_emb, nx, ny, n_eta, n_tau),
        )
