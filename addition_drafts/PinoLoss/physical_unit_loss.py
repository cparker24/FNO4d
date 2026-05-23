# ===========================================================================
# NORM: PhysicalUnitLoss (from CauseFNO4d train example draft directory!)
# Wraps any base loss (LpLoss, H1Loss) so that pred and target are
# inverse-normalised back to physical units before the loss is computed.
#
# This ensures:
#   • the optimiser targets physically meaningful errors;
#   • any PINO-style PDE residual term can be appended here in the same frame.
# ===========================================================================

class PhysicalUnitLoss(nn.Module):
    """
    Inverse-normalise (z-score then Bjorken) both prediction and target,
    then evaluate a base loss in physical units.

    The T_out τ steps in pred / target are assumed to correspond to τ indices
    tau_start_idx, tau_start_idx+1, … (default: 1, because index 0 is the IC
    which is not predicted).

    Parameters
    ----------
    base_loss     : callable(pred, target) → scalar — e.g. LpLoss or H1Loss
    mu, sigma     : (n_features, n_tau)  from NormFn3d.compute_zscore_stats
    shape_ref     : (n_eta, n_tau)       from NormFn3d.compute_ref_etau75_eta_shape
    tau_min, dtau : proper-time grid parameters [fm/c]
    tau_start_idx : first τ index present in the prediction tensor (default 1)
    eps           : floor on τ to guard against division by zero
    """

    def __init__(self, base_loss, mu, sigma, shape_ref,
                 tau_min, dtau, tau_start_idx=1, eps=1e-8):
        super().__init__()
        self.register_buffer('mu',        torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer('sigma',     torch.from_numpy(sigma.astype(np.float32)))
        self.register_buffer('shape_ref', torch.from_numpy(shape_ref.astype(np.float32)))
        self.base_loss     = base_loss
        self.tau_min       = tau_min
        self.dtau          = dtau
        self.tau_start_idx = tau_start_idx
        self.eps           = eps

    def invert(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm : (B, F, X, Y, η, T_out) — normalised model output or target
        returns: (B, F, X, Y, η, T_out) — physical units
        """
        B, F, X, Y, n_eta, T_out = x_norm.shape
        t0 = self.tau_start_idx

        # τ values for the output window  (1, 1, 1, 1, 1, T_out)
        tau_idx = torch.arange(t0, t0 + T_out,
                               device=x_norm.device, dtype=x_norm.dtype)
        tau = (self.tau_min + tau_idx * self.dtau).reshape(1, 1, 1, 1, 1, T_out)

        # Per-(feature, τ) stats for the output window  (1, F, 1, 1, 1, T_out)
        mu_w    = self.mu   [:, t0:t0 + T_out].reshape(1, F, 1, 1, 1, T_out)
        sigma_w = self.sigma[:, t0:t0 + T_out].reshape(1, F, 1, 1, 1, T_out)

        # shape_ref for the output window  (1, 1, 1, 1, η, T_out)
        sr_w = self.shape_ref[:, t0:t0 + T_out].reshape(1, 1, 1, 1, n_eta, T_out)

        # Step 1: inverse z-score on ALL features
        x_bj = x_norm * sigma_w + mu_w              # (B, F, X, Y, η, T_out)

        # Step 2: inverse Bjorken on ed (channel 0) only
        ed_bj   = x_bj[:, 0:1]                      # (B, 1, X, Y, η, T_out)
        ed_phys = (ed_bj * sr_w / (tau + self.eps)) ** (1.0 / 0.75)

        return torch.cat([ed_phys, x_bj[:, 1:]], dim=1)

    def forward(self, pred_norm: torch.Tensor,
                      target_norm: torch.Tensor) -> torch.Tensor:
        return self.base_loss(self.invert(pred_norm),
                              self.invert(target_norm))

    # ------------------------------------------------------------------
    # PINO extension point
    # ------------------------------------------------------------------
    # To add a PDE residual loss (∂_μ T^{μν} = 0 in the physical frame),
    # override forward as shown below.  The key constraint is that the
    # residual MUST be evaluated on physical-unit fields, because the
    # τ-dependent Bjorken rescaling introduces spurious source terms
    # in the rescaled frame.
    #
    #   def forward(self, pred_norm, target_norm, lambda_pde=1e-3):
    #       pred_phys   = self.invert(pred_norm)
    #       target_phys = self.invert(target_norm)
    #       loss_data   = self.base_loss(pred_phys, target_phys)
    #       loss_pde    = self._pde_residual(pred_phys)      # in physical units
    #       return loss_data + lambda_pde * loss_pde
    #
    #   def _pde_residual(self, u_phys):
    #       # u_phys: (B, F, X, Y, η, T)
    #       # F = 0: ed  [GeV/fm³]
    #       # F = 1: vx  (or u^x)
    #       # F = 2: vy  (or u^y)
    #       # F = 3: vz  (or u^η)
    #       # Milne-frame ideal conservation:
    #       #   ∂_τ ε + (ε + p)/τ + ∇_⊥·((ε+p)v_⊥) + ∂_ηs(ε+p)v_η/τ = 0
    #       # Use torch.gradient or spectral differentiation to evaluate ∂_τ, ∂_x, etc.
    #       ...
    #       raise NotImplementedError
