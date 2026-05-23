"""
Unified loss for NeuralOperator/FNO training on relativistic hydro (Milne),
with tensor layout changed to:

    [B, C, Nx, Ny, Neta, Nt]   i.e. (x, y, eta, tau)

This version provides:
- Data term: L2 or H1 or L2+H1
- Minimal physics-inspired term: R_tau = ∇_μ T^{μτ} (Milne-covariant)
- Optional constraints: timelike normalization and epsilon positivity

-----------------------------------------------------------------------
Expected tensor layout
-----------------------------------------------------------------------
pred, target: torch.Tensor, shape [B, C, Nx, Ny, Neta, Nt]

Required channels:
  0: epsilon
  1: u^x
  2: u^y
  3: u^eta
Optional:
  10: Pi_bulk (if absent, set use_bulk=False or C<=10)

Grid dictionary:
grid = {
    "tau":  torch.Tensor [Nt],   # proper-time values
    "dtau": float,
    "dx":   float,
    "dy":   float,
    "deta": float,
}
"""

import torch
import torch.nn.functional as F


# -----------------------------
# Derivative helpers for layout [B,C,Nx,Ny,Neta,Nt]
# -----------------------------
def _d_central_periodic(f, dim, h):
    return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) / (2.0 * h)


def _d_tau_nonperiodic(f, dtau):
    # f: [B, Nx, Ny, Neta, Nt]
    out = torch.zeros_like(f)
    out[..., 1:-1] = (f[..., 2:] - f[..., :-2]) / (2.0 * dtau)
    out[..., 0] = (f[..., 1] - f[..., 0]) / dtau
    out[..., -1] = (f[..., -1] - f[..., -2]) / dtau
    return out


# -----------------------------
# Data loss: l2 / h1 / l2_h1
# -----------------------------
def _data_loss(pred, target, grid, mode="l2_h1", alpha_h1=0.05, include_tau_grad=False):
    """
    pred,target: [B,C,Nx,Ny,Neta,Nt]
    """
    mode = mode.lower()
    l2 = F.mse_loss(pred, target)

    if mode == "l2":
        return l2, {"loss_l2": float(l2.detach()), "loss_h1_semi": 0.0}

    dx, dy, deta = grid["dx"], grid["dy"], grid["deta"]
    dtau = grid["dtau"]

    # spatial gradients over all channels
    gx_p = _d_central_periodic(pred, dim=2, h=dx)    # Nx dim
    gy_p = _d_central_periodic(pred, dim=3, h=dy)    # Ny dim
    ge_p = _d_central_periodic(pred, dim=4, h=deta)  # Neta dim

    gx_t = _d_central_periodic(target, dim=2, h=dx)
    gy_t = _d_central_periodic(target, dim=3, h=dy)
    ge_t = _d_central_periodic(target, dim=4, h=deta)

    h1_semi = torch.mean((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2 + (ge_p - ge_t) ** 2)

    if include_tau_grad:
        # apply tau-derivative channel-wise by reshaping to [B*C, Nx,Ny,Neta,Nt]
        B, C, Nx, Ny, Neta, Nt = pred.shape
        p = pred.reshape(B * C, Nx, Ny, Neta, Nt)
        t = target.reshape(B * C, Nx, Ny, Neta, Nt)
        gt_p = _d_tau_nonperiodic(p, dtau)
        gt_t = _d_tau_nonperiodic(t, dtau)
        h1_semi = h1_semi + torch.mean((gt_p - gt_t) ** 2)

    loss_h1 = l2 + alpha_h1 * h1_semi

    if mode in ("h1", "l2_h1"):
        return loss_h1, {"loss_l2": float(l2.detach()), "loss_h1_semi": float(h1_semi.detach())}

    raise ValueError(f"Unknown data_mode='{mode}'. Use 'l2', 'h1', or 'l2_h1'.")


# -----------------------------
# Minimal Milne physics residual: R_tau = ∇_μ T^{μτ}
# layout: fields [B,Nx,Ny,Neta,Nt]
# -----------------------------
def _milne_R_tau_minimal(pred, grid, cs2=1.0 / 3.0, use_bulk=True):
    eps = pred[:, 0]   # [B,Nx,Ny,Neta,Nt]
    ux = pred[:, 1]
    uy = pred[:, 2]
    ueta = pred[:, 3]

    # tau broadcast to [1,1,1,1,Nt]
    tau = grid["tau"].to(pred.device).view(1, 1, 1, 1, -1)
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]

    if use_bulk and pred.size(1) > 10:
        Pi = pred[:, 10]
    else:
        Pi = torch.zeros_like(eps)

    # Milne normalization:
    # u^mu u_mu = (u^tau)^2 - (u^x)^2 - (u^y)^2 - tau^2 (u^eta)^2 = 1
    utau = torch.sqrt(1.0 + ux * ux + uy * uy + (tau * tau) * (ueta * ueta) + 1e-12)

    # EOS (starter)
    P = cs2 * eps
    h = eps + P + Pi

    # Minimal T^{μν}
    # T^{μν} = h u^μ u^ν - (P+Pi) g^{μν}
    # g^{ττ}=1, g^{ηη}=-1/tau^2
    T_tautau = h * utau * utau - (P + Pi)
    T_xtau = h * ux * utau
    T_ytau = h * uy * utau
    T_etatau = h * ueta * utau
    T_etaeta = h * ueta * ueta - (P + Pi) * (-1.0 / (tau * tau).clamp_min(1e-12))

    inv_tau = 1.0 / tau.clamp_min(1e-12)

    # R_tau = (1/tau)∂_tau(tau T^{tau tau}) + ∂_x T^{x tau} + ∂_y T^{y tau}
    #       + ∂_eta T^{eta tau} + tau T^{eta eta}
    R_tau = (
        inv_tau * _d_tau_nonperiodic(tau * T_tautau, dtau)
        + _d_central_periodic(T_xtau, dim=1, h=dx)    # Nx axis in [B,Nx,Ny,Neta,Nt]
        + _d_central_periodic(T_ytau, dim=2, h=dy)    # Ny
        + _d_central_periodic(T_etatau, dim=3, h=deta)  # Neta
        + tau * T_etaeta
    )

    u_norm_minus1 = (utau * utau - ux * ux - uy * uy - (tau * tau) * (ueta * ueta)) - 1.0
    return R_tau, u_norm_minus1, eps


# -----------------------------
# Unified loss
# -----------------------------
def hydro_unified_loss(
    pred,
    target,
    grid,
    data_mode="l2_h1",        # "l2" | "h1" | "l2_h1"
    alpha_h1=0.05,
    include_tau_grad=False,
    use_bulk=True,
    cs2=1.0 / 3.0,
    eps_floor=1e-10,
    weights=None,
):
    """
    Args:
      pred,target: [B,C,Nx,Ny,Neta,Nt]
      grid: dict with tau,dtau,dx,dy,deta
      data_mode: l2 | h1 | l2_h1
      alpha_h1: H1 seminorm multiplier
      include_tau_grad: include tau-derivative mismatch in H1 part
      use_bulk: include Pi_bulk channel 10 if present
      cs2: EOS constant c_s^2 for starter physics term
      eps_floor: positivity threshold
      weights: dict with keys
          w_data, w_phys, w_unorm, w_eps_pos

    Returns:
      loss (tensor scalar), logs (dict floats)
    """
    if weights is None:
        weights = {"w_data": 1.0, "w_phys": 1e-2, "w_unorm": 1e-3, "w_eps_pos": 1e-2}

    # 1) data term
    loss_data, data_logs = _data_loss(
        pred, target, grid,
        mode=data_mode,
        alpha_h1=alpha_h1,
        include_tau_grad=include_tau_grad
    )

    # 2) minimal physics term
    R_tau, u_norm_minus1, eps = _milne_R_tau_minimal(pred, grid, cs2=cs2, use_bulk=use_bulk)
    loss_phys = torch.mean(R_tau ** 2)

    # 3) soft constraints
    loss_unorm = torch.mean(u_norm_minus1 ** 2)
    loss_eps_pos = torch.mean(F.relu(eps_floor - eps))

    # 4) total
    loss = (
        weights.get("w_data", 1.0) * loss_data
        + weights.get("w_phys", 1e-2) * loss_phys
        + weights.get("w_unorm", 1e-3) * loss_unorm
        + weights.get("w_eps_pos", 1e-2) * loss_eps_pos
    )

    logs = {
        "loss": float(loss.detach()),
        "loss_data": float(loss_data.detach()),
        "loss_phys_Rtau": float(loss_phys.detach()),
        "loss_u_norm": float(loss_unorm.detach()),
        "loss_eps_pos": float(loss_eps_pos.detach()),
        **data_logs,
    }
    return loss, logs


# ============================================================
# Proper usage (for layout [B,C,Nx,Ny,Neta,Nt])
# ============================================================
"""
Example
-------
weights = {
    "w_data": 1.0,
    "w_phys": 1e-2,
    "w_unorm": 1e-3,
    "w_eps_pos": 1e-2,
}

loss, logs = hydro_unified_loss(
    pred, target, grid,
    data_mode="l2_h1",      # "l2", "h1", or "l2_h1"
    alpha_h1=0.05,
    include_tau_grad=False,
    use_bulk=True,
    cs2=1/3,
    weights=weights,
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

Tips
----
1) Start with:
   data_mode="l2_h1", alpha_h1=0.02~0.05, w_phys=1e-2
2) After warmup, increase w_phys to 5e-2 or 1e-1.
3) If training destabilizes, reduce alpha_h1 first, then w_phys.
"""