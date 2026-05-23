import torch
import torch.nn.functional as F

# channels (minimal):
# 0 eps, 1 ux, 2 uy, 3 ueta, [optional 10 Pi_bulk]
# if Pi not present, set use_bulk=False

def d_periodic(f, dim, h):
    return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) / (2.0*h)

def d_nonperiodic_tau(f, h):
    # dim=1 assumed [B,Nt,Nx,Ny,Neta]
    out = torch.zeros_like(f)
    out[:,1:-1] = (f[:,2:] - f[:,:-2])/(2.0*h)
    out[:,0]    = (f[:,1] - f[:,0])/h
    out[:,-1]   = (f[:,-1] - f[:,-2])/h
    return out

def starter_loss_fast(pred, target, grid, weights, cs2=1/3, use_bulk=True):
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]
    tau = grid["tau"].to(pred.device).view(1,-1,1,1,1)  # [1,Nt,1,1,1]

    eps  = pred[:,0]
    ux   = pred[:,1]
    uy   = pred[:,2]
    ueta = pred[:,3]
    Pi   = pred[:,10] if (use_bulk and pred.size(1) > 10) else torch.zeros_like(eps)

    # u^tau from Milne normalization: (u^tau)^2 = 1 + ux^2 + uy^2 + tau^2 ueta^2
    utau = torch.sqrt(1.0 + ux*ux + uy*uy + (tau*tau)*ueta*ueta + 1e-12)

    # EOS
    P = cs2 * eps

    # Minimal T components needed for R_tau = ∇_μ T^{μτ}
    # T^{μν} = (eps+P+Pi)u^μu^ν - (P+Pi)g^{μν}
    h = eps + P + Pi
    T_tautau = h*utau*utau - (P+Pi)*1.0
    T_xtau   = h*ux*utau
    T_ytau   = h*uy*utau
    T_etatau = h*ueta*utau

    # Need T^{eta eta} for Christoffel term +tau*T^{eta eta}
    # g^{eta eta} = -1/tau^2
    T_etaeta = h*ueta*ueta - (P+Pi)*(-1.0/(tau*tau).clamp_min(1e-12))

    inv_tau = 1.0 / tau.clamp_min(1e-12)

    # R_tau = (1/tau)∂_tau(tau T^{tau tau}) + ∂x T^{x tau} + ∂y T^{y tau} + ∂eta T^{eta tau} + tau T^{eta eta}
    R_tau = (
        inv_tau * d_nonperiodic_tau(tau * T_tautau, dtau)
        + d_periodic(T_xtau,   2, dx)
        + d_periodic(T_ytau,   3, dy)
        + d_periodic(T_etatau, 4, deta)
        + tau * T_etaeta
    )

    # losses
    loss_data = F.mse_loss(pred, target)
    loss_cons = (R_tau**2).mean()

    # u^2=1 penalty
    # u_mu = (u_tau, -ux, -uy, -tau^2 ueta)
    u_norm = utau*utau - ux*ux - uy*uy - (tau*tau)*ueta*ueta
    loss_unorm = ((u_norm - 1.0)**2).mean()

    # positivity
    loss_pos = F.relu(1e-10 - eps).mean()

    loss = (
        weights.get("data", 1.0)*loss_data
        + weights.get("cons_tau", 0.05)*loss_cons
        + weights.get("u_norm", 0.01)*loss_unorm
        + weights.get("eps_pos", 0.1)*loss_pos
    )

    logs = {
        "loss": float(loss.detach()),
        "data": float(loss_data.detach()),
        "cons_tau": float(loss_cons.detach()),
        "u_norm": float(loss_unorm.detach()),
        "eps_pos": float(loss_pos.detach()),
    }
    return loss, logs

""" weights = {
  "data": 1.0,
  "cons_tau": 1e-2,   # start tiny
  "u_norm": 1e-3,
  "eps_pos": 1e-2,
} """
