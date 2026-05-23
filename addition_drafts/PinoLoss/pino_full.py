# ============================================================
# DNMR-like Milne tensor algebra for PINO / neuraloperator
# Choice requested: (2a) DNMR + keep current channel set
#
# Channels expected in pred: [B,C,Nt,Nx,Ny,Neta]
#   0  eps
#   1  u^x
#   2  u^y
#   3  u^eta
#   4  pi^{xx}
#   5  pi^{xy}
#   6  pi^{xeta}
#   7  pi^{yy}
#   8  pi^{yeta}
#   9  pi^{eta eta}
#   10 Pi (bulk pressure)
#
# Notes:
# - This implements full Milne covariant kinematics:
#     theta, sigma^{mu nu}, omega^{mu nu}
#   with Christoffel terms.
# - DNMR-like residual terms included:
#     tau_pi * <D pi^{mu nu}> + pi^{mu nu}
#       = 2 eta sigma^{mu nu}
#         - delta_pipi * pi^{mu nu} theta
#         - tau_pipi * pi^{<mu}_lambda sigma^{nu>lambda}
#         + 2 tau_pi * pi^{<mu}_lambda omega^{nu>lambda}
#         + lambda_piPi * Pi sigma^{mu nu}
# - Keep channel set "a": pi^{eta eta} is explicit.
# - We reconstruct pi^{tau mu} from transversality u_mu pi^{mu nu}=0
#   via a linear solve at each cell, and enforce tracelessness as penalty.
# ============================================================

import torch
import torch.nn.functional as F


# -----------------------------
# Finite differences
# -----------------------------
def d_central_periodic(f, dim, h):
    return (torch.roll(f, -1, dims=dim) - torch.roll(f, 1, dims=dim)) / (2.0 * h)


def d_central_nonperiodic(f, dim, h):
    out = torch.zeros_like(f)
    n = f.size(dim)

    sl = [slice(None)] * f.ndim
    slp = [slice(None)] * f.ndim
    slm = [slice(None)] * f.ndim
    sl[dim] = slice(1, n - 1)
    slp[dim] = slice(2, n)
    slm[dim] = slice(0, n - 2)
    out[tuple(sl)] = (f[tuple(slp)] - f[tuple(slm)]) / (2.0 * h)

    s0 = [slice(None)] * f.ndim
    s1 = [slice(None)] * f.ndim
    s0[dim] = 0
    s1[dim] = 1
    out[tuple(s0)] = (f[tuple(s1)] - f[tuple(s0)]) / h

    sN = [slice(None)] * f.ndim
    sNm = [slice(None)] * f.ndim
    sN[dim] = n - 1
    sNm[dim] = n - 2
    out[tuple(sN)] = (f[tuple(sN)] - f[tuple(sNm)]) / h
    return out


# -----------------------------
# Milne metric utilities
# -----------------------------
# index map: 0=tau,1=x,2=y,3=eta
def milne_metric_diag(tau):
    one = torch.ones_like(tau)
    g_up = [one, -one, -one, -1.0 / (tau * tau).clamp_min(1e-12)]   # g^{mu mu}
    g_dn = [one, -one, -one, -(tau * tau)]                           # g_{mu mu}
    return g_up, g_dn


def lower_vec(v_up, tau):
    _, g_dn = milne_metric_diag(tau)
    return [g_dn[i] * v_up[i] for i in range(4)]


def raise_vec(v_dn, tau):
    g_up, _ = milne_metric_diag(tau)
    return [g_up[i] * v_dn[i] for i in range(4)]


def delta_mixed(mu, nu):
    return 1.0 if mu == nu else 0.0


def normalize_u_tau(ux, uy, ueta, tau, eps=1e-12):
    # u^mu u_mu = 1 -> (u^tau)^2 = 1 + ux^2 + uy^2 + tau^2 ueta^2
    return torch.sqrt(1.0 + ux * ux + uy * uy + (tau * tau) * (ueta * ueta) + eps)


# -----------------------------
# Christoffels (Milne)
# -----------------------------
# only nonzero:
# Γ^tau_{eta eta} = tau
# Γ^eta_{tau eta} = Γ^eta_{eta tau} = 1/tau
def gamma(mu, alpha, beta, tau):
    z = torch.zeros_like(tau)
    if mu == 0 and alpha == 3 and beta == 3:
        return tau
    if mu == 3 and ((alpha == 0 and beta == 3) or (alpha == 3 and beta == 0)):
        return 1.0 / tau.clamp_min(1e-12)
    return z


# -----------------------------
# Derivative helpers
# -----------------------------
def partial_mu(f, mu, dtau, dx, dy, deta):
    # f shape [B,Nt,Nx,Ny,Neta]
    if mu == 0:
        return d_central_nonperiodic(f, dim=1, h=dtau)
    if mu == 1:
        return d_central_periodic(f, dim=2, h=dx)
    if mu == 2:
        return d_central_periodic(f, dim=3, h=dy)
    if mu == 3:
        return d_central_periodic(f, dim=4, h=deta)
    raise ValueError("mu must be 0..3")


def covariant_derivative_vector_up(v_up, alpha, beta, tau, dtau, dx, dy, deta):
    # ∇_alpha v^beta = ∂_alpha v^beta + Γ^beta_{alpha λ} v^λ
    term = partial_mu(v_up[beta], alpha, dtau, dx, dy, deta)
    add = torch.zeros_like(term)
    for lam in range(4):
        add = add + gamma(beta, alpha, lam, tau) * v_up[lam]
    return term + add


def covariant_derivative_vector_dn(v_dn, alpha, beta, tau, dtau, dx, dy, deta):
    # ∇_alpha v_beta = ∂_alpha v_beta - Γ^λ_{alpha beta} v_λ
    term = partial_mu(v_dn[beta], alpha, dtau, dx, dy, deta)
    sub = torch.zeros_like(term)
    for lam in range(4):
        sub = sub + gamma(lam, alpha, beta, tau) * v_dn[lam]
    return term - sub


def comoving_derivative_scalar(f, u_up, dtau, dx, dy, deta):
    # D f = u^mu ∂_mu f
    out = torch.zeros_like(f)
    for mu in range(4):
        out = out + u_up[mu] * partial_mu(f, mu, dtau, dx, dy, deta)
    return out


# -----------------------------
# Projectors and tensors
# -----------------------------
def delta_up(mu, nu, tau):
    # Δ^{mu nu} = g^{mu nu} - u^mu u^nu (needs u; this function not used alone)
    raise NotImplementedError


def build_Delta_up(u_up, tau):
    # returns 4x4 tensors Δ^{mu nu}
    g_up, _ = milne_metric_diag(tau)
    D = [[None for _ in range(4)] for _ in range(4)]
    z = torch.zeros_like(tau)
    for mu in range(4):
        for nu in range(4):
            gmunu = g_up[mu] if mu == nu else z
            D[mu][nu] = gmunu - u_up[mu] * u_up[nu]
    return D


def build_Delta_dn(u_dn, tau):
    # Δ_{mu nu} = g_{mu nu} - u_mu u_nu
    _, g_dn = milne_metric_diag(tau)
    D = [[None for _ in range(4)] for _ in range(4)]
    z = torch.zeros_like(tau)
    for mu in range(4):
        for nu in range(4):
            gmunu = g_dn[mu] if mu == nu else z
            D[mu][nu] = gmunu - u_dn[mu] * u_dn[nu]
    return D


def lower_rank2(Tup, tau):
    # T_{mu nu} = g_{mu a} g_{nu b} T^{ab}; diagonal metric simplifies
    _, g_dn = milne_metric_diag(tau)
    Tdn = [[None for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            Tdn[mu][nu] = g_dn[mu] * g_dn[nu] * Tup[mu][nu]
    return Tdn


def mixed_rank2_first_down(Tup, tau):
    # T^mu{}_nu = g_{nu a} T^{mu a}; diagonal => multiply column by g_{nu}
    _, g_dn = milne_metric_diag(tau)
    Tmix = [[None for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            Tmix[mu][nu] = g_dn[nu] * Tup[mu][nu]
    return Tmix


def trace_rank2_up(Tup, tau):
    # T^mu_mu = g_{mu nu} T^{mu nu} = sum_mu g_{mu mu} T^{mu mu}
    _, g_dn = milne_metric_diag(tau)
    tr = torch.zeros_like(tau)
    for mu in range(4):
        tr = tr + g_dn[mu] * Tup[mu][mu]
    return tr


def symmetrize_rank2(T):
    S = [[None for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            S[mu][nu] = 0.5 * (T[mu][nu] + T[nu][mu])
    return S


# -----------------------------
# Reconstruct pi^{tau mu} from transversality
# -----------------------------
def reconstruct_pi_full_from_spatial(u_up, tau, pi_xx, pi_xy, pi_xe, pi_yy, pi_ye, pi_ee):
    """
    Given explicit spatial components and u^mu, solve u_mu pi^{mu nu}=0 for nu=x,y,eta
    unknowns: pi^{tau x}, pi^{tau y}, pi^{tau eta}
    then solve nu=tau for pi^{tau tau}.
    """
    ut, ux, uy, ue = u_up
    u_dn = lower_vec(u_up, tau)
    u_tau, u_x, u_y, u_e = u_dn  # u_tau=ut, u_x=-ux, u_y=-uy, u_e=-tau^2 ue

    z = torch.zeros_like(pi_xx)

    # Build matrix A * X = B
    # For nu=x:
    # u_tau pi^{tau x} + u_x pi^{xx} + u_y pi^{yx} + u_e pi^{eta x} = 0
    # => u_tau ptx = -u_x pi_xx - u_y pi_xy - u_e pi_xe
    Bx = -(u_x * pi_xx + u_y * pi_xy + u_e * pi_xe)

    # nu=y
    By = -(u_x * pi_xy + u_y * pi_yy + u_e * pi_ye)

    # nu=eta
    Be = -(u_x * pi_xe + u_y * pi_ye + u_e * pi_ee)

    inv_u_tau = 1.0 / u_tau.clamp_min(1e-12)
    pi_tx = Bx * inv_u_tau
    pi_ty = By * inv_u_tau
    pi_te = Be * inv_u_tau

    # nu=tau:
    # u_tau pi^{tau tau} + u_x pi^{x tau} + u_y pi^{y tau} + u_e pi^{eta tau} = 0
    # with symmetry pi^{x tau}=pi^{tau x}, etc.
    pi_tt = -(u_x * pi_tx + u_y * pi_ty + u_e * pi_te) * inv_u_tau

    # assemble full pi^{mu nu}
    pi = [[z for _ in range(4)] for _ in range(4)]
    pi[0][0] = pi_tt
    pi[0][1] = pi_tx
    pi[1][0] = pi_tx
    pi[0][2] = pi_ty
    pi[2][0] = pi_ty
    pi[0][3] = pi_te
    pi[3][0] = pi_te

    pi[1][1] = pi_xx
    pi[1][2] = pi_xy
    pi[2][1] = pi_xy
    pi[1][3] = pi_xe
    pi[3][1] = pi_xe
    pi[2][2] = pi_yy
    pi[2][3] = pi_ye
    pi[3][2] = pi_ye
    pi[3][3] = pi_ee
    return pi


# -----------------------------
# Kinematics: theta, sigma, omega
# -----------------------------
def expansion_theta(u_up, tau, dtau, dx, dy, deta):
    # θ = ∇_mu u^mu = (1/tau)∂_tau(tau u^tau)+∂x u^x+∂y u^y+∂eta u^eta
    inv_tau = 1.0 / tau.clamp_min(1e-12)
    term_t = inv_tau * d_central_nonperiodic(tau * u_up[0], dim=1, h=dtau)
    term_x = d_central_periodic(u_up[1], dim=2, h=dx)
    term_y = d_central_periodic(u_up[2], dim=3, h=dy)
    term_e = d_central_periodic(u_up[3], dim=4, h=deta)
    return term_t + term_x + term_y + term_e


def shear_and_vorticity(u_up, tau, dtau, dx, dy, deta):
    """
    Build full sigma^{mu nu} and omega^{mu nu} in Milne:
      sigma^{mu nu} = Delta^{mu alpha} Delta^{nu beta} [ (∇_alpha u_beta + ∇_beta u_alpha)/2 - (1/3) Delta_{alpha beta} theta ]
      omega^{mu nu} = (1/2) Delta^{mu alpha} Delta^{nu beta} (∇_alpha u_beta - ∇_beta u_alpha)
    """
    u_dn = lower_vec(u_up, tau)
    Delta_up = build_Delta_up(u_up, tau)
    Delta_dn = build_Delta_dn(u_dn, tau)
    theta = expansion_theta(u_up, tau, dtau, dx, dy, deta)

    # A_{alpha beta} = (∇_alpha u_beta + ∇_beta u_alpha)/2
    A = [[None for _ in range(4)] for _ in range(4)]
    W = [[None for _ in range(4)] for _ in range(4)]  # antisymmetric base
    for a in range(4):
        for b in range(4):
            nab_a_ub = covariant_derivative_vector_dn(u_dn, a, b, tau, dtau, dx, dy, deta)
            nab_b_ua = covariant_derivative_vector_dn(u_dn, b, a, tau, dtau, dx, dy, deta)
            A[a][b] = 0.5 * (nab_a_ub + nab_b_ua)
            W[a][b] = 0.5 * (nab_a_ub - nab_b_ua)

    # S_{alpha beta} = A_{alpha beta} - (1/3) Delta_{alpha beta} theta
    S = [[None for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            S[a][b] = A[a][b] - (1.0 / 3.0) * Delta_dn[a][b] * theta

    # Project up/up:
    sigma = [[torch.zeros_like(theta) for _ in range(4)] for _ in range(4)]
    omega = [[torch.zeros_like(theta) for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            s = torch.zeros_like(theta)
            w = torch.zeros_like(theta)
            for a in range(4):
                for b in range(4):
                    s = s + Delta_up[mu][a] * Delta_up[nu][b] * S[a][b]
                    w = w + Delta_up[mu][a] * Delta_up[nu][b] * W[a][b]
            sigma[mu][nu] = s
            omega[mu][nu] = w

    # force symmetry/antisymmetry numerically
    sigma = symmetrize_rank2(sigma)
    for mu in range(4):
        for nu in range(4):
            omega[mu][nu] = 0.5 * (omega[mu][nu] - omega[nu][mu])

    return theta, sigma, omega


# -----------------------------
# Angle-bracket projection for rank-2 up tensor
# X^{<mu nu>} = Delta^{mu}_{a} Delta^{nu}_{b} (X^{ab}+X^{ba})/2 - (1/3) Delta^{mu nu} Delta_{ab} X^{ab}
# -----------------------------
def bracket_project_rank2_up(X_up, u_up, tau):
    Delta_up = build_Delta_up(u_up, tau)
    u_dn = lower_vec(u_up, tau)
    Delta_dn = build_Delta_dn(u_dn, tau)

    # sym part
    Xs = symmetrize_rank2(X_up)

    # scalar tr_perp = Delta_{ab} X^{ab}
    tr_perp = torch.zeros_like(tau)
    for a in range(4):
        for b in range(4):
            tr_perp = tr_perp + Delta_dn[a][b] * Xs[a][b]

    Y = [[torch.zeros_like(tau) for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            t = torch.zeros_like(tau)
            for a in range(4):
                for b in range(4):
                    t = t + Delta_up[mu][a] * Delta_up[nu][b] * Xs[a][b]
            Y[mu][nu] = t - (1.0 / 3.0) * Delta_up[mu][nu] * tr_perp
    return Y


# -----------------------------
# Build T^{mu nu}
# -----------------------------
def eos_pressure(eps, cs2=1.0 / 3.0):
    return cs2 * eps


def shear_viscosity(eps, P, eta_over_s=0.12):
    return eta_over_s * (eps + P).clamp_min(1e-12)


def bulk_viscosity(eps, P, zeta_over_s=0.02):
    return zeta_over_s * (eps + P).clamp_min(1e-12)


def tau_pi_from_eta(eta, eps, P, c_tau=5.0):
    return c_tau * eta / (eps + P).clamp_min(1e-12)


def tau_Pi_from_zeta(zeta, eps, P, c_tau=1.0):
    return c_tau * zeta / (eps + P).clamp_min(1e-12)


def build_Tmunu(eps, P, Pi_bulk, u_up, pi_up, tau):
    g_up, _ = milne_metric_diag(tau)
    h = eps + P + Pi_bulk
    z = torch.zeros_like(eps)
    T = [[z for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            gmunu = g_up[mu] if mu == nu else z
            T[mu][nu] = h * u_up[mu] * u_up[nu] - (P + Pi_bulk) * gmunu + pi_up[mu][nu]
    return T


def cov_div_T_milne(T, tau, dtau, dx, dy, deta):
    # R^nu = ∇_mu T^{mu nu}
    inv_tau = 1.0 / tau.clamp_min(1e-12)
    R = [None] * 4

    def d_tau(f): return d_central_nonperiodic(f, dim=1, h=dtau)
    def d_x(f): return d_central_periodic(f, dim=2, h=dx)
    def d_y(f): return d_central_periodic(f, dim=3, h=dy)
    def d_e(f): return d_central_periodic(f, dim=4, h=deta)

    for nu in range(4):
        base = inv_tau * d_tau(tau * T[0][nu]) + d_x(T[1][nu]) + d_y(T[2][nu]) + d_e(T[3][nu])
        # + Γ^nu_{muλ}T^{muλ}
        conn = torch.zeros_like(base)
        for mu in range(4):
            for lam in range(4):
                conn = conn + gamma(nu, mu, lam, tau) * T[mu][lam]
        R[nu] = base + conn
    return R


# -----------------------------
# DNMR RHS terms
# -----------------------------
def matmul_rank2_up_mixed(A_up, B_up, tau):
    # returns C^{mu nu} = A^{mu}{}_{lambda} B^{lambda nu}
    A_mixed = mixed_rank2_first_down(A_up, tau)  # A^{mu}{}_{lambda}
    z = torch.zeros_like(A_up[0][0])
    C = [[z for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            c = torch.zeros_like(z)
            for lam in range(4):
                c = c + A_mixed[mu][lam] * B_up[lam][nu]
            C[mu][nu] = c
    return C


def DNMR_residual_pi_component(mu, nu, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                               u_up, tau, dtau, dx, dy, deta,
                               eta, tau_pi, coeffs):
    """
    R^{mu nu}_pi:
      tau_pi * <D pi^{mu nu}> + pi^{mu nu}
      - 2 eta sigma^{mu nu}
      + delta_pipi pi^{mu nu} theta
      + tau_pipi pi^{<mu}_lambda sigma^{nu>lambda}
      - 2 tau_pi_vort pi^{<mu}_lambda omega^{nu>lambda}
      - lambda_piPi Pi sigma^{mu nu}
    = 0
    """
    # D pi^{ab}
    Dpi = [[comoving_derivative_scalar(pi_up[a][b], u_up, dtau, dx, dy, deta) for b in range(4)] for a in range(4)]
    Dpi_br = bracket_project_rank2_up(Dpi, u_up, tau)

    # pi^{<mu}_lambda sigma^{nu>lambda}
    pisig_raw = matmul_rank2_up_mixed(pi_up, sigma_up, tau)  # pi^mu{}_l sigma^{l nu}
    pisig_br = bracket_project_rank2_up(pisig_raw, u_up, tau)

    # pi^{<mu}_lambda omega^{nu>lambda}
    piom_raw = matmul_rank2_up_mixed(pi_up, omega_up, tau)
    piom_br = bracket_project_rank2_up(piom_raw, u_up, tau)

    delta_pipi = coeffs.get("delta_pipi", 4.0 / 3.0 * tau_pi)   # common kinetic estimate
    tau_pipi = coeffs.get("tau_pipi", 10.0 / 7.0 * tau_pi)
    tau_pi_vort = coeffs.get("tau_pi_vort", tau_pi)
    lambda_piPi = coeffs.get("lambda_piPi", 6.0 / 5.0 * tau_pi)

    R = (
        tau_pi * Dpi_br[mu][nu]
        + pi_up[mu][nu]
        - 2.0 * eta * sigma_up[mu][nu]
        + delta_pipi * pi_up[mu][nu] * theta
        + tau_pipi * pisig_br[mu][nu]
        - 2.0 * tau_pi_vort * piom_br[mu][nu]
        - lambda_piPi * Pi_bulk * sigma_up[mu][nu]
    )
    return R


# -----------------------------
# Main residual builder (2a DNMR)
# -----------------------------
def music_dnmr_milne_residuals_2a(pred, grid, cfg):
    """
    pred [B,C,Nt,Nx,Ny,Neta], C>=11
    """
    device = pred.device
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]
    tau_1d = grid["tau"].to(device)                    # [Nt]
    tau = tau_1d.view(1, -1, 1, 1, 1)                  # broadcast

    # unpack
    eps = pred[:, 0]
    ux = pred[:, 1]
    uy = pred[:, 2]
    ueta = pred[:, 3]
    pi_xx = pred[:, 4]
    pi_xy = pred[:, 5]
    pi_xe = pred[:, 6]
    pi_yy = pred[:, 7]
    pi_ye = pred[:, 8]
    pi_ee = pred[:, 9]
    Pi_bulk = pred[:, 10]

    # velocity
    utau = normalize_u_tau(ux, uy, ueta, tau)
    u_up = [utau, ux, uy, ueta]
    u_dn = lower_vec(u_up, tau)

    # thermo/transport
    P = eos_pressure(eps, cs2=cfg.get("cs2", 1.0 / 3.0))
    eta = shear_viscosity(eps, P, eta_over_s=cfg.get("eta_over_s", 0.12))
    zeta = bulk_viscosity(eps, P, zeta_over_s=cfg.get("zeta_over_s", 0.02))
    tau_pi = tau_pi_from_eta(eta, eps, P, c_tau=cfg.get("c_tau_pi", 5.0))
    tau_Pi = tau_Pi_from_zeta(zeta, eps, P, c_tau=cfg.get("c_tau_Pi", 1.0))

    # reconstruct full pi^{mu nu} from transversality
    pi_up = reconstruct_pi_full_from_spatial(u_up, tau, pi_xx, pi_xy, pi_xe, pi_yy, pi_ye, pi_ee)

    # kinematics
    theta, sigma_up, omega_up = shear_and_vorticity(u_up, tau, dtau, dx, dy, deta)

    # conservation residuals
    T = build_Tmunu(eps, P, Pi_bulk, u_up, pi_up, tau)
    R_cons = cov_div_T_milne(T, tau, dtau, dx, dy, deta)
    R_cons_tau, R_cons_x, R_cons_y, R_cons_eta = R_cons

    # bulk DNMR-like residual:
    # tau_Pi DPi + Pi = - zeta theta - delta_PiPi Pi theta + lambda_Pipi pi^{mu nu} sigma_{mu nu}
    DPi = comoving_derivative_scalar(Pi_bulk, u_up, dtau, dx, dy, deta)

    # pi^{mu nu} sigma_{mu nu}:
    # lower sigma then contract up/down
    sigma_dn = lower_rank2(sigma_up, tau)
    pi_sigma = torch.zeros_like(eps)
    for mu in range(4):
        for nu in range(4):
            pi_sigma = pi_sigma + pi_up[mu][nu] * sigma_dn[mu][nu]

    delta_PiPi = cfg.get("delta_PiPi", (2.0 / 3.0) * tau_Pi)
    lambda_Pipi = cfg.get("lambda_Pipi", (8.0 / 5.0) * tau_Pi)
    R_bulk = tau_Pi * DPi + Pi_bulk + zeta * theta + delta_PiPi * Pi_bulk * theta - lambda_Pipi * pi_sigma

    # shear DNMR residuals for predicted components (2a set)
    coeffs = {
        "delta_pipi": cfg.get("delta_pipi", None),
        "tau_pipi": cfg.get("tau_pipi", None),
        "tau_pi_vort": cfg.get("tau_pi_vort", None),
        "lambda_piPi": cfg.get("lambda_piPi", None),
    }
    # sanitize None fallback in function by replacing with missing keys
    coeffs = {k: v for k, v in coeffs.items() if v is not None}

    R_pi_xx = DNMR_residual_pi_component(1, 1, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)
    R_pi_xy = DNMR_residual_pi_component(1, 2, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)
    R_pi_xe = DNMR_residual_pi_component(1, 3, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)
    R_pi_yy = DNMR_residual_pi_component(2, 2, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)
    R_pi_ye = DNMR_residual_pi_component(2, 3, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)
    R_pi_ee = DNMR_residual_pi_component(3, 3, pi_up, sigma_up, omega_up, Pi_bulk, theta,
                                         u_up, tau, dtau, dx, dy, deta, eta, tau_pi, coeffs)

    # constraints
    # u^2=1
    u_norm = u_up[0] * u_dn[0] + u_up[1] * u_dn[1] + u_up[2] * u_dn[2] + u_up[3] * u_dn[3]
    R_unorm = u_norm - 1.0

    # transversality residual check u_mu pi^{mu nu}=0
    R_trans = []
    for nu in range(4):
        r = torch.zeros_like(eps)
        for mu in range(4):
            r = r + u_dn[mu] * pi_up[mu][nu]
        R_trans.append(r)

    # tracelessness residual pi^mu_mu = 0
    R_trace = trace_rank2_up(pi_up, tau)

    penalty_eps = F.relu(1e-12 - eps)

    return {
        "R_cons_tau": R_cons_tau,
        "R_cons_x": R_cons_x,
        "R_cons_y": R_cons_y,
        "R_cons_eta": R_cons_eta,
        "R_bulk": R_bulk,
        "R_pi_xx": R_pi_xx,
        "R_pi_xy": R_pi_xy,
        "R_pi_xe": R_pi_xe,
        "R_pi_yy": R_pi_yy,
        "R_pi_ye": R_pi_ye,
        "R_pi_ee": R_pi_ee,
        "R_unorm": R_unorm,
        "R_trans_tau": R_trans[0],
        "R_trans_x": R_trans[1],
        "R_trans_y": R_trans[2],
        "R_trans_eta": R_trans[3],
        "R_trace": R_trace,
        "penalty_eps": penalty_eps,
    }


def loss_pino_dnmr_2a(pred, target, grid, cfg, w):
    # supervised
    loss_data = F.mse_loss(pred, target)

    # physics residuals
    R = music_dnmr_milne_residuals_2a(pred, grid, cfg)

    loss_cons = (
        (R["R_cons_tau"] ** 2).mean()
        + (R["R_cons_x"] ** 2).mean()
        + (R["R_cons_y"] ** 2).mean()
        + (R["R_cons_eta"] ** 2).mean()
    )

    loss_shear = (
        (R["R_pi_xx"] ** 2).mean()
        + (R["R_pi_xy"] ** 2).mean()
        + (R["R_pi_xe"] ** 2).mean()
        + (R["R_pi_yy"] ** 2).mean()
        + (R["R_pi_ye"] ** 2).mean()
        + (R["R_pi_ee"] ** 2).mean()
    )

    loss_bulk = (R["R_bulk"] ** 2).mean()

    loss_constraints = (
        (R["R_unorm"] ** 2).mean()
        + (R["R_trans_tau"] ** 2).mean()
        + (R["R_trans_x"] ** 2).mean()
        + (R["R_trans_y"] ** 2).mean()
        + (R["R_trans_eta"] ** 2).mean()
        + (R["R_trace"] ** 2).mean()
        + R["penalty_eps"].mean()
    )

    loss_phys = (
        w.get("cons", 1.0) * loss_cons
        + w.get("shear", 0.3) * loss_shear
        + w.get("bulk", 0.15) * loss_bulk
        + w.get("constraints", 0.2) * loss_constraints
    )

    loss = w.get("data", 1.0) * loss_data + w.get("phys", 1.0) * loss_phys
    logs = {
        "loss": loss.item(),
        "loss_data": loss_data.item(),
        "loss_phys": loss_phys.item(),
        "loss_cons": loss_cons.item(),
        "loss_shear": loss_shear.item(),
        "loss_bulk": loss_bulk.item(),
        "loss_constraints": loss_constraints.item(),
    }
    return loss, logs


# -----------------------------
# training step sketch
# -----------------------------
def train_step(model, batch, optimizer, grid, cfg, weights, device="cuda"):
    model.train()
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    pred = model(x)
    loss, logs = loss_pino_dnmr_2a(pred, y, grid, cfg, weights)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return logs

### Minimal usage notes
# - `grid["tau"]` must be a 1D tensor of size `Nt` (proper-time points).
# - This is **DNMR-like** and fully Milne-covariant for kinematics and conservation.
# - If you want, next I can provide a **unit-test block** (symmetry, transversality, tracelessness, Bjorken sanity test) so you can validate this before training.