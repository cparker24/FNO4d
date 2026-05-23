# ============================================================
# Manufactured-solution tests for Milne DNMR block
# ============================================================
# What this gives you:
# 1) Exact analytic scalar-derivative check:
#    compares numerical ∂_mu f to closed-form derivatives.
# 2) Exact covariant divergence check for rank-2 tensor:
#    builds analytic T^{mu nu}(tau) with only diagonal entries,
#    then compares numerical ∇_mu T^{mu nu} to exact expression.
# 3) Energy conservation consistency check:
#    for ideal Bjorken flow (u=(1,0,0,0), pi=Pi=0, P=c_s^2 eps(tau)),
#    verifies R^tau = d_tau eps + (eps+P)/tau.
#
# This file assumes your implementation functions are available:
#   - partial_mu
#   - cov_div_T_milne
#   - music_dnmr_milne_residuals_2a
#   - normalize_u_tau
#
# Run:
#   pytest -q test_manufactured_milne.py
# ============================================================

import torch
import pytest


# -----------------------------
# Grid helpers
# -----------------------------
def make_grid(Nt=41, Nx=32, Ny=32, Neta=32,
              tau0=0.6, dtau=0.01, dx=0.2, dy=0.2, deta=0.2,
              device="cpu", dtype=torch.float64):
    tau = tau0 + dtau * torch.arange(Nt, device=device, dtype=dtype)
    x = dx * torch.arange(Nx, device=device, dtype=dtype)
    y = dy * torch.arange(Ny, device=device, dtype=dtype)
    eta = deta * torch.arange(Neta, device=device, dtype=dtype)
    return {
        "tau": tau,
        "x": x,
        "y": y,
        "eta": eta,
        "dtau": dtau,
        "dx": dx,
        "dy": dy,
        "deta": deta,
        "shape": (Nt, Nx, Ny, Neta),
    }


def mesh_5d(grid, B=1, device="cpu", dtype=torch.float64):
    Nt, Nx, Ny, Neta = grid["shape"]
    tau = grid["tau"].to(device=device, dtype=dtype).view(1, Nt, 1, 1, 1).repeat(B, 1, Nx, Ny, Neta)
    x = grid["x"].to(device=device, dtype=dtype).view(1, 1, Nx, 1, 1).repeat(B, Nt, 1, Ny, Neta)
    y = grid["y"].to(device=device, dtype=dtype).view(1, 1, 1, Ny, 1).repeat(B, Nt, Nx, 1, Neta)
    e = grid["eta"].to(device=device, dtype=dtype).view(1, 1, 1, 1, Neta).repeat(B, Nt, Nx, Ny, 1)
    return tau, x, y, e


def interior():
    # skip tau boundaries due to one-sided stencils, and one cell in periodic dims
    return (slice(None), slice(2, -2), slice(2, -2), slice(2, -2), slice(2, -2))


def l2_rel(err, ref, eps=1e-14):
    num = torch.sqrt(torch.mean(err * err))
    den = torch.sqrt(torch.mean(ref * ref)) + eps
    return (num / den).item()


# -----------------------------
# Test 1: manufactured scalar field derivatives
# -----------------------------
def test_manufactured_partial_mu_accuracy():
    # replace import path with your module
    from __main__ import partial_mu

    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]

    B = 1
    tau, x, y, e = mesh_5d(grid, B=B, device=device, dtype=dtype)

    # Manufactured scalar:
    # f = tau^2 + 0.3 sin(kx x) + 0.2 cos(ky y) + 0.1 sin(ke eta)
    Lx = grid["x"][-1] + grid["dx"]
    Ly = grid["y"][-1] + grid["dy"]
    Le = grid["eta"][-1] + grid["deta"]
    kx = 2.0 * torch.pi / Lx
    ky = 2.0 * torch.pi / Ly
    ke = 2.0 * torch.pi / Le

    f = tau**2 + 0.3 * torch.sin(kx * x) + 0.2 * torch.cos(ky * y) + 0.1 * torch.sin(ke * e)

    # exact partials
    d_tau_exact = 2.0 * tau
    d_x_exact = 0.3 * kx * torch.cos(kx * x)
    d_y_exact = -0.2 * ky * torch.sin(ky * y)
    d_e_exact = 0.1 * ke * torch.cos(ke * e)

    # numerical partials
    d_tau_num = partial_mu(f, 0, dtau, dx, dy, deta)
    d_x_num = partial_mu(f, 1, dtau, dx, dy, deta)
    d_y_num = partial_mu(f, 2, dtau, dx, dy, deta)
    d_e_num = partial_mu(f, 3, dtau, dx, dy, deta)

    sl = interior()
    assert l2_rel(d_tau_num[sl] - d_tau_exact[sl], d_tau_exact[sl]) < 3e-3
    assert l2_rel(d_x_num[sl] - d_x_exact[sl], d_x_exact[sl]) < 3e-3
    assert l2_rel(d_y_num[sl] - d_y_exact[sl], d_y_exact[sl]) < 3e-3
    assert l2_rel(d_e_num[sl] - d_e_exact[sl], d_e_exact[sl]) < 3e-3


# -----------------------------
# Test 2: manufactured analytic ∇_mu T^{mu nu}
# -----------------------------
def test_manufactured_cov_div_T_milne():
    # replace import path with your module
    from __main__ import cov_div_T_milne

    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]

    B = 1
    tau, x, y, e = mesh_5d(grid, B=B, device=device, dtype=dtype)

    # Build analytic T^{mu nu}(tau) diagonal only:
    # T^{tau tau}=A=tau^3
    # T^{x x}=B=tau^2
    # T^{y y}=C=tau
    # T^{eta eta}=D=tau^(-1)
    # others = 0
    A = tau**3
    Bxx = tau**2
    Cyy = tau
    Dee = 1.0 / tau

    z = torch.zeros_like(tau)
    T = [[z for _ in range(4)] for _ in range(4)]
    T[0][0] = A
    T[1][1] = Bxx
    T[2][2] = Cyy
    T[3][3] = Dee

    # numeric cov divergence
    R = cov_div_T_milne(T, tau, dtau, dx, dy, deta)
    Rtau_num, Rx_num, Ry_num, Reta_num = R

    # exact:
    # R^tau = (1/tau) d_tau(tau A) + tau D
    #        = (1/tau) d_tau(tau^4) + tau*(tau^-1) = 4 tau^2 + 1
    Rtau_exact = 4.0 * tau**2 + 1.0

    # R^x = 0, R^y = 0
    Rx_exact = torch.zeros_like(tau)
    Ry_exact = torch.zeros_like(tau)

    # R^eta = (2/tau) T^{tau eta} + transport terms; all zero here
    Reta_exact = torch.zeros_like(tau)

    sl = interior()
    assert l2_rel(Rtau_num[sl] - Rtau_exact[sl], Rtau_exact[sl]) < 5e-3
    assert torch.max(torch.abs(Rx_num[sl] - Rx_exact[sl])).item() < 2e-6
    assert torch.max(torch.abs(Ry_num[sl] - Ry_exact[sl])).item() < 2e-6
    assert torch.max(torch.abs(Reta_num[sl] - Reta_exact[sl])).item() < 2e-6


# -----------------------------
# Test 3: ideal Bjorken energy equation identity in residual
# -----------------------------
def test_bjorken_energy_identity_in_residual():
    # replace import path with your module
    from __main__ import music_dnmr_milne_residuals_2a

    device, dtype = "cpu", torch.float64
    grid = make_grid(Nt=81, dtau=0.005, device=device, dtype=dtype)
    B = 1
    Nt, Nx, Ny, Neta = grid["shape"]

    pred = torch.zeros((B, 11, Nt, Nx, Ny, Neta), device=device, dtype=dtype)

    # ideal Bjorken-like setup:
    # u^x=u^y=u^eta=0, pi=0, Pi=0
    # eps(tau)=eps0*(tau0/tau)^alpha ; alpha = 1 + c_s^2 gives exact ideal Bjorken solution
    cs2 = 1.0 / 3.0
    alpha = 1.0 + cs2
    tau0 = grid["tau"][0]
    eps0 = 20.0

    tau = grid["tau"].view(1, Nt, 1, 1, 1).to(device=device, dtype=dtype)
    eps_tau = eps0 * (tau0 / tau)**alpha
    pred[:, 0] = eps_tau  # eps channel

    cfg = {
        "cs2": cs2,
        "eta_over_s": 0.0,    # ideal for this test
        "zeta_over_s": 0.0,
        "c_tau_pi": 5.0,
        "c_tau_Pi": 1.0,
        "delta_pipi": 0.0,
        "tau_pipi": 0.0,
        "tau_pi_vort": 0.0,
        "lambda_piPi": 0.0,
        "delta_PiPi": 0.0,
        "lambda_Pipi": 0.0,
    }

    R = music_dnmr_milne_residuals_2a(pred, grid, cfg)
    Rtau = R["R_cons_tau"]

    # In ideal Bjorken, this should be ~0:
    # d_tau eps + (eps+P)/tau = 0
    sl = interior()
    max_err = torch.max(torch.abs(Rtau[sl])).item()
    assert max_err < 2e-3, f"Bjorken energy residual too large: {max_err:.3e}"