# ============================================================
# Unit tests for DNMR Milne residual block (2a implementation)
#
# Assumes the following are already imported/defined in scope:
#   - music_dnmr_milne_residuals_2a
#   - normalize_u_tau, reconstruct_pi_full_from_spatial
#   - shear_and_vorticity, trace_rank2_up, lower_vec
#
# Run with: pytest -q test_dnmr_milne.py
# ============================================================

import math
import torch
import pytest


# -----------------------------
# Helpers
# -----------------------------
def make_grid(Nt=9, Nx=8, Ny=8, Neta=8,
              tau0=0.6, dtau=0.05, dx=0.2, dy=0.2, deta=0.15,
              device="cpu", dtype=torch.float64):
    tau = tau0 + dtau * torch.arange(Nt, device=device, dtype=dtype)
    return {
        "tau": tau,
        "dtau": dtau,
        "dx": dx,
        "dy": dy,
        "deta": deta,
        "shape": (Nt, Nx, Ny, Neta),
    }


def broadcast_tau(grid, B=1, device="cpu", dtype=torch.float64):
    tau = grid["tau"].to(device=device, dtype=dtype)
    return tau.view(1, -1, 1, 1, 1).repeat(B, 1, 1, 1, 1)


def zeros_field(B, grid, device="cpu", dtype=torch.float64):
    Nt, Nx, Ny, Neta = grid["shape"]
    return torch.zeros((B, Nt, Nx, Ny, Neta), device=device, dtype=dtype)


def const_field(B, grid, c, device="cpu", dtype=torch.float64):
    z = zeros_field(B, grid, device=device, dtype=dtype)
    return z + c


def make_pred_tensor(B, grid, device="cpu", dtype=torch.float64):
    Nt, Nx, Ny, Neta = grid["shape"]
    C = 11
    return torch.zeros((B, C, Nt, Nx, Ny, Neta), device=device, dtype=dtype)


def interior_slice():
    # avoid boundary one-sided derivative stencils in tau
    return (slice(None), slice(1, -1), slice(1, -1), slice(1, -1), slice(1, -1))


def maxabs(x):
    return x.abs().max().item()


def assert_small(x, tol=1e-8, msg=""):
    m = maxabs(x)
    assert m < tol, f"{msg} max|.|={m:.3e} >= {tol:.3e}"


# -----------------------------
# Test 1: symmetry, transversality, tracelessness reconstruction
# -----------------------------
def test_pi_reconstruction_constraints():
    # Import from your implementation module if needed:
    # from your_module import normalize_u_tau, reconstruct_pi_full_from_spatial, lower_vec, trace_rank2_up
    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    B = 2
    tau = broadcast_tau(grid, B=B, device=device, dtype=dtype)

    Nt, Nx, Ny, Neta = grid["shape"]

    # random moderate velocities
    g = torch.Generator(device=device).manual_seed(7)
    ux = 0.15 * torch.randn((B, Nt, Nx, Ny, Neta), generator=g, device=device, dtype=dtype)
    uy = 0.12 * torch.randn_like(ux)
    ue = 0.08 * torch.randn_like(ux)

    ut = normalize_u_tau(ux, uy, ue, tau)
    u_up = [ut, ux, uy, ue]
    u_dn = lower_vec(u_up, tau)

    # random shear components (small)
    pi_xx = 0.01 * torch.randn_like(ux)
    pi_xy = 0.01 * torch.randn_like(ux)
    pi_xe = 0.01 * torch.randn_like(ux)
    pi_yy = 0.01 * torch.randn_like(ux)
    pi_ye = 0.01 * torch.randn_like(ux)
    pi_ee = 0.01 * torch.randn_like(ux)

    pi = reconstruct_pi_full_from_spatial(u_up, tau, pi_xx, pi_xy, pi_xe, pi_yy, pi_ye, pi_ee)

    # symmetry
    for mu in range(4):
        for nu in range(4):
            assert_small(pi[mu][nu] - pi[nu][mu], tol=1e-10, msg=f"pi symmetry ({mu},{nu})")

    # transversality u_mu pi^{mu nu}=0
    for nu in range(4):
        r = torch.zeros_like(ux)
        for mu in range(4):
            r = r + u_dn[mu] * pi[mu][nu]
        assert_small(r, tol=5e-9, msg=f"transversality nu={nu}")

    # tracelessness is NOT automatically guaranteed by reconstruction from only spatial set;
    # it should be enforced by loss penalty. Check finite value and shape.
    tr = trace_rank2_up(pi, tau)
    assert torch.isfinite(tr).all(), "trace has non-finite values"


# -----------------------------
# Test 2: Milne kinematics in uniform flow
# Expect sigma=0, omega=0 for u=(1,0,0,0)
# theta = 1/tau in Milne for Bjorken-like static spatial flow
# -----------------------------
def test_kinematics_uniform_flow():
    # from your_module import shear_and_vorticity, normalize_u_tau
    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    B = 1
    tau = broadcast_tau(grid, B=B, device=device, dtype=dtype)
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]

    ux = zeros_field(B, grid, device=device, dtype=dtype)
    uy = zeros_field(B, grid, device=device, dtype=dtype)
    ue = zeros_field(B, grid, device=device, dtype=dtype)
    ut = normalize_u_tau(ux, uy, ue, tau)
    u_up = [ut, ux, uy, ue]

    theta, sigma, omega = shear_and_vorticity(u_up, tau, dtau, dx, dy, deta)
    sl = interior_slice()

    # sigma ~ 0
    for mu in range(4):
        for nu in range(4):
            assert_small(sigma[mu][nu][sl], tol=1e-8, msg=f"sigma({mu},{nu})")

    # omega ~ 0 and antisymmetric
    for mu in range(4):
        for nu in range(4):
            assert_small(omega[mu][nu][sl] + omega[nu][mu][sl], tol=1e-8, msg=f"omega antisym ({mu},{nu})")
            assert_small(omega[mu][nu][sl], tol=1e-8, msg=f"omega({mu},{nu})")

    # theta ~ 1/tau
    target = (1.0 / tau.clamp_min(1e-12))
    err = theta[sl] - target[sl]
    assert_small(err, tol=1e-7, msg="theta-1/tau")


# -----------------------------
# Test 3: Bjorken-like ideal state sanity
# For constant eps, zero pi, zero Pi, zero velocity:
# nonzero residual may remain in conservation (Milne geometry expansion)
# but residuals should be finite and structured (no NaNs/inf).
# -----------------------------
def test_residuals_finite_bjorken_sanity():
    # from your_module import music_dnmr_milne_residuals_2a
    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    B = 1

    pred = make_pred_tensor(B, grid, device=device, dtype=dtype)
    pred[:, 0] = const_field(B, grid, 10.0, device=device, dtype=dtype)  # eps
    # u^x,u^y,u^eta already zero
    # pi components zero
    # Pi zero

    cfg = {
        "cs2": 1.0 / 3.0,
        "eta_over_s": 0.12,
        "zeta_over_s": 0.02,
        "c_tau_pi": 5.0,
        "c_tau_Pi": 1.0,
    }

    R = music_dnmr_milne_residuals_2a(pred, grid, cfg)

    for k, v in R.items():
        assert torch.isfinite(v).all(), f"{k} has non-finite values"


# -----------------------------
# Test 4: Near-equilibrium closure check
# Build pi ~ 2 eta sigma and Pi ~ -zeta theta from computed kinematics.
# Then relaxation residuals should be small (up to derivative terms and numerics).
# -----------------------------
def test_near_equilibrium_relaxation_small():
    # from your_module import shear_and_vorticity, normalize_u_tau, music_dnmr_milne_residuals_2a
    device, dtype = "cpu", torch.float64
    grid = make_grid(Nt=11, Nx=10, Ny=10, Neta=10, device=device, dtype=dtype)
    B = 1
    tau = broadcast_tau(grid, B=B, device=device, dtype=dtype)
    dtau, dx, dy, deta = grid["dtau"], grid["dx"], grid["dy"], grid["deta"]

    Nt, Nx, Ny, Neta = grid["shape"]

    # Smooth small velocity perturbation -> nonzero sigma/omega but gentle
    x = torch.linspace(0, 2 * math.pi, Nx, device=device, dtype=dtype).view(1, 1, Nx, 1, 1)
    y = torch.linspace(0, 2 * math.pi, Ny, device=device, dtype=dtype).view(1, 1, 1, Ny, 1)
    e = torch.linspace(0, 2 * math.pi, Neta, device=device, dtype=dtype).view(1, 1, 1, 1, Neta)

    ux = 0.02 * torch.sin(x).expand(B, Nt, Nx, Ny, Neta)
    uy = 0.02 * torch.cos(y).expand(B, Nt, Nx, Ny, Neta)
    ue = 0.01 * torch.sin(e).expand(B, Nt, Nx, Ny, Neta)
    ut = normalize_u_tau(ux, uy, ue, tau)
    u_up = [ut, ux, uy, ue]

    theta, sigma, _omega = shear_and_vorticity(u_up, tau, dtau, dx, dy, deta)

    eps = const_field(B, grid, 12.0, device=device, dtype=dtype)
    P = (1.0 / 3.0) * eps
    eta = 0.12 * (eps + P)
    zeta = 0.02 * (eps + P)

    # Navier-Stokes targets
    pi_xx = 2.0 * eta * sigma[1][1]
    pi_xy = 2.0 * eta * sigma[1][2]
    pi_xe = 2.0 * eta * sigma[1][3]
    pi_yy = 2.0 * eta * sigma[2][2]
    pi_ye = 2.0 * eta * sigma[2][3]
    pi_ee = 2.0 * eta * sigma[3][3]
    Pi = -zeta * theta

    pred = make_pred_tensor(B, grid, device=device, dtype=dtype)
    pred[:, 0] = eps
    pred[:, 1] = ux
    pred[:, 2] = uy
    pred[:, 3] = ue
    pred[:, 4] = pi_xx
    pred[:, 5] = pi_xy
    pred[:, 6] = pi_xe
    pred[:, 7] = pi_yy
    pred[:, 8] = pi_ye
    pred[:, 9] = pi_ee
    pred[:, 10] = Pi

    cfg = {
        "cs2": 1.0 / 3.0,
        "eta_over_s": 0.12,
        "zeta_over_s": 0.02,
        "c_tau_pi": 5.0,
        "c_tau_Pi": 1.0,
        # reduce extra second-order couplings for this near-NS check
        "delta_pipi": 0.0,
        "tau_pipi": 0.0,
        "tau_pi_vort": 0.0,
        "lambda_piPi": 0.0,
        "delta_PiPi": 0.0,
        "lambda_Pipi": 0.0,
    }

    R = music_dnmr_milne_residuals_2a(pred, grid, cfg)
    sl = interior_slice()

    # Shear/Bulk relaxation residuals should be small-ish
    # (not machine zero because D(pi), discretization, and geometric terms)
    assert maxabs(R["R_pi_xx"][sl]) < 5e-2
    assert maxabs(R["R_pi_xy"][sl]) < 5e-2
    assert maxabs(R["R_pi_yy"][sl]) < 5e-2
    assert maxabs(R["R_bulk"][sl]) < 5e-2


# -----------------------------
# Test 5: full loss backward pass smoke test
# -----------------------------
def test_backward_smoke():
    # from your_module import loss_pino_dnmr_2a
    device, dtype = "cpu", torch.float64
    grid = make_grid(device=device, dtype=dtype)
    B = 1
    Nt, Nx, Ny, Neta = grid["shape"]
    C = 11

    pred = torch.zeros((B, C, Nt, Nx, Ny, Neta), device=device, dtype=dtype, requires_grad=True)
    target = torch.zeros_like(pred)

    # nontrivial eps and tiny velocity to avoid all-zero Jacobian paths
    pred.data[:, 0] = 8.0
    pred.data[:, 1] = 0.01
    pred.data[:, 2] = -0.01
    pred.data[:, 3] = 0.005

    cfg = {
        "cs2": 1/3,
        "eta_over_s": 0.12,
        "zeta_over_s": 0.02,
        "c_tau_pi": 5.0,
        "c_tau_Pi": 1.0,
    }
    w = {"data": 1.0, "phys": 1.0, "cons": 1.0, "shear": 0.3, "bulk": 0.15, "constraints": 0.2}

    # import and call your function here:
    # loss, logs = loss_pino_dnmr_2a(pred, target, grid, cfg, w)
    from __main__ import loss_pino_dnmr_2a  # replace with your module import
    loss, logs = loss_pino_dnmr_2a(pred, target, grid, cfg, w)

    assert torch.isfinite(loss), "loss is not finite"
    loss.backward()
    assert pred.grad is not None, "grad is None"
    assert torch.isfinite(pred.grad).all(), "grad has non-finite entries"
