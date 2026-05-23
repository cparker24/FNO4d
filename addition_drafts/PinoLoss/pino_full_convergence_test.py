# ============================================================
# Convergence-rate tests (expect ~2nd order)
# ============================================================
# Run:
#   pytest -q test_convergence_milne.py
#
# Assumes these functions exist in your implementation module:
#   - partial_mu
#   - cov_div_T_milne
# ============================================================

import torch
import math
import pytest


# ---------- helpers ----------
def make_grid(Nt=41, Nx=16, Ny=16, Neta=16,
              tau0=0.6, dtau=0.01, dx=0.2, dy=0.2, deta=0.2,
              device="cpu", dtype=torch.float64):
    tau = tau0 + dtau * torch.arange(Nt, device=device, dtype=dtype)
    x = dx * torch.arange(Nx, device=device, dtype=dtype)
    y = dy * torch.arange(Ny, device=device, dtype=dtype)
    eta = deta * torch.arange(Neta, device=device, dtype=dtype)
    return {
        "tau": tau, "x": x, "y": y, "eta": eta,
        "dtau": dtau, "dx": dx, "dy": dy, "deta": deta,
        "shape": (Nt, Nx, Ny, Neta)
    }


def mesh5(grid, B=1, device="cpu", dtype=torch.float64):
    Nt, Nx, Ny, Neta = grid["shape"]
    tau = grid["tau"].to(device=device, dtype=dtype).view(1, Nt, 1, 1, 1).repeat(B, 1, Nx, Ny, Neta)
    x = grid["x"].to(device=device, dtype=dtype).view(1, 1, Nx, 1, 1).repeat(B, Nt, 1, Ny, Neta)
    y = grid["y"].to(device=device, dtype=dtype).view(1, 1, 1, Ny, 1).repeat(B, Nt, Nx, 1, Neta)
    e = grid["eta"].to(device=device, dtype=dtype).view(1, 1, 1, 1, Neta).repeat(B, Nt, Nx, Ny, 1)
    return tau, x, y, e


def interior():
    # more conservative interior crop
    return (slice(None), slice(3, -3), slice(3, -3), slice(3, -3), slice(3, -3))


def rel_l2(err, ref, eps=1e-14):
    num = torch.sqrt(torch.mean(err * err))
    den = torch.sqrt(torch.mean(ref * ref)) + eps
    return (num / den).item()


def observed_order(err_coarse, err_fine, refinement_ratio=2.0):
    # p = log(err_c/err_f)/log(r)
    return math.log(err_coarse / err_fine) / math.log(refinement_ratio)


# ---------- manufactured fields ----------
def manufactured_scalar(tau, x, y, e, Lx, Ly, Le):
    kx = 2.0 * math.pi / Lx
    ky = 2.0 * math.pi / Ly
    ke = 2.0 * math.pi / Le
    f = tau**2 + 0.3 * torch.sin(kx * x) + 0.2 * torch.cos(ky * y) + 0.1 * torch.sin(ke * e)

    d_tau = 2.0 * tau
    d_x = 0.3 * kx * torch.cos(kx * x)
    d_y = -0.2 * ky * torch.sin(ky * y)
    d_e = 0.1 * ke * torch.cos(ke * e)
    return f, (d_tau, d_x, d_y, d_e)


def manufactured_Tdiag(tau):
    # T^{tau tau}=tau^3, T^{xx}=tau^2, T^{yy}=tau, T^{eta eta}=tau^-1
    A = tau**3
    B = tau**2
    C = tau
    D = 1.0 / tau
    return A, B, C, D


# ---------- Test 1: partial_mu convergence ----------
def test_partial_mu_second_order_convergence():
    # replace import path with your module
    from __main__ import partial_mu

    device, dtype = "cpu", torch.float64
    B = 1

    # domain extents fixed; refine h by 2
    # coarse
    gc = make_grid(Nt=41, Nx=16, Ny=16, Neta=16, dtau=0.01, dx=0.2, dy=0.2, deta=0.2,
                   device=device, dtype=dtype)
    # fine
    gf = make_grid(Nt=81, Nx=32, Ny=32, Neta=32, dtau=0.005, dx=0.1, dy=0.1, deta=0.1,
                   device=device, dtype=dtype)

    def compute_error(g):
        tau, x, y, e = mesh5(g, B=B, device=device, dtype=dtype)
        Lx = (g["x"][-1] + g["dx"]).item()
        Ly = (g["y"][-1] + g["dy"]).item()
        Le = (g["eta"][-1] + g["deta"]).item()

        f, exact = manufactured_scalar(tau, x, y, e, Lx, Ly, Le)
        dtau, dx, dy, deta = g["dtau"], g["dx"], g["dy"], g["deta"]

        num = [
            partial_mu(f, 0, dtau, dx, dy, deta),
            partial_mu(f, 1, dtau, dx, dy, deta),
            partial_mu(f, 2, dtau, dx, dy, deta),
            partial_mu(f, 3, dtau, dx, dy, deta),
        ]

        sl = interior()
        errs = [rel_l2(num[i][sl] - exact[i][sl], exact[i][sl]) for i in range(4)]
        return errs

    ec = compute_error(gc)
    ef = compute_error(gf)

    # estimate order for each derivative
    orders = [observed_order(ec[i], ef[i], 2.0) for i in range(4)]

    # allow some slack from mixed boundary handling + periodic dims
    for i, p in enumerate(orders):
        assert p > 1.75, f"partial_mu dir={i} order too low: p={p:.3f}"


# ---------- Test 2: cov_div_T_milne convergence ----------
def test_cov_div_T_milne_second_order_convergence():
    # replace import path with your module
    from __main__ import cov_div_T_milne

    device, dtype = "cpu", torch.float64
    B = 1

    gc = make_grid(Nt=41, Nx=16, Ny=16, Neta=16, dtau=0.01, dx=0.2, dy=0.2, deta=0.2,
                   device=device, dtype=dtype)
    gf = make_grid(Nt=81, Nx=32, Ny=32, Neta=32, dtau=0.005, dx=0.1, dy=0.1, deta=0.1,
                   device=device, dtype=dtype)

    def compute_error(g):
        tau, x, y, e = mesh5(g, B=B, device=device, dtype=dtype)
        dtau, dx, dy, deta = g["dtau"], g["dx"], g["dy"], g["deta"]

        A, Bxx, Cyy, Dee = manufactured_Tdiag(tau)
        z = torch.zeros_like(tau)
        T = [[z for _ in range(4)] for _ in range(4)]
        T[0][0] = A
        T[1][1] = Bxx
        T[2][2] = Cyy
        T[3][3] = Dee

        R = cov_div_T_milne(T, tau, dtau, dx, dy, deta)
        Rtau, Rx, Ry, Reta = R

        # exact:
        # R^tau = (1/tau) d_tau(tau A) + tau D = 4 tau^2 + 1
        # R^x = 0, R^y = 0, R^eta = 0
        Rtau_ex = 4.0 * tau**2 + 1.0
        zex = torch.zeros_like(tau)

        sl = interior()
        e_tau = rel_l2(Rtau[sl] - Rtau_ex[sl], Rtau_ex[sl])
        # for zero references, use absolute L2
        e_x = torch.sqrt(torch.mean((Rx[sl] - zex[sl])**2)).item()
        e_y = torch.sqrt(torch.mean((Ry[sl] - zex[sl])**2)).item()
        e_e = torch.sqrt(torch.mean((Reta[sl] - zex[sl])**2)).item()

        return e_tau, e_x, e_y, e_e

    ec = compute_error(gc)
    ef = compute_error(gf)

    # R^tau should show ~2nd order
    p_tau = observed_order(ec[0], ef[0], 2.0)
    assert p_tau > 1.75, f"cov_div R^tau order too low: p={p_tau:.3f}"

    # zero components should also decrease ~2nd order if nonzero numerically
    for name, ec_i, ef_i in [("R^x", ec[1], ef[1]), ("R^y", ec[2], ef[2]), ("R^eta", ec[3], ef[3])]:
        # if already near machine precision at coarse, skip order check
        if ec_i > 1e-13 and ef_i > 1e-13:
            p = observed_order(ec_i, ef_i, 2.0)
            assert p > 1.5, f"{name} order too low: p={p:.3f}"


# ---------- Optional: strict regression thresholds ----------
@pytest.mark.parametrize("tol", [1e-3])
def test_regression_thresholds_fine_grid(tol):
    # replace import path with your module
    from __main__ import partial_mu, cov_div_T_milne

    device, dtype = "cpu", torch.float64
    g = make_grid(Nt=81, Nx=32, Ny=32, Neta=32, dtau=0.005, dx=0.1, dy=0.1, deta=0.1,
                  device=device, dtype=dtype)
    B = 1
    tau, x, y, e = mesh5(g, B=B, device=device, dtype=dtype)
    dtau, dx, dy, deta = g["dtau"], g["dx"], g["dy"], g["deta"]
    sl = interior()

    # scalar derivative threshold
    Lx = (g["x"][-1] + g["dx"]).item()
    Ly = (g["y"][-1] + g["dy"]).item()
    Le = (g["eta"][-1] + g["deta"]).item()
    f, exact = manufactured_scalar(tau, x, y, e, Lx, Ly, Le)

    dnum = [
        partial_mu(f, 0, dtau, dx, dy, deta),
        partial_mu(f, 1, dtau, dx, dy, deta),
        partial_mu(f, 2, dtau, dx, dy, deta),
        partial_mu(f, 3, dtau, dx, dy, deta),
    ]
    errs = [rel_l2(dnum[i][sl] - exact[i][sl], exact[i][sl]) for i in range(4)]
    assert max(errs) < 8 * tol, f"partial_mu regression failed, max err={max(errs):.3e}"

    # cov div threshold
    A, Bxx, Cyy, Dee = manufactured_Tdiag(tau)
    z = torch.zeros_like(tau)
    T = [[z for _ in range(4)] for _ in range(4)]
    T[0][0] = A
    T[1][1] = Bxx
    T[2][2] = Cyy
    T[3][3] = Dee
    Rtau, Rx, Ry, Reta = cov_div_T_milne(T, tau, dtau, dx, dy, deta)

    Rtau_ex = 4.0 * tau**2 + 1.0
    e_tau = rel_l2(Rtau[sl] - Rtau_ex[sl], Rtau_ex[sl])
    assert e_tau < 8 * tol, f"cov_div R^tau regression failed: err={e_tau:.3e}"
