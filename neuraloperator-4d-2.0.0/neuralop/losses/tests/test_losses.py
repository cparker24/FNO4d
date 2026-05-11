import math
import torch
import pytest
from torch.testing import assert_close

from ..data_losses import LpLoss, H1Loss, HdivLoss
from ..differentiation import non_uniform_fd, FiniteDiff, FourierDiff
from neuralop.layers.embeddings import regular_grid_nd


def test_lploss():
    l2_2d_mean = LpLoss(d=2, p=2, reduction="mean", measure=1.0)
    l2_2d_sum = LpLoss(d=2, p=2, reduction="sum", measure=1.0)
    x = torch.randn(10, 1, 4, 4)

    abs_0 = l2_2d_mean.abs(x, x)
    assert abs_0.item() == 0.0

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # L2 w/out normalizing constant
    # sum of items in each element in ones is 16
    # norm is 4
    mean_abs_l2_err = l2_2d_mean.abs(zeros, ones)
    assert mean_abs_l2_err.item() == 1.0

    sum_abs_l2_err = l2_2d_sum.abs(zeros, ones)
    assert sum_abs_l2_err.item() == 10.0

    # Test quadrature weights: for spatial dims, each weight should
    # be 1/the corresponding size in x, so that after applying weights,
    # lploss constitutes an average over spatial dims.
    d = 2
    default_quad_weights = l2_2d_mean.uniform_quadrature(x)
    assert default_quad_weights[-d:] == [1 / s for s in x.shape[-d:]]

    # Sanity check: ensure that both sum and mean reduction sum-reduce over spatial dims
    x = torch.arange(1, 5) * torch.ones(4, 4)
    flipped_x = x.T.unsqueeze(0).unsqueeze(0)
    x = x.unsqueeze(0).unsqueeze(0)

    zeros = torch.zeros_like(x)

    # batch and channel size of 1. squared diff should be 120, avg over spatial dims 7.5
    mean_err = l2_2d_mean.abs(flipped_x, zeros)
    sum_err = l2_2d_sum.abs(x, zeros)

    assert mean_err == sum_err
    assert_close(mean_err, torch.sqrt(torch.tensor(7.5)))


def test_h1loss():
    h1 = H1Loss(d=2, reduction="mean")
    x = torch.randn(10, 4, 4)

    abs_0 = h1.abs(x, x)
    assert abs_0.item() == 0.0

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # H1 w/o normalizing constant,
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 1 averaged in space
    mean_abs_h1 = h1.abs(zeros, ones)
    assert mean_abs_h1.item() == 1.0


def test_hdivloss():
    hdiv = HdivLoss(d=2, reduction="mean")
    x = torch.randn(10, 4, 4)

    abs_0 = hdiv.abs(x, x)
    assert abs_0.item() == 0.0

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # Hdiv without normalizing constant,
    # finite-difference derivatives of both sides are zero
    # sum of items in each element in ones is 16
    # norm is 1 averaged in space
    mean_abs_hdiv = hdiv.abs(zeros, ones)
    assert mean_abs_hdiv.item() == 1.0


@pytest.mark.parametrize("dim", [1, 2])
def test_nonuniform_fd(dim: int):
    if dim == 1:
        x = torch.sort(torch.rand(256))[0].unsqueeze(1)
        f = torch.exp(3 * x) + torch.sin(10 * x) - x**2
        df_ref = (3 * torch.exp(3 * x) + 10 * torch.cos(10 * x) - 2 * x).squeeze()
        df_dx = non_uniform_fd(
            x,
            f.squeeze(),
            num_neighbors=3,
            derivative_indices=[0],
            regularize_lstsq=False,
        )[0]

        l2 = LpLoss(d=1, p=2, reduction="mean", measure=1.0)
        assert l2.rel(df_ref, df_dx).item() < 5e-2

        # # Plot to check visually
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10,5))
        # plt.plot(x.detach().numpy().squeeze(), df_ref.detach().numpy(), 'r', linewidth=0.8, label='Reference')
        # plt.plot(x.detach().numpy(), df_dx.detach().numpy(), 'b', linewidth=0.8, label='FD')
        # plt.legend()
        # plt.xlabel(r'$x$')
        # plt.ylabel(r'$df/dx$')
        # plt.savefig('non_uniform_fd_1D.pdf')

    if dim == 2:
        num_points = 128
        x = torch.linspace(-1, 1, num_points, dtype=torch.float64)
        y = torch.linspace(-1, 1, num_points, dtype=torch.float64)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)
        f = torch.exp(Y) + 0.1 * torch.sin(10 * X) - (X**2) * (Y**2)
        dfdx_ref = torch.cos(10 * X) - 2 * X * (Y**2)
        dfdy_ref = torch.exp(Y) - 2 * (X**2) * Y
        df = non_uniform_fd(
            points,
            f.flatten(),
            num_neighbors=5,
            derivative_indices=[0, 1],
            regularize_lstsq=True,
        )
        df_dx = df[0].reshape(num_points, num_points)
        df_dy = df[1].reshape(num_points, num_points)

        l2 = LpLoss(d=2, p=2, reduction="mean", measure=1.0)
        assert l2.rel(dfdx_ref, df_dx).item() < 5e-2
        assert l2.rel(dfdy_ref, df_dy).item() < 5e-2

        # # Plot to check visually
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(7, 7))
        # plt.subplot(2, 2, 1)
        # img1 = plt.imshow(df_dx.detach().numpy())
        # plt.colorbar(img1, shrink=0.75)
        # plt.title(r'$df/dx$')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(2, 2, 2)
        # img2 = plt.imshow(dfdx_ref.detach().numpy())
        # plt.colorbar(img2, shrink=0.75)
        # plt.title(r'$df/dx$_ref')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(2, 2, 3)
        # img3 = plt.imshow(df_dy.detach().numpy())
        # plt.colorbar(img3, shrink=0.75)
        # plt.title(r'$df/dy$')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplot(2, 2, 4)
        # img4 = plt.imshow(dfdy_ref.detach().numpy().reshape(num_points, num_points))
        # plt.colorbar(img4, shrink=0.75)
        # plt.title(r'$df/dy$_ref')
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig('non_uniform_fd_2D.pdf')


@pytest.mark.parametrize("periodic_x", [True, False])
def test_finite_diff_1d(periodic_x):
    """Test the FiniteDiff class with various boundary conditions for 1D."""

    # Create a 1D test function: f(x) = cos(x) - x
    nx = 64
    x = torch.linspace(0, 2 * torch.pi, nx, dtype=torch.float64)
    f = torch.cos(x) - x

    # Initialize FiniteDiff with dim=1 and specified boundary conditions
    fd1d = FiniteDiff(dim=1, h=2 * torch.pi / nx, periodic_in_x=periodic_x)

    # Test first order derivatives
    df_dx = fd1d.dx(f)

    # Test second order derivatives
    d2f_dx2 = fd1d.dx(f, order=2)

    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert d2f_dx2.shape == f.shape

    # Test that first derivative is approximately correct for f(x) = cos(x) - x
    # df/dx ≈ -sin(x) - 1 for f(x) = cos(x) - x
    if not periodic_x:
        # For non-periodic, interior points should be close to expected value
        expected_df_dx = -torch.sin(x[2:-2]) - 1.0
        assert_close(df_dx[2:-2], expected_df_dx, atol=0.01, rtol=0.1)


@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
def test_finite_diff_2d(periodic_x, periodic_y):
    """Test the FiniteDiff class with various boundary conditions for 2D."""

    # Create a 2D test function: f(x,y) = x^2 + y^2
    nx, ny = 32, 32
    x = torch.linspace(0, 2 * torch.pi, nx, dtype=torch.float64)
    y = torch.linspace(0, 2 * torch.pi, ny, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    f = X**2 + Y**2

    # Initialize FiniteDiff with dim=2 and specified boundary conditions
    fd2d = FiniteDiff(
        dim=2, h=(0.2, 0.2), periodic_in_x=periodic_x, periodic_in_y=periodic_y
    )

    # Test first order derivatives
    df_dx = fd2d.dx(f)
    df_dy = fd2d.dy(f)

    # Test second order derivatives
    d2f_dx2 = fd2d.dx(f, order=2)
    d2f_dy2 = fd2d.dy(f, order=2)

    # Test laplacian
    laplacian = fd2d.laplacian(f)

    # Test gradient
    gradient = fd2d.gradient(f)

    # Test with vector field for divergence and curl
    # Create a simple vector field: u = [sin(x), cos(y)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u_vector = torch.stack([u1, u2], dim=0)

    divergence = fd2d.divergence(u_vector)
    curl = fd2d.curl(u_vector)

    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert d2f_dx2.shape == f.shape
    assert d2f_dy2.shape == f.shape
    assert laplacian.shape == f.shape
    assert gradient.shape == (2, nx, ny)
    assert divergence.shape == (nx, ny)
    assert curl.shape == (nx, ny)

    # Test that laplacian equals sum of second derivatives
    assert_close(laplacian, d2f_dx2 + d2f_dy2)


@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
@pytest.mark.parametrize("periodic_z", [True, False])
def test_finite_diff_3d(periodic_x, periodic_y, periodic_z):
    """Test the FiniteDiff class with various boundary conditions for 3D."""

    # Create a 3D test function: f(x,y,z) = x^2 + y^2 + z^2
    nx, ny, nz = 16, 16, 16
    x = torch.linspace(0, 2 * torch.pi, nx, dtype=torch.float64)
    y = torch.linspace(0, 2 * torch.pi, ny, dtype=torch.float64)
    z = torch.linspace(0, 2 * torch.pi, nz, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    f = X**2 + Y**2 + Z**2

    # Initialize FiniteDiff with dim=3 and specified boundary conditions
    fd3d = FiniteDiff(
        dim=3,
        h=(0.4, 0.4, 0.4),
        periodic_in_x=periodic_x,
        periodic_in_y=periodic_y,
        periodic_in_z=periodic_z,
    )

    # Test first order derivatives
    df_dx = fd3d.dx(f)
    df_dy = fd3d.dy(f)
    df_dz = fd3d.dz(f)

    # Test second order derivatives
    d2f_dx2 = fd3d.dx(f, order=2)
    d2f_dy2 = fd3d.dy(f, order=2)
    d2f_dz2 = fd3d.dz(f, order=2)

    # Test laplacian
    laplacian = fd3d.laplacian(f)

    # Test gradient
    gradient = fd3d.gradient(f)

    # Test with vector field for divergence and curl
    # Create a simple vector field: u = [sin(x), cos(y), sin(z)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u3 = torch.sin(Z)
    u_vector = torch.stack([u1, u2, u3], dim=0)

    divergence = fd3d.divergence(u_vector)
    curl = fd3d.curl(u_vector)

    # Basic shape assertions
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert df_dz.shape == f.shape
    assert d2f_dx2.shape == f.shape
    assert d2f_dy2.shape == f.shape
    assert d2f_dz2.shape == f.shape
    assert laplacian.shape == f.shape
    assert gradient.shape == (3, nz, ny, nx)
    assert divergence.shape == (nx, ny, nz)
    assert curl.shape == (3, nz, ny, nx)

    # Test that laplacian equals sum of second derivatives
    assert_close(laplacian, d2f_dx2 + d2f_dy2 + d2f_dz2)


@pytest.mark.parametrize("periodic", [True, False])
def test_fourier_diff(periodic: bool):
    if periodic:
        ## Test on periodic functions without Fourier continuation
        # Consider sin(x) and cos(x)
        L = 2 * torch.pi
        x = torch.linspace(0, L, 101)[:-1]
        f = torch.stack([torch.sin(x), torch.cos(x)], dim=0)

        # Use FourierDiff class
        fd1d = FourierDiff(dim=1, L=L, use_fc=False)
        derivatives = fd1d.compute_multiple_derivatives(f, [1, 2, 3])
        dfdx, df2dx2, df3dx3 = derivatives

        assert f.shape == dfdx.shape == df2dx2.shape == df3dx3.shape

    else:
        ## Test on non-periodic functions using Fourier continuation
        # Consider sin(16*x)-cos(8*x) and exp(-0.8x)
        L = 2 * torch.pi
        x = torch.linspace(0, L, 101)[:-1]
        f = torch.stack([torch.sin(3 * x) - torch.cos(x), torch.exp(-0.8 * x) + torch.sin(x)], dim=0)

        # Use FourierDiff class with Fourier continuation
        fd1d = FourierDiff(dim=1, L=L, use_fc="Legendre", fc_degree=4, fc_n_additional_pts=30)
        derivatives = fd1d.compute_multiple_derivatives(f, [1, 2])
        dfdx, df2dx2 = derivatives

        assert f.shape == dfdx.shape == df2dx2.shape


@pytest.mark.parametrize("use_fc", [False, "Legendre"])
def test_fourier_diff_2d(use_fc):
    if use_fc:
        fd2d = FourierDiff(
            dim=2,
            L=(2 * torch.pi, 2 * torch.pi),
            use_fc=use_fc,
            fc_degree=4,
            fc_n_additional_pts=20,
        )
    else:
        fd2d = FourierDiff(dim=2, L=(2 * torch.pi, 2 * torch.pi))

    # Create a 2D periodic function: sin(x) * cos(y)
    L_x, L_y = 2 * torch.pi, 2 * torch.pi
    nx, ny = 64, 64
    x = torch.linspace(0, L_x, nx, dtype=torch.float64)
    y = torch.linspace(0, L_y, ny, dtype=torch.float64)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Test function: f(x,y) = sin(x) * cos(y) for periodic, exp(-x) * sin(y) for non-periodic
    if use_fc:
        f = torch.exp(-X) * torch.sin(Y)  # Non-periodic function
    else:
        f = torch.sin(X) * torch.cos(Y)  # Periodic function

    # Test partial derivatives
    df_dx_computed = fd2d.dx(f)
    df_dy_computed = fd2d.dy(f)

    # Test higher order derivatives
    d2f_dx2_computed = fd2d.dx(f, order=2)
    d2f_dy2_computed = fd2d.dy(f, order=2)

    # Test laplacian
    laplacian_computed = fd2d.laplacian(f)

    # Test multiple derivatives computation
    derivatives = fd2d.compute_multiple_derivatives(f, [(1, 0), (0, 1), (2, 0), (0, 2)])
    df_dx_multi, df_dy_multi, d2f_dx2_multi, d2f_dy2_multi = derivatives

    # Test gradient
    gradient = fd2d.gradient(f)
    assert gradient.shape == (2, nx, ny)

    # Test divergence with a vector field
    # Create a vector field: u = [sin(x), cos(y)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u_vector = torch.stack([u1, u2], dim=0)
    divergence = fd2d.divergence(u_vector)

    # Test curl with a vector field
    curl = fd2d.curl(u_vector)

    # Basic shape assertions - just check that derivatives have correct shapes
    assert df_dx_computed.shape == f.shape
    assert df_dy_computed.shape == f.shape
    assert d2f_dx2_computed.shape == f.shape
    assert d2f_dy2_computed.shape == f.shape
    assert laplacian_computed.shape == f.shape
    assert len(derivatives) == 4
    assert gradient.shape == (2, nx, ny)
    assert divergence.shape == (nx, ny)
    assert curl.shape == (nx, ny)


@pytest.mark.parametrize("use_fc", [False, "Gram"])
def test_fourier_diff_3d(use_fc):
    """Test the FourierDiff class with various scenarios."""

    # Test basic functionality
    if use_fc:
        fd3d = FourierDiff(
            dim=3,
            L=(2 * torch.pi, 2 * torch.pi, 2 * torch.pi),
            use_fc=use_fc,
            fc_degree=4,
            fc_n_additional_pts=50,
        )
    else:
        fd3d = FourierDiff(dim=3, L=(2 * torch.pi, 2 * torch.pi, 2 * torch.pi))

    # Create a 3D periodic function: sin(x) * cos(y) * sin(z)
    L_x, L_y, L_z = 2 * torch.pi, 2 * torch.pi, 2 * torch.pi
    nx, ny, nz = 32, 32, 32
    x = torch.linspace(0, L_x, nx, dtype=torch.float64)
    y = torch.linspace(0, L_y, ny, dtype=torch.float64)
    z = torch.linspace(0, L_z, nz, dtype=torch.float64)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

    # Test function: f(x,y,z) = sin(x) * cos(y) * sin(z) for periodic, exp(-x) * sin(y) * cos(z) for non-periodic
    if use_fc:
        f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # Non-periodic function
    else:
        f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)  # Periodic function

    # Test partial derivatives
    df_dx_computed = fd3d.dx(f)
    df_dy_computed = fd3d.dy(f)
    df_dz_computed = fd3d.dz(f)

    # Test higher order derivatives
    d2f_dx2_computed = fd3d.dx(f, order=2)
    d2f_dy2_computed = fd3d.dy(f, order=2)
    d2f_dz2_computed = fd3d.dz(f, order=2)
    d2f_dxdy_computed = fd3d.dx(fd3d.dy(f))
    d2f_dxdz_computed = fd3d.dx(fd3d.dz(f))
    d2f_dydz_computed = fd3d.dy(fd3d.dz(f))

    # Test laplacian
    laplacian_computed = fd3d.laplacian(f)

    # Test multiple derivatives computation
    derivatives = fd3d.compute_multiple_derivatives(f, [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)])
    df_dx_multi, df_dy_multi, df_dz_multi, d2f_dx2_multi, d2f_dy2_multi, d2f_dz2_multi, d2f_dxdy_multi, d2f_dxdz_multi, d2f_dydz_multi = derivatives
    
    # Test gradient
    gradient = fd3d.gradient(f)
    assert gradient.shape == (3, nz, ny, nx)

    # Test divergence with a vector field
    # Create a vector field: u = [sin(x), cos(y), sin(z)]
    u1 = torch.sin(X)
    u2 = torch.cos(Y)
    u3 = torch.sin(Z)
    u_vector = torch.stack([u1, u2, u3], dim=0)
    divergence = fd3d.divergence(u_vector)

    # Test curl with a vector field
    curl = fd3d.curl(u_vector)

    # Basic shape assertions - just check that derivatives have correct shapes
    assert df_dx_computed.shape == f.shape
    assert df_dy_computed.shape == f.shape
    assert df_dz_computed.shape == f.shape
    assert d2f_dx2_computed.shape == f.shape
    assert d2f_dy2_computed.shape == f.shape
    assert d2f_dz2_computed.shape == f.shape
    assert d2f_dxdy_computed.shape == f.shape
    assert d2f_dxdz_computed.shape == f.shape
    assert d2f_dydz_computed.shape == f.shape
    assert laplacian_computed.shape == f.shape
    assert len(derivatives) == 9
    assert gradient.shape == (3, nz, ny, nx)
    assert divergence.shape == (nx, ny, nz)
    assert curl.shape == (3, nz, ny, nx)


# ---------------------------------------------------------------------------
# 3D regression tests — confirm the dim==3 path is unchanged after the 4D
# extension.  We use analytical functions so any regression in accuracy is
# immediately visible.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
@pytest.mark.parametrize("periodic_z", [True, False])
def test_finite_diff_3d_regression(periodic_x, periodic_y, periodic_z):
    """Regression: FiniteDiff dim=3 still gives correct analytical derivatives
    for f(x,y,z) = sin(x)*cos(y)*sin(z) with known partial derivatives."""

    nx, ny, nz = 64, 64, 64
    h = 2 * torch.pi / nx
    # Use n+1 points and drop the endpoint so the periodic spacing is exactly h
    x = torch.linspace(0, 2 * torch.pi, nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, 2 * torch.pi, ny + 1, dtype=torch.float64)[:-1]
    z = torch.linspace(0, 2 * torch.pi, nz + 1, dtype=torch.float64)[:-1]
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)

    fd3d = FiniteDiff(
        dim=3, h=h,
        periodic_in_x=periodic_x, periodic_in_y=periodic_y, periodic_in_z=periodic_z,
    )

    df_dx = fd3d.dx(f)
    df_dy = fd3d.dy(f)
    df_dz = fd3d.dz(f)
    laplacian = fd3d.laplacian(f)

    # Shape regression
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert df_dz.shape == f.shape
    assert laplacian.shape == f.shape

    # Laplacian is self-consistent: sum of 2nd derivatives
    assert_close(laplacian, fd3d.dx(f, order=2) + fd3d.dy(f, order=2) + fd3d.dz(f, order=2))

    # Analytical check on interior points only (boundary stencil differs when non-periodic)
    sl = (slice(4, -4),) * 3
    expected_dx = torch.cos(X) * torch.cos(Y) * torch.sin(Z)
    expected_dy = -torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    expected_dz = torch.sin(X) * torch.cos(Y) * torch.cos(Z)

    assert_close(df_dx[sl], expected_dx[sl], atol=2e-3, rtol=1e-2)
    assert_close(df_dy[sl], expected_dy[sl], atol=2e-3, rtol=1e-2)
    assert_close(df_dz[sl], expected_dz[sl], atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("use_fc", [False])
def test_fourier_diff_3d_regression(use_fc):
    """Regression: FourierDiff dim=3 still gives correct analytical derivatives
    for f(x,y,z) = sin(x)*cos(y)*sin(z)."""

    L = 2 * torch.pi
    nx, ny, nz = 32, 32, 32
    # Drop the endpoint so the periodic grid has exactly n points over [0, L)
    x = torch.linspace(0, L, nx + 1, dtype=torch.float64)[:-1]
    y = torch.linspace(0, L, ny + 1, dtype=torch.float64)[:-1]
    z = torch.linspace(0, L, nz + 1, dtype=torch.float64)[:-1]
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z)

    fd3d = FourierDiff(dim=3, L=(L, L, L), use_fc=use_fc)

    df_dx = fd3d.dx(f)
    df_dy = fd3d.dy(f)
    df_dz = fd3d.dz(f)
    laplacian = fd3d.laplacian(f)

    # Shape regression
    assert df_dx.shape == f.shape
    assert df_dy.shape == f.shape
    assert df_dz.shape == f.shape
    assert laplacian.shape == f.shape

    # Analytical values — Fourier differentiation is spectrally accurate
    expected_dx = torch.cos(X) * torch.cos(Y) * torch.sin(Z)
    expected_dy = -torch.sin(X) * torch.sin(Y) * torch.sin(Z)
    expected_dz = torch.sin(X) * torch.cos(Y) * torch.cos(Z)

    assert_close(df_dx, expected_dx, atol=1e-9, rtol=1e-6)
    assert_close(df_dy, expected_dy, atol=1e-9, rtol=1e-6)
    assert_close(df_dz, expected_dz, atol=1e-9, rtol=1e-6)

    # Laplacian: ∇²f = -sin(x)cos(y)sin(z) - sin(x)cos(y)sin(z) - sin(x)cos(y)sin(z) = -3f
    assert_close(laplacian, -3.0 * f, atol=1e-9, rtol=1e-6)


def test_h1loss_3d_regression():
    """Regression: H1Loss d=3 still behaves correctly after the 4D extension."""
    h1 = H1Loss(d=3, reduction="mean")
    x = torch.randn(4, 8, 8, 8)

    # Self-comparison must be zero
    assert h1.abs(x, x).item() == pytest.approx(0.0, abs=1e-6)

    # Constant vs zero: spatial derivatives vanish, only L2 term remains
    zeros = torch.zeros(4, 8, 8, 8)
    ones = torch.ones(4, 8, 8, 8)
    val = h1.abs(zeros, ones).item()
    assert val > 0.0


def test_hdivloss_3d_regression():
    """Regression: HdivLoss d=3 still behaves correctly after the 4D extension."""
    hdiv = HdivLoss(d=3, reduction="mean")
    x = torch.randn(4, 8, 8, 8)

    assert hdiv.abs(x, x).item() == pytest.approx(0.0, abs=1e-6)

    zeros = torch.zeros(4, 8, 8, 8)
    ones = torch.ones(4, 8, 8, 8)
    val = hdiv.abs(zeros, ones).item()
    assert val > 0.0


# ---------------------------------------------------------------------------
# 4D tests — FiniteDiff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic_x", [True, False])
@pytest.mark.parametrize("periodic_y", [True, False])
@pytest.mark.parametrize("periodic_z", [True, False])
@pytest.mark.parametrize("periodic_t", [True, False])
def test_finite_diff_4d_shapes(periodic_x, periodic_y, periodic_z, periodic_t):
    """FiniteDiff dim=4: all derivative methods return the correct shape."""

    nx, ny, nz, nt = 8, 8, 8, 8
    f = torch.rand(nx, ny, nz, nt, dtype=torch.float64)

    fd4d = FiniteDiff(
        dim=4, h=0.1,
        periodic_in_x=periodic_x, periodic_in_y=periodic_y,
        periodic_in_z=periodic_z, periodic_in_t=periodic_t,
    )

    assert fd4d.dx(f).shape == f.shape
    assert fd4d.dy(f).shape == f.shape
    assert fd4d.dz(f).shape == f.shape
    assert fd4d.dt(f).shape == f.shape

    assert fd4d.dx(f, order=2).shape == f.shape
    assert fd4d.dy(f, order=2).shape == f.shape
    assert fd4d.dz(f, order=2).shape == f.shape
    assert fd4d.dt(f, order=2).shape == f.shape

    assert fd4d.laplacian(f).shape == f.shape

    # gradient: stacks 4 components at dim=-5 → (4, nx, ny, nz, nt)
    grad = fd4d.gradient(f)
    assert grad.shape == (4, nx, ny, nz, nt)

    # divergence: 4-component vector field → scalar field
    v = torch.rand(4, nx, ny, nz, nt, dtype=torch.float64)
    assert fd4d.divergence(v).shape == f.shape

    # curl is not defined for 4D
    with pytest.raises(NotImplementedError):
        fd4d.curl(v)


def test_finite_diff_4d_laplacian_consistency():
    """FiniteDiff dim=4: laplacian equals sum of four 2nd-order derivatives."""

    nx = ny = nz = nt = 16
    f = torch.rand(nx, ny, nz, nt, dtype=torch.float64)
    fd4d = FiniteDiff(dim=4, h=0.1)

    lap = fd4d.laplacian(f)
    expected = (
        fd4d.dx(f, order=2)
        + fd4d.dy(f, order=2)
        + fd4d.dz(f, order=2)
        + fd4d.dt(f, order=2)
    )
    assert_close(lap, expected)


def test_finite_diff_4d_numerical_accuracy():
    """FiniteDiff dim=4 (periodic): derivatives of sin(x)cos(y)sin(z)cos(t) match
    the analytical values in the interior."""

    n = 64
    h = 2 * torch.pi / n
    # Drop endpoint so periodic spacing is exactly h
    coords = torch.linspace(0, 2 * torch.pi, n + 1, dtype=torch.float64)[:-1]
    X, Y, Z, T = torch.meshgrid(coords, coords, coords, coords, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T)

    fd4d = FiniteDiff(dim=4, h=h, periodic_in_x=True, periodic_in_y=True,
                      periodic_in_z=True, periodic_in_t=True)

    sl = (slice(4, -4),) * 4

    assert_close(fd4d.dx(f)[sl],
                 (torch.cos(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T))[sl],
                 atol=2e-3, rtol=1e-2)
    assert_close(fd4d.dy(f)[sl],
                 (-torch.sin(X) * torch.sin(Y) * torch.sin(Z) * torch.cos(T))[sl],
                 atol=2e-3, rtol=1e-2)
    assert_close(fd4d.dz(f)[sl],
                 (torch.sin(X) * torch.cos(Y) * torch.cos(Z) * torch.cos(T))[sl],
                 atol=2e-3, rtol=1e-2)
    assert_close(fd4d.dt(f)[sl],
                 (-torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.sin(T))[sl],
                 atol=2e-3, rtol=1e-2)


def test_finite_diff_4d_batched():
    """FiniteDiff dim=4 works correctly with leading batch and channel dims."""

    batch, channels, nx, ny, nz, nt = 3, 2, 8, 8, 8, 8
    f = torch.rand(batch, channels, nx, ny, nz, nt, dtype=torch.float64)
    fd4d = FiniteDiff(dim=4, h=0.1)

    for method in [fd4d.dx, fd4d.dy, fd4d.dz, fd4d.dt]:
        out = method(f)
        assert out.shape == f.shape

    assert fd4d.laplacian(f).shape == f.shape
    assert fd4d.gradient(f).shape == (batch, channels, 4, nx, ny, nz, nt)


def test_finite_diff_4d_guard():
    """FiniteDiff raises errors for invalid dim and wrong usage."""

    # dim=5 not supported
    with pytest.raises(ValueError):
        FiniteDiff(dim=5, h=0.1)

    # dt not available for dim < 4
    fd3d = FiniteDiff(dim=3, h=0.1)
    f = torch.rand(8, 8, 8)
    with pytest.raises(ValueError):
        fd3d.dt(f)


def test_central_diff_4d_compat():
    """central_diff_4d backward-compat function returns four derivative tensors."""
    from neuralop.losses.differentiation import central_diff_4d

    f = torch.rand(2, 16, 16, 16, 16, dtype=torch.float64)
    dx, dy, dz, dt = central_diff_4d(f, h=0.1)
    assert dx.shape == f.shape
    assert dy.shape == f.shape
    assert dz.shape == f.shape
    assert dt.shape == f.shape


# ---------------------------------------------------------------------------
# 4D tests — FourierDiff
# ---------------------------------------------------------------------------

def test_fourier_diff_4d_shapes():
    """FourierDiff dim=4: all derivative methods return the correct shape."""

    L = 2 * torch.pi
    nx = ny = nz = nt = 16
    # Drop endpoint for correct periodic grid
    coords = torch.linspace(0, L, nx + 1, dtype=torch.float64)[:-1]
    X, Y, Z, T = torch.meshgrid(coords, coords, coords, coords, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T)

    fd4d = FourierDiff(dim=4, L=(L, L, L, L))

    assert fd4d.dx(f).shape == f.shape
    assert fd4d.dy(f).shape == f.shape
    assert fd4d.dz(f).shape == f.shape
    assert fd4d.dt(f).shape == f.shape

    assert fd4d.dx(f, order=2).shape == f.shape
    assert fd4d.dt(f, order=2).shape == f.shape
    assert fd4d.laplacian(f).shape == f.shape

    # gradient → (4, nx, ny, nz, nt)
    grad = fd4d.gradient(f)
    assert grad.shape == (4, nx, ny, nz, nt)

    # divergence
    v = torch.stack([f, f, f, f], dim=0)  # shape (4, nx, ny, nz, nt)
    assert fd4d.divergence(v).shape == f.shape

    # curl not defined
    with pytest.raises(NotImplementedError):
        fd4d.curl(v)


def test_fourier_diff_4d_numerical_accuracy():
    """FourierDiff dim=4 (periodic, no FC): spectral accuracy for
    f(x,y,z,t) = sin(x)*cos(y)*sin(z)*cos(t) on [0, 2π]^4."""

    L = 2 * torch.pi
    n = 32
    # Drop endpoint for correct periodic grid
    coords = torch.linspace(0, L, n + 1, dtype=torch.float64)[:-1]
    X, Y, Z, T = torch.meshgrid(coords, coords, coords, coords, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T)

    fd4d = FourierDiff(dim=4, L=(L, L, L, L))

    assert_close(fd4d.dx(f), torch.cos(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T),
                 atol=1e-9, rtol=1e-6)
    assert_close(fd4d.dy(f), -torch.sin(X) * torch.sin(Y) * torch.sin(Z) * torch.cos(T),
                 atol=1e-9, rtol=1e-6)
    assert_close(fd4d.dz(f), torch.sin(X) * torch.cos(Y) * torch.cos(Z) * torch.cos(T),
                 atol=1e-9, rtol=1e-6)
    assert_close(fd4d.dt(f), -torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.sin(T),
                 atol=1e-9, rtol=1e-6)

    # Laplacian: ∇²f = -4 * sin(x)*cos(y)*sin(z)*cos(t) = -4f  (four sin/cos modes each contribute -1)
    assert_close(fd4d.laplacian(f), -4.0 * f, atol=1e-9, rtol=1e-6)


def test_fourier_diff_4d_compute_multiple():
    """FourierDiff dim=4: compute_multiple_derivatives handles multiple order tuples."""

    L = 2 * torch.pi
    n = 16
    # Drop endpoint for correct periodic grid
    coords = torch.linspace(0, L, n + 1, dtype=torch.float64)[:-1]
    X, Y, Z, T = torch.meshgrid(coords, coords, coords, coords, indexing="ij")
    f = torch.sin(X) * torch.cos(Y) * torch.sin(Z) * torch.cos(T)

    fd4d = FourierDiff(dim=4, L=(L, L, L, L))

    derivs = fd4d.compute_multiple_derivatives(f, [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])
    assert len(derivs) == 4
    for d in derivs:
        assert d.shape == f.shape

    # Results must match individual calls
    assert_close(derivs[0], fd4d.dx(f), atol=1e-12, rtol=0)
    assert_close(derivs[1], fd4d.dy(f), atol=1e-12, rtol=0)
    assert_close(derivs[2], fd4d.dz(f), atol=1e-12, rtol=0)
    assert_close(derivs[3], fd4d.dt(f), atol=1e-12, rtol=0)


def test_fourier_diff_4d_guard():
    """FourierDiff raises ValueError for dim=5 and for wrong L length."""

    with pytest.raises(ValueError):
        FourierDiff(dim=5)

    with pytest.raises(ValueError):
        FourierDiff(dim=4, L=(1.0, 1.0, 1.0))  # wrong length


# ---------------------------------------------------------------------------
# 4D tests — H1Loss and HdivLoss
# ---------------------------------------------------------------------------

def test_h1loss_4d_self_comparison():
    """H1Loss d=4: abs(x, x) == 0."""
    h1 = H1Loss(d=4, reduction="mean")
    x = torch.randn(2, 8, 8, 8, 8)
    assert h1.abs(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_h1loss_4d_constant_vs_zero():
    """H1Loss d=4: zeros vs ones — spatial derivatives vanish, L2 term is 1."""
    h1 = H1Loss(d=4, reduction="mean")
    zeros = torch.zeros(2, 8, 8, 8, 8)
    ones = torch.ones(2, 8, 8, 8, 8)
    val = h1.abs(zeros, ones).item()
    assert val == pytest.approx(1.0, abs=1e-5)


def test_h1loss_4d_rel():
    """H1Loss d=4: rel(x, x) == 0 and rel(x, y) > 0 for x != y."""
    h1 = H1Loss(d=4, reduction="mean")
    x = torch.rand(2, 8, 8, 8, 8)
    y = torch.rand(2, 8, 8, 8, 8)
    assert h1.rel(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert h1.rel(x, y).item() > 0.0


@pytest.mark.parametrize("periodic_t", [True, False])
def test_h1loss_4d_periodic_in_t(periodic_t):
    """H1Loss d=4 accepts and uses the periodic_in_t argument."""
    h1 = H1Loss(d=4, reduction="mean", periodic_in_t=periodic_t)
    x = torch.rand(2, 8, 8, 8, 8)
    y = torch.rand(2, 8, 8, 8, 8)
    val = h1(x, y)
    assert val.item() > 0.0


def test_hdivloss_4d_self_comparison():
    """HdivLoss d=4: abs(x, x) == 0."""
    hdiv = HdivLoss(d=4, reduction="mean")
    x = torch.randn(2, 8, 8, 8, 8)
    assert hdiv.abs(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_hdivloss_4d_constant_vs_zero():
    """HdivLoss d=4: zeros vs ones — divergence of constants is zero,
    so result equals the plain L2 contribution."""
    hdiv = HdivLoss(d=4, reduction="mean")
    zeros = torch.zeros(2, 8, 8, 8, 8)
    ones = torch.ones(2, 8, 8, 8, 8)
    val = hdiv.abs(zeros, ones).item()
    assert val == pytest.approx(1.0, abs=1e-5)


def test_hdivloss_4d_rel():
    """HdivLoss d=4: rel(x, x) == 0 and rel(x, y) > 0 for x != y."""
    hdiv = HdivLoss(d=4, reduction="mean")
    x = torch.rand(2, 8, 8, 8, 8)
    y = torch.rand(2, 8, 8, 8, 8)
    assert hdiv.rel(x, x).item() == pytest.approx(0.0, abs=1e-6)
    assert hdiv.rel(x, y).item() > 0.0


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_h1loss_all_dims(d):
    """H1Loss accepts d=1,2,3,4 and produces a non-negative scalar."""
    h1 = H1Loss(d=d, reduction="mean")
    shape = (2,) + (8,) * d
    x = torch.rand(*shape)
    y = torch.rand(*shape)
    val = h1(x, y)
    assert val.item() >= 0.0


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_hdivloss_all_dims(d):
    """HdivLoss accepts d=1,2,3,4 and produces a non-negative scalar."""
    hdiv = HdivLoss(d=d, reduction="mean")
    shape = (2,) + (8,) * d
    x = torch.rand(*shape)
    y = torch.rand(*shape)
    val = hdiv(x, y)
    assert val.item() >= 0.0
