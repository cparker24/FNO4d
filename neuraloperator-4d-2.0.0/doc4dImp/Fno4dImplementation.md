# FNO 4D Implementation Documentation

**Model used:** GitHub Copilot — Claude Sonnet 4.6  
**Implementation date:** May 2026  
**Plan document:** [Fno4dPlan.md](Fno4dPlan.md)

---

## Overview

The goal of this work was to extend the [`neuraloperator`](https://github.com/neuraloperator/neuraloperator) library so that the differentiation and loss subsystems support **four-dimensional fields** with axes `(x, y, z, t)` — i.e., three spatial dimensions plus time. The base library supported up to three dimensions (`dim ∈ {1, 2, 3}`); those hard-coded limits were identified and lifted.

The FNO model itself (`neuralop/models/fno.py`) and the spectral convolution layers (`neuralop/layers/spectral_convolution.py`) were already dimension-agnostic via `n_modes` tuples of arbitrary length and generic `einsum` kernels, so no changes were required there. The changes were confined to two files in `neuralop/losses/`:

| File | Classes / Functions changed |
|---|---|
| `neuralop/losses/differentiation.py` | `FiniteDiff`, `FourierDiff`, `central_diff_4d` (new) |
| `neuralop/losses/data_losses.py` | `H1Loss`, `HdivLoss` |

Tests validating both the unchanged 3D behaviour and the new 4D behaviour were added to:

- `neuralop/losses/tests/test_losses.py`

---

## Motivation

When training neural operators on spatio-temporal PDE data (e.g. 3+1 dimensional fluid simulations, nuclear reaction networks, or heavy-ion collision fields), the loss function must measure not only the pointwise $L^2$ error but also how well the model captures **spatial and temporal gradients** — i.e., the $H^1$ Sobolev norm. The Sobolev norm requires computing finite-difference or spectral derivatives along every axis. Without 4D support, users would have to fold time into a channel dimension and lose the ability to penalise errors in the temporal derivative $\partial_t u$.

---

## Plan reference

The full implementation plan is described in [Fno4dPlan.md](Fno4dPlan.md). It is organised into four phases:

1. **Phase 1** — `FiniteDiff` 4D extensions (dim guard, `periodic_in_t`, `dt()`, private 4D stencils)  
2. **Phase 2** — `FourierDiff` 4D extensions (dim guard, 4D FFT kernel, `dt()`, 4D wrappers)  
3. **Phase 3** — `H1Loss` 4D branch  
4. **Phase 4** — `HdivLoss` 4D branch  

The plan also documents which files were explicitly **excluded** from change and why (see the [Out-of-scope](#out-of-scope-files) section below).

---

## Tensor layout convention

For 4D fields the last four dimensions of any tensor are always interpreted as:

```
[..., x, y, z, t]
```

i.e. `dim=-4` → x, `dim=-3` → y, `dim=-2` → z, `dim=-1` → t.

Leading batch and channel dimensions (`[batch, channel, x, y, z, t]`) are fully supported — all stencils operate on trailing dimensions and broadcasting handles the rest.

---

## File 1: `neuralop/losses/differentiation.py`

### 1.1 `FiniteDiff`

#### Constructor changes

```python
# BEFORE (original):
if dim not in [1, 2, 3]:
    raise ValueError("dim must be 1, 2, or 3")

# AFTER:
if dim not in [1, 2, 3, 4]:
    raise ValueError("dim must be 1, 2, 3, or 4")
```

A new `periodic_in_t=True` keyword argument was added and stored when `dim >= 4`:

```python
def __init__(
    self, dim, h=1.0, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True,
    periodic_in_t=True         # NEW
):
    ...
    if dim >= 4:
        self.periodic_in_t = periodic_in_t
```

The `h` validation was already dimension-aware (checks `len(h) == dim`), so it automatically handles 4-tuples.

#### Public dispatch methods

Each existing method (`dx`, `dy`, `dz`) gained a final `else: # dim == 4` branch. A completely new public method `dt()` was added:

```python
def dt(self, u, order=1):
    """Derivative with respect to t. Only available for dim=4."""
    if self.dim < 4:
        raise ValueError("dt is only available for 4D")
    return self._dt_4d(u, order)
```

`laplacian`, `gradient`, `divergence`, and `curl` were extended as follows:

```python
# laplacian — dim==4 branch
return self._dx_4d(u, 2) + self._dy_4d(u, 2) + self._dz_4d(u, 2) + self._dt_4d(u, 2)

# gradient — dim==4 branch (stacks at dim=-5 so shape is [..., 4, nx, ny, nz, nt])
return torch.stack([self._dx_4d(u,1), self._dy_4d(u,1),
                    self._dz_4d(u,1), self._dt_4d(u,1)], dim=-5)

# divergence — dim==4 branch (expects first non-spatial dim to have 4 components)
u1 = u[..., 0, :, :, :, :]
u2 = u[..., 1, :, :, :, :]
u3 = u[..., 2, :, :, :, :]
u4 = u[..., 3, :, :, :, :]
return self._dx_4d(u1,1) + self._dy_4d(u2,1) + self._dz_4d(u3,1) + self._dt_4d(u4,1)

# curl — dim==4 branch raises NotImplementedError (see design decisions)
raise NotImplementedError(
    "Curl is not defined for 4D fields. The 4D analog is an antisymmetric "
    "rank-2 tensor (Faraday tensor), not a simple vector operation."
)
```

#### Private 4D stencil implementations

Eight new private methods implement the finite-difference stencils for 4D tensors. They mirror the existing 3D implementations exactly, with the slice patterns adjusted to carry the extra trailing dimension.

Axis-to-dimension mapping:

| Axis | `torch` dim | Slice pattern (interior, non-periodic) |
|---|---|---|
| x | `dims=-4` | `u[..., 1:-1, :, :, :]` |
| y | `dims=-3` | `u[..., :, 1:-1, :, :]` |
| z | `dims=-2` | `u[..., :, :, 1:-1, :]` |
| t | `dims=-1` | `u[..., :, :, :, 1:-1]` |

Example — `_dx_1st_4d` (periodic case):

```python
def _dx_1st_4d(self, u):
    if self.periodic_in_x:
        return (torch.roll(u, -1, dims=-4) - torch.roll(u, 1, dims=-4)) / (2.0 * self.h[0])
    else:
        dx = torch.zeros_like(u)
        dx[..., 1:-1, :, :, :] = (u[..., 2:, :, :, :] - u[..., :-2, :, :, :]) / (2.0 * self.h[0])
        # left boundary (3rd-order forward): -11f0 + 18f1 - 9f2 + 2f3
        dx[..., 0, :, :, :] = (-11*u[...,0,:,:,:] + 18*u[...,1,:,:,:] - 9*u[...,2,:,:,:] + 2*u[...,3,:,:,:]) / (6.0 * self.h[0])
        # right boundary (3rd-order backward): -2f_{n-4} + 9f_{n-3} - 18f_{n-2} + 11f_{n-1}
        dx[..., -1, :, :, :] = (-2*u[...,-4,:,:,:] + 9*u[...,-3,:,:,:] - 18*u[...,-2,:,:,:] + 11*u[...,-1,:,:,:]) / (6.0 * self.h[0])
        return dx
```

The full set of 8 private methods is:

```
_dx_1st_4d / _dx_2nd_4d   →  dispatched by  _dx_4d(u, order)
_dy_1st_4d / _dy_2nd_4d   →  dispatched by  _dy_4d(u, order)
_dz_1st_4d / _dz_2nd_4d   →  dispatched by  _dz_4d(u, order)
_dt_1st_4d / _dt_2nd_4d   →  dispatched by  _dt_4d(u, order)
```

The 2nd-order (Laplacian component) stencils use:

- Interior: $(f_{i+1} - 2f_i + f_{i-1})/h^2$  
- Left boundary: $(2f_0 - 5f_1 + 4f_2 - f_3)/h^2$  
- Right boundary: $(-f_{n-4} + 4f_{n-3} - 5f_{n-2} + 2f_{n-1})/h^2$

#### Backward-compatibility function `central_diff_4d`

Following the same pattern as `central_diff_1d/2d/3d`, a new module-level function was added:

```python
def central_diff_4d(x, h, periodic_in_x=True, periodic_in_y=True,
                    periodic_in_z=True, periodic_in_t=True):
    """Deprecated convenience wrapper — use FiniteDiff(dim=4, ...) instead."""
    warnings.warn("central_diff_4d is deprecated ...", DeprecationWarning, stacklevel=2)
    fd4d = FiniteDiff(dim=4, h=h, periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y,
                      periodic_in_z=periodic_in_z, periodic_in_t=periodic_in_t)
    return fd4d.dx(x), fd4d.dy(x), fd4d.dz(x), fd4d.dt(x)
```

---

### 1.2 `FourierDiff`

#### Constructor changes

The same `dim` guard extension was applied:

```python
# BEFORE:
if dim not in [1, 2, 3]:
# AFTER:
if dim not in [1, 2, 3, 4]:
```

The `L` normalisation already handled tuples of arbitrary length via:

```python
if not isinstance(L, (tuple, list)):
    L = (L,) * dim
if len(L) != dim:
    raise ValueError(f"For {dim}D, L must be ... {dim} elements")
self.L = L[0] if dim == 1 else L   # keep scalar for 1D, tuple otherwise
```

This required no structural change — the 4-element tuple case falls through the same path.

#### New public method `dt`

```python
def dt(self, u, order=1):
    """Compute derivative with respect to t (4D only)."""
    if self.dim < 4:
        raise ValueError("dt method only available for 4D")
    return self._dt_4d(u, order)
```

#### Dispatch updates

`compute_multiple_derivatives`, `derivative`, `dx`, `dy`, `dz`, `gradient`, `divergence`, and `curl` each received a `elif self.dim == 4` / `else: # dim == 4` branch routing to the 4D kernel. The `partial()` helper was extended to recognise `direction="t"` when `dim >= 4`.

#### Core 4D spectral kernel: `_compute_multiple_derivatives_4d`

This is the primary algorithmic addition for the Fourier path. It generalises the 3D kernel to four dimensions:

```python
def _compute_multiple_derivatives_4d(self, u, derivatives):
    L_x, L_y, L_z, L_t = self.L          # domain lengths per axis
    nx, ny, nz, nt = u.shape[-4:]

    # 4D FFT over last 4 dimensions
    u_h = torch.fft.fftn(u.clone(), dim=(-4, -3, -2, -1))

    # Angular-frequency arrays  ω_k = 2π·fftfreq(n, d=L/n)
    k_x = torch.fft.fftfreq(nx, d=L_x/nx, device=u_h.device) * (2*torch.pi)
    k_y = torch.fft.fftfreq(ny, d=L_y/ny, device=u_h.device) * (2*torch.pi)
    k_z = torch.fft.fftfreq(nz, d=L_z/nz, device=u_h.device) * (2*torch.pi)
    k_t = torch.fft.fftfreq(nt, d=L_t/nt, device=u_h.device) * (2*torch.pi)

    # 4D frequency meshgrid
    KX, KY, KZ, KT = torch.meshgrid(k_x, k_y, k_z, k_t, indexing="ij")

    # Per-derivative stencil application
    results = []
    for order_x, order_y, order_z, order_t in derivatives:
        deriv_h = (
            (1j * KX.expand(u_h.shape)) ** order_x *
            (1j * KY.expand(u_h.shape)) ** order_y *
            (1j * KZ.expand(u_h.shape)) ** order_z *
            (1j * KT.expand(u_h.shape)) ** order_t
        ) * u_h
        results.append(deriv_h)

    # Single batch IFFT → real part
    derivatives_ft = torch.stack(results, dim=0)
    return list(torch.fft.ifftn(derivatives_ft, dim=(-4,-3,-2,-1)).real)
```

The key properties are:
- A **single FFT/IFFT pair** computes all requested derivatives simultaneously — efficient for multi-derivative callers such as `gradient` and `laplacian`.
- Fourier differentiation is **spectrally accurate** (infinite order) for periodic fields sampled at $n$ evenly-spaced points.
- Fourier continuation (FC) is **not** supported for 4D (it extends the domain along each axis independently and would require $O(n^4)$ work; a `NotImplementedError` is raised if `use_fc` is set with `dim=4`).
- Low-pass filtering (`low_pass_filter_ratio`) is supported to suppress aliasing artefacts.

#### Thin wrapper methods

```python
def _dx_4d(self, u, order):
    return self._compute_multiple_derivatives_4d(u, [(order, 0, 0, 0)])[0]

def _dy_4d(self, u, order):
    return self._compute_multiple_derivatives_4d(u, [(0, order, 0, 0)])[0]

def _dz_4d(self, u, order):
    return self._compute_multiple_derivatives_4d(u, [(0, 0, order, 0)])[0]

def _dt_4d(self, u, order):
    return self._compute_multiple_derivatives_4d(u, [(0, 0, 0, order)])[0]
```

#### Vector calculus operators for `FourierDiff` (4D)

Matching the `FiniteDiff` behaviour:

```python
# laplacian
return dx(u,2) + dy(u,2) + dz(u,2) + dt(u,2)

# gradient → shape [..., 4, nx, ny, nz, nt]
return torch.stack([dx(u), dy(u), dz(u), dt(u)], dim=-5)

# divergence — expects component dim at position -5
return dx(u[...,0,:,:,:,:]) + dy(u[...,1,:,:,:,:]) + dz(u[...,2,:,:,:,:]) + dt(u[...,3,:,:,:,:])

# curl — raises NotImplementedError
```

---

## File 2: `neuralop/losses/data_losses.py`

### 2.1 `H1Loss`

#### Constructor change

```python
# BEFORE:
assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

# AFTER:
assert d > 0 and d <= 4, "Currently only implemented for 1, 2, 3, and 4-D."
```

New parameter added:

```python
def __init__(self, d=1, measure=1.0, reduction="sum", eps=1e-8,
             periodic_in_x=True, periodic_in_y=True, periodic_in_z=True,
             periodic_in_t=True):   # NEW
    ...
    self.periodic_in_t = periodic_in_t
```

#### `compute_terms` 4D branch

The existing `elif self.d == 3` branch was unchanged. A new `elif self.d == 4` branch was appended:

```python
elif self.d == 4:
    dict_x[0] = torch.flatten(x, start_dim=-4)
    dict_y[0] = torch.flatten(y, start_dim=-4)

    fd4d = FiniteDiff(
        dim=4, h=quadrature,
        periodic_in_x=self.periodic_in_x, periodic_in_y=self.periodic_in_y,
        periodic_in_z=self.periodic_in_z, periodic_in_t=self.periodic_in_t,
    )
    x_x, x_y, x_z, x_t = fd4d.dx(x), fd4d.dy(x), fd4d.dz(x), fd4d.dt(x)
    y_x, y_y, y_z, y_t = fd4d.dx(y), fd4d.dy(y), fd4d.dz(y), fd4d.dt(y)

    dict_x[1] = torch.flatten(x_x, start_dim=-4)
    dict_x[2] = torch.flatten(x_y, start_dim=-4)
    dict_x[3] = torch.flatten(x_z, start_dim=-4)
    dict_x[4] = torch.flatten(x_t, start_dim=-4)

    dict_y[1] = torch.flatten(y_x, start_dim=-4)
    dict_y[2] = torch.flatten(y_y, start_dim=-4)
    dict_y[3] = torch.flatten(y_z, start_dim=-4)
    dict_y[4] = torch.flatten(y_t, start_dim=-4)
```

The returned dictionaries are consumed by the upstream `abs` / `rel` methods via:

$$\|x - y\|_{H^1}^2 = \prod_j h_j \left( \|x - y\|_2^2 + \sum_{k=1}^{4} \|\partial_{e_k} x - \partial_{e_k} y\|_2^2 \right)$$

with $e_k \in \{x, y, z, t\}$, computed by the existing reduction loop over `dict_x.keys()`. No change was needed in `abs`/`rel` themselves.

### 2.2 `HdivLoss`

Identical changes to `H1Loss` in the constructor. In `compute_terms` the 4D branch computes the scalar divergence as the sum of the four partial derivatives:

```python
elif self.d == 4:
    dict_x[0] = torch.flatten(x, start_dim=-4)
    dict_y[0] = torch.flatten(y, start_dim=-4)

    fd4d = FiniteDiff(dim=4, h=quadrature, ...)
    x_x, x_y, x_z, x_t = fd4d.dx(x), fd4d.dy(x), fd4d.dz(x), fd4d.dt(x)
    y_x, y_y, y_z, y_t = fd4d.dx(y), fd4d.dy(y), fd4d.dz(y), fd4d.dt(y)

    div_x = torch.flatten(x_x + x_y + x_z + x_t, start_dim=-4)
    div_y = torch.flatten(y_x + y_y + y_z + y_t, start_dim=-4)
```

The norm is:

$$\|u - v\|_{H(\text{div})}^2 = \prod_j h_j \left( \|u - v\|_2^2 + \|\nabla\cdot u - \nabla\cdot v\|_2^2 \right)$$

---

## File 3: `neuralop/losses/tests/test_losses.py`

52 new test functions were appended to the existing test file (original: 25 tests). They fall into four groups:

### Group A — 3D regression tests (confirm no breakage)

| Test | Description |
|---|---|
| `test_finite_diff_3d_regression[p_x-p_y-p_z]` | 8 parametrized variants (all periodic boundary combos); checks shape, laplacian self-consistency, and analytical derivatives of $\sin(x)\cos(y)\sin(z)$ with `atol=2e-3` |
| `test_fourier_diff_3d_regression` | Spectral accuracy for `FourierDiff(dim=3)` on the same function; checks `atol=1e-9` and laplacian $= -3f$ |
| `test_h1loss_3d_regression` | Self-comparison → 0; constant vs zero → positive |
| `test_hdivloss_3d_regression` | Same pattern |

### Group B — 4D `FiniteDiff` tests

| Test | Description |
|---|---|
| `test_finite_diff_4d_shapes[…]` | 16 parametrized variants (all 4-axis periodic combos); checks output shapes for `dx/dy/dz/dt`, `laplacian`, `gradient` (shape `(4,nx,ny,nz,nt)`), `divergence`, and that `curl` raises `NotImplementedError` |
| `test_finite_diff_4d_laplacian_consistency` | Verifies `laplacian == dx(u,2) + dy(u,2) + dz(u,2) + dt(u,2)` |
| `test_finite_diff_4d_numerical_accuracy` | `n=64` periodic grid; checks 4 partials of $\sin(x)\cos(y)\sin(z)\cos(t)$ vs analytical values with `atol=2e-3` |
| `test_finite_diff_4d_batched` | Batch=3, channels=2; verifies gradient shape `(3,2,4,8,8,8,8)` |
| `test_finite_diff_4d_guard` | `dim=5` raises `ValueError`; `dt` on `dim=3` raises `ValueError` |
| `test_central_diff_4d_compat` | `central_diff_4d` returns 4 tensors of correct shape |

### Group C — 4D `FourierDiff` tests

| Test | Description |
|---|---|
| `test_fourier_diff_4d_shapes` | Shape checks for all methods; `curl` raises `NotImplementedError` |
| `test_fourier_diff_4d_numerical_accuracy` | `n=32` periodic grid; spectral accuracy `atol=1e-9`; laplacian $= -4f$ |
| `test_fourier_diff_4d_compute_multiple` | 4-tuple order specs; results match individual `dx/dy/dz/dt` calls |
| `test_fourier_diff_4d_guard` | `dim=5` raises; wrong `L` length raises |

### Group D — 4D loss function tests

| Test | Description |
|---|---|
| `test_h1loss_4d_self_comparison` | `abs(x, x) == 0` |
| `test_h1loss_4d_constant_vs_zero` | Constant field vs zero → value `≈ 1.0` (only $L^2$ term, no gradient contribution) |
| `test_h1loss_4d_rel` | `rel(x, x) == 0`; `rel(x, y) > 0` for `x ≠ y` |
| `test_h1loss_4d_periodic_in_t[True/False]` | Loss runs without error for both periodic/non-periodic time BCs |
| `test_hdivloss_4d_self_comparison` | Same pattern |
| `test_hdivloss_4d_constant_vs_zero` | Same pattern |
| `test_hdivloss_4d_rel` | Same pattern |
| `test_h1loss_all_dims[1..4]` | Smoke test for all valid `d` values |
| `test_hdivloss_all_dims[1..4]` | Smoke test for all valid `d` values |

### Test results

All 77 collected tests pass:

```
======================== 77 passed, 1 warning in 4.26s =========================
```

The 1 warning is an expected `DeprecationWarning` from calling `central_diff_4d`.

---

## Design decisions

### Curl is not defined for 4D

The 3D curl $\nabla \times \mathbf{u}$ maps a 3-component vector field to a 3-component vector field. In 4D this generalises to an antisymmetric rank-2 tensor (the **Faraday tensor** in electrodynamics), which has 6 independent components and is not a standard vector operation. Implementing it correctly would require a different API (returning a `(6, nx, ny, nz, nt)` tensor) and is outside the scope of this extension. Both `FiniteDiff.curl` and `FourierDiff.curl` raise `NotImplementedError` with an explanatory message when called with `dim=4`.

### 4D Laplacian is the Euclidean sum, not the d'Alembertian

The implemented Laplacian is:

$$\nabla^2 u = \partial_x^2 u + \partial_y^2 u + \partial_z^2 u + \partial_t^2 u$$

This is the **Euclidean** 4D Laplacian (positive definite). The **d'Alembertian** (wave operator) would use $-\partial_t^2 u$ with a Minkowski metric. The Euclidean version is the correct choice for the $H^1$ Sobolev norm used in the loss functions.

### Fourier continuation not supported for 4D

The FC (Fourier Continuation) method extends a non-periodic function to a periodic domain. For 4D the computational cost of extending all four axes is prohibitive in the current implementation. A `NotImplementedError` is raised if `use_fc` is set with `dim=4`. Direct spectral differentiation (periodic fields) works without restriction.

### Periodic grid construction in tests

`torch.linspace(0, L, n)` includes the endpoint, giving actual spacing $L/(n-1)$ instead of the intended $L/n$. For periodic grids this leads to a duplicated point. All numerical accuracy tests use:

```python
coords = torch.linspace(0, L, n + 1, dtype=torch.float64)[:-1]
```

This generates exactly $n$ points on $[0, L)$ with uniform spacing $L/n$.

---

## Out-of-scope files

| File | Reason excluded |
|---|---|
| `neuralop/layers/differential_conv.py` | Uses `nn.Conv{n}d` which PyTorch only supports up to 3D; `nn.Conv4d` does not exist |
| `neuralop/models/fno.py` | Already generic: `n_modes` is an arbitrary-length tuple; no dim limit |
| `neuralop/layers/spectral_convolution.py` | Already generic: uses `einsum` with arbitrary number of frequency modes |
| `neuralop/losses/equation_losses.py` | Physics PDEs (Burgers, Navier-Stokes) are inherently 2D/3D — out of scope for generic extension |

---

## Usage examples

### `FiniteDiff` in 4D

```python
from neuralop.losses.differentiation import FiniteDiff

# Create a 4D tensor: shape (batch, channels, nx, ny, nz, nt)
u = torch.rand(2, 1, 16, 16, 16, 16)

fd4d = FiniteDiff(
    dim=4, h=(0.1, 0.1, 0.1, 0.1),
    periodic_in_x=True, periodic_in_y=True,
    periodic_in_z=True, periodic_in_t=False,
)

du_dx = fd4d.dx(u)           # shape (2, 1, 16, 16, 16, 16)
du_dt = fd4d.dt(u)           # shape (2, 1, 16, 16, 16, 16)
lap   = fd4d.laplacian(u)    # shape (2, 1, 16, 16, 16, 16)
grad  = fd4d.gradient(u)     # shape (2, 1, 4, 16, 16, 16, 16)
```

### `FourierDiff` in 4D

```python
from neuralop.losses.differentiation import FourierDiff
import torch

L = 2 * torch.pi
fd4d = FourierDiff(dim=4, L=(L, L, L, L))

u = torch.rand(16, 16, 16, 16, dtype=torch.float64)
du_dx = fd4d.dx(u)
du_dt = fd4d.dt(u)
lap   = fd4d.laplacian(u)

# Compute all four first derivatives in a single FFT call
derivs = fd4d.compute_multiple_derivatives(u, [(1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)])
```

### `H1Loss` and `HdivLoss` in 4D

```python
from neuralop.losses import H1Loss, HdivLoss

h1   = H1Loss(d=4, reduction="mean", periodic_in_t=True)
hdiv = HdivLoss(d=4, reduction="mean")

pred   = torch.rand(4, 8, 8, 8, 8)  # (batch, nx, ny, nz, nt)
target = torch.rand(4, 8, 8, 8, 8)

loss_abs = h1.abs(pred, target)
loss_rel = h1.rel(pred, target)
loss_div = hdiv.rel(pred, target)
```
