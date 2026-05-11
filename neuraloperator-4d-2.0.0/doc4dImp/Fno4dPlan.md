# Plan: Extend neuralop to 4D (x, y, z, time)

## TL;DR
Add `dim=4` support across `FiniteDiff`, `FourierDiff`, `H1Loss`, and `HdivLoss` in the neuralop
losses/differentiation subsystem. FNO model itself already supports arbitrary n_dim via `n_modes`.
The key changes are: relax dim validation guards, add `dt()` methods, add 4D private
implementations, and extend loss `compute_terms` branches.

---

## Phase 1 — `FiniteDiff` in `neuralop/losses/differentiation.py`

**Step 1a.** Relax dim guard and add `periodic_in_t`:
- `__init__`: change `if dim not in [1, 2, 3]` → `[1, 2, 3, 4]`
- Add `periodic_in_t=True` parameter
- Store `self.periodic_in_t` when `dim >= 4`
- Update h-validation and docstring

**Step 1b.** Update public dispatch methods for dim==4:
- `dx`: add `elif self.dim == 4: return self._dx_4d(u, order)`
- `dy`: update guard `if self.dim < 2` stays, add dispatch for dim==4
- `dz`: update guard `if self.dim < 3` stays, add dispatch for dim==4
- Add new public `dt(u, order=1)` method (raises ValueError if dim < 4)
- `laplacian`: add `elif self.dim == 4: return _dx_4d(u,2)+_dy_4d(u,2)+_dz_4d(u,2)+_dt_4d(u,2)`
- `gradient`: add 4D branch returning stack of 4 components `dim=-5`
- `divergence`: add 4D branch expecting 4-component vector field
- `curl`: add `elif self.dim == 4: raise NotImplementedError("curl not defined for 4D")`

**Step 1c.** Add 4D private dispatch and leaf implementations:
Tensor layout is `[..., x, y, z, t]` so axes are: x→dim=-4, y→dim=-3, z→dim=-2, t→dim=-1

```
_dx_4d(u, order) → dispatches to _dx_1st_4d/_dx_2nd_4d
_dy_4d(u, order) → dispatches to _dy_1st_4d/_dy_2nd_4d
_dz_4d(u, order) → dispatches to _dz_1st_4d/_dz_2nd_4d
_dt_4d(u, order) → dispatches to _dt_1st_4d/_dt_2nd_4d
```

Each leaf method mirrors the 3D pattern with one extra trailing dimension in slicers:
- `_dx_1st_4d`: rolls on `dims=-4`; non-periodic: interior `u[..., 1:-1, :, :, :]`, boundaries `u[..., 0, :, :, :]` / `u[..., -1, :, :, :]`
- `_dx_2nd_4d`: same axis, second-order stencil
- `_dy_1st_4d`: rolls on `dims=-3`; interior `u[..., :, 1:-1, :, :]`
- `_dy_2nd_4d`: same
- `_dz_1st_4d`: rolls on `dims=-2`; interior `u[..., :, :, 1:-1, :]`
- `_dz_2nd_4d`: same
- `_dt_1st_4d`: rolls on `dims=-1`; interior `u[..., :, :, :, 1:-1]`
- `_dt_2nd_4d`: same

**Step 1d.** Add `central_diff_4d` backward-compat function.

---

## Phase 2 — `FourierDiff` in `neuralop/losses/differentiation.py`

**Step 2a.** Relax dim guard in `__init__`:
- Change `if dim not in [1, 2, 3]` → `[1, 2, 3, 4]`
- `self.L` logic: extend to handle 4-tuples

**Step 2b.** Update dispatch in `compute_multiple_derivatives`, `derivative`, `dx`, `dy`, `dz`:
- Add `elif self.dim == 4` branches routing to new 4D methods

**Step 2c.** Add new `dt(u, order=1)` method.

**Step 2d.** Add `_compute_multiple_derivatives_4d(u, derivatives)`:
- Uses `torch.fft.fftn(u_clone, dim=(-4, -3, -2, -1))`
- Builds 4D frequency meshgrid: `k_x, k_y, k_z, k_t`
- Applies `(1j*KX)**ox * (1j*KY)**oy * (1j*KZ)**oz * (1j*KT)**ot` stencil
- Uses `torch.fft.ifftn(..., dim=(-4,-3,-2,-1)).real`
- FC support: calls `FC(u_clone, dim=4)` (FC layers already support arbitrary dim via their API)

**Step 2e.** Add `_dx_4d`, `_dy_4d`, `_dz_4d`, `_dt_4d` thin wrappers calling `_compute_multiple_derivatives_4d` with appropriate order tuples.

**Step 2f.** Update `laplacian`, `gradient`, `divergence`, `curl` in FourierDiff to match FiniteDiff 4D behaviour.

---

## Phase 3 — `H1Loss` in `neuralop/losses/data_losses.py`

**Step 3a.** `__init__`:
- Change `assert d > 0 and d < 4` → `assert d > 0 and d <= 4`
- Add `periodic_in_t=True` parameter
- Store `self.periodic_in_t`
- Update docstring

**Step 3b.** `compute_terms`:
- Add `elif self.d == 4` branch:
  ```python
  dict_x[0] = torch.flatten(x, start_dim=-4)
  dict_y[0] = torch.flatten(y, start_dim=-4)
  fd4d = FiniteDiff(dim=4, h=quadrature, periodic_in_x=..., periodic_in_y=...,
                    periodic_in_z=..., periodic_in_t=self.periodic_in_t)
  x_x, x_y, x_z, x_t = fd4d.dx(x), fd4d.dy(x), fd4d.dz(x), fd4d.dt(x)
  y_x, y_y, y_z, y_t = fd4d.dx(y), fd4d.dy(y), fd4d.dz(y), fd4d.dt(y)
  dict_x[1..4] = torch.flatten(x_x/y/z/t, start_dim=-4)
  ```

---

## Phase 4 — `HdivLoss` in `neuralop/losses/data_losses.py`

**Step 4a.** `__init__`:
- Change `assert d > 0 and d < 4` → `assert d > 0 and d <= 4`
- Add `periodic_in_t=True` parameter

**Step 4b.** `compute_terms`:
- Add `elif self.d == 4` branch using `FiniteDiff(dim=4, ...)`:
  ```python
  div_x = x_x + x_y + x_z + x_t  # sum of all 4 partial derivatives
  ```

---

## Relevant files

- `neuralop/losses/differentiation.py` — both `FiniteDiff` and `FourierDiff` classes
- `neuralop/losses/data_losses.py` — `H1Loss` and `HdivLoss` classes

## Files NOT needing changes

- `neuralop/layers/differential_conv.py` — uses `nn.Conv{n_dim}d` which PyTorch doesn't support for 4D → already works generically via n_dim, but Conv4d doesn't exist in PyTorch; no change needed
- `neuralop/models/fno.py` — already uses `n_modes` tuple with arbitrary length; no hardcoded dim limit
- `neuralop/layers/spectral_convolution.py` — already uses generic einsum; no hardcoded dim limit
- `neuralop/losses/equation_losses.py` — Burgers/NS physics losses are inherently problem-specific (2D/3D PDE); not in scope

## Verification

1. Run existing tests: `pytest neuralop/losses/tests/` — should still pass
2. Manual smoke test: instantiate `FiniteDiff(dim=4, h=(0.1,0.1,0.1,0.1))` and call `dx`, `dy`, `dz`, `dt`
3. Manual smoke test: `FourierDiff(dim=4)` calling `dx`, `dy`, `dz`, `dt`, `laplacian`, `gradient`
4. Manual smoke test: `H1Loss(d=4)` and `HdivLoss(d=4)` with a dummy 4D tensor

## Decisions / scope

- `curl` not defined for 4D (raises `NotImplementedError`) — curl is a 3D concept; the 4D extension would be a rank-2 antisymmetric tensor (not a simple vector operation)
- `laplacian` in 4D = sum of all 4 second-order partial derivatives (spacetime Laplacian / d'Alembertian without sign flip)
- `gradient` in 4D returns a 4-component vector field stacked at `dim=-5`
- `divergence` in 4D expects a 4-component vector field
- `FiniteDifferenceConvolution` (`differential_conv.py`) relies on PyTorch's `nn.Conv{n}d` which only supports up to 3D — excluded from scope; documented as known limitation
- Physics equation losses (`equation_losses.py`) are problem-specific — excluded from scope
