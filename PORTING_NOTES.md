# MLX Porting Notes — neuraloperator to neuralop_mlx

Technical reference for continuing the port of models from
`neuraloperator-4d-2.0.0` (PyTorch) to `neuralop_mlx` (MLX / Apple Silicon).

Audience: developer picking up this work cold. Contains every non-obvious
decision made during the initial FNO port, a map of what remains, and the
concrete patterns to follow.

---

## 1. Current state

### Ported and tested (48 tests, all passing)

| Component | MLX file | PyTorch source |
|-----------|----------|---------------|
| `SpectralConvND` | `layers/spectral_convolution.py` | `layers/spectral_convolution.py : SpectralConv` |
| `FNOBlocks` | `layers/fno_block.py` | `layers/fno_block.py : FNOBlocks` |
| `ChannelMLP` | `layers/channel_mlp.py` | `layers/channel_mlp.py : ChannelMLP` |
| `GridEmbeddingND` | `layers/embeddings.py` | `layers/embeddings.py : GridEmbeddingND` |
| `LinearSkip / SoftGatingSkip / IdentitySkip` | `layers/skip_connections.py` | `layers/skip_connections.py` |
| `FNO` | `models/fno.py` | `models/fno.py : FNO` |
| `LpLoss`, `H1Loss` | `training/losses.py` | `losses/data_losses.py` |
| `Trainer` | `training/trainer.py` | `training/trainer.py` |
| `AdamW` | `training/adamw.py` | `training/adamw.py` |
| `load_from_torch` | `weight_transfer.py` | — (new) |

### Not ported

| Model / feature | PyTorch file | Blocker / notes |
|----------------|-------------|-----------------|
| `TFNO` (Tucker/CP/TT weights) | `models/fno.py : TFNO` | Requires `tltorch` FactorizedTensor — no MLX equivalent |
| `UNO` (U-shaped FNO) | `models/uno.py` | Needs `resample` (interpolation); see §5 |
| `RNO` (recurrent NO) | `models/rno.py` | Needs `RNOBlock`; MLX has `nn.LSTM` / `nn.GRU` |
| `SFNO` (spherical FNO) | `models/sfno.py` | Needs `torch_harmonics` — no MLX port exists |
| `LocalNO` | `models/local_no.py` | Diff kernels portable (2D/3D only); DISCO blocked by `torch_harmonics`; see §5 |
| `FNOGNO`, `GINO`, `CODANO` | `models/` | Require `GNOBlock` → `NeighborSearch` → no scatter ops in MLX |
| Normalization layers | `layers/normalization_layers.py` | `AdaIN`, `InstanceNorm`, `BatchNorm` — MLX has GroupNorm, LayerNorm, BatchNorm |
| `preactivation` FNOBlocks | `layers/fno_block.py` | FNO block variant; straightforward once norms are ported |
| `output_scaling_factor` | `layers/fno_block.py` | Needs `resample`; see §5 |
| `domain_padding` | `layers/padding.py` | Padding around domain before FFT — no blocker, just not done |
| `SphericalConv` | `layers/spherical_convolution.py` | Needs `torch_harmonics` |
| `GNOBlock`, `IntegralTransform` | `layers/gno_block.py` | Needs graph/scatter ops |
| `H1Loss` for d>4 | `training/losses.py` | Current MLX H1Loss is capped at d=4 |
| `wandb` logging | `training/trainer.py` | Not implemented |
| Distributed training | `training/trainer.py` | Not implemented |
| LR scheduler stepping | `training/trainer.py` | Scheduler accepted but not called per epoch |
| Data normalizers | `data/transforms/normalizers.py` | `UnitGaussianNormalizer` etc — straightforward port |

---

## 2. MLX vs PyTorch — complete API translation table

### Tensors and dtype

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.tensor(arr)` | `mx.array(arr)` | |
| `torch.FloatTensor` | `mx.array(..., dtype=mx.float32)` | |
| `torch.complex64` | `mx.complex64` | MLX supports complex64 |
| `torch.float16` (complex) | **not available** | MLX float16 has no complex support |
| `t.detach().cpu().numpy()` | `np.array(t)` then `mx.eval(t)` | MLX is lazy; eval first |
| `t.shape` | `t.shape` | Same |
| `t.numel()` | `t.size` | Property, not method |
| `t.view(...)` | `t.reshape(...)` | |
| `t.permute(...)` | `t.transpose(...)` | Same semantics |
| `t.contiguous()` | not needed | MLX has no strides concept |
| `t.to(device)` | not needed | MLX always uses Metal/CPU transparently |

### Neural network modules

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `nn.Module` | `nn.Module` | Same concept; different gradient API |
| `nn.Linear(in, out)` | `nn.Linear(in, out)` | Weight shape: both `(out, in)` |
| `nn.Conv1d(in, out, kernel_size=1)` | `nn.Linear(in, out)` | Conv1d(k=1) ≡ Linear; squeeze last weight dim |
| `nn.Conv2d` | `nn.Conv2d` | Available |
| `nn.LSTM` | `nn.LSTM` | Available |
| `nn.GRU` | `nn.GRU` | Available |
| `nn.GroupNorm` | `nn.GroupNorm` | Available |
| `nn.LayerNorm` | `nn.LayerNorm` | Available |
| `nn.BatchNorm1d/2d/3d` | `nn.BatchNorm` | MLX has single BatchNorm |
| `nn.MultiheadAttention` | `nn.MultiHeadAttention` | Available |
| `nn.ModuleList([...])` | plain Python `list` | MLX auto-traverses lists of modules |
| `nn.ModuleDict({...})` | plain Python `dict` | MLX auto-traverses dicts of modules |
| `nn.Parameter(t)` | just assign `self.x = mx.array(...)` | Any array attribute is a parameter |
| `F.gelu(x)` | `nn.gelu(x)` | Function, not method |
| `F.silu(x)` | `nn.silu(x)` | Available |
| `F.relu(x)` | `nn.relu(x)` | Available |
| `F.interpolate(x, size, mode)` | **not available** | See §5 for workaround |

### FFT operations

| PyTorch | MLX | Notes |
|---------|-----|-------|
| `torch.fft.rfftn(x, dim=dims)` | `mx.fft.rfftn(x, axes=dims)` | `dim` → `axes` |
| `torch.fft.irfftn(x, s=sz, dim=dims)` | `mx.fft.irfftn(x, s=sz, axes=dims)` | |
| `torch.fft.fftn(x, dim=dims)` | `mx.fft.fftn(x, axes=dims)` | |
| `torch.fft.ifftn(x, s=sz, dim=dims)` | `mx.fft.ifftn(x, s=sz, axes=dims)` | |
| `torch.fft.fftshift(x, dim=dims)` | `mx.fft.fftshift(x, axes=dims)` | |
| `torch.fft.ifftshift(x, dim=dims)` | `mx.fft.ifftshift(x, axes=dims)` | |
| `torch.fft.fftfreq(n)` | `mx.fft.fftfreq(n)` | Available |
| norm arg default (backward) | `norm="forward"` used in port | Match PyTorch default explicitly |

### Gradient computation

| PyTorch | MLX |
|---------|-----|
| `loss.backward()` | not available |
| `optimizer.step()` | not available |
| `optimizer.zero_grad()` | not needed |
| model forward + loss | `loss, grads = nn.value_and_grad(model, fwd)(model)` |
| `optimizer.step()` | `optimizer.update(model, grads)` |
| flush to GPU | `mx.eval(model.parameters(), optimizer.state)` |

Pattern used in Trainer:
```python
def fwd(model):
    return loss_fn(model(x), y)

loss, grads = nn.value_and_grad(model, fwd)(model)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

### Optimizer

| PyTorch | MLX |
|---------|-----|
| `torch.optim.AdamW(model.parameters(), lr, weight_decay)` | `mlx.optimizers.AdamW(lr, weight_decay=wd)` |
| `torch.optim.Adam(...)` | `mlx.optimizers.Adam(...)` |
| `torch.optim.SGD(...)` | `mlx.optimizers.SGD(...)` |
| cosine LR decay | `mlx.optimizers.cosine_decay(lr, T_max)` — passed as `learning_rate` to optimizer |
| params passed at construction | **not done** — MLX optimizers receive params at `update(model, grads)` time |

### Checkpointing

| PyTorch | MLX |
|---------|-----|
| `torch.save(model.state_dict(), path)` | `mx.save_safetensors(path, flat_dict)` |
| `model.load_state_dict(torch.load(path))` | `model.update(mlx.utils.tree_unflatten(mx.load(path)))` |
| Load PyTorch `.pt/.pth` into MLX | `load_from_torch(path, mlx_model)` |

---

## 3. MLX-specific constraints and bugs

### 3a. Lazy evaluation — CRITICAL

MLX builds a compute graph and evaluates lazily. **Always call `mx.eval()`**
before reading a value (`.item()`, `np.array()`, shape-dependent branching).
Missing `mx.eval()` silently produces wrong results in control flow that
branches on array values.

```python
out = model(x)
mx.eval(out)           # materialise before np.array() or .item()
out_np = np.array(out)
```

### 3b. `array.at[multi-dim-slice].add` stride bug (MLX 0.31.2)

**Symptom:** `out.at[:, :, 6:10, :3].add(val)` silently writes values at the
wrong positions when two or more non-trivial (non-full-range) slice dimensions
are present. Batch/channel position [0,0] is correct; [0,1], [1,0], etc. are
shifted by the size of the first sliced dimension.

**Fix:** Replace scatter with `mx.pad`. This is mathematically equivalent
when scattering into a zero buffer:

```python
# Wrong (stride bug):
out = mx.zeros(out_shape, dtype=mx.complex64)
out = out.at[:, :, s0, s1].add(contracted)

# Correct:
out = mx.pad(contracted, pad_widths)  # pad_widths computed to center contracted in out_shape
```

**Scalar-index `.at[..., 0].add(...)` is unaffected** — the bug only triggers
when multiple multi-element slices appear simultaneously.

Check whether this is fixed before removing the workaround:
```python
import mlx.core as mx
a = mx.zeros((2, 2, 4, 4))
v = mx.ones((2, 2, 2, 2))
r = a.at[:, :, 1:3, 1:3].add(v)
mx.eval(r)
# If r[0,1,1,1] == 1.0 the bug is fixed; if 0.0 it's still present
print(r[0, 1, 1, 1].item())
```

### 3c. Complex weights and AdamW — CRITICAL

MLX `AdamW` computes the variance estimate as `v = beta2 * v + (1-beta2) * g * g`.
For complex arrays `g * g` is complex multiplication (not `|g|²`), producing
NaN gradients immediately.

**Fix:** Store complex weights as two `float32` arrays (`weight_r`, `weight_i`).
Reconstruct complex for FFT contraction at runtime:

```python
# Storage
self.weight_r = mx.random.normal(shape) * init_std  # float32
self.weight_i = mx.random.normal(shape) * init_std  # float32

# Contraction (see _complex_einsum in spectral_convolution.py)
real = einsum(eq, x_r, wr) - einsum(eq, x_i, wi)
imag = einsum(eq, x_r, wi) + einsum(eq, x_i, wr)
out  = real.astype(mx.complex64) + (imag * 1j).astype(mx.complex64)
```

This pattern must be applied to **every complex-valued parameter** in any
future spectral layer.

### 3d. `mx.interpolate` does not exist

MLX 0.31.2 has no `F.interpolate` equivalent. This blocks:
- `output_scaling_factor` in FNOBlocks / UNO
- `resample()` used by UNO between layers of different resolution

Workaround options (in order of preference):
1. Spectral truncation / zero-padding in Fourier space (lossless for bandlimited signals)
2. `mx.fft.irfftn` at a different `s=` to change output resolution (already done in SpectralConvND)
3. Manual bilinear interpolation via `mx.einsum` (complex but feasible for 2D)
4. Wait for MLX to add `mx.interpolate`

### 3e. `nn.ModuleList` / `nn.ModuleDict` — not needed

MLX automatically traverses plain Python `list` and `dict` attributes when
collecting parameters. No special container class is needed. This simplifies
code significantly:

```python
self.convs = [SpectralConvND(...) for _ in range(n_layers)]  # plain list, works
```

### 3f. `state_dict()` includes `_metadata`

`model.state_dict()` in PyTorch returns a dict that includes a `_metadata` key
whose value is a plain `dict`, not a `Tensor`. Filter before processing:

```python
sd = {k: v for k, v in pt_model.state_dict().items() if isinstance(v, torch.Tensor)}
```

---

## 4. FNOBlocks activation order — critical correctness detail

The PyTorch `forward_with_postactivation` (the default) applies non-linearity:
1. After spectral conv + FNO skip: **only for non-final layers**
2. After channel MLP + MLP skip: **only for non-final layers**

Both skip connections (`fno_skip` and `channel_mlp_skip`) receive the block's
**original input** `x` — not any intermediate activated value.

The correct MLX implementation:

```python
def __call__(self, x, index):
    x_in = x                                     # save original input

    x_fno = self.convs[index](x_in)
    x_fno = x_fno + self.fno_skips[index](x_in) # fno_skip from ORIGINAL x

    if index < self.n_layers - 1:
        x = self.non_linearity(x_fno)            # gelu only for non-last
    else:
        x = x_fno

    x_mlp = self.channel_mlps[index](x)
    x_mlp = x_mlp + self.mlp_skips[index](x_in) # mlp_skip from ORIGINAL x

    if index < self.n_layers - 1:
        x = self.non_linearity(x_mlp)            # gelu only for non-last
    else:
        x = x_mlp

    return x
```

Deviating from this produces outputs that diverge from PyTorch by ~0.1–0.4
even with identical weights.

---

## 5. What each un-ported model needs

### TFNO (Tucker/CP/TT factorized FNO)

`TFNO` is `FNO` with `factorization != None`. The weight is a `tltorch`
`FactorizedTensor` instead of a dense array. There is no MLX equivalent of
`tltorch`.

To port: implement Tucker/CP/TT decomposed weight storage manually as lists of
`float32` factor matrices and the corresponding reconstruction einsum. The
contraction equations are in `layers/spectral_convolution.py:_contract_tucker`,
`_contract_cp`, `_contract_tt`.

Complexity: high (3–5 days). Most useful if memory is a hard constraint.

### UNO (U-shaped FNO)

UNO is FNO with:
- Per-layer `n_modes`, `out_channels`, `scalings` (vary by layer)
- Horizontal skip connections between encoder and decoder layers
- `resample()` to change spatial resolution between layers

The `resample()` is the main blocker: it calls `F.interpolate` which MLX lacks.
Spectral resampling (truncate/pad in Fourier space) is the natural alternative
and is already present in `SpectralConvND` via the `s=` argument to `irfftn`.

Everything else (variable channels per layer, horizontal skips) uses layers
that are already ported.

Complexity: medium (1–2 days) if using spectral resampling instead of bilinear.

### RNO (Recurrent Neural Operator)

RNO wraps FNO with `nn.GRU`-style gating. MLX has `nn.LSTM` and `nn.GRU`.
The `RNOBlock` in `layers/rno_block.py` applies a modified GRU where the
"hidden" dimension is spatial (not a flat vector). This is implemented as
element-wise operations compatible with arbitrary spatial dimensions.

All required ops (`nn.Linear`, element-wise multiply, sigmoid, tanh) are
available in MLX.

Complexity: low–medium (1 day).

### SFNO (Spherical FNO)

Uses `torch_harmonics` (Real Spherical Harmonic Transform). No MLX port of
`torch_harmonics` exists. This is a hard external dependency blocker.

Complexity: very high (requires porting `torch_harmonics` or finding an
equivalent). Defer unless spherical geometry is specifically needed.

### LocalNO (Local Neural Operator)

`LocalNO` is defined in `models/local_no.py`. Its architecture is identical to
`FNO` except the `FNOBlocks` are replaced by `LocalNOBlocks`, which add two
optional local-interaction kernels in parallel with the spectral convolution:

1. **`FiniteDifferenceConvolution`** (`layers/differential_conv.py`): a standard
   `ConvNd` that approximates differential operators (gradient, Laplacian, etc.)
   using learned finite-difference stencils. Controlled by `diff_layers=True`.

2. **`EquidistantDiscreteContinuousConv2d`** (DISCO, `layers/discrete_continuous_convolution.py`):
   a local integral operator using spherical-harmonic basis functions.
   Controlled by `disco_layers=True`. 2D only in the PyTorch implementation.

#### FiniteDifferenceConvolution portability

The `FiniteDifferenceConvolution` class uses `getattr(nn, f"Conv{n_dim}d")` to
select `Conv1d`, `Conv2d`, or `Conv3d`. All three are available in MLX ✓.

The main blocker is **padding modes**. PyTorch supports `'circular'`,
`'reflect'`, and `'replicate'` via the `padding_mode` argument. MLX `mx.pad`
supports only `'constant'` and `'edge'`. Manual workarounds:

| PyTorch padding mode | MLX workaround |
|----------------------|----------------|
| `'circular'`  | Pre-pad with `mx.concatenate` using slices from the opposite edge (equivalent to `mx.roll`-based wrap) |
| `'reflect'`   | Pre-pad by concatenating a flipped slice; `mx.flip` is available |
| `'replicate'` | Use `mx.pad(..., mode='edge')` — exact equivalent ✓ |

Circular padding pre-pad pattern:
```python
# Wrap last spatial dim by k on each side before calling nn.Conv2d
pad = k // 2
x = mx.concatenate([x[..., -pad:], x, x[..., :pad]], axis=-1)
```

The `diff_layers` feature is only implemented for `n_dim <= 3` in PyTorch
itself — `local_no_block.py` raises `NotImplementedError` for `n_dim > 3`.
So 4D diff layers are not a porting target.

#### DISCO portability

`EquidistantDiscreteContinuousConv2d` imports `torch_harmonics` for
precomputing the spherical-harmonic filter basis (`_precompute_grid`,
`filter_basis`, `disco_kernel`). This is a hard external dependency with no
MLX equivalent and no pure-Python fallback. **DISCO cannot be ported without
either porting `torch_harmonics` itself or replacing the basis precomputation
with an independent spherical-harmonic library.**

Even if the dependency were resolved, DISCO is only implemented for 2D in the
PyTorch source. It is not a 4D operator.

#### What works today without changes

`LocalNO(diff_layers=False, disco_layers=False)` is structurally identical to
`FNO`. The `LocalNOBlocks` path with no optional kernels reduces to the same
spectral conv + skips already implemented in `FNOBlocks`. Training with this
configuration works immediately using the existing `neuralop_mlx` package.

#### Porting complexity summary

| Configuration | Portability | Effort |
|--------------|-------------|--------|
| `diff_layers=False, disco_layers=False` | Already works (= FNO) | 0 |
| `diff_layers=True, disco_layers=False`, 2D or 3D | Portable; needs circular/reflect padding workarounds | Medium (~1–2 days) |
| `diff_layers=True`, 4D | Blocked in PyTorch itself (`NotImplementedError`) | N/A |
| `disco_layers=True` | Blocked by `torch_harmonics` dependency | Very high |

To port the `diff_layers` path: implement `FiniteDifferenceConvolution` in MLX
using `nn.Conv2d`/`Conv3d` with pre-padding shims for circular/reflect modes,
then wire it into a new `LocalNOBlocks` class that sums the diff kernel output
with the spectral conv output before applying the activation.

### FNOGNO / GINO / CODANO

These use `GNOBlock` → `IntegralTransform` → `NeighborSearch`, which relies
on radius-graph construction and grouped scatter-reduce operations
(`torch_scatter`). MLX has no scatter-reduce primitive.

Complexity: very high. Would require implementing a custom Metal kernel or
waiting for MLX to expose scatter ops.

### Normalization layers (AdaIN, InstanceNorm, BatchNorm)

- `BatchNorm`: MLX has `nn.BatchNorm` — direct port, trivial.
- `InstanceNorm`: equivalent to `nn.GroupNorm(num_groups=C, ...)` — available.
- `AdaIN` (Adaptive Instance Norm): uses a conditioning embedding to produce
  per-channel scale/shift. All needed ops are available. Needed before
  `preactivation` FNOBlocks or conditional models can be used.

Complexity: low (half a day for all three).

### `preactivation` FNOBlocks

Variant of `forward_with_postactivation` where norm + activation happens
*before* the convolution in each block. Only useful when normalization layers
are ported (AdaIN etc.). The logic is in `fno_block.py:forward_with_preactivation`.

Complexity: trivial once normalization layers exist.

### `domain_padding`

Pads the spatial domain before the FFT to avoid aliasing at boundaries, then
crops after irfftn. All required ops (`mx.pad`, slice) are available.
Source: `layers/padding.py`.

Complexity: low (half a day).

---

## 6. Weight transfer patterns for future models

`weight_transfer.py` handles FNO. The patterns to apply for other models:

**Conv1d (kernel=1) → Linear:**
```python
w = state_dict["some.conv.weight"].numpy()  # shape (out, in, 1)
mlx_layer.weight = mx.array(w[:, :, 0])    # squeeze last dim
```

**Complex dense weight (DenseTensor) → weight_r + weight_i:**
```python
w = state_dict["layer.weight.tensor"].numpy().astype(np.complex64)
mlx_layer.weight_r = mx.array(w.real.astype(np.float32))
mlx_layer.weight_i = mx.array(w.imag.astype(np.float32))
```

**Tucker factorized weight → not supported yet:**
```
state_dict keys: layer.weight.factors.0, layer.weight.factors.1, ..., layer.weight.core
```
These would need to be reconstructed into a dense tensor first:
```python
import tensorly as tl
factors = [state_dict[f"layer.weight.factors.{i}"].numpy() for i in range(n)]
core    = state_dict["layer.weight.core"].numpy()
dense   = tl.tucker_to_tensor((core, factors)).astype(np.complex64)
mlx_layer.weight_r = mx.array(dense.real.astype(np.float32))
mlx_layer.weight_i = mx.array(dense.imag.astype(np.float32))
```

**SoftGating per-channel weight:**
```python
w = state_dict["layer.weight"].numpy()  # shape (1, C, 1, ..., 1)
mlx_layer.alpha = mx.array(w.reshape(-1))  # shape (C,)
```

**Bias shape difference (SpectralConv):**
```python
# PT: (C, 1, 1, ...) → MLX: (1, C, 1, 1, ...)
mlx_layer.bias = mx.array(pt_bias.reshape(mlx_layer.bias.shape))
```

---

## 7. Testing strategy for new ports

Follow the pattern in `tests/test_weight_transfer.py`:

1. **Shape test**: `out.shape == x.shape` with random input — catches index errors.
2. **NaN test**: `not mx.any(mx.isnan(out)).item()` — catches exploding values.
3. **Gradient flow test**: `nn.value_and_grad(model, fwd)(model)` then check no NaN grads.
4. **Numerical parity test**: copy weights from PyTorch, compare forward outputs with `atol=1e-3`.
5. **Weight value test**: check individual layer weights match the source PT tensors.

The 1e-3 tolerance is appropriate for float32 FFT chains. Purely real-valued
layers (Linear, MLP) should match to 1e-6.

For any model using complex weights, always test gradient flow — NaN from the
AdamW complex bug (§3c) will appear in the first gradient step, not the forward.

---

## 8. Package layout conventions

Adding a new model `XNO`:

1. Create `neuralop_mlx/models/xno.py` with class `XNO(nn.Module)`.
2. Export from `neuralop_mlx/models/__init__.py`.
3. Add weight transfer logic to `weight_transfer.py` (new function
   `_transfer_xno_weights`, call from `load_from_torch` when model type is
   detected or via a separate `load_xno_from_torch` function).
4. Export `load_xno_from_torch` from `neuralop_mlx/__init__.py`.
5. Add tests:
   - `tests/test_xno_model_mlx.py` — shape, NaN, gradient tests
   - Entry in `tests/test_weight_transfer.py` or new `tests/test_xno_transfer.py`
6. Update `README_mlx.md` — tested configurations table, file layout.

The `pyproject.toml` at the repo root uses `packages.find: include = ["neuralop_mlx*"]`
so any sub-package automatically gets picked up without editing the build config.

---

## 9. Environment

```
Python      : 3.13.1
MLX         : 0.31.2  (installed in fno_env_mlx conda env)
PyTorch     : 2.11.0.dev (for comparison tests only)
conda env   : fno_env_mlx
install cmd : pip install -e .   # from FNO2Mlx/ repo root
test cmd    : python -m pytest tests/ -v
```

PyTorch source: `FNO2Mlx/neuraloperator-4d-2.0.0/`
MLX package:    `FNO2Mlx/neuralop_mlx/`
Training script: `FNO2Mlx/train/test_train_mlx.py`

---

## 10. Open MLX limitations to watch

- `mx.interpolate`: missing — blocks UNO and `output_scaling_factor`
- `mx.scatter_reduce` / `mx.segment_reduce`: missing — blocks GNO-based models
- `mx.float16` complex: not supported — float32 only for spectral layers
- `array.at[multi-slice].add` stride bug: present in 0.31.2, check each new release
- No `torch_harmonics` equivalent: blocks SFNO
- `mx.eval()` must be called explicitly: easy to forget, silently wrong in control flow
