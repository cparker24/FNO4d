# CombinedTemporalLoss

A combined, temporally-weighted L² + spatial-gradient loss for the 3+1D FNO.
Available for both PyTorch (`loc_libs/LossFn.py`) and MLX
(`neuralop_mlx/training/losses.py`) with identical API and numerics.

---

## Motivation

The standard `H1Loss(d=4)` used in early training runs treats all four
dimensions — x, y, η, τ — symmetrically. This is convenient but physically
wrong in two ways.

**Temporal dimension is not a spatial dimension.**  
The network evolves a hydrodynamic field forward in proper time τ. The spatial
dimensions (x, y, η) carry physical gradients (pressure, flow) that are
meaningful to penalise. The τ axis is an evolution axis: its finite differences
measure how fast the field changes in time, not a spatial structure we want to
regularise. Mixing spatial and temporal gradients in a 4D H¹ norm conflates
these two distinct notions of "smoothness".

**Different τ-slices may deserve different emphasis.**  
Late-time slices near freeze-out carry the observables (v₂, v₃, particle
spectra). A flat average over all τ weights early, rapidly-evolving slices
equally with the freeze-out region, under-penalising the part of the evolution
that feeds the actual physics output. `CombinedTemporalLoss` computes the loss
*per τ-slice* and sums with configurable weights.

**Spatial-gradient regularisation improves azimuthal-anisotropy observables.**  
The `grad_focus` preset (α=0.5, β=2.0) up-weights the spatial-gradient
component. The gradient term penalises differences in ∂u/∂x, ∂u/∂y, ∂u/∂η
between prediction and target, which directly captures whether the model
reproduces the spatial shape of the energy-density profile — the source of
elliptic and triangular flow.

---

## Mathematical background

### Tensor layout

All computations assume tensors of shape `(B, C, X, Y, Z, T)`:

| axis | symbol | meaning |
|------|--------|---------|
| 0 | B | batch (events) |
| 1 | C | channels: ed, vx, vy, vz |
| 2–4 | X, Y, Z | transverse x / y, spatial rapidity η |
| 5 | T | proper time τ-slices |

### Combined loss

```
L = α · L_value  +  β · L_grad
```

where α (alpha) weights the field-value component and β (beta) weights the
spatial-gradient component.  Both are computed *per τ-slice* and
weighted-summed over T:

```
L_value = Σ_t  w_t · L_value(t)
L_grad  = Σ_t  w_t · L_grad(t)
```

### Field-value component L_value(t)

For a single τ-slice, let û_t = pred[..., t] and u_t = target[..., t]
(shape B × C × X × Y × Z):

| mode | formula |
|------|---------|
| **relative** | `Σ_{x,y,η} |û_t − u_t|²  /  (Σ_{x,y,η} |u_t|²  +  ε)` |
| **absolute** | `mean_{x,y,η} |û_t − u_t|²` |

Relative mode is scale-invariant: multiplying both prediction and target by a
constant k leaves the loss unchanged. This is the recommended mode after any
field normalisation (Bjorken, cosh-η, …).

### Spatial-gradient component L_grad(t)

Spatial gradients are computed with periodic (wrap-around) central finite
differences along the three spatial axes:

```
∂_x f ≈ roll(f, −1, axis=X) − roll(f, +1, axis=X)    (similarly for Y, Z)
|∇f|² = (∂_x f)² + (∂_y f)² + (∂_η f)²
```

| mode | formula |
|------|---------|
| **relative** | `Σ |∇(û_t − u_t)|²  /  (Σ |∇u_t|²  +  ε)` |
| **absolute** | `mean |∇(û_t − u_t)|²` |

**Grid spacing cancels in relative mode.** The h factors from the finite
differences appear in both numerator and denominator and divide out exactly. No
grid metadata is required.  
In absolute mode the mean-over-cells normalisation makes h irrelevant to the
loss scale as well, so again no grid metadata is needed in either mode.

### Temporal weights w_t

By default, uniform weights `w_t = 1/T` are used — appropriate when a
Bjorken-motivated normalisation has already removed the τ-dependent amplitude
variation. Non-uniform weights can be supplied to emphasise specific τ-slices
(e.g. freeze-out):

```python
# Ramp up weight toward later slices (softmax of a linear ramp)
import torch
w = torch.softmax(torch.linspace(0, 3, ntau), dim=0)   # PyTorch
```
```python
import mlx.core as mx
w = mx.array(...)                                        # MLX
```

Weights must be L1-normalised (sum to 1) when passed in; otherwise the total
loss scale changes with T.

### Reduction over batch and channels

After computing the per-τ-slice scalar for each batch item and channel, the
default aggregation is:
1. mean over channels C → shape (B,)
2. mean over batch B → scalar

Pass `reduction='sum'` to sum over the batch instead.

---

## Presets

| preset | α | β | recommended for |
|--------|---|---|-----------------|
| `l2` | 1.0 | 0.0 | pure L² baseline; fast training |
| `h1` | 1.0 | 1.0 | balanced default; drop-in for `H1Loss` |
| `grad_focus` | 0.5 | 2.0 | improving v₂/v₃; up-weights spatial structure |
| `smooth` | 1.0 | 0.1 | mild gradient regularisation; large-batch runs |
| `boundary` | 0.0 | 1.0 | pure gradient; MG-TFNO patch-stitching |
| `custom` | — | — | set α and β directly |

---

## API reference

### Constructor

```python
CombinedTemporalLoss(
    alpha       = 1.0,          # weight on L_value component
    beta        = 1.0,          # weight on L_grad component
    preset      = 'h1',         # named (α, β) pair; overrides alpha/beta
    mode        = 'relative',   # 'relative' or 'absolute' for both components
    l2_mode     = None,         # per-component override for L_value
    grad_mode   = None,         # per-component override for L_grad
    tau_weights = None,         # L1-normalised array of shape (T,); uniform if None
    eps         = 1e-8,         # denominator stabiliser (relative mode only)
    reduction   = 'mean',       # 'mean' or 'sum' over the batch
)
```

`mode` sets both `l2_mode` and `grad_mode`.  Passing `l2_mode` or `grad_mode`
explicitly overrides the global `mode` for that component only (see mixed-mode
example below).

### Methods

| method | signature | description |
|--------|-----------|-------------|
| `__call__` | `(pred, target) → scalar` | scalar total loss; drop-in for any training loop |
| `forward_dict` | `(pred, target) → dict` | returns `{'total', 'l2', 'grad', 'h1'}` for component introspection |
| `as_eval_dict` | `() → dict` | returns `{'l2': fn, 'grad': fn, 'h1': fn}` callables for the Trainer's `eval_losses` |

`forward_dict` is useful for debugging and diagnostics — pass its output to a
`TrainingHistory` or print directly to confirm that neither component is
dominating or collapsing.

---

## PyTorch integration

### Minimal drop-in

Replace `H1Loss(d=4)` as the training loss in any PyTorch training script:

```python
import sys
sys.path.insert(0, '../loc_libs')
from LossFn import CombinedTemporalLoss

# H1-equivalent, but computed per τ-slice with spatial-only gradient
loss_fn = CombinedTemporalLoss(preset='h1', mode='relative')

# Use as training loss
train_loss = loss_fn

# Use as eval losses — logs 'l2', 'grad', 'h1' per epoch alongside the trainer
eval_losses = loss_fn.as_eval_dict()
```

### In `test_train_yaml_norm.py`

Find the optimizer / losses / trainer block and replace:

```python
# Before
l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)
train_loss  = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}
```

```python
# After
from LossFn import CombinedTemporalLoss

loss_fn     = CombinedTemporalLoss(preset='h1', mode='relative')
train_loss  = loss_fn
eval_losses = loss_fn.as_eval_dict()   # logs 'l2', 'grad', 'h1'
```

The `Trainer` only requires that `train_loss(pred, target)` returns a scalar
and that each entry of `eval_losses` is callable with `(pred, target)`.
`CombinedTemporalLoss` satisfies both.

### Grad-focus preset with freeze-out weighting

```python
import torch
from LossFn import CombinedTemporalLoss

ntau = 15   # number of τ-slices in your dataset after slicing

# Softmax ramp: later slices get more weight
w = torch.softmax(torch.linspace(0, 3, ntau), dim=0)   # shape (ntau,)

loss_fn = CombinedTemporalLoss(
    preset      = 'grad_focus',   # α=0.5, β=2.0
    mode        = 'relative',
    tau_weights = w,
)
```

### Mixed mode (relative L², absolute gradient)

Use when the field has been normalised (relative L² is appropriate) but you
also want the gradient loss to track raw error magnitude rather than a ratio:

```python
loss_fn = CombinedTemporalLoss(
    preset    = 'custom',
    alpha     = 1.0,
    beta      = 0.5,
    l2_mode   = 'relative',
    grad_mode = 'absolute',
)
```

### Component diagnostics during training

```python
loss_fn = CombinedTemporalLoss(preset='h1', mode='relative')

# Inside the training loop, once per epoch or on validation batches:
with torch.no_grad():
    d = loss_fn.forward_dict(pred, target)
print(f"total={d['total']:.4f}  l2={d['l2']:.4f}  grad={d['grad']:.4f}")
```

---

## MLX integration

The MLX class has an identical API. The only difference is the import path and
the tensor type (`mx.array` instead of `torch.Tensor`).

### Minimal drop-in

```python
from neuralop_mlx.training.losses import CombinedTemporalLoss
# or
from neuralop_mlx import CombinedTemporalLoss

loss_fn     = CombinedTemporalLoss(preset='h1', mode='relative')
train_loss  = loss_fn
eval_losses = loss_fn.as_eval_dict()
```

### In `test_train_mlx_yaml_norm.py`

Find the optimizer / losses / trainer block and replace:

```python
# Before
l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)
train_loss  = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}
```

```python
# After
from neuralop_mlx import CombinedTemporalLoss

loss_fn     = CombinedTemporalLoss(preset='h1', mode='relative')
train_loss  = loss_fn
eval_losses = loss_fn.as_eval_dict()   # logs 'l2', 'grad', 'h1'
```

The MLX `Trainer` calls `training_loss(pred, target)` and each entry of
`eval_losses(pred, target)` internally, exactly as the PyTorch Trainer does.

### Grad-focus preset with freeze-out weighting

```python
import mlx.core as mx
from neuralop_mlx import CombinedTemporalLoss

ntau = 15

# Softmax ramp over τ-slices
ramp = mx.array([float(i) / (ntau - 1) * 3 for i in range(ntau)])
w    = mx.softmax(ramp, axis=0)   # shape (ntau,)

loss_fn = CombinedTemporalLoss(
    preset      = 'grad_focus',
    mode        = 'relative',
    tau_weights = w,
)
```

### Component diagnostics during training

```python
import mlx.core as mx

loss_fn = CombinedTemporalLoss(preset='h1', mode='relative')

# After a forward pass, before or after mx.eval():
d = loss_fn.forward_dict(pred, target)
mx.eval(d['total'], d['l2'], d['grad'])
print(f"total={d['total'].item():.4f}  "
      f"l2={d['l2'].item():.4f}  "
      f"grad={d['grad'].item():.4f}")
```

---

## Choosing a preset

| situation | recommended preset | mode |
|-----------|-------------------|------|
| First run; drop-in for H1Loss | `h1` | `relative` |
| After Bjorken / cosh-η normalisation | `h1` | `relative` |
| Improving v₂/v₃ agreement | `grad_focus` | `relative` |
| Training on raw (un-normalised) fields | `h1` or `l2` | `absolute` |
| Debugging — isolate value errors | `l2` | `relative` |
| Debugging — isolate spatial-structure errors | `boundary` | `relative` |
| Mild regularisation for large batches | `smooth` | `relative` |

Start with `h1 / relative` and switch to `grad_focus` if the spatial shape of
the energy-density profile (and hence v₂) converges slowly.

---

## Files

| path | description |
|------|-------------|
| `loc_libs/LossFn.py` | PyTorch implementation |
| `neuralop_mlx/training/losses.py` | MLX implementation (appended after `H1Loss`) |
| `tests/combined_loss_test.py` | PyTorch correctness tests (10 analytical invariants) |
| `tests/combined_loss_test_mlx.py` | MLX correctness tests (same 10 invariants) |
