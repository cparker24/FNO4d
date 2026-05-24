# CausalLogLoLoss — Combined Training Objective

`combined_loss.py`

-----

## Overview

`CausalLogLoLoss` is the training objective for `CausalLogLo` and compatible
3+1D neural operators on fields of shape `(B, C, X, Y, Z, T)`. It combines
three loss terms, each targeting a distinct failure mode that a single L² loss
leaves unaddressed:

```
L_total = L_lp  +  α_h1 · L_h1  +  α_sp · L_spectral
```

|Term                          |Fixes                           |Applied over                 |
|------------------------------|--------------------------------|-----------------------------|
|`L_lp` — relative L²          |Overall amplitude accuracy      |Full 4D field (X, Y, Z, T)   |
|`L_h1` — spatial H¹ Sobolev   |Patch-boundary discontinuities  |Spatial (X, Y, Z) per τ-slice|
|`L_spectral` — radial spectral|High-frequency spectral accuracy|Spatial (X, Y, Z) per τ-slice|

-----

## Why Three Terms?

A single L² loss on the physical-space field is the standard starting point for
FNO training, but it has two blind spots that are particularly damaging for
relativistic hydrodynamics surrogates.

### Blind spot 1 — Patch-boundary discontinuities

`CausalLogLo`’s local spectral branch divides the spatial domain into
non-overlapping patches and reassembles them by direct concatenation. There is no
learned blending and no communication across patch boundaries. If the model learns
a slightly different zero-level or baseline in adjacent patches, the stitched
output will have visible discontinuities at patch edges.

L² is insensitive to this: a field with sharp edges at the patch boundaries can
have exactly the same L² value as a smooth field. The H¹ Sobolev norm adds an
explicit **spatial gradient penalty** that directly penalises these discontinuities.
This is the same reason MG-TFNO (Kossaifi et al., arXiv:2310.00120) uses H¹
training, stated verbatim in that paper:

> *“Since MG-TFNO predicts local regions which are then stitched together to form
> a global function without any communication, aliasing effects can occur where one
> output prediction does not flow smoothly into the next. To prevent this, we train
> our model using the H¹ Sobolev norm. By matching derivatives, training with this
> loss prevents any discontinuities from occurring and the output prediction is smooth.”*

### Blind spot 2 — Spectral bias suppresses high-frequency learning

Neural networks exhibit a well-documented tendency to learn low-frequency
components of a function before high-frequency ones — the so-called **spectral
bias** or frequency bias. In physical space, L² weights all spatial locations
equally. In Fourier space, this translates to weighting all wavenumbers equally,
which in practice means the optimizer reduces the dominant low-frequency error first
and only attends to high-k modes late in training, if at all.

For QGP hydrodynamics, the high-frequency spatial content is not secondary: flow
harmonics v₂ and v₃ are driven by the eccentricity of the initial energy density
distribution, which lives at moderate to high spatial frequencies. A model trained
only on L² can reproduce the overall energy profile accurately while
underestimating the anisotropy.

The spectral loss redistributes training pressure across the frequency spectrum by
computing the **power-spectrum error per radial wavenumber shell** and assigning
increasing weight to higher-frequency shells.

-----

## Term 1 — Relative L² (`L_lp`)

**What it is:** The standard FNO training loss. Measures the ratio of the
prediction error norm to the target norm, making it scale-invariant across
events with different total energy:

$$L_{\text{lp}} = \frac{|\hat{u} - u|*{L^2}}{|u|*{L^2}}$$

Applied over the full 4D field `(B, C, X, Y, Z, T)` using `LpLoss(d=4, p=2)`
from the `neuraloperator` library.

**What it minimises:** Overall pointwise prediction error averaged uniformly over
space and time.

**What it does not catch:** Patch discontinuities (zero gradient contribution from
a step that is constant within each patch) and high-frequency spectral errors when
they are small in amplitude but large in frequency content.

**Default weight:** 1.0 (anchor; the other two weights are chosen relative to this).

-----

## Term 2 — Spatial H¹ Sobolev (`L_h1`)

**What it is:** The H¹ Sobolev norm adds the squared spatial gradient of the error
to the squared amplitude error:

$$L_{H^1} = \int_D \left( |\hat{u} - u|^2 + |\nabla_{xyz}\hat{u} - \nabla_{xyz}u|^2 \right) , dx,dy,dz$$

where ∇ₓᵧᵤ denotes the gradient over the three **spatial** axes only. The time
axis τ is never included.

**Implementation:** T is folded into the batch dimension before the loss is
computed, so `H1Loss(d=3)` sees tensors of shape `(B·T, C, X, Y, Z)`. This
makes the loss equivalent to averaging the per-τ spatial H¹ error over all τ
steps, consistent with the per-step training philosophy of the causal model.

```python
# T folded into batch → H1Loss sees (B·T, C, X, Y, Z)
p = pred.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
t = target.permute(0, 5, 1, 2, 3, 4).reshape(B * T, C, X, Y, Z)
loss = H1Loss(d=3).rel(p, t)
```

**Why spatial-only, not 4D?** Two reasons. First, sharp τ-transitions are
physically real in the QGP evolution (rapid cooling at early proper time) and
should not be penalised. Second, the causal Conv1d already enforces
temporal smoothness structurally through its receptive field; adding a ∂/∂τ
gradient penalty would be redundant and potentially counterproductive.

**What it minimises:** Spatial gradient discontinuities at patch boundaries.
Forces the network to produce spatially smooth outputs at every τ-slice.

**What it does not catch:** Spectral content beyond the gradient scale. A smoothly
wrong field (correct gradients, wrong Fourier amplitudes at k > 2) can have a
small H¹ loss.

**Default weight:** `alpha_h1 = 0.03`

This value balances `α_h1 · L_h1 ≈ 0.55` against `L_lp ≈ 0.60` at ~30%
relative prediction error (mid-training typical). The raw H¹ value is ~18×
larger than L² at the same error level, so the alpha compensates for the
different numerical scale.

**Periodicity flag:** `periodic_xyz=True` (default) uses periodic finite
differences, matching the assumption of a toroidal spatial domain. Set
`periodic_xyz=False` if your grid has hard boundaries (e.g. a rapidity window
with a sharp η cutoff).

-----

## Term 3 — Radially-Binned Spectral Loss (`L_spectral`)

**What it is:** The error is transformed to Fourier space, and the power-spectrum
error is accumulated in concentric spherical shells in 3D wavenumber space
`(kx, ky, kz)`:

$$L_{\text{spectral}} = \frac{1}{N_\text{bins}} \sum_{i=0}^{N_\text{bins}-1} w_i \cdot \overline{|\hat{F}(\hat{u}-u)|^2}_{k \in \text{shell}_i}$$

where the per-shell weight increases linearly from low to high frequency:

$$w_i = 1 + (\texttt{weight_hf} - 1) \cdot \frac{i}{N_\text{bins} - 1}$$

With `weight_hf=2.0`, the highest-frequency shell is weighted twice as heavily as
the lowest. With `weight_hf=1.0` the loss reduces to a flat spectral L², which is
mathematically equivalent to physical-space MSE.

**Implementation:** T folded into batch before the FFT, so the transform runs over
`(X, Y, Z)` at each τ-step independently. Uses `torch.fft.rfftn` (real FFT, no
redundant negative-frequency computation along Z).

**What it minimises:** Frequency-by-frequency spectral accuracy. Specifically, it
raises the gradient signal for high-k modes that L² de-emphasises, pushing the
model to correctly represent fine spatial structure — eccentricity harmonics,
nucleon sub-structure, freeze-out surface texture.

**What it does not catch:** It operates in frequency space and is insensitive to
the spatial location of errors. A spectral error uniformly distributed in space
looks the same as one concentrated at a single patch boundary.

**Default weight:** `alpha_spectral = 0.001`

The raw spectral loss value is ~275× larger than L² at the same error level
(primarily because it operates on squared Fourier amplitudes which grow with grid
volume). The default alpha brings `α_sp · L_sp ≈ 0.28` into the same range as
the other two terms.

-----

## Default Parameters

```python
CausalLogLoLoss(
    alpha_spectral = 0.001,   # spectral term weight
    alpha_h1       = 0.03,    # H¹ term weight
    n_bins         = 8,       # number of radial frequency shells
    weight_hf      = 2.0,     # highest-frequency shell weight multiplier
    warmup_epochs  = 0,       # 0 = all terms active from epoch 0
    lp_p           = 2,       # L^p exponent (2 = standard L²)
    periodic_xyz   = True,    # periodic finite differences for H¹
)
```

### Scale balance at defaults

The following values were measured empirically on a `16×16×8×32` grid at ~30%
relative prediction error, which is typical for a model in mid-training:

|Term        |Raw value|Alpha|Weighted contribution|
|------------|---------|-----|---------------------|
|`L_lp`      |0.60     |1.0  |**0.60**             |
|`L_h1`      |19.2     |0.03 |**0.58**             |
|`L_spectral`|275      |0.001|**0.28**             |
|**Total**   |—        |—    |**1.46**             |

All three weighted terms contribute at the same order of magnitude. No single term
dominates the gradient signal.

### Tuning guidance

|Observation                                  |Adjustment                                    |
|---------------------------------------------|----------------------------------------------|
|Visible stripes or steps at patch boundaries |Increase `alpha_h1` (try 0.05–0.1)            |
|v₂/v₃ harmonics underestimated               |Increase `alpha_spectral` (try 0.002–0.005)   |
|Training unstable at early epochs            |Enable warm-up (`warmup_epochs=20–50`)        |
|Smooth fields but wrong large-scale structure|Decrease `alpha_h1`, increase `alpha_spectral`|
|Boundary artefacts at η edges                |Set `periodic_xyz=False`                      |
|High-viscosity regime (smooth target)        |Decrease `weight_hf` toward 1.0               |
|Low-viscosity / turbulent regime             |Increase `weight_hf` toward 3.0–5.0           |

-----

## Warm-Up Schedule

Training with all three terms active from epoch 0 can destabilise early training.
The H¹ gradient term amplifies high-frequency noise before the model has learned
the dominant low-frequency structure, which can cause gradient explosions or slow
convergence.

The built-in schedule phases in the two auxiliary terms linearly:

```
Epoch range        Active terms
─────────────────────────────────────────────────────────
[0,    W/2)        L_lp only
[W/2,  W  )        L_lp + ramp(α_h1) · L_h1
[W,    2W )        L_lp + α_h1 · L_h1 + ramp(α_sp) · L_spectral
[2W,   ∞  )        L_lp + α_h1 · L_h1 + α_sp · L_spectral
```

where W = `warmup_epochs`. Example with `warmup_epochs=20`:

|Epoch|`w_h1`|`w_spectral`|
|-----|------|------------|
|0    |0.0000|0.000000    |
|10   |0.0000|0.000000    |
|15   |0.0150|0.000000    |
|20   |0.0300|0.000000    |
|30   |0.0300|0.000500    |
|40   |0.0300|0.001000    |

Call `loss_fn.set_epoch(epoch)` at the top of each training epoch to activate the
schedule. Set `warmup_epochs=0` to disable and use all terms at full weight from
the start.

-----

## Usage

### Minimal

```python
from combined_loss import CausalLogLoLoss

loss_fn = CausalLogLoLoss()   # all defaults

total, breakdown = loss_fn(pred, target)
total.backward()
# breakdown = {'lp': 0.60, 'h1': 19.2, 'spectral': 275.0,
#              'h1_weighted': 0.58, 'spectral_weighted': 0.28, 'total': 1.46}
```

### Training loop with warm-up

```python
from combined_loss import CausalLogLoLoss, format_breakdown

loss_fn = CausalLogLoLoss(
    alpha_spectral=0.001,
    alpha_h1=0.03,
    warmup_epochs=20,
)

for epoch in range(n_epochs):
    loss_fn.set_epoch(epoch)      # must call before each epoch

    for pred, target in loader:
        optimizer.zero_grad()
        total, bd = loss_fn(pred, target)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if is_logger:
        print(format_breakdown(bd, epoch=epoch))
        # epoch=12 | total=1.2345 | lp=0.5712 | h1=18.43 (weighted=0.5529)
        #          | spectral=273.1 (weighted=0.2731)
```

### Accessing individual terms for diagnostic logging

```python
loss_fn = CausalLogLoLoss(warmup_epochs=0)

# Individual terms without combining (useful for validation)
l_lp = loss_fn.lp_loss(pred, target)
l_h1 = loss_fn.h1_spatial_loss(pred, target)   # spatial-only, T in batch
l_sp = loss_fn.spectral_loss(pred, target)

# Log to wandb / tensorboard
wandb.log({
    "loss/total":    total.item(),
    "loss/lp":       l_lp.item(),
    "loss/h1_raw":   l_h1.item(),
    "loss/sp_raw":   l_sp.item(),
})
```

### Disable individual terms

```python
# L² + H¹ only (no spectral): good for debugging patch stitching
loss_fn = CausalLogLoLoss(alpha_spectral=0.0, alpha_h1=0.03)

# L² + spectral only (no H¹): good when patches are large enough to not need smoothing
loss_fn = CausalLogLoLoss(alpha_spectral=0.001, alpha_h1=0.0)

# Pure L²: standard FNO baseline for ablation
loss_fn = CausalLogLoLoss(alpha_spectral=0.0, alpha_h1=0.0)
```

### With NormFn3d preprocessing

The loss operates on normalised fields (output of `NormFn3d`). The H¹ and spectral
losses are applied to the normalised values, which is correct: the normalisation
`ed^0.75 · τ` is smooth and invertible, so discontinuities in normalised space
correspond to discontinuities in physical space.

```python
# Normalise targets once before training
shape_ref = NormFn3d.compute_ref_etau75_eta_shape(train_arr, tau_min, dtau)
target_norm, _ = NormFn3d.etau75_eta_shape_3d(target_arr, tau_min, dtau, shape_ref)

# pred and target_norm are both normalised → loss is consistent
total, bd = loss_fn(pred, torch.tensor(target_norm))
```

-----

## Output: the `breakdown` dict

`loss_fn.forward()` returns `(total_tensor, breakdown_dict)`. The dict contains:

|Key                  |Description                                           |
|---------------------|------------------------------------------------------|
|`'lp'`               |Raw L² value (before any weighting)                   |
|`'h1'`               |Raw H¹ value (before `alpha_h1` weighting)            |
|`'spectral'`         |Raw spectral value (before `alpha_spectral` weighting)|
|`'h1_weighted'`      |`alpha_h1 * h1` — the actual contribution to the total|
|`'spectral_weighted'`|`alpha_spectral * spectral` — the actual contribution |
|`'total'`            |Sum of the three weighted terms                       |

The raw values are useful for monitoring how the loss landscape evolves: if `h1`
starts decreasing while `spectral` stays flat, the model is learning spatial
smoothness but not spectral accuracy, suggesting `alpha_spectral` should be
increased.

-----

## References

- **MG-TFNO H¹ motivation**: Kossaifi et al., *TMLR 2023*, [arXiv:2310.00120](https://arxiv.org/abs/2310.00120)
- **SPHERE spectral loss**: Kalimuthu, Holzmüller, Niepert, *TMLR 2025*, [arXiv:2504.04260](https://arxiv.org/abs/2504.04260)
- **H¹Loss / LpLoss**: NeuralOperator 2.0 library, [arXiv:2512.01421](https://arxiv.org/abs/2512.01421)
- **Spectral bias in neural networks**: Rahaman et al., *ICML 2019*
- **CausalLogLo model**: `causal_loglo_4d.py` (this project)
- **NormFn3d preprocessing**: `NormFn3d.py` (this project)