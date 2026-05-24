# CausalLogLo — 3+1D Neural Operator for Viscous Relativistic Hydrodynamics

`causal_loglo_4d.py`

-----

## Overview

`CausalLogLo` is a surrogate model for time-dependent 3D PDEs with a causal initial-value
structure, built for fields on grids of shape `(B, C, X, Y, Z, T)`.

It extends the standard [Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895)
with three architectural improvements, each targeting a distinct failure mode:

|Failure mode in standard FNO                                     |Fix in CausalLogLo                                                            |
|-----------------------------------------------------------------|------------------------------------------------------------------------------|
|Spectral bias: FNO truncates high-frequency modes globally       |**LOGLO local spectral branch**: patch-wise FFT retains all modes locally     |
|Acausal receptive field along τ; joint 4D FFT is non-physical    |**Causal temporal factorization**: 3D spatial FFT per τ-slice + causal Conv1d |
|`NormFn3d` removes amplitude cue; network cannot self-locate in τ|**τ-channel embedding**: sinusoidal sin/cos(τ) injected at input and per block|

The three ideas are **orthogonal** — they address different axes (spatial structure,
temporal causality, temporal position) and compose without conflict.

-----

## Motivation

### Why standard 4D FNO is insufficient

A standard FNO applied to a 4D field `(X, Y, Z, T)` has three problems for
relativistic hydrodynamics:

**1. Global mode truncation loses local spatial structure.**
FNO keeps only the lowest `(mx, my, mz, mt)` Fourier modes. This is an
explicit spectral filter: anything smaller than the coherence length
`L/m` in each direction is discarded. For initial conditions from TRENTo or
IP-Glasma, nucleon sub-structure and eccentricity fluctuations live at
frequencies above typical FNO cutoffs and are silently dropped.

**2. A joint 4D FFT treats τ as periodic.**
The Fourier transform assumes the signal is periodic. The proper-time axis
of an initial-value problem is not periodic — there is a single inflow
boundary at τ₀. Applying FFT along τ couples future states into past ones,
violating causality and forcing the model to learn unphysical correlations.

**3. Non-stationarity along τ is invisible to the model.**
The energy density field changes character dramatically with proper time:
early τ has steep transverse gradients near the nucleon positions; late τ is
smooth as viscous diffusion erases structure. A single set of spectral weights
must represent both regimes, which requires capacity that could be used more
efficiently if the model knew which regime it was in.

### How NormFn3d interacts with the embedding

`NormFn3d` applies the physics-motivated normalization
`ed → ed^0.75 · τ` (Bjorken scaling) before the data enters the network.
This equalizes amplitudes across τ — a training stability tool — but it
removes the dominant signal the causal Conv1d would otherwise use to
self-locate in proper time. The sinusoidal τ-embedding restores that
positional information explicitly and cleanly, as a coordinate rather than
a data-derived proxy, and the two mechanisms address complementary problems.

-----

## Architecture

### Block diagram

Each `CausalLogLoBlock` processes `(B, C, X, Y, Z, T)` through five sequential steps:

```
x  (B, C, X, Y, Z, T)
│
├─ (1) τ-bias injection
│       _TauBias: learned Linear(2→C) applied to sin/cos(τ) embedding
│       adds per-τ channel bias broadcast over (X, Y, Z)
│
├─ (2) Fold T into batch  →  x_bt  (B·T, C, X, Y, Z)
│
│   Three parallel branches on x_bt:
│
│   ├─ (3a) GLOBAL branch
│   │       SpectralConv3D(n_modes_global) + Conv3d_1x1   [Tucker compressed]
│   │       → GELU → ChannelMLP → + SoftGating(x_bt)
│   │
│   ├─ (3b) LOCAL branch
│   │       Partition (X,Y,Z) into M non-overlapping patches of size (Px,Py,Pz)
│   │       SpectralConv3D(n_modes_local) + Conv3d_1x1    [all modes per patch]
│   │       → GELU → ChannelMLP → + SoftGating(x_bt)
│   │       Reassemble patches → (B·T, C, X, Y, Z)
│   │
│   └─ (3c) HIGH-FREQUENCY stream
│           x_hf = x_bt − lowpass(x_bt)   [spectral lowpass at n_modes_global cutoffs]
│           ChannelMLP(x_hf)
│
├─ Sum of three branches → GELU
│
├─ (4) Unfold T  →  (B, C, X, Y, Z, T)
│
└─ (5) Causal dilated Conv1d along T
        Flatten spatial → (B·X·Y·Z, C, T)
        F.pad(left=(K-1)·d, right=0)   ← causality enforced structurally
        Conv1d(C, C, kernel=K, dilation=d)
        → (B, C, X, Y, Z, T)

+ outer residual:  output = block(x) + x
```

### Full model flow

```
Input (B, in_channels, X, Y, Z, T)
  │
  ├─ Concatenate sin/cos(τ) embedding → (B, in_channels+2, X, Y, Z, T)
  │
Lifting Conv3d(in_channels+2 → C, kernel=1)     [T folded into batch]
  │
  ├─ CausalLogLoBlock  [dilation=1]
  ├─ CausalLogLoBlock  [dilation=2]
  ├─ CausalLogLoBlock  [dilation=4]
  │    ...
  └─ CausalLogLoBlock  [dilation=2^(L-1)]
  │
Projection Conv3d(C → C, kernel=1) → GELU → Conv3d(C → out_channels, kernel=1)
  │
Output (B, out_channels, X, Y, Z, T)
```

-----

## Parameters

### Standard FNO parameters (unchanged meaning)

|Parameter        |Type   |Description                                                             |
|-----------------|-------|------------------------------------------------------------------------|
|`in_channels`    |`int`  |Number of input physical fields (e.g. 4 for ed, vx, vy, vz)             |
|`out_channels`   |`int`  |Number of predicted output fields                                       |
|`hidden_channels`|`int`  |Latent width C throughout the network                                   |
|`factorization`  |`str`  |Tucker / CP / TT or None — tensor factorization for SpectralConv weights|
|`rank`           |`float`|Compression rank fraction (e.g. 0.05 = 5% of full Tucker rank)          |

### New parameters beyond standard FNO

#### `n_modes_global` — `tuple[int, int, int]`

The number of Fourier modes retained in the **global spectral branch**, one per
spatial axis `(mx, my, mz)`. This is the standard FNO mode truncation applied to
the 3D spatial FFT at each τ-slice (T is no longer a spectral dimension).

- Modes beyond this cutoff are discarded by the global branch but recovered by
  the local branch and HF stream.
- Smaller values → cheaper global branch, more parameter budget for local branch.
- Typical: `(8, 8, 4)` for a 64×64×32 grid; `(12, 12, 8)` for 128×128×64.
- **Note:** There is no `n_modes_t` parameter — τ is handled by `t_kernel_size`
  and `dilation_schedule`, not by Fourier modes.

#### `patch_size` — `tuple[int, int, int]`

The spatial dimensions `(Px, Py, Pz)` of each **local branch patch**.
The domain is tiled into `M = (X/Px) × (Y/Py) × (Z/Pz)` non-overlapping
hypercubes. Each patch is processed independently by a `SpectralConv3D` that
retains **all** Fourier modes within the patch (no truncation).

**Constraints:**

- `X % Px == 0`, `Y % Py == 0`, `Z % Pz == 0` — grid must divide evenly.
- Patches should be large enough to contain the physical correlation length of
  interest. Nucleon sub-structure in TRENTo events at mid-rapidity has a typical
  scale of ~1 fm; at 0.1 fm/fm/c grid spacing this needs patches of at least 10
  cells per transverse direction.
- Smaller patches → cheaper local FFT, but shorter local receptive field.
- Typical: `(16, 16, 8)` for a 64×64×32 grid.

The local branch automatically uses `n_modes_local = (Px//2, Py//2, Pz//2)`,
which are the Nyquist limits — all accessible frequencies within the patch.

#### `t_kernel_size` — `int` (default `2`)

Kernel size `K` of the causal dilated `Conv1d` along τ. With `K=2` each block
contributes a receptive field of `1 + (K-1)·d = 1 + d` τ-steps. Larger `K`
gives wider receptive field per block at proportionally higher compute cost.

**Recommendation:** Keep `K=2`. The doubling dilation schedule grows the
receptive field exponentially with depth; large `K` is redundant.

#### `dilation_schedule` — `'double'` or `'constant'` (default `'double'`)

Controls how the dilation factor `d` grows across the `n_layers` blocks.

|Schedule    |Dilations          |Causal receptive field|
|------------|-------------------|----------------------|
|`'double'`  |1, 2, 4, …, 2^(L-1)|2^L − 1 τ-steps       |
|`'constant'`|1, 1, 1, …, 1      |L · (K−1) + 1 τ-steps |

With `K=2` and `n_layers=7`, doubling gives a receptive field of **127 τ-steps**
(covering T=128 exactly); constant gives only 7 τ-steps.

**When to use `'constant'`:** Only if τ correlations are genuinely short-range
(e.g. a purely diffusive problem at large viscosity). For QGP hydrodynamics where
the initial condition influences the full evolution, use `'double'`.

The attribute `model.temporal_receptive_field` reports the resulting value.

#### `tau_max` — `float` (default `12.0`)

The reference proper-time scale in fm/c, used to normalize the sinusoidal
embedding: `sin(π·τ/τ_max)` and `cos(π·τ/τ_max)`.

- Set this to the **final τ** of your hydrodynamic evolution (e.g. freeze-out
  time). For Au+Au at √s=200 GeV, typical values are 10–15 fm/c.
- The embedding is unique and smooth for all `τ ∈ [0, τ_max]`.
- Values `τ > τ_max` are not forbidden but extrapolate outside the trained range.
- Must match the `tau_max` used during training at inference time.

#### `use_checkpoint` — `bool` (default `False`)

If `True`, each block’s three-branch spatial forward (`_spatial_forward`) is
wrapped with `torch.utils.checkpoint`. Activations are discarded during the
forward pass and recomputed during backpropagation.

|                      |`use_checkpoint=False`|`use_checkpoint=True`              |
|----------------------|----------------------|-----------------------------------|
|Peak activation memory|~3× single-branch FNO |~1× single-branch FNO              |
|Training compute      |1×                    |~1.7× (one extra forward per block)|
|Inference             |Unchanged             |Unchanged (no effect at eval time) |

**Recommendation:** Set `True` for grids 64×64×32×128 and larger, or whenever
training on A100 40 GB.

-----

## Loss Function: `spectral_loss_3d`

```python
from causal_loglo_4d import spectral_loss_3d

loss = spectral_loss_3d(pred, target, n_bins=8, weight_hf=2.0)
```

### Motivation

The standard L² loss `||pred - target||²` in physical space weights all
spatial locations equally. In Fourier space this is equivalent to weighting
all wavenumbers equally. Because neural networks exhibit **spectral bias**
— they learn low-frequency components more easily than high-frequency ones
— the optimizer tends to minimize the low-frequency error first while
high-frequency errors persist. For flow harmonics (v₂, v₃) and freeze-out
surfaces this is particularly damaging because those quantities are sensitive
to intermediate-frequency transverse structure.

### What it does

The loss computes the **power spectrum error** in concentric spherical shells
in 3D wavenumber space `(kx, ky, kz)`, then sums with a **linearly increasing
weight** from low-frequency to high-frequency shells:

```
k_rad = sqrt(kx² + ky² + kz²)

For each radial bin i in [0, n_bins):
    w_i = 1 + (weight_hf - 1) × i / (n_bins - 1)
    loss += w_i × mean_power_error_in_bin_i
```

With `weight_hf=2.0`, the highest-frequency bin is penalized **twice as heavily**
as the lowest, linearly interpolated. With `weight_hf=1.0` the loss reduces to a
flat spectral L² loss (equivalent to physical-space MSE up to normalization).

The loss folds τ into the batch axis, so it is computed per time-step and averaged:
it is consistent with a standard per-step training objective.

### Combining with L² loss

In practice, combine with standard LpLoss for training stability:

```python
from neuralop import LpLoss
from causal_loglo_4d import spectral_loss_3d

lp_loss = LpLoss(d=4, p=2)   # 4D (3 spatial + 1 time)

def combined_loss(pred, target, alpha=0.1):
    return lp_loss(pred, target) + alpha * spectral_loss_3d(pred, target)
```

Start with `alpha=0.1` and increase if the model produces visually smooth
but spectrally inaccurate predictions (under-resolved eccentricity harmonics).

### Parameters

|Parameter  |Default|Description                                                  |
|-----------|-------|-------------------------------------------------------------|
|`n_bins`   |`8`    |Number of radial wavenumber shells                           |
|`weight_hf`|`2.0`  |Weight multiplier for the highest-frequency bin. `1.0` = flat|

-----

## Usage Examples

### Minimal

```python
import torch
from causal_loglo_4d import CausalLogLo

model = CausalLogLo(
    in_channels=4,          # ed, vx, vy, vz
    out_channels=1,         # e.g. energy density at next step
    hidden_channels=32,
    n_modes_global=(8, 8, 4),
    patch_size=(16, 16, 8),
    n_layers=7,
    tau_max=12.0,
    use_checkpoint=True,
).cuda()

# Physical τ grid: 128 steps from τ=0.4 to τ=12.0 fm/c
tau = torch.linspace(0.4, 12.0, 128, device='cuda')

# Input: one event, 4 fields, 64×64×32 transverse+rapidity grid, 128 τ steps
x = torch.randn(1, 4, 64, 64, 32, 128, device='cuda')

y = model(x, tau)   # (1, 1, 64, 64, 32, 128)
```

### Training loop with combined loss

```python
import torch
import torch.nn.functional as F
from neuralop import LpLoss
from causal_loglo_4d import CausalLogLo, spectral_loss_3d

model = CausalLogLo(
    in_channels=4,
    out_channels=4,
    hidden_channels=48,
    n_modes_global=(8, 8, 4),
    patch_size=(16, 16, 8),
    n_layers=7,
    dilation_schedule='double',
    tau_max=12.0,
    factorization='tucker',
    rank=0.05,
    use_checkpoint=True,
).cuda()

tau = torch.linspace(0.4, 12.0, 128, device='cuda')   # physical grid [fm/c]
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
lp_loss   = LpLoss(d=4, p=2)
alpha     = 0.1   # spectral loss weight

for x_batch, y_batch in train_loader:
    x_batch = x_batch.cuda()   # (B, 4, 64, 64, 32, 128)
    y_batch = y_batch.cuda()   # (B, 4, 64, 64, 32, 128)

    optimizer.zero_grad()
    y_pred = model(x_batch, tau)

    loss = lp_loss(y_pred, y_batch) + alpha * spectral_loss_3d(y_pred, y_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

### With NormFn3d preprocessing

`NormFn3d` and `CausalLogLo` are used at different stages and are fully
independent. Apply normalization to your data **before** passing it to the model:

```python
import numpy as np
from NormFn3d import NormFn3d

tau_min, dtau = 0.4, (12.0 - 0.4) / 127   # fm/c

# Normalize training data (once, before any training)
shape_ref = NormFn3d.compute_ref_etau75_eta_shape(train_arr, tau_min, dtau)
train_norm, _ = NormFn3d.etau75_eta_shape_3d(train_arr, tau_min, dtau, shape_ref)
# save shape_ref for inference

# During inference, pass physical τ to the model for the embedding
tau = torch.tensor(tau_min + dtau * np.arange(128)).float().cuda()
y_norm = model(x_norm.cuda(), tau)

# Invert normalization to recover physical fields
y_phys = NormFn3d.inv_etau75_eta_shape_3d(y_norm.cpu().numpy(), shape_ref, tau_min, dtau)
```

The τ values passed to `model(x, tau)` are the **physical** proper-time coordinates
in fm/c, not normalized. `NormFn3d` is applied only to the field values, not the
τ coordinate itself.

### FSDP multi-GPU training

`CausalLogLo` is FSDP-compatible via the existing `wrap_tfno_with_fsdp()` helper,
because complex parameters are confined to `SpectralConv` modules — the same
constraint that applies to `TFNO`.

```python
from mg_tfno_4d_fsdp import wrap_tfno_with_fsdp
from causal_loglo_4d import CausalLogLo

model = CausalLogLo(...).to(device)
model = wrap_tfno_with_fsdp(model, device)
# Training loop unchanged from mg_tfno_4d_fsdp.py
```

### Verifying causality

After building a model, confirm that future τ-slices cannot influence past ones:

```python
from causal_loglo_4d import verify_causality

ok = verify_causality(
    model,
    shape=(1, 4, 8, 8, 4, 32),   # small grid, CPU-safe
    tau_max=12.0,
    perturbation_step=16,         # perturb at τ index 16, check τ < 16
    tol=1e-5,
)
# Prints per-step delta and causal: True/False
```

Causality is enforced **structurally** by left-only padding in `Conv1d`
(`F.pad(y_t, (pad_left, 0))`), not by regularization. The check should return
`True` at machine precision (< 1e-6) for any correctly initialized model.

-----

## Grid Constraints

The grid dimensions must satisfy the following divisibility conditions:

```
X % Px == 0
Y % Py == 0
Z % Pz == 0
```

where `(Px, Py, Pz) = patch_size`. Common configurations:

|Grid (X×Y×Z×T) |Patch (Px×Py×Pz)|Patches (Mx×My×Mz)|n_modes_global|
|---------------|----------------|------------------|--------------|
|64×64×32×128   |16×16×8         |4×4×4 = 64        |(8, 8, 4)     |
|128×128×64×128 |32×32×16        |4×4×4 = 64        |(12, 12, 8)   |
|128×128×128×150|32×32×32        |4×4×4 = 64        |(16, 16, 16)  |
|32×32×16×64    |16×16×8         |2×2×2 = 8         |(6, 6, 4)     |

The number of patches does not affect model parameters — only the patch
size and `n_modes_global` do.

-----

## Memory and Compute Reference

Relative to standard 4D FNO (3 spatial + 1 temporal spectral dimension),
for a 64×64×32×128 grid:

|Component                  |Activation memory|FLOPs     |Parameters               |
|---------------------------|-----------------|----------|-------------------------|
|**4D FNO baseline**        |1×               |1×        |1×                       |
|Causal factorization       |~1×              |~0.73×    |~1/Mt (no temporal modes)|
|+ LOGLO branches           |~3×              |~1.4×     |~1.5×                    |
|+ τ-embedding              |+2/C ≈ negligible|negligible|negligible               |
|**+ `use_checkpoint=True`**|**~1×**          |~1.7×     |—                        |

The LOGLO local branch adds ~1× activation memory (same total spatial voxels,
just differently grouped). Gradient checkpointing on `_spatial_forward` (the
three-branch computation) recovers this cost at the price of a recompute pass.

-----

## Implementation Notes

### Local branch: patch split and reassemble

Patches are created via view/permute/reshape without any data copy, making
the split O(1) in time:

```
(BT, C, X, Y, Z)
→ view   (BT, C, Mx, Px, My, Py, Mz, Pz)
→ permute(0, 2, 4, 6, 1, 3, 5, 7)           # → (BT, Mx, My, Mz, C, Px, Py, Pz)
→ reshape (BT·M, C, Px, Py, Pz)             # → patch batch
```

Reassembly is the exact inverse. The round-trip is lossless (verified in the
self-test).

### HF stream: spectral lowpass

The high-frequency residual `x_hf = x − lowpass(x)` is computed by an FFT,
zeroing the modes that the global branch retains, then an inverse FFT. The mask
covers four quadrants of the `(kx, ky)` plane to handle the symmetric
negative-frequency modes, and one-sided along `kz` (rfft axis).

### τ-embedding: two injection points

- **At lifting** (once): the two embedding channels `[sin(πτ/τ_max), cos(πτ/τ_max)]`
  are concatenated to the input channels before lifting. The lifting `Conv3d`
  accepts `in_channels + 2` input channels.
- **Per block** (at each layer): `_TauBias` applies a `Linear(2 → C)` projection
  to produce a per-τ additive bias broadcast over `(X, Y, Z)`. This prevents
  the positional signal from decaying as it passes through multiple channel-mixing
  operations.

-----

## References

- **LOGLO-FNO**: Kalimuthu, Holzmüller, Niepert — *TMLR 2025*, [arXiv:2504.04260](https://arxiv.org/abs/2504.04260)
- **FNO**: Li et al. — [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- **TFNO / neuraloperator**: Kossaifi et al. — [arXiv:2310.00120](https://arxiv.org/abs/2310.00120)
- **LocalNO (NO-LIDK)**: Liu-Schiaffini et al. — *ICML 2024*, [arXiv:2402.16845](https://arxiv.org/abs/2402.16845)
- **Bjorken flow**: Bjorken, *Phys. Rev. D* **27** (1983) — motivation for `ed^0.75·τ` normalization
- **NormFn3d**: `NormFn3d.py` (this project)
- **Causal FNO**: `causal_fno_4d.py` (this project)
- **MG-TFNO + FSDP**: `_mg_tfno_4d_fsdp.py` (this project)