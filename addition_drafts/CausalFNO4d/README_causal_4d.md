# Causal FNO 4D

A 4D Fourier Neural Operator with structurally causal time axis, designed
as a drop-in replacement for `neuralop.models.TFNO` in problems where the
T axis is an initial-value direction rather than just another spatial axis.

This README explains the physical motivation, the architecture, and how
its parameters map to those of the original FNO / TFNO. It also documents
the `CausalFNO4DTauEta` variant, which adds sinusoidal τ/η positional
embeddings for Milne-coordinate hydrodynamics (see §10).

---

## 1. Physical motivation

A vanilla TFNO treats $(X, Y, Z, T)$ symmetrically: a single 4D FFT, a
single block of complex spectral weights, the same mode-truncation
schedule on every axis. That symmetric treatment is implicitly assuming
several things that are true for the spatial axes but not for time in an
initial-value problem like relativistic viscous hydrodynamics:

- **Periodicity.** The FFT assumes the signal wraps around. Spatial
  boxes can be made periodic; time cannot — $\tau_0$ is the initial
  condition and $\tau_\text{max}$ is freeze-out, and they have nothing
  to do with each other.
- **Globally two-sided receptive field.** The spectral convolution
  mixes information across all $\tau$ at once, so the prediction at
  $\tau$ can in principle depend on what happens at $\tau' > \tau$.
  That breaks causality at the representation level.
- **Stationarity along the axis.** Spatial statistics are roughly
  translation-invariant. Temporal statistics in a heavy-ion event are
  emphatically not: early-time gradients are orders of magnitude
  larger than late-time gradients, the spectral content changes
  systematically with $\tau$, and the relevant timescales shrink with
  $\tau \to 0$.
- **Spectral truncation as a low-pass filter.** Along a spatial axis,
  keeping the lowest $m$ Fourier modes filters out small-scale spatial
  structure. Along T, where the field is monotonically decaying rather
  than oscillating, the lowest modes mostly encode the overall slope.
  Truncating modes along T discards qualitatively different content
  than truncating modes along space.

The sound horizon $c_s \cdot \tau \ll \tau_\text{max}$ also tells us
that a *finite* causal receptive field is more aligned with the physics
than the global receptive field the 4D FFT provides — information
genuinely cannot travel arbitrarily far in T within the simulation
duration.

This module respects those asymmetries structurally rather than asking
the network to learn around them.

---

## 2. Architecture

Each block is split into two stages:

1. **Spatial stage — 3D spectral convolution over $(X, Y, Z)$.**
   The original `SpectralConv` from `neuralop.layers.spectral_convolution`
   is reused unchanged, but configured for 3D. It is applied at each
   $\tau$ independently by folding T into the batch dimension. This
   keeps:
   - the global spatial receptive field,
   - the Tucker / CP / TT factorization machinery,
   - the parameter savings from `rank < 1`,
   - the complex-dtype FSDP handling you already have.

2. **Temporal stage — causal dilated 1D convolution along T.**
   A standard `nn.Conv1d` with `padding=0` and an explicit *left-only*
   zero-pad of $(K-1) \cdot d$ samples. The left-only pad is what
   enforces causality: timestep $t$ in the output depends only on
   timesteps $\{t,\ t-d,\ t-2d,\ \dots,\ t-(K-1)d\}$ in the input, and
   never on any $t' > t$.

Blocks are stacked with a doubling dilation schedule $d_\ell = 2^\ell$.
With kernel size $K$, the causal receptive field after $L$ layers is

$$
\text{RF}_T = 1 + (K-1)\sum_{\ell=0}^{L-1} 2^\ell = 1 + (K-1)(2^L - 1).
$$

For $K = 2$ this gives $\text{RF}_T = 2^L$ timesteps. The model exposes
this number as `model.temporal_receptive_field` so you can check
coverage at construction time.

This is essentially the FNO-2D path from the original FNO paper (Li et
al., 2021) — spatial spectral convolution combined with explicit time
propagation — modernised by replacing the RNN with a dilated causal
convolution stack (TCN/WaveNet-style). The original paper noted that
FNO-3D was more expressive than FNO-2D + RNN but lost the ability to
extrapolate in time; the dilated causal stack recovers most of that
expressiveness in a fully parallel, easily trained form.

---

## 3. Parameter mapping: TFNO → CausalFNO4D

| TFNO argument          | CausalFNO4D argument        | Notes |
|------------------------|------------------------------|-------|
| `n_modes=(mx, my, mz, mt)` | `n_modes_spatial=(mx, my, mz)` | Drop the last entry. Temporal "bandwidth" now comes from `t_kernel_size`, `n_layers`, `dilation_schedule`. |
| `hidden_channels`      | `hidden_channels`            | Unchanged. |
| `in_channels`          | `in_channels`                | Unchanged. |
| `out_channels`         | `out_channels`               | Unchanged. |
| `factorization`        | `factorization`              | Unchanged. Forwarded directly to the inner `SpectralConv`. |
| `rank`                 | `rank`                       | Unchanged. Same semantics. |
| `implementation`       | (hard-coded `"factorized"`)  | Forwarded to `SpectralConv`. Edit the class if you need `"reconstructed"`. |
| `n_layers`             | `n_layers`                   | Same role (depth), but now ALSO sets temporal receptive field. |
| —                      | `t_kernel_size`              | NEW. Causal Conv1d kernel size along T. |
| —                      | `dilation_schedule`          | NEW. `"double"` or `"constant"`. |

A concrete migration:

```python
# BEFORE — TFNO with temporal modes
tfno = TFNO(
    n_modes=(8, 8, 8, 6),
    hidden_channels=32,
    in_channels=8,
    out_channels=1,
    factorization="tucker",
    implementation="factorized",
    rank=0.05,
    n_layers=4,
)

# AFTER — CausalFNO4D with causal temporal conv
model = CausalFNO4D(
    in_channels=8,
    out_channels=1,
    hidden_channels=32,
    n_modes_spatial=(8, 8, 8),     # was n_modes[:3]
    n_layers=7,                    # raised to cover T=128 causally
    t_kernel_size=2,
    dilation_schedule="double",
    factorization="tucker",        # unchanged
    rank=0.05,                     # unchanged
)
```

The spatial `SpectralConv` instances inside `CausalFNO4D` are
configured identically to those that would live inside a TFNO with
`n_modes=(mx, my, mz)`. The Tucker / CP / TT factorization, the
complex weight storage, the FSDP handling — all carry over with no
changes.

---

## 4. Choosing parameters

### Spatial modes (`n_modes_spatial`)

Identical considerations to the original FNO: cover the spectral
range that matters for your problem; keep below Nyquist; small
numbers (8–16) usually suffice for hydro-like fields where the
spectrum decays smoothly. If you used $(8, 8, 8, \cdot)$ in TFNO,
use $(8, 8, 8)$ here.

### Number of layers (`n_layers`)

In CausalFNO4D this number plays two roles: it sets the depth of the
trunk (as before) and, with `dilation_schedule="double"`, it sets the
temporal receptive field.

For $K = 2, d = $ doubling:

| `n_layers` | Temporal receptive field |
|-----------:|-------------------------:|
| 4          | 16  τ-steps              |
| 5          | 32  τ-steps              |
| 6          | 64  τ-steps              |
| 7          | 128 τ-steps              |
| 8          | 256 τ-steps              |

Pick the smallest `n_layers` for which `temporal_receptive_field >=`
the number of T steps in your dataset.
For full $T = 128$ coverage use `n_layers=7` (RF = 128); for $T = 150$
use `n_layers=8` (RF = 256; cheap to err on the side of more coverage).

### Temporal kernel size (`t_kernel_size`)

- `t_kernel_size=2`: pair with doubling dilation. Gives an exact
  binary tree of temporal connections, maximal parameter efficiency,
  and exponential growth in receptive field. **Default and recommended.**
- `t_kernel_size=3` or `4`: more taps per layer, larger receptive
  field per layer, more parameters per layer. Useful if you'd rather
  use fewer layers; useful if you suspect non-trivial multi-tap
  temporal structure that a binary tree might miss.

### Dilation schedule

- `"double"`: dilations are $1, 2, 4, 8, \ldots$. Best for covering
  long time ranges with few layers.
- `"constant"`: dilations are all 1. The receptive field is
  $n_\text{layers} \cdot (K - 1) + 1$, linear in depth. Use this if
  you want a strictly local temporal model — for example, if you
  believe the relevant physics is Markovian on a short window and
  longer-range "connections" would just be wasted capacity.

### Rank / factorization

Identical to TFNO. The factorization is applied to the spatial
spectral weights only — the temporal Conv1d weights are dense and
full-rank because they are tiny. A typical heavy-ion-hydro
configuration might be `factorization="tucker", rank=0.05`,
matching your current MG-TFNO setup.

---

## 5. Factorization compatibility

The inner module is `neuralop.layers.spectral_convolution.SpectralConv`,
unchanged. Anything that module supports, this model supports:

| factorization | supported | notes |
|---------------|:---------:|-------|
| `None` (dense) | ✓ | full-rank spectral weights, biggest parameter count |
| `"tucker"`     | ✓ | recommended default; matches existing MG-TFNO usage |
| `"cp"`         | ✓ | best at very high compression / low rank |
| `"tt"`         | ✓ | tensor-train, similar regime to Tucker |

The `implementation` argument of `SpectralConv` is hard-coded to
`"factorized"` in this module to match the existing MG-TFNO setup. If
you need `"reconstructed"` (slightly faster, more memory), edit the
single `implementation=` line in `CausalTimeFNOBlock`.

---

## 6. Integration with the training scripts

Select the model from the YAML config: set `model_type: causal_fno4d`
(or `causal_fno4d_tau_eta`) in `config_sel.yaml` / `config_h5_sel.yaml`,
which `train_torch_sel.py` / `train_torch_h5_sel.py` read to construct the
model. Extra constructor kwargs (`n_layers`, `t_kernel_size`,
`dilation_schedule`, `factorization`, `rank`, and for the embedding variant
`tau_max` / `eta_max`) go under a `causal_fno4d:` block in the same config.
The `n_modes` entry is reused with only its first three values
(`n_modes_spatial = n_modes[:3]`); the T entry is ignored. The patching loop,
multigrid prefiltering, FSDP wrapping, and gradient-hook logic continue to
work without modification:

- **Patching.** Spatial patches with full T preserved per patch still
  produce inputs of shape `(B, C, X_p, Y_p, Z_p, T)`, which is what
  `CausalFNO4D` expects.
- **`_spectral_lowpass_4d`.** Your multigrid prefiltering already
  FFTs only over $(X, Y, Z)$ with T folded into the batch. That is
  *exactly* the operator structure of `CausalFNO4D`, so the multigrid
  filtering is consistent with the new architecture.
- **FSDP.** The complex Tucker weights still live inside `SpectralConv`
  modules. Your `complex_mods` discovery (`any(p.is_complex() for p
  in mod.parameters(recurse=False))`) catches them the same way. The
  new `Conv1d` temporal layers have real weights and shard normally
  via `use_orig_params=True`.
- **Gradient checkpointing.** No changes needed — the block boundary
  is in the same place.

Migration is therefore just a `model_type` switch in the YAML config.

---

## 7. What you give up

Being honest about the trade-offs:

- **Resolution invariance along T is lost.** The temporal Conv1d has
  a fixed kernel size and stride. You can no longer feed a different
  $\Delta\tau$ at inference and have the operator interpolate cleanly.
  The FNO discretization invariance still holds along $(X, Y, Z)$.
- **Maximum temporal range is finite.** With doubling dilation and
  $K = 2$, $L$ layers give RF$_T = 2^L$ timesteps. Predictions that
  depend on a longer history than that will lose information, whereas
  the original 4D FNO had global receptive field along T.
- **Slightly more layers needed.** To cover $T = 128$ causally you
  need $L = 7$, vs the $L = 4$ that's common in TFNO setups. The cost
  is modest — each extra layer is one more `SpectralConv` (the
  Conv1d is negligible) — but it's not free.

---

## 8. When to use which

| Situation                                           | Recommended |
|-----------------------------------------------------|-------------|
| Initial-value problem, non-stationary T, finite information speed | **CausalFNO4D** |
| Statistically stationary in T (e.g. attractor dynamics, periodic forcing) | original TFNO |
| Need temporal super-resolution                       | original TFNO |
| Need explicit causality guarantees (safety, physics) | **CausalFNO4D** |
| Long-range temporal correlations beyond sound horizon | original TFNO |

For relativistic-hydro emulation specifically — non-stationary fields,
strict initial-value structure, finite sound horizon, no need for
temporal super-resolution — CausalFNO4D is the better-aligned choice.

---

## 9. Verification

Run the self-test:

```bash
python causal_fno_4d.py
```

This builds a small model, prints its parameter count and reported
`temporal_receptive_field`, runs a forward pass, and verifies
causality structurally by perturbing the input at a single $\tau$ and
confirming the output is *exactly* unchanged at every earlier $\tau$
(to machine precision, not just trained-to-small).

The `verify_causality` function is exposed at module level so you can
also call it against the patched + multigrid forward pass — useful as a
regression test when integrating into the training scripts.

For `CausalFNO4DTauEta`, run `verify_causality` the same way — its τ/η
embeddings are positional only, so they do not break the causal structure.

---

## 10. τ/η positional-embedding variant (`CausalFNO4DTauEta`)

For Milne-coordinate hydrodynamics it helps the network to know *where* it
is in proper time τ and spacetime rapidity η. `CausalFNO4DTauEta` wraps
`CausalFNO4D` and injects that information with the same two-part scheme as
`FNOTauEta`:

1. **Input concatenation.** Four sinusoidal channels — $[\sin, \cos]$ of
   $\pi\tau/\tau_\text{max}$ (broadcast along T) and $[\sin, \cos]$ of
   $\pi\eta/\eta_\text{max}$ (broadcast along Z) — are appended to the
   physical input before the lifting MLP, so the first projection already
   sees its $(\tau, \eta)$ location. The inner `CausalFNO4D` is therefore
   constructed with `in_channels + 4`.
2. **Per-block additive bias.** A small `Linear(2 → hidden_channels)`
   (`TauBias` / `EtaBias`) re-adds the positional signal after every block,
   so it does not decay with depth.

Because both injections depend only on the coordinates τ and η — never on
field values at other timesteps — causality is preserved exactly;
`verify_causality` passes on this model too.

### Constructor

Identical to `CausalFNO4D` plus two scale parameters:

| argument  | meaning |
|-----------|---------|
| `tau_max` | reference proper time [fm/c] for the τ embedding; also the upper bound of the default τ grid when `tau` is not passed. |
| `eta_max` | $|\eta|$ range for the η embedding; also the symmetric bound of the default η grid when `eta` is not passed. |

```python
from neuralop.models import CausalFNO4DTauEta

model = CausalFNO4DTauEta(
    in_channels=4, out_channels=4, hidden_channels=32,
    n_modes_spatial=(8, 8, 8),
    n_layers=7, t_kernel_size=2, rank=0.05,
    tau_max=12.4, eta_max=4.0,
)
```

### Forward signature

```python
y = model(x, tau, eta)   # x: (B, C, X, Y, Z, T) -> y: (B, C_out, X, Y, Z, T)
```

- `tau`: shape `(T,)`, physical proper-time values [fm/c]. Defaults to a
  uniform grid $[0, \tau_\text{max}]$ when `None`.
- `eta`: shape `(Z,)`, spacetime-rapidity values. Defaults to a uniform grid
  $[-\eta_\text{max}, +\eta_\text{max}]$ when `None`.

### Building τ/η from the data grid

The coordinate vectors should match the dataset's grid spacing, not a
uniform `[0, tau_max]`. Two helpers in `causal_fno_4d.py` build them from the
`NormFn3d` grid metadata (`tau_min`, `dtau`, `eta_min`, `deta`):

```python
from neuralop.models.causal_fno_4d import make_tau_tensor, make_eta_tensor

tau = make_tau_tensor(tau_min, dtau, n_tau, device=device)   # (n_tau,)
eta = make_eta_tensor(eta_min, deta, n_eta, device=device)   # (n_eta,)
```

`train_torch_sel.py` does exactly this and pins the resulting `tau` / `eta`
into a thin wrapper, so the neuralop `Trainer`'s `model(x)` call works
unchanged.

---

## References

- Li et al., 2021. *Fourier Neural Operator for Parametric Partial
  Differential Equations.* ICLR 2021. (FNO-2D vs FNO-3D discussion.)
- Kossaifi et al., 2023. *Multi-Grid Tensorized Fourier Neural Operator
  for High-Resolution PDEs.* arXiv:2310.00120. (Tucker factorization,
  multigrid domain decomposition.)
- Kossaifi et al. (FNO review, 2024). *A Library for Learning Neural
  Operators.* arXiv:2512.01421. (Autoregressive rollouts, RNO,
  initial-value vs operator-learning trade-offs.)
- Liu-Schiaffini et al., 2024. *Neural Operators with Localized
  Integral and Differential Kernels.* arXiv:2402.16845. (Hybrid
  global/local kernel structures.)
