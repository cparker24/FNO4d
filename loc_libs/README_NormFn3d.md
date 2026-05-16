# NormFn3d — 3-D Hydro Normalisation Helpers

`NormFn3d.py` provides normalisation functions for MUSIC/JETSCAPE hydro arrays
with shape

```
(n_events, n_features, nx, ny, n_eta, n_tau)
```

| axis | meaning |
|------|---------|
| 0 | events |
| 1 | features: 0 = `ed`, 1 = `vx`, 2 = `vy`, 3 = `vz` |
| 2 | x bins |
| 3 | y bins |
| 4 | η (spatial rapidity) bins |
| 5 | τ (proper time) bins |

Only **feature 0** (`ed`) is transformed. Velocity components are passed
through unchanged by every function.

---

## Grid parameters (dAu / AuAu datasets)

| parameter | typical value | meaning |
|-----------|--------------|---------|
| `tau_min` | 0.5 fm/c | proper time at τ bin 0 |
| `dtau`    | 0.1 fm/c | proper time step |
| `eta_min` | −5.0     | spatial rapidity at η bin 0 |
| `deta`    | 0.3125   | spatial rapidity step |

These appear in the output of `read_3d_data` and must be passed to every
function explicitly.

---

## Normalisation families

### 1. `etau75_3d` — τ scaling only

```
ed  →  ed^0.75 · τ
```

Motivated by ideal Bjorken flow where `e(τ) ~ τ^{−4/3}`, making
`e^{3/4} · τ` approximately constant in time.
**No correction is applied along η.** Energy density can vary by an order
of magnitude between mid-rapidity (η ≈ 0) and large |η|, so this
normalisation leaves the FNO seeing very different scales across η bins.

Use when: you want exact compatibility with `NormFn.etau75_3d`, or when
your model only uses a narrow η range near midrapidity.

---

### 2. `etau75_eta_3d` — τ scaling + data-driven η rescaling

```
ed  →  (ed^0.75 · τ) / ref[ieta, itau]
```

where

```
ref[ieta, itau] = mean_over_events( max_{x,y}( ed^0.75 · τ ) )
```

is computed once from the **training split** with `compute_ref_etau75_eta`.
This makes every (η, τ) cell arrive at the FNO with an O(1) typical scale
while preserving the transverse x-y spatial pattern.

Because `ref` varies with both η and τ, it absorbs the absolute amplitude
at each (η, τ) point — including any residual τ evolution of `ed^0.75 · τ`
that departs from ideal Bjorken scaling.  The FNO therefore sees a
time-independent normalised field.

**`ref` must be saved alongside the model** and reused unchanged for
validation, test, and inference — computing a fresh `ref` from each split
would leak information and break the inverse.

Use when: your model spans the full η range **and** you want the FNO to
learn only the spatial pattern, not the time evolution.

---

### 3. `cosh_etau75_3d` — τ scaling + analytic cosh(η) correction

```
ed  →  (ed · cosh(η)^α)^0.75 · τ
```

In Milne coordinates the longitudinal energy-density profile falls roughly
as `~1/cosh(η_s)` due to the finite beam rapidity and the Lorentz-frame
Jacobian. Multiplying `ed` by `cosh(η)^α` compensates this analytically,
without requiring any training data.

| `alpha` | physical meaning |
|---------|-----------------|
| 1.0 (default) | removes the leading-order cosh fall-off |
| 4/3 | absorbs the full longitudinal boost factor; pre-factor is `(e·cosh^{4/3})^{3/4}·τ`, directly analogous to the Bjorken constant of motion |

**Note:** this correction is only effective when the true η profile actually
falls as `~1/cosh(η)`.  For realistic MUSIC 3+1D output with a Gaussian-like
rapidity plateau (σ ≈ 2–3), `cosh(η)` grows much faster than the profile
falls, so the correction can worsen the η dynamic range rather than improve
it.  Use `etau75_eta_3d` or `etau75_eta_shape_3d` instead in that case.

Use when: the true η profile is known to follow `~1/cosh(η)`, or as an
analytic first-pass correction when no training data is available.

---

### 4. `etau75_eta_shape_3d` — τ scaling + shape-only η correction

```
ed  →  (ed^0.75 · τ) / shape_ref[ieta, itau]
```

where

```
raw_ref[ieta, itau]   = mean_over_events( max_{x,y}( ed^0.75 · τ ) )
shape_ref[ieta, itau] = raw_ref[ieta, itau] / max_over_η( raw_ref[:, itau] )
```

`shape_ref ∈ (0, 1]` by construction:
- **1** at the η bin with the highest mean-max energy at each τ (typically
  near midrapidity)
- smaller values at other η bins, encoding the relative η shape

#### Key distinction from `etau75_eta_3d`

The difference lies in what `ref` encodes:

| | what `ref` contains | τ variation after norm | η variation after norm |
|---|---|---|---|
| `etau75_eta_3d` | absolute scale at each (η, τ) cell | absorbed → flat | absorbed → flat |
| `etau75_eta_shape_3d` | relative η shape only (peak = 1) | **preserved** | absorbed → flat |

Because `shape_ref = 1` at the peak η bin, dividing by it leaves
`ed^0.75 · τ` untouched at that bin.  Any residual τ evolution in
`ed^0.75 · τ` that departs from ideal Bjorken scaling is therefore visible
to the FNO.  All other η bins are amplified to match the peak scale, so the
η dynamic range is equalised identically to `etau75_eta_3d`.

#### Dynamic range summary (non-Bjorken synthetic data, τ^{−1} profile)

| normalisation | η ratio | τ ratio |
|---|---|---|
| raw ed | 7.5× | 8.0× |
| `etau75_3d` | 4.5× | 1.7× |
| `etau75_eta_3d` | **1.0×** | **1.0×** ← τ evolution absorbed |
| **`etau75_eta_shape_3d`** | **1.0×** | **1.7×** ← τ evolution preserved |

> **Why τ ratio = 1 for all methods on perfect Bjorken data?**
> If `ed ~ τ^{−4/3}` exactly, then `ed^0.75 · τ = τ^{0} = const`, so there
> is no τ variation for any method to either preserve or absorb.  The
> distinction only appears for real MUSIC output that deviates from ideal
> Bjorken scaling (3D expansion, transverse flow, freeze-out effects).

Use when: your model spans the full η range **and** you want the FNO to
see the Bjorken τ evolution of the energy (learn time dynamics, not only
the spatial pattern at each fixed time step).

**`shape_ref` must be saved alongside the model** exactly like `ref` for
`etau75_eta_3d`.

---

## Function reference

| function | signature | returns |
|----------|-----------|---------|
| `etau75_3d` | `(data, tau_min, dtau)` | `normed` |
| `inv_etau75_3d` | `(data, tau_min, dtau, eps)` | `restored` |
| `compute_ref_etau75_eta` | `(train_data, tau_min, dtau, eps)` | `ref (n_eta, n_tau)` |
| `etau75_eta_3d` | `(data, tau_min, dtau, ref, eps)` | `(normed, ref)` |
| `inv_etau75_eta_3d` | `(data, ref, tau_min, dtau, eps)` | `restored` |
| `cosh_etau75_3d` | `(data, tau_min, dtau, eta_min, deta, alpha)` | `normed` |
| `inv_cosh_etau75_3d` | `(data, tau_min, dtau, eta_min, deta, alpha, eps)` | `restored` |
| `compute_ref_etau75_eta_shape` | `(train_data, tau_min, dtau, eps)` | `shape_ref (n_eta, n_tau) ∈ (0,1]` |
| `etau75_eta_shape_3d` | `(data, tau_min, dtau, shape_ref, eps)` | `(normed, shape_ref)` |
| `inv_etau75_eta_shape_3d` | `(data, shape_ref, tau_min, dtau, eps)` | `restored` |

---

## Typical workflow

### Import

```python
import sys
sys.path.insert(0, 'loc_libs')       # adjust path as needed
from NormFn3d import NormFn3d
import numpy as np
```

### Grid constants (read from read_3d_data output or hardcode)

```python
TAU_MIN = 0.5    # fm/c
DTAU    = 0.1    # fm/c
ETA_MIN = -5.0
DETA    = 0.3125
```

---

### Family 1 — `etau75_3d`

```python
# forward
normed   = NormFn3d.etau75_3d(arr, TAU_MIN, DTAU)

# inverse (e.g. after model prediction)
restored = NormFn3d.inv_etau75_3d(normed, TAU_MIN, DTAU)
```

---

### Family 2 — `etau75_eta_3d`

**Step 1 — build the reference map from training data once:**

```python
n_train   = int(0.8 * arr.shape[0])
train_arr = arr[:n_train]
val_arr   = arr[n_train:]

ref = NormFn3d.compute_ref_etau75_eta(train_arr, TAU_MIN, DTAU)
np.save('ref_etau75_eta.npy', ref)   # save alongside model checkpoint
```

**Step 2 — normalise all splits with the same ref:**

```python
train_norm, _ = NormFn3d.etau75_eta_3d(train_arr, TAU_MIN, DTAU, ref=ref)
val_norm,   _ = NormFn3d.etau75_eta_3d(val_arr,   TAU_MIN, DTAU, ref=ref)
```

**Step 3 — inverse at inference time:**

```python
ref      = np.load('ref_etau75_eta.npy')
restored = NormFn3d.inv_etau75_eta_3d(pred, ref, TAU_MIN, DTAU)
```

> **Do not** call `etau75_eta_3d` without `ref=` on validation or test data.
> If `ref` is omitted, a fresh ref is computed from that batch, which breaks
> the inverse and introduces data leakage.

---

### Family 3 — `cosh_etau75_3d`

```python
# forward (alpha=1.0 default removes leading cosh fall-off)
normed   = NormFn3d.cosh_etau75_3d(arr, TAU_MIN, DTAU, ETA_MIN, DETA, alpha=1.0)

# inverse — pass the same alpha
restored = NormFn3d.inv_cosh_etau75_3d(normed, TAU_MIN, DTAU, ETA_MIN, DETA, alpha=1.0)

# physically motivated alternative: alpha = 4/3
normed43 = NormFn3d.cosh_etau75_3d(arr, TAU_MIN, DTAU, ETA_MIN, DETA, alpha=4/3)
```

---

### Family 4 — `etau75_eta_shape_3d`

**Step 1 — build the shape reference from training data once:**

```python
n_train   = int(0.8 * arr.shape[0])
train_arr = arr[:n_train]
val_arr   = arr[n_train:]

shape_ref = NormFn3d.compute_ref_etau75_eta_shape(train_arr, TAU_MIN, DTAU)
np.save('shape_ref_etau75_eta.npy', shape_ref)   # save alongside model checkpoint
# shape_ref.shape == (n_eta, n_tau),  values in (0, 1]
```

**Step 2 — normalise all splits with the same shape_ref:**

```python
train_norm, _ = NormFn3d.etau75_eta_shape_3d(train_arr, TAU_MIN, DTAU, shape_ref=shape_ref)
val_norm,   _ = NormFn3d.etau75_eta_shape_3d(val_arr,   TAU_MIN, DTAU, shape_ref=shape_ref)
```

**Step 3 — inverse at inference time:**

```python
shape_ref = np.load('shape_ref_etau75_eta.npy')
restored  = NormFn3d.inv_etau75_eta_shape_3d(pred, shape_ref, TAU_MIN, DTAU)
```

> **Do not** omit `shape_ref=` on validation or test data for the same
> reason as `etau75_eta_3d`.

---

## Choosing between the families

```
Does your model span the full η range?
  No  → etau75_3d

  Yes →
        Should the FNO see the τ evolution of the energy
        (i.e. learn time dynamics, not just spatial patterns)?
          Yes → etau75_eta_shape_3d   η equalised, τ evolution intact
          No  → etau75_eta_3d         η and τ both absorbed → O(1) everywhere

        No training data available?
          → cosh_etau75_3d   analytic correction, no data needed
            (only effective if the η profile follows ~1/cosh(η);
             Gaussian-plateau profiles typically need etau75_eta_shape_3d)
```

---

## Inverse correctness criterion

Every family satisfies:

```
inv( norm(arr) ) ≈ arr     (round-trip error < 1e-4 for float32 inputs)
```

The small residual is float32 rounding noise from the power operations.
