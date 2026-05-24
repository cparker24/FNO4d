# Normalisation pipeline for hydro FNOs

This document describes how to use the normalisation helpers in
`NormFn3d.py` — both the original Bjorken families and the new
z-score / τ-embedding additions — when training an FNO surrogate
for relativistic hydrodynamics.

A working example is in `test_train_normalised.py`, which is a minimal
extension of `test_train.py` showing the integration end-to-end.

---

## 1. What is in `NormFn3d`

```
                          input (n_events, n_features, nx, ny, n_eta, n_tau)
                          │
                          │  ── Bjorken rescaling on ed channel only ──
                          ▼
                etau75_3d                  ed → ed^0.75 · τ
                etau75_eta_3d              ed → (ed^0.75 · τ) / ref[η, τ]
                cosh_etau75_3d             ed → (ed · cosh(η)^α)^0.75 · τ
                etau75_eta_shape_3d        ed → (ed^0.75 · τ) / shape_ref[η, τ]
                          │
                          │  ── Per-(feature, τ) z-score on ALL channels ──
                          ▼
                apply_zscore_3d(mu, sigma) data → (data − μ) / σ
                          │
                          │  ── τ sinusoidal embedding, concatenated as channels ──
                          ▼
                concatenate with tau_sinusoidal_embedding output
                          │
                          ▼
                          ready for the model
```

Each step has a matching inverse (`inv_*`) for recovering physical units.

---

## 2. Why each step exists

**Bjorken rescaling.** The energy density follows
$\varepsilon(\tau) \propto \tau^{-1-c_s^2}$ in Bjorken expansion
($\propto \tau^{-4/3}$ for an ideal conformal fluid). Multiplying by
$\tau^{4/3}$ — equivalent to raising $\varepsilon^{0.75}$ and multiplying
by $\tau$ — flattens the dominant trend so the network only has to
model deviations from it. Without this step the FNO's lowest temporal
Fourier modes are spent encoding the $1/\tau$ decay rather than the
interesting physics.

**Per-(feature, τ) z-score.** After Bjorken rescaling, three sources of
non-stationarity remain: velocity features still carry their raw scale,
$\varepsilon^{0.75}\tau$ has residual mild τ-trend from transverse
expansion and viscosity, and the spatial dynamic range varies strongly
with τ. A simple z-score with statistics computed from the *training*
split removes all three. The output has approximately zero mean and unit
variance at every (feature, τ) cell.

This step is **especially important** if you switch to a
$\tau$-translation-invariant architecture (e.g. `CausalFNO4D` with
shared Conv1d filters along τ). Time-shared filters facing
non-stationary data have to act as a gain control on top of the
dynamics, which is a non-trivial extra job; z-scoring removes it.

**τ sinusoidal embedding.** Per-τ z-scoring makes the input statistically
indistinguishable across τ. That is good for the optimisation but
removes information the model genuinely needs — namely, *which*
τ this slice is. A sinusoidal embedding with a geometric series of
periods restores absolute-τ information as extra input channels. The
construction is the same as transformer positional encodings.

---

## 3. Choosing a Bjorken family

| Family                     | When to use                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `etau75_3d`                | Quickest baseline. η variation fully visible to the network. Default for first runs. |
| `etau75_eta_shape_3d`      | **Recommended for production.** Preserves Bjorken τ evolution at peak η; equalises η dynamic range across bins. |
| `etau75_eta_3d`            | Aggressive: every (η, τ) cell becomes O(1), but the τ evolution is absorbed into the normalisation. Use only when you're confident the network should learn the spatial pattern at fixed τ rather than the τ dynamics. |
| `cosh_etau75_3d`           | Analytic η correction with no training-data dependency. Use when training stats are unstable (few events, ragged η coverage). |

All four are interchangeable in the pipeline below — they all produce
data with the same shape and Bjorken-rescaled energy density.

---

## 4. The recommended pipeline

In numpy, applied to data of shape
`(n_events, n_features, nx, ny, n_eta, n_tau)`:

```python
from NormFn3d import NormFn3d

# ── ONCE per dataset: compute statistics from the TRAINING split only ──
shape_ref = NormFn3d.compute_ref_etau75_eta_shape(train_data, tau_min, dtau)
train_bj  = NormFn3d.etau75_eta_shape_3d(train_data, tau_min, dtau, shape_ref)[0]
mu, sigma = NormFn3d.compute_zscore_stats(train_bj)
emb       = NormFn3d.tau_sinusoidal_embedding(n_tau, tau_min, dtau,
                                              n_emb=8, max_period=20.0)
np.savez('norm_stats.npz', shape_ref=shape_ref, mu=mu, sigma=sigma, emb=emb)

# ── For each split (train, val, test): apply with shared stats ──
def normalise(arr):
    arr_bj   = NormFn3d.etau75_eta_shape_3d(arr, tau_min, dtau, shape_ref)[0]
    arr_norm = NormFn3d.apply_zscore_3d(arr_bj, mu, sigma)
    return arr_norm

train_norm = normalise(train_data)
val_norm   = normalise(val_data)
# (τ embedding is broadcast per-batch inside the Dataset; see Section 6)
```

The four files you need to keep for the model lifetime are
`shape_ref`, `mu`, `sigma`, and the (deterministic) embedding
configuration `(n_tau, tau_min, dtau, n_emb, max_period)`. Save them
together so they cannot drift apart.

---

## 5. Inverting the pipeline

At inference time, invert in **reverse order**:

```python
# Model outputs `pred_norm` in normalised units (no τ embedding —
# the model only outputs physical channels).
pred_bj       = NormFn3d.inv_zscore_3d(pred_norm, mu, sigma)
pred_physical = NormFn3d.inv_etau75_eta_shape_3d(pred_bj, shape_ref,
                                                 tau_min, dtau)
```

`pred_physical` is then in GeV/fm³ (or whatever units your raw `arr`
was in) and ready for Cooper-Frye, v_n extraction, or visualisation.

---

## 6. Integrating into a training script

The cleanest split is:

1. **Apply Bjorken + z-score on numpy arrays**, once after the
   train/val split. Saves recomputing per batch.
2. **Add τ embedding inside the Dataset's `__getitem__`**. The embedding
   is τ-varying per output step, so it must be constructed against the
   target τ of each sample.
3. **Set the model's `in_channels = n_features + n_emb`** because the
   input now carries the embedding. The `out_channels` stays at
   `n_features` — the model predicts only physical fields.

The example script `test_train_normalised.py` shows this with minimal
diff against `test_train.py`. The key changes are localised to:

- Normalisation block right after the train/val split.
- A new `NormalisedFluidDataset` that attaches the τ embedding on the
  fly to match the prediction τ of each sample.
- One line in the `FNO(...)` constructor: `in_channels=4 + n_emb`.

---

## 7. Loss: normalised or physical units?

You have two reasonable choices.

**Loss in normalised units (simpler, the example default).**
Both prediction and target come out of the dataset in normalised
units. The H1Loss / LpLoss is computed directly. This is fine because
per-(feature, τ) z-scoring already balances loss contributions across
regimes — there is no longer the "early τ dominates the gradient"
problem that motivates physical-units losses. **Recommended starting
point.**

**Loss in physical units (more principled, more code).**
Inverse-normalise prediction and target before computing the loss.
Necessary if:

- You want to weight regimes by physical magnitude (e.g. emphasise the
  late-time region where Cooper-Frye observables live).
- You add a PINO-style PDE residual term $\|\partial_\mu T^{\mu\nu}\|^2$.
  **PDE residual losses MUST be evaluated in the physical frame** —
  the τ-dependent rescaling introduces spurious source terms in the
  rescaled equations.

In the example script, the comment block in the training loop shows
exactly where to insert un-normalisation if you want this.

---

## 8. Common pitfalls

- **Computing stats on the wrong split.** `mu`, `sigma`, `shape_ref`
  must come from the training split *only* — otherwise the validation
  metrics leak information about the test distribution. The helper
  `compute_*` functions take a single array; pass the train array.
- **Forgetting to save stats.** The model is useless without
  `shape_ref`, `mu`, `sigma`. Save them in the same checkpoint as the
  weights, or in a sibling `.npz` file named identically.
- **Inverting in the wrong order.** Always inverse z-score first, then
  inverse Bjorken. The order matters because z-score's μ and σ were
  computed in Bjorken-rescaled units.
- **Including the τ embedding in the output.** The model outputs only
  physical channels (`out_channels = n_features`). The embedding is
  input-only.
- **Mismatched embedding configuration across splits.** The embedding
  is a deterministic function of `(n_tau, tau_min, dtau, n_emb,
  max_period)`. If any of these change between training and inference,
  the model sees a different positional signature. Pin the values.
- **Using `etau75_eta_3d` with too few training events.** The per-(η, τ)
  reference is a mean across events; with very few events it is noisy
  and can over-fit. Prefer `etau75_eta_shape_3d` (or `cosh_etau75_3d`
  with no data dependency) when `n_train < 1000` or so.

---

## 9. Quick sanity checks before training

After normalisation, run these:

```python
print("ed channel after Bjorken (train, mean over events & space):")
print(train_bj[:, 0].mean(axis=(0, 1, 2, 3)))           # (n_tau,) — should be roughly flat in τ

print("All channels after z-score (mean, std per τ):")
print(train_norm.mean(axis=(0, 2, 3, 4)))               # (n_features, n_tau) — should be ~0
print(train_norm.std (axis=(0, 2, 3, 4)))               # (n_features, n_tau) — should be ~1

print("Embedding range:")
emb = NormFn3d.tau_sinusoidal_embedding(n_tau, tau_min, dtau, 8)
print(emb.min(), emb.max())                              # should be in [-1, 1]
```

The ed-after-Bjorken plot is the most informative: if the mean is still
trending strongly with τ, your Bjorken exponent is off (try a different
family, or check that `tau_min` is the *actual* simulation start time
in fm/c, not some normalised index).
