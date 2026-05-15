# FNO Evaluation Metric Suite — `ModelEval`

Reference documentation for `loc_libs/model_eval.py` and the two metric-focused
marimo notebooks:

| Notebook | Framework |
|---|---|
| `train/eval_model_test_metric_torch.py` | PyTorch |
| `train/eval_model_test_metric_mlx.py` | MLX (Apple Silicon) |

See [README_eval.md](README_eval.md) for the simpler contour-only notebooks.

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Data format](#2-data-format)
3. [Quick start](#3-quick-start)
4. [Pointwise metrics](#4-pointwise-metrics)
   - 4.1 [nRMSE — normalized root-mean-square error](#41-nrmse)
   - 4.2 [RMSE — absolute root-mean-square error](#42-rmse)
   - 4.3 [MaxError — worst-case pointwise deviation](#43-maxerror)
   - 4.4 [bRMSE — boundary RMSE](#44-brmse)
   - 4.5 [cRMSE — conserved-quantity RMSE](#45-crmse)
   - 4.6 [vRMSE — variance-normalized RMSE](#46-vrmse)
5. [Spectral metrics](#5-spectral-metrics)
   - 5.1 [fRMSE — frequency-band RMSE](#51-frmse)
   - 5.2 [MELR — Mean Energy Log Ratio](#52-melr)
   - 5.3 [WLR — Wasserstein Log Ratio](#53-wlr)
   - 5.4 [spectral_error_map — 2-D Fourier error map](#54-spectral_error_map)
   - 5.5 [energy_spectrum — radially-averaged power spectrum](#55-energy_spectrum)
6. [Temporal stability: autoregressive rollout](#6-temporal-stability-autoregressive-rollout)
7. [Summary and plotting](#7-summary-and-plotting)
   - 7.1 [summary()](#71-summary)
   - 7.2 [Individual plots](#72-individual-plots)
   - 7.3 [plot_dashboard()](#73-plot_dashboard)
8. [Marimo notebook UI](#8-marimo-notebook-ui)
9. [Appendix — complete API reference](#appendix--complete-api-reference)

---

## 1. Motivation

Standard training losses (L² or H¹) collapse a 6-D field into a single scalar that
conflates every scale and every spatial location.  For a QGP hydrodynamic surrogate
this is insufficient: a model that predicts the bulk flow correctly but destroys the
sub-nucleon eccentricity fluctuations will appear "good" by nRMSE but produce wrong
flow harmonics v₃ and higher.

The `ModelEval` class implements the full battery of evaluation approaches recommended
for neural fluid-dynamics surrogates, ordered from the most broadly applicable to the
most domain-specific:

| Priority | Category | Metrics |
|---|---|---|
| Must report | Pointwise | nRMSE, MaxError, bRMSE, cRMSE |
| Must report | Spectral | fRMSE(low/mid/high), MELR, AR rollout |
| Recommended | Spectral | WLR, energy-spectrum overlay |
| Optional | Spectral | 2-D spectral error map |
| Domain gate | Physics obs. | v₂, v₃, ⟨pT⟩ (via Cooper-Frye, not yet automated) |

---

## 2. Data format

All metrics work on numpy arrays with this shared convention:

```
pred, true : ndarray  shape (N, C, nx, ny, neta, T)

N    — number of test events
C    — field channels  (0=ε, 1=vx, 2=vy, 3=vη)
nx   — transverse x grid points
ny   — transverse y grid points
neta — pseudo-rapidity (η) bins
T    — output τ steps  (initial condition NOT included)
```

The arrays must be in **physical units** (GeV/fm³ for energy density, dimensionless
for velocities).  If a normalizer was used during training, call
`normalizer.inverse()` before passing data to `ModelEval`.

The spectral metrics operate on 2-D transverse slices `(nx, ny)` at each `(η, τ)`
point, which is the physically natural decomposition: the transverse plane carries the
eccentricity information that drives anisotropic flow.

---

## 3. Quick start

### Create a `ModelEval` object

```python
import sys, numpy as np
sys.path.append('../loc_libs')
from model_eval import ModelEval

# pred, true: arrays (N, C, nx, ny, neta, T) in physical units
ev = ModelEval(
    pred          = model_out_phys,    # (N, C, nx, ny, neta, T)
    true          = y_true_phys,
    tau_coords    = tau_out_coords,    # 1-D array, length T
    eta_coords    = eta_coords,        # 1-D array, length neta
    channel_names = ['ε', 'vx', 'vy', 'vη'],
)
```

### Compute everything at once

```python
# All scalar summary metrics
print(ev.summary())

# 2×3 dashboard figure
fig = ev.plot_dashboard(channel=0, ieta=0)
fig.savefig('dashboard.pdf', bbox_inches='tight')
```

### Save to JSON for reporting

```python
import json
with open('out/metrics_eval.json', 'w') as f:
    json.dump(ev.summary(), f, indent=2)
```

### Run as a marimo notebook

```bash
cd train/
marimo run eval_model_test_metric_torch.py   # PyTorch
marimo run eval_model_test_metric_mlx.py     # MLX (native or PyTorch weight transfer)
```

---

## 4. Pointwise metrics

All pointwise methods return arrays of shape `(C, neta, T)` — one number per channel,
η bin, and τ step — with averaging over the event dimension N performed internally.
The exception is `crmse()`, which returns `(C, T)` because the volume integral
collapses the η dimension.

---

### 4.1 nRMSE

**Normalized root-mean-square error** — the primary headline metric in the neural
operator literature (PDEBench, FNO, UNet-based surrogates).

#### Definition

For each event $n$, channel $c$, η-bin $i_\eta$, and τ-step $t$:

$$\text{nRMSE}_{n,c,i_\eta,t} = \sqrt{ \frac{\displaystyle\frac{1}{n_x n_y}\sum_{i,j}\bigl(f^{\text{pred}} - f^{\text{true}}\bigr)^2}{\displaystyle\frac{1}{n_x n_y}\sum_{i,j}\bigl(f^{\text{true}}\bigr)^2} }$$

The reported value is the mean over events:

$$\text{nRMSE}_{c,i_\eta,t} = \frac{1}{N}\sum_n \text{nRMSE}_{n,c,i_\eta,t}$$

This per-sample then average convention follows PDEBench rather than computing RMSE
from the pooled event distribution, making it insensitive to the absolute magnitude
of the fields and directly comparable across different physical quantities (energy
density and velocities live on very different scales).

#### Interpretation

- `nRMSE ≈ 0` — perfect prediction
- `nRMSE ≈ 1` — error as large as the typical field amplitude
- Increases with τ reveals error accumulation; large jumps at specific η bins point to
  regional failures

#### Code

```python
nrmse = ev.nrmse()          # shape (C, neta, T)

# Mean over everything → headline scalar
print(f'Mean nRMSE: {nrmse.mean():.4f}')

# Per-channel, last τ step
for c, name in enumerate(ev.channel_names):
    print(f'{name}: nRMSE at final τ = {nrmse[c, :, -1].mean():.4f}')

# Plot vs τ for channel 0 (energy density)
fig, ax = ev.plot_nrmse_vs_tau(channel=0)
```

---

### 4.2 RMSE

**Absolute root-mean-square error** — same structure as nRMSE but without dividing
by the field RMS.  Useful for dimensional analysis or when comparing predictions of
a single channel within the same dataset.

#### Definition

$$\text{RMSE}_{c,i_\eta,t} = \sqrt{ \frac{1}{N n_x n_y}\sum_{n,i,j}\bigl(f^{\text{pred}}_{n,i,j,i_\eta,t} - f^{\text{true}}_{n,i,j,i_\eta,t}\bigr)^2 }$$

#### Code

```python
rmse = ev.rmse()    # shape (C, neta, T)

# Maximum RMSE across all η bins and τ steps for each channel
for c, name in enumerate(ev.channel_names):
    print(f'{name}: peak RMSE = {rmse[c].max():.4g}')
```

---

### 4.3 MaxError

**Maximum absolute pointwise deviation** over the full test set.  Complements nRMSE
by exposing rare but large local failures that an RMSE average would wash out.

#### Definition

$$\text{MaxError}_{c,i_\eta,t} = \max_{n,i,j} \bigl|f^{\text{pred}}_{n,i,j,i_\eta,t} - f^{\text{true}}_{n,i,j,i_\eta,t}\bigr|$$

#### Interpretation

For a QGP surrogate, a large `MaxError` in the energy-density channel at late τ is
physically dangerous: the freeze-out surface is sampled at the point where the local
temperature drops below $T_c \approx 155$ MeV, so even one mis-predicted spatial
spike can place the freeze-out surface in the wrong location and corrupt all
Cooper-Frye-sampled particle spectra.

#### Code

```python
maxerr = ev.max_error()     # shape (C, neta, T)

# Where in (η, τ) space is the worst prediction error?
c = 0   # energy density
ieta_worst, itau_worst = np.unravel_index(maxerr[c].argmax(), maxerr[c].shape)
print(f'Worst MaxError: {maxerr[c].max():.4g} GeV/fm³')
print(f'  at η={ev.eta_coords[ieta_worst]:.2f}, τ={ev.tau_coords[itau_worst]:.3f} fm/c')
```

---

### 4.4 bRMSE

**Boundary RMSE** — RMSE evaluated only on the outermost ring of the transverse
`(x, y)` grid, i.e. the cells satisfying `i=0`, `i=nx−1`, `j=0`, or `j=ny−1`.

#### Definition

Let $\mathcal{B}$ denote the set of boundary cell indices:

$$\text{bRMSE}_{c,i_\eta,t} = \sqrt{ \frac{1}{N|\mathcal{B}|}\sum_{n}\sum_{(i,j)\in\mathcal{B}} \bigl(f^{\text{pred}} - f^{\text{true}}\bigr)^2 }$$

#### Physical motivation

In the Israel-Stewart hydrodynamic evolution the physical freeze-out surface is
defined by a critical energy-density (or temperature) isosurface.  At late τ the
hot QGP core has cooled and the freeze-out surface has migrated toward the transverse
boundary of the active region.  The Cooper-Frye integral for particle production is
evaluated directly on this surface, so boundary accuracy is more important than
interior accuracy at that stage.  `bRMSE` quantifies exactly this.

#### Code

```python
brmse = ev.brmse()          # shape (C, neta, T)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for ie in range(ev.neta):
    ax.plot(ev.tau_coords, brmse[0, ie], label=f'η={ev.eta_coords[ie]:.2f}')
ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('bRMSE')
ax.set_title('Boundary RMSE — ε')
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
plt.tight_layout()
```

---

### 4.5 cRMSE

**Conserved-quantity RMSE** — normalized error in the spatially integrated field,
i.e. the error in the total "charge" carried by each field component.

#### Definition

Let $Q^{\text{pred}}_{n,c,t} = \sum_{i,j,k} f^{\text{pred}}_{n,c,i,j,k,t}$ be the
volume integral (sum over all `nx × ny × neta` cells).  Then:

$$\text{cRMSE}_{c,t} = \frac{1}{N}\sum_n \frac{\bigl|Q^{\text{pred}}_{n,c,t} - Q^{\text{true}}_{n,c,t}\bigr|}{|Q^{\text{true}}_{n,c,t}| + \varepsilon}$$

Note the return shape is `(C, T)`, not `(C, neta, T)`, because summing over η is
part of the metric definition.

#### Physical motivation

For ideal relativistic hydrodynamics the energy-momentum tensor $T^{\mu\nu}$ obeys
$\partial_\mu T^{\mu\nu} = 0$.  Discretely this means that the total energy and
momentum are conserved at each τ step.  A surrogate that violates this — even while
reproducing local field profiles — will produce unphysical particle spectra.
`cRMSE` is the tightest single number testing whether the model respects global
conservation laws.

```
cRMSE ≈ 0     → energy-momentum conserved to machine precision
cRMSE > 0.01  → 1% violation of total energy — flag for investigation
```

#### Code

```python
crmse = ev.crmse()          # shape (C, T)

fig, ax = plt.subplots(figsize=(7, 4))
for c, name in enumerate(ev.channel_names):
    ax.semilogy(ev.tau_coords, crmse[c], label=name)
ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('cRMSE (log)')
ax.set_title('Conserved-quantity RMSE')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
```

---

### 4.6 vRMSE

**Variance-normalized RMSE** — each per-sample RMSE is divided by the square root of
the spatial variance of the ground-truth field for that sample.  This measures
whether the model captures spatial *contrast* rather than just the mean field level.

#### Definition

$$\text{vRMSE}_{n,c,i_\eta,t} = \sqrt{ \frac{\displaystyle\frac{1}{n_x n_y}\sum_{i,j}(f^{\text{pred}} - f^{\text{true}})^2}{\displaystyle\mathrm{Var}_{i,j}(f^{\text{true}}) + \varepsilon} }$$

$$\text{vRMSE}_{c,i_\eta,t} = \frac{1}{N}\sum_n \text{vRMSE}_{n,c,i_\eta,t}$$

#### Interpretation

- `vRMSE ≈ 0` — the model predicts the spatial structure perfectly
- `vRMSE ≈ 1` — error comparable to the spatial variance; the model is effectively
  predicting a flat field
- For events with very small spatial variance (smooth, nearly symmetric initial
  states), vRMSE can be large even for visually reasonable predictions; this is
  expected and not a failure

#### Code

```python
vrmse = ev.vrmse()          # shape (C, neta, T)
print(f'Mean vRMSE (ε): {vrmse[0].mean():.4f}')
```

---

## 5. Spectral metrics

All spectral metrics operate via `numpy.fft.rfft2` applied over the transverse
`(nx, ny)` axes.  The FFTs and Parseval-corrected power spectra are computed once and
cached internally on the first call to any spectral method.

**Parseval correction:** `rfft2` with `axes=(2,3)` applies a full complex FFT along
`nx` (all frequencies present) and a real FFT along `ny` (only positive frequencies
returned, shape `ny//2+1`).  The one-sided `ky` frequencies (indices 1 … `ny//2−1`)
each represent a conjugate pair, so their power is doubled to restore the correct
total energy:

$$P[\text{ky=1..ny/2-1}] \;\times\!=\; 2$$

This ensures `E.sum() ≈ spatial_variance` by Parseval's theorem.

---

### 5.1 fRMSE

**Frequency-band normalized RMSE** — the test-set field is decomposed in Fourier
space and the RMSE is computed independently within three radial wavenumber bands
(low, mid, high thirds of the wavenumber range).

#### Definition

Let $F^{\text{pred}}$, $F^{\text{true}}$ be the 2-D DFTs of the transverse slice and
$\mathcal{K}_b$ be the set of `(kx, ky)` frequency bins in band $b$:

$$\text{fRMSE}_b = \sqrt{ \frac{\displaystyle\sum_{k \in \mathcal{K}_b} \bigl|\hat{f}^{\text{pred}}(k) - \hat{f}^{\text{true}}(k)\bigr|^2}{\displaystyle\sum_{k \in \mathcal{K}_b} \bigl|\hat{f}^{\text{true}}(k)\bigr|^2} }$$

averaged over events and returning shape `(C, neta, T)` per band.

The three bands are equal thirds of $[0, K_\text{max}]$ by radial wavenumber:

| Band | Range | Physical interpretation |
|---|---|---|
| `low` | $K \in [0, K_\text{max}/3)$ | Bulk geometry, bulk flow, large-scale pressure gradients |
| `mid` | $K \in [K_\text{max}/3, 2K_\text{max}/3)$ | Eccentricity-driven fluctuations, v₂ and v₃ |
| `high` | $K \in [2K_\text{max}/3, K_\text{max}]$ | Sub-nucleon granularity, ν₄ and higher |

FNO is known to over-smooth high-frequency structures.  The `high` band fRMSE will be
elevated even when the `low` band nRMSE looks acceptable, and this is invisible to
the headline nRMSE number.

#### Code

```python
frmse = ev.frmse()      # dict with keys 'low', 'mid', 'high', each (C, neta, T)

c = 0   # energy density
print(f'fRMSE low  = {frmse["low"][c].mean():.4f}')
print(f'fRMSE mid  = {frmse["mid"][c].mean():.4f}')
print(f'fRMSE high = {frmse["high"][c].mean():.4f}')

# fRMSE vs τ at one η bin
fig, ax = ev.plot_frmse_vs_tau(channel=0, ieta=0)
```

#### Diagnostics

```python
# Heat-map of high-band fRMSE over (η × τ)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.5))
im = ax.pcolormesh(ev.tau_coords, ev.eta_coords, frmse['high'][0],
                   cmap='Reds', shading='auto')
plt.colorbar(im, ax=ax, label='fRMSE high')
ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('η')
ax.set_title('High-frequency band error — ε')
plt.tight_layout()
```

---

### 5.2 MELR

**Mean Energy Log Ratio** — a wavenumber-weighted summary of the spectral energy
bias, which is positive when the model over-predicts high-frequency energy and
negative (the more common FNO pathology) when the model over-smooths.

#### Definition

For each radial shell $K$ the spectral energy is:

$$E(K) = \sum_{k \in \mathcal{S}_K} \tfrac{1}{2}|\hat{f}(k)|^2$$

where $\mathcal{S}_K = \{k : K - 0.5 \le |k| < K + 0.5\}$ (nearest-integer shell
assignment).

The per-sample MELR is a wavenumber-weighted mean of the log ratio over all shells
with positive energy:

$$\text{MELR}_n = \frac{\displaystyle\sum_{K>0} K \cdot \log\!\left(\frac{E^{\text{pred}}(K)}{E^{\text{true}}(K)}\right) \cdot \mathbf{1}[\text{valid}]}{\displaystyle\sum_{K>0} K \cdot \mathbf{1}[\text{valid}]}$$

The weighting by $K$ means that errors at high wavenumbers contribute more per unit
log-ratio than errors at low wavenumbers.  The reported MELR is the mean over events.

#### Interpretation

| Sign | Meaning |
|---|---|
| `MELR ≈ 0` | Spectral energy balance preserved across scales |
| `MELR < 0` | Model over-smooths: high-K energy suppressed (typical FNO result) |
| `MELR > 0` | Model adds artificial high-frequency noise |

The log ratio is scale-invariant, so MELR is not dominated by the large low-frequency
energy peak — a 50% error at $K=8$ contributes as much as a 50% error at $K=1$.

#### Code

```python
melr = ev.melr()    # shape (C, neta, T)

# Check for over-smoothing signature
if melr[0].mean() < -0.1:
    print('Warning: FNO is over-smoothing the energy density field.')

# Plot alongside WLR
fig, ax = ev.plot_melr_wlr_vs_tau(channel=0)
```

---

### 5.3 WLR

**Wasserstein Log Ratio** — the Wasserstein-1 (Earth mover's) distance between the
normalized radial power spectra of prediction and ground truth, treated as discrete
probability distributions over shells.

#### Definition

Normalize the per-shell energy to a probability mass function:

$$p(K) = \frac{E^{\text{pred}}(K)}{\sum_{K'} E^{\text{pred}}(K')}, \qquad q(K) = \frac{E^{\text{true}}(K)}{\sum_{K'} E^{\text{true}}(K')}$$

The one-dimensional Wasserstein-1 distance equals the L¹ norm of the difference of
cumulative distribution functions:

$$\text{WLR}_n = \sum_K \bigl|P(K) - Q(K)\bigr|, \qquad P(K) = \sum_{K' \le K} p(K')$$

The reported WLR is the mean over events.

#### Interpretation

WLR measures the *shape* of the spectral energy distribution independently of the
overall scale.  A model that correctly predicts the total energy but shifts it from
high-K to low-K shells will have zero MELR but nonzero WLR.  The two metrics together
are more informative than either alone:

| MELR | WLR | Diagnosis |
|---|---|---|
| ≈ 0 | ≈ 0 | Excellent spectral fidelity |
| < 0 | large | Systematic over-smoothing (energy shifted to low K) |
| ≈ 0 | large | Spectral shape distorted without net energy bias |
| > 0 | large | Artificial noise added at high K |

#### Code

```python
wlr = ev.wlr()      # shape (C, neta, T)
print(f'Mean WLR (ε): {wlr[0].mean():.4f}')
```

---

### 5.4 spectral_error_map

**2-D Fourier-space error map** — the log₁₀ of the mean squared error
$|\hat{f}^{\text{pred}} - \hat{f}^{\text{true}}|^2$ at each `(kx, ky)` point,
averaged over the test set.

The `kx` axis is `fftshift`-ed so that the DC component is centred.  The `ky` axis
covers the one-sided range $[0, n_y/2]$ from the real FFT.

#### Physical interpretation

This is the most detailed spectral diagnostic.  If the error map is isotropic (circular
contours), the model failure is rotationally symmetric — likely a simple smoothing.
If the map is anisotropic, the model fails at specific spatial directions, which for
non-central heavy-ion collisions maps directly to errors in specific flow harmonics.
For example, errors concentrated along the $k_x=0$ or $k_y=0$ axes indicate problems
with the dominant elliptic deformation.

#### Code

```python
# Raw arrays
kx, ky, log_err = ev.spectral_error_map(channel=0, ieta=2, itau=-1)
# kx: (nx,) fftshifted,  ky: (ny//2+1,),  log_err: (nx, ny//2+1)

# Plot
fig, ax = ev.plot_spectral_error_map(channel=0, ieta=2, itau=-1)

# Compare two τ steps side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, it in zip(axes, [0, -1]):
    kx, ky, log_err = ev.spectral_error_map(0, ieta=0, itau=it)
    im = ax.pcolormesh(ky, kx, log_err, cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label='log₁₀(|ΔF|²)')
    it_abs = ev._resolve_itau(it)
    ax.set_title(f'τ = {ev.tau_coords[it_abs]:.3f} fm/c')
plt.tight_layout()
```

---

### 5.5 energy_spectrum

**Radially-averaged 1-D power spectrum** $E(K)$ vs integer wavenumber shell $K$, for
both prediction and ground truth.

Each shell $K$ collects all `(kx, ky)` bins satisfying $K - 0.5 \le \sqrt{k_x^2+k_y^2} < K + 0.5$.
Parseval correction is applied before binning (one-sided FFT factors of 2), so that
$\sum_K E(K)$ equals the mean-field spatial variance.

#### Physical interpretation

Plotting $E(K)$ on a log-log scale reveals whether the model preserves the correct
power-law cascade.  For turbulent-like initial states, the spectrum follows a power
law $E(K) \sim K^{-\alpha}$ at intermediate scales.  A model that over-smooths will
show a deficit at high $K$ that widens as $\tau$ increases.

#### Code

```python
# At a single (η, τ) point
K, E_pred, E_true = ev.energy_spectrum(channel=0, ieta=0, itau=-1)

import matplotlib.pyplot as plt
valid = (K > 0) & (E_true > 0) & (E_pred > 0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(K[valid], E_true[valid], 'k-',  label='truth')
ax.loglog(K[valid], E_pred[valid], 'r--', label='FNO pred')
ax.set_xlabel('K'); ax.set_ylabel('E(K)')
ax.legend(); ax.grid(True, alpha=0.3, which='both')

# Multiple τ steps
fig, ax = ev.plot_energy_spectrum(channel=0, ieta=0, tau_indices=[0, 5, -1])
```

---

## 6. Temporal stability: autoregressive rollout

For a 1-step trained model (input = one τ slice, output = next τ slice), error can
compound over multiple autoregressive steps.  `ModelEval.ar_rollout` wraps any
model callable for this evaluation.

### Protocol

1. Start with the physical initial condition $f(\tau_0)$
2. Call the model to predict $f(\tau_1)$
3. Feed $f(\tau_1)$ back as input to predict $f(\tau_2)$
4. Continue for all T steps
5. Evaluate the resulting trajectory against ground truth with `ModelEval`

### API

```python
@staticmethod
ModelEval.ar_rollout(model_fn, data) -> np.ndarray
```

`model_fn` is a framework-agnostic callable that accepts a numpy array of shape
`(1, C, nx, ny, neta, 1)` and returns a numpy array of the same shape.  `data` is
the full array including the initial condition, shape `(N, C, nx, ny, neta, T+1)`.

### PyTorch example

```python
import sys, numpy as np, torch
sys.path.append('../loc_libs')
from model_eval import ModelEval

device = torch.device('cpu')
model.eval()

def torch_step(x_np):
    """x_np: (1, C, nx, ny, neta, 1) numpy → same shape numpy."""
    with torch.no_grad():
        xin = torch.FloatTensor(x_np).to(device)
        return model(xin).cpu().numpy()

# data_full: (N, C, nx, ny, neta, T+1)  — index 0 is initial condition
ar_pred = ModelEval.ar_rollout(torch_step, data_full)
# ar_pred: (N, C, nx, ny, neta, T)

ev_ar = ModelEval(
    pred       = ar_pred,
    true       = y_true_phys,
    tau_coords = tau_out_coords,
    eta_coords = eta_coords,
)
print('AR rollout nRMSE:', ev_ar.nrmse().mean())

# Compare 1-step and AR error growth on the same plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ev.tau_coords,    ev.nrmse()[0].mean(axis=0),    label='Single-pass')
ax.plot(ev_ar.tau_coords, ev_ar.nrmse()[0].mean(axis=0), label='AR rollout', ls='--')
ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('nRMSE')
ax.set_title('Error accumulation — AR vs single-pass')
ax.legend(); ax.grid(True, alpha=0.3)
```

### MLX example

```python
import mlx.core as mx
model.eval()

def mlx_step(x_np):
    xin = mx.array(x_np)
    out = model(xin)
    mx.eval(out)
    return np.array(out)

ar_pred = ModelEval.ar_rollout(mlx_step, data_full)
```

---

## 7. Summary and plotting

### 7.1 summary()

Returns a flat `dict[str, float]` with the mean of every metric over all channels,
η bins, and τ steps.  `cRMSE` is averaged over channels and τ only.

```python
s = ev.summary()
# {
#   'nRMSE':      0.0842,
#   'RMSE':       0.0840,
#   'MaxError':   0.3120,
#   'bRMSE':      0.0871,
#   'cRMSE':      0.0210,
#   'vRMSE':      0.0844,
#   'fRMSE_low':  0.0701,
#   'fRMSE_mid':  0.0855,
#   'fRMSE_high': 0.1234,   ← high-band inflation flags smoothing
#   'MELR':      -0.1520,   ← negative: FNO over-smooths
#   'WLR':        0.0872,
# }
```

### 7.2 Individual plots

Every metric has a standalone plot method that returns `(fig, ax)` or `fig`:

| Method | Description |
|---|---|
| `plot_nrmse_vs_tau(channel, fig, ax)` | nRMSE vs τ, one curve per η bin |
| `plot_frmse_vs_tau(channel, ieta, fig, ax)` | fRMSE low/mid/high vs τ |
| `plot_melr_wlr_vs_tau(channel, fig, ax)` | MELR (left axis) + WLR (right axis) vs τ |
| `plot_spectral_error_map(channel, ieta, itau, fig, ax)` | 2-D log₁₀ Fourier error |
| `plot_energy_spectrum(channel, ieta, tau_indices, fig, ax)` | Log-log E(K) overlay |

All accept optional `fig` and `ax` arguments so they can be embedded in a larger
layout:

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
ev.plot_nrmse_vs_tau(channel=0, fig=fig, ax=axes[0])
ev.plot_frmse_vs_tau(channel=0, ieta=0, fig=fig, ax=axes[1])
ev.plot_melr_wlr_vs_tau(channel=0, fig=fig, ax=axes[2])
plt.tight_layout()
```

### 7.3 plot_dashboard()

A 2×3 composite figure combining all diagnostics.  The bottom-right panel is a
rendered summary table.

```
┌────────────────┬────────────────┬─────────────────┐
│  nRMSE vs τ   │  fRMSE vs τ   │  MELR & WLR vs τ│
│  (all η bins) │ (low/mid/high) │                 │
├────────────────┼────────────────┼─────────────────┤
│  Spectral err  │ Energy spectrum│  Summary table  │
│  map (last τ)  │  log-log       │  (all metrics)  │
└────────────────┴────────────────┴─────────────────┘
```

```python
fig = ev.plot_dashboard(
    channel     = 0,           # which field component
    ieta        = 0,           # which η bin for spectral plots
    tau_indices = [0, 5, -1],  # τ steps for energy spectrum
    figsize     = (19, 11),
)
fig.savefig('dashboard.pdf', bbox_inches='tight')
```

---

## 8. Marimo notebook UI

Both metric notebooks share the controls described below.  They inherit the same
config/model loading as the existing contour notebooks (see
[README_eval.md](README_eval.md)), with the data/model loading cells unchanged.

### Config and model loading

| Control | Default | Notes |
|---|---|---|
| Config YAML | `./config_test_train.yaml` | Same as training config |
| Model directory | `./checkpoints` | Must contain checkpoint + optional `normalizer.npz` |
| Checkpoint | `best` | `best`, `last`, `checkpoint_N` |
| Epoch N | `10` | Only for `checkpoint_N` |
| Separate eval file | *(empty)* | Empty = training file + test split |
| Max events | `-1` | Limit events loaded; `-1` = all |

The MLX notebook additionally supports `pytorch_best`, `pytorch_last`,
`pytorch_checkpoint_N` for weight transfer from PyTorch checkpoints.

### Metric controls

| Control | Default | Effect |
|---|---|---|
| Channel | `0 (ε)` | Which field for per-channel plots and dashboard |
| η bin | `0` | Which η bin for spectral error map and fRMSE plot |
| τ indices for energy spectrum | `0,-1` | Start and end τ steps overlaid in log-log plot |
| Save summary to JSON | unchecked | Writes `out/metrics_eval[_mlx].json` |

### Cells produced

1. **Dashboard (2×3)** — fires on any control change
2. **nRMSE heat map** — (η × τ) for every channel
3. **fRMSE heat maps** — (η × τ) for each band, for the selected channel
4. **Scalar summary table** (markdown)
5. **Training history** — reads `metrics.json` / `metrics_mlx.json` from `odir`

---

## Appendix — complete API reference

### `ModelEval.__init__`

```python
ModelEval(
    pred          : np.ndarray,           # (N, C, nx, ny, neta, T)
    true          : np.ndarray,           # (N, C, nx, ny, neta, T)
    tau_coords    : np.ndarray | None,    # (T,)  τ values in fm/c
    eta_coords    : np.ndarray | None,    # (neta,) η values
    channel_names : list[str] | None,     # length C
    eps           : float = 1e-10,        # division guard
)
```

Attributes set after construction:

| Attribute | Type | Description |
|---|---|---|
| `N, C, nx, ny, neta, T` | int | Dimensions from `pred.shape` |
| `tau_coords` | ndarray (T,) | τ axis values |
| `eta_coords` | ndarray (neta,) | η axis values |
| `channel_names` | list[str] | Field component labels |
| `eps` | float | Division guard |

---

### Pointwise metric methods

All return `np.ndarray`.  Shapes in the table below assume the constructor was called
with a `(N, C, nx, ny, neta, T)` array.

| Method | Return shape | Formula |
|---|---|---|
| `nrmse()` | `(C, neta, T)` | mean_n sqrt(MSE_xy / RMS²_xy) per sample |
| `rmse()` | `(C, neta, T)` | sqrt(mean_{n,x,y} (pred−true)²) |
| `max_error()` | `(C, neta, T)` | max_{n,x,y} |pred−true| |
| `brmse()` | `(C, neta, T)` | RMSE restricted to ∂(x,y) boundary ring |
| `crmse()` | `(C, T)` | mean_n |Σ_xyz pred − Σ_xyz true| / |Σ_xyz true| |
| `vrmse()` | `(C, neta, T)` | mean_n sqrt(MSE_xy / Var_xy(true)) per sample |

---

### Spectral metric methods

| Method | Return | Description |
|---|---|---|
| `frmse()` | `dict{'low','mid','high'}: (C,neta,T)` | Band-normalized RMSE in low/mid/high wavenumber thirds |
| `melr()` | `(C, neta, T)` | Wavenumber-weighted mean log(E_pred/E_true) |
| `wlr()` | `(C, neta, T)` | W₁ distance between normalized spectral distributions |
| `spectral_error_map(channel, ieta, itau)` | `(kx, ky, log_err)` | 2-D log₁₀ error map in Fourier space |
| `energy_spectrum(channel, ieta, itau)` | `(K, E_pred, E_true)` | Radially-averaged 1-D power spectrum |

#### `spectral_error_map` signature

```python
ev.spectral_error_map(
    channel : int = 0,    # field component index
    ieta    : int = 0,    # η bin index
    itau    : int = -1,   # τ step index (negative = wrap from end)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
# returns: kx (nx,), ky (ny//2+1,), log10_err (nx, ny//2+1)
```

#### `energy_spectrum` signature

```python
ev.energy_spectrum(
    channel     : int = 0,
    ieta        : int = 0,
    itau        : int = -1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
# returns: K_shells (n_shells,), E_pred (n_shells,), E_true (n_shells,)
```

---

### Static helper

```python
ModelEval.ar_rollout(
    model_fn : Callable[[np.ndarray], np.ndarray],
    data     : np.ndarray,   # (N, C, nx, ny, neta, T+1), index 0 = initial condition
) -> np.ndarray              # (N, C, nx, ny, neta, T)
```

`model_fn` must accept `(1, C, nx, ny, neta, 1)` and return `(1, C, nx, ny, neta, 1)`.
The function is called `N × T` times; for large datasets consider batching
internally within `model_fn`.

---

### Summary method

```python
ev.summary() -> dict[str, float]
# Keys: nRMSE, RMSE, MaxError, bRMSE, cRMSE, vRMSE,
#       fRMSE_low, fRMSE_mid, fRMSE_high, MELR, WLR
```

---

### Plot methods

| Method | Returns | Key parameters |
|---|---|---|
| `plot_nrmse_vs_tau(channel, fig, ax, title)` | `(fig, ax)` | One curve per η bin |
| `plot_frmse_vs_tau(channel, ieta, fig, ax)` | `(fig, ax)` | Three lines (low/mid/high) |
| `plot_melr_wlr_vs_tau(channel, fig, ax)` | `(fig, ax)` | Twin-axis: MELR left, WLR right |
| `plot_spectral_error_map(channel, ieta, itau, fig, ax)` | `(fig, ax)` | Inferno pcolormesh |
| `plot_energy_spectrum(channel, ieta, tau_indices, fig, ax)` | `(fig, ax)` | Log-log overlay |
| `plot_dashboard(channel, ieta, tau_indices, figsize)` | `fig` | 2×3 composite |

All plot methods accept optional `fig` and `ax` arguments for embedding in a larger
figure (see [Section 7.2](#72-individual-plots)).

---

### Internal cached arrays

These are computed lazily and stored to avoid redundant FFTs.

| Attribute | Shape | Content |
|---|---|---|
| `_F_pred`, `_F_true` | `(N,C,nx,ny//2+1,neta,T)` complex | rfft2 of pred and true |
| `_E_pred`, `_E_true` | `(N,C,neta,T,n_shells)` | Parseval-corrected radial energy per shell |
| `_kmap` | `(nx, ny//2+1)` | Radial wavenumber at each `(kx, ky)` point |
| `_k_shells` | `(n_shells,)` | Integer shell indices 0 … K_max |

---

### Module-level helpers

```python
from model_eval import _build_kmap, _boundary_mask

kmap, shells = _build_kmap(nx=32, ny=32)
# kmap: (32, 17) — radial wavenumber; shells: (0,1,...,22)

mask = _boundary_mask(nx=32, ny=32)
# mask: (32, 32) bool — True on the four outer rows/columns
```

---

### Minimum recommended reporting

For a neural-operator surrogate publication in the heavy-ion physics context the
following set is the minimum credible evaluation:

```python
s = ev.summary()
frmse = ev.frmse()

print('Minimum reporting set:')
print(f'  nRMSE:       {s["nRMSE"]:.4f}')
print(f'  fRMSE low:   {s["fRMSE_low"]:.4f}')
print(f'  fRMSE mid:   {s["fRMSE_mid"]:.4f}')
print(f'  fRMSE high:  {s["fRMSE_high"]:.4f}')
print(f'  MELR:        {s["MELR"]:.4f}  (negative = over-smoothing)')

# AR rollout requires a 1-step model
# ev_ar = ModelEval(ar_pred, y_true_phys, ...)
# ev_ar.plot_nrmse_vs_tau()   ← error accumulation curve

# Domain gate (not yet automated — requires Cooper-Frye freeze-out):
# → compare v2, v3, <pT> from FNO freeze-out surface vs MUSIC/VISH2+1
```
