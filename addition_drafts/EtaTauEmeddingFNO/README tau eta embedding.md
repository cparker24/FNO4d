# τ / η Positional Embedding for FNO Hydro Surrogates

Sinusoidal positional embeddings for the proper-time (τ) and spacetime-rapidity (η)
axes of 3+1D viscous relativistic hydrodynamics fields in Milne coordinates.  
Drop-in wrapper around `neuralop.models.FNO` — no internals of the library are modified.

-----

## The problem

Standard normalisation (e.g. `NormFn3d.etau75_eta_3d`) removes the dominant
amplitude trends from the energy density field before training:

- **Bjorken cooling** `e(τ) ∝ τ^{−4/3}` is folded out by the `e^{3/4} · τ` rescaling.
- **Forward-rapidity suppression** `e(η) ∝ 1/cosh(η)` is folded out by the `cosh(η)^α`
  or data-driven `ref(η, τ)` rescaling.

This is correct — it makes the network input O(1) everywhere, which is necessary for
stable training. But it has a side effect: **the normalised field carries no intrinsic
cue about where in (τ, η) space a given slice sits**. A spectral convolution kernel
learned at midrapidity η = 0 and early time τ_init is then applied identically at
forward rapidity η = 4 and late time τ_freeze, even though the underlying physics
(cooling rate, longitudinal flow gradients, freeze-out proximity) is qualitatively
different there.

Without a positional embedding the FNO is forced to learn a single kernel that must
serve all (τ, η) regimes simultaneously. This is fundamentally harder than learning
a kernel that knows which regime it is currently operating in.

-----

## Why τ and η are analogous problems

Both axes have the same structure: a known, large systematic variation has been
removed by normalisation, leaving the network without a positional reference.

|Axis|Removed trend                                            |Dynamic range before normalisation|What the network loses              |
|----|---------------------------------------------------------|----------------------------------|------------------------------------|
|τ   |Bjorken cooling `e ~ τ^{−4/3}`                           |orders of magnitude               |sense of “how late in the evolution”|
|η   |Geometric suppression `~ 1/cosh η` + finite beam rapidity|orders of magnitude               |sense of “how far from midrapidity” |

Both problems call for the same fix: inject a bounded, smooth sinusoidal coordinate
signal so the network can condition its spectral weights on position in (τ, η) space.

-----

## What the embedding does

A 2-channel sinusoidal encoding is computed from the physical coordinate values:

```
τ-embedding:  [sin(π · τ / τ_max),  cos(π · τ / τ_max)]     shape (T, 2)
η-embedding:  [sin(π · η / η_max),  cos(π · η / η_max)]     shape (Z, 2)
```

Both channels are bounded in `[−1, 1]` and vary smoothly with the coordinate.
The **sin** channel for η is antisymmetric (odd in η), encoding the sign of
the rapidity displacement from midrapidity.  The **cos** channel is symmetric
(even in η), encoding the magnitude.  Together they distinguish forward from
backward rapidity, which a magnitude-only encoding cannot.

The embedding is injected in two complementary places:

### 1. Input concatenation (at the lifting layer)

The four embedding channels — two for τ, two for η — are broadcast across the
full spatial grid and concatenated to the physical input channels before the
FNO lifting layer:

```
x          : (B, C_in,   X, Y, Z, T)   physical fields
τ-channels : (B, 2,      X, Y, Z, T)   broadcast from (T, 2)
η-channels : (B, 2,      X, Y, Z, T)   broadcast from (Z, 2)
─────────────────────────────────────────────────────────────
x_aug      : (B, C_in+4, X, Y, Z, T)   fed to FNO
```

The FNO lifting layer therefore receives `in_channels + 4` input channels.
Every subsequent spectral convolution operates on a hidden representation
that was formed with positional awareness from the very first linear projection.

### 2. Per-block additive bias (at every FNO layer)

Positional information injected only at the input fades as activations pass
through successive channel-mixing operations. To keep the signal alive at
every depth, a small learned `nn.Linear(2 → C)` projects each embedding into
a per-slice channel bias that is added to the hidden representation after
each FNO layer:

```
_TauBias:  (T, 2) → Linear → (T, C) → reshape → (1, C, 1, 1, 1, T)
_EtaBias:  (Z, 2) → Linear → (Z, C) → reshape → (1, C, 1, 1, Z, 1)
```

Both biases are added by **forward hooks** registered on the FNO’s internal
layers. This requires no modification of the `neuralop` library.

**Parameter count:** `n_layers × 4 × C` additional real parameters — completely
negligible compared to the Tucker-factorised spectral weight tensors.

-----

## Installation

```bash
pip install neuraloperator torch
```

The embedding module (`fno_tau_eta_embedding.py`) has no further dependencies.

-----

## Quick start

```python
import torch
from fno_tau_eta_embedding import FNOWithEmbedding, make_tau_tensor, make_eta_tensor

# ── Grid metadata matching your NormFn3d normalisation ──────────────────────
TAU_MIN, DTAU, N_TAU = 0.4,  0.076, 128   # fm/c
ETA_MIN, DETA, N_ETA = -3.5, 7.0/31, 32   # η_s

device = torch.device("cuda")

tau = make_tau_tensor(TAU_MIN, DTAU, N_TAU, device=device)  # (128,)  fm/c
eta = make_eta_tensor(ETA_MIN, DETA, N_ETA, device=device)  # (32,)   η_s

# ── Build model ─────────────────────────────────────────────────────────────
model = FNOWithEmbedding(
    in_channels=4,            # ed, vx, vy, vz
    out_channels=4,
    hidden_channels=32,
    n_modes=(8, 8, 4, 6),    # (mx, my, m_eta, m_tau) Fourier mode cutoffs
    n_layers=6,
    tau_max=float(tau[-1]),   # fm/c  — must match your training grid
    eta_max=abs(float(eta[-1])),
    factorization="tucker",   # passed verbatim to neuralop.FNO
    rank=0.05,
).to(device)

# ── Forward pass ─────────────────────────────────────────────────────────────
x = torch.randn(2, 4, 64, 64, N_ETA, N_TAU, device=device)  # (B, C, X, Y, Z, T)
y = model(x, tau, eta)   # (2, 4, 64, 64, 32, 128)
```

-----

## Full training example

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from neuralop.losses import LpLoss, H1Loss

from fno_tau_eta_embedding import FNOWithEmbedding, make_tau_tensor, make_eta_tensor
from NormFn3d import NormFn3d

# ── Normalisation setup ──────────────────────────────────────────────────────
# These values must match however you ran NormFn3d on your training data.
TAU_MIN, DTAU = 0.4, 0.076   # fm/c
ETA_MIN, DETA = -3.5, 7.0 / 31
N_TAU, N_ETA  = 128, 32
TAU_MAX       = TAU_MIN + (N_TAU - 1) * DTAU   # ~ 10.1 fm/c
ETA_MAX       = abs(ETA_MIN + (N_ETA - 1) * DETA)

# ── Coordinate tensors (fixed throughout training) ───────────────────────────
device = torch.device("cuda")
tau = make_tau_tensor(TAU_MIN, DTAU, N_TAU, device=device)
eta = make_eta_tensor(ETA_MIN, DETA, N_ETA, device=device)

# ── Model ────────────────────────────────────────────────────────────────────
model = FNOWithEmbedding(
    in_channels=4,
    out_channels=4,
    hidden_channels=32,
    n_modes=(8, 8, 4, 6),
    n_layers=6,
    tau_max=TAU_MAX,
    eta_max=ETA_MAX,
    factorization="tucker",
    rank=0.05,
).to(device)

# ── Loss and optimiser ───────────────────────────────────────────────────────
# LpLoss is the standard relative L2 used for FNO training.
# H1Loss additionally penalises gradient errors, encouraging smooth fields —
# recommended when patching or when forward-rapidity accuracy matters.
lp_loss = LpLoss(d=4, p=2, reduce_dims=(0, 1))   # 4D: (X, Y, Z, T)
h1_loss = H1Loss(d=4, reduce_dims=(0, 1))

optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# ── Training loop ────────────────────────────────────────────────────────────
# train_loader yields (x_norm, y_norm) pairs, both already normalised by
# NormFn3d and shaped (B, C, X, Y, N_ETA, N_TAU).

for epoch in range(200):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        pred = model(x_batch, tau, eta)

        # Combine Lp and H1: H1 weight can be increased during training warmup
        loss = lp_loss(pred, y_batch) + 0.1 * h1_loss(pred, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}  loss={loss.item():.4f}")

# ── Inference and denormalisation ────────────────────────────────────────────
model.eval()
with torch.no_grad():
    pred_norm = model(x_test.to(device), tau, eta)

# Recover physical energy density from normalised prediction.
# The inverse call must use the same ref / shape_ref saved during training.
pred_physical = NormFn3d.inv_etau75_eta_3d(
    pred_norm.cpu().numpy(),
    ref=ref_etau75_eta,     # (N_ETA, N_TAU) saved when building training data
    tau_min=TAU_MIN,
    dtau=DTAU,
)
```

-----

## Coordinate conventions

The code follows the same conventions as `NormFn3d.py`:

|Symbol|Axis in tensor|Physical meaning   |Typical range|
|------|--------------|-------------------|-------------|
|τ     |dim 5 (T)     |Bjorken proper time|0.4 – 12 fm/c|
|η_s   |dim 4 (Z)     |Spacetime rapidity |−4 … +4      |
|x, y  |dim 2–3 (X, Y)|Transverse plane   |±10 fm       |

`make_tau_tensor` and `make_eta_tensor` reconstruct the physical coordinate
arrays from the same `(min, step, n)` metadata used by `NormFn3d`, so the
embedding always matches the normalisation exactly.

-----

## Choosing mode cutoffs

The FNO `n_modes` tuple has **four** entries because τ and η are treated as
spatial axes by the plain `neuralop.FNO`:

```
n_modes = (mx, my, m_eta, m_tau)
```

Reasonable starting values for a 64×64×32×128 grid:

```python
n_modes = (8, 8, 4, 6)   # retains low-frequency global structure
```

- `mx, my` should capture transverse eccentricity and flow anisotropy (harmonics
  v₂, v₃ require at least modes 2–3 in transverse).
- `m_eta` is typically small (4–8) because the rapidity profile is smooth after
  normalisation.
- `m_tau` can be slightly larger (6–12) to resolve the τ-evolution dynamics,
  but keep it well below `T // 2` to avoid aliasing.

-----

## Relation to other project modules

|Module                         |Role                                                                                                                                          |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
|`NormFn3d.py`                  |Normalises raw hydro fields before the FNO sees them. Must be applied first.                                                                  |
|`fno_tau_eta_embedding.py`     |Wraps `neuralop.FNO` with (τ, η) positional embeddings.                                                                                       |
|`causal_loglo_4d_v2.py`        |Advanced architecture (LOGLO + causal) with τ-embedding built in. Use this for production training; the plain FNO wrapper is a clean baseline.|
|`combined_causal_loglo_loss.py`|Lp + H1 + spectral loss schedule; compatible with both architectures.                                                                         |

-----

## References

- **FNO**: Li et al., *Fourier Neural Operator for Parametric Partial Differential
  Equations*, ICLR 2021. [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)
- **NeuralOperator library**: Kossaifi et al., *Neural Operators for Accelerating
  Scientific Simulations and Design*, 2024.
  [arXiv:2512.01421](https://arxiv.org/abs/2512.01421)
- **MG-TFNO**: Kossaifi et al., *Multi-Grid Tensorized Fourier Neural Operator*,
  2023. [arXiv:2310.00120](https://arxiv.org/abs/2310.00120)
- **LOGLO-FNO**: Kalimuthu et al., *Local-Global Fourier Neural Operator*, TMLR 2025.
  [arXiv:2504.04260](https://arxiv.org/abs/2504.04260)
- **Bjorken flow**: Bjorken, *Hydrodynamic model for ultrarelativistic nucleus–nucleus
  collisions*, Phys. Rev. D 27 (1983) 140.