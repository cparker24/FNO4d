# neuralop_mlx — FNO on Apple Silicon (MLX)

MLX port of the 4D Fourier Neural Operator from `neuraloperator-4d-2.0.0`.  
Runs on Apple Silicon Metal GPU via [MLX](https://github.com/ml-explore/mlx).  
The original PyTorch code in `train/test_train.py` is untouched.

---

## Why this port exists

PyTorch's MPS backend does not support the complex-domain FFT operations that
FNO requires (`rfftn`, `irfftn`, `fftshift`/`ifftshift`). MLX 0.16+ provides
these natively and dispatches them to Metal, giving full GPU acceleration on
M-series chips.

---

## Installation

### 1. Create or reuse a conda environment with MLX

```bash
conda create -n fno_env_mlx python=3.13
conda activate fno_env_mlx
pip install mlx>=0.16          # Metal GPU backend
pip install numpy uproot psutil matplotlib  # data pipeline deps
```

### 2. Install the package in editable mode

```bash
cd /path/to/FNO2Mlx
pip install -e .
```

**Important:** run `pip install -e .` from the **repo root** (`FNO2Mlx/`), not
from inside the `neuralop_mlx/` sub-directory. The `pyproject.toml` at the repo
root tells setuptools where to find the package. Installing from `./neuralop_mlx`
instead produces an empty editable mapping and `import neuralop_mlx` will fail
with `ModuleNotFoundError`.

This makes `import neuralop_mlx` available anywhere in the environment.

### 3. Verify

```bash
python -m pytest tests/ -v
# Expected: 57 passed
```

---

## Quick-start: training script

`train/test_train_mlx.py` is the MLX equivalent of `train/test_train.py`.
Edit the top-of-file parameters to match your setup:

```python
input_dir   = '/path/to/FNO_Hydro_Data/flat'
input_files = ['dAu_25ev_mb.root']
n_epochs    = 50
batch_size  = 1
hidden_channels = 16
n_modes     = [32, 32, 16, 16]   # 4D: [nx, ny, nz, nt]
lr          = 8e-3
weight_decay= 1e-4
```

Then run:

```bash
conda activate fno_env_mlx
python train/test_train_mlx.py
```

Training logs are written to `training_mlx.log` and `log_mlx.txt`.
The best checkpoint is saved to `best_model.safetensors`.

---

## API reference

### Model

```python
from neuralop_mlx.models import FNO
# or
from neuralop_mlx import FNO

model = FNO(
    n_modes         = [32, 32, 16, 16],  # one entry per spatial dim
    in_channels     = 4,
    out_channels    = 4,
    hidden_channels = 16,
    n_layers        = 4,                 # number of FNO blocks
    lifting_channel_ratio   = 2,         # lifting MLP width = ratio * hidden
    projection_channel_ratio= 2,         # projection MLP width = ratio * out
    positional_embedding    = "grid",    # "grid" or None
)

import mlx.core as mx
x   = mx.array(...)     # shape (batch, in_channels, d0, d1, d2, d3)
out = model(x)          # shape (batch, out_channels, d0, d1, d2, d3)
mx.eval(out)            # MLX is lazy; materialise before reading values
```

`n_modes` length determines dimensionality: 2-element → 2D, 4-element → 4D, etc.

### Weight transfer from a trained PyTorch model

`load_from_torch` copies all weights from a trained PyTorch FNO into an MLX
FNO that was constructed with the same architecture. Both models must share
the same `n_modes`, `in_channels`, `out_channels`, `hidden_channels`,
`n_layers`, and `projection_channel_ratio`.

```python
from neuralop_mlx import load_from_torch

# From a live PyTorch model object
mlx_model = load_from_torch(pt_model, mlx_model)

# From a checkpoint saved with torch.save(model.state_dict(), path)
mlx_model = load_from_torch("best_model.pth", mlx_model)

# From a wrapped checkpoint dict {"model_state_dict": state_dict, ...}
mlx_model = load_from_torch("checkpoint.pt", mlx_model)
```

The function handles all layout differences automatically:

| PyTorch parameter | MLX parameter | Transform applied |
|---|---|---|
| `convs[i].weight.tensor` (complex64) | `convs[i].weight_r` + `weight_i` (float32) | split real/imag |
| `convs[i].weight.core` + `weight.factors.factor_N` (Tucker/TFNO) | `convs[i].weight_r` + `weight_i` (float32) | `tl.tucker_to_tensor` then split real/imag |
| `convs[i].bias` shape `(C,1,…,1)` | `convs[i].bias` shape `(1,C,1,…,1)` | reshape |
| `fno_skips[i].conv.weight` shape `(out,in,1)` | `fno_skips[i].linear.weight` shape `(out,in)` | squeeze last dim |
| `channel_mlp[i].fcs[j].weight` shape `(out,in,1)` | `channel_mlps[i].fcs[j].weight` shape `(out,in)` | squeeze last dim |
| `channel_mlp_skips[i].weight` shape `(1,C,1,…,1)` | `mlp_skips[i].alpha` shape `(C,)` | flatten |
| `lifting.fcs[j].weight` shape `(out,in,1)` | `lifting.fcs[j].weight` shape `(out,in)` | squeeze last dim |

`load_from_torch` returns the same `mlx_model` object with weights updated
in place and calls `mx.eval()` so arrays are immediately materialised.

### Losses

```python
from neuralop_mlx import LpLoss, H1Loss

l2   = LpLoss(d=4, p=2)              # relative L2, last 4 spatial dims
h1   = H1Loss(d=4, periodic_in_x=True, periodic_in_y=True,
              periodic_in_z=True, periodic_in_t=True)

loss = h1(pred, target)   # both shape (batch, channels, d0, d1, d2, d3)
```

`d` must match the number of spatial dimensions (max 4).

### Optimizer

```python
from neuralop_mlx import AdamW

optimizer = AdamW(lr=8e-3, weight_decay=1e-4)
# betas and eps can also be set: AdamW(lr=..., betas=(0.9, 0.999), eps=1e-8)
```

This is a thin wrapper around `mlx.optimizers.AdamW`. The `params` argument
(present in the PyTorch version) is accepted but ignored — MLX optimizers
receive parameters at update time, not at construction.

### Trainer

```python
from neuralop_mlx import Trainer
import mlx.optimizers as optim_mlx

scheduler = optim_mlx.cosine_decay(lr, T_max=30)

trainer = Trainer(
    model         = model,
    n_epochs      = 50,
    device        = 'gpu',       # 'gpu' targets Metal; 'cpu' for CPU fallback
    eval_interval = 2,
    verbose       = True,
)

trainer.train(
    train_loader  = train_loader,   # PyTorch DataLoader returning dicts {x, y}
    test_loaders  = {100: test_loader},
    optimizer     = optimizer,
    scheduler     = scheduler,
    training_loss = h1,
    eval_losses   = {'h1': h1, 'l2': l2},
    save_dir      = './ckpt',
    save_best     = '100_l2',       # key from eval_losses to track for best model
)
```

#### Checkpoint save / load (MLX native format)

```python
trainer.checkpoint(save_dir='./ckpt', name='my_model')
# writes ./ckpt/my_model.safetensors

trainer.load_checkpoint(save_dir='./ckpt', name='my_model')
```

Checkpoints are stored in safetensors format via `mx.save_safetensors` /
`mx.load`. To instead load a checkpoint from the PyTorch training run, use
`load_from_torch` (see above).

### DataLoader compatibility

The Trainer accepts PyTorch `DataLoader` objects. Batches can be:
- `dict` with keys `"x"` and `"y"` (used in `test_train_mlx.py`)
- `(x, y)` tuples

Values can be numpy arrays or `mx.array`; the Trainer converts numpy to
`mx.array` automatically.

---

## Migrating from the PyTorch version

| PyTorch (`neuralop`)           | MLX (`neuralop_mlx`)                   | Notes |
|-------------------------------|----------------------------------------|-------|
| `from neuralop.models import FNO` | `from neuralop_mlx.models import FNO` | Same constructor signature |
| `from neuralop.training.trainer import Trainer` | `from neuralop_mlx import Trainer` | Same `train()` signature |
| `from neuralop.losses import LpLoss, H1Loss` | `from neuralop_mlx import LpLoss, H1Loss` | Same call signature |
| `torch.optim.AdamW(model.parameters(), lr=…)` | `AdamW(lr=…)` | Params not passed at construction |
| `out = model(x)` | `out = model(x); mx.eval(out)` | MLX is lazy: must call `mx.eval()` |
| `loss.backward(); opt.step()` | handled by `Trainer` internally | Uses `mlx.nn.value_and_grad` |
| `torch.mps.empty_cache()` | not needed | MLX manages Metal memory automatically |
| `torch.FloatTensor(arr)` | `mx.array(arr.astype(np.float32))` | |
| `model.state_dict()` / `torch.save` | `trainer.checkpoint(...)` | MLX saves safetensors |
| loading a `.pt` / `.pth` checkpoint | `load_from_torch(path, mlx_model)` | automatic layout conversion |

### Constructor arguments present in PyTorch FNO but not in MLX FNO

These PyTorch-only features are **not implemented** (all default to their
original default values, i.e. dense weights):

| Argument | PyTorch default | Status in MLX |
|----------|----------------|---------------|
| `factorization` | `None` | Not a constructor arg in MLX; TFNO checkpoints load via `load_from_torch` (see Known gaps §1) |
| `rank` | `1.0` | Not applicable |
| `fixed_rank_modes` | `False` | Not applicable |
| `max_n_modes` | `None` | Not implemented |
| `conv_module` | `SpectralConv` | Not swappable |
| `fno_block_precision` | `"full"` | Not implemented |
| `stabilizer` | `None` | Not implemented |
| `output_scaling_factor` | `None` | Not implemented |

If your PyTorch script passes any of these, remove them when switching to the
MLX version — the MLX constructor will raise `TypeError`.

---

## Tested configurations

| Scenario | Test file | Status |
|----------|-----------|--------|
| 4D forward pass, arbitrary batch/spatial sizes | `test_fno_model_mlx.py` | Passing |
| 4D gradient flow (no NaN) | `test_fno_model_mlx.py` | Passing |
| 2D / 3D / 4D forward pass shape | `test_fno_model_mlx.py` | Passing |
| 4D training loop (loss decreases, checkpoint round-trip) | `test_training_mlx.py` | Passing |
| End-to-end training on real ROOT data (dAu 25 GeV, 4D hydro) | `train/test_train_mlx.py` | Runs to completion; loss decreases; physics not validated |
| Per-layer shape and NaN checks (2D and 4D) | `test_layers_mlx.py` | Passing |
| SpectralConv 2D numerical parity vs PyTorch (max_diff < 1e-3) | `test_compare_torch_mlx.py` | Passing |
| Full FNO 2D output shape parity vs PyTorch | `test_compare_torch_mlx.py` | Passing |
| Gradient norm parity 2D vs PyTorch (within 20%) | `test_compare_torch_mlx.py` | Passing |
| Weight transfer from live PyTorch model (2D, 3D, 4D) | `test_weight_transfer.py` | Passing |
| Weight transfer from `.pth` checkpoint (bare and wrapped) | `test_weight_transfer.py` | Passing |
| Weight values match per-layer after transfer | `test_weight_transfer.py` | Passing |
| File/key error handling in `load_from_torch` | `test_weight_transfer.py` | Passing |
| TFNO Tucker weights → MLX inference, 2D/3D/4D parity (max diff < 1e-3) | `test_weight_transfer.py` | Passing |
| TFNO Tucker weights at very low rank (0.1) → MLX inference parity | `test_weight_transfer.py` | Passing |
| TFNO Tucker weights → MLX gradient flow (no NaN) | `test_weight_transfer.py` | Passing |
| TFNO Tucker checkpoint file round-trip | `test_weight_transfer.py` | Passing |

---

## Known gaps and open issues

### 1. TFNO inference is supported; TFNO training is not

MLX can load a pre-trained Tucker-factorized TFNO checkpoint and run inference
with outputs numerically identical to the original model (max diff < 1e-3 over
float32 FFT chains). What it cannot do is **train** with Tucker/CP/TT
factorized weights, because no MLX equivalent of `tltorch.FactorizedTensor`
exists.

**Why PyTorch MPS cannot help here either:** PyTorch's MPS backend has no
working `complex64` support — `torch.einsum` with complex tensors crashes at
the Metal gather layer (PyTorch issues [#90019][mps-c64],
[#119088][mps-c64-2], [#143140][mps-einsum]). Tucker contractions in
`tltorch` go through `tl.einsum` on complex weights, so TFNO is also broken on
MPS. MLX avoids this entirely by splitting weights into real/imaginary `float32`
pairs at the storage level.

**Loading a TFNO checkpoint for inference:**

```python
from neuralop_mlx import FNO, load_from_torch

# Construct a standard dense MLX FNO with the same architecture as the TFNO
mlx_model = FNO(
    n_modes=[32, 32, 16, 16], in_channels=4, out_channels=4,
    hidden_channels=64, n_layers=4, projection_channel_ratio=2,
)

# load_from_torch detects Tucker keys automatically and reconstructs
# the dense weight via tl.tucker_to_tensor before assigning weight_r/weight_i
load_from_torch("tfno_checkpoint.pth", mlx_model)

# Inference output is numerically identical to the TFNO forward pass
out = mlx_model(x)
```

`tensorly` must be installed (`pip install tensorly`) for the Tucker
reconstruction. It is already present in any environment that has `tltorch`.

**What is and isn't preserved after loading:**

| Property | Preserved? |
|----------|-----------|
| Output values (forward-pass parity, max diff < 1e-3) | ✅ Yes |
| Parameter count / memory footprint | ❌ No — MLX stores full dense weights |
| Tucker structure for continued training | ❌ No — training updates dense weights |

The compression benefit of Tucker factorization is realised during PyTorch
training. Once loaded into MLX, the model runs at full (dense) memory
footprint. For pure inference on Apple Silicon this is fine; for continued
training with Tucker regularisation, the original PyTorch TFNO must be used.

[mps-c64]: https://github.com/pytorch/pytorch/issues/90019
[mps-c64-2]: https://github.com/pytorch/pytorch/issues/119088
[mps-einsum]: https://github.com/pytorch/pytorch/issues/143140

### 2. LR scheduler is not called by the Trainer

The `scheduler` argument to `trainer.train()` is accepted but not stepped.
The cosine decay schedule in `test_train_mlx.py` is wired up at optimizer
construction time (`mlx.optimizers.cosine_decay`), which MLX handles
internally. If you need explicit per-epoch stepping of a custom scheduler,
call it yourself outside the Trainer.

### 3. No mixed-precision (float16) path

`Trainer(mixed_precision=True)` is accepted but does nothing. MLX's
`mx.float16` does not support complex numbers, so the spectral contraction
runs in float32 throughout. No float16 speed-up is available.

### 4. No wandb or distributed training

`Trainer(wandb_log=True)` and `Trainer(use_distributed=True)` print a warning
and are otherwise ignored.

### 5. H1Loss periodic assumption

`H1Loss` uses `mx.roll` (periodic finite differences) by default for all
spatial axes. Non-periodic boundaries use a replicated-boundary approximation
that differs slightly from the PyTorch H1Loss non-periodic implementation.

### 6. MLX 0.31.2 `at[].add` multi-dim stride bug (workaround in place)

In MLX 0.31.2, `array.at[s1, s2, slice1, slice2].add(val)` computes wrong
source strides when two or more non-trivial slice dimensions are present.
`SpectralConvND` works around this by using `mx.pad` instead of a scatter
operation. If a future MLX release fixes the bug, the `mx.pad` approach
remains correct but could be replaced with a direct scatter for clarity.

### 7. `test_train_mlx.py` not run in CI

The end-to-end training script requires ROOT data files at
`/Users/du8478/FNO_Hydro_Data/flat/`. It is not part of the automated test
suite.

Training has been run to completion on real data (dAu 25 GeV minimum-bias,
4D hydrodynamic fields). The training loop runs without errors and the loss
decreases over epochs. **No physics validity check has been performed yet** —
the model output has not been compared to reference hydrodynamic solutions or
checked for conservation law compliance.

---

## Implementation notes

### FNOBlocks activation order

The MLX FNOBlocks forward exactly matches the PyTorch
`forward_with_postactivation` logic:

- Both skip connections (`fno_skip` and `mlp_skip`) are computed from the
  block's **original input** `x` before any operations in that layer.
- GELU is applied after the spectral conv + FNO skip, and again after the
  channel MLP + MLP skip, **but only for non-final layers**. The last layer
  returns without any trailing activation.

### Complex weight storage

PyTorch stores SpectralConv weights as a single `complex64` tensor
(`DenseTensor`). MLX's AdamW uses `g * g` (complex multiply) rather than
`|g|²` for the variance estimate, which produces NaN gradients with complex
weights. The MLX implementation stores `weight_r` and `weight_i` as separate
`float32` arrays so that standard real-valued optimizers work correctly.

---

## File layout

```
FNO2Mlx/
├── pyproject.toml              # project root — install from here
├── neuralop_mlx/               # installable MLX package
│   ├── __init__.py             # exports FNO, Trainer, LpLoss, H1Loss, AdamW, load_from_torch
│   ├── models/fno.py           # FNO model
│   ├── layers/
│   │   ├── spectral_convolution.py   # SpectralConvND (core FFT layer)
│   │   ├── fno_block.py              # FNOBlocks
│   │   ├── channel_mlp.py            # ChannelMLP
│   │   ├── embeddings.py             # GridEmbeddingND
│   │   └── skip_connections.py       # linear / soft-gating / identity skips
│   ├── training/
│   │   ├── trainer.py          # Trainer
│   │   ├── losses.py           # LpLoss, H1Loss
│   │   └── adamw.py            # AdamW wrapper
│   ├── weight_transfer.py      # load_from_torch
│   └── utils.py                # count_model_params
├── train/
│   ├── test_train.py           # original PyTorch script (unchanged)
│   └── test_train_mlx.py       # MLX training script
└── tests/
    ├── test_layers_mlx.py      # per-layer unit tests
    ├── test_fno_model_mlx.py   # full FNO shape/gradient tests
    ├── test_training_mlx.py    # Trainer + loss unit tests
    ├── test_compare_torch_mlx.py  # numerical PyTorch vs MLX comparison
    └── test_weight_transfer.py    # load_from_torch tests (2D/3D/4D, file/live)
```
