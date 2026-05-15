# FNO Evaluation Scripts

Four [marimo](https://marimo.io) notebooks for evaluating trained FNO models on test data.
See the main README for environment setup and dependencies.

| Script | Framework | Visualisation | Checkpoint formats |
|---|---|---|---|
| `train/eval_model_test_torch.py` | PyTorch | `imshow` (truth / FNO rows) | `.pt` files written by the PyTorch training scripts |
| `train/eval_model_test_mlx.py` | MLX | `imshow` (truth / FNO rows) | MLX `.safetensors` (native) or PyTorch `.pt` (via weight transfer) |
| `train/eval_model_test_cont_torch.py` | PyTorch | skinnycontour (CDF contour lines) | `.pt` files written by the PyTorch training scripts |
| `train/eval_model_test_cont_mlx.py` | MLX | skinnycontour (CDF contour lines) | MLX `.safetensors` (native) or PyTorch `.pt` (via weight transfer) |

The `_cont_*` scripts use the **skinnycontour** approach from `apply.py`: truth is shown as a
background image with white/black CDF-level contour lines overlaid, and the FNO prediction is
overlaid as red contour lines at the same levels.  Three τ snapshots per η bin are shown
side-by-side.

---

## Installing marimo (optional)

Marimo is not installed by the base `install.sh` so that headless or batch
setups are not forced to carry the dependency.  If you want the interactive
notebook UI, run the helper script **after** activating the FNO4d environment:

```bash
source venv/bin/activate          # or: conda activate <env>
bash install_marimo.sh
```

The script is idempotent: it skips the install if marimo is already present and
prints the resolved binary path on success.

---

## Quick start

Run as an interactive notebook (recommended):

```bash
cd train/

# imshow-based scripts
marimo run eval_model_test_torch.py
marimo run eval_model_test_mlx.py

# skinnycontour scripts
marimo run eval_model_test_cont_torch.py
marimo run eval_model_test_cont_mlx.py
```

Run non-interactively (executes all cells top-to-bottom, shows plots inline):

```bash
marimo run eval_model_test_cont_torch.py --headless
```

Edit the source while keeping live reactivity (developer mode):

```bash
marimo edit eval_model_test_cont_torch.py
```

---

## UI controls

All four notebooks share the core set of controls described below.
The `_cont_*` scripts add four additional controls (see
[Skinnycontour controls](#skinnycontour-controls-cont_-scripts-only))
and replace the per-τ `τ output indices` field with a fixed three-panel layout.

### Config YAML
**Default:** `./config_test_train.yaml`

Path to the training config file used when training the model being evaluated.
All data-preparation parameters (`train_ratio`, `xyslice`, `tauslice`, `input_dir`,
`input_files`, `hidden_channels`, `n_modes`) are read from this file so that the
eval pipeline is guaranteed to be consistent with training.

### Model directory
**Default:** `./checkpoints`

Directory that contains the saved checkpoint files and, optionally, `normalizer.npz`.
Must match the `model_dir` used during training.

```
checkpoints/
├── best_model_state_dict.pt      # PyTorch best model (neuralop trainer)
├── last_model.pt                 # PyTorch last epoch  (save_last_checkpoint)
├── best_model.safetensors        # MLX best model
├── last_model.safetensors        # MLX last epoch
├── last_optimizer.safetensors    # MLX optimizer state
├── last_state.json               # MLX epoch metadata
├── normalizer.npz                # normalizer state (if normalization was used)
└── checkpoint_10/                # periodic checkpoint at epoch 10
    ├── best_model_state_dict.pt  # PyTorch
    └── best_model.safetensors    # MLX
```

### Checkpoint type

#### PyTorch scripts (`eval_model_test_torch.py`, `eval_model_test_cont_torch.py`)

| Value | File loaded | When to use |
|---|---|---|
| `best` | `best_model_state_dict.pt` | Best validation-loss model saved by neuralop `Trainer` |
| `last` | `last_model.pt` | Final epoch saved by `save_last_checkpoint` |
| `checkpoint_N` | `checkpoint_N/best_model_state_dict.pt` | Periodic intermediate checkpoint (set **Epoch N** below) |

#### MLX scripts (`eval_model_test_mlx.py`, `eval_model_test_cont_mlx.py`)

| Value | File loaded | Mechanism |
|---|---|---|
| `best` | `best_model.safetensors` | MLX native (`mx.load`) |
| `last` | `last_model.safetensors` | MLX native |
| `checkpoint_N` | `checkpoint_N/best_model.safetensors` | MLX native |
| `pytorch_best` | `best_model_state_dict.pt` | PyTorch → MLX weight transfer |
| `pytorch_last` | `last_model.pt` | PyTorch → MLX weight transfer |
| `pytorch_checkpoint_N` | `checkpoint_N/best_model_state_dict.pt` | PyTorch → MLX weight transfer |

The `pytorch_*` options use `neuralop_mlx.weight_transfer.load_from_torch`, which
handles all layout differences between the two frameworks (complex spectral weights,
Conv1d kernels, soft-gating scalars, etc.) automatically.

### Epoch N
**Default:** `10`

The epoch number used when `checkpoint_N` or `pytorch_checkpoint_N` is selected.
Ignored for all other checkpoint types.

### Separate eval file
**Default:** *(empty)*

- **Empty** — load the same data file(s) listed in the config YAML, apply the same
  `train_ratio` split, and evaluate on the **test portion only** (events `[n_train:]`).
- **Path** — load this file instead and use **all events** in it as the eval set
  (no train/test split applied).

```
# example: evaluate on a different dataset
/path/to/new_run/dAu_100ev_central.root
```

Only the first file is used if a comma-separated list is given.

### Max events
**Default:** `-1` (all events in the file)

Limit the number of events read from the ROOT file. Useful for quick checks:

```
nevents: 20   # fast smoke-test
nevents: -1   # full dataset
```

### Event index for plots
**Default:** `0`

Which event (0-based index into the test set) to use for the contour plots.
The L² error plot always averages over **all** test events regardless of this setting.

### τ output indices  *(imshow scripts only)*
**Default:** `5,10,15,-1`

Comma-separated list of τ output indices to visualise. Each index refers to a
column in the model output array (0-based, where index 0 = second data time step
= τ_min + Δτ). Negative indices wrap around: `-1` selects the last τ step.

Physical τ for output index `t`: **τ = τ_min + (t + 1) × Δτ** fm/c.

```
5,10,15,-1        # four snapshots including the last
0                 # earliest predicted step only
-1                # final predicted step only
0,5,10,15,20,25   # six snapshots
```

### Channel
**Default:** `0`

Which field component to plot.

| Index | Field | Units |
|---|---|---|
| 0 | Energy density *e* | GeV/fm³ |
| 1 | *v*x (transverse velocity) | dimensionless |
| 2 | *v*y (transverse velocity) | dimensionless |
| 3 | *v*η (longitudinal velocity) | dimensionless |

---

## Skinnycontour controls (`_cont_*` scripts only)

### η bin
**Default:** `-1` (all η bins)

Which η bin to plot. Set to `-1` to loop over every η bin and produce one 3-panel
figure per bin. Set to a specific index (e.g. `2`) to produce a single figure for
that bin — useful for quick checks.

### 3 τ-step indices for contour panels
**Default:** `5,10,-1`

Exactly three comma-separated output-step indices selecting the three side-by-side
τ snapshots. The same index convention applies as for the imshow scripts: 0-based,
`-1` = last non-zero time step (the function searches backwards from the end to skip
freeze-out steps).

```
5,10,-1      # early, mid, final
0,10,20      # three explicit steps
```

### CDF ratios for contour levels
**Default:** `0.05,0.1,0.2,0.3,0.4,0.6`

Comma-separated list of cumulative-energy fractions used to determine the contour
level values.  For each ratio *r*, the contour is drawn at the field value that
accounts for the top-*r* fraction of the total integrated field (energy density, or
whichever channel is selected).  Smaller values → tighter (hotter) contours.

### Rescale energy: (e/τ)^(4/3)
**Default:** checked

When enabled, energy-density values are rescaled by `(e / τ)^(4/3)` before computing
contours, matching the Bjorken-scaling convention used in `apply.py`.  This highlights
deviations from ideal hydrodynamic scaling and is the standard display mode.  Uncheck
to plot raw energy density.

---

## Output

### Imshow plots (`eval_model_test_torch.py`, `eval_model_test_mlx.py`)

One figure is produced per η bin. Each figure has:
- **Rows:** Truth (top) / FNO prediction (bottom)
- **Columns:** one per selected τ index
- **Axes:** physical x and y coordinates in fm
- **Colour scale:** shared between truth and prediction within each (η, τ) panel
- **Colour map:** `hot`

Title format: `η = {value}  (event {N})`

### Skinnycontour plots (`eval_model_test_cont_torch.py`, `eval_model_test_cont_mlx.py`)

One figure is produced per η bin (or for the single selected bin).  Each figure has:
- **Layout:** 1 row × 3 columns, one column per selected τ-step index
- **Background:** truth field rendered with `imshow` (viridis colourmap)
- **White/black contour lines:** truth field at the selected CDF levels
- **Red contour lines:** FNO prediction at the same CDF levels, allowing direct comparison
- **Annotations:** CDF percentile labels (left side) and the predicted-minus-truth fractional
  deviation at each level (right side, red)
- **τ label:** physical τ in fm/c shown in the upper-right corner of each panel

Title format: `{channel} Energy/Vy/Vx/Vz: η={value} Event {N}. Contours: Truth: black. FNO: red.`

### L² error plot

Relative L² error as a function of physical τ, with one curve per η bin,
averaged over the full test set:

```
error(η, τ) = sqrt( mean_events( mean_xy( (pred - true)² ) ) /
                    mean_events( mean_xy( true² ) ) )
```

### Training metrics plot

If `metrics.json` (PyTorch) or `metrics_mlx.json` (MLX) exists in `odir`
(from the training config), the training/validation loss curves are shown.

---

## Physical coordinate conventions

All grid parameters are read directly from the ROOT data file.

| Quantity | Formula | Typical range |
|---|---|---|
| x | x_min + i × Δx | −15 … +15 fm |
| y | y_min + j × Δy | −15 … +15 fm |
| η (space-time rapidity) | η_min + k × Δη | −2 … +2 |
| τ (initial condition) | τ_min | ≈ 0.6 – 0.7 fm/c |
| τ (output step t) | τ_min + (t + 1) × Δτ | τ_min … τ_freeze |

The spatial slicing applied during training (`xyslice`, `tauslice` in the YAML)
drops the last x, y, and η bin respectively and is reproduced identically here.

---

## Normalisation

If `normalizer.npz` is present in the model directory it is loaded automatically
and the **inverse transform** is applied to both the model output and the ground
truth before plotting, so all displayed values are in physical units (GeV/fm³ for
energy density).

If no normalizer file is found, raw (unnormalised) values are plotted with a
warning printed to the cell output.

Supported normalisation schemes (set at training time via the `normalization.name`
YAML key):

| Name | Description |
|---|---|
| `none` | Identity — no normalisation |
| `etau75_3d` | *e*^0.75 × τ (Bjorken tau scaling) |
| `etau75_eta_3d` | Above ÷ data-driven per-(η, τ) reference |
| `etau75_eta_shape_3d` | Above with shape-only η correction |
| `cosh_etau75_3d` | (*e* × cosh(η)^α)^0.75 × τ (analytic η correction) |

---

## Examples

### Evaluate the best PyTorch model on the training-data test split (imshow)

```bash
cd train/
marimo run eval_model_test_torch.py
# In the UI:
#   Config YAML:    ./config_test_train.yaml
#   Model dir:      ./checkpoints
#   Checkpoint:     best
#   Eval file:      (empty)
#   τ indices:      5,10,15,-1
#   Channel:        0
```

### Skinnycontour evaluation with the PyTorch model

```bash
marimo run eval_model_test_cont_torch.py
# In the UI:
#   Config YAML:    ./config_test_train.yaml
#   Model dir:      ./checkpoints
#   Checkpoint:     best
#   Contour τ idx:  5,10,-1
#   CDF ratios:     0.05,0.1,0.2,0.3,0.4,0.6
#   Rescale energy: checked
#   η bin:          -1   (all bins)
```

### Skinnycontour evaluation with the MLX model (native checkpoint)

```bash
marimo run eval_model_test_cont_mlx.py
# In the UI:
#   Checkpoint:     best
#   Model dir:      ./checkpoints
```

### Evaluate the last MLX checkpoint on a separate file

```bash
marimo run eval_model_test_mlx.py
# In the UI:
#   Checkpoint:     last
#   Eval file:      /data/dAu_test_50ev.root
#   Max events:     50
```

### Load a PyTorch checkpoint into the MLX model (weight transfer)

```bash
marimo run eval_model_test_cont_mlx.py
# In the UI:
#   Checkpoint:     pytorch_best
#   Model dir:      ./checkpoints
```

### Inspect a periodic checkpoint at epoch 20

```bash
# PyTorch (imshow)
marimo run eval_model_test_torch.py
# Checkpoint: checkpoint_N,  Epoch N: 20

# PyTorch (skinnycontour)
marimo run eval_model_test_cont_torch.py
# Checkpoint: checkpoint_N,  Epoch N: 20

# MLX native
marimo run eval_model_test_mlx.py
# Checkpoint: checkpoint_N,  Epoch N: 20

# MLX loading from the PyTorch checkpoint at epoch 20
marimo run eval_model_test_cont_mlx.py
# Checkpoint: pytorch_checkpoint_N,  Epoch N: 20
```

### Plot a single η bin at three specific τ steps (skinnycontour)

```
3 τ-step indices:  5,10,-1
η bin:             2
Channel:           0
Rescale energy:    checked
```

One 3-panel figure is produced for η bin 2 only.

### Plot only the final τ step for a specific event (imshow)

```
τ indices:   -1
Event index: 3
Channel:     0
```

A single column of plots is produced (Truth / FNO) for every η bin at the last
predicted τ step.

### Non-interactive batch run

Marimo does not yet have a headless batch mode that accepts UI overrides from the
command line. For scripted evaluation, call the underlying logic directly:

```python
#!/usr/bin/env python
import sys
sys.path.append('../loc_libs')

import yaml, numpy as np, torch
from pathlib import Path
from read_3d_root import read_3d_data
from norm_utils import Normalizer
from neuralop.models import FNO

cfg_path  = './config_test_train.yaml'
model_dir = Path('./checkpoints')
ckpt_path = model_dir / 'best_model_state_dict.pt'

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

def _s(v): return slice(None) if v is None else slice(*v)
xyslice  = _s(cfg.get('xyslice',  [None, -1]))
etaslice = _s(cfg.get('tauslice', [None, -1]))

raw = read_3d_data(f"{cfg['input_dir']}/{cfg['input_files'][0]}", nevents=-1)
arr = raw['arr'][:, :, xyslice, xyslice, etaslice, :]

n_train   = int(arr.shape[0] * cfg.get('train_ratio', 0.8))
data_test = arr[n_train:]

norm_path  = model_dir / 'normalizer.npz'
normalizer = Normalizer.load(str(norm_path)) if norm_path.exists() else None
data_norm  = normalizer.transform(data_test) if normalizer else data_test

device = torch.device('cpu')
state  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
model  = FNO(in_channels=4, out_channels=4,
             n_modes=cfg['n_modes'],
             hidden_channels=cfg['hidden_channels'],
             projection_channel_ratio=2).to(device)
model.load_state_dict(state)
model.eval()

preds = []
with torch.no_grad():
    for i in range(data_norm.shape[0]):
        x0  = data_norm[i, :, :, :, :, :1]
        T   = data_norm.shape[-1] - 1
        x   = np.repeat(x0, T, axis=-1)
        out = model(torch.FloatTensor(x).unsqueeze(0)).squeeze(0).cpu().numpy()
        preds.append(out)
model_out = np.stack(preds)   # (N, 4, nx, ny, neta, T)

if normalizer:
    x0_all    = data_norm[:, :, :, :, :, :1]
    full      = np.concatenate([x0_all, model_out], axis=-1)
    pred_phys = normalizer.inverse(full)[:, :, :, :, :, 1:]
    true_phys = normalizer.inverse(data_norm)[:, :, :, :, :, 1:]
else:
    pred_phys = model_out
    true_phys = data_norm[:, :, :, :, :, 1:]

print('pred shape:', pred_phys.shape)
print('true shape:', true_phys.shape)
# → save, analyse, plot as needed
```
