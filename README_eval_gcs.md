# FNO Evaluation — GCS-enabled

Three files make up the GCS-enabled evaluation stack:

| File | Purpose |
|---|---|
| [`eval_model/eval_gcs_clean.py`](eval_model/eval_gcs_clean.py) | **Recommended notebook** — clean, 4-cell setup + UI/plot cells |
| [`eval_model/eval_gcs_setup.py`](eval_model/eval_gcs_setup.py) | Plain-Python backend module; imported by `eval_gcs_clean.py` |
| [`eval_model/eval_gcs.py`](eval_model/eval_gcs.py) | Full self-contained notebook (all setup logic inline) |

`eval_gcs_clean.py` delegates all heavy work (config, GCS auth, data loading, model
loading, inference) to `eval_gcs_setup.run_all()` via a single setup cell, then
exposes only the visualization UI.  `eval_gcs.py` is kept for reference and contains
the identical logic inline.

Both notebooks extend [`eval_model/eval.py`](eval_model/eval.py) with support for
loading models and data from **Google Cloud Storage** and keep the same three
evaluation modes (`basic`, `cont`, `metric`).

**Key differences from `eval.py`:**

| Feature | `eval.py` | `eval_gcs_clean.py` / `eval_gcs.py` |
|---|---|---|
| Source configuration | Marimo UI text fields | YAML config file |
| Model from GCS | — | Download prefix or stream `.pt` via gcsfs |
| Data from GCS | Partial (`gs://` path in UI) | Download or stream (ROOT) |
| HDF5 from GCS | — | Download to local cache |
| Backend selection | UI dropdown | YAML (`backend: torch \| mlx \| torch_to_mlx`) |
| Marimo UI scope | All settings | **Only** eval visualization controls |

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Config file reference](#2-config-file-reference)
3. [GCS setup](#3-gcs-setup)
4. [Model access modes](#4-model-access-modes)
5. [Data access modes](#5-data-access-modes)
6. [Backends and checkpoint types](#6-backends-and-checkpoint-types)
7. [Marimo UI controls](#7-marimo-ui-controls)
8. [Output](#8-output)
9. [Examples](#9-examples)
10. [Dependencies](#10-dependencies)

---

## 1. Quick start

### All-local evaluation (no GCS)

```bash
cd eval_model/

# copy and edit the template config
cp eval_gcs_config.yaml my_run.yaml
# set model_dir, train_config, backend in my_run.yaml

EVAL_GCS_CONFIG=my_run.yaml marimo run eval_gcs_clean.py
```

The default config at `eval_model/eval_gcs_config.yaml` already uses local paths
with `model_access: local` and `data_access: local`, so running without the env var
will work if `./checkpoints/` exists.

### Evaluate a model stored on GCS

```bash
# Edit eval_gcs_config.yaml:
#   gcs_key_file:   ./my-service-account.json
#   model_access:   gcs_download
#   model_dir:      gs://my-bucket/runs/run01/checkpoints
#   data_access:    gcs_stream
#   data_format:    root

marimo run eval_gcs_clean.py
```

### Headless / non-interactive

```bash
EVAL_GCS_CONFIG=my_run.yaml marimo run eval_gcs_clean.py --headless
```

### Using `eval_gcs_setup` as a plain Python module

`eval_gcs_setup.py` can be used outside marimo for scripted evaluation:

```python
import sys; sys.path.insert(0, 'eval_model')
import eval_gcs_setup as setup

ctx = setup.run_all('eval_model/eval_gcs_config.yaml')
# ctx.model_out_phys  shape: (N, C, nx, ny, neta, T)
# ctx.y_true_phys     shape: (N, C, nx, ny, neta, T)
# ctx.eta_coords, ctx.tau_out_coords, ...
```

---

## 2. Config file reference

The config is a YAML file.  Its path is resolved in this order:

1. `EVAL_GCS_CONFIG` environment variable
2. `eval_gcs_config.yaml` in the current working directory

A fully-annotated template is provided at
[`eval_model/eval_gcs_config.yaml`](eval_model/eval_gcs_config.yaml).
The table below documents every key.

### GCS authentication

| Key | Type | Default | Description |
|---|---|---|---|
| `gcs_key_file` | str or null | `null` | Path to a GCS service-account JSON key file. Required when any `*_access` value starts with `gcs_`. |

### Model source

| Key | Type | Default | Description |
|---|---|---|---|
| `model_access` | str | `local` | `local` · `gcs_download` · `gcs_stream` |
| `model_dir` | str | `./checkpoints` | Local path **or** `gs://bucket/prefix` (for `gcs_*` modes) |
| `model_local_cache` | str | `./cache/models` | Local directory where GCS-downloaded model files are cached |
| `train_config` | str | `./checkpoints/config.yaml` | Path to the training config YAML. May be a `gs://` URI (downloaded to `model_local_cache`). |

### Data source

| Key | Type | Default | Description |
|---|---|---|---|
| `data_access` | str | `local` | `local` · `gcs_download` · `gcs_stream` |
| `data_files` | list | `[]` | Override the file list. Empty → use the paths from the training config. Must be full `gs://` URIs when using `gcs_*` modes. |
| `data_format` | str | `auto` | `auto` · `root` · `h5`. `auto` infers from the file extension. |
| `data_local_cache` | str | `./cache/data` | Local directory where GCS-downloaded data files are cached |
| `gcs_overwrite_cache` | bool | `false` | Re-download even if a size-matched local copy exists |
| `gcs_show_progress` | bool or null | `null` | `null` = auto (progress bar in TTY only), `true` = always, `false` = never |

### Eval options

| Key | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `torch` | Inference backend: `torch` · `mlx` · `torch_to_mlx` |
| `checkpoint_type` | str | `best` | See [Section 6](#6-backends-and-checkpoint-types) |
| `checkpoint_epoch` | int | `10` | Epoch number for `checkpoint_N` / `pytorch_checkpoint_N` variants |
| `nevents` | int | `-1` | Maximum events to load from the data file. `-1` = all. |
| `use_train_split` | bool | `true` | `true` → evaluate on the test split `[n_train:]`. `false` → use all loaded events. |
| `eval_mode` | str | `basic` | Default mode shown in the UI dropdown (`basic` · `cont` · `metric`) |
| `active_threshold` | float | `0.3` | ε threshold (GeV/fm³) for active-cell masking in `metric` mode; `0` = off |
| `metric_backend` | str | `numpy` | Metric computation backend: `numpy` (float64, reference) or `torch` (float32, GPU-accelerated; auto-selects CUDA → MPS → CPU). Only used in `metric` mode. See [README_eval_metric.md §3.1](README_eval_metric.md#31-choosing-a-backend). |

### Output

| Key | Type | Default | Description |
|---|---|---|---|
| `odir` | str | `./out` | Directory for optional JSON metric dumps and training history plots |

---

## 3. GCS setup

### Service account key

All GCS access uses a service-account JSON key file.  The account needs at minimum:

| Operation | Required role |
|---|---|
| Downloading / streaming data and models | Storage Object Viewer |
| Uploading results (not done by `eval_gcs.py`) | Storage Object Creator |

Set the path in the config:

```yaml
gcs_key_file: ./my-service-account.json
```

### Required Python packages

GCS support requires packages beyond the base FNO4d environment:

```bash
pip install gcsfs google-cloud-storage   # streaming + downloads
# or (if loc_libs extras are wired up):
pip install -e '../loc_libs[gcs]'
# or:
pip install -r requirements-gcs.txt
```

`gcsfs` is needed for `gcs_stream` (model `.pt` streaming and ROOT data streaming).
`google-cloud-storage` is needed for `gcs_download` (blob downloads with progress).

If the key file is set but the packages are not installed, the notebook stops with a
clear error message pointing to the install command.

---

## 4. Model access modes

### `local` (default)

`model_dir` is treated as a local filesystem path.  No GCS connection is made.

```yaml
model_access: local
model_dir: ./checkpoints
train_config: ./checkpoints/config.yaml
```

### `gcs_download`

The entire `model_dir` GCS prefix is downloaded to `model_local_cache` before any
files are read.  Subsequent runs reuse the cached files (size-matched) unless
`gcs_overwrite_cache: true`.

```yaml
model_access:      gcs_download
model_dir:         gs://my-bucket/runs/run01/checkpoints
model_local_cache: ./cache/models
gcs_key_file:      ./my-service-account.json
```

The training config (`train_config`) is downloaded separately to `model_local_cache`
if it is also a `gs://` URI.  The normalizer (`normalizer.npz`) is handled the same
way.

### `gcs_stream`

Checkpoint files are fetched from GCS on demand — no permanent local copy is made.

- **PyTorch `.pt` files** — streamed directly into `torch.load()` via a `gcsfs` file
  object.  No temporary file is written.
- **MLX safetensors / PyTorch `.pt` for MLX weight transfer** — downloaded to a
  `tempfile.NamedTemporaryFile` (deleted after the process exits).  `mlx.core.load`
  and `load_from_torch` require a local path.
- **`normalizer.npz`** — downloaded to a temporary file via `gcsfs.cat()`.

```yaml
model_access: gcs_stream
model_dir:    gs://my-bucket/runs/run01/checkpoints
gcs_key_file: ./my-service-account.json
```

Use `gcs_download` instead if you plan to run multiple evaluations against the same
checkpoint directory — downloading once is faster than repeated temp-file fetches.

---

## 5. Data access modes

### `local` (default)

Paths are resolved from `data_files` (if set) or from the training config's
`input_dir`/`input_files` (ROOT) or `hdf5_dir`/`hdf5_files` (HDF5) keys.

```yaml
data_access: local
# data_files: []   # empty = use training config paths
```

### `gcs_stream`

ROOT files are read directly from GCS without downloading, using `uproot` +
`gcsfs` via the `storage_options` argument.  The `gs://` URIs must be in
`data_files` (or already in the training config).

```yaml
data_access:  gcs_stream
data_format:  root          # must be root; h5 streaming is not supported
data_files:
  - gs://my-bucket/data/dAu_25ev_mb.root
gcs_key_file: ./my-service-account.json
```

> **HDF5 + GCS streaming is not supported.** `h5py` requires a local filesystem
> path.  Use `gcs_download` for `.h5` files.

### `gcs_download`

Data files are downloaded to `data_local_cache` before reading.  Works for both
ROOT and HDF5 formats.  Subsequent runs reuse cached files.

```yaml
data_access:      gcs_download
data_format:      auto        # auto | root | h5
data_local_cache: ./cache/data
data_files:
  - gs://my-bucket/data/dAu_25ev_mb.h5
gcs_key_file: ./my-service-account.json
```

### `data_files` override

When `data_files` is set it replaces the file list from the training config entirely.
Leave it empty (`[]`) to inherit the training config's file list.  All other
data-preparation parameters (`train_ratio`, `xyslice`, `tauslice`, `nevents`) are
always taken from the training config or from `nevents`/`use_train_split` in the
eval config.

---

## 6. Backends and checkpoint types

### Backend

| Value | FNO implementation | Weight source |
|---|---|---|
| `torch` | `neuralop.models.FNO` | PyTorch `.pt` state dict |
| `mlx` | `neuralop_mlx.models.FNO` | MLX `.safetensors` (auto-fallback to PyTorch transfer) |
| `torch_to_mlx` | `neuralop_mlx.models.FNO` | Always loads from PyTorch `.pt` via weight transfer |

### Checkpoint type

#### PyTorch (`backend: torch`)

| `checkpoint_type` | File loaded |
|---|---|
| `best` or `pytorch_best` | `best_model_state_dict.pt` |
| `last` or `pytorch_last` | `last_model.pt` |
| `checkpoint_N` or `pytorch_checkpoint_N` | `checkpoint_<epoch>/best_model_state_dict.pt` |

#### MLX (`backend: mlx` or `torch_to_mlx`)

| `checkpoint_type` | File loaded | Mechanism |
|---|---|---|
| `best` | `best_model.safetensors` | MLX native (`mx.load`) |
| `last` | `last_model.safetensors` | MLX native |
| `checkpoint_N` | `checkpoint_<epoch>/best_model.safetensors` | MLX native |
| `pytorch_best` | `best_model_state_dict.pt` | PyTorch → MLX weight transfer |
| `pytorch_last` | `last_model.pt` | PyTorch → MLX weight transfer |
| `pytorch_checkpoint_N` | `checkpoint_<epoch>/best_model_state_dict.pt` | PyTorch → MLX weight transfer |

When `checkpoint_type: best` is used with `backend: mlx` and no
`best_model.safetensors` is present, the notebook automatically falls back to
`best_model_state_dict.pt` with PyTorch weight transfer and prints a notice.

When `backend: torch_to_mlx` is used, the PyTorch `.pt` checkpoint is always
loaded regardless of `checkpoint_type` (the `pytorch_*` prefix is implied).

Set `checkpoint_epoch` to the relevant epoch number when using any `*_N` type:

```yaml
checkpoint_type:  checkpoint_N
checkpoint_epoch: 20
```

### Expected checkpoint directory layout

```
checkpoints/
├── config.yaml                       # training config (used if train_config points here)
├── normalizer.npz                    # normalizer state (loaded automatically if present)
├── best_model_state_dict.pt          # PyTorch best model
├── last_model.pt                     # PyTorch last epoch
├── best_model.safetensors            # MLX best model
├── last_model.safetensors            # MLX last epoch
└── checkpoint_20/
    ├── best_model_state_dict.pt
    └── best_model.safetensors
```

---

## 7. Marimo UI controls

All file locations, GCS auth, data format, model backend, and checkpoint selection
are read from the YAML config and are **not** exposed in the UI.

The UI contains only the mode selector and per-mode visualization controls.

### Mode selector (always visible)

| Control | Default | Description |
|---|---|---|
| Eval mode | value of `eval_mode` in config | `basic` · `cont` · `metric` |

Switching modes re-renders the control panel and the relevant plot cells.
Inference is not re-run — all three modes share the same `model_out_phys` and
`y_true_phys` arrays computed once at startup.

### `basic` mode controls

| Control | Default | Description |
|---|---|---|
| Event index | `0` | Which test-set event to plot |
| η bin index | `0` | Which η bin to display |
| τ output indices | `5,10,15,-1` | Comma-separated list; `-1` = last step |
| Channel | `0` | `0`=ε · `1`=vy · `2`=vx · `3`=vη |

Each selected (η, τ) pair produces a 2-row panel (Truth / FNO) with a shared
colour scale and physical x/y axis labels.

### `cont` mode controls

| Control | Default | Description |
|---|---|---|
| Event (slider) | `0` | Event index into the test set |
| η bin (slider) | `0` | η bin index |
| Feature / channel | `0 — ε` | Channel dropdown |
| 2nd and 3rd τ-step indices | `10,-1` | Indices of the 2nd and 3rd τ panels (1st panel is always index 0) |
| CDF ratios | `0.05,0.1,0.2,0.3,0.4,0.6` | Comma-separated CDF fractions for contour levels |
| Rescale energy: (e/τ)^(4/3) | depends on normalizer | Bjorken-scaling convention |

Produces a 1 × 3 figure with CDF contour overlays (truth in white/black, FNO in
red).  See [README_eval.md](README_eval.md) for a detailed description of the
skinnycontour approach.

### `metric` mode controls

| Control | Default | Description |
|---|---|---|
| Channel | `0 (ε)` | Field component for per-channel plots and the dashboard |
| η bin index | `0` | η bin for spectral error map and fRMSE plot |
| Active ε threshold | `0.3` | Cells below this GeV/fm³ are masked; `0` = no masking |
| τ indices for energy spectrum | `0,-1` | τ steps overlaid in the log-log spectrum plot |
| Save summary to JSON | unchecked | Writes `<odir>/metrics_eval[_mlx].json` |

The metric computation backend (`numpy` vs `torch`) is **not** a UI control — set it
via `metric_backend` in the config file.  Switch to `metric_backend: torch` to run
all metrics on the GPU (10–50× faster for N > 10 events).

See [README_eval_metric.md](README_eval_metric.md) for full documentation of
nRMSE, fRMSE, MELR, WLR, and the known failure modes (sparse-field blow-up,
cRMSE divergence, freeze-out leakage).

---

## 8. Output

### Cells produced (all modes)

| Cell | Always shown | Mode guard |
|---|---|---|
| Relative L² error vs τ (all η bins) | basic + cont | hidden in metric mode |
| Training history plot | if `<odir>/metrics[_mlx].json` exists | — |

### `basic` mode

One 2-row × n-τ figure per invocation showing truth (top) and FNO prediction
(bottom) for the selected event, η bin, and τ indices.  Colour scale is shared
between truth and prediction within each column.

### `cont` mode

One 1 × 3 skinnycontour figure.  Each column is one τ snapshot.  Truth is the
background image; CDF contours are drawn in white/black over truth and in red
for the FNO prediction.

### `metric` mode

| Output | Description |
|---|---|
| 2×3 dashboard | nRMSE vs τ · fRMSE vs τ · MELR+WLR · spectral error map · energy spectrum · summary table |
| nRMSE heat map | (η × τ) for every channel |
| fRMSE heat maps | (η × τ) for low/mid/high bands, selected channel |
| Scalar summary table | All `summary()` keys rendered as markdown |
| `metrics_eval[_mlx].json` | Written if **Save summary to JSON** is checked |

---

## 9. Examples

### Evaluate best PyTorch checkpoint (all local)

```yaml
# eval_gcs_config.yaml
model_access:    local
model_dir:       ./checkpoints
train_config:    ./checkpoints/config.yaml
data_access:     local
backend:         torch
checkpoint_type: best
eval_mode:       basic
```

```bash
cd eval_model/
marimo run eval_gcs_clean.py
```

### Evaluate MLX model with PyTorch weight transfer (local)

```yaml
model_access:    local
model_dir:       ./checkpoints
train_config:    ./checkpoints/config.yaml
data_access:     local
backend:         torch_to_mlx
checkpoint_type: best
eval_mode:       cont
```

### Stream ROOT data directly from GCS, load model locally

```yaml
gcs_key_file:    ./my-service-account.json
model_access:    local
model_dir:       ./checkpoints
train_config:    ./checkpoints/config.yaml
data_access:     gcs_stream
data_format:     root
data_files:
  - gs://my-bucket/data/dAu_50ev_mb.root
nevents:         50
use_train_split: false   # evaluate all 50 events
eval_mode:       metric
```

### Download model and HDF5 data from GCS, then evaluate

```yaml
gcs_key_file:      ./my-service-account.json
model_access:      gcs_download
model_dir:         gs://my-bucket/runs/run01/checkpoints
train_config:      gs://my-bucket/runs/run01/checkpoints/config.yaml
model_local_cache: ./cache/models
data_access:       gcs_download
data_format:       h5
data_files:
  - gs://my-bucket/data/dAu_250ev_mb.h5
data_local_cache:  ./cache/data
gcs_overwrite_cache: false
backend:           torch
checkpoint_type:   best
eval_mode:         metric
metric_backend:    torch   # GPU-accelerated metrics (10-50× faster for large N)
```

```bash
EVAL_GCS_CONFIG=gcs_eval_h5.yaml marimo run eval_gcs_clean.py
```

### Stream model `.pt` from GCS (no local copy)

```yaml
gcs_key_file:    ./my-service-account.json
model_access:    gcs_stream
model_dir:       gs://my-bucket/runs/run01/checkpoints
train_config:    gs://my-bucket/runs/run01/checkpoints/config.yaml
data_access:     local
backend:         torch
checkpoint_type: pytorch_best
eval_mode:       basic
```

The `.pt` state dict is loaded directly into `torch.load()` via a `gcsfs` file
object — no file is written to disk.

### Inspect a periodic checkpoint at epoch 30

```yaml
checkpoint_type:  checkpoint_N
checkpoint_epoch: 30
backend:          torch
```

### Headless batch run with metric output

```bash
EVAL_GCS_CONFIG=my_run.yaml marimo run eval_gcs_clean.py --headless
# Plots are rendered inline; JSON is written if save_json_ui would be checked.
# For fully scripted evaluation without marimo, use eval_gcs_setup directly (see Quick Start).
```

---

## 10. Dependencies

### Base (always required)

| Package | Used for |
|---|---|
| `marimo` | notebook runtime |
| `numpy` | array operations |
| `matplotlib` | all plots |
| `torch` | PyTorch backend (if `backend: torch` or `torch_to_mlx`) |
| `mlx` | MLX backend (if `backend: mlx` or `torch_to_mlx`) |
| `neuralop` / `neuralop_mlx` | FNO model implementations |
| `h5py` | HDF5 data reading |
| `uproot`, `awkward` | ROOT data reading |
| `yaml` | config parsing |

### GCS (optional, required for `gcs_*` access modes)

| Package | Used for |
|---|---|
| `gcsfs` | `gcs_stream` — fsspec filesystem for `gs://` URIs |
| `google-cloud-storage` | `gcs_download` — blob download API |
| `tqdm` | download progress bars (optional; silently skipped if absent) |

Install:

```bash
pip install gcsfs google-cloud-storage tqdm
# or:
pip install -r requirements-gcs.txt
```

### Local libraries (`loc_libs/`)

| Module | Used for |
|---|---|
| `read_3d_root` | ROOT data reader (uproot-based) |
| `read_3d_hdf5` | HDF5 data reader |
| `norm_utils.Normalizer` | normalizer save/load and transforms |
| `gcs_utils` | GCS download helpers (`download_gcs_file`, `download_gcs_dir`) |
| `skinnycontour` | `cont` mode contour plots |
| `model_eval.ModelEval` | `metric` mode full metric suite (numpy or torch backend, controlled by `metric_backend`) |
| `parse_training_log` | training history plot |

---

## See also

- [README_eval.md](README_eval.md) — original per-script evaluation notebooks
- [README_eval_metric.md](README_eval_metric.md) — full documentation of the metric suite
- [README_train_gcs.md](README_train_gcs.md) — training with GCS data sources
- [`eval_model/eval_gcs_config.yaml`](eval_model/eval_gcs_config.yaml) — annotated config template
- [`eval_model/eval_gcs_clean.py`](eval_model/eval_gcs_clean.py) — clean marimo notebook (recommended)
- [`eval_model/eval_gcs_setup.py`](eval_model/eval_gcs_setup.py) — plain-Python setup backend
- [`eval_model/eval_gcs.py`](eval_model/eval_gcs.py) — self-contained marimo notebook (all logic inline)
