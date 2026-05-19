# train_generic — Unified FNO Training Script

## Change log

### `min_save_epoch` — moved from monkey-patch to native trainer parameter

**Problem (bug).**  The previous implementation used a monkey-patch
(`patch_trainer_min_save_epoch` in `loc_libs/training_utils.py`) that blocked
the `trainer.checkpoint()` call before epoch `min_save_epoch`.  However, the
`best_metric_value` tracker inside `trainer.train()` was a plain local variable
that was updated from epoch 0 regardless.  This meant the running best could be
set to a low value during the early unstable epochs, so once the guard lifted
the very first eligible save might never trigger (nothing ever beat the
already-recorded early minimum).

**Fix.**  `min_save_epoch` is now a first-class parameter of
`neuralop/training/trainer.py` `Trainer.train()`.  The condition that updates
`best_metric_value` **and** calls `checkpoint()` is gated together:

```python
# neuraloperator-4d-2.0.0/neuralop/training/trainer.py  (train method)

# Before
if eval_metrics[save_best] < best_metric_value:
    best_metric_value = eval_metrics[save_best]
    self.checkpoint(save_dir)

# After
if epoch >= min_save_epoch and eval_metrics[save_best] < best_metric_value:
    best_metric_value = eval_metrics[save_best]
    self.checkpoint(save_dir)
```

The new `min_save_epoch` parameter (default `0`, meaning no restriction) was
added to `Trainer.train()`:

```python
# neuraloperator-4d-2.0.0/neuralop/training/trainer.py  (train signature)

# Before
def train(self, ..., save_best: int = None, save_dir: Union[str, Path] = "./ckpt", ...):

# After
def train(self, ..., save_best: int = None, min_save_epoch: int = 0,
          save_dir: Union[str, Path] = "./ckpt", ...):
```

**All training scripts** now pass the config value directly to `trainer.train()`:

```python
# All training scripts (train_h5.py, train.py, resume_train_h5.py,
#   train_torch/*, train_gcs/*, train/*)

# Before — monkey-patch applied before trainer.train()
if min_save_epoch is not None and min_save_epoch > 0:
    patch_trainer_min_save_epoch(trainer, Trainer, min_save_epoch)

trainer.train(...,
              save_best=save_best_key,
              ...)

# After — passed directly; or 0 converts None (key absent from config) to the default
trainer.train(...,
              save_best=save_best_key,
              min_save_epoch=min_save_epoch or 0,
              ...)
```

**`patch_trainer_min_save_epoch` has been removed** from
`loc_libs/training_utils.py` entirely.  There is no longer any risk of two
independent guards with different thresholds being active at the same time.

**Config usage is unchanged** — set `min_save_epoch` in your YAML as before:

```yaml
min_save_epoch: 50   # do not save best model before epoch 50
```

Omitting the key (or setting it to `null`) is equivalent to `0` — best-model
saving is active from the first evaluation epoch.

Three training scripts cover all supported input formats:

| Script | Purpose | Input format | Config |
|--------|---------|-------------|--------|
| `train.py` | Fresh training | ROOT files (`.root`) — local or GCS | `config.yaml` |
| `train_h5.py` | Fresh training | Pre-converted HDF5 files (`.h5`) | `config_h5.yaml` |
| `resume_train_h5.py` | Resume from checkpoint | Pre-converted HDF5 files (`.h5`) | `config_h5_resume.yaml` |

## Quick start

```bash
cd train_generic

# ROOT-based (local or GCS)
python train.py --config config.yaml

# HDF5-based (multiple files, lazy or eager)
python train_h5.py --config config_h5.yaml

# Resume HDF5 training from last checkpoint
python resume_train_h5.py --config config_h5_resume.yaml
```

## train.py — ROOT input

### Mode selection

Two keys at the top of `config.yaml` steer the run:

```yaml
backend:     torch        # torch | mlx
data_source: local        # local | gcs_download | gcs_stream
```

| `backend` | `data_source`  | Equivalent old script                         |
|-----------|----------------|-----------------------------------------------|
| `torch`   | `local`        | `train/test_train_yaml_norm.py`               |
| `mlx`     | `local`        | `train/test_train_mlx_yaml_norm.py`           |
| `torch`   | `gcs_download` | `train_gcs/test_train_yaml_norm_gcs_download.py` |
| `mlx`     | `gcs_download` | `train_gcs/test_train_mlx_yaml_norm_gcs_download.py` |
| `torch`   | `gcs_stream`   | `train_gcs/test_train_yaml_norm_gcs_stream.py` |
| `mlx`     | `gcs_stream`   | `train_gcs/test_train_mlx_yaml_norm_gcs_stream.py` |

## train_h5.py — HDF5 input

`train_h5.py` reads pre-converted `.h5` files produced by
`convert_root_to_hdf5()` (see `loc_libs/read_3d_hdf5.py`).  Multiple files
are concatenated transparently — they must share the same grid dimensions
(`nFeatures`, `nx`, `ny`, `neta`, `ntau`).

### Why HDF5?

ROOT stores branch data in baskets.  Without per-event basket flushing, uproot
must decompress the entire basket even for a small slice, causing a ≈ 2× memory
spike.  HDF5 with per-event chunks gives true random-access I/O: only the
requested events are decompressed.

### Converting ROOT → HDF5 (one-time)

```python
from read_3d_hdf5 import convert_root_to_hdf5

convert_root_to_hdf5('dAu_250ev_mb.root', 'dAu_250ev_mb.h5',
                     nevents=-1, chunk_events=1)
```

### Mode selection

```yaml
backend:     torch          # torch | mlx
data_source: local          # local | gcs_download
```

`gcs_stream` is **not** supported for HDF5 because `h5py` requires local file
paths.  When `data_source: gcs_download`, the `.h5` files are downloaded to
`local_data_dir` first (re-runs skip the download if the file size matches),
then read locally — identical caching behaviour to `train.py`.

### Loading strategy (`lazy`)

| `lazy` | `dat['arr']` type | Memory during training | I/O pattern |
|--------|-------------------|------------------------|-------------|
| `true` | `MultiH5Array` | ≈ one batch at a time | per-batch disk reads |
| `false` | `numpy.ndarray` | full dataset in RAM | single up-front load |

`lazy: true` is recommended for large datasets where RAM is the bottleneck.
`lazy: false` is faster per-epoch once the data is loaded.

### Normalisation in lazy mode

| Normalizer | Fitting cost in lazy mode |
|-----------|--------------------------|
| `none` | zero — identity transform |
| `etau75_3d`, `cosh_etau75_3d` | zero — analytic, applied per-event |
| `etau75_eta_3d`, `etau75_eta_shape_3d` | one materialisation pass over the training split to compute the reference array; result is discarded and normalisation is applied per-event thereafter |

---

## Configuration reference — `config.yaml` (ROOT)

All parameters live in `config.yaml`.  Keys that are unused in the active mode
are silently ignored (e.g. `gcs_key_file` is ignored when `data_source: local`).

### Mode

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `backend` | `torch` \| `mlx` | `torch` | Deep learning framework |
| `data_source` | `local` \| `gcs_download` \| `gcs_stream` | `local` | Where data is read from |

### Output

| Key | Default | Description |
|-----|---------|-------------|
| `odir` | `./out` | Directory for logs, metrics JSON, and training-curve SVG |
| `model_dir` | `./checkpoints` | Directory for saved model weights |

### Input data

| Key | Default | Description |
|-----|---------|-------------|
| `input_dir` | — | **[local only]** Directory prepended to each entry in `input_files` |
| `input_files` | — | List of filenames (`local`) or full `gs://` URIs (`gcs_*`) |
| `nevents_in` | `-1` | Events to load per file; `-1` loads all |

### GCS authentication

Required when `data_source` is `gcs_download` or `gcs_stream`.

| Key | Description |
|-----|-------------|
| `gcs_key_file` | Path to a GCS service-account JSON key file |

The service account needs at minimum the **Storage Object Viewer** role.
Add **Storage Object Creator** (or **Storage Object User**) to also upload
outputs after training.

### GCS download cache

Only used when `data_source: gcs_download`.

| Key | Default | Description |
|-----|---------|-------------|
| `local_data_dir` | `./data` | Directory where downloaded `.root` files are cached |
| `overwrite_cache` | `false` | `true` forces re-download even if file size matches |
| `gcs_show_progress` | `null` | `null` = auto-detect, `true` = always show bar, `false` = suppress |

Re-runs skip the download automatically when the local file size matches the
remote blob size.

### Post-training GCS upload

Optional; applies to `gcs_download` and `gcs_stream`.

| Key | Default | Description |
|-----|---------|-------------|
| `gcs_output_prefix` | `null` | If set, `odir` is uploaded to `{gcs_output_prefix}/logs` after training |

Remove the key or set it to `null` to skip the upload.

### Normalisation

| Key | Default | Description |
|-----|---------|-------------|
| `normalization.name` | `none` | Normaliser to apply; see options below |
| `normalization.alpha` | `1.0` | Exponent for `cosh_etau75_3d` only |

Available normalisers: `none`, `etau75_3d`, `etau75_eta_3d`,
`etau75_eta_shape_3d`, `cosh_etau75_3d`.

The normaliser is fit on the training split only, applied to both splits, and
saved to `{model_dir}/normalizer.npz` for use at inference time.

### Data preparation

| Key | Default | Description |
|-----|---------|-------------|
| `train_ratio` | `0.8` | Fraction of events used for training |
| `xyslice` | `[null, -1]` | Spatial slice applied after loading (`slice(None, -1)` drops last bin) |
| `tauslice` | `[null, -1]` | Tau slice applied after loading |

### Model

| Key | Default | Description |
|-----|---------|-------------|
| `hidden_channels` | `16` | FNO hidden dimension |
| `n_modes` | `[32, 32, 16, 16]` | Fourier modes along `[x, y, eta, tau]` |

### Training

| Key | Default | Description |
|-----|---------|-------------|
| `n_epochs` | — | Total training epochs |
| `batch_size` | — | Training batch size |
| `eval_interval` | `2` | Run evaluation every N epochs |
| `test_batch_sizes` | `[10]` | Eval batch size per test resolution |
| `checkpoint_freq` | `null` | Save a checkpoint every N epochs; `null` disables |
| `min_save_epoch` | `null` | Suppress best-model and checkpoint saves before this epoch |
| `save_last` | `true` | Save final epoch state to `model_dir/last_model.*` |

### Optimizer

| Key | Default | Description |
|-----|---------|-------------|
| `lr` | — | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |

The learning-rate schedule is cosine annealing with `T_max = 30` for both
backends.

### Logging

| Key | Default | Description |
|-----|---------|-------------|
| `log_directly` | `true` | Mirror console output to `training.log` (torch) or `training_mlx.log` (mlx) |

## Outputs

All outputs land in `odir` and `model_dir` as configured.

| File | Backend | Description |
|------|---------|-------------|
| `training.log` | torch | Full run log |
| `training_mlx.log` | mlx | Full run log |
| `metrics.json` | torch | Per-epoch loss and metric history |
| `metrics_mlx.json` | mlx | Per-epoch loss and metric history |
| `training_curves.svg` | torch | Loss/metric plot |
| `training_curves_mlx.svg` | mlx | Loss/metric plot |
| `best_model.*` | both | Best checkpoint (lowest validation L2) |
| `last_model.*` | both | Final epoch checkpoint (when `save_last: true`) |
| `normalizer.npz` | both | Normaliser state for inference |

## Dependencies

### Base (always required)

```bash
pip install torch neuralop pyyaml
# or for MLX backend:
pip install mlx neuralop-mlx pyyaml
```

The `loc_libs/` directory is imported from `../loc_libs` relative to this
script.  It must be present at that path (it is in the standard repo layout).

### GCS (only for `gcs_download` or `gcs_stream`)

```bash
pip install -e '../loc_libs[gcs]'
# or:
pip install -r ../requirements-gcs.txt
```

## Examples

### Local, PyTorch

```yaml
backend:     torch
data_source: local
input_dir:   /data/flat
input_files:
  - dAu_25ev_mb.root
```

### Local, MLX

```yaml
backend:     mlx
data_source: local
input_dir:   /data/flat
input_files:
  - dAu_25ev_mb.root
```

### GCS download cache, PyTorch

```yaml
backend:       torch
data_source:   gcs_download
gcs_key_file:  ./fnotest-wayne-gcs.json
local_data_dir: ../data
input_files:
  - gs://my-bucket/data/dAu_25ev_mb.root
```

### GCS streaming, MLX

```yaml
backend:      mlx
data_source:  gcs_stream
gcs_key_file: ./fnotest-wayne-gcs.json
input_files:
  - gs://my-bucket/data/dAu_25ev_mb.root
```

---

## Configuration reference — `config_h5.yaml` (HDF5)

All keys shared with `config.yaml` (model, training, optimizer, logging,
normalisation, data preparation) behave identically.  The table below lists
only the keys that differ or are new.

### Input data

| Key | Default | Description |
|-----|---------|-------------|
| `hdf5_dir` | `.` | Directory prepended to each entry in `hdf5_files` |
| `hdf5_files` | — | List of `.h5` filenames; all must share grid dimensions |
| `nevents_in` | `-1` | Events to load across all files combined; `-1` = all |
| `start_event` | `0` | First global event index (useful for held-out test sets) |

### Loading strategy

| Key | Default | Description |
|-----|---------|-------------|
| `lazy` | `true` | `true` = `MultiH5Array` (on-demand reads); `false` = full ndarray |

### GCS output upload (optional)

HDF5 input files are always local.  Outputs (`odir`) can still be uploaded
after training.

| Key | Default | Description |
|-----|---------|-------------|
| `gcs_key_file` | — | Service-account JSON key (only needed for upload) |
| `gcs_output_prefix` | `null` | Upload `odir` to `{prefix}/logs`; `null` skips |
| `gcs_show_progress` | `null` | `null` = auto, `true` = always, `false` = suppress |

## Examples — HDF5

### Single file, PyTorch, lazy

```yaml
backend:    torch
hdf5_dir:   /data/flat
hdf5_files:
  - dAu_250ev_mb.h5
lazy: true
```

### Multiple files, MLX, lazy

```yaml
backend:    mlx
hdf5_dir:   /data/flat
hdf5_files:
  - dAu_250ev_mb.h5
  - dAu_500ev_mb.h5
lazy: true
nevents_in: -1
```

### Multiple files, PyTorch, eager (full load)

```yaml
backend:    torch
hdf5_dir:   /data/flat
hdf5_files:
  - dAu_250ev_mb.h5
  - dAu_500ev_mb.h5
lazy:        false
nevents_in:  2000
train_ratio: 0.8
```

### Multiple files, GCS download, PyTorch, lazy

```yaml
backend:        torch
data_source:    gcs_download
gcs_key_file:   ./fnotest-wayne-gcs.json
local_data_dir: ../data
hdf5_files:
  - gs://my-bucket/data/dAu_250ev_mb.h5
  - gs://my-bucket/data/dAu_500ev_mb.h5
lazy: true
```

### Subset of events (e.g. skip first 500 for held-out test)

```yaml
lazy:        true
start_event: 500
nevents_in:  -1   # all events from index 500 onward
```

---

## resume_train_h5.py — Resume from checkpoint

`resume_train_h5.py` picks up where a `train_h5.py` run left off.  It shares
the identical data pipeline and model construction with `train_h5.py`; the
only difference is that model weights (and, where available, optimizer state)
are loaded from a saved checkpoint before training begins.

The log file is opened in **append mode** so the prior training history is
preserved.

### Resume scenarios

**1. Same machine, same backend (most common)**

Set `total_epochs` to the final target epoch count and point
`resume_model_dir` at the original `model_dir`:

```yaml
backend:             mlx      # or torch
resume_model_source: local
resume_model_dir:    ./checkpoints
checkpoint_type:     last
total_epochs:        200      # must exceed the last saved epoch
```

**2. Resume from a periodic checkpoint**

```yaml
resume_model_source: local
resume_model_dir:    ./checkpoints
checkpoint_type:     periodic
checkpoint_epoch:    50       # loads ./checkpoints/checkpoint_50/
total_epochs:        200
```

**3. Resume from GCS (download checkpoint, train locally)**

```yaml
resume_model_source: gcs_download
resume_model_dir:    gs://my-bucket/runs/run01/checkpoints
local_model_cache:   ./cache/checkpoints
gcs_key_file:        ./my-service-account.json
checkpoint_type:     last
total_epochs:        200
```

Only the files needed for the selected `checkpoint_type` are downloaded:
- `last` — downloads the full checkpoint directory (contains `last_model.*`).
- `periodic` — downloads only the `checkpoint_N/` subdirectory.

**4. Cross-backend: torch → mlx weight transfer**

Train on a CUDA machine with `backend: torch`, then continue on an Apple
Silicon Mac with `backend: mlx`:

```yaml
backend:             mlx
resume_backend:      torch
resume_model_source: local    # or gcs_download
resume_model_dir:    ./checkpoints
checkpoint_type:     last     # only 'last' is supported cross-backend
total_epochs:        200
```

Weights are transferred via `neuralop_mlx.weight_transfer.load_from_torch`.
Optimizer state is **not** transferred (incompatible across frameworks); the
AdamW optimizer restarts from the configured `lr`.

### Checkpoint optimizer state availability

| Backend | `checkpoint_type` | Optimizer restored? |
|---------|-------------------|---------------------|
| torch | `last` | yes — from `last_model.pt` |
| torch | `periodic` | no — neuralop's `checkpoint()` saves model weights only |
| mlx | `last` | yes — from `last_optimizer.safetensors` |
| mlx | `periodic` | yes — if `patch_trainer_checkpoint` was called with `optimizer=` |
| torch → mlx | `last` | no — cross-backend transfer restarts optimizer |

### Normalizer handling

The script searches for `normalizer.npz` in this order:
1. `ckpt_dir/normalizer.npz`
2. `model_dir/normalizer.npz`
3. Falls back to building fresh from the training data (prints a warning).

For data-driven normalizers (`etau75_eta_3d`, `etau75_eta_shape_3d`) the saved
`ref` array is restored from disk — no re-fitting is needed.

The normalizer is always re-saved to `model_dir/normalizer.npz` after loading
so it stays in sync with the resumed weights.

---

## Configuration reference — `config_h5_resume.yaml`

All keys from `config_h5.yaml` are valid.  The table below describes the
resume-specific additions only.

### Resume control

| Key | Default | Description |
|-----|---------|-------------|
| `total_epochs` | — | **Required.** Final epoch target. `remaining = total_epochs − resume_epoch`. Must exceed the last saved epoch. |
| `resume_model_source` | `local` | `local` \| `gcs_download` — where the checkpoint lives |
| `resume_model_dir` | `model_dir` | Local path or `gs://` URI of the checkpoint directory. Defaults to `model_dir` when omitted. |
| `local_model_cache` | `./cache/checkpoints` | `[gcs_download]` Where to cache the downloaded checkpoint |
| `checkpoint_type` | `last` | `last` \| `periodic` — which saved state to load |
| `checkpoint_epoch` | — | `[periodic]` Epoch number N of the `checkpoint_N/` subdirectory |
| `resume_backend` | `null` | Backend that produced the checkpoint. `null` = same as `backend`. Set to `torch` for torch→mlx transfer. |

### Outputs

In addition to the standard `train_h5.py` outputs, the resume script writes:

| File | Backend | Description |
|------|---------|-------------|
| `metrics_resume.json` | torch | Metrics for the resumed portion only |
| `metrics_mlx_resume.json` | mlx | Metrics for the resumed portion only |
| `training_curves_resume.svg` | torch | Loss/metric plot for the resumed portion |
| `training_curves_mlx_resume.svg` | mlx | Loss/metric plot for the resumed portion |

The full training history (including the original run) is in the appended
`training.log` / `training_mlx.log`.

## Examples — Resume

### Same machine, MLX, resume from last

```yaml
backend:             mlx
data_source:         local
hdf5_dir:            ../data
hdf5_files:
  - dAu_250ev_mb.h5
total_epochs:        200
resume_model_source: local
resume_model_dir:    ./checkpoints
checkpoint_type:     last
```

### Same machine, PyTorch, resume from periodic checkpoint

```yaml
backend:             torch
data_source:         local
hdf5_dir:            /data/flat
hdf5_files:
  - dAu_250ev_mb.h5
total_epochs:        300
resume_model_source: local
resume_model_dir:    ./checkpoints
checkpoint_type:     periodic
checkpoint_epoch:    100
```

### GCS download checkpoint, MLX

```yaml
backend:             mlx
data_source:         local
hdf5_dir:            ../data
hdf5_files:
  - dAu_250ev_mb.h5
total_epochs:        200
resume_model_source: gcs_download
resume_model_dir:    gs://my-bucket/runs/run01/checkpoints
local_model_cache:   ./cache/checkpoints
gcs_key_file:        ./fnotest-wayne-gcs.json
checkpoint_type:     last
```

### Cross-backend: torch → mlx

```yaml
backend:             mlx
resume_backend:      torch
data_source:         local
hdf5_dir:            ../data
hdf5_files:
  - dAu_250ev_mb.h5
total_epochs:        200
resume_model_source: local
resume_model_dir:    ./checkpoints   # contains last_model.pt from torch run
checkpoint_type:     last
lr:                  8.0e-3          # optimizer restarts; set desired lr
```
