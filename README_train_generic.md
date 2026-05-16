# train_generic — Unified FNO Training Script

Two training scripts cover all supported input formats:

| Script | Input format | Config |
|--------|-------------|--------|
| `train.py` | ROOT files (`.root`) — local or GCS | `config.yaml` |
| `train_h5.py` | Pre-converted HDF5 files (`.h5`) | `config_h5.yaml` |

## Quick start

```bash
cd train_generic

# ROOT-based (local or GCS)
python train.py --config config.yaml

# HDF5-based (multiple files, lazy or eager)
python train_h5.py --config config_h5.yaml
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
