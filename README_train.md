# FNO4d Training

Trains a 4-D Fourier Neural Operator (FNO) on heavy-ion fluid-dynamics simulation
data stored in ROOT files. The model learns to evolve the hydrodynamic fields
(energy density, transverse velocities, longitudinal velocity) from an initial
τ-slice forward in proper time.

Four training scripts are provided — two backends × two configuration styles:

| Script | Config style | Package | Backend | Target |
|--------|-------------|---------|---------|--------|
| `test_train.py` | hardcoded at top | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx.py` | hardcoded at top | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |
| `test_train_yaml.py` | YAML file | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx_yaml.py` | YAML file | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |

---

## Environment

```bash
conda activate fno_env_mlx
```

All scripts require the local helper library `loc_libs`.

---

## Setup

Install `loc_libs` once so all modules are importable everywhere:

```bash
pip install -e ./loc_libs
```

If `loc_libs` is **not** installed the scripts fall back to
`sys.path.append('../loc_libs')` — no further action needed when running from
the `train/` directory.

---

## Running

### Hardcoded-parameter scripts

Edit the parameter block at the top of the script, then run:

```bash
cd train/

# PyTorch (CUDA / CPU)
python test_train.py

# MLX (Apple Silicon Metal)
python test_train_mlx.py
```

Notebook equivalents: `test_train.ipynb`, `test_train_mlx.ipynb`.

### YAML-config scripts

Both YAML scripts share a single config file:

```bash
cd train/
mkdir models          # model_dir must exist before running

# PyTorch (CUDA / CPU)
python test_train_yaml.py                               # uses config_test_train.yaml
python test_train_yaml.py --config my_run.yaml          # override

# MLX (Apple Silicon Metal)
python test_train_mlx_yaml.py                          # uses config_test_train.yaml
python test_train_mlx_yaml.py --config my_run.yaml     # override
```

---

## Input data

ROOT files are read with `read_3d_data` from `loc_libs/read_3d_root.py`.
Each file must contain a flat 6-D array of shape

```
(n_events, n_features, nx, ny, n_eta, n_tau)
```

| axis | meaning |
|------|---------|
| 0 | events |
| 1 | features: `ed`, `vx`, `vy`, `vz` |
| 2–3 | transverse x / y bins |
| 4 | η (spatial rapidity) bins |
| 5 | τ (proper time) bins |

`xyslice` and `tauslice` (default `slice(None, -1)`) trim one bin off the
spatial and time edges before training.

### Memory diagnostics during loading

`read_3d_data` and `read_3d_data_hdf5` print `[Memory tracemalloc]` lines only
when `verbose=True` (default `False`):

```python
arr_data = read_3d_data(f'{input_dir}/{f}', nevents=nevents_in, verbose=True)
```

---

## Training parameters

For the hardcoded scripts all parameters live at the **top of the script**.
For the YAML scripts they live in `config_test_train.yaml` (or a custom file
passed via `--config`).

| parameter | default | description |
|-----------|---------|-------------|
| `odir` | `./` | output directory for logs, metrics, and training curves |
| `model_dir` | `./` | output directory for `best_model.safetensors` and checkpoints; **must exist** |
| `input_dir` | `/Users/du8478/FNO_Hydro_Data/flat` | directory containing ROOT files |
| `input_files` | `['dAu_25ev_mb.root']` | list of ROOT files to load |
| `nevents_in` | `10` | events to load; `-1` = all |
| `n_epochs` | `1` | number of training epochs |
| `batch_size` | `1` | events per mini-batch |
| `train_ratio` | `0.8` | fraction of events used for training |
| `hidden_channels` | `16` | FNO channel width |
| `n_modes` | `[32,32,16,16]` | Fourier modes per spatial/time dimension `[x, y, η, τ]` |
| `lr` | `8e-3` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `test_batch_sizes` | `[10]` | eval batch size per test resolution |
| `eval_interval` | `2` | run evaluation every N training epochs |
| `checkpoint_freq` | `None` | save a periodic checkpoint every N epochs; `None` disables |
| `save_last` | `True` | save a full resumable checkpoint after the last epoch |
| `log_directly` | `True` | mirror all output to a log file (see Logging) |
| `xyslice` | `[null, -1]` | YAML form of `slice(None, -1)` — drop last spatial bin |
| `tauslice` | `[null, -1]` | YAML form of `slice(None, -1)` — drop last tau bin |

`test_resolutions` is not a parameter — it is derived automatically from
`arr.shape[2]` (the x-dimension of the loaded, sliced data) so the eval metric
keys always match the actual grid size.

### `model_dir` validation

The YAML scripts check that `model_dir` exists at startup and abort immediately
if it does not — before any data is loaded:

```
FileNotFoundError: model_dir './models' does not exist — please create it before running.
```

---

## Model

```python
FNO(in_channels=4, out_channels=4,
    n_modes=[32,32,16,16], hidden_channels=16,
    projection_channel_ratio=2)
```

`in_channels` / `out_channels` correspond to the 4 hydrodynamic features.
`n_modes` sets how many low-frequency Fourier coefficients are kept per dimension;
reducing these lowers memory at the cost of representational capacity.

### Optimizer and scheduler

| script | optimizer | scheduler |
|--------|-----------|-----------|
| PyTorch | `AdamW(lr, weight_decay)` | `CosineAnnealingLR(T_max=30)` |
| MLX | `AdamW(lr, weight_decay)` | `cosine_decay(lr, T_max=30)` |

### Loss functions

| loss | used for | description |
|------|----------|-------------|
| `H1Loss(d=4)` | training | Sobolev H¹ loss — penalises value and gradient errors |
| `LpLoss(d=4, p=2)` | evaluation | relative L² loss |

`d=4` evaluates the loss across all four spatial+time dimensions.

---

## Datasets (`loc_libs/datasets.py`)

Both dataset classes share the same indexing logic via `_FluidDatasetBase`:
the first τ-slice is repeated as input, and τ-steps 1…T−1 are the target.

```
x  : initial τ-slice repeated (n_tau−1) times  →  shape (4, nx, ny, n_eta, n_tau−1)
y  : τ-slices 1 … n_tau−1                       →  shape (4, nx, ny, n_eta, n_tau−1)
```

| class | used by | return type |
|-------|---------|-------------|
| `FluidDataset` | `test_train.py`, `test_train_yaml.py` | `torch.FloatTensor` |
| `FluidDatasetMLX` | `test_train_mlx.py`, `test_train_mlx_yaml.py` | `numpy.float32` (MLX Trainer converts to `mx.array`) |

---

## Logging

All scripts use `RunLog` from `loc_libs/training_utils.py`:

```python
log = RunLog(f'{odir}/training.log' if log_directly else None)
```

- `log_directly=True` — every `log(...)` call prints to the terminal **and**
  appends the same line to the log file.
- `log_directly=False` — plain `print` only, no file written.

`patch_trainer_logging(trainer, log, history=history)` routes the Trainer's
per-epoch output through the same `log` object and into the `TrainingHistory`
collector, so all metrics appear in the log file alongside the configuration
summary.

---

## Metrics and visualisation

Per-epoch metrics are collected into a `TrainingHistory` object
(from `loc_libs/parse_training_log.py`) and saved automatically after training:

```python
history = TrainingHistory()
patch_trainer_logging(trainer, log, history=history)
trainer.train(...)
history.save(f'{odir}/metrics.json')
history.plot(save_path=f'{odir}/training_curves.svg', title='FNO Training')
```

### `metrics.json` structure

```json
{
  "train": {"epoch": [...], "time": [...], "avg_loss": [...], "train_err": [...]},
  "eval":  {"epoch": [...], "<nx>_h1": [...], "<nx>_l2": [...]}
}
```

where `<nx>` is the x-dimension bin count derived from the data.

### Reloading and replotting

```python
from parse_training_log import TrainingHistory
history = TrainingHistory.load('metrics.json')
history.plot(save_path='training_curves.svg', title='My run')
```

### Legacy text-log plotting

The existing `plot_training_log` function in `parse_training_log.py` still
works for plotting directly from `training.log` text files produced by earlier
runs.

---

## Memory and runtime reporting

After training completes, all scripts log:

```
peak CPU memory (RSS): 4.32 GB
peak Metal memory:     2.18 GB   # MLX scripts
peak CUDA memory:      6.07 GB   # PyTorch scripts (n/a if no CUDA device)
total time: 142.3s
```

Peak counters are reset just before `trainer.train()` so they reflect training
only (not data loading). Reading the counters has no measurable impact on
training performance — they are maintained as a side-effect of normal
allocation.

---

## Resuming training

When `save_last=True` (default), a full resumable checkpoint is written to
`model_dir` after the last epoch. To continue a run, load the checkpoint at
the start of the next script and adjust `n_epochs`:

```python
from training_utils import load_last_checkpoint

resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, model_dir)
# then pass remaining epochs to trainer:
trainer.n_epochs = total_epochs - resume_epoch
trainer.train(...)
```

### Files written by `save_last`

| framework | file | contents |
|-----------|------|---------|
| PyTorch | `last_model.pt` | model, optimizer, scheduler state dicts + epoch number |
| MLX | `last_model.safetensors` | model weights |
| MLX | `last_optimizer.safetensors` | optimizer state |
| MLX | `last_state.json` | epoch number and total epochs |

### Periodic checkpoints (`checkpoint_freq`)

Both frameworks write a subdirectory `checkpoint_<epoch>/` in `model_dir`
every `checkpoint_freq` epochs. Contents are fully resumable:

| framework | files in `checkpoint_<epoch>/` |
|-----------|-------------------------------|
| PyTorch | neuralop standard save — model weights, optimizer, scheduler, metadata |
| MLX | `model.safetensors`, `optimizer.safetensors`, `state.json` |

---

## Output files

Outputs are split between `odir` (logs and metrics) and `model_dir` (weights).

### PyTorch

| file | location | description |
|------|----------|-------------|
| `training.log` | `odir` | config summary + per-epoch train/eval metrics |
| `metrics.json` | `odir` | structured JSON of all per-epoch metrics |
| `training_curves.svg` | `odir` | figure: avg_loss, train_err, H¹, L² per epoch |
| `best_model.safetensors` | `model_dir` | best checkpoint (lowest L² eval loss) |
| `last_model.pt` | `model_dir` | last-epoch resumable checkpoint (when `save_last=True`) |
| `checkpoint_<epoch>/` | `model_dir` | periodic resumable checkpoints (when `checkpoint_freq` is set) |

### MLX

| file | location | description |
|------|----------|-------------|
| `training_mlx.log` | `odir` | config summary + per-epoch train/eval metrics |
| `metrics_mlx.json` | `odir` | structured JSON of all per-epoch metrics |
| `training_curves_mlx.svg` | `odir` | figure: avg_loss, train_err, H¹, L² per epoch |
| `best_model.safetensors` | `model_dir` | best checkpoint (lowest L² eval loss) |
| `last_model.safetensors` | `model_dir` | last-epoch model weights (when `save_last=True`) |
| `last_optimizer.safetensors` | `model_dir` | last-epoch optimizer state (when `save_last=True`) |
| `last_state.json` | `model_dir` | epoch metadata for resuming (when `save_last=True`) |
| `checkpoint_<epoch>/` | `model_dir` | periodic resumable checkpoints (when `checkpoint_freq` is set) |

---

## loc_libs modules used

| module | symbol | purpose |
|--------|--------|---------|
| `read_3d_root` | `read_3d_data` | load ROOT → numpy array (`verbose=False`) |
| `datasets` | `FluidDataset` | PyTorch Dataset wrapping the 6-D array |
| `datasets` | `FluidDatasetMLX` | MLX Dataset — returns numpy float32 |
| `memory_utils` | `calculate_model_memory` | parameter count and MB footprint |
| `memory_utils` | `MemoryPrinter` | callable RSS printer for ad-hoc profiling |
| `memory_utils` | `reset_peak_memory` | reset CPU/GPU peak counters before training |
| `memory_utils` | `log_peak_memory` | log peak CPU RSS and GPU memory after training |
| `training_utils` | `RunLog` | unified terminal + file logger |
| `training_utils` | `patch_trainer_logging` | route Trainer output through `RunLog` and `TrainingHistory` |
| `training_utils` | `patch_trainer_checkpoint` | periodic checkpoint saving via `on_epoch_start` |
| `training_utils` | `save_last_checkpoint` | save full resumable state after last epoch |
| `training_utils` | `load_last_checkpoint` | load saved state and return resume epoch |
| `parse_training_log` | `TrainingHistory` | in-memory metrics collection, JSON save/load, SVG plot |
| `parse_training_log` | `plot_training_log` | legacy text-log plotter |

---

## Platform notes

| platform | status |
|----------|--------|
| Dell DGX GB10 (ARM + CUDA) | working |
| Mac M3 — MLX | working (full Metal GPU via `test_train_mlx.py`) |
| Mac M3 — PyTorch | CPU only — MPS FFT not yet supported upstream |
| rhicML / Lambda instances | to be verified |
