# FNO4d Training

Trains a 4-D Fourier Neural Operator (FNO) on heavy-ion fluid-dynamics simulation
data stored in ROOT files. The model learns to evolve the hydrodynamic fields
(energy density, transverse velocities, longitudinal velocity) from an initial
τ-slice forward in proper time.

Two training scripts are provided — one per compute backend:

| Script | Package | Backend | Target |
|--------|---------|---------|--------|
| `test_train.py` | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx.py` | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |

---

## Environment

```bash
conda activate fno_env_mlx
```

Both scripts require the local helper library `loc_libs`.

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

```bash
cd train/

# PyTorch (CUDA / CPU)
python test_train.py

# MLX (Apple Silicon Metal)
python test_train_mlx.py
```

Notebook equivalents: `test_train.ipynb`, `test_train_mlx.ipynb`.

---

## Input data

| parameter | PyTorch default | MLX default | description |
|-----------|----------------|-------------|-------------|
| `input_dir` | `./` | `/Users/du8478/FNO_Hydro_Data/flat` | directory containing ROOT files |
| `input_files` | `['dAu_25ev_mb.root']` | `['dAu_25ev_mb.root']` | list of ROOT files to load |
| `nevents_in` | `-1` (all) | `10` | events to load; `-1` = all |

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

All parameters live at the **top of the script** for easy editing before each run.

| parameter | default | description |
|-----------|---------|-------------|
| `n_epochs` | `1` | number of training epochs |
| `batch_size` | `1` | events per mini-batch |
| `train_ratio` | `0.8` | fraction of events used for training |
| `hidden_channels` | `16` | FNO channel width |
| `n_modes` | `[32,32,16,16]` | Fourier modes per spatial/time dimension `[x, y, η, τ]` |
| `lr` | `8e-3` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `checkpoint_freq` | `None` | save a checkpoint every N epochs; `None` disables |
| `log_directly` | `True` | mirror all output to a log file (see Logging) |

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
| `FluidDataset` | `test_train.py` | `torch.FloatTensor` |
| `FluidDatasetMLX` | `test_train_mlx.py` | `numpy.float32` (MLX Trainer converts to `mx.array`) |

---

## Logging

Both scripts use `RunLog` from `loc_libs/training_utils.py`:

```python
log = RunLog(f'{odir}/training.log' if log_directly else None)
```

- `log_directly=True` — every `log(...)` call prints to the terminal **and**
  appends the same line to the log file.
- `log_directly=False` — plain `print` only, no file written.

`patch_trainer_logging(trainer, log)` routes the Trainer's per-epoch output
(train error, eval metrics) through the same `log` object, so all metrics
appear in the log file alongside the configuration summary.

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
  "eval":  {"epoch": [...], "100_h1": [...], "100_l2": [...]}
}
```

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

## Output files

### PyTorch (`test_train.py`)

| file | description |
|------|-------------|
| `training.log` | config summary + per-epoch train/eval metrics |
| `metrics.json` | structured JSON of all per-epoch metrics |
| `training_curves.svg` | 4-panel figure: avg_loss, train_err, 100_h1, 100_l2 |
| `best_model.safetensors` | best checkpoint (lowest `100_l2`) |
| `checkpoint_<epoch>/` | periodic checkpoints (when `checkpoint_freq` is set) |

### MLX (`test_train_mlx.py`)

| file | description |
|------|-------------|
| `training_mlx.log` | config summary + per-epoch train/eval metrics |
| `metrics_mlx.json` | structured JSON of all per-epoch metrics |
| `training_curves_mlx.svg` | 4-panel figure |
| `best_model.safetensors` | best checkpoint (lowest `100_l2`) |

---

## loc_libs modules used

| module | symbol | purpose |
|--------|--------|---------|
| `read_3d_root` | `read_3d_data` | load ROOT → numpy array (`verbose=False`) |
| `datasets` | `FluidDataset` | PyTorch Dataset wrapping the 6-D array |
| `datasets` | `FluidDatasetMLX` | MLX Dataset — returns numpy float32 |
| `memory_utils` | `calculate_model_memory` | parameter count and MB footprint |
| `memory_utils` | `MemoryPrinter` | callable RSS printer for ad-hoc profiling |
| `training_utils` | `RunLog` | unified terminal + file logger |
| `training_utils` | `patch_trainer_logging` | route Trainer verbose output through `RunLog` and `TrainingHistory` |
| `training_utils` | `patch_trainer_checkpoint` | periodic checkpoint saving via `on_epoch_start` |
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
