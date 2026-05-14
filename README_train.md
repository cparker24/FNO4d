# FNO4d Training — `test_train.py`

Trains a 4-D Fourier Neural Operator (FNO) on heavy-ion fluid-dynamics simulation
data stored in ROOT files. The model learns to evolve the hydrodynamic fields
(energy density, transverse velocities, longitudinal velocity) from an initial
τ-slice forward in proper time.

---

## Environment

```bash
conda activate fno_env_mlx
```

The script requires the custom 4-D neuralop fork
(`neuraloperator-4d-2.0.0`) and the local helper library `loc_libs`.

---

## Setup

Install `loc_libs` once so the local modules are importable everywhere:

```bash
pip install -e ./loc_libs
```

If `loc_libs` is **not** installed, the script falls back to
`sys.path.append('../loc_libs')` — no further action needed when running from
the `train/` directory.

---

## Running

```bash
cd train/
python test_train.py
```

Or as a Jupyter notebook: `test_train.ipynb`.

---

## Input data

| parameter | default | description |
|-----------|---------|-------------|
| `input_dir` | `./` | directory containing ROOT files |
| `input_files` | `['dAu_25ev_mb.root']` | list of ROOT files to load |
| `nevents_in` | `-1` (all) | number of events to load; set to a positive integer to limit |

ROOT files are read with `read_3d_data` from `loc_libs/read_3d_root.py`.
Each file must contain a tree with a flat 6-D array of shape

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
spatial and time edges before training — adjust to match your grid boundaries.

---

## Training parameters

| parameter | default | description |
|-----------|---------|-------------|
| `n_epochs` | `1` | number of training epochs |
| `batch_size` | `1` | events per mini-batch |
| `train_ratio` | `0.8` | fraction of events used for training |
| `hidden_channels` | `16` | FNO channel width |
| `n_modes` | `[32,32,16,16]` | Fourier modes retained per spatial/time dimension |
| `lr` | `8e-3` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `checkpoint_freq` | `None` | save a checkpoint every N epochs; `None` disables |
| `log_directly` | `True` | tee stdout to `training.log` during training |

---

## Model

```python
FNO(in_channels=4, out_channels=4,
    n_modes=[32,32,16,16], hidden_channels=16,
    projection_channel_ratio=2)
```

`in_channels` and `out_channels` correspond to the 4 hydrodynamic features
(`ed`, `vx`, `vy`, `vz`).

`n_modes` has one entry per spatial/temporal dimension — `[x, y, η, τ]` — and
sets how many low-frequency Fourier coefficients are kept in each FNO layer.
Reducing these values lowers memory and compute cost at the expense of
representational capacity.

### Optimizer and scheduler

- **AdamW** with cosine-annealing LR schedule (`T_max=30`)

### Loss functions

| loss | used for | description |
|------|----------|-------------|
| `H1Loss(d=4)` | training | Sobolev H¹ loss — penalises both value and gradient errors |
| `LpLoss(d=4, p=2)` | evaluation | relative L² loss |

Both are evaluated on all four spatial+time dimensions (`d=4`).

---

## Dataset — `FluidDataset` (`loc_libs/datasets.py`)

The dataset takes a single τ-slice as input and predicts the remaining
τ-steps as the target:

```
x  : initial τ-slice repeated (n_tau−1) times  →  shape (4, nx, ny, n_eta, n_tau−1)
y  : τ-slices 1 … n_tau−1                       →  shape (4, nx, ny, n_eta, n_tau−1)
```

---

## Output files

| file | description |
|------|-------------|
| `log.log` | run configuration and data-shape summary |
| `training.log` | full stdout from the neuralop `Trainer` (when `log_directly=True`) |
| `best_model.safetensors` | best checkpoint according to `100_l2` metric |
| `checkpoint_<epoch>/` | periodic checkpoints (when `checkpoint_freq` is set) |

---

## loc_libs modules used

| module | symbol | purpose |
|--------|--------|---------|
| `read_3d_root` | `read_3d_data` | load ROOT → numpy array |
| `datasets` | `FluidDataset` | PyTorch Dataset wrapping the 6-D array |
| `memory_utils` | `calculate_model_memory` | print model parameter count and MB footprint |
| `memory_utils` | `MemoryPrinter` | callable that prints process RSS at any point |
| `training_utils` | `enable_log_tee` | redirect stdout to both terminal and a log file |
| `training_utils` | `patch_trainer_checkpoint` | monkey-patch `Trainer.on_epoch_start` to save periodic checkpoints |
| `parse_training_log` | `plot_training_log` | post-hoc plotting of loss curves from `training.log` |

---

## Platform notes

| platform | status |
|----------|--------|
| Dell DGX GB10 (ARM + CUDA) | working |
| Mac M3 (CPU only) | working — MPS FFT not yet supported upstream in PyTorch |
| rhicML / Lambda instances | to be verified |
