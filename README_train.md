# FNO4d Training

Trains a 4-D Fourier Neural Operator (FNO) on heavy-ion fluid-dynamics simulation
data stored in ROOT files. The model learns to evolve the hydrodynamic fields
(energy density, transverse velocities, longitudinal velocity) from an initial
τ-slice forward in proper time.

> **Setup and environment** — see the main [README.md](README.md) for installation
> instructions, environment activation, pre-flight checks, and cloud storage setup.

---

## Training scripts

Six training scripts are provided — two backends × three configuration styles:

| Script | Config style | Package | Backend | Target |
|--------|-------------|---------|---------|--------|
| `test_train.py` | hardcoded at top | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx.py` | hardcoded at top | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |
| `test_train_yaml.py` | YAML file | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx_yaml.py` | YAML file | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |
| `test_train_yaml_norm.py` | YAML file + normalisation | `neuralop` (local 4D fork) | CUDA / CPU | Lambda / HPC |
| `test_train_mlx_yaml_norm.py` | YAML file + normalisation | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |

The `_norm` scripts are drop-in replacements for the plain YAML scripts —
they accept the same config file and add an optional `normalization:` block.
With `name: none` (the default) their behaviour is identical to the plain YAML scripts.

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

All YAML scripts share a single config file:

```bash
cd train/
mkdir checkpoints          # model_dir must exist before running

# PyTorch (CUDA / CPU)
python test_train_yaml.py                               # uses config_test_train.yaml
python test_train_yaml.py --config my_run.yaml          # override

# MLX (Apple Silicon Metal)
python test_train_mlx_yaml.py                          # uses config_test_train.yaml
python test_train_mlx_yaml.py --config my_run.yaml     # override

# With normalisation (see Data normalisation section below)
python test_train_yaml_norm.py --config my_run.yaml
python test_train_mlx_yaml_norm.py --config my_run.yaml
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
| `n_modes` | `[32,32,16,16]` | Fourier modes per spatial/time dimension `[x, y, η, τ]`; for `causal_fno4d*` only the first 3 entries are used |
| `model_type` | `fno` | architecture: `fno` (default) \| `causal_fno4d` \| `causal_fno4d_tau_eta` — see [Causal FNO](#causal-fno--causalfno4d--causalfno4dtaueta) in README.md |
| `causal_fno4d` | _(map)_ | extra kwargs for `CausalFNO4D` / `CausalFNO4DTauEta` when `model_type` is `causal_fno4d*`; see config.yaml comments for available keys |
| `lr` | `8e-3` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `test_batch_sizes` | `[10]` | eval batch size per test resolution |
| `eval_interval` | `2` | run evaluation every N training epochs |
| `checkpoint_freq` | `None` | save a periodic checkpoint every N epochs; `None` disables |
| `min_save_epoch` | `None` | skip `best_model` and periodic checkpoint saves until this epoch is reached; `None` saves from epoch 0; `save_last` is unaffected |
| `save_last` | `True` | save a full resumable checkpoint after the last epoch |
| `log_directly` | `True` | mirror all output to a log file (see Logging) |
| `xyslice` | `[null, -1]` | YAML form of `slice(None, -1)` — drop last spatial bin |
| `tauslice` | `[null, -1]` | YAML form of `slice(None, -1)` — drop last tau bin |
| `normalization` | `{name: none}` | data normalisation (see below); `_norm` scripts only |

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

## Data normalisation (`_norm` scripts only)

The `_norm` scripts normalise the raw energy-density field before it reaches the
network and save a `normalizer.npz` file alongside the model so predictions can
be inverted back to physical units.

Normalisation is always fit on the **training split only**; the same parameters
are then applied to the validation split — preventing data leakage from
data-driven schemes.

### Selecting a normalisation

Add a `normalization:` block to the YAML config:

```yaml
normalization:
  name: none              # default — identity, no change in behaviour
```

Available names:

| name | transform | notes |
|------|-----------|-------|
| `none` | identity | default; identical to non-`_norm` scripts |
| `etau75_3d` | ed → ed^0.75 · τ | tau scaling only; motivated by ideal Bjorken flow |
| `etau75_eta_3d` | ed → (ed^0.75 · τ) / ref[η, τ] | data-driven per-(η,τ) rescaling; ref computed from training data |
| `etau75_eta_shape_3d` | ed → (ed^0.75 · τ) / shape\_ref[η, τ] | shape-only η correction; preserves Bjorken tau evolution |
| `cosh_etau75_3d` | ed → (ed · cosh(η)^α)^0.75 · τ | analytic η correction; no training data needed |

For the `cosh_etau75_3d` variant, set the exponent via:

```yaml
normalization:
  name: cosh_etau75_3d
  alpha: 1.0              # default 1.0; physically motivated alternatives: 4/3
```

Grid parameters (`tau_min`, `dtau`, `eta_min`, `deta`) are read automatically
from the data file — they do not need to be repeated in the YAML.

### Saved normalizer

After training, `{model_dir}/normalizer.npz` contains everything needed to
invert predictions:

```python
from norm_utils import Normalizer

normalizer = Normalizer.load('checkpoints/normalizer.npz')
pred_physical = normalizer.inverse(model_output_array)
```

The `.npz` stores: `name`, `tau_min`, `dtau`, `alpha`, and (for data-driven
schemes) the `ref` array computed from training data.

### Normalisation math

See `loc_libs/NormFn3d.py` for full derivations and docstrings. All families
operate only on feature 0 (energy density `ed`); velocity channels are passed
through unchanged.

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

## Datasets (`loc_libs/fluid_datasets.py`)

Both dataset classes share the same indexing logic via `_FluidDatasetBase`:
the first τ-slice is repeated as input, and τ-steps 1…T−1 are the target.

```
x  : initial τ-slice repeated (n_tau−1) times  →  shape (4, nx, ny, n_eta, n_tau−1)
y  : τ-slices 1 … n_tau−1                       →  shape (4, nx, ny, n_eta, n_tau−1)
```

| class | used by | return type |
|-------|---------|-------------|
| `FluidDataset` | PyTorch scripts | `torch.FloatTensor` |
| `FluidDatasetMLX` | MLX scripts | `numpy.float32` (MLX Trainer converts to `mx.array`) |

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
`model_dir` after the last epoch. The model architecture and hyperparameters
are **not** stored in the checkpoint — the same config (or parameter block)
used for the original run must be used when resuming.

For `_norm` scripts, also reload the normalizer before constructing datasets:

```python
from norm_utils import Normalizer
normalizer  = Normalizer.load(f'{model_dir}/normalizer.npz')
data_train  = normalizer.transform(data_train)
data_verify = normalizer.transform(data_verify)
```

### PyTorch — resume from last epoch

Use the same script but set `n_epochs` to the **total** number of epochs you
want in the end (not the remaining count). The snippet below replaces the
training block:

```python
from training_utils import load_last_checkpoint

total_epochs = 200       # final target — must be > epochs already trained

model = FNO(
    in_channels=4, out_channels=4,
    n_modes=n_modes, hidden_channels=hidden_channels,
    projection_channel_ratio=2,
).to(device)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

trainer = Trainer(model=model, n_epochs=total_epochs,
                  device=device, wandb_log=False,
                  eval_interval=eval_interval,
                  use_distributed=False, verbose=True)

patch_trainer_logging(trainer, log, history=history)

# --- resume ---
resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, model_dir)
trainer.n_epochs = total_epochs - resume_epoch   # remaining epochs only

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer, scheduler=scheduler,
              regularizer=False, save_dir=model_dir,
              save_best=save_best_key,
              training_loss=train_loss, eval_losses=eval_losses)
```

`load_last_checkpoint` reads `model_dir/last_model.pt` and restores model
weights, AdamW moments, and the cosine LR schedule position into the objects
already constructed above.

To resume from a **periodic checkpoint** instead, pass the subdirectory:

```python
resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler,
                                    f'{model_dir}/checkpoint_50')
trainer.n_epochs = total_epochs - resume_epoch
```

---

### MLX — resume from last epoch

```python
from training_utils import load_last_checkpoint

total_epochs = 200       # final target — must be > epochs already trained

model = FNO(
    in_channels=4, out_channels=4,
    n_modes=n_modes, hidden_channels=hidden_channels,
    projection_channel_ratio=2,
)

optimizer = AdamW(lr=lr, weight_decay=weight_decay)
scheduler = optim_mlx.cosine_decay(lr, T_max := 30)

trainer = Trainer(
    model=model, n_epochs=total_epochs,
    device=device, wandb_log=False,
    eval_interval=eval_interval,
    use_distributed=False, verbose=True,
)

patch_trainer_logging(trainer, log, history=history)

# --- resume ---
resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, model_dir)
trainer.n_epochs = total_epochs - resume_epoch   # remaining epochs only

trainer.train(
    train_loader=train_loader, test_loaders=test_loaders,
    optimizer=optimizer, scheduler=scheduler,
    regularizer=False, save_dir=model_dir,
    save_best=save_best_key,
    training_loss=train_loss, eval_losses=eval_losses,
)
```

`load_last_checkpoint` reads `model_dir/last_model.safetensors` (model
weights) and `model_dir/last_optimizer.safetensors` (AdamW state), and
returns the next epoch number from `model_dir/last_state.json`.

To resume from a **periodic checkpoint**:

```python
resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler,
                                    f'{model_dir}/checkpoint_50')
trainer.n_epochs = total_epochs - resume_epoch
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
| `normalizer.npz` | `model_dir` | normalizer state for inverse transform (`_norm` scripts only) |
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
| `normalizer.npz` | `model_dir` | normalizer state for inverse transform (`_norm` scripts only) |
| `checkpoint_<epoch>/` | `model_dir` | periodic resumable checkpoints (when `checkpoint_freq` is set) |

---

## loc_libs modules used

| module | symbol | purpose |
|--------|--------|---------|
| `read_3d_root` | `read_3d_data` | load ROOT → numpy array (`verbose=False`) |
| `fluid_datasets` | `FluidDataset` | PyTorch Dataset wrapping the 6-D array |
| `fluid_datasets` | `FluidDatasetMLX` | MLX Dataset — returns numpy float32 |
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
| `NormFn3d` | `NormFn3d` | physics-motivated normalisation functions and their inverses |
| `norm_utils` | `build_normalizer`, `Normalizer` | stateful normalizer — fit/transform/inverse/save/load (`_norm` scripts only) |

---

## Platform notes

| platform | status |
|----------|--------|
| Dell DGX GB10 (ARM + CUDA) | working |
| Mac M3 — MLX | working (full Metal GPU via `test_train_mlx.py`) |
| Mac M3 — PyTorch | CPU only — MPS FFT not yet supported upstream |
| rhicML / Lambda instances | to be verified |
