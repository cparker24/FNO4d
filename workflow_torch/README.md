# workflow_torch — end-to-end PyTorch FNO workflow on the new data pipeline

A self-contained example of the full lifecycle — **train → resume → eval** —
for the 3+1D heavy-ion FNO surrogate, built entirely on the unified data
pipeline in [`loc_libs/data`](../loc_libs/data) (`DATA_PIPELINE_PLAN.md`).

It is PyTorch-only and deliberately small: three CLI scripts, one shared model
selector, and YAML configs.  The headline feature is that **HDF5 and ROOT input
are interchangeable** — the file extension picks the right lazy source, and
everything downstream (split, normalisation, first-tau windowing, DataLoaders)
is identical.

## Why this exists

The older `train_torch/*` scripts hand-rolled the dataset, normalizer fit, and
DataLoaders, and had separate code paths per input type. This workflow replaces
all of that with two shared entry points:

| Concern | Old | Here |
|---|---|---|
| Input source | per-type branches | `data.build_source(paths)` — `.root` → `RootSource`, `.h5` → `MultiH5Array` |
| Split + norm + window + loaders | inline per script | `data.build_dataloaders(...)` (normalisation + windowing in a batched collate) |
| Inference (eval) | per-event Python loop | `loc_libs/eval.run_inference` — batched, reuses the *training* collate (the CLI is `evaluate.py`; a script named `eval.py` cannot import `loc_libs/eval.py`) |
| Model construction | duplicated 3× | `_model.build_model(...)` shared by train/resume/eval |

## Files

```
workflow_torch/
  train.py          train an FNO (HDF5 or ROOT) on the new pipeline
  resume.py         resume training from a checkpoint
  evaluate.py       score a checkpoint with model_eval.ModelEval (CLI, no notebook)
  _model.py         shared model selector (fno | causal_fno4d | causal_fno4d_tau_eta)
  config.yaml       train config — HDF5 example
  config_root.yaml  train config — ROOT example
  config_resume.yaml
  config_eval.yaml
  README.md         this file
```

## Usage

```bash
cd workflow_torch

# 1) train (HDF5)
python train.py --config config.yaml
#    ... or ROOT, no code change:
python train.py --config config_root.yaml

# 2) resume to a higher epoch target (reuses the checkpoint's normalizer)
python resume.py --config config_resume.yaml

# 3) evaluate the trained checkpoint
python evaluate.py --config config_eval.yaml
#    -> out_eval/metrics_eval.json, dashboard.png, nrmse_heatmap.png
```

Scripts insert `../loc_libs` (and this dir, for `_model`) on `sys.path`, so run
them from anywhere — paths in the configs resolve relative to your working
directory.

## Input: HDF5 vs ROOT

`build_source` infers the type from the extension (all files in a run must
match):

* `.h5` / `.hdf5` → `read_3d_data_hdf5(..., lazy=True)` → `MultiH5Array`.
  `data_source: local` or `gcs_download` (h5py needs local paths).
* `.root` → `RootSource` (always lazy). Additionally supports
  `data_source: gcs_stream` (uproot streams `gs://` directly — no download).

Switch by changing `input_files`; nothing else in the config or scripts changes.

## Config keys

`config.yaml` / `config_root.yaml` document every key inline. The essentials:

* **`data_source`** — `local` | `gcs_download` | `gcs_stream` (ROOT only).
* **`model_type`** — `fno` | `causal_fno4d` | `causal_fno4d_tau_eta`
  (causal variants take extra kwargs under `causal_fno4d:`).
* **`normalization.name`** — `none` | `etau75_3d` | `etau75_eta_3d` |
  `etau75_eta_shape_3d` | `cosh_etau75_3d`. Data-driven variants are fitted
  chunked on the train split by `build_dataloaders`.
* **`downsample:`** — optional grid coarsening, auto-applied before the
  `xy/eta/tau` slices.
* **`loader:`** — `num_workers`, `prefetch_factor`, `locality_block`.

### Resume

`config_resume.yaml` adds `total_epochs`, `resume_model_source`,
`resume_model_dir`, `checkpoint_type` (`last`/`periodic`). The normalizer is
loaded from the checkpoint dir and passed straight into `build_dataloaders`
(`normalizer=`), so it is reused verbatim — never refit. `save_last: true` in
the original run is required for `checkpoint_type: last`.

### Eval

`config_eval.yaml` points at the run's `train_config` for architecture + data
prep, plus a `model_dir` and `checkpoint_type` (`best`/`last`/`checkpoint_N`).
By default it scores the held-out val split of the training files
(`use_train_split: true`); set `eval_input_files` to score a separate hold-out
set instead. Outputs `metrics_eval.json` and (if `save_figures`) a dashboard +
nRMSE heatmap.

## Shared eval helpers

The reusable, backend-agnostic eval pieces live in
[`loc_libs/eval.py`](../loc_libs/eval.py) so the notebooks
(`eval_model/eval_gcs_setup.py`) and this CLI share one implementation:

* `run_inference(model, backend, device, data_test, normalizer, batch_size)` —
  batched inference through `FluidDataset` + the training collate (first-tau
  repeat + same normalizer), so eval preprocessing is bit-identical to training.
* `compute_grid_coords(data_test, grid)` — physical (x, y, η, τ_out) arrays.
* `load_torch_model / load_torch_state_dict` — tolerant checkpoint loading.

The metric suite itself stays in [`model_eval.py`](../loc_libs/model_eval.py)
(`ModelEval`).
