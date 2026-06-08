# FNO Model Evaluation — Panel Web App

Interactive evaluation dashboard for FNO surrogate models of heavy-ion
hydrodynamics.  Panel-based alternative to the marimo notebook at
`../eval_model/eval_gcs.py`, with identical GCS support and the same
three evaluation modes.

## Installation

Activate your conda or venv environment first, then:

```bash
# Install base + GCS dependencies from the repo root (if not done already)
pip install -r requirements-base.txt
pip install -r requirements-gcs.txt   # only needed for GCS access

# Install Panel (the only new dependency)
bash eval_model_web/install.sh
```

## Running

Use the launcher script — it handles conda/venv activation automatically:

```bash
bash eval_model_web/run.sh
```

The app opens at `http://localhost:5006/app`.

### Launcher options

| Variable / flag | Effect |
|---|---|
| `CONDA_ENV_NAME=my_env bash run.sh` | Use a different conda env (default: `fno_env_mlx`) |
| `VENV_PATH=~/my_venv bash run.sh` | Use a venv instead of conda |
| `PORT=8080 bash run.sh` | Custom port (default: 5006) |
| `bash run.sh --show` | Open browser tab automatically |
| `bash run.sh --address 0.0.0.0` | Expose on the network (LAN access) |
| `EVAL_GCS_CONFIG=/path/to/config.yaml bash run.sh` | Override config path |

If your environment is already activated, you can also run Panel directly:

```bash
panel serve eval_model_web/app.py --show
```

## Startup behaviour

On first load the browser shows a **Pipeline loading** banner and a live log
terminal.  All heavy work (GCS downloads, data loading, model loading,
inference) runs in the background.  The page refreshes automatically when
the pipeline completes — no manual reload needed.

If the pipeline fails the banner turns red and the full traceback appears in
the log.  Fix the config or environment issue, then reload the page to retry.

> **Maintenance note (2026-06-08):** `app.py` still carries its own
> *self-contained* inference pipeline — a per-event loop plus an upfront
> full-array `normalizer.transform` (its local `data_test_norm`).  The marimo
> eval paths (`../eval_model/eval.py`, `eval_gcs_setup.py`) were migrated to a
> batched `DataLoader` over the shared `loc_libs/data` pipeline (`FluidDataset`
> \+ collate), which reuses the training first-tau-repeat windowing/normalizer
> and batches the forward pass.  `app.py` was intentionally left unchanged then
> and is now the last remaining copy of the old pattern (alongside the
> standalone `eval_gcs.py` and `mlx_vs_torch_inf_test.py`).  Porting it to the
> same batched path is a worthwhile future cleanup for speed and consistency —
> verified equivalent (bit-identical pred/true) in the marimo migration — but
> not required for the app to function.

## Config file

The app reads the same `eval_gcs_config.yaml` as the marimo notebook.

**Default lookup order:**

1. `EVAL_GCS_CONFIG` environment variable
2. `../eval_model/eval_gcs_config.yaml` (relative to `app.py`)

You can also type a new path directly in the **Config file path** field in
the sidebar — the pipeline re-runs automatically on change.

All relative paths inside the config are resolved relative to the config
file's own directory (not the working directory), matching the marimo
notebook's behaviour.

## Eval modes

| Mode | Description |
|---|---|
| `basic` | Truth / prediction image grids per η bin at selected τ steps |
| `cont` | CDF-based contour overlays (3 τ snapshots per η bin) |
| `metric` | Full metric suite: nRMSE · fRMSE · MELR · WLR dashboard |

All modes also show the relative L² error vs τ curve below the main plot.
If a `metrics.json` training-history file is found in `odir`, a training
curve is appended at the bottom.

## Caching

All expensive work runs once per `(config_path, config_mtime)` pair and is
cached for the lifetime of the server process.  Widget changes (event index,
η bin, τ steps, …) only re-render the matplotlib figures — typically
sub-second.

Editing the config file on disk invalidates the cache and re-runs the
pipeline on the next page interaction.

## File layout

```
eval_model_web/
  app.py                  main Panel application
  run.sh                  launcher (conda + venv, env activation, port)
  install.sh              install Panel dependencies into active env
  requirements-panel.txt  Panel requirement (panel>=1.4)
  README.md               this file
```

The app imports from `../loc_libs/` (added to `sys.path` at startup) and
uses `gcs_utils`, `norm_utils`, `model_eval`, `skinnycontour`, `read_3d_hdf5`,
and optionally `read_3d_root` (requires `uproot`/`awkward` for ROOT files).
