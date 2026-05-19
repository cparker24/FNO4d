# FNO Model Evaluation — Panel Web App

Interactive evaluation dashboard for FNO surrogate models of heavy-ion
hydrodynamics.  This is a Panel-based alternative to the marimo notebook
at `../eval_model/eval_gcs.py`, with identical GCS support and the same
three evaluation modes.

## Installation

```bash
# From the repo root — install base + GCS dependencies first (if not done)
pip install -r requirements-base.txt
pip install -r requirements-gcs.txt   # only needed for GCS access

# Then install Panel (the only new dependency)
bash eval_model_web/install.sh
```

## Running

```bash
cd eval_model_web
panel serve app.py --show
```

The app opens at `http://localhost:5006/app`.  To expose it on the network:

```bash
panel serve app.py --show --port 5006 --address 0.0.0.0
```

## Config file

The app reads the same `eval_gcs_config.yaml` as the marimo notebook.

**Default lookup order:**

1. `EVAL_GCS_CONFIG` environment variable
2. `../eval_model/eval_gcs_config.yaml` (relative to this file)

Override at launch:

```bash
EVAL_GCS_CONFIG=/path/to/my_config.yaml panel serve app.py --show
```

Or type a new path directly in the **Config file path** field in the
sidebar — the pipeline re-runs automatically.

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

All expensive work (GCS downloads, data loading, model loading, inference)
runs once and is cached for the lifetime of the server process.  Widget
changes (event index, η bin, τ steps, …) only re-render the matplotlib
figures — typically sub-second.

The cache is keyed by `(config_path, config_mtime)`.  Editing the config
file on disk (which changes its mtime) automatically invalidates the cache
and re-runs the pipeline on the next interaction.

## File layout

```
eval_model_web/
  app.py                  main Panel application
  install.sh              install Panel dependencies
  requirements-panel.txt  pinned Panel requirement (panel>=1.4)
  README.md               this file
```

The app imports from `../loc_libs/` (added to `sys.path` automatically) and
uses the same `gcs_utils`, `norm_utils`, `model_eval`, `skinnycontour`, and
reader libraries as the marimo notebook.
