# FNO4d — 4D Fourier Neural Operator for Heavy-Ion Collisions

Training pipeline for a 4-dimensional Fourier Neural Operator (FNO) applied to
relativistic heavy-ion collision data (hydrodynamic initial conditions → final
state observables). 

The PyTorch FNO implementation in [neuraloperator-4d-2.0.0/](neuraloperator-4d-2.0.0/) is a
**minimal fork** of the upstream [neuraloperator](https://github.com/neuraloperator/neuraloperator)
library by Kossaifi, Kovachki, Li, et al. and all credit goes to authors [Credits & Attribution](#credits--attribution). 

The MLX port (heavy usage of Claude Sonnet 4.6) is following the PyTorch implementation as closely as possible, some constraints still exist and not every model can be ported as this moment (see [PORTING_NOTES.md](PORTING_NOTES.md)). Since no real physics tests and direct comparison to the PyTorch version is performed yet, consider the MLX **experimental** and should without out further verification only used to develop infra structure and other aspects localy on a Mac and not physics!

Two parallel training stacks are provided:

| Stack | Package | GPU backend | Target |
|---|---|---|---|
| PyTorch | `neuralop` (local fork) | CUDA (Linux) / CPU (Mac) | Lambda GPU instances, HPC |
| MLX | `neuralop_mlx` (experimental)| Metal (Apple Silicon) | Mac M-series |

---

## Table of Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [One-shot install](#one-shot-install)
  - [Fresh isolated environment (`install_full.sh`)](#fresh-isolated-environment-install_fullsh)
  - [Manual install](#manual-install-without-the-script)
  - [Pre-flight checks](#pre-flight-checks)
  - [Verify the install](#verify-the-install)
- [Cloud storage](#cloud-storage)
- [Training](#training)
  - [Edit parameters](#edit-parameters)
  - [PyTorch path (Linux / Lambda)](#pytorch-path-linux--lambda)
  - [MLX path (Mac Silicon)](#mlx-path-mac-silicon)
  - [GCS training (cloud data)](#gcs-training-cloud-data)
  - [Inference / apply](#inference--apply)
  - [Evaluation](#evaluation)
- [Model evaluation](#model-evaluation)
  - [Metric categories](#metric-categories)
  - [Quick usage](#quick-usage)
  - [Minimum publication checklist](#minimum-publication-checklist)
  - [Documentation](#documentation)
- [Packages](#packages)
  - [`neuralop_mlx` — MLX FNO (Apple Silicon)](#neuralop_mlx--mlx-fno-apple-silicon)
  - [`neuralop` — PyTorch neuraloperator (local 4D fork)](#neuralop--pytorch-neuraloperator-local-4d-fork)
  - [`loc_libs` — data I/O helpers](#loc_libs--data-io-helpers)
- [Tests](#tests)
- [Porting status (PyTorch → MLX)](#porting-status-pytorch--mlx)
- [Credits & Attribution](#credits--attribution)

---

## Repository layout

```
FNO4d/
├── README_train.md              # training scripts — parameters, data format, outputs
├── README_train_gcs.md          # GCS training — cloud data, streaming vs download, IAM setup
├── README_eval.md               # evaluation / inference scripts — UI controls, outputs, examples
├── README_eval_metric.md        # ModelEval metric suite — math, diagnostics, API reference
├── README_mlx.md                # MLX port overview — setup, differences from PyTorch stack
├── PORTING_NOTES.md             # technical notes for continuing the neuralop → MLX port
├── install.sh                   # install into current env (inherits system packages)
├── install_full.sh              # create a fresh isolated env (conda or virtualenv)
├── install_kernel.sh            # register active env as a Jupyter kernel
├── install_marimo.sh            # install marimo into the active env (optional)
├── check_install.sh             # pre-flight checks for install.sh (mirrors its probes)
├── check_install_full.sh        # pre-flight checks for install_full.sh
├── setup_nbstripout.sh          # install nbstripout and register git filter for notebooks
├── requirements-base.txt        # shared deps (both platforms)
├── requirements-mac.txt         # Mac Silicon: adds torch + mlx
├── requirements-linux.txt       # Linux/Lambda: adds nothing (torch auto-detected)
├── requirements-s3.txt          # opt-in: S3 / S3-compatible (--s3 flag)
├── requirements-gcs.txt         # opt-in: Google Cloud Storage (--gcs flag)
│
├── neuralop_mlx/                # MLX FNO implementation (Apple Silicon)
│   ├── models/fno.py            # FNO model (N-dimensional)
│   ├── layers/                  # spectral conv, channel MLP, embeddings, …
│   ├── training/                # Trainer, AdamW, LpLoss, H1Loss
│   └── weight_transfer.py       # load_from_torch() — PyTorch → MLX weight migration
│
├── neuraloperator-4d-2.0.0/     # PyTorch neuraloperator (local 4D fork)
│
├── tests/                       # MLX unit and integration tests
│   ├── test_fno_model_mlx.py    # forward-pass shape and output checks for MLX FNO
│   ├── test_layers_mlx.py       # spectral conv, channel MLP, embedding layer tests
│   ├── test_training_mlx.py     # one-step training loop smoke test (MLX)
│   ├── test_weight_transfer.py  # PyTorch → MLX weight transfer correctness
│   └── test_compare_torch_mlx.py # numerical parity check: PyTorch vs MLX outputs
│
├── loc_libs/                    # local helper library (data I/O, normalisation, utilities)
│   ├── read_3d_root.py          # read events from ROOT files (local or cloud)
│   ├── read_3d_hdf5.py          # ROOT → HDF5 conversion + streaming reads
│   ├── root_pytorch_loader.py   # IterableDataset for large ROOT files
│   ├── datasets.py              # in-memory PyTorch Dataset classes
│   ├── training_utils.py        # RunLog and shared training helpers
│   ├── parse_training_log.py    # parse and plot training-loss logs
│   ├── memory_utils.py          # memory profiling utilities
│   ├── NormFn3d.py              # physics-motivated normalisation functions and inverses
│   ├── NormFn.py                # older 3-channel normalizers (pre-NormFn3d, backward compat)
│   ├── README_NormFn3d.md       # extended docs for the NormFn3d normalizer family
│   ├── norm_utils.py            # stateful Normalizer wrapper (fit/transform/inverse/save/load)
│   ├── gcs_utils.py             # GCS helpers: streaming storage_options + file download (opt-in)
│   ├── skinnycontour.py         # skinny-contour CDF overlay utilities
│   ├── cont.py                  # low-level contour mask helpers (backend for skinnycontour)
│   ├── contour_v_plot.py        # velocity-channel contour plotting
│   ├── plot_imshow.py           # imshow-based field snapshot plotting helpers
│   └── model_eval.py            # ModelEval — full metric suite (nRMSE, fRMSE, MELR, WLR, …)
│
├── train/                       # training entry points and evaluation notebooks
    ├── test_train.py            # PyTorch training script (hardcoded params)
    ├── test_train.ipynb         # PyTorch training notebook
    ├── test_train_mlx.py        # MLX training script (hardcoded params, Mac)
    ├── test_train_mlx.ipynb     # MLX training notebook (Mac)
    ├── test_train_yaml.py       # PyTorch training script (YAML config)
    ├── test_train_mlx_yaml.py   # MLX training script (YAML config, Mac)
    ├── test_train_yaml_norm.py  # PyTorch training script (YAML config + normalisation)
    ├── test_train_mlx_yaml_norm.py # MLX training script (YAML config + normalisation, Mac)
    ├── config_test_train.yaml   # shared YAML config for all YAML-based scripts
    ├── test_train_hdf5.ipynb    # HDF5-backed PyTorch training notebook
    ├── test_mem_data_load.py    # memory profiling — ROOT loader
    ├── test_mem_data_load_hdf5.py # memory profiling — HDF5 loader
    ├── apply.py                 # marimo inference app
    ├── eval_model_test_torch.py # PyTorch eval — imshow field snapshots
    ├── eval_model_test_mlx.py   # MLX eval — imshow field snapshots
    ├── eval_model_test_cont_torch.py  # PyTorch eval — skinnycontour CDF contours
    ├── eval_model_test_cont_mlx.py    # MLX eval — skinnycontour CDF contours
    ├── eval_model_test_metric_torch.py # PyTorch eval — full ModelEval metric suite
    └── eval_model_test_metric_mlx.py   # MLX eval — full ModelEval metric suite
│
└── train_gcs/                   # GCS-backed training scripts (data stays in the bucket)
    ├── test_train_yaml_norm_gcs_stream.py     # PyTorch — stream directly from gs://
    ├── test_train_mlx_yaml_norm_gcs_stream.py # MLX — stream directly from gs://
    ├── test_train_yaml_norm_gcs_download.py   # PyTorch — download to local_data_dir first
    ├── test_train_mlx_yaml_norm_gcs_download.py # MLX — download to local_data_dir first
    ├── config_gcs_stream.yaml   # example config for streaming scripts
    └── config_gcs_download.yaml # example config for download scripts
```

---

## Installation

### Prerequisites

- Python ≥ 3.9
- **Mac:** Xcode command-line tools (`xcode-select --install`)
- **Linux/Lambda:** NVIDIA driver + `nvidia-smi` in `PATH` (for CUDA detection)
- `git clone` this repo and `cd` into it

### One-shot install

```bash
bash install.sh                 # default: always creates/reuses venv/
bash install.sh --conda         # opt-in: install into active named conda env
bash install.sh --s3            # + AWS S3 / S3-compatible storage (s3fs, boto3)
bash install.sh --gcs           # + Google Cloud Storage (gcsfs, google-cloud-storage)
bash install.sh --s3 --gcs      # + both cloud backends
```

Flags compose freely, e.g. `bash install.sh --conda --s3 --gcs`.

#### Environment modes

| Command | Behaviour |
|---|---|
| `bash install.sh` | Creates (or reuses) `venv/` in the repo root with `--system-site-packages`. Works in a bare shell, inside a conda env, or in a Docker container — packages already installed in the parent Python are visible without reinstalling. |
| `bash install.sh --conda` | Installs directly into the **currently active named conda env** (no venv created). Refused if the active env is `base` or no conda env is active. |
| `bash install.sh --s3` | Also installs `requirements-s3.txt`: `s3fs` + `boto3` for S3/S3-compatible access. |
| `bash install.sh --gcs` | Also installs `requirements-gcs.txt`: `gcsfs` + `google-cloud-storage` for GCS access. |

> **conda base:** the `base` env is never modified directly. With `bash install.sh`
> a venv is created that inherits base's packages via `--system-site-packages`.
> With `--conda`, base is explicitly rejected — activate a named env first.

---

### Fresh isolated environment (`install_full.sh`)

Use `install_full.sh` when you want a completely self-contained environment with
**no** packages inherited from the system or a parent conda env.  It creates the
environment from scratch, then installs all dependencies inside it.

```bash
# virtualenv (default) — stored inside the repo as venv_fno4d/
bash install_full.sh
bash install_full.sh --name .venv                        # custom path/name
bash install_full.sh --name .venv --s3                   # + S3 storage
bash install_full.sh --name .venv --gcs                  # + Google Cloud Storage
bash install_full.sh --name .venv --torch 2.5.1          # pin a torch version
bash install_full.sh --name .venv --torch 2.5.1 --s3     # pin torch + S3

# conda — creates a new named conda env
bash install_full.sh --conda                                    # default name: fno4d
bash install_full.sh --conda --name fno4d_cu124                 # custom name
bash install_full.sh --conda --name fno4d_251 --torch 2.5.1     # pin torch
bash install_full.sh --conda --name fno4d_s3 --s3               # + S3
bash install_full.sh --conda --name fno4d_all --s3 --gcs        # + both cloud
```

Flags may be freely combined.  Run `bash install_full.sh --help` for details.

#### Environment options

| Flag | Behaviour |
|---|---|
| *(none)* | Creates `venv_fno4d/` in the repo root — a plain virtualenv with no inherited packages. |
| `--venv` | Same as above; explicit form of the default. |
| `--conda` | Creates a new named conda env (`python=3.11`) via `conda create`. |
| `--name NAME` | Override the environment name/path (`venv_fno4d` or `fno4d` by default). |
| `--torch VERSION` | Pin a specific PyTorch version, e.g. `--torch 2.5.1`. Default: latest available for the detected CUDA / platform. `torchvision` is resolved automatically. |
| `--s3` | Also installs `requirements-s3.txt`: `s3fs` + `boto3`. |
| `--gcs` | Also installs `requirements-gcs.txt`: `gcsfs` + `google-cloud-storage`. |

> The environment must not already exist — the script exits with an error if it
> does, so accidental overwrites are impossible.

#### Activate after install

```bash
# virtualenv
source venv_fno4d/bin/activate   # or whatever name you chose

# conda
conda activate fno4d             # or whatever name you chose
```

#### Register as a Jupyter kernel

After activating the environment, run `install_kernel.sh` to register it as a
Jupyter kernel so it appears in JupyterLab / Jupyter Notebook:

```bash
bash install_kernel.sh
```

The kernel name is derived automatically from environment variables:

| Active environment | Name source |
|---|---|
| `conda activate fno4d` | `$CONDA_DEFAULT_ENV` → `fno4d` |
| `source venv_fno4d/bin/activate` | `basename $VIRTUAL_ENV` → `venv_fno4d` |

Both the name and the display label can be overridden:

```bash
# Override the kernel identifier (used internally by Jupyter)
KERNEL_NAME=fno4d_cu124 bash install_kernel.sh

# Override the display name shown in the Jupyter UI
KERNEL_DISPLAY_NAME="FNO4d CUDA 12.4" bash install_kernel.sh

# Override both
KERNEL_NAME=fno4d_cu124 KERNEL_DISPLAY_NAME="FNO4d CUDA 12.4" bash install_kernel.sh
```

The script installs `ipykernel` automatically if it is not already present and
registers the kernel for the current user (`--user`), so it is visible
regardless of how Jupyter was launched.

```bash
jupyter kernelspec list              # verify the kernel was registered
jupyter kernelspec remove fno4d     # remove it later if needed
```

---

#### Mac Silicon (arm64)

Installs `torch` + `torchvision` (MPS-capable, standard PyPI), `mlx>=0.16`
(Metal GPU), all base deps, and three editable local packages:
`loc_libs`, `neuraloperator-4d-2.0.0`, and `neuralop_mlx` (repo root).

#### Linux / Lambda (x86\_64 or aarch64)

1. Checks if `torch` is already importable (Docker image / conda / system site
   inherited via `--system-site-packages`).
2. If not found, detects CUDA version via `nvidia-smi` and installs the
   matching PyTorch wheel:

   | CUDA version | Wheel tag |
   |---|---|
   | 11.8 | `cu118` |
   | 12.1 – 12.3 | `cu121` |
   | 12.4 – 12.7 | `cu124` |
   | 12.8 – 12.x | `cu128` |
   | 13.x | `cu130` (probed first, falls back to `cu128` with a warning) |
   | No GPU | CPU-only wheel (training will be slow) |

3. Installs `torch-harmonics` (needed for DISCO layers in `neuraloperator`).
4. Installs base deps, `loc_libs`, and `neuraloperator-4d-2.0.0`.

> **Note:** `mlx` and `neuralop_mlx` are **not** installed on Linux —
> they are Apple Silicon only.

#### What else `install.sh` sets up

Beyond Python packages the script also configures two developer-experience tools:

**nbstripout** — strips Jupyter notebook outputs before every `git commit`.
Notebook outputs (cell results, tracebacks, memory stats) are never stored in
git history, keeping diffs readable and the repo small.  The filter is
registered in the local `.git/config` and the `.gitattributes` file (committed
to the repo) tells git which files to filter.

```bash
# strip + refresh all notebooks manually (only needed once after install,
# or if you add a new notebook that already has outputs):
git add --renormalize '*.ipynb'
git commit -m "strip notebook outputs"
```

**VS Code — Run on Save** (`emeraldwalk.runonsave`) — automatically runs
`git add <notebook>` every time you save a `.ipynb` file in VS Code.  This
keeps git's stat cache in sync with the nbstripout filter so notebooks never
show as "modified" in the Source Control panel after running cells.

The extension is listed in `.vscode/extensions.json` so VS Code will prompt
you to install it when you open the repo.  To install it manually:

```
Extensions panel → search "Run on Save" (emeraldwalk.runonsave) → Install
```

> **JupyterLab users:** the Run on Save extension only works inside VS Code.
> In JupyterLab the same effect can be achieved by adding a post-save hook to
> `~/.jupyter/jupyter_server_config.py`:
>
> ```python
> import subprocess, os
> def _strip_on_save(model, os_path, contents_manager, **kwargs):
>     if model.get("type") == "notebook":
>         subprocess.run(["git", "add", os_path],
>                        cwd=os.path.dirname(os_path) or ".",
>                        capture_output=True, timeout=10)
> c.FileContentsManager.post_save_hook = _strip_on_save
> ```

#### Enabling nbstripout without running `install.sh`

The `.gitattributes` rule (`*.ipynb filter=nbstripout`) is already committed to
the repo and clones correctly, but the **filter driver** (the command git
actually calls) must be registered once per clone in `.git/config`.  The
install script does this automatically; if you skip the script, do it manually:

**1. Install the package** (skip if already installed)

```bash
pip install nbstripout        # inside your active conda env / virtualenv
# or
conda install -c conda-forge nbstripout
```

**2. Register the filter driver for this clone**

```bash
cd /path/to/FNO4d
nbstripout --install
```

This writes the driver into `.git/config` — do **not** pass `--attributes`,
because `.gitattributes` already has the correct entries tracked in the repo.

**3. Verify**

```bash
nbstripout --status
# → nbstripout is installed in repository '...'
```

**Optional — install globally** so the driver is active for every repo on your
machine and you never need to repeat step 2 after a clone:

```bash
nbstripout --install --global
```

> **Conda users:** make sure the env that contains `nbstripout` is active
> before running `nbstripout --install`, or use the full path, e.g.
> `/opt/homebrew/Caskroom/miniconda/base/bin/nbstripout --install`.

---

### Manual install (without the script)

```bash
# Activate your env first (conda or virtualenv), then:

# Mac Silicon
pip install -r requirements-mac.txt

# Linux
pip install -r requirements-linux.txt
# then handle torch manually if not already present

# Both platforms — local editable packages
pip install -e ./loc_libs
pip install -e ./neuraloperator-4d-2.0.0

# Mac only
pip install -e .          # installs neuralop_mlx from repo root

# Cloud storage (either platform, opt-in)
pip install -r requirements-s3.txt   # S3 / S3-compatible (s3fs, boto3)
pip install -r requirements-gcs.txt  # Google Cloud Storage
```

> **Important (Mac / neuralop\_mlx):** always run `pip install -e .` from the
> **repo root**, not from inside `neuralop_mlx/`. The `pyproject.toml` at the
> root is the correct entry point.

### Pre-flight checks

Before running an install script you can do a dry run that mirrors every probe
the installer performs — without creating an environment or downloading
anything.  Exit code equals the number of failures (0 = all clear).

| Script | Pairs with | Notes |
|---|---|---|
| `check_install.sh` | `install.sh` | Creates a temporary `--system-site-packages` probe venv so already-installed packages are visible, matching `install.sh`'s behaviour. |
| `check_install_full.sh` | `install_full.sh` | Probes the resolution of a fully isolated environment (no inherited packages). |

Pass the **same flags** you plan to use with the matching installer:

```bash
# Pair with install.sh
bash check_install.sh                   # default venv mode
bash check_install.sh --conda           # --conda mode
bash check_install.sh --s3 --gcs        # + cloud backends
bash check_install.sh --conda --s3      # conda + S3

# Pair with install_full.sh
bash check_install_full.sh --venv                           # virtualenv (default)
bash check_install_full.sh --conda                          # conda env
bash check_install_full.sh --conda --name fno4d_s3 --s3     # conda + S3
bash check_install_full.sh --venv  --torch 2.5.1 --s3 --gcs # pin torch + both cloud
```

### Verify the install

```bash
# Mac
python -c "import neuralop_mlx; import neuralop; import mlx.core; import torch; print('OK')"

# Linux
python -c "import neuralop; import torch; print('CUDA:', torch.cuda.is_available())"

# Both — check loc_libs
python -c "from loc_libs.read_3d_root import read_3d_data; print('loc_libs OK')"
```

---

## Cloud storage

Cloud storage backends are **opt-in** — install them with `--s3` / `--gcs` flags
or the corresponding requirements files:

| Backend | Install flag | Requirements file | Packages |
|---|---|---|---|
| AWS S3 / S3-compatible | `--s3` | `requirements-s3.txt` | `s3fs`, `boto3` |
| Google Cloud Storage | `--gcs` | `requirements-gcs.txt` | `gcsfs`, `google-cloud-storage` |

Once installed, `read_3d_root.py` and `read_3d_hdf5.py` accept a `storage_options`
dict forwarded directly to `uproot`, enabling transparent remote I/O via fsspec.

```python
from loc_libs.read_3d_root import read_3d_data

# AWS S3 or S3-compatible (Lambda storage, MinIO, …)
data = read_3d_data(
    "s3://my-bucket/dAu_250ev_mb.root",
    storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY",
                     "endpoint_url": "https://..."},  # omit for AWS
)

# Google Cloud Storage
data = read_3d_data(
    "gs://my-bucket/dAu_250ev_mb.root",
    storage_options={"token": "path/to/service_account.json"},
)
```

The packages `s3fs`, `boto3`, `gcsfs`, and `google-cloud-storage` are
installed only when the corresponding `--s3` / `--gcs` flag is passed to
`install.sh` or `install_full.sh` (or the matching requirements file is
installed manually).

For GCS specifically, dedicated end-to-end training scripts are available in
`train_gcs/` — they handle authentication and the choice between streaming
and downloading automatically.  See [README_train_gcs.md](README_train_gcs.md).

---

## Training

### Edit parameters

All training scripts keep their parameters at the **top of the file** for easy
editing before each run:

```python
input_dir   = '/path/to/data'
input_files = ['dAu_250ev_mb.root']
n_epochs    = 50
batch_size  = 1
hidden_channels = 16
n_modes     = [32, 32, 16, 16]   # one entry per spatial dimension
lr          = 8e-3
weight_decay= 1e-4
nevents_in  = -1                  # -1 = all events
train_ratio = 0.8
```

### PyTorch path (Linux / Lambda)

See [README_train.md](README_train.md) for a full description of parameters,
input format, output files, and the `loc_libs` helpers used.

```bash
cd train
python test_train.py
# or open test_train.ipynb / test_train_hdf5.ipynb in Jupyter
```

Output: `log.log`, `best_model_state_dict.pt`

For large datasets, the HDF5-backed pipeline avoids loading all events into
memory at once:

```bash
# First run: converts ROOT → HDF5 (one-time, accepts 2× peak memory)
# Subsequent runs: true partial I/O — reads only the requested events
python -c "
from loc_libs.read_3d_hdf5 import convert_root_to_hdf5
convert_root_to_hdf5('data/dAu_250ev_mb.root', 'data/dAu_250ev_mb.h5')
"
# then run test_train_hdf5.ipynb
```

### MLX path (Mac Silicon)

```bash
cd train
python test_train_mlx.py
# or open test_train_mlx.ipynb in Jupyter
```

Output: `log_mlx.log`, `best_model.safetensors`

### GCS training (cloud data)

When ROOT files live in Google Cloud Storage, the `train_gcs/` scripts read
them directly without requiring a local copy.  Two modes are available:

| Mode | When to use |
|------|------------|
| **Streaming** | One-off runs, low-disk instances — no data lands on disk |
| **Download** | Repeated runs on the same files — downloads once, reuses the local cache |

```bash
cd train_gcs/

# edit config_gcs_stream.yaml: set gcs_key_file and gs:// input_files
python test_train_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml     # PyTorch
python test_train_mlx_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml # MLX

# edit config_gcs_download.yaml: set gcs_key_file, local_data_dir, gs:// input_files
python test_train_yaml_norm_gcs_download.py --config config_gcs_download.yaml     # PyTorch
python test_train_mlx_yaml_norm_gcs_download.py --config config_gcs_download.yaml # MLX
```

GCS packages must be installed first (`bash install.sh --gcs`), and a
service-account JSON key file is required.  Full setup instructions and all
new YAML keys are documented in **[README_train_gcs.md](README_train_gcs.md)**.

---

### Inference / apply

`train/apply.py` is a [marimo](https://marimo.io) reactive notebook for
loading a saved checkpoint and visualising predictions:

```bash
marimo run train/apply.py
# or to edit:
marimo edit train/apply.py
```

### Evaluation

Six marimo evaluation notebooks live in `train/`.  See [README_eval.md](README_eval.md)
for the visual/contour notebooks and [README_eval_metric.md](README_eval_metric.md)
for the full metric suite.

| Script | Framework | Output |
|---|---|---|
| `eval_model_test_torch.py` | PyTorch | `imshow` field snapshots |
| `eval_model_test_mlx.py` | MLX | `imshow` field snapshots |
| `eval_model_test_cont_torch.py` | PyTorch | skinnycontour CDF overlays |
| `eval_model_test_cont_mlx.py` | MLX | skinnycontour CDF overlays |
| `eval_model_test_metric_torch.py` | PyTorch | nRMSE, fRMSE, MELR, WLR, spectral maps |
| `eval_model_test_metric_mlx.py` | MLX | nRMSE, fRMSE, MELR, WLR, spectral maps |

```bash
cd train/
marimo run eval_model_test_metric_torch.py   # full metric suite — PyTorch
marimo run eval_model_test_metric_mlx.py     # full metric suite — MLX
```

Marimo is not installed by default; run `bash install_marimo.sh` after
activating the FNO4d environment (see [README_eval.md](README_eval.md#installing-marimo-optional)).

---

## Model evaluation

`loc_libs/model_eval.py` provides a framework-agnostic `ModelEval` class that
computes the complete evaluation battery recommended for neural fluid-dynamics
surrogates.  It works on raw NumPy arrays and has no dependency on PyTorch or MLX.

### Metric categories

| Category | Metrics | What they catch |
|---|---|---|
| **Pointwise** | nRMSE, RMSE | Overall field accuracy — the headline number |
| **Pointwise** | MaxError | Worst-case local failure (freeze-out surface spikes) |
| **Pointwise** | bRMSE | Accuracy at the domain boundary (freeze-out proxy) |
| **Pointwise** | cRMSE | Violations of global energy-momentum conservation (ε channel) |
| **Pointwise** | vRMSE | Whether spatial contrast (not just mean level) is reproduced |
| **Spectral** | fRMSE (low/mid/high) | Scale-resolved errors; catches FNO over-smoothing |
| **Spectral** | MELR | Spectral energy bias (negative = over-smooth; positive = freeze-out leakage) |
| **Spectral** | WLR | Spectral shape distortion independent of total energy |
| **Spectral** | Spectral error map | 2-D Fourier-space failure pattern |
| **Spectral** | Energy spectrum | log-log E(K) cascade fidelity over time |
| **Temporal** | AR rollout | Error accumulation under autoregressive stepping |

Standard training losses (L² or H¹) collapse all of the above into a single scalar
that conflates scales and locations.  A model that predicts bulk flow correctly but
destroys sub-nucleon eccentricity fluctuations will appear "good" by nRMSE while
producing wrong flow harmonics v₃ and higher.  The spectral metrics are the
quantitative check.

### Quick usage

```python
import sys, numpy as np
sys.path.append('loc_libs')
from model_eval import ModelEval

# pred, true: (N, C, nx, ny, neta, T)  — physical units, initial condition excluded
ev = ModelEval(
    pred             = model_out_phys,
    true             = y_true_phys,
    tau_coords       = tau_out_coords,   # fm/c
    eta_coords       = eta_coords,
    channel_names    = ['ε', 'vx', 'vy', 'vη'],
    active_threshold = 0.3,  # mask freeze-out cells (≈ T_c → ε_c ≈ 0.3 GeV/fm³)
)

# All scalar metrics at once
print(ev.summary())
# {'nRMSE': 0.084, 'MaxError': 0.31, 'bRMSE': 0.087,
#  'cRMSE_e0': 0.021,      ← ε channel only — conservation law test
#  'cRMSE_all': 0.035,     ← all channels, L1-floored denominator
#  'fRMSE_low': 0.070, 'fRMSE_mid': 0.086, 'fRMSE_high': 0.123,
#  'MELR': -0.152, 'WLR': 0.087,
#  'active_frac': 0.215}   ← 21.5 % of cells above freeze-out threshold

# 2×3 dashboard figure
fig = ev.plot_dashboard(channel=0, ieta=0)
fig.savefig('dashboard.pdf', bbox_inches='tight')

# Save for reporting
import json
with open('out/metrics_eval.json', 'w') as f:
    json.dump(ev.summary(), f, indent=2)
```

> **Sparse-field note:** without `active_threshold`, normalized metrics (nRMSE,
> vRMSE, fRMSE) can reach thousands on real hydro data because the many freeze-out
> zero cells collapse the denominator.  Set `active_threshold` to the QCD crossover
> energy density (≈ 0.3 GeV/fm³) to restrict averages to the hot active core.
> The marimo notebooks expose this as an interactive slider (default 0.3).

The `fRMSE_high` being noticeably larger than `fRMSE_low` and a negative `MELR`
are the standard FNO signature of high-frequency over-smoothing.  A positive `MELR`
indicates the model is leaking predictions into freeze-out cells.

### Minimum publication checklist

```
active_frac        — quote so readers know what fraction of the field was evaluated
nRMSE              — headline pointwise accuracy
fRMSE low/mid/high — scale-resolved error; exposes smoothing invisible to nRMSE
MELR               — signed spectral energy bias (negative = over-smooth; positive = leakage)
cRMSE_e0           — energy conservation test (ε channel only)
AR rollout curve   — nRMSE vs τ under autoregressive stepping (1-step models)
```

Add v₂/v₃/⟨pT⟩ (via Cooper-Frye freeze-out, not yet automated) as the final
physics gate.

### Documentation

Full mathematical definitions, physical motivation, code examples, and API
reference: **[README_eval_metric.md](README_eval_metric.md)**

---

## Packages

### `neuralop_mlx` — MLX FNO (Apple Silicon)

```python
from neuralop_mlx.models import FNO
import mlx.core as mx

model = FNO(
    n_modes=[32, 32, 16, 16],   # 4D
    in_channels=4,
    out_channels=4,
    hidden_channels=16,
    n_layers=4,
    positional_embedding="grid",
)

x   = mx.array(...)            # shape: (batch, channels, d0, d1, d2, d3)
out = model(x)
mx.eval(out)                   # MLX is lazy — materialise before reading
```

**Weight transfer from a trained PyTorch model:**

```python
from neuralop_mlx import load_from_torch

mlx_model = load_from_torch("best_model_state_dict.pt", mlx_model)
# also accepts a live torch.nn.Module or a {'model_state_dict': …} dict
```

**Losses and optimiser:**

```python
from neuralop_mlx import LpLoss, H1Loss, AdamW

loss_fn = LpLoss(d=4, p=2)           # relative L2 over 4 spatial dims
loss_fn = H1Loss(d=4)                # Sobolev H1 loss

optimizer = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
```

### `neuralop` — PyTorch neuraloperator (local 4D fork)

```python
from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.layers.embeddings import GridEmbeddingND

model = FNO(n_modes=(32, 32, 16, 16), hidden_channels=16,
            in_channels=4, out_channels=4)
```

### `loc_libs` — data I/O helpers

```python
from loc_libs.read_3d_root  import read_3d_data, coarsen_array
from loc_libs.read_3d_hdf5  import convert_root_to_hdf5, read_3d_data_hdf5
from loc_libs.root_pytorch_loader import make_dataloaders
from loc_libs.parse_training_log  import plot_training_log
from loc_libs.skinnycontour       import find_skinny_contours

# GCS helpers (requires pip install -r requirements-gcs.txt or --gcs flag)
from loc_libs.gcs_utils import get_gcs_storage_options  # streaming via fsspec
from loc_libs.gcs_utils import download_gcs_files        # download specific files
from loc_libs.gcs_utils import download_gcs_dir          # download a GCS prefix (with pattern filter)
from loc_libs.gcs_utils import download_gcs_prefix       # download a GCS prefix (flat or structured)
from loc_libs.gcs_utils import upload_gcs_file           # upload a single file
from loc_libs.gcs_utils import upload_gcs_dir            # upload a local directory (with pattern filter)
```

`read_3d_data` returns a dict with:
- `arr` — NumPy array of shape `(nevents, nFeatures, nx, ny, neta, ntau)`
- Grid parameters: `nx`, `ny`, `neta`, `dx`, `dy`, `deta`, `tau_min`, `dtau`, …

---

## Tests

```bash
# from repo root
python -m pytest tests/ -v
# expected: 48 passed (neuralop_mlx unit + integration tests)
```

---

## Porting status (PyTorch → MLX)

| Component | Status |
|---|---|
| `FNO` (dense weights, N-dimensional) | ✅ Ported, tested |
| `SpectralConvND`, `FNOBlocks`, `ChannelMLP` | ✅ Ported, tested |
| `GridEmbeddingND`, skip connections | ✅ Ported, tested |
| `Trainer`, `AdamW`, `LpLoss`, `H1Loss` | ✅ Ported, tested |
| `load_from_torch` (weight migration) | ✅ Ported, tested |
| `TFNO` (Tucker/CP/TT factorized) | ❌ Blocked — no `tltorch` in MLX |
| `UNO`, `RNO` | ❌ Not yet ported |
| `SFNO`, DISCO layers (`LocalNO`) | ❌ Blocked — depends on `torch_harmonics` |
| `FNOGNO`, `GINO`, `CODANO` | ❌ Blocked — requires scatter ops (not in MLX) |

---

## Credits & Attribution

### `neuraloperator` — PyTorch package

The PyTorch FNO implementation in [neuraloperator-4d-2.0.0/](neuraloperator-4d-2.0.0/) is a
**minimal fork** of the upstream [neuraloperator](https://github.com/neuraloperator/neuraloperator)
library by Kossaifi, Kovachki, Li, et al.  Only the **differentiation and loss subsystems** were
extended to support **four-dimensional fields** (x, y, z, t); all other code is unchanged from the
original.  The specific modifications are documented in:

- [neuraloperator-4d-2.0.0/doc4dImp/Fno4dPlan.md](neuraloperator-4d-2.0.0/doc4dImp/Fno4dPlan.md) — implementation plan
- [neuraloperator-4d-2.0.0/doc4dImp/Fno4dImplementation.md](neuraloperator-4d-2.0.0/doc4dImp/Fno4dImplementation.md) — detailed record of every change made

For full documentation, tutorials, and the upstream API reference see the
[neuraloperator documentation site](https://neuraloperator.github.io/dev/index.html).

If you use the PyTorch FNO in your work, please cite the upstream library papers:

**Library paper** ([arXiv:2412.10354](https://arxiv.org/abs/2412.10354))
```bibtex
@article{kossaifi2025librarylearningneuraloperators,
  author  = {Jean Kossaifi and Nikola Kovachki and Zongyi Li and David Pitt and
             Miguel Liu-Schiaffini and Robert Joseph George and Boris Bonev and
             Kamyar Azizzadenesheli and Julius Berner and Valentin Duruisseaux and
             Anima Anandkumar},
  title   = {A Library for Learning Neural Operators},
  journal = {arXiv preprint arXiv:2412.10354},
  year    = {2025},
}
```

**FNO practical guide** ([arXiv:2512.01421](https://arxiv.org/abs/2512.01421))
```bibtex
@article{duruisseaux2025guide,
  author  = {Valentin Duruisseaux and Jean Kossaifi and Anima Anandkumar},
  title   = {Fourier Neural Operators Explained: A Practical Perspective},
  journal = {arXiv preprint arXiv:2512.01421},
  year    = {2025},
}
```

**Neural operator theory** ([arXiv:2108.08481](https://arxiv.org/abs/2108.08481))
```bibtex
@article{kovachki2023neuraloperator,
  author    = {Nikola Kovachki and Zongyi Li and Burigede Liu and
               Kamyar Azizzadenesheli and Kaushik Bhattacharya and
               Andrew Stuart and Anima Anandkumar},
  title     = {Neural Operator: Learning Maps Between Function Spaces
               with Applications to PDEs},
  journal   = {Journal of Machine Learning Research},
  volume    = {24},
  number    = {1},
  articleno = {89},
  year      = {2023},
}
```

### `neuralop_mlx` — MLX port (experimental)

> **Warning:** The MLX port (`neuralop_mlx/`) is **experimental**.  The API may change, not
> all upstream `neuraloperator` features are ported (see [Porting status](#porting-status-pytorch--mlx)),
> and it has not been validated on the same range of datasets as the PyTorch stack.

The MLX port is an independent reimplementation of the FNO and its training infrastructure in
[Apple MLX](https://github.com/ml-explore/mlx), written to match the behaviour of the PyTorch
stack on Apple Silicon.  It is not a mechanical translation of the upstream `neuraloperator`
source.  Weight transfer from trained PyTorch checkpoints is supported via `load_from_torch`
(see [README_mlx.md](README_mlx.md) and [PORTING_NOTES.md](PORTING_NOTES.md)).
