# FNO4d — 4D Fourier Neural Operator

Training pipeline for a 4-dimensional Fourier Neural Operator (FNO) applied to
relativistic heavy-ion collision data (hydrodynamic initial conditions → final
state observables).

Two parallel training stacks are provided:

| Stack | Package | GPU backend | Target |
|---|---|---|---|
| PyTorch | `neuralop` (local fork) | CUDA (Linux) / CPU (Mac) | Lambda GPU instances, HPC |
| MLX | `neuralop_mlx` | Metal (Apple Silicon) | Mac M-series |

---

## Repository layout

```
FNO4d/
├── install.sh                   # one-shot install script (see below)
├── requirements-base.txt        # shared deps (both platforms)
├── requirements-mac.txt         # Mac Silicon: adds torch + mlx
├── requirements-linux.txt       # Linux/Lambda: adds nothing (torch auto-detected)
│
├── neuralop_mlx/                # MLX FNO implementation (Apple Silicon)
│   ├── models/fno.py            # FNO model (N-dimensional)
│   ├── layers/                  # spectral conv, channel MLP, embeddings, …
│   ├── training/                # Trainer, AdamW, LpLoss, H1Loss
│   └── weight_transfer.py       # load_from_torch() — PyTorch → MLX migration
│
├── neuraloperator-4d-2.0.0/     # PyTorch neuraloperator (local 4D fork)
│
├── loc_libs/                    # local helper library (data I/O, visualisation)
│   ├── read_3d_root.py          # read events from ROOT files (local or cloud)
│   ├── read_3d_hdf5.py          # ROOT → HDF5 conversion + streaming reads
│   ├── root_pytorch_loader.py   # IterableDataset for large ROOT files
│   ├── parse_training_log.py    # parse and plot training logs
│   ├── skinnycontour.py         # contour utilities
│   └── NormFn.py                # normalisation helpers
│
└── train/                       # training entry points
    ├── test_train.py            # PyTorch training script
    ├── test_train.ipynb         # PyTorch training notebook
    ├── test_train_mlx.py        # MLX training script (Mac)
    ├── test_train_mlx.ipynb     # MLX training notebook (Mac)
    ├── test_train_hdf5.ipynb    # HDF5-backed PyTorch training notebook
    ├── test_mem_data_load.py    # memory profiling — ROOT loader
    ├── test_mem_data_load_hdf5.py # memory profiling — HDF5 loader
    └── apply.py                 # marimo inference app
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- **Mac:** Xcode command-line tools (`xcode-select --install`)
- **Linux/Lambda:** NVIDIA driver + `nvidia-smi` in `PATH` (for CUDA detection)
- `git clone` this repo and `cd` into it

### One-shot install

```bash
bash install.sh
```

The script auto-detects your platform and environment:

| Condition | Behaviour |
|---|---|
| `$CONDA_PREFIX` is set | Installs into the **active conda env** — no venv created |
| No conda env active | Creates `venv/` in the repo root, activates it |
| Linux venv | Created with `--system-site-packages` to inherit container torch |

#### Mac Silicon (arm64)

Installs `torch` + `torchvision` (MPS-capable, standard PyPI), `mlx>=0.16`
(Metal GPU), all base deps, and three editable local packages:
`loc_libs`, `neuraloperator-4d-2.0.0`, and `neuralop_mlx` (repo root).

#### Linux / Lambda (x86\_64 or aarch64)

1. Checks if `torch` is already importable (Docker image / conda / system site).
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
```

> **Important (Mac / neuralop\_mlx):** always run `pip install -e .` from the
> **repo root**, not from inside `neuralop_mlx/`. The `pyproject.toml` at the
> root is the correct entry point.

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

`read_3d_root.py` and `read_3d_hdf5.py` accept a `storage_options` dict that
is forwarded directly to `uproot`, enabling transparent remote I/O via fsspec.

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
included in `requirements-base.txt` and installed on both platforms.

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

```bash
cd train
python test_train.py
# or open test_train.ipynb / test_train_hdf5.ipynb in Jupyter
```

Output: `log.txt`, `best_model_state_dict.pt`

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

Output: `log_mlx.txt`, `best_model.safetensors`

### Inference / apply

`train/apply.py` is a [marimo](https://marimo.io) reactive notebook for
loading a saved checkpoint and visualising predictions:

```bash
marimo run train/apply.py
# or to edit:
marimo edit train/apply.py
```

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
