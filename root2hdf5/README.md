# root2hdf5 — Jagged ROOT to Fixed-Length HDF5

Conversion guide for heavy-ion hydrodynamic data:
from variable-length (jagged) ROOT files to fixed-shape, chunked HDF5
files suitable for machine learning pipelines.

---

## Which example to use

Two scripts are provided; they produce HDF5 files in the **same format**
so `read_3d_data_hdf5()` works on the output of either:

| | [`convert_example.py`](convert_example.py) | [`convert_example_production.py`](convert_example_production.py) |
|---|---|---|
| **Dependencies** | `uproot`, `awkward`, `h5py`, `numpy` only | Full FNO4d install (`loc_libs`) |
| **FNO4d required** | No | Yes |
| **Use on non-GPU batch machine** | Yes — lightweight, easy to `pip install` | Needs PyTorch / MLX stack |
| **ROOT format** | Synthetic / custom (adapt as needed) | MUSIC output format (TParameter scalars, `t/user_res` branch) |
| **Remote files (S3/GCS)** | No | Yes |
| **int32-safe large-file batching** | No | Yes |
| **Memory tracing** | Basic (tracemalloc peak per step) | Full (tracemalloc snapshots before/after every batch) |
| **When to reach for it** | Quick one-off conversions; batch nodes without the full ML stack; learning the mechanics | Routine use in the FNO4d project with real MUSIC data |

**Recommended workflow:**
- Convert once on a data-processing machine (CPU node, login node, etc.)
  using `convert_example.py` or `loc_libs/read_3d_hdf5.py` — whichever is
  easier to install there.
- Train and evaluate on the GPU node reading from the HDF5 file.

---

## The problem in one sentence

Each hydro event runs until its own freeze-out time τ_fo, so the time
dimension is *variable* — and neural networks cannot batch tensors of
different shapes.

---

## Background: why ROOT files are jagged here

The MUSIC hydrodynamics code writes one ROOT entry per collision event.
Each event is stored as a flat float32 vector of length

```
ntau_i  ×  nx  ×  ny  ×  neta  ×  nFeatures
```

where

| Symbol      | Meaning                                              | Typical value |
|-------------|------------------------------------------------------|---------------|
| `ntau_i`    | number of hydrodynamic time steps for event *i*      | varies (e.g. 5–12) |
| `nx`, `ny`  | transverse grid points (x, y)                        | 20–40 each |
| `neta`      | longitudinal grid points (space-time rapidity η)     | 10–20 |
| `nFeatures` | physical fields per cell: ε, v_x, v_y, v_η           | 4 |

Because each nucleus–nucleus collision reaches QCD freeze-out at a
different proper time, `ntau_i` differs from event to event.
uproot exposes this branch as an *awkward-array* — a ragged list of
1-D arrays with unequal lengths:

```python
import uproot, awkward as ak
f      = uproot.open("data.root")
arr_ak = f["t"]["user_res"].array(library="ak")
# len(arr_ak[0]) != len(arr_ak[1])  ← jagged
```

---

## Why this is a problem for machine learning

### 1. Fixed-shape tensors are required for batching

Every layer in a neural network — convolutions, attention, Fourier
transforms — operates on tensors with a compile-time-fixed shape.
PyTorch's `DataLoader` calls `torch.stack` under the hood, which
requires all samples in a batch to have identical dimensions:

```python
torch.stack([event_0, event_1, event_2])   # raises RuntimeError if shapes differ
```

There is no standard path from a jagged array to a batched tensor
without first imposing a common length along the variable dimension.

### 2. ROOT baskets make random-access reads expensive

The MUSIC writer flushes all events into a single ROOT basket (no
`SetAutoFlush(1)` call).  When uproot reads *any* slice of that branch,
it must decompress the entire basket first:

```
requested: events [10:20]   (10 events)
actual I/O: decompress entire basket  (all N events)
peak memory ≈ 2× full array size
```

This is workable for a one-time conversion, but a training loop that reads
many small random batches hits the same 2× peak on every call.

### 3. HDF5 chunked storage gives true O(chunk) random access

HDF5 lets you specify a *chunk shape* for each dataset.  With
`chunk_events=1`, every event occupies its own compressed chunk:

```
requested: events [10:20]
actual I/O: decompress exactly 10 chunks  (10 events)
peak memory ∝ batch_size,  not full_file_size
```

A shuffled `DataLoader` that picks 32 random events decompresses
exactly 32 chunks — the rest of the file is never touched.

### 4. HDF5 is universally supported by ML frameworks

| Framework     | HDF5 support |
|---------------|--------------|
| PyTorch       | `h5py` + standard `Dataset` |
| TensorFlow    | `tf.data` HDF5 connector |
| JAX / MLX     | load via `h5py`, pass numpy arrays |
| scikit-learn  | `h5py` directly |

ROOT requires uproot (Python) or the C++ ROOT runtime — neither is a
standard ML dependency.

### 5. Zero-padding is physically motivated

A cell that has passed freeze-out carries zero energy density and zero
velocity.  Padding with 0.0 is not an arbitrary bookkeeping choice —
it exactly represents the physical state of the plasma after
hadronization.  A neural network trained on padded data implicitly
learns that the freeze-out surface is the boundary between active and
dead regions.

---

## Conversion strategy

```
┌──────────────────────────────────────────────────────────────┐
│  ROOT file (jagged)                                          │
│  t/user_res branch: list of float32 vectors, lengths vary    │
│                                                              │
│  event 0: [f0, ..., f_{nt0 × K - 1}]   nt0 = 7              │
│  event 1: [f0, ..., f_{nt1 × K - 1}]   nt1 = 12             │
│  event 2: [f0, ..., f_{nt2 × K - 1}]   nt2 = 5              │
│  ...                                                         │
│  K = nx × ny × neta × nFeatures                             │
└──────────────────────────────────────────────────────────────┘
         │
         │  1. choose_ntau = max(ntau_i)  (or user-supplied)
         │  2. pad shorter events with 0.0 to length choose_ntau × K
         │  3. clip longer events to choose_ntau × K
         │  4. reshape: (N, choose_ntau × K) → (N, ntau, nx, ny, neta, nF)
         │  5. reorder axes: → (N, nFeatures, nx, ny, neta, ntau)
         │     standard ML layout: (batch, channels, spatial..., time)
         ▼
┌──────────────────────────────────────────────────────────────┐
│  HDF5 file (fixed-shape, chunked)                            │
│  dataset 'arr': shape (N, nF, nx, ny, neta, ntau)            │
│                 dtype float32                                 │
│                 chunks (1, nF, nx, ny, neta, ntau)  per event │
│                 compression: lzf (fast) or gzip (smaller)    │
│                                                              │
│  attributes: nx, ny, neta, nFeatures, dtau, tau_min,         │
│              choose_ntau, nevents                            │
│  datasets:   ntau_freezeout, tau_freezeout  (per-event)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Quick start

### Standalone (lightweight, no FNO4d required)

Install the minimal dependencies — no PyTorch, no GPU stack:

```bash
pip install uproot awkward h5py numpy
```

Run the self-contained example (creates a synthetic ROOT file, converts
it, reads back a slice, and verifies round-trip correctness):

```bash
cd root2hdf5
python convert_example.py
```

Sample output:

```
Creating synthetic ROOT file: synthetic.root
  20 events, ntau range [5, 12]
  grid: nx=8, ny=8, neta=4, nFeatures=4

--- Inspecting ROOT file: synthetic.root ---
  ntau per event: min=5, max=12  ← variable-length (jagged) dimension

  Per-event flat lengths (first 5 events):
    event 0: ntau=5   flat_len=5120   (= 5 × 8 × 8 × 4 × 4 = 5120)
    event 1: ntau=11  flat_len=11264  (= 11 × 8 × 8 × 4 × 4 = 11264)
    ...

--- Converting synthetic.root → synthetic.h5 ---
  choose_ntau = 12  (19 events zero-padded, 0 clipped)
  output shape : (20, 4, 8, 8, 4, 12)

--- Partial read from HDF5: events [0, 5) ---
  arr shape   : (5, 4, 8, 8, 4, 12)
  Peak memory : 0.24 MiB  (only 5 events loaded, not the full file)

--- Round-trip verification ---
  event 0: OK  nt=5  padding [5:12]: OK (zeros)
  All checked events match the original ROOT data.
```

Options:

```
--nev N          number of synthetic events (default: 20)
--root PATH      ROOT file path (default: synthetic.root)
--h5   PATH      HDF5 file path (default: synthetic.h5)
--chunk N        HDF5 chunk size in events (default: 1)
--ntau N         override choose_ntau; default: max(ntau_i) across all events
--skip-create    skip ROOT creation if the file already exists (use your own)
```

To convert your own ROOT file (any layout that matches the synthetic format):

```bash
python convert_example.py --root /path/to/your.root --h5 /path/to/out.h5 --skip-create
```

Adapt `convert_to_hdf5()` inside the script for a different ROOT tree/branch layout.

---

### Production (FNO4d / loc_libs)

For real MUSIC ROOT files and the full feature set, use
`loc_libs/read_3d_hdf5.py` which is part of the FNO4d project:

```python
from loc_libs.read_3d_hdf5 import convert_root_to_hdf5, read_3d_data_hdf5

# One-time conversion (accepts the 2× ROOT peak)
convert_root_to_hdf5(
    "data/dAu_250ev_mb.root",
    "data/dAu_250ev_mb.h5",
    nevents      = -1,
    chunk_events = 1,
    verbose      = True,
)

# Every subsequent training run (true partial I/O, bounded peak memory)
dat = read_3d_data_hdf5("data/dAu_250ev_mb.h5", nevents=50, start_event=0)
arr = dat["arr"]   # shape: (50, nFeatures, nx, ny, neta, ntau)
```

The runnable example with remote file patterns and a PyTorch DataLoader
is in `convert_example_production.py`:

```bash
# demo mode — builds a small synthetic HDF5 automatically
python convert_example_production.py

# use an existing converted file
python convert_example_production.py --h5 /path/to/dAu_250ev_mb.h5
```

---

## Using the HDF5 file in a PyTorch training loop

Both converters write the same HDF5 schema, so this Dataset works
with either:

```python
import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class HydroHDF5Dataset(Dataset):
    """One h5py open per __getitem__; decompresses exactly one HDF5 chunk."""

    def __init__(self, h5_path, indices=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as hf:
            n = int(hf.attrs["nevents"])
            self.ntau = int(hf.attrs["choose_ntau"])
        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        with h5py.File(self.h5_path, "r") as hf:
            arr = hf["arr"][idx]            # decompresses one chunk only
        x = np.repeat(arr[..., :1], self.ntau - 1, axis=-1)   # initial condition
        y = arr[..., 1:]                                        # target steps
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

n = 250
train_set = HydroHDF5Dataset("data/dAu_250ev_mb.h5", indices=range(200))
val_set   = HydroHDF5Dataset("data/dAu_250ev_mb.h5", indices=range(200, 250))

# num_workers > 0: each worker opens its own h5py handle (safe)
loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)

for x_batch, y_batch in loader:
    # x_batch: (8, nF, nx, ny, neta, ntau-1)  initial condition, tiled
    # y_batch: (8, nF, nx, ny, neta, ntau-1)  target tau steps 1..T
    ...
```

---

## Choosing chunk size

| `chunk_events` | Read granularity                  | Typical use case |
|----------------|-----------------------------------|------------------|
| 1              | One event per chunk (finest)      | Shuffled DataLoader, large datasets |
| 4–16           | Small batches per chunk           | Sequential reads, better compression ratio |
| 64+            | Large blocks                      | Sequential scans, archival storage |

With `chunk_events=1` and `compression="lzf"`, each DataLoader worker
reads exactly `batch_size` chunks per iteration.  Increasing
`chunk_events` improves the compression ratio but forces extra chunk reads
when the DataLoader's random batch indices straddle chunk boundaries.

---

## Memory comparison: ROOT vs HDF5

| Scenario                              | ROOT (one basket)      | HDF5 (chunk_events=1) |
|---------------------------------------|------------------------|-----------------------|
| Load 10 events from 1000-event file   | decompress all 1000    | decompress 10 chunks  |
| Peak memory (N=1000, shape per event) | ~2× full array         | ~10× single event     |
| Shuffled DataLoader batch=32          | 2× full array per step | 32 chunks per step    |
| Multi-worker DataLoader               | each worker: 2× peak   | each worker: N_w chunks |

---

## Array layout

Both converters write arrays in the layout expected by the FNO model:

```
(N, nFeatures, nx, ny, neta, ntau)
 │   │          │   │   │     │
 │   │          │   │   │     └─ temporal (proper time τ)
 │   │          └───┴───┴─── spatial (x, y, η)
 │   └── physics channels: ε, v_x, v_y, v_η
 └── batch (event index)
```

This is PyTorch convention: `(batch, channels, spatial...)`.
The ROOT file stores data in `(ntau, nx, ny, neta, nFeatures)` row-major
order within each flat event vector.  The converters apply `np.moveaxis`
(or equivalently `np.transpose`) to reorder from ROOT layout to ML layout.

---

## HDF5 schema

Both converters write files with this schema so they are interchangeable:

```
/                     (root group)
  attrs:
    nx, ny, neta, nFeatures   grid dimensions
    dtau, tau_min             time-step size and starting proper time (fm/c)
    choose_ntau               fixed tau length (post-conversion)
    nevents                   number of events in this file
  datasets:
    arr               float32  (nevents, nFeatures, nx, ny, neta, choose_ntau)
                               chunked (1, nFeatures, nx, ny, neta, choose_ntau)
    ntau_freezeout    int32    (nevents,)  per-event actual tau step count
    tau_freezeout     float32  (nevents,)  per-event freeze-out proper time (fm/c)
```

---

## Files

| File                              | Description |
|-----------------------------------|-------------|
| `convert_example.py`              | Standalone: synthetic ROOT, conversion, readback, verification. Minimal dependencies — suitable for non-GPU batch machines. |
| `convert_example_production.py`   | loc_libs API: full `convert_root_to_hdf5` / `read_3d_data_hdf5` usage, remote files, PyTorch DataLoader. Requires FNO4d install. |
| `README.md`                       | This document |

Production converter (MUSIC ROOT format, remote files, large-file batching):
[`loc_libs/read_3d_hdf5.py`](../loc_libs/read_3d_hdf5.py)
