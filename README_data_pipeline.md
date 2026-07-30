# Unified data pipeline (`loc_libs/data/`)

One reading-and-preparation path for FNO 3+1D fluid data, shared by the **torch**
and **MLX** backends and by every storage backend (ROOT / HDF5 / Zarr, local or
`gs://`/`s3://`). It replaces the old forked stack — ROOT-eager vs HDF5-lazy,
four near-duplicate `Dataset` classes, per-backend loaders, per-event
normalisation — with a single chain:

```
ArraySource  →  FluidDataset  →  collate_fn  →  DataLoader
 (ROOT/HDF5/Zarr,   (raw event       (normalize +     (workers, prefetch,
  local or gs://,     reads only)      window +         pin, locality sampler;
  lazy)                                to-tensor)        shared by torch & MLX)
```

Implementation: [`loc_libs/data/`](loc_libs/data/). This is the realisation of
[`DATA_PIPELINE_PLAN.md`](DATA_PIPELINE_PLAN.md).

---

## Why

The previous code had two readers (ROOT, HDF5), four `Dataset` classes
(`FluidDataset`, `FluidDatasetMLX`, `FluidDatasetH5`, `FluidDatasetH5MLX`),
per-backend loader blocks, normalisation applied **per event** inside
`__getitem__`, and a manual "recompute the coarse grid after downsampling" step
that was easy to forget. The unified pipeline collapses all of that:

- **One `FluidDataset`** that returns raw numpy events.
- **Normalisation + windowing move into a batched `collate_fn`** — one
  vectorised call per batch instead of one per event.
- **Multi-worker reads are fork-safe** (lazy HDF5/ROOT handle caches are keyed
  by pid; see [Fork-safety](#fork-safety-multi-worker-reads)).
- **MLX gets real DataLoader workers** for the first time (it used to be hard-
  capped at `num_workers=0`).
- **Downsampling auto-binds the coarse grid** — pass one `downsample:` block and
  the normalizer is built on the coarsened grid automatically.

---

## Quick start

The single entry point is `build_dataloaders()`. Give it a *source* and the
training settings; it returns `(train_loader, test_loaders, normalizer)`.

```python
import sys; sys.path.insert(0, '../loc_libs')   # repo convention: loc_libs on path
from data import build_dataloaders
from read_3d_hdf5 import read_3d_data_hdf5

dat    = read_3d_data_hdf5(['a.h5', 'b.h5'], lazy=True)   # arr is a MultiH5Array
source = dat['arr']
grid   = {k: dat[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}

train_loader, test_loaders, normalizer = build_dataloaders(
    source,
    n_total=len(source), train_ratio=0.8,
    xyslice=slice(None, -1), etaslice=slice(None), tauslice=slice(None, -1),
    norm_cfg={'name': 'etau75_eta_3d'}, grid_params=grid,
    backend='torch', batch_size=8, test_batch_sizes=[8],
    num_workers=4,
)

for batch in train_loader:
    x, y = batch['x'], batch['y']          # torch tensors (backend='torch')
```

For MLX, pass `backend='mlx'` — the collate then returns float32 numpy arrays and
the MLX `Trainer` converts them to `mx.array`. **Everything else is identical**,
including the worker prefetch.

In the training scripts this is already wired: see
[`train_generic/train_h5.py`](train_generic/train_h5.py) (HDF5) and
[`train_generic/train.py`](train_generic/train.py) (ROOT). You configure it
entirely from YAML — see [Configuration](#configuration).

---

## The pieces

### Sources — [`data/sources.py`](loc_libs/data/sources.py)

A *source* is anything exposing `.shape`, `.dtype`, `__len__`, and
`__getitem__` (int → 5-D event `(F, nx, ny, neta, ntau)`; slice → 6-D batch).
This is the `ArraySource` protocol — structural, no inheritance needed. A plain
`np.ndarray` already satisfies it. The lazy sources:

| Source | Backend | Local | `gs://` / `s3://` | Notes |
|---|---|---|---|---|
| `np.ndarray` | in-RAM | ✓ | — | eager; just pass it |
| `MultiH5Array` | HDF5 | ✓ | download only¹ | true random-access I/O (per-event chunks) |
| `RootSource` | ROOT | ✓ | **stream** | reads only the requested entry range; pads to `choose_ntau` |
| `ZarrSource` | Zarr | ✓ | **stream** | thin wrapper over a chunked Zarr array |
| `DownsampledArrayView` | any | ✓ | ✓ | wraps any source; coarsens on read |

¹ h5py needs a local path, so HDF5 over GCS uses `data_source: gcs_download`.
ROOT and Zarr stream directly via fsspec — no download.

```python
from data import RootSource, ZarrSource
src = RootSource(['evt.root'], choose_ntau=None)            # None = global max ntau
src = RootSource(['gs://bucket/evt.root'], storage_options=get_gcs_storage_options(key))
src = ZarrSource('gs://bucket/data.zarr', key='arr', storage_options=...)
```

`RootSource[a:b]` is byte-for-byte equal to the matching slice of the old eager
`read_3d_data` array, including the pad-to-`choose_ntau`.

### `FluidDataset` — [`data/dataset.py`](loc_libs/data/dataset.py)

Index/slice view over a source. Returns a **raw, unnormalised, un-windowed**
event with the spatial/eta/tau slices applied. Train/val splits are expressed as
index ranges (`np.arange(n_train)` / `np.arange(n_train, n_total)`) — no array
copies.

### `make_collate` — [`data/collate.py`](loc_libs/data/collate.py)

Runs **once per batch, inside the worker**: stack → normalise (one vectorised
`Normalizer.transform`) → first-tau-repeat window → to-tensor. Because
`Normalizer.transform` is elementwise per `(eta, tau)`, the batched result is
identical (within fp tolerance) to the old per-event path. The window is:

```
x = repeat(batch[..., :1], T-1)     # initial τ slice, repeated
y = batch[..., 1:T]                 # τ steps 1 … T-1
```

where `T` is the number of τ steps after `tauslice`.

### `build_dataloaders` — [`data/loader.py`](loc_libs/data/loader.py)

Ties it together (see [signature](#build_dataloaders-arguments)): optional
downsample wrap + coarse-grid bind → train/val split → normalizer (chunked
`fit_lazy` on the train split for data-driven variants) → datasets + collate →
torch `DataLoader`s for both backends.

---

## Configuration

YAML keys consumed by `train_generic/train_h5.py` and `train_generic/train.py`:

```yaml
# --- normalisation ---
normalization:
  name: etau75_eta_3d        # none | etau75_3d | etau75_eta_3d |
                             # etau75_eta_shape_3d | cosh_etau75_3d
  # alpha: 1.0               # only for cosh_etau75_3d
norm_chunk_size: 64          # events per chunk when fitting data-driven refs

# --- data preparation ---
train_ratio: 0.8
xyslice:  [null, -1]         # slice(None, -1) on both nx and ny
etaslice: [null, null]       # slice on neta
tauslice: [null, -1]         # slice on ntau

# --- loaders (new) ---
loader:
  num_workers:     4         # DataLoader workers (0 = synchronous)
  prefetch_factor: 2         # batches prefetched per worker (ignored if workers=0)
  locality_block:  256       # null -> plain shuffle; int -> BlockBatchSampler

# --- downsampling (now auto-applied here; coarse grid bound automatically) ---
downsample:
  spatial_factor: 2          # or spatial_factors: [2,2,1] / spatial_target: [32,32,16]
  tau_stride:     2          # keep every 2nd τ step
  # reduce: mean             # mean | subsample
  # boundary: crop           # crop | pad   (see README_downsample.md)
```

Notes:
- `num_workers` may also be set at the top level (back-compat); the `loader:`
  block takes precedence.
- `downsample:` is the same block documented in
  [`README_downsample.md`](README_downsample.md). It used to require a separate
  manual call to recompute the normalizer grid — now `build_dataloaders` applies
  it to the source **and** the grid params in one place.
- Setting `normalization.name: none` makes the collate a pure window + to-tensor
  (no transform).

---

## Locality sampler

ROOT baskets and HDF5 chunks decompress as a unit, so scattered single-event
reads are wasteful. `BlockBatchSampler` (`locality_block: <int>`) shuffles
**blocks** of contiguous indices, then yields contiguous batch-sized ranges — so
each batch maps to one contiguous slice read instead of N random reads. It
trades looser shuffling for sequential I/O. `locality_block: null` falls back to
ordinary shuffled batching. Every index is still visited exactly once per epoch.

A good default for `block` is roughly the HDF5 chunk size (or a few ROOT
baskets).

---

## Fork-safety (multi-worker reads)

h5py / uproot file handles must **not** be shared across `fork()`. The pipeline
handles this two ways (`DATA_PIPELINE_PLAN.md` §9):

1. **pid-keyed handle cache** — `MultiH5Array._open` and `RootSource._tree_for`
   key their open handles by `os.getpid()`, so a forked worker transparently
   reopens its own.
2. **`worker_init_fn`** (passed automatically) drops any cache a worker
   inherited before its first read, unwrapping `DownsampledArrayView` to reach
   the underlying source.

The training scripts call `multiprocessing.set_start_method('fork')` at startup,
which is the mode this design targets — `fork` also avoids re-importing
(re-running) the training script in each worker, which `spawn` does. The lazy
sources are additionally pickle-safe (handles excluded from `__getstate__`), so
they survive `spawn` too if you change the start method yourself.

Verified: `num_workers=4` over a multi-file `MultiH5Array` yields batches
identical to `num_workers=0`.

---

## `build_dataloaders` arguments

```python
build_dataloaders(
    source, *,
    n_total, train_ratio,          # split: first int(n_total*train_ratio) = train
    xyslice, etaslice, tauslice,   # per-event slices (slice objects)
    norm_cfg, grid_params,         # normalizer name/cfg + {tau_min,dtau,eta_min,deta}
    backend,                       # 'torch' | 'mlx'
    batch_size, test_batch_sizes,  # train batch size; list of eval batch sizes
    num_workers=0, prefetch_factor=2,
    downsample_cfg=None,           # the YAML 'downsample' mapping (or None)
    locality_block=None,           # int -> BlockBatchSampler; None -> plain shuffle
    norm_chunk_size=64,
) -> (train_loader, test_loaders, normalizer)
```

`test_loaders` is a dict keyed by spatial resolution (`nx` after slicing /
downsampling), one `DataLoader` per entry in `test_batch_sizes`.

---

## Backward compatibility

The legacy modules are untouched, so existing imports keep working:

- `from fluid_datasets import FluidDataset, FluidDatasetH5, FluidDatasetMLX, FluidDatasetH5MLX`
- `from read_3d_root import read_3d_data`, `from read_3d_hdf5 import read_3d_data_hdf5`

The new package also re-exports the legacy classes for incremental migration:

```python
from data import FluidDatasetH5, FluidDatasetMLX, FluidDatasetH5MLX
from data import FluidDatasetEager      # the old eager FluidDataset
```

Note: `data.FluidDataset` is the **new** raw-event class (§7), which supersedes
the old eager `fluid_datasets.FluidDataset` (available as `FluidDatasetEager`).

Migrated to the new path so far: `train_generic/train_h5.py` (HDF5) and
`train_generic/train.py` (ROOT). The specialized scripts under `train_torch/`
and `train_gcs/` still run on the legacy modules and can be migrated the same
way when needed.

---

## Tests

[`tests/test_data_pipeline.py`](tests/test_data_pipeline.py) (`DATA_PIPELINE_PLAN.md`
§15). Data-backed tests auto-skip when the sample `.root` / `.h5` files are
absent.

```bash
cd tests && python -m pytest test_data_pipeline.py -q
```

Covers: `BlockBatchSampler` coverage/length, `RootSource` vs eager parity (+
pickle round-trip), `build_dataloaders` parity vs the legacy per-event path for
every normalizer, multi-worker fork-safety (`workers=4 == workers=0`), and
downsample coarse-grid auto-bind.
