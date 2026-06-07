# Data Pipeline — Unified Dataset & DataLoader Implementation Plan

> Concrete implementation plan for §4 of `REORGANIZATION.md`. Scope: the data-reading and
> preparation path only. This is a design/implementation spec — the code skeletons below are
> the target API, not yet committed source.

## 1. Goal & scope

Replace the forked data path — **ROOT-eager vs HDF5-lazy**, **4 near-duplicate `Dataset`
classes**, **per-backend loaders**, **per-event normalization** — with a single stack:

```
ArraySource  →  FluidDataset  →  collate_fn  →  DataLoader
 (ROOT/HDF5/Zarr,   (raw event       (normalize +     (workers, prefetch,
  local or gs://,     reads only)      window +         pin, locality sampler;
  lazy)                                to-tensor)        shared by torch & MLX)
```

Goals: one code path for both backends; correct multi-worker reads; batched (vectorized)
normalization; lazy ROOT/HDF5/Zarr with direct `gs://` streaming; and removal of the
"forgot to recompute coarse grid params" trap.

## 2. Most of this already exists (reuse, don't reinvent)

| Concern | Existing code to reuse | Notes |
|---|---|---|
| Lazy array protocol | `read_3d_hdf5.MultiH5Array` (`read_3d_hdf5.py:186-324`) | Already has `.shape`, `.dtype`, `__len__`, int/slice/fancy `__getitem__`, `__array__`, `load()`, per-process handle cache `_open()` (`:272-278`). |
| Lazy downsample decorator | `downsample.DownsampledArrayView` (`downsample.py:526-584`) | Wraps any source; `base[int]`→5-D event, `base[slice]`→6-D batch. |
| Coarse-grid bookkeeping | `downsample.downsampled_grid_params` (`:365-450`), `parse_downsample_cfg` (`:453-520`) | Works on both full `.attrs` grids and the reduced normalizer dict. Docstring already shows the exact wiring. |
| Normalization lifecycle | `norm_utils.Normalizer` (`norm_utils.py:52-289`), `build_normalizer` (`:277`) | `fit_lazy` (chunked, `:70`), `fit_transform`/`transform` (batched over `(N,F,nx,ny,neta,ntau)`), `inverse`, `save`/`load`. |
| Dataset (index split + slicing) | `fluid_datasets._FluidDatasetH5Base` (`fluid_datasets.py:64-125`) | Already supports `indices`, `xy/eta/tau` slices, optional normalizer. The 4 public classes differ **only** in `_to_tensor` and eager-vs-lazy. |
| ROOT read/pad/reshape logic | `read_3d_root.read_3d_data` (`read_3d_root.py:148-203`) | Variable-tau pad-to-`choose_ntau`, reshape, the two `moveaxis` — lift into `RootSource`. |
| Loader construction to replace | `train_generic/train_h5.py:259-345` | The eager/lazy + per-backend block this plan unifies. |

The net change is small: name the protocol, add `RootSource`, **collapse 4 datasets → 1**,
**move normalization + windowing into `collate_fn`**, **make handle caches fork-safe**, add a
**locality `BatchSampler`** and one **`build_dataloaders()`**.

## 3. Target module layout

New subpackage `loc_libs/data/` (part of the §2a `loc_libs` restructure):

```
loc_libs/data/
  sources.py    # ArraySource protocol; RootSource (new); ZarrSource (optional);
                #   re-export MultiH5Array, DownsampledArrayView
  dataset.py    # FluidDataset (single class; returns raw numpy events)
  collate.py    # make_collate(normalizer, time_steps, backend)
  loader.py     # build_dataloaders(...), BlockBatchSampler, worker_init_fn
  __init__.py   # re-exports + back-compat shims for old class names
```

## 4. `ArraySource` protocol — `sources.py`

Formalizes the contract `MultiH5Array`, `DownsampledArrayView`, and plain `np.ndarray` already
satisfy. Used only for typing/documentation (structural; no inheritance required).

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class ArraySource(Protocol):
    shape: tuple[int, ...]          # (nevents, nFeatures, nx, ny, neta, ntau)
    dtype: np.dtype
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> np.ndarray: ...   # int -> 5-D event; slice -> 6-D batch
    # optional, used for multi-worker safety (see §8):
    # def reset_handles(self) -> None: ...
```

## 5. `RootSource` — lazy ROOT — `sources.py`

uproot supports lazy/chunked reads; only the current *assembly* into a full array is eager
(`read_3d_root.py:152`). `RootSource` reads just the requested event range and pads/reshapes
on the fly, reusing the proven logic at `read_3d_root.py:148-203`.

```python
import uproot, awkward as ak, numpy as np, os

class RootSource:
    """Lazy ArraySource over one or more .root files (local or gs:///s3://)."""
    def __init__(self, file_paths, *, choose_ntau=None, storage_options=None):
        self._paths = list(file_paths)
        self._open_kw = {} if storage_options is None else {"storage_options": storage_options}
        # cheap metadata only — no bulk-branch read:
        #   nx, ny, neta, nFeatures, NT(member fVal), per-event ntau_freezeout, branch layout
        # choose_ntau defaults to the global max over ntau_freezeout (padding target)
        # build self.grid_params = {tau_min, dtau, eta_min, deta}
        # set self._cumlen across files (like MultiH5Array) and self._item_shape
        self._trees = {}            # pid-keyed cache (see §8)

    @property
    def shape(self):  return (self._total,) + self._item_shape
    @property
    def dtype(self):  return np.float32
    def __len__(self): return self._total

    def __getitem__(self, idx):
        a, b, squeeze = _range(idx, self._total)     # int -> (i, i+1, True); slice -> (a, b, False)
        chunk = self._tree_for(a).arrays(            # entry-range read; fsspec for gs://
            [self._branch], entry_start=a, entry_stop=b, **self._open_kw)
        out = _pad_reshape(chunk[self._branch], self.choose_ntau, self._dims)  # reuse :148-203
        return out[0] if squeeze else out

    def reset_handles(self):                         # §8
        self._trees = {}
```

- **Lazy mechanisms (documented):** `uproot.iterate(step_size=...)` (chunked sequential —
  already used in `read_3d_root.py:164`); `uproot.dask(...)` (parallel chunked collection that
  **replaced the deprecated `uproot.lazy`**); entry-range `tree.arrays(entry_start, entry_stop)`
  for random access (used above).
- **Basket-alignment caveat:** arbitrary `entry_start/stop` can re-read TBaskets when they
  straddle basket boundaries → pair with `BlockBatchSampler` (§9) for contiguous batches.
- **Cloud:** uproot reads `gs://`/`s3://` via fsspec `storage_options` (from
  `gcs_utils.get_gcs_storage_options`) → ROOT streams lazily with no download (h5py can't).

## 6. `ZarrSource` (optional) — `sources.py`

Secondary/preferred-store option. Thin protocol wrapper over a chunked Zarr array; `store` may
be a local path or an fsspec mapper for `gs://` (lazy partial + parallel reads, no download).
Mirror the upstream `neuraloperator-4d-2.0.0/neuralop/data/` zarr dataset.

```python
class ZarrSource:
    def __init__(self, store, key="arr", storage_options=None):
        import zarr
        self._z = zarr.open(_as_mapper(store, storage_options), mode="r")[key]
    @property
    def shape(self): return self._z.shape
    @property
    def dtype(self): return self._z.dtype
    def __len__(self):  return self._z.shape[0]
    def __getitem__(self, idx): return self._z[idx]
```

## 7. Unified `FluidDataset` — `dataset.py`

One class. **Returns raw numpy events** — normalization, x/y windowing, and tensor conversion
all move to `collate_fn` (§7→§8). Keeps the proven index-remap + slicing from
`_FluidDatasetH5Base`.

```python
import numpy as np, torch
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self, source, indices=None, xyslice=None, etaslice=None, tauslice=None):
        self.source  = source
        self.indices = None if indices is None else np.asarray(indices, dtype=np.intp)
        self.xy   = xyslice  or slice(None)
        self.eta  = etaslice or slice(None)
        self.tau  = tauslice or slice(None)
        self.n    = len(self.indices) if self.indices is not None else len(source)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        gi = int(self.indices[idx]) if self.indices is not None else idx
        ev = self.source[gi]                                   # (F, nx, ny, neta, ntau)
        return ev[:, self.xy, self.xy, self.eta, self.tau]     # raw, unnormalized, unwindowed
```

Eager vs lazy disappears — an in-RAM `np.ndarray` is just a source. Drops the MLX subclasses
and the eager `FluidDataset` (4 → 1).

## 8. Batched `collate_fn` — `collate.py`

Normalization (vectorized over the whole batch) + windowing + tensor conversion happen **once
per batch inside the worker**, replacing the per-event `transform` at `fluid_datasets.py:110`.

```python
import numpy as np, torch

def make_collate(normalizer, time_steps, backend):
    T = time_steps
    def collate(events):
        batch = np.stack(events, axis=0)                       # (B, F, nx, ny, neta, ntau)
        if normalizer is not None and normalizer.name != "none":
            batch = normalizer.transform(batch)                # one vectorized call
        x = np.repeat(batch[..., :1], T - 1, axis=-1)          # _repeat_first_tau, batched
        y = batch[..., 1:T]
        if backend == "torch":
            return {"x": torch.from_numpy(np.ascontiguousarray(x)),
                    "y": torch.from_numpy(np.ascontiguousarray(y))}
        return {"x": x.astype(np.float32, copy=False),         # MLX: numpy; trainer -> mx.array
                "y": y.astype(np.float32, copy=False)}
    return collate
```

`Normalizer.transform` is elementwise per `(eta, tau)`, so the batched result is identical to
today's per-event path; val/test reuse the train-fitted `ref` exactly as now.

## 9. Fork-safety for workers — `sources.py` + `loader.py`

h5py/uproot handles must not be shared across `fork()`. Today `MultiH5Array._open` caches in
`self._handles`, first opened in the **main** process during the dataset `__init__` probe
(`fluid_datasets.py:101`) — so forked workers would inherit a live handle. Two fixes:

- **Primary — pid-keyed cache.** Make `MultiH5Array._open` (and `RootSource._tree_for`) key
  the cache by `os.getpid()`, so a forked worker transparently reopens its own handles:
  ```python
  def _open(self, fi):
      pid = os.getpid()
      cache = self._handles.setdefault(pid, {})
      if fi not in cache:
          cache[fi] = self._h5py.File(self._paths[fi], "r")
      return cache[fi]
  ```
- **Belt-and-suspenders — `worker_init_fn`** that drops the cache on the source (unwrapping
  `DownsampledArrayView._base`). Add `reset_handles()` = clear the cache dict (do **not** close
  inherited fds).

```python
def _worker_init(_worker_id):
    info = torch.utils.data.get_worker_info()
    src = info.dataset.source
    for obj in (src, getattr(src, "_base", None)):     # unwrap DownsampledArrayView
        if hasattr(obj, "reset_handles"):
            obj.reset_handles()
```

## 10. Locality `BlockBatchSampler` — `loader.py`

Opt-in. Shuffles **blocks** of contiguous indices, then yields contiguous batch-size ranges,
so each batch maps to a single `MultiH5Array._read_global` slice (or aligned TBaskets) instead
of scattered single-event reads.

```python
from torch.utils.data import Sampler

class BlockBatchSampler(Sampler):
    def __init__(self, n, batch_size, block=None, shuffle=True, drop_last=False): ...
    def __iter__(self):
        # 1) partition [0, n) into contiguous blocks of size `block`
        # 2) shuffle block order (if shuffle)
        # 3) within each block, emit contiguous slices of `batch_size`
    def __len__(self): ...
```

`block=None` ⇒ fall back to standard shuffled sampling. Default `block` ≈ HDF5 chunk / a few
baskets. Trade-off: looser shuffling for sequential I/O — expose via config.

## 11. `build_dataloaders()` — `loader.py`

Single entry replacing `train_h5.py:259-345`, used by all backends and sources.

```python
import numpy as np
from torch.utils.data import DataLoader
from loc_libs.downsample import parse_downsample_cfg, downsampled_grid_params, DownsampledArrayView
from loc_libs.norm_utils import build_normalizer

_DATA_DRIVEN = ("etau75_eta_3d", "etau75_eta_shape_3d")

def build_dataloaders(source, *, n_total, train_ratio, xyslice, etaslice, tauslice,
                      norm_cfg, grid_params, backend, batch_size, test_batch_sizes,
                      num_workers=0, prefetch_factor=2, downsample_cfg=None,
                      locality_block=None, norm_chunk_size=64):
    # 1) optional downsample wrap — auto-bind coarse grid (kills the manual-recompute trap)
    dkw = parse_downsample_cfg(downsample_cfg)
    if dkw:
        source      = DownsampledArrayView(source, **dkw)
        grid_params = downsampled_grid_params(grid_params, **dkw)

    # 2) split (index remap; no array copies)
    n_train = int(n_total * train_ratio)

    # 3) normalizer (+ chunked fit on the train range for data-driven variants)
    normalizer = build_normalizer(norm_cfg, grid_params)
    if normalizer.name in _DATA_DRIVEN:
        normalizer.fit_lazy(source, n_train, xyslice, etaslice, tauslice,
                            chunk_size=norm_chunk_size)

    # 4) datasets (raw reads) + collate
    train_ds = FluidDataset(source, np.arange(n_train),          xyslice, etaslice, tauslice)
    val_ds   = FluidDataset(source, np.arange(n_train, n_total), xyslice, etaslice, tauslice)
    T = train_ds[0].shape[-1]
    collate = make_collate(normalizer, T, backend)

    # 5) loaders — torch DataLoader for BOTH backends (numpy collate for MLX)
    nw = num_workers
    common = dict(collate_fn=collate, num_workers=nw,
                  pin_memory=(backend == "torch" and nw >= 0),
                  persistent_workers=(nw > 0),
                  prefetch_factor=(prefetch_factor if nw > 0 else None),
                  worker_init_fn=_worker_init)
    if locality_block:
        train_loader = DataLoader(train_ds,
                                  batch_sampler=BlockBatchSampler(len(train_ds), batch_size,
                                                                  block=locality_block, shuffle=True),
                                  **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)

    res = train_ds[0].shape[1]                      # nx after xyslice (test_loaders key)
    test_loaders = {res: DataLoader(val_ds, batch_size=tb, shuffle=False, **common)
                    for tb in test_batch_sizes}
    return train_loader, test_loaders, normalizer
```

MLX uses the same torch `DataLoader` (workers do the I/O; collate returns numpy; the MLX
trainer converts to `mx.array`) → MLX finally gets the prefetch/parallel reads it lacks today
(`train_h5.py:324` hard-codes `num_workers=0` for MLX).

## 12. Config additions (YAML)

```yaml
loader:
  num_workers: 4
  prefetch_factor: 2
  locality_block: 256        # null -> plain shuffle; int -> BlockBatchSampler
data_source: local           # local | gcs_download | gcs_stream   (existing)
downsample: { spatial_factor: 2, tau_stride: 2 }   # existing; now auto-applied here
normalization: { name: etau75_eta_3d }             # existing
```

## 13. Migration / integration steps

1. Add `loc_libs/data/` modules above; in `__init__.py` re-export old names
   (`FluidDataset`, `FluidDatasetH5`, `FluidDatasetMLX`, `FluidDatasetH5MLX`) as thin shims so
   existing imports keep working.
2. Make `MultiH5Array._open` pid-keyed; add `reset_handles()` to `MultiH5Array` and
   `DownsampledArrayView`.
3. **Pilot:** switch `train_generic/train_h5.py` to call `build_dataloaders()`; verify parity
   against the current path.
4. Roll out to `train_generic/train.py` (ROOT path → `RootSource`) and the `train_torch/` /
   `train_gcs/` variants. (Consumes nicely into the §3 training-runner extraction.)

## 14. Backward compatibility

`loc_libs/data/__init__.py` re-exports the legacy class names; the old per-event normalization
path remains importable until all call sites migrate. Distribution name stays `loc_libs`.

## 15. Testing & verification

- **Parity:** for each normalizer name, assert `build_dataloaders` batches equal the current
  `FluidDatasetH5` + `DataLoader` output (same seed, `num_workers=0`) within fp tolerance.
- **Fork-safety:** `num_workers=4` over a multi-file `MultiH5Array` — no h5py
  "already open"/corruption; batches identical to `num_workers=0`.
- **RootSource:** `RootSource[a:b]` equals the matching slice of the eager `read_3d_data`
  array (including pad-to-`choose_ntau`), on a small local `.root`.
- **Locality sampler:** `BlockBatchSampler` yields every index exactly once per epoch.
- **GCS (network-gated):** smoke-test `RootSource`/`ZarrSource` against a `gs://` URI via
  `get_gcs_storage_options`; skip when no creds.
- **Regression:** existing `tests/` (e.g. `test_training_mlx.py`, the torch↔mlx parity tests)
  stay green.

## 16. Expected outcome

2 readers + 4 Dataset classes + scattered norm/downsample wiring →
**1 source interface + 1 Dataset + 1 collate + 1 loader builder** in `loc_libs/data/`, with
correct multi-worker HDF5, lazy ROOT/Zarr incl. `gs://` streaming, batched normalization, and
a single loader shared by the torch and MLX backends.
