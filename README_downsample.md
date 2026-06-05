# Spatial & temporal downsampling

Coarsen the spatial grid (and optionally skip τ steps) of the 3+1D FNO fluid
data — either **once, into a new HDF5 file**, or **on the fly inside the
DataLoader**. Single implementation in [`loc_libs/downsample.py`](loc_libs/downsample.py);
both paths call the same `downsample()` core so an offline file and an
in-loader view produce bit-identical arrays.

---

## Why

Training at full resolution (`nx=ny=65`, `neta=33`, `ntau≈39`) is expensive.
Halving the spatial grid cuts the per-event tensor ~8× and lets you iterate on
models/normalisers/losses quickly, then scale back up. Two wrinkles this module
handles for you:

**The production grids are odd.** A factor of 2 (or 3, …) does not divide
`65` or `33` evenly, so each axis must first be made divisible:

| `boundary` | what it does | `65 →` (factor 2) | notes |
|---|---|---|---|
| **`crop`** *(default)* | trim down to the previous multiple, then block-reduce | `64 → 32` | no fabricated data — cleanest |
| `pad` | pad up to the next multiple, then block-reduce | `66 → 33` | keeps full extent; adds boundary cells (`pad_mode`) |

**Time is not a spatial axis.** τ is downsampled by *skipping* steps
(`tau_stride`) — keep every k-th slice, **no averaging and no interpolation** —
which simply increases the effective `dtau → k·dtau`. Spatial axes use a
block **mean** (`reduce='mean'`) by default, or a strided pick
(`reduce='subsample'`).

---

## Tensor layout

All functions assume the array shape returned by `read_3d_data` /
`read_3d_data_hdf5`:

```
arr.shape = (nevents, nFeatures, nx, ny, neta, ntau)
   axis        0         1       2   3    4     5
                                 └───┴────┴── spatial  (coarsened)   = (-4,-3,-2)
                         └── feature/channel (untouched)
                                              ntau = time/τ (skipped) = (-1)
```

Because the spatial axes sit at `(-4, -3, -2)` and τ at `-1` for **both** the
6-D batch and a single 5-D event `(nFeatures, nx, ny, neta, ntau)`, the same
call works in the offline writer and in the per-event DataLoader path.

---

## Way 1 — make a coarse file (h5 **or** root in → h5 out)

```bash
# halve x,y,eta and skip every other τ step (crop boundary is the default)
python loc_libs/downsample.py data/dAu_25ev_mb.h5 -o data/dAu_ds2.h5 \
       --spatial-factor 2 --tau-stride 2

# keep eta, only coarsen x & y, pad instead of crop, reflect-fill
python loc_libs/downsample.py data/dAu_25ev_mb.h5 -o data/dAu_xy.h5 \
       --spatial-factors 2 2 1 --boundary pad --pad-mode reflect

# source can be ROOT too (output is always HDF5 — the repo's fast-IO format)
python loc_libs/downsample.py data/dAu_25ev_mb.root -o data/dAu_ds2.h5 --spatial-factor 2
```

Equivalent Python:

```python
from downsample import write_downsampled_h5
write_downsampled_h5('data/dAu_25ev_mb.h5', 'data/dAu_ds2.h5',
                     spatial_factors=2, tau_stride=2)        # boundary='crop' default
```

The output is a **drop-in** for everything that reads the standard format
(`read_3d_data_hdf5`, `MultiH5Array`, `FluidDatasetH5`, the eval tools), just at
the coarser resolution. The writer streams `chunk_events` at a time, so peak RAM
is one chunk regardless of dataset size.

### CLI options

| flag | default | meaning |
|---|---|---|
| `INPUT...` | — | one or more `.h5` / `.root` source files (same grid, concatenated) |
| `-o, --output` | — | output `.h5` path (required) |
| `--spatial-factor F` | — | one factor for `nx, ny, neta` |
| `--spatial-factors FX FY FETA` | — | per-axis factors (use `1` to skip an axis) |
| `--tau-stride K` | `1` | keep every K-th τ step (`dtau → K·dtau`); `1` = off |
| `--reduce` | `mean` | spatial reduction: `mean` (block-average) \| `subsample` (strided) |
| `--boundary` | `crop` | make spatial dims divisible: `crop` (trim) \| `pad` (pad up) |
| `--pad-mode` | `edge` | `[--boundary pad]` `edge` \| `reflect` \| `zero` \| any `numpy.pad` mode |
| `--side` | `end` | place the pad/crop cells at the `end`, or split across `both` edges |
| `--nevents N` | `-1` | cap events (`-1` = all) |
| `--chunk-events N` | `8` | events processed/written per step (memory bound) |
| `--compression` | `lzf` | HDF5 compression for `arr` (`lzf` \| `gzip` \| …) |
| `--force` | off | overwrite the output if it exists |

### What the file records

Beyond the recomputed grid scalars (`nx/ny/neta`, `dx/dy/deta`, `x_min/y_min/eta_min`,
`dtau`, `choose_ntau`), the output stores provenance so a coarse file is
self-describing:

```
attrs:  downsample_spatial_factors=[2 2 2]  downsample_tau_stride=2
        downsample_reduce='mean'  downsample_boundary='crop'  downsample_side='end'
        orig_nx=65  orig_ny=65  orig_neta=33  source_files=[...]
data:   arr             (nevents, nFeatures, nx', ny', neta', ntau')
        ntau_freezeout  recomputed to ceil(orig / tau_stride)
        tau_freezeout   unchanged (physical freeze-out time)
```

---

## Way 2 — downsample in the DataLoader (no new file)

Wired into [`train_torch/train_torch_h5_sel.py`](train_torch/train_torch_h5_sel.py)
exactly like the `xyslice` / `etaslice` / `tauslice` options, and **off by
default**. Add a `downsample:` block to the config and it is applied to the lazy
(or eager) `arr` *before* the slices — so the slices then act on the coarse grid:

```yaml
# train_torch/config_h5_sel.yaml
downsample:
  spatial_factor: 2       # or: spatial_factors: [2, 2, 1]  to keep eta
  tau_stride:     2       # keep every other τ step (K=1 = off)
  reduce:         mean    # mean | subsample
  boundary:       crop    # crop (default) | pad
  #pad_mode:      edge    # [boundary: pad] edge | reflect | zero | ...
  #side:          end     # end | both

xyslice:  [null, -1]      # these still run, now on the coarsened grid
etaslice: [null, -1]
tauslice: [null, null]
```

Under the hood the script does the same three lines you can reuse anywhere:

```python
from downsample import (DownsampledArrayView, downsampled_grid_params,
                        parse_downsample_cfg)

downsample_kw = parse_downsample_cfg(cfg.get('downsample'))   # None = off
...
if downsample_kw:
    arr         = DownsampledArrayView(arr, **downsample_kw)        # lazy, per-event
    grid_params = downsampled_grid_params(grid_params, **downsample_kw)  # for the Normalizer
```

`DownsampledArrayView` wraps any base exposing `len`/`shape`/`dtype`/`__getitem__`
(an ndarray **or** a `MultiH5Array`) and downsamples on read, so it plugs
straight into `FluidDatasetH5` without materialising a second copy — `view[i]`
reads exactly event *i* from disk and returns it coarse. Updating `grid_params`
keeps a data-driven Normalizer consistent (`deta`/`eta_min`/`dtau` change;
`tau_min` does not).

---

## API reference

All importable from `downsample` (`import` with `loc_libs` on the path, as the
training scripts do).

| object | purpose |
|---|---|
| `downsample(arr, spatial_factors=1, tau_stride=1, reduce='mean', boundary='crop', pad_mode='edge', side='end')` | combined spatial coarsen + τ skip on a 5-D event or 6-D batch |
| `downsample_spatial(arr, factors=2, ...)` | spatial coarsen only (`nx,ny,neta`) |
| `skip_tau(arr, stride, axis=-1)` | keep every k-th τ step (`arr[..., ::stride]`) |
| `coarsen_axis(arr, axis, factor, ...)` | single-axis primitive (pad/crop + mean/subsample) |
| `downsampled_grid_params(grid, ...)` | recompute grid scalars for the coarse resolution |
| `parse_downsample_cfg(cfg)` | turn a YAML `downsample:` block into kwargs (or `None`) |
| `DownsampledArrayView(base, ...)` | lazy drop-in array that downsamples on read |
| `write_downsampled_h5(src, dst, ...)` | read h5/root, downsample, write a new h5 |

`factors` / `spatial_factors` accept a scalar (same factor for `nx,ny,neta`) or
`(fx, fy, feta)`; use `1` to leave an axis untouched.

### Grid-parameter math

For a spatial axis of length `L`, spacing `d`, origin `min`, factor `f`
(`side='end'`):

| mode | new count | new spacing | new origin |
|---|---|---|---|
| `mean` + `crop` | `L // f` | `f·d` | `min + (f−1)/2·d` |
| `mean` + `pad` | `⌈L/f⌉` | `f·d` | `min + (f−1)/2·d` |
| `subsample` | `⌈L/f⌉` | `f·d` | `min` |

τ skip by stride `k`: `ntau → ⌈ntau/k⌉`, `dtau → k·dtau`, `tau_min` unchanged,
`ntau_freezeout → ⌈ntau_freezeout/k⌉`. (`side='both'` additionally shifts the
origin by the front pad/crop offset.)

---

## Notes

- **Output is HDF5**, even from a ROOT source — this repo converts ROOT → HDF5
  for fast partial I/O (see [`README_train.md`](README_train.md) /
  `loc_libs/read_3d_hdf5.py`). A coarse-ROOT writer is not provided; ask if you
  need one.
- **τ padding is preserved.** Events shorter than `choose_ntau` are zero-padded
  along τ; spatial block-mean of zeros stays zero and τ striding keeps the
  alignment, so the freeze-out bookkeeping (`ntau_freezeout`) stays correct.
- **`crop` vs `pad`.** `crop` (default) drops a thin boundary shell rather than
  averaging real cells against fabricated ones — for a centred fireball on a
  `[-10, 10]` grid the discarded edge is ≈0. Use `pad` (with `pad_mode`) when you
  must preserve the exact physical extent. `side='both'` keeps the trim/pad
  symmetric about the grid centre.
- Generalises the original divisible-only `coarsen_array()` in
  `loc_libs/read_3d_root.py` (pad/crop + τ stride + lazy view).
