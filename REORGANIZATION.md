# FNO4dDev — Reorganization & Consolidation Roadmap

> **Status:** Proposal for review. **No files have been moved and no code changed by this document.**
> It records a concrete, phased plan so the cleanup can be executed (and reviewed) in safe steps.

This repo has grown organically into a capable but cluttered 4D Fourier Neural Operator
project: a PyTorch stack (upstream `neuraloperator-4d-2.0.0/` fork) plus an MLX port
(`neuralop_mlx/`), a shared helper library (`loc_libs/`), several training entry-point
trees, an evaluation suite, and experimental drafts. The clutter is mostly **duplication
and a flat namespace**, not missing structure — and some of the groundwork for cleanup is
already in place (see "What's already good" below).

---

## What's already good (don't redo)

- **`loc_libs/` is already an installable package** — it has its own `loc_libs/pyproject.toml`
  (`name = "loc_libs"`, v0.1.0) with optional `[gcs]` / `[s3]` extras and an `__init__.py`
  documenting its modules. Install is `pip install -e ./loc_libs`. So "make it a real package"
  is *mostly done*; the remaining work is **internal structure**, not packaging.
- **Docs are already wired through MkDocs** — `docs/*.md` are thin `--8<-- "README_*.md"`
  include stubs, organized by the `nav:` tree in `mkdocs.yml`. The root `README_*.md` files
  are the source of truth feeding the site; they are not unmanaged sprawl.
- **Tests are centralized** in `tests/` (MLX layer/model/training tests, torch↔mlx parity,
  weight-transfer, memory profiling).
- **Backend & data-source are already YAML-driven** (`backend: torch|mlx`,
  `data_source: local|gcs_download|gcs_stream`), so a shared runner only has to branch on
  config — no new switch design needed.

---

## 1. Pain points

| # | Problem | Evidence |
|---|---------|----------|
| 1 | **Duplicated training scripts across trees** | `train_generic/train.py` ≈ `train_torch/train_torch.py`; `train_generic/train_h5.py` ≈ `train_torch/train_h5_torch.py`; `train_generic/resume_train_h5.py` ≈ `train_torch/resume_train_h5_torch.py`; plus `*_sel.py` variants and several near-identical `train_gcs/test_train_*_gcs_*.py` scripts. |
| 2 | **`loc_libs/` is a flat namespace** | 17 modules in one directory (data I/O, normalization, losses, eval, viz, utils all mixed). Importable but hard to navigate; the `__init__.py` docstring has drifted from filenames (mentions `datasets` while the file is `fluid_datasets.py`). |
| 3 | **Copy-pasted utilities in drafts** | `addition_drafts/CausalFNO4d/NormFn3d.py` duplicates `loc_libs/NormFn3d.py`. |
| 4 | **Root directory visually noisy** | 9 `README_*.md` files + many top-level dirs at the repo root. They feed MkDocs, but make the root hard to scan. |
| 5 | **Conversion/profiling scripts at top level** | `root2hdf5/`, `flatSplitTime/`, `cloud_util/` sit beside core packages with no `tools/` grouping. |

---

## 2. Target layout

Keep the two model stacks separate (they're intentionally distinct):
`neuralop_mlx/` (MLX port) and `neuraloperator-4d-2.0.0/` (upstream PyTorch fork). The
reorganization is about the **helper library, the training entry points, and top-level tidiness**.

### 2a. Give `loc_libs/` internal structure (submodules by concern)

The package already exists; restructure its flat module list into subpackages. (Optionally
rename the distribution from `loc_libs` to something descriptive like `fno4d` at the same time —
see Phase notes; the name change is cosmetic and can be deferred.)

```
loc_libs/                      (or fno4d/)
  data/      ← read_3d_root.py, read_3d_hdf5.py, root_pytorch_loader.py,
               fluid_datasets.py, downsample.py, gcs_utils.py
  norm/      ← NormFn3d.py, NormFn.py, norm_utils.py
  losses/    ← LossFn.py
  eval/      ← model_eval.py, parse_training_log.py
  viz/       ← contour_v_plot.py, skinnycontour.py
  utils/     ← training_utils.py, memory_utils.py
  training/  ← NEW: shared training internals extracted from the train_* scripts (see §3)
  __init__.py   ← re-export the common names so existing `from loc_libs import X` keeps working
```

Update `loc_libs/pyproject.toml` `[tool.setuptools.packages.find]` to pick up the new
subpackages, and refresh the `__init__.py` module map (fix the `datasets`/`fluid_datasets` drift).

### 2b. Group top-level scripts under `tools/`

```
tools/
  root2hdf5/     ← move root2hdf5/  (root_to_hdf5.py, convert_example*.py, inspect_file.py)
  profiling/     ← move flatSplitTime/  (profiling notebooks/scripts)
  cloud/         ← move cloud_util/
```

---

## 3. Training de-duplication (keep entry points, extract shared logic)

Per the chosen direction: **keep the separate entry points**, but lift the copy-pasted
internals into `loc_libs/training/` so each script becomes a thin wrapper.

**Shared concerns to extract** (currently duplicated across `train_generic/`, `train_torch/`,
`train_gcs/`):

- `config.py` — YAML parsing + validation/defaults (one schema).
- `pipeline.py` — build the data pipeline for `data_source ∈ {local, gcs_download, gcs_stream}`
  and `local|HDF5` sources; reuse existing `loc_libs.data` readers and `fluid_datasets`.
- `model.py` — construct the model from config, branching on `backend ∈ {torch, mlx}`.
- `runner.py` — the epoch/train loop, reusing `loc_libs.utils.training_utils`
  (`RunLog`, `TrainingHistory`) instead of re-implementing logging/history.
- `checkpoint.py` — save / resume (covers the `resume_*` scripts).

**Resulting entry points** stay where they are but shrink to: parse args/YAML →
`loc_libs.training.runner.run(cfg)`. So:

- `train_generic/train.py`, `train_torch/train_torch.py`, `train_gcs/test_train_*_gcs_*.py`
  → thin wrappers (the `train_torch/*` ones simply default `backend: torch`).
- `*_h5*` variants → select the HDF5 path in `pipeline.py`.
- `resume_*` variants → call the runner with `resume=True`.
- `*_sel.py` selection variants → a `selection:` block in config consumed by `pipeline.py`.

This removes the bulk of the duplication while preserving the familiar per-backend launch
commands and configs.

---

## 4. Data pipeline: a unified Dataset/DataLoader layer

The data path is the other big source of duplication, and it's where the most performance is
left on the table. This section proposes folding reading + preparation into one efficient
`loc_libs/data/` layer built around a PyTorch `Dataset`/`DataLoader` (with a numpy boundary so
the MLX backend reuses it too).

### Where it stands today
- **Two paradigms in practice.** ROOT yields an **eager** result; HDF5 is **lazy**
  (`read_3d_hdf5.MultiH5Array`, random-access per item). Worth noting: the ROOT reader
  (`read_3d_root.read_3d_data`) already reads **in chunks** via
  `uproot.iterate(..., step_size=...)` (`read_3d_root.py:164`) to bound peak memory — it just
  then **assembles a full eager array** (`np.zeros(...)` at `read_3d_root.py:152`, filled
  chunk-by-chunk). So the read is chunked; only the final assembly is eager. **This is a code
  choice, not an uproot limitation.**
- **4 near-identical Dataset classes** in `loc_libs/fluid_datasets.py` — `FluidDataset`,
  `FluidDatasetMLX`, `FluidDatasetH5`, `FluidDatasetH5MLX` — differing only by eager/lazy and a
  one-line tensor conversion (`_to_tensor`). x/y windowing is `_repeat_first_tau`.
- **Normalization runs per item inside `__getitem__`** (numpy ops on tiny 5-D arrays every
  sample); fitting uses the streaming `norm_utils.Normalizer.fit_lazy`.
- **Downsampling** is either lazy (`downsample.DownsampledArrayView`) or offline
  (`write_downsampled_h5`); the lazy path makes the caller rebuild grid params manually via
  `downsample.downsampled_grid_params` — a silent-bug trap.
- **GCS:** `gcs_utils.get_gcs_storage_options` streams via uproot/fsspec, but **h5py cannot
  stream `gs://`**, so the HDF5 path forces a full download.
- **DataLoader:** the torch path only enables `num_workers`/`pin_memory`/`persistent_workers`/
  `prefetch_factor` when `num_workers>0`; the **MLX path uses no DataLoader features at all**.
- The upstream `neuraloperator-4d-2.0.0/neuralop/data/` already has a **Zarr** dataset to mirror.

### Proposed design

1. **One lazy `ArraySource` interface** (generalize `MultiH5Array`): `.shape`, `.dtype`, fancy
   `__getitem__`. Implementations: in-memory ndarray (eager), `H5Source`, `ZarrSource`, a new
   lazy **`RootSource`**, and `DownsampledArrayView` as a decorator. The eager/lazy split
   disappears — eager is just an in-RAM source.
   - **`RootSource` (lazy ROOT).** uproot *does* support lazy/chunked reads:
     `uproot.iterate(step_size=...)` (chunked sequential — already used here),
     `uproot.dask(...)` (the modern lazy + parallel collection that **replaced the now-deprecated
     `uproot.lazy`**), and entry-range reads `tree.arrays(entry_start, entry_stop)` for random
     access. `RootSource.__getitem__` reads an event range, then pads to the global
     `choose_ntau` and applies the reshape/moveaxis on the fly (reuse the existing logic at
     `read_3d_root.py:148-203`). Compute `choose_ntau` once from the cheap
     `ntau_freezeout`/`NT` metadata — no bulk-branch read. **Caveat:** arbitrary
     `entry_start/stop` can re-read TBaskets when they straddle basket boundaries, so pair this
     with the basket-aligned sampler in (4). This removes the eager full-array assembly entirely.
2. **One `FluidDataset(source, indices, transform)`.** Slicing, `_repeat_first_tau` windowing,
   and normalization become composable transforms; return numpy and convert at the boundary.
   Collapses the 4 classes into 1.
3. **Normalization as a batched `collate_fn`** instead of per item: vectorized over the whole
   batch and executed inside DataLoader workers. Keep `fit_lazy` for the data-driven refs, and
   **auto-bind coarse grid params** when the source is a `DownsampledArrayView` (kills the
   manual-recompute trap).
4. **One DataLoader builder for both backends.** torch `DataLoader` + a `worker_init_fn` that
   **reopens HDF5/Zarr handles per worker** (h5py handles are not fork-safe — required for
   correct `num_workers>0`), plus `pin_memory`, `persistent_workers`, `prefetch_factor`, and a
   **chunk-aware `BatchSampler`** that shuffles blocks of contiguous indices to exploit chunk
   (and TBasket) locality. Emit **numpy** batches so the **MLX path reuses the same loader**
   (convert to `mx.array` in the loop) — this is the cheapest way to give MLX the prefetch it
   currently lacks.
5. **Format strategy (ROOT is *not* forced to be eager).** With a lazy `RootSource`, ROOT is a
   first-class lazy training source, and uproot reads `gs://`/`s3://` directly via fsspec — so
   ROOT *streams* from cloud lazily, unlike h5py. Converting ROOT → a chunked store (reuse the
   `root2hdf5/` tooling) becomes an **optimization, not a requirement**: a store with
   training-tuned chunking + compression gives faster random access and better worker
   parallelism. Prefer **Zarr** as the canonical store (lazy `gs://` via fsspec — no mandatory
   download, parallel/compressed chunks, mirrors the upstream `neuralop/data/` precedent); keep
   HDF5 and lazy-ROOT both supported.
6. **Optional epoch caching** (LRU, or full in-RAM when it fits) to avoid re-decompressing the
   same events every epoch — the biggest win for ROOT/compressed sources.

**Net effect:** 2 readers + 4 Dataset classes + scattered norm/downsample wiring →
**1 source interface + 1 Dataset + 1 collate + 1 loader builder**, all in `loc_libs/data/`,
with correct multi-worker HDF5, real lazy GCS for both ROOT and Zarr, and a single loader
shared by the torch and MLX backends.

---

## 5. Drafts cleanup

`addition_drafts/*` should import shared utilities instead of carrying copies:

- Delete `addition_drafts/CausalFNO4d/NormFn3d.py`; import `from loc_libs.norm import NormFn3d`.
- Audit the other draft dirs (`CausalLogLo4d/`, `EtaTauEmeddingFNO/`, `PinoLoss/`) for the
  same pattern and repoint them at `loc_libs.norm` / `loc_libs.losses`.
- Keep the experimental models themselves — only the duplicated helpers go away.

---

## 6. Documentation (lower priority — already half-done)

Docs already build via MkDocs include-stubs, so this is optional polish, not a fix:

- **Option A (recommended, low effort):** leave the include indirection as-is; just reduce
  root visual noise by adding a short "Docs" index section to the top-level `README.md`
  linking into the MkDocs pages.
- **Option B (more invasive):** move each `README_*.md`'s *content* into the matching
  `docs/*.md` and drop the `--8<--` includes, leaving a single root `README.md`. This is
  cleaner long-term but rewrites the doc source layout and the `nav:` paths.

Either way, do **not** simply `git mv README_*.md docs/` — that would break the existing
`--8<--` includes.

---

## 7. Tests & examples

- Once `loc_libs` is subdivided, update test imports to the new submodule paths
  (`from loc_libs.data import ...`). Re-exports in `__init__.py` keep most imports working.
- Fold the scattered demo/test scripts under `train/` (`train/eval_model_test_*.py`,
  `train/test_train*.ipynb`, `train/apply.py`) into either `tests/` (real tests) or a new
  top-level `examples/` (demos), so `train/` stops being a catch-all.

---

## 8. Phased migration (safe ordering)

Each phase is independently shippable and leaves the repo working.

1. **Restructure `loc_libs` internals.** Create the subpackages (§2a), move modules, and
   **re-export everything from `loc_libs/__init__.py`** so existing `from loc_libs import X`
   call sites keep working. Update `pyproject` `find` + the module-map docstring. Run `tests/`.
2. **Unify the data pipeline** (§4) inside the new `loc_libs/data/`: introduce the
   `ArraySource` interface + `RootSource`/`H5Source`/`ZarrSource`, collapse the 4 Dataset
   classes into one, move normalization into a batched `collate_fn`, and add the shared
   DataLoader builder (`worker_init_fn` + chunk-aware `BatchSampler`). Do this before §3's
   runner extraction so the runner consumes the unified loader. Verify parity against the
   current loaders on a small sample.
3. **Extract training internals** into `loc_libs/training/` (§3). Convert **one** entry point
   (`train_generic/train.py`) as a pilot wrapper; verify a short training run matches prior
   behavior.
4. **Migrate remaining entry points** to wrappers; **dedupe drafts** (§5).
5. **Tidy top level**: move scripts under `tools/` (§2b); apply the chosen docs option (§6);
   migrate demos to `examples/` (§7).
6. **(Optional) Rename** the package `loc_libs → fno4d` and update import sites + the deep-link
   in `addition_drafts`. Do this last, as one mechanical pass, once everything else is stable.

### Risk notes
- **Biggest risk:** many scripts import `loc_libs` by name (`train_generic/`, `train_torch/`,
  `train_gcs/`, `eval_model/`). The `__init__.py` re-export shims in Phase 1 prevent a
  big-bang break; remove them only after all call sites move (Phase 6).
- Keep the package distribution name `loc_libs` until Phase 6 so `pip install -e ./loc_libs`
  and all imports remain valid during the transition.
- The two model stacks and their `pyproject.toml`s are untouched — no risk to the
  MLX/PyTorch packages.

---

## Out of scope (this document)

Actually moving files, creating subpackages, rewriting imports, deleting duplicates, and any
rename — all deferred until this roadmap is approved and scheduled as implementation PRs
(ideally one PR per phase above).
