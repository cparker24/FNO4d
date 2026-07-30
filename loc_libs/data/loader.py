"""DataLoader construction shared by the torch and MLX backends.

A single :func:`build_dataloaders` replaces the per-backend, eager-vs-lazy
loader block in ``train_generic/train_h5.py`` (and the ROOT / GCS variants).
Also provides the locality :class:`BlockBatchSampler` (§10) and the
fork-safe :func:`worker_init_fn` (§9).  See ``DATA_PIPELINE_PLAN.md`` §10–§11.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from downsample import (parse_downsample_cfg, downsampled_grid_params,
                        DownsampledArrayView)
from norm_utils import build_normalizer

from .dataset import FluidDataset
from .collate import make_collate

# Normalizer names that need a data-driven ref fitted on the train split.
_DATA_DRIVEN = ('etau75_eta_3d', 'etau75_eta_shape_3d')


# ---------------------------------------------------------------------------
# fork-safety worker init (DATA_PIPELINE_PLAN.md §9)
# ---------------------------------------------------------------------------
def worker_init_fn(_worker_id):
    """Drop any lazy file handles a forked DataLoader worker inherited.

    h5py / uproot handles must not be shared across ``fork()``.  The lazy
    sources already key their handle cache by pid (so a worker reopens its own
    handles transparently); this hook is the belt-and-suspenders measure that
    clears any inherited cache before the first read.  Safe with eager ndarray
    sources (no ``reset_handles``) and never called when ``num_workers == 0``.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    src = getattr(info.dataset, 'source', None)
    # unwrap DownsampledArrayView -> base, resetting both levels if present
    for obj in (src, getattr(src, '_base', None)):
        if obj is not None and hasattr(obj, 'reset_handles'):
            obj.reset_handles()


# ---------------------------------------------------------------------------
# locality sampler (DATA_PIPELINE_PLAN.md §10)
# ---------------------------------------------------------------------------
class BlockBatchSampler(Sampler):
    """Shuffle contiguous blocks, then emit contiguous batch-size ranges.

    Each yielded batch maps to a single ``MultiH5Array._read_global`` slice (or
    aligned ROOT TBaskets) instead of scattered single-event reads, trading
    looser shuffling for sequential I/O.

    Partitions ``[0, n)`` into contiguous blocks of ``block`` events, shuffles
    the block order (when ``shuffle``), and within each block yields contiguous
    slices of ``batch_size``.  Every index is yielded exactly once per epoch
    (unless ``drop_last`` drops a block's final partial batch).

    ``block=None`` falls back to standard (shuffled) single-index batching.
    """

    def __init__(self, n, batch_size, block=None, shuffle=True, drop_last=False,
                 generator=None):
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.block = None if block is None else int(block)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.generator = generator
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.block is not None and self.block <= 0:
            raise ValueError("block must be positive or None")

    def _block_starts(self):
        block = self.block if self.block is not None else self.n
        block = max(block, 1)
        starts = list(range(0, self.n, block))
        if self.shuffle:
            if self.generator is not None:
                perm = torch.randperm(len(starts), generator=self.generator).tolist()
            else:
                perm = np.random.permutation(len(starts)).tolist()
            starts = [starts[i] for i in perm]
        return starts, block

    def __iter__(self):
        starts, block = self._block_starts()
        bs = self.batch_size
        for s in starts:
            end = min(s + block, self.n)
            for b in range(s, end, bs):
                batch = list(range(b, min(b + bs, end)))
                if self.drop_last and len(batch) < bs:
                    continue
                yield batch

    def __len__(self):
        block = self.block if self.block is not None else self.n
        block = max(block, 1)
        bs = self.batch_size
        total = 0
        for s in range(0, self.n, block):
            blen = min(s + block, self.n) - s
            total += blen // bs if self.drop_last else -(-blen // bs)
        return total


# ---------------------------------------------------------------------------
# build_dataloaders (DATA_PIPELINE_PLAN.md §11)
# ---------------------------------------------------------------------------
def build_dataloaders(source, *, n_total, train_ratio, xyslice, etaslice, tauslice,
                      norm_cfg, grid_params, backend, batch_size, test_batch_sizes,
                      num_workers=0, prefetch_factor=2, downsample_cfg=None,
                      locality_block=None, norm_chunk_size=64, normalizer=None):
    """Build (train_loader, test_loaders, normalizer) for any backend/source.

    Used by both torch and MLX: workers do the I/O and the collate returns
    numpy for MLX (the MLX trainer converts to ``mx.array``), so MLX gets the
    same parallel prefetch as torch.

    Parameters mirror the YAML training config; see ``DATA_PIPELINE_PLAN.md``
    §11–§12.  ``downsample_cfg`` (a YAML ``downsample`` mapping) is auto-applied
    here and the coarse grid is auto-derived — removing the manual
    "recompute coarse grid params" trap.

    ``normalizer`` — pass an already-built (and, for data-driven variants,
    already-fitted) Normalizer to use it verbatim instead of building one from
    ``norm_cfg``.  Used by the resume path to reuse the checkpoint's
    ``normalizer.npz`` rather than refitting on the train split.  ``None``
    (default) builds + fits from ``norm_cfg`` as before.
    """
    # 1) optional downsample wrap — auto-bind the coarse grid
    dkw = parse_downsample_cfg(downsample_cfg)
    if dkw:
        source      = DownsampledArrayView(source, **dkw)
        grid_params = downsampled_grid_params(grid_params, **dkw)

    # 2) split (index remap; no array copies)
    n_train = int(n_total * train_ratio)

    # 3) normalizer (+ chunked fit on the train range for data-driven variants).
    #    A caller-supplied normalizer (e.g. loaded from a checkpoint) is used
    #    as-is — no rebuild, no refit.
    if normalizer is None:
        normalizer = build_normalizer(norm_cfg, grid_params)
        if normalizer.name in _DATA_DRIVEN:
            normalizer.fit_lazy(source, n_train, xyslice, etaslice, tauslice,
                                chunk_size=norm_chunk_size)

    # 4) datasets (raw reads) + collate
    train_ds = FluidDataset(source, np.arange(n_train),          xyslice, etaslice, tauslice)
    val_ds   = FluidDataset(source, np.arange(n_train, n_total), xyslice, etaslice, tauslice)
    T = train_ds[0].shape[-1]
    collate = make_collate(normalizer, T, backend)

    # 5) loaders — one torch DataLoader for BOTH backends (numpy collate for MLX)
    nw = num_workers
    common = dict(collate_fn=collate, num_workers=nw,
                  pin_memory=(backend == 'torch'),
                  persistent_workers=(nw > 0),
                  prefetch_factor=(prefetch_factor if nw > 0 else None),
                  worker_init_fn=worker_init_fn)

    if locality_block:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=BlockBatchSampler(len(train_ds), batch_size,
                                            block=locality_block, shuffle=True),
            **common)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)

    res = train_ds[0].shape[1]                      # nx after xyslice (test_loaders key)
    test_loaders = {res: DataLoader(val_ds, batch_size=tb, shuffle=False, **common)
                    for tb in test_batch_sizes}
    return train_loader, test_loaders, normalizer
