"""Unified ``FluidDataset`` — one class for every backend and source.

Replaces the four near-duplicate ``Dataset`` classes in ``fluid_datasets.py``.
**Returns raw, unnormalised numpy events** — normalisation, the
first-tau-repeat windowing and tensor conversion all move into the batched
``collate_fn`` (see ``collate.make_collate`` and ``DATA_PIPELINE_PLAN.md`` §7).

Keeps the proven index-remap + per-event slicing of the old
``_FluidDatasetH5Base`` so train/val splits and axis trimming need no extra
array copies.
"""

import numpy as np
from torch.utils.data import Dataset


class FluidDataset(Dataset):
    """Index/slice view over an ArraySource that returns raw numpy events.

    Layout of each event: ``(nFeatures, nx, ny, neta, ntau)``.

    Parameters
    ----------
    source   : ArraySource
        Any object satisfying the ArraySource protocol (ndarray, MultiH5Array,
        RootSource, DownsampledArrayView, ZarrSource, ...).
    indices  : array-like | None
        Global event indices to expose.  None = all events.  Pass
        ``np.arange(n_train)`` / ``np.arange(n_train, n_total)`` to express
        train/val splits without copying the source.
    xyslice  : slice | None
        Applied to both the nx and ny axes of each event.  None = no trim.
    etaslice : slice | None
        Applied to the neta axis.  None = no trim.
    tauslice : slice | None
        Applied to the ntau axis.  None = no trim.
    """

    def __init__(self, source, indices=None, xyslice=None, etaslice=None, tauslice=None):
        self.source  = source
        self.indices = None if indices is None else np.asarray(indices, dtype=np.intp)
        self.xy   = xyslice  if xyslice  is not None else slice(None)
        self.eta  = etaslice if etaslice is not None else slice(None)
        self.tau  = tauslice if tauslice is not None else slice(None)
        self.n    = len(self.indices) if self.indices is not None else len(source)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        gi = int(self.indices[idx]) if self.indices is not None else int(idx)
        ev = self.source[gi]                                   # (F, nx, ny, neta, ntau)
        return ev[:, self.xy, self.xy, self.eta, self.tau]     # raw, unnormalised, unwindowed
