"""
Drop-in replacement for ROOT.TMVA.Experimental.CreatePyTorchGenerators
using uproot — works in environments that have torch + uproot but not PyROOT.

Usage
-----
from root_pytorch_loader import CreatePyTorchGenerators

train_ds, val_ds = CreatePyTorchGenerators(
    "data.root",
    batch_size=8,
    chunk_size=128,
    validation_split=0.2,
)
train_loader = DataLoader(train_ds, batch_size=None)  # batching handled internally

Each yielded item is a dict:
    'x' : FloatTensor (batch, nFeatures, nx', ny', neta', ntau-1)
          — initial tau step repeated ntau-1 times (autoregressive input)
    'y' : FloatTensor (batch, nFeatures, nx', ny', neta', ntau-1)
          — tau steps 1 .. ntau-1 (targets)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset
import uproot


def _read_grid_params(file_path: str) -> dict:
    """Read grid scalars stored at the top level of a flat ROOT file."""
    with uproot.open(file_path) as f:
        return {
            "nFeatures": int(f["nFeatures"].member("fVal")),
            "nx":        int(f["nx"].member("fVal")),
            "ny":        int(f["ny"].member("fVal")),
            "neta":      int(f["neta"].member("fVal")),
            "ntau":      int(f["NT"].member("fVal")),
        }


class _FlatRootDataset(IterableDataset):
    """
    Reads the 'flat_data' branch from a ROOT file in sequential chunks,
    reshapes each chunk to (nFeatures, nx, ny, neta, ntau) per event,
    applies spatial/eta slices, then yields (x, y) dict batches.

    Within each chunk the order is shuffled (training) or kept (validation)
    so that every call to __iter__ gives a different permutation — one epoch
    of data per iteration.
    """

    def __init__(
        self,
        file_path: str,
        start_entry: int,
        stop_entry: int,
        *,
        chunk_size: int,
        batch_size: int,
        ntau: int,
        nx: int,
        ny: int,
        neta: int,
        nFeatures: int,
        xyslice: slice,
        etaslice: slice,
        shuffle: bool,
        drop_remainder: bool,
    ):
        self.file_path     = file_path
        self.start_entry   = start_entry
        self.stop_entry    = stop_entry
        self.chunk_size    = chunk_size
        self.batch_size    = batch_size
        self.ntau          = ntau
        self.nx            = nx
        self.ny            = ny
        self.neta          = neta
        self.nFeatures     = nFeatures
        self.xyslice       = xyslice
        self.etaslice      = etaslice
        self.shuffle       = shuffle
        self.drop_remainder = drop_remainder

    # ------------------------------------------------------------------
    # Approximate length (used by tqdm / DataLoader progress)
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        n = self.stop_entry - self.start_entry
        return n // self.batch_size if self.drop_remainder else math.ceil(n / self.batch_size)

    # ------------------------------------------------------------------
    # Reshape flat (B, flat_size) → (B, nFeatures, nx, ny, neta, ntau)
    # and apply spatial / eta slices.
    # ------------------------------------------------------------------
    def _reshape(self, flat: np.ndarray) -> np.ndarray:
        B = flat.shape[0]
        # (B, ntau, nx, ny, neta, nFeatures)
        arr = flat.reshape(B, self.ntau, self.nx, self.ny, self.neta, self.nFeatures)
        arr = np.moveaxis(arr, 1, -1)   # tau → last  : (B, nx, ny, neta, nFeatures, ntau)
        arr = np.moveaxis(arr, -2, 1)   # feats → front: (B, nFeatures, nx, ny, neta, ntau)
        # Apply slices matching train_model.py: arr[:, :, xyslice, xyslice, etaslice, :]
        return arr[:, :, self.xyslice, self.xyslice, self.etaslice, :]

    # ------------------------------------------------------------------
    # Build (x, y) pair from a shaped batch array.
    # x = initial tau step repeated; y = subsequent tau steps.
    # ------------------------------------------------------------------
    @staticmethod
    def _make_xy(arr: np.ndarray) -> dict:
        n_t = arr.shape[-1]
        x = np.repeat(arr[:, :, :, :, :, :1], n_t - 1, axis=-1)
        y = arr[:, :, :, :, :, 1:]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
        }

    # ------------------------------------------------------------------
    def __iter__(self):
        with uproot.open(self.file_path) as f:
            branch = f["t"]["flat_data"]

            pos = self.start_entry
            while pos < self.stop_entry:
                end = min(pos + self.chunk_size, self.stop_entry)

                flat_chunk = branch.array(
                    entry_start=pos, entry_stop=end, library="np"
                ).astype(np.float32)

                arr = self._reshape(flat_chunk)

                if self.shuffle:
                    arr = arr[np.random.permutation(len(arr))]

                # yield full batches
                b = 0
                while b + self.batch_size <= len(arr):
                    yield self._make_xy(arr[b : b + self.batch_size])
                    b += self.batch_size

                # optionally yield partial last batch
                if not self.drop_remainder and b < len(arr):
                    yield self._make_xy(arr[b:])

                pos = end


def CreatePyTorchGenerators(
    file_path: str,
    batch_size: int = 8,
    chunk_size: int = 128,
    columns: list[str] | None = None,          # API compat — only flat_data used
    max_vec_sizes: dict | None = None,          # API compat — ignored (size auto-read)
    validation_split: float = 0.2,
    shuffle: bool = True,
    drop_remainder: bool = True,
    xyslice: slice = slice(None, -1),
    etaslice: slice = slice(None, -1),
) -> Tuple[_FlatRootDataset, _FlatRootDataset]:
    """
    uproot-based replacement for ROOT.TMVA.Experimental.CreatePyTorchGenerators.

    Reads flat hydrodynamics data from a ROOT TTree ('t', branch 'flat_data')
    in memory-efficient chunks and returns two PyTorch IterableDatasets.

    Parameters
    ----------
    file_path : str
    batch_size : int
    chunk_size : int
        Events held in RAM at once. Larger → better intra-chunk shuffle,
        higher peak memory. Rule of thumb: keep < total RAM / event_size.
    columns, max_vec_sizes : ignored
        Present only for drop-in API compatibility with the ROOT version.
    validation_split : float
        Fraction of entries reserved for validation (taken from the end of
        the file, before any shuffling).
    shuffle : bool
        Shuffle within each chunk every epoch (training only).
    drop_remainder : bool
        Drop the last batch if it is smaller than batch_size.
    xyslice, etaslice : slice
        Applied to the nx/ny axes and the neta axis respectively.
        Default slice(None,-1) drops the last bin on each axis, matching
        the convention in train_model.py (where the eta slice is named
        'tauslice' — kept here for clarity).

    Returns
    -------
    (train_dataset, val_dataset) : Tuple[IterableDataset, IterableDataset]
        Use with DataLoader(dataset, batch_size=None) — batching is done
        internally to avoid double-batching.
    """
    params = _read_grid_params(file_path)
    ntau, nx, ny, neta, nFeatures = (
        params["ntau"], params["nx"], params["ny"],
        params["neta"], params["nFeatures"],
    )

    with uproot.open(file_path) as f:
        n_total = f["t"]["flat_data"].num_entries

    n_val   = int(n_total * validation_split)
    n_train = n_total - n_val

    shared = dict(
        chunk_size=chunk_size,
        batch_size=batch_size,
        ntau=ntau, nx=nx, ny=ny, neta=neta, nFeatures=nFeatures,
        xyslice=xyslice,
        etaslice=etaslice,
        drop_remainder=drop_remainder,
    )

    train_ds = _FlatRootDataset(
        file_path, start_entry=0,       stop_entry=n_train, shuffle=shuffle,  **shared
    )
    val_ds = _FlatRootDataset(
        file_path, start_entry=n_train, stop_entry=n_total, shuffle=False,    **shared
    )
    return train_ds, val_ds
