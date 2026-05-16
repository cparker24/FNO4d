import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helper shared by both base classes
# ---------------------------------------------------------------------------
def _repeat_first_tau(ev, time_steps):
    """Return (x, y) pair from a single event array (..., ntau)."""
    x = np.repeat(ev[..., :1], time_steps - 1, axis=-1)
    y = ev[..., 1:time_steps]
    return x, y


class _FluidDatasetBase(Dataset):
    """Shared logic for fluid-simulation datasets backed by a materialised ndarray.

    Layout: (events, channels, x, y, eta, tau).
    Returns the first tau step repeated as input x and steps 1..T-1 as target y.
    Subclasses implement _to_tensor() to control the return type.
    """

    def __init__(self, data):
        self.data = data
        self.time_steps = data.shape[-1]
        self.n_samples = data.shape[0]
        print("Numpy data shape: ", data.shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_initial = self.data[idx, :, :, :, :, :1]
        x = np.repeat(x_initial, self.time_steps - 1, axis=-1)
        y = self.data[idx, :, :, :, :, 1:self.time_steps]
        return {'x': self._to_tensor(x), 'y': self._to_tensor(y)}

    def _to_tensor(self, arr):
        raise NotImplementedError

    def values(self):
        return self.data.shape


class FluidDataset(_FluidDatasetBase):
    """PyTorch variant — returns torch.FloatTensor."""

    def _to_tensor(self, arr):
        return torch.FloatTensor(arr)


class FluidDatasetMLX(_FluidDatasetBase):
    """MLX variant — returns float32 numpy arrays (MLX Trainer converts to mx.array)."""

    def _to_tensor(self, arr):
        return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# HDF5 / lazy-array variants
# ---------------------------------------------------------------------------

class _FluidDatasetH5Base(Dataset):
    """Base class for datasets backed by an ndarray or MultiH5Array.

    Supports on-demand reads via index remapping and optional per-event
    spatial slicing — so train/val splits and axis trimming do not require
    materialising extra copies of the full array.

    Parameters
    ----------
    data     : ndarray or MultiH5Array
                Full event array with shape (nevents, nFeatures, nx, ny, neta, ntau).
    indices  : array-like or None
                Global event indices to expose.  None = all events.
                Pass np.arange(n_train) / np.arange(n_train, n_total) to express
                train/val splits without slicing the source array.
    xyslice  : slice or None
                Applied to the nx and ny axes of each event.  None = no trim.
    tauslice : slice or None
                Applied to the neta axis of each event.  None = no trim.
    """

    def __init__(self, data, indices=None, xyslice=None, tauslice=None):
        self.data     = data
        self.indices  = np.asarray(indices, dtype=np.intp) if indices is not None else None
        self.xyslice  = xyslice  if xyslice  is not None else slice(None)
        self.tauslice = tauslice if tauslice is not None else slice(None)
        self.n_samples = len(indices) if indices is not None else len(data)
        sample = self._get_raw(0)
        self.time_steps = sample.shape[-1]
        print(f'{type(self).__name__}: n_samples={self.n_samples}  event_shape={sample.shape}')

    def _get_raw(self, local_idx):
        gi = int(self.indices[local_idx]) if self.indices is not None else local_idx
        ev = self.data[gi]   # (nFeatures, nx, ny, neta, ntau)
        return ev[:, self.xyslice, self.xyslice, self.tauslice, :]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ev = self._get_raw(idx)
        x, y = _repeat_first_tau(ev, self.time_steps)
        return {'x': self._to_tensor(x), 'y': self._to_tensor(y)}

    def _to_tensor(self, arr):
        raise NotImplementedError

    def values(self):
        return self.data.shape


class FluidDatasetH5(_FluidDatasetH5Base):
    """PyTorch variant for HDF5/lazy data — returns torch.FloatTensor."""

    def _to_tensor(self, arr):
        return torch.FloatTensor(arr)


class FluidDatasetH5MLX(_FluidDatasetH5Base):
    """MLX variant for HDF5/lazy data — returns float32 numpy arrays."""

    def _to_tensor(self, arr):
        return arr.astype(np.float32)
