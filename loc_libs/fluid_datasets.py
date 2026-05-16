import numpy as np
import torch
from torch.utils.data import Dataset


class _FluidDatasetBase(Dataset):
    """Shared logic for fluid-simulation datasets.

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
