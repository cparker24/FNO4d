import numpy as np
import torch
from torch.utils.data import Dataset


class FluidDataset(Dataset):
    """PyTorch Dataset wrapping a 6-D numpy array of fluid-simulation snapshots.

    Layout: (events, channels, x, y, eta, tau).
    Returns the first tau step repeated as input x and steps 1..T-1 as target y.
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
        return {'x': torch.FloatTensor(x), 'y': torch.FloatTensor(y)}

    def values(self):
        return self.data.shape
