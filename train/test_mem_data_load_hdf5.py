#!/usr/bin/env python

# ************************************************************************************************************
# Same as test_mem_data_load.py but uses HDF5 for data loading to avoid the
# ROOT basket memory doubling problem.
#
# On the first run the ROOT file is converted to HDF5 once (accepts the
# one-time 2x peak from the full ROOT basket read).  On every subsequent run
# read_3d_data_hdf5() performs true partial I/O — peak memory stays at
# ~1x arr size regardless of chunk size.
# ************************************************************************************************************

# training parameters
odir = f'./'
input_dir  = '/Users/du8478/FNO_Hydro_Data/flat'
input_files = ['dAu_250ev_mb.root']
hdf5_dir   = input_dir   # HDF5 files written next to the ROOT files

nevents_in    = 100
log_directly  = True
checkpoint_freq = None
train_ratio   = 0.8
xyslice = slice(None, -1)
tauslice = slice(None, -1)

# optimizer parameters
lr           = 8e-3
weight_decay = 1e-4
batch_size   = 1
hidden_channels = 16
n_modes      = [32, 32, 16, 16]
n_epochs     = 1

log = open(f'{odir}/log_hdf5.txt', 'w')
log.write(f'input dir: {input_dir}\n')
log.write(f'input files: {input_files}\n')
log.write(f'n_epochs: {n_epochs}\n')
log.write(f'batch_size: {batch_size}\n')
log.write(f'hidden_channels: {hidden_channels}\n')
log.write(f'n_modes: {n_modes}\n')
log.write(f'train_ratio: {train_ratio}\n')
log.write(f'xyslice: {xyslice}\n')
log.write(f'tauslice: {tauslice}\n')
log.write(f'nevents(in): {nevents_in}\n')
log.write(f'AdamW: lr: {lr}\n')
log.write(f'AdamW: weight_decay: {weight_decay}\n')

import tracemalloc
tracemalloc.start()

import os
import gc
import sys
import atexit

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from time import perf_counter

import psutil

def print_process_memory(label=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    print(f"[Memory rss] {label}: {mem:.3f} GiB")

def print_tracemalloc(label=""):
    current, peak = tracemalloc.get_traced_memory()
    print(f"[Memory tracemalloc] {label}  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

t0 = perf_counter()

sys.path.append('/Users/du8478/FNO/Dave/loc_libs')
from read_3d_hdf5 import convert_root_to_hdf5, read_3d_data_hdf5

# ---------------------------------------------------------------------------
# Convert ROOT → HDF5 if needed (once per file)
# ---------------------------------------------------------------------------
hdf5_files = []
for root_fname in input_files:
    root_path = f'{input_dir}/{root_fname}'
    h5_path   = f'{hdf5_dir}/{os.path.splitext(root_fname)[0]}.h5'
    hdf5_files.append(h5_path)

    if os.path.exists(h5_path):
        print(f'HDF5 file already exists, skipping conversion: {h5_path}')
    else:
        print(f'HDF5 file not found — converting {root_path} → {h5_path}')
        print_process_memory("Before ROOT→HDF5 conversion")
        print_tracemalloc("Before ROOT→HDF5 conversion")

        convert_root_to_hdf5(
            root_file_path=root_path,
            hdf5_file_path=h5_path,
            nevents=-1,         # convert all events
            chunk_events=1,     # one basket per event (finest read granularity)
        )

        print_process_memory("After ROOT→HDF5 conversion")
        print_tracemalloc("After ROOT→HDF5 conversion")
        gc.collect()

# ---------------------------------------------------------------------------
# Load data from HDF5 — true partial I/O, no 2x peak
# ---------------------------------------------------------------------------
print(f'\nEvents requested: {nevents_in}')
print_process_memory("Before HDF5 load")
print_tracemalloc("Before HDF5 load")

arr_data = [
    read_3d_data_hdf5(h5_path, nevents=nevents_in)
    for h5_path in hdf5_files
]

print_process_memory("After HDF5 load")
print_tracemalloc("After HDF5 load")

if not arr_data:
    raise ValueError("No data found.")

log.write('Data parameters:\n')
for key in arr_data[0].keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout', 'nans'):
        log.write(f'  {key}: {arr_data[0][key]}\n')

# ---------------------------------------------------------------------------
# Slice and split
# ---------------------------------------------------------------------------
arr = arr_data[0]['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]

del arr_data
gc.collect()

print_tracemalloc("After del arr_data + gc.collect()")

log.write(f'nevents (loaded): {arr.shape[0]}\n')

n_train     = int(arr.shape[0] * train_ratio)
data_train  = arr[:n_train]
data_verify = arr[n_train:]

gc.collect()

np.random.shuffle(data_train)
np.random.shuffle(data_verify)

print(f'shapes train:  {data_train.shape}')
print(f'shapes verify: {data_verify.shape}')
log.write(f'data_train shape: {data_train.shape}\n')
log.write(f'data_verify shape: {data_verify.shape}\n')

print_tracemalloc("After train/verify split")

# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'Using device: {device}')

class FluidDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.time_steps = data.shape[-1]
        self.n_samples  = data.shape[0]
        print(f"FluidDataset: numpy data shape = {data.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_initial = self.data[idx, :, :, :, :, :1]
        x = np.repeat(x_initial, self.time_steps - 1, axis=-1)
        y = self.data[idx, :, :, :, :, 1:self.time_steps]
        return {'x': torch.FloatTensor(x), 'y': torch.FloatTensor(y)}

    def values(self):
        return self.data.shape

dataset_train = FluidDataset(data_train)
dataset_test  = FluidDataset(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

test_resolutions = [100]
test_batch_sizes = [1]
test_loaders = {
    res: DataLoader(dataset_test, batch_size=bs, shuffle=False,
                    num_workers=0, pin_memory=True, persistent_workers=False)
    for res, bs in zip(test_resolutions, test_batch_sizes)
}

print_process_memory("After creating datasets and dataloaders")
print_tracemalloc("After creating datasets and dataloaders")

gc.collect()
print(f'\nTotal data prep time: {perf_counter() - t0:.1f} s')
log.write(f'Total data prep time: {perf_counter() - t0:.1f} s\n')
log.close()

tracemalloc.stop()
