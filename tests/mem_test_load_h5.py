#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Memory test for read_3d_data_hdf5 with multiple HDF5 files.
# Mirrors mem_test_load.py but takes pre-existing .h5 files as input
# (no ROOT-to-HDF5 conversion) and exercises the multi-file lazy path.
#
# Usage:
#   python mem_test_load_h5.py [--config config_h5.yaml]
#
# lazy=True  → dat['arr'] is a MultiH5Array; data is read from disk only
#              when sliced.  Peak RSS stays near the batch window, not the
#              full array size.
# lazy=False → dat['arr'] is a contiguous ndarray loaded immediately.
#              Same behaviour as the single-file version, just concatenated.
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
import os
from time import perf_counter

parser = argparse.ArgumentParser(description='HDF5 multi-file memory test')
parser.add_argument('--config', default='config_h5.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config_h5.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

# ---- config helpers -------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

# ---- extract config -------------------------------------------------------
hdf5_dir        = cfg.get('hdf5_dir', '.')
hdf5_files      = cfg['hdf5_files']
lazy            = cfg.get('lazy', True)
nevents_in      = cfg.get('nevents_in', -1)
start_event     = cfg.get('start_event', 0)
train_ratio     = cfg.get('train_ratio', 0.8)
xyslice         = _slice(cfg.get('xyslice',  [None, -1]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
batch_size      = cfg.get('batch_size', 4)
test_batch_size = cfg.get('test_batch_size', 4)
log_directly    = cfg.get('log_directly', True)

# ---- tracemalloc starts before any heavy imports --------------------------
import tracemalloc
tracemalloc.start()

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, '../loc_libs')
from read_3d_hdf5 import read_3d_data_hdf5
from fluid_datasets import FluidDatasetH5

# ---- logging --------------------------------------------------------------
log_path = cfg.get('log_path', './mem_test_load_h5.log')
_logf = open(log_path, 'w')

def L(msg):
    print(msg)
    _logf.write(msg + '\n')
    _logf.flush()

# ---- memory helpers -------------------------------------------------------
def rss_gib(label=''):
    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    tag = f'  [{label}]' if label else ''
    L(f'[RSS]{tag} {mem:.3f} GiB')

def tmalloc(label=''):
    current, peak = tracemalloc.get_traced_memory()
    tag = f'  [{label}]' if label else ''
    L(f'[tracemalloc]{tag}  current={current/1024**3:.3f} GiB  peak={peak/1024**3:.3f} GiB')

# ---- configuration summary ------------------------------------------------
L(f'config:          {args.config}')
L(f'hdf5_dir:        {hdf5_dir}')
L(f'hdf5_files:      {hdf5_files}')
L(f'lazy:            {lazy}')
L(f'nevents_in:      {nevents_in}')
L(f'start_event:     {start_event}')
L(f'train_ratio:     {train_ratio}')
L(f'xyslice:         {xyslice}')
L(f'tauslice:        {tauslice}')
L(f'batch_size:      {batch_size}')
L(f'test_batch_size: {test_batch_size}')

t0 = perf_counter()

# ---- resolve full paths ---------------------------------------------------
full_paths = [os.path.join(hdf5_dir, f) for f in hdf5_files]
for p in full_paths:
    if not os.path.exists(p):
        raise FileNotFoundError(f'HDF5 file not found: {p}')

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
rss_gib('before HDF5 load')
tmalloc('before HDF5 load')

dat = read_3d_data_hdf5(
    full_paths,
    nevents=nevents_in,
    start_event=start_event,
    lazy=lazy,
    verbose=True,
)

rss_gib('after HDF5 load')
tmalloc('after HDF5 load')

L('\nGrid parameters:')
for k, v in dat.items():
    if k not in ('arr', 'tau_freezeout', 'ntau_freezeout', 'nans'):
        L(f'  {k}: {v}')

arr = dat['arr']
L(f'\narr type:  {type(arr).__name__}')
L(f'arr shape: {arr.shape}')

# ---------------------------------------------------------------------------
# Train/val split
#
# FluidDatasetH5 handles xyslice/tauslice per event, so no pre-slicing needed
# in either mode.  In lazy mode we pass index arrays instead of slicing the
# MultiH5Array (which would materialise data); in eager mode the split produces
# two numpy subarrays and the dataset trims axes on each __getitem__ call.
# ---------------------------------------------------------------------------
del dat
gc.collect()

rss_gib('after del dat + gc.collect()')
tmalloc('after del dat + gc.collect()')

n_total = len(arr)
n_train = int(n_total * train_ratio)
L(f'\ntotal events: {n_total}  →  train={n_train}  val={n_total - n_train}')

if lazy:
    train_indices = np.arange(n_train)
    val_indices   = np.arange(n_train, n_total)
    dataset_train = FluidDatasetH5(arr, indices=train_indices,
                                   xyslice=xyslice, tauslice=tauslice)
    dataset_val   = FluidDatasetH5(arr, indices=val_indices,
                                   xyslice=xyslice, tauslice=tauslice)
else:
    data_train  = arr[:n_train]
    data_verify = arr[n_train:]
    del arr
    gc.collect()
    dataset_train = FluidDatasetH5(data_train, xyslice=xyslice, tauslice=tauslice)
    dataset_val   = FluidDatasetH5(data_verify, xyslice=xyslice, tauslice=tauslice)

rss_gib('after train/val split + dataset creation')
tmalloc('after train/val split + dataset creation')

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
L(f'\ndevice: {device}')

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(dataset_val,   batch_size=test_batch_size,
                          shuffle=False, num_workers=0, pin_memory=False,
                          persistent_workers=False)

gc.collect()
rss_gib('after DataLoaders')
tmalloc('after DataLoaders')

# ---------------------------------------------------------------------------
# Smoke-test: one batch from each loader
# ---------------------------------------------------------------------------
L('\nSanity check — loading one batch from each loader:')

batch = next(iter(train_loader))
L(f'  train  x: {tuple(batch["x"].shape)}  y: {tuple(batch["y"].shape)}')
rss_gib('after first train batch')
tmalloc('after first train batch')

batch_v = next(iter(val_loader))
L(f'  val    x: {tuple(batch_v["x"].shape)}  y: {tuple(batch_v["y"].shape)}')
rss_gib('after first val batch')
tmalloc('after first val batch')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
L(f'\nTotal data-prep time: {perf_counter() - t0:.1f} s')

tracemalloc.stop()
_logf.close()
