#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Unified FNO training script — backend and data source steered by YAML.
#
# Usage:
#   python train.py [--config config.yaml]
#
# Two keys in the YAML select the execution mode:
#
#   backend:     torch          # torch | mlx
#   data_source: local          # local | gcs_download | gcs_stream
#
# All six combinations are supported.  See config.yaml for the full list of
# available options and which keys apply to each mode.
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
import resource
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader   # used by both backends

def rss_gib():
    """Resident set size in GiB — tracks all memory including numpy C-level allocs."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is kilobytes on Linux, bytes on macOS
    import platform
    scale = 1024 if platform.system() == 'Linux' else 1
    return ru.ru_maxrss * scale / 1024**3

# ---------------------------------------------------------------------------
# parse args and load config — before framework imports so backend is known
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO from YAML config')
parser.add_argument('--config', default='config.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

backend     = cfg.get('backend', 'torch')      # torch | mlx
data_source = cfg.get('data_source', 'local')  # local | gcs_download | gcs_stream

if backend not in ('torch', 'mlx'):
    raise ValueError(f"backend must be 'torch' or 'mlx', got: {backend!r}")
if data_source not in ('local', 'gcs_download', 'gcs_stream'):
    raise ValueError(f"data_source must be 'local', 'gcs_download', or 'gcs_stream', got: {data_source!r}")

# ---------------------------------------------------------------------------
# framework-specific imports
# ---------------------------------------------------------------------------
if backend == 'torch':
    import torch
    from neuralop.models import FNO
    from neuralop import Trainer
    from neuralop.training import AdamW
    from neuralop import LpLoss, H1Loss
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', category=UserWarning, module='neuralop')
else:  # mlx
    import mlx.optimizers as optim_mlx
    from neuralop_mlx.models import FNO
    from neuralop_mlx import Trainer, LpLoss, H1Loss, AdamW
    from neuralop_mlx.utils import count_model_params

sys.path.insert(0, '../loc_libs')
from read_3d_root import read_3d_data
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                             patch_trainer_min_save_epoch, save_last_checkpoint)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer

if backend == 'torch':
    from fluid_datasets import FluidDataset
else:
    from fluid_datasets import FluidDatasetMLX

# ---------------------------------------------------------------------------
# GCS-specific imports
# ---------------------------------------------------------------------------
if data_source == 'gcs_download':
    from gcs_utils import download_gcs_files, upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)
elif data_source == 'gcs_stream':
    from gcs_utils import get_gcs_storage_options, upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=False)

# ---------------------------------------------------------------------------
# common config
# ---------------------------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

odir            = cfg['odir']
input_files     = cfg['input_files']
n_epochs        = cfg['n_epochs']
batch_size      = cfg['batch_size']
hidden_channels = cfg['hidden_channels']
n_modes         = cfg['n_modes']
nevents_in      = cfg.get('nevents_in', -1)
log_directly    = cfg.get('log_directly', True)
checkpoint_freq = cfg.get('checkpoint_freq', None)
min_save_epoch  = cfg.get('min_save_epoch', None)
train_ratio     = cfg.get('train_ratio', 0.8)
xyslice         = _slice(cfg.get('xyslice', [None, -1]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
lr              = cfg['lr']
weight_decay    = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
eval_interval   = cfg.get('eval_interval', 2)
save_last       = cfg.get('save_last', True)
model_dir       = cfg.get('model_dir', odir)
norm_cfg        = cfg.get('normalization', {'name': 'none'})

# local-only param
input_dir = cfg.get('input_dir', '.')

# GCS params (ignored when data_source == 'local')
gcs_key_file      = cfg.get('gcs_key_file', None)
gcs_output_prefix = cfg.get('gcs_output_prefix', None)
local_data_dir    = cfg.get('local_data_dir', './data')
overwrite_cache   = cfg.get('overwrite_cache', False)
gcs_show_progress = cfg.get('gcs_show_progress', None)

# ---------------------------------------------------------------------------
# output dirs + logging
# ---------------------------------------------------------------------------
_sfx = '_mlx' if backend == 'mlx' else ''
log = RunLog(f'./mem_test{_sfx}.log' if log_directly else None)

t0 = perf_counter()

# ---------------------------------------------------------------------------
# configuration summary
# ---------------------------------------------------------------------------
log(f'config:          {args.config}')
log(f'backend:         {backend}')
log(f'data_source:     {data_source}')
log(f'input files:     {input_files}')
if data_source == 'local':
    log(f'input dir:       {input_dir}')
elif data_source == 'gcs_download':
    log(f'gcs_key_file:    {gcs_key_file}')
    log(f'local_data_dir:  {local_data_dir}')
    log(f'overwrite_cache: {overwrite_cache}')
    log(f'gcs_output_prefix: {gcs_output_prefix}')
elif data_source == 'gcs_stream':
    log(f'gcs_key_file:    {gcs_key_file}')
    log(f'gcs_output_prefix: {gcs_output_prefix}')
log(f'n_epochs:        {n_epochs}')
log(f'batch_size:      {batch_size}')
log(f'hidden_channels: {hidden_channels}')
log(f'checkpoint_freq: {checkpoint_freq}')
log(f'min_save_epoch:  {min_save_epoch}')
log(f'n_modes:         {n_modes}')
log(f'train_ratio:     {train_ratio}')
log(f'xyslice:         {xyslice}')
log(f'tauslice:        {tauslice}')
log(f'log_directly:    {log_directly}')
log(f'nevents(in):     {nevents_in}')
log(f'AdamW: lr:         {lr}')
log(f'AdamW: weight_decay: {weight_decay}')
log(f'test_batch_sizes:  {test_batch_sizes}')
log(f'eval_interval:     {eval_interval}')
log(f'model_dir:         {model_dir}')
log(f'normalization:     {norm_cfg}')

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
log(f'Events called for {nevents_in}')

if not input_files:
    raise ValueError("No data found in the specified input files.")

if data_source == 'local':
    arr_data = read_3d_data([f'{input_dir}/{f}' for f in input_files], nevents=nevents_in, local_rank=None, verbose=True)

elif data_source == 'gcs_download':
    log(f'Downloading {len(input_files)} file(s) from GCS ...')
    t_dl = perf_counter()
    local_files = download_gcs_files(
        input_files, local_data_dir, gcs_key_file, overwrite=overwrite_cache,
        show_progress=gcs_show_progress,
    )
    log(f'Download complete in {perf_counter() - t_dl:.1f}s')
    for gs_uri, local_path in zip(input_files, local_files):
        log(f'  {gs_uri}  →  {local_path}')
    arr_data = read_3d_data(local_files, nevents=nevents_in, local_rank=None, verbose=True)

else:  # gcs_stream
    storage_options = get_gcs_storage_options(gcs_key_file)
    arr_data = read_3d_data(input_files, nevents=nevents_in, local_rank=None, storage_options=storage_options)

log('Data parameters from input file:')
for key in arr_data.keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[key]}')

arr = arr_data['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]

grid_params = {k: arr_data[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
del arr_data
gc.collect()

log(f'nevents (constrained by input file(s)): {arr.shape[0]}')
test_resolutions = [arr.shape[2]]

print(f"[RSS] after data load:  {rss_gib():.2f} GiB")

# ---------------------------------------------------------------------------
# train / val split
# ---------------------------------------------------------------------------
n_train     = int(arr.shape[0] * train_ratio)
data_train  = arr[:n_train]
data_verify = arr[n_train:]

print(f"[RSS] after train/val split: {rss_gib():.2f} GiB")

del arr
gc.collect()

print(f"[RSS] after del arr + gc:    {rss_gib():.2f} GiB")

log(f'shapes train:  {data_train.shape}')
log(f'shapes verify: {data_verify.shape}')

# ---------------------------------------------------------------------------
# normalisation  (fit on train only, apply to both splits)
# ---------------------------------------------------------------------------
normalizer = build_normalizer(norm_cfg, grid_params)
log(f'normalizer:      {normalizer}')

data_train  = normalizer.fit_transform(data_train)
data_verify = normalizer.transform(data_verify)

print(f"[RSS] after normalizer:      {rss_gib():.2f} GiB")
# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
if backend == 'torch':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')
else:
    device = 'gpu'
    log(f'Using device: {device} (MLX Metal)')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
if backend == 'torch':
    dataset_train = FluidDataset(data_train)
    dataset_test  = FluidDataset(data_verify)
else:
    dataset_train = FluidDatasetMLX(data_train)
    dataset_test  = FluidDatasetMLX(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    log(f'res {res}')
    test_loaders[res] = DataLoader(dataset_test,
                                   batch_size=test_bsize,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False,
                                   persistent_workers=False)
gc.collect()

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
if backend == 'torch':
    model = FNO(
        in_channels=4,
        out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
    ).to(device)
    memory_usage = calculate_model_memory(model)
else:
    model = FNO(
        in_channels=4,
        out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
    )
    memory_usage = calculate_model_memory(model, count_fn=count_model_params)

log(f"Model memory usage: {memory_usage:.2f} MB")

print(f"[RSS] after model (before training):      {rss_gib():.2f} GiB")