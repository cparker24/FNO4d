#!/usr/bin/env python

# ---------------------------------------------------------------------------
# FNO training script — reads pre-converted HDF5 files.
# Multiple files are treated as one concatenated dataset via MultiH5Array.
#
# Usage:
#   python train_h5.py [--config config_h5.yaml]
#
# Two keys in config_h5.yaml select the execution mode:
#   backend:     torch          # torch | mlx
#   data_source: local          # local | gcs_download
#
# Key YAML additions vs config.yaml (ROOT-based):
#   hdf5_dir    : [local] directory containing .h5 files
#   hdf5_files  : filenames (local) or full gs:// URIs (gcs_download)
#   lazy        : true  → on-demand reads per batch; minimal RAM
#                 false → load full array immediately; faster iteration
#   start_event : first global event index across the concatenated files
#
# gcs_download downloads .h5 files from GCS to a local cache, then reads
# them locally.  gcs_stream is not supported because h5py requires local paths.
#
# Normalizer in lazy mode
# -----------------------
# Analytic normalizers (none, etau75_3d, cosh_etau75_3d) are applied
# per-event inside __getitem__ — no up-front materialisation needed.
# Data-driven normalizers (etau75_eta_3d, etau75_eta_shape_3d) must compute
# a reference array from the training split; the training events are
# materialised once for fitting, then discarded before the training loop.
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
import multiprocessing as _mp
import numpy as np
from pathlib import Path
from time import perf_counter

# DataLoader workers must use 'fork' (DATA_PIPELINE_PLAN.md §9): the lazy
# sources key their handle cache by pid so a forked worker reopens its own
# handles, and fork copies this already-running script instead of re-importing
# it (which 'spawn' does — re-executing the whole training script per worker).
try:
    _mp.set_start_method('fork')
except RuntimeError:
    pass  # already set by an earlier import

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO from HDF5 files')
parser.add_argument('--config', default='config_h5.yaml', metavar='FILE',
                    help='YAML configuration file (default: config_h5.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

backend     = cfg.get('backend',     'torch')   # torch | mlx
data_source = cfg.get('data_source', 'local')   # local | gcs_download

if backend not in ('torch', 'mlx'):
    raise ValueError(f"backend must be 'torch' or 'mlx', got: {backend!r}")
if data_source not in ('local', 'gcs_download'):
    raise ValueError(f"data_source must be 'local' or 'gcs_download', got: {data_source!r}")

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
else:
    import mlx.optimizers as optim_mlx
    from neuralop_mlx.models import FNO
    from neuralop_mlx import Trainer, LpLoss, H1Loss, AdamW
    from neuralop_mlx.utils import count_model_params

sys.path.insert(0, '../loc_libs')
from read_3d_hdf5 import read_3d_data_hdf5
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                             save_last_checkpoint)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer

from data import build_dataloaders

# ---------------------------------------------------------------------------
# GCS imports
# ---------------------------------------------------------------------------
gcs_key_file      = cfg.get('gcs_key_file', None)
gcs_output_prefix = cfg.get('gcs_output_prefix', None)
gcs_show_progress = cfg.get('gcs_show_progress', None)

if data_source == 'gcs_download':
    from gcs_utils import download_gcs_files, upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)
elif gcs_output_prefix:
    from gcs_utils import upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

odir            = cfg['odir']
model_dir       = cfg.get('model_dir', odir)
hdf5_dir        = cfg.get('hdf5_dir', '.')
hdf5_files      = cfg['hdf5_files']
lazy            = cfg.get('lazy', True)
nevents_in      = cfg.get('nevents_in', -1)
start_event     = cfg.get('start_event', 0)

# GCS download params (ignored when data_source == 'local')
local_data_dir  = cfg.get('local_data_dir', './data')
overwrite_cache = cfg.get('overwrite_cache', False)
n_epochs        = cfg['n_epochs']
batch_size      = cfg['batch_size']
hidden_channels = cfg['hidden_channels']
n_modes         = cfg['n_modes']
log_directly    = cfg.get('log_directly', True)
checkpoint_freq = cfg.get('checkpoint_freq', None)
min_save_epoch  = cfg.get('min_save_epoch', None)
train_ratio     = cfg.get('train_ratio', 0.8)
xyslice         = _slice(cfg.get('xyslice',  [None, -1]))
etaslice        = _slice(cfg.get('etaslice', [None, None]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
lr              = cfg['lr']
weight_decay    = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
num_workers     = cfg.get('num_workers', 0)
eval_interval   = cfg.get('eval_interval', 2)
save_last       = cfg.get('save_last', True)
norm_cfg        = cfg.get('normalization', {'name': 'none'})
norm_chunk_size = cfg.get('norm_chunk_size', 64)

# loader / downsample (DATA_PIPELINE_PLAN.md §11–§12); num_workers may be set at
# the top level (back-compat) or under a `loader:` block.
downsample_cfg  = cfg.get('downsample')
loader_cfg      = cfg.get('loader', {}) or {}
num_workers     = loader_cfg.get('num_workers', num_workers)
prefetch_factor = loader_cfg.get('prefetch_factor', 2)
locality_block  = loader_cfg.get('locality_block')

# ---------------------------------------------------------------------------
# output dirs + logging
# ---------------------------------------------------------------------------
_odir_created_msg = None
if not Path(odir).exists():
    Path(odir).mkdir(parents=True, exist_ok=True)
    _odir_created_msg = f"[INFO] created odir '{odir}' (did not exist)"
    print(_odir_created_msg)

_sfx = '_mlx' if backend == 'mlx' else ''
log = RunLog(f'{odir}/training{_sfx}.log' if log_directly else None)

if _odir_created_msg and log._f:
    log._f.write(_odir_created_msg + '\n')
    log._f.flush()

if not Path(model_dir).exists():
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    log(f"[INFO] created model_dir '{model_dir}' (did not exist)")

t0 = perf_counter()

# ---------------------------------------------------------------------------
# configuration summary
# ---------------------------------------------------------------------------
log(f'config:           {args.config}')
log(f'backend:          {backend}')
log(f'data_source:      {data_source}')
log(f'hdf5_files:       {hdf5_files}')
if data_source == 'local':
    log(f'hdf5_dir:         {hdf5_dir}')
elif data_source == 'gcs_download':
    log(f'gcs_key_file:     {gcs_key_file}')
    log(f'local_data_dir:   {local_data_dir}')
    log(f'overwrite_cache:  {overwrite_cache}')
log(f'lazy:             {lazy}')
log(f'nevents_in:       {nevents_in}')
log(f'start_event:      {start_event}')
log(f'n_epochs:         {n_epochs}')
log(f'batch_size:       {batch_size}')
log(f'hidden_channels:  {hidden_channels}')
log(f'checkpoint_freq:  {checkpoint_freq}')
log(f'min_save_epoch:   {min_save_epoch}')
log(f'n_modes:          {n_modes}')
log(f'train_ratio:      {train_ratio}')
log(f'xyslice:          {xyslice}')
log(f'etaslice:         {etaslice}')
log(f'tauslice:         {tauslice}')
log(f'log_directly:     {log_directly}')
log(f'AdamW lr:         {lr}')
log(f'AdamW weight_decay: {weight_decay}')
log(f'test_batch_sizes: {test_batch_sizes}')
if backend == 'torch':
    log(f'num_workers:      {num_workers}')
log(f'eval_interval:    {eval_interval}')
log(f'model_dir:        {model_dir}')
log(f'normalization:    {norm_cfg}')
if gcs_output_prefix:
    log(f'gcs_output_prefix: {gcs_output_prefix}')

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
if not hdf5_files:
    raise ValueError("hdf5_files is empty — set at least one file in the YAML config.")

if data_source == 'local':
    full_paths = [str(Path(hdf5_dir) / f) for f in hdf5_files]
    for p in full_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"HDF5 file not found: {p}")

elif data_source == 'gcs_download':
    log(f'Downloading {len(hdf5_files)} HDF5 file(s) from GCS ...')
    t_dl = perf_counter()
    full_paths = download_gcs_files(
        hdf5_files, local_data_dir, gcs_key_file,
        overwrite=overwrite_cache, show_progress=gcs_show_progress,
    )
    log(f'Download complete in {perf_counter() - t_dl:.1f}s')
    for gs_uri, local_path in zip(hdf5_files, full_paths):
        log(f'  {gs_uri}  →  {local_path}')

log(f'\nLoading {len(full_paths)} HDF5 file(s)  lazy={lazy} ...')

dat = read_3d_data_hdf5(
    full_paths,
    nevents=nevents_in,
    start_event=start_event,
    lazy=lazy,
    verbose=True,
)

log('Grid parameters:')
for k, v in dat.items():
    if k not in ('arr', 'tau_freezeout', 'ntau_freezeout', 'nans'):
        log(f'  {k}: {v}')

arr         = dat['arr']
grid_params = {k: dat[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta') if k in dat}
del dat
gc.collect()

log(f'arr type:  {type(arr).__name__}')
log(f'arr shape: {arr.shape}')

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
# train / val split + normalisation + datasets + loaders
# ---------------------------------------------------------------------------
# build_dataloaders (loc_libs/data) is the single entry shared by both backends
# and all sources (DATA_PIPELINE_PLAN.md §11).  It auto-applies any `downsample:`
# config and binds the coarse grid, fits data-driven normalizers chunked on the
# train split, moves normalisation + first-tau windowing into a batched collate,
# and uses torch DataLoader workers for MLX too (numpy collate; the MLX trainer
# converts to mx.array).
n_total = len(arr)
log(f'total events: {n_total}  →  train={int(n_total * train_ratio)}  '
    f'val={n_total - int(n_total * train_ratio)}')
log(f'num_workers={num_workers}  prefetch_factor={prefetch_factor}  '
    f'locality_block={locality_block}  downsample={bool(downsample_cfg)}')

train_loader, test_loaders, normalizer = build_dataloaders(
    arr, n_total=n_total, train_ratio=train_ratio,
    xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
    norm_cfg=norm_cfg, grid_params=grid_params,
    backend=backend, batch_size=batch_size, test_batch_sizes=test_batch_sizes,
    num_workers=num_workers, prefetch_factor=prefetch_factor,
    downsample_cfg=downsample_cfg, locality_block=locality_block,
    norm_chunk_size=norm_chunk_size)
del arr
gc.collect()

log(f'normalizer: {normalizer}')
if normalizer.ref is not None:
    log(f'[norm] data-driven ref shape={normalizer.ref.shape}')

normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

# resolutions are the test_loaders keys (nx after xyslice / downsample)
test_resolutions = list(test_loaders.keys())
_sample = next(iter(train_loader))
log(f'batch x shape: {tuple(_sample["x"].shape)}  y shape: {tuple(_sample["y"].shape)}')
del _sample
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

log(f'Model memory usage: {memory_usage:.2f} MB')

# ---------------------------------------------------------------------------
# optimizer, losses, trainer
# ---------------------------------------------------------------------------
if backend == 'torch':
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
else:
    optimizer = AdamW(lr=lr, weight_decay=weight_decay)
    scheduler = optim_mlx.cosine_decay(lr, 30)

l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)
train_loss  = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

trainer = Trainer(model=model, n_epochs=n_epochs,
                  device=device,
                  wandb_log=False,
                  eval_interval=eval_interval,
                  use_distributed=False,
                  verbose=True)

history = TrainingHistory()
patch_trainer_logging(trainer, log, history=history)

if checkpoint_freq is not None and checkpoint_freq > 0:
    if backend == 'torch':
        patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, model_dir)
    else:
        patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, model_dir,
                                 optimizer=optimizer)


save_best_key = f'{test_resolutions[0]}_l2'

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
if backend == 'torch':
    reset_peak_memory(cuda=True)
else:
    reset_peak_memory(mlx=True)

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              save_dir=model_dir,
              save_best=save_best_key,
              min_save_epoch=min_save_epoch or 0,
              training_loss=train_loss,
              eval_losses=eval_losses)

if save_last:
    save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, model_dir,
                         config_path=args.config)

if backend == 'torch':
    log_peak_memory(log, cuda=True)
else:
    log_peak_memory(log, mlx=True)

_title_parts = ['FNO Training (HDF5)']
if backend == 'mlx':
    _title_parts.append('MLX')
if lazy:
    _title_parts.append('lazy')
_title = _title_parts[0] + (f' ({", ".join(_title_parts[1:])})' if len(_title_parts) > 1 else '')

history.save(f'{odir}/metrics{_sfx}.json')
history.plot(save_path=f'{odir}/training_curves{_sfx}.svg', title=_title)
log(f'metrics   → {odir}/metrics{_sfx}.json')
log(f'curves    → {odir}/training_curves{_sfx}.svg')
log(f'total time: {perf_counter() - t0:.1f}s')

# ---------------------------------------------------------------------------
# upload outputs to GCS (optional)
# ---------------------------------------------------------------------------
if gcs_output_prefix:
    log(f'Uploading outputs to GCS: {gcs_output_prefix}')
    if log._f:
        log._f.flush()
    for _fname in ['optimizer.pt', 'scheduler.pt', 'regularizer.pt']:
        _p = Path(model_dir) / _fname
        if _p.exists():
            _p.unlink()
            log(f'Removed {_fname} from {model_dir} before upload (redundant: resume state is in last_model.pt)')
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file,
                   show_progress=gcs_show_progress)
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file, show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
