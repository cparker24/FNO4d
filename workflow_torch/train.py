#!/usr/bin/env python

# ---------------------------------------------------------------------------
# workflow_torch/train.py — unified PyTorch FNO training on the new data
# pipeline (loc_libs/data).  One script, HDF5 *or* ROOT input, model selector.
#
# Usage:
#   python train.py [--config config.yaml]
#
# Highlights vs the older train_torch/* scripts:
#   * Input is read through loc_libs/data.build_source, which picks the right
#     lazy ArraySource from the file extension (.root -> RootSource,
#     .h5/.hdf5 -> MultiH5Array).  Switching between ROOT and HDF5 is just a
#     change of input_files — no code path branching here.
#   * Datasets, train/val split, normalisation, first-tau windowing and the
#     DataLoaders are all built by the single build_dataloaders() entry point
#     (DATA_PIPELINE_PLAN.md §11): normalisation + windowing run in a batched
#     collate, data-driven normalizers fit chunked on the train split, and
#     downsample/locality are auto-applied from the YAML.
#
# Config keys that select the execution mode:
#   data_source : local | gcs_download | gcs_stream
#       gcs_stream is ROOT-only (uproot streams gs://; h5py needs local paths).
#   model_type  : fno | causal_fno4d | causal_fno4d_tau_eta
#
# See config.yaml (HDF5) and config_root.yaml (ROOT) for the full key list.
# ---------------------------------------------------------------------------

import argparse
import gc
import sys
import multiprocessing as _mp
from pathlib import Path
from time import perf_counter

import yaml

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
parser = argparse.ArgumentParser(description='Train FNO (torch) on the new data pipeline')
parser.add_argument('--config', default='config.yaml', metavar='FILE',
                    help='YAML configuration file (default: config.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')   # local | gcs_download | gcs_stream
if data_source not in ('local', 'gcs_download', 'gcs_stream'):
    raise ValueError(
        f"data_source must be 'local', 'gcs_download', or 'gcs_stream', got: {data_source!r}")

# ---------------------------------------------------------------------------
# torch / neuralop imports
# ---------------------------------------------------------------------------
import torch
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='neuralop')

sys.path.insert(0, str(Path(__file__).parent.parent / 'loc_libs'))
sys.path.insert(0, str(Path(__file__).parent))      # for the local _model helper
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                            save_last_checkpoint)
from parse_training_log import TrainingHistory
from downsample import parse_downsample_cfg, downsampled_grid_params
from data import build_source, build_dataloaders
from _model import build_model

# ---------------------------------------------------------------------------
# GCS imports
# ---------------------------------------------------------------------------
gcs_key_file      = cfg.get('gcs_key_file', None)
gcs_output_prefix = cfg.get('gcs_output_prefix', None)
gcs_show_progress = cfg.get('gcs_show_progress', None)

if data_source == 'gcs_download':
    from gcs_utils import download_gcs_files, upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)
elif data_source == 'gcs_stream':
    from gcs_utils import get_gcs_storage_options, upload_gcs_dir, check_gcs_dependencies
    check_gcs_dependencies(require_storage=False)
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
input_dir       = cfg.get('input_dir', '.')
input_files     = cfg['input_files']
lazy            = cfg.get('lazy', True)
nevents_in      = cfg.get('nevents_in', -1)
start_event     = cfg.get('start_event', 0)
choose_ntau     = cfg.get('choose_ntau', None)      # ROOT-only tau pad target

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
eval_interval   = cfg.get('eval_interval', 2)
save_last       = cfg.get('save_last', True)
norm_cfg        = cfg.get('normalization', {'name': 'none'})
norm_chunk_size = cfg.get('norm_chunk_size', 64)
model_type      = cfg.get('model_type', 'fno')      # fno | causal_fno4d | causal_fno4d_tau_eta
causal_cfg      = cfg.get('causal_fno4d', {})       # extra kwargs for CausalFNO4D variants

# loader / downsample (DATA_PIPELINE_PLAN.md §11–§12)
downsample_cfg  = cfg.get('downsample')
loader_cfg      = cfg.get('loader', {}) or {}
num_workers     = loader_cfg.get('num_workers', cfg.get('num_workers', 0))
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

log = RunLog(f'{odir}/training.log' if log_directly else None)
if _odir_created_msg and log._f:
    log._f.write(_odir_created_msg + '\n')
    log._f.flush()

if not Path(model_dir).exists():
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    log(f"[INFO] created model_dir '{model_dir}' (did not exist)")

t0 = perf_counter()

# ---------------------------------------------------------------------------
# resolve input paths by data_source + classify file type
# ---------------------------------------------------------------------------
if not input_files:
    raise ValueError("input_files is empty — set at least one file in the YAML config.")

_is_h5 = all(str(f).lower().endswith(('.h5', '.hdf5')) for f in input_files)

if data_source == 'gcs_stream' and _is_h5:
    raise ValueError(
        "data_source 'gcs_stream' is ROOT-only — h5py cannot read gs:// paths. "
        "Use data_source 'gcs_download' for HDF5 inputs.")

storage_options = None
if data_source == 'local':
    source_paths = [str(Path(input_dir) / f) for f in input_files]
    for p in source_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input file not found: {p}")
elif data_source == 'gcs_download':
    log(f'Downloading {len(input_files)} file(s) from GCS ...')
    _t_dl = perf_counter()
    source_paths = download_gcs_files(
        input_files, local_data_dir, gcs_key_file,
        overwrite=overwrite_cache, show_progress=gcs_show_progress)
    log(f'Download complete in {perf_counter() - _t_dl:.1f}s')
    for _gs, _lp in zip(input_files, source_paths):
        log(f'  {_gs}  →  {_lp}')
else:  # gcs_stream (ROOT)
    storage_options = get_gcs_storage_options(gcs_key_file)
    source_paths = list(input_files)

# ---------------------------------------------------------------------------
# configuration summary
# ---------------------------------------------------------------------------
log(f'config:           {args.config}')
log(f'backend:          torch')
log(f'data_source:      {data_source}')
log(f'input type:       {"HDF5" if _is_h5 else "ROOT"}')
log(f'input_files:      {input_files}')
if data_source == 'local':
    log(f'input_dir:        {input_dir}')
elif data_source == 'gcs_download':
    log(f'gcs_key_file:     {gcs_key_file}')
    log(f'local_data_dir:   {local_data_dir}')
    log(f'overwrite_cache:  {overwrite_cache}')
elif data_source == 'gcs_stream':
    log(f'gcs_key_file:     {gcs_key_file}')
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
log(f'AdamW lr:         {lr}')
log(f'AdamW weight_decay: {weight_decay}')
log(f'test_batch_sizes: {test_batch_sizes}')
log(f'num_workers:      {num_workers}')
log(f'eval_interval:    {eval_interval}')
log(f'model_dir:        {model_dir}')
log(f'normalization:    {norm_cfg}')
log(f'model_type:       {model_type}')
if model_type != 'fno':
    log(f'causal_fno4d cfg: {causal_cfg}')
if gcs_output_prefix:
    log(f'gcs_output_prefix: {gcs_output_prefix}')

# ---------------------------------------------------------------------------
# build the lazy source (ROOT or HDF5) via the unified factory
# ---------------------------------------------------------------------------
log(f'\nBuilding source from {len(source_paths)} file(s)  lazy={lazy} ...')
source, grid_params = build_source(
    source_paths, storage_options=storage_options, lazy=lazy,
    start_event=start_event, nevents=nevents_in, choose_ntau=choose_ntau,
    verbose=True)
log(f'source:      {type(source).__name__}  shape={tuple(source.shape)}')
log(f'grid_params: {grid_params}')

# nevents_in caps events (for ROOT it is applied here; for HDF5 build_source
# already limited the window — min() is a no-op in that case).
n_total = len(source) if nevents_in is None or nevents_in <= 0 else min(nevents_in, len(source))
log(f'total events used: {n_total}  →  train={int(n_total * train_ratio)}  '
    f'val={n_total - int(n_total * train_ratio)}')

# Effective grid for grid-aware models (Normalizer/loaders apply the same
# downsample internally; mirror it here so the embedding uses the coarse grid).
_dkw = parse_downsample_cfg(downsample_cfg)
grid_params_eff = downsampled_grid_params(grid_params, **_dkw) if _dkw else grid_params

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# split + normalisation + datasets + loaders (single unified builder, §11)
# ---------------------------------------------------------------------------
log(f'num_workers={num_workers}  prefetch_factor={prefetch_factor}  '
    f'locality_block={locality_block}  downsample={bool(downsample_cfg)}')

train_loader, test_loaders, normalizer = build_dataloaders(
    source, n_total=n_total, train_ratio=train_ratio,
    xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
    norm_cfg=norm_cfg, grid_params=grid_params,
    backend='torch', batch_size=batch_size, test_batch_sizes=test_batch_sizes,
    num_workers=num_workers, prefetch_factor=prefetch_factor,
    downsample_cfg=downsample_cfg, locality_block=locality_block,
    norm_chunk_size=norm_chunk_size)

log(f'normalizer:      {normalizer}')
if normalizer.ref is not None:
    log(f'[norm] data-driven ref shape={normalizer.ref.shape}')

normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

test_resolutions = list(test_loaders.keys())
_sample = next(iter(train_loader))
_sample_x_shape = tuple(_sample['x'].shape)        # (B, C, nx, ny, neta, T-1)
log(f'batch x shape: {_sample_x_shape}  y shape: {tuple(_sample["y"].shape)}')
del _sample
gc.collect()

# ---------------------------------------------------------------------------
# model (selector: fno | causal_fno4d | causal_fno4d_tau_eta)
# ---------------------------------------------------------------------------
model = build_model(
    model_type, n_modes=n_modes, hidden_channels=hidden_channels, device=device,
    sample_x_shape=_sample_x_shape, grid_params=grid_params_eff,
    causal_cfg=causal_cfg, model_dir=model_dir, save_embedding=True, log=log)

memory_usage = calculate_model_memory(model)
log(f'Model memory usage: {memory_usage:.2f} MB')

# ---------------------------------------------------------------------------
# optimizer, losses, trainer
# ---------------------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

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
    patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, model_dir)

save_best_key = f'{test_resolutions[0]}_l2'

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
reset_peak_memory(cuda=True)

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

log_peak_memory(log, cuda=True)

# Shut down DataLoader workers explicitly so interpreter exit is immediate.
log('Shutting down DataLoader workers ...')
del train_loader
test_loaders.clear()
gc.collect()

_title_parts = [f'FNO Training ({model_type}, {"HDF5" if _is_h5 else "ROOT"})']
if lazy:
    _title_parts.append('lazy')
if data_source == 'gcs_download':
    _title_parts.append('GCS download')
elif data_source == 'gcs_stream':
    _title_parts.append('GCS stream')
_title = _title_parts[0] + (f' ({", ".join(_title_parts[1:])})' if len(_title_parts) > 1 else '')

history.save(f'{odir}/metrics.json')
history.plot(save_path=f'{odir}/training_curves.svg', title=_title)
log(f'metrics   → {odir}/metrics.json')
log(f'curves    → {odir}/training_curves.svg')
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
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file, show_progress=gcs_show_progress)
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file, show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
