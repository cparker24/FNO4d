#!/usr/bin/env python

# ---------------------------------------------------------------------------
# FNO training script — pure PyTorch, HDF5 input, model-type selector.
#
# Usage:
#   python train_torch_h5_sel.py [--config config_h5_sel.yaml]
#
# One key in config_h5_sel.yaml selects the data source:
#   data_source: local          # local | gcs_download
#
# One key selects the model architecture:
#   model_type: fno | causal_fno4d | causal_fno4d_tau_eta
#
# Key YAML options (see config_h5_sel.yaml for the full list):
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
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO (model-type selector) from HDF5 files')
parser.add_argument('--config', default='config_h5_sel.yaml', metavar='FILE',
                    help='YAML configuration file (default: config_h5_sel.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')   # local | gcs_download

if data_source not in ('local', 'gcs_download'):
    raise ValueError(f"data_source must be 'local' or 'gcs_download', got: {data_source!r}")

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='neuralop')

sys.path.insert(0, '../loc_libs')
from read_3d_hdf5 import read_3d_data_hdf5
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                             save_last_checkpoint, save_embedding_grid)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer
from fluid_datasets import FluidDatasetH5, worker_init_fn
from downsample import (DownsampledArrayView, downsampled_grid_params,
                        parse_downsample_cfg)

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
downsample_kw   = parse_downsample_cfg(cfg.get('downsample'))  # None = off; coarsen before slicing
lr              = cfg['lr']
weight_decay    = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
num_workers     = cfg.get('num_workers', 0)
eval_interval   = cfg.get('eval_interval', 2)
save_last       = cfg.get('save_last', True)
norm_cfg        = cfg.get('normalization', {'name': 'none'})
norm_chunk_size = cfg.get('norm_chunk_size', 64)
model_type      = cfg.get('model_type', 'fno')      # fno | causal_fno4d | causal_fno4d_tau_eta
causal_cfg      = cfg.get('causal_fno4d', {})       # extra kwargs for CausalFNO4D variants

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
# configuration summary
# ---------------------------------------------------------------------------
log(f'config:           {args.config}')
log(f'backend:          torch')
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

# Optional spatial/temporal downsampling — applied to the lazy/eager arr before
# the xy/eta/tau slicing below, exactly like those slices act on the (now
# coarsened) grid.  grid_params is updated so the Normalizer is built at the
# matching resolution (deta/dtau/eta_min change; tau_min does not).  Works for
# both lazy (MultiH5Array) and eager (ndarray) arr without an extra full copy.
if downsample_kw:
    arr         = DownsampledArrayView(arr, **downsample_kw)
    grid_params = downsampled_grid_params(grid_params, **downsample_kw)
    log(f'downsample:       {downsample_kw}')

log(f'arr type:  {type(arr).__name__}')
log(f'arr shape: {arr.shape}')

# ---------------------------------------------------------------------------
# train / val split  +  normalisation  +  datasets
# ---------------------------------------------------------------------------
n_total = len(arr)
n_train = int(n_total * train_ratio)
log(f'total events: {n_total}  →  train={n_train}  val={n_total - n_train}')

normalizer = build_normalizer(norm_cfg, grid_params)
log(f'normalizer: {normalizer}')

if lazy:
    train_indices = np.arange(n_train)
    val_indices   = np.arange(n_train, n_total)

    _needs_fit = normalizer.name in ('etau75_eta_3d', 'etau75_eta_shape_3d')
    if _needs_fit:
        log(f'[norm] data-driven normalizer: computing ref in chunks of {norm_chunk_size} events ...')
        normalizer.fit_lazy(arr, n_train, xyslice, etaslice, tauslice, chunk_size=norm_chunk_size)
        gc.collect()
        log(f'[norm] ref computed  shape={normalizer.ref.shape}')

    dataset_train = FluidDatasetH5(arr, indices=train_indices,
                                   xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
                                   normalizer=normalizer)
    dataset_val   = FluidDatasetH5(arr, indices=val_indices,
                                   xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
                                   normalizer=normalizer)

else:
    data_train  = arr[0:n_train][:, :, xyslice, xyslice, etaslice, tauslice]
    data_verify = arr[n_train:n_total][:, :, xyslice, xyslice, etaslice, tauslice]
    del arr
    gc.collect()

    data_train  = normalizer.fit_transform(data_train)
    data_verify = normalizer.transform(data_verify)

    dataset_train = FluidDatasetH5(data_train)
    dataset_val   = FluidDatasetH5(data_verify)

normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

# resolution after xyslice (used as test_loaders dict key)
_sample_x = dataset_train[0]['x']
test_resolutions = [_sample_x.shape[1]]   # nx after xyslice

log(f'shapes train:  {dataset_train.n_samples} events  event_shape={_sample_x.shape}')
log(f'shapes val:    {dataset_val.n_samples} events')

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------
_nw      = num_workers
_pin     = True
_persist = _nw > 0
_prefetch = 2 if _nw > 0 else None

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                          num_workers=_nw,
                          persistent_workers=_persist,
                          prefetch_factor=_prefetch,
                          pin_memory=_pin,
                          worker_init_fn=worker_init_fn)

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    log(f'test resolution: {res}')
    test_loaders[res] = DataLoader(dataset_val,
                                   batch_size=test_bsize,
                                   shuffle=False,
                                   num_workers=_nw,
                                   persistent_workers=_persist,
                                   prefetch_factor=_prefetch,
                                   pin_memory=_pin,
                                   worker_init_fn=worker_init_fn)
gc.collect()

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
# Event shape from the dataset sample: (C, X, Y, Z, T)
# Z = _sample_x.shape[-2], T = _sample_x.shape[-1]

if model_type == 'fno':
    model = FNO(
        in_channels=4,
        out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
    ).to(device)

elif model_type == 'causal_fno4d':
    from neuralop.models import CausalFNO4D
    model = CausalFNO4D(
        in_channels=4,
        out_channels=4,
        hidden_channels=hidden_channels,
        n_modes_spatial=tuple(n_modes[:3]),
        **causal_cfg,
    ).to(device)
    log(f'CausalFNO4D temporal_receptive_field: {model.temporal_receptive_field}')

elif model_type == 'causal_fno4d_tau_eta':
    from neuralop.models import CausalFNO4DTauEta
    from neuralop.models.causal_fno_4d import make_tau_tensor, make_eta_tensor
    # Derive grid dimensions from the dataset sample shape (C, X, Y, Z, T)
    n_eta = _sample_x.shape[-2]   # Z dimension after slicing
    n_tau = _sample_x.shape[-1]   # T dimension after slicing
    tau_vec = make_tau_tensor(grid_params['tau_min'], grid_params['dtau'], n_tau, device=device)
    eta_vec = make_eta_tensor(grid_params['eta_min'], grid_params['deta'], n_eta, device=device)
    _base_model = CausalFNO4DTauEta(
        in_channels=4,
        out_channels=4,
        hidden_channels=hidden_channels,
        n_modes_spatial=tuple(n_modes[:3]),
        tau_min=float(grid_params['tau_min']), dtau=float(grid_params['dtau']),
        eta_min=float(grid_params['eta_min']), deta=float(grid_params['deta']),
        **causal_cfg,
    ).to(device)
    log(f'CausalFNO4DTauEta temporal_receptive_field: {_base_model.temporal_receptive_field}')
    save_embedding_grid(
        model_dir, 'CausalFNO4DTauEta',
        grid_params['tau_min'], grid_params['dtau'], n_tau, _base_model.tau_max,
        grid_params['eta_min'], grid_params['deta'], n_eta, _base_model.eta_max,
    )
    log(f'embedding grid  →  {model_dir}/embedding_grid.json')
    # Thin wrapper: the neuralop Trainer calls model(x); pin tau/eta from the data grid.
    class _TauEtaTrainingWrapper(nn.Module):
        def __init__(self, m, tau, eta):
            super().__init__()
            self.m = m
            self.register_buffer('tau', tau)
            self.register_buffer('eta', eta)
        def forward(self, x):
            return self.m(x, self.tau, self.eta)
    model = _TauEtaTrainingWrapper(_base_model, tau_vec, eta_vec)

else:
    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Must be 'fno', 'causal_fno4d', or 'causal_fno4d_tau_eta'."
    )

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

_title_parts = [f'FNO Training ({model_type}, HDF5)']
if lazy:
    _title_parts.append('lazy')
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
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file,
                   show_progress=gcs_show_progress)
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file, show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
