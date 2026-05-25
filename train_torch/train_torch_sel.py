#!/usr/bin/env python

# ---------------------------------------------------------------------------
# FNO training script — pure PyTorch, model-type selector, data source steered by YAML.
#
# Usage:
#   python train_torch_sel.py [--config config_sel.yaml]
#
# One key in the YAML selects the data source:
#
#   data_source: local          # local | gcs_download | gcs_stream
#
# One key selects the model architecture:
#
#   model_type: fno                  # standard 4D FNO (default)
#              causal_fno4d          # causal FNO: 3D SpectralConv + causal Conv1d along tau
#              causal_fno4d_tau_eta  # causal FNO + sinusoidal tau/eta positional embeddings
#
# All three data source options are supported.  See config_sel.yaml for the full
# list of available options and which keys apply to each mode.
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
import torch
import torch.nn as nn
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO (model-type selector) from YAML config')
parser.add_argument('--config', default='config_sel.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config_sel.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')  # local | gcs_download | gcs_stream

if data_source not in ('local', 'gcs_download', 'gcs_stream'):
    raise ValueError(f"data_source must be 'local', 'gcs_download', or 'gcs_stream', got: {data_source!r}")

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
from read_3d_root import read_3d_data
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                             save_last_checkpoint)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer
from fluid_datasets import FluidDataset

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
xyslice         = _slice(cfg.get('xyslice',  [None, -1]))
etaslice        = _slice(cfg.get('etaslice', [None, None]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
lr              = cfg['lr']
weight_decay    = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
eval_interval   = cfg.get('eval_interval', 2)
save_last       = cfg.get('save_last', True)
model_dir       = cfg.get('model_dir', odir)
norm_cfg        = cfg.get('normalization', {'name': 'none'})
model_type      = cfg.get('model_type', 'fno')      # fno | causal_fno4d | causal_fno4d_tau_eta
causal_cfg      = cfg.get('causal_fno4d', {})       # extra kwargs for CausalFNO4D variants

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
log(f'config:          {args.config}')
log(f'backend:         torch')
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
log(f'etaslice:        {etaslice}')
log(f'tauslice:        {tauslice}')
log(f'log_directly:    {log_directly}')
log(f'nevents(in):     {nevents_in}')
log(f'AdamW: lr:         {lr}')
log(f'AdamW: weight_decay: {weight_decay}')
log(f'test_batch_sizes:  {test_batch_sizes}')
log(f'eval_interval:     {eval_interval}')
log(f'model_dir:         {model_dir}')
log(f'normalization:     {norm_cfg}')
log(f'model_type:        {model_type}')
if model_type != 'fno':
    log(f'causal_fno4d cfg:  {causal_cfg}')

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
log(f'Events called for {nevents_in}')

if not input_files:
    raise ValueError("No data found in the specified input files.")

if data_source == 'local':
    arr_data = read_3d_data([f'{input_dir}/{f}' for f in input_files], nevents=nevents_in, local_rank=None)

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
    arr_data = read_3d_data(local_files, nevents=nevents_in, local_rank=None)

else:  # gcs_stream
    storage_options = get_gcs_storage_options(gcs_key_file)
    arr_data = read_3d_data(input_files, nevents=nevents_in, local_rank=None, storage_options=storage_options)

log('Data parameters from input file:')
for key in arr_data.keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[key]}')

arr = arr_data['arr']
arr = arr[:, :, xyslice, xyslice, etaslice, tauslice]

grid_params = {k: arr_data[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
del arr_data
gc.collect()

log(f'nevents (constrained by input file(s)): {arr.shape[0]}')
test_resolutions = [arr.shape[2]]

# ---------------------------------------------------------------------------
# train / val split
# ---------------------------------------------------------------------------
n_train     = int(arr.shape[0] * train_ratio)
data_train  = arr[:n_train]
data_verify = arr[n_train:]
del arr
gc.collect()

log(f'shapes train:  {data_train.shape}')
log(f'shapes verify: {data_verify.shape}')

# ---------------------------------------------------------------------------
# normalisation  (fit on train only, apply to both splits)
# ---------------------------------------------------------------------------
normalizer = build_normalizer(norm_cfg, grid_params)
log(f'normalizer:      {normalizer}')

data_train  = normalizer.fit_transform(data_train)
data_verify = normalizer.transform(data_verify)

normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
dataset_train = FluidDataset(data_train)
dataset_test  = FluidDataset(data_verify)

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
    n_eta = data_train.shape[-2]   # Z dimension after slicing
    n_tau = data_train.shape[-1]   # T dimension after slicing
    tau_vec = make_tau_tensor(grid_params['tau_min'], grid_params['dtau'], n_tau, device=device)
    eta_vec = make_eta_tensor(grid_params['eta_min'], grid_params['deta'], n_eta, device=device)
    _base_model = CausalFNO4DTauEta(
        in_channels=4,
        out_channels=4,
        hidden_channels=hidden_channels,
        n_modes_spatial=tuple(n_modes[:3]),
        **causal_cfg,
    ).to(device)
    log(f'CausalFNO4DTauEta temporal_receptive_field: {_base_model.temporal_receptive_field}')
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
log(f"Model memory usage: {memory_usage:.2f} MB")

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
    save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, model_dir, config_path=args.config)

log_peak_memory(log, cuda=True)

# build title from active modes
_title_parts = [f'FNO Training ({model_type})']
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
# upload outputs to GCS (optional — set gcs_output_prefix in the YAML config)
# ---------------------------------------------------------------------------
if data_source in ('gcs_download', 'gcs_stream') and gcs_output_prefix:
    log(f'Uploading outputs to GCS: {gcs_output_prefix}')
    if log._f:
        log._f.flush()
    for _fname in ['optimizer.pt', 'scheduler.pt', 'regularizer.pt']:
        _p = Path(model_dir) / _fname
        if _p.exists():
            _p.unlink()
            log(f'Removed {_fname} from {model_dir} before upload (redundant: resume state is in last_model.pt)')
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file, show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
