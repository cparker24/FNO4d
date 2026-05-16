#!/usr/bin/env python

# ---------------------------------------------------------------------------
# PyTorch FNO training — parameters loaded from a YAML config file.
# Extends test_train_yaml.py with optional data normalisation via NormFn3d.
#
# Usage: python test_train_yaml_norm.py --config config_test_train.yaml
#
# Add a  normalization:  block to the YAML to activate (default: none):
#
#   normalization:
#     name: none              # none | etau75_3d | etau75_eta_3d |
#                             # etau75_eta_shape_3d | cosh_etau75_3d
#     alpha: 1.0              # only used by cosh_etau75_3d
#
# The normalizer state (grid params + ref) is saved to
#   {model_dir}/normalizer.npz
# and can be loaded at inference time with Normalizer.load(path).
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
import torch
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss

sys.path.insert(0, '../loc_libs')
from read_3d_root import read_3d_data
from fluid_datasets import FluidDataset
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import RunLog, patch_trainer_logging, patch_trainer_checkpoint, patch_trainer_min_save_epoch, save_last_checkpoint
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='neuralop')

# ---------------------------------------------------------------------------
# load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO (PyTorch) from YAML config')
parser.add_argument('--config', default='config_test_train.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config_test_train.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

def _slice(val):
    return slice(None) if val is None else slice(*val)

odir            = cfg['odir']
input_dir       = cfg['input_dir']
input_files     = cfg['input_files']
n_epochs        = cfg['n_epochs']
batch_size      = cfg['batch_size']
hidden_channels = cfg['hidden_channels']
n_modes         = cfg['n_modes']
nevents_in      = cfg.get('nevents_in', -1)
log_directly    = cfg.get('log_directly', True)
checkpoint_freq  = cfg.get('checkpoint_freq', None)
min_save_epoch   = cfg.get('min_save_epoch', None)
train_ratio      = cfg.get('train_ratio', 0.8)
xyslice         = _slice(cfg.get('xyslice', [None, -1]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
lr               = cfg['lr']
weight_decay     = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
eval_interval    = cfg.get('eval_interval', 2)
save_last        = cfg.get('save_last', True)
model_dir        = cfg.get('model_dir', odir)
norm_cfg         = cfg.get('normalization', {'name': 'none'})

# ---------------------------------------------------------------------------
# ensure output directories exist
# ---------------------------------------------------------------------------
_odir_created_msg = None
if not Path(odir).exists():
    Path(odir).mkdir(parents=True, exist_ok=True)
    _odir_created_msg = f"[INFO] created odir '{odir}' (did not exist)"
    print(_odir_created_msg)

# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
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
log(f'input dir:       {input_dir}')
log(f'input files:     {input_files}')
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
# load data
# ---------------------------------------------------------------------------
log(f'Events called for {nevents_in}')

if not input_files:
    raise ValueError("No data found in the specified input files.")
arr_data = read_3d_data([f'{input_dir}/{f}' for f in input_files], nevents=nevents_in, local_rank=None)

log('Data parameters from input file:')
for key in arr_data.keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[key]}')

arr = arr_data['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]

grid_params = {k: arr_data[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
del arr_data
gc.collect()

log(f'nevents (constrained by input file): {arr.shape[0]}')
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
normalizer  = build_normalizer(norm_cfg, grid_params)
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
model = FNO(
    in_channels=4,
    out_channels=4,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
).to(device)

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

if min_save_epoch is not None and min_save_epoch > 0:
    patch_trainer_min_save_epoch(trainer, Trainer, min_save_epoch)

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
              training_loss=train_loss,
              eval_losses=eval_losses)

if save_last:
    save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, model_dir, config_path=args.config)
log_peak_memory(log, cuda=True)
history.save(f'{odir}/metrics.json')
history.plot(save_path=f'{odir}/training_curves.svg', title='FNO Training')
log(f'metrics   → {odir}/metrics.json')
log(f'curves    → {odir}/training_curves.svg')
log(f'total time: {perf_counter() - t0:.1f}s')
