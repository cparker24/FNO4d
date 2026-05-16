#!/usr/bin/env python

# ---------------------------------------------------------------------------
# MLX FNO training — GCS streaming variant.
#
# Reads ROOT files directly from Google Cloud Storage (gs:// URIs) via
# uproot + fsspec + gcsfs.  No local copy of the data files is made.
#
# Usage:
#   python test_train_mlx_yaml_norm_gcs_stream.py --config config_gcs_stream.yaml
#
# Required extra in YAML (in addition to the standard training keys):
#
#   gcs_key_file: /path/to/service_account.json
#
#   input_files:
#     - gs://my-bucket/data/file1.root
#     - gs://my-bucket/data/file2.root
#
# The 'input_dir' key is ignored — input_files must be full gs:// URIs.
#
# GCS dependencies:  pip install -e '../loc_libs[gcs]'
#                or: pip install -r ../requirements-gcs.txt
#
# Normalisation block (optional, default: none):
#   normalization:
#     name: none   # none | etau75_3d | etau75_eta_3d | etau75_eta_shape_3d
#                  # | cosh_etau75_3d
#     alpha: 1.0   # only used by cosh_etau75_3d
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
from pathlib import Path
import mlx.core as mx
import mlx.optimizers as optim_mlx
from time import perf_counter
from torch.utils.data import DataLoader

from neuralop_mlx.models import FNO
from neuralop_mlx import Trainer, LpLoss, H1Loss, AdamW
from neuralop_mlx.utils import count_model_params

sys.path.insert(0, '../loc_libs')
from read_3d_root import read_3d_data
from fluid_datasets import FluidDatasetMLX
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import RunLog, patch_trainer_logging, patch_trainer_checkpoint, save_last_checkpoint
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer
from gcs_utils import get_gcs_storage_options, upload_gcs_dir, check_gcs_dependencies

# ---------------------------------------------------------------------------
# check GCS deps early so the error is clear before any data loading
# ---------------------------------------------------------------------------
check_gcs_dependencies(require_storage=False)

# ---------------------------------------------------------------------------
# load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO (MLX) from GCS — streaming mode')
parser.add_argument('--config', default='config_gcs_stream.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config_gcs_stream.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

def _slice(val):
    return slice(None) if val is None else slice(*val)

odir            = cfg['odir']
input_files     = cfg['input_files']   # full gs:// URIs
n_epochs        = cfg['n_epochs']
batch_size      = cfg['batch_size']
hidden_channels = cfg['hidden_channels']
n_modes         = cfg['n_modes']
nevents_in      = cfg.get('nevents_in', 10)
log_directly    = cfg.get('log_directly', True)
checkpoint_freq = cfg.get('checkpoint_freq', None)
train_ratio     = cfg.get('train_ratio', 0.8)
xyslice         = _slice(cfg.get('xyslice', [None, -1]))
tauslice        = _slice(cfg.get('tauslice', [None, -1]))
lr               = cfg['lr']
weight_decay     = cfg.get('weight_decay', 1e-4)
test_batch_sizes = cfg.get('test_batch_sizes', [10])
eval_interval    = cfg.get('eval_interval', 2)
save_last        = cfg.get('save_last', True)
model_dir        = cfg.get('model_dir', odir)
norm_cfg         = cfg.get('normalization', {'name': 'none'})
gcs_key_file        = cfg['gcs_key_file']
gcs_output_prefix   = cfg.get('gcs_output_prefix', None)

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
log = RunLog(f'{odir}/training_mlx.log' if log_directly else None)

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
log(f'input files:     {input_files}')
log(f'gcs_key_file:    {gcs_key_file}')
log(f'gcs_output_prefix: {gcs_output_prefix}')
log(f'mode:            GCS streaming (gs:// via fsspec/gcsfs)')
log(f'n_epochs:        {n_epochs}')
log(f'batch_size:      {batch_size}')
log(f'hidden_channels: {hidden_channels}')
log(f'checkpoint_freq: {checkpoint_freq}')
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
# GCS storage options (passed through to uproot.open via fsspec)
# ---------------------------------------------------------------------------
storage_options = get_gcs_storage_options(gcs_key_file)

# ---------------------------------------------------------------------------
# load data from GCS (streamed — no local copy)
# ---------------------------------------------------------------------------
log(f'Events called for {nevents_in}')

arr_data = [
    read_3d_data(f, nevents=nevents_in, local_rank=None, storage_options=storage_options)
    for f in input_files
]

if len(arr_data) == 0:
    raise ValueError("No data found in the specified input files.")

log('Data parameters from input file:')
for key in arr_data[0].keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[0][key]}')

arr = arr_data[0]['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]

grid_params = {k: arr_data[0][k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
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
device = 'gpu'
log(f'Using device: {device} (MLX Metal)')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
dataset_train = FluidDatasetMLX(data_train)
dataset_test  = FluidDatasetMLX(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    log(f'res {res}')
    test_loaders[res] = DataLoader(
        dataset_test,
        batch_size=test_bsize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
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
)

memory_usage = calculate_model_memory(model, count_fn=count_model_params)
log(f"Model memory usage: {memory_usage:.2f} MB")

# ---------------------------------------------------------------------------
# optimizer, losses, trainer
# ---------------------------------------------------------------------------
optimizer = AdamW(lr=lr, weight_decay=weight_decay)
scheduler = optim_mlx.cosine_decay(lr, T_max := 30)

l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)
train_loss  = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

trainer = Trainer(
    model=model,
    n_epochs=n_epochs,
    device=device,
    wandb_log=False,
    eval_interval=eval_interval,
    use_distributed=False,
    verbose=True,
)

history = TrainingHistory()
patch_trainer_logging(trainer, log, history=history)

if checkpoint_freq is not None and checkpoint_freq > 0:
    patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, model_dir, optimizer=optimizer)

save_best_key = f'{test_resolutions[0]}_l2'

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
reset_peak_memory(mlx=True)
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    save_dir=model_dir,
    save_best=save_best_key,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

if save_last:
    save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, model_dir)
log_peak_memory(log, mlx=True)
history.save(f'{odir}/metrics_mlx.json')
history.plot(save_path=f'{odir}/training_curves_mlx.svg', title='FNO Training MLX (GCS stream)')
log(f'metrics   → {odir}/metrics_mlx.json')
log(f'curves    → {odir}/training_curves_mlx.svg')
log(f'total time: {perf_counter() - t0:.1f}s')

# ---------------------------------------------------------------------------
# upload outputs to GCS (optional — set gcs_output_prefix in the YAML config)
# ---------------------------------------------------------------------------
if gcs_output_prefix:
    log(f'Uploading outputs to GCS: {gcs_output_prefix}')
    if log._f:
        log._f.flush()   # ensure the log file is complete before uploading
    #upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file)
    upload_gcs_dir(odir,      f'{gcs_output_prefix}/logs',        gcs_key_file)
    log(f'GCS upload complete → {gcs_output_prefix}')
