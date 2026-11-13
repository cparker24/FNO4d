#!/usr/bin/env python

# ---------------------------------------------------------------------------
# FNO training script — pure PyTorch, data source steered by YAML.
#
# Usage:
#   python train_torch.py [--config config.yaml]
#
# One key in the YAML selects the data source:
#
#   data_source: local          # local | gcs_download | gcs_stream
#
# All three data source options are supported.  See config.yaml for the full
# list of available options and which keys apply to each mode.
# ---------------------------------------------------------------------------

import argparse
import yaml
import sys
import gc
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train FNO from YAML config')
parser.add_argument('--config', default='config.yaml', metavar='FILE',
                    help='path to YAML configuration file (default: config.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')  # local | gcs_download | gcs_stream

if data_source not in ('local', 'gcs_download', 'gcs_stream'):
    raise ValueError(f"data_source must be 'local', 'gcs_download', or 'gcs_stream', got: {data_source!r}")

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import torch
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
from masked_loss import TailMaskedLoss

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
# Dedicated validation files, read and normalized the same way as
# input_files but never seen during training -- an alternative to
# train_ratio's positional split of input_files itself. When set,
# train_ratio is ignored and ALL of input_files is used for training.
# Intended for held-out validation events generated separately (different
# random seed) from the training data, so the eval split is a genuine
# distinct draw rather than the tail slice of the same files.
validation_files = cfg.get('validation_files', None)
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
min_ntau_freezeout = cfg.get('min_ntau_freezeout', 10)
drop_channels   = cfg.get('drop_channels', None)
# Storage dtype for the master train/val arrays held in host RAM -- 'float32'
# (default, unchanged behavior) or 'float16' to halve their resident size on
# a host-RAM-bound run. Every per-batch consumer already upcasts to float32
# before the model sees it (FluidDataset._to_tensor / collate.py), so this
# only trades precision in the long-lived host array, not model input dtype.
storage_dtype   = np.dtype(cfg.get('storage_dtype', 'float32'))
lr              = cfg['lr']
weight_decay    = cfg.get('weight_decay', 1e-4)
scheduler_cfg   = cfg.get('scheduler', {'name': 'cosine_annealing'})
scheduler_name  = scheduler_cfg.get('name', 'cosine_annealing')
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
log(f'validation_files: {validation_files}')
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
log(f'xyslice:         {xyslice}')
log(f'etaslice:        {etaslice}')
log(f'tauslice:        {tauslice}')
log(f'min_ntau_freezeout: {min_ntau_freezeout}')
log(f'drop_channels:   {drop_channels}')
log(f'train_ratio:     {train_ratio}' + (' (ignored -- validation_files set)' if validation_files else ''))
log(f'log_directly:    {log_directly}')
log(f'nevents(in):     {nevents_in}')
log(f'AdamW: lr:         {lr}')
log(f'AdamW: weight_decay: {weight_decay}')
log(f'scheduler:         {scheduler_cfg}')
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

# read_3d_data pads every event to choose_ntau tau steps *before* tauslice is
# applied below. Left to auto-detect, choose_ntau is the global max
# ntau_freezeout across all events -- which can be far larger than what
# tauslice actually keeps (e.g. 0-10% centrality events freeze out much later
# than peripheral ones), wasting memory on tau steps that get sliced off
# immediately after. Pad only to what tauslice.stop will keep instead.
choose_ntau = tauslice.stop if isinstance(tauslice.stop, int) and tauslice.stop > 0 else None
log(f'choose_ntau (pre-slice padding target): {choose_ntau}')

# xyslice/etaslice/tauslice and min_ntau_freezeout are now passed straight
# into read_3d_data, which crops and filters DURING the chunked ROOT read
# (per-chunk, a few events at a time) instead of after returning the full,
# all-events, raw-resolution array. Previously the crop/filter happened here,
# as a fused mask+slice index on the full returned array -- correct, but
# that index still has to hold both the full array AND its sliced output in
# memory at once (numpy indexing can't do this in place). On the doubled
# (eta4/n65) grid that peak (~123GiB + ~116GiB =~ 239GiB) exceeded the
# instance's RAM and OOM-killed the process; see
# js-contrib contribs/FnoHydro/PARTICLIZATION_INVESTIGATION.md's "widened
# eta4/n65 retrain: host-RAM OOM during data loading" entry. Doing it inside
# read_3d_data means the full raw-resolution array is never materialised at
# all -- only the already-cropped/filtered one, at roughly its final size.
_read_kwargs = dict(xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
                     min_ntau_freezeout=min_ntau_freezeout, drop_channels=drop_channels,
                     dtype=storage_dtype)

def _load(files, label):
    """Dispatch a file list through the configured data_source, exactly as
    the single input_files read used to -- factored out so validation_files
    (a second, independent file list) can be read the same way without
    duplicating the data_source if/elif/else three times."""
    if data_source == 'local':
        return read_3d_data([f'{input_dir}/{f}' for f in files], nevents=nevents_in, choose_ntau=choose_ntau, local_rank=None, **_read_kwargs)
    elif data_source == 'gcs_download':
        log(f'Downloading {len(files)} {label} file(s) from GCS ...')
        t_dl = perf_counter()
        local_files = download_gcs_files(
            files, local_data_dir, gcs_key_file, overwrite=overwrite_cache,
            show_progress=gcs_show_progress,
        )
        log(f'Download complete in {perf_counter() - t_dl:.1f}s')
        for gs_uri, local_path in zip(files, local_files):
            log(f'  {gs_uri}  →  {local_path}')
        return read_3d_data(local_files, nevents=nevents_in, choose_ntau=choose_ntau, local_rank=None, **_read_kwargs)
    else:  # gcs_stream
        storage_options = get_gcs_storage_options(gcs_key_file)
        return read_3d_data(files, nevents=nevents_in, choose_ntau=choose_ntau, local_rank=None, storage_options=storage_options, **_read_kwargs)

arr_data = _load(input_files, 'training')
arr_data_val = _load(validation_files, 'validation') if validation_files else None

log('Data parameters from input file:')
for key in arr_data.keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[key]}')
if arr_data_val is not None:
    log('Data parameters from validation file:')
    for key in arr_data_val.keys():
        if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
            log(f'  {key}: {arr_data_val[key]}')

# arr_data['arr'] is already cropped (xyslice/etaslice/tauslice) and filtered
# (min_ntau_freezeout) by read_3d_data -- no further indexing needed here.
_t = perf_counter()
arr = arr_data['arr']
grid_params = {k: arr_data[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
del arr_data
if arr_data_val is not None:
    arr_val = arr_data_val['arr']
    # Must match on everything but the event-count axis for the same model/
    # normalizer/test_loader to apply to both -- a real mismatch here (e.g.
    # validation data generated on a different grid) would silently corrupt
    # training rather than error out downstream, so check it explicitly.
    assert arr_val.shape[1:] == arr.shape[1:], (
        f'validation_files grid/feature shape {arr_val.shape[1:]} does not '
        f'match input_files shape {arr.shape[1:]}')
    del arr_data_val
gc.collect()
log(f'[TIMING] del arr_data + gc.collect(): {perf_counter() - _t:.1f}s')

# drop_channels is applied INSIDE read_3d_data (via _read_kwargs above), not
# here -- it has to be, to keep the "crop during the chunked read, never
# materialise the undropped array" property every other crop option
# (xyslice/etaslice/tauslice/min_ntau_freezeout) already relies on. A
# post-hoc np.delete() on the full returned array would briefly hold both the
# original and the smaller copy in memory at once, the exact failure mode
# that OOM-killed the eta4/n65 host-RAM read (see the comment above
# _read_kwargs). By the time arr reaches here it is already at its final,
# post-drop feature count.
if drop_channels:
    log(f'Dropped feature channel(s) {drop_channels} during the chunked read '
        f'(full order is [energy_density, temperature, vx, vy, vz]) -- tests '
        f'whether explicit temperature supervision helps or hurts vx/vy/vz '
        f'reproduction, see FNO_VZ_BIAS_INVESTIGATION.md. A model trained this '
        f'way has n_features=4, directly loadable by FnoHydro.cc\'s existing '
        f'n_features==4 inference path (implicit temperature from channel 0 '
        f'via the EOS on read-back).')

log(f'nevents (constrained by input file(s)): {arr.shape[0]}')
n_features = arr.shape[1]
log(f'n_features (from data): {n_features}')
test_resolutions = [arr.shape[2]]

# ---------------------------------------------------------------------------
# train / val split
# ---------------------------------------------------------------------------
if validation_files:
    # All of input_files trains; validation_files (read separately above) is
    # the entire eval set -- no positional split of input_files itself.
    data_train  = arr
    data_verify = arr_val
    del arr, arr_val
else:
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

# in_place=True: data_train/data_verify's old references are discarded
# immediately below, so normalizing in place (vs. allocating a full second
# copy) is safe here -- see NormFn3d.etau75_3d's docstring for why that
# second large allocation matters on this grid size.
_t = perf_counter()
data_train  = normalizer.fit_transform(data_train, in_place=True)
log(f'[TIMING] normalizer.fit_transform(data_train): {perf_counter() - _t:.1f}s')
_t = perf_counter()
data_verify = normalizer.transform(data_verify, in_place=True)
log(f'[TIMING] normalizer.transform(data_verify): {perf_counter() - _t:.1f}s')

_t = perf_counter()
normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'[TIMING] normalizer.save(): {perf_counter() - _t:.1f}s')
log(f'normalizer  →  {normalizer_path}')

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
_t = perf_counter()
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
log(f'[TIMING] dataset/dataloader construction: {perf_counter() - _t:.1f}s')

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
model = FNO(
    in_channels=n_features,
    out_channels=n_features,
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
if scheduler_name == 'cosine_annealing':
    # Single smooth decay from lr down to eta_min (default 0) over the whole run.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
elif scheduler_name == 'cosine_warm_restarts':
    # Periodically jumps lr back up to its initial value every T_0 epochs
    # (each subsequent cycle T_mult times as long as the last), instead of
    # decaying once to ~0 -- lets the optimizer re-explore a wider region
    # after settling into a minimum, at the cost of repeatedly disturbing
    # whatever it just converged to.
    T_0    = scheduler_cfg.get('T_0', 20)
    T_mult = scheduler_cfg.get('T_mult', 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)
else:
    raise ValueError(f"Unknown scheduler.name {scheduler_name!r}; expected 'cosine_annealing' or 'cosine_warm_restarts'")

l2loss = TailMaskedLoss(LpLoss(d=4, p=2))
h1loss = TailMaskedLoss(H1Loss(d=4))
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
# Source - https://stackoverflow.com/a/62556666
# Posted by SHAGUN SHARMA, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-10, License - CC BY-SA 4.0

torch.cuda.memory_summary(device=None, abbreviated=False)

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
_title_parts = ['FNO Training']
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
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file, show_progress=gcs_show_progress)
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file, show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
