#!/usr/bin/env python

# ---------------------------------------------------------------------------
# workflow_torch/resume.py — resume PyTorch FNO training from a checkpoint.
# Same data pipeline and model selector as train.py; HDF5 *or* ROOT input.
#
# Usage:
#   python resume.py [--config config_resume.yaml]
#
# It restores model weights, optimizer + scheduler state and the epoch counter,
# then runs the remaining epochs up to total_epochs.  The normalizer is loaded
# from the checkpoint directory and reused verbatim (build_dataloaders is given
# normalizer=... so it neither rebuilds nor refits), guaranteeing the resumed
# run sees the same transform as the original.
#
# Resume-specific YAML keys (all others match config.yaml):
#
#   total_epochs        : 200
#       Final epoch target.  remaining = total_epochs − resume_epoch.
#   resume_model_source : local | gcs_download
#   resume_model_dir    : ./checkpoints | gs://bucket/run/checkpoints
#       Directory holding the saved checkpoint.  Defaults to model_dir.
#   local_model_cache   : ./cache/checkpoints        [gcs_download only]
#   checkpoint_type     : last | periodic
#       'last'     — last_model.pt (model + optimizer + scheduler + epoch).
#       'periodic' — checkpoint_N/ subdir written by checkpoint_freq
#                    (model weights only; optimizer restarts from configured lr).
#   checkpoint_epoch    : 50                          [periodic only]
# ---------------------------------------------------------------------------

import argparse
import gc
import json
import sys
import multiprocessing as _mp
from pathlib import Path
from time import perf_counter

import yaml

try:
    _mp.set_start_method('fork')
except RuntimeError:
    pass

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Resume FNO (torch) training on the new data pipeline')
parser.add_argument('--config', default='config_resume.yaml', metavar='FILE',
                    help='YAML configuration file (default: config_resume.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')   # local | gcs_download | gcs_stream
if data_source not in ('local', 'gcs_download', 'gcs_stream'):
    raise ValueError(
        f"data_source must be 'local', 'gcs_download', or 'gcs_stream', got: {data_source!r}")

# --- resume-specific config ------------------------------------------------
resume_model_source = cfg.get('resume_model_source', 'local')
checkpoint_type     = cfg.get('checkpoint_type', 'last')
checkpoint_epoch    = cfg.get('checkpoint_epoch', None)
total_epochs        = int(cfg['total_epochs'])

if resume_model_source not in ('local', 'gcs_download'):
    raise ValueError(
        f"resume_model_source must be 'local' or 'gcs_download', got: {resume_model_source!r}")
if checkpoint_type not in ('last', 'periodic'):
    raise ValueError(f"checkpoint_type must be 'last' or 'periodic', got: {checkpoint_type!r}")
if checkpoint_type == 'periodic' and checkpoint_epoch is None:
    raise ValueError("checkpoint_type is 'periodic' but checkpoint_epoch is not set.")

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
                            save_last_checkpoint, load_last_checkpoint)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer, Normalizer
from downsample import parse_downsample_cfg, downsampled_grid_params
from data import build_source, build_dataloaders
from _model import build_model

# ---------------------------------------------------------------------------
# GCS imports
# ---------------------------------------------------------------------------
gcs_key_file      = cfg.get('gcs_key_file', None)
gcs_output_prefix = cfg.get('gcs_output_prefix', None)
gcs_show_progress = cfg.get('gcs_show_progress', None)

_needs_gcs_data  = data_source in ('gcs_download', 'gcs_stream')
_needs_gcs_model = resume_model_source == 'gcs_download'

if _needs_gcs_data or _needs_gcs_model or gcs_output_prefix:
    from gcs_utils import check_gcs_dependencies
    check_gcs_dependencies(require_storage=(data_source != 'gcs_stream'))
if data_source == 'gcs_download':
    from gcs_utils import download_gcs_files
if data_source == 'gcs_stream':
    from gcs_utils import get_gcs_storage_options
if gcs_output_prefix:
    from gcs_utils import upload_gcs_dir
if _needs_gcs_model:
    from gcs_utils import download_gcs_dir

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

odir              = cfg['odir']
model_dir         = cfg.get('model_dir', odir)
input_dir         = cfg.get('input_dir', '.')
input_files       = cfg['input_files']
lazy              = cfg.get('lazy', True)
nevents_in        = cfg.get('nevents_in', -1)
start_event       = cfg.get('start_event', 0)
choose_ntau       = cfg.get('choose_ntau', None)
local_data_dir    = cfg.get('local_data_dir',   './data')
local_model_cache = cfg.get('local_model_cache', './cache/checkpoints')
overwrite_cache   = cfg.get('overwrite_cache', False)
batch_size        = cfg['batch_size']
hidden_channels   = cfg['hidden_channels']
n_modes           = cfg['n_modes']
log_directly      = cfg.get('log_directly', True)
checkpoint_freq   = cfg.get('checkpoint_freq', None)
min_save_epoch    = cfg.get('min_save_epoch', None)
train_ratio       = cfg.get('train_ratio', 0.8)
xyslice           = _slice(cfg.get('xyslice',  [None, -1]))
etaslice          = _slice(cfg.get('etaslice', [None, None]))
tauslice          = _slice(cfg.get('tauslice', [None, -1]))
lr                = cfg['lr']
weight_decay      = cfg.get('weight_decay', 1e-4)
test_batch_sizes  = cfg.get('test_batch_sizes', [10])
eval_interval     = cfg.get('eval_interval', 2)
save_last         = cfg.get('save_last', True)
norm_cfg          = cfg.get('normalization', {'name': 'none'})
norm_chunk_size   = cfg.get('norm_chunk_size', 64)
model_type        = cfg.get('model_type', 'fno')
causal_cfg        = cfg.get('causal_fno4d', {})

downsample_cfg  = cfg.get('downsample')
loader_cfg      = cfg.get('loader', {}) or {}
num_workers     = loader_cfg.get('num_workers', cfg.get('num_workers', 0))
prefetch_factor = loader_cfg.get('prefetch_factor', 2)
locality_block  = loader_cfg.get('locality_block')

_resume_model_dir_raw = cfg.get('resume_model_dir', model_dir)

# ---------------------------------------------------------------------------
# output dirs + logging (append mode to preserve prior training history)
# ---------------------------------------------------------------------------
_odir_created_msg = None
if not Path(odir).exists():
    Path(odir).mkdir(parents=True, exist_ok=True)
    _odir_created_msg = f"[INFO] created odir '{odir}' (did not exist)"
    print(_odir_created_msg)


class _AppendRunLog(RunLog):
    """RunLog variant that opens the log file in append mode."""
    def __init__(self, path=None):
        import atexit
        self._f = open(path, 'a') if path else None
        if self._f:
            atexit.register(self.close)


log = _AppendRunLog(f'{odir}/training.log' if log_directly else None)
log('\n' + '=' * 72)
log('RESUME RUN')
log('=' * 72)
if _odir_created_msg and log._f:
    log._f.write(_odir_created_msg + '\n'); log._f.flush()

if not Path(model_dir).exists():
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    log(f"[INFO] created model_dir '{model_dir}' (did not exist)")

t0 = perf_counter()

# ---------------------------------------------------------------------------
# resolve input paths + classify type
# ---------------------------------------------------------------------------
if not input_files:
    raise ValueError("input_files is empty — set at least one file in the YAML config.")

_is_h5 = all(str(f).lower().endswith(('.h5', '.hdf5')) for f in input_files)
if data_source == 'gcs_stream' and _is_h5:
    raise ValueError("data_source 'gcs_stream' is ROOT-only — use 'gcs_download' for HDF5.")

storage_options = None
if data_source == 'local':
    source_paths = [str(Path(input_dir) / f) for f in input_files]
    for p in source_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input file not found: {p}")
elif data_source == 'gcs_download':
    log(f'Downloading {len(input_files)} file(s) from GCS ...')
    _t_dl = perf_counter()
    source_paths = download_gcs_files(input_files, local_data_dir, gcs_key_file,
                                      overwrite=overwrite_cache, show_progress=gcs_show_progress)
    log(f'Download complete in {perf_counter() - _t_dl:.1f}s')
else:  # gcs_stream (ROOT)
    storage_options = get_gcs_storage_options(gcs_key_file)
    source_paths = list(input_files)

# ---------------------------------------------------------------------------
# configuration summary
# ---------------------------------------------------------------------------
log(f'config:              {args.config}')
log(f'backend:             torch')
log(f'data_source:         {data_source}')
log(f'input type:          {"HDF5" if _is_h5 else "ROOT"}')
log(f'input_files:         {input_files}')
log(f'lazy:                {lazy}')
log(f'nevents_in:          {nevents_in}')
log(f'total_epochs:        {total_epochs}')
log(f'batch_size:          {batch_size}')
log(f'hidden_channels:     {hidden_channels}')
log(f'n_modes:             {n_modes}')
log(f'train_ratio:         {train_ratio}')
log(f'model_dir:           {model_dir}')
log(f'normalization:       {norm_cfg}')
log(f'model_type:          {model_type}')
log(f'resume_model_source: {resume_model_source}')
log(f'resume_model_dir:    {_resume_model_dir_raw}')
log(f'checkpoint_type:     {checkpoint_type}')
if checkpoint_type == 'periodic':
    log(f'checkpoint_epoch:    {checkpoint_epoch}')

# ---------------------------------------------------------------------------
# resolve checkpoint directory (download from GCS if needed)
# ---------------------------------------------------------------------------
if resume_model_source == 'gcs_download':
    if not gcs_key_file:
        raise ValueError("gcs_key_file is required when resume_model_source is 'gcs_download'")
    _cache = Path(local_model_cache)
    _cache.mkdir(parents=True, exist_ok=True)
    if checkpoint_type == 'periodic':
        gs_ckpt   = f"{str(_resume_model_dir_raw).rstrip('/')}/checkpoint_{checkpoint_epoch}"
        local_sub = _cache / f'checkpoint_{checkpoint_epoch}'
        log(f'Downloading periodic checkpoint from GCS: {gs_ckpt}')
        download_gcs_dir(gs_ckpt, str(local_sub), gcs_key_file,
                         overwrite=overwrite_cache, show_progress=gcs_show_progress)
        ckpt_dir = local_sub
    else:
        log(f'Downloading checkpoint directory from GCS: {_resume_model_dir_raw}')
        download_gcs_dir(str(_resume_model_dir_raw), str(_cache), gcs_key_file,
                         overwrite=overwrite_cache, show_progress=gcs_show_progress)
        ckpt_dir = _cache
else:  # local
    ckpt_dir = (Path(_resume_model_dir_raw) / f'checkpoint_{checkpoint_epoch}'
                if checkpoint_type == 'periodic' else Path(_resume_model_dir_raw))

if not ckpt_dir.exists():
    raise FileNotFoundError(
        f"Checkpoint directory not found: {ckpt_dir.resolve()}\n"
        f"  resume_model_dir={_resume_model_dir_raw}, checkpoint_type={checkpoint_type}"
        + (f", checkpoint_epoch={checkpoint_epoch}" if checkpoint_type == 'periodic' else ""))
log(f'Checkpoint dir: {ckpt_dir.resolve()}')

# ---------------------------------------------------------------------------
# build source
# ---------------------------------------------------------------------------
log(f'\nBuilding source from {len(source_paths)} file(s)  lazy={lazy} ...')
source, grid_params = build_source(
    source_paths, storage_options=storage_options, lazy=lazy,
    start_event=start_event, nevents=nevents_in, choose_ntau=choose_ntau, verbose=True)
log(f'source:      {type(source).__name__}  shape={tuple(source.shape)}')
log(f'grid_params: {grid_params}')

n_total = len(source) if nevents_in is None or nevents_in <= 0 else min(nevents_in, len(source))
log(f'total events used: {n_total}  →  train={int(n_total * train_ratio)}  '
    f'val={n_total - int(n_total * train_ratio)}')

_dkw = parse_downsample_cfg(downsample_cfg)
grid_params_eff = downsampled_grid_params(grid_params, **_dkw) if _dkw else grid_params

# ---------------------------------------------------------------------------
# normalizer — load from checkpoint and reuse verbatim (no rebuild / refit)
# ---------------------------------------------------------------------------
_norm_candidates = [ckpt_dir / 'normalizer.npz', Path(model_dir) / 'normalizer.npz']
_norm_path = next((p for p in _norm_candidates if p.exists()), None)
if _norm_path is not None:
    log(f'Loading normalizer from checkpoint: {_norm_path}')
    normalizer = Normalizer.load(str(_norm_path))
else:
    log('[WARN] normalizer.npz not found in checkpoint or model_dir — building fresh from norm_cfg.')
    normalizer = None     # build_dataloaders will build + fit from norm_cfg

# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# split + datasets + loaders (reuse loaded normalizer)
# ---------------------------------------------------------------------------
train_loader, test_loaders, normalizer = build_dataloaders(
    source, n_total=n_total, train_ratio=train_ratio,
    xyslice=xyslice, etaslice=etaslice, tauslice=tauslice,
    norm_cfg=norm_cfg, grid_params=grid_params,
    backend='torch', batch_size=batch_size, test_batch_sizes=test_batch_sizes,
    num_workers=num_workers, prefetch_factor=prefetch_factor,
    downsample_cfg=downsample_cfg, locality_block=locality_block,
    norm_chunk_size=norm_chunk_size, normalizer=normalizer)

log(f'normalizer: {normalizer}')
# Re-save so model_dir stays in sync with the resumed weights.
normalizer.save(f'{model_dir}/normalizer.npz')
log(f'normalizer  →  {model_dir}/normalizer.npz')

test_resolutions = list(test_loaders.keys())
_sample = next(iter(train_loader))
_sample_x_shape = tuple(_sample['x'].shape)
log(f'batch x shape: {_sample_x_shape}  y shape: {tuple(_sample["y"].shape)}')
del _sample
gc.collect()

# ---------------------------------------------------------------------------
# model (selector — must match the checkpoint's architecture)
# ---------------------------------------------------------------------------
model = build_model(
    model_type, n_modes=n_modes, hidden_channels=hidden_channels, device=device,
    sample_x_shape=_sample_x_shape, grid_params=grid_params_eff,
    causal_cfg=causal_cfg, model_dir=model_dir, save_embedding=False, log=log)

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

trainer = Trainer(model=model, n_epochs=total_epochs,
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
# load checkpoint and determine resume epoch
# ---------------------------------------------------------------------------
log('\n--- Loading checkpoint ---')
if checkpoint_type == 'last':
    resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)
else:  # periodic — neuralop checkpoint() saves model weights only
    _last_pt = ckpt_dir / 'last_model.pt'
    _sf_path = ckpt_dir / 'best_model.safetensors'
    if _last_pt.exists():
        resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)
    elif _sf_path.exists():
        from safetensors.torch import load_file as _sf_load
        model.load_state_dict(_sf_load(str(_sf_path)))
        log(f'Torch model weights loaded from {_sf_path}')
        log('[WARN] Periodic checkpoint has no optimizer state — optimizer restarts from configured lr.')
        _state_f = ckpt_dir / 'state.json'
        if _state_f.exists():
            with open(_state_f) as _fs:
                resume_epoch = int(json.load(_fs).get('epoch', checkpoint_epoch)) + 1
        else:
            resume_epoch = checkpoint_epoch + 1
    else:
        raise FileNotFoundError(
            f"No recognized model file in periodic checkpoint dir: {ckpt_dir}\n"
            "Expected one of: best_model.safetensors, last_model.pt")

remaining_epochs = total_epochs - resume_epoch
if remaining_epochs <= 0:
    log(f'[INFO] Already at or past epoch {resume_epoch - 1} (total_epochs={total_epochs}). '
        'Increase total_epochs to continue.')
    sys.exit(0)

trainer.n_epochs = remaining_epochs
log(f'Resuming from epoch {resume_epoch}  →  training {remaining_epochs} more epochs '
    f'(target total: {total_epochs})')

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
    save_last_checkpoint(trainer, optimizer, scheduler, total_epochs, model_dir,
                         config_path=args.config)

log_peak_memory(log, cuda=True)

log('Shutting down DataLoader workers ...')
del train_loader
test_loaders.clear()
gc.collect()

_title_parts = [f'FNO Resume ({model_type}, {"HDF5" if _is_h5 else "ROOT"})']
if lazy:
    _title_parts.append('lazy')
_title = _title_parts[0] + (f' ({", ".join(_title_parts[1:])})' if len(_title_parts) > 1 else '')

history.save(f'{odir}/metrics_resume.json')
history.plot(save_path=f'{odir}/training_curves_resume.svg', title=_title)
log(f'metrics   → {odir}/metrics_resume.json')
log(f'curves    → {odir}/training_curves_resume.svg')
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
