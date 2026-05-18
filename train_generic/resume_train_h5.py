#!/usr/bin/env python

# ---------------------------------------------------------------------------
# resume_train_h5.py — Resume FNO training from a saved checkpoint.
# Reads pre-converted HDF5 files; identical data pipeline to train_h5.py.
#
# Usage:
#   python resume_train_h5.py [--config config_h5_resume.yaml]
#
# This script is identical to train_h5.py except that instead of starting
# from a randomly-initialised model, it:
#   1. Downloads (optionally via GCS) and loads a saved checkpoint.
#   2. Restores model weights, optimizer state, and epoch counter.
#   3. Runs only the remaining epochs up to total_epochs.
#
# Resume-specific YAML keys (all others are identical to config_h5.yaml):
#
#   total_epochs        : 200
#       Final epoch target.  remaining = total_epochs − resume_epoch.
#       Must be > last saved epoch.
#
#   resume_model_source : local | gcs_download
#       'local'        — checkpoint lives on the local filesystem.
#       'gcs_download' — download from GCS first, then load locally.
#
#   resume_model_dir    : ./checkpoints | gs://bucket/runs/run01/checkpoints
#       Local path or GCS URI of the directory containing the saved
#       checkpoint files.  Defaults to model_dir if omitted.
#
#   local_model_cache   : ./cache/checkpoints
#       [gcs_download only] Where to cache the downloaded checkpoint.
#
#   checkpoint_type     : last | periodic
#       'last'     — loads last_model.pt (torch) or
#                    last_model.safetensors + last_state.json (mlx).
#                    Written by save_last_checkpoint / save_last: true.
#       'periodic' — loads from checkpoint_N/ subdirectory written by
#                    checkpoint_freq.  Use checkpoint_epoch to name N.
#
#   checkpoint_epoch    : 50
#       [periodic only] Epoch number N of the checkpoint subdirectory.
#
#   resume_backend      : null | torch | mlx
#       Backend that produced the saved checkpoint.  null (default) means
#       the same as backend.
#       Set to 'torch' when backend is 'mlx' for a torch→mlx weight
#       transfer via neuralop_mlx.weight_transfer.load_from_torch.
#       Cross-backend transfer only supports checkpoint_type: last and
#       does NOT restore optimizer state (optimizer restarts fresh).
#
# Periodic-checkpoint optimizer state
# ------------------------------------
# MLX   — restored if optimizer.safetensors is present (written by
#          patch_trainer_checkpoint when optimizer= is passed).
# Torch — neuralop's checkpoint() saves model weights only; optimizer
#          state is NOT included in periodic saves, so the optimizer
#          restarts from the configured lr.
# ---------------------------------------------------------------------------

import argparse
import json
import sys
import gc
import numpy as np
from pathlib import Path
from time import perf_counter
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# parse args and load config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Resume FNO training from HDF5 checkpoint')
parser.add_argument('--config', default='config_h5_resume.yaml', metavar='FILE',
                    help='YAML configuration file (default: config_h5_resume.yaml)')
args = parser.parse_args()

import yaml
with open(args.config) as f:
    cfg = yaml.safe_load(f)

backend     = cfg.get('backend',     'torch')   # torch | mlx
data_source = cfg.get('data_source', 'local')   # local | gcs_download

if backend not in ('torch', 'mlx'):
    raise ValueError(f"backend must be 'torch' or 'mlx', got: {backend!r}")
if data_source not in ('local', 'gcs_download'):
    raise ValueError(f"data_source must be 'local' or 'gcs_download', got: {data_source!r}")

# --- resume-specific config ------------------------------------------------
resume_model_source = cfg.get('resume_model_source', 'local')
resume_backend      = cfg.get('resume_backend', None) or backend
checkpoint_type     = cfg.get('checkpoint_type', 'last')
checkpoint_epoch    = cfg.get('checkpoint_epoch', None)
total_epochs        = int(cfg['total_epochs'])

if resume_model_source not in ('local', 'gcs_download'):
    raise ValueError(
        f"resume_model_source must be 'local' or 'gcs_download', got: {resume_model_source!r}")
if checkpoint_type not in ('last', 'periodic'):
    raise ValueError(
        f"checkpoint_type must be 'last' or 'periodic', got: {checkpoint_type!r}")
if checkpoint_type == 'periodic' and checkpoint_epoch is None:
    raise ValueError("checkpoint_type is 'periodic' but checkpoint_epoch is not set.")
if resume_backend not in ('torch', 'mlx'):
    raise ValueError(f"resume_backend must be 'torch' or 'mlx', got: {resume_backend!r}")

cross_backend = (resume_backend != backend)
if cross_backend and checkpoint_type != 'last':
    raise ValueError(
        "Cross-backend weight transfer (torch→mlx) only supports "
        "checkpoint_type: last.  Set checkpoint_type to 'last' or use the same backend.")

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

sys.path.insert(0, str(Path(__file__).parent.parent / 'loc_libs'))
from read_3d_hdf5 import read_3d_data_hdf5
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import (RunLog, patch_trainer_logging, patch_trainer_checkpoint,
                             save_last_checkpoint,
                             load_last_checkpoint)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer, Normalizer

if backend == 'torch':
    from fluid_datasets import FluidDatasetH5
else:
    from fluid_datasets import FluidDatasetH5MLX

# ---------------------------------------------------------------------------
# GCS imports
# ---------------------------------------------------------------------------
gcs_key_file      = cfg.get('gcs_key_file', None)
gcs_output_prefix = cfg.get('gcs_output_prefix', None)
gcs_show_progress = cfg.get('gcs_show_progress', None)

_needs_gcs_data  = data_source == 'gcs_download'
_needs_gcs_model = resume_model_source == 'gcs_download'

if _needs_gcs_data or _needs_gcs_model or gcs_output_prefix:
    from gcs_utils import check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)

if _needs_gcs_data or gcs_output_prefix:
    from gcs_utils import download_gcs_files, upload_gcs_dir

if _needs_gcs_model:
    from gcs_utils import download_gcs_dir

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

odir              = cfg['odir']
model_dir         = cfg.get('model_dir', odir)
hdf5_dir          = cfg.get('hdf5_dir', '.')
hdf5_files        = cfg['hdf5_files']
lazy              = cfg.get('lazy', True)
nevents_in        = cfg.get('nevents_in', -1)
start_event       = cfg.get('start_event', 0)
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
tauslice          = _slice(cfg.get('tauslice', [None, -1]))
lr                = cfg['lr']
weight_decay      = cfg.get('weight_decay', 1e-4)
test_batch_sizes  = cfg.get('test_batch_sizes', [10])
num_workers       = cfg.get('num_workers', 0)
eval_interval     = cfg.get('eval_interval', 2)
save_last         = cfg.get('save_last', True)
norm_cfg          = cfg.get('normalization', {'name': 'none'})

# resume_model_dir defaults to model_dir when not specified
_resume_model_dir_raw = cfg.get('resume_model_dir', model_dir)

# ---------------------------------------------------------------------------
# output dirs + logging (append mode to preserve prior training history)
# ---------------------------------------------------------------------------
_odir_created_msg = None
if not Path(odir).exists():
    Path(odir).mkdir(parents=True, exist_ok=True)
    _odir_created_msg = f"[INFO] created odir '{odir}' (did not exist)"
    print(_odir_created_msg)

_sfx = '_mlx' if backend == 'mlx' else ''


class _AppendRunLog(RunLog):
    """RunLog variant that opens the log file in append mode."""

    def __init__(self, path=None):
        import atexit
        self._f = open(path, 'a') if path else None
        if self._f:
            atexit.register(self.close)


log_path = f'{odir}/training{_sfx}.log' if log_directly else None
log = _AppendRunLog(log_path)
log('\n' + '=' * 72)
log('RESUME RUN')
log('=' * 72)

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
log(f'config:              {args.config}')
log(f'backend:             {backend}')
log(f'data_source:         {data_source}')
log(f'hdf5_files:          {hdf5_files}')
if data_source == 'local':
    log(f'hdf5_dir:            {hdf5_dir}')
elif data_source == 'gcs_download':
    log(f'gcs_key_file:        {gcs_key_file}')
    log(f'local_data_dir:      {local_data_dir}')
    log(f'overwrite_cache:     {overwrite_cache}')
log(f'lazy:                {lazy}')
log(f'nevents_in:          {nevents_in}')
log(f'start_event:         {start_event}')
log(f'total_epochs:        {total_epochs}')
log(f'batch_size:          {batch_size}')
log(f'hidden_channels:     {hidden_channels}')
log(f'checkpoint_freq:     {checkpoint_freq}')
log(f'min_save_epoch:      {min_save_epoch}')
log(f'n_modes:             {n_modes}')
log(f'train_ratio:         {train_ratio}')
log(f'xyslice:             {xyslice}')
log(f'tauslice:            {tauslice}')
log(f'log_directly:        {log_directly}')
log(f'AdamW lr:            {lr}')
log(f'AdamW weight_decay:  {weight_decay}')
log(f'test_batch_sizes:    {test_batch_sizes}')
if backend == 'torch':
    log(f'num_workers:         {num_workers}')
log(f'eval_interval:       {eval_interval}')
log(f'model_dir:           {model_dir}')
log(f'normalization:       {norm_cfg}')
log(f'resume_model_source: {resume_model_source}')
log(f'resume_model_dir:    {_resume_model_dir_raw}')
log(f'resume_backend:      {resume_backend}  (cross_backend={cross_backend})')
log(f'checkpoint_type:     {checkpoint_type}')
if checkpoint_type == 'periodic':
    log(f'checkpoint_epoch:    {checkpoint_epoch}')
if gcs_output_prefix:
    log(f'gcs_output_prefix:   {gcs_output_prefix}')

# ---------------------------------------------------------------------------
# resolve checkpoint directory (download from GCS if needed)
# ---------------------------------------------------------------------------
if resume_model_source == 'gcs_download':
    if not gcs_key_file:
        raise ValueError(
            "gcs_key_file is required when resume_model_source is 'gcs_download'")
    local_model_cache_path = Path(local_model_cache)
    local_model_cache_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_type == 'periodic':
        # Download only the specific checkpoint_N subdirectory
        gs_ckpt   = f"{str(_resume_model_dir_raw).rstrip('/')}/checkpoint_{checkpoint_epoch}"
        local_sub = local_model_cache_path / f'checkpoint_{checkpoint_epoch}'
        log(f'Downloading periodic checkpoint from GCS: {gs_ckpt}')
        t_dl = perf_counter()
        download_gcs_dir(gs_ckpt, str(local_sub), gcs_key_file,
                         overwrite=overwrite_cache, show_progress=gcs_show_progress)
        ckpt_dir = local_sub
    else:
        # Download the full checkpoint directory (contains last_model.*)
        log(f'Downloading checkpoint directory from GCS: {_resume_model_dir_raw}')
        t_dl = perf_counter()
        download_gcs_dir(str(_resume_model_dir_raw), str(local_model_cache_path),
                         gcs_key_file, overwrite=overwrite_cache,
                         show_progress=gcs_show_progress)
        ckpt_dir = local_model_cache_path

    log(f'Checkpoint downloaded in {perf_counter() - t_dl:.1f}s  →  {ckpt_dir}')

else:  # local
    if checkpoint_type == 'periodic':
        ckpt_dir = Path(_resume_model_dir_raw) / f'checkpoint_{checkpoint_epoch}'
    else:
        ckpt_dir = Path(_resume_model_dir_raw)

if not ckpt_dir.exists():
    raise FileNotFoundError(
        f"Checkpoint directory not found: {ckpt_dir.resolve()}\n"
        f"  resume_model_dir={_resume_model_dir_raw}, "
        f"checkpoint_type={checkpoint_type}"
        + (f", checkpoint_epoch={checkpoint_epoch}" if checkpoint_type == 'periodic' else ""))

log(f'Checkpoint dir: {ckpt_dir.resolve()}')

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
# train / val split + normalisation + datasets
#
# Load normalizer from the checkpoint directory so the same transform is
# applied as in the original run.  Falls back to building fresh (with a
# warning) if no normalizer.npz is found.
# ---------------------------------------------------------------------------
n_total = len(arr)
n_train = int(n_total * train_ratio)
log(f'total events: {n_total}  →  train={n_train}  val={n_total - n_train}')

# Search order: ckpt_dir → model_dir → build fresh
_norm_candidates = [ckpt_dir / 'normalizer.npz', Path(model_dir) / 'normalizer.npz']
_norm_path = next((p for p in _norm_candidates if p.exists()), None)

if _norm_path is not None:
    log(f'Loading normalizer from checkpoint: {_norm_path}')
    normalizer = Normalizer.load(str(_norm_path))
else:
    log('[WARN] normalizer.npz not found in checkpoint or model_dir — building from training data.')
    normalizer = build_normalizer(norm_cfg, grid_params)

log(f'normalizer: {normalizer}')

DatasetCls = FluidDatasetH5 if backend == 'torch' else FluidDatasetH5MLX

if lazy:
    train_indices = np.arange(n_train)
    val_indices   = np.arange(n_train, n_total)

    # Data-driven normalizers: only re-fit if the ref was not loaded from disk.
    _needs_fit = normalizer.name in ('etau75_eta_3d', 'etau75_eta_shape_3d')
    if _needs_fit and normalizer.ref is None:
        log('[norm] data-driven normalizer: materialising training split for ref ...')
        normalizer.fit_transform(arr[0:n_train])
        gc.collect()
        log(f'[norm] ref computed  shape={normalizer.ref.shape}')

    dataset_train = DatasetCls(arr, indices=train_indices,
                               xyslice=xyslice, tauslice=tauslice,
                               normalizer=normalizer)
    dataset_val   = DatasetCls(arr, indices=val_indices,
                               xyslice=xyslice, tauslice=tauslice,
                               normalizer=normalizer)

else:
    data_train  = arr[0:n_train]
    data_verify = arr[n_train:n_total]
    del arr
    gc.collect()

    _needs_fit = normalizer.name in ('etau75_eta_3d', 'etau75_eta_shape_3d')
    if _needs_fit and normalizer.ref is None:
        data_train = normalizer.fit_transform(data_train)
    else:
        data_train = normalizer.transform(data_train)
    data_verify = normalizer.transform(data_verify)

    dataset_train = DatasetCls(data_train, xyslice=xyslice, tauslice=tauslice)
    dataset_val   = DatasetCls(data_verify, xyslice=xyslice, tauslice=tauslice)

# Always (re-)save the normalizer to model_dir so it stays in sync with the
# resumed model weights.
normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

_sample_x = dataset_train[0]['x']
test_resolutions = [_sample_x.shape[1]]   # nx after xyslice

log(f'shapes train: {dataset_train.n_samples} events  event_shape={dataset_train[0]["x"].shape}')
log(f'shapes val:   {dataset_val.n_samples} events')

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
# data loaders
# ---------------------------------------------------------------------------
_nw      = num_workers if backend == 'torch' else 0
_pin     = backend == 'torch'
_persist = _nw > 0
_prefetch = 2 if _nw > 0 else None

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                          num_workers=_nw,
                          persistent_workers=_persist,
                          prefetch_factor=_prefetch,
                          pin_memory=_pin)

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    log(f'test resolution: {res}')
    test_loaders[res] = DataLoader(dataset_val,
                                   batch_size=test_bsize,
                                   shuffle=False,
                                   num_workers=_nw,
                                   persistent_workers=_persist,
                                   prefetch_factor=_prefetch,
                                   pin_memory=_pin)
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

trainer = Trainer(model=model, n_epochs=total_epochs,
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
# load checkpoint and determine resume epoch
# ---------------------------------------------------------------------------
log(f'\n--- Loading checkpoint ---')
log(f'  ckpt_dir        : {ckpt_dir}')
log(f'  checkpoint_type : {checkpoint_type}')
log(f'  resume_backend  : {resume_backend}')
log(f'  cross_backend   : {cross_backend}')

if cross_backend:
    # ---- torch → mlx weight transfer ----------------------------------------
    if resume_backend == 'torch' and backend == 'mlx':
        from neuralop_mlx.weight_transfer import load_from_torch
        torch_ckpt = ckpt_dir / 'last_model.pt'
        if not torch_ckpt.exists():
            raise FileNotFoundError(
                f"Cross-backend transfer expects last_model.pt at: {torch_ckpt}\n"
                "Make sure the original torch run used save_last: true.")
        log(f'Transferring PyTorch weights → MLX model: {torch_ckpt}')
        load_from_torch(str(torch_ckpt), model)
        # Read epoch from the torch checkpoint
        import torch as _torch
        _pt_state = _torch.load(str(torch_ckpt), map_location='cpu', weights_only=False)
        resume_epoch = int(_pt_state.get('epoch', 0)) + 1
        log(f'[cross-backend] Epoch {resume_epoch - 1} restored. '
            'Optimizer state NOT transferred — starts fresh from configured lr.')
    else:
        raise NotImplementedError(
            f"Cross-backend transfer {resume_backend}→{backend} is not supported.\n"
            "Only torch→mlx is implemented.  "
            "To go mlx→torch, re-train with backend: torch.")

elif checkpoint_type == 'last':
    # ---- standard last checkpoint (same backend) ----------------------------
    resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)

elif checkpoint_type == 'periodic':
    # ---- periodic checkpoint (same backend) ---------------------------------
    if backend == 'mlx':
        import mlx.core as mx
        import mlx.utils as mlx_utils

        _model_f = ckpt_dir / 'best_model.safetensors'
        if not _model_f.exists():
            raise FileNotFoundError(
                f"MLX periodic checkpoint model not found: {_model_f}")
        params = mx.load(str(_model_f))
        nested = mlx_utils.tree_unflatten(list(params.items()))
        trainer.model.update(nested)
        mx.eval(trainer.model.parameters())
        log(f'MLX model weights loaded from {_model_f}')

        _opt_f = ckpt_dir / 'optimizer.safetensors'
        if _opt_f.exists():
            opt_params = mx.load(str(_opt_f))
            opt_nested = mlx_utils.tree_unflatten(list(opt_params.items()))
            optimizer.state.update(opt_nested)
            log(f'MLX optimizer state loaded from {_opt_f}')
        else:
            log('[WARN] optimizer.safetensors not found in periodic checkpoint — '
                'optimizer restarts from configured lr.')

        _state_f = ckpt_dir / 'state.json'
        if _state_f.exists():
            with open(_state_f) as _fs:
                _meta = json.load(_fs)
            resume_epoch = int(_meta['epoch']) + 1
        else:
            resume_epoch = checkpoint_epoch + 1

    else:
        # PyTorch periodic: neuralop checkpoint() saves best_model.safetensors
        # with model weights only (no optimizer state).
        _last_pt = ckpt_dir / 'last_model.pt'
        _sf_path = ckpt_dir / 'best_model.safetensors'

        if _last_pt.exists():
            # Checkpoint subdir happened to contain a full last_model.pt
            resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)
        elif _sf_path.exists():
            from safetensors.torch import load_file as _sf_load
            state_dict = _sf_load(str(_sf_path))
            model.load_state_dict(state_dict)
            log(f'Torch model weights loaded from {_sf_path}')
            log('[WARN] Periodic torch checkpoint has no optimizer state — '
                'optimizer restarts from configured lr.')
            _state_f = ckpt_dir / 'state.json'
            if _state_f.exists():
                with open(_state_f) as _fs:
                    _meta = json.load(_fs)
                resume_epoch = int(_meta.get('epoch', checkpoint_epoch)) + 1
            else:
                resume_epoch = checkpoint_epoch + 1
        else:
            raise FileNotFoundError(
                f"No recognized model file in periodic checkpoint dir: {ckpt_dir}\n"
                "Expected one of: best_model.safetensors, last_model.pt")
else:
    raise ValueError(f"Unknown checkpoint_type: {checkpoint_type!r}")

remaining_epochs = total_epochs - resume_epoch
if remaining_epochs <= 0:
    log(f'[INFO] Already at or past epoch {resume_epoch - 1} '
        f'(total_epochs={total_epochs}).  Increase total_epochs to continue.')
    sys.exit(0)

trainer.n_epochs = remaining_epochs
log(f'Resuming from epoch {resume_epoch}  →  training {remaining_epochs} more epochs '
    f'(target total: {total_epochs})')

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
    save_last_checkpoint(trainer, optimizer, scheduler, total_epochs, model_dir,
                         config_path=args.config)

if backend == 'torch':
    log_peak_memory(log, cuda=True)
else:
    log_peak_memory(log, mlx=True)

_title_parts = ['FNO Resume (HDF5)']
if backend == 'mlx':
    _title_parts.append('MLX')
if lazy:
    _title_parts.append('lazy')
if cross_backend:
    _title_parts.append(f'{resume_backend}→{backend}')
_title = _title_parts[0] + (f' ({", ".join(_title_parts[1:])})' if len(_title_parts) > 1 else '')

history.save(f'{odir}/metrics{_sfx}_resume.json')
history.plot(save_path=f'{odir}/training_curves{_sfx}_resume.svg', title=_title)
log(f'metrics   → {odir}/metrics{_sfx}_resume.json')
log(f'curves    → {odir}/training_curves{_sfx}_resume.svg')
log(f'total time: {perf_counter() - t0:.1f}s')

# ---------------------------------------------------------------------------
# upload outputs to GCS (optional)
# ---------------------------------------------------------------------------
if gcs_output_prefix:
    log(f'Uploading outputs to GCS: {gcs_output_prefix}')
    if log._f:
        log._f.flush()
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file,
                   show_progress=gcs_show_progress)
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file,
                   show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
