#!/usr/bin/env python

# ---------------------------------------------------------------------------
# resume_train_h5_torch_sel.py — Resume training from a checkpoint.
# Pure PyTorch, HDF5 input, model-type selector (fno | causal_fno4d |
# causal_fno4d_tau_eta).  Torch-only resume counterpart of
# train_torch_h5_sel.py, with the τ/η reconstruction path.
#
# Usage:
#   python resume_train_h5_torch_sel.py [--config config_h5_resume_sel.yaml]
#
# It mirrors train_torch_h5_sel.py's data pipeline and model selector, but
# instead of a random init it:
#   1. Resolves a checkpoint directory (local or GCS download).
#   2. Rebuilds the model for the configured model_type.
#   3. Restores weights (+ optimizer/scheduler for checkpoint_type: last).
#   4. Trains the remaining epochs up to total_epochs.
#
# τ/η reconstruction (model_type: causal_fno4d_tau_eta)
# -----------------------------------------------------
# The CausalFNO4DTauEta forward() needs physical τ/η coordinate vectors. These
# are rebuilt from the persisted embedding_grid.json sidecar (written by the
# original training run via save_embedding_grid), searched in the checkpoint
# dir then model_dir.  If the sidecar is missing, the grid is taken from the
# current dataset's grid params with a warning.  The model is then wrapped in
# the same _TauEtaTrainingWrapper used at training time so the saved
# state_dict (which includes the pinned tau/eta buffers) loads cleanly.
#
# Resume-specific YAML keys (others match config_h5_sel.yaml):
#   total_epochs        : final epoch target (remaining = total − resume_epoch)
#   resume_model_source : local | gcs_download
#   resume_model_dir    : dir (or gs:// URI) with the checkpoint; default model_dir
#   local_model_cache   : [gcs_download] where to cache the download
#   checkpoint_type     : last | periodic
#   checkpoint_epoch    : [periodic only] epoch N of checkpoint_N/
# ---------------------------------------------------------------------------

import argparse
import json
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
parser = argparse.ArgumentParser(
    description='Resume FNO training (torch, model-type selector) from HDF5 checkpoint')
parser.add_argument('--config', default='config_h5_resume_sel.yaml', metavar='FILE',
                    help='YAML configuration file (default: config_h5_resume_sel.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

data_source = cfg.get('data_source', 'local')   # local | gcs_download
if data_source not in ('local', 'gcs_download'):
    raise ValueError(f"data_source must be 'local' or 'gcs_download', got: {data_source!r}")

# --- resume-specific config ------------------------------------------------
resume_model_source = cfg.get('resume_model_source', 'local')
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
                             save_last_checkpoint, load_last_checkpoint,
                             save_embedding_grid, load_embedding_grid)
from parse_training_log import TrainingHistory
from norm_utils import build_normalizer, Normalizer
from fluid_datasets import FluidDatasetH5, worker_init_fn

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
etaslice          = _slice(cfg.get('etaslice', [None, None]))
tauslice          = _slice(cfg.get('tauslice', [None, -1]))
lr                = cfg['lr']
weight_decay      = cfg.get('weight_decay', 1e-4)
test_batch_sizes  = cfg.get('test_batch_sizes', [10])
num_workers       = cfg.get('num_workers', 0)
eval_interval     = cfg.get('eval_interval', 2)
save_last         = cfg.get('save_last', True)
norm_cfg          = cfg.get('normalization', {'name': 'none'})
norm_chunk_size   = cfg.get('norm_chunk_size', 64)
model_type        = cfg.get('model_type', 'fno')   # fno | causal_fno4d | causal_fno4d_tau_eta
causal_cfg        = cfg.get('causal_fno4d', {})

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
log(f'backend:             torch')
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
log(f'etaslice:            {etaslice}')
log(f'tauslice:            {tauslice}')
log(f'log_directly:        {log_directly}')
log(f'AdamW lr:            {lr}')
log(f'AdamW weight_decay:  {weight_decay}')
log(f'test_batch_sizes:    {test_batch_sizes}')
log(f'num_workers:         {num_workers}')
log(f'eval_interval:       {eval_interval}')
log(f'model_dir:           {model_dir}')
log(f'normalization:       {norm_cfg}')
log(f'model_type:          {model_type}')
if model_type != 'fno':
    log(f'causal_fno4d cfg:    {causal_cfg}')
log(f'resume_model_source: {resume_model_source}')
log(f'resume_model_dir:    {_resume_model_dir_raw}')
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
        raise ValueError("gcs_key_file is required when resume_model_source is 'gcs_download'")
    local_model_cache_path = Path(local_model_cache)
    local_model_cache_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_type == 'periodic':
        gs_ckpt   = f"{str(_resume_model_dir_raw).rstrip('/')}/checkpoint_{checkpoint_epoch}"
        local_sub = local_model_cache_path / f'checkpoint_{checkpoint_epoch}'
        log(f'Downloading periodic checkpoint from GCS: {gs_ckpt}')
        t_dl = perf_counter()
        download_gcs_dir(gs_ckpt, str(local_sub), gcs_key_file,
                         overwrite=overwrite_cache, show_progress=gcs_show_progress)
        ckpt_dir = local_sub
    else:
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
        f"  resume_model_dir={_resume_model_dir_raw}, checkpoint_type={checkpoint_type}"
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
# Load the normalizer from the checkpoint dir so the same transform is applied
# as in the original run.  Falls back to building fresh (with a warning).
# ---------------------------------------------------------------------------
n_total = len(arr)
n_train = int(n_total * train_ratio)
log(f'total events: {n_total}  →  train={n_train}  val={n_total - n_train}')

_norm_candidates = [ckpt_dir / 'normalizer.npz', Path(model_dir) / 'normalizer.npz']
_norm_path = next((p for p in _norm_candidates if p.exists()), None)

if _norm_path is not None:
    log(f'Loading normalizer from checkpoint: {_norm_path}')
    normalizer = Normalizer.load(str(_norm_path))
else:
    log('[WARN] normalizer.npz not found in checkpoint or model_dir — building from training data.')
    normalizer = build_normalizer(norm_cfg, grid_params)

log(f'normalizer: {normalizer}')

if lazy:
    train_indices = np.arange(n_train)
    val_indices   = np.arange(n_train, n_total)

    _needs_fit = normalizer.name in ('etau75_eta_3d', 'etau75_eta_shape_3d')
    if _needs_fit and normalizer.ref is None:
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

    _needs_fit = normalizer.name in ('etau75_eta_3d', 'etau75_eta_shape_3d')
    if _needs_fit and normalizer.ref is None:
        data_train = normalizer.fit_transform(data_train)
    else:
        data_train = normalizer.transform(data_train)
    data_verify = normalizer.transform(data_verify)

    dataset_train = FluidDatasetH5(data_train)
    dataset_val   = FluidDatasetH5(data_verify)

# Always (re-)save the normalizer to model_dir so it stays in sync with the
# resumed model weights.
normalizer_path = f'{model_dir}/normalizer.npz'
normalizer.save(normalizer_path)
log(f'normalizer  →  {normalizer_path}')

_sample_x = dataset_train[0]['x']
test_resolutions = [_sample_x.shape[1]]   # nx after xyslice

log(f'shapes train: {dataset_train.n_samples} events  event_shape={_sample_x.shape}')
log(f'shapes val:   {dataset_val.n_samples} events')

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
# model  (reconstructed to match the saved checkpoint structure)
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
    # Grid dimensions from the dataset sample shape (C, X, Y, Z, T).
    n_eta = _sample_x.shape[-2]   # Z dimension after slicing
    n_tau = _sample_x.shape[-1]   # T dimension after slicing

    # Prefer the persisted embedding grid (written by the original run);
    # search the checkpoint dir first, then model_dir.
    _grid = load_embedding_grid(ckpt_dir) or load_embedding_grid(model_dir)
    if _grid is not None:
        log(f'Loaded embedding grid from sidecar: {_grid}')
        eg_tau_min, eg_dtau = _grid['tau_min'], _grid['dtau']
        eg_eta_min, eg_deta = _grid['eta_min'], _grid['deta']
        if (_grid.get('n_tau') not in (None, n_tau)
                or _grid.get('n_eta') not in (None, n_eta)):
            log(f"[WARN] embedding_grid n_tau/n_eta ({_grid.get('n_tau')}/{_grid.get('n_eta')}) "
                f"differ from current data ({n_tau}/{n_eta}); using current data dims.")
    else:
        log('[WARN] embedding_grid.json not found in checkpoint or model_dir — '
            'using grid params from the current dataset.')
        eg_tau_min, eg_dtau = grid_params['tau_min'], grid_params['dtau']
        eg_eta_min, eg_deta = grid_params['eta_min'], grid_params['deta']

    tau_vec = make_tau_tensor(eg_tau_min, eg_dtau, n_tau, device=device)
    eta_vec = make_eta_tensor(eg_eta_min, eg_deta, n_eta, device=device)
    _base_model = CausalFNO4DTauEta(
        in_channels=4,
        out_channels=4,
        hidden_channels=hidden_channels,
        n_modes_spatial=tuple(n_modes[:3]),
        tau_min=float(eg_tau_min), dtau=float(eg_dtau),
        eta_min=float(eg_eta_min), deta=float(eg_deta),
        **causal_cfg,
    ).to(device)
    log(f'CausalFNO4DTauEta temporal_receptive_field: {_base_model.temporal_receptive_field}')
    # Re-save the sidecar to model_dir so it stays in sync with resumed weights.
    save_embedding_grid(
        model_dir, 'CausalFNO4DTauEta',
        eg_tau_min, eg_dtau, n_tau, _base_model.tau_max,
        eg_eta_min, eg_deta, n_eta, _base_model.eta_max,
    )
    log(f'embedding grid  →  {model_dir}/embedding_grid.json')

    # Same wrapper as training: the neuralop Trainer calls model(x), and the
    # pinned tau/eta buffers are part of the saved/restored state_dict.
    class _TauEtaTrainingWrapper(nn.Module):
        def __init__(self, m, tau, eta):
            super().__init__()
            self.m = m
            self.register_buffer('tau', tau)
            self.register_buffer('eta', eta)
        def forward(self, x):
            return self.m(x, self.tau, self.eta)
        def load_state_dict(self, state_dict, strict=True, assign=False):
            # The inner BaseModel injects a top-level '_metadata' key into its
            # state_dict; strip it here since this wrapper is a plain nn.Module
            # and would otherwise reject it as an unexpected key.
            state_dict = {k: v for k, v in state_dict.items() if k != '_metadata'}
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
    model = _TauEtaTrainingWrapper(_base_model, tau_vec, eta_vec)

else:
    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Must be 'fno', 'causal_fno4d', or 'causal_fno4d_tau_eta'.")

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
log(f'\n--- Loading checkpoint ---')
log(f'  ckpt_dir        : {ckpt_dir}')
log(f'  checkpoint_type : {checkpoint_type}')

if checkpoint_type == 'last':
    resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)

else:  # periodic — neuralop checkpoint() saves model weights only (no optimizer)
    _last_pt = ckpt_dir / 'last_model.pt'
    _sd_pt   = ckpt_dir / 'best_model_state_dict.pt'
    _sf_path = ckpt_dir / 'best_model.safetensors'

    if _last_pt.exists():
        resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, ckpt_dir)
    elif _sd_pt.exists():
        state_dict = torch.load(str(_sd_pt), map_location=device, weights_only=False)
        state_dict.pop('_metadata', None)
        model.load_state_dict(state_dict)
        log(f'Torch model weights loaded from {_sd_pt}')
        log('[WARN] Periodic torch checkpoint has no optimizer state — '
            'optimizer restarts from configured lr.')
        resume_epoch = checkpoint_epoch + 1
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
            "Expected one of: last_model.pt, best_model_state_dict.pt, best_model.safetensors")

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

_title_parts = [f'FNO Resume ({model_type}, HDF5)']
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
    upload_gcs_dir(odir, f'{gcs_output_prefix}/logs', gcs_key_file,
                   show_progress=gcs_show_progress)
    upload_gcs_dir(model_dir, f'{gcs_output_prefix}/checkpoints', gcs_key_file,
                   show_progress=gcs_show_progress)
    log(f'GCS upload complete → {gcs_output_prefix}')
