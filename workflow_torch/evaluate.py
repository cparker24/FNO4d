#!/usr/bin/env python

# ---------------------------------------------------------------------------
# workflow_torch/evaluate.py — standalone PyTorch FNO evaluation (CLI, no notebook).
#
# Usage:
#   python evaluate.py [--config config_eval.yaml]
#
# NB: named evaluate.py (not eval.py) on purpose — a script called eval.py
# cannot import loc_libs/eval.py: Python puts the running script's own directory
# first on sys.path, so `from eval import ...` would re-import the script itself.
#
# Reuses the training config for the model architecture + data slicing, loads a
# checkpoint, runs batched inference through the SHARED pipeline
# (loc_libs/eval.run_inference → FluidDataset + collate, bit-identical to the
# train-time windowing/normalizer), and scores it with model_eval.ModelEval.
#
# Reused building blocks:
#   loc_libs/eval.py   run_inference, compute_grid_coords, load_torch_model
#   loc_libs/data      build_source (HDF5 or ROOT)
#   _model.py          build_model (same selector as train.py/resume.py)
#   model_eval.py      ModelEval (metric suite + dashboard)
#
# Eval config keys (see config_eval.yaml):
#   train_config     : path to the run's training config.yaml (architecture,
#                      slicing, normalization, downsample).
#   model_dir        : checkpoint directory (default: train_config's model_dir/odir).
#   checkpoint_type  : best | last | checkpoint_N
#   checkpoint_epoch : N                         [checkpoint_N only]
#   data_source      : local | gcs_download
#   eval_input_files : optional override file list (else train_config's inputs)
#   eval_input_dir   : directory for eval_input_files (local)
#   use_train_split  : true → score the held-out val split of the training files
#   nevents          : cap on events read (-1 = all)
#   active_threshold : ε mask for normalized metrics (0 = off)
#   eval_batch_size  : inference batch size
#   metric_backend   : numpy | torch
#   odir             : output dir for metrics_eval.json + figures
#   save_figures     : true → write dashboard + nRMSE heatmap PNGs
# ---------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# parse args + configs
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Evaluate a trained FNO (torch) checkpoint')
parser.add_argument('--config', default='config_eval.yaml', metavar='FILE',
                    help='YAML eval configuration file (default: config_eval.yaml)')
args = parser.parse_args()

with open(args.config) as f:
    eval_cfg = yaml.safe_load(f)

_train_cfg_path = Path(eval_cfg['train_config'])
if not _train_cfg_path.exists():
    raise FileNotFoundError(f"train_config not found: {_train_cfg_path.resolve()}")
with open(_train_cfg_path) as f:
    train_cfg = yaml.safe_load(f)

data_source = eval_cfg.get('data_source', 'local')
if data_source not in ('local', 'gcs_download'):
    raise ValueError(f"data_source must be 'local' or 'gcs_download', got: {data_source!r}")

# ---------------------------------------------------------------------------
# torch + loc_libs imports
# ---------------------------------------------------------------------------
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='neuralop')

sys.path.insert(0, str(Path(__file__).parent.parent / 'loc_libs'))
sys.path.insert(0, str(Path(__file__).parent))      # for the local _model helper
from norm_utils import Normalizer
from downsample import parse_downsample_cfg, downsampled_grid_params, DownsampledArrayView
from data import build_source
from eval import run_inference, compute_grid_coords, load_torch_model
from model_eval import ModelEval
from _model import build_model

gcs_key_file      = eval_cfg.get('gcs_key_file', None)
gcs_show_progress = eval_cfg.get('gcs_show_progress', None)
if data_source == 'gcs_download':
    from gcs_utils import download_gcs_files, check_gcs_dependencies
    check_gcs_dependencies(require_storage=True)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def _slice(val):
    return slice(None) if val is None else slice(*val)

# architecture + data prep come from the TRAINING config
hidden_channels = train_cfg['hidden_channels']
n_modes         = train_cfg['n_modes']
model_type      = train_cfg.get('model_type', 'fno')
causal_cfg      = train_cfg.get('causal_fno4d', {})
train_ratio     = train_cfg.get('train_ratio', 0.8)
xyslice         = _slice(train_cfg.get('xyslice',  [None, -1]))
etaslice        = _slice(train_cfg.get('etaslice', [None, None]))
tauslice        = _slice(train_cfg.get('tauslice', [None, -1]))
downsample_cfg  = train_cfg.get('downsample')

# eval-specific
model_dir        = Path(eval_cfg.get('model_dir',
                                     train_cfg.get('model_dir', train_cfg.get('odir', './checkpoints'))))
checkpoint_type  = eval_cfg.get('checkpoint_type', 'best')
checkpoint_epoch = int(eval_cfg.get('checkpoint_epoch', 10))
use_train_split  = eval_cfg.get('use_train_split', True)
nevents          = int(eval_cfg.get('nevents', -1))
active_threshold = eval_cfg.get('active_threshold', 0.0)
active_threshold = float(active_threshold) if active_threshold and active_threshold > 0 else None
eval_batch_size  = int(eval_cfg.get('eval_batch_size', 8))
metric_backend   = eval_cfg.get('metric_backend', 'numpy')
odir             = Path(eval_cfg.get('odir', './out_eval'))
save_figures     = eval_cfg.get('save_figures', True)
local_data_dir   = eval_cfg.get('local_data_dir', './cache/eval_data')
overwrite_cache  = eval_cfg.get('overwrite_cache', False)

# data source: separate eval files, or the training files (+ held-out split)
if eval_cfg.get('eval_input_files'):
    input_files = eval_cfg['eval_input_files']
    input_dir   = eval_cfg.get('eval_input_dir', '.')
    using_eval_files = True
else:
    input_files = train_cfg['input_files']
    input_dir   = train_cfg.get('input_dir', '.')
    using_eval_files = False

odir.mkdir(parents=True, exist_ok=True)
t0 = perf_counter()

print(f'eval_config:     {args.config}')
print(f'train_config:    {_train_cfg_path}')
print(f'model_dir:       {model_dir}')
print(f'checkpoint_type: {checkpoint_type}'
      + (f' (epoch {checkpoint_epoch})' if checkpoint_type == 'checkpoint_N' else ''))
print(f'model_type:      {model_type}')
print(f'input type:      {"override eval files" if using_eval_files else "training files"}')
print(f'use_train_split: {use_train_split}')

# ---------------------------------------------------------------------------
# resolve input paths
# ---------------------------------------------------------------------------
_is_h5 = all(str(f).lower().endswith(('.h5', '.hdf5')) for f in input_files)
if data_source == 'local':
    source_paths = [str(Path(input_dir) / f) for f in input_files]
    for p in source_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Input file not found: {p}")
else:  # gcs_download
    print(f'Downloading {len(input_files)} file(s) from GCS ...')
    source_paths = download_gcs_files(input_files, local_data_dir, gcs_key_file,
                                      overwrite=overwrite_cache, show_progress=gcs_show_progress)

# ---------------------------------------------------------------------------
# build source, apply downsample, materialise the test split
# ---------------------------------------------------------------------------
source, grid_params = build_source(source_paths, lazy=True,
                                   nevents=nevents, verbose=True)
_dkw = parse_downsample_cfg(downsample_cfg)
if _dkw:
    source      = DownsampledArrayView(source, **_dkw)
    grid_params = downsampled_grid_params(grid_params, **_dkw)
    print(f'downsample:      {_dkw}')

n_total = len(source) if nevents is None or nevents <= 0 else min(nevents, len(source))
n_train = int(n_total * train_ratio)

if use_train_split and not using_eval_files:
    lo, hi = n_train, n_total
    if hi <= lo:
        raise ValueError(f"Empty val split (n_train={n_train}, n_total={n_total}); "
                         "set use_train_split: false or lower train_ratio.")
    print(f'Scoring held-out val split: events [{lo}, {hi})  ({hi - lo} events)')
else:
    lo, hi = 0, n_total
    print(f'Scoring all events: [{lo}, {hi})  ({hi - lo} events)')

# Materialise the requested events (lazy slice -> dense ndarray) and apply the
# same xy/eta/tau trims training used, so the model sees an identical grid.
data_test = source[lo:hi][:, :, xyslice, xyslice, etaslice, tauslice]
data_test = np.ascontiguousarray(data_test)
print(f'data_test shape: {data_test.shape}')

# ---------------------------------------------------------------------------
# normalizer (loaded — must match the trained model)
# ---------------------------------------------------------------------------
norm_path = model_dir / 'normalizer.npz'
if norm_path.exists():
    normalizer = Normalizer.load(str(norm_path))
    print(f'normalizer:      {normalizer}')
else:
    normalizer = None
    print('normalizer:      none found — evaluating on raw data')

# ---------------------------------------------------------------------------
# device + model
# ---------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device:          {device}')

# (B, C, nx, ny, neta, T-1) — the model input window after the first-tau repeat.
sample_x_shape = (1,) + tuple(data_test.shape[1:-1]) + (data_test.shape[-1] - 1,)
model = build_model(
    model_type, n_modes=n_modes, hidden_channels=hidden_channels, device=device,
    sample_x_shape=sample_x_shape, grid_params=grid_params,
    causal_cfg=causal_cfg, save_embedding=False, log=print)

# resolve checkpoint file
if checkpoint_type == 'best':
    ckpt_path = model_dir / 'best_model_state_dict.pt'
elif checkpoint_type == 'last':
    ckpt_path = model_dir / 'last_model.pt'
elif checkpoint_type == 'checkpoint_N':
    ckpt_path = model_dir / f'checkpoint_{checkpoint_epoch}' / 'best_model_state_dict.pt'
else:
    raise ValueError(f"checkpoint_type must be 'best', 'last', or 'checkpoint_N', "
                     f"got: {checkpoint_type!r}")
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
load_torch_model(model, ckpt_path, device)
n_params = sum(p.numel() for p in model.parameters())
print(f'model:           {n_params:,} params loaded from {ckpt_path.name}')

# ---------------------------------------------------------------------------
# batched inference (shared pipeline) + physical-unit reconstruction
# ---------------------------------------------------------------------------
print('Running batched inference ...')
model_out_phys, y_true_phys = run_inference(
    model, 'torch', device, data_test, normalizer, batch_size=eval_batch_size)
print(f'pred:  {model_out_phys.shape}\ntrue:  {y_true_phys.shape}')

coords = compute_grid_coords(data_test, grid_params)
eta_coords     = coords['eta_coords']
tau_out_coords = coords['tau_out_coords']

# ---------------------------------------------------------------------------
# metrics (model_eval.ModelEval)
# ---------------------------------------------------------------------------
evaluator = ModelEval(
    pred=model_out_phys, true=y_true_phys,
    tau_coords=tau_out_coords, eta_coords=eta_coords,
    channel_names=['ε', 'vx', 'vy', 'vη'],
    active_threshold=active_threshold,
    backend=metric_backend,
)
summary = evaluator.summary()

print('\n=== Metric summary (mean over C, η, τ) ===')
for k, v in summary.items():
    print(f'  {k:<12} {v:.6g}')

metrics_path = odir / 'metrics_eval.json'
with open(metrics_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nsummary → {metrics_path}')

# ---------------------------------------------------------------------------
# figures (optional)
# ---------------------------------------------------------------------------
if save_figures:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _ieta     = int(np.argmin(np.abs(eta_coords - 0.0)))      # η closest to 0
    _itau_3fm = int(np.argmin(np.abs(tau_out_coords - 3.0)))  # τ closest to 3 fm/c

    fig_dash = evaluator.plot_dashboard(channel=0, ieta=_ieta, itau=_itau_3fm)
    _dash_path = odir / 'dashboard.png'
    fig_dash.savefig(_dash_path, dpi=120, bbox_inches='tight')
    plt.close(fig_dash)
    print(f'dashboard → {_dash_path}')

    nrmse = evaluator.nrmse()
    fig2, axes2 = plt.subplots(1, evaluator.C, figsize=(5 * evaluator.C, 3.5))
    if evaluator.C == 1:
        axes2 = [axes2]
    for c, ax in enumerate(axes2):
        im = ax.pcolormesh(tau_out_coords, eta_coords, nrmse[c], cmap='hot_r', shading='auto')
        plt.colorbar(im, ax=ax, label='nRMSE')
        ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('η')
        ax.set_title(f'nRMSE — {evaluator.channel_names[c]}')
    fig2.suptitle('nRMSE heat map (η × τ)', fontsize=12)
    fig2.tight_layout()
    _nrmse_path = odir / 'nrmse_heatmap.png'
    fig2.savefig(_nrmse_path, dpi=120, bbox_inches='tight')
    plt.close(fig2)
    print(f'nRMSE map → {_nrmse_path}')

print(f'\ntotal time: {perf_counter() - t0:.1f}s')
