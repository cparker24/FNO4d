"""
FNO Model Evaluation — Panel Web App

Run from the eval_model_web/ directory:

    panel serve app.py --show
    panel serve app.py --show --port 5006

Override the config file via environment variable:

    EVAL_GCS_CONFIG=/path/to/eval_gcs_config.yaml panel serve app.py --show

The default config path is ../eval_model/eval_gcs_config.yaml (relative to
this file), matching the marimo notebook's search order.

All GCS operations (gcs_download, gcs_stream) work identically to the
marimo version — Panel runs server-side Python with no browser sandbox.

Caching strategy
----------------
  _run_pipeline() is decorated with @pn.cache(max_items=2).  It is keyed
  by (cfg_path, cfg_mtime), so editing the config file on disk invalidates
  the cache automatically.  Data loading, normalisation, model loading, and
  inference run only once per unique (config, mtime) pair; widget changes
  only trigger the fast visualisation layer.
"""

import gc
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import param as _param

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import yaml

pn.extension('terminal', sizing_mode='stretch_width', notifications=True)

# ---------------------------------------------------------------------------
# Path setup — loc_libs is one level above this file
# ---------------------------------------------------------------------------
_HERE     = Path(__file__).resolve().parent
_REPO     = _HERE.parent
_LOC_LIBS = _REPO / 'loc_libs'

if not _LOC_LIBS.is_dir():
    raise RuntimeError(
        f'loc_libs not found at {_LOC_LIBS}.\n'
        'Make sure you are running from the eval_model_web/ directory and '
        'that the repo layout is intact.'
    )

# Add loc_libs/ directly so bare imports work:  from gcs_utils import ...
# Add the repo root as well so package imports work: from loc_libs import ...
for _p in (str(_LOC_LIBS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# loc_libs imports — all hoisted to module level so the sys.path insert above
# is guaranteed to have run before any import attempt, and failures surface
# immediately at startup rather than inside a background thread.
from skinnycontour import plot_three_bins_contour  # noqa: E402
from model_eval import ModelEval                   # noqa: E402
import parse_training_log                          # noqa: E402
from gcs_utils import (                            # noqa: E402
    download_gcs_file as _dl_file,
    download_gcs_dir  as _dl_dir,
)
from norm_utils import Normalizer                  # noqa: E402
from read_3d_hdf5 import read_3d_data_hdf5 as _read_h5   # noqa: E402
try:
    from read_3d_root import read_3d_data as _read_root   # noqa: E402
except ImportError as _e:
    _read_root = None
    print(f'[app] read_3d_root unavailable ({_e}) — ROOT files not supported')

# ---------------------------------------------------------------------------
# Log terminal — shown in the browser, captures all print() output
# ---------------------------------------------------------------------------
log_terminal = pn.widgets.Terminal(
    height=260,
    sizing_mode='stretch_width',
    options={'cursorBlink': False, 'disableStdin': True, 'convertEol': True},
    margin=(0, 0, 0, 0),
)

_orig_stdout = sys.__stdout__

class _TeeWriter:
    """Forward every write to both the server console and the browser terminal."""
    def write(self, s: str):
        _orig_stdout.write(s)
        _orig_stdout.flush()
        try:
            log_terminal.write(s)
        except Exception:
            pass
    def flush(self):
        _orig_stdout.flush()
    def __getattr__(self, name):
        return getattr(_orig_stdout, name)

sys.stdout = _TeeWriter()

# ---------------------------------------------------------------------------
# Default config resolution  (matches marimo notebook search order)
# ---------------------------------------------------------------------------
_env_cfg = os.environ.get('EVAL_GCS_CONFIG', '')
_DEFAULT_CFG = str(
    Path(_env_cfg) if _env_cfg
    else _HERE.parent / 'eval_model' / 'eval_gcs_config.yaml'
)


# ---------------------------------------------------------------------------
# Cached pipeline — all heavy work in one function
# ---------------------------------------------------------------------------

@pn.cache(max_items=2)
def _run_pipeline(cfg_path: str, cfg_mtime: float) -> dict:
    """Load config → GCS → data → model → inference.

    Keyed by (cfg_path, cfg_mtime) so the cache is invalidated automatically
    whenever the config file is edited on disk.  cfg_mtime is passed by the
    caller; it is not used inside the function but drives cache keying.
    """
    print(f'[Pipeline] Starting  cfg={cfg_path}')

    # 1. Eval config
    cfg_p = Path(cfg_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f'Config not found: {cfg_p.resolve()}')
    with open(cfg_p) as fh:
        eval_cfg = yaml.safe_load(fh)
    cfg_dir = cfg_p.resolve().parent
    eval_cfg['_cfg_dir'] = str(cfg_dir)

    def _cfg_rel(val, default=''):
        """Resolve a config value as a path relative to the config file's directory.
        GCS URIs and absolute paths are returned unchanged."""
        s = str(val or default)
        if not s or s.startswith('gs://'):
            return s
        p = Path(s)
        return str(p if p.is_absolute() else (cfg_dir / p).resolve())

    model_access = eval_cfg.get('model_access', 'local')
    data_access  = eval_cfg.get('data_access',  'local')
    needs_gcs    = model_access.startswith('gcs') or data_access.startswith('gcs')
    gcs_key_raw  = eval_cfg.get('gcs_key_file')
    overwrite    = eval_cfg.get('gcs_overwrite_cache', False)
    show_prog    = eval_cfg.get('gcs_show_progress', None)

    # 2. GCS authentication
    gcs_key = _cfg_rel(Path(gcs_key_raw).expanduser()) if gcs_key_raw else None
    gcs_fs       = None
    storage_opts = {}
    if gcs_key:
        if not Path(gcs_key).exists():
            raise FileNotFoundError(f'GCS key not found: {gcs_key}')
        import gcsfs
        gcs_fs       = gcsfs.GCSFileSystem(token=gcs_key)
        storage_opts = {'token': gcs_key}
        print(f'[GCS] authenticated via {gcs_key}')
    elif needs_gcs:
        raise ValueError('gcs_key_file is required when model_access or data_access is gcs_*')

    # 3. Training config
    src         = _cfg_rel(eval_cfg.get('train_config', './checkpoints/config.yaml'))
    model_cache = Path(_cfg_rel(eval_cfg.get('model_local_cache', './cache/models')))

    if str(src).startswith('gs://'):
        model_cache.mkdir(parents=True, exist_ok=True)
        local_cfg = Path(_dl_file(src, str(model_cache), gcs_key,
                                  overwrite=overwrite, show_progress=show_prog))
    else:
        local_cfg = Path(src)

    if not local_cfg.exists():
        raise FileNotFoundError(f'Training config not found: {local_cfg.resolve()}')
    with open(local_cfg) as fh:
        train_cfg = yaml.safe_load(fh)
    print(f'[Pipeline] Training config: {local_cfg}')

    # 4. Model directory
    model_dir = _cfg_rel(eval_cfg.get('model_dir', './checkpoints'))
    if model_access == 'gcs_download':
        _dl_dir(str(model_dir), str(model_cache), gcs_key,
                overwrite=overwrite, show_progress=show_prog)
        model_dir_resolved = str(model_cache)
    elif model_access == 'gcs_stream':
        model_dir_resolved = str(model_dir).rstrip('/')
    else:
        model_dir_resolved = str(Path(model_dir))
    print(f'[Pipeline] Model dir: {model_dir_resolved}')

    # 5. Data loading
    def _s(v):
        return slice(None) if v is None else slice(*v)

    data_format    = eval_cfg.get('data_format', 'auto')
    override_files = eval_cfg.get('data_files', []) or []
    data_cache     = Path(_cfg_rel(eval_cfg.get('data_local_cache', './cache/data')))
    nevents_load   = int(eval_cfg.get('nevents', -1))
    use_split      = eval_cfg.get('use_train_split', True)
    train_ratio    = train_cfg.get('train_ratio', 0.8)
    xyslice        = _s(train_cfg.get('xyslice',  [None, -1]))
    etaslice       = _s(train_cfg.get('tauslice', [None, -1]))

    is_h5_cfg = 'hdf5_files' in train_cfg
    tcfg_dir  = train_cfg.get('hdf5_dir'   if is_h5_cfg else 'input_dir',   '.')
    cfg_files = train_cfg.get('hdf5_files' if is_h5_cfg else 'input_files', [])
    eval_cfg_dir = Path(eval_cfg.get('_cfg_dir', '.'))

    def _resolve_local(p: str) -> str:
        if p.startswith('gs://'):
            return p
        pp = Path(p)
        return str(pp if pp.is_absolute() else (eval_cfg_dir / pp).resolve())

    src_files = (
        [_resolve_local(str(f).strip()) for f in override_files]
        if override_files
        else [
            f if str(f).startswith('gs://') else str(Path(tcfg_dir) / f)
            for f in cfg_files
        ]
    )
    file0 = src_files[0]

    def _ext_h5(p): return str(p).endswith('.h5') or str(p).endswith('.hdf5')
    if data_format == 'h5':
        use_h5 = True
    elif data_format == 'root':
        use_h5 = False
    else:
        use_h5 = _ext_h5(file0) or is_h5_cfg

    if data_access == 'gcs_download':
        data_cache.mkdir(parents=True, exist_ok=True)
        load_paths = [
            _dl_file(f, str(data_cache), gcs_key,
                     overwrite=overwrite, show_progress=show_prog)
            for f in src_files
        ]
        file0 = load_paths[0]
    elif data_access == 'gcs_stream':
        if use_h5:
            raise ValueError('HDF5 GCS streaming is not supported. '
                             'Use data_access: gcs_download for .h5 files.')
        load_paths = src_files
    else:
        if not file0.startswith('gs://') and not Path(file0).exists():
            raise FileNotFoundError(f'Data file not found: {file0}')
        load_paths = src_files

    print(f'[Pipeline] Loading: {file0}  ({"h5" if use_h5 else "root"})')
    so = storage_opts if data_access == 'gcs_stream' else None
    if use_h5:
        h5_arg = load_paths if len(load_paths) > 1 else load_paths[0]
        raw = _read_h5(h5_arg, nevents=nevents_load, lazy=True)
    else:
        if _read_root is None:
            raise ImportError(
                'uproot/awkward not installed — ROOT files are not supported. '
                'Install them or set data_format: h5 in the config.'
            )
        raw = _read_root(file0, nevents=nevents_load, storage_options=so)

    grid = {k: raw[k]
            for k in ('tau_min', 'dtau', 'eta_min', 'deta', 'x_min', 'dx', 'y_min', 'dy')
            if k in raw}

    arr     = raw['arr']
    n_train = int(arr.shape[0] * train_ratio)
    data_test = (arr[n_train:] if use_split else arr)
    data_test = data_test[:, :, xyslice, xyslice, etaslice, :]
    print(f'[Pipeline] Data shape: {data_test.shape}')
    del arr; gc.collect()

    # 6. Normalizer
    norm_local = None
    if model_access == 'gcs_stream' and str(model_dir_resolved).startswith('gs://'):
        try:
            tf = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
            tf.close()
            gcs_fs.get(model_dir_resolved + '/normalizer.npz', tf.name)
            norm_local = Path(tf.name)
        except Exception as exc:
            print(f'[Pipeline] No normalizer at GCS: {exc}')
    else:
        cand = Path(model_dir_resolved) / 'normalizer.npz'
        norm_local = cand if cand.exists() else None

    if norm_local is not None and norm_local.exists():
        normalizer     = Normalizer.load(str(norm_local))
        data_test_norm = normalizer.transform(data_test)
        print(f'[Pipeline] Normalizer: {normalizer}')
    else:
        normalizer     = None
        data_test_norm = data_test
        print('[Pipeline] No normalizer — using raw data')

    # 7. Model checkpoint
    backend_cfg     = eval_cfg.get('backend', 'torch')
    ckpt_type       = eval_cfg.get('checkpoint_type', 'best')
    epoch           = int(eval_cfg.get('checkpoint_epoch', 10))
    hidden_channels = train_cfg['hidden_channels']
    n_modes         = train_cfg['n_modes']

    def _local(rel):
        return Path(model_dir_resolved) / rel

    def _gcs_to_tmp(rel):
        uri = model_dir_resolved.rstrip('/') + '/' + rel
        tf  = tempfile.NamedTemporaryFile(suffix=Path(rel).suffix, delete=False)
        tf.close()
        gcs_fs.get(uri, tf.name)
        return Path(tf.name)

    def _resolve_ckpt(rel):
        return _gcs_to_tmp(rel) if model_access == 'gcs_stream' else _local(rel)

    if backend_cfg == 'torch':
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from neuralop.models import FNO

        if ckpt_type in ('best', 'pytorch_best'):
            rel = 'best_model_state_dict.pt'
        elif ckpt_type in ('last', 'pytorch_last'):
            rel = 'last_model.pt'
        else:
            rel = f'checkpoint_{epoch}/best_model_state_dict.pt'

        if model_access == 'gcs_stream':
            uri = model_dir_resolved.rstrip('/') + '/' + rel
            with gcs_fs.open(uri, 'rb') as fh:
                raw_ckpt = torch.load(fh, map_location=device, weights_only=False)
        else:
            ckpt = _local(rel)
            if not ckpt.exists():
                raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
            raw_ckpt = torch.load(str(ckpt), map_location=device, weights_only=False)

        if isinstance(raw_ckpt, dict) and 'model_state_dict' in raw_ckpt:
            state = raw_ckpt['model_state_dict']
        elif isinstance(raw_ckpt, dict) and all(
                isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
            state = raw_ckpt
        else:
            state = {(k[7:] if k.startswith('module.') else k): v
                     for k, v in raw_ckpt.items() if isinstance(v, torch.Tensor)}

        model = FNO(in_channels=4, out_channels=4, n_modes=n_modes,
                    hidden_channels=hidden_channels,
                    projection_channel_ratio=2).to(device)
        model.load_state_dict(state)
        model.eval()
        backend = 'torch'
        print(f'[Pipeline] Torch model  ckpt={ckpt_type}  device={device}')

    else:
        import mlx.core as mx
        import mlx.utils as mlx_utils
        from neuralop_mlx.models import FNO
        from neuralop_mlx.weight_transfer import load_from_torch
        device  = None
        backend = 'mlx'

        force_pt = (backend_cfg == 'torch_to_mlx') or ckpt_type.startswith('pytorch_')
        if ckpt_type in ('best', 'pytorch_best') or backend_cfg == 'torch_to_mlx':
            rel_st, rel_pt = 'best_model.safetensors', 'best_model_state_dict.pt'
        elif ckpt_type in ('last', 'pytorch_last'):
            rel_st, rel_pt = 'last_model.safetensors', 'last_model.pt'
        else:
            rel_st = f'checkpoint_{epoch}/best_model.safetensors'
            rel_pt = f'checkpoint_{epoch}/best_model_state_dict.pt'

        model = FNO(in_channels=4, out_channels=4, n_modes=n_modes,
                    hidden_channels=hidden_channels, projection_channel_ratio=2)

        if force_pt:
            ckpt = _resolve_ckpt(rel_pt)
            if not ckpt.exists():
                raise FileNotFoundError(f'Checkpoint not found: {ckpt}')
            load_from_torch(str(ckpt), model)
            print(f'[Pipeline] MLX ← PyTorch weights: {ckpt}')
        else:
            ckpt = _resolve_ckpt(rel_st)
            if not ckpt.exists():
                print(f'[Pipeline] No safetensors — fallback to PyTorch transfer')
                ckpt = _resolve_ckpt(rel_pt)
                if not ckpt.exists():
                    raise FileNotFoundError(f'No checkpoint found for ckpt_type={ckpt_type}')
                load_from_torch(str(ckpt), model)
            else:
                params = mx.load(str(ckpt))
                nested = mlx_utils.tree_unflatten(list(params.items()))
                model.update(nested)
                mx.eval(model.parameters())
                print(f'[Pipeline] MLX native weights: {ckpt}')
        model.eval()

    # 8. Inference
    print(f'[Pipeline] Running inference  N={data_test_norm.shape[0]} …')
    if backend == 'torch':
        preds = []
        with torch.no_grad():
            for i in range(data_test_norm.shape[0]):
                x0  = data_test_norm[i, :, :, :, :, :1]
                T   = data_test_norm.shape[-1] - 1
                x   = np.repeat(x0, T, axis=-1)
                xin = torch.FloatTensor(x).unsqueeze(0).to(device)
                preds.append(model(xin).squeeze(0).cpu().numpy())
        model_out = np.stack(preds)
    else:
        preds = []
        for i in range(data_test_norm.shape[0]):
            x0  = data_test_norm[i, :, :, :, :, :1].astype(np.float32)
            T   = data_test_norm.shape[-1] - 1
            x   = np.repeat(x0, T, axis=-1)
            xin = mx.array(x[np.newaxis])
            out = model(xin)
            mx.eval(out)
            preds.append(np.array(out[0]))
        model_out = np.stack(preds)
    print(f'[Pipeline] Inference done: {model_out.shape}')

    if normalizer is not None:
        x0_all         = data_test_norm[:, :, :, :, :, :1]
        model_full     = np.concatenate([x0_all, model_out], axis=-1)
        model_out_phys = normalizer.inverse(model_full)[:, :, :, :, :, 1:]
        y_true_phys    = normalizer.inverse(data_test_norm)[:, :, :, :, :, 1:]
    else:
        model_out_phys = model_out
        y_true_phys    = data_test_norm[:, :, :, :, :, 1:]

    # 9. Grid coordinates
    nx_eff, ny_eff, neta_eff = (data_test_norm.shape[2],
                                 data_test_norm.shape[3],
                                 data_test_norm.shape[4])
    ntau_out = data_test_norm.shape[5] - 1

    x_min   = grid.get('x_min',   0.0);  dx   = grid.get('dx',   1.0)
    y_min   = grid.get('y_min',   0.0);  dy   = grid.get('dy',   1.0)
    eta_min = grid.get('eta_min', 0.0);  deta = grid.get('deta', 1.0)
    tau_min = grid.get('tau_min', 0.6);  dtau = grid.get('dtau', 0.1)

    x_coords       = x_min   + np.arange(nx_eff)   * dx
    y_coords       = y_min   + np.arange(ny_eff)   * dy
    eta_coords     = eta_min + np.arange(neta_eff) * deta
    tau_out_coords = tau_min + (np.arange(ntau_out) + 1) * dtau
    xy_extent      = [x_coords[0] - dx / 2, x_coords[-1] + dx / 2,
                      y_coords[0] - dy / 2, y_coords[-1] + dy / 2]

    norm_cfg  = train_cfg.get('normalization', {})
    norm_name = (norm_cfg.get('name', 'none') if isinstance(norm_cfg, dict)
                 else str(norm_cfg))

    # Resolve odir relative to config file so JSON output goes to the right place
    eval_cfg['odir'] = _cfg_rel(eval_cfg.get('odir', './out'))

    print('[Pipeline] Complete.')
    return {
        'model_out_phys': model_out_phys,
        'y_true_phys':    y_true_phys,
        'data_test_norm': data_test_norm,
        'grid':           grid,
        'train_cfg':      train_cfg,
        'eval_cfg':       eval_cfg,
        'backend':        backend,
        'neta_eff':       neta_eff,
        'ntau_out':       ntau_out,
        'n_test':         data_test_norm.shape[0],
        'norm_name':      norm_name,
        'coords': {
            'x_coords':       x_coords,
            'y_coords':       y_coords,
            'eta_coords':     eta_coords,
            'tau_out_coords': tau_out_coords,
            'xy_extent':      xy_extent,
            'dtau':           dtau,
            'tau_min':        tau_min,
        },
    }


def _cfg_mtime(cfg_path: str) -> float:
    try:
        return Path(cfg_path).stat().st_mtime
    except OSError:
        return 0.0


def _pipeline(cfg_path: str) -> dict:
    """Thin wrapper — returns cached result instantly once the pipeline has run."""
    return _run_pipeline(cfg_path, _cfg_mtime(cfg_path))


# ---------------------------------------------------------------------------
# Background pipeline execution
#
# _run_pipeline() is heavy (GCS + data + inference).  Running it in Panel's
# Tornado I/O thread would freeze every connected browser until it finishes.
# Instead we launch it in a daemon thread and use a param trigger to notify
# Panel when the result is ready, causing _render to re-execute.
# ---------------------------------------------------------------------------

class _PipelineState(_param.Parameterized):
    """Trigger param — incremented by background thread to force a re-render."""
    trigger = _param.Integer(default=0)

_pipe_state   = _PipelineState()
_pipe_lock    = threading.Lock()
_pipe_ready   : set  = set()   # keys that completed successfully
_pipe_failed  : dict = {}      # key -> error string for failed runs
_pipe_threads : dict = {}      # cfg_path -> running Thread


def _launch_pipeline(cfg_path: str) -> None:
    """Start the pipeline in a background thread if not already cached or running."""
    key = (cfg_path, _cfg_mtime(cfg_path))

    with _pipe_lock:
        if key in _pipe_ready or key in _pipe_failed:
            return
        if cfg_path in _pipe_threads and _pipe_threads[cfg_path].is_alive():
            return

        def _worker():
            try:
                _run_pipeline(*key)
                print('[Pipeline] Ready — plots will refresh automatically.')
                with _pipe_lock:
                    _pipe_ready.add(key)
            except Exception as exc:
                import traceback
                print(f'[Pipeline] ERROR: {exc}')
                print(traceback.format_exc())
                with _pipe_lock:
                    _pipe_failed[key] = str(exc)
            finally:
                with _pipe_lock:
                    _pipe_threads.pop(cfg_path, None)
                # NOTE: do NOT call pn.state.execute here — it is unreliable
                # from daemon threads because it has no session context.
                # Trigger is fired by the per-session periodic callback instead.

        t = threading.Thread(target=_worker, daemon=True)
        _pipe_threads[cfg_path] = t
        t.start()


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

cfg_w = pn.widgets.TextInput(
    name='Config file path',
    value=_DEFAULT_CFG,
    placeholder='Path to eval_gcs_config.yaml',
)

eval_mode_w = pn.widgets.Select(
    name='Eval mode',
    options=['basic', 'cont', 'metric'],
    value='basic',
    width=160,
)

# -- basic --
event_w   = pn.widgets.IntInput(name='Event index',                       value=0, start=0, step=1, width=130)
ieta_w    = pn.widgets.IntInput(name='η bin index',                       value=0, start=0, step=1, width=130)
channel_w = pn.widgets.IntInput(name='Channel  (0=ε  1=vx  2=vy  3=vη)', value=0, start=0, step=1, width=220)
tau_idx_w = pn.widgets.TextInput(name='τ output indices (comma-sep, −1=last)',
                                  value='5,10,15,-1', width=240)

# -- metric --
channel_drop_w  = pn.widgets.Select(
    name='Channel', options=['0 (ε)', '1 (vx)', '2 (vy)', '3 (vη)'],
    value='0 (ε)', width=150)
active_thresh_w = pn.widgets.FloatInput(
    name='Active ε threshold (GeV/fm³, 0 = no mask)',
    value=0.3, start=0.0, step=0.05, width=230)
spec_tau_w  = pn.widgets.TextInput(name='τ indices for spectrum', value='0,-1', width=180)
save_json_w = pn.widgets.Checkbox(name='Save summary to JSON', value=False)

# -- cont --
cont_event_w   = pn.widgets.IntSlider(name='Event',   start=0, end=9, value=0, step=1)
cont_ieta_w    = pn.widgets.IntSlider(name='η bin',   start=0, end=9, value=0, step=1)
cont_channel_w = pn.widgets.Select(
    name='Feature / channel',
    options=['0 — ε (energy)', '1 — vx', '2 — vy', '3 — vη'],
    value='0 — ε (energy)', width=200)
contour_iT_w = pn.widgets.TextInput(
    name='2nd and 3rd τ-step indices (comma-sep; 1st panel always τ=0; −1=last non-zero)',
    value='10,-1', width=280)
cdfrats_w    = pn.widgets.TextInput(
    name='CDF ratios for contour levels (comma-sep)',
    value='0.05,0.1,0.2,0.3,0.4,0.6', width=300)
rescale_e_w  = pn.widgets.Checkbox(name='Rescale energy: (ε/τ)^(4/3)', value=True)


# ---------------------------------------------------------------------------
# Sidebar controls — reactive to eval mode
# ---------------------------------------------------------------------------

def _sidebar_controls(mode):
    if mode == 'basic':
        return pn.Column(
            pn.Row(event_w, ieta_w),
            pn.Row(channel_w),
            pn.Row(tau_idx_w),
        )
    elif mode == 'cont':
        return pn.Column(
            cont_event_w,
            cont_ieta_w,
            cont_channel_w,
            contour_iT_w,
            cdfrats_w,
            rescale_e_w,
        )
    else:
        return pn.Column(
            pn.Row(channel_drop_w, ieta_w),
            active_thresh_w,
            pn.Row(spec_tau_w, save_json_w),
        )


# ---------------------------------------------------------------------------
# Plot helpers — each takes the pipeline dict and widget values
# ---------------------------------------------------------------------------

def _err(exc: Exception) -> pn.pane.Alert:
    return pn.pane.Alert(
        f'**{type(exc).__name__}:** {exc}',
        alert_type='danger',
    )


def _mpl(fig) -> pn.pane.Matplotlib:
    pane = pn.pane.Matplotlib(fig, tight=True, format='png', dpi=110)
    plt.close(fig)
    return pane


def _make_basic(p, event, ieta, channel, tau_idx_str):
    model_out_phys = p['model_out_phys']
    y_true_phys    = p['y_true_phys']
    ntau_out       = p['ntau_out']
    coords         = p['coords']
    eta_coords     = coords['eta_coords']
    tau_out_coords = coords['tau_out_coords']
    xy_extent      = coords['xy_extent']

    raw_idx  = [t.strip() for t in tau_idx_str.split(',') if t.strip()]
    tau_idxs = [int(t) % ntau_out for t in raw_idx]
    tau_idxs = sorted({t for t in tau_idxs if 0 <= t < ntau_out})
    if not tau_idxs:
        return pn.pane.Alert('No valid τ indices — check the τ output indices field.',
                             alert_type='warning')

    ch_label = (['ε (GeV/fm³)', 'vx', 'vy', 'vη'][channel]
                if channel < 4 else f'ch{channel}')
    eta_phys = eta_coords[ieta] if ieta < len(eta_coords) else ieta
    n_sel    = len(tau_idxs)

    fig, axes = plt.subplots(2, n_sel, figsize=(4.5 * n_sel, 7), squeeze=False)
    fig.suptitle(f'η = {eta_phys:.2f}  (event {event})', fontsize=14, y=1.01)

    for col, t in enumerate(tau_idxs):
        tau_phys = tau_out_coords[t]
        true = y_true_phys[event, channel, :, :, ieta, t]
        pred = model_out_phys[event, channel, :, :, ieta, t]
        vmin, vmax = min(true.min(), pred.min()), max(true.max(), pred.max())
        kw = dict(origin='lower', extent=xy_extent, aspect='equal',
                  vmin=vmin, vmax=vmax, cmap='hot')
        for row, (dat, lbl) in enumerate([(y_true_phys, 'Truth'),
                                           (model_out_phys, 'FNO')]):
            ax = axes[row, col]
            im = ax.imshow(dat[event, channel, :, :, ieta, t].T, **kw)
            ax.set_title(f'{lbl}   τ = {tau_phys:.3f} fm/c', fontsize=10)
            ax.set_xlabel('x (fm)')
            ax.set_ylabel('y (fm)')
            plt.colorbar(im, ax=ax, label=ch_label, fraction=0.046, pad=0.04)

    fig.tight_layout()
    return _mpl(fig)


def _make_l2(p, channel):
    model_out_phys = p['model_out_phys']
    y_true_phys    = p['y_true_phys']
    coords         = p['coords']
    eta_coords     = coords['eta_coords']
    tau_out_coords = coords['tau_out_coords']
    neta_eff       = p['neta_eff']
    sfx            = ' — MLX' if p['backend'] == 'mlx' else ''

    denom  = np.mean(y_true_phys[:, channel] ** 2, axis=(0, 1, 2))
    num    = np.mean((model_out_phys[:, channel] - y_true_phys[:, channel]) ** 2,
                     axis=(0, 1, 2))
    rel_l2 = np.sqrt(num / np.maximum(denom, 1e-30))

    fig, ax = plt.subplots(figsize=(8, 4))
    for ie in range(neta_eff):
        ax.plot(tau_out_coords, rel_l2[ie], label=f'η={eta_coords[ie]:.2f}')
    ax.set_xlabel('τ (fm/c)')
    ax.set_ylabel('Relative L² error')
    ax.set_title(f'Relative L² error vs τ (all η bins, test set){sfx}')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _mpl(fig)


def _make_cont(p, cont_event, cont_ieta, cont_channel_str,
               contour_iT_str, cdfrats_str, rescale_energy):
    model_out_phys = p['model_out_phys']
    y_true_phys    = p['y_true_phys']
    coords         = p['coords']
    eta_coords     = coords['eta_coords']
    x_coords       = coords['x_coords']
    y_coords       = coords['y_coords']
    dtau           = coords['dtau']
    tau_min        = coords['tau_min']

    channel  = int(cont_channel_str.split()[0])
    raw_iT   = [t.strip() for t in contour_iT_str.split(',') if t.strip()]
    raw_iT   = (raw_iT + ['-1', '-1'])[:2]
    cont_iT  = (0,) + tuple(int(t) for t in raw_iT)
    cdfrats  = [float(x.strip()) for x in cdfrats_str.split(',') if x.strip()]
    tau0     = tau_min + dtau
    eta_phys = eta_coords[cont_ieta] if cont_ieta < len(eta_coords) else cont_ieta

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
    plot_three_bins_contour(
        model_out_phys, y_true_phys,
        iparam=channel, event=cont_event,
        tau0=tau0, taudelta=dtau, iT=cont_iT,
        cdfrats=cdfrats, ieta=cont_ieta,
        tag=f'η={eta_phys:.2f}', rescale_energy=not rescale_energy,
        x_coords=x_coords, y_coords=y_coords,
        fig=fig, axes=axes,
    )
    fig.tight_layout()
    return _mpl(fig)


def _make_metric(p, channel_str, ieta, active_thresh, spec_tau_str, save_json):
    model_out_phys = p['model_out_phys']
    y_true_phys    = p['y_true_phys']
    coords         = p['coords']
    tau_out_coords = coords['tau_out_coords']
    eta_coords     = coords['eta_coords']
    backend        = p['backend']
    eval_cfg       = p['eval_cfg']

    ch       = int(channel_str.split()[0])
    thresh   = active_thresh if active_thresh > 0 else None
    tau_idx  = [int(t.strip()) for t in spec_tau_str.split(',') if t.strip()]
    metric_backend = eval_cfg.get('metric_backend', 'numpy')
    sfx      = ' — MLX' if backend == 'mlx' else ''

    evaluator = ModelEval(
        pred=model_out_phys, true=y_true_phys,
        tau_coords=tau_out_coords, eta_coords=eta_coords,
        channel_names=['ε', 'vx', 'vy', 'vη'],
        active_threshold=thresh,
        backend=metric_backend,
    )

    # Dashboard
    fig_dash = evaluator.plot_dashboard(channel=ch, ieta=ieta, tau_indices=tau_idx)
    fig_dash.tight_layout()

    # nRMSE heatmap
    nrmse = evaluator.nrmse()
    fig_nrmse, axes_n = plt.subplots(1, evaluator.C, figsize=(5 * evaluator.C, 3.5))
    if evaluator.C == 1:
        axes_n = [axes_n]
    for c, ax in enumerate(axes_n):
        im = ax.pcolormesh(tau_out_coords, eta_coords, nrmse[c],
                           cmap='hot_r', shading='auto')
        plt.colorbar(im, ax=ax, label='nRMSE')
        ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('η')
        ax.set_title(f'nRMSE — {evaluator.channel_names[c]}')
    fig_nrmse.suptitle(f'nRMSE heat map (η × τ){sfx}', fontsize=12)
    fig_nrmse.tight_layout()

    # fRMSE heatmap
    frmse = evaluator.frmse()
    fig_frmse, axes_f = plt.subplots(1, 3, figsize=(15, 3.5))
    for ax, (band, cmap) in zip(axes_f,
            [('low', 'Blues'), ('mid', 'Oranges'), ('high', 'Reds')]):
        im = ax.pcolormesh(tau_out_coords, eta_coords, frmse[band][ch],
                           cmap=cmap, shading='auto')
        plt.colorbar(im, ax=ax, label='fRMSE')
        ax.set_xlabel('τ (fm/c)'); ax.set_ylabel('η')
        ax.set_title(f'fRMSE ({band}) — {evaluator.channel_names[ch]}')
    fig_frmse.suptitle(f'Spectral RMSE heat maps (η × τ){sfx}', fontsize=12)
    fig_frmse.tight_layout()

    # Scalar summary
    s = evaluator.summary()
    table = ('## Scalar summary\n\n| Metric | Value |\n|---|---|\n'
             + '\n'.join(f'| `{k}` | {v:.6g} |' for k, v in s.items()))

    if save_json:
        sfx2  = '_mlx' if backend == 'mlx' else ''
        opath = Path(eval_cfg.get('odir', './out')) / f'metrics_eval{sfx2}.json'
        opath.parent.mkdir(parents=True, exist_ok=True)
        with open(opath, 'w') as fh:
            json.dump(s, fh, indent=2)
        print(f'[Metric] Summary saved → {opath}')

    return pn.Column(
        _mpl(fig_dash),
        _mpl(fig_nrmse),
        _mpl(fig_frmse),
        pn.pane.Markdown(table),
    )


def _make_training_history(p):
    eval_cfg = p['eval_cfg']
    backend  = p['backend']
    sfx  = '_mlx' if backend == 'mlx' else ''
    odir = Path(eval_cfg.get('odir', './out'))
    hist_path = odir / f'metrics{sfx}.json'
    if not hist_path.exists():
        return pn.pane.Markdown(
            f'*No training history found at `{hist_path}`*', margin=(0, 0, 10, 0))
    hist  = parse_training_log.TrainingHistory.load(str(hist_path))
    label = 'MLX' if backend == 'mlx' else 'PyTorch'
    hist.plot(title=f'FNO Training ({label})')
    fig = plt.gcf()
    fig.tight_layout()
    return _mpl(fig)


# ---------------------------------------------------------------------------
# Status banner — shown above plots
# ---------------------------------------------------------------------------

def _status_banner(p):
    coords = p['coords']
    eta    = coords['eta_coords']
    tau    = coords['tau_out_coords']
    return pn.pane.Alert(
        f"**Loaded** — {p['n_test']} events · "
        f"{p['neta_eff']} η bins  [{eta[0]:.2f}, {eta[-1]:.2f}] · "
        f"{p['ntau_out']} τ steps [{tau[0]:.3f}, {tau[-1]:.3f}] fm/c · "
        f"backend={p['backend']} · norm={p['norm_name']}",
        alert_type='success',
        margin=(0, 0, 10, 0),
    )


# ---------------------------------------------------------------------------
# Main render — single bound function driving all plot output
# ---------------------------------------------------------------------------

def _render(cfg_path, eval_mode,
            event, ieta, channel, tau_idx_str,
            cont_event, cont_ieta, cont_channel_str,
            contour_iT_str, cdfrats_str, rescale_energy,
            channel_drop_str, active_thresh, spec_tau_str, save_json,
            _trigger):  # _trigger is only used to schedule re-renders; not read

    key = (cfg_path, _cfg_mtime(cfg_path))
    with _pipe_lock:
        ready  = key in _pipe_ready
        errmsg = _pipe_failed.get(key)

    if errmsg:
        return pn.Column(
            pn.pane.Alert(
                f'**Pipeline failed:** {errmsg}\n\n'
                'Fix the error shown in the log, then reload the page to retry.',
                alert_type='danger', margin=(5, 0, 10, 0),
            ),
            pn.pane.Markdown('### Log'),
            log_terminal,
            sizing_mode='stretch_width',
        )

    if not ready:
        _launch_pipeline(cfg_path)
        return pn.Column(
            pn.pane.Alert(
                '**Pipeline loading** — GCS auth · data · model · inference are '
                'running in the background.  The page will refresh automatically '
                'when done.',
                alert_type='info', margin=(5, 0, 10, 0),
            ),
            pn.pane.Markdown('### Live log'),
            log_terminal,
            sizing_mode='stretch_width',
        )

    # Pipeline cached — retrieve result (instant) and render plots
    try:
        p = _pipeline(cfg_path)
    except Exception as exc:
        return _err(exc)

    log_card = pn.Card(
        log_terminal,
        title='Pipeline log',
        collapsed=True,
        sizing_mode='stretch_width',
        margin=(10, 0, 0, 0),
    )

    items = [_status_banner(p)]

    if eval_mode == 'basic':
        items.append(_make_basic(p, event, ieta, channel, tau_idx_str))
        items.append(_make_l2(p, channel))
    elif eval_mode == 'cont':
        items.append(_make_cont(p, cont_event, cont_ieta, cont_channel_str,
                                contour_iT_str, cdfrats_str, rescale_energy))
        items.append(_make_l2(p, int(cont_channel_str.split()[0])))
    else:
        items.append(_make_metric(p, channel_drop_str, ieta, active_thresh,
                                  spec_tau_str, save_json))

    items += [pn.layout.Divider(), log_card, _make_training_history(p)]

    return pn.Column(*items, sizing_mode='stretch_width')


main_view = pn.bind(
    _render,
    cfg_w, eval_mode_w,
    event_w, ieta_w, channel_w, tau_idx_w,
    cont_event_w, cont_ieta_w, cont_channel_w,
    contour_iT_w, cdfrats_w, rescale_e_w,
    channel_drop_w, active_thresh_w, spec_tau_w, save_json_w,
    _trigger=_pipe_state.param.trigger,
)

sidebar_controls = pn.bind(_sidebar_controls, eval_mode_w)

# ---------------------------------------------------------------------------
# Slider bound sync — called after pipeline completes, never inside _render.
# ---------------------------------------------------------------------------
def _sync_slider_bounds(cfg_path: str) -> None:
    key = (cfg_path, _cfg_mtime(cfg_path))
    with _pipe_lock:
        if key not in _pipe_ready:
            return
    try:
        p = _pipeline(cfg_path)
        cont_event_w.param.update(end=max(p['n_test'] - 1, 0))
        cont_ieta_w.param.update(end=max(p['neta_eff'] - 1, 0))
    except Exception:
        pass

# Sync slider bounds whenever the pipeline-ready trigger fires (session-safe)
_pipe_state.param.watch(lambda e: _sync_slider_bounds(cfg_w.value), 'trigger')

# ---------------------------------------------------------------------------
# Per-session onload: start pipeline + poll until done, then fire trigger.
#
# pn.state.execute() from a background thread has no session context and
# does not reliably wake up pn.bind.  Instead we use add_periodic_callback,
# which runs on the session's own Tornado IO loop and can safely mutate params.
# ---------------------------------------------------------------------------
def _on_session_load():
    _launch_pipeline(cfg_w.value)
    _already_triggered: set = set()

    def _poll_pipeline():
        cfg_path = cfg_w.value
        key = (cfg_path, _cfg_mtime(cfg_path))
        if key in _already_triggered:
            return  # already fired for this key
        with _pipe_lock:
            done  = key in _pipe_ready or key in _pipe_failed
            alive = cfg_path in _pipe_threads and _pipe_threads[cfg_path].is_alive()
        if done or not alive:
            _already_triggered.add(key)
            _sync_slider_bounds(cfg_path)
            _pipe_state.param.update(trigger=_pipe_state.trigger + 1)

    def _on_cfg_change(event):
        _launch_pipeline(event.new)

    cfg_w.param.watch(_on_cfg_change, 'value')
    pn.state.add_periodic_callback(_poll_pipeline, period=500)

pn.state.onload(_on_session_load)

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

template = pn.template.FastListTemplate(
    title='FNO Model Evaluation',
    sidebar=[
        pn.pane.Markdown('## Config'),
        cfg_w,
        pn.layout.Divider(),
        pn.pane.Markdown('## Eval mode'),
        eval_mode_w,
        pn.layout.Divider(),
        pn.pane.Markdown('## Plot controls'),
        sidebar_controls,
        pn.layout.Divider(),
        pn.pane.Markdown(
            '_Pipeline runs in the background on first load. '
            'Progress is shown in the log on the main panel. '
            'Widget changes only re-render plots once cached._',
            styles={'color': '#888', 'font-size': '0.85em'},
        ),
    ],
    main=[main_view],
    sidebar_width=340,
)

template.servable()
