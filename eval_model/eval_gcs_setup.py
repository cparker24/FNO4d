"""
eval_gcs_setup.py — backend for eval_gcs_clean.py

All heavy setup stages (config loading, GCS auth, data loading, model loading,
inference, grid computation) live here so the marimo notebook stays clean.

Usage inside the notebook:
    import eval_gcs_setup as setup
    ctx = setup.run_all()        # raises on any error
    # then unpack ctx.model_out_phys, ctx.eta_coords, …
"""

import gc
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Locate loc_libs relative to this file so imports work regardless of CWD
_HERE     = Path(__file__).parent
_LOC_LIBS = _HERE.parent / 'loc_libs'
if str(_LOC_LIBS) not in sys.path:
    sys.path.insert(0, str(_LOC_LIBS))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    """Everything the eval notebook needs after setup completes."""
    eval_cfg:       dict
    train_cfg:      dict
    backend:        str            # 'torch' | 'mlx'
    data_test:      np.ndarray     # raw (unnormalised) test events
    normalizer:     Any            # Normalizer | None
    model_out_phys: np.ndarray
    y_true_phys:    np.ndarray
    # grid coordinates
    eta_coords:     np.ndarray
    tau_out_coords: np.ndarray
    x_coords:       np.ndarray
    y_coords:       np.ndarray
    xy_extent:      list
    dtau:           float
    tau_min:        float
    neta_eff:       int
    ntau_out:       int


# ---------------------------------------------------------------------------
# Stage 1: eval config
# ---------------------------------------------------------------------------

def load_eval_config(cfg_path: Optional[str] = None) -> dict:
    import yaml
    if cfg_path is None:
        _env = os.environ.get('EVAL_GCS_CONFIG', '')
        cfg_path = _env if _env else 'eval_gcs_config.yaml'
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Eval config not found: {p.resolve()}\n"
            "Create eval_gcs_config.yaml in the working directory or set "
            "EVAL_GCS_CONFIG=/path/to/config.yaml"
        )
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg['_cfg_dir'] = str(p.resolve().parent)
    print(f'Eval config: {p.resolve()}')
    return cfg


# ---------------------------------------------------------------------------
# Stage 2: GCS auth
# ---------------------------------------------------------------------------

def setup_gcs(eval_cfg: dict):
    """Return (gcs_fs, gcs_key, storage_opts). gcs_fs is None for local-only."""
    _model_access = eval_cfg.get('model_access', 'local')
    _data_access  = eval_cfg.get('data_access',  'local')
    _needs_gcs    = _model_access.startswith('gcs') or _data_access.startswith('gcs')
    _key_raw      = eval_cfg.get('gcs_key_file')

    gcs_key = str(Path(_key_raw).expanduser()) if _key_raw else None

    if _needs_gcs:
        if not gcs_key:
            raise ValueError(
                "gcs_key_file is required when model_access or data_access is gcs_*"
            )
        if not Path(gcs_key).exists():
            raise FileNotFoundError(f"GCS key file not found: {gcs_key}")

    gcs_fs       = None
    storage_opts = {}
    if gcs_key:
        try:
            import gcsfs
            gcs_fs       = gcsfs.GCSFileSystem(token=gcs_key)
            storage_opts = {'token': gcs_key}
            print(f'[GCS] authenticated via {gcs_key}')
        except ImportError:
            if _needs_gcs:
                raise ImportError(
                    "gcsfs is required for GCS access.\n"
                    "Install: pip install gcsfs   or   pip install -r requirements-gcs.txt"
                )
            print('[GCS] gcsfs unavailable — local-only mode')

    return gcs_fs, gcs_key, storage_opts


# ---------------------------------------------------------------------------
# Stage 3: training config
# ---------------------------------------------------------------------------

def load_train_config(eval_cfg: dict, gcs_fs, gcs_key: Optional[str]) -> dict:
    import yaml
    from gcs_utils import download_gcs_file

    src         = eval_cfg.get('train_config', './checkpoints/config.yaml')
    model_cache = Path(eval_cfg.get('model_local_cache', './cache/models'))

    if str(src).startswith('gs://'):
        if gcs_fs is None:
            raise RuntimeError(f"GCS auth required to fetch training config: {src}")
        model_cache.mkdir(parents=True, exist_ok=True)
        local_cfg = Path(download_gcs_file(
            src, str(model_cache), gcs_key,
            overwrite=eval_cfg.get('gcs_overwrite_cache', False),
        ))
    else:
        local_cfg = Path(src)
        # If the local file doesn't exist yet and the model is being downloaded from
        # GCS, try fetching config.yaml directly from the GCS model directory.
        # This handles the common case where train_config points inside
        # model_local_cache but resolve_model_dir hasn't run yet.
        if not local_cfg.exists() and eval_cfg.get('model_access', 'local') == 'gcs_download':
            model_dir = eval_cfg.get('model_dir', '')
            if str(model_dir).startswith('gs://'):
                if gcs_fs is None:
                    raise RuntimeError(
                        f"GCS auth required to fetch training config from model_dir: {model_dir}"
                    )
                gcs_cfg_uri = str(model_dir).rstrip('/') + '/config.yaml'
                print(f'[train_config] {local_cfg} not found — fetching from {gcs_cfg_uri}')
                model_cache.mkdir(parents=True, exist_ok=True)
                local_cfg = Path(download_gcs_file(
                    gcs_cfg_uri, str(model_cache), gcs_key,
                    overwrite=eval_cfg.get('gcs_overwrite_cache', False),
                ))

    if not local_cfg.exists():
        raise FileNotFoundError(
            f"Training config not found: {local_cfg.resolve()}\n"
            "Check train_config in eval_gcs_config.yaml"
        )
    with open(local_cfg) as f:
        cfg = yaml.safe_load(f)
    print(f'Training config: {local_cfg}')
    return cfg


# ---------------------------------------------------------------------------
# Stage 4: model directory
# ---------------------------------------------------------------------------

def resolve_model_dir(eval_cfg: dict, gcs_fs, gcs_key: Optional[str]) -> str:
    from gcs_utils import download_gcs_dir

    model_access = eval_cfg.get('model_access', 'local')
    model_dir    = eval_cfg.get('model_dir', './checkpoints')
    model_cache  = Path(eval_cfg.get('model_local_cache', './cache/models'))
    overwrite    = eval_cfg.get('gcs_overwrite_cache', False)
    show_prog    = eval_cfg.get('gcs_show_progress', None)

    if model_access == 'gcs_download':
        if gcs_fs is None:
            raise RuntimeError("GCS auth required for model_access: gcs_download")
        download_gcs_dir(str(model_dir), str(model_cache), gcs_key,
                         overwrite=overwrite, show_progress=show_prog)
        resolved = str(model_cache)
        print(f'Model dir (downloaded from GCS): {resolved}')
    elif model_access == 'gcs_stream':
        resolved = str(model_dir).rstrip('/')
        print(f'Model dir (GCS stream): {resolved}')
    else:
        resolved = str(Path(model_dir))
        print(f'Model dir (local): {resolved}')
    return resolved


# ---------------------------------------------------------------------------
# Stage 5: data loading
# ---------------------------------------------------------------------------

def load_data(eval_cfg: dict, gcs_fs, gcs_key: Optional[str],
              storage_opts: dict, train_cfg: dict):
    from read_3d_root import read_3d_data
    from read_3d_hdf5 import read_3d_data_hdf5
    from gcs_utils    import download_gcs_file

    def _s(v): return slice(None) if v is None else slice(*v)

    data_access     = eval_cfg.get('data_access',  'local')
    data_format     = eval_cfg.get('data_format',  'auto')
    override_files  = eval_cfg.get('data_files',   []) or []
    data_cache      = Path(eval_cfg.get('data_local_cache', './cache/data'))
    nevents_load    = int(eval_cfg.get('nevents', -1))
    use_split       = eval_cfg.get('use_train_split', True)
    overwrite       = eval_cfg.get('gcs_overwrite_cache', False)
    show_prog       = eval_cfg.get('gcs_show_progress', None)
    train_ratio     = train_cfg.get('train_ratio', 0.8)
    xyslice         = _s(train_cfg.get('xyslice', [None, -1]))
    if 'etaslice' in train_cfg:
        etaslice = _s(train_cfg['etaslice'])
        tauslice = _s(train_cfg.get('tauslice', [None, None]))
    else:
        # old config: 'tauslice' was applied to the eta axis
        etaslice = _s(train_cfg.get('tauslice', [None, -1]))
        tauslice = slice(None, None)

    is_h5_cfg  = 'hdf5_files' in train_cfg
    tcfg_dir   = train_cfg.get('hdf5_dir'   if is_h5_cfg else 'input_dir',   '.')
    cfg_files  = train_cfg.get('hdf5_files' if is_h5_cfg else 'input_files', [])
    cfg_dir    = Path(eval_cfg.get('_cfg_dir', '.'))

    def _resolve_local(p: str) -> str:
        if p.startswith('gs://'):
            return p
        pp = Path(p)
        return str(pp if pp.is_absolute() else (cfg_dir / pp).resolve())

    # Build source file list
    if override_files:
        src_files = [_resolve_local(str(f).strip()) for f in override_files]
    else:
        src_files = [
            f if str(f).startswith('gs://') else str(Path(tcfg_dir) / f)
            for f in cfg_files
        ]

    file0 = src_files[0]

    def _ext_h5(p): return str(p).endswith('.h5') or str(p).endswith('.hdf5')
    if data_format == 'h5':
        use_h5 = True
    elif data_format == 'root':
        use_h5 = False
    else:
        use_h5 = _ext_h5(file0) or is_h5_cfg

    # Resolve access mode
    if data_access == 'gcs_download':
        if gcs_fs is None:
            raise RuntimeError("GCS auth required for data_access: gcs_download")
        data_cache.mkdir(parents=True, exist_ok=True)
        load_paths = [
            download_gcs_file(f, str(data_cache), gcs_key,
                              overwrite=overwrite, show_progress=show_prog)
            for f in src_files
        ]
        file0 = load_paths[0]

    elif data_access == 'gcs_stream':
        if use_h5:
            raise ValueError(
                "HDF5 GCS streaming is not supported. "
                "Use data_access: gcs_download for .h5 files."
            )
        load_paths = src_files

    else:  # local
        if not file0.startswith('gs://') and not Path(file0).exists():
            raise FileNotFoundError(
                f"Data file not found: {file0}\n"
                "Check data_files or training config paths in eval_gcs_config.yaml"
            )
        load_paths = src_files

    print(f'Loading data: {file0}  (format: {"h5" if use_h5 else "root"})')

    so = storage_opts if data_access == 'gcs_stream' else None
    if use_h5:
        h5_arg = load_paths if len(load_paths) > 1 else load_paths[0]
        raw = read_3d_data_hdf5(h5_arg, nevents=nevents_load, lazy=True)
    else:
        raw = read_3d_data(file0, nevents=nevents_load, storage_options=so)

    grid = {
        k: raw[k]
        for k in ('tau_min', 'dtau', 'eta_min', 'deta', 'x_min', 'dx', 'y_min', 'dy')
        if k in raw
    }
    print('Grid:', {k: f'{v:.4g}' for k, v in grid.items()})

    arr     = raw['arr']
    n_train = int(arr.shape[0] * train_ratio)
    if use_split:
        data_test = arr[n_train:][:, :, xyslice, xyslice, etaslice, tauslice]
        print(f'Test split: {data_test.shape[0]} events  shape={data_test.shape}')
    else:
        data_test = arr[:, :, xyslice, xyslice, etaslice, tauslice]
        print(f'All events: {data_test.shape[0]}  shape={data_test.shape}')
    del arr; gc.collect()
    return data_test, grid


# ---------------------------------------------------------------------------
# Stage 6: normalizer
# ---------------------------------------------------------------------------

def load_normalizer(eval_cfg: dict, gcs_fs, model_dir_resolved: str):
    """Load the normalizer only — no full-array transform.

    The per-batch collate in :func:`run_inference` applies the normalizer
    during inference, so we no longer materialise a full normalised copy of the
    test array here (keeps peak eval memory to one raw + pred + true set).
    """
    from norm_utils import Normalizer

    model_access = eval_cfg.get('model_access', 'local')

    if model_access == 'gcs_stream' and str(model_dir_resolved).startswith('gs://'):
        if gcs_fs is None:
            raise RuntimeError("GCS fs required to fetch normalizer.npz")
        norm_uri = model_dir_resolved.rstrip('/') + '/normalizer.npz'
        norm_local = None
        try:
            tf = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
            tf.close()
            gcs_fs.get(norm_uri, tf.name)
            norm_local = Path(tf.name)
        except Exception as e:
            print(f'No normalizer at {norm_uri}: {e}')
    else:
        norm_local = Path(model_dir_resolved) / 'normalizer.npz'
        if not norm_local.exists():
            norm_local = None

    if norm_local is not None and norm_local.exists():
        normalizer = Normalizer.load(str(norm_local))
        print(f'Normalizer: {normalizer}')
    else:
        normalizer = None
        print('No normalizer found — using raw data')
    return normalizer


# ---------------------------------------------------------------------------
# Stage 7: model
# ---------------------------------------------------------------------------

def load_model(eval_cfg: dict, gcs_fs, model_dir_resolved: str, train_cfg: dict):
    model_access    = eval_cfg.get('model_access',    'local')
    backend_cfg     = eval_cfg.get('backend',         'torch')
    ckpt_type       = eval_cfg.get('checkpoint_type', 'best')
    epoch           = int(eval_cfg.get('checkpoint_epoch', 10))
    hidden_channels = train_cfg['hidden_channels']
    n_modes         = train_cfg['n_modes']

    def _local_path(rel):
        return Path(model_dir_resolved) / rel

    def _gcs_fetch(rel):
        uri = model_dir_resolved.rstrip('/') + '/' + rel
        sf  = Path(rel).suffix
        tf  = tempfile.NamedTemporaryFile(suffix=sf, delete=False)
        tf.close()
        gcs_fs.get(uri, tf.name)
        return Path(tf.name)

    def _get_path(rel):
        return _gcs_fetch(rel) if model_access == 'gcs_stream' else _local_path(rel)

    def _check(p, label):
        if not p.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                f"Check model_dir and checkpoint_type in eval_gcs_config.yaml"
            )

    # ---- PyTorch ------------------------------------------------------------
    if backend_cfg == 'torch':
        import torch
        from neuralop.models import FNO

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')

        if ckpt_type in ('best', 'pytorch_best'):
            rel = 'best_model_state_dict.pt'
        elif ckpt_type in ('last', 'pytorch_last'):
            rel = 'last_model.pt'
        else:
            rel = f'checkpoint_{epoch}/best_model_state_dict.pt'

        if model_access == 'gcs_stream':
            if gcs_fs is None:
                raise RuntimeError("GCS fs required for model_access: gcs_stream")
            uri = model_dir_resolved.rstrip('/') + '/' + rel
            with gcs_fs.open(uri, 'rb') as f:
                raw = torch.load(f, map_location=device, weights_only=False)
        else:
            ckpt = _local_path(rel)
            _check(ckpt, rel)
            raw = torch.load(str(ckpt), map_location=device, weights_only=False)

        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state = raw['model_state_dict']
        elif isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
            state = raw
        else:
            state = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in raw.items() if isinstance(v, torch.Tensor)
            }

        model = FNO(
            in_channels=4, out_channels=4,
            n_modes=n_modes, hidden_channels=hidden_channels,
            projection_channel_ratio=2,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        backend = 'torch'
        print(f'Torch model: {sum(p.numel() for p in model.parameters()):,} params  ckpt={ckpt_type}')

    # ---- MLX / torch_to_mlx -------------------------------------------------
    else:
        import mlx.core as mx
        import mlx.utils as mlx_utils
        from neuralop_mlx.models import FNO
        from neuralop_mlx.utils import count_model_params
        from neuralop_mlx.weight_transfer import load_from_torch

        device   = None
        backend  = 'mlx'
        force_pt = (backend_cfg == 'torch_to_mlx') or ckpt_type.startswith('pytorch_')

        if ckpt_type in ('best', 'pytorch_best') or backend_cfg == 'torch_to_mlx':
            rel_st, rel_pt = 'best_model.safetensors', 'best_model_state_dict.pt'
        elif ckpt_type in ('last', 'pytorch_last'):
            rel_st, rel_pt = 'last_model.safetensors', 'last_model.pt'
        else:
            rel_st = f'checkpoint_{epoch}/best_model.safetensors'
            rel_pt = f'checkpoint_{epoch}/best_model_state_dict.pt'

        model = FNO(
            in_channels=4, out_channels=4,
            n_modes=n_modes, hidden_channels=hidden_channels,
            projection_channel_ratio=2,
        )

        if force_pt:
            ckpt = _get_path(rel_pt)
            _check(ckpt, rel_pt)
            print(f'MLX ← PyTorch weights: {ckpt}')
            load_from_torch(str(ckpt), model)
        else:
            ckpt = _get_path(rel_st)
            if not ckpt.exists():
                print(f'No safetensors at {rel_st} — fallback to PyTorch weight transfer')
                ckpt = _get_path(rel_pt)
                _check(ckpt, rel_pt)
                load_from_torch(str(ckpt), model)
            else:
                params = mx.load(str(ckpt))
                nested = mlx_utils.tree_unflatten(list(params.items()))
                model.update(nested)
                mx.eval(model.parameters())
                print(f'MLX native weights: {ckpt}')

        model.eval()
        print(f'MLX model: {count_model_params(model):,} params  ckpt={ckpt_type}')

    return model, device, backend


# ---------------------------------------------------------------------------
# Stages 8–9: inference + grid coordinates
# ---------------------------------------------------------------------------
# Both are shared, backend-agnostic helpers and now live in loc_libs/eval.py so
# the workflow_torch CLI and the eval notebooks use one implementation.  Thin
# wrappers below keep run_all()'s call sites and console output unchanged.

from eval import run_inference as _run_inference_impl
from eval import compute_grid_coords as _compute_grid_coords_impl


def run_inference(model, backend: str, device, data_test: np.ndarray, normalizer,
                  batch_size: int = 8):
    """Batched inference (loc_libs/eval.run_inference) with a progress print."""
    model_out_phys, y_true_phys = _run_inference_impl(
        model, backend, device, data_test, normalizer, batch_size=batch_size)
    print(f'Batched inference done (batch={batch_size}) — '
          f'pred: {model_out_phys.shape}, true: {y_true_phys.shape}')
    return model_out_phys, y_true_phys


def compute_grid_coords(data_test: np.ndarray, grid: dict) -> dict:
    """Physical grid coordinates (loc_libs/eval.compute_grid_coords) + summary print."""
    coords = _compute_grid_coords_impl(data_test, grid)
    xc, yc, ec, tc = (coords['x_coords'], coords['y_coords'],
                      coords['eta_coords'], coords['tau_out_coords'])
    print(f'Grid  x: [{xc[0]:.2f}, {xc[-1]:.2f}] fm  '
          f'y: [{yc[0]:.2f}, {yc[-1]:.2f}] fm')
    print(f'      η: [{ec[0]:.2f}, {ec[-1]:.2f}]  '
          f'τ_out: [{tc[0]:.3f}, {tc[-1]:.3f}] fm/c')
    return coords


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_all(cfg_path: Optional[str] = None) -> EvalContext:
    """Run the full setup pipeline and return an EvalContext."""
    eval_cfg           = load_eval_config(cfg_path)
    gcs_fs, gcs_key, storage_opts = setup_gcs(eval_cfg)
    train_cfg          = load_train_config(eval_cfg, gcs_fs, gcs_key)
    model_dir_resolved = resolve_model_dir(eval_cfg, gcs_fs, gcs_key)
    data_test, grid    = load_data(eval_cfg, gcs_fs, gcs_key, storage_opts, train_cfg)
    normalizer         = load_normalizer(eval_cfg, gcs_fs, model_dir_resolved)
    model, device, backend     = load_model(eval_cfg, gcs_fs, model_dir_resolved, train_cfg)
    batch_size         = int(eval_cfg.get('eval_batch_size', 8))
    model_out_phys, y_true_phys = run_inference(
        model, backend, device, data_test, normalizer, batch_size)
    coords = compute_grid_coords(data_test, grid)

    return EvalContext(
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        backend=backend,
        data_test=data_test,
        normalizer=normalizer,
        model_out_phys=model_out_phys,
        y_true_phys=y_true_phys,
        **coords,
    )
