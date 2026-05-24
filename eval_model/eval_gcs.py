import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # FNO Model Evaluation — GCS-enabled

    All source settings (file locations, GCS auth, data format, model backend) are
    read from a YAML config file.

    **Config lookup order:**
    1. `EVAL_GCS_CONFIG` environment variable
    2. `eval_gcs_config.yaml` in the working directory

    Marimo UI below controls **only** visualization (basic / cont / metric).

    | Mode | Description |
    |---|---|
    | `basic`  | Truth / prediction image grids per η bin at selected τ steps |
    | `cont`   | CDF-based contour overlays (3 τ snapshots per η bin) |
    | `metric` | Full metric suite: nRMSE · fRMSE · MELR · WLR dashboard |
    """)
    return


# ---------------------------------------------------------------------------
# Cell: imports + load eval config YAML
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    import os, sys, gc
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    sys.path.insert(0, '../loc_libs')

    _cfg_env  = os.environ.get('EVAL_GCS_CONFIG', '')
    _cfg_path = Path(_cfg_env) if _cfg_env else Path('eval_gcs_config.yaml')
    mo.stop(
        not _cfg_path.exists(),
        mo.callout(
            mo.md(
                f"**Eval config not found:** `{_cfg_path.resolve()}`\n\n"
                "Create `eval_gcs_config.yaml` in the working directory or set "
                "`EVAL_GCS_CONFIG=/path/to/config.yaml`."
            ),
            kind="warn",
        ),
    )
    with open(_cfg_path) as _f:
        eval_cfg = yaml.safe_load(_f)
    # Stash the config directory so data/model paths can be resolved relative to it
    eval_cfg['_cfg_dir'] = str(_cfg_path.resolve().parent)
    print(f'Eval config: {_cfg_path.resolve()}')
    return eval_cfg, gc, mo, np, os, plt, sys, yaml, Path


# ---------------------------------------------------------------------------
# Cell: GCS filesystem + storage options
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, mo):
    from pathlib import Path as _Path

    _gcs_key_raw  = eval_cfg.get('gcs_key_file')
    _model_access = eval_cfg.get('model_access', 'local')
    _data_access  = eval_cfg.get('data_access',  'local')
    _needs_gcs    = _model_access.startswith('gcs') or _data_access.startswith('gcs')

    gcs_key = str(_Path(_gcs_key_raw).expanduser()) if _gcs_key_raw else None

    if _needs_gcs:
        mo.stop(
            not gcs_key,
            mo.callout(
                mo.md("**`gcs_key_file` is required** when `model_access` or `data_access` is `gcs_*`."),
                kind="danger",
            ),
        )
        mo.stop(
            not _Path(gcs_key).exists(),
            mo.callout(mo.md(f"**GCS key not found:** `{gcs_key}`"), kind="danger"),
        )

    gcs_fs       = None
    storage_opts = {}
    if gcs_key:
        try:
            import gcsfs as _gcsfs
            gcs_fs       = _gcsfs.GCSFileSystem(token=gcs_key)
            storage_opts = {'token': gcs_key}
            print(f'[GCS] authenticated via {gcs_key}')
        except ImportError:
            if _needs_gcs:
                mo.stop(
                    True,
                    mo.callout(
                        mo.md("**gcsfs not installed.**  `pip install gcsfs`  or "
                              "`pip install -r requirements-gcs.txt`"),
                        kind="danger",
                    ),
                )
            print('[GCS] gcsfs unavailable — local-only mode')
    return gcs_fs, gcs_key, storage_opts


# ---------------------------------------------------------------------------
# Cell: load training config (local or from GCS)
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, gcs_fs, gcs_key, mo):
    import yaml as _yaml
    from pathlib import Path as _Path
    from gcs_utils import download_gcs_file as _dl_file

    _src         = eval_cfg.get('train_config', './checkpoints/config.yaml')
    _model_cache = _Path(eval_cfg.get('model_local_cache', './cache/models'))

    if str(_src).startswith('gs://'):
        mo.stop(
            gcs_fs is None,
            mo.callout(mo.md(f"**GCS auth needed** to fetch training config `{_src}`"), kind="danger"),
        )
        _model_cache.mkdir(parents=True, exist_ok=True)
        _local_cfg = _Path(
            _dl_file(_src, str(_model_cache), gcs_key,
                     overwrite=eval_cfg.get('gcs_overwrite_cache', False))
        )
    else:
        _local_cfg = _Path(_src)

    mo.stop(
        not _local_cfg.exists(),
        mo.callout(
            mo.md(f"**Training config not found:** `{_local_cfg.resolve()}`\n\n"
                  "Check `train_config` in `eval_gcs_config.yaml`."),
            kind="warn",
        ),
    )
    with open(_local_cfg) as _f:
        train_cfg = _yaml.safe_load(_f)
    print(f'Training config: {_local_cfg}')
    return (train_cfg,)


# ---------------------------------------------------------------------------
# Cell: resolve model directory (download prefix from GCS if needed)
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, gcs_fs, gcs_key, mo):
    from pathlib import Path as _Path
    from gcs_utils import download_gcs_dir as _dl_dir

    _model_access = eval_cfg.get('model_access', 'local')
    _model_dir    = eval_cfg.get('model_dir', './checkpoints')
    _model_cache  = _Path(eval_cfg.get('model_local_cache', './cache/models'))
    _overwrite    = eval_cfg.get('gcs_overwrite_cache', False)
    _show_prog    = eval_cfg.get('gcs_show_progress', None)

    if _model_access == 'gcs_download':
        mo.stop(
            gcs_fs is None,
            mo.callout(mo.md("**GCS auth needed** for `model_access: gcs_download`."), kind="danger"),
        )
        _dl_dir(str(_model_dir), str(_model_cache), gcs_key,
                overwrite=_overwrite, show_progress=_show_prog)
        model_dir_resolved = str(_model_cache)
        print(f'Model dir (downloaded from GCS): {model_dir_resolved}')
    elif _model_access == 'gcs_stream':
        model_dir_resolved = str(_model_dir).rstrip('/')  # keep as gs:// prefix
        print(f'Model dir (GCS stream): {model_dir_resolved}')
    else:  # local
        model_dir_resolved = str(_Path(_model_dir))
        print(f'Model dir (local): {model_dir_resolved}')
    return (model_dir_resolved,)


# ---------------------------------------------------------------------------
# Cell: load and slice evaluation data
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, gcs_fs, gcs_key, mo, storage_opts, train_cfg):
    import gc as _gc
    from pathlib import Path as _Path
    from read_3d_root  import read_3d_data        as _read_root
    from read_3d_hdf5  import read_3d_data_hdf5   as _read_h5
    from gcs_utils     import download_gcs_file    as _dl_file

    def _s(v): return slice(None) if v is None else slice(*v)

    _data_access  = eval_cfg.get('data_access',  'local')
    _data_format  = eval_cfg.get('data_format',  'auto')
    _override_files = eval_cfg.get('data_files', []) or []
    _data_cache   = _Path(eval_cfg.get('data_local_cache', './cache/data'))
    _nevents_load = int(eval_cfg.get('nevents', -1))
    _use_split    = eval_cfg.get('use_train_split', True)
    _overwrite    = eval_cfg.get('gcs_overwrite_cache', False)
    _show_prog    = eval_cfg.get('gcs_show_progress', None)
    _train_ratio  = train_cfg.get('train_ratio', 0.8)
    _xyslice      = _s(train_cfg.get('xyslice', [None, -1]))
    if 'etaslice' in train_cfg:
        _etaslice = _s(train_cfg['etaslice'])
        _tauslice = _s(train_cfg.get('tauslice', [None, None]))
    else:
        # old config: 'tauslice' was applied to the eta axis
        _etaslice = _s(train_cfg.get('tauslice', [None, -1]))
        _tauslice = slice(None, None)

    _is_h5_cfg  = 'hdf5_files' in train_cfg
    _tcfg_dir   = train_cfg.get('hdf5_dir'   if _is_h5_cfg else 'input_dir',   '.')
    _cfg_files  = train_cfg.get('hdf5_files' if _is_h5_cfg else 'input_files', [])
    _eval_cfg_dir = _Path(eval_cfg.get('_cfg_dir', '.'))

    def _resolve_local(p: str) -> str:
        """Resolve a non-GCS path relative to the eval config file's directory."""
        if p.startswith('gs://'):
            return p
        pp = _Path(p)
        return str(pp if pp.is_absolute() else (_eval_cfg_dir / pp).resolve())

    # Build source file list
    if _override_files:
        _src_files = [_resolve_local(str(f).strip()) for f in _override_files]
    else:
        _src_files = [
            f if str(f).startswith('gs://') else str(_Path(_tcfg_dir) / f)
            for f in _cfg_files
        ]

    _file0 = _src_files[0]

    # Determine format
    def _ext_h5(p): return str(p).endswith('.h5') or str(p).endswith('.hdf5')
    if _data_format == 'h5':
        _use_h5 = True
    elif _data_format == 'root':
        _use_h5 = False
    else:  # auto
        _use_h5 = _ext_h5(_file0) or _is_h5_cfg

    # Resolve access mode → _load_paths (always local after this block)
    if _data_access == 'gcs_download':
        mo.stop(
            gcs_fs is None,
            mo.callout(mo.md("**GCS auth needed** for `data_access: gcs_download`."), kind="danger"),
        )
        _data_cache.mkdir(parents=True, exist_ok=True)
        _load_paths = [
            _dl_file(f, str(_data_cache), gcs_key,
                     overwrite=_overwrite, show_progress=_show_prog)
            for f in _src_files
        ]
        _file0 = _load_paths[0]

    elif _data_access == 'gcs_stream':
        mo.stop(
            _use_h5,
            mo.callout(
                mo.md("**HDF5 GCS streaming is not supported.**  "
                      "Use `data_access: gcs_download` for `.h5` files."),
                kind="danger",
            ),
        )
        # ROOT files: uproot streams via storage_options; no local copy needed
        _load_paths = _src_files

    else:  # local
        mo.stop(
            not _file0.startswith('gs://') and not _Path(_file0).exists(),
            mo.callout(
                mo.md(f"**Data file not found:** `{_file0}`\n\n"
                      "Check `data_files` or training config paths in "
                      "`eval_gcs_config.yaml`."),
                kind="warn",
            ),
        )
        _load_paths = _src_files

    print(f'Loading data: {_file0}  (format: {"h5" if _use_h5 else "root"})')

    _so = storage_opts if _data_access == 'gcs_stream' else None
    if _use_h5:
        _h5_arg = _load_paths if len(_load_paths) > 1 else _load_paths[0]
        raw = _read_h5(_h5_arg, nevents=_nevents_load, lazy=True)
    else:
        raw = _read_root(_file0, nevents=_nevents_load, storage_options=_so)

    grid = {
        k: raw[k]
        for k in ('tau_min', 'dtau', 'eta_min', 'deta', 'x_min', 'dx', 'y_min', 'dy')
        if k in raw
    }
    print('Grid:', {k: f'{v:.4g}' for k, v in grid.items()})

    arr = raw['arr']
    n_train = int(arr.shape[0] * _train_ratio)
    if _use_split:
        data_test = arr[n_train:]
        data_test = data_test[:, :, _xyslice, _xyslice, _etaslice, _tauslice]
        print(f'Test split: {data_test.shape[0]} events  shape={data_test.shape}')
    else:
        data_test = arr
        data_test = data_test[:, :, _xyslice, _xyslice, _etaslice, _tauslice]
        print(f'All events: {data_test.shape[0]}  shape={data_test.shape}')
    del arr; _gc.collect()
    return data_test, grid


# ---------------------------------------------------------------------------
# Cell: load normalizer
# ---------------------------------------------------------------------------
@app.cell
def _(data_test, eval_cfg, gcs_fs, model_dir_resolved, mo):
    import tempfile as _tmp
    from pathlib import Path as _Path
    from norm_utils import Normalizer

    _model_access = eval_cfg.get('model_access', 'local')

    if _model_access == 'gcs_stream' and str(model_dir_resolved).startswith('gs://'):
        mo.stop(
            gcs_fs is None,
            mo.callout(mo.md("**GCS fs required** to stream normalizer.npz."), kind="danger"),
        )
        _norm_uri = model_dir_resolved.rstrip('/') + '/normalizer.npz'
        _norm_local = None
        try:
            _tf = _tmp.NamedTemporaryFile(suffix='.npz', delete=False)
            _tf.close()
            gcs_fs.get(_norm_uri, _tf.name)
            _norm_local = _Path(_tf.name)
        except Exception as _e:
            print(f'No normalizer at {_norm_uri}: {_e}')
    else:
        _norm_local = _Path(model_dir_resolved) / 'normalizer.npz'
        if not _norm_local.exists():
            _norm_local = None

    if _norm_local is not None and _norm_local.exists():
        normalizer     = Normalizer.load(str(_norm_local))
        data_test_norm = normalizer.transform(data_test)
        print(f'Normalizer: {normalizer}')
    else:
        normalizer     = None
        data_test_norm = data_test
        print('No normalizer found — using raw data')
    return data_test_norm, normalizer


# ---------------------------------------------------------------------------
# Cell: load model checkpoint
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, gcs_fs, mo, model_dir_resolved, train_cfg):
    import tempfile as _tmp
    from pathlib import Path as _Path

    _model_access    = eval_cfg.get('model_access',    'local')
    _backend_cfg     = eval_cfg.get('backend',         'torch')
    _ckpt_type       = eval_cfg.get('checkpoint_type', 'best')
    _epoch           = int(eval_cfg.get('checkpoint_epoch', 10))
    _hidden_channels = train_cfg['hidden_channels']
    _n_modes         = train_cfg['n_modes']

    # ---- helpers -----------------------------------------------------------
    def _local_path(rel: str) -> _Path:
        return _Path(model_dir_resolved) / rel

    def _gcs_fetch_to_temp(rel: str) -> _Path:
        """Download a single file from GCS to a temp file, return local Path."""
        _uri = model_dir_resolved.rstrip('/') + '/' + rel
        _sf  = _Path(rel).suffix
        _tf  = _tmp.NamedTemporaryFile(suffix=_sf, delete=False)
        _tf.close()
        gcs_fs.get(_uri, _tf.name)
        return _Path(_tf.name)

    def _resolve_local(rel: str) -> _Path:
        """Return a local Path for *rel*, fetching from GCS if model_access=gcs_stream."""
        if _model_access == 'gcs_stream':
            return _gcs_fetch_to_temp(rel)
        return _local_path(rel)

    def _check_local(p: _Path, label: str):
        mo.stop(
            not p.exists(),
            mo.callout(mo.md(f"**Checkpoint not found:** `{p}`\n\n"
                             "Check `model_dir` and `checkpoint_type` in "
                             "`eval_gcs_config.yaml`."), kind="warn"),
        )

    # ---- PyTorch -----------------------------------------------------------
    if _backend_cfg == 'torch':
        import torch as _torch
        device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        from neuralop.models import FNO as _FNO

        if _ckpt_type in ('best', 'pytorch_best'):
            _rel = 'best_model_state_dict.pt'
        elif _ckpt_type in ('last', 'pytorch_last'):
            _rel = 'last_model.pt'
        else:  # checkpoint_N / pytorch_checkpoint_N
            _rel = f'checkpoint_{_epoch}/best_model_state_dict.pt'

        if _model_access == 'gcs_stream':
            _uri = model_dir_resolved.rstrip('/') + '/' + _rel
            mo.stop(gcs_fs is None,
                    mo.callout(mo.md("GCS fs required for streaming."), kind="danger"))
            with gcs_fs.open(_uri, 'rb') as _f:
                _raw = _torch.load(_f, map_location=device, weights_only=False)
        else:
            _ckpt = _local_path(_rel)
            _check_local(_ckpt, _rel)
            _raw = _torch.load(str(_ckpt), map_location=device, weights_only=False)

        if isinstance(_raw, dict) and 'model_state_dict' in _raw:
            _state = _raw['model_state_dict']
        elif isinstance(_raw, dict) and all(isinstance(v, _torch.Tensor) for v in _raw.values()):
            _state = _raw
        else:
            _state = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in _raw.items() if isinstance(v, _torch.Tensor)
            }

        model = _FNO(
            in_channels=4, out_channels=4,
            n_modes=_n_modes, hidden_channels=_hidden_channels,
            projection_channel_ratio=2,
        ).to(device)
        model.load_state_dict(_state)
        model.eval()
        backend = 'torch'
        print(f'Torch model: {sum(p.numel() for p in model.parameters()):,} params  '
              f'ckpt={_ckpt_type}')

    # ---- MLX / torch_to_mlx ------------------------------------------------
    else:
        import mlx.core as _mx
        import mlx.utils as _mlx_utils
        from neuralop_mlx.models import FNO as _FNO
        from neuralop_mlx.utils import count_model_params
        from neuralop_mlx.weight_transfer import load_from_torch
        device = None
        backend = 'mlx'

        # torch_to_mlx: force PyTorch-source checkpoint regardless of ckpt_type
        _force_pt = (_backend_cfg == 'torch_to_mlx') or _ckpt_type.startswith('pytorch_')

        if _ckpt_type in ('best', 'pytorch_best') or _backend_cfg == 'torch_to_mlx':
            _rel_st = 'best_model.safetensors'
            _rel_pt = 'best_model_state_dict.pt'
        elif _ckpt_type in ('last', 'pytorch_last'):
            _rel_st = 'last_model.safetensors'
            _rel_pt = 'last_model.pt'
        else:  # checkpoint_N / pytorch_checkpoint_N
            _rel_st = f'checkpoint_{_epoch}/best_model.safetensors'
            _rel_pt = f'checkpoint_{_epoch}/best_model_state_dict.pt'

        model = _FNO(
            in_channels=4, out_channels=4,
            n_modes=_n_modes, hidden_channels=_hidden_channels,
            projection_channel_ratio=2,
        )

        if _force_pt:
            _ckpt = _resolve_local(_rel_pt)
            _check_local(_ckpt, _rel_pt)
            print(f'MLX ← PyTorch weights: {_ckpt}')
            load_from_torch(str(_ckpt), model)
        else:
            _ckpt = _resolve_local(_rel_st)
            if not _ckpt.exists():
                # auto-fallback to PyTorch weight transfer
                print(f'No safetensors at {_rel_st} — fallback to PyTorch weight transfer')
                _ckpt = _resolve_local(_rel_pt)
                _check_local(_ckpt, _rel_pt)
                load_from_torch(str(_ckpt), model)
            else:
                _params = _mx.load(str(_ckpt))
                _nested = _mlx_utils.tree_unflatten(list(_params.items()))
                model.update(_nested)
                _mx.eval(model.parameters())
                print(f'MLX native weights: {_ckpt}')

        model.eval()
        print(f'MLX model: {count_model_params(model):,} params  ckpt={_ckpt_type}')
    return backend, device, model


# ---------------------------------------------------------------------------
# Cell: inference
# ---------------------------------------------------------------------------
@app.cell
def _(backend, data_test_norm, device, model, normalizer, np):
    if backend == 'torch':
        import torch as _torch

        def _run_inference(mdl, data, dev):
            preds = []
            with _torch.no_grad():
                for i in range(data.shape[0]):
                    x0  = data[i, :, :, :, :, :1]
                    T   = data.shape[-1] - 1
                    x   = np.repeat(x0, T, axis=-1)
                    xin = _torch.FloatTensor(x).unsqueeze(0).to(dev)
                    out = mdl(xin).squeeze(0).cpu().numpy()
                    preds.append(out)
            return np.stack(preds)  # (N, C, nx, ny, neta, T)

        model_out = _run_inference(model, data_test_norm, device)

    else:  # mlx
        import mlx.core as _mx

        def _run_inference(mdl, data):
            preds = []
            for i in range(data.shape[0]):
                x0  = data[i, :, :, :, :, :1].astype(np.float32)
                T   = data.shape[-1] - 1
                x   = np.repeat(x0, T, axis=-1)
                xin = _mx.array(x[np.newaxis])
                out = mdl(xin)
                _mx.eval(out)
                preds.append(np.array(out[0]))
            return np.stack(preds)

        model_out = _run_inference(model, data_test_norm)

    print(f'Inference done: model_out shape {model_out.shape}')

    if normalizer is not None:
        x0_all         = data_test_norm[:, :, :, :, :, :1]
        _model_full    = np.concatenate([x0_all, model_out], axis=-1)
        model_out_phys = normalizer.inverse(_model_full)[:, :, :, :, :, 1:]
        y_true_phys    = normalizer.inverse(data_test_norm)[:, :, :, :, :, 1:]
    else:
        model_out_phys = model_out
        y_true_phys    = data_test_norm[:, :, :, :, :, 1:]

    print(f'Physical shapes — pred: {model_out_phys.shape}, true: {y_true_phys.shape}')
    return model_out_phys, y_true_phys


# ---------------------------------------------------------------------------
# Cell: grid coordinates
# ---------------------------------------------------------------------------
@app.cell
def _(data_test_norm, grid, np):
    nx_eff   = data_test_norm.shape[2]
    ny_eff   = data_test_norm.shape[3]
    neta_eff = data_test_norm.shape[4]
    ntau_out = data_test_norm.shape[5] - 1

    x_min   = grid.get('x_min',   0.0);  dx   = grid.get('dx',   1.0)
    y_min   = grid.get('y_min',   0.0);  dy   = grid.get('dy',   1.0)
    eta_min = grid.get('eta_min', 0.0);  deta = grid.get('deta', 1.0)
    tau_min = grid.get('tau_min', 0.6);  dtau = grid.get('dtau', 0.1)

    x_coords       = x_min   + np.arange(nx_eff)   * dx
    y_coords       = y_min   + np.arange(ny_eff)   * dy
    eta_coords     = eta_min + np.arange(neta_eff) * deta
    tau_out_coords = tau_min + (np.arange(ntau_out) + 1) * dtau

    xy_extent = [x_coords[0] - dx / 2, x_coords[-1] + dx / 2,
                 y_coords[0] - dy / 2, y_coords[-1] + dy / 2]

    print(f'Grid  x: [{x_coords[0]:.2f}, {x_coords[-1]:.2f}] fm  '
          f'y: [{y_coords[0]:.2f}, {y_coords[-1]:.2f}] fm')
    print(f'      η: [{eta_coords[0]:.2f}, {eta_coords[-1]:.2f}]  '
          f'τ_out: [{tau_out_coords[0]:.3f}, {tau_out_coords[-1]:.3f}] fm/c')
    return (
        dtau,
        eta_coords,
        neta_eff,
        ntau_out,
        tau_min,
        tau_out_coords,
        x_coords,
        xy_extent,
        y_coords,
    )


# ---------------------------------------------------------------------------
# Cell: Marimo UI — create all widgets (no .value access here)
# ---------------------------------------------------------------------------
@app.cell
def _(data_test_norm, eval_cfg, mo, neta_eff, train_cfg):
    _norm     = train_cfg.get('normalization', {})
    norm_name = _norm.get('name', 'none') if isinstance(_norm, dict) else str(_norm)
    _n_test    = data_test_norm.shape[0]
    _default_mode = eval_cfg.get('eval_mode', 'basic')

    # --- mode selector ---
    eval_mode_ui = mo.ui.dropdown(
        ['basic', 'cont', 'metric'],
        value=_default_mode,
        label='Eval mode',
    )

    # --- basic / metric controls ---
    event_ui         = mo.ui.number(value=0, start=0, step=1,
                                     label='Event index for plots')
    channel_num_ui   = mo.ui.number(value=0, start=0, step=1,
                                     label='Channel (0=ε, 1=vy, 2=vx, 3=vη)')
    channel_drop_ui  = mo.ui.dropdown(['0 (ε)', '1 (vx)', '2 (vy)', '3 (vη)'],
                                       value='0 (ε)', label='Channel')
    tau_idx_ui       = mo.ui.text(value='5,10,15,-1',
                                   label='τ output indices to plot (comma-sep, -1=last)')
    ieta_ui          = mo.ui.number(value=0, start=0, step=1, stop=neta_eff - 1,
                                     label='η bin index')
    spec_tau_ui      = mo.ui.text(value='0,-1',
                                   label='τ indices for energy spectrum (comma-sep)')
    active_thresh_ui = mo.ui.number(value=0.3, start=0.0, step=0.05, stop=5.0,
                                     label='Active ε threshold (GeV/fm³, 0=no mask)')
    save_json_ui     = mo.ui.checkbox(value=False, label='Save summary to JSON')

    # --- cont controls ---
    cont_event_ui    = mo.ui.slider(start=0, stop=_n_test - 1, value=0, step=1,
                                     label='Event')
    cont_ieta_ui     = mo.ui.slider(start=0, stop=neta_eff - 1, value=0, step=1,
                                     label='η bin')
    cont_channel_ui  = mo.ui.dropdown(
        ['0 — ε (energy)', '1 — vx', '2 — vy', '3 — vη'],
        value='0 — ε (energy)', label='Feature / channel',
    )
    contour_iT_ui    = mo.ui.text(value='10,-1',
                                   label='2nd and 3rd τ-step indices (comma-sep; '
                                         '1st panel is always 0; -1=last non-zero)')
    cdfrats_ui       = mo.ui.text(value='0.05,0.1,0.2,0.3,0.4,0.6',
                                   label='CDF ratios for contour levels (comma-sep)')
    rescale_energy_ui = mo.ui.checkbox(
        value=(norm_name != 'none'),
        label='Rescale energy: (e/τ)^(4/3)',
    )
    return (
        active_thresh_ui,
        cdfrats_ui,
        channel_drop_ui,
        channel_num_ui,
        cont_channel_ui,
        cont_event_ui,
        cont_ieta_ui,
        contour_iT_ui,
        eval_mode_ui,
        event_ui,
        ieta_ui,
        rescale_energy_ui,
        save_json_ui,
        spec_tau_ui,
        tau_idx_ui,
        norm_name,
    )


# ---------------------------------------------------------------------------
# Cell: Marimo UI — render the mode-appropriate control panel
# ---------------------------------------------------------------------------
@app.cell
def _(
    active_thresh_ui,
    cdfrats_ui,
    channel_drop_ui,
    channel_num_ui,
    cont_channel_ui,
    cont_event_ui,
    cont_ieta_ui,
    contour_iT_ui,
    eval_mode_ui,
    event_ui,
    ieta_ui,
    mo,
    rescale_energy_ui,
    save_json_ui,
    spec_tau_ui,
    tau_idx_ui,
    norm_name,
):
    if eval_mode_ui.value == 'basic':
        _controls = mo.vstack([
            mo.hstack([eval_mode_ui]),
            mo.hstack([event_ui, ieta_ui, tau_idx_ui, channel_num_ui]),
        ])
    elif eval_mode_ui.value == 'cont':
        _controls = mo.vstack([
            mo.hstack([eval_mode_ui]),
            mo.hstack([cont_event_ui, cont_ieta_ui, cont_channel_ui]),
            mo.hstack([contour_iT_ui, cdfrats_ui]),
            mo.hstack([mo.md(f'**Normalizer:** `{norm_name}`'), rescale_energy_ui]),
        ])
    else:  # metric
        _controls = mo.vstack([
            mo.hstack([eval_mode_ui]),
            mo.hstack([channel_drop_ui, ieta_ui]),
            mo.hstack([active_thresh_ui, spec_tau_ui, save_json_ui]),
        ])
    _controls
    return


# ---------------------------------------------------------------------------
# Cell: basic plots
# ---------------------------------------------------------------------------
@app.cell
def _(
    channel_num_ui,
    eta_coords,
    eval_mode_ui,
    event_ui,
    ieta_ui,
    mo,
    model_out_phys,
    ntau_out,
    plt,
    tau_idx_ui,
    tau_out_coords,
    xy_extent,
    y_true_phys,
):
    mo.stop(eval_mode_ui.value != 'basic')

    _event   = int(event_ui.value)
    _ieta    = int(ieta_ui.value)
    _channel = int(channel_num_ui.value)

    _raw_idx     = [t.strip() for t in tau_idx_ui.value.split(',') if t.strip()]
    _tau_indices = [int(t) % ntau_out for t in _raw_idx]
    _tau_indices = [t for t in _tau_indices if 0 <= t < ntau_out]

    _n_tau_sel = len(_tau_indices)
    _ch_label  = ['e (GeV/fm³)', 'vx', 'vy', 'vη'][_channel] if _channel < 4 else f'ch{_channel}'
    _eta_phys  = eta_coords[_ieta]

    print(f'Plotting event {_event}, η bin {_ieta} (η={_eta_phys:.2f}), '
          f'channel {_channel}, τ indices: {_tau_indices}')

    _fig, _axes = plt.subplots(2, _n_tau_sel, figsize=(4.5 * _n_tau_sel, 7), squeeze=False)
    _fig.suptitle(f'η = {_eta_phys:.2f}  (event {_event})', fontsize=14, y=1.01)

    for _col, _t in enumerate(_tau_indices):
        _tau_phys = tau_out_coords[_t]
        _true = y_true_phys[_event, _channel, :, :, _ieta, _t]
        _pred = model_out_phys[_event, _channel, :, :, _ieta, _t]
        _vmin = min(_true.min(), _pred.min())
        _vmax = max(_true.max(), _pred.max())
        _kw   = dict(origin='lower', extent=xy_extent, aspect='equal',
                     vmin=_vmin, vmax=_vmax, cmap='hot')
        for _row, (_dat, _lbl) in enumerate([(y_true_phys, 'Truth'),
                                              (model_out_phys, 'FNO')]):
            _ax = _axes[_row, _col]
            _im = _ax.imshow(_dat[_event, _channel, :, :, _ieta, _t].T, **_kw)
            _ax.set_title(f'{_lbl}   τ = {_tau_phys:.3f} fm/c', fontsize=10)
            _ax.set_xlabel('x (fm)')
            _ax.set_ylabel('y (fm)')
            plt.colorbar(_im, ax=_ax, label=_ch_label, fraction=0.046, pad=0.04)
    _fig.tight_layout()
    _fig
    return


# ---------------------------------------------------------------------------
# Cell: cont plots
# ---------------------------------------------------------------------------
@app.cell
def _(
    cdfrats_ui,
    cont_channel_ui,
    cont_event_ui,
    cont_ieta_ui,
    contour_iT_ui,
    dtau,
    eta_coords,
    eval_mode_ui,
    mo,
    model_out_phys,
    plt,
    rescale_energy_ui,
    tau_min,
    x_coords,
    y_coords,
    y_true_phys,
):
    mo.stop(eval_mode_ui.value != 'cont')
    from skinnycontour import plot_three_bins_contour

    _event          = int(cont_event_ui.value)
    _channel        = int(cont_channel_ui.value.split()[0])
    _rescale_energy = bool(rescale_energy_ui.value)

    _raw_iT = [t.strip() for t in contour_iT_ui.value.split(',') if t.strip()]
    if len(_raw_iT) < 2:
        _raw_iT = (_raw_iT + ['-1', '-1'])[:2]
    elif len(_raw_iT) > 2:
        print(f'Warning: expected 2 τ indices, got {len(_raw_iT)} — using first 2.')
        _raw_iT = _raw_iT[:2]
    _contour_iT = (0,) + tuple(int(t) for t in _raw_iT)

    _cdfrats  = [float(x.strip()) for x in cdfrats_ui.value.split(',') if x.strip()]
    _tau0     = tau_min + dtau
    _ieta     = int(cont_ieta_ui.value)
    _eta_phys = eta_coords[_ieta]

    print(f'Contour plots — event {_event}, channel {_channel}, '
          f'η bin {_ieta} (η={_eta_phys:.2f}), '
          f'iT={_contour_iT}, τ0={_tau0:.3f} fm/c, Δτ={dtau:.3f} fm/c')
    print(f'CDF ratios: {_cdfrats},  rescale_energy={_rescale_energy}')

    _fig, _axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
    plot_three_bins_contour(
        model_out_phys, y_true_phys,
        iparam=_channel,
        event=_event,
        tau0=_tau0,
        taudelta=dtau,
        iT=_contour_iT,
        cdfrats=_cdfrats,
        ieta=_ieta,
        tag=f'η={_eta_phys:.2f}',
        rescale_energy=_rescale_energy,
        x_coords=x_coords,
        y_coords=y_coords,
        fig=_fig,
        axes=_axes,
    )
    mo.output.append(_fig)
    plt.close(_fig)
    return


# ---------------------------------------------------------------------------
# Cell: metric — ModelEval init
# ---------------------------------------------------------------------------
@app.cell
def _(
    active_thresh_ui,
    eta_coords,
    eval_mode_ui,
    mo,
    model_out_phys,
    tau_out_coords,
    y_true_phys,
):
    mo.stop(eval_mode_ui.value != 'metric')
    from model_eval import ModelEval

    _thresh = active_thresh_ui.value if active_thresh_ui.value > 0 else None
    evaluator = ModelEval(
        pred=model_out_phys,
        true=y_true_phys,
        tau_coords=tau_out_coords,
        eta_coords=eta_coords,
        channel_names=['ε', 'vx', 'vy', 'vη'],
        active_threshold=_thresh,
    )
    print('ModelEval ready — '
          f'N={evaluator.N}, C={evaluator.C}, '
          f'nx={evaluator.nx}, ny={evaluator.ny}, '
          f'neta={evaluator.neta}, T={evaluator.T}, '
          f'active_frac={evaluator.active_frac():.3f}')
    return (evaluator,)


# ---------------------------------------------------------------------------
# Cell: metric — dashboard + optional JSON save
# ---------------------------------------------------------------------------
@app.cell
def _(
    backend,
    channel_drop_ui,
    eval_cfg,
    evaluator,
    ieta_ui,
    mo,
    plt,
    save_json_ui,
    spec_tau_ui,
):
    import json
    from pathlib import Path as _Path

    _ch      = int(channel_drop_ui.value.split()[0])
    _ieta    = int(ieta_ui.value)
    _tau_idx = [int(t.strip()) for t in spec_tau_ui.value.split(',') if t.strip()]

    print(f'Computing metrics — channel={_ch}, η bin={_ieta} …')
    _fig = evaluator.plot_dashboard(channel=_ch, ieta=_ieta, tau_indices=_tau_idx)
    mo.output.append(_fig)
    plt.close(_fig)

    if save_json_ui.value:
        _sfx      = '_mlx' if backend == 'mlx' else ''
        _out_path = _Path(eval_cfg.get('odir', './out')) / f'metrics_eval{_sfx}.json'
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_out_path, 'w') as _f:
            json.dump(evaluator.summary(), _f, indent=2)
        print(f'Summary saved → {_out_path}')
    return


# ---------------------------------------------------------------------------
# Cell: metric — nRMSE heatmap
# ---------------------------------------------------------------------------
@app.cell
def _(backend, eta_coords, evaluator, mo, plt, tau_out_coords):
    _nrmse    = evaluator.nrmse()
    _ch_names = evaluator.channel_names
    _sfx      = ' — MLX' if backend == 'mlx' else ''

    _fig2, _axes2 = plt.subplots(1, evaluator.C, figsize=(5 * evaluator.C, 3.5))
    if evaluator.C == 1:
        _axes2 = [_axes2]
    for _c, _ax in enumerate(_axes2):
        _im = _ax.pcolormesh(
            tau_out_coords, eta_coords, _nrmse[_c],
            cmap='hot_r', shading='auto',
        )
        plt.colorbar(_im, ax=_ax, label='nRMSE')
        _ax.set_xlabel('τ (fm/c)')
        _ax.set_ylabel('η')
        _ax.set_title(f'nRMSE — {_ch_names[_c]}')
    _fig2.suptitle(f'nRMSE heat map (η × τ){_sfx}', fontsize=12)
    _fig2.tight_layout()
    mo.output.append(_fig2)
    plt.close(_fig2)
    return


# ---------------------------------------------------------------------------
# Cell: metric — fRMSE heatmap
# ---------------------------------------------------------------------------
@app.cell
def _(backend, channel_drop_ui, eta_coords, evaluator, mo, plt, tau_out_coords):
    _ch    = int(channel_drop_ui.value.split()[0])
    _frmse = evaluator.frmse()
    _sfx   = ' — MLX' if backend == 'mlx' else ''

    _fig3, _axes3 = plt.subplots(1, 3, figsize=(15, 3.5))
    for _ax, (band, cmap) in zip(_axes3,
            [('low', 'Blues'), ('mid', 'Oranges'), ('high', 'Reds')]):
        _im = _ax.pcolormesh(
            tau_out_coords, eta_coords, _frmse[band][_ch],
            cmap=cmap, shading='auto',
        )
        plt.colorbar(_im, ax=_ax, label='fRMSE')
        _ax.set_xlabel('τ (fm/c)')
        _ax.set_ylabel('η')
        _ax.set_title(f'fRMSE ({band}) — {evaluator.channel_names[_ch]}')
    _fig3.suptitle(f'Spectral RMSE heat maps (η × τ){_sfx}', fontsize=12)
    _fig3.tight_layout()
    mo.output.append(_fig3)
    plt.close(_fig3)
    return


# ---------------------------------------------------------------------------
# Cell: metric — scalar summary table
# ---------------------------------------------------------------------------
@app.cell
def _(evaluator, mo):
    _s = evaluator.summary()
    mo.md(
        '## Scalar summary\n\n'
        '| Metric | Value |\n|---|---|\n'
        + '\n'.join(f'| `{k}` | {v:.6g} |' for k, v in _s.items())
    )
    return


# ---------------------------------------------------------------------------
# Cell: basic / cont — relative L² error vs τ
# ---------------------------------------------------------------------------
@app.cell
def _(
    backend,
    channel_num_ui,
    eta_coords,
    eval_mode_ui,
    mo,
    model_out_phys,
    neta_eff,
    np,
    plt,
    tau_out_coords,
    y_true_phys,
):
    mo.stop(eval_mode_ui.value == 'metric')

    _ch    = int(channel_num_ui.value)
    _denom = np.mean(y_true_phys[:, _ch] ** 2, axis=(0, 1, 2))
    _num   = np.mean((model_out_phys[:, _ch] - y_true_phys[:, _ch]) ** 2, axis=(0, 1, 2))
    _rel_l2 = np.sqrt(_num / np.maximum(_denom, 1e-30))  # (neta, T)

    _sfx  = ' — MLX' if backend == 'mlx' else ''
    _fig4, _ax4 = plt.subplots(figsize=(8, 4))
    for _ie in range(neta_eff):
        _ax4.plot(tau_out_coords, _rel_l2[_ie], label=f'η={eta_coords[_ie]:.2f}')
    _ax4.set_xlabel('τ (fm/c)')
    _ax4.set_ylabel('Relative L² error')
    _ax4.set_title(f'Relative L² error vs τ (all η bins, test set){_sfx}')
    _ax4.legend(fontsize=7, ncol=2)
    _ax4.grid(True, alpha=0.3)
    _fig4.tight_layout()
    return


# ---------------------------------------------------------------------------
# Cell: training history plot (if available)
# ---------------------------------------------------------------------------
@app.cell
def _(backend, eval_cfg, plt):
    import parse_training_log
    from pathlib import Path as _Path

    _sfx  = '_mlx' if backend == 'mlx' else ''
    _odir = eval_cfg.get('odir', './out')
    _p    = _Path(_odir) / f'metrics{_sfx}.json'
    if _p.exists():
        _hist       = parse_training_log.TrainingHistory.load(str(_p))
        _back_label = 'MLX' if backend == 'mlx' else 'PyTorch'
        _hist.plot(title=f'FNO Training ({_back_label})')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No {_p.name} found at {_p}')
    return


if __name__ == '__main__':
    app.run()
