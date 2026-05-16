import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # FNO Model Evaluation — Unified

    Set the **checkpoint directory** and **training config YAML** explicitly.
    Model architecture and data-prep parameters are read from the training config;
    checkpoints are loaded from the checkpoint directory.

    | Mode | Description |
    |---|---|
    | `basic`  | Truth/prediction image grids per η bin at selected τ steps |
    | `cont`   | CDF-based skinny contour overlays (3 τ snapshots per η bin) |
    | `metric` | Full metric suite: nRMSE, fRMSE, MELR, WLR, dashboard + JSON |

    *conda env: `fno_env_mlx` (both backends)*
    """)
    return


# ── Common UI ────────────────────────────────────────────────────────────────

@app.cell
def _(mo):
    model_dir_ui   = mo.ui.text(value="./checkpoints",
                                 label="Checkpoint directory")
    train_cfg_ui   = mo.ui.text(value="./checkpoints/config.yaml",
                                 label="Training config YAML")
    backend_ui     = mo.ui.dropdown(["torch", "mlx"],
                                     value="torch",
                                     label="Backend  (mlx enables PyTorch→MLX weight transfer)")
    eval_mode_ui   = mo.ui.dropdown(["basic", "cont", "metric"],
                                     value="basic",
                                     label="Eval mode")
    ckpt_type_ui   = mo.ui.dropdown(
        ["best", "last", "checkpoint_N",
         "pytorch_best", "pytorch_last", "pytorch_checkpoint_N"],
        value="best",
        label="Checkpoint  (pytorch_* → weight transfer from PyTorch .pt into MLX model)",
    )
    ckpt_epoch_ui  = mo.ui.number(value=10, start=1, step=1, label="Epoch N (checkpoint_N only)")
    eval_file_ui   = mo.ui.text(
        value="", label="Separate eval file (empty → use training files + split)"
    )
    nevents_ui     = mo.ui.number(value=-1, step=1, label="Max events to load (-1 = all)")
    mo.vstack([
        mo.hstack([model_dir_ui, train_cfg_ui]),
        mo.hstack([backend_ui, eval_mode_ui]),
        mo.hstack([ckpt_type_ui, ckpt_epoch_ui]),
        mo.hstack([eval_file_ui, nevents_ui]),
    ])
    return (model_dir_ui, train_cfg_ui, backend_ui, eval_mode_ui, ckpt_type_ui,
            ckpt_epoch_ui, eval_file_ui, nevents_ui)


# ── Data loading ──────────────────────────────────────────────────────────────

@app.cell
def _(mo, model_dir_ui, train_cfg_ui, backend_ui, eval_mode_ui, eval_file_ui, nevents_ui):
    import sys, gc
    sys.path.insert(0, '../loc_libs')

    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    from read_3d_root import read_3d_data
    from norm_utils import Normalizer
    import parse_training_log

    backend   = backend_ui.value    # torch | mlx
    eval_mode = eval_mode_ui.value  # basic | cont | metric

    # --- training config: explicit path from UI ---
    _train_cfg_path = Path(train_cfg_ui.value)
    mo.stop(
        not _train_cfg_path.exists(),
        mo.callout(
            mo.md(f"**Training config not found:** `{_train_cfg_path}`\n\n"
                  "Set a valid path in the **Training config YAML** field above."),
            kind="warn",
        ),
    )
    with open(_train_cfg_path) as _f:
        train_cfg = yaml.safe_load(_f)
    print(f'Training config: {_train_cfg_path}')

    if backend == 'torch':
        from neuralop.models import FNO
    else:
        from neuralop_mlx.models import FNO

    print(f'backend: {backend},  eval_mode: {eval_mode}')

    def _s(v):
        return slice(None) if v is None else slice(*v)

    input_dir       = train_cfg.get('input_dir', '.')
    input_files     = train_cfg['input_files']
    train_ratio     = train_cfg.get('train_ratio', 0.8)
    xyslice         = _s(train_cfg.get('xyslice',  [None, -1]))
    etaslice        = _s(train_cfg.get('tauslice', [None, -1]))
    hidden_channels = train_cfg['hidden_channels']
    n_modes         = train_cfg['n_modes']
    odir            = train_cfg.get('odir', './out')
    nevents_load    = int(nevents_ui.value)

    eval_str = eval_file_ui.value.strip()
    if eval_str:
        data_file       = eval_str.split(',')[0].strip()
        use_train_split = False
    else:
        data_file       = f'{input_dir}/{input_files[0]}'
        use_train_split = True

    _data_path = Path(data_file) if not data_file.startswith('gs://') else None
    mo.stop(
        _data_path is not None and not _data_path.exists(),
        mo.callout(
            mo.md(f"**Data file not found:** `{data_file}`\n\n"
                  "Check `input_dir` / `input_files` in the training config, "
                  "or set a path in **Separate eval file**."),
            kind="warn",
        ),
    )
    print(f'Loading data: {data_file}')
    raw = read_3d_data(data_file, nevents=nevents_load)

    grid = {}
    for _k in ('tau_min', 'dtau', 'eta_min', 'deta', 'x_min', 'dx', 'y_min', 'dy'):
        if _k in raw:
            grid[_k] = raw[_k]
    print('Grid params:', {k: f'{v:.4g}' for k, v in grid.items()})

    arr = raw['arr']
    arr = arr[:, :, xyslice, xyslice, etaslice, :]
    print(f'Array shape after slicing: {arr.shape}')

    n_train = int(arr.shape[0] * train_ratio)
    if use_train_split:
        data_test = arr[n_train:]
        print(f'Train/test split at {n_train}: using {data_test.shape[0]} test events')
    else:
        data_test = arr
        print(f'Using all {data_test.shape[0]} events from eval file')

    del arr; gc.collect()
    return (train_cfg, grid, data_test, hidden_channels, n_modes, odir,
            backend, eval_mode,
            np, plt, Path, Normalizer, FNO, parse_training_log)


# ── Normalizer ────────────────────────────────────────────────────────────────

@app.cell
def _(model_dir_ui, data_test, Normalizer, Path, np):
    model_dir = Path(model_dir_ui.value)
    norm_path = model_dir / 'normalizer.npz'

    if norm_path.exists():
        normalizer     = Normalizer.load(str(norm_path))
        data_test_norm = normalizer.transform(data_test)
        print(f'Normalizer: {normalizer}')
    else:
        normalizer     = None
        data_test_norm = data_test
        print('No normalizer.npz found — using raw data')

    return (model_dir, normalizer, data_test_norm)


# ── Model loading ─────────────────────────────────────────────────────────────

@app.cell
def _(mo, model_dir, ckpt_type_ui, ckpt_epoch_ui,
      hidden_channels, n_modes, FNO, backend):
    ckpt_type = ckpt_type_ui.value
    _epoch    = int(ckpt_epoch_ui.value)

    if backend == 'torch':
        import torch as _torch
        device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')

        if ckpt_type in ('best', 'pytorch_best'):
            ckpt_path = model_dir / 'best_model_state_dict.pt'
        elif ckpt_type in ('last', 'pytorch_last'):
            ckpt_path = model_dir / 'last_model.pt'
        else:  # checkpoint_N / pytorch_checkpoint_N
            ckpt_path = model_dir / f'checkpoint_{_epoch}' / 'best_model_state_dict.pt'

        print(f'Checkpoint: {ckpt_path}')
        mo.stop(
            not ckpt_path.exists(),
            mo.callout(mo.md(f"**Checkpoint not found:** `{ckpt_path}`\n\n"
                             "Check the **Checkpoint directory** and **Checkpoint** type fields."),
                       kind="warn"),
        )

        _raw = _torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(_raw, dict) and 'model_state_dict' in _raw:
            state_dict = _raw['model_state_dict']
        elif isinstance(_raw, dict) and all(isinstance(v, _torch.Tensor) for v in _raw.values()):
            state_dict = _raw
        else:
            state_dict = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in _raw.items() if isinstance(v, _torch.Tensor)
            }

        model = FNO(
            in_channels=4, out_channels=4,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=2,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f'Model: {n_params:,} parameters, checkpoint: {ckpt_type}')

    else:  # mlx
        import mlx.core as _mx
        import mlx.utils as _mlx_utils
        from neuralop_mlx.utils import count_model_params
        from neuralop_mlx.weight_transfer import load_from_torch
        device = None
        _is_pt = ckpt_type.startswith('pytorch_')

        if ckpt_type == 'best':
            ckpt_path    = model_dir / 'best_model.safetensors'
            _pt_fallback = model_dir / 'best_model_state_dict.pt'
        elif ckpt_type == 'last':
            ckpt_path    = model_dir / 'last_model.safetensors'
            _pt_fallback = model_dir / 'last_model.pt'
        elif ckpt_type == 'checkpoint_N':
            ckpt_path    = model_dir / f'checkpoint_{_epoch}' / 'best_model.safetensors'
            _pt_fallback = model_dir / f'checkpoint_{_epoch}' / 'best_model_state_dict.pt'
        elif ckpt_type == 'pytorch_best':
            ckpt_path    = model_dir / 'best_model_state_dict.pt'
            _pt_fallback = None
        elif ckpt_type == 'pytorch_last':
            ckpt_path    = model_dir / 'last_model.pt'
            _pt_fallback = None
        else:  # pytorch_checkpoint_N
            ckpt_path    = model_dir / f'checkpoint_{_epoch}' / 'best_model_state_dict.pt'
            _pt_fallback = None

        if not _is_pt and not ckpt_path.exists() and _pt_fallback is not None and _pt_fallback.exists():
            print(f'No safetensors at {ckpt_path.name} — auto-fallback to PyTorch weight transfer')
            ckpt_path = _pt_fallback
            _is_pt = True

        print(f'Checkpoint: {ckpt_path}  ({"PyTorch→MLX transfer" if _is_pt else "MLX native"})')
        mo.stop(
            not ckpt_path.exists(),
            mo.callout(mo.md(f"**Checkpoint not found:** `{ckpt_path}`\n\n"
                             "Check the **Checkpoint directory** and **Checkpoint** type fields."),
                       kind="warn"),
        )

        model = FNO(
            in_channels=4, out_channels=4,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channel_ratio=2,
        )

        if _is_pt:
            load_from_torch(str(ckpt_path), model)
        else:
            _params = _mx.load(str(ckpt_path))
            _nested = _mlx_utils.tree_unflatten(list(_params.items()))
            model.update(_nested)
            _mx.eval(model.parameters())

        model.eval()
        print(f'Model: {count_model_params(model):,} parameters, checkpoint: {ckpt_type}')

    return (model, device)


# ── Inference ─────────────────────────────────────────────────────────────────

@app.cell
def _(model, device, data_test_norm, normalizer, backend, np):
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
            return np.stack(preds)   # (N, C, nx, ny, neta, T)

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
            return np.stack(preds)   # (N, C, nx, ny, neta, T)

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
    return (model_out, model_out_phys, y_true_phys)


# ── Grid coordinates ──────────────────────────────────────────────────────────

@app.cell
def _(grid, data_test_norm, np):
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

    xy_extent = [x_coords[0] - dx/2, x_coords[-1] + dx/2,
                 y_coords[0] - dy/2, y_coords[-1] + dy/2]

    print(f'Grid  x: [{x_coords[0]:.2f}, {x_coords[-1]:.2f}] fm  '
          f'y: [{y_coords[0]:.2f}, {y_coords[-1]:.2f}] fm')
    print(f'      η: [{eta_coords[0]:.2f}, {eta_coords[-1]:.2f}]  '
          f'τ_out: [{tau_out_coords[0]:.3f}, {tau_out_coords[-1]:.3f}] fm/c')

    return (nx_eff, ny_eff, neta_eff, ntau_out, dtau, tau_min,
            x_coords, y_coords, eta_coords, tau_out_coords, xy_extent)


# ── Mode-specific controls ────────────────────────────────────────────────────

@app.cell
def _(mo, neta_eff, data_test_norm, train_cfg, eval_mode):
    _norm      = train_cfg.get('normalization', {})
    _norm_name = _norm.get('name', 'none') if isinstance(_norm, dict) else str(_norm)
    _n_test    = data_test_norm.shape[0]

    # basic / metric controls
    event_ui          = mo.ui.number(value=0, start=0, step=1,
                                      label="Event index for plots")
    channel_num_ui    = mo.ui.number(value=0, start=0, step=1,
                                      label="Channel (0=energy, 1=vy, 2=vx, 3=vz)")
    channel_drop_ui   = mo.ui.dropdown(['0 (ε)', '1 (vx)', '2 (vy)', '3 (vη)'],
                                        value='0 (ε)', label='Channel')
    tau_idx_ui        = mo.ui.text(value="5,10,15,-1",
                                    label="τ output indices to plot (comma-sep, -1=last)")
    ieta_ui           = mo.ui.number(value=0, start=0, step=1, stop=neta_eff - 1,
                                      label='η bin index')
    spec_tau_ui       = mo.ui.text(value='0,-1',
                                    label='τ indices for energy spectrum (comma-sep)')
    active_thresh_ui  = mo.ui.number(value=0.3, start=0.0, step=0.05, stop=5.0,
                                      label='Active ε threshold (GeV/fm³, 0 = no mask)')
    save_json_ui      = mo.ui.checkbox(value=False, label='Save summary to JSON')

    # cont-specific controls (sliders for quick scrubbing)
    cont_event_ui     = mo.ui.slider(start=0, stop=_n_test - 1, value=0, step=1,
                                      label="Event")
    cont_ieta_ui      = mo.ui.slider(start=0, stop=neta_eff - 1, value=0, step=1,
                                      label="η bin")
    cont_channel_ui   = mo.ui.dropdown(['0 — ε (energy)', '1 — vx', '2 — vy', '3 — vη'],
                                        value='0 — ε (energy)', label='Feature / channel')
    contour_iT_ui     = mo.ui.text(value="10,-1",
                                    label="2nd and 3rd τ-step indices (comma-sep; 1st panel is always 0; -1=last non-zero)")
    cdfrats_ui        = mo.ui.text(value="0.05,0.1,0.2,0.3,0.4,0.6",
                                    label="CDF ratios for contour levels (comma-sep)")
    rescale_energy_ui = mo.ui.checkbox(value=(_norm_name != 'none'),
                                        label="Rescale energy: (e/τ)^(4/3)")

    if eval_mode == 'basic':
        _controls = mo.vstack([
            mo.hstack([event_ui, ieta_ui, tau_idx_ui, channel_num_ui]),
        ])
    elif eval_mode == 'cont':
        _controls = mo.vstack([
            mo.hstack([cont_event_ui, cont_ieta_ui, cont_channel_ui]),
            mo.hstack([contour_iT_ui, cdfrats_ui]),
            mo.hstack([mo.md(f"**Normalizer:** `{_norm_name}`"), rescale_energy_ui]),
        ])
    else:  # metric
        _controls = mo.vstack([
            mo.hstack([channel_drop_ui, ieta_ui]),
            mo.hstack([active_thresh_ui, spec_tau_ui, save_json_ui]),
        ])
    _controls

    return (event_ui, channel_num_ui, channel_drop_ui, tau_idx_ui, ieta_ui,
            cont_event_ui, cont_ieta_ui, cont_channel_ui,
            contour_iT_ui, cdfrats_ui, rescale_energy_ui,
            spec_tau_ui, active_thresh_ui, save_json_ui)


# ── Basic: truth/prediction image grids per η bin ────────────────────────────

@app.cell
def _(model_out_phys, y_true_phys, eta_coords, tau_out_coords, xy_extent,
      ntau_out, eval_mode,
      event_ui, ieta_ui, tau_idx_ui, channel_num_ui, plt, mo):
    mo.stop(eval_mode != 'basic')

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


# ── Cont: CDF-based skinny contour plots ─────────────────────────────────────

@app.cell
def _(model_out_phys, y_true_phys, eta_coords, x_coords, y_coords,
      tau_min, dtau, eval_mode,
      cont_event_ui, cont_ieta_ui, cont_channel_ui,
      contour_iT_ui, cdfrats_ui, rescale_energy_ui,
      np, plt, mo):
    mo.stop(eval_mode != 'cont')
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


# ── Metric: ModelEval initialisation ─────────────────────────────────────────

@app.cell
def _(model_out_phys, y_true_phys, tau_out_coords, eta_coords,
      eval_mode, active_thresh_ui, mo):
    mo.stop(eval_mode != 'metric')
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
    print('ModelEval ready —',
          f'N={evaluator.N}, C={evaluator.C}, '
          f'nx={evaluator.nx}, ny={evaluator.ny}, '
          f'neta={evaluator.neta}, T={evaluator.T}, '
          f'active_frac={evaluator.active_frac():.3f}')
    return (evaluator,)


# ── Metric: dashboard ─────────────────────────────────────────────────────────

@app.cell
def _(evaluator, channel_drop_ui, ieta_ui, spec_tau_ui,
      save_json_ui, odir, backend, mo, plt, Path):
    import json

    _ch      = int(channel_drop_ui.value.split()[0])
    _ieta    = int(ieta_ui.value)
    _tau_idx = [int(t.strip()) for t in spec_tau_ui.value.split(',') if t.strip()]

    print(f'Computing metrics — channel={_ch}, η bin={_ieta} ...')
    _fig = evaluator.plot_dashboard(channel=_ch, ieta=_ieta, tau_indices=_tau_idx)
    mo.output.append(_fig)
    plt.close(_fig)

    if save_json_ui.value:
        _sfx      = '_mlx' if backend == 'mlx' else ''
        _out_path = Path(odir) / f'metrics_eval{_sfx}.json'
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_out_path, 'w') as _f:
            json.dump(evaluator.summary(), _f, indent=2)
        print(f'Summary saved → {_out_path}')
    return


# ── Metric: per-channel nRMSE heatmap ────────────────────────────────────────

@app.cell
def _(evaluator, tau_out_coords, eta_coords, backend, plt, mo):
    _nrmse    = evaluator.nrmse()
    _ch_names = evaluator.channel_names
    _sfx      = ' — MLX' if backend == 'mlx' else ''

    _fig2, _axes2 = plt.subplots(1, evaluator.C, figsize=(5 * evaluator.C, 3.5))
    if evaluator.C == 1:
        _axes2 = [_axes2]
    for _c, _ax in enumerate(_axes2):
        _im = _ax.pcolormesh(
            tau_out_coords, eta_coords, _nrmse[_c],
            cmap='hot_r', shading='auto'
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


# ── Metric: fRMSE heatmaps ────────────────────────────────────────────────────

@app.cell
def _(evaluator, tau_out_coords, eta_coords, channel_drop_ui, backend, plt, mo):
    _ch    = int(channel_drop_ui.value.split()[0])
    _frmse = evaluator.frmse()
    _sfx   = ' — MLX' if backend == 'mlx' else ''

    _fig3, _axes3 = plt.subplots(1, 3, figsize=(15, 3.5))
    for _ax, (band, cmap) in zip(_axes3,
            [('low', 'Blues'), ('mid', 'Oranges'), ('high', 'Reds')]):
        _im = _ax.pcolormesh(
            tau_out_coords, eta_coords, _frmse[band][_ch],
            cmap=cmap, shading='auto'
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


# ── Metric: scalar summary table ─────────────────────────────────────────────

@app.cell
def _(evaluator, mo):
    _s = evaluator.summary()
    mo.md(
        '## Scalar summary\n\n'
        '| Metric | Value |\n|---|---|\n'
        + '\n'.join(f'| `{k}` | {v:.6g} |' for k, v in _s.items())
    )


# ── Relative L² error (basic + cont) ─────────────────────────────────────────

@app.cell
def _(model_out_phys, y_true_phys, eta_coords, tau_out_coords,
      neta_eff, channel_num_ui, backend, eval_mode, np, plt, mo):
    mo.stop(eval_mode == 'metric')

    _ch     = int(channel_num_ui.value)
    _denom  = np.mean(y_true_phys[:, _ch] ** 2, axis=(0, 1, 2))
    _num    = np.mean((model_out_phys[:, _ch] - y_true_phys[:, _ch]) ** 2, axis=(0, 1, 2))
    _rel_l2 = np.sqrt(_num / np.maximum(_denom, 1e-30))   # (neta, T)

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
    return _fig4


# ── Training history ──────────────────────────────────────────────────────────

@app.cell
def _(odir, backend, parse_training_log, plt, Path):
    _sfx = '_mlx' if backend == 'mlx' else ''
    _p   = Path(odir) / f'metrics{_sfx}.json'
    if _p.exists():
        _hist = parse_training_log.TrainingHistory.load(str(_p))
        _back_label = 'MLX' if backend == 'mlx' else 'PyTorch'
        _hist.plot(title=f'FNO Training ({_back_label})')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No {_p.name} found at {_p}')
    return


if __name__ == "__main__":
    app.run()
