import marimo

# conda env: fno_env_mlx
__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # FNO Model Evaluation — MLX

    Evaluates a trained MLX FNO checkpoint on the test split (or a separate eval file).
    Contour plots are shown for every η bin at user-selected τ output steps.
    Axes are in physical units: x/y in fm, η dimensionless, τ in fm/c.

    *conda env: `fno_env_mlx`*
    """)
    return


@app.cell
def _(mo):
    cfg_path_ui   = mo.ui.text(value="./config_test_train.yaml", label="Config YAML")
    model_dir_ui  = mo.ui.text(value="./checkpoints",            label="Model directory")
    ckpt_type_ui  = mo.ui.dropdown(
        ["best", "last", "checkpoint_N",
         "pytorch_best", "pytorch_last", "pytorch_checkpoint_N"],
        value="best",
        label="Checkpoint  (pytorch_* → weight transfer from PyTorch .pt file)",
    )
    ckpt_epoch_ui = mo.ui.number(value=10, start=1, step=1, label="Epoch N (checkpoint_N only)")
    eval_file_ui  = mo.ui.text(
        value="", label="Separate eval file (empty → use training files + split)"
    )
    nevents_ui    = mo.ui.number(value=-1, step=1, label="Max events to load (-1 = all)")
    event_ui      = mo.ui.number(value=0, start=0, step=1, label="Event index for plots")
    tau_idx_ui    = mo.ui.text(value="5,10,15,-1", label="τ output indices to plot (comma-sep, -1=last)")
    channel_ui    = mo.ui.number(value=0, start=0, step=1, label="Channel (0 = energy density)")
    mo.vstack([
        mo.hstack([cfg_path_ui, model_dir_ui]),
        mo.hstack([ckpt_type_ui, ckpt_epoch_ui]),
        mo.hstack([eval_file_ui, nevents_ui]),
        mo.hstack([event_ui, tau_idx_ui, channel_ui]),
    ])
    return (cfg_path_ui, model_dir_ui, ckpt_type_ui, ckpt_epoch_ui,
            eval_file_ui, nevents_ui, event_ui, tau_idx_ui, channel_ui)


@app.cell
def _(cfg_path_ui, eval_file_ui, nevents_ui):
    import sys, gc, os
    sys.path.append('../loc_libs')

    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    import mlx.core as mx
    import mlx.utils as mlx_utils
    from pathlib import Path

    from read_3d_root import read_3d_data
    from norm_utils import Normalizer
    from neuralop_mlx.models import FNO
    import parse_training_log

    with open(cfg_path_ui.value) as _f:
        cfg = yaml.safe_load(_f)

    def _s(v):
        return slice(None) if v is None else slice(*v)

    input_dir    = cfg['input_dir']
    input_files  = cfg['input_files']
    train_ratio  = cfg.get('train_ratio', 0.8)
    xyslice      = _s(cfg.get('xyslice',  [None, -1]))
    etaslice     = _s(cfg.get('tauslice', [None, -1]))   # 'tauslice' in YAML slices the eta dim
    hidden_channels = cfg['hidden_channels']
    n_modes      = cfg['n_modes']
    odir         = cfg.get('odir', './out')
    nevents_load = int(nevents_ui.value)

    # resolve data file
    eval_str = eval_file_ui.value.strip()
    if eval_str:
        data_file       = eval_str.split(',')[0].strip()
        use_train_split = False
    else:
        data_file       = f'{input_dir}/{input_files[0]}'
        use_train_split = True

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
    return (cfg, grid, data_test, hidden_channels, n_modes, odir,
            use_train_split, sys, np, plt, mx, mlx_utils, Path,
            Normalizer, FNO, parse_training_log)


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


@app.cell
def _(model_dir, ckpt_type_ui, ckpt_epoch_ui,
       hidden_channels, n_modes, mx, mlx_utils, FNO):
    from neuralop_mlx.utils import count_model_params
    from neuralop_mlx.weight_transfer import load_from_torch

    ckpt_type = ckpt_type_ui.value
    _epoch    = int(ckpt_epoch_ui.value)
    _is_pt    = ckpt_type.startswith('pytorch_')

    if ckpt_type == 'best':
        ckpt_path = model_dir / 'best_model.safetensors'
    elif ckpt_type == 'last':
        ckpt_path = model_dir / 'last_model.safetensors'
    elif ckpt_type == 'checkpoint_N':
        ckpt_path = model_dir / f'checkpoint_{_epoch}' / 'best_model.safetensors'
    elif ckpt_type == 'pytorch_best':
        ckpt_path = model_dir / 'best_model_state_dict.pt'
    elif ckpt_type == 'pytorch_last':
        ckpt_path = model_dir / 'last_model.pt'
    else:  # pytorch_checkpoint_N
        ckpt_path = model_dir / f'checkpoint_{_epoch}' / 'best_model_state_dict.pt'

    print(f'Checkpoint: {ckpt_path}  ({"PyTorch→MLX transfer" if _is_pt else "MLX native"})')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model = FNO(
        in_channels=4, out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
    )

    if _is_pt:
        # weight transfer: load PyTorch state dict and map weights into MLX model
        load_from_torch(str(ckpt_path), model)
    else:
        # native MLX: flat safetensors saved by MLX trainer / save_last_checkpoint
        params = mx.load(str(ckpt_path))
        nested = mlx_utils.tree_unflatten(list(params.items()))
        model.update(nested)
        mx.eval(model.parameters())

    model.eval()
    print(f'Model: {count_model_params(model):,} parameters, checkpoint: {ckpt_type}')

    return (model,)


@app.cell
def _(model, data_test_norm, normalizer, np, mx):
    def _run_inference(mdl, data):
        """Single-pass inference: predict all τ steps from the initial condition."""
        preds = []
        for i in range(data.shape[0]):
            x0  = data[i, :, :, :, :, :1].astype(np.float32)  # (C, nx, ny, neta, 1)
            T   = data.shape[-1] - 1
            x   = np.repeat(x0, T, axis=-1)                     # (C, nx, ny, neta, T)
            xin = mx.array(x[np.newaxis])                        # (1, C, nx, ny, neta, T)
            out = mdl(xin)
            mx.eval(out)
            preds.append(np.array(out[0]))
        return np.stack(preds)   # (N, C, nx, ny, neta, T)

    model_out = _run_inference(model, data_test_norm)
    print(f'Inference done: model_out shape {model_out.shape}')

    if normalizer is not None:
        x0_all = data_test_norm[:, :, :, :, :, :1]
        _model_full    = np.concatenate([x0_all, model_out], axis=-1)
        model_out_phys = normalizer.inverse(_model_full)[:, :, :, :, :, 1:]
        y_true_phys    = normalizer.inverse(data_test_norm)[:, :, :, :, :, 1:]
    else:
        model_out_phys = model_out
        y_true_phys    = data_test_norm[:, :, :, :, :, 1:]

    print(f'Physical shapes — pred: {model_out_phys.shape}, true: {y_true_phys.shape}')
    return (model_out, model_out_phys, y_true_phys)


@app.cell
def _(grid, data_test_norm, np):
    nx_eff   = data_test_norm.shape[2]
    ny_eff   = data_test_norm.shape[3]
    neta_eff = data_test_norm.shape[4]
    ntau_out = data_test_norm.shape[5] - 1

    x_min  = grid.get('x_min',   0.0);  dx   = grid.get('dx',   1.0)
    y_min  = grid.get('y_min',   0.0);  dy   = grid.get('dy',   1.0)
    eta_min = grid.get('eta_min', 0.0); deta = grid.get('deta',  1.0)
    tau_min = grid.get('tau_min', 0.6); dtau = grid.get('dtau',  0.1)

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

    return (nx_eff, ny_eff, neta_eff, ntau_out,
            x_coords, y_coords, eta_coords, tau_out_coords, xy_extent)


@app.cell
def _(model_out_phys, y_true_phys, eta_coords, tau_out_coords, xy_extent,
       ntau_out, neta_eff, event_ui, tau_idx_ui, channel_ui, np, plt, mo):
    event   = int(event_ui.value)
    channel = int(channel_ui.value)

    _raw_idx    = [t.strip() for t in tau_idx_ui.value.split(',') if t.strip()]
    tau_indices = [int(t) % ntau_out for t in _raw_idx]
    tau_indices = [t for t in tau_indices if 0 <= t < ntau_out]
    print(f'Plotting event {event}, channel {channel}, tau indices: {tau_indices}')

    n_tau_sel = len(tau_indices)
    ch_label  = ['e (GeV/fm³)', 'vx', 'vy', 'vη'][channel] if channel < 4 else f'ch{channel}'

    for _ieta in range(neta_eff):
        eta_phys = eta_coords[_ieta]

        _fig, _axes = plt.subplots(2, n_tau_sel, figsize=(4.5 * n_tau_sel, 7),
                                   squeeze=False)
        _fig.suptitle(f'η = {eta_phys:.2f}  (event {event})', fontsize=14, y=1.01)

        for _col, _t in enumerate(tau_indices):
            tau_phys = tau_out_coords[_t]
            _true = y_true_phys[event, channel, :, :, _ieta, _t]
            _pred = model_out_phys[event, channel, :, :, _ieta, _t]
            _vmin = min(_true.min(), _pred.min())
            _vmax = max(_true.max(), _pred.max())
            _kw   = dict(origin='lower', extent=xy_extent, aspect='equal',
                         vmin=_vmin, vmax=_vmax, cmap='hot')

            for _row, (_dat, _lbl) in enumerate([(y_true_phys, 'Truth'), (model_out_phys, 'FNO')]):
                _ax = _axes[_row, _col]
                _d  = _dat[event, channel, :, :, _ieta, _t]
                _im = _ax.imshow(_d.T, **_kw)
                _ax.set_title(f'{_lbl}   τ = {tau_phys:.3f} fm/c', fontsize=10)
                _ax.set_xlabel('x (fm)')
                _ax.set_ylabel('y (fm)')
                plt.colorbar(_im, ax=_ax, label=ch_label, fraction=0.046, pad=0.04)

        _fig.tight_layout()
        mo.output.append(_fig)
        plt.close(_fig)

    return


@app.cell
def _(model_out_phys, y_true_phys, eta_coords, tau_out_coords,
      neta_eff, channel_ui, np, plt):
    _ch = int(channel_ui.value)
    # average over events (0), x (1), y (2); keep neta (3) and tau (4)
    _denom  = np.mean(y_true_phys[:, _ch] ** 2, axis=(0, 1, 2))                          # (neta, T)
    _num    = np.mean((model_out_phys[:, _ch] - y_true_phys[:, _ch]) ** 2, axis=(0, 1, 2))
    _rel_l2 = np.sqrt(_num / np.maximum(_denom, 1e-30))                                   # (neta, T)

    _fig2, _ax2 = plt.subplots(figsize=(8, 4))
    for _ie in range(neta_eff):
        _ax2.plot(tau_out_coords, _rel_l2[_ie], label=f'η={eta_coords[_ie]:.2f}')
    _ax2.set_xlabel('τ (fm/c)')
    _ax2.set_ylabel('Relative L² error')
    _ax2.set_title('Relative L² error vs τ (all η bins, test set) — MLX')
    _ax2.legend(fontsize=7, ncol=2)
    _ax2.grid(True, alpha=0.3)
    _fig2.tight_layout()
    return _fig2


@app.cell
def _(odir, parse_training_log, plt, Path):
    import json
    _metrics_path = Path(odir) / 'metrics_mlx.json'
    if _metrics_path.exists():
        history = parse_training_log.TrainingHistory.load(str(_metrics_path))
        history.plot(title='FNO Training (MLX)')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No metrics_mlx.json found at {_metrics_path}')
    return


if __name__ == "__main__":
    app.run()
