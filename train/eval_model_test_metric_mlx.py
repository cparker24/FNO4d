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
    # FNO Model Evaluation — MLX · Full Metric Suite

    Computes and visualises the complete evaluation battery described in the PDEBench /
    heavy-ion surrogate literature:

    | Category | Metrics |
    |---|---|
    | Pointwise | nRMSE, RMSE, MaxError, bRMSE, cRMSE, vRMSE |
    | Spectral  | fRMSE(low/mid/high), MELR, WLR |
    | Spectral  | Spectral error map, Energy spectrum |
    | Dashboard | 2×3 composite figure + JSON summary |

    Supports both native MLX `.safetensors` checkpoints and PyTorch→MLX weight transfer.

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
    mo.vstack([
        mo.hstack([cfg_path_ui, model_dir_ui]),
        mo.hstack([ckpt_type_ui, ckpt_epoch_ui]),
        mo.hstack([eval_file_ui, nevents_ui]),
    ])
    return (cfg_path_ui, model_dir_ui, ckpt_type_ui, ckpt_epoch_ui,
            eval_file_ui, nevents_ui)


@app.cell
def _(cfg_path_ui, eval_file_ui, nevents_ui):
    import sys, gc
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

    input_dir       = cfg['input_dir']
    input_files     = cfg['input_files']
    train_ratio     = cfg.get('train_ratio', 0.8)
    xyslice         = _s(cfg.get('xyslice',  [None, -1]))
    etaslice        = _s(cfg.get('tauslice', [None, -1]))
    hidden_channels = cfg['hidden_channels']
    n_modes         = cfg['n_modes']
    odir            = cfg.get('odir', './out')
    nevents_load    = int(nevents_ui.value)

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
        print(f'Using {data_test.shape[0]} test events (train/test split at {n_train})')
    else:
        data_test = arr
        print(f'Using all {data_test.shape[0]} events from eval file')

    del arr; gc.collect()
    return (cfg, grid, data_test, hidden_channels, n_modes, odir,
            sys, np, plt, mx, mlx_utils, Path, Normalizer, FNO, parse_training_log)


@app.cell
def _(model_dir_ui, data_test, Normalizer, Path, np):
    model_dir = Path(model_dir_ui.value)
    norm_path = model_dir / 'normalizer.npz'

    if norm_path.exists():
        normalizer     = Normalizer.load(str(norm_path))
        data_test_norm = normalizer.transform(data_test)
        print(f'Normalizer loaded: {normalizer}')
    else:
        normalizer     = None
        data_test_norm = data_test
        print('No normalizer.npz — using raw data')

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
    else:   # pytorch_checkpoint_N
        ckpt_path = model_dir / f'checkpoint_{_epoch}' / 'best_model_state_dict.pt'

    print(f'Checkpoint: {ckpt_path}  ({"PyTorch→MLX" if _is_pt else "MLX native"})')
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model = FNO(
        in_channels=4, out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        projection_channel_ratio=2,
    )

    if _is_pt:
        load_from_torch(str(ckpt_path), model)
    else:
        _params = mx.load(str(ckpt_path))
        _nested = mlx_utils.tree_unflatten(list(_params.items()))
        model.update(_nested)
        mx.eval(model.parameters())

    model.eval()
    print(f'Model: {count_model_params(model):,} params, ckpt: {ckpt_type}')

    return (model,)


@app.cell
def _(model, data_test_norm, normalizer, np, mx):
    def _run_inference(mdl, data):
        preds = []
        for i in range(data.shape[0]):
            x0  = data[i, :, :, :, :, :1].astype(np.float32)
            T   = data.shape[-1] - 1
            x   = np.repeat(x0, T, axis=-1)
            xin = mx.array(x[np.newaxis])
            out = mdl(xin)
            mx.eval(out)
            preds.append(np.array(out[0]))
        return np.stack(preds)   # (N, C, nx, ny, neta, T)

    model_out = _run_inference(model, data_test_norm)
    print(f'Inference done: {model_out.shape}')

    if normalizer is not None:
        x0_all         = data_test_norm[:, :, :, :, :, :1]
        _full          = np.concatenate([x0_all, model_out], axis=-1)
        model_out_phys = normalizer.inverse(_full)[:, :, :, :, :, 1:]
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

    x_min   = grid.get('x_min',   0.0);  dx   = grid.get('dx',   1.0)
    y_min   = grid.get('y_min',   0.0);  dy   = grid.get('dy',   1.0)
    eta_min = grid.get('eta_min', 0.0);  deta = grid.get('deta', 1.0)
    tau_min = grid.get('tau_min', 0.6);  dtau = grid.get('dtau', 0.1)

    x_coords       = x_min   + np.arange(nx_eff)   * dx
    y_coords       = y_min   + np.arange(ny_eff)   * dy
    eta_coords     = eta_min + np.arange(neta_eff) * deta
    tau_out_coords = tau_min + (np.arange(ntau_out) + 1) * dtau

    print(f'x: [{x_coords[0]:.2f}, {x_coords[-1]:.2f}] fm  '
          f'y: [{y_coords[0]:.2f}, {y_coords[-1]:.2f}] fm')
    print(f'η: [{eta_coords[0]:.2f}, {eta_coords[-1]:.2f}]  '
          f'τ_out: [{tau_out_coords[0]:.3f}, {tau_out_coords[-1]:.3f}] fm/c')

    return (nx_eff, ny_eff, neta_eff, ntau_out, dtau, tau_min,
            x_coords, y_coords, eta_coords, tau_out_coords)


# ── ModelEval controls ──────────────────────────────────────────────────────

@app.cell
def _(mo, neta_eff):
    channel_ui   = mo.ui.dropdown(['0 (ε)', '1 (vx)', '2 (vy)', '3 (vη)'],
                                  value='0 (ε)', label='Channel')
    ieta_ui      = mo.ui.number(value=0, start=0, step=1, stop=neta_eff - 1,
                                label='η bin for spectral plots')
    spec_tau_ui  = mo.ui.text(value='0,-1',
                              label='τ indices for energy spectrum (comma-sep)')
    save_json_ui = mo.ui.checkbox(value=False, label='Save summary to out/metrics_eval_mlx.json')
    mo.vstack([
        mo.hstack([channel_ui, ieta_ui]),
        mo.hstack([spec_tau_ui, save_json_ui]),
    ])
    return (channel_ui, ieta_ui, spec_tau_ui, save_json_ui)


@app.cell
def _(model_out_phys, y_true_phys, tau_out_coords, eta_coords):
    from model_eval import ModelEval

    evaluator = ModelEval(
        pred=model_out_phys,
        true=y_true_phys,
        tau_coords=tau_out_coords,
        eta_coords=eta_coords,
        channel_names=['ε', 'vx', 'vy', 'vη'],
    )
    print('ModelEval ready —',
          f'N={evaluator.N}, C={evaluator.C}, '
          f'nx={evaluator.nx}, ny={evaluator.ny}, '
          f'neta={evaluator.neta}, T={evaluator.T}')
    return (evaluator,)


# ── Dashboard ───────────────────────────────────────────────────────────────

@app.cell
def _(evaluator, channel_ui, ieta_ui, spec_tau_ui,
       save_json_ui, odir, mo, plt, Path):
    import json

    _ch   = int(channel_ui.value.split()[0])
    _ieta = int(ieta_ui.value)
    _tau_idx = [int(t.strip()) for t in spec_tau_ui.value.split(',') if t.strip()]

    print(f'Computing metrics — channel={_ch}, η bin={_ieta} ...')
    _fig = evaluator.plot_dashboard(
        channel=_ch, ieta=_ieta, tau_indices=_tau_idx
    )
    mo.output.append(_fig)
    plt.close(_fig)

    if save_json_ui.value:
        _out_path = Path(odir) / 'metrics_eval_mlx.json'
        _out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(_out_path, 'w') as _f:
            json.dump(evaluator.summary(), _f, indent=2)
        print(f'Summary saved → {_out_path}')
    return


# ── Per-channel nRMSE heatmap ────────────────────────────────────────────────

@app.cell
def _(evaluator, tau_out_coords, eta_coords, plt, mo):
    _nrmse    = evaluator.nrmse()
    _ch_names = evaluator.channel_names

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
    _fig2.suptitle('nRMSE heat map (η × τ) — MLX', fontsize=12)
    _fig2.tight_layout()
    mo.output.append(_fig2)
    plt.close(_fig2)
    return


# ── fRMSE heat maps ──────────────────────────────────────────────────────────

@app.cell
def _(evaluator, tau_out_coords, eta_coords, channel_ui, plt, mo):
    _ch   = int(channel_ui.value.split()[0])
    _frmse = evaluator.frmse()

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
    _fig3.suptitle('Spectral RMSE heat maps (η × τ) — MLX', fontsize=12)
    _fig3.tight_layout()
    mo.output.append(_fig3)
    plt.close(_fig3)
    return


# ── Summary table ────────────────────────────────────────────────────────────

@app.cell
def _(evaluator, mo):
    _s = evaluator.summary()
    mo.md(
        '## Scalar summary\n\n'
        '| Metric | Value |\n|---|---|\n'
        + '\n'.join(f'| `{k}` | {v:.6g} |' for k, v in _s.items())
    )


# ── Training history ─────────────────────────────────────────────────────────

@app.cell
def _(odir, parse_training_log, plt, Path):
    _p = Path(odir) / 'metrics_mlx.json'
    if _p.exists():
        _hist = parse_training_log.TrainingHistory.load(str(_p))
        _hist.plot(title='FNO Training (MLX)')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No metrics_mlx.json at {_p}')
    return


if __name__ == "__main__":
    app.run()
