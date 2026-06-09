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

    All source settings live in **`eval_gcs_config.yaml`** (or the `EVAL_GCS_CONFIG`
    environment variable).  Edit that file to change model/data sources, backend, and
    checkpoint type — no code changes needed.

    **Eval modes:**
    | Mode | Description |
    |---|---|
    | `basic`  | Truth / prediction image grids per η bin at selected τ steps |
    | `cont`   | CDF-based contour overlays (3 τ snapshots per η bin) |
    | `metric` | Full metric suite: nRMSE · fRMSE · MELR · WLR dashboard |
    """)
    return


# ---------------------------------------------------------------------------
# Setup cell — runs the full pipeline via eval_gcs_setup.run_all()
# ---------------------------------------------------------------------------
@app.cell
def _(mo):
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent) if '__file__' in dir() else '.')

    import eval_gcs_setup as _setup
    try:
        _ctx = _setup.run_all()
    except Exception as _e:
        mo.stop(True, mo.callout(mo.md(f"**Setup failed:** {_e}"), kind="danger"))

    eval_cfg       = _ctx.eval_cfg
    train_cfg      = _ctx.train_cfg
    backend        = _ctx.backend
    # Keep only the event count, not the full raw array: the rest of the
    # notebook needs nothing else from data_test, and exporting the array would
    # pin ~4.5 GB resident (the whole point is to free it once _ctx is dropped).
    n_test         = _ctx.data_test.shape[0]
    model_out_phys = _ctx.model_out_phys
    y_true_phys    = _ctx.y_true_phys
    eta_coords     = _ctx.eta_coords
    tau_out_coords = _ctx.tau_out_coords
    x_coords       = _ctx.x_coords
    y_coords       = _ctx.y_coords
    xy_extent      = _ctx.xy_extent
    dtau           = _ctx.dtau
    tau_min        = _ctx.tau_min
    neta_eff       = _ctx.neta_eff
    ntau_out       = _ctx.ntau_out
    return (
        backend,
        dtau,
        eta_coords,
        eval_cfg,
        model_out_phys,
        n_test,
        neta_eff,
        np,
        ntau_out,
        plt,
        tau_min,
        tau_out_coords,
        train_cfg,
        x_coords,
        xy_extent,
        y_coords,
        y_true_phys,
    )


# ---------------------------------------------------------------------------
# UI — create all widgets (no .value access in this cell)
# ---------------------------------------------------------------------------
@app.cell
def _(eval_cfg, mo, n_test, neta_eff, train_cfg):
    _norm     = train_cfg.get('normalization', {})
    norm_name = _norm.get('name', 'none') if isinstance(_norm, dict) else str(_norm)
    _n_test        = n_test
    _default_mode  = eval_cfg.get('eval_mode', 'basic')
    _default_thresh = float(eval_cfg.get('active_threshold', 0.3))

    eval_mode_ui = mo.ui.dropdown(
        ['basic', 'cont', 'metric'],
        value=_default_mode,
        label='Eval mode',
    )

    # active threshold — shown in ALL modes (affects metric masking)
    active_thresh_ui = mo.ui.number(value=_default_thresh, start=0.0, step=0.05, stop=5.0,
                                    label='Active ε threshold (GeV/fm³, 0=off)')

    # basic / metric controls
    event_ui         = mo.ui.number(value=0, start=0, step=1,
                                    label='Event index for plots')
    channel_num_ui   = mo.ui.number(value=0, start=0, step=1,
                                    label='Channel (0=ε, 1=vy, 2=vx, 3=vη)')
    channel_drop_ui  = mo.ui.dropdown(['0 (ε)', '1 (vx)', '2 (vy)', '3 (vη)'],
                                      value='0 (ε)', label='Channel')
    tau_idx_ui       = mo.ui.text(value='5,10,15,-1',
                                  label='τ output indices (comma-sep, -1=last)')
    ieta_ui          = mo.ui.number(value=0, start=0, step=1, stop=neta_eff - 1,
                                    label='η bin index')
    spec_tau_ui      = mo.ui.text(value='0,-1',
                                  label='τ indices for energy spectrum (comma-sep)')
    save_json_ui     = mo.ui.checkbox(value=False, label='Save summary to JSON')

    # cont controls
    cont_event_ui   = mo.ui.slider(start=0, stop=_n_test - 1, value=0, step=1,
                                   label='Event')
    cont_ieta_ui    = mo.ui.slider(start=0, stop=neta_eff - 1, value=0, step=1,
                                   label='η bin')
    cont_channel_ui = mo.ui.dropdown(
        ['0 — ε (energy)', '1 — vx', '2 — vy', '3 — vη'],
        value='0 — ε (energy)', label='Feature / channel',
    )
    contour_iT_ui   = mo.ui.text(value='10,-1',
                                  label='2nd and 3rd τ-step indices (comma-sep; '
                                        '1st panel is always 0; -1=last non-zero)')
    cdfrats_ui      = mo.ui.text(value='0.05,0.1,0.2,0.3,0.4,0.6',
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
        norm_name,
        rescale_energy_ui,
        save_json_ui,
        spec_tau_ui,
        tau_idx_ui,
    )


# ---------------------------------------------------------------------------
# UI + output — controls and plots rendered together in one cell so the
# output appears directly below the controls without any gap.
# ---------------------------------------------------------------------------
@app.cell
def _(
    active_thresh_ui,
    backend,
    cdfrats_ui,
    channel_drop_ui,
    channel_num_ui,
    cont_channel_ui,
    cont_event_ui,
    cont_ieta_ui,
    contour_iT_ui,
    dtau,
    eta_coords,
    eval_cfg,
    eval_mode_ui,
    event_ui,
    ieta_ui,
    mo,
    model_out_phys,
    neta_eff,
    norm_name,
    np,
    ntau_out,
    plt,
    rescale_energy_ui,
    save_json_ui,
    spec_tau_ui,
    tau_idx_ui,
    tau_min,
    tau_out_coords,
    x_coords,
    xy_extent,
    y_coords,
    y_true_phys,
):
    mode = eval_mode_ui.value
    parts = []          # layout sections assembled below

    # ---- controls --------------------------------------------------------
    # Top row always shows mode selector + active threshold
    _top_row = mo.hstack([eval_mode_ui, active_thresh_ui])

    if mode == 'basic':
        _controls = mo.vstack([
            _top_row,
            mo.hstack([event_ui, ieta_ui, tau_idx_ui, channel_num_ui]),
        ])
    elif mode == 'cont':
        _controls = mo.vstack([
            _top_row,
            mo.hstack([cont_event_ui, cont_ieta_ui, cont_channel_ui]),
            mo.hstack([contour_iT_ui, cdfrats_ui]),
            mo.hstack([mo.md(f'**Normalizer:** `{norm_name}`'), rescale_energy_ui]),
        ])
    else:
        _controls = mo.vstack([
            _top_row,
            mo.hstack([channel_drop_ui, ieta_ui]),
            mo.hstack([spec_tau_ui, save_json_ui]),
        ])
    parts.append(_controls)

    # ---- basic plots -----------------------------------------------------
    if mode == 'basic':
        _event   = int(event_ui.value)
        _ieta    = int(ieta_ui.value)
        _channel = int(channel_num_ui.value)
        _raw_idx     = [t.strip() for t in tau_idx_ui.value.split(',') if t.strip()]
        _tau_indices = [int(t) % ntau_out for t in _raw_idx]
        _tau_indices = [t for t in _tau_indices if 0 <= t < ntau_out]
        _ch_label  = ['e (GeV/fm³)', 'vx', 'vy', 'vη'][_channel] if _channel < 4 else f'ch{_channel}'
        _eta_phys  = eta_coords[_ieta]

        _fig, _axes = plt.subplots(2, len(_tau_indices),
                                   figsize=(4.5 * len(_tau_indices), 7), squeeze=False)
        _fig.suptitle(f'η = {_eta_phys:.2f}  (event {_event})', fontsize=14, y=1.01)
        for _col, _t in enumerate(_tau_indices):
            _tau_phys = tau_out_coords[_t]
            _true = y_true_phys[_event, _channel, :, :, _ieta, _t]
            _pred = model_out_phys[_event, _channel, :, :, _ieta, _t]
            _vmin, _vmax = min(_true.min(), _pred.min()), max(_true.max(), _pred.max())
            _kw = dict(origin='lower', extent=xy_extent, aspect='equal',
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
        parts.append(mo.as_html(_fig))
        plt.close(_fig)

    # ---- cont plots ------------------------------------------------------
    elif mode == 'cont':
        from skinnycontour import plot_three_bins_contour
        _event          = int(cont_event_ui.value)
        _channel        = int(cont_channel_ui.value.split()[0])
        _rescale_energy = bool(rescale_energy_ui.value)
        _raw_iT = [t.strip() for t in contour_iT_ui.value.split(',') if t.strip()]
        if len(_raw_iT) < 2:
            _raw_iT = (_raw_iT + ['-1', '-1'])[:2]
        elif len(_raw_iT) > 2:
            _raw_iT = _raw_iT[:2]
        _contour_iT = (0,) + tuple(int(t) for t in _raw_iT)
        _cdfrats  = [float(x.strip()) for x in cdfrats_ui.value.split(',') if x.strip()]
        _tau0     = tau_min + dtau
        _ieta     = int(cont_ieta_ui.value)
        _eta_phys = eta_coords[_ieta]

        _fig, _axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
        plot_three_bins_contour(
            model_out_phys, y_true_phys,
            iparam=_channel, event=_event,
            tau0=_tau0, taudelta=dtau, iT=_contour_iT,
            cdfrats=_cdfrats, ieta=_ieta, tag=f'η={_eta_phys:.2f}',
            rescale_energy=_rescale_energy,
            x_coords=x_coords, y_coords=y_coords,
            fig=_fig, axes=_axes,
        )
        parts.append(mo.as_html(_fig))
        plt.close(_fig)

    # ---- metric plots ----------------------------------------------------
    else:
        import json
        from pathlib import Path as _Path
        from model_eval import ModelEval

        _thresh          = active_thresh_ui.value if active_thresh_ui.value > 0 else None
        _ch              = int(channel_drop_ui.value.split()[0])
        _tau_idx         = [int(t.strip()) for t in spec_tau_ui.value.split(',') if t.strip()]
        _sfx             = ' — MLX' if backend == 'mlx' else ''
        _metric_backend  = eval_cfg.get('metric_backend', 'numpy')

        # η bin closest to η=0 (physically meaningful default, independent of bin ordering)
        _ieta = int(np.argmin(np.abs(eta_coords - 0.0)))

        # τ step closest to 3 fm/c for the spectral error map panel
        _itau_3fm = int(np.argmin(np.abs(tau_out_coords - 3.0)))

        _evaluator = ModelEval(
            pred=model_out_phys, true=y_true_phys,
            tau_coords=tau_out_coords, eta_coords=eta_coords,
            channel_names=['ε', 'vx', 'vy', 'vη'],
            active_threshold=_thresh,
            backend=_metric_backend,
        )

        # dashboard
        _fig_dash = _evaluator.plot_dashboard(
            channel=_ch, ieta=_ieta, tau_indices=_tau_idx, itau=_itau_3fm,
        )
        parts.append(mo.as_html(_fig_dash))
        plt.close(_fig_dash)

        # nRMSE heatmap (all channels)
        _nrmse = _evaluator.nrmse()
        _fig2, _axes2 = plt.subplots(1, _evaluator.C, figsize=(5 * _evaluator.C, 3.5))
        if _evaluator.C == 1:
            _axes2 = [_axes2]
        for _c, _ax in enumerate(_axes2):
            _im = _ax.pcolormesh(tau_out_coords, eta_coords, _nrmse[_c],
                                 cmap='hot_r', shading='auto')
            plt.colorbar(_im, ax=_ax, label='nRMSE')
            _ax.set_xlabel('τ (fm/c)'); _ax.set_ylabel('η')
            _ax.set_title(f'nRMSE — {_evaluator.channel_names[_c]}')
        _fig2.suptitle(f'nRMSE heat map (η × τ){_sfx}', fontsize=12)
        _fig2.tight_layout()
        parts.append(mo.as_html(_fig2))
        plt.close(_fig2)

        # fRMSE heatmaps (selected channel)
        _frmse = _evaluator.frmse()
        _fig3, _axes3 = plt.subplots(1, 3, figsize=(15, 3.5))
        for _ax, (band, cmap) in zip(_axes3,
                [('low', 'Blues'), ('mid', 'Oranges'), ('high', 'Reds')]):
            _im = _ax.pcolormesh(tau_out_coords, eta_coords, _frmse[band][_ch],
                                 cmap=cmap, shading='auto')
            plt.colorbar(_im, ax=_ax, label='fRMSE')
            _ax.set_xlabel('τ (fm/c)'); _ax.set_ylabel('η')
            _ax.set_title(f'fRMSE ({band}) — {_evaluator.channel_names[_ch]}')
        _fig3.suptitle(f'Spectral RMSE heat maps (η × τ){_sfx}', fontsize=12)
        _fig3.tight_layout()
        parts.append(mo.as_html(_fig3))
        plt.close(_fig3)

        # scalar summary table
        _s = _evaluator.summary()
        parts.append(mo.md(
            '## Scalar summary\n\n'
            '| Metric | Value |\n|---|---|\n'
            + '\n'.join(f'| `{k}` | {v:.6g} |' for k, v in _s.items())
        ))

        # optional JSON save
        if save_json_ui.value:
            _out_path = _Path(eval_cfg.get('odir', './out')) / f'metrics_eval{"_mlx" if backend == "mlx" else ""}.json'
            _out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_out_path, 'w') as _f:
                json.dump(_s, _f, indent=2)
            parts.append(mo.callout(mo.md(f'Saved → `{_out_path}`'), kind='success'))

    # ---- relative L² error vs τ (basic + cont) ---------------------------
    if mode != 'metric':
        _ch     = int(channel_num_ui.value)
        _denom  = np.mean(y_true_phys[:, _ch] ** 2, axis=(0, 1, 2))
        _num    = np.mean((model_out_phys[:, _ch] - y_true_phys[:, _ch]) ** 2, axis=(0, 1, 2))
        _rel_l2 = np.sqrt(_num / np.maximum(_denom, 1e-30))
        _sfx    = ' — MLX' if backend == 'mlx' else ''
        _fig4, _ax4 = plt.subplots(figsize=(8, 4))
        for _ie in range(neta_eff):
            _ax4.plot(tau_out_coords, _rel_l2[_ie], label=f'η={eta_coords[_ie]:.2f}')
        _ax4.set_xlabel('τ (fm/c)')
        _ax4.set_ylabel('Relative L² error')
        _ax4.set_title(f'Relative L² error vs τ (all η bins, test set){_sfx}')
        _ax4.legend(fontsize=7, ncol=2)
        _ax4.grid(True, alpha=0.3)
        _fig4.tight_layout()
        parts.append(mo.as_html(_fig4))
        plt.close(_fig4)

    mo.vstack(parts)
    return


# ---------------------------------------------------------------------------
# Training history plot (if available)
# ---------------------------------------------------------------------------
@app.cell
def _(backend, eval_cfg, plt):
    import parse_training_log
    from pathlib import Path as _Path

    _sfx  = '_mlx' if backend == 'mlx' else ''
    _odir = eval_cfg.get('odir', './out')
    _p    = _Path(_odir) / f'metrics{_sfx}.json'
    if _p.exists():
        _hist = parse_training_log.TrainingHistory.load(str(_p))
        _hist.plot(title=f'FNO Training ({"MLX" if backend == "mlx" else "PyTorch"})')
        plt.tight_layout()
        plt.show()
    else:
        print(f'No {_p.name} found at {_p}')
    return


if __name__ == '__main__':
    app.run()
