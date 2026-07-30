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
    # FNO Model Evaluation — Contour overlays (GCS-enabled)

    Source settings live in **`eval_gcs_config.yaml`** (or the `EVAL_GCS_CONFIG`
    environment variable) — same backend as `eval_gcs_clean.py`, but this notebook
    renders **only the `cont` eval**: CDF-based skinny-contour overlays of
    truth vs. prediction (3 τ snapshots per η bin), plus the relative L² error vs τ.
    """)
    return


# ---------------------------------------------------------------------------
# Setup — runs the full pipeline via eval_gcs_setup.run_all()
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

    backend        = _ctx.backend
    train_cfg      = _ctx.train_cfg
    model_out_phys = _ctx.model_out_phys
    y_true_phys    = _ctx.y_true_phys
    eta_coords     = _ctx.eta_coords
    tau_out_coords = _ctx.tau_out_coords
    x_coords       = _ctx.x_coords
    y_coords       = _ctx.y_coords
    dtau           = _ctx.dtau
    tau_min        = _ctx.tau_min
    neta_eff       = _ctx.neta_eff
    return (
        backend,
        dtau,
        eta_coords,
        model_out_phys,
        neta_eff,
        np,
        plt,
        tau_min,
        tau_out_coords,
        train_cfg,
        x_coords,
        y_coords,
        y_true_phys,
    )


# ---------------------------------------------------------------------------
# UI — contour controls
# ---------------------------------------------------------------------------
@app.cell
def _(mo, model_out_phys, neta_eff, train_cfg):
    _norm     = train_cfg.get('normalization', {})
    norm_name = _norm.get('name', 'none') if isinstance(_norm, dict) else str(_norm)
    _n_test   = model_out_phys.shape[0]

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
        cdfrats_ui,
        cont_channel_ui,
        cont_event_ui,
        cont_ieta_ui,
        contour_iT_ui,
        norm_name,
        rescale_energy_ui,
    )


# ---------------------------------------------------------------------------
# Controls + contour plots + relative L² error (rendered together)
# ---------------------------------------------------------------------------
@app.cell
def _(
    backend,
    cdfrats_ui,
    cont_channel_ui,
    cont_event_ui,
    cont_ieta_ui,
    contour_iT_ui,
    dtau,
    eta_coords,
    mo,
    model_out_phys,
    neta_eff,
    norm_name,
    np,
    plt,
    rescale_energy_ui,
    tau_min,
    tau_out_coords,
    x_coords,
    y_coords,
    y_true_phys,
):
    from skinnycontour import plot_three_bins_contour

    parts = []          # layout sections assembled below

    # ---- controls --------------------------------------------------------
    parts.append(mo.vstack([
        mo.hstack([cont_event_ui, cont_ieta_ui, cont_channel_ui]),
        mo.hstack([contour_iT_ui, cdfrats_ui]),
        mo.hstack([mo.md(f'**Normalizer:** `{norm_name}`'), rescale_energy_ui]),
    ]))

    # ---- contour overlay -------------------------------------------------
    _event          = int(cont_event_ui.value)
    _channel        = int(cont_channel_ui.value.split()[0])
    _rescale_energy = bool(rescale_energy_ui.value)

    _raw_iT = [t.strip() for t in contour_iT_ui.value.split(',') if t.strip()]
    if len(_raw_iT) < 2:
        _raw_iT = (_raw_iT + ['-1', '-1'])[:2]
    elif len(_raw_iT) > 2:
        _raw_iT = _raw_iT[:2]
    _contour_iT = (0,) + tuple(int(t) for t in _raw_iT)
    _cdfrats    = [float(x.strip()) for x in cdfrats_ui.value.split(',') if x.strip()]
    _tau0       = tau_min + dtau
    _ieta       = int(cont_ieta_ui.value)
    _eta_phys   = eta_coords[_ieta]

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

    # ---- relative L² error vs τ (selected channel, all η bins) -----------
    _denom = np.mean(y_true_phys[:, _channel] ** 2, axis=(0, 1, 2))
    _num   = np.mean((model_out_phys[:, _channel] - y_true_phys[:, _channel]) ** 2,
                     axis=(0, 1, 2))
    _rel_l2 = np.sqrt(_num / np.maximum(_denom, 1e-30))
    _sfx    = ' — MLX' if backend == 'mlx' else ''
    _fig2, _ax2 = plt.subplots(figsize=(8, 4))
    for _ie in range(neta_eff):
        _ax2.plot(tau_out_coords, _rel_l2[_ie], label=f'η={eta_coords[_ie]:.2f}')
    _ax2.set_xlabel('τ (fm/c)')
    _ax2.set_ylabel('Relative L² error')
    _ax2.set_title(f'Relative L² error vs τ (all η bins, test set){_sfx}')
    _ax2.legend(fontsize=7, ncol=2)
    _ax2.grid(True, alpha=0.3)
    _fig2.tight_layout()
    parts.append(mo.as_html(_fig2))
    plt.close(_fig2)

    mo.vstack(parts)
    return


# ---------------------------------------------------------------------------
# Memory diagnostic — splits process RSS into live arrays vs allocator cache.
# An "(other)" row is a real resident array to chase (e.g. a stray data_test
# or a float64 output); a large "unaccounted" with no "(other)" row is just
# free-but-retained allocator memory (harmless, reused next run).
# ---------------------------------------------------------------------------
@app.cell
def _(mo, model_out_phys, np, y_true_phys):
    import gc
    import psutil

    gc.collect()

    _named     = [('model_out_phys', model_out_phys), ('y_true_phys', y_true_phys)]
    _named_ids = {id(_a) for _, _a in _named}
    _rows      = [(_nm, _a.nbytes / 1e9, str(_a.dtype), tuple(_a.shape))
                  for _nm, _a in _named]
    _total     = sum(_r[1] for _r in _rows)

    # any OTHER owned ndarray > 50 MB still resident
    for _o in gc.get_objects():
        if (isinstance(_o, np.ndarray) and _o.base is None
                and _o.nbytes > 50e6 and id(_o) not in _named_ids):
            _rows.append(('(other)', _o.nbytes / 1e9, str(_o.dtype), tuple(_o.shape)))
            _total += _o.nbytes / 1e9

    _rss = psutil.Process().memory_info().rss / 1e9
    _mlx_active = _mlx_cache = _mlx_peak = 0.0
    try:
        import mlx.core as _mx
        _mlx_active = _mx.get_active_memory() / 1e9
        _mlx_cache  = _mx.get_cache_memory()  / 1e9
        _mlx_peak   = _mx.get_peak_memory()   / 1e9
    except Exception:
        pass

    _tbl = '\n'.join(f'| `{_nm}` | {_gb:.2f} | `{_dt}` | {_sh} |'
                     for _nm, _gb, _dt, _sh in _rows)
    mo.md(
        '### Memory diagnostic\n\n'
        '| array | size (GB) | dtype | shape |\n|---|---|---|---|\n'
        + _tbl + '\n\n'
        + f'- **live ndarrays (owned, >50 MB):** {_total:.2f} GB\n'
        + f'- **MLX:** active {_mlx_active:.2f} · cache {_mlx_cache:.2f} · '
          f'peak {_mlx_peak:.2f} GB\n'
        + f'- **process RSS:** {_rss:.2f} GB\n'
        + f'- **unaccounted (allocator cache / fragmentation):** '
          f'{_rss - _total - _mlx_active:.2f} GB\n\n'
        + '_Large `unaccounted` with no `(other)` row → free-but-retained '
          'allocator memory. An `(other)` row → a real resident array._'
    )
    return


if __name__ == '__main__':
    app.run()
