"""Reusable evaluation/inference helpers shared by the eval notebooks and the
``workflow_torch`` CLI scripts.

This centralises the two pieces that were previously copy-pasted across
``eval_model/eval.py``, ``eval_model/eval_gcs_setup.py`` and
``eval_model_web/app.py``:

* :func:`run_inference`     — batched inference through the shared data pipeline
                              (``FluidDataset`` + collate), bit-identical to the
                              training first-tau-repeat windowing + normalizer.
* :func:`compute_grid_coords` — physical (x, y, η, τ_out) coordinate arrays from
                              the test array shape + grid params.

plus a small torch checkpoint loader (:func:`load_torch_state_dict` /
:func:`load_torch_model`) that understands the several state-dict layouts the
training scripts emit.

The heavy metric machinery still lives in :mod:`model_eval` (``ModelEval``);
this module only covers the data → pred/true → coordinates path that feeds it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# batched inference (DATA_PIPELINE_PLAN.md §7–§8 collate reuse)
# ---------------------------------------------------------------------------
def run_inference(model, backend: str, device, data_test, normalizer,
                  batch_size: int = 8):
    """Batched inference via the SHARED data pipeline (FluidDataset + collate).

    The collate builds the model input exactly as training does (first-tau
    repeat) and applies the SAME normalizer, so eval preprocessing is
    bit-identical to the train path — no hand-rolled repeat/normalize.  Batching
    the forward pass replaces the old one-event-at-a-time loop (the main
    GPU-utilisation win).  ``shuffle=False`` keeps pred/true aligned with event
    order; ``num_workers=0`` because ``data_test`` is typically already an
    in-memory ndarray (workers would only add fork/IPC overhead).

    Parameters
    ----------
    model      : torch.nn.Module | mlx model
        Tau-preserving model mapping (B, C, nx, ny, neta, T-1) -> same shape.
    backend    : {'torch', 'mlx'}
    device     : torch.device | None
        Target device for the torch path; ignored for MLX.
    data_test  : ArraySource | np.ndarray
        Test events ``(N, C, nx, ny, neta, full_T)``.  Any object the
        ``FluidDataset`` accepts (ndarray, MultiH5Array, RootSource, …).
    normalizer : Normalizer | None
        Applied per batch by the collate; ``None`` (or a ``name == 'none'``
        normalizer) is a no-op and inference runs on raw data.
    batch_size : int
        Forward-pass batch size.

    Returns
    -------
    (model_out_phys, y_true_phys) : tuple[np.ndarray, np.ndarray]
        Physical-unit prediction and ground truth, each
        ``(N, C, nx, ny, neta, full_T - 1)`` (initial condition stripped).
    """
    from data import FluidDataset, make_collate
    from torch.utils.data import DataLoader

    # Window length so the collate emits all post-IC steps: it builds
    # x = repeat(IC, T-1) and y = steps 1..T-1.  data_test has full_T tau
    # slices, so T = full_T gives full_T-1 output steps (IC excluded).
    T      = data_test.shape[-1]
    ds     = FluidDataset(data_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=make_collate(normalizer, T, backend))

    if backend == 'torch':
        import torch
    else:
        import mlx.core as mx

    pred_phys, true_phys = [], []
    for b in loader:
        x, y = b['x'], b['y']                          # normalised: x=IC-repeat, y=truth
        if backend == 'torch':
            with torch.no_grad():
                out = model(x.to(device)).cpu().numpy()
            x = x.numpy(); y = y.numpy()
        else:
            o = model(mx.array(x)); mx.eval(o)
            out = np.array(o)                          # x, y already numpy

        # x0 = first slice of the repeated input = the normalised initial
        # condition; concat it back so normalizer.inverse sees the IC at
        # tau-index 0 (its per-tau params line up), then strip the IC.
        x0 = x[..., :1]
        if normalizer is not None:
            pred_phys.append(normalizer.inverse(np.concatenate([x0, out], axis=-1))[..., 1:])
            true_phys.append(normalizer.inverse(np.concatenate([x0, y],   axis=-1))[..., 1:])
        else:
            pred_phys.append(out)
            true_phys.append(y)

    model_out_phys = np.concatenate(pred_phys, axis=0)   # (N, C, nx, ny, neta, T)
    y_true_phys    = np.concatenate(true_phys, axis=0)
    return model_out_phys, y_true_phys


# ---------------------------------------------------------------------------
# grid coordinates
# ---------------------------------------------------------------------------
def compute_grid_coords(data_test, grid: dict) -> dict:
    """Physical coordinate arrays for an eval array + grid params.

    ``data_test`` is shaped ``(N, C, nx, ny, neta, full_T)`` and the returned
    ``tau_out_coords`` corresponds to the ``full_T - 1`` predicted steps (the
    initial condition at tau-index 0 is excluded).

    Returns a dict with ``eta_coords, tau_out_coords, x_coords, y_coords,
    xy_extent, dtau, tau_min, neta_eff, ntau_out``.
    """
    neta_eff = data_test.shape[4]
    ntau_out = data_test.shape[5] - 1

    x_min   = grid.get('x_min',   0.0);  dx   = grid.get('dx',   1.0)
    y_min   = grid.get('y_min',   0.0);  dy   = grid.get('dy',   1.0)
    eta_min = grid.get('eta_min', 0.0);  deta = grid.get('deta', 1.0)
    tau_min = grid.get('tau_min', 0.6);  dtau = grid.get('dtau', 0.1)

    x_coords       = x_min   + np.arange(data_test.shape[2]) * dx
    y_coords       = y_min   + np.arange(data_test.shape[3]) * dy
    eta_coords     = eta_min + np.arange(neta_eff) * deta
    tau_out_coords = tau_min + (np.arange(ntau_out) + 1) * dtau
    xy_extent      = [x_coords[0] - dx/2, x_coords[-1] + dx/2,
                      y_coords[0] - dy/2, y_coords[-1] + dy/2]

    return dict(
        eta_coords=eta_coords, tau_out_coords=tau_out_coords,
        x_coords=x_coords, y_coords=y_coords, xy_extent=xy_extent,
        dtau=dtau, tau_min=tau_min, neta_eff=neta_eff, ntau_out=ntau_out,
    )


# ---------------------------------------------------------------------------
# torch checkpoint loading
# ---------------------------------------------------------------------------
def load_torch_state_dict(raw):
    """Extract a plain ``{name: tensor}`` state dict from a loaded checkpoint.

    Handles the layouts the training scripts emit:
      * a full checkpoint dict with a ``'model_state_dict'`` key,
      * a bare state dict (all values are tensors),
      * a ``module.``-prefixed (DataParallel) dict — the prefix is stripped.
    """
    import torch
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        return raw['model_state_dict']
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    return {
        (k[7:] if k.startswith('module.') else k): v
        for k, v in raw.items() if isinstance(v, torch.Tensor)
    }


def load_torch_model(model, ckpt_path, device, *, eval_mode: bool = True):
    """Load a torch checkpoint file into ``model`` in place and return it.

    ``ckpt_path`` may point at any of the layouts :func:`load_torch_state_dict`
    understands (``best_model_state_dict.pt``, ``last_model.pt``, …).
    """
    import torch
    raw = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(load_torch_state_dict(raw))
    if eval_mode:
        model.eval()
    return model
