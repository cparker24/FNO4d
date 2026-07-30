"""Shared FNO model selector for the workflow_torch scripts.

train.py, resume.py and eval.py must build the model identically — otherwise a
checkpoint trained with one construction will not load cleanly into another
(notably the CausalFNO4DTauEta wrapper, whose state dict is keyed ``m.*`` plus
``tau``/``eta`` buffers).  Keeping one builder guarantees that.

``loc_libs`` must already be on ``sys.path`` (the workflow scripts insert it)
for the ``training_utils`` import used by the embedding-grid save.
"""

import torch.nn as nn
from neuralop.models import FNO


def build_model(model_type, *, n_modes, hidden_channels, device,
                sample_x_shape=None, grid_params=None, causal_cfg=None,
                model_dir=None, save_embedding=False, log=print):
    """Build the selected model and move it to ``device``.

    Parameters
    ----------
    model_type   : {'fno', 'causal_fno4d', 'causal_fno4d_tau_eta'}
    n_modes      : list[int]            Fourier modes (spatial = first 3 for causal).
    hidden_channels : int
    device       : torch.device
    sample_x_shape : tuple | None       (B, C, nx, ny, neta, T) — required for
        causal_fno4d_tau_eta (gives n_eta, n_tau for the tau/eta embedding).
    grid_params  : dict | None          tau_min/dtau/eta_min/deta — required for
        causal_fno4d_tau_eta.  Pass the *effective* (post-downsample) grid.
    causal_cfg   : dict | None          extra kwargs for the CausalFNO4D variants.
    model_dir    : str | None           where to write embedding_grid.json.
    save_embedding : bool               write the embedding grid (training only).
    log          : callable             logger (defaults to print).

    Returns
    -------
    torch.nn.Module
        For causal_fno4d_tau_eta this is a thin wrapper pinning tau/eta so the
        Trainer / inference loop can call ``model(x)``.
    """
    causal_cfg = causal_cfg or {}

    if model_type == 'fno':
        return FNO(
            in_channels=4, out_channels=4,
            n_modes=n_modes, hidden_channels=hidden_channels,
            projection_channel_ratio=2,
        ).to(device)

    if model_type == 'causal_fno4d':
        from neuralop.models import CausalFNO4D
        model = CausalFNO4D(
            in_channels=4, out_channels=4,
            hidden_channels=hidden_channels,
            n_modes_spatial=tuple(n_modes[:3]),
            **causal_cfg,
        ).to(device)
        log(f'CausalFNO4D temporal_receptive_field: {model.temporal_receptive_field}')
        return model

    if model_type == 'causal_fno4d_tau_eta':
        from neuralop.models import CausalFNO4DTauEta
        from neuralop.models.causal_fno_4d import make_tau_tensor, make_eta_tensor
        if sample_x_shape is None or grid_params is None:
            raise ValueError(
                "model_type 'causal_fno4d_tau_eta' requires sample_x_shape and grid_params")
        n_eta = sample_x_shape[-2]     # Z after slicing/downsample
        n_tau = sample_x_shape[-1]     # T after windowing
        tau_vec = make_tau_tensor(grid_params['tau_min'], grid_params['dtau'], n_tau, device=device)
        eta_vec = make_eta_tensor(grid_params['eta_min'], grid_params['deta'], n_eta, device=device)
        base = CausalFNO4DTauEta(
            in_channels=4, out_channels=4,
            hidden_channels=hidden_channels,
            n_modes_spatial=tuple(n_modes[:3]),
            tau_min=float(grid_params['tau_min']), dtau=float(grid_params['dtau']),
            eta_min=float(grid_params['eta_min']), deta=float(grid_params['deta']),
            **causal_cfg,
        ).to(device)
        log(f'CausalFNO4DTauEta temporal_receptive_field: {base.temporal_receptive_field}')
        if save_embedding and model_dir is not None:
            from training_utils import save_embedding_grid
            save_embedding_grid(
                model_dir, 'CausalFNO4DTauEta',
                grid_params['tau_min'], grid_params['dtau'], n_tau, base.tau_max,
                grid_params['eta_min'], grid_params['deta'], n_eta, base.eta_max,
            )
            log(f'embedding grid  →  {model_dir}/embedding_grid.json')

        class _TauEtaWrapper(nn.Module):
            """Pin tau/eta from the data grid so callers use model(x)."""
            def __init__(self, m, tau, eta):
                super().__init__()
                self.m = m
                self.register_buffer('tau', tau)
                self.register_buffer('eta', eta)
            def forward(self, x):
                return self.m(x, self.tau, self.eta)

        return _TauEtaWrapper(base, tau_vec, eta_vec)

    raise ValueError(
        f"Unknown model_type: {model_type!r}. "
        "Must be 'fno', 'causal_fno4d', or 'causal_fno4d_tau_eta'.")
