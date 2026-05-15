"""
norm_utils.py

Thin wrapper around NormFn3d that manages the full normalisation lifecycle
for a training run:

    fit_transform(train_data)   — forward pass; computes ref from training data
    transform(data)             — forward pass with the training ref (val/test)
    inverse(data)               — undo normalisation (physical units)
    save(path)                  — persist params + ref to .npz alongside the model
    Normalizer.load(path)       — restore a saved normalizer at inference time

Usage in a training script
--------------------------
    grid_params = {k: arr_data[0][k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')}
    normalizer  = build_normalizer(cfg.get('normalization'), grid_params)

    data_train  = normalizer.fit_transform(data_train)
    data_verify = normalizer.transform(data_verify)
    normalizer.save(f'{model_dir}/normalizer.npz')

    # after inference:
    pred_physical = normalizer.inverse(model_output)

Supported names (set via YAML key  normalization: {name: ...})
--------------------------------------------------------------
    none                — identity; no-op in all directions
    etau75_3d           — ed^0.75 * tau  (tau scaling only)
    etau75_eta_3d       — (ed^0.75 * tau) / ref[ieta, itau]  (data-driven)
    etau75_eta_shape_3d — (ed^0.75 * tau) / shape_ref        (shape-only eta)
    cosh_etau75_3d      — (ed * cosh(eta)^alpha)^0.75 * tau  (analytic eta)

YAML example
------------
    normalization:
      name: none          # default; change to one of the names above
      alpha: 1.0          # only used by cosh_etau75_3d
"""

import numpy as np
from NormFn3d import NormFn3d

_VALID = frozenset({
    'none',
    'etau75_3d',
    'etau75_eta_3d',
    'etau75_eta_shape_3d',
    'cosh_etau75_3d',
})


class Normalizer:
    """Stateful normalizer; holds grid params and (optionally) a data-driven ref."""

    def __init__(self, name: str, grid_params: dict, norm_cfg: dict):
        if name not in _VALID:
            raise ValueError(f"Unknown normalisation {name!r}. Choose from {sorted(_VALID)}")
        self.name    = name
        self.tau_min = float(grid_params['tau_min'])
        self.dtau    = float(grid_params['dtau'])
        self.eta_min = float(grid_params['eta_min']) if 'eta_min' in grid_params else None
        self.deta    = float(grid_params['deta'])    if 'deta'    in grid_params else None
        self.alpha   = float(norm_cfg.get('alpha', 1.0))
        self.ref     = None  # populated by fit_transform for data-driven norms

    # ------------------------------------------------------------------
    # forward passes
    # ------------------------------------------------------------------

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Normalise *training* data.  Computes and stores ref for data-driven norms."""
        if self.name == 'none':
            return data
        if self.name == 'etau75_3d':
            return NormFn3d.etau75_3d(data, self.tau_min, self.dtau)
        if self.name == 'etau75_eta_3d':
            self.ref = NormFn3d.compute_ref_etau75_eta(data, self.tau_min, self.dtau)
            out, _   = NormFn3d.etau75_eta_3d(data, self.tau_min, self.dtau, ref=self.ref)
            return out
        if self.name == 'etau75_eta_shape_3d':
            self.ref = NormFn3d.compute_ref_etau75_eta_shape(data, self.tau_min, self.dtau)
            out, _   = NormFn3d.etau75_eta_shape_3d(data, self.tau_min, self.dtau, shape_ref=self.ref)
            return out
        if self.name == 'cosh_etau75_3d':
            self._require_eta_params()
            return NormFn3d.cosh_etau75_3d(
                data, self.tau_min, self.dtau, self.eta_min, self.deta, self.alpha)
        raise AssertionError('unreachable')

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalise *validation/test* data using the ref from fit_transform."""
        if self.name == 'none':
            return data
        if self.name == 'etau75_3d':
            return NormFn3d.etau75_3d(data, self.tau_min, self.dtau)
        if self.name == 'etau75_eta_3d':
            self._require_ref()
            out, _ = NormFn3d.etau75_eta_3d(data, self.tau_min, self.dtau, ref=self.ref)
            return out
        if self.name == 'etau75_eta_shape_3d':
            self._require_ref()
            out, _ = NormFn3d.etau75_eta_shape_3d(data, self.tau_min, self.dtau, shape_ref=self.ref)
            return out
        if self.name == 'cosh_etau75_3d':
            self._require_eta_params()
            return NormFn3d.cosh_etau75_3d(
                data, self.tau_min, self.dtau, self.eta_min, self.deta, self.alpha)
        raise AssertionError('unreachable')

    # ------------------------------------------------------------------
    # inverse
    # ------------------------------------------------------------------

    def inverse(self, data: np.ndarray) -> np.ndarray:
        """Undo normalisation — recover physical energy density units."""
        if self.name == 'none':
            return data
        if self.name == 'etau75_3d':
            return NormFn3d.inv_etau75_3d(data, self.tau_min, self.dtau)
        if self.name == 'etau75_eta_3d':
            self._require_ref()
            return NormFn3d.inv_etau75_eta_3d(data, self.ref, self.tau_min, self.dtau)
        if self.name == 'etau75_eta_shape_3d':
            self._require_ref()
            return NormFn3d.inv_etau75_eta_shape_3d(data, self.ref, self.tau_min, self.dtau)
        if self.name == 'cosh_etau75_3d':
            self._require_eta_params()
            return NormFn3d.inv_cosh_etau75_3d(
                data, self.tau_min, self.dtau, self.eta_min, self.deta, self.alpha)
        raise AssertionError('unreachable')

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save normalizer state to *path* (a .npz file)."""
        kw = dict(
            name    = np.array(self.name),
            tau_min = np.array(self.tau_min),
            dtau    = np.array(self.dtau),
            alpha   = np.array(self.alpha),
        )
        if self.eta_min is not None:
            kw['eta_min'] = np.array(self.eta_min)
        if self.deta is not None:
            kw['deta'] = np.array(self.deta)
        if self.ref is not None:
            kw['ref'] = self.ref
        np.savez(path, **kw)

    @classmethod
    def load(cls, path: str) -> 'Normalizer':
        """Restore a saved normalizer from *path*."""
        d    = np.load(path, allow_pickle=False)
        name = str(d['name'])
        grid = {'tau_min': float(d['tau_min']), 'dtau': float(d['dtau'])}
        if 'eta_min' in d:
            grid['eta_min'] = float(d['eta_min'])
        if 'deta' in d:
            grid['deta'] = float(d['deta'])
        cfg = {'alpha': float(d['alpha'])}
        obj = cls(name, grid, cfg)
        if 'ref' in d:
            obj.ref = d['ref']
        return obj

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _require_ref(self):
        if self.ref is None:
            raise RuntimeError(
                f"Normalizer '{self.name}' requires a ref array. "
                "Call fit_transform() on training data first, or load() a saved normalizer."
            )

    def _require_eta_params(self):
        if self.eta_min is None or self.deta is None:
            raise RuntimeError(
                f"Normalizer '{self.name}' requires eta_min and deta in grid_params."
            )

    def __repr__(self):
        ref_shape = None if self.ref is None else self.ref.shape
        return (f"Normalizer(name={self.name!r}, tau_min={self.tau_min}, dtau={self.dtau}, "
                f"alpha={self.alpha}, ref={ref_shape})")


# ------------------------------------------------------------------
# factory
# ------------------------------------------------------------------

def build_normalizer(norm_cfg, grid_params: dict) -> Normalizer:
    """Create a Normalizer from a YAML config dict and data grid params.

    Parameters
    ----------
    norm_cfg    : dict from YAML  normalization: {...}, or None for identity
    grid_params : dict with keys tau_min, dtau (and eta_min, deta for cosh variant)
                  — typically taken directly from read_3d_data() output
    """
    if norm_cfg is None:
        norm_cfg = {}
    name = norm_cfg.get('name', 'none')
    return Normalizer(name, grid_params, norm_cfg)
