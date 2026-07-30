#!/usr/bin/env python
"""Tests for the unified data pipeline (loc_libs/data/) — DATA_PIPELINE_PLAN.md §15.

Covers:
  * BlockBatchSampler — every index exactly once per epoch (§10)
  * RootSource[a:b] == matching slice of eager read_3d_data (§5)
  * build_dataloaders parity vs the legacy FluidDatasetH5 path (§8, §15)
  * fork-safety: num_workers=4 batches == num_workers=0 (§9, §15)
  * downsample auto-bind: coarse grid wired without manual recompute (§11)

Data-backed tests are skipped automatically when the sample .root / .h5 files
are not present (CI without the large data artifacts).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'loc_libs'))

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
_H5  = os.path.join(_DATA_DIR, 'dAu_25ev_mb.h5')
_H5b = os.path.join(_DATA_DIR, 'dAu_25ev_mb_2.h5')
_ROOT = os.path.join(_DATA_DIR, 'dAu_25ev_mb.root')

_GRID = {'tau_min': 0.58, 'dtau': 0.1, 'eta_min': -5.0, 'deta': 0.3125}
_XY, _ETA, _TAU = slice(None, -1), slice(None), slice(None, -1)


# ---------------------------------------------------------------------------
# §10  locality sampler — no data needed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('block', [None, 1, 7, 10, 256])
@pytest.mark.parametrize('n,bs', [(53, 4), (40, 8), (1, 4)])
def test_block_sampler_covers_every_index_once(block, n, bs):
    from data import BlockBatchSampler
    s = BlockBatchSampler(n, batch_size=bs, block=block, shuffle=True)
    flat = [i for batch in s for i in batch]
    assert sorted(flat) == list(range(n))
    assert len(list(s)) == len(s)          # __len__ matches actual batch count


def test_block_sampler_drop_last():
    from data import BlockBatchSampler
    s = BlockBatchSampler(10, batch_size=4, block=10, shuffle=False, drop_last=True)
    batches = list(s)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == len(s) == 2     # 10 // 4, last partial dropped


# ---------------------------------------------------------------------------
# §5  RootSource parity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_ROOT), reason='sample .root not present')
def test_rootsource_matches_eager():
    from read_3d_root import read_3d_data
    from data import RootSource

    n = 6
    eager = read_3d_data(_ROOT, nevents=n)['arr']
    rs = RootSource(_ROOT)

    assert rs.shape[1:] == eager.shape[1:]
    sl = np.asarray(rs[0:n])
    assert sl.shape == eager.shape
    assert np.array_equal(sl, eager)             # exact incl. pad-to-choose_ntau
    assert np.array_equal(rs[0], eager[0])       # int -> 5-D event


@pytest.mark.skipif(not os.path.exists(_ROOT), reason='sample .root not present')
def test_rootsource_pickle_roundtrip():
    import pickle
    from data import RootSource
    rs = RootSource(_ROOT)
    rs2 = pickle.loads(pickle.dumps(rs))         # survives spawn
    assert rs2.shape == rs.shape
    assert np.array_equal(rs2[0], rs[0])


# ---------------------------------------------------------------------------
# §8/§15  loader parity vs the legacy per-event path
# ---------------------------------------------------------------------------
def _legacy_batches(norm_cfg, n_total=20):
    from read_3d_hdf5 import MultiH5Array
    from norm_utils import build_normalizer
    from fluid_datasets import FluidDatasetH5
    from torch.utils.data import DataLoader

    src = MultiH5Array([_H5], global_start=0, global_stop=n_total)
    n_train = int(n_total * 0.8)
    norm = build_normalizer(norm_cfg, _GRID)
    if norm.name in ('etau75_eta_3d', 'etau75_eta_shape_3d'):
        norm.fit_lazy(src, n_train, _XY, _ETA, _TAU, chunk_size=8)
    ds = FluidDatasetH5(src, indices=np.arange(n_train),
                        xyslice=_XY, etaslice=_ETA, tauslice=_TAU, normalizer=norm)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    return [(b['x'].numpy(), b['y'].numpy()) for b in dl], norm


def _new_batches(norm_cfg, n_total=20):
    from read_3d_hdf5 import MultiH5Array
    from norm_utils import build_normalizer
    from data import FluidDataset, make_collate
    from torch.utils.data import DataLoader

    src = MultiH5Array([_H5], global_start=0, global_stop=n_total)
    n_train = int(n_total * 0.8)
    norm = build_normalizer(norm_cfg, _GRID)
    if norm.name in ('etau75_eta_3d', 'etau75_eta_shape_3d'):
        norm.fit_lazy(src, n_train, _XY, _ETA, _TAU, chunk_size=8)
    ds = FluidDataset(src, np.arange(n_train), _XY, _ETA, _TAU)
    T = ds[0].shape[-1]
    dl = DataLoader(ds, batch_size=4, shuffle=False,
                    collate_fn=make_collate(norm, T, 'torch'))
    return [(b['x'].numpy(), b['y'].numpy()) for b in dl], norm


@pytest.mark.skipif(not os.path.exists(_H5), reason='sample .h5 not present')
@pytest.mark.parametrize('name', ['none', 'etau75_3d', 'etau75_eta_3d', 'etau75_eta_shape_3d'])
def test_loader_parity_vs_legacy(name):
    legacy, lnorm = _legacy_batches({'name': name})
    new, nnorm = _new_batches({'name': name})
    assert len(legacy) == len(new) and len(new) > 0
    for (lx, ly), (nx_, ny_) in zip(legacy, new):
        assert lx.shape == nx_.shape
        np.testing.assert_allclose(lx, nx_, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(ly, ny_, rtol=1e-6, atol=1e-6)
    if lnorm.ref is not None:
        np.testing.assert_allclose(lnorm.ref, nnorm.ref, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# §9/§15  fork-safety: multi-worker batches identical to single-process
# ---------------------------------------------------------------------------
def _val_batches(num_workers, paths, n_total=30):
    from read_3d_hdf5 import MultiH5Array
    from data import build_dataloaders
    src = MultiH5Array(paths, global_start=0, global_stop=n_total)
    tr, te, norm = build_dataloaders(
        src, n_total=n_total, train_ratio=0.5,
        xyslice=_XY, etaslice=_ETA, tauslice=_TAU,
        norm_cfg={'name': 'etau75_eta_3d'}, grid_params=_GRID,
        backend='torch', batch_size=4, test_batch_sizes=[5],
        num_workers=num_workers, norm_chunk_size=8)
    res = next(iter(te))
    return [(b['x'].numpy(), b['y'].numpy()) for b in te[res]]


@pytest.mark.skipif(not (os.path.exists(_H5) and os.path.exists(_H5b)),
                    reason='both sample .h5 files required for multi-file fork test')
def test_fork_safety_multiworker_matches_single():
    paths = [_H5, _H5b]
    b0 = _val_batches(0, paths)
    b4 = _val_batches(4, paths)
    assert len(b0) == len(b4) and len(b0) > 0
    for (x0, y0), (x4, y4) in zip(b0, b4):
        np.testing.assert_array_equal(x0, x4)
        np.testing.assert_array_equal(y0, y4)


# ---------------------------------------------------------------------------
# §11  downsample auto-bind (coarse grid wired without manual recompute)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(_H5), reason='sample .h5 not present')
def test_downsample_autobind_shapes_and_grid():
    from read_3d_hdf5 import MultiH5Array
    from downsample import downsampled_grid_params, parse_downsample_cfg
    from data import build_dataloaders

    n_total = 16
    src = MultiH5Array([_H5], global_start=0, global_stop=n_total)
    dcfg = {'spatial_factor': 2, 'tau_stride': 2}
    tr, te, norm = build_dataloaders(
        src, n_total=n_total, train_ratio=0.75,
        xyslice=_XY, etaslice=_ETA, tauslice=_TAU,
        norm_cfg={'name': 'etau75_eta_3d'}, grid_params=_GRID,
        backend='torch', batch_size=2, test_batch_sizes=[2],
        num_workers=0, downsample_cfg=dcfg, norm_chunk_size=4)

    # normalizer must have been built on the *coarse* grid (auto-bound)
    coarse = downsampled_grid_params(_GRID, **parse_downsample_cfg(dcfg))
    assert norm.dtau == pytest.approx(coarse['dtau'])
    assert norm.deta == pytest.approx(coarse['deta'])
    # a batch is produced and is finite
    b = next(iter(tr))
    assert b['x'].ndim == 6 and np.isfinite(b['x'].numpy()).all()
