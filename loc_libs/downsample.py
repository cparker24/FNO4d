'''
Spatial (and temporal) downsampling for FNO4d fluid-simulation data.

Data layout — identical to read_3d_data() / read_3d_data_hdf5():

    arr.shape = (nevents, nFeatures, nx, ny, neta, ntau)
                    |         |       \\__\\____\\__ spatial axes  (block-reduced here)
                    |         \\__ feature / channel axis        (never touched)
                    \\__ event axis                              (never touched)
    ntau  (last axis) is the time / tau axis                    (optionally strided)

The three spatial axes (nx, ny, neta) sit at positions (-4, -3, -2) and tau at
(-1) for BOTH the full 6-D batch and a single 5-D event
(nFeatures, nx, ny, neta, ntau), so the *same* downsample() call works in the
offline file-writer and inside the DataLoader's per-event path.

Two things this module does
---------------------------
1. Spatial coarsening (nx, ny, neta) by an integer factor, with block-mean
   ('mean') or strided pick ('subsample').  Production grids have ODD spatial
   extents (nx=ny=65, neta=33) that a factor of 2/3/... does not divide evenly,
   so each axis is first made divisible by either
       boundary='crop'  crop down to a multiple        (default; no fake data,
                        drops a few boundary cells, e.g. 65 -> 64 -> 32)
       boundary='crop'   pad up to the next multiple   (keeps the full extent by
                        adding boundary cells -> "padded downsampled")

2. Temporal skipping (tau): keep every `tau_stride`-th step with NO averaging
   and NO interpolation, which simply increases the effective dtau by the
   stride (dtau -> tau_stride * dtau).

X-SCAPE-style interpolation (optional)
--------------------------------------
For *resampling* (as opposed to anti-aliased coarsening) the module also offers
the exact scheme X-SCAPE's EvolutionHistory uses to read the hydro grid
(src/framework/{LinearInterpolation.h, FluidEvolutionHistory.cc}):

  * tau_interp='linear'  — linear interpolation between bracketing tau steps
                           (Jetscape::LinearInt); allows a fractional tau_stride.
  * spatial_target=(NX,NY,NETA) — trilinear resample of nx/ny/neta to an
                           arbitrary shape over the same extent (TrilinearInt).

Both clamp at the edges (no extrapolation) and act per field/channel, matching
X-SCAPE.  Prefer block-mean for plain down-resolution; use these to re-grid to a
non-integer-factor resolution or a different dtau consistently with X-SCAPE.

Two entry points
----------------
A. Make a new (padded/cropped) downsampled HDF5 file — offline, once:

       from downsample import write_downsampled_h5
       write_downsampled_h5('data/dAu_25ev_mb.h5',  'data/dAu_25ev_mb_ds2.h5',
                            spatial_factors=2, tau_stride=2, boundary='crop')
       write_downsampled_h5('data/dAu_25ev_mb.root', 'data/dAu_25ev_mb_ds2.h5',
                            spatial_factors=2)

   or from the command line:

       python downsample.py data/dAu_25ev_mb.h5 -o data/dAu_25ev_mb_ds2.h5 \
              --spatial-factor 2 --tau-stride 2 --boundary crop

B. Downsample on the fly in the DataLoader — no new file written:

       from read_3d_hdf5 import read_3d_data_hdf5
       from fluid_datasets import FluidDatasetH5
       from downsample import DownsampledArrayView, downsampled_grid_params

       dat  = read_3d_data_hdf5('data/dAu_25ev_mb.h5', lazy=True)
       view = DownsampledArrayView(dat['arr'], spatial_factors=2, tau_stride=2)
       # rebuild the Normalizer at the new resolution (deta/eta_min/dtau change):
       grid = {k: dat[k] for k in ('tau_min','dtau','eta_min','deta') if k in dat}
       new_grid = downsampled_grid_params(grid, spatial_factors=2, tau_stride=2)
       ds = FluidDatasetH5(view, indices=train_idx, normalizer=norm)

See also read_3d_root.coarsen_array() — the original divisible-only block-mean
primitive that this module generalises (pad/crop + tau stride + lazy view).
'''

import os
import sys
import gc

import numpy as np

__all__ = [
    'SPATIAL_AXES',
    'coarsen_axis',
    'skip_tau',
    'interp_tau',
    'downsample_spatial',
    'resample_spatial',
    'downsample',
    'downsampled_grid_params',
    'parse_downsample_cfg',
    'DownsampledArrayView',
    'write_downsampled_h5',
]

# Spatial axes for the standard (..., nx, ny, neta, ntau) layout — valid for
# both the 6-D batch and a single 5-D event.
SPATIAL_AXES = (-4, -3, -2)
TAU_AXIS = -1

# Scalar grid keys carried over to the output file (recomputed where needed).
_SCALAR_KEYS = [
    'nFeatures', 'nx', 'ny', 'neta',
    'x_min', 'y_min', 'eta_min',
    'dx', 'dy', 'deta',
    'tau_min', 'tau_min_MUSIC', 'dtau',
]

# (count, spacing, origin) triples for the three spatial axes, in axis order.
_AXIS_GRID_KEYS = (
    ('nx',   'dx',   'x_min'),
    ('ny',   'dy',   'y_min'),
    ('neta', 'deta', 'eta_min'),
)


# ---------------------------------------------------------------------------
# factor helpers
# ---------------------------------------------------------------------------
def _as_factor_triple(factors):
    '''Normalise `factors` to a (fx, fy, feta) int tuple.

    A scalar is broadcast to all three spatial axes; a length-3 iterable is
    used as-is.  Use 1 on an axis to leave it untouched (e.g. (2, 2, 1) keeps
    the eta resolution while halving x and y).
    '''
    if np.isscalar(factors):
        return (int(factors),) * 3
    f = tuple(int(x) for x in factors)
    if len(f) != 3:
        raise ValueError(
            f"spatial factors must be a scalar or length-3 (fx, fy, feta); got {factors!r}")
    return f


# ---------------------------------------------------------------------------
# single-axis primitives
# ---------------------------------------------------------------------------
def coarsen_axis(arr, axis, factor, reduce='mean', boundary='crop',
                 pad_mode='edge', side='end'):
    '''Reduce `arr` along one `axis` by an integer `factor`.

    Parameters
    ----------
    reduce   : 'mean'      block-average each group of `factor` cells (default).
               'subsample' keep every `factor`-th cell (pure stride, no fill).
    boundary : how to make the axis length divisible by `factor` when it is not
               (only used by reduce='mean'):
               'crop'  crop down to the previous multiple (default; drops cells).
               'pad'   pad up to the next multiple with `pad_mode`.
    pad_mode : numpy.pad mode for boundary='crop' — 'edge' (replicate, default),
               'reflect', 'zero'/'zeros'/'constant' (zero-fill), ...
    side     : where the pad/crop cells go — 'end' (default, append/trim at the
               far edge) or 'both' (split front = amount // 2).

    `factor == 1` returns `arr` unchanged.
    '''
    if factor == 1:
        return arr
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    axis = axis % arr.ndim
    L = arr.shape[axis]

    if reduce == 'subsample':
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(None, None, factor)
        return arr[tuple(sl)]

    if reduce != 'mean':
        raise ValueError(f"reduce must be 'mean' or 'subsample', got {reduce!r}")

    if pad_mode in ('zero', 'zeros'):
        pad_mode = 'constant'

    rem = L % factor
    if rem:
        if boundary == 'crop':
            drop  = rem
            front = drop // 2 if side == 'both' else 0
            back  = drop - front
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(front, L - back)
            arr = arr[tuple(sl)]
        elif boundary == 'pad':
            pad   = factor - rem
            front = pad // 2 if side == 'both' else 0
            back  = pad - front
            pad_width = [(0, 0)] * arr.ndim
            pad_width[axis] = (front, back)
            arr = np.pad(arr, pad_width, mode=pad_mode)
        else:
            raise ValueError(f"boundary must be 'pad' or 'crop', got {boundary!r}")

    nblk = arr.shape[axis] // factor
    grouped = list(arr.shape)
    grouped[axis] = nblk
    grouped.insert(axis + 1, factor)
    return arr.reshape(grouped).mean(axis=axis + 1)


def skip_tau(arr, stride, axis=TAU_AXIS):
    '''Keep every `stride`-th tau step — no averaging, no interpolation.

    Equivalent to arr[..., ::stride] on the tau axis; effectively multiplies
    dtau by `stride`.  `stride == 1` returns `arr` unchanged.
    '''
    if stride == 1:
        return arr
    if stride < 1:
        raise ValueError(f"tau_stride must be >= 1, got {stride}")
    axis = axis % arr.ndim
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(None, None, stride)
    return arr[tuple(sl)]


# ---------------------------------------------------------------------------
# linear (X-SCAPE-style) interpolation primitives
# ---------------------------------------------------------------------------
# These mirror Jetscape::LinearInt / TrilinearInt in X-SCAPE's
# src/framework/{LinearInterpolation.h, FluidEvolutionHistory.cc}: linear weights
# with the lower index taken by truncation and the upper neighbour CLAMPED to the
# last cell (edge value, no extrapolation), applied independently per field/channel.
def _lerp_axis(arr, axis, src_pos):
    '''Linearly interpolate `arr` along `axis` at fractional source indices.

    `src_pos` is a 1-D array of positions in *index units* of `axis`.  Positions
    are clamped to [0, L-1] (edge clamp, no extrapolation) and each output sample
    is `(1-f)*arr[i0] + f*arr[i1]` with `i0 = floor(pos)`, `i1 = min(i0+1, L-1)`,
    `f = pos - i0` — exactly X-SCAPE's `LinearInt`.  Returns an array whose `axis`
    length is `len(src_pos)`.
    '''
    axis = axis % arr.ndim
    L = arr.shape[axis]
    pos = np.clip(np.asarray(src_pos, dtype=np.float64), 0.0, L - 1.0)
    i0 = np.floor(pos).astype(np.intp)
    i1 = np.minimum(i0 + 1, L - 1)
    a0 = np.take(arr, i0, axis=axis)
    a1 = np.take(arr, i1, axis=axis)
    fshape = [1] * arr.ndim
    fshape[axis] = pos.shape[0]
    f = (pos - i0).reshape(fshape)
    if np.issubdtype(arr.dtype, np.floating):
        f = f.astype(arr.dtype)
    return (1.0 - f) * a0 + f * a1


def interp_tau(arr, stride, axis=TAU_AXIS):
    '''Linearly re-grid the tau axis to spacing `stride`*dtau (X-SCAPE LinearInt).

    Unlike skip_tau() this *interpolates* between bracketing stored steps, so
    `stride` may be fractional (e.g. 1.5).  tau_min is preserved; the new step
    count is `floor((L-1)/stride) + 1` (all new steps stay within the original
    tau span, so no extrapolation).  `stride == 1` returns `arr` unchanged.
    '''
    if stride == 1:
        return arr
    if stride <= 0:
        raise ValueError(f"tau stride must be > 0, got {stride}")
    axis = axis % arr.ndim
    L = arr.shape[axis]
    M = int(np.floor((L - 1) / stride)) + 1
    return _lerp_axis(arr, axis, np.arange(M) * float(stride))


# ---------------------------------------------------------------------------
# array-level downsampling
# ---------------------------------------------------------------------------
def downsample_spatial(arr, factors=2, spatial_axes=SPATIAL_AXES,
                       reduce='mean', boundary='crop', pad_mode='edge', side='end'):
    '''Coarsen the three spatial axes (nx, ny, neta) of `arr`.

    Works on the full 6-D batch (nevents, nFeatures, nx, ny, neta, ntau) and on
    a single 5-D event (nFeatures, nx, ny, neta, ntau): the default
    spatial_axes=(-4, -3, -2) selects nx/ny/neta in both cases.

    `factors` is a scalar (same factor for all three) or (fx, fy, feta); use 1
    to skip an axis.  Remaining args are forwarded to coarsen_axis().
    '''
    fx = _as_factor_triple(factors)
    out = np.ascontiguousarray(arr)
    for axis, factor in zip(spatial_axes, fx):
        out = coarsen_axis(out, axis, factor, reduce=reduce, boundary=boundary,
                           pad_mode=pad_mode, side=side)
    return out


def resample_spatial(arr, target_shape, spatial_axes=SPATIAL_AXES):
    '''Trilinear-resample the spatial axes (nx, ny, neta) to `target_shape`.

    This is X-SCAPE's `TrilinearInt` (done as separable linear interpolation per
    axis, which is identical): the target grid spans the SAME physical extent as
    the source, with endpoints inclusive — i.e. it maps the first/last cells onto
    themselves and linearly interpolates in between.  Each field/channel is
    interpolated independently.  Use it to resample to an arbitrary resolution
    (e.g. a power-of-two grid that is not an integer divisor of nx).

    Unlike downsample_spatial(reduce='mean'), this does NOT area-average, so it is
    a *resampler*, not an anti-aliasing coarsener — prefer block-mean when simply
    reducing resolution.

    target_shape : (NX, NY, NETA); use None for an axis to leave it unchanged.
    '''
    out = np.ascontiguousarray(arr)
    for axis, N in zip(spatial_axes, target_shape):
        if N is None:
            continue
        N = int(N)
        if N < 1:
            raise ValueError(f"target axis size must be >= 1, got {N}")
        ax = axis % out.ndim
        L = out.shape[ax]
        if N == L:
            continue
        pos = np.zeros(N) if (L == 1 or N == 1) else np.linspace(0.0, L - 1.0, N)
        out = _lerp_axis(out, ax, pos)
    return out


def downsample(arr, spatial_factors=1, tau_stride=1, reduce='mean',
               boundary='crop', pad_mode='edge', side='end',
               tau_interp='skip', spatial_target=None,
               spatial_axes=SPATIAL_AXES, tau_axis=TAU_AXIS):
    '''Combined spatial + temporal downsampling in one call.

    Spatial — pick one:
      spatial_factors : scalar or (fx, fy, feta)  — block coarsen (1 = off),
                        with `reduce`/`boundary`/`pad_mode`/`side` (the default).
      spatial_target  : (NX, NY, NETA)            — trilinear-resample to this
                        shape instead (X-SCAPE TrilinearInt); overrides
                        spatial_factors/reduce/boundary when given.

    Temporal:
      tau_stride : keep/resample tau by this stride (1 = off).
      tau_interp : 'skip'   strided pick, no interpolation (integer stride only).
                   'linear' X-SCAPE LinearInt between steps (fractional stride OK).
    '''
    if spatial_target is not None:
        out = resample_spatial(arr, spatial_target, spatial_axes=spatial_axes)
    else:
        out = downsample_spatial(arr, spatial_factors, spatial_axes=spatial_axes,
                                 reduce=reduce, boundary=boundary,
                                 pad_mode=pad_mode, side=side)
    if tau_interp == 'linear':
        out = interp_tau(out, tau_stride, axis=tau_axis)
    elif tau_interp == 'skip':
        s = int(tau_stride)
        if s != tau_stride:
            raise ValueError("tau_stride must be an integer when tau_interp='skip'; "
                             "use tau_interp='linear' for fractional strides")
        out = skip_tau(out, s, axis=tau_axis)
    else:
        raise ValueError(f"tau_interp must be 'skip' or 'linear', got {tau_interp!r}")
    return out


# ---------------------------------------------------------------------------
# grid-parameter bookkeeping
# ---------------------------------------------------------------------------
def downsampled_grid_params(grid, spatial_factors=1, tau_stride=1,
                            reduce='mean', boundary='crop', side='end',
                            pad_mode='edge', tau_interp='skip',
                            spatial_target=None):
    '''Return a copy of `grid` describing the coarsened grid.

    Updates nx/ny/neta, dx/dy/deta, x_min/y_min/eta_min for the spatial
    coarsening and dtau / choose_ntau for tau skipping.  tau_min is unchanged
    (the first tau step is always kept).  Each value is updated only if its key
    is present, so this works both on a full grid (h5 .attrs) and on the reduced
    Normalizer grid dict {tau_min, dtau, eta_min, deta} — exactly what you pass
    to build_normalizer() at the downsampled resolution.

    `pad_mode` is accepted (and ignored — grid coordinates do not depend on the
    fill values) so the same kwargs can be forwarded here and to
    DownsampledArrayView / write_downsampled_h5.

    Note: when a spacing/origin is present but its count key is NOT (the reduced
    Normalizer dict), the origin shift is exact for side='end' (the default);
    for side='both' it falls back to the 'end' offset.
    '''
    out = dict(grid)

    if spatial_target is not None:
        # trilinear resample to an explicit shape over the same physical extent
        # (endpoints inclusive): count -> N, spacing -> d*(L-1)/(N-1), origin kept.
        for (n_key, d_key, min_key), N in zip(_AXIS_GRID_KEYS, spatial_target):
            if N is None or n_key not in grid:
                continue
            L = int(grid[n_key]); N = int(N)
            out[n_key] = N
            if d_key in grid and L > 1 and N > 1:
                out[d_key] = float(grid[d_key]) * (L - 1) / (N - 1)
        fx = (1, 1, 1)                       # skip the block-coarsen loop below
    else:
        fx = _as_factor_triple(spatial_factors)

    for (n_key, d_key, min_key), factor in zip(_AXIS_GRID_KEYS, fx):
        if factor == 1:
            continue
        have_count = n_key in grid
        d = float(grid[d_key]) if d_key in grid else None

        if reduce == 'subsample':
            if have_count:
                out[n_key] = -(-int(grid[n_key]) // factor)   # ceil(L / factor)
            if d is not None:
                out[d_key] = factor * d
            # origin unchanged: the first cell of each stride is kept
            continue

        # reduce == 'mean'
        origin_shift = 0          # in cells, relative to the original origin
        if have_count:
            L   = int(grid[n_key])
            rem = L % factor
            if boundary == 'crop':
                front = (rem // 2) if side == 'both' else 0
                out[n_key]   = (L - rem) // factor
                origin_shift = front            # cropped origin moves +front cells
            else:  # pad
                pad   = (factor - rem) % factor
                front = (pad // 2) if side == 'both' else 0
                out[n_key]   = (L + pad) // factor
                origin_shift = -front           # padded origin moves -front cells

        if d is not None:
            out[d_key] = factor * d
            if min_key in grid:
                # new cell centre = shifted origin + half-block offset
                out[min_key] = (float(grid[min_key])
                                + origin_shift * d
                                + (factor - 1) / 2.0 * d)

    # tau bookkeeping is identical for 'skip' and 'linear' (both keep tau_min,
    # scale dtau by the stride, and yield floor((L-1)/stride)+1 steps); tau_interp
    # is accepted only so the same kwargs forward here and to downsample().
    del tau_interp
    if tau_stride and tau_stride != 1:
        if 'dtau' in grid:
            out['dtau'] = float(grid['dtau']) * tau_stride
        if 'choose_ntau' in grid:
            Lt = int(grid['choose_ntau'])
            out['choose_ntau'] = int(np.floor((Lt - 1) / tau_stride)) + 1

    return out


def parse_downsample_cfg(cfg):
    '''Normalise a YAML 'downsample' mapping into downsample() kwargs, or None.

    Mirrors the `_slice(cfg.get('xyslice', ...))` pattern in the training
    scripts: turns a small config block into ready-to-forward kwargs that both
    DownsampledArrayView and downsampled_grid_params accept verbatim.

    Recognised keys (all optional)::

        downsample:
          spatial_factor:  2          # int  -> nx, ny, neta share one factor
          spatial_factors: [2, 2, 1]  # list -> per axis (overrides spatial_factor)
          spatial_target:  [32,32,16] # trilinear-resample to this shape instead
                                      #   (use 0/null to keep an axis); overrides
                                      #   spatial_factor(s)/reduce/boundary
          tau_stride:      1          # tau stride (K=1 = off; may be fractional
                                      #   when tau_interp: linear)
          tau_interp:      skip       # skip (decimate) | linear (X-SCAPE LinearInt)
          reduce:          mean       # mean | subsample
          boundary:        crop       # crop | pad
          pad_mode:        edge       # [boundary: pad] edge | reflect | zero | ...
          side:            end        # end  | both

    Returns None when `cfg` is falsy or a no-op (no spatial change and tau_stride
    1), so callers can write::

        downsample_kw = parse_downsample_cfg(cfg.get('downsample'))
        if downsample_kw:
            arr         = DownsampledArrayView(arr, **downsample_kw)
            grid_params = downsampled_grid_params(grid_params, **downsample_kw)
    '''
    if not cfg:
        return None
    sf_list = cfg.get('spatial_factors')
    if sf_list is not None:
        spatial_factors = tuple(int(x) for x in sf_list)
    else:
        spatial_factors = int(cfg.get('spatial_factor', 1))

    st = cfg.get('spatial_target')
    if st is not None:
        # 0 / null on an axis means "leave unchanged"
        spatial_target = tuple(None if (v in (0, None, 'null')) else int(v) for v in st)
        if all(v is None for v in spatial_target):
            spatial_target = None

    else:
        spatial_target = None

    tau_stride = cfg.get('tau_stride', 1)
    tau_stride = int(tau_stride) if float(tau_stride).is_integer() else float(tau_stride)
    tau_interp = cfg.get('tau_interp', 'skip')

    no_spatial = (spatial_target is None and
                  all(f == 1 for f in _as_factor_triple(spatial_factors)))
    if no_spatial and tau_stride == 1:
        return None

    return dict(
        spatial_factors=spatial_factors,
        spatial_target=spatial_target,
        tau_stride=tau_stride,
        tau_interp=tau_interp,
        reduce=cfg.get('reduce', 'mean'),
        boundary=cfg.get('boundary', 'crop'),
        pad_mode=cfg.get('pad_mode', 'edge'),
        side=cfg.get('side', 'end'),
    )


# ---------------------------------------------------------------------------
# lazy DataLoader view
# ---------------------------------------------------------------------------
class DownsampledArrayView:
    '''Lazy, drop-in array view that downsamples on read.

    Wraps any base object exposing len(), .shape, .dtype and __getitem__
    returning numpy arrays (an ndarray, a read_3d_hdf5.MultiH5Array, ...).  Each
    int / slice access reads exactly the requested events from the base and
    returns them downsampled, so it plugs straight into FluidDatasetH5 without
    materialising a second copy of the dataset:

        view = DownsampledArrayView(dat['arr'], spatial_factors=2, tau_stride=2)
        ds   = FluidDatasetH5(view, indices=idx, xyslice=..., normalizer=norm)

    base[int]  -> a single downsampled 5-D event   (nFeatures, nx', ny', neta', ntau')
    base[a:b]  -> a downsampled 6-D batch

    The downsampling parameters mirror downsample().  If a Normalizer depends on
    deta / eta_min / dtau, build it from downsampled_grid_params(grid, ...).
    '''

    def __init__(self, base, spatial_factors=2, tau_stride=1, reduce='mean',
                 boundary='crop', pad_mode='edge', side='end',
                 tau_interp='skip', spatial_target=None):
        self._base = base
        self._kw = dict(spatial_factors=spatial_factors, tau_stride=tau_stride,
                        reduce=reduce, boundary=boundary, pad_mode=pad_mode,
                        side=side, tau_interp=tau_interp,
                        spatial_target=spatial_target)
        if len(base) == 0:
            raise ValueError("base array is empty; nothing to downsample")
        probe = self._down(base[0])           # single 5-D event
        self._item_shape = probe.shape
        self._dtype = probe.dtype

    def _down(self, a):
        return downsample(np.asarray(a), **self._kw)

    @property
    def shape(self):
        return (len(self._base),) + self._item_shape

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return len(self._base)

    def __repr__(self):
        return (f"DownsampledArrayView(shape={self.shape}, dtype={self.dtype}, "
                f"{self._kw})")

    def __getitem__(self, idx):
        # base[int] -> 5-D event -> downsample -> 5-D event
        # base[slice]/array -> 6-D batch -> downsample -> 6-D batch
        return self._down(self._base[idx])

    def __array__(self, dtype=None, copy=None):  # noqa: ARG002
        out = self._down(self._base[:])
        return out if dtype is None else out.astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# offline file writer:  h5 / root  ->  padded/cropped downsampled h5
# ---------------------------------------------------------------------------
def _load_source(paths, nevents, verbose):
    '''Load source metadata + an indexable `arr` from .h5 or .root path(s).

    Returns the `dat` dict from read_3d_data (root, eager ndarray) or
    read_3d_data_hdf5 (h5, lazy MultiH5Array).  All paths must be the same
    format and share one grid, matching read_3d_data*() semantics.
    '''
    # make sibling loc_libs modules importable whether this is imported as
    # `downsample`, `loc_libs.downsample`, or run as a script.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    is_root = any(str(p).endswith('.root') for p in paths)
    arg = paths[0] if len(paths) == 1 else paths
    if is_root:
        try:
            from read_3d_root import read_3d_data
        except ImportError:
            from loc_libs.read_3d_root import read_3d_data
        return read_3d_data(arg, nevents=nevents, verbose=verbose)
    try:
        from read_3d_hdf5 import read_3d_data_hdf5
    except ImportError:
        from loc_libs.read_3d_hdf5 import read_3d_data_hdf5
    return read_3d_data_hdf5(arg, nevents=nevents, lazy=True, verbose=verbose)


def write_downsampled_h5(src, dst, spatial_factors=2, tau_stride=1, reduce='mean',
                         boundary='crop', pad_mode='edge', side='end',
                         tau_interp='skip', spatial_target=None,
                         nevents=-1, chunk_events=8, compression='lzf',
                         force=False, verbose=False):
    '''Read h5/root source(s), spatially/temporally downsample, write a new h5.

    The output mirrors the convert_root_to_hdf5() / merge layout: a chunked,
    compressed `arr` dataset plus ntau_freezeout / tau_freezeout and grid
    parameters as attributes — so the result is a drop-in for read_3d_data_hdf5,
    MultiH5Array, FluidDatasetH5, etc., just at the coarser resolution.

    Parameters
    ----------
    src             : str | list[str]  — .h5 or .root source path(s) (one grid).
    dst             : str              — output .h5 path.
    spatial_factors : scalar or (fx, fy, feta) spatial coarsening factor(s).
    spatial_target  : (NX, NY, NETA) or None — trilinear-resample to this shape
                      instead of block-coarsening (overrides spatial_factors).
    tau_stride      : tau stride (1 = no time downsample; fractional allowed when
                      tau_interp='linear').
    tau_interp      : 'skip' (decimate) | 'linear' (X-SCAPE LinearInt in tau).
    reduce, boundary, pad_mode, side : see downsample() / coarsen_axis().
    nevents         : int   — cap events (-1 = all).
    chunk_events    : int   — events processed & written per step (memory bound).
    compression     : str   — HDF5 compression for `arr` (e.g. 'lzf', 'gzip').
    force           : bool  — overwrite dst if it exists.

    Extra provenance is stored as attributes (downsample_*, orig_nx/ny/neta).
    Returns the output path.
    '''
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    paths = [str(src)] if isinstance(src, (str, os.PathLike)) else [str(p) for p in src]
    if os.path.exists(dst) and not force:
        raise FileExistsError(f"{dst} already exists (pass force=True to overwrite)")

    factors_t = _as_factor_triple(spatial_factors)
    kw = dict(spatial_factors=spatial_factors, spatial_target=spatial_target,
              tau_stride=tau_stride, tau_interp=tau_interp, reduce=reduce,
              boundary=boundary, pad_mode=pad_mode, side=side)

    # ---- load source ------------------------------------------------------
    dat  = _load_source(paths, nevents, verbose)
    base = dat['arr']
    n    = len(base)
    if n == 0:
        raise ValueError("no events loaded from source")

    # ---- grid params: original -> downsampled -----------------------------
    grid     = {k: dat[k] for k in _SCALAR_KEYS if k in dat}
    new_grid = downsampled_grid_params(grid, spatial_factors=spatial_factors,
                                       spatial_target=spatial_target,
                                       tau_stride=tau_stride, tau_interp=tau_interp,
                                       reduce=reduce, boundary=boundary, side=side)

    # ---- probe output shape ----------------------------------------------
    sample      = downsample(np.asarray(base[0:1]), **kw)
    item_shape  = sample.shape[1:]                      # (nFeatures, nx', ny', neta', ntau')
    ntau_new    = int(item_shape[-1])
    final_shape = (n,) + item_shape
    chunk_shape = (min(chunk_events, n),) + item_shape

    print(f"Downsampling {n} events  ->  {dst}")
    if spatial_target is not None:
        print(f"  spatial: trilinear resample (X-SCAPE) to target {tuple(spatial_target)}")
    else:
        print(f"  spatial factors (nx,ny,neta) = {factors_t}   reduce={reduce}   "
              f"boundary={boundary}" + (f"  pad_mode={pad_mode}" if boundary == 'pad' else "")
              + f"  side={side}")
    print(f"  tau_stride = {tau_stride}  (interp={tau_interp})"
          + (f"  dtau {float(grid.get('dtau', 0)):.4g} -> {float(new_grid.get('dtau', 0)):.4g}"
             if 'dtau' in grid and tau_stride != 1 else ""))
    print(f"  spatial grid: ({grid.get('nx')},{grid.get('ny')},{grid.get('neta')})"
          f" -> ({new_grid.get('nx')},{new_grid.get('ny')},{new_grid.get('neta')})")
    print(f"  ntau: {grid.get('choose_ntau', base.shape[-1])} -> {ntau_new}")
    print(f"  output arr shape {final_shape}  chunk {chunk_shape}"
          f"  (~{np.prod(chunk_shape) * 4 / 1024**2:.1f} MiB/chunk)")

    # ---- per-event metadata ----------------------------------------------
    ntau_fo = np.asarray(dat['ntau_freezeout'])[:n].astype(np.int64)
    if tau_stride != 1:                                 # active steps -> floor((fo-1)/stride)+1
        ntau_fo = np.maximum(np.floor((ntau_fo - 1) / tau_stride).astype(np.int64) + 1, 0)
    tau_fo = np.asarray(dat['tau_freezeout'])[:n]       # physical freeze-out time: unchanged

    # ---- write ------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(dst))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(dst, 'w') as hf:
        for k in _SCALAR_KEYS:                          # recomputed grid scalars
            if k in new_grid:
                hf.attrs[k] = new_grid[k]
        hf.attrs['choose_ntau'] = ntau_new
        hf.attrs['nevents']     = int(n)

        # provenance / downsample metadata
        hf.attrs['downsample_tau_stride'] = float(tau_stride)
        hf.attrs['downsample_tau_interp'] = str(tau_interp)
        if spatial_target is not None:
            hf.attrs['downsample_spatial_method'] = 'trilinear'
            hf.attrs['downsample_spatial_target'] = np.asarray(
                [int(item_shape[i + 1]) for i in range(3)], dtype=np.int32)
        else:
            hf.attrs['downsample_spatial_method']  = str(reduce)
            hf.attrs['downsample_spatial_factors'] = np.asarray(factors_t, dtype=np.int32)
            hf.attrs['downsample_boundary']        = str(boundary)
            hf.attrs['downsample_side']            = str(side)
            if boundary == 'pad':
                hf.attrs['downsample_pad_mode'] = str(pad_mode)
        for (nk, _, _) in _AXIS_GRID_KEYS:
            if nk in grid:
                hf.attrs[f'orig_{nk}'] = int(grid[nk])
        hf.attrs['source_files'] = [os.path.abspath(p) for p in paths]

        hf.create_dataset('ntau_freezeout', data=ntau_fo.astype(np.int32))
        hf.create_dataset('tau_freezeout',  data=tau_fo)

        ds = hf.create_dataset('arr', shape=final_shape, dtype=np.float32,
                               chunks=chunk_shape, compression=compression)

        for start in range(0, n, chunk_events):
            stop  = min(start + chunk_events, n)
            chunk = np.asarray(base[start:stop])        # reads only this event range
            ds[start:stop] = downsample(chunk, **kw).astype(np.float32, copy=False)
            print(f"  events {start}-{stop} written")
            del chunk
            gc.collect()

    size_mb = os.path.getsize(dst) / 1024**2
    print(f"Done. Wrote {dst}  ({size_mb:.1f} MiB)")
    return dst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Spatially/temporally downsample FNO4d h5/root data into a "
                    "new padded-or-cropped HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('inputs', nargs='+', metavar='INPUT',
                   help='.h5 or .root source file(s) (same grid; treated as one dataset)')
    p.add_argument('-o', '--output', required=True, metavar='OUT.h5',
                   help='output .h5 path')

    g = p.add_mutually_exclusive_group()
    g.add_argument('--spatial-factor', type=int, default=None, metavar='F',
                   help='single spatial factor applied to nx, ny and neta')
    g.add_argument('--spatial-factors', type=int, nargs=3, default=None,
                   metavar=('FX', 'FY', 'FETA'),
                   help='per-axis spatial factors (use 1 to skip an axis)')
    g.add_argument('--spatial-target', type=int, nargs=3, default=None,
                   metavar=('NX', 'NY', 'NETA'),
                   help='trilinear-resample (X-SCAPE) to this target shape over the '
                        'same extent (use 0 to keep an axis); overrides --spatial-factor(s)')

    p.add_argument('--tau-stride', type=float, default=1, metavar='K',
                   help='tau stride (increases effective dtau; fractional allowed with '
                        '--tau-interp linear)')
    p.add_argument('--tau-interp', choices=['skip', 'linear'], default='skip',
                   help="skip (decimate, integer stride) or linear (X-SCAPE LinearInt)")
    p.add_argument('--reduce', choices=['mean', 'subsample'], default='mean',
                   help="spatial reduction: block-mean or strided pick")
    p.add_argument('--boundary', choices=['crop', 'pad'], default='crop',
                   help="make spatial dims divisible by cropping down (default) or padding up")
    p.add_argument('--pad-mode', default='edge', metavar='MODE',
                   help="numpy.pad mode when --boundary pad (edge|reflect|zero|...)")
    p.add_argument('--side', choices=['end', 'both'], default='end',
                   help="place pad/crop cells at the end, or split across both edges")

    p.add_argument('--nevents', type=int, default=-1, metavar='N',
                   help='cap events (-1 = all)')
    p.add_argument('--chunk-events', type=int, default=8, metavar='N',
                   help='events processed & written per step (memory bound)')
    p.add_argument('--compression', default='lzf',
                   help='HDF5 compression for the arr dataset')
    p.add_argument('--force', action='store_true', help='overwrite output if it exists')
    p.add_argument('--verbose', action='store_true', help='print read diagnostics')
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    spatial_target = None
    if args.spatial_target is not None:
        spatial_target = tuple(None if v == 0 else v for v in args.spatial_target)
        if all(v is None for v in spatial_target):
            spatial_target = None

    if args.spatial_factors is not None:
        spatial_factors = tuple(args.spatial_factors)
    elif args.spatial_factor is not None:
        spatial_factors = args.spatial_factor
    else:
        spatial_factors = 1                      # no block-coarsen (tau-only / target mode)

    tau_stride = int(args.tau_stride) if float(args.tau_stride).is_integer() else args.tau_stride

    no_spatial = (spatial_target is None and spatial_factors == 1)
    if no_spatial and tau_stride == 1:
        print("Nothing to do: set --spatial-factor(s)/--spatial-target and/or --tau-stride")
        return 2

    write_downsampled_h5(
        args.inputs if len(args.inputs) > 1 else args.inputs[0],
        args.output,
        spatial_factors=spatial_factors,
        spatial_target=spatial_target,
        tau_stride=tau_stride,
        tau_interp=args.tau_interp,
        reduce=args.reduce,
        boundary=args.boundary,
        pad_mode=args.pad_mode,
        side=args.side,
        nevents=args.nevents,
        chunk_events=args.chunk_events,
        compression=args.compression,
        force=args.force,
        verbose=args.verbose,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
