"""ArraySource protocol + lazy readers for the unified data pipeline.

This module formalises the contract that ``MultiH5Array``,
``DownsampledArrayView`` and a plain ``np.ndarray`` already satisfy, and adds a
lazy ROOT reader (:class:`RootSource`) and an optional Zarr reader
(:class:`ZarrSource`).  See ``DATA_PIPELINE_PLAN.md`` §4–§6.

An *ArraySource* is any object exposing::

    .shape : tuple[int, ...]   # (nevents, nFeatures, nx, ny, neta, ntau)
    .dtype : np.dtype
    __len__()                  # nevents
    __getitem__(idx)           # int -> 5-D event; slice -> 6-D batch

Optionally it may expose ``reset_handles()`` for multi-worker fork-safety
(see ``DATA_PIPELINE_PLAN.md`` §9).
"""

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# Re-export the existing lazy sources so callers have a single import location.
from read_3d_hdf5 import MultiH5Array          # noqa: F401
from downsample import DownsampledArrayView    # noqa: F401


@runtime_checkable
class ArraySource(Protocol):
    """Structural protocol for a lazy event array.

    Used for typing/documentation only — no inheritance is required.  ``int``
    indexing returns a single 5-D event ``(nFeatures, nx, ny, neta, ntau)``;
    slice indexing returns a 6-D batch ``(N, nFeatures, nx, ny, neta, ntau)``.
    """

    shape: tuple[int, ...]
    dtype: np.dtype

    def __len__(self) -> int: ...
    def __getitem__(self, idx): ...
    # optional, used for multi-worker safety (see DATA_PIPELINE_PLAN.md §9):
    # def reset_handles(self) -> None: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _range(idx, total):
    """Normalise an int/slice index into (start, stop, squeeze).

    int      -> (i, i+1, True)   so a single 5-D event is returned
    slice    -> (start, stop, False) with step == 1 only

    Mirrors the int/slice handling in MultiH5Array.__getitem__.
    """
    if isinstance(idx, (int, np.integer)):
        i = int(idx)
        if i < 0:
            i += total
        if not (0 <= i < total):
            raise IndexError(f"index {i} out of bounds for axis 0 with size {total}")
        return i, i + 1, True
    if isinstance(idx, slice):
        start, stop, step = idx.indices(total)
        if step != 1:
            raise ValueError("RootSource only supports unit-stride slices; "
                             "use a BlockBatchSampler for batched reads")
        return start, max(start, stop), False
    raise TypeError(f"unsupported index type {type(idx).__name__!r} for RootSource")


# ---------------------------------------------------------------------------
# build_source — one factory for .root and .h5 inputs
# ---------------------------------------------------------------------------
_H5_EXTS   = ('.h5', '.hdf5')
_ROOT_EXTS = ('.root',)


def _classify_paths(paths):
    """Return 'root' or 'h5' for a homogeneous path list (raises if mixed)."""
    kinds = set()
    for p in paths:
        suf = Path(str(p)).suffix.lower()
        if suf in _ROOT_EXTS:
            kinds.add('root')
        elif suf in _H5_EXTS:
            kinds.add('h5')
        else:
            raise ValueError(
                f"Cannot infer input type from extension: {p!r}. "
                f"Expected one of {_ROOT_EXTS + _H5_EXTS}.")
    if len(kinds) != 1:
        raise ValueError(f"Mixed input file types {kinds} — all paths must be "
                         "the same type (all .root or all .h5/.hdf5).")
    return kinds.pop()


def build_source(paths, *, storage_options=None, lazy=True, start_event=0,
                 nevents=-1, choose_ntau=None, verbose=False):
    """Build the right :class:`ArraySource` for ``.root`` or ``.h5`` inputs.

    The unified pipeline (``build_dataloaders``) accepts any ArraySource, so the
    only thing that differs between ROOT and HDF5 input is how the source is
    constructed — this factory hides that behind a single call.

    The file type is inferred from the extension (all paths must match):
      * ``.root``        -> :class:`RootSource` (lazy; supports ``gs://``
        streaming via ``storage_options``).
      * ``.h5``/``.hdf5`` -> ``read_3d_data_hdf5(..., lazy=lazy)`` ->
        :class:`MultiH5Array` (lazy) or a dense ``np.ndarray`` (eager).

    Parameters
    ----------
    paths           : str | list[str]
        Local path(s) or ``gs://`` URI(s).  ROOT streams remotely; HDF5 must be
        downloaded to local paths first (h5py limitation).
    storage_options : dict | None
        Forwarded to ``uproot.open`` for remote ROOT files.
    lazy            : bool
        HDF5 only — on-demand per-event reads (True) vs. load-all (False).
        Ignored for ROOT (always lazy).
    start_event     : int
        HDF5 only — first global event index across the concatenated files.
    nevents         : int
        HDF5 only — cap on events read (-1 = all).  For ROOT, cap via the
        ``n_total`` argument to ``build_dataloaders`` instead.
    choose_ntau     : int | None
        ROOT only — pad/clip the tau axis (None -> global max ntau_freezeout).
    verbose         : bool
        HDF5 only — forwarded to ``read_3d_data_hdf5``.

    Returns
    -------
    (source, grid_params) : tuple[ArraySource, dict]
        ``grid_params`` holds the ``tau_min, dtau, eta_min, deta`` (and, for
        HDF5, ``x_min, dx, y_min, dy``) consumed by the normalizer / downsample
        / coordinate helpers.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [str(paths)]
    else:
        paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("build_source requires at least one file path")

    kind = _classify_paths(paths)

    if kind == 'root':
        source = RootSource(paths, storage_options=storage_options,
                            choose_ntau=choose_ntau)
        return source, dict(source.grid_params)

    # h5 / hdf5
    from read_3d_hdf5 import read_3d_data_hdf5
    h5_arg = paths if len(paths) > 1 else paths[0]
    dat = read_3d_data_hdf5(h5_arg, nevents=nevents, start_event=start_event,
                            lazy=lazy, verbose=verbose)
    grid_params = {k: dat[k] for k in
                   ('tau_min', 'dtau', 'eta_min', 'deta', 'x_min', 'dx', 'y_min', 'dy')
                   if k in dat}
    return dat['arr'], grid_params


# ---------------------------------------------------------------------------
# RootSource — lazy ROOT reader (DATA_PIPELINE_PLAN.md §5)
# ---------------------------------------------------------------------------
class RootSource:
    """Lazy ArraySource over one or more variable-tau ``.root`` files.

    Reads only the requested entry range and pads/reshapes on the fly, reusing
    the proven assembly logic of ``read_3d_root.read_3d_data`` (the eager path)
    so a slice equals the matching slice of the eager array, including the
    pad-to-``choose_ntau``.

    Files (local or ``gs://`` / ``s3://``) are treated as one concatenated
    dataset, like :class:`MultiH5Array`.  ``choose_ntau`` defaults to the global
    maximum of ``ntau_freezeout`` across all files so padding is consistent.

    Parameters
    ----------
    file_paths      : str | list[str]
        Path(s) to the ``.root`` file(s).
    choose_ntau     : int | None
        Pad/clip the tau axis to this length.  None -> global max ntau_freezeout.
    storage_options : dict | None
        Forwarded to ``uproot.open`` for remote files (e.g. from
        ``gcs_utils.get_gcs_storage_options``).
    tree            : str
        TTree name (default ``"t"``).
    branch          : str
        Bulk branch name (default ``"user_res"``).
    """

    _SCALAR_KEYS = ('nFeatures', 'nx', 'ny', 'neta',
                    'x_min', 'y_min', 'eta_min',
                    'dx', 'dy', 'deta', 'tau_min', 'dtau')

    def __init__(self, file_paths, *, choose_ntau=None, storage_options=None,
                 tree='t', branch='user_res'):
        import uproot          # local import: heavy and optional
        self._uproot = uproot

        if isinstance(file_paths, (str, os.PathLike)):
            self._paths = [str(file_paths)]
        else:
            self._paths = [str(p) for p in file_paths]
        if not self._paths:
            raise ValueError("RootSource requires at least one file path")

        self._open_kw = {} if storage_options is None else {'storage_options': storage_options}
        self._tree_name = tree
        self._branch = branch

        # cheap metadata only — open each file, read scalar grid params (file0)
        # and per-file entry counts + ntau_freezeout (for the padding target).
        f0 = uproot.open(self._paths[0], **self._open_kw)
        scal = {k: float(f0[k].member('fVal')) for k in self._SCALAR_KEYS if k in f0}
        self._nx        = int(scal['nx'])
        self._ny        = int(scal['ny'])
        self._neta      = int(scal['neta'])
        self._nFeatures = int(scal['nFeatures'])
        # grid params consumed by build_normalizer / downsampled_grid_params
        self.grid_params = {k: scal[k] for k in ('tau_min', 'dtau', 'eta_min', 'deta')
                            if k in scal}

        import awkward as ak
        lengths = []
        max_ntau = 0
        for p in self._paths:
            f = uproot.open(p, **self._open_kw)
            t = f[self._tree_name]
            lengths.append(int(t.num_entries))
            nf = ak.to_numpy(t['ntau_freezeout'].array(library='ak'))
            if nf.size:
                max_ntau = max(max_ntau, int(np.max(nf)))

        self._cumlen = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._total  = int(self._cumlen[-1])
        self.choose_ntau = int(choose_ntau) if choose_ntau else max_ntau
        self._item_shape = (self._nFeatures, self._nx, self._ny, self._neta, self.choose_ntau)
        # pid-keyed tree-handle cache (fork safety, DATA_PIPELINE_PLAN.md §9)
        self._trees = {}

    # ------------------------------------------------------------------ protocol
    @property
    def shape(self):
        return (self._total,) + self._item_shape

    @property
    def dtype(self):
        return np.dtype(np.float32)

    def __len__(self):
        return self._total

    def __repr__(self):
        return (f"RootSource(files={len(self._paths)}, shape={self.shape}, "
                f"choose_ntau={self.choose_ntau})")

    def __getitem__(self, idx):
        a, b, squeeze = _range(idx, self._total)
        if a >= b:
            out = np.empty((0,) + self._item_shape, dtype=np.float32)
            return out[0] if squeeze else out

        pieces = []
        for fi in range(len(self._paths)):
            fstart = int(self._cumlen[fi])
            fstop  = int(self._cumlen[fi + 1])
            lo = max(a, fstart) - fstart
            hi = min(b, fstop)  - fstart
            if lo >= hi:
                continue
            chunk = self._tree_for(fi).arrays(
                [self._branch], entry_start=lo, entry_stop=hi, library='ak')
            pieces.append(self._pad_reshape(chunk[self._branch]))

        out = (pieces[0] if len(pieces) == 1
               else np.concatenate(pieces, axis=0))
        return out[0] if squeeze else out

    # ------------------------------------------------------------------ internals
    def _tree_for(self, fi):
        """Return a cached TTree handle for file index ``fi``.

        Keyed by os.getpid() so a forked DataLoader worker reopens its own
        handles instead of reusing the parent's (DATA_PIPELINE_PLAN.md §9).
        """
        cache = self._trees.setdefault(os.getpid(), {})
        if fi not in cache:
            cache[fi] = self._uproot.open(self._paths[fi], **self._open_kw)[self._tree_name]
        return cache[fi]

    def _pad_reshape(self, branch_ak):
        """Pad a variable-tau awkward branch to a dense ``(n, F, nx, ny, neta, ntau)``.

        Reuses the read_3d_root.read_3d_data assembly (pad_none -> fill 0 ->
        reshape -> two moveaxis) so the result matches the eager array exactly.
        """
        import awkward as ak
        target_len = self._nx * self._ny * self._neta * self._nFeatures * self.choose_ntau
        padded = ak.to_numpy(
            ak.fill_none(ak.pad_none(branch_ak, target_len, axis=-1, clip=True),
                         np.float32(0.))
        ).astype(np.float32, copy=False)
        n = padded.shape[0]
        arr = padded.reshape(n, self.choose_ntau, self._nx, self._ny,
                             self._neta, self._nFeatures)
        arr = np.moveaxis(arr, 1, -1)    # tau -> last axis
        arr = np.moveaxis(arr, -2, 1)    # nFeatures -> axis 1
        return np.ascontiguousarray(arr)

    def reset_handles(self):
        """Drop cached tree handles for the current process (fork safety).

        Handles inherited from the parent are dropped, not closed (closing would
        tear down the parent's still-open fds).
        """
        self._trees = {}

    # ------------------------------------------------------------------ pickling
    # uproot handles are not picklable, so exclude them so the instance can
    # survive a DataLoader 'spawn' start method (macOS default).
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_trees', None)
        state.pop('_uproot', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        import uproot
        self._uproot = uproot
        self._trees = {}


# ---------------------------------------------------------------------------
# ZarrSource (optional) — DATA_PIPELINE_PLAN.md §6
# ---------------------------------------------------------------------------
def _as_mapper(store, storage_options):
    """Return a zarr-openable store: a local path, or an fsspec mapper for gs://."""
    if isinstance(store, (str, os.PathLike)) and str(store).startswith(
            ("gs://", "s3://", "http://", "https://")):
        import fsspec
        return fsspec.get_mapper(str(store), **(storage_options or {}))
    return store


class ZarrSource:
    """Thin ArraySource over a chunked Zarr array (local or ``gs://``, lazy).

    Parameters
    ----------
    store           : str | path | MutableMapping
        Local path, fsspec URI (gs:// / s3://), or an already-opened mapper.
    key             : str
        Array name within the Zarr group (default ``"arr"``).
    storage_options : dict | None
        fsspec options for remote stores.
    """

    def __init__(self, store, key='arr', storage_options=None):
        import zarr
        self._z = zarr.open(_as_mapper(store, storage_options), mode='r')[key]

    @property
    def shape(self):
        return tuple(self._z.shape)

    @property
    def dtype(self):
        return self._z.dtype

    def __len__(self):
        return self._z.shape[0]

    def __repr__(self):
        return f"ZarrSource(shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, idx):
        return np.asarray(self._z[idx])
