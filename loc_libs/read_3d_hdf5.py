'''
Utilities for converting ROOT bulk-writer files to HDF5 and reading them back.

Why HDF5?
---------
ROOT stores branch data in baskets.  When all events are written in one job
without explicit basket flushing (i.e. without SetAutoFlush(1) in the C++
writer), every event lands in the same basket.  uproot must decompress and
decode the entire basket even when only a slice is requested, causing peak
memory ≈ 2× the full array size on every chunked read.

HDF5 with per-event (or small-batch) chunks has true random-access I/O:
  dataset[10:20]  →  decompresses exactly that range, nothing more.

Workflow
--------
  # Once (accepts the one-time 2× ROOT peak):
  from read_3d_hdf5 import convert_root_to_hdf5
  convert_root_to_hdf5('dAu_250ev_mb.root', 'dAu_250ev_mb.h5')

  # Every subsequent run (true partial I/O, no 2× peak):
  from read_3d_hdf5 import read_3d_data_hdf5
  dat = read_3d_data_hdf5('dAu_250ev_mb.h5', nevents=100)

The returned dict has the same keys and arr shape as read_3d_data() in
read_3d_root.py:  arr shape = (nevents, nFeatures, nx, ny, neta, ntau)
'''

import os
import gc
import tracemalloc

import uproot as up
import awkward as ak
import numpy as np


# ---------------------------------------------------------------------------
# Scalar grid-parameter keys stored in the ROOT file
# ---------------------------------------------------------------------------
_SCALAR_KEYS = [
    'nFeatures', 'nx', 'ny', 'neta',
    'x_min', 'y_min', 'eta_min',
    'dx', 'dy', 'deta',
    'tau_min', 'tau_min_MUSIC', 'dtau',
]


# see also: https://github.com/dev-geof/root2hdf5

def convert_root_to_hdf5(root_file_path, hdf5_file_path,
                          nevents=-1, choose_ntau=None,
                          chunk_events=1,
                          storage_options=None,
                          verbose=False):
    '''Convert a variable-length-tau ROOT file to HDF5 with chunked storage.

    Parameters
    ----------
    root_file_path  : str  – source .root file (local or remote s3/gs/http)
    hdf5_file_path  : str  – output .h5 file path (created or overwritten)
    nevents         : int  – number of events to convert (-1 = all)
    choose_ntau     : int  – pad/clip tau axis to this length (None = auto max)
    chunk_events    : int  – HDF5 chunk size along the event axis.
                             1 = finest granularity (one basket per event),
                             matches the effect of SetAutoFlush(1) in the C++
                             writer.  Increase for better compression at the
                             cost of coarser read granularity.
    storage_options : dict – passed to uproot.open (for remote files)
    '''
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    open_kwargs = storage_options or {}
    file = up.open(root_file_path, **open_kwargs)

    # --- read scalar grid parameters ---
    dat = {k: file[k].member('fVal') for k in _SCALAR_KEYS if k in file}
    nx        = int(dat['nx'])
    ny        = int(dat['ny'])
    neta      = int(dat['neta'])
    nFeatures = int(dat['nFeatures'])

    ntau_freezeout = ak.to_numpy(file['t']['ntau_freezeout'].array(library='ak'))
    tau_freezeout  = ak.to_numpy(file['t']['tau_freezeout'].array(library='ak'))

    if not choose_ntau:
        choose_ntau = int(np.max(ntau_freezeout))
    target_len = int(nx * ny * neta * nFeatures * choose_ntau)

    branch   = file['t']['user_res']
    nentries = branch.num_entries
    if nevents <= 0 or nevents > nentries:
        nevents = nentries

    print(f"Converting {nevents} events  →  {hdf5_file_path}")
    print(f"  arr shape per event: ({choose_ntau}, {nx}, {ny}, {neta}, {nFeatures})")
    print(f"  HDF5 chunk size: {chunk_events} event(s)  "
          f"(~{chunk_events * nx * ny * neta * nFeatures * choose_ntau * 4 / 1024**2:.1f} MB/chunk)")

    # Output shape after moveaxis: (nevents, nFeatures, nx, ny, neta, choose_ntau)
    # — identical to what read_3d_data() in read_3d_root.py returns.
    final_shape = (nevents, nFeatures, nx, ny, neta, choose_ntau)
    chunk_shape = (min(chunk_events, nevents), nFeatures, nx, ny, neta, choose_ntau)

    with h5py.File(hdf5_file_path, 'w') as hf:
        # scalar grid parameters as HDF5 attributes
        for k, v in dat.items():
            hf.attrs[k] = v
        hf.attrs['choose_ntau'] = choose_ntau
        hf.attrs['nevents']     = nevents

        hf.create_dataset('ntau_freezeout', data=ntau_freezeout[:nevents])
        hf.create_dataset('tau_freezeout',  data=tau_freezeout[:nevents])

        ds = hf.create_dataset(
            'arr',
            shape=final_shape,
            dtype=np.float32,
            chunks=chunk_shape,
            compression='lzf',   # fast lossless; swap for 'gzip' for smaller files
        )

        # uproot accumulates per-event element counts in a variable that starts
        # as Python int but becomes numpy int32 after the first += with a numpy
        # scalar.  Reading all events at once overflows int32 when the total
        # element count exceeds 2^31 (~2.15B).  Fix: read in batches whose
        # total element count (= batch_size × max_elements_per_event) stays
        # safely below 2^31.  Since all events are usually in one ROOT basket,
        # each batch read still decompresses the full basket, but this is
        # unavoidable for the one-time conversion and correctness takes priority.
        int32_max    = 2**31 - 1
        read_batch   = max(1, int(int32_max // target_len))  # max events per read
        write_chunk  = max(chunk_events, 10)                 # min write granularity
        read_batch   = max(read_batch, write_chunk)          # align to write_chunk

        print(f"  int32 safe read batch: {read_batch} events  "
              f"(~{read_batch * target_len * 4 / 1024**3:.2f} GiB/batch)")

        for rd_start in range(0, nevents, read_batch):
            rd_stop = min(rd_start + read_batch, nevents)
            print(f"Reading ROOT events {rd_start}–{rd_stop} ...")

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory] before read  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

            arr_ak_batch = branch.array(entry_start=rd_start, entry_stop=rd_stop, library='ak')

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory] after  read  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

            for wr_start in range(rd_start, rd_stop, write_chunk):
                wr_stop  = min(wr_start + write_chunk, rd_stop)
                local_start = wr_start - rd_start
                local_stop  = wr_stop  - rd_start
                print(f"  writing events {wr_start}–{wr_stop} ...")

                arr_ak = arr_ak_batch[local_start:local_stop]
                arr_ak_padded = ak.pad_none(arr_ak, target_len, axis=-1, clip=True)
                del arr_ak
                arr_ak_filled = ak.fill_none(arr_ak_padded, np.float32(0.))
                del arr_ak_padded
                padded = ak.to_numpy(arr_ak_filled).astype(np.float32, copy=False)
                del arr_ak_filled

                n = wr_stop - wr_start
                chunk = padded.reshape(n, choose_ntau, nx, ny, neta, nFeatures)
                del padded
                chunk = np.moveaxis(chunk, 1, -1)   # tau → last axis
                chunk = np.moveaxis(chunk, -2, 1)   # nFeatures → axis 1
                ds[wr_start:wr_stop] = chunk
                del chunk

            del arr_ak_batch
            gc.collect()

    print(f"Done. Written to {hdf5_file_path}")


# ---------------------------------------------------------------------------

class MultiH5Array:
    '''Lazy concatenated view over the "arr" dataset across multiple HDF5 files.

    No data is read from disk until the object is indexed.  Supports the
    numpy array protocol (shape, dtype, __len__, __getitem__, __array__) so
    it can be used as a drop-in wherever a numpy array is expected, while
    keeping memory usage bounded to exactly the slice that is requested.

    Usage
    -----
        mha = MultiH5Array([f1, f2, f3])
        batch = mha[0:32]       # reads only 32 events; returns ndarray
        full  = mha.load()      # materialises everything
        arr   = np.asarray(mha) # same as load()

    Parameters
    ----------
    file_paths   : list[str] – ordered list of .h5 files
    global_start : int       – first global event index exposed (default 0)
    global_stop  : int|None  – one-past-last global event index (None = all)
    '''

    # grid-parameter attrs that must agree across files
    _DIM_KEYS = ('nFeatures', 'nx', 'ny', 'neta', 'choose_ntau')

    def __init__(self, file_paths, global_start=0, global_stop=None):
        try:
            import h5py as _h5py
        except ImportError:
            raise ImportError("h5py is required: pip install h5py")
        self._h5py = _h5py
        self._paths = list(file_paths)

        # read shape / dtype from first file; validate rest
        with _h5py.File(self._paths[0], 'r') as f:
            self._dtype      = f['arr'].dtype
            self._item_shape = f['arr'].shape[1:]  # (nFeatures, nx, ny, neta, ntau)
            ref_dims = {k: f.attrs[k] for k in self._DIM_KEYS if k in f.attrs}

        lengths = []
        for p in self._paths:
            with _h5py.File(p, 'r') as f:
                # dimension check
                for k, ref in ref_dims.items():
                    if k in f.attrs and f.attrs[k] != ref:
                        raise ValueError(
                            f"Dimension mismatch for '{k}': "
                            f"{self._paths[0]} has {ref}, {p} has {f.attrs[k]}"
                        )
                lengths.append(f['arr'].shape[0])

        self._cumlen   = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._raw_total = int(self._cumlen[-1])
        self._offset   = int(global_start)
        stop           = int(global_stop) if global_stop is not None else self._raw_total
        self._length   = max(0, stop - self._offset)

    # ------------------------------------------------------------------ numpy protocol

    @property
    def shape(self):
        return (self._length,) + self._item_shape

    @property
    def dtype(self):
        return self._dtype

    def __len__(self):
        return self._length

    def __repr__(self):
        return (f"MultiH5Array(files={len(self._paths)}, "
                f"shape={self.shape}, dtype={self.dtype})")

    def __array__(self, dtype=None, copy=None):  # noqa: ARG002
        out = self._read_global(self._offset, self._offset + self._length)
        return out if dtype is None else out.astype(dtype, copy=False)

    # ------------------------------------------------------------------ public helpers

    def load(self):
        '''Materialise the entire view into one contiguous ndarray.'''
        return self._read_global(self._offset, self._offset + self._length)

    # ------------------------------------------------------------------ indexing

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            if idx < 0:
                idx += self._length
            if not (0 <= idx < self._length):
                raise IndexError(f"index {idx} out of bounds for axis 0 with size {self._length}")
            g  = self._offset + int(idx)
            fi = self._file_for_global(g)
            li = g - int(self._cumlen[fi])
            with self._h5py.File(self._paths[fi], 'r') as f:
                return f['arr'][li]

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(self._length)
            if step != 1:
                # non-unit stride: materialise then fancy-index
                return self.load()[idx]
            return self._read_global(self._offset + start, self._offset + stop)

        else:
            # boolean mask, integer array, tuple of indices, …
            return self.load()[idx]

    # ------------------------------------------------------------------ internals

    def _file_for_global(self, gi):
        '''Return file index for global event index gi.'''
        fi = int(np.searchsorted(self._cumlen[1:], gi, side='right'))
        return min(fi, len(self._paths) - 1)

    def _read_global(self, g_start, g_stop):
        '''Read global event range [g_start, g_stop) from the file set.'''
        if g_start >= g_stop:
            return np.empty((0,) + self._item_shape, dtype=self._dtype)
        pieces = []
        for fi in range(len(self._paths)):
            fstart = int(self._cumlen[fi])
            fstop  = int(self._cumlen[fi + 1])
            lo = max(g_start, fstart) - fstart
            hi = min(g_stop,  fstop)  - fstart
            if lo >= hi:
                continue
            with self._h5py.File(self._paths[fi], 'r') as f:
                pieces.append(f['arr'][lo:hi])
        if not pieces:
            return np.empty((0,) + self._item_shape, dtype=self._dtype)
        return np.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0].copy()


# ---------------------------------------------------------------------------

def read_3d_data_hdf5(hdf5_file_path, nevents=-1, start_event=0,
                      verbose=False, lazy=True):
    '''Load data from one or more pre-converted HDF5 files.

    When a list of files is supplied every file is treated as a contiguous
    segment of one logical dataset — analogous to uproot.concatenate().
    All files must have identical grid dimensions (nFeatures, nx, ny, neta,
    choose_ntau); a ValueError is raised if they differ.

    Parameters
    ----------
    hdf5_file_path : str or list[str]
        Path(s) to .h5 file(s) produced by convert_root_to_hdf5().
    nevents        : int  – events to load (-1 = all across all files).
    start_event    : int  – first global event index.
    lazy           : bool – True  → dat['arr'] is a MultiH5Array; data is read
                                    from disk only when the object is sliced.
                                    Ideal for training loops (one batch at a time).
                            False → dat['arr'] is a numpy ndarray loaded immediately.
    verbose        : bool – print memory diagnostics.

    Returns
    -------
    dict with the same keys as read_3d_data() in read_3d_root.py:
        'arr'             – MultiH5Array (lazy=True) or ndarray (lazy=False)
                            shape (n_events, nFeatures, nx, ny, neta, ntau)
        'ntau_freezeout'  – 1D ndarray (always loaded eagerly; small)
        'tau_freezeout'   – 1D ndarray (always loaded eagerly; small)
        plus scalar grid parameters as individual keys (from first file attrs)
    '''
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    # ---- normalise input to a list ----------------------------------------
    if isinstance(hdf5_file_path, (str, os.PathLike)):
        file_paths = [str(hdf5_file_path)]
    else:
        file_paths = [str(p) for p in hdf5_file_path]

    for p in file_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    multi = len(file_paths) > 1

    # ---- read attrs from each file; validate dimensions -------------------
    _DIM_KEYS = ('nFeatures', 'nx', 'ny', 'neta', 'choose_ntau')
    all_attrs  = []
    per_file_nevents = []
    for p in file_paths:
        with h5py.File(p, 'r') as hf:
            all_attrs.append(dict(hf.attrs))
            per_file_nevents.append(int(hf.attrs['nevents']))

    if multi:
        ref = all_attrs[0]
        for attrs, p in zip(all_attrs[1:], file_paths[1:]):
            for k in _DIM_KEYS:
                if k in ref and k in attrs and attrs[k] != ref[k]:
                    raise ValueError(
                        f"Dimension mismatch for '{k}': "
                        f"{file_paths[0]} has {ref[k]}, {p} has {attrs[k]}"
                    )
        print(f"Multi-file mode: {len(file_paths)} files, "
              f"total events = {sum(per_file_nevents)}")

    # merged attrs: scalar grid params from first file
    dat = dict(all_attrs[0])

    # ---- compute global event range ---------------------------------------
    total_events = sum(per_file_nevents)
    end_event    = total_events if nevents <= 0 else min(start_event + nevents, total_events)
    n_loaded     = end_event - start_event

    # ---- ntau_freezeout / tau_freezeout: always load eagerly (1D, small) --
    ntau_pieces = []
    tau_pieces  = []
    # iterate files, tracking cumulative offset to clip to [start_event, end_event)
    cum = 0
    for p, nev in zip(file_paths, per_file_nevents):
        fstart, fstop = cum, cum + nev
        lo = max(start_event, fstart) - fstart
        hi = min(end_event,   fstop)  - fstart
        if lo < hi:
            with h5py.File(p, 'r') as hf:
                ntau_pieces.append(hf['ntau_freezeout'][lo:hi])
                tau_pieces.append(hf['tau_freezeout'][lo:hi])
        cum = fstop

    dat['ntau_freezeout'] = (np.concatenate(ntau_pieces) if len(ntau_pieces) > 1
                             else ntau_pieces[0])
    dat['tau_freezeout']  = (np.concatenate(tau_pieces)  if len(tau_pieces) > 1
                             else tau_pieces[0])

    # ---- arr: lazy or eager -----------------------------------------------
    if lazy:
        dat['arr'] = MultiH5Array(file_paths,
                                  global_start=start_event,
                                  global_stop=end_event)
        label = f"({file_paths[0]}{'…' if multi else ''})"
        print(f"Lazy MultiH5Array: {n_loaded} events {label}; "
              f"arr shape: {dat['arr'].shape}")
        dat['nans'] = None  # NaN scan would force full load; skip in lazy mode
    else:
        if verbose:
            tracemalloc.reset_peak()
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory] before HDF5 read  current={current/1024**3:.2f} GiB  "
                  f"peak={peak/1024**3:.2f} GiB")

        mha = MultiH5Array(file_paths, global_start=start_event, global_stop=end_event)
        dat['arr'] = mha.load()
        del mha

        if verbose:
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory] after  HDF5 read  current={current/1024**3:.2f} GiB  "
                  f"peak={peak/1024**3:.2f} GiB")

        label = (f"{file_paths[0]}{'…' if multi else ''}")
        print(f"Loaded {n_loaded} events from {label}; arr shape: {dat['arr'].shape}")

        num_nan = int(np.sum(np.isnan(dat['arr'])))
        if num_nan:
            print(f"Warning: {num_nan} NaN entries in loaded array")
        dat['nans'] = np.argwhere(np.isnan(dat['arr'])) if num_nan else None

    return dat


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    root_file = sys.argv[1] if len(sys.argv) > 1 else 'input.root'
    h5_file   = sys.argv[2] if len(sys.argv) > 2 else root_file.replace('.root', '.h5')

    tracemalloc.start()
    convert_root_to_hdf5(root_file, h5_file, nevents=-1, chunk_events=1, verbose=True)
    dat = read_3d_data_hdf5(h5_file, nevents=10, verbose=True)
    print('arr shape:', dat['arr'].shape)
    tracemalloc.stop()
