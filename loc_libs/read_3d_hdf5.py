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
                          storage_options=None):
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

            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory] before read  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

            arr_ak_batch = branch.array(entry_start=rd_start, entry_stop=rd_stop, library='ak')

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

def read_3d_data_hdf5(hdf5_file_path, nevents=-1, start_event=0):
    '''Load data from a pre-converted HDF5 file with true partial reads.

    Unlike read_3d_data() on a ROOT file, this never loads more than the
    requested events because HDF5 chunked storage supports real random-access
    I/O — only the requested HDF5 chunks are decompressed.

    Parameters
    ----------
    hdf5_file_path : str – path to .h5 file produced by convert_root_to_hdf5()
    nevents        : int – number of events to load (-1 = all)
    start_event    : int – first event index to load

    Returns
    -------
    dict with the same keys as read_3d_data() in read_3d_root.py:
        'arr'             – ndarray (nevents, nFeatures, nx, ny, neta, ntau)
        'ntau_freezeout'  – 1D array of per-event freeze-out tau indices
        'tau_freezeout'   – 1D array of per-event freeze-out tau values
        plus all scalar grid parameters as individual keys
    '''
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install h5py")

    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"File {hdf5_file_path} does not exist.")

    with h5py.File(hdf5_file_path, 'r') as hf:
        total = int(hf.attrs['nevents'])
        end_event = total if nevents <= 0 else min(start_event + nevents, total)

        dat = {k: hf.attrs[k] for k in hf.attrs}

        # True partial read: HDF5 only decompresses the chunks that overlap
        # [start_event, end_event) — no full-file load.
        # Reset peak so the reported peak reflects THIS read only, not any
        # prior operations (e.g. a ROOT→HDF5 conversion earlier in the run).
        tracemalloc.reset_peak()
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory] before HDF5 read  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

        dat['arr']            = hf['arr'][start_event:end_event]
        dat['ntau_freezeout'] = hf['ntau_freezeout'][start_event:end_event]
        dat['tau_freezeout']  = hf['tau_freezeout'][start_event:end_event]

        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory] after  HDF5 read  current={current/1024**3:.2f} GiB  peak={peak/1024**3:.2f} GiB")

    n_loaded = end_event - start_event
    print(f"Loaded {n_loaded} events from {hdf5_file_path}; arr shape: {dat['arr'].shape}")

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
    convert_root_to_hdf5(root_file, h5_file, nevents=-1, chunk_events=1)
    dat = read_3d_data_hdf5(h5_file, nevents=10)
    print('arr shape:', dat['arr'].shape)
    tracemalloc.stop()
