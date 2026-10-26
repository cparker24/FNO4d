import os
import tracemalloc
import uproot  as up
import awkward as ak
import numpy as np

def read_3d_data(file_path, nevents=-1, choose_ntau=None, local_rank=None,
                 storage_options=None, verbose=False):
    ''' read events from a 3d root file; file_path may be a single path or a list of paths
    returns:
      {'data' : 6D numpy array,
       'tau0' : 1D numpy array of initial tau values,
        # several parameters
        x_min, y_min, eta_min, dx, dy, deta, dtau : grid parameters}
    '''

    if verbose:
        tracemalloc.start()

    # Normalize to list so multi-file and single-file share the same code path
    if isinstance(file_path, (str, bytes)):
        file_paths = [file_path]
    else:
        file_paths = list(file_path)

    open_kwargs = storage_options if storage_options else {}

    for fp in file_paths:
        _is_remote = isinstance(fp, str) and fp.startswith(("s3://", "gs://", "http://", "https://"))
        if not _is_remote and not os.path.exists(fp):
            raise FileNotFoundError(f"File {fp} does not exist.")

    # Scalar metadata is read from the first file (assumed consistent across files)
    file0 = up.open(file_paths[0], **open_kwargs)
    dat = {
        'nFeatures' : file0['nFeatures'].member('fVal'),
        'nx' : file0['nx'].member('fVal'),
        'ny' : file0['ny'].member('fVal'),
        'neta' : file0['neta'].member('fVal'),
        'x_min' : file0['x_min'].member('fVal'),
        'y_min' : file0['y_min'].member('fVal'),
        'eta_min' : file0['eta_min'].member('fVal'),
        'dx' : file0['dx'].member('fVal'),
        'dy' : file0['dy'].member('fVal'),
        'deta' : file0['deta'].member('fVal'),
        'tau_min' : file0['tau_min'].member('fVal'),
        'tau_min_MUSIC' : file0['tau_min_MUSIC'].member('fVal'),
        'dtau' : file0['dtau'].member('fVal'),
        'ntau' : file0['ntau'].member('fVal'), # if variable length, then is 0
    }

    if verbose:
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

    # Chain all files so ntau_freezeout reflects the global max across every file —
    # this is critical for consistent padding when file_paths has more than one entry
    tree_paths = [f"{fp}:t" for fp in file_paths]
    combined_meta = up.concatenate(tree_paths,
                                   filter_name=["tau_freezeout", "ntau_freezeout"],
                                   **open_kwargs)
    dat['tau_freezeout'] = ak.to_numpy(combined_meta['tau_freezeout'])
    dat['ntau_freezeout'] = ak.to_numpy(combined_meta['ntau_freezeout'])

    print('Grid parameters:')
    print(f"  nx: {dat['nx']}, ny: {dat['ny']}, neta: {dat['neta']}, ntau: {dat['ntau']}")
    print(f"  x_min: {dat['x_min']}, y_min: {dat['y_min']}, eta_min: {dat['eta_min']}")
    print(f"  dx: {dat['dx']}, dy: {dat['dy']}, deta: {dat['deta']}, tau_min: {dat['tau_min']}, dtau: {dat['dtau']}")
    print(f"  ntau_freezeout: {dat['ntau_freezeout']}, tau_min_MUSIC: {dat['tau_min_MUSIC']}")

    # Total entries across all chained files
    total_entries = len(dat['ntau_freezeout'])
    print(f'Total events available across {len(file_paths)} file(s): {total_entries}')

    if local_rank is not None:
        half = nevents // 2
        start_event = local_rank * half
        end_event = start_event + half
    else:
        start_event = 0
        end_event = nevents

    use_flat = 'flat_data' in file0['t'].keys()

    if use_flat:
        print("reading flat_data")
        if end_event <= 0:
            end_event = total_entries
        elif end_event > total_entries:
            print(f'Requested more events ({end_event}) than available ({total_entries}); using max available ({total_entries})')
            end_event = total_entries
        print(f'Selected events: {end_event - start_event}')

        flat_combined = up.concatenate(tree_paths,
                                       filter_name=["flat_data", "ntau_start"],
                                       entry_stop=end_event,
                                       **open_kwargs)
        dat['ntau_start'] = ak.to_numpy(flat_combined['ntau_start'])
        dat['ntau'] = file0['NT'].member('fVal')

        arr = ak.to_numpy(flat_combined['flat_data'][start_event:end_event]).astype(np.float32)
        ex_size = dat['ntau']*dat['nx']*dat['ny']*dat['neta']*dat['nFeatures']

        if arr.shape[-1] != ex_size:
            raise ValueError(f"Unexpected data size: got {arr.shape[-1]}, expected {ex_size}")
        arr = np.reshape(arr, (end_event-start_event, dat['ntau'], dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']))
        arr = np.moveaxis(arr, 1, -1)
        arr = np.moveaxis(arr, -2, 1)

        n_nan = np.sum(np.isnan(arr))
        print(f' :: number of nan: {n_nan} out of {arr.size} entries')
        print('Read array of shape:', arr.shape)
        dat['arr'] = arr
        num_nan = np.sum(np.isnan(arr))
        bad_tups = []
        if num_nan:
            print(f'   Warning: there are {num_nan} NaN entries in the data array')
            nan_locs = np.argwhere(np.isnan(arr))
            print('Locations of NaNs in arr:')
            print(type(nan_locs[0]))
            for loc in nan_locs:
                print(tuple(int(i) for i in loc))
            if len(nan_locs) > 100:
                print(f'... and {len(nan_locs) - 100} more')
        dat['nans'] = np.argwhere(np.isnan(arr)) if num_nan else None

        print('Read array of shape:', arr.shape)
        if verbose:
            tracemalloc.stop()
        return dat
    else:
        ''' must be reading straight from .root, presumably with variable length tau dimension'''
        # choose_ntau must be the global max so padding is consistent across all files
        if not choose_ntau:
            choose_ntau = int(np.max(dat['ntau_freezeout']))

        if nevents <= 0:
            nevents = total_entries
        elif nevents > total_entries:
            print(f'Requested more events ({nevents}) than available ({total_entries}); using max available ({total_entries})')
            nevents = total_entries
        print(f'Selected events: {nevents}')

        if verbose:
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

        target_len = int(dat['nx']*dat['ny']*dat['neta']*dat['nFeatures']*choose_ntau)
        print(f"Processing {nevents} events with variable-length tau dimension; padding to {choose_ntau} taus per event (total size per event: {target_len} entries)")

        # Allocate the full output array once, up front
        arr = np.zeros(
            (nevents, choose_ntau, dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']), dtype=np.float32
        )

        if verbose:
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

        # uproot.iterate chains multiple files transparently; step_size controls peak memory.
        # entry_stop in iterate is per-file, so we enforce the global cap by clipping each chunk.
        # Capped at 5 rather than scaling with nevents: the per-chunk transient (raw chunk +
        # awkward pad_none/fill_none/to_numpy intermediates) scales with chunk_size * per-event
        # size, and per-event size grew substantially with the widened (neta=33) eta grid --
        # nevents//10 (~19-20 events/chunk for a 199-event combined dataset) OOM-killed the
        # process here even though the final target array alone fit comfortably in RAM.
        chunk_size = min(nevents // 10, 5) if nevents >= 10 else 1
        out_idx = 0
        for chunk in up.iterate(tree_paths, filter_name="user_res",
                                step_size=chunk_size, **open_kwargs):
            remaining = nevents - out_idx
            if remaining <= 0:
                break
            arr_ak = chunk['user_res'][:remaining]  # clip last partial chunk to exact count
            n_chunk = len(arr_ak)
            print(f"Processing events {out_idx} to {out_idx + n_chunk}...")

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            padded = ak.to_numpy(
                ak.fill_none(ak.pad_none(arr_ak, target_len, axis=-1, clip=True),
                            np.float32(0.))
            ).astype(np.float32, copy=False)

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            arr[out_idx:out_idx + n_chunk] = padded.reshape(
                n_chunk, choose_ntau, dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']
            )
            out_idx += n_chunk
            if out_idx >= nevents:
                del arr_ak, padded
                break

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            del arr_ak, padded   # intermediate freed immediately

        print('Before moveaxis - Read array of shape:', arr.shape)

        arr = np.moveaxis(arr, 1, -1)
        arr = np.moveaxis(arr, -2, 1)
        dat['arr'] = arr

        if verbose:
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] All done with padding Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

        num_nan = np.sum(np.isnan(arr))
        if num_nan:
            print(f'   Warning: there are {num_nan} NaN entries in the data array')
            raise ValueError(f"{num_nan} NaN entries found in data array")

        print('Read array of shape:', arr.shape)
        if verbose:
            tracemalloc.stop()
        return dat

def coarsen_array(arr, i_axis, factor):
    ''' coarsen the array by averaging every "factor" elements (2, 3, etc) along
    the specified axis (i_axis)
    '''
    print(arr.shape)
    L = arr.shape[i_axis]
    if L % factor:
        raise ValueError(f"Cannot reduces axis {i_axis} by ratio of {factor}; remainder exists ({L} % {factor} = {L%factor}")
    L_new = L // factor

    # group consecutive pairs along the axis: shape -> (..., L//2, 2, ...)
    pre_shape = list(arr.shape)
    pre_shape[i_axis] = L_new
    pre_shape.insert(i_axis + 1, factor)
    arr_newshape = arr.reshape(*pre_shape)
    mean_arr = arr_newshape.mean(axis=i_axis + 1)
    print('   Old dim:', arr.shape, '\n-> New dim:', mean_arr.shape)
    return mean_arr

if __name__ == "__main__":
    data_0 = read_3d_data('/home/davidstewart/flat/flat3d_noscale.root', nevents=20, verbose=True)
    print('minimum value: ', np.min(data_0['arr'][:,0,:,:,:]))

    print('\nTesting with flat3d file:')
    file_path = '/home/davidstewart/flat/flat3d_tauep75.root'
    data = read_3d_data(file_path, nevents=20, verbose=True)
    print(np.sum(np.isnan(data['arr'])))

    for loc in data['nans']:
        print(type(loc))
        print(tuple(int(i) for i in loc))
        print(data_0['arr'][tuple(int(i) for i in loc)])
