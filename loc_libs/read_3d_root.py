import os
import tracemalloc
import uproot  as up
import awkward as ak
import numpy as np

def read_3d_data(file_path, nevents=-1, choose_ntau=None, local_rank=None,
                 storage_options=None, verbose=False,
                 xyslice=None, etaslice=None, tauslice=None,
                 min_ntau_freezeout=None, drop_channels=None, dtype=np.float32):
    ''' read events from a 3d root file; file_path may be a single path or a list of paths
    returns:
      {'data' : 6D numpy array,
       'tau0' : 1D numpy array of initial tau values,
        # several parameters
        x_min, y_min, eta_min, dx, dy, deta, dtau : grid parameters}

    xyslice, etaslice, tauslice : optional slice objects, applied to the nx/ny,
        neta, and (post-padding) tau axes respectively while the data is still
        being read in per-chunk pieces, so the full raw-resolution array is
        never materialised. None (the default) keeps that axis uncropped --
        unchanged behavior for any existing caller that doesn't pass these.
    min_ntau_freezeout : optional int. If given, events with ntau_freezeout
        below this are dropped (same per-event filter train_torch.py used to
        apply *after* this function returned the full array -- doing it here
        instead, before allocating the output array, is what avoids ever
        holding both the full and the cropped/filtered array in memory at
        once; see the "variable-length tau dimension" branch below for the
        full explanation). None (the default) keeps every event, unchanged
        behavior for any existing caller that doesn't pass this.
    drop_channels : optional list of feature-axis indices to remove (e.g. [1]
        for temperature, given RootBulkWriter.cc's fixed [energy_density,
        temperature, vx, vy, vz] write order), applied per-chunk alongside
        xyslice/etaslice/tauslice above for the same reason -- a post-hoc
        drop on the full returned array would briefly hold both the original
        and the smaller copy in memory at once. None (the default) keeps
        every feature channel, unchanged behavior for any existing caller
        that doesn't pass this. Only supported on the variable-length-tau
        ("straight from .root") path below, not the legacy flat_data path.
    dtype : storage dtype for the master output array (default np.float32,
        unchanged behavior for any existing caller). Pass np.float16 to halve
        the resident size of a dataset large enough to be host-RAM-bound --
        every downstream consumer already upcasts back to float32 per-batch
        before it reaches the model (FluidDataset._to_tensor does
        torch.FloatTensor(arr); collate.py's np.ascontiguousarray(..., dtype=
        np.float32) does the same for the lazy path), so this only affects
        the precision of the long-lived host array, not what the model sees.
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
        if drop_channels:
            raise NotImplementedError(
                "drop_channels is only supported on the variable-length-tau "
                "('straight from .root') read path, not the legacy flat_data "
                "format this file uses.")
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

        # Resolve the crop slices against the RAW grid dims up front, so the
        # output array below can be allocated at its final CROPPED size
        # instead of the full raw resolution. int()'d defensively since
        # slice.indices() requires true ints and dat['nx']/etc. come from
        # uproot's TParameter<float>.member('fVal').
        nx_raw, ny_raw, neta_raw = int(dat['nx']), int(dat['ny']), int(dat['neta'])
        xyslice_r  = xyslice  if xyslice  is not None else slice(None)
        etaslice_r = etaslice if etaslice is not None else slice(None)
        tauslice_r = tauslice if tauslice is not None else slice(None)
        nx_out   = len(range(*xyslice_r.indices(nx_raw)))
        ny_out   = len(range(*xyslice_r.indices(ny_raw)))
        neta_out = len(range(*etaslice_r.indices(neta_raw)))
        ntau_out = len(range(*tauslice_r.indices(choose_ntau)))

        # Same reasoning as the spatial/tau crops above, applied to the
        # feature axis: resolve which channels to keep up front so the
        # output array is allocated at its final (post-drop) feature count,
        # and only ever index the small per-chunk array below, never the
        # full one.
        nFeatures_raw = int(dat['nFeatures'])
        if drop_channels:
            keep_channels = [i for i in range(nFeatures_raw) if i not in set(drop_channels)]
            nFeatures_out = len(keep_channels)
            print(f'Dropping feature channel(s) {sorted(drop_channels)} -> '
                  f'keeping {keep_channels} ({nFeatures_raw} -> {nFeatures_out} features)')
        else:
            keep_channels = None  # keep all; skip the extra per-chunk fancy-index below
            nFeatures_out = nFeatures_raw

        # Event-level filtering (dropping blank/near-blank records left by a
        # MUSIC crash partway through generation) is also done here, before
        # allocation, for the same reason -- this used to be applied by the
        # caller (train_torch.py) *after* this function returned the full,
        # all-events array, which needed a second, almost-as-large copy held
        # in memory at the same time as the first (both briefly resident).
        # That "fused mask+slice" step is exactly what OOM-killed the retrain
        # on the doubled (eta4/n65) grid: ~123GiB original + ~116GiB sliced
        # copy =~ 239GiB peak, more than the instance's RAM, even though the
        # final (cropped) array alone would have fit comfortably. Filtering
        # and cropping HERE, per-chunk, as each small piece is read, means
        # the only thing ever held at full dataset size is the final,
        # already-cropped array -- the raw-resolution/all-events array is
        # never materialised at all.
        windowed_ntau_freezeout = dat['ntau_freezeout'][:nevents]
        if min_ntau_freezeout is not None:
            good_mask = windowed_ntau_freezeout >= min_ntau_freezeout
        else:
            good_mask = np.ones(nevents, dtype=bool)
        n_dropped = int((~good_mask).sum())
        if n_dropped > 0:
            print(f'Dropping {n_dropped}/{nevents} event(s) with '
                  f'ntau_freezeout < {min_ntau_freezeout} -- these are blank or '
                  f'near-blank records left behind when MUSIC crashed partway '
                  f'through generating the data file, and training on them '
                  f'corrupts the loss (a near-zero-norm target blows up the '
                  f'relative L2/H1 loss computation).')
        nevents_out = int(good_mask.sum())

        print(f"Cropping during load: nx {nx_raw}->{nx_out}, ny {ny_raw}->{ny_out}, "
              f"neta {neta_raw}->{neta_out}, ntau {choose_ntau}->{ntau_out}, "
              f"nevents {nevents}->{nevents_out} (output array allocated at the cropped size)")

        # Allocate the output array ONCE, up front -- already at its final
        # CROPPED/FILTERED shape (see above), not the full raw resolution.
        arr = np.zeros(
            (nevents_out, ntau_out, nx_out, ny_out, neta_out, nFeatures_out), dtype=dtype
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
        out_idx_raw = 0   # index into the nevents-windowed (pre-filter) event sequence
        out_idx = 0       # index into the cropped/filtered output array
        for chunk in up.iterate(tree_paths, filter_name="user_res",
                                step_size=chunk_size, **open_kwargs):
            remaining = nevents - out_idx_raw
            if remaining <= 0:
                break
            arr_ak = chunk['user_res'][:remaining]  # clip last partial chunk to exact count
            n_chunk = len(arr_ak)
            print(f"Processing events {out_idx_raw} to {out_idx_raw + n_chunk}...")

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            padded = ak.to_numpy(
                ak.fill_none(ak.pad_none(arr_ak, target_len, axis=-1, clip=True),
                            np.float32(0.))
            ).astype(dtype, copy=False)
            padded = padded.reshape(
                n_chunk, choose_ntau, dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']
            )

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            # Crop + filter THIS CHUNK ONLY (a few events' worth, per
            # chunk_size above) rather than the whole dataset at once -- the
            # transient here is chunk_size events wide, not nevents wide.
            # Fused into one indexing call (boolean mask on axis 0 + slices
            # on the rest) so this itself doesn't briefly double even the
            # small per-chunk array.
            chunk_mask = good_mask[out_idx_raw:out_idx_raw + n_chunk]
            n_good_chunk = int(chunk_mask.sum())
            if n_good_chunk > 0:
                cropped = padded[chunk_mask, tauslice_r, xyslice_r, xyslice_r, etaslice_r, :]
                # Channel selection is a SEPARATE indexing step (not folded
                # into the call above) because mixing a boolean mask with a
                # fancy-index list, with plain slices in between, triggers
                # numpy's "advanced indices not contiguous" broadcasting rule
                # and silently reorders the output's axes -- doing it as its
                # own step on the already-chunk-sized `cropped` array sidesteps
                # that entirely, at the cost of one more chunk-sized (not
                # dataset-sized) transient, which is the same trade every
                # other per-chunk crop above already makes.
                if keep_channels is not None:
                    cropped = cropped[..., keep_channels]
                arr[out_idx:out_idx + n_good_chunk] = cropped
                out_idx += n_good_chunk
            out_idx_raw += n_chunk

            if out_idx_raw >= nevents:
                del arr_ak, padded
                break

            if verbose:
                current, peak = tracemalloc.get_traced_memory()
                print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")

            del arr_ak, padded   # intermediate freed immediately

        print('Before moveaxis - Read array of shape (already cropped/filtered):', arr.shape)

        arr = np.moveaxis(arr, 1, -1)
        arr = np.moveaxis(arr, -2, 1)
        dat['arr'] = arr
        dat['nFeatures'] = nFeatures_out  # reflects any drop_channels applied above

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
