import os
import tracemalloc
import uproot  as up
import awkward as ak
import numpy as np

def read_3d_data(file_path, nevents=-1, choose_ntau=None, local_rank=None, storage_options=None):
    ''' read events from a 3d root file
    returns:
      {'data' : 6D numpy array,
       'tau0' : 1D numpy array of initial tau values,
        # several parameters
        x_min, y_min, eta_min, dx, dy, deta, dtau : grid parameters}
    '''
    _is_remote = isinstance(file_path, str) and file_path.startswith(("s3://", "gs://", "http://", "https://"))
    if not _is_remote and not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    open_kwargs = storage_options if storage_options else {}
    file = up.open(file_path, **open_kwargs)

    dat = {
        'nFeatures' : file['nFeatures'].member('fVal'),
        'nx' : file['nx'].member('fVal'),
        #'NT' : file['NT'].member('fVal'), #JP: Check!!!!!
        'ny' : file['ny'].member('fVal'),
        'neta' : file['neta'].member('fVal'),
        'x_min' : file['x_min'].member('fVal'),
        'y_min' : file['y_min'].member('fVal'),
        'eta_min' : file['eta_min'].member('fVal'),
        'dx' : file['dx'].member('fVal'),
        'dy' : file['dy'].member('fVal'),
        'deta' : file['deta'].member('fVal'),
        'tau_min' : file['tau_min'].member('fVal'),
        'tau_min_MUSIC' : file['tau_min_MUSIC'].member('fVal'),
        'dtau' : file['dtau'].member('fVal'),
        'ntau' : file['ntau'].member('fVal'), # if variable length, then is 0
    }

    current, peak = tracemalloc.get_traced_memory()
    print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
    
    dat['tau_freezeout'] = ak.to_numpy(file['t']['tau_freezeout'].array(library='ak'))
    dat['ntau_freezeout'] = ak.to_numpy(file['t']['ntau_freezeout'].array(library='ak'))

    print('Grid parameters:')
    print(f"  nx: {dat['nx']}, ny: {dat['ny']}, neta: {dat['neta']}, ntau: {dat['ntau']}")
    print(f"  x_min: {dat['x_min']}, y_min: {dat['y_min']}, eta_min: {dat['eta_min']}")
    print(f"  dx: {dat['dx']}, dy: {dat['dy']}, deta: {dat['deta']}, tau_min: {dat['tau_min']}, dtau: {dat['dtau']}")
    print(f"  ntau_freezeout: {dat['ntau_freezeout']}, tau_min_MUSIC: {dat['tau_min_MUSIC']}")

    if local_rank is not None:
        half = nevents // 2
        start_event = local_rank * half
        end_event = start_event + half
    else:
        start_event = 0
        end_event = nevents 


    if 'flat_data' in file['t'].keys():
        print("reading flat_data")
        branch = file['t']['flat_data']
        nentries = branch.num_entries
        if end_event <= 0:
            end_event = nentries
        elif end_event > nentries:
            print(f'Requested more events ({end_event}) than available({nentries}); so using max available ({nentries})')
            end_event = nentries
        dat['ntau_start'] = file['t']['ntau_start'].array(entry_stop=end_event,library='np')
        dat['ntau'] = file['NT'].member('fVal')

        arr = branch.array(entry_start=start_event,entry_stop=end_event,library='np').astype(np.float32)
        ex_size = dat['ntau']*dat['nx']*dat['ny']*dat['neta']*dat['nFeatures']
        
        if arr.shape[-1] != ex_size:
            raise ValueError(f"Unexpected data size: got {arr.shape[-1]}, expected {ex_size}")
        # print(arr.shape)
        # print('size supposed: ', nevents*dat['ntau']*dat['nx']*dat['ny']*dat['neta']*dat['nFeatures'])
        # arr = ak.to_numpy(branch.array(entry_stop=nevents)).astype(np.float32)
        # print(arr.shape)
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
            # print locations of NaN entries (6D array: event, tau, nx, ny, neta, feature)
            nan_locs = np.argwhere(np.isnan(arr))
            print('Locations of NaNs in arr:')
            print(type(nan_locs[0]))
            for loc in nan_locs:
                print(tuple(int(i) for i in loc))
            if len(nan_locs) > 100:
                print(f'... and {len(nan_locs) - 100} more')
            # raise ValueError(f"{num_nan} NaN entries found in data array")
        dat['nans'] = np.argwhere(np.isnan(arr)) if num_nan else None
        
        print('Read array of shape:', arr.shape)
        return dat
    else:
        ''' must be reading straight from .root, presumable with variable length tau dimension'''
        branch = file['t']['user_res']
        if not choose_ntau:
            choose_ntau = np.max(dat['ntau_freezeout'])
            #choose_ntau = int(np.percentile(dat['ntau_freezeout'], 99)) #better if outlier events are not necessay!? Unclear!?
        
        nentries = branch.num_entries

        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
        
        if nevents <= 0:
            nevents = nentries
        elif nevents > nentries:
            print(f'Requested more events ({nevents}) than available({nentries}); so using max available ({nentries})')
            nevents = nentries

        target_len = int(dat['nx']*dat['ny']*dat['neta']*dat['nFeatures']*choose_ntau)
        print(f'There were {nevents} events')

        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
        print(f"Processing {nevents} events with variable-length tau dimension; padding to {choose_ntau} taus per event (total size per event: {target_len} entries)")
        
        # Allocate the full output array once, up front
        arr = np.zeros(
            (nevents, choose_ntau, dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']), dtype=np.float32
        )

        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
            
        # Process in chunks to bound peak memory
        chunk_size = nevents // 10 if nevents >= 10 else 1
        for start in range(0, nevents, chunk_size):
            print(f"Processing events {start} to {min(start + chunk_size, nevents)}...")
            stop = min(start + chunk_size, nevents)
            
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
            
            arr_ak = branch.array(entry_start=start, entry_stop=stop)
            padded = ak.to_numpy(
                ak.fill_none(ak.pad_none(arr_ak, target_len, axis=-1, clip=True),
                            np.float32(0.))
            ).astype(np.float32, copy=False)
            
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
            # write directly into the pre-allocated slice — no accumulation
            arr[start:stop] = padded.reshape(
                stop - start, choose_ntau, dat['nx'], dat['ny'], dat['neta'], dat['nFeatures']
            )
            current, peak = tracemalloc.get_traced_memory()
            print(f"[Memory tracemalloc] Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
            
            del arr_ak, padded   # intermediate freed immediately
        
        current, peak = tracemalloc.get_traced_memory()
        
        print('Before moveaxis - Read array of shape:', arr.shape)
        
        arr = np.moveaxis(arr, 1, -1)
        arr = np.moveaxis(arr, -2, 1)
        dat['arr'] = arr
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"[Memory tracemalloc] All done with padding Current: {current / 1024**3:.2f} GiB; Peak: {peak / 1024**3:.2f} GiB")
        
        num_nan = np.sum(np.isnan(arr))
        if num_nan:
            print(f'   Warning: there are {num_nan} NaN entries in the data array')
            raise ValueError(f"{num_nan} NaN entries found in data array") 
        
        print('Read array of shape:', arr.shape)
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
    # print('Testing with RootBulkWriter:')
    # file_path = '/home/davidstewart/xscape-docker/X-SCAPE/build/dAu_seed0_250ev.root'
    data_0 = read_3d_data('/home/davidstewart/flat/flat3d_noscale.root', nevents=20)
    # keys = data.keys()
    # for key in keys:
    #     if key == 'arr':
    #         print(f"{key}: array of shape {data[key].shape}")
    #     else:
    #         print(key, ":", data[key])
    print('minimum value: ', np.min(data_0['arr'][:,0,:,:,:]))
    
    print('\nTesting with flat3d file:')
    file_path = '/home/davidstewart/flat/flat3d_tauep75.root'
    data = read_3d_data(file_path, nevents=20)
    keys = data.keys()
    # for key in keys:
    #     if key == 'arr':
    #         print(f"{key}: array of shape {data[key].shape}")
    #     else:
    #         print(key, ":", data[key])
    print(np.sum(np.isnan(data['arr'])))

    for loc in data['nans']:
        print(type(loc))
        print(tuple(int(i) for i in loc))
        print(data_0['arr'][tuple(int(i) for i in loc)])
