import numpy as np
import uproot  as up
import read_3d_root

def load_event(file_name, tree_name, branch_name, ievent):
    with up.open(file_name) as file:
        tree = file[tree_name]
        branch = tree[branch_name]
        event_data = branch.array(library="ak", entry_start=ievent, entry_stop=ievent + 1)[0].to_numpy()
        nx = file['nx'].member('fVal')
        ny = file['ny'].member('fVal')
        neta = file['neta'].member('fVal')
        nFeatures = file['nFeatures'].member('fVal')
        dat_len = nx * ny * neta * nFeatures
        nevents = len(event_data) // dat_len
        arr = event_data.reshape((nevents, nx, ny, neta, nFeatures))
        arr = np.moveaxis(arr, -1, 1)
        arr = np.moveaxis(arr, 0, -1)
    return arr

if __name__ == "__main__":
    data = load_event('/home/davidstewart/xscape-docker/X-SCAPE/build/dAu_seed0_25ev_10cen.root', 't', 'user_res',ievent=0)
    print(' -> XSCAPE event shape: ', data.shape)
    
    for i in range(5):
        print('time ',i,'XSCAPE ROOT FILE: ',np.max(data[0,:,:,2,i]))

    arr = read_3d_root.read_3d_data('/home/davidstewart/flat/dAu3d/25_ev_full.root', nevents=5)['arr']
    print(' -> flat event shape: ', arr.shape)
    print(np.max(arr[0,0,:,:,2,1]))
    for i in range(5):
        print('time ',i,'FLAT ROOT FILE: ',np.max(arr[0,0,:,:,2,i]))
    print('first five: ', arr[0,0,0,0,:,0])
    # for i in range(arr.shape[0]):
        # print(np.max(arr[i,0,:,:,2,1]))

    # for key in keys:
        # if key == 'arr':
            # print(f"{key}: array of shape {data[key].shape}")
        # else:
            # print(key, ":", data[key])


