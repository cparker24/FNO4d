import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Results for ckpt_n1
    """)
    return


@app.cell
def _():
    import os
    print(os.getcwd())
    import sys 
    import torch 

    sys.path.append('../../loc_libs')
    #import contour_v_plot
    import read_3d_root
    from os import path
    import numpy as np
    import matplotlib.pyplot as plt
    import parse_training_log

    input_stub = '.' # local file
    input_file = '/home/davidstewart/flat/dAu3d/full_50ev_b_lt_2fm_B.root'

    indir = f'.' # current run
    #model_input = f'{indir}/best_model_state_dict.pt'
    model_input = f'./best_model_state_dict.pt'
    log_file = f'{indir}/log.log'
    with open(log_file, 'r') as f:
        for line in f.readlines():
            #print(line)
            if 'n_modes' in line:
                n_modes = [int(x) for x in line.strip().split('[')[-1][:-1].split(',')]
            if 'hidden_channels' in line:
                hidden_channels = int(line.split()[-1])
            if 'NT:' in line:
                model_steps = int(line.split()[-1])

    arr = read_3d_root.read_3d_data(input_file, nevents=25)['arr']
    #arr = arr[:,:,:-1,:-1,:-1,:]
    arr = arr[:,:, :-1, :-1, :-1, :]
    print(f'Array shape: {arr.shape}')
    #arr = arr.mean(axis=4)
    #arr = arr[:,:,:,:,np.newaxis,:]

    print(f'Final data shape: {arr.shape}')

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)
    print(f'n_modes: {n_modes}, hidden_channels: {hidden_channels}, steps {model_steps}')
    return (
        arr,
        device,
        hidden_channels,
        model_input,
        model_steps,
        n_modes,
        np,
        parse_training_log,
        plt,
        sys,
        torch,
    )


@app.cell
def _(arr, np, plt):
    plt.imshow(arr[1,0,:,:,4,10])
    for i in range(0):
        print(i,np.max(arr[1,0,:,:,i,10]))
    #rint(arr[1,0,5,5,0,0I])
    plt.colorbar()
    plt.show()
    print(type(arr))
    return


@app.cell
def _(device, hidden_channels, model_input, n_modes, torch):
    # get the input model
    from neuralop.models import FNO

    # this is really messy -- make cleaner
    in_dict = torch.load(model_input, map_location=device,weights_only=False)
    new_dict = {}
    for key in in_dict.keys():
        #print(key)
        if key[:7] == "module.":
            new_dict[key[7:]] = in_dict[key]
        else:
            new_dict[key] = in_dict[key]
    del in_dict


    model = FNO(
        in_channels=4,        # Input channels (e.g., velocity field)
        out_channels=4,       # Output channels
        # positional_embedding=GridEmbeddingND(in_channels=3, dim=3, grid_boundaries=[[-15,15],[-15,15],[3.5,3.5001]]),
        #  positional_embedding=GridEmbeddingND(in_channels=3, dim=3, grid_boundaries=[[-15,15],[-15,15],[0.6,15.5]]),
        n_modes=n_modes, # [60,60,50],    # Number of modes in each layer
        hidden_channels=hidden_channels,   # 20             # Width of the network
        projection_channel_ratio=2
    ).to(device)
    #
    #
    #
    model.load_state_dict(new_dict)
    model.eval()
    return (model,)


@app.cell
def _(arr, device, model, model_steps, np, torch):
    # use the model on the input data
    def get_xy(in_data, i):
        time_steps=in_data.shape[-1]
        x = in_data[i,:,:,:,:,:1]
        x = np.repeat(x, time_steps-1,axis=-1)
        y = in_data[i,:,:,:,:,1:time_steps]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        return x, y

    def model_result(model, data, model_print=True):
        x_out = []
        y_out = []
        model_out = []
        print(data.shape)
        print(data[0].shape)
        for i in range(data.shape[0]):
            x, y = get_xy(data, i)
            xin = x.unsqueeze(0).to(device)
            out = model(xin).detach().cpu().numpy()
            x_out.append( x[:,:,:,:,0].cpu().numpy() )
            y_out.append(y[:,:,:,:,:].cpu().numpy())
            # y_out.append(_y_out[..., np.newaxis])
            model_out.append( out[0] )

        x_out = np.stack(x_out)
        y_out = np.stack(y_out)
        model_out = np.stack(model_out)

        if model_print:
            print('x_out shape:', x_out.shape)
            print('y_out shape:', y_out.shape)
            print('model_out shape:', model_out.shape)

        return {'x':x_out, 'y':y_out, 'model':model_out}

    def iter_model_result(model, data, model_steps=model_steps, model_print=True):
        ''' Like model_result, but iterates using steps of size model_steps, until it
        gets up to at least the actual steps of the input '''
        print (f'Model Steps: {model_steps}')
        actual_steps = data.shape[-1]
        n_iter = (actual_steps-1)//(model_steps-1)
        print(f'n_iter {n_iter} from {actual_steps-1} / {model_steps-1}')

        x_out = []
        y_out = []
        model_out = []

        for i in range(data.shape[0]):
            x0 = data[i,:,:,:,:,:1]
            x_out.append(x0)
            x0 = np.repeat(x0,model_steps-1,axis=-1)

            print(type(x0))
            x = torch.FloatTensor(x0)
            xin = x.unsqueeze(0).to(device)
            out = model(xin).detach().cpu().numpy()
            out_arr = [out,]
            for j in range(n_iter):
                #print(f'bb {len(out_arr)} and size {out_arr[-1].shape}')
                xj = out_arr[-1][0,:,:,:,:,-1:]
                xj = np.repeat(xj,model_steps-1,axis=-1)
                xj = torch.FloatTensor(xj).unsqueeze(0).to(device)
                out_arr.append(model(xj).detach().cpu().numpy())

            out_arr = np.concatenate(out_arr,axis=-1)
            out_arr = out_arr[0]
            model_out.append(out_arr)
            #print(f'AA {out_arr.shape}')

        x_out = np.stack(x_out)
        y_out = data[:,:,:,:,:,1:]
        #print('AA0 ',len(model_out), model_out[0].shape, type(model_out[0]))
        model_out = np.stack(model_out,axis=0)
        print(f'shape model_out : {model_out.shape}')

        return {'x':x_out, 'y':y_out, 'model':model_out}

    pred = iter_model_result(model, arr, model_print=True)
    print(arr.shape)
    print(pred['x'].shape, pred['y'].shape, pred['model'].shape)
    return (pred,)


@app.cell
def _(plt, pred):
    for _i in range(25):
        plt.imshow(pred['y'][_i,0,:,:,3,0])
        plt.title(f'Event {_i}')
        plt.show()
    return


@app.cell
def _(pred, sys):
    sys.path.append('../../loc_libs')
    from skinnycontour import plot_three_bins_contour, plot_contour

    #print(np.all(np.abs(pred['y'][0,0,:,:,0,35])<0.0001))
    #print(pred['y'][0,0,:,:,0,35])
    iT= (11, 15,-1)
    #iT = (11,15,17)
    if True:
      for i_ in (3,):
        print(i_)
        for eta_ in range(pred['model'].shape[4]):
            plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=i_, tau0=0.7, ieta=eta_, tag=f'eta={eta_-5}', rescale_energy=True)
            #plt.imshow(pred['model'][i_,0,:,:,eta_,14],origin='lower')
            #plt.title(f'Event {i_} eta {eta_-5} tau 1.7 fm/c')
            #plt.colorbar()
            #plt.show()
        #plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=i_, tau0=0.7, ieta=0, tag=f'eta={-2}')
        #plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=i_, tau0=0.7, ieta=1, tag=f'eta={-1}')
        #plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=i_, tau0=0.7, ieta=2, tag=f'eta={0}' )
        #plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=i_, tau0=0.7, ieta=3, tag=f'eta={1}' )
    #plot_three_bins_contour(pred['model'], pred['y'], iT=iT, event=1, tau0=0.7, ieta=0, tag='eta=-2', rescale=2., xlim=(20,40), ylim=(20,40))
    #plot_three_bins_contour(pred['model'], pred['y'], iT=(10,20,-1), event=2, tau0=0.7)
    #plot_three_bins_contour(pred['model'], pred['y'], iT=(10,20,-1), event=3, tau0=0.7)
    #plot_three_bins_contour(pred['model'], pred['y'], iT=(10,20,-1), event=4, tau0=0.7)
    return


@app.cell
def _(parse_training_log, plt):
    data = parse_training_log.plot_training_log('./training.log','',first_entry=0, times=10)
    print(type(data))
    print(len(data))
    print(data)
    plt.show()
    return


@app.cell
def _(arr, plt):
    import skinnycontour as sk
    print(arr.shape)
    itau = 14
    _dat = arr[4,0,:,:,0,itau].copy()
    _dat[_dat>0] = (_dat[_dat>0]/(itau/10.+0.7))**(4./3.)
    fig, axis = plt.subplots(1,1)
    axis.imshow(_dat,origin='lower')
    plt.colorbar(axis.images[0])
    sk.draw_contour_lines(_dat, cdfrat=0.9)
    plt.show()
    return


@app.cell
def _(arr, arr_super, device, model, model_steps, np, torch):
    # XY projection super resolution
    #def get_xy_super(in_data, i):
    #    time_steps=in_data.shape[-1]
    #    x = in_data[i,:,:,:,:,:1]
    #    x = np.repeat(x, time_steps-1,axis=-1)
    #    print(f'x shape before: {x.shape}')
    #    x = np.repeat(x,2,axis=-2)
    #    print(f'x shape after: {x.shape}')
    #    print(f'y before {in_data[i,:,:,:,1:time_steps].shape}')
    #    y = np.repeat(in_data[i,:,:,:,:,1:time_steps],2,axis=-2) # just double the steps in y
    #    print(f'y after {y.shape}')
    #
    #    x = torch.FloatTensor(x)
    #    y = torch.FloatTensor(y)
    #
    #    return x, y
    #
    #def model_result_super(model, data, model_print=True):
    #    print('fixme a0')
    #    x_out = []
    #    y_out = []
    #    model_out = []
    #    print(data.shape)
    #    print(data[0].shape)
    #    for i in range(data.shape[0]):
    #        x, y = get_xy_super(data, i)
    #        xin = x.unsqueeze(0).to(device)
    #        out = model(xin).detach().cpu().numpy()
    #        x_out.append( x[:,:,:,:,0].cpu().numpy() )
    #        y_out.append(y[:,:,:,:,:].cpu().numpy())
    #        # y_out.append(_y_out[..., np.newaxis])
    #        model_out.append( out[0] )
    #
    #    x_out = np.stack(x_out)
    #    y_out = np.stack(y_out)
    #    model_out = np.stack(model_out)
    #
    #    if model_print:
    #        print('x_out shape:', x_out.shape)
    #        print('y_out shape:', y_out.shape)
    #        print('model_out shape:', model_out.shape)
    #
    #    return {'x':x_out, 'y':y_out, 'model':model_out}

    def iter_model_result_super(model, data, model_steps=model_steps, model_print=True):
        ''' Like model_result, but iterates using steps of size model_steps, until it
        gets up to at least the actual steps of the input '''
        print (f'Model Steps: {model_steps}')
        actual_steps = data.shape[-1]
        n_iter = (actual_steps-1)//(model_steps-1)
        print(f'n_iter {n_iter} from {actual_steps-1} / {model_steps-1}')

        x_out = []
        y_out = []
        model_out = []

        for i in range(data.shape[0]):
            x0 = data[i,:,:,:,:,:1]
            x_out.append(x0)
            x0 = np.repeat(x0,model_steps-1,axis=-1)
            print(f'x0 before: {x0.shape}')
            x0 = np.repeat(x0,2,axis=-2)
            print(f'x0 after: {x0.shape}')

            print(type(x0))
            x = torch.FloatTensor(x0)
            xin = x.unsqueeze(0).to(device)
            out = model(xin).detach().cpu().numpy()
            out_arr = [out,]
            for j in range(n_iter):
                #print(f'bb {len(out_arr)} and size {out_arr[-1].shape}')
                xj = out_arr[-1][0,:,:,:,:,-1:]
                xj = np.repeat(xj,model_steps-1,axis=-1)
                xj = torch.FloatTensor(xj).unsqueeze(0).to(device)
                out_arr.append(model(xj).detach().cpu().numpy())

            out_arr = np.concatenate(out_arr,axis=-1)
            out_arr = out_arr[0]
            model_out.append(out_arr)
            #print(f'AA {out_arr.shape}')

        x_out = np.stack(x_out)
        y_out = data[:,:,:,:,:,1:]
        #print('AA0 ',len(model_out), model_out[0].shape, type(model_out[0]))
        model_out = np.stack(model_out,axis=0)
        print(f'shape model_out : {model_out.shape}')

        return {'x':x_out, 'y':y_out, 'model':model_out}

    print("FIXME B0")
    pred_super = iter_model_result_super(model, arr[:6], model_print=True)
    print("FIXME B1")
    print(arr_super.shape)
    print(pred_super['x'].shape, pred_super['y'].shape, pred_super['model'].shape)
    return (pred_super,)


@app.cell
def _(plt, pred_super):
    sdat = pred_super['model']
    print(sdat.shape)

    for z in range(8):
        plt.imshow(sdat[4,0,:,:,z,50])
        plt.show()
    return


if __name__ == "__main__":
    app.run()
