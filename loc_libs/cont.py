import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from skimage.measure import find_contours


def get_mask_cdf(X, cdfval=None, threshold=None, *, doprint=False):
    ''' 
    input: 
        X: tensor, cdfval: float in 0..1 
    output: 
        mask: tensor of same shape as X, threshold
        
    The mask is true for all highest valued entries in X
    up to such a number that the sum of these values
    is X.sum() * cdfval
    '''
    # Flatten the tensor
    flat_X = X.flatten()
    if doprint: 
        print(flat_X)
    
    # Sort the flattened tensor
    sorted_indices = np.argsort(flat_X)[::-1]

    if doprint: 
        print(sorted_indices)

    sorted_X = flat_X[sorted_indices]

    if doprint: 
        print(sorted_X)

    if threshold is None:
        # Calculate the cumulative sum
        cumulative_sum = np.cumsum(sorted_X)
        if doprint: 
            print(['cum_sum',cumulative_sum])
        
        # Find the threshold value for the top cdfval
        total_sum = cumulative_sum[-1]
        threshold_index = np.searchsorted(cumulative_sum, cdfval * total_sum)

        threshold = sorted_X[threshold_index]

    

    if doprint:
        print(f'cdfval: {cdfval}, threshold: {threshold} threshold_index: {threshold_index}')
    
    # Create a mask for the top cdfval% values
    mask = np.zeros_like(flat_X, dtype=bool)
    mask[sorted_indices[:threshold_index + 1]] = True
    
    # Reshape the mask to the original tensor shape
    mask = mask.reshape(X.shape)
    
    return mask, threshold

def get_topcdf_value(X, cdf_ratio):
    '''
    input: X: tensor, cdf_ratio: float in 0..1
    output: float
    Return the value of the cdf_ratio-th percentile of the
    values in X
    '''
    # Flatten the tensor
    flat_X = X.flatten()
    
    # Sort the flattened tensor
    sorted_X = np.sort(flat_X)[::-1]
    
    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(sorted_X)
    
    # Find the threshold value for the top cdfval
    total_sum = cumulative_sum[-1]
    threshold_index = np.searchsorted(cumulative_sum, cdf_ratio * total_sum)
    
    return sorted_X[threshold_index]

def draw_contour_lines(dat, *,  rescale=None, val=None, cdfrat=None, linestyle='dashed',color='red',linewidth=1):
    ''''
    input: mask: tensor
    output: None
    Draw dotted lines around the values that are True in
    the mask
    '''
    if val is None and cdfrat is None:
        raise ValueError('Either val or cdfrat must be specified')

    if val is None:
        val = get_topcdf_value(dat, cdfrat)

    contours = find_contours(dat, level=val)
    if rescale:
        try:
            contours[0] = contours[0]*rescale
        except:
            print('failed to find contrours for rescale')
    
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth)

    return val

def plot_one_bin_contour(T, iparam=0, event=0, iT=2,
    cdfrats=[0.05,.1,.2,.3,.4], ylim=(15,45), xlim=(15,45), topadjust=0.85, 
    tag='', fn=None, rescale=None, save=None):

    if iparam == 0:
        tag = f'Energy: {tag}'
    # elif iparam == 1:
        # tag = f'Temp: {tag}'
    elif iparam == 1:
        tag = f'Vy: {tag}'
    elif iparam == 2:
        tag = f'Vx: {tag}'


    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    i=0

    if True:
        # title = f'{0.6+iT*0.1:.1f} fm/$c$'

        dat_truth = T[event,iparam,:,:,iT]
        # dat_model = P[event,iparam,:,:,iT]

        axes.imshow(dat_truth, origin='lower', cmap='viridis', extent=(0, 60, 0, 60))
        plt.sca(axes)

        for cdf in cdfrats:
            Tval = draw_contour_lines(dat_truth, rescale=rescale, cdfrat=cdf, color='black', linestyle='solid')

    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_xlabel('')

    im = axes.images[0]
    im.set_norm(plt.matplotlib.colors.LogNorm(vmin=0.001, vmax=14))
    axes.set_ylabel('')

    # fig.suptitle(f'{tag} Event {event}. Contours: Truth: black. FNO: red.', fontsize=16)
    if save:
        fig.savefig(save)
    plt.show()

def draw_contour_lines(dat, *,  rescale=None, val=None, cdfrat=None, linestyle='dashed',color='red',linewidth=1, styles=[],alpha=1):
    '''
    input: mask: tensor
    output: None
    Draw dotted lines around the values that are True in
    the mask
    '''
    if val is None and cdfrat is None:
        raise ValueError('Either val or cdfrat must be specified')

    if val is None:
        val = get_topcdf_value(dat, cdfrat)

    contours = find_contours(dat, level=val)
    if rescale:
        try:
            contours[0] = contours[0]*rescale
        except:
            print('failed to find contrours for rescale')
    
    if styles:
        for n, contour in enumerate(contours):
            for linestyle, color, linewidth in styles:
                plt.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)
    else:
        for n, contour in enumerate(contours):
            plt.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)

    return val

def plot_three_coarse (P, T, iparam=0, event=0, iT=(0, 25, 48),
   cdfrats=[0.05,.1,.2,.3,.4,.6], ylim=(15,45), xlim=(15,45), topadjust=0.85, 
   tag='', show_FNO=True, show_cdflines=True, 
   show_cdfvals=True, fn=None, rescale=None, save=None, fig=None, axes=None):

    if show_cdfvals and not show_FNO:
      raise ValueError('show_FNO requires show_cdfvals to be True')

    if iparam == 0:
        tag = f'Energy: {tag}'
    # elif iparam == 1:
        # tag = f'Temp: {tag}'
    elif iparam == 1:
        tag = f'Vy: {tag}'
    elif iparam == 2:
        tag = f'Vx: {tag}'


    if not fig and not axes:
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))

    assert len(iT) == 3

    for i, iT in enumerate(iT):
        title = ''

        dat_truth = T[event,iparam,:,:,iT]
        dat_model = P[event,iparam,:,:,iT]

        if fn is not None:
            dat_truth = fn(dat_truth, iT)
            dat_model = fn(dat_model, iT)

        axes[i].imshow(dat_truth, origin='lower', cmap='viridis', extent=(0, 30, 0, 30))
        plt.sca(axes[i])

        # Rescale tick marks
        if ylim == (15, 45):
            ticks = np.arange(15, 50, 5)  # Original tick positions
            new_labels = ticks / 2  # Rescale the labels
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(new_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(new_labels)
            plt.sca(axes[i])
            axes[i].tick_params(axis='both', which='major', length=8, width=2, labelsize=15)
            axes[i].tick_params(axis='both', which='minor', length=5, width=1)
        elif ylim == (0, 60):
            ticks = np.arange(0, 61, 10)
            new_labels = ticks / 2  # Rescale the labels
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(new_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(new_labels)
            plt.sca(axes[i])

        cnt = 0
        xcntl = 0.015
        xcnt = {0: xcntl, 1: xcntl, 2: xcntl, 3: xcntl, 4: xcntl, 5: xcntl}
        ycnt = {0: 0.93, 1: 0.87, 2: 0.81, 3: 0.13, 4: 0.07, 5: 0.01}

        tcolor = "#a9e4f3"  # lighter red

        axes[i].text( 
            0.655,
            0.94,
            r'$\mathbf{\tau}$'+f'={0.6+iT*0.1:.1f} '+r'fm/$\mathbf{c}$',
            color='white',
            fontsize=13,
            ha='left',
            va='bottom',
            transform=axes[i].transAxes,
            bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
            usetex=False,
            fontstyle='normal',
            fontweight='bold'
        )

        if show_cdflines:
            Tvals = []
            cnt = 0
            # draw truth lines first, and then the model lines
            for cdf in cdfrats:
                Tval=draw_contour_lines(dat_truth, rescale=rescale, cdfrat=cdf, color=tcolor, linestyle='solid', styles=[('solid', 'white', 1.5),('dotted','black',1.5)])
                Tvals.append(Tval)

                if show_cdfvals:
                    textw = f'{Tval:4.2f}'

                    axes[i].text(
                        xcnt[cnt],
                        ycnt[cnt],
                        f"{int(cdf*100):.0f}$^{{\\mathbf{{th}}}}$",
                        color='white',
                        fontsize=13,
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
                        usetex=False,
                        fontstyle='normal',
                        fontweight='bold'
                    )
                    axes[i].text(
                        xcnt[cnt]+0.15,
                        ycnt[cnt],
                        textw,
                        color='white',
                        fontsize=13,
                        fontweight='bold',
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                    )
                    cnt += 1

            if show_FNO:
                for n, Tval in enumerate(Tvals):
                    cdf = cdfrats[n]

                    Pval = draw_contour_lines(dat_model, rescale=rescale, cdfrat=cdf, color='red', linestyle='solid',linewidth=1.8, alpha=0.5)
                    # title += f'\n{cdf*100:2.0f}%: P={Pval:5.3f}'
                    delta = Pval - Tval
                    drat = (delta / Tval)*100
                    if abs(drat) < 1.:
                        textp = f'{drat:.2f}'
                    elif abs(drat) < 10:
                        textp = f'{drat:.1f}'
                    else:
                        textp = f'{drat:.0f}'
                    if drat < 0:
                        textp = f'({textp}%)'
                    else:
                        textp = f'(+{textp}%)'

                    axes[i].text(
                        xcnt[n]+0.32,
                        ycnt[n],
                        textp,
                        color="#fc5858",  # lighter red
                        fontsize=13,
                        fontweight='bold',
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                    )

            axes[i].set_title(title)
            if ylim:
                axes[i].set_ylim(ylim)
            if xlim:
                axes[i].set_xlim(xlim)
            axes[i].set_xlabel('x [fm]')
            axes[i].set_ylabel('y [fm]', fontsize=16)
            axes[i].set_xlabel('x [fm]', fontsize=16)
            cbar = plt.colorbar(axes[i].images[0], ax=axes[i])
            cbar.ax.tick_params(labelsize=16)
            # plt.colorbar(axes[i].imagesg[0], ax=axes[i])


    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(hspace=.00)
    plt.subplots_adjust(wspace=.1)
    plt.subplots_adjust(top=0.88)
    plt.subplots_adjust(bottom=0.16)  # Add space at the bottom for x-axis labels2
    plt.subplots_adjust(left=0.05)  # Add space at the bottom for x-axis labels2

    fig.suptitle(f'{tag} Event {event}. Contours: Truth: black. FNO: red.', fontsize=16)
    if save:
        fig.savefig(save)
    plt.show()


def comp_2hist(A, B, cdf0, cdf1, fn=None, vprint=False):
    '''
    input: A, B: 2D numpy arrays

    if cdf_ratio and not value, get the value from cdf_ratio.

    Get the masks for A and B: mask_A, mask_B
    Get mu and std for differences between A and B
    Get N_b_no
    Get the 
    

    return: {   'cdf_ratio' : float,
                'val' : float,
                'mask_A' : 2D numpy array,
                'mask_B' : 2D numpy array,
                'mu_delta_AandB' : float,
                'std_delta_AandB' : float,
                'mu_delta_BnotA' : float,
                'std_delta_BnotA' : float,
                'mu_delta_AnotB' : float,
                'std_delta_AnotB' : float,
                'nmask_rat' : float, # nm_B_and_A / nm_A
                'nmask_extra_rat' : float, # nm_B_and_not_A / nm_A
                'nmake_miss_rat' : float, # nm_A_and_not_B / nm_A
             '}
    '''

    mask_A, val_A = get_mask_cdf(A, cdfval=cdf1)
    mask_B, val_B = get_mask_cdf(B, cdfval=cdf1)


    if vprint:
        print (f' mask_A   size: {np.sum(mask_A)}, mask_B size: {np.sum(mask_B)}')
    if cdf0 is not None:
        _, _val_A = get_mask_cdf(A, cdfval=cdf0)
        mask_A = mask_A & ~_

        _, _val_B = get_mask_cdf(B, cdfval=cdf0)
        mask_B = mask_B & ~_

    # print (f' mask_A resize: {np.sum(mask_A)}, mask_B size: {np.sum(mask_B)}')

    mask_A_and_B = mask_A & mask_B
    mask_A_or_B = mask_A | mask_B


    return {
        'cdfcutoff_A' : val_A,
        'cdfcutoff_B' : val_B,
        'diff_cdfcutoff' : val_B - val_A,
        'mask_A' : mask_A,
        'mask_B' : mask_B,
        'mu_delta_AandB' : np.mean(B[mask_A_and_B] - A[mask_A_and_B]),
        'std_delta_AandB' : np.std(B[mask_A_and_B] - A[mask_A_and_B]),
        'mu_delta_AorB' : np.mean(B[mask_A_or_B] - A[mask_A_or_B]),
        'std_delta_AorB' : np.std(B[mask_A_or_B] - A[mask_A_or_B]),
        'mu_A' : np.mean(A[mask_A]),
        'mu_B' : np.mean(B[mask_B]),
        'nbins_kept' : np.sum(mask_A_and_B) ,
        'nbins_extra' : np.sum(mask_B & ~mask_A) ,
        'nbins_lost' : np.sum(mask_A & ~mask_B) ,
        'nbins_A' : np.sum(mask_A),
        'nbins_B' : np.sum(mask_B),
        'nbins_A_and_B' : np.sum(mask_A_and_B),
        'nbins_A_or_B' : np.sum(mask_A_or_B),
        'arr_mu_A' : A[mask_A],
        'arr_delta_AorB' :  A[mask_A_or_B] - B[mask_A_or_B],
    }

def collect_comp_2hist(dat_T, dat_P, iTime, iParam=0, cdf0=None, cdf1=None, fn=None):
    '''
    collect the values of of com_2hist to all comparable events
    '''
    nevents = dat_T.shape[0]
    if nevents < 2:
        raise ValueError('Need at least 2 events to compare')
    
    # if cdf1 is None:
        # cdf1 = 0.1

    if iTime is None:
        raise ValueError('iTime must be specified')

    T = dat_T[0,iParam,:,:,iTime]
    P = dat_P[0,iParam,:,:,iTime]

    if fn is not None:
        T = fn(T, iTime)
        P = fn(P, iTime)

    out_dict = comp_2hist(T, P, cdf0=cdf0, cdf1=cdf1, fn=fn)

    # convert the entries to a list
    for key in out_dict.keys():
        out_dict[key] = [out_dict[key],] + [None,]*(nevents-1)

    for iEvent in range(1, nevents):

        T = dat_T[iEvent,iParam,:,:,iTime]
        P = dat_P[iEvent,iParam,:,:,iTime]

        if fn is not None:
            T = fn(T, iTime)
            P = fn(P, iTime)

        out_dict_i = comp_2hist(T, P, cdf0=cdf0, cdf1=cdf1, fn=fn)

        for key in out_dict.keys():
            out_dict[key][iEvent] = out_dict_i[key]

    for key in out_dict.keys():
        if key[:4] == 'arr_':
            out_dict[key] = np.array([item for sublist in out_dict[key] for item in sublist])
    
    return out_dict

def collect_alltimes_comp_2hist(dat_T, dat_P, iParam=0, cdf0=0., cdf1=0.1, fn=None):
    olist = []
    for iTime in range(dat_T.shape[-1]):
        out_dict = collect_comp_2hist(dat_T, dat_P, iTime=iTime, iParam=iParam,
        cdf0=cdf0, cdf1=cdf1, fn=fn)
        olist.append(out_dict)
    return olist

def collect_mu_std(all_lists, key):
    x = []
    y = []
    y_err = []
    #FIXME
    for iTime in range(len(all_lists)):
        # print(f'key: {key} iTime {iTime}  len: {len(all_lists[iTime][key])})')
        x.append(iTime+1)
        y.append(np.mean(all_lists[iTime][key]))
        y_err.append(np.std(all_lists[iTime][key]))
    return x, y, y_err

def plot_vel_in_ring(data, iEvent, R0=5, R1 = 10, iTime=10, tag=''):
    '''
    Plot the velocity in a ring of radius R0 and R1
    '''
    vx = data['y'][iEvent,1,:,:,iTime]
    vy = data['y'][iEvent,2,:,:,iTime]

    n = np.arange(0, vx.shape[0])
    X, Y = np.meshgrid(n, n)   # Create a grid

    mask = (X**2 + Y**2 >= R0**2) & (X**2 + Y**2 <= R1**2)

    vx_ring = vx[mask]
    vy_ring = vy[mask]

    plt.quiver(X[mask], Y[mask], vx_ring, vy_ring, scale=0.8)
    plt.title(f'{tag}: Velocity field in ring at time {iTime+2} for event {iEvent}')
    plt.show()


def plot_quiver(data, key, iEvent, iTime,scale=0.8,tag=''):
    vx = data[key][iEvent,3,:,:,iTime]
    vy = data[key][iEvent,2,:,:,iTime]
    n = np.arange(0, vx.shape[0])
    X, Y = np.meshgrid(n, n)   # Create a grid
    cax = plt.quiver(X,Y,vx,vy,scale=1.,scale_units='xy', angles='xy')
    plt.imshow(np.sqrt(vx**2 + vy**2), cmap='viridis', alpha=0.5)
    plt.title(f'{tag}: Velocity field at time {iTime+2} for event {iEvent}')
    plt.colorbar(label='Velocity magnitude')
    plt.show()
    return cax

def get_vel_err(dat, iTime, iEvent):
    vx_T = dat['y'][iEvent,3,:,:,iTime]
    vy_T = dat['y'][iEvent,2,:,:,iTime]
    vx_M = dat['model'][iEvent,3,:,:,iTime]
    vy_M = dat['model'][iEvent,2,:,:,iTime]

    abs_vel_err = np.sqrt((vx_T-vx_M)**2 + (vy_T-vy_M)**2)
    phi_vel_err = np.arctan2(vy_M, vx_M) - np.arctan2(vy_T, vx_T)
    phi_vel_err = np.degrees(phi_vel_err)
    phi_vel_err = (phi_vel_err + 180) % 360 - 180
    
    return abs_vel_err, phi_vel_err, {'vx_T': vx_T, 'vy_T': vy_T, 'vx_M': vx_M, 'vy_M': vy_M}

def plot_vel_err(dat, iTime, iEvent, range_abs=None, range_phi=None):
    abs_err, phi_err, dat = get_vel_err(dat, iTime=iTime, iEvent=iEvent)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot absolute velocity error
    if range_abs is not None:
        im0 = ax[0].imshow(abs_err, cmap='viridis', alpha=0.5, vmin=range_abs[0], vmax=range_abs[1])
    else:
        im0 = ax[0].imshow(abs_err, cmap='viridis', alpha=0.5)
    ax[0].set_title(f'Absolute velocity error at time {iTime+2} for event {iEvent}')
    fig.colorbar(im0, ax=ax[0], label='Absolute Error')  # Add colorbar to ax[0]

    # Plot phi velocity error
    if range_phi is not None:
        im1 = ax[1].imshow(phi_err, cmap='viridis', alpha=0.5, vmin=range_phi[0], vmax=range_phi[1])
    else:
        im1 = ax[1].imshow(phi_err, cmap='viridis', alpha=0.5)  # Example range for angles
    ax[1].set_title(f'Phi velocity error at time {iTime+2} for event {iEvent}')
    fig.colorbar(im1, ax=ax[1], label='Phi Error')  # Add colorbar to ax[1]
    
    plt.show()

def print_area_and_E_density(data, cdf0, cdf1, tag='',fn=None):
    diffs= collect_alltimes_comp_2hist(data['model'], data['y'], cdf0=cdf0, cdf1=cdf1, fn=fn)

    print(len(diffs))
    area_lost = collect_mu_std(diffs, 'nbins_lost')
    area_extra = collect_mu_std(diffs, 'nbins_extra')
    area_A     = collect_mu_std(diffs, 'nbins_A')

    import matplotlib.pyplot as plt


    # plt.errorbar(area_lost[0], area_lost[1], yerr=area_lost[2], fmt='o',label='ratio area lost')
    # plt.errorbar(area_extra[0], area_extra[1], yerr=area_extra[2], fmt='o', label='ratio area extra')
    # plt.errorbar(area_A[0], area_A[1], yerr=area_A[2], fmt='o', label='area A')

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(15, 8))
    ax1, ax3, ax2, ax4 = axes.flatten()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(12, 10))
    # return fig, (ax1, ax2, ax3, ax4)
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, sharex=True, figsize=(12, 5))

    # Top panel
    ax1.errorbar(area_A[0], area_A[1], yerr=area_A[2], fmt='o', label='nbins truth')
    ax1.set_ylabel("N bins")
    ax1.legend()
    ax1.grid()

    # Bottom panel
    ax2.errorbar(area_lost[0], area_lost[1], yerr=area_lost[2], fmt='o', label='nbins lost')
    ax2.errorbar(area_extra[0], area_extra[1], yerr=area_extra[2], fmt='o', label='nbins extra')
    ax2.set_xlabel("time step")
    ax2.set_ylabel("Difference in N bins")
    ax2.legend()
    ax2.grid()

    if tag != '':
        tag = f"{tag}, "

    cdf_title = f"{cdf0}-{cdf1}" if cdf0 else f"0-{cdf1}"

    if cdf0:
        ax1.set_title(f"{tag} Results for bins representing {cdf_title} of total energy")
    else:
        ax1.set_title(f"{tag} Results for bins representing 0.-{cdf1} of total energy")
    # plt.tight_layout()
    # plt.show()

    # also plot the average energies
    mu_A = collect_mu_std(diffs, 'arr_mu_A')
    mu_delta_AorB = collect_mu_std(diffs, 'arr_delta_AorB')


    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    # Top panel
    ax3.errorbar(mu_A[0], mu_A[1], yerr=mu_A[2], fmt='o', label='mean energy density, truth')
    ax3.set_ylabel(r"Mean energy density")
    ax3.set_title(f'{tag} Results for bins representing {cdf_title} of total energy')
    ax3.legend()
    ax3.grid()
    # Bottom panel
    ax4.errorbar(mu_delta_AorB[0], mu_delta_AorB[1], yerr=mu_delta_AorB[2], fmt='o',
    label='per-cell difference in energy density (all cells truth and modelled)')
    ax4.set_xlabel("time step")
    ax4.set_ylabel(r"Mean energy density difference")
    ax4.legend()
    plt.tight_layout
    plt.show()


def get_cdf_rat_data(data, cdfrat_range=(0.0,0.1), iparam=(1,2), event=0, iT=0):
    '''
    Get the cells in in the cdfrat_range for energy. Find the x,y position of
    the energy weighted center of mass of these cells (ECOM).  Find the velocity
    distributions of these cells going radially outward from ECOM and curling
    around ECOM.
    Return these distributions:
        {'ecom_x': float, 'ecom_y', float, 'vradial', array, 'vcurl', array}
    '''
    return {}







def comp_2hist(A, B, cdf0, cdf1, fn=None, vprint=False):
    '''
    input: A, B: 2D numpy arrays

    if cdf_ratio and not value, get the value from cdf_ratio.

    Get the masks for A and B: mask_A, mask_B
    Get mu and std for differences between A and B
    Get N_b_no
    Get the 
    

    return: {   'cdf_ratio' : float,
                'val' : float,
                'mask_A' : 2D numpy array,
                'mask_B' : 2D numpy array,
                'mu_delta_AandB' : float,
                'std_delta_AandB' : float,
                'mu_delta_BnotA' : float,
                'std_delta_BnotA' : float,
                'mu_delta_AnotB' : float,
                'std_delta_AnotB' : float,
                'nmask_rat' : float, # nm_B_and_A / nm_A
                'nmask_extra_rat' : float, # nm_B_and_not_A / nm_A
                'nmake_miss_rat' : float, # nm_A_and_not_B / nm_A
             '}
    '''

    mask_A, val_A = get_mask_cdf(A, cdfval=cdf1)
    mask_B, val_B = get_mask_cdf(B, cdfval=cdf1)


    if vprint:
        print (f' mask_A   size: {np.sum(mask_A)}, mask_B size: {np.sum(mask_B)}')
    if cdf0 is not None:
        _, _val_A = get_mask_cdf(A, cdfval=cdf0)
        mask_A = mask_A & ~_

        _, _val_B = get_mask_cdf(B, cdfval=cdf0)
        mask_B = mask_B & ~_

    # print (f' mask_A resize: {np.sum(mask_A)}, mask_B size: {np.sum(mask_B)}')

    mask_A_and_B = mask_A & mask_B
    mask_A_or_B = mask_A | mask_B


    return {
        'cdfcutoff_A' : val_A,
        'cdfcutoff_B' : val_B,
        'diff_cdfcutoff' : val_B - val_A,
        'mask_A' : mask_A,
        'mask_B' : mask_B,
        'mu_delta_AandB' : np.mean(B[mask_A_and_B] - A[mask_A_and_B]),
        'std_delta_AandB' : np.std(B[mask_A_and_B] - A[mask_A_and_B]),
        'mu_delta_AorB' : np.mean(B[mask_A_or_B] - A[mask_A_or_B]),
        'std_delta_AorB' : np.std(B[mask_A_or_B] - A[mask_A_or_B]),
        'mu_A' : np.mean(A[mask_A]),
        'mu_B' : np.mean(B[mask_B]),
        'nbins_kept' : np.sum(mask_A_and_B) ,
        'nbins_extra' : np.sum(mask_B & ~mask_A) ,
        'nbins_lost' : np.sum(mask_A & ~mask_B) ,
        'nbins_A' : np.sum(mask_A),
        'nbins_B' : np.sum(mask_B),
        'nbins_A_and_B' : np.sum(mask_A_and_B),
        'nbins_A_or_B' : np.sum(mask_A_or_B),
        'arr_mu_A' : A[mask_A],
        'arr_delta_AorB' :  A[mask_A_or_B] - B[mask_A_or_B],
    }

def collect_comp_2hist(dat_T, dat_P, iTime, iParam=0, cdf0=None, cdf1=None, fn=None):
    '''
    collect the values of of com_2hist to all comparable events
    '''
    nevents = dat_T.shape[0]
    if nevents < 2:
        raise ValueError('Need at least 2 events to compare')
    
    # if cdf1 is None:
        # cdf1 = 0.1

    if iTime is None:
        raise ValueError('iTime must be specified')

    T = dat_T[0,iParam,:,:,iTime]
    P = dat_P[0,iParam,:,:,iTime]

    if fn is not None:
        T = fn(T, iTime)
        P = fn(P, iTime)

    out_dict = comp_2hist(T, P, cdf0=cdf0, cdf1=cdf1, fn=fn)

    # convert the entries to a list
    for key in out_dict.keys():
        out_dict[key] = [out_dict[key],] + [None,]*(nevents-1)

    for iEvent in range(1, nevents):

        T = dat_T[iEvent,iParam,:,:,iTime]
        P = dat_P[iEvent,iParam,:,:,iTime]

        if fn is not None:
            T = fn(T, iTime)
            P = fn(P, iTime)

        out_dict_i = comp_2hist(T, P, cdf0=cdf0, cdf1=cdf1, fn=fn)

        for key in out_dict.keys():
            out_dict[key][iEvent] = out_dict_i[key]

    for key in out_dict.keys():
        if key[:4] == 'arr_':
            out_dict[key] = np.array([item for sublist in out_dict[key] for item in sublist])
    
    return out_dict

def collect_alltimes_comp_2hist(dat_T, dat_P, iParam=0, cdf0=0., cdf1=0.1, fn=None):
    olist = []
    for iTime in range(dat_T.shape[-1]):
        out_dict = collect_comp_2hist(dat_T, dat_P, iTime=iTime, iParam=iParam,
        cdf0=cdf0, cdf1=cdf1, fn=fn)
        olist.append(out_dict)
    return olist

def collect_mu_std(all_lists, key):
    x = []
    y = []
    y_err = []
    #FIXME
    for iTime in range(len(all_lists)):
        # print(f'key: {key} iTime {iTime}  len: {len(all_lists[iTime][key])})')
        x.append(iTime+1)
        y.append(np.mean(all_lists[iTime][key]))
        y_err.append(np.std(all_lists[iTime][key]))
    return x, y, y_err

def plot_vel_in_ring(data, iEvent, R0=5, R1 = 10, iTime=10, tag=''):
    '''
    Plot the velocity in a ring of radius R0 and R1
    '''
    vx = data['y'][iEvent,1,:,:,iTime]
    vy = data['y'][iEvent,2,:,:,iTime]

    n = np.arange(0, vx.shape[0])
    X, Y = np.meshgrid(n, n)   # Create a grid

    mask = (X**2 + Y**2 >= R0**2) & (X**2 + Y**2 <= R1**2)

    vx_ring = vx[mask]
    vy_ring = vy[mask]

    plt.quiver(X[mask], Y[mask], vx_ring, vy_ring, scale=0.8)
    plt.title(f'{tag}: Velocity field in ring at time {iTime+2} for event {iEvent}')
    plt.show()


def plot_quiver(data, key, iEvent, iTime,scale=0.8,tag=''):
    vx = data[key][iEvent,3,:,:,iTime]
    vy = data[key][iEvent,2,:,:,iTime]
    n = np.arange(0, vx.shape[0])
    X, Y = np.meshgrid(n, n)   # Create a grid
    cax = plt.quiver(X,Y,vx,vy,scale=1.,scale_units='xy', angles='xy')
    plt.imshow(np.sqrt(vx**2 + vy**2), cmap='viridis', alpha=0.5)
    plt.title(f'{tag}: Velocity field at time {iTime+2} for event {iEvent}')
    plt.colorbar(label='Velocity magnitude')
    plt.show()
    return cax

def get_vel_err(dat, iTime, iEvent):
    vx_T = dat['y'][iEvent,3,:,:,iTime]
    vy_T = dat['y'][iEvent,2,:,:,iTime]
    vx_M = dat['model'][iEvent,3,:,:,iTime]
    vy_M = dat['model'][iEvent,2,:,:,iTime]

    abs_vel_err = np.sqrt((vx_T-vx_M)**2 + (vy_T-vy_M)**2)
    phi_vel_err = np.arctan2(vy_M, vx_M) - np.arctan2(vy_T, vx_T)
    phi_vel_err = np.degrees(phi_vel_err)
    phi_vel_err = (phi_vel_err + 180) % 360 - 180
    
    return abs_vel_err, phi_vel_err, {'vx_T': vx_T, 'vy_T': vy_T, 'vx_M': vx_M, 'vy_M': vy_M}

def plot_vel_err(dat, iTime, iEvent, range_abs=None, range_phi=None):
    abs_err, phi_err, dat = get_vel_err(dat, iTime=iTime, iEvent=iEvent)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot absolute velocity error
    if range_abs is not None:
        im0 = ax[0].imshow(abs_err, cmap='viridis', alpha=0.5, vmin=range_abs[0], vmax=range_abs[1])
    else:
        im0 = ax[0].imshow(abs_err, cmap='viridis', alpha=0.5)
    ax[0].set_title(f'Absolute velocity error at time {iTime+2} for event {iEvent}')
    fig.colorbar(im0, ax=ax[0], label='Absolute Error')  # Add colorbar to ax[0]

    # Plot phi velocity error
    if range_phi is not None:
        im1 = ax[1].imshow(phi_err, cmap='viridis', alpha=0.5, vmin=range_phi[0], vmax=range_phi[1])
    else:
        im1 = ax[1].imshow(phi_err, cmap='viridis', alpha=0.5)  # Example range for angles
    ax[1].set_title(f'Phi velocity error at time {iTime+2} for event {iEvent}')
    fig.colorbar(im1, ax=ax[1], label='Phi Error')  # Add colorbar to ax[1]
    
    plt.show()

def print_area_and_E_density(data, cdf0, cdf1, tag='',fn=None):
    diffs= collect_alltimes_comp_2hist(data['model'], data['y'], cdf0=cdf0, cdf1=cdf1, fn=fn)

    print(len(diffs))
    area_lost = collect_mu_std(diffs, 'nbins_lost')
    area_extra = collect_mu_std(diffs, 'nbins_extra')
    area_A     = collect_mu_std(diffs, 'nbins_A')

    import matplotlib.pyplot as plt


    # plt.errorbar(area_lost[0], area_lost[1], yerr=area_lost[2], fmt='o',label='ratio area lost')
    # plt.errorbar(area_extra[0], area_extra[1], yerr=area_extra[2], fmt='o', label='ratio area extra')
    # plt.errorbar(area_A[0], area_A[1], yerr=area_A[2], fmt='o', label='area A')

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(15, 8))
    ax1, ax3, ax2, ax4 = axes.flatten()

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, figsize=(12, 10))
    # return fig, (ax1, ax2, ax3, ax4)
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2, sharex=True, figsize=(12, 5))

    # Top panel
    ax1.errorbar(area_A[0], area_A[1], yerr=area_A[2], fmt='o', label='nbins truth')
    ax1.set_ylabel("N bins")
    ax1.legend()
    ax1.grid()

    # Bottom panel
    ax2.errorbar(area_lost[0], area_lost[1], yerr=area_lost[2], fmt='o', label='nbins lost')
    ax2.errorbar(area_extra[0], area_extra[1], yerr=area_extra[2], fmt='o', label='nbins extra')
    ax2.set_xlabel("time step")
    ax2.set_ylabel("Difference in N bins")
    ax2.legend()
    ax2.grid()

    if tag != '':
        tag = f"{tag}, "

    cdf_title = f"{cdf0}-{cdf1}" if cdf0 else f"0-{cdf1}"

    if cdf0:
        ax1.set_title(f"{tag} Results for bins representing {cdf_title} of total energy")
    else:
        ax1.set_title(f"{tag} Results for bins representing 0.-{cdf1} of total energy")
    # plt.tight_layout()
    # plt.show()

    # also plot the average energies
    mu_A = collect_mu_std(diffs, 'arr_mu_A')
    mu_delta_AorB = collect_mu_std(diffs, 'arr_delta_AorB')


    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
    # Top panel
    ax3.errorbar(mu_A[0], mu_A[1], yerr=mu_A[2], fmt='o', label='mean energy density, truth')
    ax3.set_ylabel(r"Mean energy density")
    ax3.set_title(f'{tag} Results for bins representing {cdf_title} of total energy')
    ax3.legend()
    ax3.grid()
    # Bottom panel
    ax4.errorbar(mu_delta_AorB[0], mu_delta_AorB[1], yerr=mu_delta_AorB[2], fmt='o',
    label='per-cell difference in energy density (all cells truth and modelled)')
    ax4.set_xlabel("time step")
    ax4.set_ylabel(r"Mean energy density difference")
    ax4.legend()
    plt.tight_layout
    plt.show()


_COM_dx_dy = namedtuple('_COM_dx_dy', ['cdf', 'dx', 'dy','com_x', 'com_y'])
def get_COM_dx_dy(e_M, cdfrat_range=(0.,0.1)):
    # e_M = data[i_event,i_e,:,:,iT]
    cdf_1 = get_mask_cdf(e_M, cdfrat_range[1], doprint=False)[0]
    if (cdfrat_range[0] == 0):
        cdf = cdf_1
    else:
        cdf_0 = get_mask_cdf(e_M, cdfrat_range[0], doprint=False)[0]
        cdf = ((~cdf_0) & cdf_1)
    indices = np.argwhere(cdf)

    weights = e_M[cdf]
    ecom_x = np.sum(indices[:, 0] * weights) / np.sum(weights)
    ecom_y = np.sum(indices[:, 1] * weights) / np.sum(weights)

    dx = indices[:, 0] - ecom_x
    dy = indices[:, 1] - ecom_y
    dr = np.sqrt(dx**2 + dy**2)
    dx /= dr
    dy /= dr

    return _COM_dx_dy(cdf=cdf, dx=dx, dy=dy, com_x = ecom_x, com_y=ecom_y)

def get_vrad_vcurl(vx_M, vy_M, cdf_dx_dy):
    vx = vx_M[cdf_dx_dy.cdf]
    vy = vy_M[cdf_dx_dy.cdf]

    vradial = vx * cdf_dx_dy.dx + vy * cdf_dx_dy.dy
    vcurl   = vy * cdf_dx_dy.dx - vx * cdf_dx_dy.dy

    return vradial, vcurl

def get_dotbin(vrad, bins=np.linspace(-.2,1.,21), norm=True):
    hist, edges = np.histogram(vrad, bins=bins, density=norm)
    return hist, 0.5*(edges[:-1]+edges[1:])

def get_curlbin(vcurl, bins=np.linspace(-0.5,0.5,21), norm=True):
    hist, edges = np.histogram(vcurl, bins=bins, density=norm)
    return hist, 0.5*(edges[:-1]+edges[1:])

def plot_dotcurl(P,T, ievxvy=(0,1,2), event=0, iT=10,
    cdfrats=((0,0.05), (0.05,0.1), (0.1, 0.3), (0.6,0.8), (0.9,1.)),
    save=None, fig=None, axes=None, draw_legend=True):

    prior_axis = (fig and  axes)
    if not prior_axis:
        fig, axes = plt.subplots(2, 1, figsize=(6,6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, cdf_rat in enumerate(cdfrats):
        com_dx_dy = get_COM_dx_dy(T[event,ievxvy[0],:,:,iT], cdfrat_range=cdf_rat)
        T_vrad, T_vcurl = get_vrad_vcurl(T[event,ievxvy[1],:,:,iT], T[event,ievxvy[2],:,:,iT], com_dx_dy)

        T_data_rad, r_bins = get_dotbin(T_vrad)
        T_data_curl, c_bins = get_curlbin(T_vcurl)

        P_vrad, P_vcurl = get_vrad_vcurl(P[event,ievxvy[1],:,:,iT], P[event,ievxvy[2],:,:,iT], com_dx_dy)
        P_data_rad, _ = get_dotbin(P_vrad)
        P_data_curl, _ = get_curlbin(P_vcurl)

        axes[0].plot(r_bins, T_data_rad, alpha=0.7, color=colors[i%len(colors)], label=f'True CDF {cdf_rat}', linestyle='-', linewidth=1)
        axes[0].plot(r_bins, P_data_rad, alpha=0.7, color=colors[i%len(colors)], linestyle=':', linewidth=4)

        axes[1].plot(c_bins, T_data_curl, alpha=0.7, color=colors[i%len(colors)], label=f'True CDF {cdf_rat}', linestyle='-', linewidth=1)
        axes[1].plot(c_bins, P_data_curl, alpha=0.7, color=colors[i%len(colors)], linestyle=':', linewidth=4)


    if not prior_axis:
        axes[0].set_ylim((0.,21.))
        axes[0].set_xlabel('velocity [c]')
        axes[0].set_ylabel('rad.vel. (eCOM) dist.')
        axes[1].set_ylim((0.,21.))
        axes[1].set_xlabel('velocity [c]')
        axes[1].set_ylabel('curl.vel. (eCOM) dist.')
    if draw_legend:
        axes[0].legend()
        # axes[1].legend()

def plot_contour(P: np.ndarray, # predicted 2D array
                 T: np.ndarray, # truth 2D array 
                 cdfrats = [0.05,.1,.2,.3,.4,.6],
                 axis: plt.Axes = None, # if none, create new figure
                 xlim: tuple = (15, 45),
                 ylim: tuple = None, # default to xlim
                 tau: float = 0.6,
                 #topadjust=0.85, 
                 #tag='', 
                 #show_FNO=True, 
                 show_cdflines=True, 
                 show_cdfvals=True, 
                 # fn=None, 
                 rescale=None, 
                 save=None, 
                 # fig=None, 
                 #axes=None,
                 taulabel_offset=0):

    if not axis:
        fig, axes = plt.subplots(1, 1, figsize=(4.5, 4.5))

    dat_truth = T
    dat_model = P

    axis.imshow(dat_truth, origin='lower', cmap='viridis', extent=(0, 60, 0, 60))
    plt.sca(axis)

    # Rescale tick marks
    if ylim == (15, 45):
        ticks = np.arange(15, 50, 5)  # Original tick positions
        new_labels = ticks / 2  # Rescale the labels
        axis.set_xticks(ticks)
        axis.set_xticklabels(new_labels)
        axis.set_yticks(ticks)
        axis.set_yticklabels(new_labels)
        plt.sca(axis)
        axis.tick_params(axis='both', which='major', length=8, width=2, labelsize=15)
        axis.tick_params(axis='both', which='minor', length=5, width=1)
    elif ylim == (0, 60):
        ticks = np.arange(0, 61, 10)
        new_labels = ticks / 2  # Rescale the labels
        axis.set_xticks(ticks)
        axis.set_xticklabels(new_labels)
        axis.set_yticks(ticks)
        axis.set_yticklabels(new_labels)
        plt.sca(axis)

    cnt = 0
    xcntl = 0.015
    xcnt = {0: xcntl, 1: xcntl, 2: xcntl, 3: xcntl, 4: xcntl, 5: xcntl}
    ycnt = {0: 0.93, 1: 0.87, 2: 0.81, 3: 0.13, 4: 0.07, 5: 0.01}

    tcolor = "#a9e4f3"  # lighter red

    axis.text( 
        0.655,
        0.94,
        r'$\mathbf{\tau}$'+f'={0.6+taulabel_offset*0.1+tau:.1f} '+r'fm/$\mathbf{c}$',
        color='white',
        fontsize=13,
        ha='left',
        va='bottom',
        transform=axes[i].transAxes,
        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
        usetex=False,
        fontstyle='normal',
        fontweight='bold'
    )

    if show_cdflines:
        Tvals = []
        cnt = 0
        # draw truth lines first, and then the model lines
        for cdf in cdfrats:
            Tval=draw_contour_lines(dat_truth, rescale=rescale, cdfrat=cdf, color=tcolor, linestyle='solid', styles=[('solid', 'white', 1.5),('dotted','black',1.5)])
            Tvals.append(Tval)

            if show_cdfvals:
                textw = f'{Tval:4.2f}'

                axes[i].text(
                    xcnt[cnt],
                    ycnt[cnt],
                    f"{int(cdf*100):.0f}$^{{\\mathbf{{th}}}}$",
                    color='white',
                    fontsize=13,
                    ha='left',
                    va='bottom',
                    transform=axes[i].transAxes,
                    bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
                    usetex=False,
                    fontstyle='normal',
                    fontweight='bold'
                )
                axes[i].text(
                    xcnt[cnt]+0.15,
                    ycnt[cnt],
                    textw,
                    color='white',
                    fontsize=13,
                    fontweight='bold',
                    ha='left',
                    va='bottom',
                    transform=axes[i].transAxes,
                    bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                )
                cnt += 1

        if show_FNO:
            for n, Tval in enumerate(Tvals):
                cdf = cdfrats[n]

                Pval = draw_contour_lines(dat_model, rescale=rescale, cdfrat=cdf, color='red', linestyle='solid',linewidth=1.8, alpha=0.5)
                # title += f'\n{cdf*100:2.0f}%: P={Pval:5.3f}'
                delta = Pval - Tval
                drat = (delta / Tval)*100
                if abs(drat) < 1.:
                    textp = f'{drat:.2f}'
                elif abs(drat) < 10:
                    textp = f'{drat:.1f}'
                else:
                    textp = f'{drat:.0f}'
                if drat < 0:
                    textp = f'({textp}%)'
                else:
                    textp = f'(+{textp}%)'

                axes[i].text(
                    xcnt[n]+0.32,
                    ycnt[n],
                    textp,
                    color="#fc5858",  # lighter red
                    fontsize=13,
                    fontweight='bold',
                    ha='left',
                    va='bottom',
                    transform=axes[i].transAxes,
                    bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                )

        axes[i].set_title(title)
        if ylim:
            axes[i].set_ylim(ylim)
        if xlim:
            axes[i].set_xlim(xlim)
        axes[i].set_xlabel('x [fm]')
        axes[i].set_ylabel('y [fm]', fontsize=16)
        axes[i].set_xlabel('x [fm]', fontsize=16)
        cbar = plt.colorbar(axes[i].images[0], ax=axes[i])
        cbar.ax.tick_params(labelsize=16)
        # plt.colorbar(axes[i].imagesg[0], ax=axes[i])


    # plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    # plt.subplots_adjust(hspace=.00)
    # plt.subplots_adjust(wspace=.1)
    # plt.subplots_adjust(top=0.88)
    # plt.subplots_adjust(bottom=0.16)  # Add space at the bottom for x-axis labels2
    # plt.subplots_adjust(left=0.05)  # Add space at the bottom for x-axis labels2

    # fig.suptitle(f'{tag} Event {event}. Contours: Truth: black. FNO: red.', fontsize=16)
    # if save:
        # fig.savefig(save)
    # plt.show()
                 

def plot_three_bins_contour (P, T, iparam=0, event=0, iT=(0, 25, 48),
   cdfrats=[0.05,.1,.2,.3,.4,.6], ylim=(15,45), xlim=(15,45),
                             topadjust=0.85, 
                             tag='', show_FNO=True, show_cdflines=True, 
                             show_cdfvals=True, fn=None, rescale=None, save=None, fig=None, axes=None,
                             taulabel_offset=0):

    if show_cdfvals and not show_FNO:
      raise ValueError('show_FNO requires show_cdfvals to be True')

    if iparam == 0:
        tag = f'Energy: {tag}'
    # elif iparam == 1:
        # tag = f'Temp: {tag}'
    elif iparam == 1:
        tag = f'Vy: {tag}'
    elif iparam == 2:
        tag = f'Vx: {tag}'


    if not fig and not axes:
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))

    assert len(iT) == 3

    for i, iT in enumerate(iT):
        title = ''

        dat_truth = T[event,iparam,:,:,iT]
        dat_model = P[event,iparam,:,:,iT]

        if fn is not None:
            dat_truth = fn(dat_truth, iT)
            dat_model = fn(dat_model, iT)

        axes[i].imshow(dat_truth, origin='lower', cmap='viridis', extent=(0, 60, 0, 60))
        plt.sca(axes[i])

        # Rescale tick marks
        if ylim == (15, 45):
            ticks = np.arange(15, 50, 5)  # Original tick positions
            new_labels = ticks / 2  # Rescale the labels
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(new_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(new_labels)
            plt.sca(axes[i])
            axes[i].tick_params(axis='both', which='major', length=8, width=2, labelsize=15)
            axes[i].tick_params(axis='both', which='minor', length=5, width=1)
        elif ylim == (0, 60):
            ticks = np.arange(0, 61, 10)
            new_labels = ticks / 2  # Rescale the labels
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(new_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(new_labels)
            plt.sca(axes[i])

        cnt = 0
        xcntl = 0.015
        xcnt = {0: xcntl, 1: xcntl, 2: xcntl, 3: xcntl, 4: xcntl, 5: xcntl}
        ycnt = {0: 0.93, 1: 0.87, 2: 0.81, 3: 0.13, 4: 0.07, 5: 0.01}

        tcolor = "#a9e4f3"  # lighter red

        axes[i].text( 
            0.655,
            0.94,
            r'$\mathbf{\tau}$'+f'={0.6+taulabel_offset*0.1+iT*0.1:.1f} '+r'fm/$\mathbf{c}$',
            color='white',
            fontsize=13,
            ha='left',
            va='bottom',
            transform=axes[i].transAxes,
            bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
            usetex=False,
            fontstyle='normal',
            fontweight='bold'
        )

        if show_cdflines:
            Tvals = []
            cnt = 0
            # draw truth lines first, and then the model lines
            for cdf in cdfrats:
                Tval=draw_contour_lines(dat_truth, rescale=rescale, cdfrat=cdf, color=tcolor, linestyle='solid', styles=[('solid', 'white', 1.5),('dotted','black',1.5)])
                Tvals.append(Tval)

                if show_cdfvals:
                    textw = f'{Tval:4.2f}'

                    axes[i].text(
                        xcnt[cnt],
                        ycnt[cnt],
                        f"{int(cdf*100):.0f}$^{{\\mathbf{{th}}}}$",
                        color='white',
                        fontsize=13,
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none'),
                        usetex=False,
                        fontstyle='normal',
                        fontweight='bold'
                    )
                    axes[i].text(
                        xcnt[cnt]+0.15,
                        ycnt[cnt],
                        textw,
                        color='white',
                        fontsize=13,
                        fontweight='bold',
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                    )
                    cnt += 1

            if show_FNO:
                for n, Tval in enumerate(Tvals):
                    cdf = cdfrats[n]

                    Pval = draw_contour_lines(dat_model, rescale=rescale, cdfrat=cdf, color='red', linestyle='solid',linewidth=1.8, alpha=0.5)
                    # title += f'\n{cdf*100:2.0f}%: P={Pval:5.3f}'
                    delta = Pval - Tval
                    drat = (delta / Tval)*100
                    if abs(drat) < 1.:
                        textp = f'{drat:.2f}'
                    elif abs(drat) < 10:
                        textp = f'{drat:.1f}'
                    else:
                        textp = f'{drat:.0f}'
                    if drat < 0:
                        textp = f'({textp}%)'
                    else:
                        textp = f'(+{textp}%)'

                    axes[i].text(
                        xcnt[n]+0.32,
                        ycnt[n],
                        textp,
                        color="#fc5858",  # lighter red
                        fontsize=13,
                        fontweight='bold',
                        ha='left',
                        va='bottom',
                        transform=axes[i].transAxes,
                        bbox=dict(facecolor='black', alpha=0.0, edgecolor='none')
                    )

            axes[i].set_title(title)
            if ylim:
                axes[i].set_ylim(ylim)
            if xlim:
                axes[i].set_xlim(xlim)
            axes[i].set_xlabel('x [fm]')
            axes[i].set_ylabel('y [fm]', fontsize=16)
            axes[i].set_xlabel('x [fm]', fontsize=16)
            cbar = plt.colorbar(axes[i].images[0], ax=axes[i])
            cbar.ax.tick_params(labelsize=16)
            # plt.colorbar(axes[i].imagesg[0], ax=axes[i])


    plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(hspace=.00)
    plt.subplots_adjust(wspace=.1)
    plt.subplots_adjust(top=0.88)
    plt.subplots_adjust(bottom=0.16)  # Add space at the bottom for x-axis labels2
    plt.subplots_adjust(left=0.05)  # Add space at the bottom for x-axis labels2

    fig.suptitle(f'{tag} Event {event}. Contours: Truth: black. FNO: red.', fontsize=16)
    if save:
        fig.savefig(save)
    plt.show()
