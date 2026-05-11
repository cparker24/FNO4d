# ipynb are annoying, but I cannot get Marimo to work in visual code, and I need to develop
# some quite plotting functions that having visual code's Github copilot support for will
# be very helpful.

# Purpose of this notebook is to put together a short library that will:
# 1. function to return a grid of subplots
# 2. plot imshow plots on them
# 3. plot the colorbars for the plots with the "topographical" lines too
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
    # print(f'threshold_index: {threshold_index}')
    
    return sorted_X[threshold_index]

# def draw_contour_lines(ax, dat, *,  rescale=None, val=None, cdfrat=None, linestyle='dashed',color='red',linewidth=1):
#     ''''
#     input: mask: tensor
#     output: None
#     Draw dotted lines around the values that are True in
#     the mask
#     '''
#     if val is None and cdfrat is None:
#         raise ValueError('Either val or cdfrat must be specified')

#     if val is None:
#         val = get_topcdf_value(dat, cdfrat)

#     print(f'val is {val}')

#     contours = find_contours(dat, level=val)
#     if rescale:
#         try:
#             contours[0] = contours[0]*rescale
#         except:
#             print('failed to find contrours for rescale')
    
#     for n, contour in enumerate(contours):
#         ax.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth)

#     return val

def draw_contour_lines(ax, dat, *,  rescale=None, val=None, cdfrat=None, linestyle='dashed',color='red',linewidth=1, styles=[],alpha=1):
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
        # print(f'cdf value: {val}')

    contours = find_contours(dat, level=val)
    if rescale:
        try:
            contours[0] = contours[0]*rescale
        except:
            print('failed to find contrours for rescale')
    
    if styles:
        for n, contour in enumerate(contours):
            for linestyle, color, linewidth in styles:
                ax.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)
    else:
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linestyle=linestyle, color=color, linewidth=linewidth, alpha=alpha)

    return val

def create_subplot_grid(nrows, ncols, figsize=(10, 8)):
    ''' ok, this one is probably silly'''
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    return fig, axes

def plot_imshow_with_colorbar(ax, data, cmap='viridis', colorbar=True, **imshow_kwargs):
    ''' plot an imshow on the given axis, with optional colorbar '''
    import matplotlib.pyplot as plt
    im = ax.imshow(data, cmap=cmap, **imshow_kwargs, origin='lower')
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im

def text_on_axis(ax, text, posx=0.1, posy=0.1, transform=None,
   fontsize=14,
   color='white',
   ha='left',
   va='bottom',
   fontweight='bold',
   bbox=dict(facecolor='black',alpha=0.0, edgecolor='none'
   )):
    ''' Add text to the given axis at the specified position '''
    if transform is None:
        transform = ax.transAxes
    ax.text(posx, posy, text, transform=transform, fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        fontweight=fontweight,
        bbox=bbox)

def plot2_compare(datA, datB, axA=None, axB=None, tau=None, iter=None, cdfs=(0.15, 0.3, 0.5),pre='',
                  xrange=(15.,45.), yrange=(15.,45.)):
    if axA is None or axB is None:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 6))

    imA = axA.imshow(datA, cmap='viridis', origin='lower')
    axA.set_xlim(xrange)
    axA.set_ylim(yrange)
    plt.colorbar(imA, ax=axA)
    # im = axB.imshow(datB, cmap='viridis', origin='lower')

    imB = axB.imshow(datB, cmap='viridis', origin='lower')
    axB.set_xlim(xrange)
    axB.set_ylim(yrange)
    plt.colorbar(imB, ax=axB)


    cdfA = []
    i_A = 0.0
    for cdf in cdfs:
        _A = draw_contour_lines(axA, datA, cdfrat=cdf, styles=(('solid','white',1.5),('dotted','black',1.5)))
        _ = draw_contour_lines(axB, datA, cdfrat=cdf, styles=(('solid','white',1.5),('dotted','black',1.5)))
        cdfA.append(_A)
        text_on_axis(axA, f'{int(cdf*100):}th: {cdfA[-1]:.2f}', posy=0.92 - i_A, posx=0.03, color='white')

        _B = draw_contour_lines(axB, datB, cdfrat=cdf, styles=(('dashed','red',1.5),))
        text_on_axis(axB, f'{int(cdf*100):}th: {_B:.2f}', posy=0.92 - i_A, posx=0.03, color='red')
        text_on_axis(axB, f'({'+' if _B>_A else ''}{(_B-_A)/_B*100:.2f}%)', posy=0.92 - i_A, posx=0.43, color='red')

        i_A += 0.05
        fig.suptitle(f'{pre} {"tau:"+str(tau) if tau is not None else ""} {"iter:"+str(iter) if iter is not None else ""}', fontsize=16 )

        fig.tight_layout=([0,0,1,0.95])


    # fig.suptitle("this is my title", fontsize=16)
    # fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#def plot_topographical_lines(ax, data, levels=10, **contour_kwargs):

