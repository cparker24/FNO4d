import re

import numpy as np
import matplotlib.pyplot as plt
def parse_training_log(filename):
    """
    Parse the training log file and extract time, avg_loss, train_err, 100_h1, and 100_l2 values.
    
    Returns:
        dict: Dictionary containing lists of floats for each metric    """
    epochs = []
    time = []
    avg_loss = []
    train_err = []
    h1_100 = []
    l2_100 = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    # print('lines:', lines, len(lines))
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
          # Look for training metrics lines (pattern: [epoch] time=..., avg_loss=..., train_err=...)
        train_match = re.match(r'\[(\d+)\] time=([\d.]+), avg_loss=([\d.]+), train_err=([\d.]+)', line)
        if train_match:
            epochs.append(int(train_match.group(1)))  # Extract epoch number
            time.append(float(train_match.group(2)))
            avg_loss.append(float(train_match.group(3)))
            train_err.append(float(train_match.group(4)))
              # Look for the corresponding eval line on the next line
            if i + 1 < len(lines):
                eval_line = lines[i + 1].strip()
                eval_match = re.match(r'Eval: 100_h1=([\d.]+), 100_l2=([\d.]+)', eval_line)
                if eval_match:
                    h1_100.append(float(eval_match.group(1)))
                    l2_100.append(float(eval_match.group(2)))
        
        i += 1
    
    return {
        'epochs': epochs,
        'time': time,
        'avg_loss': avg_loss,
        'train_err': train_err,
        '100_h1': h1_100,
        '100_l2': l2_100
    }

def plot_training_log(filename='training.log', title='', first_entry=None, times=False, axes=None, fig=None, save=True, plot_opts={}):

    plot_kwargs = {k: v for k, v in plot_opts.items() if v is not None}
    if 'marker' not in plot_kwargs:
        plot_kwargs['marker'] = 'o'

    # Prepare epochs
    data = parse_training_log(filename)
    epochs = data['epochs']

    # Compute cumulative minutes
    cum_min = np.cumsum(data['time']) / 60

    # Create 4-panel vertical plot
    if axes is None:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 7))

    if first_entry is not None:
        start_index = first_entry
    else:
        start_index = 0

    # Top panel: average loss
    # Filter out None values from plot_opt
    axes[0].plot(epochs[start_index:], data['avg_loss'][start_index:], **plot_kwargs)
    axes[0].set_ylabel('Avg Loss')

    # Second panel: training error
    axes[1].plot(epochs[start_index:], data['train_err'][start_index:], **plot_kwargs)
    axes[1].set_ylabel('Train Err')

    if times:
        # Annotate each point with cumulative time as (hr:min)
        entry: int=0
        for x, y, m in zip(epochs, data['train_err'], cum_min):
            entry += 1
            if entry % times:
                continue
            hr = int(m // 60)
            mn = int(m % 60)
            sc = int((m - int(m)) * 60)
            axes[1].text(x, y, f"{hr}:{mn:02d}:{sc:02d}", va='bottom', ha='center', fontsize=8)

    # Third panel: 100_h1
    axes[2].plot(epochs[start_index:], data['100_h1'][start_index:], **plot_kwargs)
    axes[2].set_ylabel('100_h1')

    # Fourth panel: 100_l2
    axes[3].plot(epochs[start_index:], data['100_l2'][start_index:], **plot_kwargs)
    axes[3].set_ylabel('100_l2')
    axes[3].set_xlabel('Epochs')

    if title:
        plt.suptitle(title)

    # Remove spacing between panels
    plt.tight_layout(h_pad=0)

    # Show plot
    if save:
        plt.savefig(filename.replace('.log', '.svg'), bbox_inches='tight')
    # plt.show()


    return data, fig, axes