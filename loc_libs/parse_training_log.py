import re
import json

import numpy as np
import matplotlib.pyplot as plt


class TrainingHistory:
    """Accumulate per-epoch train/eval metrics in memory, with save/load/plot.

    Populated automatically by patch_trainer_logging(..., history=history).

    Structure::

        history.train  →  {'epoch': [...], 'avg_loss': [...], 'train_err': [...]}
        history.eval   →  {'epoch': [...], '100_h1': [...], '100_l2': [...]}

    Usage::

        history = TrainingHistory()
        patch_trainer_logging(trainer, log, history=history)
        trainer.train(...)
        history.save(f'{odir}/metrics.json')
        history.plot(save_path=f'{odir}/training_curves.svg')
    """

    def __init__(self):
        self.train = {}
        self.eval = {}

    def record_train(self, epoch, *, time, avg_loss, train_err,
                     avg_lasso_loss=None, **kwargs):
        self.train.setdefault('epoch', []).append(epoch)
        self.train.setdefault('time',  []).append(time)
        self.train.setdefault('avg_loss',  []).append(avg_loss)
        self.train.setdefault('train_err', []).append(train_err)
        if avg_lasso_loss is not None:
            self.train.setdefault('avg_lasso_loss', []).append(avg_lasso_loss)

    def record_eval(self, epoch, eval_metrics):
        self.eval.setdefault('epoch', []).append(epoch)
        for k, v in eval_metrics.items():
            if isinstance(v, float) or hasattr(v, 'item'):
                self.eval.setdefault(k, []).append(float(v))

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'train': self.train, 'eval': self.eval}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        h = cls()
        h.train = d.get('train', {})
        h.eval  = d.get('eval', {})
        return h

    # Human-readable labels for known metric keys
    _YLABEL_MAP = {
        'avg_loss':       'avg loss',
        'train_err':      'train error',
        'avg_lasso_loss': 'lasso loss',
    }

    @staticmethod
    def _fmt_ylabel(key):
        """Format a metric key for axis labels and legend entries.

        Examples: 'avg_loss' → 'avg loss',
                  '100_h1'   → 'H¹  (res 100)',
                  '100_l2'   → 'L²  (res 100)'
        """
        if key in TrainingHistory._YLABEL_MAP:
            return TrainingHistory._YLABEL_MAP[key]
        m = re.match(r'^(\d+)_(\w+)$', key)
        if m:
            res, metric = m.groups()
            pretty = {'h1': 'H¹', 'l2': 'L²'}.get(metric, metric)
            return f'{pretty}  (res {res})'
        return key.replace('_', ' ')

    def plot(self, save_path=None, title=''):
        """Plot train and eval metrics, optionally saving to save_path.

        Returns (fig, axes).
        """
        train_keys = [k for k in ('avg_loss', 'train_err', 'avg_lasso_loss')
                      if k in self.train]
        eval_keys  = [k for k in self.eval if k != 'epoch']
        plot_keys  = train_keys + eval_keys

        if not plot_keys:
            return None, None

        n = len(plot_keys)
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(8, 2 * n))
        if n == 1:
            axes = [axes]

        train_epochs = self.train.get('epoch', [])
        eval_epochs  = self.eval.get('epoch', [])

        for ax, key in zip(axes, plot_keys):
            ylabel = self._fmt_ylabel(key)
            if key in self.train:
                ax.plot(train_epochs, self.train[key],
                        marker='o', ms=3, lw=1, color='tab:blue',
                        label=f'{ylabel}  [train]')
            if key in self.eval:
                ax.plot(eval_epochs, self.eval[key],
                        marker='s', ms=3, lw=1, color='tab:orange',
                        label=f'{ylabel}  [eval]')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8, loc='upper right',
                      framealpha=0.8, edgecolor='0.8')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Epoch')
        if title:
            plt.suptitle(title, y=1.01)
        plt.tight_layout(h_pad=0.5)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig, axes
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