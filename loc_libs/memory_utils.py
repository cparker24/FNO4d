import os
import psutil


def process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # bytes


class MemoryPrinter:
    """Print current process RSS to stdout and optionally to a log file."""

    def __init__(self, log=None, logfile=None):
        self.log = log
        if log:
            if not logfile:
                raise ValueError("If log is provided, logfile must also be provided.")
            self.logfile = logfile
            if not os.path.isfile(os.path.dirname(logfile)):
                with open(self.logfile, 'w') as f:
                    f.write("")

    def __call__(self, label=""):
        mem = process_memory() / (1024 ** 3)
        output = f"[Memory] {label}: {mem:.3f} GB"
        if self.log is not None:
            try:
                self.log.write(output + "\n")
            except Exception:
                with open(self.logfile, 'a') as f:
                    f.write(output + "\n")
        print(output)


def reset_peak_memory(*, cuda=False, mlx=False):
    """Reset peak-memory counters before a training run."""
    if cuda:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    if mlx:
        import mlx.core as mx
        if hasattr(mx, 'reset_peak_memory'):
            mx.reset_peak_memory()


def log_peak_memory(log_fn, *, cuda=False, mlx=False):
    """Log peak CPU RSS and, if requested, peak GPU memory after training."""
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    log_fn(f'peak CPU memory (RSS): {rss:.2f} GB')
    if cuda:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            log_fn(f'peak CUDA memory:      {peak:.2f} GB')
        else:
            log_fn('peak CUDA memory:      n/a (no CUDA device)')
    if mlx:
        import mlx.core as mx
        if hasattr(mx, 'get_peak_memory'):
            peak = mx.get_peak_memory() / 1024 ** 3
            log_fn(f'peak Metal memory:     {peak:.2f} GB')


def calculate_model_memory(model, count_fn=None):
    """Return model memory footprint in MB assuming float32 parameters.

    count_fn : callable(model) -> int, optional.
        Defaults to neuralop.utils.count_model_params (PyTorch).
        Pass neuralop_mlx.utils.count_model_params for MLX models.
    """
    if count_fn is None:
        from neuralop.utils import count_model_params
        count_fn = count_model_params
    total_params = count_fn(model)
    print(f'Total parameters: {total_params}')
    return (total_params * 4) / (1024 ** 2)  # float32 bytes → MB
