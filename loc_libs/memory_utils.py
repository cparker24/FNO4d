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
