import os
import sys
import atexit


class RunLog:
    """Print to stdout and, if path is given, mirror every line to a log file.

    With path=None this is a plain print wrapper — no file I/O, no stdout
    patching, normal terminal behaviour.

    Usage::

        log = RunLog(f'{odir}/training.log')   # log to file + terminal
        log = RunLog()                          # terminal only

        log('some message')
        log(f'value = {x}')
        log.close()   # optional — also called automatically at exit
    """

    def __init__(self, path=None):
        self._f = open(path, 'w') if path else None
        if self._f:
            atexit.register(self.close)

    def __call__(self, *args, **kwargs):
        print(*args, **kwargs)
        if self._f:
            # mirror the same formatted string to the file
            import io
            buf = io.StringIO()
            print(*args, **kwargs, file=buf)
            self._f.write(buf.getvalue())
            self._f.flush()

    def close(self):
        if self._f:
            self._f.close()
            self._f = None


class Logger:
    """Tee stdout to a file while keeping console output."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def enable_log_tee(log_path):
    """Redirect stdout so output goes to both the terminal and log_path."""
    sys.stdout = Logger(log_path)

    def _restore():
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal

    atexit.register(_restore)


def patch_trainer_logging(trainer, log_fn, history=None):
    """Route trainer verbose output (train + eval metrics) through log_fn
    and optionally collect structured metrics into a TrainingHistory.

    Works with both the MLX Trainer (also sets trainer.log_fn) and the
    PyTorch neuralop Trainer (patches log_training and log_eval methods).

    Parameters
    ----------
    trainer   : Trainer instance
    log_fn    : callable — e.g. a RunLog instance
    history   : TrainingHistory, optional — receives record_train/record_eval calls
    """
    import types

    # MLX Trainer: also update log_fn for messages outside log_training/log_eval
    if hasattr(trainer, 'log_fn'):
        trainer.log_fn = log_fn

    def _log_training(self, epoch, time, avg_loss, train_err,
                      avg_lasso_loss=None, lr=None):
        msg = f"[{epoch}] time={time:.2f}, avg_loss={avg_loss:.4f}, train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        log_fn(msg)
        if history is not None:
            history.record_train(epoch, time=time, avg_loss=avg_loss,
                                 train_err=train_err, avg_lasso_loss=avg_lasso_loss)

    def _log_eval(self, epoch, eval_metrics):
        parts = [
            f"{k}={float(v):.4f}" for k, v in eval_metrics.items()
            if isinstance(v, float) or hasattr(v, 'item')
        ]
        log_fn("Eval: " + ", ".join(parts))
        if history is not None:
            history.record_eval(epoch, eval_metrics)

    trainer.log_training = types.MethodType(_log_training, trainer)
    trainer.log_eval = types.MethodType(_log_eval, trainer)


def save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, save_dir):
    """Save a complete training state so training can be resumed later.

    PyTorch : writes ``last_model.pt`` containing model, optimizer, and
              scheduler state dicts plus the last epoch number.
    MLX     : writes ``last_model.safetensors`` (model weights),
              ``last_optimizer.safetensors`` (optimizer state), and
              ``last_state.json`` (epoch metadata).

    To resume, call ``load_last_checkpoint`` at the start of the next run
    and adjust ``n_epochs`` to the remaining number of epochs.
    """
    import json
    from pathlib import Path

    save_dir = Path(save_dir)
    last_epoch = n_epochs - 1
    is_mlx = hasattr(trainer, 'log_fn')

    if is_mlx:
        import mlx.core as mx
        import mlx.utils as mlx_utils
        flat_model = dict(mlx_utils.tree_flatten(trainer.model.parameters()))
        mx.save_safetensors(str(save_dir / 'last_model.safetensors'), flat_model)
        flat_opt = dict(mlx_utils.tree_flatten(optimizer.state))
        if flat_opt:
            mx.save_safetensors(str(save_dir / 'last_optimizer.safetensors'), flat_opt)
        with open(save_dir / 'last_state.json', 'w') as f:
            json.dump({'epoch': last_epoch, 'n_epochs': n_epochs}, f, indent=2)
    else:
        import torch
        state = {
            'epoch': last_epoch,
            'n_epochs': n_epochs,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(state, save_dir / 'last_model.pt')


def load_last_checkpoint(trainer, optimizer, scheduler, save_dir):
    """Load a training state saved by ``save_last_checkpoint``.

    Returns the next epoch to train (last saved epoch + 1) so the caller
    can adjust ``n_epochs`` accordingly::

        resume_epoch = load_last_checkpoint(trainer, optimizer, scheduler, model_dir)
        # set n_epochs = total_epochs - resume_epoch before calling trainer.train()
    """
    import json
    from pathlib import Path

    save_dir = Path(save_dir)
    is_mlx = hasattr(trainer, 'log_fn')

    if is_mlx:
        import mlx.core as mx
        import mlx.utils as mlx_utils
        params = mx.load(str(save_dir / 'last_model.safetensors'))
        nested = mlx_utils.tree_unflatten(list(params.items()))
        trainer.model.update(nested)
        mx.eval(trainer.model.parameters())
        opt_path = save_dir / 'last_optimizer.safetensors'
        if opt_path.exists():
            opt_params = mx.load(str(opt_path))
            opt_nested = mlx_utils.tree_unflatten(list(opt_params.items()))
            optimizer.state.update(opt_nested)
        with open(save_dir / 'last_state.json') as f:
            meta = json.load(f)
        return meta['epoch'] + 1
    else:
        import torch
        state = torch.load(save_dir / 'last_model.pt')
        trainer.model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in state:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        return state['epoch'] + 1


def patch_trainer_checkpoint(trainer, trainer_cls, checkpoint_freq, odir,
                              optimizer=None):
    """Monkey-patch trainer.on_epoch_start to save checkpoints every checkpoint_freq epochs.

    For MLX trainers pass ``optimizer`` to include optimizer state in each
    checkpoint, making them fully resumable (matching PyTorch behaviour).
    """
    import json
    from pathlib import Path

    is_mlx = hasattr(trainer, 'log_fn')

    def _on_epoch_start(self, epoch):
        self.epoch = epoch
        if self.epoch % checkpoint_freq == 0 and self.epoch > 0:
            ckpt_dir = Path(f'{odir}/checkpoint_{self.epoch}')
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint(str(ckpt_dir))
            if is_mlx and optimizer is not None:
                import mlx.core as mx
                import mlx.utils as mlx_utils
                flat_opt = dict(mlx_utils.tree_flatten(optimizer.state))
                if flat_opt:
                    mx.save_safetensors(
                        str(ckpt_dir / 'optimizer.safetensors'), flat_opt)
                with open(ckpt_dir / 'state.json', 'w') as f:
                    json.dump({'epoch': self.epoch}, f, indent=2)
            print('saving checkpoint to ', ckpt_dir)

    trainer.on_epoch_start = _on_epoch_start.__get__(trainer, trainer_cls)
