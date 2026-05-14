import os
import sys
import atexit


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


def patch_trainer_checkpoint(trainer, trainer_cls, checkpoint_freq, odir):
    """Monkey-patch trainer.on_epoch_start to save checkpoints every checkpoint_freq epochs."""

    def _on_epoch_start(self, epoch):
        self.epoch = epoch
        if self.epoch % checkpoint_freq == 0 and self.epoch > 0:
            save_dir = f'{odir}/checkpoint_{self.epoch}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.checkpoint(save_dir)
            print('saving checkpoint to ', save_dir)

    trainer.on_epoch_start = _on_epoch_start.__get__(trainer, trainer_cls)
