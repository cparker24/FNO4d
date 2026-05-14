"""Minimal MLX Trainer — mirrors the public interface of neuralop Trainer.

Key differences from PyTorch version:
- Uses mlx.nn.value_and_grad for gradient computation (no .backward())
- mx.eval() must be called to materialise lazy computations
- Checkpoint via mx.save_safetensors / mx.load
- No DDP or wandb (minimal scope)
- Mixed precision: model runs in float32; SpectralConv weight contraction
  can be optionally cast to float16 (handled in SpectralConvND if requested)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


class Trainer:
    def __init__(
        self,
        *,
        model,
        n_epochs: int,
        device: str = "gpu",
        eval_interval: int = 1,
        verbose: bool = True,
        mixed_precision: bool = False,
        wandb_log: bool = False,
        use_distributed: bool = False,
        log_fn=None,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.device = device
        self.eval_interval = eval_interval
        self.verbose = verbose
        self.mixed_precision = mixed_precision
        self.log_fn = log_fn if log_fn is not None else print
        # wandb and distributed are accepted but not implemented
        if wandb_log:
            self.log_fn("[Trainer] wandb_log=True is not supported in MLX trainer, ignoring.")
        if use_distributed:
            self.log_fn("[Trainer] use_distributed=True is not supported in MLX trainer, ignoring.")

    # ------------------------------------------------------------------
    # Public API (matches neuralop Trainer.train)
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader,
        test_loaders: Dict,
        optimizer,
        scheduler=None,
        regularizer=False,
        training_loss=None,
        eval_losses: Dict = None,
        save_dir: Union[str, Path] = "./ckpt",
        save_best: Optional[str] = None,
        **kwargs,
    ):
        from neuralop_mlx.training.losses import LpLoss

        if training_loss is None:
            training_loss = LpLoss(d=2, p=2)
        if eval_losses is None:
            eval_losses = {"l2": training_loss}

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_metric = float("inf")
        epoch_metrics = {}

        for epoch in range(self.n_epochs):
            t0 = time.perf_counter()
            train_err, avg_loss = self._train_one_epoch(
                epoch, train_loader, training_loss, optimizer, scheduler
            )
            epoch_time = time.perf_counter() - t0

            if self.verbose:
                self.log_training(epoch, epoch_time, avg_loss, train_err)

            epoch_metrics = {"train_err": train_err, "avg_loss": avg_loss}

            if epoch % self.eval_interval == 0:
                eval_metrics = self._evaluate_all(test_loaders, eval_losses)
                epoch_metrics.update(eval_metrics)

                if self.verbose:
                    self.log_eval(epoch, eval_metrics)

                if save_best is not None:
                    metric_val = eval_metrics.get(save_best, None)
                    if metric_val is None:
                        # try to find a matching key
                        for k, v in eval_metrics.items():
                            if save_best in k:
                                metric_val = v
                                break
                    if metric_val is not None and metric_val < best_metric:
                        best_metric = metric_val
                        self.checkpoint(save_dir, name="best_model")
                        if self.verbose:
                            self.log_fn(f"         Saved best model (metric={best_metric:.6f})")

        return epoch_metrics

    # ------------------------------------------------------------------
    # Logging (mirrors the PyTorch neuralop Trainer interface so that
    # patch_trainer_logging works identically for both frameworks)
    # ------------------------------------------------------------------

    def log_training(self, epoch, time, avg_loss, train_err,
                     avg_lasso_loss=None, lr=None):
        self.log_fn(
            f"Epoch {epoch:4d} | train_err={train_err:.6f} | "
            f"avg_loss={avg_loss:.6f} | time={time:.2f}s"
        )

    def log_eval(self, epoch, eval_metrics):
        metrics_str = " | ".join(f"{k}={v:.6f}" for k, v in eval_metrics.items())
        self.log_fn(f"         eval: {metrics_str}")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def checkpoint(self, save_dir: Union[str, Path], name: str = "model"):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        flat = dict(mlx.utils.tree_flatten(self.model.parameters()))
        mx.save_safetensors(str(save_dir / f"{name}.safetensors"), flat)

    def load_checkpoint(self, save_dir: Union[str, Path], name: str = "model"):
        save_dir = Path(save_dir)
        path = save_dir / f"{name}.safetensors"
        params = mx.load(str(path))
        # Rebuild the nested structure and load into model
        nested = mlx.utils.tree_unflatten(list(params.items()))
        self.model.update(nested)
        mx.eval(self.model.parameters())

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch, train_loader, training_loss, optimizer, scheduler):
        self.model.train(True)
        total_loss = 0.0
        n_batches = 0

        def forward(model, batch_x, batch_y):
            out = model(batch_x)
            return training_loss(out, batch_y)

        loss_and_grad = nn.value_and_grad(self.model, forward)

        for batch in train_loader:
            batch_x, batch_y = self._prepare_batch(batch)
            loss, grads = loss_and_grad(self.model, batch_x, batch_y)
            optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), optimizer.state)

            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0

        avg_loss = total_loss / n_batches
        train_err = avg_loss  # consistent with neuralop output
        return train_err, avg_loss

    def _evaluate_all(self, test_loaders: Dict, eval_losses: Dict) -> Dict:
        self.model.train(False)
        metrics = {}
        for loader_name, loader in test_loaders.items():
            for loss_name, loss_fn in eval_losses.items():
                key = f"{loader_name}_{loss_name}"
                val = self._evaluate_loader(loader, loss_fn)
                metrics[key] = val
        return metrics

    def _evaluate_loader(self, loader, loss_fn) -> float:
        total = 0.0
        n = 0
        for batch in loader:
            batch_x, batch_y = self._prepare_batch(batch)
            out = self.model(batch_x)
            loss = loss_fn(out, batch_y)
            mx.eval(loss)
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _prepare_batch(self, batch):
        """Extract (x, y) from a batch dict or tuple and convert to mx.array."""
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        else:
            x, y = batch[0], batch[1]

        if not isinstance(x, mx.array):
            import numpy as np
            x = mx.array(np.asarray(x, dtype=np.float32))
        if not isinstance(y, mx.array):
            import numpy as np
            y = mx.array(np.asarray(y, dtype=np.float32))

        return x, y
