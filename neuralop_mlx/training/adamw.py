"""Thin wrapper around mlx.optimizers.AdamW with a PyTorch-compatible interface."""

import mlx.optimizers as optim


def AdamW(params=None, lr: float = 1e-3, weight_decay: float = 1e-2, betas=(0.9, 0.999), eps: float = 1e-8):
    """Create an MLX AdamW optimizer.

    The `params` argument is accepted for API compatibility with the PyTorch version
    but is ignored (MLX optimizers receive parameters at update time).
    """
    return optim.AdamW(
        learning_rate=lr,
        betas=list(betas),
        eps=eps,
        weight_decay=weight_decay,
    )
