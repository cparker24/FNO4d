import mlx.utils


def count_model_params(model):
    """Count total trainable parameters.

    SpectralConv weights are stored as weight_r + weight_i (both float32),
    so together they count as 2x a real weight of the same shape — matching
    the PyTorch convention of counting complex params as 2 reals.
    """
    return sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
