"""Load weights from a trained PyTorch FNO into an MLX FNO.

Typical usage
-------------
From a live PyTorch model object::

    from neuralop_mlx.weight_transfer import load_from_torch
    mlx_model = load_from_torch(pt_model, mlx_model)

From a saved checkpoint (.pt / .pth)::

    mlx_model = load_from_torch("best_model.pt", mlx_model)

The function returns the same mlx_model with weights updated in place and
calls mx.eval() so the arrays are immediately materialised.

Architecture constraints
------------------------
Both models must have been constructed with the same hyperparameters:
  n_modes, in_channels, out_channels, hidden_channels, n_layers,
  lifting_channel_ratio, projection_channel_ratio, use_channel_mlp.

Weight layout differences handled automatically
-----------------------------------------------
| PyTorch                                  | MLX                                  |
|------------------------------------------|--------------------------------------|
| convs[i].weight.tensor  (complex64)      | convs[i].weight_r + weight_i (float) |
| convs[i].bias  shape (C,1,..,1)          | convs[i].bias  shape (1,C,1,..,1)    |
| fno_skips[i].conv.weight  (out,in,1)     | fno_skips[i].linear.weight (out,in)  |
| channel_mlp[i].fcs[j].weight (out,in,1) | channel_mlps[i].fcs[j].weight (out,in)|
| channel_mlp_skips[i].weight (1,C,1,..)  | mlp_skips[i].alpha  (C,)             |
| lifting/projection fcs[j].weight (o,i,1)| same but (o,i) — Conv1d→Linear       |
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_from_torch(source, mlx_model, map_location="cpu"):
    """Transfer weights from a PyTorch FNO to an MLX FNO.

    Parameters
    ----------
    source : str | Path | torch.nn.Module
        Either a path to a checkpoint saved with ``torch.save(model.state_dict(), path)``
        or a live PyTorch model object.
    mlx_model : neuralop_mlx.models.FNO
        The target MLX model (already constructed with matching architecture).
    map_location : str
        Passed to ``torch.load`` when loading from a file. Default "cpu".

    Returns
    -------
    mlx_model : neuralop_mlx.models.FNO
        Same object, weights updated in place.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch is required for weight transfer.") from e

    # --- resolve state dict ---
    if isinstance(source, (str, __import__("pathlib").Path)):
        raw = torch.load(str(source), map_location=map_location, weights_only=False)
        # support both bare state_dict and {'model_state_dict': ...} checkpoint dicts
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif isinstance(raw, dict) and not _looks_like_state_dict(raw):
            # try common checkpoint wrapper keys
            for key in ("state_dict", "model", "net"):
                if key in raw:
                    state_dict = raw[key]
                    break
            else:
                state_dict = raw
        else:
            state_dict = raw
        pt_model = None
    else:
        # live model object
        state_dict = source.state_dict()
        pt_model = source

    # state_dict() may include non-tensor entries (e.g. _metadata); keep only tensors
    import torch
    state_dict = {k: v.cpu() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}

    n_layers = mlx_model.n_layers

    # --- spectral conv layers ---
    for i in range(n_layers):
        prefix = f"fno_blocks.convs.{i}"
        mlx_conv = mlx_model.fno_blocks.convs[i]

        w = _get_complex_weight(state_dict, prefix)
        mlx_conv.weight_r = mx.array(w.real.astype(np.float32))
        mlx_conv.weight_i = mx.array(w.imag.astype(np.float32))

        bias_key = f"{prefix}.bias"
        if bias_key in state_dict and mlx_conv.bias is not None:
            pt_bias = state_dict[bias_key].numpy().astype(np.float32)
            # PT shape: (C, 1, ..., 1) → MLX shape: (1, C, 1, ..., 1)
            mlx_conv.bias = mx.array(pt_bias.reshape(mlx_conv.bias.shape))

    # --- FNO skip connections (linear) ---
    for i in range(n_layers):
        pt_key_base = f"fno_blocks.fno_skips.{i}"
        mlx_skip = mlx_model.fno_blocks.fno_skips[i]

        # PyTorch: Flattened1dConv stores weight as (out, in, 1) Conv1d kernel
        w = _find_weight(state_dict, [
            f"{pt_key_base}.conv.weight",
            f"{pt_key_base}.weight",
        ])
        if w is not None:
            w_np = w.numpy().astype(np.float32)
            if w_np.ndim == 3:          # (out, in, 1) → (out, in)
                w_np = w_np[:, :, 0]
            mlx_skip.linear.weight = mx.array(w_np)

    # --- channel MLP layers ---
    if mlx_model.fno_blocks.channel_mlps is not None:
        for i in range(n_layers):
            # PyTorch uses 'channel_mlp' (no s); MLX uses 'channel_mlps'
            pt_prefix = f"fno_blocks.channel_mlp.{i}"
            mlx_mlp = mlx_model.fno_blocks.channel_mlps[i]
            _transfer_channel_mlp_weights(state_dict, pt_prefix, mlx_mlp)

    # --- MLP skip connections (soft-gating) ---
    if mlx_model.fno_blocks.mlp_skips is not None:
        for i in range(n_layers):
            mlx_skip = mlx_model.fno_blocks.mlp_skips[i]
            w = _find_weight(state_dict, [
                f"fno_blocks.channel_mlp_skips.{i}.weight",
                f"fno_blocks.mlp_skips.{i}.weight",
                f"fno_blocks.mlp_skips.{i}.alpha",
            ])
            if w is not None:
                # PT shape: (1, C, 1, ..., 1) → flatten to (C,)
                mlx_skip.alpha = mx.array(w.numpy().astype(np.float32).reshape(-1))

    # --- lifting MLP ---
    _transfer_channel_mlp_weights(state_dict, "lifting", mlx_model.lifting)

    # --- projection MLP ---
    _transfer_channel_mlp_weights(state_dict, "projection", mlx_model.projection)

    mx.eval(mlx_model.parameters())
    return mlx_model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _looks_like_state_dict(d: dict) -> bool:
    """Heuristic: a state dict has tensor values, not nested dicts."""
    try:
        import torch
        return all(isinstance(v, torch.Tensor) for v in d.values())
    except ImportError:
        return False


def _find_weight(state_dict: dict, candidates: list):
    """Return the first key that exists in state_dict, or None."""
    for key in candidates:
        if key in state_dict:
            return state_dict[key]
    return None


def _get_complex_weight(state_dict: dict, prefix: str) -> np.ndarray:
    """Extract complex weight from a PyTorch SpectralConv layer.

    Handles three storage formats:
    - DenseTensor (.weight.tensor) — neuralop default when factorization=None
    - Plain complex tensor (.weight)
    - Tucker factorized tensor (.weight.core + .weight.factors.factor_N) — TFNO
    """
    import torch

    # --- Dense / DenseTensor ---
    candidates = [
        f"{prefix}.weight.tensor",   # DenseTensor (neuralop default)
        f"{prefix}.weight",          # plain complex tensor
    ]
    for key in candidates:
        if key in state_dict:
            t = state_dict[key]
            if t.is_complex():
                return t.numpy().astype(np.complex64)
            # Some checkpoints store real/imag stacked on last dim
            if t.shape[-1] == 2:
                arr = t.numpy()
                return arr[..., 0].astype(np.float32) + 1j * arr[..., 1].astype(np.float32)
            return t.numpy().astype(np.complex64)

    # --- Tucker factorized (TFNO) ---
    # Keys: {prefix}.weight.core  and  {prefix}.weight.factors.factor_0, factor_1, ...
    core_key = f"{prefix}.weight.core"
    if core_key in state_dict:
        try:
            import tensorly as tl
        except ImportError as exc:
            raise ImportError(
                "tensorly is required to load Tucker-factorized TFNO weights. "
                "Install it with: pip install tensorly"
            ) from exc

        # Convert to numpy before reconstruction. Use a temporary numpy backend
        # to avoid interfering with any PyTorch-backed tensorly calls elsewhere
        # (tl.set_backend is global; restore it after reconstruction).
        core = state_dict[core_key].numpy()

        factors = []
        i = 0
        while True:
            fkey = f"{prefix}.weight.factors.factor_{i}"
            if fkey not in state_dict:
                break
            factors.append(state_dict[fkey].numpy())
            i += 1

        if not factors:
            raise KeyError(
                f"Found Tucker core at '{core_key}' but no factor matrices. "
                f"Available keys: {[k for k in state_dict if k.startswith(prefix)]}"
            )

        _prev_backend = tl.get_backend()
        try:
            tl.set_backend("numpy")
            dense = tl.tucker_to_tensor((core, factors)).astype(np.complex64)
        finally:
            tl.set_backend(_prev_backend)

        return dense

    raise KeyError(
        f"Cannot find spectral conv weight for prefix '{prefix}'. "
        f"Tried dense keys {candidates} and Tucker key '{core_key}'. "
        f"Available keys with that prefix: {[k for k in state_dict if k.startswith(prefix)]}"
    )


def _transfer_channel_mlp_weights(state_dict: dict, pt_prefix: str, mlx_mlp):
    """Copy ChannelMLP weights: Conv1d (out, in, 1) → Linear (out, in)."""
    for j, fc in enumerate(mlx_mlp.fcs):
        w = _find_weight(state_dict, [
            f"{pt_prefix}.fcs.{j}.weight",
            f"{pt_prefix}.fc{j}.weight",
        ])
        if w is not None:
            w_np = w.numpy().astype(np.float32)
            if w_np.ndim == 3:          # Conv1d kernel (out, in, 1) → (out, in)
                w_np = w_np[:, :, 0]
            fc.weight = mx.array(w_np)

        b = _find_weight(state_dict, [
            f"{pt_prefix}.fcs.{j}.bias",
            f"{pt_prefix}.fc{j}.bias",
        ])
        if b is not None:
            fc.bias = mx.array(b.numpy().astype(np.float32))
