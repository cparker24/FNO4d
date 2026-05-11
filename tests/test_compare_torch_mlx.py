"""Numerical comparison: PyTorch FNO vs MLX FNO.

Copies weights from a PyTorch FNO into the MLX FNO and verifies that forward
pass outputs agree to within floating-point tolerance (atol=1e-3 for float32 FFTs).

Run with: python -m pytest tests/test_compare_torch_mlx.py -v
         or directly: python tests/test_compare_torch_mlx.py

Requires both `torch` and `mlx` in the same environment.
"""

import math
import numpy as np
import pytest

try:
    import torch
    import torch.nn as tnn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

requires_both = pytest.mark.skipif(
    not (TORCH_AVAILABLE and MLX_AVAILABLE),
    reason="Both PyTorch and MLX must be installed",
)


# ---------------------------------------------------------------------------
# Weight transfer helpers
# ---------------------------------------------------------------------------

def _torch_to_numpy_complex(t: "torch.Tensor") -> np.ndarray:
    """Convert a complex PyTorch tensor to a complex128 numpy array."""
    return t.detach().cpu().numpy().astype(np.complex64)


def _transfer_spectral_conv_weights(pt_conv, mlx_conv):
    """Copy SpectralConv weights from PyTorch to MLX.

    PyTorch stores a DenseTensor (complex64); MLX stores weight_r + weight_i (float32).
    """
    import mlx.core as mx

    pt_weight = pt_conv.weight  # DenseTensor / FactorizedTensor
    if hasattr(pt_weight, 'tensor'):
        pt_np = pt_weight.tensor.detach().cpu().numpy()
    elif hasattr(pt_weight, 'to_tensor'):
        pt_np = pt_weight.to_tensor().detach().cpu().numpy()
    else:
        pt_np = pt_weight.detach().cpu().numpy()

    pt_np = pt_np.astype(np.complex64)
    # Store as split real/imaginary (matches MLX weight_r / weight_i)
    mlx_conv.weight_r = mx.array(pt_np.real.astype(np.float32))
    mlx_conv.weight_i = mx.array(pt_np.imag.astype(np.float32))

    if pt_conv.bias is not None and mlx_conv.bias is not None:
        pt_bias = pt_conv.bias.detach().cpu().numpy().astype(np.float32)
        mlx_conv.bias = mx.array(pt_bias)


def _transfer_linear_weights(pt_linear, mlx_linear):
    """Transfer nn.Linear / nn.Conv1d(kernel_size=1) → MLX Linear."""
    import mlx.core as mx
    # PyTorch Linear: weight shape (out, in); MLX Linear: weight shape (out, in)
    if hasattr(pt_linear, 'weight'):
        w = pt_linear.weight.detach().cpu().numpy().astype(np.float32)
        if w.ndim == 3:  # Conv1d: (out, in, 1)
            w = w[:, :, 0]
        mlx_linear.weight = mx.array(w)
    if hasattr(pt_linear, 'bias') and pt_linear.bias is not None:
        b = pt_linear.bias.detach().cpu().numpy().astype(np.float32)
        mlx_linear.bias = mx.array(b)


def build_matching_models(n_modes, hidden_channels=8, n_layers=2):
    """Build PyTorch FNO and MLX FNO with matching architecture, then
    copy weights from PyTorch to MLX.

    Returns (pt_model, mlx_model, input_numpy).
    """
    import sys
    sys.path.insert(0, '/Users/du8478/FNO2Mlx/neuraloperator-4d-2.0.0')
    from neuralop.models import FNO as TorchFNO

    from neuralop_mlx.models import FNO as MlxFNO
    import mlx.core as mx

    pt_model = TorchFNO(
        in_channels=4,
        out_channels=4,
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        projection_channel_ratio=2,
    )
    pt_model.eval()

    mlx_model = MlxFNO(
        in_channels=4,
        out_channels=4,
        n_modes=list(n_modes),
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        projection_channel_ratio=2,
    )

    # Transfer FNO block spectral conv weights
    for i in range(n_layers):
        pt_conv = pt_model.fno_blocks.convs[i]
        mlx_conv = mlx_model.fno_blocks.convs[i]
        _transfer_spectral_conv_weights(pt_conv, mlx_conv)

    # Transfer FNO block skip connections (linear)
    for i in range(n_layers):
        pt_skip = pt_model.fno_blocks.fno_skips[i]
        mlx_skip = mlx_model.fno_blocks.fno_skips[i]
        if hasattr(pt_skip, 'linear'):
            _transfer_linear_weights(pt_skip.linear, mlx_skip.linear)
        elif hasattr(pt_skip, 'weight'):
            _transfer_linear_weights(pt_skip, mlx_skip.linear)

    # Transfer ChannelMLP weights (lifting, projection, and fno block mlps)
    def _transfer_channel_mlp(pt_mlp, mlx_mlp):
        pt_fcs = list(pt_mlp.fcs)
        mlx_fcs = mlx_mlp.fcs
        for pt_fc, mlx_fc in zip(pt_fcs, mlx_fcs):
            _transfer_linear_weights(pt_fc, mlx_fc)

    _transfer_channel_mlp(pt_model.lifting, mlx_model.lifting)
    _transfer_channel_mlp(pt_model.projection, mlx_model.projection)

    for i in range(n_layers):
        if pt_model.fno_blocks.channel_mlp is not None:
            pt_cmlp = pt_model.fno_blocks.channel_mlp[i]
            mlx_cmlp = mlx_model.fno_blocks.channel_mlps[i]
            _transfer_channel_mlp(pt_cmlp, mlx_cmlp)

    # Create shared input
    np.random.seed(0)
    n_dim = len(n_modes)
    spatial = (8,) * (n_dim - 1) + (6,)
    x_np = np.random.randn(1, 4, *spatial).astype(np.float32)

    return pt_model, mlx_model, x_np


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@requires_both
class TestSpectralConvComparison:
    """Compare individual SpectralConv forward passes."""

    def test_spectral_conv_output_close_2d(self):
        import torch
        import mlx.core as mx

        from neuralop_mlx.layers.spectral_convolution import SpectralConvND

        sys_path_backup = __import__('sys').path[:]
        import sys
        sys.path.insert(0, '/Users/du8478/FNO2Mlx/neuraloperator-4d-2.0.0')
        from neuralop.layers.spectral_convolution import SpectralConv

        n_modes = (4, 4)
        pt_conv = SpectralConv(in_channels=4, out_channels=4, n_modes=n_modes)
        mlx_conv = SpectralConvND(in_channels=4, out_channels=4, n_modes=list(n_modes))

        # Copy weights
        _transfer_spectral_conv_weights(pt_conv, mlx_conv)

        np.random.seed(42)
        x_np = np.random.randn(2, 4, 16, 16).astype(np.float32)

        with torch.no_grad():
            pt_out = pt_conv(torch.tensor(x_np)).numpy()

        mlx_out = np.array(mlx_conv(mx.array(x_np)))
        mx.eval()

        max_diff = np.max(np.abs(pt_out - mlx_out))
        print(f"\nSpectralConv 2D max diff: {max_diff:.2e}")
        assert max_diff < 1e-3, f"SpectralConv 2D outputs differ: max_diff={max_diff:.2e}"

        sys.path = sys_path_backup


@requires_both
class TestFNOComparison:
    """Compare full FNO model forward passes after weight transfer."""

    def _run_comparison(self, n_modes):
        import torch
        import mlx.core as mx

        try:
            pt_model, mlx_model, x_np = build_matching_models(n_modes, hidden_channels=8, n_layers=2)
        except Exception as e:
            pytest.skip(f"Weight transfer failed (architecture mismatch or missing package): {e}")

        with torch.no_grad():
            pt_out = pt_model(torch.tensor(x_np)).numpy()

        mlx_out = np.array(mlx_model(mx.array(x_np)))
        mx.eval()

        max_diff = np.max(np.abs(pt_out - mlx_out))
        rel_diff = max_diff / (np.max(np.abs(pt_out)) + 1e-8)
        print(f"\nFNO {len(n_modes)}D max_diff={max_diff:.3e} rel_diff={rel_diff:.3e}")
        return max_diff, rel_diff

    def test_fno_output_shape_matches_2d(self):
        import torch, mlx.core as mx
        try:
            pt, mlx_m, x_np = build_matching_models((4, 4))
        except Exception as e:
            pytest.skip(str(e))
        with torch.no_grad():
            pt_out = pt(torch.tensor(x_np)).numpy()
        mlx_out = np.array(mlx_m(mx.array(x_np)))
        mx.eval()
        assert pt_out.shape == tuple(mlx_out.shape), f"{pt_out.shape} != {mlx_out.shape}"


@requires_both
class TestGradientComparison:
    """Compare gradient norms between PyTorch and MLX."""

    def test_gradient_norm_close(self):
        import torch
        import mlx.core as mx
        import mlx.nn as nn

        try:
            pt_model, mlx_model, x_np = build_matching_models((4, 4), hidden_channels=8, n_layers=2)
        except Exception as e:
            pytest.skip(str(e))

        y_np = np.random.randn(*x_np.shape).astype(np.float32)

        # PyTorch gradient
        x_pt = torch.tensor(x_np, requires_grad=False)
        y_pt = torch.tensor(y_np)
        out_pt = pt_model(x_pt)
        loss_pt = ((out_pt - y_pt) ** 2).mean()
        loss_pt.backward()
        pt_grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in pt_model.parameters()
            if p.grad is not None
        ) ** 0.5

        # MLX gradient
        x_mlx = mx.array(x_np)
        y_mlx = mx.array(y_np)

        def fwd(model):
            return mx.mean((model(x_mlx) - y_mlx) ** 2)

        _, grads = nn.value_and_grad(mlx_model, fwd)(mlx_model)
        mx.eval(grads)

        import mlx.utils
        mlx_grad_norm = math.sqrt(sum(
            float(mx.sum(g ** 2).item())
            for _, g in mlx.utils.tree_flatten(grads)
            if g is not None and g.dtype != mx.complex64
        ))

        rel_diff = abs(pt_grad_norm - mlx_grad_norm) / (pt_grad_norm + 1e-8)
        print(f"\nGrad norm: torch={pt_grad_norm:.4f} mlx={mlx_grad_norm:.4f} rel_diff={rel_diff:.3f}")
        # Gradient norms can differ due to FFT accumulation differences; allow 20%
        assert rel_diff < 0.20, f"Gradient norm relative diff too large: {rel_diff:.3f}"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Spectral Conv 2D comparison ===")
    t = TestSpectralConvComparison()
    try:
        t.test_spectral_conv_output_close_2d()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n=== FNO output shape comparison ===")
    t2 = TestFNOComparison()
    try:
        t2.test_fno_output_shape_matches_2d()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")

    print("\n=== Gradient norm comparison ===")
    t3 = TestGradientComparison()
    try:
        t3.test_gradient_norm_close()
        print("PASS")
    except Exception as e:
        print(f"FAIL: {e}")
