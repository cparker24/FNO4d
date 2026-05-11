"""Unit tests for individual MLX FNO layers.

Run with: python -m pytest tests/test_layers_mlx.py -v
"""

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# GridEmbeddingND
# ---------------------------------------------------------------------------

class TestGridEmbeddingND:
    def test_output_shape_4d(self):
        from neuralop_mlx.layers.embeddings import GridEmbeddingND
        emb = GridEmbeddingND(in_channels=4, dim=4)
        x = mx.zeros([2, 4, 8, 8, 4, 4])
        out = emb(x)
        mx.eval(out)
        assert out.shape == (2, 8, 8, 8, 4, 4), f"Expected (2,8,8,8,4,4), got {out.shape}"

    def test_output_shape_2d(self):
        from neuralop_mlx.layers.embeddings import GridEmbeddingND
        emb = GridEmbeddingND(in_channels=3, dim=2)
        x = mx.zeros([1, 3, 16, 16])
        out = emb(x)
        mx.eval(out)
        assert out.shape == (1, 5, 16, 16)

    def test_grid_range(self):
        """Grid values must lie in [0, 1)."""
        from neuralop_mlx.layers.embeddings import GridEmbeddingND
        emb = GridEmbeddingND(in_channels=1, dim=2, grid_boundaries=[[0, 1], [0, 1]])
        x = mx.zeros([1, 1, 8, 8])
        out = emb(x)
        mx.eval(out)
        grids = np.array(out[0, 1:])  # grid channels
        assert grids.min() >= 0.0
        assert grids.max() < 1.0

    def test_grid_cached(self):
        """Grid is reused on second call with same resolution."""
        from neuralop_mlx.layers.embeddings import GridEmbeddingND
        emb = GridEmbeddingND(in_channels=2, dim=2)
        x = mx.zeros([1, 2, 8, 8])
        _ = emb(x)
        grid_ref = emb._grid
        _ = emb(x)
        assert emb._grid is grid_ref


# ---------------------------------------------------------------------------
# ChannelMLP
# ---------------------------------------------------------------------------

class TestChannelMLP:
    def test_output_shape_4d(self):
        from neuralop_mlx.layers.channel_mlp import ChannelMLP
        mlp = ChannelMLP(in_channels=8, out_channels=4, hidden_channels=16, n_layers=2)
        x = mx.array(np.random.randn(2, 8, 6, 6, 4, 4).astype(np.float32))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (2, 4, 6, 6, 4, 4)

    def test_output_shape_2d(self):
        from neuralop_mlx.layers.channel_mlp import ChannelMLP
        mlp = ChannelMLP(in_channels=4, out_channels=8)
        x = mx.array(np.random.randn(3, 4, 16, 16).astype(np.float32))
        out = mlp(x)
        mx.eval(out)
        assert out.shape == (3, 8, 16, 16)

    def test_no_nan(self):
        from neuralop_mlx.layers.channel_mlp import ChannelMLP
        mlp = ChannelMLP(in_channels=4, out_channels=4)
        x = mx.array(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = mlp(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()


# ---------------------------------------------------------------------------
# SpectralConvND
# ---------------------------------------------------------------------------

class TestSpectralConvND:
    def test_output_shape_4d(self):
        from neuralop_mlx.layers.spectral_convolution import SpectralConvND
        conv = SpectralConvND(in_channels=4, out_channels=6, n_modes=[4, 4, 3, 3])
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 6, 8, 8, 4, 4)

    def test_output_shape_2d(self):
        from neuralop_mlx.layers.spectral_convolution import SpectralConvND
        conv = SpectralConvND(in_channels=4, out_channels=4, n_modes=[8, 8])
        x = mx.array(np.random.randn(2, 4, 16, 16).astype(np.float32))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (2, 4, 16, 16)

    def test_no_nan_4d(self):
        from neuralop_mlx.layers.spectral_convolution import SpectralConvND
        conv = SpectralConvND(in_channels=4, out_channels=4, n_modes=[4, 4, 3, 3])
        x = mx.array(np.random.randn(1, 4, 8, 8, 4, 6).astype(np.float32))
        out = conv(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item()

    def test_modes_smaller_than_input(self):
        """n_modes much smaller than input size should still work."""
        from neuralop_mlx.layers.spectral_convolution import SpectralConvND
        conv = SpectralConvND(in_channels=2, out_channels=2, n_modes=[4, 4, 2, 2])
        x = mx.array(np.random.randn(1, 2, 32, 32, 16, 16).astype(np.float32))
        out = conv(x)
        mx.eval(out)
        assert out.shape == (1, 2, 32, 32, 16, 16)

    def test_weight_split_real(self):
        from neuralop_mlx.layers.spectral_convolution import SpectralConvND
        conv = SpectralConvND(in_channels=2, out_channels=2, n_modes=[4, 4])
        assert conv.weight_r.dtype == mx.float32
        assert conv.weight_i.dtype == mx.float32
        # Property returns complex view
        assert conv.weight.dtype == mx.complex64


# ---------------------------------------------------------------------------
# FNOBlocks
# ---------------------------------------------------------------------------

class TestFNOBlocks:
    def test_output_shape_4d(self):
        from neuralop_mlx.layers.fno_block import FNOBlocks
        blocks = FNOBlocks(in_channels=8, out_channels=8, n_modes=[4, 4, 3, 3], n_layers=3)
        x = mx.array(np.random.randn(2, 8, 8, 8, 4, 4).astype(np.float32))
        for i in range(3):
            x = blocks(x, index=i)
        mx.eval(x)
        assert x.shape == (2, 8, 8, 8, 4, 4)

    def test_no_nan(self):
        from neuralop_mlx.layers.fno_block import FNOBlocks
        blocks = FNOBlocks(in_channels=4, out_channels=4, n_modes=[4, 4, 3, 3], n_layers=2)
        x = mx.array(np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32))
        for i in range(2):
            x = blocks(x, index=i)
        mx.eval(x)
        assert not mx.any(mx.isnan(x)).item()


# ---------------------------------------------------------------------------
# Skip connections
# ---------------------------------------------------------------------------

class TestSkipConnections:
    def test_linear_skip(self):
        from neuralop_mlx.layers.skip_connections import build_skip
        skip = build_skip("linear", 4, 8)
        x = mx.array(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = skip(x)
        mx.eval(out)
        assert out.shape == (2, 8, 8, 8)

    def test_soft_gating(self):
        from neuralop_mlx.layers.skip_connections import build_skip
        skip = build_skip("soft-gating", 4, 4)
        x = mx.array(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = skip(x)
        mx.eval(out)
        assert out.shape == (2, 4, 8, 8)

    def test_identity(self):
        from neuralop_mlx.layers.skip_connections import build_skip
        skip = build_skip("identity", 4, 4)
        x = mx.array(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = skip(x)
        assert out is x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
