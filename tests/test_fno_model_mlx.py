"""Tests for the full FNO model in MLX.

Run with: python -m pytest tests/test_fno_model_mlx.py -v
"""

import numpy as np
import pytest
import mlx.core as mx

from neuralop_mlx.models import FNO
from neuralop_mlx.utils import count_model_params


class TestFNO:
    def _make_model(self, n_modes, hidden_channels=8, n_layers=2):
        return FNO(
            in_channels=4,
            out_channels=4,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            projection_channel_ratio=2,
        )

    def test_4d_forward_shape(self):
        model = self._make_model([4, 4, 3, 3])
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 4, 8, 8, 4, 4)

    def test_3d_forward_shape(self):
        model = self._make_model([8, 8, 6])
        x = mx.array(np.random.randn(2, 4, 16, 16, 8).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (2, 4, 16, 16, 8)

    def test_2d_forward_shape(self):
        model = self._make_model([8, 8])
        x = mx.array(np.random.randn(3, 4, 16, 16).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (3, 4, 16, 16)

    def test_no_nan_4d(self):
        model = self._make_model([4, 4, 3, 3])
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 6).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert not mx.any(mx.isnan(out)).item(), "NaN in output"

    def test_param_count_positive(self):
        model = self._make_model([4, 4, 3, 3])
        n = count_model_params(model)
        assert n > 0

    def test_no_positional_embedding(self):
        model = FNO(
            in_channels=4, out_channels=4,
            n_modes=[4, 4, 3, 3],
            hidden_channels=8, n_layers=2,
            positional_embedding=None,
        )
        x = mx.array(np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 4, 8, 8, 4, 4)

    def test_batch_size_1(self):
        """Edge case: batch_size=1 (matches test_train.py default)."""
        model = self._make_model([32, 32, 16, 16], hidden_channels=16, n_layers=4)
        x = mx.array(np.random.randn(1, 4, 40, 40, 20, 20).astype(np.float32))
        out = model(x)
        mx.eval(out)
        assert out.shape == (1, 4, 40, 40, 20, 20)
        assert not mx.any(mx.isnan(out)).item()

    def test_gradient_flows(self):
        """A single gradient step should not produce NaN gradients."""
        import mlx.nn as nn
        from neuralop_mlx.training.losses import H1Loss

        model = self._make_model([4, 4, 3, 3])
        loss_fn = H1Loss(d=4)
        x = mx.array(np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32))
        y = mx.array(np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32))

        def forward(m):
            return loss_fn(m(x), y)

        loss, grads = nn.value_and_grad(model, forward)(model)
        mx.eval(loss, grads)
        assert not mx.isnan(loss).item(), "NaN loss"

        import mlx.utils
        for name, g in mlx.utils.tree_flatten(grads):
            if g is not None:
                assert not mx.any(mx.isnan(g)).item(), f"NaN grad in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
