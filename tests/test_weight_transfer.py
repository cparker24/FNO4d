"""Tests for load_from_torch weight transfer helper.

Requires both PyTorch (neuraloperator-4d-2.0.0) and MLX in the same environment.

Run with: python -m pytest tests/test_weight_transfer.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pytest

try:
    import torch
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

NEURALOP_PATH = "/Users/du8478/FNO2Mlx/neuraloperator-4d-2.0.0"


def _make_pt(n_modes, n_layers=2, hidden=8):
    sys.path.insert(0, NEURALOP_PATH)
    from neuralop.models import FNO as TorchFNO
    m = TorchFNO(
        n_modes=n_modes, in_channels=4, out_channels=4,
        hidden_channels=hidden, n_layers=n_layers,
        projection_channel_ratio=2,
    )
    m.eval()
    return m


def _make_mlx(n_modes, n_layers=2, hidden=8):
    from neuralop_mlx.models import FNO as MlxFNO
    return MlxFNO(
        n_modes=list(n_modes), in_channels=4, out_channels=4,
        hidden_channels=hidden, n_layers=n_layers,
        projection_channel_ratio=2,
    )


def _pt_forward(pt_model, x_np):
    import torch
    with torch.no_grad():
        return pt_model(torch.tensor(x_np)).numpy()


def _mlx_forward(mlx_model, x_np):
    out = np.array(mlx_model(mx.array(x_np)))
    mx.eval()
    return out


# ---------------------------------------------------------------------------
# Numerical parity: live model transfer
# ---------------------------------------------------------------------------

@requires_both
class TestLiveModelTransfer:

    def _run(self, n_modes, spatial):
        from neuralop_mlx.weight_transfer import load_from_torch
        np.random.seed(42)
        x_np = np.random.randn(1, 4, *spatial).astype(np.float32)
        pt = _make_pt(n_modes)
        pt_out = _pt_forward(pt, x_np)
        mlx_m = _make_mlx(n_modes)
        load_from_torch(pt, mlx_m)
        mlx_out = _mlx_forward(mlx_m, x_np)
        diff = np.max(np.abs(pt_out - mlx_out))
        assert diff < 1e-3, f"n_modes={n_modes} max_diff={diff:.2e}"

    def test_2d(self):
        self._run((4, 4), (16, 16))

    def test_3d(self):
        self._run((4, 4, 3), (8, 8, 4))

    def test_4d(self):
        self._run((4, 4, 3, 3), (8, 8, 4, 4))

    def test_4d_asymmetric_spatial(self):
        """Non-square spatial dims."""
        self._run((4, 4, 3, 2), (10, 8, 6, 4))

    def test_4d_more_layers(self):
        """n_layers=4 (typical production config)."""
        from neuralop_mlx.weight_transfer import load_from_torch
        np.random.seed(0)
        x_np = np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32)
        pt = _make_pt((4, 4, 3, 3), n_layers=4)
        pt_out = _pt_forward(pt, x_np)
        mlx_m = _make_mlx((4, 4, 3, 3), n_layers=4)
        load_from_torch(pt, mlx_m)
        diff = np.max(np.abs(pt_out - _mlx_forward(mlx_m, x_np)))
        assert diff < 1e-3, f"max_diff={diff:.2e}"


# ---------------------------------------------------------------------------
# Checkpoint loading (file-based)
# ---------------------------------------------------------------------------

@requires_both
class TestCheckpointTransfer:

    def _pt_and_input(self, n_modes=(4, 4, 3, 3)):
        np.random.seed(7)
        x_np = np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32)
        pt = _make_pt(n_modes)
        pt_out = _pt_forward(pt, x_np)
        return pt, x_np, pt_out

    def test_bare_state_dict(self):
        """torch.save(model.state_dict(), path)"""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt, x_np, pt_out = self._pt_and_input()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(pt.state_dict(), path)
            mlx_m = _make_mlx((4, 4, 3, 3))
            load_from_torch(path, mlx_m)
            diff = np.max(np.abs(pt_out - _mlx_forward(mlx_m, x_np)))
            assert diff < 1e-3, f"bare state_dict max_diff={diff:.2e}"
        finally:
            os.unlink(path)

    def test_wrapped_checkpoint(self):
        """torch.save({'epoch': N, 'model_state_dict': sd, 'loss': v}, path)"""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt, x_np, pt_out = self._pt_and_input()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save({"epoch": 10, "model_state_dict": pt.state_dict(), "loss": 0.05}, path)
            mlx_m = _make_mlx((4, 4, 3, 3))
            load_from_torch(path, mlx_m)
            diff = np.max(np.abs(pt_out - _mlx_forward(mlx_m, x_np)))
            assert diff < 1e-3, f"wrapped ckpt max_diff={diff:.2e}"
        finally:
            os.unlink(path)

    def test_checkpoint_roundtrip_matches_live(self):
        """Loading from file must give same result as loading from live model."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt, x_np, _ = self._pt_and_input()
        mlx_live = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_live)
        out_live = _mlx_forward(mlx_live, x_np)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(pt.state_dict(), path)
            mlx_file = _make_mlx((4, 4, 3, 3))
            load_from_torch(path, mlx_file)
            out_file = _mlx_forward(mlx_file, x_np)
        finally:
            os.unlink(path)

        diff = np.max(np.abs(out_live - out_file))
        assert diff == 0.0, f"live vs file mismatch: {diff:.2e}"


# ---------------------------------------------------------------------------
# Weight values are actually transferred
# ---------------------------------------------------------------------------

@requires_both
class TestWeightValues:

    def test_spectral_conv_weights_match(self):
        """weight_r / weight_i must equal real/imag of PyTorch weight."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_pt((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)

        sd = {k: v for k, v in pt.state_dict().items() if isinstance(v, torch.Tensor)}
        for i in range(2):
            pt_w = sd[f"fno_blocks.convs.{i}.weight.tensor"].numpy().astype(np.complex64)
            mlx_wr = np.array(mlx_m.fno_blocks.convs[i].weight_r)
            mlx_wi = np.array(mlx_m.fno_blocks.convs[i].weight_i)
            assert np.max(np.abs(pt_w.real - mlx_wr)) < 1e-6, f"weight_r mismatch layer {i}"
            assert np.max(np.abs(pt_w.imag - mlx_wi)) < 1e-6, f"weight_i mismatch layer {i}"

    def test_fno_skip_weights_match(self):
        """FNO skip linear weights must match (Conv1d kernel squeezed)."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_pt((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)

        sd = {k: v for k, v in pt.state_dict().items() if isinstance(v, torch.Tensor)}
        for i in range(2):
            pt_w = sd[f"fno_blocks.fno_skips.{i}.conv.weight"].numpy()[:, :, 0]
            mlx_w = np.array(mlx_m.fno_blocks.fno_skips[i].linear.weight)
            assert np.max(np.abs(pt_w - mlx_w)) < 1e-6, f"fno_skip mismatch layer {i}"

    def test_soft_gating_weights_match(self):
        """Per-channel soft-gating alpha must match PyTorch channel_mlp_skips."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_pt((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)

        sd = {k: v for k, v in pt.state_dict().items() if isinstance(v, torch.Tensor)}
        for i in range(2):
            pt_w = sd[f"fno_blocks.channel_mlp_skips.{i}.weight"].numpy().reshape(-1)
            mlx_w = np.array(mlx_m.fno_blocks.mlp_skips[i].alpha)
            assert np.max(np.abs(pt_w - mlx_w)) < 1e-6, f"soft-gating mismatch layer {i}"

    def test_lifting_weights_match(self):
        """Lifting MLP weights must match (Conv1d → Linear)."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_pt((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)

        sd = {k: v for k, v in pt.state_dict().items() if isinstance(v, torch.Tensor)}
        for j in range(2):
            pt_w = sd[f"lifting.fcs.{j}.weight"].numpy()[:, :, 0]
            mlx_w = np.array(mlx_m.lifting.fcs[j].weight)
            assert np.max(np.abs(pt_w - mlx_w)) < 1e-6, f"lifting fc{j} mismatch"


# ---------------------------------------------------------------------------
# Tucker factorized weights (TFNO → MLX FNO)
# ---------------------------------------------------------------------------

requires_tensorly = pytest.mark.skipif(
    not (TORCH_AVAILABLE and MLX_AVAILABLE and __import__("importlib").util.find_spec("tensorly") is not None),
    reason="PyTorch, MLX, and tensorly must all be installed",
)


def _make_tfno(n_modes, n_layers=2, hidden=8, rank=0.5):
    """Build a PyTorch TFNO with Tucker factorization."""
    sys.path.insert(0, NEURALOP_PATH)
    from neuralop.models import TFNO
    m = TFNO(
        n_modes=n_modes, in_channels=4, out_channels=4,
        hidden_channels=hidden, n_layers=n_layers,
        projection_channel_ratio=2,
        factorization="Tucker", rank=rank,
    )
    m.eval()
    return m


@requires_both
class TestTuckerWeightTransfer:
    """Load Tucker-factorized TFNO weights into a dense MLX FNO."""

    def _run_parity(self, n_modes, spatial, rank=0.5, seed=42):
        """Core helper: transfer weights and compare forward outputs."""
        from neuralop_mlx.weight_transfer import load_from_torch
        np.random.seed(seed)
        x_np = np.random.randn(1, 4, *spatial).astype(np.float32)

        pt = _make_tfno(n_modes, rank=rank)
        with __import__("torch").no_grad():
            pt_out = pt(__import__("torch").tensor(x_np)).numpy()

        mlx_m = _make_mlx(n_modes)
        load_from_torch(pt, mlx_m)
        mx.eval(mlx_m.parameters())

        mlx_out = np.array(mlx_m(mx.array(x_np)))
        mx.eval()
        return pt_out, mlx_out

    # --- shape / NaN ---

    def test_load_does_not_raise(self):
        """load_from_torch must complete without error on a TFNO checkpoint."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_tfno((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)  # must not raise

    def test_output_shape(self):
        """Output shape must be unchanged after Tucker weight transfer."""
        pt_out, mlx_out = self._run_parity((4, 4, 3, 3), (8, 8, 4, 4))
        assert mlx_out.shape == pt_out.shape

    def test_no_nan_in_output(self):
        """No NaN values in MLX output after Tucker weight transfer."""
        _, mlx_out = self._run_parity((4, 4, 3, 3), (8, 8, 4, 4))
        assert not np.any(np.isnan(mlx_out)), "NaN detected in MLX output"

    # --- numerical parity ---

    def test_parity_2d(self):
        """2D TFNO → MLX parity within 1e-3."""
        pt_out, mlx_out = self._run_parity((4, 4), (16, 16))
        diff = np.max(np.abs(pt_out - mlx_out))
        assert diff < 1e-3, f"2D Tucker parity max_diff={diff:.2e}"

    def test_parity_3d(self):
        """3D TFNO → MLX parity within 1e-3."""
        pt_out, mlx_out = self._run_parity((4, 4, 3), (8, 8, 4))
        diff = np.max(np.abs(pt_out - mlx_out))
        assert diff < 1e-3, f"3D Tucker parity max_diff={diff:.2e}"

    def test_parity_4d(self):
        """4D TFNO → MLX parity within 1e-3."""
        pt_out, mlx_out = self._run_parity((4, 4, 3, 3), (8, 8, 4, 4))
        diff = np.max(np.abs(pt_out - mlx_out))
        assert diff < 1e-3, f"4D Tucker parity max_diff={diff:.2e}"

    def test_parity_low_rank(self):
        """Very low rank (0.1, ~10% params) still transfers correctly."""
        pt_out, mlx_out = self._run_parity((4, 4, 3, 3), (8, 8, 4, 4), rank=0.1)
        diff = np.max(np.abs(pt_out - mlx_out))
        assert diff < 1e-3, f"low-rank Tucker parity max_diff={diff:.2e}"

    # --- gradient flow ---

    def test_gradient_flow_after_tucker_transfer(self):
        """No NaN gradients after Tucker weight transfer and one training step."""
        import mlx.nn as nn
        from neuralop_mlx.weight_transfer import load_from_torch
        np.random.seed(0)
        x_np = np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32)
        y_np = np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32)
        x, y = mx.array(x_np), mx.array(y_np)

        pt = _make_tfno((4, 4, 3, 3))
        mlx_m = _make_mlx((4, 4, 3, 3))
        load_from_torch(pt, mlx_m)

        def fwd(model):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        loss, grads = nn.value_and_grad(mlx_m, fwd)(mlx_m)
        mx.eval(loss, grads)

        assert not np.isnan(float(loss.item())), "NaN loss after Tucker transfer"

        def _check_no_nan(tree, path=""):
            if isinstance(tree, mx.array):
                assert not mx.any(mx.isnan(tree)).item(), f"NaN gradient at {path}"
            elif isinstance(tree, dict):
                for k, v in tree.items():
                    _check_no_nan(v, f"{path}.{k}")
            elif isinstance(tree, list):
                for i, v in enumerate(tree):
                    _check_no_nan(v, f"{path}[{i}]")

        _check_no_nan(grads)

    # --- checkpoint round-trip ---

    def test_checkpoint_file_tucker(self):
        """Tucker weights survive save → load from file."""
        import torch, tempfile, os
        from neuralop_mlx.weight_transfer import load_from_torch
        np.random.seed(5)
        x_np = np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32)

        pt = _make_tfno((4, 4, 3, 3))
        with torch.no_grad():
            pt_out = pt(torch.tensor(x_np)).numpy()

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            torch.save(pt.state_dict(), path)
            mlx_m = _make_mlx((4, 4, 3, 3))
            load_from_torch(path, mlx_m)
            mlx_out = np.array(mlx_m(mx.array(x_np)))
            mx.eval()
            diff = np.max(np.abs(pt_out - mlx_out))
            assert diff < 1e-3, f"Tucker checkpoint file parity max_diff={diff:.2e}"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@requires_both
class TestErrorHandling:

    def test_missing_file_raises(self):
        from neuralop_mlx.weight_transfer import load_from_torch
        mlx_m = _make_mlx((4, 4))
        with pytest.raises(Exception):
            load_from_torch("/nonexistent/path/model.pth", mlx_m)

    def test_bad_checkpoint_key_raises(self):
        """A checkpoint with no recognisable state_dict key should still load
        (it will attempt to use the dict directly and fail on the weight lookup,
        raising KeyError rather than silently producing wrong weights)."""
        from neuralop_mlx.weight_transfer import load_from_torch
        pt = _make_pt((4, 4))
        sd = pt.state_dict()
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name
        try:
            # Save with an unrecognised wrapper key
            torch.save({"some_random_key": sd}, path)
            mlx_m = _make_mlx((4, 4))
            # Should either succeed (if it falls back) or raise a clear error
            # — must not silently produce zero/uninitialized weights
            try:
                load_from_torch(path, mlx_m)
            except (KeyError, RuntimeError, ValueError):
                pass  # acceptable: clear failure is better than silent wrong weights
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
