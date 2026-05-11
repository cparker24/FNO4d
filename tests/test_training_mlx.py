"""Training loop tests for MLX FNO.

Run with: python -m pytest tests/test_training_mlx.py -v
"""

import numpy as np
import pytest
import mlx.core as mx
from torch.utils.data import DataLoader, Dataset


class SyntheticFluidDataset(Dataset):
    """Synthetic dataset matching FluidDataset in test_train.py.

    Data shape: (n, in_channels, nx, ny, neta, ntau)
    x: initial field repeated ntau-1 times along last axis
    y: evolution field (tau=1..end)
    """

    def __init__(self, n_samples=8, in_channels=4, nx=8, ny=8, neta=4, ntau=5):
        self.n = n_samples
        self.c = in_channels
        self.T = ntau
        np.random.seed(42)
        # shape: (n, in_channels, nx, ny, neta, ntau) — 4 spatial/temporal dims
        self.data = np.random.randn(n_samples, in_channels, nx, ny, neta, ntau).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Matches FluidDataset.__getitem__ from test_train.py
        x_init = self.data[idx, :, :, :, :, :1]          # (c, nx, ny, neta, 1)
        x = np.repeat(x_init, self.T - 1, axis=-1)        # (c, nx, ny, neta, T-1)
        y = self.data[idx, :, :, :, :, 1:self.T]          # (c, nx, ny, neta, T-1)
        return {"x": x, "y": y}


@pytest.fixture
def setup():
    from neuralop_mlx.models import FNO
    from neuralop_mlx.training.losses import H1Loss, LpLoss
    from neuralop_mlx.training.adamw import AdamW

    model = FNO(in_channels=4, out_channels=4, n_modes=[4, 4, 3, 3], hidden_channels=8, n_layers=2)
    optimizer = AdamW(lr=1e-3, weight_decay=1e-4)
    h1loss = H1Loss(d=4)
    l2loss = LpLoss(d=4, p=2)

    ds = SyntheticFluidDataset(n_samples=8)
    train_loader = DataLoader(ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(ds, batch_size=2, shuffle=False)
    test_loaders = {100: test_loader}

    return model, optimizer, h1loss, l2loss, train_loader, test_loaders


class TestTrainer:
    def test_loss_decreases(self, setup):
        """Loss should drop after a few epochs on the same synthetic data."""
        from neuralop_mlx.training.trainer import Trainer

        model, optimizer, h1loss, l2loss, train_loader, test_loaders = setup

        # Evaluate initial loss
        total_init = 0.0
        n = 0
        for batch in train_loader:
            x = mx.array(np.array(batch["x"]))
            y = mx.array(np.array(batch["y"]))
            loss_val = h1loss(model(x), y)
            mx.eval(loss_val)
            total_init += loss_val.item()
            n += 1
        avg_init = total_init / n

        trainer = Trainer(model=model, n_epochs=5, eval_interval=5, verbose=False)
        trainer.train(
            train_loader=train_loader,
            test_loaders=test_loaders,
            optimizer=optimizer,
            training_loss=h1loss,
            eval_losses={"h1": h1loss, "l2": l2loss},
            save_dir="/tmp/fno_mlx_trainer_test",
            save_best=None,
        )

        total_final = 0.0
        n = 0
        for batch in train_loader:
            x = mx.array(np.array(batch["x"]))
            y = mx.array(np.array(batch["y"]))
            loss_val = h1loss(model(x), y)
            mx.eval(loss_val)
            total_final += loss_val.item()
            n += 1
        avg_final = total_final / n

        assert avg_final < avg_init, f"Loss did not decrease: {avg_init:.4f} → {avg_final:.4f}"

    def test_checkpoint_roundtrip(self, setup, tmp_path):
        """Saved and loaded model should produce identical outputs."""
        from neuralop_mlx.training.trainer import Trainer
        import mlx.utils

        model, optimizer, h1loss, l2loss, train_loader, test_loaders = setup

        x = mx.array(np.random.randn(1, 4, 8, 8, 4, 4).astype(np.float32))
        out_before = model(x)
        mx.eval(out_before)

        trainer = Trainer(model=model, n_epochs=1, eval_interval=1, verbose=False)
        trainer.checkpoint(tmp_path, name="ckpt")
        trainer.load_checkpoint(tmp_path, name="ckpt")

        out_after = model(x)
        mx.eval(out_after)

        diff = mx.max(mx.abs(out_before - out_after)).item()
        assert diff < 1e-6, f"Checkpoint roundtrip mismatch: max_diff={diff}"


class TestLosses:
    def test_lp_loss_zero_on_identical(self):
        from neuralop_mlx.training.losses import LpLoss
        loss = LpLoss(d=4, p=2)
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        val = loss(x, x)
        mx.eval(val)
        assert abs(val.item()) < 1e-5

    def test_h1_loss_zero_on_identical(self):
        from neuralop_mlx.training.losses import H1Loss
        loss = H1Loss(d=4)
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        val = loss(x, x)
        mx.eval(val)
        assert abs(val.item()) < 1e-5

    def test_lp_loss_nonzero_on_different(self):
        from neuralop_mlx.training.losses import LpLoss
        loss = LpLoss(d=4, p=2)
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        y = mx.zeros_like(x)
        val = loss(x, y)
        mx.eval(val)
        assert val.item() > 0

    def test_h1_loss_positive(self):
        from neuralop_mlx.training.losses import H1Loss
        loss = H1Loss(d=4)
        x = mx.array(np.random.randn(2, 4, 8, 8, 4, 4).astype(np.float32))
        y = mx.zeros_like(x)
        val = loss(x, y)
        mx.eval(val)
        assert val.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
