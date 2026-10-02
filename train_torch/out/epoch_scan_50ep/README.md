# Epoch-count scan (2026-10-02)

Full dataset (200 events, 160 train / 40 verify), `batch_size: 1`,
`test_batch_sizes: [5]`, `hidden_channels: 16`, `n_modes: [32, 32, 16, 16]`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, single RTX 3090.

Run at both `tauslice: [null, 90]` and `tauslice: [null, 100]`, 50 epochs
each, to answer two questions: how far GPU memory climbs with more epochs,
and how many epochs are actually useful before overfitting.

## Memory: unchanged from 10 epochs to 50 epochs

| tauslice upper | peak CUDA @ 10 epochs | peak CUDA @ 50 epochs |
|---|---|---|
| 90  | 19.48 GB | 19.48 GB (identical) |
| 100 | 21.15 GB | 21.15 GB (identical) |

`torch.cuda.max_memory_allocated()` lands on the same high-water mark
regardless of epoch count once `expandable_segments:True` is set -- the
mid-run nvidia-smi fluctuation observed while these were running was normal
allocator caching behavior between the batch=1 training step and the
batch=5 eval step, not a genuine upward trend across epochs.

## Both runs overfit well before epoch 50

| tauslice | best eval l2 | at epoch | eval l2 @ epoch 48 |
|---|---|---|---|
| 90  | 0.3272 | 30 | 0.3702 |
| 100 | 0.4751 | 30 | 0.5251 |

See `overfitting_check.svg` -- held-out eval loss bottoms out right around
epoch 30 for both tauslice values, then rises while training loss keeps
falling (tau=100's training loss even ticks back up from epoch 36 on).
Training past ~epoch 30 at these settings buys nothing and measurably hurts
generalization.

**Conclusion: 30 epochs is the practical target** -- it sits right at the
observed overfitting onset, so training beyond it does not help with this
data/architecture.

Raw per-epoch logs: `tau90_50ep.log`, `tau100_50ep.log`.
