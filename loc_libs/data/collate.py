"""Batched ``collate_fn`` — normalise + window + to-tensor, once per batch.

Replaces the per-event ``transform`` + first-tau-repeat that the old
``_FluidDatasetH5Base.__getitem__`` ran on every single event
(``fluid_datasets.py``).  Because ``Normalizer.transform`` is elementwise per
``(eta, tau)``, the batched result is bit-for-bit identical to the per-event
path (within fp tolerance).  See ``DATA_PIPELINE_PLAN.md`` §8.

The collate is a small *callable class* (not a closure) so it pickles cleanly —
required when the DataLoader uses the ``spawn`` / ``forkserver`` start method
(the macOS default), which pickles ``collate_fn`` to each worker.
"""

import numpy as np
import torch


class _Collate:
    """Picklable batched collate: stack -> normalise -> window -> to-tensor."""

    def __init__(self, normalizer, time_steps, backend):
        self.normalizer = normalizer
        self.T = int(time_steps)
        self.backend = backend
        self.apply_norm = (normalizer is not None and
                           getattr(normalizer, 'name', 'none') != 'none')
        self.is_torch = backend == 'torch'

    def __call__(self, events):
        T = self.T
        batch = np.stack(events, axis=0)                       # (B, F, nx, ny, neta, ntau)
        if self.apply_norm:
            batch = self.normalizer.transform(batch)           # one vectorised call
        x = np.repeat(batch[..., :1], T - 1, axis=-1)          # first-tau repeat, batched
        y = batch[..., 1:T]
        if self.is_torch:
            return {'x': torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                    'y': torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))}
        return {'x': np.ascontiguousarray(x, dtype=np.float32),
                'y': np.ascontiguousarray(y, dtype=np.float32)}


def make_collate(normalizer, time_steps, backend):
    """Build a picklable ``collate_fn``.

    Parameters
    ----------
    normalizer : Normalizer | None
        Applied (vectorised) to the whole stacked batch.  ``None`` or a
        ``name == 'none'`` normalizer is a no-op.
    time_steps : int
        Window length T.  Input ``x`` is the first tau step repeated ``T-1``
        times; target ``y`` is tau steps ``1..T-1``.
    backend    : str
        ``"torch"`` -> torch tensors; otherwise float32 numpy arrays (the MLX
        trainer converts to ``mx.array``).
    """
    return _Collate(normalizer, time_steps, backend)
