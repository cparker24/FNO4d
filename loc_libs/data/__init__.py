"""Unified dataset & DataLoader stack (``DATA_PIPELINE_PLAN.md``).

One source interface + one Dataset + one collate + one loader builder:

    ArraySource  ->  FluidDataset  ->  make_collate  ->  build_dataloaders

Primary API
-----------
    from data import build_dataloaders, FluidDataset, make_collate
    from data import RootSource, ZarrSource, MultiH5Array, DownsampledArrayView
    from data import BlockBatchSampler, worker_init_fn, ArraySource

Back-compat
-----------
The legacy per-event dataset classes (``FluidDatasetH5``, ``FluidDatasetMLX``,
``FluidDatasetH5MLX``) remain importable from here as thin shims so call sites
can migrate incrementally; they still live in ``fluid_datasets``.  Note that
``data.FluidDataset`` is the *new* raw-event dataset (§7), which replaces the
old eager ``fluid_datasets.FluidDataset``; the legacy eager class is available
as :data:`FluidDatasetEager`.
"""

from .sources import (ArraySource, RootSource, ZarrSource,
                      MultiH5Array, DownsampledArrayView, build_source)
from .dataset import FluidDataset
from .collate import make_collate
from .loader import build_dataloaders, BlockBatchSampler, worker_init_fn

# ---------------------------------------------------------------------------
# back-compat shims for the legacy per-event dataset classes
# ---------------------------------------------------------------------------
from fluid_datasets import (FluidDatasetH5, FluidDatasetMLX, FluidDatasetH5MLX)
from fluid_datasets import FluidDataset as FluidDatasetEager

__all__ = [
    'ArraySource', 'RootSource', 'ZarrSource',
    'MultiH5Array', 'DownsampledArrayView', 'build_source',
    'FluidDataset', 'make_collate',
    'build_dataloaders', 'BlockBatchSampler', 'worker_init_fn',
    # legacy
    'FluidDatasetEager', 'FluidDatasetH5', 'FluidDatasetMLX', 'FluidDatasetH5MLX',
]
