"""
Example: Using the production loc_libs converter (loc_libs/read_3d_hdf5.py).

When to use this vs. the standalone convert_example.py
-------------------------------------------------------
convert_example.py (standalone)
  - Minimal dependencies: uproot, awkward, h5py, numpy only
  - No FNO4d installation required
  - Ideal for a batch-conversion node or CI machine that does not need
    PyTorch, MLX, or the full FNO4d stack
  - Good starting point for adapting to a different ROOT layout

convert_example_production.py  ← this file
  - Uses loc_libs/read_3d_hdf5.py which ships with the FNO4d project
  - Full MUSIC ROOT format support (TParameter scalars, t/user_res branch)
  - Remote file support: s3://, gs://, http://
  - int32-safe batching for very large files (> ~2 B total elements)
  - Integrated memory tracing via tracemalloc
  - read_3d_data_hdf5() returns the same dict as read_3d_root.py, so
    training scripts need no changes when switching from ROOT to HDF5

Both converters produce HDF5 files in the same format, so read_3d_data_hdf5()
works on output from either script.

Script structure
----------------
  Part A  Conversion  — convert_root_to_hdf5() API and real-data usage
  Part B  Read-back   — read_3d_data_hdf5() with partial/streaming reads
  Part C  DataLoader  — PyTorch Dataset wrapping the HDF5 file
  Part D  Remote      — S3 / GCS conversion patterns

For Parts B–C this script uses a synthetic HDF5 file created by
convert_example.py, since the production converter requires a real MUSIC
ROOT file.  Swap the h5_path for the path to your converted file.

Run:
    python convert_example_production.py
    python convert_example_production.py --h5 /data/dAu_250ev_mb.h5

Dependencies (loc_libs path):
    pip install -e ../loc_libs   # or: sys.path.insert as done below
Additional (this demo only):
    pip install uproot awkward    # to build the synthetic HDF5
"""

import argparse
import os
import sys
import tracemalloc

import numpy as np

# Resolve loc_libs from the repo root regardless of working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)

from loc_libs.read_3d_hdf5 import convert_root_to_hdf5, read_3d_data_hdf5


# ---------------------------------------------------------------------------
# Part A — Conversion: convert_root_to_hdf5()
# ---------------------------------------------------------------------------

def show_conversion_api():
    """Print the documented API for the production converter."""
    print("=" * 70)
    print("Part A — Production converter: convert_root_to_hdf5()")
    print("=" * 70)
    print("""
Signature:
    convert_root_to_hdf5(
        root_file_path,          # local path or s3:// / gs:// / http://
        hdf5_file_path,          # output .h5 (created or overwritten)
        nevents      = -1,       # -1 = all events
        choose_ntau  = None,     # None = auto-detect max(ntau_freezeout)
        chunk_events = 1,        # HDF5 events per chunk (1 = finest)
        storage_options = None,  # dict forwarded to uproot.open
        verbose      = False,    # print tracemalloc memory snapshots
    )

Expected ROOT layout (MUSIC output):
    top-level TParameter<double> objects for each scalar grid parameter:
        nx, ny, neta, nFeatures, x_min, y_min, eta_min,
        dx, dy, deta, tau_min, tau_min_MUSIC, dtau
    tree 't' containing:
        ntau_freezeout  — int array, per-event tau step count  (jagged key)
        tau_freezeout   — float array, per-event freeze-out proper time (fm/c)
        user_res        — jagged float32 branch, length per event =
                          ntau_i × nx × ny × neta × nFeatures

Typical call for a local MUSIC file:

    from loc_libs.read_3d_hdf5 import convert_root_to_hdf5

    convert_root_to_hdf5(
        "data/dAu_250ev_mb.root",
        "data/dAu_250ev_mb.h5",
        nevents      = -1,       # convert all events
        choose_ntau  = None,     # auto: use max freeze-out steps found
        chunk_events = 1,        # one event per HDF5 chunk
        verbose      = True,     # print memory snapshots during conversion
    )

This is a one-time operation.  The 2× ROOT peak memory is accepted here
so that all subsequent reads (training, evaluation) use the HDF5 path.
""")


def example_conversion_local():
    """Code snippet — local file conversion."""
    snippet = """\
from loc_libs.read_3d_hdf5 import convert_root_to_hdf5

# --- local MUSIC ROOT file ---
convert_root_to_hdf5(
    root_file_path = "data/dAu_250ev_mb.root",
    hdf5_file_path = "data/dAu_250ev_mb.h5",
    nevents        = -1,     # all events
    choose_ntau    = None,   # auto-detect from ntau_freezeout
    chunk_events   = 1,      # finest read granularity
    verbose        = True,
)
"""
    print("Local conversion snippet:")
    print(snippet)


def example_conversion_remote():
    """Code snippet — remote file conversion (S3 / GCS)."""
    snippet = """\
from loc_libs.read_3d_hdf5 import convert_root_to_hdf5

# --- AWS S3 or S3-compatible (Lambda storage, MinIO, …) ---
convert_root_to_hdf5(
    root_file_path = "s3://my-bucket/dAu_250ev_mb.root",
    hdf5_file_path = "data/dAu_250ev_mb.h5",       # local output
    storage_options = {
        "key":          "ACCESS_KEY",
        "secret":       "SECRET_KEY",
        "endpoint_url": "https://storage.example.com",   # omit for AWS
    },
    verbose = True,
)

# --- Google Cloud Storage ---
convert_root_to_hdf5(
    root_file_path  = "gs://my-bucket/dAu_250ev_mb.root",
    hdf5_file_path  = "data/dAu_250ev_mb.h5",
    storage_options = {"token": "path/to/service_account.json"},
    verbose         = True,
)
"""
    print("Remote conversion snippets (S3 / GCS):")
    print(snippet)
    print("Note: install the optional cloud backends first:")
    print("  pip install -r requirements-s3.txt    # S3 / S3-compatible")
    print("  pip install -r requirements-gcs.txt   # Google Cloud Storage")
    print()


# ---------------------------------------------------------------------------
# Part B — Read-back: read_3d_data_hdf5()
# ---------------------------------------------------------------------------

def demo_readback(h5_path: str):
    """Demonstrate read_3d_data_hdf5() with partial reads and verbose mode."""
    print("=" * 70)
    print("Part B — read_3d_data_hdf5(): partial reads and memory tracing")
    print("=" * 70)

    # B1: read the first 5 events (partial I/O — only those HDF5 chunks decompress)
    print("\nB1. Load first 5 events:")
    dat = read_3d_data_hdf5(h5_path, nevents=5, start_event=0)
    arr = dat["arr"]
    print(f"  arr shape : {arr.shape}   (nevents, nF, nx, ny, neta, ntau)")
    print(f"  arr dtype : {arr.dtype}")
    print(f"  Grid      : nx={int(dat['nx'])}, ny={int(dat['ny'])}, "
          f"neta={int(dat['neta'])}, nFeatures={int(dat['nFeatures'])}")
    print(f"  dtau      : {dat['dtau']:.4f} fm/c")

    # B2: load a different slice (events 5–9) — reads a disjoint set of chunks
    print("\nB2. Load events 5–9 (disjoint slice):")
    dat2 = read_3d_data_hdf5(h5_path, nevents=5, start_event=5)
    print(f"  arr shape : {dat2['arr'].shape}")

    # B3: verbose mode — prints tracemalloc peak before and after the HDF5 read
    print("\nB3. Verbose mode (tracemalloc memory tracing):")
    tracemalloc.start()
    dat3 = read_3d_data_hdf5(h5_path, nevents=3, start_event=0, verbose=True)
    tracemalloc.stop()
    print(f"  Loaded arr shape: {dat3['arr'].shape}")

    # B4: the returned dict is drop-in compatible with read_3d_data() in read_3d_root.py
    print("\nB4. Dict keys returned (same as read_3d_root.read_3d_data):")
    for k, v in dat.items():
        if k == "arr":
            print(f"  arr            → ndarray {v.shape}")
        elif k == "nans":
            print(f"  nans           → {v}")
        elif hasattr(v, "__len__"):
            print(f"  {k:<14} → array len={len(v)}")
        else:
            print(f"  {k:<14} → {v}")


# ---------------------------------------------------------------------------
# Part C — PyTorch DataLoader integration
# ---------------------------------------------------------------------------

def demo_dataloader(h5_path: str):
    """Show a minimal PyTorch Dataset backed by the HDF5 file."""
    print("\n" + "=" * 70)
    print("Part C — PyTorch DataLoader integration")
    print("=" * 70)

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
    except ImportError:
        print("  PyTorch not available — showing code only.\n")
        _print_dataloader_snippet(h5_path)
        return

    import h5py

    class HydroHDF5Dataset(Dataset):
        """Lazy-loading Dataset: one h5py open per __getitem__ call.

        Each call decompresses exactly one HDF5 chunk (chunk_events=1),
        so peak memory per worker is proportional to batch_size, not
        the full file size.

        Input  x: initial condition  — arr[:, :, :, :, :1]  (tau step 0 repeated)
        Target y: subsequent steps   — arr[:, :, :, :, 1:]
        """

        def __init__(self, h5_path, indices=None):
            self.h5_path = h5_path
            with h5py.File(h5_path, "r") as hf:
                n = int(hf.attrs["nevents"])
                self.ntau = int(hf.attrs["choose_ntau"])
            self.indices = list(range(n)) if indices is None else list(indices)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            with h5py.File(self.h5_path, "r") as hf:
                arr = hf["arr"][idx]              # decompresses one chunk only
            # (nF, nx, ny, neta, ntau) → split tau into x / y
            x = np.repeat(arr[..., :1], self.ntau - 1, axis=-1)  # initial condition
            y = arr[..., 1:]                                      # target steps
            return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

    import h5py
    with h5py.File(h5_path, "r") as hf:
        n_total = int(hf.attrs["nevents"])

    # 80/20 train/val split
    n_train = int(0.8 * n_total)
    idx_all = list(range(n_total))
    train_set = HydroHDF5Dataset(h5_path, indices=idx_all[:n_train])
    val_set   = HydroHDF5Dataset(h5_path, indices=idx_all[n_train:])

    # num_workers > 0: each worker opens its own h5py handle (safe because
    # h5py opens the file independently in each forked process)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False, num_workers=0)

    print(f"\n  Train events: {len(train_set)},  Val events: {len(val_set)}")
    print(f"  Train batches per epoch: {len(train_loader)}")

    x_batch, y_batch = next(iter(train_loader))
    print(f"\n  x_batch shape : {tuple(x_batch.shape)}   (initial condition, tiled)")
    print(f"  y_batch shape : {tuple(y_batch.shape)}   (target tau steps 1..T)")
    print(f"  x dtype       : {x_batch.dtype}")

    _print_dataloader_snippet(h5_path)


def _print_dataloader_snippet(h5_path):
    print("""
  ── Drop into your training script ──────────────────────────────────
  from torch.utils.data import Dataset, DataLoader
  import h5py, numpy as np, torch

  class HydroHDF5Dataset(Dataset):
      def __init__(self, h5_path, indices=None):
          self.h5_path = h5_path
          with h5py.File(h5_path, "r") as hf:
              n = int(hf.attrs["nevents"])
              self.ntau = int(hf.attrs["choose_ntau"])
          self.indices = list(range(n)) if indices is None else list(indices)

      def __len__(self):
          return len(self.indices)

      def __getitem__(self, i):
          idx = self.indices[i]
          with h5py.File(self.h5_path, "r") as hf:
              arr = hf["arr"][idx]          # one chunk decompressed
          x = np.repeat(arr[..., :1], self.ntau - 1, axis=-1)
          y = arr[..., 1:]
          return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())

  dataset = HydroHDF5Dataset("data/dAu_250ev_mb.h5")
  loader  = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
  ─────────────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# Part D — Remote file snippet (S3 / GCS)
# ---------------------------------------------------------------------------

def show_remote_snippets():
    print("=" * 70)
    print("Part D — Remote file patterns")
    print("=" * 70)
    example_conversion_remote()


# ---------------------------------------------------------------------------
# Helpers: build a demo HDF5 if no real file is provided
# ---------------------------------------------------------------------------

def _build_demo_h5(h5_path: str):
    """Use the standalone converter to create a small synthetic HDF5 for demo."""
    import warnings
    warnings.filterwarnings("ignore", message=".*RNTuple.*", category=FutureWarning)

    # Import from the standalone script in the same directory
    _dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _dir)
    from convert_example import make_synthetic_root, convert_to_hdf5

    root_path = h5_path.replace(".h5", "_synthetic_demo.root")
    print(f"Building demo HDF5 via standalone converter ...")
    print(f"  ROOT : {root_path}")
    print(f"  HDF5 : {h5_path}")
    make_synthetic_root(root_path, n_events=20)
    convert_to_hdf5(root_path, h5_path, chunk_events=1)
    os.remove(root_path)   # tidy up the intermediate ROOT file
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--h5", type=str, default=None,
        help="path to an existing HDF5 file produced by either converter; "
             "if omitted a small synthetic demo file is created automatically",
    )
    p.add_argument(
        "--keep-demo", action="store_true",
        help="do not delete the synthetic demo HDF5 after the script finishes",
    )
    return p.parse_args()


def main():
    args = parse_args()

    demo_created = False
    if args.h5 is None:
        h5_path    = os.path.join(os.path.dirname(__file__), "demo_production.h5")
        demo_created = True
        _build_demo_h5(h5_path)
    else:
        h5_path = args.h5
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        print(f"Using existing HDF5 file: {h5_path}\n")

    # Part A: conversion API (printed, not executed — requires a real MUSIC ROOT file)
    show_conversion_api()
    example_conversion_local()

    # Part B: read_3d_data_hdf5 — actually executed on the demo / real HDF5
    demo_readback(h5_path)

    # Part C: PyTorch DataLoader
    demo_dataloader(h5_path)

    # Part D: remote snippets (printed only)
    show_remote_snippets()

    if demo_created and not args.keep_demo:
        os.remove(h5_path)
        print(f"Demo file removed: {h5_path}")
    else:
        print(f"\nHDF5 file retained: {h5_path}")


if __name__ == "__main__":
    main()
