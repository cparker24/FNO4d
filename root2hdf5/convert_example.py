"""
Example: Convert a jagged ROOT file to a fixed-length, chunked HDF5 file.

Heavy-ion hydro codes (e.g. MUSIC) emit one ROOT event per hydrodynamic
trajectory.  Each event runs until its own freeze-out time tau_fo, so the
tau dimension is *variable*: event i has ntau_i time steps.  uproot exposes
this as a jagged (ragged) array — a list of flat float32 vectors of different
lengths.

Neural networks require fixed-shape tensors and efficient random-access I/O.
This script shows the complete pipeline:

  Step 1  Create a synthetic jagged ROOT file that mimics the MUSIC output
  Step 2  Inspect the jagged structure to understand the problem
  Step 3  Convert: pad/clip tau → fixed shape, write to chunked HDF5
  Step 4  Read back a slice with true partial I/O (no full-file decompression)
  Step 5  Verify round-trip correctness against the original ROOT data

Run:
    python convert_example.py
    python convert_example.py --nev 50 --chunk 4
    python convert_example.py --root my_data.root --h5 my_data.h5 --skip-create

Dependencies:
    pip install uproot awkward h5py numpy
"""

import argparse
import os
import tracemalloc
import warnings

import uproot
import awkward as ak
import numpy as np
import h5py

# Uproot 5.7+ defaults to RNTuple for new writes; suppress the informational
# warning since we intentionally create TTrees (same format as MUSIC output).
warnings.filterwarnings("ignore", message=".*RNTuple.*", category=FutureWarning)


# ---------------------------------------------------------------------------
# Step 1 — Create a synthetic jagged ROOT file
# ---------------------------------------------------------------------------

def make_synthetic_root(
    root_path: str,
    n_events: int = 20,
    nx: int = 8,
    ny: int = 8,
    neta: int = 4,
    n_features: int = 4,
    ntau_min: int = 5,
    ntau_max: int = 12,
    seed: int = 42,
):
    """Write a minimal ROOT file with a jagged tau dimension.

    Mimics the MUSIC bulk-writer format:
    - 'meta' tree: scalar grid parameters (one entry each)
    - 'events' tree:
        - 'data'          jagged float32 branch, length = ntau_i * nx * ny * neta * nF
        - 'ntau'          int32 per-event tau-step count  (the jagged key)
        - 'tau_freezeout' float32 per-event freeze-out proper time (fm/c)

    The per-event flat vector layout is:
        [tau_0_x_0_y_0_eta_0_f0, tau_0_x_0_y_0_eta_0_f1, ..., tau_{nt-1}_x_{nx-1}_...]
    i.e. row-major (tau, x, y, eta, feature).
    """
    rng = np.random.default_rng(seed)
    ntau_per_event = rng.integers(ntau_min, ntau_max + 1, size=n_events).astype(np.int32)

    # Each element of jagged is a 1-D float32 array whose length differs per event
    jagged = [
        rng.standard_normal(int(nt) * nx * ny * neta * n_features).astype(np.float32)
        for nt in ntau_per_event
    ]

    dtau      = 0.1   # fm/c per time step
    tau_min   = 0.4   # fm/c starting proper time
    tau_fo    = (ntau_per_event * dtau + tau_min).astype(np.float32)

    print(f"Creating synthetic ROOT file: {root_path}")
    print(f"  {n_events} events, ntau range [{ntau_min}, {ntau_max}]")
    print(f"  grid: nx={nx}, ny={ny}, neta={neta}, nFeatures={n_features}")

    with uproot.recreate(root_path) as f:
        f["meta"] = {
            "nx":         np.array([nx],         dtype=np.float64),
            "ny":         np.array([ny],         dtype=np.float64),
            "neta":       np.array([neta],       dtype=np.float64),
            "nFeatures":  np.array([n_features], dtype=np.float64),
            "dtau":       np.array([dtau],       dtype=np.float64),
            "tau_min":    np.array([tau_min],    dtype=np.float64),
            "ntau_max":   np.array([ntau_max],   dtype=np.float64),
        }
        f["events"] = {
            "data":          ak.Array(jagged),
            "ntau":          ntau_per_event,
            "tau_freezeout": tau_fo,
        }

    flat_lengths = [len(x) for x in jagged]
    print(f"  Flat branch lengths: min={min(flat_lengths)}, "
          f"max={max(flat_lengths)}, mean={np.mean(flat_lengths):.0f}")
    print(f"  Written: {root_path}")
    return ntau_per_event


# ---------------------------------------------------------------------------
# Step 2 — Inspect the jagged structure
# ---------------------------------------------------------------------------

def inspect_root(root_path: str):
    """Print a summary that makes the jagged problem concrete."""
    print(f"\n--- Inspecting ROOT file: {root_path} ---")

    with uproot.open(root_path) as f:
        meta = f["meta"]
        nx         = int(meta["nx"].array(library="np")[0])
        ny         = int(meta["ny"].array(library="np")[0])
        neta       = int(meta["neta"].array(library="np")[0])
        n_features = int(meta["nFeatures"].array(library="np")[0])

        branch   = f["events"]["data"]
        ntau_arr = f["events"]["ntau"].array(library="np")
        n_events = branch.num_entries

        print(f"  grid : nx={nx}, ny={ny}, neta={neta}, nFeatures={n_features}")
        print(f"  events: {n_events}")
        print(f"  ntau per event: min={ntau_arr.min()}, max={ntau_arr.max()}  "
              f"← variable-length (jagged) dimension")

        # Show the first few events to make the jaggedness visible
        sample_ak = branch.array(entry_stop=5, library="ak")
        print()
        print("  Per-event flat lengths (first 5 events):")
        for i in range(len(sample_ak)):
            nt  = int(ntau_arr[i])
            expected = nt * nx * ny * neta * n_features
            print(f"    event {i}: ntau={nt}  flat_len={len(sample_ak[i])}  "
                  f"(= {nt} × {nx} × {ny} × {neta} × {n_features} = {expected})")

    print()
    print("  Problem: different events have different sizes.")
    print("  torch.stack([event_0, event_1, ...]) fails — tensors must have")
    print("  identical shapes.  A DataLoader batch cannot be formed directly.")
    print("  Solution: pad shorter events with zeros, clip longer ones,")
    print("            producing a fixed (N, nF, nx, ny, neta, ntau) tensor.")


# ---------------------------------------------------------------------------
# Step 3 — Convert jagged ROOT → fixed-length chunked HDF5
# ---------------------------------------------------------------------------

def convert_to_hdf5(
    root_path: str,
    h5_path: str,
    choose_ntau: int = None,
    chunk_events: int = 1,
):
    """Pad/clip the jagged tau axis and write to a chunked HDF5 dataset.

    Parameters
    ----------
    root_path    : input .root file (local path)
    h5_path      : output .h5 file (created or overwritten)
    choose_ntau  : target tau length; None = max(ntau) across all events
    chunk_events : number of events per HDF5 chunk along the event axis.
                   1 gives the finest possible read granularity — reading
                   event k decompresses only that one chunk, equivalent to
                   SetAutoFlush(1) in the C++ writer.
    """
    print(f"\n--- Converting {root_path} → {h5_path} ---")

    # ---- read metadata and per-event arrays from ROOT ----
    with uproot.open(root_path) as f:
        meta       = f["meta"]
        nx         = int(meta["nx"].array(library="np")[0])
        ny         = int(meta["ny"].array(library="np")[0])
        neta       = int(meta["neta"].array(library="np")[0])
        n_features = int(meta["nFeatures"].array(library="np")[0])
        dtau       = float(meta["dtau"].array(library="np")[0])
        tau_min    = float(meta["tau_min"].array(library="np")[0])

        branch   = f["events"]["data"]
        ntau_arr = f["events"]["ntau"].array(library="np")
        tau_fo   = f["events"]["tau_freezeout"].array(library="np")
        n_events = branch.num_entries

        # Determine the target tau length
        if choose_ntau is None:
            choose_ntau = int(ntau_arr.max())

        # The flat vector length every event is padded / clipped to
        target_flat = int(choose_ntau * nx * ny * neta * n_features)

        n_padded  = int((ntau_arr < choose_ntau).sum())
        n_clipped = int((ntau_arr > choose_ntau).sum())
        print(f"  choose_ntau = {choose_ntau}  "
              f"({n_padded} events zero-padded, {n_clipped} clipped)")

        # Output shape: (N, nFeatures, nx, ny, neta, ntau)  — ML-standard layout
        # (batch, channels, spatial..., time)
        final_shape = (n_events, n_features, nx, ny, neta, choose_ntau)
        chunk_shape = (min(chunk_events, n_events), n_features, nx, ny, neta, choose_ntau)

        print(f"  output shape : {final_shape}")
        print(f"  HDF5 chunk   : {chunk_shape}  "
              f"(~{np.prod(chunk_shape) * 4 / 1024:.1f} KiB/chunk)")

        # ---- read the jagged array from ROOT ----
        print("  Reading jagged data from ROOT ...")
        tracemalloc.start()
        arr_ak = branch.array(library="ak")          # list of variable-length arrays
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  ROOT read peak memory: {peak / 1024**2:.1f} MiB")

        # ---- pad / clip every event to target_flat elements ----
        # ak.pad_none pads with None, clip=True discards elements beyond target_flat
        arr_padded = ak.pad_none(arr_ak, target_flat, axis=-1, clip=True)
        # fill_none replaces None with 0.0 — physically correct: cells past freeze-out
        # have zero energy density, so zero-padding is not just a bookkeeping trick
        arr_filled = ak.fill_none(arr_padded, np.float32(0.))
        flat_np    = ak.to_numpy(arr_filled).astype(np.float32)  # shape (N, target_flat)

        # ---- reshape and reorder axes into ML-standard layout ----
        # ROOT layout inside each event: (ntau, nx, ny, neta, nFeatures)  — row-major
        arr_nd = flat_np.reshape(n_events, choose_ntau, nx, ny, neta, n_features)
        arr_nd = np.moveaxis(arr_nd, 1, -1)   # ntau axis: position 1 → last
        arr_nd = np.moveaxis(arr_nd, -2, 1)   # nFeatures: second-to-last → position 1
        # arr_nd shape is now (N, nFeatures, nx, ny, neta, ntau)  ✓

        # ---- write to HDF5 ----
        with h5py.File(h5_path, "w") as hf:
            # scalar grid parameters as attributes (fast lookup, no dataset overhead)
            hf.attrs["nx"]          = nx
            hf.attrs["ny"]          = ny
            hf.attrs["neta"]        = neta
            hf.attrs["nFeatures"]   = n_features
            hf.attrs["dtau"]        = dtau
            hf.attrs["tau_min"]     = tau_min
            hf.attrs["choose_ntau"] = choose_ntau
            # use 'nevents' to match the production converter (read_3d_hdf5.py)
            hf.attrs["nevents"]     = n_events

            # per-event metadata — use 'ntau_freezeout' to match production format
            hf.create_dataset("ntau_freezeout", data=ntau_arr)
            hf.create_dataset("tau_freezeout",  data=tau_fo)

            # main array: chunked + compressed
            # chunk_events=1 → one HDF5 chunk per event → reading event k decompresses
            # only that chunk, not the entire file.  This is what enables efficient
            # shuffled DataLoader reads during training.
            ds = hf.create_dataset(
                "arr",
                shape=final_shape,
                dtype=np.float32,
                chunks=chunk_shape,
                compression="lzf",   # fast lossless; swap for "gzip" for smaller files
            )
            ds[:] = arr_nd

    size_mb = os.path.getsize(h5_path) / 1024**2
    print(f"  Written: {h5_path}  ({size_mb:.1f} MiB on disk)")


# ---------------------------------------------------------------------------
# Step 4 — Read back a slice with true partial I/O
# ---------------------------------------------------------------------------

def read_hdf5_partial(h5_path: str, start: int = 0, n: int = 5):
    """Load n events from the HDF5 file without touching the rest.

    With chunk_events=1, HDF5 decompresses exactly n chunks.
    Compare this to ROOT where reading any slice decompresses the full basket.
    """
    print(f"\n--- Partial read from HDF5: events [{start}, {start + n}) ---")

    tracemalloc.start()
    with h5py.File(h5_path, "r") as hf:
        attrs  = dict(hf.attrs)
        # This slice decompresses only the chunks that overlap [start, start+n)
        arr    = hf["arr"][start: start + n]
        ntau   = hf["ntau_freezeout"][start: start + n]
        tau_fo = hf["tau_freezeout"][start: start + n]
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  arr shape   : {arr.shape}")
    print(f"  arr dtype   : {arr.dtype}")
    print(f"  ntau loaded : {ntau}")
    print(f"  Peak memory : {peak / 1024**2:.2f} MiB  "
          f"(only {n} events loaded, not the full file)")
    print(f"  Grid attrs  : nx={int(attrs['nx'])}, ny={int(attrs['ny'])}, "
          f"neta={int(attrs['neta'])}, nFeatures={int(attrs['nFeatures'])}, "
          f"choose_ntau={int(attrs['choose_ntau'])}")
    return arr, ntau, tau_fo


# ---------------------------------------------------------------------------
# Step 5 — Verify round-trip correctness
# ---------------------------------------------------------------------------

def verify_roundtrip(root_path: str, h5_path: str, n_check: int = 5):
    """Assert that the HDF5 values match the original ROOT data (pre-padding region)."""
    print(f"\n--- Round-trip verification (first {n_check} events) ---")

    with uproot.open(root_path) as f:
        meta       = f["meta"]
        nx         = int(meta["nx"].array(library="np")[0])
        ny         = int(meta["ny"].array(library="np")[0])
        neta       = int(meta["neta"].array(library="np")[0])
        n_features = int(meta["nFeatures"].array(library="np")[0])
        ntau_arr   = f["events"]["ntau"].array(library="np")[:n_check]
        raw_ak     = f["events"]["data"].array(entry_stop=n_check, library="ak")

    with h5py.File(h5_path, "r") as hf:
        choose_ntau = int(hf.attrs["choose_ntau"])
        arr_h5      = hf["arr"][:n_check]   # (n_check, nF, nx, ny, neta, ntau)

    all_ok = True
    for i in range(n_check):
        nt     = int(ntau_arr[i])
        raw_np = np.asarray(raw_ak[i], dtype=np.float32)

        # Reproduce the same reshape + transpose the converter applied.
        # ROOT flat layout per event: (ntau, nx, ny, neta, nFeatures) row-major.
        # Target layout: (nFeatures, nx, ny, neta, ntau).
        # np.transpose axes: map position 0→4 1→1 2→2 3→3 4→0
        ref   = raw_np.reshape(nt, nx, ny, neta, n_features)
        ref   = np.transpose(ref, (4, 1, 2, 3, 0))
        # ref: (nF, nx, ny, neta, nt)

        h5sl  = arr_h5[i, :, :, :, :, :nt]  # skip zero-padded region

        match = np.allclose(ref, h5sl, atol=1e-5)
        if not match:
            all_ok = False
            print(f"  event {i}: MISMATCH  max_diff={np.max(np.abs(ref - h5sl)):.2e}")
        else:
            print(f"  event {i}: OK  nt={nt}")

        # Confirm that the padded region is exactly zero
        if nt < choose_ntau:
            pad_val = np.max(np.abs(arr_h5[i, :, :, :, :, nt:]))
            pad_ok  = "OK (zeros)" if pad_val == 0.0 else f"NONZERO ({pad_val:.2e})"
            print(f"          padding [{nt}:{choose_ntau}]: {pad_ok}")

    if all_ok:
        print("  All checked events match the original ROOT data.")
    else:
        print("  WARNING: mismatches detected — check the reshape/moveaxis logic.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Jagged ROOT → fixed-length HDF5 conversion example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nev",         type=int, default=20,              help="synthetic event count")
    p.add_argument("--root",        type=str, default="synthetic.root", help="ROOT file path")
    p.add_argument("--h5",          type=str, default="synthetic.h5",   help="HDF5 file path")
    p.add_argument("--chunk",       type=int, default=1,               help="HDF5 chunk size (events)")
    p.add_argument("--ntau",        type=int, default=None,            help="target tau steps (default: auto max)")
    p.add_argument("--skip-create", action="store_true",
                   help="skip ROOT creation if the file already exists")
    return p.parse_args()


def main():
    args = parse_args()

    # Step 1: create a synthetic ROOT file with variable-length tau per event
    if not (args.skip_create and os.path.exists(args.root)):
        make_synthetic_root(args.root, n_events=args.nev)
    else:
        print(f"Using existing ROOT file: {args.root}")

    # Step 2: inspect the jagged structure to understand the problem
    inspect_root(args.root)

    # Step 3: convert to fixed-length HDF5
    convert_to_hdf5(args.root, args.h5, choose_ntau=args.ntau, chunk_events=args.chunk)

    # Step 4: partial read — load only 5 events, leaving the rest on disk
    read_hdf5_partial(args.h5, start=0, n=5)

    # Step 5: round-trip correctness check
    verify_roundtrip(args.root, args.h5)

    print("\nDone.")
    print(f"  HDF5 file ready for ML training: {args.h5}")
    print("  Load a batch in PyTorch:")
    print("    with h5py.File(h5_path, 'r') as hf:")
    print("        x = torch.from_numpy(hf['arr'][batch_indices])")


if __name__ == "__main__":
    main()
