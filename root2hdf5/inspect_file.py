"""
Quick inspector for ROOT and HDF5 hydro data files.

Usage:
    python inspect_file.py file.root
    python inspect_file.py file.h5
    python inspect_file.py file.root file.h5 ...
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*RNTuple.*", category=FutureWarning)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# HDF5 inspection
# ---------------------------------------------------------------------------

def inspect_h5(path: str) -> None:
    import h5py
    import numpy as np

    size_mb = os.path.getsize(path) / 1024**2
    print(f"  Type        : HDF5")
    print(f"  Size        : {size_mb:.1f} MiB")

    with h5py.File(path, "r") as hf:
        attrs = dict(hf.attrs)

        # Grid parameters
        grid_keys = ["nx", "ny", "neta", "nFeatures",
                     "dtau", "tau_min", "tau_min_MUSIC",
                     "dx", "dy", "deta", "x_min", "y_min", "eta_min",
                     "choose_ntau", "nevents"]
        print("\n  Grid / metadata attributes:")
        for k in grid_keys:
            if k in attrs:
                v = attrs[k]
                print(f"    {k:<20} {v}")

        # Extra attributes not in the standard list
        extra = {k: v for k, v in attrs.items()
                 if k not in grid_keys and k != "source_files"}
        for k, v in extra.items():
            print(f"    {k:<20} {v}")

        # Main array
        if "arr" in hf:
            ds = hf["arr"]
            shape  = ds.shape
            chunks = ds.chunks
            size_gib = ds.size * 4 / 1024**3   # float32
            print(f"\n  Array 'arr':")
            print(f"    shape       : {shape}  (nevents, nF, nx, ny, neta, ntau)")
            print(f"    dtype       : {ds.dtype}")
            print(f"    chunks      : {chunks}")
            print(f"    uncompressed: {size_gib:.2f} GiB")

        # Per-event tau info
        if "ntau_freezeout" in hf:
            ntau = hf["ntau_freezeout"][:]
            print(f"\n  ntau_freezeout (per-event actual tau steps):")
            print(f"    min / max / mean : {ntau.min()} / {ntau.max()} / {ntau.mean():.1f}")
            choose = int(attrs.get("choose_ntau", ntau.max()))
            n_padded  = int((ntau < choose).sum())
            n_clipped = int((ntau > choose).sum())
            n_exact   = int((ntau == choose).sum())
            print(f"    vs choose_ntau={choose}: "
                  f"{n_exact} exact, {n_padded} zero-padded, {n_clipped} clipped")

        # Merge provenance
        if "source_files" in attrs:
            src = list(attrs["source_files"])
            print(f"\n  Merged from {len(src)} file(s):")
            for i, p in enumerate(src):
                print(f"    [{i}] {p}")

        # All datasets
        print(f"\n  Datasets:")
        def _print_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"    {name:<22} shape={obj.shape}  dtype={obj.dtype}")
        hf.visititems(_print_ds)


# ---------------------------------------------------------------------------
# ROOT inspection
# ---------------------------------------------------------------------------

_PROD_SCALAR_KEYS = [
    "nx", "ny", "neta", "nFeatures",
    "x_min", "y_min", "eta_min",
    "dx", "dy", "deta",
    "tau_min", "tau_min_MUSIC", "dtau", "ntau",
]


def _detect_root_mode(f) -> str:
    return "standalone" if "meta" in f else "production"


def inspect_root(path: str) -> None:
    import uproot
    import awkward as ak
    import numpy as np

    size_mb = os.path.getsize(path) / 1024**2
    print(f"  Type        : ROOT")
    print(f"  Size        : {size_mb:.1f} MiB")

    with uproot.open(path) as f:
        mode = _detect_root_mode(f)
        print(f"  Format      : {mode}")

        # --- grid parameters ---
        print("\n  Grid / metadata:")
        if mode == "standalone":
            m = f["meta"]
            for k in ["nx", "ny", "neta", "nFeatures", "dtau", "tau_min", "ntau_max"]:
                if k in m:
                    print(f"    {k:<20} {m[k].array(library='np')[0]}")
            ntau_arr = f["events"]["ntau"].array(library="np")
            n_events = f["events"]["ntau"].num_entries
            fixed_ntau = "ntau_max" in m   # synthetic format stores a fixed max
        else:
            for k in _PROD_SCALAR_KEYS:
                if k in f:
                    print(f"    {k:<20} {f[k].member('fVal')}")
            ntau_arr = ak.to_numpy(f["t"]["ntau_freezeout"].array(library="ak"))
            n_events = len(ntau_arr)
            fixed_ntau = False

        # --- event / tau summary ---
        print(f"\n  Events      : {n_events}")

        ntau_min  = int(ntau_arr.min())
        ntau_max  = int(ntau_arr.max())
        ntau_mean = float(ntau_arr.mean())
        is_jagged = ntau_min != ntau_max

        if is_jagged:
            print(f"  τ dimension : VARIABLE (jagged)")
            print(f"    ntau min / max / mean : {ntau_min} / {ntau_max} / {ntau_mean:.1f}")
            print(f"    → choose_ntau for HDF5 conversion: {ntau_max}  (max)")
        else:
            print(f"  τ dimension : fixed  (ntau = {ntau_min})")

        # --- size estimates ---
        if mode == "standalone":
            nx        = int(f["meta"]["nx"].array(library="np")[0])
            ny        = int(f["meta"]["ny"].array(library="np")[0])
            neta      = int(f["meta"]["neta"].array(library="np")[0])
            nFeatures = int(f["meta"]["nFeatures"].array(library="np")[0])
        else:
            nx        = int(f["nx"].member("fVal"))
            ny        = int(f["ny"].member("fVal"))
            neta      = int(f["neta"].member("fVal"))
            nFeatures = int(f["nFeatures"].member("fVal"))

        cells_per_event_actual = ntau_arr * nx * ny * neta * nFeatures
        raw_gib = float(cells_per_event_actual.sum()) * 4 / 1024**3

        padded_shape = (n_events, nFeatures, nx, ny, neta, ntau_max)
        padded_gib   = int(np.prod(padded_shape)) * 4 / 1024**3

        print(f"\n  Array size estimates (float32):")
        print(f"    HDF5 output shape   : {padded_shape}")
        print(f"    Uncompressed padded : {padded_gib:.2f} GiB")
        print(f"    Raw data (no pad)   : {raw_gib:.2f} GiB")
        print(f"    Padding overhead    : {(padded_gib - raw_gib) / padded_gib * 100:.1f}%")

        # --- top-level keys ---
        print(f"\n  Top-level keys : {list(f.keys())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="+", metavar="FILE",
                   help=".root or .h5 / .hdf5 file(s) to inspect")
    return p.parse_args()


def main():
    args = parse_args()
    sep  = "=" * 60

    for path in args.files:
        print(sep)
        print(f"  {path}")
        print(sep)

        if not os.path.exists(path):
            print(f"  ERROR: file not found: {path}")
            print()
            continue

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".h5", ".hdf5"):
                inspect_h5(path)
            elif ext == ".root":
                inspect_root(path)
            else:
                print(f"  ERROR: unrecognised extension '{ext}' — expected .root, .h5, or .hdf5")
        except Exception as exc:
            print(f"  ERROR: {exc}")

        print()


if __name__ == "__main__":
    main()
