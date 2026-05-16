"""
Batch ROOT → HDF5 converter with optional multi-file merge.

Accepts one or more ROOT files or directories.

Default mode — one HDF5 per ROOT file:
  Each .root file is converted to a .h5 file in the same directory.
  Files whose .h5 already exists are skipped (--force to overwrite).

Merge mode (--merge OUTPUT.h5):
  All ROOT files are combined into a single HDF5 file.
  Events are padded to the global maximum tau length across all input files
  (or the value given by --choose-ntau).

Two converter backends are supported:

  auto (default)
               Peeks at each ROOT file and selects the right backend.

  standalone   Expects a ROOT layout with a 'meta' tree and an 'events' tree
               (ntau, data, tau_freezeout branches).  Minimal dependencies:
               uproot, awkward, h5py, numpy.

  production   Full MUSIC ROOT format (TParameter scalars, t/user_res branch).
               Requires the loc_libs package from this repo.

Examples
--------
# convert one file (backend auto-detected)
python root_to_hdf5.py data/run01.root

# convert all ROOT files in a directory (skip existing HDF5)
python root_to_hdf5.py data/

# convert recursively, force overwrite
python root_to_hdf5.py --force --recursive data/

# merge all ROOT files in a directory into one HDF5
python root_to_hdf5.py --merge data/all_merged.h5 data/

# merge explicit files into one HDF5
python root_to_hdf5.py --merge out/merged.h5 a.root b.root c.root

# dry-run: show what would happen without doing anything
python root_to_hdf5.py --dry-run --merge out/merged.h5 data/

# custom tau steps and chunk size
python root_to_hdf5.py --choose-ntau 20 --chunk-events 4 data/run01.root
"""

import argparse
import gc
import os
import sys
import warnings

import awkward as ak
import h5py
import numpy as np
import uproot

warnings.filterwarnings("ignore", message=".*RNTuple.*", category=FutureWarning)

# Repo root on sys.path so loc_libs is importable regardless of cwd
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _THIS_DIR)

# TParameter scalar keys used by MUSIC / production format
_PROD_SCALAR_KEYS = [
    "nx", "ny", "neta", "nFeatures",
    "x_min", "y_min", "eta_min",
    "dx", "dy", "deta",
    "tau_min", "tau_min_MUSIC", "dtau",
]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _collect_root_files(inputs: list[str], recursive: bool) -> list[str]:
    """Return a deduplicated list of .root paths from files and/or directories."""
    seen  = set()
    files = []

    for inp in inputs:
        if os.path.isfile(inp):
            if not inp.endswith(".root"):
                print(f"  [warn] skipping non-ROOT file: {inp}")
                continue
            abs_p = os.path.abspath(inp)
            if abs_p not in seen:
                seen.add(abs_p)
                files.append(abs_p)

        elif os.path.isdir(inp):
            walker = os.walk(inp) if recursive else [(inp, [], os.listdir(inp))]
            for dirpath, _dirs, filenames in walker:
                for fn in sorted(filenames):
                    if fn.endswith(".root"):
                        abs_p = os.path.abspath(os.path.join(dirpath, fn))
                        if abs_p not in seen:
                            seen.add(abs_p)
                            files.append(abs_p)
        else:
            print(f"  [warn] input not found: {inp}")

    return files


def _h5_path_for(root_path: str) -> str:
    if root_path.endswith(".root"):
        return root_path[:-5] + ".h5"
    return root_path + ".h5"


# ---------------------------------------------------------------------------
# Format detection and metadata reading
# ---------------------------------------------------------------------------

def _detect_mode(root_path: str) -> str:
    """Return 'standalone' if the file has a 'meta' tree, else 'production'."""
    with uproot.open(root_path) as f:
        return "standalone" if "meta" in f else "production"


def _read_file_meta(root_path: str, mode: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Lightweight first pass: read grid params, ntau array, and tau_fo array.

    Returns (grid_dict, ntau_arr, tau_fo_arr).  Does NOT load event data.
    """
    with uproot.open(root_path) as f:
        if mode == "standalone":
            m = f["meta"]
            grid = dict(
                nx        = int(m["nx"].array(library="np")[0]),
                ny        = int(m["ny"].array(library="np")[0]),
                neta      = int(m["neta"].array(library="np")[0]),
                nFeatures = int(m["nFeatures"].array(library="np")[0]),
                dtau      = float(m["dtau"].array(library="np")[0]),
                tau_min   = float(m["tau_min"].array(library="np")[0]),
            )
            ntau_arr = f["events"]["ntau"].array(library="np")
            tau_fo   = f["events"]["tau_freezeout"].array(library="np")
        else:
            grid = {k: f[k].member("fVal") for k in _PROD_SCALAR_KEYS if k in f}
            ntau_arr = ak.to_numpy(f["t"]["ntau_freezeout"].array(library="ak"))
            tau_fo   = ak.to_numpy(f["t"]["tau_freezeout"].array(library="ak"))

    return grid, ntau_arr.astype(np.int32), tau_fo.astype(np.float32)


def _grid_key(grid: dict) -> tuple:
    """Return a tuple of the four shape-defining grid parameters."""
    return (int(grid["nx"]), int(grid["ny"]), int(grid["neta"]), int(grid["nFeatures"]))


# ---------------------------------------------------------------------------
# Single-file converters (unchanged behaviour from before)
# ---------------------------------------------------------------------------

def _convert_standalone(root_path: str, h5_path: str, args) -> None:
    from convert_example import convert_to_hdf5
    convert_to_hdf5(
        root_path,
        h5_path,
        choose_ntau  = args.choose_ntau,
        chunk_events = args.chunk_events,
    )


def _convert_production(root_path: str, h5_path: str, args) -> None:
    from loc_libs.read_3d_hdf5 import convert_root_to_hdf5
    convert_root_to_hdf5(
        root_file_path = root_path,
        hdf5_file_path = h5_path,
        nevents        = args.nevents,
        choose_ntau    = args.choose_ntau,
        chunk_events   = args.chunk_events,
        verbose        = args.verbose,
    )


_SINGLE_CONVERTERS = {
    "standalone": _convert_standalone,
    "production": _convert_production,
}


# ---------------------------------------------------------------------------
# Merge: multiple ROOT files → one HDF5
# ---------------------------------------------------------------------------

def _merge_files(root_files: list[str], h5_path: str, args) -> None:
    """Merge all root_files into a single HDF5 at h5_path.

    Pass 1: read grid params and ntau arrays (no event data loaded).
    Pass 2: read and pad events from each file, write incrementally.

    The global choose_ntau is max(ntau) across all events in all files
    unless overridden by --choose-ntau.
    """
    print(f"Merge: {len(root_files)} ROOT file(s) → {h5_path}")

    # --- Pass 1: metadata + ntau arrays ---------------------------------
    print("\nPass 1: reading metadata from all files ...")
    modes      = []
    grids      = []
    ntau_lists = []
    tau_lists  = []
    nevents_per_file = []

    for i, rp in enumerate(root_files, 1):
        mode = args.mode if args.mode != "auto" else _detect_mode(rp)
        grid, ntau_arr, tau_fo = _read_file_meta(rp, mode)

        cap = args.nevents
        if cap > 0:
            ntau_arr = ntau_arr[:cap]
            tau_fo   = tau_fo[:cap]

        modes.append(mode)
        grids.append(grid)
        ntau_lists.append(ntau_arr)
        tau_lists.append(tau_fo)
        nevents_per_file.append(len(ntau_arr))
        print(f"  [{i}/{len(root_files)}] {os.path.basename(rp)}"
              f"  mode={mode}  events={len(ntau_arr)}"
              f"  ntau=[{ntau_arr.min()}, {ntau_arr.max()}]")

    # --- Validate mode and grid consistency -----------------------------
    if len(set(modes)) > 1:
        raise ValueError(
            f"Mixed ROOT formats detected across input files: {set(modes)}. "
            "All files must use the same format for merging."
        )
    mode = modes[0]

    ref_key  = _grid_key(grids[0])
    ref_grid = grids[0]
    for i, (g, rp) in enumerate(zip(grids[1:], root_files[1:]), 1):
        if _grid_key(g) != ref_key:
            raise ValueError(
                f"Grid mismatch between file 0 and file {i}:\n"
                f"  {root_files[0]}: nx={ref_key[0]}, ny={ref_key[1]}, "
                f"neta={ref_key[2]}, nFeatures={ref_key[3]}\n"
                f"  {rp}: nx={_grid_key(g)[0]}, ny={_grid_key(g)[1]}, "
                f"neta={_grid_key(g)[2]}, nFeatures={_grid_key(g)[3]}"
            )

    # uproot.iterate chain: one entry per file, tree and branch by format
    tree_suffix = ":events" if mode == "standalone" else ":t"
    branch_name = "data"     if mode == "standalone" else "user_res"
    tree_paths  = [f"{rp}{tree_suffix}" for rp in root_files]

    # --- Determine global choose_ntau ------------------------------------
    all_ntau    = np.concatenate(ntau_lists)
    choose_ntau = args.choose_ntau if args.choose_ntau else int(all_ntau.max())
    total_ev    = sum(nevents_per_file)

    nx        = int(ref_grid["nx"])
    ny        = int(ref_grid["ny"])
    neta      = int(ref_grid["neta"])
    nFeatures = int(ref_grid["nFeatures"])

    n_padded  = int((all_ntau < choose_ntau).sum())
    n_clipped = int((all_ntau > choose_ntau).sum())

    print(f"\nGlobal choose_ntau = {choose_ntau}"
          f"  ({n_padded} events zero-padded, {n_clipped} clipped)")
    print(f"Total events       = {total_ev}")
    final_shape = (total_ev, nFeatures, nx, ny, neta, choose_ntau)
    chunk_shape = (min(args.chunk_events, total_ev), nFeatures, nx, ny, neta, choose_ntau)
    print(f"Output shape       = {final_shape}")
    print(f"HDF5 chunk         = {chunk_shape}"
          f"  (~{np.prod(chunk_shape) * 4 / 1024**2:.1f} MiB/chunk)\n")

    if args.dry_run:
        print("[dry-run] would write:", h5_path)
        return

    # --- Create HDF5 ----------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    ntau_all = np.concatenate(ntau_lists)
    tau_all  = np.concatenate(tau_lists)

    # source_file index: integer per event pointing into the file list
    src_idx  = np.concatenate([
        np.full(n, i, dtype=np.int32)
        for i, n in enumerate(nevents_per_file)
    ])

    with h5py.File(h5_path, "w") as hf:
        # grid attributes
        for k, v in ref_grid.items():
            hf.attrs[k] = v
        hf.attrs["choose_ntau"] = choose_ntau
        hf.attrs["nevents"]     = total_ev

        # source file list as a variable-length string attribute
        hf.attrs["source_files"] = [os.path.abspath(rp) for rp in root_files]

        hf.create_dataset("ntau_freezeout", data=ntau_all)
        hf.create_dataset("tau_freezeout",  data=tau_all)
        hf.create_dataset("source_file",    data=src_idx,
                          compression="lzf")

        ds = hf.create_dataset(
            "arr",
            shape=final_shape,
            dtype=np.float32,
            chunks=chunk_shape,
            compression="lzf",
        )

        # --- Pass 2: iterate across all chained files --------------------
        print("Pass 2: reading and writing events ...")
        target_len = int(choose_ntau * nx * ny * neta * nFeatures)
        step_size  = max(1, total_ev // 10)
        out_idx    = 0
        for batch in uproot.iterate(tree_paths, filter_name=branch_name,
                                    step_size=step_size):
            remaining = total_ev - out_idx
            if remaining <= 0:
                break
            arr_ak = batch[branch_name][:remaining]
            n      = len(arr_ak)
            print(f"  writing events {out_idx}–{out_idx + n - 1} ...")

            padded = ak.to_numpy(
                ak.fill_none(ak.pad_none(arr_ak, target_len, axis=-1, clip=True),
                             np.float32(0.))
            ).astype(np.float32, copy=False)
            del arr_ak

            chunk = padded.reshape(n, choose_ntau, nx, ny, neta, nFeatures)
            del padded
            chunk = np.moveaxis(chunk, 1, -1)   # ntau → last
            chunk = np.moveaxis(chunk, -2, 1)   # nFeatures → axis 1
            ds[out_idx: out_idx + n] = chunk
            del chunk
            out_idx += n
            gc.collect()

    size_mb = os.path.getsize(h5_path) / 1024 ** 2
    print(f"\nDone.  Written: {h5_path}  ({size_mb:.1f} MiB)")
    print(f"  source_files attribute lists the {len(root_files)} origin file(s)")
    print(f"  source_file  dataset  maps each event to its origin file index")


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def run(args):
    root_files = _collect_root_files(args.inputs, args.recursive)

    if not root_files:
        print("No ROOT files found — nothing to do.")
        return 0

    # ── Merge mode ──────────────────────────────────────────────────────
    if args.merge:
        h5_path = args.merge
        if os.path.exists(h5_path) and not args.force:
            print(f"SKIP  {h5_path} already exists  (--force to overwrite)")
            return 0
        try:
            _merge_files(root_files, h5_path, args)
        except Exception as exc:
            print(f"ERROR during merge: {exc}")
            return 1
        return 0

    # ── Per-file conversion mode ─────────────────────────────────────────
    print(f"Found {len(root_files)} ROOT file(s) to process.")
    print(f"Backend : {args.mode}")
    print(f"Force   : {args.force}")
    print(f"Dry-run : {args.dry_run}")
    print()

    n_converted = n_skipped = n_failed = 0

    for i, root_path in enumerate(root_files, 1):
        h5_path = _h5_path_for(root_path)
        prefix  = f"[{i}/{len(root_files)}]"

        if os.path.exists(h5_path) and not args.force:
            print(f"{prefix} SKIP  {root_path}")
            print(f"         → {h5_path} already exists  (--force to overwrite)")
            n_skipped += 1
            continue

        mode = args.mode if args.mode != "auto" else _detect_mode(root_path)

        action = "WOULD CONVERT" if args.dry_run else "CONVERT"
        print(f"{prefix} {action} [{mode}]  {root_path}")
        print(f"         → {h5_path}")

        if args.dry_run:
            n_converted += 1
            continue

        try:
            _SINGLE_CONVERTERS[mode](root_path, h5_path, args)
            size_mb = os.path.getsize(h5_path) / 1024 ** 2
            print(f"         done  ({size_mb:.1f} MiB)\n")
            n_converted += 1
        except Exception as exc:
            print(f"         ERROR: {exc}\n")
            n_failed += 1

    print("-" * 60)
    verb = "would be " if args.dry_run else ""
    print(f"  {n_converted} file(s) {verb}converted")
    if n_skipped:
        print(f"  {n_skipped} file(s) skipped (HDF5 already exists)")
    if n_failed:
        print(f"  {n_failed} file(s) FAILED")

    return 1 if n_failed else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "inputs", nargs="+", metavar="INPUT",
        help="ROOT file(s) or director(ies) to convert",
    )
    p.add_argument(
        "--merge", metavar="OUTPUT.h5", default=None,
        help="merge all input ROOT files into one HDF5 at OUTPUT.h5",
    )
    p.add_argument(
        "--mode", choices=["auto", "standalone", "production"], default="auto",
        help="converter backend: auto detects from ROOT layout (default: auto)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="overwrite existing HDF5 file(s)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what would be done without writing anything",
    )
    p.add_argument(
        "--recursive", "-r", action="store_true",
        help="search directories recursively for .root files",
    )

    conv = p.add_argument_group("conversion options")
    conv.add_argument(
        "--nevents", type=int, default=-1, metavar="N",
        help="max events per file (-1 = all)",
    )
    conv.add_argument(
        "--choose-ntau", type=int, default=None, metavar="N",
        help="target tau steps; None = auto (global max across all input events)",
    )
    conv.add_argument(
        "--chunk-events", type=int, default=1, metavar="N",
        help="HDF5 chunk size in events (default: 1 = finest granularity)",
    )
    conv.add_argument(
        "--verbose", action="store_true",
        help="print memory snapshots during conversion  [production single-file only]",
    )

    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
