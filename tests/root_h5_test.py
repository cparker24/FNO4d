#!/usr/bin/env python

# ---------------------------------------------------------------------------
# Verify that each .h5 file produced by convert_root_to_hdf5() stores exactly
# the same data as its source .root file.
#
# Two test modes (both run by default when multiple pairs are present):
#
#   per-file  : each (root, h5) pair is loaded and compared independently.
#   combined  : all ROOT files are concatenated via read_3d_data() and all
#               H5 files are concatenated via MultiH5Array / read_3d_data_hdf5(),
#               then compared as one dataset.  This catches ordering or offset
#               bugs that only appear when files are chained together.
#
# Usage:
#   python root_h5_test.py [--data-dir ../data] [--nevents -1] [--verbose]
#   python root_h5_test.py --no-per-file      # combined only
#   python root_h5_test.py --no-combined      # per-file only
#
# Pairs are auto-detected: every .root file whose stem has a matching .h5
# file in the same directory is tested.  Pass explicit pairs with --pairs:
#   python root_h5_test.py --pairs file.root file.h5 [file2.root file2.h5 ...]
# ---------------------------------------------------------------------------

import argparse
import sys
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description='Compare ROOT vs HDF5 data arrays')
parser.add_argument('--data-dir', default='../data', metavar='DIR',
                    help='directory containing .root and .h5 files (default: ../data)')
parser.add_argument('--nevents', type=int, default=-1,
                    help='events to load per file (-1 = all, default: -1)')
parser.add_argument('--pairs', nargs='+', metavar='FILE',
                    help='explicit root/h5 pairs: root1 h5_1 root2 h5_2 ...')
parser.add_argument('--no-per-file', action='store_true',
                    help='skip per-file comparison')
parser.add_argument('--no-combined', action='store_true',
                    help='skip combined multi-file comparison')
parser.add_argument('--verbose', action='store_true',
                    help='print per-event differences')
args = parser.parse_args()

sys.path.insert(0, '../loc_libs')
from read_3d_root import read_3d_data
from read_3d_hdf5 import read_3d_data_hdf5

# ---------------------------------------------------------------------------
# resolve file pairs
# ---------------------------------------------------------------------------
if args.pairs:
    if len(args.pairs) % 2 != 0:
        parser.error('--pairs requires an even number of arguments (root h5 root h5 ...)')
    pairs = [(Path(args.pairs[i]), Path(args.pairs[i + 1]))
             for i in range(0, len(args.pairs), 2)]
else:
    data_dir = Path(args.data_dir)
    root_files = sorted(data_dir.glob('*.root'))
    pairs = []
    for rf in root_files:
        hf = rf.with_suffix('.h5')
        if hf.exists():
            pairs.append((rf, hf))
        else:
            print(f'[SKIP] no matching .h5 for {rf.name}')

if not pairs:
    print('No (root, h5) pairs found. Exiting.')
    sys.exit(1)

print(f'Testing {len(pairs)} pair(s)\n')

# ---------------------------------------------------------------------------
# scalar grid keys to compare
# ---------------------------------------------------------------------------
_GRID_KEYS = ('nFeatures', 'nx', 'ny', 'neta', 'tau_min', 'dtau',
              'x_min', 'y_min', 'eta_min', 'dx', 'dy', 'deta')

# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def _fmt(val):
    return f'{val:.6g}' if isinstance(val, float) else str(val)


def compare_scalars(root_dat, h5_dat, label=''):
    failures = []
    for k in _GRID_KEYS:
        rv = root_dat.get(k)
        hv = h5_dat.get(k)
        if rv is None or hv is None:
            continue
        rv, hv = float(rv), float(hv)
        if not np.isclose(rv, hv, rtol=1e-6, atol=0):
            failures.append(f'  {k}: ROOT={_fmt(rv)}  H5={_fmt(hv)}  diff={abs(rv-hv):.3e}')
    if failures:
        print(f'[FAIL] {label} — grid parameter mismatch:')
        for f in failures:
            print(f)
        return False
    print(f'[OK]   {label} — grid parameters match')
    return True


def compare_1d(root_arr, h5_arr, name, label='', atol=1e-6):
    n = min(len(root_arr), len(h5_arr))
    root_arr, h5_arr = np.asarray(root_arr[:n]), np.asarray(h5_arr[:n])
    if np.array_equal(root_arr, h5_arr):
        print(f'[OK]   {label} — {name} exactly equal  (n={n})')
        return True
    max_diff = float(np.max(np.abs(root_arr - h5_arr)))
    if np.allclose(root_arr, h5_arr, rtol=0, atol=atol):
        print(f'[OK]   {label} — {name} within atol={atol}  max_diff={max_diff:.3e}  (n={n})')
        return True
    bad = np.where(~np.isclose(root_arr, h5_arr, rtol=0, atol=atol))[0]
    print(f'[FAIL] {label} — {name} mismatch  max_diff={max_diff:.3e}  n_bad={len(bad)}')
    for i in bad[:5]:
        print(f'       event {i}: ROOT={root_arr[i]:.6g}  H5={h5_arr[i]:.6g}')
    return False


def compare_arrays(arr_root, arr_h5, ntau_fo, label='', verbose=False):
    '''Compare arr event by event, clipping to the active (non-padded) tau range.'''
    n_events = min(arr_root.shape[0], arr_h5.shape[0])
    ntau_min = min(arr_root.shape[-1], arr_h5.shape[-1])

    if arr_root.shape[1:] != arr_h5.shape[1:]:
        # shapes differ only in ntau — clip to common length and warn
        if arr_root.shape[1:-1] == arr_h5.shape[1:-1]:
            print(f'  [NOTE] ntau differs: ROOT={arr_root.shape[-1]}  H5={arr_h5.shape[-1]}'
                  f' — comparing first {ntau_min} tau steps')
        else:
            print(f'[FAIL] {label} — incompatible shapes:'
                  f' ROOT={arr_root.shape}  H5={arr_h5.shape}')
            return False

    r = arr_root[:n_events, ..., :ntau_min].astype(np.float32)
    h = arr_h5[:n_events,   ..., :ntau_min].astype(np.float32)

    n_exact   = 0
    n_close   = 0
    n_fail    = 0
    max_diffs = []

    for i in range(n_events):
        # active tau range for this event
        tau_end = int(ntau_fo[i]) if ntau_fo is not None and i < len(ntau_fo) else ntau_min
        tau_end = min(tau_end, ntau_min)

        ri = r[i, ..., :tau_end]
        hi = h[i, ..., :tau_end]

        if np.array_equal(ri, hi):
            n_exact += 1
            max_diffs.append(0.0)
        elif np.allclose(ri, hi, rtol=0, atol=1e-6):
            n_close += 1
            max_diffs.append(float(np.max(np.abs(ri - hi))))
            if verbose:
                print(f'       event {i}: near-equal  max_diff={max_diffs[-1]:.3e}')
        else:
            n_fail += 1
            md = float(np.max(np.abs(ri - hi)))
            max_diffs.append(md)
            if verbose or n_fail <= 3:
                bad_idx = np.unravel_index(np.argmax(np.abs(ri - hi)), ri.shape)
                print(f'       event {i}: MISMATCH  max_diff={md:.3e}'
                      f'  worst index={bad_idx}'
                      f'  ROOT={float(ri[bad_idx]):.6g}  H5={float(hi[bad_idx]):.6g}')

    global_max = max(max_diffs) if max_diffs else 0.0

    if n_fail == 0:
        tag = '[OK]  '
        detail = (f'exact={n_exact}  near-equal={n_close}'
                  f'  global_max_diff={global_max:.3e}')
    else:
        tag = '[FAIL]'
        detail = (f'exact={n_exact}  near-equal={n_close}  FAIL={n_fail}'
                  f'  global_max_diff={global_max:.3e}')

    print(f'{tag} {label} — arr comparison  {detail}  (events={n_events})')
    return n_fail == 0


# ---------------------------------------------------------------------------
# main loop — per-file
# ---------------------------------------------------------------------------
results = []

if not args.no_per_file:
    for root_path, h5_path in pairs:
        label = root_path.stem
        print(f'─── {label} '.ljust(72, '─'))
        print(f'  ROOT: {root_path}')
        print(f'  H5:   {h5_path}')

        # ---- load ROOT --------------------------------------------------------
        print('\nLoading ROOT ...')
        try:
            root_dat = read_3d_data(str(root_path), nevents=args.nevents, local_rank=None)
        except Exception as e:
            print(f'[ERROR] ROOT load failed: {e}')
            results.append((label, False))
            continue

        # ---- load HDF5 --------------------------------------------------------
        print('\nLoading HDF5 ...')
        try:
            h5_dat = read_3d_data_hdf5(str(h5_path), nevents=args.nevents, lazy=False)
        except Exception as e:
            print(f'[ERROR] HDF5 load failed: {e}')
            results.append((label, False))
            continue

        print()

        # ---- compare ----------------------------------------------------------
        ok_scalars = compare_scalars(root_dat, h5_dat, label)

        ok_ntau = compare_1d(root_dat['ntau_freezeout'], h5_dat['ntau_freezeout'],
                             'ntau_freezeout', label)
        ok_tau  = compare_1d(root_dat['tau_freezeout'],  h5_dat['tau_freezeout'],
                             'tau_freezeout',  label, atol=1e-5)

        ntau_fo = root_dat.get('ntau_freezeout')
        ok_arr  = compare_arrays(root_dat['arr'], h5_dat['arr'],
                                 ntau_fo, label, verbose=args.verbose)

        pair_ok = all([ok_scalars, ok_ntau, ok_tau, ok_arr])
        results.append((label, pair_ok))
        print()

# ---------------------------------------------------------------------------
# combined multi-file test
# ---------------------------------------------------------------------------
if not args.no_combined and len(pairs) > 1:
    label = 'COMBINED (' + ' + '.join(p[0].stem for p in pairs) + ')'
    print(f'─── combined '.ljust(72, '─'))
    print(f'  ROOT files: {[str(p[0]) for p in pairs]}')
    print(f'  H5 files:   {[str(p[1]) for p in pairs]}')

    root_paths = [str(p[0]) for p in pairs]
    h5_paths   = [str(p[1]) for p in pairs]

    # ---- load combined ROOT -----------------------------------------------
    print('\nLoading combined ROOT ...')
    try:
        root_dat = read_3d_data(root_paths, nevents=args.nevents, local_rank=None)
    except Exception as e:
        print(f'[ERROR] combined ROOT load failed: {e}')
        results.append((label, False))
        root_dat = None

    # ---- load combined HDF5 (eager so we get a plain ndarray) -------------
    if root_dat is not None:
        print('\nLoading combined HDF5 ...')
        try:
            h5_dat = read_3d_data_hdf5(h5_paths, nevents=args.nevents, lazy=False)
        except Exception as e:
            print(f'[ERROR] combined HDF5 load failed: {e}')
            results.append((label, False))
            h5_dat = None

        if h5_dat is not None:
            print()
            ok_scalars = compare_scalars(root_dat, h5_dat, label)

            ok_ntau = compare_1d(root_dat['ntau_freezeout'], h5_dat['ntau_freezeout'],
                                 'ntau_freezeout', label)
            ok_tau  = compare_1d(root_dat['tau_freezeout'],  h5_dat['tau_freezeout'],
                                 'tau_freezeout',  label, atol=1e-5)

            ntau_fo = root_dat.get('ntau_freezeout')
            ok_arr  = compare_arrays(root_dat['arr'], h5_dat['arr'],
                                     ntau_fo, label, verbose=args.verbose)

            results.append((label, all([ok_scalars, ok_ntau, ok_tau, ok_arr])))
            print()

elif not args.no_combined and len(pairs) == 1:
    print('[NOTE] only one pair — skipping combined test (needs ≥ 2 pairs)\n')

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
print('═' * 72)
print('SUMMARY')
print('─' * 72)
n_pass = sum(1 for _, ok in results if ok)
for label, ok in results:
    print(f'  {"PASS" if ok else "FAIL"}  {label}')
print('─' * 72)
print(f'  {n_pass}/{len(results)} pair(s) passed')
print('═' * 72)

sys.exit(0 if n_pass == len(results) else 1)
