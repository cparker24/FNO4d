"""
build_avg_temperature_profiles.py

Builds average (mean + std) peak-temperature-vs-tau profiles per centrality
bin from the training data files themselves (ground truth, not model
predictions), for use as reference curves in comparison plots -- a
statistically robust alternative to the single-event matched-seed
comparisons used throughout BENCHMARKS.md.

The original 4-feature files (ed, vx, vy, vz) have no explicit temperature
channel, so temperature is derived from energy density via the same hotQCD
EOS (id 91) MUSIC/FnoHydro use (loc_libs/hotqcd_eos.py), not approximated.

Averaging handles each event's own freeze-out time correctly: tau steps
past an event's own ntau_freezeout are zero-padding in the stored array,
not physical zeros, and are excluded from that event's contribution to the
average at that tau (so the per-tau mean is always an average over events
still "alive" at that tau, with the contributing count recorded alongside).

Usage:
    python build_avg_temperature_profiles.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'loc_libs'))
from read_3d_root import read_3d_data
from hotqcd_eos import HotQCDEOS

INPUT_DIR = Path('/home/cparker/jetscape-docker/rootfiles')
OUT_DIR = Path(__file__).resolve().parent / 'out' / 'avg_temperature_profiles'

# One file per centrality -- the "restored original setup" dataset
# (4-feature, implicit temperature, original coarse grid, 50 events each).
CENTRALITY_FILES = {
    '0-10%':  'AuAu_50ev_0_10-1.root',
    '10-20%': 'AuAu_50ev_10_20-1.root',
    '20-30%': 'AuAu_50ev_20_30-1.root',
    '30-40%': 'AuAu_50ev_30_40-1.root',
}


def build_profile(file_path: Path, eos: HotQCDEOS):
    """Returns dict with tau (ntau,), mean_T/std_T/n_events (ntau,) arrays,
    plus mean_ed/std_ed for the energy-density channel alongside."""
    data = read_3d_data(str(file_path), nevents=-1, local_rank=None)
    arr = data['arr'][:, :, :-1, :-1, :-1, :]  # trim last x/y/eta bin, matching
    # train_torch.py's xyslice=[None,-1]/etaslice=[None,-1] -> FNO's 64x64x16 grid
    ntau_freezeout = data['ntau_freezeout']
    tau_min = data['tau_min']
    dtau = data['dtau']

    nevents, nfeat, nx, ny, neta, ntau = arr.shape
    tau_axis = np.arange(ntau) * dtau + tau_min

    # max over (x, y, eta) of channel 0 (energy density) at every tau,
    # for every event -- cheap, and this is the convention used
    # throughout this investigation's live-test diagnostics.
    max_ed = arr[:, 0].max(axis=(1, 2, 3))  # (nevents, ntau)
    max_T = eos.temperature_from_energy_density(max_ed)  # (nevents, ntau)

    # Mask out each event's own zero-padded tail (tau steps at or past
    # its own ntau_freezeout are padding, not physical zeros).
    valid = np.arange(ntau)[None, :] < ntau_freezeout[:, None]  # (nevents, ntau)

    n_contrib = valid.sum(axis=0)  # (ntau,)
    with np.errstate(invalid='ignore'):
        mean_T = np.where(n_contrib > 0,
                          np.sum(np.where(valid, max_T, 0.0), axis=0) / np.maximum(n_contrib, 1),
                          np.nan)
        mean_ed = np.where(n_contrib > 0,
                           np.sum(np.where(valid, max_ed, 0.0), axis=0) / np.maximum(n_contrib, 1),
                           np.nan)
        # std via E[x^2] - E[x]^2, masked the same way
        mean_T2 = np.where(n_contrib > 0,
                           np.sum(np.where(valid, max_T**2, 0.0), axis=0) / np.maximum(n_contrib, 1),
                           np.nan)
        std_T = np.sqrt(np.clip(mean_T2 - mean_T**2, 0, None))

    return {
        'tau': tau_axis,
        'mean_T': mean_T,
        'std_T': std_T,
        'mean_ed': mean_ed,
        'n_events': n_contrib,
        'n_total_events': nevents,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eos = HotQCDEOS()

    all_profiles = {}
    for centrality, fname in CENTRALITY_FILES.items():
        fpath = INPUT_DIR / fname
        print(f'Processing {centrality}: {fname} ...')
        profile = build_profile(fpath, eos)
        all_profiles[centrality] = profile

        safe_name = centrality.replace('%', 'pct').replace('-', '_')
        csv_path = OUT_DIR / f'avg_temperature_{safe_name}.csv'
        with open(csv_path, 'w') as f:
            f.write('tau_fm,mean_max_T_GeV,std_max_T_GeV,mean_max_ed_GeV_fm3,n_events_contributing\n')
            for i in range(len(profile['tau'])):
                f.write(f"{profile['tau'][i]:.4f},{profile['mean_T'][i]:.6f},"
                       f"{profile['std_T'][i]:.6f},{profile['mean_ed'][i]:.6f},"
                       f"{profile['n_events'][i]}\n")
        print(f'  -> {csv_path}  ({profile["n_total_events"]} events, '
              f'tau range {profile["tau"][0]:.2f}-{profile["tau"][-1]:.2f} fm/c)')

    # Quick summary printed to console
    print()
    print(f'{"centrality":>10} {"tau=1":>10} {"tau=2":>10} {"tau=3":>10} {"tau=4":>10} {"tau=5":>10} {"tau=6":>10}')
    for centrality, profile in all_profiles.items():
        tau = profile['tau']
        row = [centrality]
        for target in [1, 2, 3, 4, 5, 6]:
            idx = np.argmin(np.abs(tau - target))
            row.append(f'{profile["mean_T"][idx]*1000:.1f}')
        print(f'{row[0]:>10} ' + ' '.join(f'{v:>10}' for v in row[1:]))


if __name__ == '__main__':
    main()
