"""
hotqcd_eos.py

Python replica of MUSIC's EOS_hotQCD (eos_id=91, the hotQCD+SMASH hadron
resonance gas table) energy-density -> temperature conversion, matching
FnoHydro::GetTemperatureFromEos bit-for-bit in algorithm (linear
interpolation on an equally-spaced energy grid, temperature tabulated as
T^5 for a smooth curve -- see X-SCAPE's external_packages/music/src/
eos_hotQCD.cpp:initialize_eos/get_temperature and eos_base.cpp:interpolate1D).

Needed because the original 4-feature training data (ed, vx, vy, vz) has
no explicit temperature channel -- temperature must be derived from energy
density the same way MUSIC/FnoHydro do, not approximated.

Usage:
    from hotqcd_eos import HotQCDEOS
    eos = HotQCDEOS()  # reads the table once
    T_GeV = eos.temperature_from_energy_density(ed_GeV_fm3)  # vectorized
"""

import numpy as np

HBARC = 0.19733  # GeV*fm, matches MUSIC's Util::hbarc exactly

DEFAULT_TABLE_PATH = (
    '/home/cparker/jetscape-docker/X-SCAPE/external_packages/music/EOS/'
    'hotQCD/hrg_hotqcd_eos_SMASH_binary.dat'
)


class HotQCDEOS:
    def __init__(self, table_path: str = DEFAULT_TABLE_PATH):
        raw = np.fromfile(table_path, dtype='<f8')
        if raw.size != 100000 * 4:
            raise ValueError(
                f"Unexpected EOS table size: {raw.size} doubles, "
                f"expected {100000*4} (100000 rows x [e,P,s,T])"
            )
        rows = raw.reshape(100000, 4)
        e_col = rows[:, 0] / HBARC  # GeV/fm^3 -> 1/fm^4
        T_col = rows[:, 3] / HBARC  # GeV -> 1/fm

        self.e0 = e_col[0]
        self.delta_e = e_col[1] - e_col[0]
        self.n = len(e_col)
        # Tabulated as T^5 for a smooth interpolated curve, matching
        # eos_hotQCD.cpp's storage convention exactly.
        self.T5_table = T_col ** 5

    def _get_temperature_fm(self, e_fm4: np.ndarray) -> np.ndarray:
        """interpolate1D + T=(T^5)^0.2, matching eos_base.cpp exactly."""
        e_fm4 = np.asarray(e_fm4, dtype=np.float64)
        idx = np.floor((e_fm4 - self.e0) / self.delta_e).astype(np.int64)
        idx = np.clip(idx, 0, self.n - 2)
        frac = (e_fm4 - (idx * self.delta_e + self.e0)) / self.delta_e
        t1 = self.T5_table[idx]
        t2 = self.T5_table[idx + 1]
        T5 = t1 * (1.0 - frac) + t2 * frac

        # Underflow (e below the table's first grid point): linear
        # extrapolation through the origin, matching interpolate1D exactly.
        underflow = e_fm4 < self.e0
        T5_underflow = self.T5_table[0] * e_fm4 / self.e0
        T5 = np.where(underflow, T5_underflow, T5)

        T5 = np.clip(T5, 0.0, None)
        return np.power(T5, 0.2)

    def temperature_from_energy_density(self, ed_gev_fm3) -> np.ndarray:
        """ed in GeV/fm^3 (scalar or array) -> temperature in GeV.

        Matches FnoHydro::GetTemperatureFromEos exactly:
            fnoEOS->get_temperature(ed/hbarc, 0) * hbarc
        """
        ed_gev_fm3 = np.asarray(ed_gev_fm3, dtype=np.float64)
        e_fm4 = ed_gev_fm3 / HBARC
        T_fm = self._get_temperature_fm(e_fm4)
        return T_fm * HBARC


if __name__ == '__main__':
    eos = HotQCDEOS()
    # Sanity check against a real MUSIC-computed ground-truth point:
    # AuAu_50ev_30_40-1.root event 0, k=0: ed=36.139904 GeV/fm^3 ->
    # MUSIC's own recorded T=0.382287 GeV (see BENCHMARKS.md).
    test_ed = 36.139904
    expected_T = 0.382287
    got_T = eos.temperature_from_energy_density(test_ed)
    print(f'ed={test_ed} GeV/fm^3 -> T={got_T:.6f} GeV (expected ~{expected_T})')
    print(f'relative error: {abs(got_T - expected_T) / expected_T * 100:.3f}%')
