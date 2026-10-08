# FNO model benchmarks (as of 2026-10-07)

Reference point before investigating model capacity. All same-initial-state
comparisons use seed=424242 at 30-40% centrality (`xmls/AuAu_50ev_30-40.xml`'s
generation seed; same seed re-used for the X-SCAPE live-test configs so the
FNO and MUSIC arms see an identical MCGlauber/MUSIC-primed initial
condition). All FNO-arm numbers are post the two normalization-bug fixes in
`js-contrib` commit `91c4177` (forward-normalize the primed IC; stop
denormalizing in place in `BuildNextWindowIC`).

## Training runs

| Run | Events | tauslice | n_epochs | Final eval l2 | Peak CUDA | Total time | Checkpoint |
|---|---|---|---|---|---|---|---|
| A | 200 (0-10/10-20/20-30/30-40, 50 each) | 100 | 30 | 0.969 (epoch 28) | 21.15 GB | 3488s | superseded |
| B | 400 (100 each) | 70 | 30 | **0.627** (epoch 28) | 17.44 GB | 5627s | **current best by training loss** |
| C (capacity: n_modes tau 16->32, 570M params vs B's 302M) | 400 (100 each) | 70 | 30 | 0.638 (epoch 28) | 21.44 GB | 6526s | superseded -- no improvement |

Run B's eval-loss curve is also qualitatively cleaner (smooth decrease,
1.24->0.627) vs run A's noisier one -- more data clearly helps the training
metric. See `epoch_scan_50ep/README.md` for the separate finding that ~30
epochs is the overfitting onset at this data scale (established before run
B's data expansion; not re-verified at 400 events).

## Same-IC fidelity vs real MUSIC (seed=424242, 30-40% centrality)

Three FNO versions compared against one real full-dynamical MUSIC run
(`music_cen30_40_event0.csv` / `fno_vs_music_cen3040/music_peak_temperature.csv`,
froze out at tau=8.76 fm/c):

| FNO version | Peak T (tau~0.5) | Mid-evolution (tau 2-6) | Freeze-out crossing | Post-freeze-out artifact |
|---|---|---|---|---|
| Pre-fix (buggy input norm) | 494 MeV (MUSIC: 381) -- **over**-predicts | stays ~30-100+ MeV hotter than MUSIC throughout | never (still ~168 MeV at window end) | none observed (never gets there) |
| Run A (200 ev, tau=100) | 346 MeV -- close | runs ~20-40 MeV **cooler** than MUSIC | tau~6.8 (MUSIC: 8.76) -- early | dips to ~65-75 MeV then partially recovers, tau~9 |
| Run B (400 ev, tau=70) | 341 MeV -- close | **same under-prediction as run A**, no improvement | tau~6.7 (window ends tau~7.3, too short to compare tail) | smaller dip (~95-100 MeV), earlier, shorter window to observe it in |
| Run C (capacity: tau-modes=32) | 310 MeV -- close | **tracks run B almost exactly, no improvement** | tau~6.7, same as run B | same pattern as run B, no change |

**Headline finding: expanding training data 2x and shrinking tauslice from
100->70 did not close the mid-evolution cooling-rate gap, and neither did
nearly doubling model capacity (run C, 570M vs 302M params, tau-axis
Fourier modes 16->32) targeted specifically at the temporal-resolution
hypothesis.** Runs A, B, and C all track each other almost exactly relative
to MUSIC -- run C's eval l2 (0.638) wasn't even measurably better than run
B's (0.627). This is a fairly decisive negative result: the mid-evolution
gap is not a capacity bottleneck in this dimension. Whatever's driving it
is more likely the loss function's behavior in this regime, the
normalization scheme's structure, a systematic bias in the training data
itself, or a genuine limitation of this model class/training setup rather
than something more data or more parameters fixes.

Plots: `fno_vs_music_cen3040/fno_vs_music_cen3040_seed424242.png` (buggy
norm, superseded by the finding below -- kept for the historical
over-prediction record), `fno_vs_music_cen3040/fno_vs_music_cen3040_seed424242_v2.png`
(fixed norm, run A), `fno_vs_music_cen3040/fno_vs_music_cen3040_seed424242_v3.png`
(run A vs run B vs MUSIC, 3-way), `fno_vs_music_cen3040/fno_vs_music_cen3040_seed424242_v4.png`
(run A vs run B vs run C vs MUSIC, 4-way).

## Post-fix 20-event centrality sweep (run A checkpoint, before run B existed)

5 events each at 0-10/10-20/20-30/30-40% centrality, fixed-normalization
model (run A). All 20 reached freeze-out within the native window --
confirms the normalization fix wasn't a one-centrality fluke. Final
temperatures at the window's last step (tau~10.34 fm/c):

| Centrality | Final T range |
|---|---|
| 0-10% | 100.6 - 115.7 MeV |
| 10-20% | 81.8 - 103.8 MeV |
| 20-30% | 82.0 - 86.0 MeV |
| 30-40% | 68.1 - 74.5 MeV |

Physically sensible ordering (hotter/most-central finishes closest to
threshold). Every one of the 20 curves showed the same post-freeze-out
dip-and-recover artifact (likely Gibbs-ringing from the hard zero-padding
edge in training targets past each event's own freeze-out -- see
conversation history for the ed/T co-variation check that ruled out the
EOS conversion as the source). Data: `centrality_sweep_fixed_norm/`,
overlay plot `centrality_sweep_fixed_norm/sweep_all_centralities_vs_music.png`.

## Training data (as of run B)

100 clean events per centrality bin (400 total), validated via the
`ntau_freezeout` filter (`min_ntau_freezeout: 10` in config.yaml, added to
`train_torch.py` after discovering MUSIC/ROOT-writer crashes left blank
events in the first generation attempt -- see `js-contrib`/X-SCAPE commit
history around Oct 6-7 2026 for the ROOT `TFile::Flush()` fix):
- `AuAu_100ev_0_10-1.root` (100 events)
- `AuAu_50ev_10_20-1.root` + `-2` (100 events)
- `AuAu_50ev_20_30-1.root` + `-2` (100 events)
- `AuAu_50ev_30_40-1.root` + `-2` (100 events)

## Grid-resolution hypothesis (tested, refuted)

`RootBulkWriter`'s configured output grid (dx=0.3125, nx=65) is coarser
than MUSIC's native solver grid (dx=0.20202, nx=99) -- a real rebinning
step the training ground truth goes through that the live
`WriteMusicPeakTemperatureDiagnostics` comparison (reads `bulk_info`
directly, no rebinning) doesn't. Initial single-event check looked like it
confirmed this: training event 0 of `AuAu_50ev_30_40-1.root` (coarse grid)
showed ~5-15% lower max-ed than the live seed=424242 MUSIC comparison at
every matched tau, and its peak at tau=0.44 (36.14) was close to the live
comparison's (36.52), suggesting they were the same underlying event.

Generated 20 new events at MUSIC's exact native resolution to test this
properly (`AuAu_20ev_30_40_finegrid.root`, nx=ny=99, dx=0.20202 --
`RootBulkWriter`'s `x_min=0`/`dx=0` auto-match XML option resolves to the
wrong thing, the externally-configured IS/MCGlauber grid, not MUSIC's true
solver grid, so the native dx had to be hardcoded instead). Averaged
properly across all 20 fine-grid events vs all 50 coarse-grid events:
peak energy density is statistically the same (51.4 vs 50.9 GeV/fm^3, ~1%
apart) and the normalized (relative-to-own-peak) decay shape is nearly
indistinguishable at every tau checked. **Refuted**: rebinning doesn't
meaningfully touch peak height or decay rate. The original single-event
"confirmation" was almost certainly coincidental similarity between two
different events (same centrality bin, different nucleon configuration),
not the same event, given peak values vary by several percent from natural
event-to-event fluctuation alone.

Data kept for reference: `AuAu_20ev_30_40_finegrid.root` in `rootfiles/`.

## Open issue carried forward

Mid-evolution (tau 2-6 fm/c) systematic under-prediction of ~20-40 MeV
relative to real MUSIC on an identical initial condition (true matched-seed
comparison, not a coincidental one), consistent across data scale (runs
A/B), model capacity (run C), and now also grid resolution -- all four
ruled out. Remaining candidates: loss function behavior in this specific
regime, the normalization scheme's own structure, or some other mismatch
between the training data pipeline and the live inference pipeline not yet
identified.
