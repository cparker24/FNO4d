#!/usr/bin/env python
"""Batch-build fixed-IC FNO and ground-truth-passthrough TorchScript modules
for the full 20-event, 4-centrality matched-initial-condition particlization
comparison (old-FNO vs. doubled-FNO vs. MUSIC, see
PARTICLIZATION_INVESTIGATION.md's "Redone with a matched initial condition"
section for the single-centrality/5-event precedent this generalizes).

Unlike calling build_fixed_ic_fno_model.py / build_ground_truth_passthrough_model.py
once per event (each call re-reads events 0..N from the ROOT file via
read_3d_data's nevents=event+1 convention, an O(n^2) re-read over 20 events --
and for the ground-truth tensor, which needs choose_ntau=120 instead of 1,
that re-read is ~0.5GB/event), this script reads each centrality's ROOT file
exactly ONCE for the fixed-IC case and ONCE for the ground-truth case (all 20
events in one read_3d_data call), then slices per-event tensors out of the
in-memory array. It also loads each FNO checkpoint's network weights exactly
ONCE (not once per centrality/event) since the trained network itself doesn't
depend on centrality.

Both model checkpoints here (checkpoints_lambda_symgrid_eta4_n65_notemp_extended_90split_etamodes32
and checkpoints_lambda_symgrid_doubled) were trained with drop_channels: [1]
(temperature dropped, see their config.yaml) on the symmetric grid (no [:-1]
edge-bin crop -- RootBulkWriter.cc commit da3ea4cc), so every read here uses
drop_channels=[1], crop_edge_bin=False, channels=4 -- unlike the original
5-channel, cropped-grid precedent this generalizes.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "loc_libs"))

from read_3d_root import read_3d_data  # noqa: E402
from build_fixed_ic_fno_model import build_model, load_state_dict, FixedInputModel  # noqa: E402
from build_ground_truth_passthrough_model import ConstantOutput  # noqa: E402

ROOTFILES_DIR = Path("/home/cparker/jetscape-docker/rootfiles")
MODELS_DIR = Path(
    "/home/cparker/jetscape-docker/X-SCAPE/external_packages/js-contrib/contribs/FnoHydro/models/matched_ic"
)
CHECKPOINTS = {
    "old": Path("/home/cparker/jetscape-docker/FNO4d/train_torch/"
                "checkpoints_lambda_symgrid_eta4_n65_notemp_extended_90split_etamodes32"),
    "doubled": Path("/home/cparker/jetscape-docker/FNO4d/train_torch/checkpoints_lambda_symgrid_doubled"),
}
CENTRALITIES = ["0_10", "10_20", "20_30", "30_40"]
N_EVENTS = 20
CHANNELS = 4
DROP_CHANNELS = [1]
NTAU_FNO = 119
TAU0 = 0.44
DTAU = 0.1
NX = NY = NETA = 64
ATOL = 1e-5


def build_fixed_input_batch(root_file):
    """One read_3d_data call for all N_EVENTS events' tau-index-0 IC."""
    arr_data = read_3d_data(
        root_file, nevents=N_EVENTS, choose_ntau=1,
        xyslice=None, etaslice=None, tauslice=slice(None, 1),
        min_ntau_freezeout=None, drop_channels=DROP_CHANNELS,
    )
    arr = arr_data["arr"]  # (N_EVENTS, 4, nx, ny, neta, ntau=1)
    assert arr.shape[2:5] == (NX, NY, NETA), f"unexpected spatial shape {arr.shape}"
    assert abs(arr_data["tau_min"] - TAU0) < 1e-3, (
        f"tau0 {TAU0} doesn't match this ROOT file's tau_min {arr_data['tau_min']}"
    )
    out = []
    for event in range(N_EVENTS):
        ic = arr[event, :, :, :, :, 0].copy()  # (4, nx, ny, neta)
        ed = np.clip(ic[0], 0.0, None)
        ic[0] = ed ** 0.75 * TAU0
        ic_t = torch.from_numpy(ic).float()
        repeated = ic_t.unsqueeze(-1).repeat(1, 1, 1, 1, NTAU_FNO)
        out.append(repeated.unsqueeze(0))  # (1, 4, nx, ny, neta, ntau_fno)
    return out


def build_ground_truth_batch(root_file):
    """One read_3d_data call for all N_EVENTS events' full (tau0..tau0+119*dtau) evolution."""
    choose_ntau = NTAU_FNO + 1
    arr_data = read_3d_data(
        root_file, nevents=N_EVENTS, choose_ntau=choose_ntau,
        xyslice=None, etaslice=None, tauslice=slice(None, choose_ntau),
        min_ntau_freezeout=None, drop_channels=DROP_CHANNELS,
    )
    arr = arr_data["arr"]  # (N_EVENTS, 4, nx, ny, neta, choose_ntau)
    assert arr.shape[2:5] == (NX, NY, NETA), f"unexpected spatial shape {arr.shape}"
    assert abs(arr_data["tau_min"] - TAU0) < 1e-3, (
        f"tau0 {TAU0} doesn't match this ROOT file's tau_min {arr_data['tau_min']}"
    )
    assert abs(arr_data["dtau"] - DTAU) < 1e-6, (
        f"dtau {DTAU} doesn't match this ROOT file's dtau {arr_data['dtau']}"
    )
    tau_phys = TAU0 + np.arange(1, NTAU_FNO + 1) * DTAU
    out = []
    for event in range(N_EVENTS):
        raw = arr[event]  # (4, nx, ny, neta, choose_ntau)
        o = raw[:, :, :, :, 1:NTAU_FNO + 1].copy()  # (4, nx, ny, neta, ntau_fno)
        ed = np.clip(o[0], 0.0, None)
        o[0] = ed ** 0.75 * tau_phys.reshape(1, 1, 1, NTAU_FNO)
        print(f"  event {event}: ntau_freezeout={arr_data['ntau_freezeout'][event]}, "
              f"ed range [{o[0].min():.4g}, {o[0].max():.4g}]")
        out.append(torch.from_numpy(o).float().unsqueeze(0))  # (1, 4, nx, ny, neta, ntau_fno)
    return out


def trace_and_check_fixed_ic(wrapper):
    """Full verification (eager-vs-traced match + ignores-input check): ~4x
    the cost of a single forward pass through the real 302M-param FNO
    network (~300s each on this host's CPU). Used once per arm (see
    FULL_CHECK_ARMS in main()) -- the network's traced graph structure
    cannot depend on the embedded fixed_input's VALUES (no data-dependent
    control flow in an FNO: fixed sequence of spectral conv / MLP layers),
    only on its shape, which is identical across all centralities/events --
    so one verified build per arm is sufficient, not one per event."""
    example = torch.randn(1, CHANNELS, NX, NY, NETA, NTAU_FNO)
    with torch.no_grad():
        eager_out = wrapper(example)
        traced = torch.jit.trace(wrapper, example)
        traced_out = traced(example)
    max_diff = (eager_out - traced_out).abs().max().item()
    if max_diff > ATOL:
        raise RuntimeError(f"Traced module diverges from eager by {max_diff:.3e} > atol={ATOL}")
    other = torch.zeros_like(example)
    other_out = traced(other)
    diff_vs_other = (other_out - traced_out).abs().max().item()
    assert diff_vs_other == 0.0, "fixed-IC wrapper depends on input -- not a true fixed-IC wrapper!"
    return traced


def trace_fixed_ic_fast(wrapper):
    """Single torch.jit.trace call, no redundant eager/ignores-input
    verification passes (already confirmed once per arm by
    trace_and_check_fixed_ic) -- cuts ~4 forward passes down to the 1
    torch.jit.trace() already has to do internally."""
    example = torch.randn(1, CHANNELS, NX, NY, NETA, NTAU_FNO)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
    return traced


def trace_and_check_ground_truth(model):
    example = torch.randn(1, CHANNELS, NX, NY, NETA, NTAU_FNO)
    with torch.no_grad():
        eager_out = model(example)
        traced = torch.jit.trace(model, example)
        traced_out = traced(example)
    max_diff = (eager_out - traced_out).abs().max().item()
    assert max_diff == 0.0, "ground-truth passthrough should be an EXACT constant function"
    other = torch.zeros_like(example)
    other_out = traced(other)
    assert torch.equal(other_out, traced_out), "ground-truth module depends on input -- not a true passthrough!"
    return traced


def load_fno_network(checkpoint_dir):
    cfg = yaml.safe_load((checkpoint_dir / "config.yaml").read_text())
    print(f"  {checkpoint_dir.name}: n_modes={cfg['n_modes']} hidden_channels={cfg['hidden_channels']}")
    model = build_model(cfg, CHANNELS, CHANNELS)
    model.load_state_dict(load_state_dict(checkpoint_dir / "best_model_state_dict.pt", torch.device("cpu")))
    model.eval()
    return model


FULL_CHECK_DONE = set()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FNO networks (once each, reused across all centralities/events)...")
    networks = {arm: load_fno_network(path) for arm, path in CHECKPOINTS.items()}

    for cen in CENTRALITIES:
        root_file = ROOTFILES_DIR / f"AuAu_valid20ev_{cen}_eta4_n65_symgrid.root"
        assert root_file.exists(), f"missing validation ROOT file: {root_file}"

        print(f"\n=== {cen}: reading fixed-IC tensors (all {N_EVENTS} events, one read) ===")
        fixed_inputs = build_fixed_input_batch(str(root_file))

        print(f"=== {cen}: reading ground-truth tensors (all {N_EVENTS} events, one read) ===")
        ground_truths = build_ground_truth_batch(str(root_file))

        for event in range(N_EVENTS):
            for arm, net in networks.items():
                wrapper = FixedInputModel(net, fixed_inputs[event])
                wrapper.eval()
                checked_key = (cen, arm)
                if checked_key not in FULL_CHECK_DONE:
                    traced = trace_and_check_fixed_ic(wrapper)
                    FULL_CHECK_DONE.add(checked_key)
                    print(f"  {cen}/{arm}: full eager-vs-traced + ignores-input check passed (event {event})")
                else:
                    traced = trace_fixed_ic_fast(wrapper)
                out_path = MODELS_DIR / f"fixed_ic_{arm}_symgrid_{cen}_event{event}.pt"
                traced.save(str(out_path))

            gt_model = ConstantOutput(ground_truths[event])
            gt_model.eval()
            traced_gt = trace_and_check_ground_truth(gt_model)
            gt_path = MODELS_DIR / f"ground_truth_passthrough_symgrid_{cen}_event{event}.pt"
            traced_gt.save(str(gt_path))

            print(f"  {cen} event {event}: saved old/doubled fixed-IC + ground-truth passthrough modules")

    print("\nAll matched-IC model files built.")


if __name__ == "__main__":
    main()
