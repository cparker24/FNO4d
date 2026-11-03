#!/usr/bin/env python
"""Build a "ground-truth passthrough" TorchScript module for FnoHydro: a
fake FNO model that ignores its input entirely and always returns the real
MUSIC evolution for one specific training event, taken directly from a
training ROOT file, downsampled/rebinned onto exactly the same (nx, ny,
neta, dtau, deta) grid an FNO model trains and predicts on.

Why: isolates whether FNO's yield deficit vs. MUSIC (documented throughout
PARTICLIZATION_INVESTIGATION.md) comes from the model's own prediction
error, or simply from running Cooper-Frye particlization on the coarser
FNO-sized grid instead of MUSIC's native grid (dx=dy=0.3fm/67x67,
deta=0.2/60 pts, dtau=0.02fm/c -- see "How fine of a grid does MUSIC use?"
in that doc) -- a pure resolution effect that has nothing to do with the
neural network at all. Running this through the exact same
FnoHydro.cc -> surface-finder -> iSS pipeline as every other comparison in
this investigation gives a clean A/B: same downstream code, only the
"model" differs (real network vs. an oracle that already knows the right
answer at this resolution).

Physical content: this is genuine MUSIC output, not a model prediction --
RootBulkWriter.cc writes (energy_density, temperature, vx, vy, vz) by
querying MUSIC's own hydro-info interpolator at exactly the FNO training
grid's (x, y, eta, tau) points, i.e. the grid is coarser but the physics
at each sampled point is MUSIC's real answer, not FNO's guess.

Normalisation: FnoHydro.cc's FNO output convention is "input = repeated
IC, output slot j (0-indexed, j=0..ntau_fno-1) = predicted state at
tau_phys = tau0 + (j+1)*dtau" (see FnoHydro.cc's DenormalizeEnergyDensity
call sites, e.g. around PassHydroEvolutionHistoryToFramework: tau_phys =
bulk_info.Tau0() + bulk_info.dtau*k_local, with k_local the GLOBAL tau
index after the IC is prepended at index 0). This exactly matches
NormFn3d.etau75_3d's own convention (tau = tau_min + k*dtau, tau_min ==
tau0, k the array's raw tau index) -- so output slot j pulls RAW tau index
k=j+1 from the ROOT file, and only channel 0 (energy density) gets the
etau75_3d forward transform (ed^0.75 * tau_phys); temperature/vx/vy/vz
(channels 1-4) are untouched, matching FnoHydro.cc's NormalizeEnergyDensity
being called on channel 0 only. See "Charge lookup bug" and "Three-way
comparison" sections of PARTICLIZATION_INVESTIGATION.md for the broader
pattern of carefully matching this codebase's existing conventions exactly
rather than re-deriving them.

Usage:
    python build_ground_truth_passthrough_model.py \\
        --root-file /home/cparker/jetscape-docker/rootfiles/AuAu_50ev_10_20_eta4_n65.root \\
        --event 0 --ntau-fno 119 --tau0 0.44 --dtau 0.1 \\
        --output /home/cparker/jetscape-docker/X-SCAPE/external_packages/js-contrib/contribs/FnoHydro/models/ground_truth_passthrough_10_20_eta4_n65_event0.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loc_libs"))
from read_3d_root import read_3d_data  # noqa: E402


class ConstantOutput(torch.nn.Module):
    """forward(x) ignores x and always returns the embedded ground-truth
    tensor -- traces to a genuine constant-output TorchScript graph."""

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("out", tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A bare `return self.out` traces to a graph node with no real op
        # depending on it, which trips a PyTorch tracer internal assert
        # ("outputs_[i]->uses().empty()"). Multiplying by a harmless
        # x-independent 1.0 gives the tracer a genuine op to attach the
        # output to while keeping forward() an exact constant function.
        return self.out * 1.0


def build_ground_truth_tensor(root_file, event, ntau_fno, tau0, dtau,
                               nx=64, ny=64, neta=64):
    # choose_ntau must cover raw tau index ntau_fno (= output slot
    # ntau_fno-1's raw index j+1 = ntau_fno), so request ntau_fno+1 raw
    # steps; read_3d_data zero-pads past this event's own ntau_freezeout,
    # which is physically correct (already-frozen-out cells read e=0).
    choose_ntau = ntau_fno + 1
    arr_data = read_3d_data(
        root_file, nevents=event + 1, choose_ntau=choose_ntau,
        xyslice=slice(None, -1), etaslice=slice(None, -1),
        tauslice=slice(None, choose_ntau), min_ntau_freezeout=None,
    )
    arr = arr_data["arr"]  # (nevents, nFeatures=5, nx, ny, neta, ntau)
    assert arr.shape[2:5] == (nx, ny, neta), f"unexpected spatial shape {arr.shape}"
    assert abs(arr_data["tau_min"] - tau0) < 1e-3, (
        f"--tau0 {tau0} doesn't match this ROOT file's tau_min "
        f"{arr_data['tau_min']} -- pass the correct --tau0."
    )
    assert abs(arr_data["dtau"] - dtau) < 1e-6, (
        f"--dtau {dtau} doesn't match this ROOT file's dtau {arr_data['dtau']}."
    )

    raw = arr[event]  # (5, nx, ny, neta, choose_ntau)
    # output slot j (0..ntau_fno-1) <- raw tau index j+1
    out = raw[:, :, :, :, 1:ntau_fno + 1].copy()  # (5, nx, ny, neta, ntau_fno)

    tau_phys = tau0 + np.arange(1, ntau_fno + 1) * dtau  # (ntau_fno,)
    ed = out[0]  # (nx, ny, neta, ntau_fno)
    ed_clipped = np.clip(ed, 0.0, None)
    out[0] = ed_clipped ** 0.75 * tau_phys.reshape(1, 1, 1, ntau_fno)

    print(f"Ground-truth tensor built: event {event}, ntau_freezeout="
          f"{arr_data['ntau_freezeout'][event]}, shape {out.shape}")
    print(f"  channel0 (normalized ed) range: [{out[0].min():.4g}, {out[0].max():.4g}]")
    print(f"  channel1 (T) range: [{out[1].min():.4g}, {out[1].max():.4g}]")

    return torch.from_numpy(out).float().unsqueeze(0)  # (1, 5, nx, ny, neta, ntau_fno)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-file", required=True)
    p.add_argument("--event", type=int, default=0)
    p.add_argument("--ntau-fno", type=int, default=119,
                   help="must match the XML <ntau> / export_torchscript.py --ntau used for the real FNO model on this grid")
    p.add_argument("--tau0", type=float, default=0.44)
    p.add_argument("--dtau", type=float, default=0.1)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--neta", type=int, default=64)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    out_tensor = build_ground_truth_tensor(
        args.root_file, args.event, args.ntau_fno, args.tau0, args.dtau,
        nx=args.nx, ny=args.ny, neta=args.neta,
    )

    model = ConstantOutput(out_tensor)
    model.eval()
    example = torch.randn(1, 5, args.nx, args.ny, args.neta, args.ntau_fno)
    with torch.no_grad():
        eager_out = model(example)
        traced = torch.jit.trace(model, example)
        traced_out = traced(example)
    max_diff = (eager_out - traced_out).abs().max().item()
    print(f"Sanity check: max |eager - traced| = {max_diff:.3e}")
    assert max_diff == 0.0, "traced module should be an EXACT constant function"

    # Confirm it really is constant (ignores input), not just matching on
    # this one example -- call again with a totally different input.
    other = torch.zeros_like(example)
    other_out = traced(other)
    assert torch.equal(other_out, traced_out), "module output depends on input -- not a true passthrough!"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Saved ground-truth passthrough module -> {out_path}")
    print(
        "\nXML <FNO> block should use:\n"
        f"  <n_features>5</n_features>\n"
        f"  <nx>{args.nx}</nx>\n"
        f"  <ny>{args.ny}</ny>\n"
        f"  <neta>{args.neta}</neta>\n"
        f"  <ntau>{args.ntau_fno}</ntau>\n"
        "plus <x_min>/<y_min>/<eta_min>/<dtau>/<deta> matching the grid this "
        "ground truth was read from -- same as the real FNO model it stands in for."
    )


if __name__ == "__main__":
    main()
