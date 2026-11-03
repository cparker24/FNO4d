#!/usr/bin/env python
"""Build a traced TorchScript module that runs the REAL trained FNO model,
but forces its input to be one specific training event's own initial
condition (read straight from a training ROOT file) instead of whatever
the live MCGlauber+MusicPrimedPreEq priming stage hands FnoHydro.cc.

Why: build_ground_truth_passthrough_model.py isolated the effect of grid
RESOLUTION alone (real MUSIC physics particlized on FNO's grid). But that
comparison's "FNO prediction" arm (the existing 20-event ensemble) used a
live-primed initial condition, i.e. a DIFFERENT physical event than
whichever training event the ground truth came from -- confounding the
comparison with genuine event-to-event fluctuation on top of the
model's own prediction error. This module removes that confound: for a
given training event, the model now predicts from EXACTLY that event's own
tau0 state, not an unrelated one, so ground truth and FNO prediction can
be compared one-for-one on the same initial state.

This does NOT fully eliminate the FnoHydro.cc tau0-seam artifact found in
build_ground_truth_passthrough_model.py -- FnoHydro.cc always prepends the
live pre_eq_ptr snapshot as the trajectory's position-0 entry regardless of
what the model itself returns (no XML/config hook to override that one
slice without patching the C++ source), so that single slice still comes
from the live (mismatched) priming run. But the model's entire OWN
forward-pass computation (all 119 predicted tau slices, i.e. 119/120 of
the full trajectory) is now genuinely conditioned on the correct, matching
initial state -- the part that actually matters for whether the trained
network is doing a good job on ITS OWN dynamics, not an artifact of being
fed an unrelated starting point. See
PARTICLIZATION_INVESTIGATION.md's "Particlizing MUSIC on the FNO grid"
section for the quantified size of that one-slice artifact (~0.1% of
surface cells, already shown to be small and non-fatal).

Normalisation and tau-index convention: identical to
build_ground_truth_passthrough_model.py (NormalizeEnergyDensity's
etau75_3d forward transform on channel 0 only, tau0 + k*dtau convention);
here only the raw tau index k=0 (the IC itself) is needed, then repeated
across the tau axis the same way FnoHydro.cc repeats its own IC
(`fno_input_tensor.repeat({1,1,1,1,ntau_fno})`).

Usage:
    python build_fixed_ic_fno_model.py \\
        --root-file /home/cparker/jetscape-docker/rootfiles/AuAu_50ev_10_20_eta4_n65.root \\
        --event 0 \\
        --fno-config /home/cparker/jetscape-docker/FNO4d/train_torch/checkpoints_lambda_widened_eta4_n65/config.yaml \\
        --fno-checkpoint /home/cparker/jetscape-docker/FNO4d/train_torch/checkpoints_lambda_widened_eta4_n65/best_model_state_dict.pt \\
        --ntau-fno 119 --tau0 0.44 --dtau 0.1 \\
        --output /home/cparker/jetscape-docker/X-SCAPE/external_packages/js-contrib/contribs/FnoHydro/models/fixed_ic_fno_eta4_n65_event0.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loc_libs"))
from read_3d_root import read_3d_data  # noqa: E402


def build_model(cfg, in_channels, out_channels):
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=cfg["n_modes"],
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
    )


def load_state_dict(checkpoint_path, device):
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw.items() if isinstance(v, torch.Tensor)
    }


class FixedInputModel(torch.nn.Module):
    """forward(x) ignores x and runs the real FNO model on the embedded
    fixed initial condition instead."""

    def __init__(self, fno_model: torch.nn.Module, fixed_input: torch.Tensor):
        super().__init__()
        self.fno_model = fno_model
        self.register_buffer("fixed_input", fixed_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno_model(self.fixed_input)


def build_fixed_input(root_file, event, ntau_fno, tau0, dtau, nx=64, ny=64, neta=64):
    arr_data = read_3d_data(
        root_file, nevents=event + 1, choose_ntau=1,
        xyslice=slice(None, -1), etaslice=slice(None, -1), tauslice=slice(None, 1),
        min_ntau_freezeout=None,
    )
    arr = arr_data["arr"]  # (nevents, nFeatures=5, nx, ny, neta, ntau=1)
    assert arr.shape[2:5] == (nx, ny, neta), f"unexpected spatial shape {arr.shape}"
    assert abs(arr_data["tau_min"] - tau0) < 1e-3, (
        f"--tau0 {tau0} doesn't match this ROOT file's tau_min {arr_data['tau_min']}."
    )

    ic = arr[event, :, :, :, :, 0].copy()  # (5, nx, ny, neta) -- raw tau index 0, the IC
    ed = np.clip(ic[0], 0.0, None)
    ic[0] = ed ** 0.75 * tau0  # NormalizeEnergyDensity(ed, tau0), channel 0 only

    print(f"Fixed IC built: event {event}, channel0 (normalized ed) range "
          f"[{ic[0].min():.4g}, {ic[0].max():.4g}], channel1 (T) range "
          f"[{ic[1].min():.4g}, {ic[1].max():.4g}]")

    ic_t = torch.from_numpy(ic).float()  # (5, nx, ny, neta)
    # Repeat across the tau axis, matching FnoHydro.cc's own
    # fno_input_tensor.repeat({1,1,1,1,ntau_fno}), then add the batch dim.
    repeated = ic_t.unsqueeze(-1).repeat(1, 1, 1, 1, ntau_fno)  # (5, nx, ny, neta, ntau_fno)
    return repeated.unsqueeze(0)  # (1, 5, nx, ny, neta, ntau_fno)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-file", required=True)
    p.add_argument("--event", type=int, default=0)
    p.add_argument("--fno-config", required=True)
    p.add_argument("--fno-checkpoint", required=True)
    p.add_argument("--channels", type=int, default=5)
    p.add_argument("--ntau-fno", type=int, default=119)
    p.add_argument("--tau0", type=float, default=0.44)
    p.add_argument("--dtau", type=float, default=0.1)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--neta", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=1e-5)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = yaml.safe_load(Path(args.fno_config).read_text())
    print(f"n_modes={cfg['n_modes']}  hidden_channels={cfg['hidden_channels']}  channels={args.channels}")

    fno_model = build_model(cfg, args.channels, args.channels).to(device)
    fno_model.load_state_dict(load_state_dict(Path(args.fno_checkpoint), device))
    fno_model.eval()

    fixed_input = build_fixed_input(
        args.root_file, args.event, args.ntau_fno, args.tau0, args.dtau,
        nx=args.nx, ny=args.ny, neta=args.neta,
    ).to(device)

    wrapper = FixedInputModel(fno_model, fixed_input)
    wrapper.eval()

    example = torch.randn(1, args.channels, args.nx, args.ny, args.neta, args.ntau_fno, device=device)
    with torch.no_grad():
        eager_out = wrapper(example)
        traced = torch.jit.trace(wrapper, example)
        traced_out = traced(example)
    max_diff = (eager_out - traced_out).abs().max().item()
    print(f"Sanity check (traced vs eager): max diff = {max_diff:.3e}")
    if max_diff > args.atol:
        raise RuntimeError(f"Traced module diverges from eager by {max_diff:.3e} > atol={args.atol}")

    # Confirm the wrapper really ignores x (same real output on a totally
    # different input) -- different from build_ground_truth_passthrough_model.py's
    # exact-equality check only because fno_model(fixed_input) involves
    # floating-point ops that could in principle vary call-to-call; an
    # identical result here is still the expected/correct behavior.
    other = torch.zeros_like(example)
    other_out = traced(other)
    diff_vs_other = (other_out - traced_out).abs().max().item()
    print(f"Ignores-input check: max diff when called with a different x = {diff_vs_other:.3e}")
    assert diff_vs_other == 0.0, "module output depends on input -- not a true fixed-IC wrapper!"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Saved fixed-IC FNO module -> {out_path}")


if __name__ == "__main__":
    main()
