#!/usr/bin/env python
"""Build a "ground-truth bank playback" TorchScript module for FnoHydro: a
fake FNO model that ignores its input entirely and, on each successive
forward() call, returns the real MUSIC evolution for the NEXT event in a
stack of N events read from a training-format ROOT file, each resampled
onto exactly the same (nx, ny, neta, dtau, deta) grid an FNO model trains
and predicts on.

This extends build_ground_truth_passthrough_model.py's single-event
ConstantOutput module (which embeds one event as a hardcoded constant) to
a full N-event ensemble: a stateful module holding all N events' tensors
stacked in a buffer, plus a persistent integer counter buffer that
advances by 1 on every forward() call (in-place buffer mutation, so it
survives across separate calls to the loaded TorchScript module within
one live JETSCAPE process run across N events). See that script's own
docstring for the broader motivation (isolating resolution/domain-crop
effects from genuine FNO model error) and the exact tensor/normalization
conventions reused here unchanged.

Run with: python build_ground_truth_bank_playback_model.py \\
    --root-file /path/to/some_eta4_n65.root --n-events 3 \\
    --ntau-fno 119 --tau0 0.44 --dtau 0.1 --output /tmp/bank_playback_smoketest.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loc_libs"))
from read_3d_root import read_3d_data  # noqa: E402

from build_ground_truth_passthrough_model import build_ground_truth_tensor  # noqa: E402


class BankPlayback(torch.nn.Module):
    """forward(x) ignores x; returns bank[counter] and advances counter by
    1 (wrapping modulo N), via in-place buffer mutation so state persists
    across separate forward() calls on the loaded, traced module."""

    def __init__(self, bank: torch.Tensor):
        super().__init__()
        self.register_buffer("bank", bank)  # (N, 5, nx, ny, neta, ntau_fno)
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer("n_events", torch.tensor(bank.shape[0], dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.counter % self.n_events
        out = self.bank.index_select(0, idx.reshape(1)).squeeze(0) * 1.0
        self.counter.add_(1)
        return out.unsqueeze(0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-file", required=True)
    p.add_argument("--n-events", type=int, required=True)
    p.add_argument("--event-offset", type=int, default=0,
                   help="index of the first event to read from the ROOT file")
    p.add_argument("--ntau-fno", type=int, default=119)
    p.add_argument("--tau0", type=float, default=0.44)
    p.add_argument("--dtau", type=float, default=0.1)
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--neta", type=int, default=64)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tensors = []
    for i in range(args.n_events):
        ev = args.event_offset + i
        t = build_ground_truth_tensor(
            args.root_file, ev, args.ntau_fno, args.tau0, args.dtau,
            nx=args.nx, ny=args.ny, neta=args.neta,
        )  # (1, 5, nx, ny, neta, ntau_fno)
        tensors.append(t.squeeze(0))
    bank = torch.stack(tensors, dim=0)  # (N, 5, nx, ny, neta, ntau_fno)
    print(f"Bank built: {bank.shape[0]} events, shape {tuple(bank.shape)}")

    model = BankPlayback(bank)
    model.eval()
    example = torch.randn(1, 5, args.nx, args.ny, args.neta, args.ntau_fno)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    # torch.jit.trace() itself executes forward() once to record the graph,
    # which advances the counter buffer as a side effect -- reset it to 0
    # here so the SAVED module's first real forward() call returns event 0,
    # not event 1. Confirmed empirically: without this reset, call N returns
    # bank[(N+1) % n_events], an off-by-one that silently shifts every
    # event's ground truth onto the wrong live JETSCAPE event.
    traced.counter.zero_()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Saved bank playback module -> {out_path}")


if __name__ == "__main__":
    main()
