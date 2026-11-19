#!/usr/bin/env python
"""Build a "fixed-IC FNO bank playback" TorchScript module for FnoHydro:
the REAL trained FNO network (stored once), run on the NEXT event in a
stack of N events' own fixed initial conditions on each successive
forward() call, instead of whatever the live pipeline hands it.

Extends build_fixed_ic_fno_model.py's single-event FixedInputModel (one
real-network trace per event, which duplicates the full network ~3.4GB per
event -- not viable at N=20 events x 4 centralities x 2 arms, see
PARTICLIZATION_INVESTIGATION.md) to a full N-event ensemble sharing ONE
copy of the network, using the same persistent-counter buffer-mutation
pattern as build_ground_truth_bank_playback_model.py.

Usage:
    python build_fixed_ic_fno_bank_playback_model.py \\
        --root-file /path/to/AuAu_valid20ev_10_20_eta4_n65_symgrid.root \\
        --n-events 20 \\
        --fno-config /path/to/checkpoints_.../config.yaml \\
        --fno-checkpoint /path/to/checkpoints_.../best_model_state_dict.pt \\
        --ntau-fno 119 --tau0 0.44 --dtau 0.1 \\
        --channels 4 --drop-channels 1 --no-crop-edge-bin \\
        --output /path/to/fixed_ic_bank_10_20_old.pt
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from neuralop.models import FNO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_fixed_ic_fno_model import build_model, load_state_dict, build_fixed_input  # noqa: E402


class FixedICBankPlayback(torch.nn.Module):
    """forward(x) ignores x; runs the REAL fno_model on bank[counter]'s
    fixed IC, advances counter by 1 (wrapping modulo N) via in-place
    buffer mutation, same persistence-across-calls pattern as
    BankPlayback in build_ground_truth_bank_playback_model.py.

    The bank stores only each event's tau0 slice (N, channels, nx, ny,
    neta) -- NOT pre-repeated across the tau axis, unlike
    build_fixed_ic_fno_model.py's single-event FixedInputModel. Repeating
    119x per event before storage would bloat an N=20 bank ~119x for zero
    informational gain (every tau slice is identical copies of the IC);
    the repeat happens here in forward() instead, at negligible cost."""

    def __init__(self, fno_model: torch.nn.Module, bank: torch.Tensor, ntau_fno: int):
        super().__init__()
        self.fno_model = fno_model
        self.register_buffer("bank", bank)  # (N, channels, nx, ny, neta) -- tau0 only
        self.register_buffer("counter", torch.zeros((), dtype=torch.int64))
        self.register_buffer("n_events", torch.tensor(bank.shape[0], dtype=torch.int64))
        self.ntau_fno = ntau_fno

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = self.counter % self.n_events
        ic = self.bank.index_select(0, idx.reshape(1))  # (1, channels, nx, ny, neta)
        fixed_input = ic.unsqueeze(-1).repeat(1, 1, 1, 1, 1, self.ntau_fno)
        self.counter.add_(1)
        return self.fno_model(fixed_input)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-file", required=True)
    p.add_argument("--n-events", type=int, required=True)
    p.add_argument("--event-offset", type=int, default=0)
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
    p.add_argument("--drop-channels", default=None)
    p.add_argument("--no-crop-edge-bin", action="store_true")
    args = p.parse_args()
    drop_channels = [int(x) for x in args.drop_channels.split(",")] if args.drop_channels else None

    device = torch.device(args.device)
    cfg = yaml.safe_load(Path(args.fno_config).read_text())
    print(f"n_modes={cfg['n_modes']}  hidden_channels={cfg['hidden_channels']}  channels={args.channels}")

    fno_model = build_model(cfg, args.channels, args.channels).to(device)
    fno_model.load_state_dict(load_state_dict(Path(args.fno_checkpoint), device))
    fno_model.eval()

    tensors = []
    for i in range(args.n_events):
        ev = args.event_offset + i
        fixed_input = build_fixed_input(
            args.root_file, ev, args.ntau_fno, args.tau0, args.dtau,
            nx=args.nx, ny=args.ny, neta=args.neta,
            drop_channels=drop_channels, crop_edge_bin=not args.no_crop_edge_bin,
        )  # (1, channels, nx, ny, neta, ntau_fno) -- all ntau_fno slices identical copies
        tensors.append(fixed_input[0, ..., 0])  # (channels, nx, ny, neta) -- tau0 slice only
    bank = torch.stack(tensors, dim=0).to(device)  # (N, channels, nx, ny, neta)
    print(f"Bank built: {bank.shape[0]} events, shape {tuple(bank.shape)}")

    model = FixedICBankPlayback(fno_model, bank, args.ntau_fno)
    model.eval()

    example = torch.randn(1, args.channels, args.nx, args.ny, args.neta, args.ntau_fno, device=device)
    with torch.no_grad():
        eager_out0 = model(example)
        traced = torch.jit.trace(model, example)
    # torch.jit.trace executes forward() once, advancing the counter as a
    # side effect -- reset so the SAVED module's first real call returns
    # event (event_offset + 0), matching build_ground_truth_bank_playback_model.py's
    # documented fix for the same off-by-one.
    traced.counter.zero_()

    with torch.no_grad():
        traced_out0 = traced(example)
    max_diff = (eager_out0 - traced_out0).abs().max().item()
    print(f"Sanity check (traced vs eager, event 0): max diff = {max_diff:.3e}")
    if max_diff > args.atol:
        raise RuntimeError(f"Traced module diverges from eager by {max_diff:.3e} > atol={args.atol}")

    # Confirm distinct events really are distinct (counter genuinely
    # advances and selects different bank rows) -- call N times with a
    # fresh, differently-seeded module copy loaded from disk, same as a
    # live JETSCAPE process would.
    traced.save(str(args.output) + ".selftest")
    check = torch.jit.load(str(args.output) + ".selftest")
    outs = []
    with torch.no_grad():
        for _ in range(min(3, args.n_events)):
            outs.append(check(example).clone())
    distinct = all((outs[i] - outs[i + 1]).abs().max().item() > 0 for i in range(len(outs) - 1))
    print(f"Distinct-events check (first {len(outs)} calls all differ): {distinct}")
    if not distinct:
        raise RuntimeError("Bank playback returned identical output across successive calls -- counter not advancing!")
    Path(str(args.output) + ".selftest").unlink()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Saved fixed-IC FNO bank playback module ({args.n_events} events) -> {out_path}")


if __name__ == "__main__":
    main()
