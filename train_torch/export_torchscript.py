#!/usr/bin/env python
"""Export a trained FNO checkpoint (state_dict) to a JIT-traced TorchScript
module consumable by X-SCAPE's FnoHydro contrib
(external_packages/js-contrib/contribs/FnoHydro).

FnoHydro builds its input tensor by repeating the initial condition across
the tau axis and calling module.forward on a single tensor of shape
    (1, n_features, nx, ny, neta, ntau)
(see FnoHydro.cc: fno_input_tensor.repeat({1,1,1,1,ntau_fno}).unsqueeze(0)).
This matches how eval.py feeds the model during evaluation (x = repeat(IC,
T-1)), so we trace against exactly that shape.

Usage:
    python export_torchscript.py \\
        --config ./checkpoints/config.yaml \\
        --checkpoint ./checkpoints/best_model_state_dict.pt \\
        --output ./checkpoints/traced_fno.pt \\
        --nx 64 --ny 64 --neta 16 --ntau 31
"""
import argparse
from pathlib import Path

import torch
import yaml
from neuralop.models import FNO


def build_model(cfg: dict, in_channels: int, out_channels: int) -> FNO:
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=cfg["n_modes"],
        hidden_channels=cfg["hidden_channels"],
        projection_channel_ratio=2,
    )


def load_state_dict(checkpoint_path: Path, device: torch.device) -> dict:
    raw = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw.items() if isinstance(v, torch.Tensor)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="./checkpoints/config.yaml")
    p.add_argument("--checkpoint", default="./checkpoints/best_model_state_dict.pt")
    p.add_argument("--output", default="./checkpoints/traced_fno.pt")
    p.add_argument("--channels", type=int, default=4,
                    help="in_channels == out_channels for this model (eps, vx, vy, veta)")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--neta", type=int, default=16)
    p.add_argument("--ntau", type=int, default=34,
                    help="output tau steps == FnoHydro's <ntau> == tauslice_upper - 1")
    p.add_argument("--device", default="cpu",
                    help="Trace on cpu for a portable model unless FnoHydro will always run on the same GPU arch")
    p.add_argument("--atol", type=float, default=1e-5,
                    help="max allowed |traced - eager| output difference in the sanity check")
    args = p.parse_args()

    device = torch.device(args.device)

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Config:     {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"n_modes={cfg['n_modes']}  hidden_channels={cfg['hidden_channels']}  channels={args.channels}")

    model = build_model(cfg, args.channels, args.channels).to(device)
    state = load_state_dict(ckpt_path, device)
    model.load_state_dict(state)
    model.eval()

    n_params = sum(v.numel() for v in model.parameters())
    print(f"Loaded model: {n_params:,} params")

    example_shape = (1, args.channels, args.nx, args.ny, args.neta, args.ntau)
    print(f"Tracing with example input shape {example_shape} "
          f"(repeated-IC convention matching FnoHydro.cc / eval.py)")
    example = torch.randn(*example_shape, device=device)

    with torch.no_grad():
        eager_out = model(example)
        traced = torch.jit.trace(model, example)
        traced_out = traced(example)

    max_diff = (eager_out - traced_out).abs().max().item()
    print(f"Sanity check: max |eager - traced| = {max_diff:.3e}")
    if max_diff > args.atol:
        raise RuntimeError(
            f"Traced module diverges from eager model by {max_diff:.3e} > atol={args.atol}. "
            "The model likely has shape-dependent control flow that trace() can't capture correctly — "
            "inspect before shipping this .pt file."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Saved traced TorchScript module -> {out_path}")
    print(
        "\nXML <FNO> block should use:\n"
        f"  <n_features>{args.channels}</n_features>\n"
        f"  <nx>{args.nx}</nx>\n"
        f"  <ny>{args.ny}</ny>\n"
        f"  <neta>{args.neta}</neta>\n"
        f"  <ntau>{args.ntau}</ntau>\n"
        "plus <x_min>/<y_min>/<eta_min>/<dtau>/<tau0> matching the grid the model was trained on."
    )


if __name__ == "__main__":
    main()
