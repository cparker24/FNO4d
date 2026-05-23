#!/usr/bin/env python3
"""
tools/export_weights.py
=======================
Convert a trained FNO model (PyTorch .pt checkpoint or PyTorch .safetensors)
into the MLX safetensors format expected by neuralop_mlx_cpp, and write a
companion model_config.json.

Usage
-----
  # From a PyTorch .pt checkpoint + training YAML config:
  python tools/export_weights.py \
      --pt path/to/best_model.pt \
      --train-config path/to/config.yaml \
      --out weights.safetensors \
      --config-out model_config.json

  # From a PyTorch safetensors file + training YAML config:
  python tools/export_weights.py \
      --safetensors path/to/weights.safetensors \
      --train-config path/to/config.yaml \
      --out weights_mlx.safetensors \
      --config-out model_config.json

The output weights.safetensors can be loaded directly by neuralop_mlx_cpp
(load_weights auto-detects the MLX format from the key names).

Dependencies: mlx, neuralop (neuralop_mlx), pyyaml, safetensors (optional,
              for reading PyTorch safetensors).
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from tools/ directory
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

import mlx.core as mx
import mlx.utils
import yaml


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_training_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _resolve_fno_params(cfg: dict) -> dict:
    """Extract the FNO hyper-parameters from a training config dict."""
    # Training configs nest differently across versions; try common paths.
    fno = cfg.get("fno", cfg.get("model", cfg))
    return {
        "n_modes":                  fno.get("n_modes", [16, 16, 16, 16]),
        "in_channels":              fno.get("in_channels",  4),
        "out_channels":             fno.get("out_channels", 4),
        "hidden_channels":          fno.get("hidden_channels", 64),
        "n_layers":                 fno.get("n_layers", 4),
        "lifting_channel_ratio":    float(fno.get("lifting_channel_ratio",    2.0)),
        "projection_channel_ratio": float(fno.get("projection_channel_ratio", 2.0)),
        "positional_embedding":     bool( fno.get("positional_embedding",     True)),
        "use_channel_mlp":          bool( fno.get("use_channel_mlp",          True)),
        "channel_mlp_expansion":    float(fno.get("channel_mlp_expansion",    0.5)),
        "fno_skip":                       fno.get("fno_skip",          "linear"),
        "channel_mlp_skip":               fno.get("channel_mlp_skip",  "soft-gating"),
        "fft_norm":                       fno.get("fft_norm",           "forward"),
    }


def _build_mlx_model(params: dict):
    """Construct an MLX FNO model with the given hyper-parameters."""
    from neuralop_mlx.models.fno import FNO
    return FNO(
        n_modes=params["n_modes"],
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        hidden_channels=params["hidden_channels"],
        n_layers=params["n_layers"],
        lifting_channel_ratio=params["lifting_channel_ratio"],
        projection_channel_ratio=params["projection_channel_ratio"],
        positional_embedding=params.get("positional_embedding", True),
        use_channel_mlp=params.get("use_channel_mlp", True),
        channel_mlp_expansion=params.get("channel_mlp_expansion", 0.5),
        fno_skip=params.get("fno_skip", "linear"),
        channel_mlp_skip=params.get("channel_mlp_skip", "soft-gating"),
        fft_norm=params.get("fft_norm", "forward"),
    )


def _load_from_pytorch_pt(model, pt_path: str):
    from neuralop_mlx.weight_transfer import load_from_torch
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    load_from_torch(model, ckpt)


def _load_from_pytorch_safetensors(model, st_path: str):
    """Load from a PyTorch safetensors file."""
    try:
        from safetensors import safe_open
        import torch
        tensors = {}
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        # Build a fake checkpoint dict that load_from_torch understands
        from neuralop_mlx.weight_transfer import load_from_torch
        load_from_torch(model, {"model_state_dict": tensors})
    except ImportError:
        raise RuntimeError(
            "The 'safetensors' Python package is required to read "
            "PyTorch safetensors files.  Install with:  pip install safetensors"
        )


def _flatten_params(model) -> dict:
    """Flatten the MLX model parameter tree to a flat {key: array} dict."""
    return dict(mlx.utils.tree_flatten(model.parameters()))


def _write_config(params: dict, path: str):
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Config written to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export FNO weights to MLX safetensors format")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt",          metavar="FILE",
                     help="PyTorch .pt checkpoint (model_state_dict wrapper "
                          "or raw state_dict)")
    src.add_argument("--safetensors", metavar="FILE",
                     help="PyTorch safetensors file")

    parser.add_argument("--train-config", required=True, metavar="YAML",
                        help="Training YAML config (model hyper-parameters)")
    parser.add_argument("--out", required=True, metavar="FILE",
                        help="Output MLX safetensors path")
    parser.add_argument("--config-out", default=None, metavar="FILE",
                        help="Output JSON config path (default: <out>.json)")

    args = parser.parse_args()

    config_out = args.config_out or (
        str(Path(args.out).with_suffix("")) + "_config.json"
    )

    print(f"Loading training config from {args.train_config}")
    train_cfg = _load_training_config(args.train_config)
    params    = _resolve_fno_params(train_cfg)

    print("Building MLX FNO model …")
    model = _build_mlx_model(params)

    if args.pt:
        print(f"Loading PyTorch checkpoint from {args.pt} …")
        _load_from_pytorch_pt(model, args.pt)
    else:
        print(f"Loading PyTorch safetensors from {args.safetensors} …")
        _load_from_pytorch_safetensors(model, args.safetensors)

    print("Evaluating MLX parameters …")
    mx.eval(model.parameters())

    flat_params = _flatten_params(model)
    print(f"Saving {len(flat_params)} weight tensors to {args.out} …")
    mx.save_safetensors(args.out, flat_params)
    print(f"Weights saved to {args.out}")

    _write_config(params, config_out)


if __name__ == "__main__":
    main()
