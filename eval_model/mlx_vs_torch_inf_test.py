"""Inference comparison: PyTorch FNO vs MLX FNO (weight transfer from the same checkpoint).

Loads the best_model_state_dict.pt from eval_model/best_model_compare/,
runs inference with both backends on the same test data, and compares:
  - 2-D x/y energy figures for eta bin 16, t bins 1, 5, 10, 30
  - Numerical tensor-level comparison (max abs diff, RMSE, cosine similarity, etc.)

Run:
    python eval_model/mlx_vs_torch_inf_test.py [config_mlx_torch.yaml]
or from inside eval_model/:
    python mlx_vs_torch_inf_test.py [config_mlx_torch.yaml]

A path to an alternative config may be passed as the first CLI argument.
"""

import sys
import gc
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")          # no interactive display needed
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent   # …/eval_model
REPO_ROOT  = SCRIPT_DIR.parent                 # …/FNO4d

sys.path.insert(0, str(REPO_ROOT / "loc_libs"))
sys.path.insert(0, str(REPO_ROOT / "neuraloperator-4d-2.0.0"))
sys.path.insert(0, str(REPO_ROOT))

from read_3d_root import read_3d_data
from norm_utils import Normalizer

# ── load comparison config ────────────────────────────────────────────────────

_cfg_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "config_mlx_torch.yaml"
# resolve relative paths against SCRIPT_DIR
if not _cfg_arg.is_absolute():
    _cfg_arg = SCRIPT_DIR / _cfg_arg

print(f"Comparison config : {_cfg_arg}")
with open(_cfg_arg) as _f:
    _cmp = yaml.safe_load(_f)

def _resolve(p):
    """Resolve a path from the config relative to SCRIPT_DIR."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()

COMPARE_DIR = _resolve(_cmp["compare_dir"])
CKPT_PATH   = COMPARE_DIR / _cmp["checkpoint_file"]
CONFIG_PATH = _resolve(_cmp["train_config"])
NORM_PATH   = _resolve(_cmp.get("normalizer"))       # None → skip normaliser
OUTPUT_DIR  = _resolve(_cmp.get("output_dir", "./compare_output"))

IETA_PLOT   = int(_cmp.get("ieta_plot", 16))
T_BINS_PLOT = list(_cmp.get("t_bins_plot", [1, 5, 10, 30]))
CHANNEL     = int(_cmp.get("channel", 0))
EVENT       = int(_cmp.get("event_index", 0))

CH_LABEL    = ["ε (GeV/fm³)", "vx", "vy", "vη"][CHANNEL]

print(f"Checkpoint        : {CKPT_PATH}")
print(f"Train config      : {CONFIG_PATH}")
print(f"Normalizer        : {NORM_PATH}")
print(f"Output dir        : {OUTPUT_DIR}")
print(f"η bin             : {IETA_PLOT}")
print(f"τ bins            : {T_BINS_PLOT}")
print(f"Channel           : {CHANNEL} ({CH_LABEL})")

# ── load training config ──────────────────────────────────────────────────────

with open(CONFIG_PATH) as _f:
    cfg = yaml.safe_load(_f)

def _s(v):
    return slice(None) if v is None else slice(*v)

input_dir       = cfg.get("input_dir", ".")
input_files     = cfg["input_files"]
train_ratio     = cfg.get("train_ratio", 0.8)
xyslice         = _s(cfg.get("xyslice", [None, -1]))
if 'etaslice' in cfg:
    etaslice = _s(cfg['etaslice'])
    tauslice = _s(cfg.get('tauslice', [None, None]))
else:
    # old config: 'tauslice' was applied to the eta axis
    etaslice = _s(cfg.get('tauslice', [None, -1]))
    tauslice = slice(None, None)
hidden_channels = cfg["hidden_channels"]
n_modes         = cfg["n_modes"]

# resolve input_dir relative to COMPARE_DIR (config lives there)
data_path = Path(input_dir)
if not data_path.is_absolute():
    data_path = COMPARE_DIR / data_path
data_file = str(data_path / input_files[0])

# ── load & prepare data ───────────────────────────────────────────────────────

print(f"Data   : {data_file}")
raw = read_3d_data(data_file)
grid = {k: raw[k] for k in ("tau_min","dtau","eta_min","deta","x_min","dx","y_min","dy")
        if k in raw}
print("Grid   :", {k: f"{v:.4g}" for k, v in grid.items()})

arr  = raw["arr"]
arr  = arr[:, :, xyslice, xyslice, etaslice, tauslice]
print(f"Shape  : {arr.shape}  (N, C, nx, ny, neta, tau)")

n_train   = int(arr.shape[0] * train_ratio)
data_test = arr[n_train:]
print(f"Test   : {data_test.shape[0]} events  (split at {n_train})")
del arr; gc.collect()

# normalizer (optional)
if NORM_PATH is not None and NORM_PATH.exists():
    normalizer     = Normalizer.load(str(NORM_PATH))
    data_test_norm = normalizer.transform(data_test)
    print(f"Norm   : {normalizer}")
else:
    normalizer     = None
    data_test_norm = data_test
    print("Norm   : none")

# grid coordinates
nx_eff  = data_test_norm.shape[2]
ny_eff  = data_test_norm.shape[3]
ntau    = data_test_norm.shape[5] - 1

x_min  = grid.get("x_min", 0.0);  dx   = grid.get("dx",   1.0)
y_min  = grid.get("y_min", 0.0);  dy   = grid.get("dy",   1.0)
tau_min = grid.get("tau_min", 0.6); dtau = grid.get("dtau", 0.1)
eta_min = grid.get("eta_min", 0.0); deta = grid.get("deta", 1.0)

x_coords   = x_min  + np.arange(nx_eff) * dx
y_coords   = y_min  + np.arange(ny_eff) * dy
tau_coords = tau_min + (np.arange(ntau) + 1) * dtau
eta_coords = eta_min + np.arange(data_test_norm.shape[4]) * deta

xy_extent  = [x_coords[0] - dx/2, x_coords[-1] + dx/2,
              y_coords[0] - dy/2, y_coords[-1] + dy/2]

# ── validate requested bins ───────────────────────────────────────────────────

assert IETA_PLOT < data_test_norm.shape[4], (
    f"eta bin {IETA_PLOT} out of range (neta={data_test_norm.shape[4]})")

t_valid = []
for t in T_BINS_PLOT:
    if t < ntau:
        t_valid.append(t)
    else:
        print(f"Warning: t bin {t} >= ntau={ntau}, skipping")
T_BINS_PLOT = t_valid

# ── PyTorch inference ─────────────────────────────────────────────────────────

print("\n── PyTorch inference ─────────────────────────────────────────────")
import torch
from neuralop.models import FNO as TorchFNO

device = torch.device("cpu")

raw_ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
    state_dict = raw_ckpt["model_state_dict"]
elif isinstance(raw_ckpt, dict) and all(isinstance(v, torch.Tensor) for v in raw_ckpt.values()):
    state_dict = raw_ckpt
else:
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_ckpt.items() if isinstance(v, torch.Tensor)
    }

torch_model = TorchFNO(
    in_channels=4, out_channels=4,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
).to(device)
torch_model.load_state_dict(state_dict)
torch_model.eval()
n_params_pt = sum(p.numel() for p in torch_model.parameters())
print(f"Params : {n_params_pt:,}")

def run_torch_inference(model, data, dev):
    preds = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            x0  = data[i, :, :, :, :, :1]
            T   = data.shape[-1] - 1
            x   = np.repeat(x0, T, axis=-1)
            xin = torch.FloatTensor(x).unsqueeze(0).to(dev)
            out = model(xin).squeeze(0).cpu().numpy()
            preds.append(out)
    return np.stack(preds)   # (N, C, nx, ny, neta, T)

torch_out = run_torch_inference(torch_model, data_test_norm, device)
print(f"Output : {torch_out.shape}")

# ── MLX inference ─────────────────────────────────────────────────────────────

print("\n── MLX inference ─────────────────────────────────────────────────")
import mlx.core as mx
import mlx.utils as mlx_utils
from neuralop_mlx.models import FNO as MlxFNO
from neuralop_mlx.weight_transfer import load_from_torch

mlx_model = MlxFNO(
    in_channels=4, out_channels=4,
    n_modes=list(n_modes),
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
)
load_from_torch(str(CKPT_PATH), mlx_model)
mlx_model.eval()
_all_mlx    = list(mlx_utils.tree_flatten(mlx_model.parameters()))
_mlx_total  = sum(int(np.prod(p.shape)) for _, p in _all_mlx)
_mlx_ri     = sum(int(np.prod(p.shape)) for k, p in _all_mlx
                  if k.endswith("weight_r") or k.endswith("weight_i"))
_mlx_real   = _mlx_total - _mlx_ri
_mlx_cmplx  = _mlx_ri // 2 + _mlx_real   # only halve the r/i pairs, leave real params as-is
print(f"Params : {_mlx_total:,}")
print(f"       = {_mlx_cmplx:,} complex-equivalent  "
      f"(spectral conv weight_r+weight_i pairs halved: {_mlx_ri:,} → {_mlx_ri//2:,}; "
      f"real params unchanged: {_mlx_real:,})")

def run_mlx_inference(model, data):
    preds = []
    for i in range(data.shape[0]):
        x0  = data[i, :, :, :, :, :1].astype(np.float32)
        T   = data.shape[-1] - 1
        x   = np.repeat(x0, T, axis=-1)
        xin = mx.array(x[np.newaxis])
        out = model(xin)
        mx.eval(out)
        preds.append(np.array(out[0]))
    return np.stack(preds)   # (N, C, nx, ny, neta, T)

mlx_out = run_mlx_inference(mlx_model, data_test_norm)
print(f"Output : {mlx_out.shape}")

# ── inverse normalisation → physical units ────────────────────────────────────

def to_physical(model_out, data_norm, norm):
    if norm is not None:
        x0_all      = data_norm[:, :, :, :, :, :1]
        full        = np.concatenate([x0_all, model_out], axis=-1)
        phys        = norm.inverse(full)[:, :, :, :, :, 1:]
        truth_phys  = norm.inverse(data_norm)[:, :, :, :, :, 1:]
    else:
        phys        = model_out
        truth_phys  = data_norm[:, :, :, :, :, 1:]
    return phys, truth_phys

torch_phys, truth_phys = to_physical(torch_out,  data_test_norm, normalizer)
mlx_phys,   _          = to_physical(mlx_out,    data_test_norm, normalizer)
print(f"\nPhysical shapes — torch: {torch_phys.shape}, mlx: {mlx_phys.shape}, truth: {truth_phys.shape}")

# ── numerical comparison ──────────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════════")
print("  Numerical tensor comparison  (raw model outputs, all events)")
print("══════════════════════════════════════════════════════════")

diff       = mlx_phys - torch_phys
abs_diff   = np.abs(diff)
rel_base   = np.maximum(np.abs(torch_phys), 1e-12)

max_abs    = float(abs_diff.max())
mean_abs   = float(abs_diff.mean())
rms_diff   = float(np.sqrt(np.mean(diff**2)))
max_rel    = float((abs_diff / rel_base).max())
mean_rel   = float((abs_diff / rel_base).mean())

# flatten for cosine similarity
pt_flat    = torch_phys.ravel()
mx_flat    = mlx_phys.ravel()
cos_sim    = float(np.dot(pt_flat, mx_flat) /
                   (np.linalg.norm(pt_flat) * np.linalg.norm(mx_flat) + 1e-30))

# per-channel stats (channel axis = 1)
ch_names   = ["ε", "vx", "vy", "vη"]
print(f"\n{'Metric':<28} {'Value':>14}")
print("-" * 44)
print(f"{'Max abs diff':<28} {max_abs:>14.6e}")
print(f"{'Mean abs diff':<28} {mean_abs:>14.6e}")
print(f"{'RMS diff':<28} {rms_diff:>14.6e}")
print(f"{'Max rel diff':<28} {max_rel:>14.6e}")
print(f"{'Mean rel diff':<28} {mean_rel:>14.6e}")
print(f"{'Cosine similarity':<28} {cos_sim:>14.10f}")
print(f"{'Fraction |diff|<1e-4':<28} {float((abs_diff < 1e-4).mean()):>14.6f}")
print(f"{'Fraction |diff|<1e-3':<28} {float((abs_diff < 1e-3).mean()):>14.6f}")
print(f"{'Fraction |diff|<1e-2':<28} {float((abs_diff < 1e-2).mean()):>14.6f}")

print("\nPer-channel max abs diff:")
for c, name in enumerate(ch_names):
    print(f"  {name:>4}  {float(abs_diff[:, c].max()):>.6e}")

print("\nPer-channel mean abs diff:")
for c, name in enumerate(ch_names):
    print(f"  {name:>4}  {float(abs_diff[:, c].mean()):>.6e}")

# histogram of absolute differences
print("\nAbsolute diff histogram (log10 bins):")
edges    = np.logspace(-8, 0, 10)
counts, _ = np.histogram(abs_diff.ravel(), bins=edges)
total    = abs_diff.size
for lo, hi, cnt in zip(edges[:-1], edges[1:], counts):
    bar = "█" * int(40 * cnt / max(counts))
    print(f"  [{lo:.0e}, {hi:.0e})  {cnt:>10d}  ({100*cnt/total:5.1f}%)  {bar}")

# ── output directory ──────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 2-D x/y energy figures — one subdirectory per event ──────────────────────

n_events = truth_phys.shape[0]
eta_val  = eta_coords[IETA_PLOT]
n_t      = len(T_BINS_PLOT)

print(f"\n── Generating 2-D x/y energy figures ({n_events} events) ────────────")

for event in range(n_events):
    ev_dir = OUTPUT_DIR / f"ev_{event}"
    ev_dir.mkdir(exist_ok=True)
    print(f"  Event {event} → {ev_dir.name}/")

    # Figure 1: Truth | PyTorch | MLX | |Δ| — one file per t bin
    for t in T_BINS_PLOT:
        tau_val = tau_coords[t]
        tr  = truth_phys[event, CHANNEL, :, :, IETA_PLOT, t]
        pt  = torch_phys[event, CHANNEL, :, :, IETA_PLOT, t]
        mx_ = mlx_phys[event,  CHANNEL, :, :, IETA_PLOT, t]
        dif = np.abs(mx_ - pt)

        vmin = min(tr.min(), pt.min(), mx_.min())
        vmax = max(tr.max(), pt.max(), mx_.max())
        kw   = dict(origin="lower", extent=xy_extent, aspect="equal",
                    vmin=vmin, vmax=vmax, cmap="hot")

        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        fig.suptitle(
            f"η bin {IETA_PLOT} (η={eta_val:.2f}),  τ step {t}  (τ={tau_val:.3f} fm/c),  "
            f"event {event}",
            fontsize=12, y=1.02,
        )
        for ax, dat, title in [
            (axes[0], tr,  "Truth"),
            (axes[1], pt,  "PyTorch"),
            (axes[2], mx_, "MLX"),
        ]:
            im = ax.imshow(dat.T, **kw)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("x (fm)"); ax.set_ylabel("y (fm)")
            plt.colorbar(im, ax=ax, label=CH_LABEL, fraction=0.046, pad=0.04)

        im_d = axes[3].imshow(dif.T, origin="lower", extent=xy_extent, aspect="equal",
                              cmap="viridis")
        axes[3].set_title(f"|MLX − PyTorch|\nmax={dif.max():.3e}", fontsize=11)
        axes[3].set_xlabel("x (fm)"); axes[3].set_ylabel("y (fm)")
        plt.colorbar(im_d, ax=axes[3], label=CH_LABEL, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(ev_dir / f"xy_energy_eta{IETA_PLOT}_t{t:03d}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Figure 2: truth vs backend grid — all t bins in one figure
    for label, data in [("torch_vs_truth", torch_phys), ("mlx_vs_truth", mlx_phys)]:
        backend_label = "PyTorch" if "torch" in label else "MLX"
        fig, axes = plt.subplots(2, n_t, figsize=(4.8 * n_t, 8.5), squeeze=False)
        fig.suptitle(
            f"{backend_label} vs Truth — η bin {IETA_PLOT} (η={eta_val:.2f}),  event {event}",
            fontsize=13, y=1.01,
        )
        for col, t in enumerate(T_BINS_PLOT):
            tau_val = tau_coords[t]
            tr = truth_phys[event, CHANNEL, :, :, IETA_PLOT, t]
            pr = data[event,       CHANNEL, :, :, IETA_PLOT, t]
            vmin = min(tr.min(), pr.min())
            vmax = max(tr.max(), pr.max())
            kw = dict(origin="lower", extent=xy_extent, aspect="equal",
                      vmin=vmin, vmax=vmax, cmap="hot")

            im = axes[0, col].imshow(tr.T, **kw)
            axes[0, col].set_title(f"Truth  τ={tau_val:.3f} fm/c", fontsize=10)
            axes[0, col].set_xlabel("x (fm)"); axes[0, col].set_ylabel("y (fm)")
            plt.colorbar(im, ax=axes[0, col], label=CH_LABEL, fraction=0.046, pad=0.04)

            im = axes[1, col].imshow(pr.T, **kw)
            axes[1, col].set_title(f"{backend_label}  τ={tau_val:.3f} fm/c", fontsize=10)
            axes[1, col].set_xlabel("x (fm)"); axes[1, col].set_ylabel("y (fm)")
            plt.colorbar(im, ax=axes[1, col], label=CH_LABEL, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(ev_dir / f"grid_{label}_eta{IETA_PLOT}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Figure 3: PyTorch vs MLX direct comparison — all t bins
    fig, axes = plt.subplots(3, n_t, figsize=(4.8 * n_t, 11), squeeze=False)
    fig.suptitle(
        f"PyTorch vs MLX (weight transfer)  —  "
        f"η bin {IETA_PLOT} (η={eta_val:.2f}),  event {event}",
        fontsize=13, y=1.01,
    )
    for col, t in enumerate(T_BINS_PLOT):
        tau_val = tau_coords[t]
        pt  = torch_phys[event, CHANNEL, :, :, IETA_PLOT, t]
        mx_ = mlx_phys[event,  CHANNEL, :, :, IETA_PLOT, t]
        dif = np.abs(mx_ - pt)
        vmin = min(pt.min(), mx_.min())
        vmax = max(pt.max(), mx_.max())
        kw = dict(origin="lower", extent=xy_extent, aspect="equal",
                  vmin=vmin, vmax=vmax, cmap="hot")

        im = axes[0, col].imshow(pt.T, **kw)
        axes[0, col].set_title(f"PyTorch  τ={tau_val:.3f} fm/c", fontsize=10)
        axes[0, col].set_xlabel("x (fm)"); axes[0, col].set_ylabel("y (fm)")
        plt.colorbar(im, ax=axes[0, col], label=CH_LABEL, fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(mx_.T, **kw)
        axes[1, col].set_title(f"MLX  τ={tau_val:.3f} fm/c", fontsize=10)
        axes[1, col].set_xlabel("x (fm)"); axes[1, col].set_ylabel("y (fm)")
        plt.colorbar(im, ax=axes[1, col], label=CH_LABEL, fraction=0.046, pad=0.04)

        im_d = axes[2, col].imshow(dif.T, origin="lower", extent=xy_extent, aspect="equal",
                                   cmap="viridis")
        axes[2, col].set_title(f"|Δ| max={dif.max():.3e}", fontsize=10)
        axes[2, col].set_xlabel("x (fm)"); axes[2, col].set_ylabel("y (fm)")
        plt.colorbar(im_d, ax=axes[2, col], label=CH_LABEL, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(ev_dir / f"torch_vs_mlx_eta{IETA_PLOT}_direct.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

# ── per-t numerical comparison table ─────────────────────────────────────────

print(f"\n── Per-τ-bin comparison (energy channel, η bin {IETA_PLOT}, all test events) ──")
print(f"{'t':>4}  {'τ (fm/c)':>9}  {'max|Δ|':>12}  {'mean|Δ|':>12}  {'RMS Δ':>12}  {'cos-sim':>12}")
print("-" * 70)
for t in range(ntau):
    pt_t  = torch_phys[:, CHANNEL, :, :, IETA_PLOT, t]
    mx_t  = mlx_phys[:,  CHANNEL, :, :, IETA_PLOT, t]
    d     = np.abs(mx_t - pt_t)
    rms   = float(np.sqrt(np.mean((mx_t - pt_t)**2)))
    cs    = float(np.dot(pt_t.ravel(), mx_t.ravel()) /
                  (np.linalg.norm(pt_t) * np.linalg.norm(mx_t) + 1e-30))
    marker = " ◄" if t in T_BINS_PLOT else ""
    print(f"{t:>4}  {tau_coords[t]:>9.4f}  {d.max():>12.4e}  {d.mean():>12.4e}  "
          f"{rms:>12.4e}  {cs:>12.10f}{marker}")

# ── save numerical summary ────────────────────────────────────────────────────

import json
summary = {
    "checkpoint": str(CKPT_PATH),
    "n_test_events": int(data_test_norm.shape[0]),
    "output_shape": list(torch_phys.shape),
    "max_abs_diff": max_abs,
    "mean_abs_diff": mean_abs,
    "rms_diff": rms_diff,
    "max_rel_diff": max_rel,
    "mean_rel_diff": mean_rel,
    "cosine_similarity": cos_sim,
    "frac_diff_lt_1e-4": float((abs_diff < 1e-4).mean()),
    "frac_diff_lt_1e-3": float((abs_diff < 1e-3).mean()),
    "frac_diff_lt_1e-2": float((abs_diff < 1e-2).mean()),
    "per_channel_max_abs": {ch_names[c]: float(abs_diff[:, c].max()) for c in range(4)},
    "per_channel_mean_abs": {ch_names[c]: float(abs_diff[:, c].mean()) for c in range(4)},
}
summary_path = OUTPUT_DIR / "comparison_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved → {summary_path}")
print("\nDone.")
