#!/usr/bin/env python
"""
test_train_normalised.py
========================

Minimal extension of test_train.py that integrates the NormFn3d normalisation
pipeline.  Read README_normalisation.md alongside this file.

The diff against test_train.py is concentrated in three places, each marked
with `# ── NORM:` comments:

  1. After the train/val split:  compute Bjorken + z-score stats from TRAIN,
     apply to both splits, save the stats for inference.
  2. Replace FluidDataset with NormalisedFluidDataset, which attaches the
     τ sinusoidal embedding on the fly per sample.
  3. In the FNO constructor:  in_channels grows by n_emb (the embedding
     dimension).  out_channels stays the same (model predicts physical
     fields only, not the embedding).

Everything else — data loading, optimizer, scheduler, Trainer setup,
checkpointing, logging — is unchanged from test_train.py.
"""

# ===========================================================================
# Training parameters  (unchanged from test_train.py, except `n_emb` added)
# ===========================================================================

odir       = './'
input_dir  = '/Users/du8478/FNO_Hydro_Data/flat'
input_files = ['dAu_25ev_mb.root']

n_epochs        = 1
batch_size      = 1
hidden_channels = 16
n_modes         = [32, 32, 16, 16]
nevents_in      = 10
log_directly    = True
checkpoint_freq = None
train_ratio     = 0.8
xyslice         = slice(None, -1)
tauslice        = slice(None, -1)

lr              = 8e-3
weight_decay    = 1e-4

# ── NORM: τ embedding configuration ─────────────────────────────────────────
# n_emb must be even; max_period in fm/c should bracket the full sim duration.
n_emb           = 8
max_period      = 20.0
# ─────────────────────────────────────────────────────────────────────────────

log = open(f'{odir}/log.txt', 'w')
log.write(f'input dir: {input_dir}\n')
log.write(f'input files: {input_files}\n')
log.write(f'n_epochs: {n_epochs}\n')
log.write(f'batch_size: {batch_size}\n')
log.write(f'hidden_channels: {hidden_channels}\n')
log.write(f'checkpoint_freq: {checkpoint_freq}\n')
log.write(f'n_modes: {n_modes}\n')
log.write(f'train_ratio: {train_ratio}\n')
log.write(f'xyslice: {xyslice}\n')
log.write(f'tauslice: {tauslice}\n')
log.write(f'log_directly: {log_directly}\n')
log.write(f'nevents(in): {nevents_in}\n')
log.write(f'AdamW: lr: {lr}\n')
log.write(f'AdamW: weight_decay: {weight_decay}\n')
log.write(f'n_emb: {n_emb}\n')
log.write(f'max_period: {max_period}\n')


# ===========================================================================
# Imports
# ===========================================================================

import os
import sys
import gc
import atexit
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import uproot as up
import awkward as ak
import matplotlib.pyplot as plt

from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.utils import count_model_params

sys.path.append('/Users/du8478//FNO/Dave/loc_libs')
from read_3d_root import read_3d_data

# ── NORM: import the normalisation helpers ──────────────────────────────────
from NormFn3d import NormFn3d
# ─────────────────────────────────────────────────────────────────────────────

t0 = perf_counter()


# ===========================================================================
# Data loading  (unchanged from test_train.py)
# ===========================================================================

print(f'Events called for {nevents_in}')
arr_data = [read_3d_data(f'{input_dir}/{f}', nevents=nevents_in, local_rank=None)
            for f in input_files]
if len(arr_data) == 0:
    raise ValueError("No data found in the specified input files.")

# ── NORM: read tau_min and dtau from the data dictionary ────────────────────
# read_3d_data exposes the proper-time grid metadata; we need it for Bjorken
# rescaling and for the τ embedding.  Adjust the key names if your reader uses
# different conventions.
tau_min = float(arr_data[0].get('tau_min', 0.4))
dtau    = float(arr_data[0].get('dtau',    0.1))
log.write(f'tau_min: {tau_min}\n')
log.write(f'dtau: {dtau}\n')
# ─────────────────────────────────────────────────────────────────────────────

arr = arr_data[0]['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]
del arr_data
gc.collect()

log.write(f'nevents (constrained by input file): {arr.shape[0]}\n')

n_train     = int(arr.shape[0] * train_ratio)
data_train  = arr[:n_train]
data_verify = arr[n_train:]
np.random.shuffle(data_train)
np.random.shuffle(data_verify)

print('shapes train: ',  data_train.shape)
print('shapes verify: ', data_verify.shape)
log.write(f'data_train  shape: {data_train.shape}\n')
log.write(f'data_verify shape: {data_verify.shape}\n')

n_events_train, n_features, nx, ny, n_eta, n_tau = data_train.shape


# ===========================================================================
# NORM: Compute normalisation statistics from the TRAIN split only,
#       apply to both splits, save stats for inference.
# ===========================================================================

print('\n── Computing normalisation statistics from train split ──')

# Step 1: Bjorken family on ed.  Using shape-preserving η correction; switch
# to etau75_3d if you do not want any η rescaling.
shape_ref = NormFn3d.compute_ref_etau75_eta_shape(data_train, tau_min, dtau)
train_bj, _ = NormFn3d.etau75_eta_shape_3d(data_train,  tau_min, dtau, shape_ref)
verify_bj, _ = NormFn3d.etau75_eta_shape_3d(data_verify, tau_min, dtau, shape_ref)

# Step 2: per-(feature, τ) z-score on ALL channels, computed on Bjorken-rescaled
# training data, applied to both splits with the same stats.
mu, sigma = NormFn3d.compute_zscore_stats(train_bj)
train_norm  = NormFn3d.apply_zscore_3d(train_bj,  mu, sigma)
verify_norm = NormFn3d.apply_zscore_3d(verify_bj, mu, sigma)

# Step 3: build the τ embedding once.  It does not depend on the data.
tau_emb = NormFn3d.tau_sinusoidal_embedding(n_tau, tau_min, dtau,
                                            n_emb=n_emb, max_period=max_period)

# Save stats so inference can use the same normalisation.
np.savez(f'{odir}/norm_stats.npz',
         shape_ref=shape_ref, mu=mu, sigma=sigma,
         tau_emb=tau_emb,
         tau_min=tau_min, dtau=dtau, n_emb=n_emb, max_period=max_period)
print(f'Saved norm stats to {odir}/norm_stats.npz')

# Quick sanity checks (printed; useful for catching off-by-one or wrong tau_min)
mean_per_tau = train_norm.mean(axis=(0, 2, 3, 4))      # (n_features, n_tau)
std_per_tau  = train_norm.std (axis=(0, 2, 3, 4))      # (n_features, n_tau)
print(f'After z-score:  mean per (feat, τ) max abs = {np.abs(mean_per_tau).max():.2e}  (~0)')
print(f'After z-score:  std  per (feat, τ) mean    = {std_per_tau.mean():.4f}        (~1)')

del data_train, data_verify, train_bj, verify_bj
gc.collect()


# ===========================================================================
# Dataset
# ===========================================================================

class NormalisedFluidDataset(Dataset):
    """Variant of FluidDataset that:

      - operates on already-normalised numpy data (physical features only,
        no τ embedding yet);
      - constructs τ-embedding channels on the fly per sample so the
        embedding aligns with the prediction τ of each output step;
      - returns x with shape (n_features + n_emb, X, Y, η, T-1) and
        y with shape (n_features, X, Y, η, T-1).
    """

    def __init__(self, data_norm: np.ndarray, tau_emb: np.ndarray):
        """
        Parameters
        ----------
        data_norm : (n_events, n_features, nx, ny, n_eta, n_tau)
                    Already normalised (Bjorken + z-score applied).
        tau_emb   : (n_emb, n_tau)
                    From NormFn3d.tau_sinusoidal_embedding.
        """
        self.data       = data_norm
        self.tau_emb    = tau_emb
        self.n_samples  = data_norm.shape[0]
        self.n_features = data_norm.shape[1]
        self.nx         = data_norm.shape[2]
        self.ny         = data_norm.shape[3]
        self.n_eta      = data_norm.shape[4]
        self.n_tau      = data_norm.shape[5]
        self.n_emb      = tau_emb.shape[0]
        print(f'NormalisedFluidDataset: {self.n_samples} events, '
              f'{self.n_features} physical + {self.n_emb} embed channels, '
              f'τ steps {self.n_tau}')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Physical IC at τ=0, repeated across the T-1 prediction steps.
        x_ic = self.data[idx, :, :, :, :, :1]                          # (F, X, Y, η, 1)
        x_ic = np.repeat(x_ic, self.n_tau - 1, axis=-1)                # (F, X, Y, η, T-1)

        # τ embedding for the prediction steps τ=1 … T-1, broadcast spatially.
        emb = self.tau_emb[:, 1:self.n_tau]                            # (E, T-1)
        emb_b = np.broadcast_to(
            emb[:, np.newaxis, np.newaxis, np.newaxis, :],
            (self.n_emb, self.nx, self.ny, self.n_eta, self.n_tau - 1)
        )

        # Concat along channel axis: physical IC + τ embedding.
        x = np.concatenate([x_ic, emb_b], axis=0)                      # (F+E, X, Y, η, T-1)

        # Target: actual normalised evolution at τ=1 … T-1.
        y = self.data[idx, :, :, :, :, 1:self.n_tau]                   # (F,   X, Y, η, T-1)

        return {'x': torch.FloatTensor(x), 'y': torch.FloatTensor(y)}


dataset_train = NormalisedFluidDataset(train_norm,  tau_emb)
dataset_test  = NormalisedFluidDataset(verify_norm, tau_emb)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False)

# neuralop Trainer expects a dict of test loaders keyed by resolution.
test_loaders = {
    100: DataLoader(dataset_test, batch_size=10, shuffle=False,
                    num_workers=0, pin_memory=False, persistent_workers=False),
}


# ===========================================================================
# Device
# ===========================================================================

device = 'cpu'   # adjust as in test_train.py
print(f'Using device: {device}')


# ===========================================================================
# Model
# ===========================================================================

# ── NORM: in_channels is now n_features + n_emb.  out_channels is unchanged ──
in_channels  = n_features + n_emb
out_channels = n_features
print(f'FNO in_channels  = {in_channels}  ({n_features} physical + {n_emb} τ-embed)')
print(f'FNO out_channels = {out_channels} (physical only)')

model = FNO(
    in_channels=in_channels,
    out_channels=out_channels,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
).to(device)
# ─────────────────────────────────────────────────────────────────────────────

total_params = count_model_params(model)
memory_MB    = total_params * 4 / (1024 ** 2)
print(f'Total parameters: {total_params:,}')
print(f'Model memory:     {memory_MB:.2f} MB')
log.write(f'Total parameters: {total_params}\n')
log.write(f'Model memory: {memory_MB:.2f} MB\n')
log.close()


# ===========================================================================
# Optimizer, loss, trainer
# ===========================================================================

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# ── NORM: loss is computed in NORMALISED units (recommended starting point) ──
#
# Per-(feature, τ) z-scoring already balances loss contributions across regimes,
# so an LpLoss in normalised units is a sensible default.
#
# If you want to compute the loss in PHYSICAL units instead (e.g. to add a
# PINO-style PDE residual term, or to weight late-τ regions more heavily),
# wrap the loss to inverse-normalise prediction and target first.  Sketch:
#
#     class PhysicalLpLoss(torch.nn.Module):
#         def __init__(self, mu, sigma, shape_ref, tau_min, dtau, base_loss):
#             super().__init__()
#             self.register_buffer('mu',    torch.from_numpy(mu).float())
#             self.register_buffer('sigma', torch.from_numpy(sigma).float())
#             # ... store shape_ref, tau_min, dtau, base_loss ...
#         def forward(self, pred_norm, target_norm):
#             pred_phys   = inv_norm_torch(pred_norm,   ...)
#             target_phys = inv_norm_torch(target_norm, ...)
#             return self.base_loss(pred_phys, target_phys)
#
# The torch versions of inv_zscore_3d and inv_etau75_eta_shape_3d are simple
# vectorised transcriptions of the numpy code in NormFn3d.

l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)

train_loss  = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}
# ─────────────────────────────────────────────────────────────────────────────


trainer = Trainer(
    model=model,
    n_epochs=n_epochs,
    device=device,
    wandb_log=False,
    eval_interval=2,
    use_distributed=False,
    verbose=True,
)

if checkpoint_freq is not None and checkpoint_freq > 0:
    def custom_on_epoch_start(self, epoch):
        self.epoch = epoch
        if self.epoch % checkpoint_freq == 0 and self.epoch > 0:
            save_dir = f'{odir}/checkpoint_{self.epoch}'
            os.makedirs(save_dir, exist_ok=True)
            self.checkpoint(save_dir)
            print('saving checkpoint to ', save_dir)
    trainer.on_epoch_start = custom_on_epoch_start.__get__(trainer, Trainer)


# ===========================================================================
# Logging
# ===========================================================================

log_path = f'{odir}/training.log'
if log_directly:
    class _Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message); self.log.write(message)
        def flush(self):
            self.terminal.flush(); self.log.flush()

    sys.stdout = _Logger(log_path)
    atexit.register(lambda: (sys.stdout.log.close(),
                             setattr(sys, 'stdout', sys.stdout.terminal)))


# ===========================================================================
# Train
# ===========================================================================

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    save_dir=odir,
    save_best='100_l2',
    training_loss=train_loss,
    eval_losses=eval_losses,
)


# ===========================================================================
# NORM: how to use this model for inference / visualisation
# ===========================================================================
#
# At inference time, load the model AND the normalisation stats together:
#
#     stats = np.load('norm_stats.npz')
#     shape_ref = stats['shape_ref']
#     mu, sigma = stats['mu'], stats['sigma']
#     tau_emb   = stats['tau_emb']
#     tau_min   = float(stats['tau_min']); dtau = float(stats['dtau'])
#
# Then for a new IC `ic_phys` of shape (1, n_features, nx, ny, n_eta, n_tau):
#
#     ic_bj   = NormFn3d.etau75_eta_shape_3d(ic_phys, tau_min, dtau, shape_ref)[0]
#     ic_norm = NormFn3d.apply_zscore_3d(ic_bj, mu, sigma)
#
#     ds = NormalisedFluidDataset(ic_norm, tau_emb)
#     x  = ds[0]['x'].unsqueeze(0).to(device)         # (1, F+E, X, Y, η, T-1)
#
#     with torch.no_grad():
#         pred_norm = model(x).cpu().numpy()           # (1, F, X, Y, η, T-1)
#
#     # Inverse-normalise to physical units (REVERSE order: z-score first):
#     pred_bj   = NormFn3d.inv_zscore_3d(pred_norm, mu, sigma)
#     pred_phys = NormFn3d.inv_etau75_eta_shape_3d(pred_bj, shape_ref,
#                                                  tau_min, dtau)
#     # `pred_phys` is now in the same units as the original MUSIC output.
