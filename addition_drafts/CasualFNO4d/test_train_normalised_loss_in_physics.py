#!/usr/bin/env python
"""
test_train_normalised.py
========================

Extension of test_train.py with:
  1. Bjorken + z-score normalisation (NormFn3d pipeline).
  2. τ sinusoidal embedding as extra input channels.
  3. Loss and evaluation computed in PHYSICAL units via PhysicalUnitLoss.

Differences against test_train.py are marked with  # ── NORM:  comments.
"""

# ===========================================================================
# Training parameters
# ===========================================================================

odir         = './'
input_dir    = '/Users/du8478/FNO_Hydro_Data/flat'
input_files  = ['dAu_25ev_mb.root']

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

lr            = 8e-3
weight_decay  = 1e-4

# ── NORM: τ embedding configuration ─────────────────────────────────────────
n_emb       = 8        # must be even
max_period  = 20.0     # fm/c — should cover the full simulation duration
# ─────────────────────────────────────────────────────────────────────────────

log = open(f'{odir}/log.txt', 'w')
for k, v in dict(input_dir=input_dir, input_files=input_files,
                  n_epochs=n_epochs, batch_size=batch_size,
                  hidden_channels=hidden_channels, n_modes=n_modes,
                  train_ratio=train_ratio, xyslice=xyslice, tauslice=tauslice,
                  nevents_in=nevents_in, lr=lr, weight_decay=weight_decay,
                  n_emb=n_emb, max_period=max_period).items():
    log.write(f'{k}: {v}\n')


# ===========================================================================
# Imports
# ===========================================================================

import os, sys, gc, atexit
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralop.models import FNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.training import AdamW
from neuralop.utils import count_model_params

sys.path.append('/Users/du8478//FNO/Dave/loc_libs')
from read_3d_root import read_3d_data

# ── NORM ─────────────────────────────────────────────────────────────────────
from NormFn3d import NormFn3d
# ─────────────────────────────────────────────────────────────────────────────

t0 = perf_counter()


# ===========================================================================
# NORM: PhysicalUnitLoss
# Wraps any base loss (LpLoss, H1Loss) so that pred and target are
# inverse-normalised back to physical units before the loss is computed.
#
# This ensures:
#   • the optimiser targets physically meaningful errors;
#   • any PINO-style PDE residual term can be appended here in the same frame.
# ===========================================================================

class PhysicalUnitLoss(nn.Module):
    """
    Inverse-normalise (z-score then Bjorken) both prediction and target,
    then evaluate a base loss in physical units.

    The T_out τ steps in pred / target are assumed to correspond to τ indices
    tau_start_idx, tau_start_idx+1, … (default: 1, because index 0 is the IC
    which is not predicted).

    Parameters
    ----------
    base_loss     : callable(pred, target) → scalar — e.g. LpLoss or H1Loss
    mu, sigma     : (n_features, n_tau)  from NormFn3d.compute_zscore_stats
    shape_ref     : (n_eta, n_tau)       from NormFn3d.compute_ref_etau75_eta_shape
    tau_min, dtau : proper-time grid parameters [fm/c]
    tau_start_idx : first τ index present in the prediction tensor (default 1)
    eps           : floor on τ to guard against division by zero
    """

    def __init__(self, base_loss, mu, sigma, shape_ref,
                 tau_min, dtau, tau_start_idx=1, eps=1e-8):
        super().__init__()
        self.register_buffer('mu',        torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer('sigma',     torch.from_numpy(sigma.astype(np.float32)))
        self.register_buffer('shape_ref', torch.from_numpy(shape_ref.astype(np.float32)))
        self.base_loss     = base_loss
        self.tau_min       = tau_min
        self.dtau          = dtau
        self.tau_start_idx = tau_start_idx
        self.eps           = eps

    def invert(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm : (B, F, X, Y, η, T_out) — normalised model output or target
        returns: (B, F, X, Y, η, T_out) — physical units
        """
        B, F, X, Y, n_eta, T_out = x_norm.shape
        t0 = self.tau_start_idx

        # τ values for the output window  (1, 1, 1, 1, 1, T_out)
        tau_idx = torch.arange(t0, t0 + T_out,
                               device=x_norm.device, dtype=x_norm.dtype)
        tau = (self.tau_min + tau_idx * self.dtau).reshape(1, 1, 1, 1, 1, T_out)

        # Per-(feature, τ) stats for the output window  (1, F, 1, 1, 1, T_out)
        mu_w    = self.mu   [:, t0:t0 + T_out].reshape(1, F, 1, 1, 1, T_out)
        sigma_w = self.sigma[:, t0:t0 + T_out].reshape(1, F, 1, 1, 1, T_out)

        # shape_ref for the output window  (1, 1, 1, 1, η, T_out)
        sr_w = self.shape_ref[:, t0:t0 + T_out].reshape(1, 1, 1, 1, n_eta, T_out)

        # Step 1: inverse z-score on ALL features
        x_bj = x_norm * sigma_w + mu_w              # (B, F, X, Y, η, T_out)

        # Step 2: inverse Bjorken on ed (channel 0) only
        ed_bj   = x_bj[:, 0:1]                      # (B, 1, X, Y, η, T_out)
        ed_phys = (ed_bj * sr_w / (tau + self.eps)) ** (1.0 / 0.75)

        return torch.cat([ed_phys, x_bj[:, 1:]], dim=1)

    def forward(self, pred_norm: torch.Tensor,
                      target_norm: torch.Tensor) -> torch.Tensor:
        return self.base_loss(self.invert(pred_norm),
                              self.invert(target_norm))

    # ------------------------------------------------------------------
    # PINO extension point
    # ------------------------------------------------------------------
    # To add a PDE residual loss (∂_μ T^{μν} = 0 in the physical frame),
    # override forward as shown below.  The key constraint is that the
    # residual MUST be evaluated on physical-unit fields, because the
    # τ-dependent Bjorken rescaling introduces spurious source terms
    # in the rescaled frame.
    #
    #   def forward(self, pred_norm, target_norm, lambda_pde=1e-3):
    #       pred_phys   = self.invert(pred_norm)
    #       target_phys = self.invert(target_norm)
    #       loss_data   = self.base_loss(pred_phys, target_phys)
    #       loss_pde    = self._pde_residual(pred_phys)      # in physical units
    #       return loss_data + lambda_pde * loss_pde
    #
    #   def _pde_residual(self, u_phys):
    #       # u_phys: (B, F, X, Y, η, T)
    #       # F = 0: ed  [GeV/fm³]
    #       # F = 1: vx  (or u^x)
    #       # F = 2: vy  (or u^y)
    #       # F = 3: vz  (or u^η)
    #       # Milne-frame ideal conservation:
    #       #   ∂_τ ε + (ε + p)/τ + ∇_⊥·((ε+p)v_⊥) + ∂_ηs(ε+p)v_η/τ = 0
    #       # Use torch.gradient or spectral differentiation to evaluate ∂_τ, ∂_x, etc.
    #       ...
    #       raise NotImplementedError


# ===========================================================================
# Data loading
# ===========================================================================

print(f'Events called for {nevents_in}')
arr_data = [read_3d_data(f'{input_dir}/{f}', nevents=nevents_in, local_rank=None)
            for f in input_files]
if not arr_data:
    raise ValueError('No data found in the specified input files.')

# ── NORM: read τ grid metadata ───────────────────────────────────────────────
tau_min = float(arr_data[0].get('tau_min', 0.4))
dtau    = float(arr_data[0].get('dtau',    0.1))
log.write(f'tau_min: {tau_min}\ndtau: {dtau}\n')
# ─────────────────────────────────────────────────────────────────────────────

arr = arr_data[0]['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]
del arr_data
gc.collect()

n_train    = int(arr.shape[0] * train_ratio)
data_train = arr[:n_train]
data_val   = arr[n_train:]
np.random.shuffle(data_train)
np.random.shuffle(data_val)

log.write(f'data_train  shape: {data_train.shape}\n')
log.write(f'data_verify shape: {data_val.shape}\n')

_, n_features, nx, ny, n_eta, n_tau = data_train.shape


# ===========================================================================
# NORM: compute statistics on train split, apply to both splits, save stats
# ===========================================================================

print('── Computing normalisation statistics ──')

shape_ref         = NormFn3d.compute_ref_etau75_eta_shape(data_train, tau_min, dtau)
train_bj, _       = NormFn3d.etau75_eta_shape_3d(data_train, tau_min, dtau, shape_ref)
val_bj, _         = NormFn3d.etau75_eta_shape_3d(data_val,   tau_min, dtau, shape_ref)
mu, sigma         = NormFn3d.compute_zscore_stats(train_bj)
train_norm        = NormFn3d.apply_zscore_3d(train_bj, mu, sigma)
val_norm          = NormFn3d.apply_zscore_3d(val_bj,   mu, sigma)
tau_emb           = NormFn3d.tau_sinusoidal_embedding(n_tau, tau_min, dtau,
                                                      n_emb=n_emb,
                                                      max_period=max_period)

np.savez(f'{odir}/norm_stats.npz',
         shape_ref=shape_ref, mu=mu, sigma=sigma, tau_emb=tau_emb,
         tau_min=tau_min, dtau=dtau, n_emb=n_emb, max_period=max_period)
print(f'Saved norm_stats.npz')

del data_train, data_val, train_bj, val_bj
gc.collect()


# ===========================================================================
# Dataset
# ===========================================================================

class NormalisedFluidDataset(Dataset):
    """
    Operates on already-normalised data (physical features only).
    Attaches τ-sinusoidal embedding channels on the fly, aligned to the
    prediction τ of each output step.

    Returns
    -------
    x : (n_features + n_emb, X, Y, η, T-1)
        IC repeated across T-1 steps  +  τ embedding for steps 1…T-1.
    y : (n_features, X, Y, η, T-1)
        Actual normalised evolution at τ=1…T-1.
    """

    def __init__(self, data_norm: np.ndarray, tau_emb: np.ndarray):
        self.data       = data_norm
        self.tau_emb    = tau_emb
        self.n_samples  = data_norm.shape[0]
        self.n_features = data_norm.shape[1]
        self.nx         = data_norm.shape[2]
        self.ny         = data_norm.shape[3]
        self.n_eta      = data_norm.shape[4]
        self.n_tau      = data_norm.shape[5]
        self.n_emb      = tau_emb.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Physical IC repeated across T-1 prediction steps
        x_ic = self.data[idx, :, :, :, :, :1]
        x_ic = np.repeat(x_ic, self.n_tau - 1, axis=-1)        # (F, X, Y, η, T-1)

        # τ embedding for τ steps 1 … T-1, broadcast to spatial dims
        emb = self.tau_emb[:, 1:self.n_tau]                     # (E, T-1)
        emb_b = np.broadcast_to(
            emb[:, np.newaxis, np.newaxis, np.newaxis, :],
            (self.n_emb, self.nx, self.ny, self.n_eta, self.n_tau - 1),
        )

        x = np.concatenate([x_ic, emb_b], axis=0)               # (F+E, X, Y, η, T-1)
        y = self.data[idx, :, :, :, :, 1:self.n_tau]            # (F,   X, Y, η, T-1)
        return {'x': torch.FloatTensor(x), 'y': torch.FloatTensor(y)}


dataset_train = NormalisedFluidDataset(train_norm, tau_emb)
dataset_val   = NormalisedFluidDataset(val_norm,   tau_emb)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loaders = {
    100: DataLoader(dataset_val, batch_size=10, shuffle=False,
                    num_workers=0, pin_memory=False, persistent_workers=False),
}


# ===========================================================================
# Device
# ===========================================================================

device = 'cpu'
print(f'Using device: {device}')


# ===========================================================================
# Model
# ===========================================================================

# ── NORM: in_channels = physical features + τ embedding channels ─────────────
in_channels  = n_features + n_emb
out_channels = n_features
# ─────────────────────────────────────────────────────────────────────────────

model = FNO(
    in_channels=in_channels,
    out_channels=out_channels,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
).to(device)

total_params = count_model_params(model)
memory_MB    = total_params * 4 / (1024 ** 2)
print(f'Parameters: {total_params:,}  ({memory_MB:.2f} MB)')
log.write(f'in_channels: {in_channels}\nout_channels: {out_channels}\n')
log.write(f'Parameters: {total_params}\nModel memory: {memory_MB:.2f} MB\n')
log.close()


# ===========================================================================
# NORM: Loss in physical units
# ===========================================================================

_base_l2 = LpLoss(d=4, p=2)
_base_h1 = H1Loss(d=4)

# PhysicalUnitLoss inverts z-score and Bjorken rescaling on both pred and
# target before evaluating the underlying LpLoss / H1Loss.
# tau_start_idx=1 because the model output corresponds to τ steps 1…T-1.
phys_l2 = PhysicalUnitLoss(_base_l2, mu, sigma, shape_ref,
                            tau_min, dtau, tau_start_idx=1).to(device)
phys_h1 = PhysicalUnitLoss(_base_h1, mu, sigma, shape_ref,
                            tau_min, dtau, tau_start_idx=1).to(device)

train_loss  = phys_h1
eval_losses = {'h1': phys_h1, 'l2': phys_l2}
# ─────────────────────────────────────────────────────────────────────────────


# ===========================================================================
# Optimizer, scheduler, trainer
# ===========================================================================

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

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
    def _on_epoch_start(self, epoch):
        self.epoch = epoch
        if epoch % checkpoint_freq == 0 and epoch > 0:
            save_dir = f'{odir}/checkpoint_{epoch}'
            os.makedirs(save_dir, exist_ok=True)
            self.checkpoint(save_dir)
    trainer.on_epoch_start = _on_epoch_start.__get__(trainer, Trainer)


# ===========================================================================
# Logging
# ===========================================================================

if log_directly:
    class _Logger:
        def __init__(self, fn):
            self.terminal = sys.stdout
            self.log = open(fn, 'w')
        def write(self, m): self.terminal.write(m); self.log.write(m)
        def flush(self):    self.terminal.flush();  self.log.flush()

    sys.stdout = _Logger(f'{odir}/training.log')
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
# Inference template
# ===========================================================================
#
#   stats     = np.load(f'{odir}/norm_stats.npz')
#   shape_ref = stats['shape_ref']
#   mu, sigma = stats['mu'], stats['sigma']
#   tau_emb   = stats['tau_emb']
#
#   ic_bj     = NormFn3d.etau75_eta_shape_3d(ic_phys, tau_min, dtau, shape_ref)[0]
#   ic_norm   = NormFn3d.apply_zscore_3d(ic_bj, mu, sigma)
#   ds        = NormalisedFluidDataset(ic_norm, tau_emb)
#   x         = ds[0]['x'].unsqueeze(0).to(device)
#
#   with torch.no_grad():
#       pred_norm = model(x)
#
#   # Invert normalisation to physical units (z-score first, then Bjorken):
#   pred_bj   = NormFn3d.inv_zscore_3d(pred_norm.cpu().numpy(), mu, sigma)
#   pred_phys = NormFn3d.inv_etau75_eta_shape_3d(pred_bj, shape_ref, tau_min, dtau)
