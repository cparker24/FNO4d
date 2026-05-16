#!/usr/bin/env python

# ---------------------------------------------------------------------------
# training parameters
# ---------------------------------------------------------------------------
odir = './'
input_dir = '/Users/du8478/FNO_Hydro_Data/flat'
input_files = ['dAu_25ev_mb.root']

n_epochs = 1
batch_size = 1
hidden_channels = 16
n_modes = [32, 32, 16, 16]

nevents_in = 10
log_directly = True
checkpoint_freq = None
train_ratio = 0.8
xyslice = slice(None, -1)
tauslice = slice(None, -1)

lr = 8e-3
weight_decay = 1e-4
test_batch_sizes = [10]
eval_interval = 2
save_last = True
model_dir = './'

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import sys
import gc
import torch
from time import perf_counter
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss

sys.path.insert(0, '../loc_libs')
from read_3d_root import read_3d_data
from fluid_datasets import FluidDataset
from memory_utils import calculate_model_memory, reset_peak_memory, log_peak_memory
from training_utils import RunLog, patch_trainer_logging, patch_trainer_checkpoint, save_last_checkpoint
from parse_training_log import TrainingHistory

log = RunLog(f'{odir}/training.log' if log_directly else None)

t0 = perf_counter()

# ---------------------------------------------------------------------------
# configuration summary
# ---------------------------------------------------------------------------
log(f'input dir: {input_dir}')
log(f'input files: {input_files}')
log(f'n_epochs: {n_epochs}')
log(f'batch_size: {batch_size}')
log(f'hidden_channels: {hidden_channels}')
log(f'checkpoint_freq: {checkpoint_freq}')
log(f'n_modes: {n_modes}')
log(f'train_ratio: {train_ratio}')
log(f'xyslice: {xyslice}')
log(f'tauslice: {tauslice}')
log(f'log_directly: {log_directly}')
log(f'nevents(in): {nevents_in}')
log(f'AdamW: lr: {lr}')
log(f'AdamW: weight_decay: {weight_decay}')

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
log(f'Events called for {nevents_in}')

if not input_files:
    raise ValueError("No data found in the specified input files.")
arr_data = read_3d_data([f'{input_dir}/{f}' for f in input_files], nevents=nevents_in, local_rank=None)

log('Data parameters from input file:')
for key in arr_data.keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log(f'  {key}: {arr_data[key]}')

arr = arr_data['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]
del arr_data
gc.collect()

log(f'nevents (constrained by input file): {arr.shape[0]}')
test_resolutions = [arr.shape[2]]

n_train = int(arr.shape[0] * train_ratio)
data_train = arr[:n_train]
data_verify = arr[n_train:]

log(f'shapes train:  {data_train.shape}')
log(f'shapes verify: {data_verify.shape}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
log(f'Using device: {device}')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
dataset_train = FluidDataset(data_train)
dataset_test = FluidDataset(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    log(f'res {res}')
    test_loaders[res] = DataLoader(dataset_test,
                                   batch_size=test_bsize,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False,
                                   persistent_workers=False)
gc.collect()

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
model = FNO(
    in_channels=4,
    out_channels=4,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2
).to(device)

memory_usage = calculate_model_memory(model)
log(f"Model memory usage: {memory_usage:.2f} MB")

# ---------------------------------------------------------------------------
# optimizer, losses, trainer
# ---------------------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)
train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

trainer = Trainer(model=model, n_epochs=n_epochs,
                  device=device,
                  wandb_log=False,
                  eval_interval=eval_interval,
                  use_distributed=False,
                  verbose=True)

history = TrainingHistory()
patch_trainer_logging(trainer, log, history=history)

if checkpoint_freq is not None and checkpoint_freq > 0:
    patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, model_dir)

save_best_key = f'{test_resolutions[0]}_l2'

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
reset_peak_memory(cuda=True)
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              save_dir=model_dir,
              save_best=save_best_key,
              training_loss=train_loss,
              eval_losses=eval_losses)

if save_last:
    save_last_checkpoint(trainer, optimizer, scheduler, n_epochs, model_dir)
log_peak_memory(log, cuda=True)
history.save(f'{odir}/metrics.json')
history.plot(save_path=f'{odir}/training_curves.svg', title='FNO Training')
log(f'metrics   → {odir}/metrics.json')
log(f'curves    → {odir}/training_curves.svg')
log(f'total time: {perf_counter() - t0:.1f}s')
