#!/usr/bin/env python

# ---------------------------------------------------------------------------
# training parameters
# ---------------------------------------------------------------------------
odir = './'
input_dir = './'
input_files = ['dAu_25ev_mb.root']

n_epochs = 1
batch_size = 1
hidden_channels = 16
n_modes = [32, 32, 16, 16]

nevents_in = -1
log_directly = True
checkpoint_freq = None
train_ratio = 0.8
xyslice = slice(None, -1)
tauslice = slice(None, -1)

lr = 8e-3
weight_decay = 1e-4

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import sys
import gc
import numpy as np
import torch
from time import perf_counter
from torch.utils.data import DataLoader

from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss

sys.path.append('../loc_libs')
from read_3d_root import read_3d_data
from parse_training_log import plot_training_log
from datasets import FluidDataset
from memory_utils import calculate_model_memory
from training_utils import enable_log_tee, patch_trainer_checkpoint

t0 = perf_counter()

# ---------------------------------------------------------------------------
# log file
# ---------------------------------------------------------------------------
log = open(f'{odir}/log.log', 'w')
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
log.write(f'cpu_model_forward: {log_directly}\n')
log.write(f'nevents(in): {nevents_in}\n')
log.write(f'AdamW: lr: {lr}\n')
log.write(f'AdamW: weight_decay: {weight_decay}\n')

# ---------------------------------------------------------------------------
# load data
# ---------------------------------------------------------------------------
print(f'Events called for {nevents_in}')

arr_data = [read_3d_data(f'{input_dir}/{f}', nevents=nevents_in, local_rank=None) for f in input_files]

if len(arr_data) == 0:
    raise ValueError("No data found in the specified input files.")

log.write('Data parameters from {first_file}:\n')
for key in arr_data[0].keys():
    if key not in ('arr', 'tau_freezeout', 'ntau_freezeout'):
        log.write(f'  {key}: {arr_data[0][key]}\n')

arr = arr_data[0]['arr']
arr = arr[:, :, xyslice, xyslice, tauslice, :]
del arr_data
gc.collect()

log.write(f'nevents (constrained by input file): {arr.shape[0]}')

n_train = int(arr.shape[0] * train_ratio)
data_train = arr[:n_train]
data_verify = arr[n_train:]

np.random.shuffle(data_train)
np.random.shuffle(data_verify)

print('shapes train: ', data_train.shape)
print('shapes verify: ', data_verify.shape)
log.write(f'data_train shape: {data_train.shape}\n')
log.write(f'data_verify shape: {data_verify.shape}\n')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ---------------------------------------------------------------------------
# datasets and loaders
# ---------------------------------------------------------------------------
dataset_train = FluidDataset(data_train)
dataset_test = FluidDataset(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

test_resolutions = [100]
test_batch_sizes = [10]
test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    print("res", res)
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
print(f"Model memory usage: {memory_usage:.2f} MB")
log.write(f"Model memory usage: {memory_usage:.2f} MB\n")
log.close()

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
                  eval_interval=2,
                  use_distributed=False,
                  verbose=True)

if checkpoint_freq is not None and checkpoint_freq > 0:
    patch_trainer_checkpoint(trainer, Trainer, checkpoint_freq, odir)

if log_directly:
    enable_log_tee(f'{odir}/training.log')

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              save_dir=odir,
              save_best='100_l2',
              training_loss=train_loss,
              eval_losses=eval_losses)
