#!/usr/bin/env python

# ************************************************************************************************************
# MLX version of test_train.py — runs FNO training on Apple Silicon with full Metal GPU support.
# Port of test_train.py using neuralop_mlx instead of the PyTorch neuraloperator package.
# ************************************************************************************************************

# training parameters
odir = f'./'
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

# optimizer parameters
lr = 8e-3
weight_decay = 1e-4

log = open(f'{odir}/log_mlx.log', 'w')
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

import uproot as up
import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.optimizers as optim_mlx
import os
import atexit
import gc

from neuralop_mlx.models import FNO
from neuralop_mlx import Trainer, LpLoss, H1Loss, AdamW
from neuralop_mlx.utils import count_model_params

import psutil
from time import perf_counter

t0 = perf_counter()

import sys
sys.path.append('../loc_libs')
from read_3d_root import read_3d_data, coarsen_array
from parse_training_log import plot_training_log

from torch.utils.data import DataLoader, Dataset

def process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

class MemoryPrinter:
    def __init__(self, log=None, logfile=None):
        self.log = log
        if log:
            if not logfile:
                raise ValueError("If log is provided, logfile must also be provided.")
            self.logfile = logfile
            if not os.path.isfile(os.path.dirname(logfile)):
                with open(self.logfile, 'w') as f:
                    f.write("")

    def __call__(self, label=""):
        mem = process_memory() / (1024 ** 3)
        output = f"[Memory] {label}: {mem:.3f} GB"
        if self.log is not None:
            try:
                self.log.write(output + "\n")
            except:
                with open(self.logfile, 'a') as f:
                    f.write(output + "\n")
        print(output)


# input data
in_tree = 't'
in_branch = 'flat_data'
ratio = 0.8

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

n_train = int(arr.shape[0] * ratio)
data_train = arr[:n_train]
data_verify = arr[n_train:]

np.random.shuffle(data_train)
np.random.shuffle(data_verify)

print('shapes train: ', data_train.shape)
print('shapes verify: ', data_verify.shape)

log.write(f'data_train shape: {data_train.shape}\n')
log.write(f'data_verify shape: {data_verify.shape}\n')

#plt.imshow(data_train[0, 0, :, :, 0, 0])

# MLX uses 'gpu' for Metal acceleration (no MPS cache to clear)
device = 'gpu'
print(f'Using device: {device} (MLX Metal)')
print(f'Data train shape: {data_train.shape}\nData verify shape: {data_verify.shape}')


class FluidDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.time_steps = self.data.shape[-1]
        self.n_samples = self.data.shape[0]
        print("Numpy data shape: ", self.data.shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_initial = self.data[idx, :, :, :, :, :1]
        x = np.repeat(x_initial, self.time_steps - 1, axis=-1)
        y = self.data[idx, :, :, :, :, 1:self.time_steps]
        # Return numpy arrays; Trainer converts to mx.array
        return {'x': x.astype(np.float32), 'y': y.astype(np.float32)}

    def values(self):
        return self.data.shape


dataset_train = FluidDataset(data_train)
dataset_test = FluidDataset(data_verify)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

dataset_train[0]['x'].shape, dataset_test[0]['y'].shape, dataset_train[0]['y'].shape
dataset_test.values(), dataset_train.values()

test_resolutions = [100]
test_batch_sizes = [10]

test_loaders = {}
for res, test_bsize in zip(test_resolutions, test_batch_sizes):
    print("res", res)
    test_loaders[res] = DataLoader(
        dataset_test,
        batch_size=test_bsize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

gc.collect()

# Build model
model = FNO(
    in_channels=4,
    out_channels=4,
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    projection_channel_ratio=2,
)

def calculate_model_memory(model):
    total_params = count_model_params(model)
    print(f'Total parameters: {total_params}')
    total_memory_MB = total_params * 4 / (1024 ** 2)
    return total_memory_MB

memory_usage = calculate_model_memory(model)
print(f"Model memory usage: {memory_usage:.2f} MB")
log.write(f"Model memory usage: {memory_usage:.2f} MB\n")
log.close()

optimizer = AdamW(lr=lr, weight_decay=weight_decay)

# Cosine annealing scheduler
scheduler = optim_mlx.cosine_decay(lr, T_max := 30)

l2loss = LpLoss(d=4, p=2)
h1loss = H1Loss(d=4)

train_loss = h1loss
eval_losses = {'h1': h1loss, 'l2': l2loss}

trainer = Trainer(
    model=model,
    n_epochs=n_epochs,
    device=device,
    wandb_log=False,
    eval_interval=2,
    use_distributed=False,
    verbose=True,
)

log_path = f'{odir}/training_mlx.log'

if log_directly:
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_path)

    def restore_stdout():
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
    atexit.register(restore_stdout)

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
