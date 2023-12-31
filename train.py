import os
import random
import time
import math
from contextlib import nullcontext

# Hacks to make DP work on consumer GPU
# os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_P2P_DISABLE"] = "1"

# ML
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchaudio.transforms as T

# Local
from utils import spectogram, spectogram_raw, spectogram_mel_log, sliding_window
from datasets import preprocessed_audio_dataset, sample_dataset
from model import SuperVAD, Config

#
# Parameters
# 

init_from = "scratch" # or resume 
experiment = "supervad_lr_decay_1v"
device = "cuda:0"
model_config = Config()
train_batch = 16
train_epochs = 120
train_epoch_samples = 100000
train_lr = 1e-4
train_lr_min = 1e-5 # minimum learning rate, should be ~= train_lr/10 per Chinchilla
train_lr_decay = True
train_lr_warmup = 6000 # ~ 1 epoch
train_lr_decay = 600000 # ~ 100 epochs
train_weight_decay = 1e-2
train_betas = (0.9, 0.95)
augment_time_masking = 5
augment_freq_masking = 5
augment_rescale= (0.9, 1.1)
validation_batch = 64

#
# Device
# 

if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
enable_autocast = True
enable_anomaly_detection = False
parallel = False
compile = True

#
# Precision
# 

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # Using float32 since float16 sometimes not that stable
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
autocast = nullcontext() if device_type == 'cpu' or not enable_autocast else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
anomaly_detection = torch.autograd.detect_anomaly() if enable_anomaly_detection else nullcontext()
torch.set_float32_matmul_precision('high')

# Logging

writer = SummaryWriter(f'runs/{experiment}')

#
# Dataset
#

dataset_train = preprocessed_audio_dataset('./datasets/supervad-1/vad_train')
dataset_test = preprocessed_audio_dataset('./datasets/supervad-1/vad_test')

time_masking = T.TimeMasking(time_mask_param=augment_time_masking)
freq_masking = T.FrequencyMasking(freq_mask_param=augment_freq_masking)

def prepare_labels(labels):    

    # Sliding window
    labels = labels.to(device, non_blocking=True)
    labels = sliding_window(labels, model_config.ctx_length, 1)

    # Use the last element in each window as target (we want to predict current token, not anything else)
    A, B, C = labels.shape
    labels = labels.reshape(A * B, C)
    # labels = torch.mean(labels, dim = 1)    # Average labels over each window
    labels = labels[..., -1] # Get last value
    labels = labels.reshape(labels.shape[0], 1) # To suppress warnings
    
    return labels

def prepare_samples(samples, train=True):

    # Spectogram
    samples = samples.to(device, non_blocking=True)

    # Randomly rescale
    if train:
        scaling_factors = augment_rescale[0] + torch.rand(samples.shape[0], 1, device=device) * (augment_rescale[1] - augment_rescale[0])
        samples = samples * scaling_factors
    
    samples = spectogram_raw(samples)
    # samples = samples.to(device, non_blocking=True)

    # SpecAugment
    # if train:
    #     samples = time_masking(samples)
    #     samples = freq_masking(samples)

    # Log Mel
    samples = spectogram_mel_log(samples)
    # samples = spectogram(samples)
    
    # Sliding window
    samples = sliding_window(samples, model_config.ctx_length * 2, 2)

    # Normalize
    # samples = naive_normalize_spectogram(samples)

    # Reshape sliding window to batch
    A, B, C, D = samples.shape
    samples = samples.reshape(A * B, C, D)
    return samples

#
# Model
#

base = SuperVAD(model_config)
vad = base
if compile:
    vad = torch.compile(vad)
if parallel:
    vad = nn.DataParallel(vad)
vad.to(device)

#
# Optimizer
#

# Creating weight decay regularization
param_dict = {pn: p for pn, p in base.named_parameters()} # All parameters
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # Filter out non-grad parameters (like positional embedding)

# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': train_weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
print(f"Decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
print(f"Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

epoch = 0
iter = 0
current_lr = train_lr
optimizer = torch.optim.AdamW(optim_groups, lr=train_lr, betas=train_betas, fused=True)

#
# Data Loader
#

sampler = torch.utils.data.RandomSampler(dataset_train, replacement=True, num_samples=train_epoch_samples) # We are limiting number of samples
loader = DataLoader(dataset_train, batch_size=train_batch, num_workers=8, sampler=sampler, pin_memory=True)

#
# Save/Load
# 

def save():
    torch.save({ 'model_state_dict': base.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'iter': iter },  f'./checkpoints/{experiment}.pt')
    torch.save({ 'model_state_dict': base.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch, 'iter': iter},  f'./checkpoints/{experiment}_{epoch}.pt')
    
def load():
    global epoch
    global iter
    checkpoint = torch.load(f'./checkpoints/{experiment}.pt')
    base.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    iter = checkpoint['iter']
    print(f'Loaded at #{epoch}/{iter}')

# Do load if needed
if init_from == "resume":
    load()

#
# Validation
#

random.seed(42) # Predictable
validation_samples, validation_labels = [], []
for i in range(0, validation_batch):
    s, l = sample_dataset(dataset_test)
    validation_samples.append(s)
    validation_labels.append(l)
random.seed(None)
validation_labels = torch.stack(validation_labels)
validation_labels = prepare_labels(validation_labels)
validation_samples = torch.stack(validation_samples)
validation_samples = prepare_samples(validation_samples, train=False)

# Validation function
def validate():

    # Switch to validation
    vad.eval()

    # Validation loss
    with torch.no_grad():
        with autocast:
            prediction = vad(validation_samples)
        loss = F.mse_loss(validation_labels, prediction.float())
        return loss.item() / validation_labels.shape[0]

# Test validation
print(validate())

#
# Training
#

def resolve_train_lr(it):
    if train_lr_decay:
        # 1) linear warmup for train_lr_warmup steps
        if it < train_lr_warmup:
            return train_lr * (it + 1) / train_lr_warmup
        # 2) if it > train_lr_decay, return min learning rate
        if it > train_lr_decay:
            return train_lr_min
        
        # 3) cosine learning rate decay
        decay_ratio = (it - train_lr_warmup) / (train_lr_decay - train_lr_warmup)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return train_lr_min + coeff * (train_lr - train_lr_min)
    else:
        return train_lr
    
def train_batch(samples, labels):
    # Switch to training
    vad.train()

    # Prepare
    labels = prepare_labels(labels)
    samples = prepare_samples(samples)
    
    # Train
    with anomaly_detection:
        optimizer.zero_grad()
        with autocast:
            prediction = vad(samples)
        loss = F.mse_loss(labels, prediction.float())
        loss.backward()
        optimizer.step()

    # Stats
    return loss.detach(), labels.shape[0]

def train_epoch():
    global current_lr
    global iter
    total_loss = torch.tensor(0.0).to(device)
    total_items = torch.tensor(0).to(device)
    for i, data in enumerate(loader):
        samples, labels = data

        # Update learning rate
        current_lr = resolve_train_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Train
        l, c = train_batch(samples, labels)

        # Stats
        iter = iter + 1
        total_loss = total_loss + l
        total_items = total_items + c
    return total_loss / total_items

#
# Star Training
#

print(f'Training {experiment} on {device} with {dtype} precision')
for i in range(epoch, train_epochs):
    epoch = epoch + 1

    # Train
    start = time.perf_counter()
    training_loss = train_epoch()
    duration = round((time.perf_counter() - start) * 1000)

    # Validate
    validation_loss = validate()

    # Stats
    print(f'#{epoch}: {training_loss}/{validation_loss} in {duration} ms')
    writer.add_scalar('training loss', training_loss, epoch)
    writer.add_scalar('validation loss', validation_loss, epoch)
    writer.add_scalar('learning rate', current_lr, epoch)

    # Save
    save()