import os
import torch
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
from datasets import SAMPLE_RATE, DATASET_SAMPLE_LENGTH, TOKENS_PER_SECOND, preprocessed_audio_dataset, sample_dataset

#
# Plotting
#

def plot_waveform(waveform, sample_rate=16000, title="Waveform", xlim=(0,5)):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

def plot_specgram(waveform, sample_rate=16000, title="Spectrogram", xlim=(0,5)):
    waveform = waveform.numpy()

    num_channels, _ = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)

def plot_labels(waveform, sample_rate=50, title="Labels", xlim=(0,5), threshold=None):
    waveform = waveform.numpy()

    num_frames = waveform.shape[0]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform, linewidth=1)
    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.25,1.25)

    # Highlight intervals where values are higher than the threshold
    if threshold is not None:
        ax.axhline(y=threshold, color='green', linestyle='--')
        above_threshold = waveform > threshold
        for start, stop in contiguous_regions(above_threshold):
            ax.axvspan(time_axis[start], time_axis[stop], color='green', alpha=0.3)
    
    figure.suptitle(title)

def play_audio(waveform):
    display(Audio(data=waveform, rate=16000))

def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition".
    This function returns a 2D array where the first column is the
    start index of the region and the second column is the end index.
    """
    # Find the indices of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size - 1] 

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx

#
# 
#

mel_filters = torch.from_numpy(np.load("./mel_filters.npz", allow_pickle=False)["mel_80"])

def spectogram(audio):
    window = torch.hann_window(400, device=audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=False)
    magnitudes = torch.sum((stft ** 2), dim=-1)[..., :-1]

    # Mel
    mel_spec = mel_filters.to(audio.device) @ magnitudes

    # Log
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    
    return log_spec

def naive_normalize_spectogram(audio):
    audio = torch.maximum(audio, audio.max() - 8.0)
    audio = (audio + 4.0) / 4.0
    return audio

def sliding_window(tensor, window_size, step):

    # Load last dimension
    last_dim = tensor.size(-1)
    if window_size > last_dim:
         raise ValueError("Window size is larger than the tensor's last dimension")

    # Create sliding window
    unfolded = tensor.unfold(-1, window_size, step)

    # Permute dimensions
    total_dims = tensor.dim()
    dims = []
    dims.append(total_dims-1)
    for i in range(total_dims - 1):
        dims.append(i)
    dims.append(total_dims)
    unfolded = unfolded.permute(*dims)

    return unfolded

def clamp(value, lower, upper):
    return lower if value < lower else upper if value > upper else value