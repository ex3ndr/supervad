import json
import torch
import torchaudio
from torch.utils.data import Dataset
import random
import math

SAMPLE_RATE = 16000
DATASET_SAMPLE_LENGTH = 5
TOKENS_PER_SECOND = 50 # == 20ms

def sample_dataset(dataset):
    return dataset[random.randint(0, len(dataset) - 1)]

class PreprocessedAudioDataset(Dataset):
    def __init__(self, meta, dir):
        self.data = meta
        self.dir = dir
        self.keys = list(meta.keys())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):

        # Load session
        intervals = self.data[self.keys[idx]]

        # Source
        audio, _ = torchaudio.load(self.dir + "/" + self.keys[idx])
        audio = audio[0]

        # Target
        target = torch.zeros(DATASET_SAMPLE_LENGTH * TOKENS_PER_SECOND)
        for k in intervals:
            sstart = k[0]
            sstop = k[1]
            sstart = math.floor(sstart * TOKENS_PER_SECOND)
            sstop = math.floor(sstop * TOKENS_PER_SECOND)
            target[sstart:sstop] = 1

        return audio, target
        
def preprocessed_audio_dataset(dir):
    with open(dir + "/meta.json") as meta_file:
        return PreprocessedAudioDataset(json.load(meta_file), dir)