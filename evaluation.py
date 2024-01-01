import torch
import webrtcvad
import numpy
import onnxruntime as rt
from utils import spectogram, sliding_window, naive_normalize_spectogram
from model import SuperVAD, Config

#
# WebRTC
#

def webrtc_vad(wav):
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    speech_probs = []
    window_size_samples = 320
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+window_size_samples]
        if len(chunk) < window_size_samples:
            break
        is_speech = vad.is_speech(chunk.numpy(), 16000)
        speech_probs.append(is_speech)
    return torch.tensor(speech_probs)

#
# Silero
#

silero_model, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)

def silero_vad(wav):
    speech_probs = []
    silero_model.reset_states()
    window_size_samples = 512
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+window_size_samples]
        if len(chunk) < window_size_samples:
            break
        speech_prob = silero_model(chunk, 16000).item()
        speech_probs.append(speech_prob)
    return torch.tensor(speech_probs)

#
# Supervad
#

super_vad_model = rt.InferenceSession("./supervad.onnx")

def super_vad(wav):
    predictions = []
    for i in range(3200, len(wav), 320):
        predicted = super_vad_model.run(["output"],{'input': wav[i-3200:i].unsqueeze(0).numpy()})[0][0][0]
        predictions.append(predicted)
    return torch.tensor(predictions)
    

# Supervad Torch

super_vad_torch_model = SuperVAD()
checkpoint = torch.load("./supervad.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
super_vad_torch_model.load_state_dict(checkpoint)
super_vad_torch_model.to(device)

def super_vad_torch(wav):
    predictions = []
    for i in range(3200, len(wav), 320):
        p = super_vad_torch_model(naive_normalize_spectogram(spectogram(wav[i-3200:i].to(device))).unsqueeze(0))
        predictions.append(p[0][0])
    return torch.tensor(predictions)