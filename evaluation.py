import torch
import webrtcvad
import numpy
import onnxruntime as rt
from utils import spectogram, sliding_window, naive_normalize_spectogram
from model import SuperVAD, Config

#
# Init Code
#

webrtc_mode_v = None
silero_model = None
super_vad_model = None
super_vad_torch_model = None
super_vad_torch_device = None

def init_evaluation(supervad_pytorch = "./supervad.pt", supervad_pytorch_chk=False, supervad_onnx = "./supervad.onnx", webrtc_mode = 3, device = None):
    global webrtc
    global super_vad_model
    global super_vad_torch_model
    global super_vad_torch_device
    global silero_model
    global webrtc_mode_v

    # WebRTC
    webrtc_mode_v = webrtc_mode

    # Silero
    silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=True, onnx=False)

    # SuperVAD pytorch
    super_vad_torch_model = SuperVAD()
    checkpoint = torch.load(supervad_pytorch)
    super_vad_torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device is not None:
        super_vad_torch_device = device
    if supervad_pytorch_chk:
        super_vad_torch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        super_vad_torch_model.load_state_dict(checkpoint)
    super_vad_torch_model.to(super_vad_torch_device)
    super_vad_torch_model.eval()

    # SuperVAD ONNX
    super_vad_model = rt.InferenceSession(supervad_onnx)

def reload_torch(supervad_pytorch = "./supervad.pt", supervad_pytorch_chk=False, device = None):
    global super_vad_torch_model
    global super_vad_torch_device

    # SuperVAD pytorch
    super_vad_torch_model = SuperVAD()
    checkpoint = torch.load(supervad_pytorch)
    super_vad_torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device is not None:
        super_vad_torch_device = device
    if supervad_pytorch_chk:
        super_vad_torch_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        super_vad_torch_model.load_state_dict(checkpoint)
    super_vad_torch_model.to(super_vad_torch_device)
    super_vad_torch_model.eval()

#
# WebRTC
#

def webrtc_vad(wav):
    webrtc = webrtcvad.Vad()
    webrtc.set_mode(webrtc_mode_v)
    speech_probs = []
    window_size_samples = 320
    for i in range(0, len(wav), window_size_samples):
        chunk = wav[i: i+window_size_samples]
        if len(chunk) < window_size_samples:
            break
        is_speech = webrtc.is_speech(chunk.numpy(), 16000)
        speech_probs.append(is_speech)
    return torch.tensor(speech_probs)

#
# Silero
#

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

def super_vad(wav):
    predictions = []
    for i in range(3200, len(wav), 320):
        predicted = super_vad_model.run(["output"],{'input': wav[i-3200:i].unsqueeze(0).numpy()})[0][0][0]
        predictions.append(predicted)
    return torch.tensor(predictions)
    

# Supervad Torch

def super_vad_torch(wav):
    predictions = []
    wav = torch.nn.functional.pad(wav, (3200-320, 0), "constant", 0) # Pad zeros
    wav = sliding_window(spectogram(wav).to(super_vad_torch_device), 20, 2)
    p = super_vad_torch_model(wav)
    return p.cpu().squeeze()