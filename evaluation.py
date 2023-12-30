import torch

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
