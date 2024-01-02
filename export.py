import torch
import os
from torch import nn
from model import SuperVAD, Config
from utils import spectogram

PARAM_CHECKPOINT = "checkpoints/vad_lnorm_1e4s.pt"
PARAM_OUTPUT_NAME = "supervad"

# Load model
model_config = Config()
base = SuperVAD(model_config)
base.load_state_dict(torch.load(PARAM_CHECKPOINT)['model_state_dict'])

# Wrap model for ONNX
class ONNXSuperVAD(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, audio):
        return self.inner(spectogram(audio)).to(torch.float32)
class ONNXSuperVADDirect(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, audio):
        return self.inner(audio).to(torch.float32)
onnexed_vad = ONNXSuperVAD(base)
onnexed_vad_direct = ONNXSuperVADDirect(base)
base.eval()
onnexed_vad.eval()
onnexed_vad_direct.eval()

# Export
if not os.path.exists("./export"):
    os.mkdir("./export")
dummy = torch.zeros((model_config.ctx_length*320)).unsqueeze(0)
torch.onnx.export(onnexed_vad, dummy, 
                  f'./export/{PARAM_OUTPUT_NAME}.onnx', 
                  export_params=True, 
                  input_names = ['input'], 
                  output_names = ['output'], 
                  dynamic_axes={
                      'input' : {0 : 'batch_size'}, 
                      'output' : {0 : 'batch_size'}
                  })

dummy2 = torch.zeros((model_config.ctx_width, model_config.ctx_length * 2)).unsqueeze(0)
torch.onnx.export(onnexed_vad_direct, dummy2, 
                  f'./export/{PARAM_OUTPUT_NAME}_direct.onnx', 
                  export_params=True, 
                  input_names = ['input'], 
                  output_names = ['output'], 
                  dynamic_axes={
                      'input' : {0 : 'batch_size'}, 
                      'output' : {0 : 'batch_size'}
                  })
torch.save(base.state_dict(),  f'./export/{PARAM_OUTPUT_NAME}.pt')

# traced_gpu = torch.jit.trace(gpu_model, sample_input_gpu)
# torch.jit.save(traced_gpu, "gpu.pt")
traced = torch.jit.trace(onnexed_vad_direct, dummy2)
torch.jit.save(traced, f'./export/{PARAM_OUTPUT_NAME}_script.pt')
traced = torch.jit.trace(onnexed_vad, dummy)
torch.jit.save(traced, f'./export/{PARAM_OUTPUT_NAME}_script2.pt')
# script = torch.jit.script(base, dummy)
# script.save(f'./export/{PARAM_OUTPUT_NAME}_script.pt')