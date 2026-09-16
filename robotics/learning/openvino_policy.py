"""Execute the learned policy through OpenVINO; no PyTorch model forward pass."""
from pathlib import Path
import json
import openvino as ov
import torch

class OpenVINOPolicy:
    def __init__(self,device='CPU',precision='fp32'):
        root=Path(__file__).resolve().parent/'checkpoints'
        self.device=device
        settings={'INFERENCE_PRECISION_HINT':'f32'} if device=='CPU' else {}
        self.compiled=ov.Core().compile_model(str(root/f'policy-{precision}.xml'),device,settings)
        self.digest=json.loads((root/'openvino-benchmark.json').read_text())['checkpoint_sha256']
    def __call__(self,image,state,tokens,step):
        inputs={name:value.detach().cpu().numpy() for name,value in zip(['image','state','tokens','step'],[image,state,tokens,step])}
        result=self.compiled(inputs)
        return torch.from_numpy(result['actions'].copy()),torch.from_numpy(result['language_logits'].copy())
