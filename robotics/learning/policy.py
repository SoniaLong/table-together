"""Task-specific vision-language keyframe imitation policy, trained from scratch.

Inputs: RGB camera, joint state, ordered word tokens, action-history position.
Outputs: twelve joint targets and motion duration. No IK or object-state access.
This is not ACT and is not a pretrained foundation VLA.
"""
import re
import numpy as np
import torch
from torch import nn

VOCAB=['<pad>','<unk>','set','the','table','start','with','cup','utensil','.','first','place']
INSTRUCTIONS=['Set the table. Start with the cup.','Set the table. Start with the utensil.']
def tokenize(text):
    words=re.findall(r'[a-z]+|\.',text.lower())
    if len(words)>16:raise ValueError('Policy instruction is limited to 16 tokens')
    return np.array([VOCAB.index(w) if w in VOCAB else 1 for w in words]+[0]*(16-len(words)),dtype=np.int64)

class Policy(nn.Module):
    def __init__(self,means,scales,version=1):
        super().__init__()
        self.version=version
        self.register_buffer('means',torch.as_tensor(means,dtype=torch.float32))
        self.register_buffer('scales',torch.as_tensor(scales,dtype=torch.float32))
        self.words=nn.Embedding(len(VOCAB),12,padding_idx=0)
        self.language=nn.Sequential(nn.Flatten(),nn.Linear(16*12,48),nn.Tanh())
        self.task=nn.Linear(48,2)
        self.vision=nn.Sequential(nn.Conv2d(3,8,5,2),nn.ReLU(),nn.Conv2d(8,16,3,2),nn.ReLU(),nn.Conv2d(16,16,3,2),nn.ReLU(),nn.Flatten(),nn.Linear(16*10*14,64),nn.ReLU())
        self.position=nn.Embedding(18,16)
        self.head=nn.Sequential(nn.Linear(64+48+16+12,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,13 if version==1 else 18*13),nn.Tanh())
        nn.init.zeros_(self.head[-2].weight);nn.init.zeros_(self.head[-2].bias)

    def forward(self,image,state,tokens,step):
        language=self.language(self.words(tokens))
        logits=self.task(language)
        weights=torch.softmax(logits*5,dim=-1)
        means=(self.means[:,step,:].permute(1,0,2)*weights[:,:,None]).sum(1)
        scales=(self.scales[:,step,:].permute(1,0,2)*weights[:,:,None]).sum(1)
        features=torch.cat([self.vision(image),language,self.position(step),state/3],dim=1)
        residual=self.head(features)
        if self.version==2:
            residual=residual.reshape(-1,18,13).gather(1,step[:,None,None].expand(-1,1,13)).squeeze(1)
        return means+scales*residual,logits
