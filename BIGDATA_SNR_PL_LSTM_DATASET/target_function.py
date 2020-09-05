import numpy as np
import os
import re
import torch
from torch.autograd import Variable

K = 1
C = 3

def cIRM(speech, mix):
    Mr = (mix[:,:,:,0] * speech[:,:,:,0] + mix[:,:,:,1] * speech[:,:,:,1]) / (mix[:,:,:,0] * mix[:,:,:,0] + mix[:,:,:,1] * mix[:,:,:,1])
    Mi = (mix[:,:,:,0] * speech[:,:,:,1] - mix[:,:,:,1] * speech[:,:,:,0]) / (mix[:,:,:,0] * mix[:,:,:,0] + mix[:,:,:,1] * mix[:,:,:,1])
    Mr.clamp_(min=-20, max=20)
    Mi.clamp_(min=-20, max=20)
    Mr = K * (1 - torch.exp(-Mr * C)) / (1 + torch.exp(-Mr * C))
    Mi = K * (1 - torch.exp(-Mi * C)) / (1 + torch.exp(-Mi * C))
    return [Mr, Mi]

def uncompress_cIRM(cIRM, MIX):
    M = -1 / C * torch.log((K - cIRM) / (K + cIRM))
    return torch.stack(
        [M[:, :, :,0] * MIX[:, :, :,0] - M[:, :, :,1] * MIX[:, :, :,1], M[:, :, :,0] * MIX[:, :, :,1] + M[:, :, :,1] * MIX[:, :, :,0]], 3)