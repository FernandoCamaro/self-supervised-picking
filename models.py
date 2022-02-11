import torch
from torch import nn
from torchvision import models
import numpy as np

class Model(nn.Module):

    def __init__(self, gamma, max_reward):
        super(Model, self).__init__()
        "assume that minimum reward is 0."
        self.model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)
        self.gamma = gamma
        self.max_reward = max_reward
        self.max_possible_qvalue = max_reward*(1+gamma/(1-gamma))
    
    def forward(self, x):
        is_np_array = False
        if type(x) == np.ndarray:
            is_np_array = True
            x = torch.Tensor(x)
            x = x.unsqueeze(0)

        x = x.to(self.device)
        y = self.model(x)
        y = torch.sigmoid(y['out'])
        y = y*self.max_possible_qvalue # [bs, 1, H, W]
        y = y.squeeze(1)
        bs = y.shape[0]
        y = y.reshape(bs,-1) # [bs, H*W]

        if is_np_array:
            y = y.squeeze(0).detach().cpu().numpy() #[H*W]
        
        return y

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)