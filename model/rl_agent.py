import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
            #nn.Sigmoid()
        )
        self.soft = nn.Softmax(dim=0)

    def forward(self, x, tags):
        x = self.net(x)
        #print(x.shape, '-'*3, type(x))
        #print(tags.shape, '-'*3, type(tags))
        x = torch.sub(x, tags)
        x = self.soft(x)
        return x
