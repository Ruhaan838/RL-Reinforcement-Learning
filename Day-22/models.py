import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        dummy = torch.zeros(1, *input_dim)
        n_flatten = self.feature(dummy).shape[1]
        if dueling:
            self.value = nn.Sequential(
                nn.Linear(n_flatten, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.advantage = nn.Sequential(
                nn.Linear(n_flatten, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
        else:
            self.q = nn.Sequential(
                nn.Linear(n_flatten, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
            
    def forward(self, x):
        x = self.feature(x)
        if self.dueling:
            value = self.value(x)
            advan = self.advantage(x)
            return value + advan - advan.mean(dim=1, keepdim=True)
        else:
            return self.q(x)