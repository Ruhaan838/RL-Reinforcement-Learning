import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        logits = self.net(x)
        value = self.value_head(x)
        return logits, value
