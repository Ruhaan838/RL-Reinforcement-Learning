import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class MAMLPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, params=None):
        if params is None:
            x = F.relu(self.fc1(x))
            return self.fc2(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            return F.linear(x, params['fc2.weight'], params['fc2.bias'])

    def clone_parameters(self):
        return {k: v.clone().detach().requires_grad_(True) for k, v in self.named_parameters()}

model = MAMLPolicy(input_size=10, hidden_size=64, output_size=5)
summary(model, input_size=(1, 10))