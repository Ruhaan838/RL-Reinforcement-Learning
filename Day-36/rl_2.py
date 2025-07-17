import torch.nn as nn
from torchinfo import summary

class RL2Policy(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        logits = self.fc(out)
        return logits, h

model = RL2Policy(input_size=10, hidden_size=64, action_size=5)
summary(model, input_size=(1, 20, 10))