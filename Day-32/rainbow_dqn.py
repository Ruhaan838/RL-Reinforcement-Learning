import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.017)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.017)

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features).sign() * torch.sqrt(torch.abs(torch.randn(self.in_features)))
        epsilon_out = torch.randn(self.out_features).sign() * torch.sqrt(torch.abs(torch.randn(self.out_features)))
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, obs_dim, action_dim, atom_size=51, Vmin=-10, Vmax=10):
        super().__init__()
        self.atom_size = atom_size
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.support = torch.linspace(Vmin, Vmax, atom_size)
        self.action_dim = action_dim

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.advantage = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim * atom_size)
        )
        self.value = nn.Sequential(
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, atom_size)
        )

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q

    def dist(self, x):
        x = self.feature(x)
        adv = self.advantage(x).view(-1, self.action_dim, self.atom_size)
        val = self.value(x).view(-1, 1, self.atom_size)
        q_atoms = val + adv - adv.mean(1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1)
        return dist


obs_dim = 8
action_dim = 4
model = RainbowDQN(obs_dim, action_dim)


summary(model, input_size=(1, obs_dim), 
        col_names=["input_size", "output_size", "num_params", "trainable"], 
        depth=3)