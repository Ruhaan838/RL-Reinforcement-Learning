import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_eps', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_eps', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameter()
        self.reset_noise()
        
    def reset_parameter(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / self.out_features ** 0.5)
        
    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)
        
    def _scale_noise(self, size):
        x = torch.rand(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        
        weight = self.weight_mu
        bias = self.bias_mu
        
        if self.training:
            weight = weight + self.weight_sigma * self.weight_eps
            bias = bias + self.bias_sigma * self.bias_eps
            
        return F.linear(x, weight, bias)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, action_dim)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.noisy1(x))
        return self.noisy2(x)
    
    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        

if __name__ == "__main__":
    state_dim = 4
    action_dim = 2
    n_dqn = DQN(state_dim, action_dim)
    dummy_in = torch.randn(2, state_dim)
    summary(n_dqn, input_data=dummy_in, batch_dim=2)