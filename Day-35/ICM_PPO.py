import torch
import torch.nn as nn
from torchinfo import summary

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

class RNDTarget(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

class RNDPredictor(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.net(x)

class ICMEncoder(nn.Module):
    def __init__(self, obs_dim, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        return self.net(x)

class ICMInverseModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, phi_s, phi_s_next):
        x = torch.cat([phi_s, phi_s_next], dim=-1)
        return self.net(x)

class ICMForwardModel(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, phi_s, a):
        x = torch.cat([phi_s, a], dim=-1)
        return self.net(x)

obs_dim = 10
action_dim = 4
feature_dim = 64

ppo = PPOActorCritic(obs_dim, action_dim)
rnd_target = RNDTarget(obs_dim)
rnd_predictor = RNDPredictor(obs_dim)
icm_encoder = ICMEncoder(obs_dim, feature_dim)
icm_inverse = ICMInverseModel(feature_dim, action_dim)
icm_forward = ICMForwardModel(feature_dim, action_dim)

summary(ppo, input_size=(1, obs_dim))
summary(rnd_predictor, input_size=(1, obs_dim))
summary(icm_encoder, input_size=(1, obs_dim))

