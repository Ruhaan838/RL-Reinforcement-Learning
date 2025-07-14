import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self, input_channels=3, feature_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.out_dim = feature_dim

    def forward(self, x):
        return self.conv(x)

class RSSM(nn.Module):
    def __init__(self, stoch_size=30, deter_size=200, action_size=6, hidden_size=200):
        super().__init__()
        self.rnn = nn.GRUCell(hidden_size, deter_size)
        self.fc_embed = nn.Linear(stoch_size + action_size, hidden_size)
        self.fc_mean = nn.Linear(deter_size, stoch_size)
        self.fc_std = nn.Linear(deter_size, stoch_size)

    def forward(self, prev_stoch, prev_deter, action):
        x = torch.cat([prev_stoch, action], dim=-1)
        h = self.rnn(self.fc_embed(x), prev_deter)
        mean = self.fc_mean(h)
        std = F.softplus(self.fc_std(h)) + 1e-4
        stoch = mean + torch.randn_like(std) * std
        return stoch, h, mean, std

class Decoder(nn.Module):
    def __init__(self, deter_size=200, stoch_size=30, output_channels=3):
        super().__init__()
        self.fc = nn.Linear(deter_size + stoch_size, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_channels, 6, 2),
            nn.Sigmoid()
        )

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch], dim=-1)
        x = self.fc(x).view(-1, 64, 4, 4)
        return self.deconv(x)

class Actor(nn.Module):
    def __init__(self, deter_size=200, stoch_size=30, action_size=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(deter_size + stoch_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Tanh()
        )

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch], dim=-1)
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, deter_size=200, stoch_size=30):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(deter_size + stoch_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, deter, stoch):
        x = torch.cat([deter, stoch], dim=-1)
        return self.fc(x)

class Dreamer(nn.Module):
    def __init__(self, obs_channels=3, action_size=6, stoch_size=30, deter_size=200):
        super().__init__()
        self.encoder = Encoder(obs_channels)
        self.rssm = RSSM(stoch_size, deter_size, action_size)
        self.decoder = Decoder(deter_size, stoch_size, obs_channels)
        self.actor = Actor(deter_size, stoch_size, action_size)
        self.critic = Critic(deter_size, stoch_size)

    def forward(self, obs, prev_stoch, prev_deter, prev_action):

        embedded_obs = self.encoder(obs)
        stoch, deter, mean, std = self.rssm(prev_stoch, prev_deter, prev_action)
        recon_obs = self.decoder(deter, stoch)
        action = self.actor(deter, stoch)
        value = self.critic(deter, stoch)
        return {
            "embedded_obs": embedded_obs,
            "stoch": stoch,
            "deter": deter,
            "mean": mean,
            "std": std,
            "recon_obs": recon_obs,
            "action": action,
            "value": value
        }

model = Dreamer()
summary(
    model,  
    input_data=(
        torch.zeros(4, 3, 64, 64),   
        torch.zeros(4, 30),          
        torch.zeros(4, 200),         
        torch.zeros(4, 6)            
    ),
    col_names=["input_size", "output_size", "num_params"],
    depth=3
)