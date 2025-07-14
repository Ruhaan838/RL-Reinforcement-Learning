import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),  # [B, 32, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [B, 64, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # [B, 128, H/8, W/8]
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z).view(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=1, hidden_dim=256, num_gaussians=5):
        super().__init__()
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.num_gaussians = num_gaussians
        self.z_dim = latent_dim

        self.fc_pi = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, a, h=None):
        inp = torch.cat([z, a], dim=-1).unsqueeze(1)  # [B, 1, latent+action]
        output, h = self.rnn(inp, h)

        pi = self.fc_pi(output).view(-1, self.num_gaussians, self.z_dim)
        mu = self.fc_mu(output).view(-1, self.num_gaussians, self.z_dim)
        sigma = torch.exp(self.fc_sigma(output).view(-1, self.num_gaussians, self.z_dim))

        pi = F.softmax(pi, dim=1)
        return pi, mu, sigma, h

class Controller(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256, action_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, z, hidden):
        hidden_state = hidden[0][-1]  # Take last hidden state from LSTM
        x = torch.cat([z, hidden_state], dim=-1)
        return self.fc(x)
    
col_names=["input_size", "output_size", "num_params"]

print("="*10," Summary of the VAE model ","="*10)
vae = VAE()
summary(
    vae,
    input_size=(4, 3, 64, 64),
    col_names=col_names,
    depth=3
)
print("="*10," Summary of the MDN-RNN model ","="*10)
mdnrnn = MDNRNN()
summary(
    mdnrnn,
    input_data=(torch.zeros(4, 32), torch.zeros(4, 1)),
    col_names=col_names,
    depth=3
)

print("="*10," Summary of the Controller model ","="*10)
controller = Controller()
summary(
    controller,
    input_data=(torch.zeros(4, 32), (torch.zeros(1, 4, 256), torch.zeros(1, 4, 256))),
    col_names=col_names,
    depth=3
)




