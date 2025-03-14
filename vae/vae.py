import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class TimeConditionedVAE(nn.Module):
    def __init__(self, latent_dim, time_embedding_dim=32):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Encoder with time conditioning
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2, padding=1),  # -> 32x32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # -> 64x16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # -> 128x8x8
            nn.LeakyReLU(0.2),
            nn.Flatten()  # -> 8192
        )
        
        # Calculate encoder output size exactly (4->32->64->128 channels, spatial 64->32->16->8)
        self.encoder_output_size = 128 * 8 * 8
        
        # Bottleneck with time conditioning
        self.fc_mu = nn.Linear(self.encoder_output_size + time_embedding_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size + time_embedding_dim, latent_dim)
        
        # Decoder with time conditioning
        self.decoder_input = nn.Linear(latent_dim + time_embedding_dim, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # -> 64x16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x32x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),   # -> 4x64x64
            nn.Tanh()  # Output range [-1, 1] to match normalized input
        )

    def encode_time(self, t):
        # Convert time to [batch_size, 1] shape and embed it
        t = t.unsqueeze(1)
        return self.time_embed(t)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, t_emb):
        x_encoded = self.encoder(x)
        # Concatenate time embedding

        x_with_time = torch.cat([x_encoded, t_emb], dim=1)
        mu = self.fc_mu(x_with_time)
        logvar = self.fc_logvar(x_with_time)
        return mu, logvar
        
    def decode(self, z, t_emb):
        # Concatenate time embedding with latent code
        z_with_time = torch.cat([z, t_emb], dim=1)
        x = self.decoder_input(z_with_time)
        x = self.decoder(x)
        return x

    def forward(self, x, t):
        t_emb = self.encode_time(t)
        mu, logvar = self.encode(x, t_emb)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, t_emb)
        return x_recon, mu, logvar