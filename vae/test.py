import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vae import VAE

model_name = "sd1.4"
latent_dim = 64

vae = VAE(latent_dim)
vae.load_state_dict(torch.load(f"models/{model_name}_dim{latent_dim}.pth"))
vae.eval()

# Extract encoder and decoder
encoder = lambda x: vae.fc_mu(vae.encoder(x))
decoder = lambda z: vae.decoder(vae.decoder_fc(z).view(-1, 64, 8, 8))