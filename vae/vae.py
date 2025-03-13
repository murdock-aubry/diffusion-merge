import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Hyperparameters
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100
BETA_START = 0.0
BETA_END = 1.0
BETA_STEPS = 50

model_name = "sd1.4-cocotuned"
layer = -1

# Make latent dim a function of the input size
def calculate_latent_dim(input_shape, compression_factor=0.1):
    """Calculate latent dimension as a fraction of flattened input size"""
    flattened_size = input_shape[0] * input_shape[1] * input_shape[2]
    return int(flattened_size * compression_factor)

def load_data(file_path, split=0.8, layer=-1):
    data = torch.load(file_path)
    
    prompts = data["prompts"]
    inputs = data["inputs"]
    outputs = data["outputs"]
    
    data = torch.stack(inputs, dim=0)
    
    if layer == -1:
        data = data.reshape(-1, *data.shape[2:])
    else:
        data = data[:, layer, :, :]
    
    # Store original data range for reconstruction evaluation
    data_min = data.min()
    data_max = data.max()
    
    # Normalize the dataset
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    std = data.std(dim=(0, 2, 3), keepdim=True)
    data = (data - mean) / (std + 1e-6)
    
    # Split into train and test sets
    train_size = int(split * len(data))
    test_size = len(data) - train_size
    indices = torch.randperm(len(data))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, (mean, std)

data_path = f"/w/383/murdock/hidden_reps/{model_name}/representations.pt"
train_data, test_data, norm_params = load_data(data_path, layer=layer)

# Get sample shape for autoconfig
sample_shape = next(iter(train_data))[0].shape
print(f"Input data shape: {sample_shape}", flush = True)

# Calculate latent dimension based on compression factor
compression_factor = 0.1  # 10x compression
LATENT_DIM = calculate_latent_dim(sample_shape, compression_factor)
print(f"Using latent dimension: {LATENT_DIM} (compression factor: {compression_factor})", flush = True)

# VAE Model with fixed encoder/decoder architecture for 4x64x64 inputs
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        # Fixed for 4x64x64 inputs
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
        
        # Bottleneck
        self.fc_mu = nn.Linear(self.encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_size, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.encoder_output_size)
        
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # -> 64x16x16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # -> 32x32x32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),   # -> 4x64x64
            nn.Tanh()  # Output range [-1, 1] to match normalized input
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
        
    def decode(self, z):
        x = self.decoder_input(z)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# Loss function with weighted components
def vae_loss(x, x_recon, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kl_loss, recon_loss + beta * kl_loss

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush = True)
vae = VAE(LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LR)

# Training with tracking
train_losses = []
test_losses = []
recon_losses = []
kl_losses = []

# Beta annealing schedule
def get_beta(epoch):
    if epoch < BETA_STEPS:
        return BETA_START + (BETA_END - BETA_START) * epoch / BETA_STEPS
    return BETA_END

# Training loop
print("Starting training...", flush = True)
for epoch in range(EPOCHS):
    vae.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    num_batches = 0
    
    # Get current beta value
    current_beta = get_beta(epoch)
    
    for x_batch, in train_data:
        x_batch = x_batch.to(device).float()
        x_recon, mu, logvar = vae(x_batch)
        recon_loss, kl_loss, loss = vae_loss(x_batch, x_recon, mu, logvar, current_beta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        num_batches += 1
    
    # Evaluate on test set
    vae.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for x_batch, in test_data:
            x_batch = x_batch.to(device).float()
            x_recon, mu, logvar = vae(x_batch)
            _, _, loss = vae_loss(x_batch, x_recon, mu, logvar, current_beta)
            test_loss += loss.item()
            test_batches += 1
    
    # Calculate average losses
    avg_train_loss = epoch_loss / num_batches
    avg_recon_loss = epoch_recon_loss / num_batches
    avg_kl_loss = epoch_kl_loss / num_batches
    avg_test_loss = test_loss / test_batches
    
    # Track losses
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(avg_kl_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}, Beta: {current_beta:.3f}, Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}), Test Loss: {avg_test_loss:.4f}", flush=True)
    
    # Optional: Save model periodically
    if (epoch + 1) % 50 == 0:
        model_path = f"/w/284/murdock/merge/vae/models/{model_name}_dim{LATENT_DIM}_epoch{epoch+1}_layer{layer}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'latent_dim': LATENT_DIM,
            'compression_factor': compression_factor
        }, model_path)

# Plot loss curves
print("Training complete, generating plots...", flush = True)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.ylabel('Total Loss')
plt.legend()
plt.title(f'VAE Training (Latent Dim: {LATENT_DIM}, Compression: {compression_factor})')

plt.subplot(2, 1, 2)
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Component Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f"/w/284/murdock/merge/vae/plots/{model_name}_dim{LATENT_DIM}_loss.png")
plt.close()



# Visualize reconstructions
def visualize_reconstructions(model, data_loader, n_samples=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        x_batch, = next(iter(data_loader))
        x_batch = x_batch[:n_samples].to(device).float()
        
        # Get reconstructions
        x_recon, _, _ = model(x_batch)
        
        # Move to CPU for plotting
        x_batch = x_batch.cpu()
        x_recon = x_recon.cpu()
        
        plt.figure(figsize=(15, 6))
        for i in range(n_samples):
            # Show each channel of the original
            for c in range(4):
                plt.subplot(4, n_samples*2, i*2 + 1 + c*n_samples*2)
                plt.imshow(x_batch[i, c].numpy(), cmap='viridis')
                plt.axis('off')
                if i == 0:
                    plt.title(f'Original Ch.{c}')
            
            # Show each channel of the reconstruction
            for c in range(4):
                plt.subplot(4, n_samples*2, i*2 + 2 + c*n_samples*2)
                plt.imshow(x_recon[i, c].numpy(), cmap='viridis')
                plt.axis('off')
                if i == 0:
                    plt.title(f'Recon Ch.{c}')
        
        plt.tight_layout()
        plt.savefig(f"/w/284/murdock/merge/vae/plots/{model_name}_dim{LATENT_DIM}_reconstructions.png")
        plt.close()

# Save final model
final_model_path = f"/w/383/murdock/models/vae/{model_name}_dim{LATENT_DIM}_final_layer{layer}.pth"
torch.save({
    'model_state_dict': vae.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'latent_dim': LATENT_DIM,
    'compression_factor': compression_factor,
    'final_train_loss': train_losses[-1],
    'final_test_loss': test_losses[-1],
    'norm_params': norm_params
}, final_model_path)

# Visualize some reconstructions
visualize_reconstructions(vae, test_data)

print(f"Training complete. Final model saved to {final_model_path}", flush = True)

# Try different compression factors
def test_compression_factors():
    """Test how different compression factors affect the VAE's performance"""
    factors = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    results = []
    
    for factor in factors:
        print(f"\nTesting compression factor: {factor}")
        latent_dim = calculate_latent_dim(sample_shape, factor)
        print(f"Latent dimension: {latent_dim}")
        
        # Create model with this dimension
        model = VAE(latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        # Train for just 20 epochs to get a sense of convergence
        for epoch in range(20):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for x_batch, in train_data:
                x_batch = x_batch.to(device).float()
                x_recon, mu, logvar = model(x_batch)
                recon_loss, kl_loss, loss = vae_loss(x_batch, x_recon, mu, logvar, 1.0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/20, Loss: {avg_loss:.4f}")
        
        # Evaluate final performance
        model.eval()
        with torch.no_grad():
            total_recon_loss = 0
            total_kl_loss = 0
            total_batches = 0
            
            for x_batch, in test_data:
                x_batch = x_batch.to(device).float()
                x_recon, mu, logvar = model(x_batch)
                recon_loss, kl_loss, _ = vae_loss(x_batch, x_recon, mu, logvar, 1.0)
                
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_batches += 1
            
            avg_recon_loss = total_recon_loss / total_batches
            avg_kl_loss = total_kl_loss / total_batches
            
            results.append({
                'factor': factor,
                'latent_dim': latent_dim,
                'recon_loss': avg_recon_loss,
                'kl_loss': avg_kl_loss,
                'total_loss': avg_recon_loss + avg_kl_loss
            })
    
    # Plot results
    plt.figure(figsize=(12, 6))
    factors = [r['factor'] for r in results]
    recon_losses = [r['recon_loss'] for r in results]
    
    plt.plot(factors, recon_losses, 'o-')
    plt.xlabel('Compression Factor')
    plt.ylabel('Reconstruction Loss')
    plt.xscale('log')
    plt.title('Reconstruction Loss vs. Compression Factor')
    plt.grid(True)
    plt.savefig(f"/w/284/murdock/merge/vae/plots/compression_analysis.png")
    
    # Print tabular results
    print("\nCompression Factor Analysis:", flush = True)
    print("-" * 60, flush = True)
    print(f"{'Factor':<10} {'Latent Dim':<12} {'Recon Loss':<12} {'KL Loss':<12} {'Total Loss':<12}", flush = True)
    print("-" * 60, flush = True)
    for r in results:
        print(f"{r['factor']:<10.3f} {r['latent_dim']:<12d} {r['recon_loss']:<12.4f} {r['kl_loss']:<12.4f} {r['total_loss']:<12.4f}", flush = True)
    
    return results

# Uncomment to run the compression analysis
# compression_results = test_compression_factors()