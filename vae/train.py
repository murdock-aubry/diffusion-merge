import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from vae import TimeConditionedVAE
from utils import *

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30
BETA_START = 0.0
BETA_END = 0.03
BETA_STEPS = 30

model_name = "sd1.4"
# model_name = "sd1.4-cocotuned"
# model_name = "sd1.4-dogtuned"


data_path = f"/w/383/murdock/hidden_reps/{model_name}/representations_50.pt"
train_data, test_data, norm_params, num_time_steps = load_data(data_path, batch_size = BATCH_SIZE)

# Get sample shape for autoconfig
sample_data, _ = next(iter(train_data))
sample_shape = sample_data.shape
print(f"Input data shape: {sample_shape}", flush=True)
print(f"Number of time steps: {num_time_steps}", flush=True)

# Calculate latent dimension based on compression factor
compression_factor = 0.2  # 10x compression
print(compression_factor, flush = True)
LATENT_DIM = calculate_latent_dim(sample_shape, compression_factor = compression_factor)
print(f"Using latent dimension: {LATENT_DIM} (compression factor: {compression_factor})", flush=True)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
vae = TimeConditionedVAE(LATENT_DIM).to(device)
optimizer = optim.Adam(vae.parameters(), lr=LR)

# Training with tracking
train_losses = []
test_losses = []
recon_losses = []
kl_losses = []
ssim_losses = []

print("Starting training...", flush=True)
for epoch in range(EPOCHS):
    vae.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    epoch_ssim_loss = 0
    num_batches = 0
    time_scaled = False
    
    # Get current beta value
    current_beta = get_beta(epoch, beta_start = BETA_START, beta_end = BETA_END, beta_steps = BETA_STEPS)
    
    for (x_batch, t_batch) in train_data:
        x_batch = x_batch.to(device).float()
        t_batch = t_batch.to(device).float()
        
        x_recon, mu, logvar = vae(x_batch, t_batch)
        recon_loss, kl_loss, ssim_loss, loss = vae_loss(x_batch, x_recon, t_batch, mu, logvar, current_beta, time_scaled = time_scaled)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_ssim_loss += ssim_loss.item()
        num_batches += 1
    
    # Evaluate on test set
    vae.eval()
    test_loss = 0
    test_batches = 0
    with torch.no_grad():
        for (x_batch, t_batch) in test_data:
            x_batch = x_batch.to(device).float()
            t_batch = t_batch.to(device).float()
            
            x_recon, mu, logvar = vae(x_batch, t_batch)
            _, _, _, loss = vae_loss(x_batch, x_recon, t_batch, mu, logvar, current_beta, time_scaled = time_scaled)
            test_loss += loss.item()
            test_batches += 1
    
    # Calculate average losses
    avg_train_loss = epoch_loss / num_batches
    avg_recon_loss = epoch_recon_loss / num_batches
    avg_kl_loss = epoch_kl_loss / num_batches
    avg_ssim_loss = epoch_ssim_loss / num_batches
    avg_test_loss = test_loss / test_batches
    
    # Track losses
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    recon_losses.append(avg_recon_loss)
    kl_losses.append(avg_kl_loss)
    ssim_losses.append(avg_ssim_loss)
    
    # Print progress
    print(f"Epoch {epoch+1}/{EPOCHS}, Beta: {current_beta:.3f}, Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}, SSIM: {avg_ssim_loss:.4f}), Test Loss: {avg_test_loss:.4f}", flush=True)
    
    # Optional: Save model periodically
    if (epoch + 1) % 50 == 0:
        model_path = f"/w/383/murdock/models/vae/{model_name}/dim{LATENT_DIM}_epoch{epoch+1}_time_conditioned.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'latent_dim': LATENT_DIM,
            'compression_factor': compression_factor,
            'num_time_steps': num_time_steps
        }, model_path)


# Plot loss curves
print("Training complete, generating plots...", flush=True)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.ylabel('Total Loss')
plt.legend()
plt.title(f'Time-Conditioned VAE Training (Latent Dim: {LATENT_DIM}, Compression: {compression_factor})')

plt.subplot(2, 1, 2)
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.plot(ssim_losses, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Component Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f"/w/284/murdock/merge/vae/plots/{model_name}_dim{LATENT_DIM}_epoch{EPOCHS}_time_conditioned_loss.png")
plt.close()



final_model_path = f"/w/383/murdock/models/vae/{model_name}/dim{LATENT_DIM}_time.pth"
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
visualize_reconstructions(vae, test_data, model_name, LATENT_DIM, device = device)

print(f"Training complete. Final model saved to {final_model_path}", flush = True)
