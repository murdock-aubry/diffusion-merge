import torch
import torch.nn as nn
import torch.optim as optim
from piq import ssim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def calculate_latent_dim(input_shape, compression_factor=0.1):
    """Calculate latent dimension as a fraction of flattened input size"""
    flattened_size = input_shape[0] * input_shape[1] * input_shape[2]
    return int(flattened_size * compression_factor)

def load_data(file_path, batch_size = 32, split=0.8, layers = [], shuffle = True):
    data = torch.load(file_path)
    
    prompts = data["prompts"]
    inputs = data["inputs"]
    outputs = data["outputs"]
    
    data = torch.stack(inputs, dim=0)
    torch.manual_seed(42)


    # Keep all layers (time steps) as part of the data
    # Each sample will now have shape [n_layers, channels, height, width]
    original_shape = data.shape
    
    # Store original data range for reconstruction evaluation
    data_min = data.min()
    data_max = data.max()
    
    # Normalize the dataset across all layers
    mean = data.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = data.std(dim=(0, 1, 3, 4), keepdim=True)
    data = (data - mean) / (std + 1e-6)
    
    # Creating time indices as conditioning vector
    # Assuming dimensions are [batch, layers, channels, height, width]
    num_layers = data.shape[1]
    time_indices = torch.arange(num_layers).float()
    
    # Normalize time indices to [0, 1]
    normalized_time = time_indices / (num_layers - 1)
    
    # Create dataset with both data and its corresponding time step
    datasets = []

    if len(layers) > 0: # Either loop through the provided layers only
        for i in layers:
            layer_data = data[:, i]  # Get data for layer i across all samples
            
            # Create tensor of the same time step repeated for all samples in this layer
            time_conditioning = torch.full((len(layer_data),), normalized_time[i])
            datasets.append(TensorDataset(layer_data, time_conditioning))
    
    else: # or grab all layers
        for i in range(num_layers):
            layer_data = data[:, i]  # Get data for layer i across all samples
            
            # Create tensor of the same time step repeated for all samples in this layer
            time_conditioning = torch.full((len(layer_data),), normalized_time[i])
            datasets.append(TensorDataset(layer_data, time_conditioning))
    
    # Combine all layer datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Split into train and test sets
    train_size = int(split * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        combined_dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, (mean, std), num_layers

def vae_loss(x, x_recon, t_batch, mu, logvar, beta, alpha=0.1, time_scaled=False):
    # Calculate reconstruction loss (MSE) in a vectorized way
    recon_loss_per_sample = torch.mean((x_recon - x)**2, dim=list(range(1, len(x.shape))))
    
    # Calculate KL divergence in a vectorized way (already efficient)
    kl_loss_per_sample = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    
    if time_scaled:
        # For time-scaled version, we need per-sample SSIM
        # Use batch SSIM if your library supports it
        x_clamped = torch.clamp(x, 0, 1)
        x_recon_clamped = torch.clamp(x_recon, 0, 1)
        
        # SSIM is tricky to vectorize fully, but we can make it more efficient
        ssim_losses = torch.zeros(x.shape[0], device=x.device)
        for i in range(x.shape[0]):
            ssim_losses[i] = ssim(x_clamped[i:i+1], x_recon_clamped[i:i+1], data_range=1.0)
        ssim_loss_per_sample = 1 - ssim_losses
        
        # Apply time scaling
        recon_loss = torch.mean(recon_loss_per_sample * t_batch ** 2)
        kl_loss = torch.mean(kl_loss_per_sample * t_batch ** 2)
        ssim_loss = torch.mean(ssim_loss_per_sample * t_batch ** 2)
    else:
        # For non-time-scaled, use batch calculations
        recon_loss = torch.mean(recon_loss_per_sample)
        kl_loss = torch.mean(kl_loss_per_sample)
        ssim_loss = 1 - ssim(torch.clamp(x, 0, 1), torch.clamp(x_recon, 0, 1), data_range=1.0)
    
    # Calculate total loss
    total_loss = recon_loss + beta * kl_loss + alpha * ssim_loss
    
    return recon_loss, kl_loss, ssim_loss, total_loss


def get_beta(epoch, beta_start = 0.0, beta_end = 1.0, beta_steps = 100):
    if epoch < beta_steps:
        return beta_start + (beta_end - beta_start) * epoch / beta_steps
    return beta_end


def visualize_reconstructions(model, data_loader, model_name, latent_dim, n_samples=5, device = "cuda"):
    model.eval()
    with torch.no_grad():
        # Get a batch of data
        batch = next(iter(data_loader))
        x_batch, t_batch = batch  # Unpack both data and time values
        
        # Only use the first n_samples
        x_batch = x_batch[:n_samples].to(device).float()
        t_batch = t_batch[:n_samples].to(device).float()
        
        # Get reconstructions
        x_recon, _, _ = model(x_batch, t_batch)
        
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
        plt.savefig(f"/w/284/murdock/merge/vae/plots/{model_name}_dim{latent_dim}_time_conditioned_reconstructions.png")
        plt.close()
