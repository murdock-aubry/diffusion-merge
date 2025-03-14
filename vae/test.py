import torch
import argparse
import os
from vae import TimeConditionedVAE
from torch.utils.data import DataLoader, TensorDataset
from utils import load_data
from utils import visualize_reconstructions

def parse_args():
    parser = argparse.ArgumentParser(description='Encode test data with a trained Time-Conditioned VAE')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save encoded data (.pt file)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    return parser.parse_args()
    

def encode_data(model, data_loader, device):
    model.eval()
    encoded_data = []
    original_data = []
    time_steps_data = []
    
    with torch.no_grad():
        for batch_idx, (data, time_steps) in enumerate(data_loader):
            data = data.to(device).float()
            time_steps = time_steps.to(device).float()
            
            # Encode the data
            _, mu, logvar = model(data, time_steps)
            
            # Store the mean vector (mu) as the encoded representation
            encoded_data.append(mu.cpu())
            original_data.append(data.cpu())
            time_steps_data.append(time_steps.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Encoded {batch_idx * len(data)} samples...", flush = True)
    
    # Concatenate all batches
    encoded_data = torch.cat(encoded_data, dim=0)
    original_data = torch.cat(original_data, dim=0)
    time_steps_data = torch.cat(time_steps_data, dim=0)
    
    return encoded_data, original_data, time_steps_data

def main():
    args = parse_args()

    model_name = args.model_path.split('/')[-2]
    
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush = True)
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}", flush = True)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract model parameters
    latent_dim = checkpoint['latent_dim']
    print(f"Latent dimension: {latent_dim}", flush = True)
    
    # Load data
    print(f"Loading data from {args.data_path}", flush = True)
    train_loader, test_loader, norm_params, num_time_steps = load_data(args.data_path, batch_size=args.batch_size, shuffle = False)
    # Combine train and test loaders into one
    combined_data = torch.utils.data.ConcatDataset([train_loader.dataset, test_loader.dataset])
    data_loader = DataLoader(combined_data, batch_size=args.batch_size, shuffle=False)
    
    
    # Initialize model
    model = TimeConditionedVAE(latent_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Encode test data
    print("Encoding test data...", flush = True)
    encoded_data, original_data, time_steps_data = encode_data(model, data_loader, device)
    
    print(f"Encoded data shape: {encoded_data.shape}", flush = True)
    print(f"Original data shape: {original_data.shape}", flush = True)
    print(f"Time steps data shape: {time_steps_data.shape}", flush = True)
    
    # Save encoded data as PyTorch .pt file
    print(f"Saving encoded data to {args.output_path}", flush = True)
    torch.save({
        'encoded_data': encoded_data,
        # 'original_data': original_data,
        'time_steps': time_steps_data,
        'latent_dim': latent_dim,
        'model_path': args.model_path
    }, args.output_path)
    
    print("Encoding complete!", flush = True)


    print("Compiling reconstruction visualizations.", flush = True)

    visualize_reconstructions(model, test_loader, model_name, latent_dim, device = device)

    print("Visualizations computed.", flush = True)

if __name__ == "__main__":
    main()