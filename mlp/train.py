import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from time_conditioned_mlp import TimeConditionedMLP  # Import the model we created earlier

class TimeConditionedDataset(Dataset):
    """
    Dataset for time-conditioned MLP training.
    
    Expects:
    - input_features: tensor of shape [num_samples, feature_dim]
    - output_features: tensor of shape [num_samples, feature_dim]
    - times: tensor of shape [num_samples]
    """
    def __init__(self, input_features, output_features, times):
        self.input_features = input_features
        self.output_features = output_features
        self.times = times
        
        assert len(input_features) == len(output_features) == len(times), "All inputs must have the same number of samples"
        assert input_features.shape[1] == output_features.shape[1], "Input and output feature dimensions must match"
        
    def __len__(self):
        return len(self.input_features)
    
    def __getitem__(self, idx):
        return {
            'input': self.input_features[idx],
            'output': self.output_features[idx],
            'time': self.times[idx]
        }

def train_model(
    model, 
    train_loader, 
    val_loader=None, 
    epochs=100, 
    lr=1e-4, 
    weight_decay=1e-5,
    device="cuda" if torch.cuda.is_available() else "cpu",
    scheduler_patience=10,
    early_stopping_patience=20,
    save_path="/w/383/murdock/models/time_conditioned_mlp.pt"
):
    """
    Training function for the TimeConditionedMLP model.
    """
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_loader else None
    }
    
    # Early stopping
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            times = batch['time'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, times)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch['input'].to(device)
                    targets = batch['output'].to(device)
                    times = batch['time'].to(device)
                    
                    outputs = model(inputs, times)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                # Save best model
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}")
    
    # Load best model if validation was used
    if val_loader:
        model.load_state_dict(torch.load(save_path))
    else:
        # Save final model if no validation
        torch.save(model.state_dict(), save_path)
    
    return model, history

def plot_training_history(history):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    
    if history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Evaluate the model on test data.
    """
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0.0
    
    all_inputs = []
    all_outputs = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input'].to(device)
            targets = batch['output'].to(device)
            times = batch['time'].to(device)
            
            outputs = model(inputs, times)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # Store for visualization
            all_inputs.append(inputs.cpu())
            all_outputs.append(targets.cpu())
            all_predictions.append(outputs.cpu())
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")
    
    # Combine batches
    all_inputs = torch.cat(all_inputs, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    return {
        'test_loss': avg_test_loss,
        'inputs': all_inputs,
        'targets': all_outputs,
        'predictions': all_predictions
    }

# Example usage
if __name__ == "__main__":
    # Parameters
    feature_dim = 16  # Number of features (N)
    batch_size = 32
    hidden_dims = [128, 256, 128]
    time_embed_dim = 64
    num_samples = 1000
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio will be 0.15
    
    # Generate synthetic data for demonstration
    # In practice, replace this with your actual data loading code
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create synthetic dataset
    times = torch.linspace(0, 1, num_samples)
    
    # Example synthetic function: features evolve based on time
    # Replace this with your actual data
    input_features = torch.randn(num_samples, feature_dim)
    
    # Create time-dependent output (example transformation)
    # In a real scenario, this would be your ground truth data
    output_features = torch.zeros_like(input_features)
    for i, t in enumerate(times):
        factor = 1 + t.item() * 0.5  # Time-dependent factor
        noise = 0.1 * torch.randn_like(input_features[i])
        output_features[i] = factor * input_features[i] + noise
    
    # Split data
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
    
    train_inputs = input_features[:train_size]
    train_outputs = output_features[:train_size]
    train_times = times[:train_size]
    
    val_inputs = input_features[train_size:train_size+val_size]
    val_outputs = output_features[train_size:train_size+val_size]
    val_times = times[train_size:train_size+val_size]
    
    test_inputs = input_features[train_size+val_size:]
    test_outputs = output_features[train_size+val_size:]
    test_times = times[train_size+val_size:]
    
    # Create datasets
    train_dataset = TimeConditionedDataset(train_inputs, train_outputs, train_times)
    val_dataset = TimeConditionedDataset(val_inputs, val_outputs, val_times)
    test_dataset = TimeConditionedDataset(test_inputs, test_outputs, test_times)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = TimeConditionedMLP(
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        dropout=0.1
    )
    
    # Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-5,
        save_path="best_time_conditioned_mlp.pt"
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    results = evaluate_model(trained_model, test_loader)
    
    print("Training completed successfully!")
    print(f"Final test loss: {results['test_loss']:.6f}")