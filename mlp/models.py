import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSinusoidalEmbedding(nn.Module):
    """
    Sinusoidal time embedding for conditioning the MLP.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Pad if dimension is odd
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1), mode='constant')
        
        return embeddings

class TimeConditionedMLP(nn.Module):
    """
    Time-conditioned MLP that maps N features to N features.
    """
    def __init__(self, 
                 feature_dim, 
                 hidden_dims=[256, 512, 256], 
                 time_embed_dim=128,
                 dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeSinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, hidden_dims[0])
        
        # Hidden layers
        self.layers = nn.ModuleList([])
        for i in range(len(hidden_dims) - 1):
            # Add time conditioning to each hidden layer
            time_proj = nn.Linear(time_embed_dim, hidden_dims[i])
            layer = nn.ModuleList([
                time_proj,
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.Dropout(dropout)
            ])
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], feature_dim)
        
        # Final activation - can be modified based on requirements
        self.final_activation = nn.Identity()
    
    def forward(self, x, time):
        """
        Forward pass of the time-conditioned MLP.
        
        Args:
            x: Input features [batch_size, feature_dim]
            time: Time conditioning [batch_size]
            
        Returns:
            Output features [batch_size, feature_dim]
        """
        # Embed time
        time_embed = self.time_embed(time)
        
        # Project input
        h = self.input_proj(x)
        h = F.silu(h)
        
        # Apply hidden layers with time conditioning
        for time_proj, linear, dropout in self.layers:
            # Add time conditioning
            time_emb = time_proj(time_embed)
            h = h + time_emb
            
            # Apply linear, activation, and dropout
            h = linear(h)
            h = F.silu(h)
            h = dropout(h)
        
        # Project to output dimension
        output = self.output_proj(h)
        output = self.final_activation(output)
        
        return output
