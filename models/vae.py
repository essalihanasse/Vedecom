"""
Cleaned VAE model definition for mixed numerical and categorical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class VAE(nn.Module):
    """
    Variational Autoencoder for mixed numerical and categorical data.
    
    Args:
        input_dim: Total input dimension
        num_numerical: Number of numerical features
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        cat_dict: Dictionary with categorical feature information
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_numerical: int, 
        hidden_dim: int = 64, 
        latent_dim: int = 2, 
        cat_dict: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.2
    ):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_numerical = num_numerical
        self.cat_dict = cat_dict or {}
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (common hidden layers)
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Separate output for numerical features
        self.decoder_numerical = nn.Linear(hidden_dim, self.num_numerical)
        
        # Separate decoders for each categorical variable
        self.decoder_categoricals = nn.ModuleDict()
        for feature, info in self.cat_dict.items():
            self.decoder_categoricals[feature] = nn.Linear(hidden_dim, info['cardinality'])
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent space parameters.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (mu, logvar) of the latent space
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent space.
        
        Args:
            mu: Mean of latent space
            logvar: Log variance of latent space
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent vector to reconstructed input.
        
        Args:
            z: Latent vector
            
        Returns:
            Dictionary with reconstructed numerical and categorical outputs
        """
        h = self.decoder_hidden(z)
        
        # Decode numerical features
        x_numerical = self.decoder_numerical(h)
        
        decoded_outputs = {'numerical': x_numerical}
        
        # Decode categorical features with softmax activation
        for feature, decoder in self.decoder_categoricals.items():
            logits = decoder(h)
            probs = F.softmax(logits, dim=1)
            decoded_outputs[feature] = probs
        
        return decoded_outputs
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (decoded_outputs, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_outputs = self.decode(z)
        return decoded_outputs, mu, logvar

class VAELoss:
    """VAE loss function for mixed numerical and categorical variables."""
    
    def __init__(self, cat_dict: Dict[str, Any], num_numerical: int):
        self.cat_dict = cat_dict
        self.num_numerical = num_numerical
    
    def __call__(
        self, 
        decoded_outputs: Dict[str, torch.Tensor], 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            decoded_outputs: Dictionary with reconstruction outputs
            x: Original input data
            mu: Mean of latent space
            logvar: Log variance of latent space
            beta: Weight for KL divergence term
            
        Returns:
            Tuple of (total_loss, numerical_loss, categorical_loss, kl_loss)
        """
        # MSE loss for numerical features
        x_numerical = x[:, :self.num_numerical]
        numerical_output = decoded_outputs['numerical']
        numerical_loss = F.mse_loss(numerical_output, x_numerical, reduction='sum')
        
        # Cross-entropy loss for categorical features
        categorical_loss = torch.tensor(0.0, device=x.device)
        for feature, info in self.cat_dict.items():
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            # Extract the one-hot encoded target for this categorical feature
            target_onehot = x[:, start_idx:end_idx]
            predicted_probs = decoded_outputs[feature]
            
            # Convert one-hot to class indices
            target_indices = torch.argmax(target_onehot, dim=1)
            
            # Compute cross entropy loss
            cat_loss = F.cross_entropy(predicted_probs, target_indices, reduction='sum')
            categorical_loss += cat_loss
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply beta factor to KL divergence term
        weighted_kl_loss = beta * kl_loss
        
        # Total loss
        total_loss = numerical_loss + categorical_loss + weighted_kl_loss
        
        return total_loss, numerical_loss, categorical_loss, weighted_kl_loss

class BetaScheduler:
    """Beta annealing scheduler for VAE training."""
    
    @staticmethod
    def get_schedule(
        beta_start: float, 
        beta_end: float, 
        num_epochs: int, 
        strategy: str = 'linear'
    ) -> List[float]:
        """
        Generate beta annealing schedule.
        
        Args:
            beta_start: Starting beta value
            beta_end: Target beta value
            num_epochs: Total number of epochs
            strategy: Annealing strategy ('constant', 'linear', 'exponential', 'cyclical')
            
        Returns:
            List of beta values for each epoch
        """
        epochs = np.arange(num_epochs)
        
        if strategy == 'constant':
            return [beta_end] * num_epochs
        
        elif strategy == 'linear':
            return np.linspace(beta_start, beta_end, num_epochs).tolist()
        
        elif strategy == 'exponential':
            t = epochs / (num_epochs - 1)
            beta_schedule = beta_start + (beta_end - beta_start) * \
                           (np.exp(3 * t) - 1) / (np.exp(3) - 1)
            return beta_schedule.tolist()
        
        elif strategy == 'cyclical':
            cycles = 3
            t = cycles * epochs / (num_epochs - 1)
            cycle_pos = t - np.floor(t)
            beta_schedule = beta_start + (beta_end - beta_start) * \
                           (0.5 + 0.5 * np.sin(2 * np.pi * cycle_pos - np.pi/2))
            return beta_schedule.tolist()
        
        else:
            logger.warning(f"Unknown annealing strategy: {strategy}. Using constant.")
            return [beta_end] * num_epochs

def get_latent_encoding(
    model: VAE, 
    data_tensor: torch.Tensor, 
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:
    """
    Get latent space encodings for input data in batches.
    
    Args:
        model: The VAE model
        data_tensor: Input data tensor
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Mean encodings in latent space as numpy array
    """
    model.eval()
    data_tensor = data_tensor.to(device)
    
    encodings = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            mu, _ = model.encode(batch)
            encodings.append(mu.cpu().numpy())
    
    return np.vstack(encodings)

def create_model(
    input_dim: int,
    num_numerical: int,
    cat_dict: Dict[str, Any],
    hidden_dim: int = 64,
    latent_dim: int = 2,
    dropout_rate: float = 0.2
) -> VAE:
    """
    Factory function to create a VAE model.
    
    Args:
        input_dim: Total input dimension
        num_numerical: Number of numerical features
        cat_dict: Dictionary with categorical feature information
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout_rate: Dropout rate
        
    Returns:
        Initialized VAE model
    """
    return VAE(
        input_dim=input_dim,
        num_numerical=num_numerical,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        cat_dict=cat_dict,
        dropout_rate=dropout_rate
    )