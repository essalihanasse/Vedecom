"""
Enhanced VAE model definition supporting multiple latent dimensions with adaptive architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class AdaptiveVAE(nn.Module):
    """
    Enhanced Variational Autoencoder with adaptive architecture for different latent dimensions.
    
    Args:
        input_dim: Total input dimension
        num_numerical: Number of numerical features
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        cat_dict: Dictionary with categorical feature information
        dropout_rate: Dropout rate for regularization
        architecture_type: Type of architecture ('adaptive', 'deep', 'wide')
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_numerical: int, 
        hidden_dim: int = 64, 
        latent_dim: int = 2, 
        cat_dict: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.2,
        architecture_type: str = 'adaptive'
    ):
        super(AdaptiveVAE, self).__init__()
        
        self.input_dim = input_dim
        self.num_numerical = num_numerical
        self.latent_dim = latent_dim
        self.cat_dict = cat_dict or {}
        self.dropout_rate = dropout_rate
        self.architecture_type = architecture_type
        
        # Adaptive architecture based on latent dimension
        self.encoder_layers = self._build_adaptive_encoder(input_dim, hidden_dim, latent_dim)
        self.decoder_layers = self._build_adaptive_decoder(latent_dim, hidden_dim, architecture_type)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(self.encoder_layers[-1].out_features, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_layers[-1].out_features, latent_dim)
        
        # Separate output for numerical features
        self.decoder_numerical = nn.Linear(self.decoder_layers[-1].out_features, self.num_numerical)
        
        # Separate decoders for each categorical variable
        self.decoder_categoricals = nn.ModuleDict()
        for feature, info in self.cat_dict.items():
            self.decoder_categoricals[feature] = nn.Linear(
                self.decoder_layers[-1].out_features, 
                info['cardinality']
            )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"AdaptiveVAE initialized: latent_dim={latent_dim}, "
                   f"architecture={architecture_type}, total_params={self._count_parameters()}")
    
    def _build_adaptive_encoder(self, input_dim: int, hidden_dim: int, latent_dim: int) -> nn.ModuleList:
        """Build adaptive encoder based on latent dimension."""
        layers = nn.ModuleList()
        
        if latent_dim <= 4:
            # Shallow architecture for low-dimensional latent spaces
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ])
        elif latent_dim <= 16:
            # Medium depth architecture
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ])
        else:
            # Deeper architecture for high-dimensional latent spaces
            layers.extend([
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            ])
        
        return layers
    
    def _build_adaptive_decoder(self, latent_dim: int, hidden_dim: int, architecture_type: str) -> nn.ModuleList:
        """Build adaptive decoder based on latent dimension and architecture type."""
        layers = nn.ModuleList()
        
        if architecture_type == 'deep' and latent_dim > 8:
            # Deep architecture for complex reconstructions
            layers.extend([
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU()
            ])
        elif architecture_type == 'wide':
            # Wide architecture
            wide_dim = hidden_dim * 2 if latent_dim > 8 else hidden_dim
            layers.extend([
                nn.Linear(latent_dim, wide_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(wide_dim, wide_dim),
                nn.ReLU()
            ])
        else:
            # Adaptive standard architecture
            if latent_dim <= 4:
                layers.extend([
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU()
                ])
            elif latent_dim <= 16:
                layers.extend([
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU()
                ])
        
        return layers
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent space parameters.
        
        Args:
            x: Input data tensor
            
        Returns:
            Tuple of (mu, logvar) of the latent space
        """
        # Forward through encoder layers
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Apply latent dimension specific constraints
        if self.latent_dim > 16:
            # For high-dimensional latent spaces, apply gentle regularization
            mu = torch.tanh(mu) * 3  # Constrain to reasonable range
            logvar = torch.clamp(logvar, min=-10, max=2)  # Prevent extreme variances
        
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
        
        # Add noise scaling for different latent dimensions
        if self.latent_dim <= 2:
            noise_scale = 1.0
        elif self.latent_dim <= 8:
            noise_scale = 0.9
        else:
            noise_scale = 0.8
        
        return mu + eps * std * noise_scale
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode latent vector to reconstructed input.
        
        Args:
            z: Latent vector
            
        Returns:
            Dictionary with reconstructed numerical and categorical outputs
        """
        # Forward through decoder layers
        h = z
        for layer in self.decoder_layers:
            h = layer(h)
        
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
    
    def get_latent_representation(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Get latent representation of input data.
        
        Args:
            x: Input data tensor
            use_mean: If True, use mean of latent distribution, else sample
            
        Returns:
            Latent representation
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)

class EnhancedVAELoss:
    """Enhanced VAE loss function with latent dimension specific weighting."""
    
    def __init__(self, cat_dict: Dict[str, Any], num_numerical: int, latent_dim: int = 2):
        self.cat_dict = cat_dict
        self.num_numerical = num_numerical
        self.latent_dim = latent_dim
        
        # Latent dimension specific loss weights
        self.recon_weight = self._get_reconstruction_weight(latent_dim)
        self.kl_weight_base = self._get_kl_weight_base(latent_dim)
    
    def _get_reconstruction_weight(self, latent_dim: int) -> float:
        """Get reconstruction loss weight based on latent dimension."""
        if latent_dim <= 2:
            return 1.0
        elif latent_dim <= 8:
            return 1.1
        else:
            return 1.2
    
    def _get_kl_weight_base(self, latent_dim: int) -> float:
        """Get base KL divergence weight based on latent dimension."""
        if latent_dim <= 2:
            return 1.0
        elif latent_dim <= 8:
            return 0.9
        else:
            return 0.8
    
    def __call__(
        self, 
        decoded_outputs: Dict[str, torch.Tensor], 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute enhanced VAE loss with latent dimension adaptation.
        
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
        numerical_loss = F.mse_loss(numerical_output, x_numerical, reduction='sum') * self.recon_weight
        
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
        
        categorical_loss = categorical_loss * self.recon_weight
        
        # Enhanced KL divergence loss with dimension-specific weighting
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply dimension-specific and beta weighting
        effective_beta = beta * self.kl_weight_base
        
        # For high-dimensional latent spaces, add capacity constraint
        if self.latent_dim > 8:
            # Gradually increase KL weight to prevent posterior collapse
            capacity = min(self.latent_dim * 0.1, 2.0)
            kl_loss = torch.clamp(kl_loss - capacity, min=0.0)
        
        weighted_kl_loss = effective_beta * kl_loss
        
        # Total loss
        total_loss = numerical_loss + categorical_loss + weighted_kl_loss
        
        return total_loss, numerical_loss, categorical_loss, weighted_kl_loss

def create_adaptive_model(
    input_dim: int,
    num_numerical: int,
    cat_dict: Dict[str, Any],
    latent_dim: int = 2,
    hidden_dim: int = 64,
    dropout_rate: float = 0.2,
    architecture_type: str = 'adaptive'
) -> AdaptiveVAE:
    """
    Factory function to create an adaptive VAE model.
    
    Args:
        input_dim: Total input dimension
        num_numerical: Number of numerical features
        cat_dict: Dictionary with categorical feature information
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        dropout_rate: Dropout rate
        architecture_type: Architecture type ('adaptive', 'deep', 'wide')
        
    Returns:
        Initialized AdaptiveVAE model
    """
    return AdaptiveVAE(
        input_dim=input_dim,
        num_numerical=num_numerical,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        cat_dict=cat_dict,
        dropout_rate=dropout_rate,
        architecture_type=architecture_type
    )

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

def get_latent_encoding_adaptive(
    model: AdaptiveVAE, 
    data_tensor: torch.Tensor, 
    device: torch.device,
    batch_size: int = 512,
    use_mean: bool = True
) -> np.ndarray:
    """
    Get latent space encodings for input data in batches (adaptive version).
    
    Args:
        model: The AdaptiveVAE model
        data_tensor: Input data tensor
        device: Device to run on
        batch_size: Batch size for processing
        use_mean: Whether to use mean or sample from latent distribution
        
    Returns:
        Latent encodings as numpy array
    """
    model.eval()
    data_tensor = data_tensor.to(device)
    
    encodings = []
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            latent_repr = model.get_latent_representation(batch, use_mean=use_mean)
            encodings.append(latent_repr.cpu().numpy())
    
    return np.vstack(encodings)

# For backward compatibility, also create the original functions
def create_model(
    input_dim: int,
    num_numerical: int,
    cat_dict: Dict[str, Any],
    hidden_dim: int = 64,
    latent_dim: int = 2,
    dropout_rate: float = 0.2
) -> AdaptiveVAE:
    """
    Factory function to create a VAE model (backward compatibility).
    
    Args:
        input_dim: Total input dimension
        num_numerical: Number of numerical features
        cat_dict: Dictionary with categorical feature information
        hidden_dim: Hidden layer dimension
        latent_dim: Latent space dimension
        dropout_rate: Dropout rate
        
    Returns:
        Initialized AdaptiveVAE model
    """
    return create_adaptive_model(
        input_dim=input_dim,
        num_numerical=num_numerical,
        cat_dict=cat_dict,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        architecture_type='adaptive'
    )

def get_latent_encoding(
    model: AdaptiveVAE, 
    data_tensor: torch.Tensor, 
    device: torch.device,
    batch_size: int = 512
) -> np.ndarray:
    """
    Get latent space encodings for input data in batches (backward compatibility).
    
    Args:
        model: The VAE model
        data_tensor: Input data tensor
        device: Device to run on
        batch_size: Batch size for processing
        
    Returns:
        Mean encodings in latent space as numpy array
    """
    return get_latent_encoding_adaptive(model, data_tensor, device, batch_size, use_mean=True)

# Legacy alias for VAE
VAE = AdaptiveVAE

# Legacy alias for VAELoss
VAELoss = EnhancedVAELoss