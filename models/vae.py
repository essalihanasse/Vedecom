"""
Fixed VAE model definition with corrected architecture.
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
        
        # Build encoder - fix the out_features issue
        self.encoder = self._build_encoder(input_dim, hidden_dim, latent_dim)
        
        # Get the output dimension of the last encoder layer
        encoder_output_dim = self._get_encoder_output_dim()
        
        # Latent space parameters
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        
        # Build decoder
        self.decoder = self._build_decoder(latent_dim, hidden_dim, architecture_type)
        
        # Get decoder output dimension
        decoder_output_dim = self._get_decoder_output_dim()
        
        # Separate output for numerical features
        self.decoder_numerical = nn.Linear(decoder_output_dim, self.num_numerical)
        
        # Separate decoders for each categorical variable
        self.decoder_categoricals = nn.ModuleDict()
        for feature, info in self.cat_dict.items():
            self.decoder_categoricals[feature] = nn.Linear(
                decoder_output_dim, 
                info['cardinality']
            )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"AdaptiveVAE initialized: latent_dim={latent_dim}, "
                   f"architecture={architecture_type}, total_params={self._count_parameters()}")
    
    def _build_encoder(self, input_dim: int, hidden_dim: int, latent_dim: int) -> nn.Sequential:
        """Build encoder with proper layer tracking."""
        layers = []
        
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
        
        return nn.Sequential(*layers)
    
    def _get_encoder_output_dim(self) -> int:
        """Get the output dimension of the encoder."""
        # Find the last Linear layer in the encoder
        for layer in reversed(self.encoder):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        
        # Fallback - should never happen with proper architecture
        raise ValueError("No Linear layer found in encoder")
    
    def _build_decoder(self, latent_dim: int, hidden_dim: int, architecture_type: str) -> nn.Sequential:
        """Build decoder with proper layer tracking."""
        layers = []
        
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
        
        return nn.Sequential(*layers)
    
    def _get_decoder_output_dim(self) -> int:
        """Get the output dimension of the decoder."""
        # Find the last Linear layer in the decoder
        for layer in reversed(self.decoder):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        
        # Fallback - should never happen with proper architecture
        raise ValueError("No Linear layer found in decoder")
    
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
        """
        # Forward through encoder
        h = self.encoder(x)
        
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
        """
        # Forward through decoder
        h = self.decoder(z)
        
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
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_outputs = self.decode(z)
        return decoded_outputs, mu, logvar
    
    def get_latent_representation(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        """
        Get latent representation of input data.
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)


# Factory functions for backwards compatibility
def create_adaptive_model(
    input_dim: int,
    num_numerical: int,
    cat_dict: Dict[str, Any],
    latent_dim: int = 2,
    hidden_dim: int = 64,
    dropout_rate: float = 0.2,
    architecture_type: str = 'adaptive'
) -> AdaptiveVAE:
    """Factory function to create an adaptive VAE model."""
    return AdaptiveVAE(
        input_dim=input_dim,
        num_numerical=num_numerical,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        cat_dict=cat_dict,
        dropout_rate=dropout_rate,
        architecture_type=architecture_type
    )

def create_model(
    input_dim: int,
    num_numerical: int,
    cat_dict: Dict[str, Any],
    hidden_dim: int = 64,
    latent_dim: int = 2,
    dropout_rate: float = 0.2
) -> AdaptiveVAE:
    """Factory function to create a VAE model (backward compatibility)."""
    return create_adaptive_model(
        input_dim=input_dim,
        num_numerical=num_numerical,
        cat_dict=cat_dict,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        architecture_type='adaptive'
    )

# Legacy aliases
VAE = AdaptiveVAE