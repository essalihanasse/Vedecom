"""
Enhanced configuration settings for the VAE pipeline with multiple latent dimensions support.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class PathConfig:
    """Path configuration settings."""
    DATA_DIR: str = 'data'
    OUTPUT_DIR: str = 'output'
    
    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'models')
    
    @property
    def VISUALIZATIONS_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'visualizations')
    
    @property
    def SAMPLES_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'samples')
    
    @property
    def TESTS_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, 'tests')
    
    @property
    def DATA_FILE(self) -> str:
        return os.path.join(self.DATA_DIR, 'data.csv')
    
    @property
    def PREPROCESSED_FILE(self) -> str:
        return os.path.join(self.DATA_DIR, 'preprocessed_data.csv')
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        for directory in [self.DATA_DIR, self.OUTPUT_DIR, self.MODELS_DIR, 
                         self.VISUALIZATIONS_DIR, self.SAMPLES_DIR, self.TESTS_DIR]:
            os.makedirs(directory, exist_ok=True)

@dataclass
class ModelConfig:
    """Model architecture and training configuration with multiple latent dimensions."""
    # Architecture - Now supports multiple latent dimensions
    HIDDEN_DIM: int = 16
    LATENT_DIMS: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])  # Multiple latent dimensions to test
    
    # Training
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 300
    
    # Latent dimension specific settings
    LATENT_DIM_CONFIGS: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        2: {'hidden_multiplier': 1.0, 'lr_multiplier': 1.0},
        4: {'hidden_multiplier': 1.2, 'lr_multiplier': 1.0},
        8: {'hidden_multiplier': 1.5, 'lr_multiplier': 0.9},
        16: {'hidden_multiplier': 2.0, 'lr_multiplier': 0.8},
        32: {'hidden_multiplier': 2.5, 'lr_multiplier': 0.7},
    })
    
    def get_config_for_latent_dim(self, latent_dim: int) -> Dict[str, Any]:
        """Get configuration parameters for specific latent dimension."""
        base_config = {
            'hidden_dim': self.HIDDEN_DIM,
            'latent_dim': latent_dim,
            'batch_size': self.BATCH_SIZE,
            'learning_rate': self.LEARNING_RATE,
            'num_epochs': self.NUM_EPOCHS
        }
        
        # Apply dimension-specific modifications
        if latent_dim in self.LATENT_DIM_CONFIGS:
            dim_config = self.LATENT_DIM_CONFIGS[latent_dim]
            base_config['hidden_dim'] = int(self.HIDDEN_DIM * dim_config['hidden_multiplier'])
            base_config['learning_rate'] = self.LEARNING_RATE * dim_config['lr_multiplier']
        
        return base_config

@dataclass  
class TrainingConfig:
    """Training configuration."""
    BETA_VALUES: List[float] = field(default_factory=lambda: [1.0, 10, 100])
    ANNEALING_STRATEGIES: List[str] = field(default_factory=lambda: ['linear', 'exponential'])
    SAMPLE_SIZES: List[int] = field(default_factory=lambda: [100, 400, 900])
    
    # Early stopping
    EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 20
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    RESTORE_BEST_WEIGHTS: bool = True
    
    # Checkpointing
    SAVE_BEST_ONLY: bool = True
    CHECKPOINT_MONITOR: str = 'val_loss'
    CHECKPOINT_MODE: str = 'min'
    
    # Latent dimension specific training settings
    LATENT_DIM_TRAINING: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        2: {'patience': 10, 'min_delta': 0.001},
        4: {'patience': 12, 'min_delta': 0.0008},
        8: {'patience': 15, 'min_delta': 0.0005},
        16: {'patience': 20, 'min_delta': 0.0003},
        32: {'patience': 25, 'min_delta': 0.0001},
    })

@dataclass
class DataConfig:
    """Data configuration."""
    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        'code_country', 'T1_climate_day_period', 'T1_climate_dazzled', 
        'T1_climate_fog', 'T1_climate_precipitation', 'NumberOfLanesInPrincipalRoad'
    ])
    
    NUMERICAL_COLS: List[str] = field(default_factory=lambda: [
        'FrontCurvature', 'T1_ego_speed', 'T1_climate_outside_temperature', 
        'T1_V1 (CIPV)_pos_x', 'T1_V1 (CIPV)_pos_y', 'T1_V1 (CIPV)_absolute_velocity_x', 
        'T1_V1 (CIPV)_absolute_acceleration_x', 'T2_V1 (CIPV)_absolute_velocity_x'
    ])

@dataclass
class SamplingConfig:
    """Sampling configuration."""
    DEFAULT_METHODS: List[str] = field(default_factory=lambda: ['equiprobable', 'cluster_based', 'latin_hypercube'])
    
    # Core parameters
    INFO_WEIGHT: float = 1.0
    REDUNDANCY_WEIGHT: float = 1.0
    COVERAGE_RADIUS: float = 0.2
    
    # Clustering parameters
    CLUSTER_METHOD: str = 'kmeans'
    CLUSTER_SIZING_METHOD: str = 'adaptive'
    MIN_CLUSTERS: int = 5
    MAX_CLUSTERS: int = 500
    
    # Hybrid parameters
    CLUSTER_FRACTION: float = 0.7
    DISTANCE_FRACTION: float = 0.3
    
    # Latent dimension specific sampling parameters
    LATENT_DIM_SAMPLING: Dict[int, Dict[str, Any]] = field(default_factory=lambda: {
        2: {'coverage_radius': 0.2, 'min_clusters': 5},
        4: {'coverage_radius': 0.25, 'min_clusters': 8},
        8: {'coverage_radius': 0.3, 'min_clusters': 12},
        16: {'coverage_radius': 0.35, 'min_clusters': 20},
        32: {'coverage_radius': 0.4, 'min_clusters': 30},
    })

class Config:
    """Main configuration class with latent dimension support."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.sampling = SamplingConfig()
        
        # Create directories on initialization
        self.paths.create_directories()
    
    def get_model_params(self, latent_dim: int) -> Dict[str, Any]:
        """Get model parameters as dictionary for specific latent dimension."""
        return self.model.get_config_for_latent_dim(latent_dim)
    
    def get_sampling_params(self, latent_dim: int) -> Dict[str, Any]:
        """Get sampling parameters as dictionary for specific latent dimension."""
        base_params = {
            'info_weight': self.sampling.INFO_WEIGHT,
            'redundancy_weight': self.sampling.REDUNDANCY_WEIGHT,
            'coverage_radius': self.sampling.COVERAGE_RADIUS,
            'cluster_method': self.sampling.CLUSTER_METHOD,
            'cluster_sizing_method': self.sampling.CLUSTER_SIZING_METHOD,
            'min_clusters': self.sampling.MIN_CLUSTERS,
            'max_clusters': self.sampling.MAX_CLUSTERS,
            'cluster_fraction': self.sampling.CLUSTER_FRACTION,
            'distance_fraction': self.sampling.DISTANCE_FRACTION
        }
        
        # Apply latent dimension specific parameters
        if latent_dim in self.sampling.LATENT_DIM_SAMPLING:
            dim_config = self.sampling.LATENT_DIM_SAMPLING[latent_dim]
            base_params.update(dim_config)
        
        return base_params
    
    def get_training_params(self, latent_dim: int) -> Dict[str, Any]:
        """Get training parameters for specific latent dimension."""
        base_params = {
            'early_stopping': self.training.EARLY_STOPPING,
            'patience': self.training.EARLY_STOPPING_PATIENCE,
            'min_delta': self.training.EARLY_STOPPING_MIN_DELTA,
            'restore_best_weights': self.training.RESTORE_BEST_WEIGHTS
        }
        
        # Apply latent dimension specific parameters
        if latent_dim in self.training.LATENT_DIM_TRAINING:
            dim_config = self.training.LATENT_DIM_TRAINING[latent_dim]
            base_params.update(dim_config)
        
        return base_params

# Global config instance
config = Config()