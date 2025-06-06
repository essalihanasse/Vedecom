"""
Optimized configuration settings for the VAE pipeline.
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
    """Model architecture and training configuration."""
    # Architecture
    HIDDEN_DIM: int = 16
    LATENT_DIM: int = 2
    
    # Training
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 50

@dataclass  
class TrainingConfig:
    """Training configuration."""
    BETA_VALUES: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    ANNEALING_STRATEGIES: List[str] = field(default_factory=lambda: ['linear', 'exponential'])
    SAMPLE_SIZES: List[int] = field(default_factory=lambda: [100, 400, 900])
    
    # Early stopping
    EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    RESTORE_BEST_WEIGHTS: bool = True
    
    # Checkpointing
    SAVE_BEST_ONLY: bool = True
    CHECKPOINT_MONITOR: str = 'val_loss'
    CHECKPOINT_MODE: str = 'min'

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
    DEFAULT_METHODS: List[str] = field(default_factory=lambda: ['equiprobable', 'cluster_based'])
    
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

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.sampling = SamplingConfig()
        
        # Create directories on initialization
        self.paths.create_directories()
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters as dictionary."""
        return {
            'hidden_dim': self.model.HIDDEN_DIM,
            'latent_dim': self.model.LATENT_DIM,
            'batch_size': self.model.BATCH_SIZE,
            'learning_rate': self.model.LEARNING_RATE,
            'num_epochs': self.model.NUM_EPOCHS
        }
    
    def get_sampling_params(self) -> Dict[str, Any]:
        """Get sampling parameters as dictionary."""
        return {
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

# Global config instance
config = Config()