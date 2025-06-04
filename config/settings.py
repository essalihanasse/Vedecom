"""
Centralized configuration settings for the VAE pipeline.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

@dataclass
class PathConfig:
    """Path configuration settings."""
    # Base directories
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
        directories = [
            self.DATA_DIR, self.OUTPUT_DIR, self.MODELS_DIR,
            self.VISUALIZATIONS_DIR, self.SAMPLES_DIR, self.TESTS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    HIDDEN_DIM: int = 16
    LATENT_DIM: int = 2
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 100

@dataclass
class TrainingConfig:
    """Training configuration."""
    BETA_VALUES: List[float] = None
    ANNEALING_STRATEGIES: List[str] = None
    SAMPLE_SIZES: List[int] = None
    
    # Early stopping parameters
    EARLY_STOPPING: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.001
    RESTORE_BEST_WEIGHTS: bool = True
    
    # Model checkpoint parameters
    SAVE_BEST_ONLY: bool = True
    CHECKPOINT_MONITOR: str = 'val_loss'
    CHECKPOINT_MODE: str = 'min'
    
    def __post_init__(self):
        if self.BETA_VALUES is None:
            self.BETA_VALUES = [0.1, 0.5, 1.0, 3.0, 5.0]
        
        if self.ANNEALING_STRATEGIES is None:
            self.ANNEALING_STRATEGIES = ['linear', 'exponential']
        
        if self.SAMPLE_SIZES is None:
            self.SAMPLE_SIZES = [100, 400, 900, 1225]

@dataclass
class DataConfig:
    """Data configuration."""
    CATEGORICAL_COLS: List[str] = None
    NUMERICAL_COLS: List[str] = None
    
    def __post_init__(self):
        if self.CATEGORICAL_COLS is None:
            self.CATEGORICAL_COLS = [
                'code_country', 
                'T1_climate_day_period', 
                'T1_climate_dazzled', 
                'T1_climate_fog', 
                'T1_climate_precipitation',
                'NumberOfLanesInPrincipalRoad'
            ]
        
        if self.NUMERICAL_COLS is None:
            self.NUMERICAL_COLS = [
                'FrontCurvature', 
                'T1_ego_speed', 
                'T1_climate_outside_temperature', 
                'T1_V1 (CIPV)_pos_x', 
                'T1_V1 (CIPV)_pos_y', 
                'T1_V1 (CIPV)_absolute_velocity_x', 
                'T1_V1 (CIPV)_absolute_acceleration_x', 
                'T2_V1 (CIPV)_absolute_velocity_x'
            ]

@dataclass
class SamplingConfig:
    """Sampling configuration."""
    DEFAULT_METHODS: List[str] = None
    
    # Representative sampling parameters
    INFO_WEIGHT: float = 1.0
    REDUNDANCY_WEIGHT: float = 1.0
    COVERAGE_RADIUS: float = 0.2
    CANDIDATE_FRACTION: float = 1.0
    
    # Cluster-based parameters
    N_CLUSTERS_FACTOR: float = 0.1
    MIN_CLUSTERS: int = 5
    MAX_CLUSTERS: int = 500  # Increased from 50
    CLUSTER_METHOD: str = 'kmeans'
    WITHIN_CLUSTER_METHOD: str = 'centroid_distance'
    CLUSTER_SIZING_METHOD: str = 'adaptive'
    
    # Hybrid parameters
    CLUSTER_FRACTION: float = 0.7
    DISTANCE_FRACTION: float = 0.3
    
    def __post_init__(self):
        if self.DEFAULT_METHODS is None:
            self.DEFAULT_METHODS = ['equiprobable', 'cluster_based']

# Global configuration instance
class Config:
    """Main configuration class combining all sub-configurations."""
    
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
            'candidate_fraction': self.sampling.CANDIDATE_FRACTION,
            'n_clusters_factor': self.sampling.N_CLUSTERS_FACTOR,
            'min_clusters': self.sampling.MIN_CLUSTERS,
            'max_clusters': self.sampling.MAX_CLUSTERS,
            'cluster_method': self.sampling.CLUSTER_METHOD,
            'within_cluster_method': self.sampling.WITHIN_CLUSTER_METHOD,
            'cluster_sizing_method': self.sampling.CLUSTER_SIZING_METHOD,
            'cluster_fraction': self.sampling.CLUSTER_FRACTION,
            'distance_fraction': self.sampling.DISTANCE_FRACTION
        }

# Global config instance
config = Config()

# Backward compatibility - expose commonly used paths and settings
DATA_DIR = config.paths.DATA_DIR
OUTPUT_DIR = config.paths.OUTPUT_DIR
MODELS_DIR = config.paths.MODELS_DIR
VISUALIZATIONS_DIR = config.paths.VISUALIZATIONS_DIR
SAMPLES_DIR = config.paths.SAMPLES_DIR
TESTS_DIR = config.paths.TESTS_DIR
DATA_FILE = config.paths.DATA_FILE
PREPROCESSED_FILE = config.paths.PREPROCESSED_FILE

HIDDEN_DIM = config.model.HIDDEN_DIM
LATENT_DIM = config.model.LATENT_DIM
BATCH_SIZE = config.model.BATCH_SIZE
LEARNING_RATE = config.model.LEARNING_RATE
NUM_EPOCHS = config.model.NUM_EPOCHS

BETA_VALUES = config.training.BETA_VALUES
ANNEALING_STRATEGIES = config.training.ANNEALING_STRATEGIES
SAMPLE_SIZES = config.training.SAMPLE_SIZES

CATEGORICAL_COLS = config.data.CATEGORICAL_COLS
NUMERICAL_COLS = config.data.NUMERICAL_COLS