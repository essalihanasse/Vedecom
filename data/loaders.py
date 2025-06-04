

# data/loaders.py
"""
Data loading utilities for the VAE pipeline.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)

class VAEDataset(Dataset):
    """
    Custom dataset for VAE training.
    """
    
    def __init__(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: Input data array
            transform: Optional transform function
        """
        if isinstance(data, np.ndarray):
            self.data = torch.FloatTensor(data)
        else:
            self.data = data.float()
        
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class DataLoader:
    """
    Enhanced data loader for VAE pipeline.
    """
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw data from file.
        
        Args:
            filepath: Path to data file (uses config default if None)
            
        Returns:
            Loaded DataFrame
        """
        if filepath is None:
            filepath = self.config.paths.DATA_FILE
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Support different file formats
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif file_ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif file_ext == '.json':
            df = pd.read_json(filepath)
        else:
            # Default to CSV
            df = pd.read_csv(filepath)
        
        logger.info(f"Loaded raw data: {df.shape} from {filepath}")
        return df
    
    def load_preprocessed_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load preprocessed data.
        
        Args:
            filepath: Path to preprocessed file (uses config default if None)
            
        Returns:
            Preprocessed DataFrame
        """
        if filepath is None:
            filepath = self.config.paths.PREPROCESSED_FILE
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessed data not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded preprocessed data: {df.shape}")
        return df
    
    def load_preprocessing_objects(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Load preprocessing objects (scalers, encoders, etc.).
        
        Args:
            directory: Directory containing preprocessing objects
            
        Returns:
            Dictionary with preprocessing objects
        """
        if directory is None:
            directory = self.config.paths.DATA_DIR
        
        objects_file = os.path.join(directory, 'preprocessing_objects.pkl')
        
        if not os.path.exists(objects_file):
            raise FileNotFoundError(f"Preprocessing objects not found: {objects_file}")
        
        with open(objects_file, 'rb') as f:
            objects = pickle.load(f)
        
        logger.info("Loaded preprocessing objects")
        return objects
    
    def create_data_loaders(
        self, 
        train_data: np.ndarray, 
        val_data: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
        """
        Create PyTorch data loaders.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            batch_size: Batch size (uses config default if None)
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader or tuple of (train_loader, val_loader)
        """
        if batch_size is None:
            batch_size = self.config.model.BATCH_SIZE
        
        # Create training dataset and loader
        train_dataset = VAEDataset(train_data)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        if val_data is not None:
            # Create validation dataset and loader
            val_dataset = VAEDataset(val_data)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            return train_loader, val_loader
        
        return train_loader
    
    def load_model_checkpoint(self, strategy: str, beta: float) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            strategy: Annealing strategy
            beta: Beta value
            
        Returns:
            Loaded checkpoint dictionary
        """
        model_path = os.path.join(
            self.config.paths.MODELS_DIR,
            strategy,
            f'beta_{beta}',
            'vae_model_final.pth'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        logger.info(f"Loaded model checkpoint: {model_path}")
        return checkpoint
    
    def load_sampling_results(
        self, 
        strategy: str, 
        beta: float, 
        method: str, 
        sample_size: int
    ) -> pd.DataFrame:
        """
        Load sampling results.
        
        Args:
            strategy: Annealing strategy
            beta: Beta value
            method: Sampling method
            sample_size: Sample size
            
        Returns:
            DataFrame with sampling results
        """
        results_path = os.path.join(
            self.config.paths.SAMPLES_DIR,
            strategy,
            f'beta_{beta}',
            f'method_{method}',
            f'samples_{sample_size}',
            'selected_points.csv'
        )
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Sampling results not found: {results_path}")
        
        df = pd.read_csv(results_path)
        logger.info(f"Loaded sampling results: {df.shape}")
        return df
    
    def get_available_models(self) -> List[Tuple[str, float]]:
        """
        Get list of available trained models.
        
        Returns:
            List of (strategy, beta) tuples
        """
        models = []
        models_dir = self.config.paths.MODELS_DIR
        
        if not os.path.exists(models_dir):
            return models
        
        for strategy in os.listdir(models_dir):
            strategy_dir = os.path.join(models_dir, strategy)
            if os.path.isdir(strategy_dir):
                for beta_dir in os.listdir(strategy_dir):
                    if beta_dir.startswith('beta_'):
                        try:
                            beta_value = float(beta_dir.replace('beta_', ''))
                            model_file = os.path.join(strategy_dir, beta_dir, 'vae_model_final.pth')
                            if os.path.exists(model_file):
                                models.append((strategy, beta_value))
                        except ValueError:
                            continue
        
        return sorted(models)
    
    def get_available_sampling_results(self) -> List[Tuple[str, float, str, int]]:
        """
        Get list of available sampling results.
        
        Returns:
            List of (strategy, beta, method, sample_size) tuples
        """
        results = []
        samples_dir = self.config.paths.SAMPLES_DIR
        
        if not os.path.exists(samples_dir):
            return results
        
        for strategy in os.listdir(samples_dir):
            strategy_dir = os.path.join(samples_dir, strategy)
            if os.path.isdir(strategy_dir):
                for beta_dir in os.listdir(strategy_dir):
                    if beta_dir.startswith('beta_'):
                        try:
                            beta_value = float(beta_dir.replace('beta_', ''))
                            beta_path = os.path.join(strategy_dir, beta_dir)
                            
                            for method_dir in os.listdir(beta_path):
                                if method_dir.startswith('method_'):
                                    method = method_dir.replace('method_', '')
                                    method_path = os.path.join(beta_path, method_dir)
                                    
                                    for sample_dir in os.listdir(method_path):
                                        if sample_dir.startswith('samples_'):
                                            try:
                                                sample_size = int(sample_dir.replace('samples_', ''))
                                                results_file = os.path.join(method_path, sample_dir, 'selected_points.csv')
                                                if os.path.exists(results_file):
                                                    results.append((strategy, beta_value, method, sample_size))
                                            except ValueError:
                                                continue
                        except ValueError:
                            continue
        
        return sorted(results)

def create_data_loader(config) -> DataLoader:
    """
    Factory function to create data loader.
    
    Args:
        config: Configuration object
        
    Returns:
        Initialized DataLoader
    """
    return DataLoader(config)