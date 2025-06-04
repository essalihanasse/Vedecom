# utils/__init__.py
"""
Utility modules for the VAE pipeline.
"""

from .math_utils import *
from .file_utils import *

# utils/math_utils.py
"""
Mathematical utility functions for the VAE pipeline.
"""

import numpy as np
import torch
from typing import Union, Tuple, List, Optional
import math

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_grid_dimensions(sample_size: int) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for a given sample size.
    
    Args:
        sample_size: Number of samples
        
    Returns:
        Tuple of (rows, cols)
    """
    # For perfect squares, use square grid
    sqrt_size = int(math.sqrt(sample_size))
    if sqrt_size * sqrt_size == sample_size:
        return sqrt_size, sqrt_size
    
    # Find closest factorization
    best_diff = float('inf')
    best_rows, best_cols = 1, sample_size
    
    for i in range(1, int(math.sqrt(sample_size)) + 1):
        if sample_size % i == 0:
            rows, cols = i, sample_size // i
            diff = abs(rows - cols)
            if diff < best_diff:
                best_diff = diff
                best_rows, best_cols = rows, cols
    
    return best_rows, best_cols

def normalize_array(arr: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize array using different methods.
    
    Args:
        arr: Input array
        method: Normalization method ('zscore', 'minmax', 'unit')
        
    Returns:
        Normalized array
    """
    if method == 'zscore':
        return (arr - np.mean(arr)) / np.std(arr)
    elif method == 'minmax':
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    elif method == 'unit':
        return arr / np.linalg.norm(arr)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def safe_log(x: Union[np.ndarray, torch.Tensor], eps: float = 1e-8) -> Union[np.ndarray, torch.Tensor]:
    """
    Safe logarithm that avoids log(0).
    
    Args:
        x: Input values
        eps: Small epsilon to add for numerical stability
        
    Returns:
        Safe logarithm of x
    """
    if isinstance(x, torch.Tensor):
        return torch.log(torch.clamp(x, min=eps))
    else:
        return np.log(np.maximum(x, eps))

def compute_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence for VAE.
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        
    Returns:
        KL divergence
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def batch_pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distances between two sets of points.
    
    Args:
        x: First set of points (batch_size, dim)
        y: Second set of points (batch_size, dim)
        
    Returns:
        Pairwise distances
    """
    return torch.cdist(x, y)

def entropy_from_probabilities(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute entropy from probability distributions.
    
    Args:
        probs: Probability distributions
        axis: Axis along which to compute entropy
        
    Returns:
        Entropy values
    """
    return -np.sum(probs * safe_log(probs), axis=axis)

# utils/file_utils.py
"""
File handling utilities for the VAE pipeline.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)

def ensure_dir(path: Union[str, Path]) -> None:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)

def safe_filename(filename: str) -> str:
    """
    Create safe filename by removing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    # Remove or replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'[^\w\-_\.]', '_', safe_name)
    return safe_name

def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_data = {}
    for key, value in data.items():
        serializable_data[key] = convert_numpy(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object as pickle file.
    
    Args:
        obj: Object to save
        filepath: Output file path
    """
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def find_files(directory: Union[str, Path], pattern: str = "*", recursive: bool = True) -> List[Path]:
    """
    Find files matching pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (glob style)
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))

def get_file_size(filepath: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: File path
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(filepath)

def cleanup_directory(directory: Union[str, Path], keep_latest: int = 5, pattern: str = "*.pth") -> None:
    """
    Clean up directory by keeping only the latest N files matching pattern.
    
    Args:
        directory: Directory to clean
        keep_latest: Number of latest files to keep
        pattern: File pattern to match
    """
    files = find_files(directory, pattern, recursive=False)
    
    if len(files) <= keep_latest:
        return
    
    # Sort by modification time, newest first
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Remove old files
    for file_path in files[keep_latest:]:
        try:
            file_path.unlink()
            logger.info(f"Removed old file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not remove file {file_path}: {e}")

def archive_directory(source_dir: Union[str, Path], archive_path: Union[str, Path]) -> None:
    """
    Create archive of directory.
    
    Args:
        source_dir: Directory to archive
        archive_path: Path for the archive file (without extension)
    """
    shutil.make_archive(str(archive_path), 'zip', str(source_dir))
    logger.info(f"Created archive: {archive_path}.zip")

class CSVLogger:
    """Simple CSV logger for tracking metrics."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.columns = None
        ensure_dir(self.filepath.parent)
    
    def log(self, data: Dict[str, Any]) -> None:
        """Log data to CSV file."""
        df = pd.DataFrame([data])
        
        if not self.filepath.exists():
            # First write - create file with headers
            df.to_csv(self.filepath, index=False)
            self.columns = list(data.keys())
        else:
            # Append to existing file
            df.to_csv(self.filepath, mode='a', header=False, index=False)
    
    def read(self) -> pd.DataFrame:
        """Read the logged data."""
        if self.filepath.exists():
            return pd.read_csv(self.filepath)
        else:
            return pd.DataFrame()

def create_experiment_directory(base_dir: Union[str, Path], experiment_name: str) -> Path:
    """
    Create directory for experiment with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to created experiment directory
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    ensure_dir(exp_dir)
    
    return exp_dir