"""
Extended Equiprobable grid sampling for 3D and higher dimensions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult
import os

logger = logging.getLogger(__name__)

class EquiprobableSampler(BaseSampler):
    """
    Equiprobable grid sampling method using Gaussian quantiles.
    
    Supports 2D, 3D, and higher dimensional sampling.
    """
    
    def __init__(self, **kwargs):
        """Initialize equiprobable sampler."""
        super().__init__('equiprobable', **kwargs)
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample points using equiprobable grid approach with Gaussian quantiles.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        n_dims = z_latent.shape[1]
        logger.info(f"Starting equiprobable grid sampling (Gaussian quantiles) for {sample_size} samples in {n_dims}D")
        
        # Calculate grid dimensions for any number of dimensions
        grid_dims = self._calculate_grid_dimensions_nd(sample_size, n_dims)
        total_cells = np.prod(grid_dims)
        
        logger.info(f"Using {' × '.join(map(str, grid_dims))} grid ({total_cells} cells)")
        
        # Create equiprobable grid using Gaussian quantiles
        grid_assignments, grid_info = self._create_gaussian_equiprobable_grid_nd(
            z_latent, grid_dims
        )
        
        # Select representatives from each cell
        selected_indices = self._select_from_grid(
            z_latent, grid_assignments, total_cells
        )
        
        # Ensure we don't exceed requested size
        selected_indices = selected_indices[:sample_size]
        
        # Additional method info
        additional_info = {
            'grid_dimensions': grid_dims,
            'total_cells': total_cells,
            'empty_cells': grid_info['empty_cells'],
            'grid_assignments': grid_assignments.tolist(),
            'samples_per_cell': grid_info['samples_per_cell'],
            'quantile_method': 'gaussian',
            'gaussian_parameters': grid_info['gaussian_parameters'],
            'n_dimensions': n_dims
        }
        
        logger.info(f"Equiprobable grid sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _calculate_grid_dimensions_nd(self, sample_size: int, n_dims: int) -> List[int]:
        """
        Calculate optimal grid dimensions for n-dimensional space.
        
        Args:
            sample_size: Number of samples needed
            n_dims: Number of dimensions
            
        Returns:
            List of grid dimensions for each axis
        """
        if n_dims == 1:
            return [sample_size]
        
        if n_dims == 2:
            # Use existing 2D logic
            sqrt_size = int(math.sqrt(sample_size))
            if sqrt_size * sqrt_size == sample_size:
                return [sqrt_size, sqrt_size]
            
            # Find closest factorization
            best_diff = float('inf')
            best_dims = [1, sample_size]
            
            for i in range(1, int(math.sqrt(sample_size)) + 1):
                if sample_size % i == 0:
                    dims = [i, sample_size // i]
                    diff = abs(dims[0] - dims[1])
                    if diff < best_diff:
                        best_diff = diff
                        best_dims = dims
            
            return best_dims
        
        # For 3D and higher dimensions
        if n_dims == 3:
            return self._factorize_for_3d(sample_size)
        else:
            return self._factorize_for_nd(sample_size, n_dims)
    
    def _factorize_for_3d(self, sample_size: int) -> List[int]:
        """
        Find good 3D factorization of sample_size.
        
        For 100 samples, prioritizes balanced divisions like [5, 5, 4] or [4, 5, 5].
        """
        # Try to find factors close to cube root
        cube_root = int(round(sample_size ** (1/3)))
        
        best_dims = None
        best_score = float('inf')
        
        # Search around the cube root
        for a in range(max(1, cube_root - 2), cube_root + 3):
            if sample_size % a == 0:
                remaining = sample_size // a
                
                # Factor the remaining into 2 dimensions
                sqrt_remaining = int(math.sqrt(remaining))
                
                for b in range(max(1, sqrt_remaining - 2), sqrt_remaining + 3):
                    if remaining % b == 0:
                        c = remaining // b
                        
                        # Calculate balance score (prefer more balanced dimensions)
                        dims = sorted([a, b, c])
                        score = dims[2] - dims[0]  # Range between max and min
                        
                        if score < best_score:
                            best_score = score
                            best_dims = [a, b, c]
        
        if best_dims is None:
            # Fallback: use simple factorization
            factors = self._get_factors(sample_size)
            if len(factors) >= 3:
                return factors[:3]
            elif len(factors) == 2:
                return factors + [1]
            else:
                return [sample_size, 1, 1]
        
        return best_dims
    
    def _factorize_for_nd(self, sample_size: int, n_dims: int) -> List[int]:
        """
        Factorize sample_size into n_dims dimensions.
        """
        # Start with the n-th root
        nth_root = int(round(sample_size ** (1/n_dims)))
        
        # Get all factors
        factors = self._get_factors(sample_size)
        
        if len(factors) >= n_dims:
            # Use first n_dims factors
            dims = factors[:n_dims]
        else:
            # Pad with 1s and adjust
            dims = factors + [1] * (n_dims - len(factors))
            
            # Redistribute the product to balance dimensions
            remaining_product = sample_size // np.prod(dims)
            dims[0] *= remaining_product
        
        return dims
    
    def _get_factors(self, n: int) -> List[int]:
        """Get factors of n, trying to keep them balanced."""
        factors = []
        temp = n
        divisor = 2
        
        while divisor * divisor <= temp:
            while temp % divisor == 0:
                factors.append(divisor)
                temp //= divisor
            divisor += 1
        
        if temp > 1:
            factors.append(temp)
        
        return factors
    
    def _create_gaussian_equiprobable_grid_nd(
        self, 
        z_latent: np.ndarray, 
        grid_dims: List[int]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create equiprobable grid assignments using Gaussian quantiles for n-D.
        
        Args:
            z_latent: Latent coordinates
            grid_dims: Grid dimensions for each axis
            
        Returns:
            Tuple of (grid_assignments, grid_info)
        """
        n_points, n_dims = z_latent.shape
        
        # Fit Gaussian parameters to each dimension
        gaussian_params = {}
        all_quantiles = {}
        
        for dim in range(n_dims):
            dim_data = z_latent[:, dim]
            mean = np.mean(dim_data)
            std = np.std(dim_data)
            
            gaussian_params[f'dim_{dim}'] = {'mean': mean, 'std': std}
            
            # Compute Gaussian quantiles for this dimension
            n_bins = grid_dims[dim]
            quantiles = norm.ppf(
                np.linspace(0, 1, n_bins + 1)[1:-1], 
                loc=mean, 
                scale=std
            )
            
            # Add extreme values to ensure all points are covered
            quantiles = np.concatenate([
                [dim_data.min() - 1], 
                quantiles, 
                [dim_data.max() + 1]
            ])
            
            all_quantiles[f'dim_{dim}'] = quantiles
            
            logger.info(f"Dim {dim}: μ={mean:.3f}, σ={std:.3f}, {n_bins} bins")
        
        # Assign each point to a grid cell
        grid_assignments = np.zeros(n_points, dtype=int)
        
        for i in range(n_points):
            cell_coords = []
            
            for dim in range(n_dims):
                value = z_latent[i, dim]
                quantiles = all_quantiles[f'dim_{dim}']
                
                # Find which bin this value falls into
                bin_idx = np.digitize(value, quantiles) - 1
                bin_idx = min(max(bin_idx, 0), grid_dims[dim] - 1)
                
                cell_coords.append(bin_idx)
            
            # Convert n-D coordinates to single index
            grid_idx = self._coords_to_index(cell_coords, grid_dims)
            grid_assignments[i] = grid_idx
        
        # Calculate grid statistics
        total_cells = np.prod(grid_dims)
        samples_per_cell = {}
        empty_cells = []
        
        for grid_idx in range(total_cells):
            count = np.sum(grid_assignments == grid_idx)
            if count > 0:
                samples_per_cell[grid_idx] = count
            else:
                empty_cells.append(grid_idx)
        
        grid_info = {
            'quantiles': all_quantiles,
            'samples_per_cell': samples_per_cell,
            'empty_cells': empty_cells,
            'grid_dims': grid_dims,
            'gaussian_parameters': gaussian_params,
            'n_dimensions': n_dims
        }
        
        if empty_cells:
            logger.warning(f"Found {len(empty_cells)} empty grid cells with Gaussian quantiles")
        
        return grid_assignments, grid_info
    
    def _coords_to_index(self, coords: List[int], grid_dims: List[int]) -> int:
        """
        Convert n-dimensional coordinates to single index.
        
        Uses row-major order (C-style indexing).
        """
        index = 0
        multiplier = 1
        
        for i in reversed(range(len(coords))):
            index += coords[i] * multiplier
            multiplier *= grid_dims[i]
        
        return index
    
    def _index_to_coords(self, index: int, grid_dims: List[int]) -> List[int]:
        """
        Convert single index back to n-dimensional coordinates.
        """
        coords = []
        temp_index = index
        
        for i in reversed(range(len(grid_dims))):
            coords.append(temp_index % grid_dims[i])
            temp_index //= grid_dims[i]
        
        return list(reversed(coords))
    
    def _select_from_grid(
        self,
        z_latent: np.ndarray,
        grid_assignments: np.ndarray,
        total_cells: int
    ) -> List[int]:
        """
        Select one representative from each grid cell.
        
        Args:
            z_latent: Latent coordinates
            grid_assignments: Grid cell assignments for each point
            total_cells: Total number of grid cells
            
        Returns:
            List of selected indices
        """
        selected_indices = []
        
        for grid_idx in range(total_cells):
            points_in_cell = np.where(grid_assignments == grid_idx)[0]
            
            if len(points_in_cell) > 0:
                # Select point closest to cell median
                cell_points = z_latent[points_in_cell]
                cell_median = np.median(cell_points, axis=0)
                
                # Calculate distances to median
                distances = np.sqrt(np.sum((cell_points - cell_median) ** 2, axis=1))
                
                # Find closest point
                closest_idx = points_in_cell[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return selected_indices
