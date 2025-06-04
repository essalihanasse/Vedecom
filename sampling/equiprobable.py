"""
Equiprobable grid sampling implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult
import os
logger = logging.getLogger(__name__)

class EquiprobableSampler(BaseSampler):
    """
    Equiprobable grid sampling method.
    
    Divides the latent space into equiprobable regions and selects
    one representative from each region.
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
        Sample points using equiprobable grid approach.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting equiprobable grid sampling for {sample_size} samples")
        
        # Calculate grid dimensions
        n_rows, n_cols = self._calculate_grid_dimensions(sample_size)
        total_cells = n_rows * n_cols
        
        logger.info(f"Using {n_rows}x{n_cols} grid ({total_cells} cells)")
        
        # Create equiprobable grid
        grid_assignments, grid_info = self._create_equiprobable_grid(
            z_latent, n_rows, n_cols
        )
        
        # Select representatives from each cell
        selected_indices = self._select_from_grid(
            z_latent, grid_assignments, total_cells
        )
        
        # Ensure we don't exceed requested size
        selected_indices = selected_indices[:sample_size]
        
        # Additional method info
        additional_info = {
            'grid_dimensions': (n_rows, n_cols),
            'total_cells': total_cells,
            'empty_cells': grid_info['empty_cells'],
            'grid_assignments': grid_assignments.tolist(),
            'samples_per_cell': grid_info['samples_per_cell']
        }
        
        logger.info(f"Equiprobable grid sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _calculate_grid_dimensions(self, sample_size: int) -> Tuple[int, int]:
        """
        Calculate optimal grid dimensions for a given sample size.
        
        Args:
            sample_size: Number of samples needed
            
        Returns:
            Tuple of (n_rows, n_cols)
        """
        # For perfect squares, use square grid
        sqrt_size = int(math.sqrt(sample_size))
        if sqrt_size * sqrt_size == sample_size:
            return sqrt_size, sqrt_size
        
        # For non-perfect squares, find closest factorization
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
    
    def _create_equiprobable_grid(
        self, 
        z_latent: np.ndarray, 
        n_rows: int, 
        n_cols: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create equiprobable grid assignments.
        
        Args:
            z_latent: Latent coordinates
            n_rows: Number of grid rows
            n_cols: Number of grid columns
            
        Returns:
            Tuple of (grid_assignments, grid_info)
        """
        # Compute quantiles for each dimension
        x_quantiles = np.quantile(z_latent[:, 0], np.linspace(0, 1, n_cols + 1))
        y_quantiles = np.quantile(z_latent[:, 1], np.linspace(0, 1, n_rows + 1))
        
        # Assign each point to a grid cell
        grid_assignments = np.zeros(len(z_latent), dtype=int)
        for i in range(len(z_latent)):
            x, y = z_latent[i]
            
            # Find which bin each coordinate falls into
            x_bin = np.digitize(x, x_quantiles) - 1
            y_bin = np.digitize(y, y_quantiles) - 1
            
            # Ensure bounds are correct
            x_bin = min(max(x_bin, 0), n_cols - 1)
            y_bin = min(max(y_bin, 0), n_rows - 1)
            
            # Compute single index for the grid
            grid_idx = y_bin * n_cols + x_bin
            grid_assignments[i] = grid_idx
        
        # Calculate grid statistics
        total_cells = n_rows * n_cols
        samples_per_cell = {}
        empty_cells = []
        
        for grid_idx in range(total_cells):
            count = np.sum(grid_assignments == grid_idx)
            if count > 0:
                samples_per_cell[grid_idx] = count
            else:
                empty_cells.append(grid_idx)
        
        grid_info = {
            'x_quantiles': x_quantiles,
            'y_quantiles': y_quantiles,
            'samples_per_cell': samples_per_cell,
            'empty_cells': empty_cells,
            'n_rows': n_rows,
            'n_cols': n_cols
        }
        
        if empty_cells:
            logger.warning(f"Found {len(empty_cells)} empty grid cells")
        
        return grid_assignments, grid_info
    
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
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """Create equiprobable-specific visualizations."""
        # Call parent visualization first
        super().create_visualization(result, output_dir, title_suffix, **plot_kwargs)
        
        # Create equiprobable-specific plots
        self._create_grid_visualization(result, output_dir)
        self._create_cell_count_heatmap(result, output_dir)
    
    def _create_grid_visualization(self, result: SamplingResult, output_dir: str) -> None:
        """Create grid visualization with quantile lines."""
        try:
            grid_info = result.method_info
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            
            plt.figure(figsize=(12, 10))
            
            # Plot all points
            plt.scatter(
                z_latent[:, 0], z_latent[:, 1], 
                alpha=0.1, s=10, color='blue', label='All points'
            )
            
            # Plot selected points
            plt.scatter(
                z_latent[selected_indices, 0], z_latent[selected_indices, 1], 
                alpha=1.0, s=50, color='red', label='Selected samples'
            )
            
            # Draw grid lines if quantiles available
            if 'x_quantiles' in grid_info and 'y_quantiles' in grid_info:
                x_quantiles = grid_info['x_quantiles']
                y_quantiles = grid_info['y_quantiles']
                
                for q in x_quantiles:
                    plt.axvline(q, color='gray', linestyle='--', alpha=0.5)
                for q in y_quantiles:
                    plt.axhline(q, color='gray', linestyle='--', alpha=0.5)
            
            n_rows, n_cols = grid_info['grid_dimensions']
            plt.title(f'Equiprobable Grid Sampling\n'
                     f'Grid: {n_rows}x{n_cols}, Samples: {len(selected_indices)}')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'equiprobable_grid.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create grid visualization: {e}")
    
    def _create_cell_count_heatmap(self, result: SamplingResult, output_dir: str) -> None:
        """Create heatmap showing distribution of points in grid cells."""
        try:
            grid_info = result.method_info
            grid_assignments = np.array(grid_info['grid_assignments'])
            n_rows, n_cols = grid_info['grid_dimensions']
            
            # Create heatmap data
            total_cells = n_rows * n_cols
            cell_counts = np.bincount(grid_assignments, minlength=total_cells)
            heatmap_data = cell_counts.reshape(n_rows, n_cols)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(heatmap_data, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Number of points')
            
            # Add cell count text for smaller grids
            if n_rows * n_cols <= 400:
                for i in range(n_rows):
                    for j in range(n_cols):
                        count = int(heatmap_data[i, j])
                        color = 'white' if count > np.max(heatmap_data)/2 else 'black'
                        plt.text(
                            j, i, str(count), ha='center', va='center', 
                            color=color, fontsize=max(6, 12 - n_rows // 3)
                        )
            
            plt.title(f'Distribution of Points in Grid Cells\n'
                     f'Grid: {n_rows}x{n_cols}')
            plt.xlabel(f'Latent Dimension 1 (bins, {n_cols} total)')
            plt.ylabel(f'Latent Dimension 2 (bins, {n_rows} total)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'grid_cell_counts.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create cell count heatmap: {e}")