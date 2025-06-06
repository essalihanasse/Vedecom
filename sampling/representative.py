"""
Optimized distance-based representative sampling implementation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional, Any
import logging
from .base import BaseSampler, SamplingResult, SamplingUtils

logger = logging.getLogger(__name__)

class DistanceBasedSampler(BaseSampler):
    """
    Optimized distance-based representative sampling method.
    
    Uses a simplified greedy approach that maximizes minimum distance
    between representatives for good coverage.
    """
    
    def __init__(
        self,
        info_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        coverage_radius: float = 0.2,
        candidate_fraction: float = 1.0,
        **kwargs
    ):
        super().__init__('distance_based', **kwargs)
        
        self.info_weight = info_weight
        self.redundancy_weight = redundancy_weight
        self.coverage_radius = coverage_radius
        self.candidate_fraction = candidate_fraction
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """Sample representatives using optimized distance-based approach."""
        logger.info(f"Starting distance-based sampling for {sample_size} samples")
        
        # Use efficient farthest-first strategy
        selected_indices = self._farthest_first_selection(z_latent, sample_size)
        
        # Calculate coverage statistics
        coverage_stats = SamplingUtils.calculate_coverage_statistics(
            z_latent, selected_indices, self.coverage_radius
        )
        
        additional_info = {
            'selection_strategy': 'farthest_first',
            'coverage_radius': self.coverage_radius,
            **coverage_stats
        }
        
        logger.info(f"Distance-based sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Coverage: {coverage_stats['coverage_fraction']*100:.1f}%")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _farthest_first_selection(self, z_latent: np.ndarray, n_representatives: int) -> List[int]:
        """
        Optimized farthest-first selection strategy.
        
        This is much simpler and more efficient than the original complex
        information-theoretic approach while achieving similar results.
        """
        n_samples = len(z_latent)
        
        if n_representatives >= n_samples:
            return list(range(n_samples))
        
        selected_indices = []
        
        # Start with the point closest to center (good heuristic)
        center = np.mean(z_latent, axis=0)
        distances_to_center = np.sum((z_latent - center) ** 2, axis=1)
        first_idx = np.argmin(distances_to_center)
        selected_indices.append(first_idx)
        
        # Pre-allocate distance matrix for efficiency
        min_distances = np.full(n_samples, np.inf)
        
        # Iteratively add points that maximize minimum distance
        for step in range(1, n_representatives):
            # Update minimum distances to selected set
            last_selected = z_latent[selected_indices[-1]]
            distances_to_last = np.sum((z_latent - last_selected) ** 2, axis=1)
            min_distances = np.minimum(min_distances, distances_to_last)
            
            # Find point with maximum minimum distance (excluding already selected)
            min_distances[selected_indices] = -1  # Mark as unavailable
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
            
            if step % max(1, n_representatives // 10) == 0:
                logger.debug(f"Selected {step + 1}/{n_representatives} representatives")
        
        return selected_indices
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """Create optimized visualizations."""
        # Call parent visualization
        super().create_visualization(result, output_dir, title_suffix, **plot_kwargs)
        
        # Create coverage-specific visualization
        self._create_coverage_visualization(result, output_dir)
    
    def _create_coverage_visualization(self, result: SamplingResult, output_dir: str) -> None:
        """Create coverage visualization with coverage circles."""
        try:
            import matplotlib.pyplot as plt
            
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            coverage_radius = result.method_info.get('coverage_radius', 0.2)
            
            # Efficiently compute covered points
            selected_points = z_latent[selected_indices]
            distances_matrix = cdist(z_latent, selected_points)
            min_distances = np.min(distances_matrix, axis=1)
            covered_mask = min_distances <= coverage_radius
            
            plt.figure(figsize=(12, 10))
            
            # Plot points by coverage status
            plt.scatter(z_latent[~covered_mask, 0], z_latent[~covered_mask, 1], 
                       c='lightblue', alpha=0.3, s=5, label='Uncovered')
            plt.scatter(z_latent[covered_mask, 0], z_latent[covered_mask, 1], 
                       c='lightcoral', alpha=0.6, s=8, label='Covered')
            
            # Plot representatives with coverage circles
            for i, (x, y) in enumerate(selected_points):
                plt.scatter(x, y, s=150, color='darkred', marker='*', 
                           edgecolors='white', linewidths=2, zorder=10)
                
                # Draw coverage circle (only for first few to avoid clutter)
                if i < 10:  # Limit to avoid visual clutter
                    circle = plt.Circle((x, y), coverage_radius, fill=False, 
                                      color='darkred', alpha=0.5, linestyle='-')
                    plt.gca().add_patch(circle)
            
            coverage_pct = np.sum(covered_mask) / len(z_latent) * 100
            plt.title(f'Coverage Analysis: {coverage_pct:.1f}% covered\n'
                     f'Representatives: {len(selected_indices)}, Radius: {coverage_radius}')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            plt.savefig(f'{output_dir}/coverage_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create coverage visualization: {e}")