"""
Latin Hypercube Sampling implementation for latent space sampling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult, SamplingUtils
import os

logger = logging.getLogger(__name__)

class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampling (LHS) for representative sampling in latent space.
    
    LHS ensures that each dimension is sampled uniformly by dividing each dimension
    into equal-probability intervals and sampling exactly once from each interval.
    This provides excellent space-filling properties and uniform coverage.
    """
    
    def __init__(
        self,
        criterion: str = 'maximin',
        iterations: int = 10,
        random_state: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Latin Hypercube sampler.
        
        Args:
            criterion: Optimization criterion ('maximin', 'correlation', 'centermaximin')
            iterations: Number of optimization iterations
            random_state: Random seed for reproducibility
        """
        super().__init__('latin_hypercube', **kwargs)
        
        self.criterion = criterion
        self.iterations = iterations
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample representatives using Latin Hypercube Sampling.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting Latin Hypercube sampling for {sample_size} samples")
        logger.info(f"Criterion: {self.criterion}, Iterations: {self.iterations}")
        
        n_samples, n_dims = z_latent.shape
        
        if sample_size >= n_samples:
            logger.warning(f"Sample size ({sample_size}) >= dataset size ({n_samples}), returning all points")
            selected_indices = list(range(n_samples))
        else:
            # Generate Latin Hypercube design
            lhs_design = self._generate_lhs_design(sample_size, n_dims)
            
            # Map LHS design to actual data points
            selected_indices = self._map_lhs_to_data(lhs_design, z_latent)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_lhs_quality(z_latent, selected_indices)
        
        # Additional method info
        additional_info = {
            'criterion': self.criterion,
            'iterations': self.iterations,
            'n_dimensions': n_dims,
            'design_quality': quality_metrics,
            'space_filling_efficiency': len(selected_indices) / n_samples
        }
        
        logger.info(f"Latin Hypercube sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Space-filling efficiency: {quality_metrics.get('space_filling_score', 0):.3f}")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _generate_lhs_design(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        Generate Latin Hypercube design.
        
        Args:
            n_samples: Number of samples
            n_dims: Number of dimensions
            
        Returns:
            LHS design matrix with values in [0, 1]
        """
        # Generate initial random LHS design
        lhs = np.zeros((n_samples, n_dims))
        
        for j in range(n_dims):
            # Create permutation of integers 0 to n_samples-1
            perm = np.random.permutation(n_samples)
            # Add random uniform within each interval
            lhs[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
        
        # Optimize design based on criterion
        if self.iterations > 0:
            lhs = self._optimize_lhs_design(lhs)
        
        return lhs
    
    def _optimize_lhs_design(self, initial_design: np.ndarray) -> np.ndarray:
        """
        Optimize LHS design using specified criterion.
        
        Args:
            initial_design: Initial LHS design
            
        Returns:
            Optimized LHS design
        """
        best_design = initial_design.copy()
        best_score = self._evaluate_design(best_design)
        
        n_samples, n_dims = initial_design.shape
        
        for iteration in range(self.iterations):
            # Create candidate design by swapping elements
            candidate_design = best_design.copy()
            
            # Randomly select dimension and two points to swap
            dim = np.random.randint(n_dims)
            i, j = np.random.choice(n_samples, 2, replace=False)
            
            # Swap the values
            candidate_design[i, dim], candidate_design[j, dim] = \
                candidate_design[j, dim], candidate_design[i, dim]
            
            # Evaluate candidate
            candidate_score = self._evaluate_design(candidate_design)
            
            # Accept if better (for maximin, higher is better)
            if self.criterion == 'maximin' and candidate_score > best_score:
                best_design = candidate_design
                best_score = candidate_score
            elif self.criterion == 'correlation' and candidate_score < best_score:
                best_design = candidate_design
                best_score = candidate_score
        
        logger.debug(f"LHS optimization: {self.criterion} score improved from "
                    f"{self._evaluate_design(initial_design):.4f} to {best_score:.4f}")
        
        return best_design
    
    def _evaluate_design(self, design: np.ndarray) -> float:
        """
        Evaluate LHS design quality based on criterion.
        
        Args:
            design: LHS design matrix
            
        Returns:
            Design quality score
        """
        if self.criterion == 'maximin':
            # Maximize minimum distance between points
            distances = cdist(design, design)
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            return np.min(distances)
        
        elif self.criterion == 'correlation':
            # Minimize maximum absolute correlation between dimensions
            if design.shape[1] < 2:
                return 0.0
            corr_matrix = np.corrcoef(design.T)
            # Get upper triangle excluding diagonal
            upper_tri = np.triu(corr_matrix, k=1)
            return np.max(np.abs(upper_tri))
        
        elif self.criterion == 'centermaximin':
            # Maximize minimum distance to center
            center = np.full(design.shape[1], 0.5)  # Center of unit hypercube
            distances_to_center = cdist([center], design)[0]
            return np.min(distances_to_center)
        
        else:
            return 0.0
    
    def _map_lhs_to_data(self, lhs_design: np.ndarray, z_latent: np.ndarray) -> List[int]:
        """
        Map LHS design points to actual data points.
        
        Args:
            lhs_design: LHS design in [0, 1] space
            z_latent: Actual latent coordinates
            
        Returns:
            List of selected data point indices
        """
        n_samples, n_dims = z_latent.shape
        
        # Transform LHS design to latent space bounds
        min_vals = np.min(z_latent, axis=0)
        max_vals = np.max(z_latent, axis=0)
        
        # Scale LHS design to latent space
        lhs_scaled = lhs_design * (max_vals - min_vals) + min_vals
        
        # Find nearest data points to LHS points
        selected_indices = []
        used_indices = set()
        
        # Use KDTree for efficient nearest neighbor search
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(10, n_samples), algorithm='auto')
        nbrs.fit(z_latent)
        
        for lhs_point in lhs_scaled:
            # Find nearest neighbors
            distances, neighbor_indices = nbrs.kneighbors([lhs_point])
            
            # Select first unused neighbor
            for idx in neighbor_indices[0]:
                if idx not in used_indices:
                    selected_indices.append(idx)
                    used_indices.add(idx)
                    break
            else:
                # If all neighbors are used, find globally nearest unused point
                remaining_indices = [i for i in range(n_samples) if i not in used_indices]
                if remaining_indices:
                    distances_to_remaining = cdist([lhs_point], z_latent[remaining_indices])[0]
                    best_remaining_idx = remaining_indices[np.argmin(distances_to_remaining)]
                    selected_indices.append(best_remaining_idx)
                    used_indices.add(best_remaining_idx)
        
        return selected_indices
    
    def _calculate_lhs_quality(self, z_latent: np.ndarray, selected_indices: List[int]) -> Dict[str, float]:
        """
        Calculate quality metrics for the LHS sampling.
        
        Args:
            z_latent: Latent coordinates
            selected_indices: Selected point indices
            
        Returns:
            Dictionary with quality metrics
        """
        if not selected_indices:
            return {'space_filling_score': 0.0}
        
        selected_points = z_latent[selected_indices]
        n_dims = z_latent.shape[1]
        
        metrics = {}
        
        # Space-filling score (based on minimum distance)
        if len(selected_indices) > 1:
            distances = cdist(selected_points, selected_points)
            np.fill_diagonal(distances, np.inf)
            min_distance = np.min(distances)
            
            # Normalize by expected distance for random sampling
            space_volume = np.prod(np.max(z_latent, axis=0) - np.min(z_latent, axis=0))
            expected_distance = (space_volume / len(selected_indices)) ** (1/n_dims)
            metrics['space_filling_score'] = min_distance / expected_distance if expected_distance > 0 else 0
            metrics['min_pairwise_distance'] = min_distance
            metrics['avg_pairwise_distance'] = np.mean(distances[distances != np.inf])
        else:
            metrics['space_filling_score'] = 1.0
            metrics['min_pairwise_distance'] = 0.0
            metrics['avg_pairwise_distance'] = 0.0
        
        # Coverage uniformity (check if each dimension is uniformly covered)
        uniformity_scores = []
        for dim in range(n_dims):
            dim_values = selected_points[:, dim]
            dim_range = np.max(z_latent[:, dim]) - np.min(z_latent[:, dim])
            
            if dim_range > 0:
                # Calculate uniformity using Kolmogorov-Smirnov test against uniform distribution
                normalized_values = (dim_values - np.min(z_latent[:, dim])) / dim_range
                sorted_values = np.sort(normalized_values)
                expected_values = np.linspace(0, 1, len(sorted_values))
                ks_statistic = np.max(np.abs(sorted_values - expected_values))
                uniformity_scores.append(1 - ks_statistic)  # Higher is better
            else:
                uniformity_scores.append(1.0)
        
        metrics['dimension_uniformity'] = np.mean(uniformity_scores)
        
        # Correlation between dimensions (should be low for good LHS)
        if n_dims > 1 and len(selected_indices) > 1:
            corr_matrix = np.corrcoef(selected_points.T)
            upper_tri = np.triu(corr_matrix, k=1)
            max_correlation = np.max(np.abs(upper_tri)) if upper_tri.size > 0 else 0
            metrics['max_correlation'] = max_correlation
            metrics['correlation_score'] = 1 - max_correlation  # Higher is better
        else:
            metrics['max_correlation'] = 0.0
            metrics['correlation_score'] = 1.0
        
        return metrics
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """Create LHS-specific visualizations."""
        # Call parent visualization first
        super().create_visualization(result, output_dir, title_suffix, **plot_kwargs)
        
        # Create LHS-specific plots
        self._create_lhs_quality_plot(result, output_dir)
        self._create_dimension_coverage_plot(result, output_dir)
    
    def _create_lhs_quality_plot(self, result: SamplingResult, output_dir: str) -> None:
        """Create visualization of LHS quality metrics."""
        try:
            quality_metrics = result.method_info.get('design_quality', {})
            
            if not quality_metrics:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Selected points with Voronoi-like regions
            ax = axes[0, 0]
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            
            ax.scatter(z_latent[:, 0], z_latent[:, 1], alpha=0.2, s=5, color='lightblue', label='All points')
            ax.scatter(z_latent[selected_indices, 0], z_latent[selected_indices, 1], 
                      alpha=0.8, s=50, color='red', edgecolors='darkred', 
                      linewidths=1, label=f'LHS samples ({len(selected_indices)})')
            
            ax.set_title('Latin Hypercube Sampling')
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Distance distribution
            ax = axes[0, 1]
            if len(selected_indices) > 1:
                selected_points = z_latent[selected_indices]
                distances = cdist(selected_points, selected_points)
                upper_tri_distances = distances[np.triu_indices_from(distances, k=1)]
                
                ax.hist(upper_tri_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(quality_metrics.get('min_pairwise_distance', 0), 
                          color='red', linestyle='--', linewidth=2,
                          label=f"Min distance: {quality_metrics.get('min_pairwise_distance', 0):.3f}")
                ax.set_title('Pairwise Distance Distribution')
                ax.set_xlabel('Distance')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Need ≥2 points\nfor distance analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Pairwise Distance Distribution')
            
            # Plot 3: Quality metrics bar chart
            ax = axes[1, 0]
            metric_names = []
            metric_values = []
            
            for key, value in quality_metrics.items():
                if key in ['space_filling_score', 'dimension_uniformity', 'correlation_score']:
                    metric_names.append(key.replace('_', ' ').title())
                    metric_values.append(value)
            
            if metric_names:
                bars = ax.bar(metric_names, metric_values, alpha=0.7, color=['skyblue', 'lightgreen', 'orange'])
                ax.set_title('LHS Quality Metrics\n(Higher is Better)')
                ax.set_ylabel('Score')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3, axis='y')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Plot 4: Summary statistics
            ax = axes[1, 1]
            ax.axis('off')
            
            # Create summary text
            summary_text = f"""
LHS Sampling Summary:

Criterion: {result.method_info.get('criterion', 'N/A')}
Iterations: {result.method_info.get('iterations', 'N/A')}
Dimensions: {result.method_info.get('n_dimensions', 'N/A')}

Quality Metrics:
Space Filling: {quality_metrics.get('space_filling_score', 0):.3f}
Uniformity: {quality_metrics.get('dimension_uniformity', 0):.3f}
Correlation: {quality_metrics.get('correlation_score', 0):.3f}

Distance Stats:
Min Distance: {quality_metrics.get('min_pairwise_distance', 0):.4f}
Avg Distance: {quality_metrics.get('avg_pairwise_distance', 0):.4f}
Max Correlation: {quality_metrics.get('max_correlation', 0):.4f}

Efficiency: {result.method_info.get('space_filling_efficiency', 0)*100:.1f}%
"""
            
            ax.text(0.05, 0.95, summary_text.strip(), transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Latin Hypercube Sampling Quality Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'lhs_quality_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create LHS quality plot: {e}")
    
    def _create_dimension_coverage_plot(self, result: SamplingResult, output_dir: str) -> None:
        """Create visualization of dimension coverage uniformity."""
        try:
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            
            if len(selected_indices) < 2:
                return
            
            selected_points = z_latent[selected_indices]
            n_dims = min(z_latent.shape[1], 4)  # Limit to first 4 dimensions for visualization
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for dim in range(n_dims):
                ax = axes[dim]
                
                # Get values for this dimension
                all_values = z_latent[:, dim]
                selected_values = selected_points[:, dim]
                
                # Create histogram comparison
                ax.hist(all_values, bins=30, alpha=0.3, color='blue', 
                       density=True, label='All points')
                ax.hist(selected_values, bins=15, alpha=0.7, color='red', 
                       density=True, label='LHS samples')
                
                # Add uniform distribution for comparison
                x_range = np.linspace(np.min(all_values), np.max(all_values), 100)
                uniform_density = np.ones_like(x_range) / (np.max(all_values) - np.min(all_values))
                ax.plot(x_range, uniform_density, 'g--', linewidth=2, 
                       label='Ideal uniform')
                
                ax.set_title(f'Dimension {dim + 1} Coverage')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Calculate and display uniformity score
                dim_range = np.max(all_values) - np.min(all_values)
                if dim_range > 0:
                    normalized_values = (selected_values - np.min(all_values)) / dim_range
                    sorted_values = np.sort(normalized_values)
                    expected_values = np.linspace(0, 1, len(sorted_values))
                    ks_statistic = np.max(np.abs(sorted_values - expected_values))
                    uniformity_score = 1 - ks_statistic
                    
                    ax.text(0.02, 0.98, f'Uniformity: {uniformity_score:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for dim in range(n_dims, len(axes)):
                axes[dim].set_visible(False)
            
            plt.suptitle('Latin Hypercube Dimension Coverage Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'lhs_dimension_coverage.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create dimension coverage plot: {e}")


class AdaptiveLatinHypercubeSampler(LatinHypercubeSampler):
    """
    Adaptive Latin Hypercube Sampling that considers data density.
    
    This variant modifies the standard LHS approach to account for
    the actual distribution of data in the latent space.
    """
    
    def __init__(
        self,
        density_weight: float = 0.3,
        adaptive_iterations: int = 20,
        **kwargs
    ):
        """
        Initialize adaptive LHS sampler.
        
        Args:
            density_weight: Weight for density-based adjustment (0-1)
            adaptive_iterations: Number of density-aware optimization iterations
        """
        super().__init__(**kwargs)
        self.method_name = 'adaptive_latin_hypercube'
        self.density_weight = density_weight
        self.adaptive_iterations = adaptive_iterations
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """Sample using adaptive Latin Hypercube approach."""
        logger.info(f"Starting Adaptive Latin Hypercube sampling for {sample_size} samples")
        logger.info(f"Density weight: {self.density_weight}")
        
        n_samples, n_dims = z_latent.shape
        
        if sample_size >= n_samples:
            selected_indices = list(range(n_samples))
        else:
            # Generate initial LHS design
            lhs_design = self._generate_lhs_design(sample_size, n_dims)
            
            # Apply density-aware adaptation
            adapted_design = self._adapt_design_to_density(lhs_design, z_latent)
            
            # Map to actual data points
            selected_indices = self._map_lhs_to_data(adapted_design, z_latent)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_lhs_quality(z_latent, selected_indices)
        
        # Additional method info
        additional_info = {
            'criterion': self.criterion,
            'iterations': self.iterations,
            'density_weight': self.density_weight,
            'adaptive_iterations': self.adaptive_iterations,
            'n_dimensions': n_dims,
            'design_quality': quality_metrics,
            'space_filling_efficiency': len(selected_indices) / n_samples
        }
        
        logger.info(f"Adaptive LHS sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _adapt_design_to_density(self, lhs_design: np.ndarray, z_latent: np.ndarray) -> np.ndarray:
        """
        Adapt LHS design to account for data density.
        
        Args:
            lhs_design: Initial LHS design
            z_latent: Actual data points
            
        Returns:
            Density-adapted LHS design
        """
        # Estimate data density using KDE
        from sklearn.neighbors import KernelDensity
        
        kde = KernelDensity(bandwidth='scott')
        kde.fit(z_latent)
        
        # Transform LHS design to data space
        min_vals = np.min(z_latent, axis=0)
        max_vals = np.max(z_latent, axis=0)
        lhs_scaled = lhs_design * (max_vals - min_vals) + min_vals
        
        # Evaluate density at LHS points
        log_densities = kde.score_samples(lhs_scaled)
        densities = np.exp(log_densities)
        
        # Normalize densities
        densities = densities / np.max(densities)
        
        # Adaptive optimization
        adapted_design = lhs_scaled.copy()
        
        for iteration in range(self.adaptive_iterations):
            # Select a point to adjust
            point_idx = np.random.randint(len(adapted_design))
            
            # Generate candidate position
            candidate = adapted_design.copy()
            
            # Small perturbation considering density
            density_factor = 1.0 - self.density_weight * densities[point_idx]
            perturbation_scale = 0.1 * density_factor
            
            candidate[point_idx] += np.random.normal(0, perturbation_scale, adapted_design.shape[1])
            
            # Ensure bounds
            candidate[point_idx] = np.clip(candidate[point_idx], min_vals, max_vals)
            
            # Evaluate improvement (distance-based with density weighting)
            current_score = self._evaluate_adaptive_design(adapted_design, densities)
            candidate_densities = densities.copy()
            candidate_densities[point_idx] = np.exp(kde.score_samples([candidate[point_idx]])[0])
            candidate_score = self._evaluate_adaptive_design(candidate, candidate_densities)
            
            # Accept if better
            if candidate_score > current_score:
                adapted_design = candidate
                densities = candidate_densities
        
        # Transform back to [0,1] space
        adapted_design_normalized = (adapted_design - min_vals) / (max_vals - min_vals)
        
        return adapted_design_normalized
    
    def _evaluate_adaptive_design(self, design: np.ndarray, densities: np.ndarray) -> float:
        """Evaluate design considering both space-filling and density."""
        # Standard space-filling score
        distances = cdist(design, design)
        np.fill_diagonal(distances, np.inf)
        min_distance = np.min(distances)
        
        # Density balance score (prefer more uniform density coverage)
        density_variance = np.var(densities)
        density_score = 1.0 / (1.0 + density_variance)
        
        # Combined score
        return min_distance * (1 - self.density_weight) + density_score * self.density_weight