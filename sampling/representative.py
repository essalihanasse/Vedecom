"""
Distance-based representative sampling implementation.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm

from .base import BaseSampler, SamplingResult, SamplingUtils
import os
logger = logging.getLogger(__name__)

class DistanceBasedSampler(BaseSampler):
    """
    Distance-based representative sampling method.
    
    Uses information-theoretic measures adapted for continuous latent space
    to select representatives that maximize information gain while minimizing
    redundancy.
    """
    
    def __init__(
        self,
        info_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        coverage_radius: float = 0.2,
        candidate_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initialize distance-based sampler.
        
        Args:
            info_weight: Weight for information gain term
            redundancy_weight: Weight for redundancy penalty term
            coverage_radius: Radius for coverage calculation
            candidate_fraction: Fraction of candidates to consider (for speed)
        """
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
        """
        Sample representatives using distance-based approach.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting distance-based sampling for {sample_size} samples")
        
        # Perform greedy selection
        selected_indices = self._greedy_representative_selection(
            z_latent, sample_size
        )
        
        # Calculate coverage statistics
        coverage_stats = SamplingUtils.calculate_coverage_statistics(
            z_latent, selected_indices, self.coverage_radius
        )
        
        # Additional method info
        additional_info = {
            'info_weight': self.info_weight,
            'redundancy_weight': self.redundancy_weight,
            'coverage_radius': self.coverage_radius,
            'candidate_fraction': self.candidate_fraction,
            'selection_method': 'greedy' if self.candidate_fraction >= 1.0 else 'heuristic',
            **coverage_stats
        }
        
        logger.info(f"Distance-based sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Coverage: {coverage_stats['coverage_fraction']*100:.1f}%")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _greedy_representative_selection(
        self, 
        z_latent: np.ndarray, 
        n_representatives: int
    ) -> List[int]:
        """
        Greedy algorithm for representative selection in latent space.
        
        Args:
            z_latent: Latent space coordinates
            n_representatives: Number of representatives to select
            
        Returns:
            List of selected representative indices
        """
        n_samples = len(z_latent)
        representatives = []
        remaining_candidates = list(range(n_samples))
        covered_points = set()
        
        logger.info(f"Selecting {n_representatives} representatives from {n_samples} latent points")
        
        for step in tqdm(range(n_representatives), desc="Selecting representatives"):
            best_score = -float('inf')
            best_candidate = None
            
            # Optionally limit candidates for speed
            if self.candidate_fraction < 1.0:
                n_candidates = max(1, int(len(remaining_candidates) * self.candidate_fraction))
                candidates = np.random.choice(remaining_candidates, n_candidates, replace=False)
            else:
                candidates = remaining_candidates
            
            # Evaluate each candidate
            for candidate_idx in candidates:
                score = self._compute_objective_score(
                    z_latent, representatives, candidate_idx, covered_points
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_idx
            
            # Add best candidate
            if best_candidate is not None:
                representatives.append(best_candidate)
                remaining_candidates.remove(best_candidate)
                
                # Update covered points
                covered_points = self._update_covered_points(z_latent, representatives)
                
                if (step + 1) % max(1, n_representatives // 10) == 0:
                    coverage_pct = len(covered_points) / n_samples * 100
                    logger.debug(f"Step {step + 1}: Selected point {best_candidate} "
                               f"(score: {best_score:.4f}, coverage: {coverage_pct:.1f}%)")
        
        return representatives
    
    def _compute_objective_score(
        self,
        z_latent: np.ndarray,
        current_representatives: List[int],
        candidate_idx: int,
        covered_points: set
    ) -> float:
        """
        Compute the objective function score for a candidate representative.
        
        Args:
            z_latent: Latent space coordinates
            current_representatives: Current representative indices
            candidate_idx: Index of candidate representative
            covered_points: Set of currently covered point indices
            
        Returns:
            Objective function value (higher is better)
        """
        # Information gain component
        info_gain = self._compute_information_gain(z_latent, candidate_idx, covered_points)
        
        # Redundancy penalty component
        redundancy_penalty = self._compute_redundancy_penalty(z_latent, candidate_idx, current_representatives)
        
        # Combined objective
        objective = self.info_weight * info_gain + self.redundancy_weight * redundancy_penalty
        
        return objective
    
    def _compute_information_gain(
        self, 
        z_latent: np.ndarray, 
        point_idx: int, 
        covered_points: set
    ) -> float:
        """
        Compute information gain when adding a new representative point.
        
        Args:
            z_latent: Latent space coordinates
            point_idx: Index of candidate point
            covered_points: Set of currently covered points
            
        Returns:
            Information gain value
        """
        n_samples = len(z_latent)
        
        if len(covered_points) == 0:
            # For first point, return inverse of local density
            nbrs = NearestNeighbors(n_neighbors=min(4, n_samples)).fit(z_latent)
            distances, _ = nbrs.kneighbors([z_latent[point_idx]])
            avg_distance = np.mean(distances[0][1:])  # Exclude self
            return avg_distance
        
        # Find uncovered points
        uncovered_indices = set(range(n_samples)) - covered_points
        
        if not uncovered_indices:
            return 0.0
        
        uncovered_points = z_latent[list(uncovered_indices)]
        candidate_point = z_latent[point_idx]
        
        # Information gain inversely related to average distance to uncovered points
        distances_to_uncovered = cdist([candidate_point], uncovered_points)[0]
        avg_distance_to_uncovered = np.mean(distances_to_uncovered)
        information_gain = 1.0 / (1.0 + avg_distance_to_uncovered)
        
        return information_gain
    
    def _compute_redundancy_penalty(
        self, 
        z_latent: np.ndarray, 
        candidate_idx: int, 
        representative_indices: List[int]
    ) -> float:
        """
        Compute redundancy penalty for a candidate point.
        
        Args:
            z_latent: Latent space coordinates
            candidate_idx: Index of candidate point
            representative_indices: List of existing representative indices
            
        Returns:
            Redundancy penalty (higher = less redundant = better)
        """
        if len(representative_indices) == 0:
            return 1.0  # No redundancy for first point
        
        candidate_point = z_latent[candidate_idx]
        representative_points = z_latent[representative_indices]
        
        # Calculate distances to all existing representatives
        distances = cdist([candidate_point], representative_points)[0]
        
        # Minimum distance to existing representatives (higher = less redundant)
        min_distance = np.min(distances)
        
        return min_distance
    
    def _update_covered_points(
        self, 
        z_latent: np.ndarray, 
        representative_indices: List[int]
    ) -> set:
        """
        Update the set of points that are well-covered by current representatives.
        
        Args:
            z_latent: Latent space coordinates
            representative_indices: Current representative indices
            
        Returns:
            Set of covered point indices
        """
        if not representative_indices:
            return set()
        
        covered_points = set()
        representative_points = z_latent[representative_indices]
        
        # For each point, check if it's within coverage radius of any representative
        for i, point in enumerate(z_latent):
            distances = cdist([point], representative_points)[0]
            if np.min(distances) <= self.coverage_radius:
                covered_points.add(i)
        
        return covered_points
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """Create distance-based specific visualizations."""
        # Call parent visualization first
        super().create_visualization(result, output_dir, title_suffix, **plot_kwargs)
        
        # Create distance-based specific plots
        self._create_coverage_visualization(result, output_dir)
        self._create_distance_distribution_plot(result, output_dir)
    
    def _create_coverage_visualization(self, result: SamplingResult, output_dir: str) -> None:
        """Create coverage visualization with coverage circles."""
        try:
            import matplotlib.pyplot as plt
            
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            coverage_radius = result.method_info.get('coverage_radius', 0.2)
            
            # Find covered points
            covered_points = set()
            selected_points = z_latent[selected_indices]
            
            for i, point in enumerate(z_latent):
                distances = cdist([point], selected_points)[0]
                if np.min(distances) <= coverage_radius:
                    covered_points.add(i)
            
            plt.figure(figsize=(14, 10))
            
            # Color points based on coverage
            for i, point in enumerate(z_latent):
                color = 'lightcoral' if i in covered_points else 'lightblue'
                alpha = 0.6 if i in covered_points else 0.2
                plt.scatter(point[0], point[1], c=color, alpha=alpha, s=5)
            
            # Plot representatives with coverage circles
            for x, y in selected_points:
                plt.scatter(x, y, s=150, color='darkred', marker='*', 
                           edgecolors='white', linewidths=2, zorder=10)
                
                # Draw coverage circle
                circle = plt.Circle((x, y), coverage_radius, fill=False, 
                                  color='darkred', alpha=0.7, linestyle='-', linewidth=2)
                plt.gca().add_patch(circle)
            
            coverage_pct = len(covered_points) / len(z_latent) * 100
            plt.title(f'Coverage Analysis: {coverage_pct:.1f}% of points covered\n'
                     f'Coverage radius: {coverage_radius}')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            
            # Custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightcoral', alpha=0.6, label=f'Covered points ({len(covered_points)})'),
                Patch(facecolor='lightblue', alpha=0.2, label=f'Uncovered points ({len(z_latent) - len(covered_points)})'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='darkred', 
                          markersize=12, label='Representatives')
            ]
            plt.legend(handles=legend_elements)
            
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.savefig(os.path.join(output_dir, 'coverage_analysis.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create coverage visualization: {e}")
    
    def _create_distance_distribution_plot(self, result: SamplingResult, output_dir: str) -> None:
        """Create distribution plot of distances to nearest representative."""
        try:
            import matplotlib.pyplot as plt
            
            z_latent = result.latent_coordinates
            selected_indices = result.selected_indices
            selected_points = z_latent[selected_indices]
            coverage_radius = result.method_info.get('coverage_radius', 0.2)
            
            # Calculate distances from each point to nearest representative
            distances_to_nearest = []
            for point in z_latent:
                distances = cdist([point], selected_points)[0]
                distances_to_nearest.append(np.min(distances))
            
            plt.figure(figsize=(12, 8))
            
            plt.hist(distances_to_nearest, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(coverage_radius, color='red', linestyle='--', linewidth=2, 
                       label=f'Coverage radius ({coverage_radius})')
            plt.axvline(np.mean(distances_to_nearest), color='green', linestyle='--', linewidth=2,
                       label=f'Mean distance ({np.mean(distances_to_nearest):.3f})')
            
            plt.title('Distribution of Distances to Nearest Representative')
            plt.xlabel('Distance to Nearest Representative')
            plt.ylabel('Number of Points')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'distance_distribution.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create distance distribution plot: {e}")