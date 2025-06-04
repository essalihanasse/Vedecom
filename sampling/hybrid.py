"""
Hybrid sampling implementation combining cluster-based and distance-based approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult
from .cluster_based import ClusterBasedSampler
from .representative import DistanceBasedSampler
import os
logger = logging.getLogger(__name__)

class HybridSampler(BaseSampler):
    """
    Hybrid representative sampling method.
    
    Combines cluster-based and distance-based approaches to get the benefits
    of both structured coverage from clustering and diversity from distance-based selection.
    """
    
    def __init__(
        self,
        cluster_fraction: float = 0.7,
        distance_fraction: float = 0.3,
        # Cluster-based parameters
        cluster_method: str = 'kmeans',
        cluster_sizing_method: str = 'adaptive',
        within_cluster_method: str = 'centroid_distance',
        min_clusters: int = 2,
        max_clusters: int = 500,
        n_clusters_factor: float = 0.1,
        # Distance-based parameters
        info_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        coverage_radius: float = 0.2,
        candidate_fraction: float = 1.0,
        **kwargs
    ):
        """
        Initialize hybrid sampler.
        
        Args:
            cluster_fraction: Fraction of samples from clustering approach
            distance_fraction: Fraction of samples from distance-based approach
            **kwargs: Parameters for underlying methods
        """
        super().__init__('hybrid', **kwargs)
        
        # Validate fractions
        if abs(cluster_fraction + distance_fraction - 1.0) > 1e-6:
            logger.warning(f"Fractions don't sum to 1.0: {cluster_fraction} + {distance_fraction}")
            # Normalize
            total = cluster_fraction + distance_fraction
            cluster_fraction = cluster_fraction / total
            distance_fraction = distance_fraction / total
        
        self.cluster_fraction = cluster_fraction
        self.distance_fraction = distance_fraction
        
        # Initialize component samplers
        self.cluster_sampler = ClusterBasedSampler(
            cluster_method=cluster_method,
            cluster_sizing_method=cluster_sizing_method,
            within_cluster_method=within_cluster_method,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            n_clusters_factor=n_clusters_factor,
            info_weight=info_weight,
            redundancy_weight=redundancy_weight
        )
        
        self.distance_sampler = DistanceBasedSampler(
            info_weight=info_weight,
            redundancy_weight=redundancy_weight,
            coverage_radius=coverage_radius,
            candidate_fraction=candidate_fraction
        )
        
        logger.info(f"Hybrid sampler initialized: {cluster_fraction:.1%} clustering, {distance_fraction:.1%} distance-based")
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample representatives using hybrid approach.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting hybrid sampling for {sample_size} samples")
        
        # Calculate sample sizes for each method
        cluster_sample_size = int(sample_size * self.cluster_fraction)
        distance_sample_size = sample_size - cluster_sample_size
        
        logger.info(f"Allocation: {cluster_sample_size} from clustering, {distance_sample_size} from distance-based")
        
        # Phase 1: Get representatives from clustering
        cluster_indices = []
        cluster_info = {}
        
        if cluster_sample_size > 0:
            logger.info("Phase 1: Cluster-based selection")
            cluster_result = self.cluster_sampler.sample(z_latent, cluster_sample_size, original_df)
            cluster_indices = cluster_result.selected_indices
            cluster_info = cluster_result.method_info
        
        # Phase 2: Get additional representatives using distance-based method
        distance_indices = []
        distance_info = {}
        
        if distance_sample_size > 0:
            logger.info("Phase 2: Distance-based selection from remaining points")
            
            # Create list of remaining candidates (excluding cluster representatives)
            remaining_indices = [i for i in range(len(z_latent)) if i not in cluster_indices]
            
            if remaining_indices:
                distance_indices = self._select_distance_based_from_remaining(
                    z_latent, remaining_indices, cluster_indices, distance_sample_size
                )
                
                # Get some statistics for the distance-based selection
                if distance_indices:
                    from .base import SamplingUtils
                    all_selected = cluster_indices + distance_indices
                    coverage_stats = SamplingUtils.calculate_coverage_statistics(
                        z_latent, all_selected, self.distance_sampler.coverage_radius
                    )
                    distance_info = {
                        'method': 'distance_based_hybrid',
                        'selected_from_remaining': len(distance_indices),
                        'coverage_stats': coverage_stats
                    }
            else:
                logger.warning("No remaining points for distance-based selection")
        
        # Combine results
        selected_indices = cluster_indices + distance_indices
        
        # Ensure we don't exceed requested size
        if len(selected_indices) > sample_size:
            selected_indices = selected_indices[:sample_size]
            logger.warning(f"Truncated selection to {sample_size} samples")
        
        # Create comprehensive method info
        additional_info = {
            'cluster_fraction': self.cluster_fraction,
            'distance_fraction': self.distance_fraction,
            'cluster_sample_size': cluster_sample_size,
            'distance_sample_size': distance_sample_size,
            'cluster_representatives': len(cluster_indices),
            'distance_representatives': len(distance_indices),
            'total_representatives': len(selected_indices),
            'cluster_info': cluster_info,
            'distance_info': distance_info,
            'combination_method': 'sequential'
        }
        
        logger.info(f"Hybrid sampling completed: {len(cluster_indices)} + {len(distance_indices)} = {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _select_distance_based_from_remaining(
        self,
        z_latent: np.ndarray,
        remaining_indices: List[int],
        selected_indices: List[int],
        n_additional: int
    ) -> List[int]:
        """
        Select additional representatives from remaining points using distance-based approach.
        
        Args:
            z_latent: Latent space coordinates
            remaining_indices: Indices of remaining candidate points
            selected_indices: Indices of already selected points
            n_additional: Number of additional representatives needed
            
        Returns:
            List of additional selected indices
        """
        if not remaining_indices or n_additional <= 0:
            return []
        
        additional_selected = []
        candidates = remaining_indices.copy()
        current_selected = selected_indices.copy()
        
        logger.debug(f"Selecting {n_additional} additional points from {len(candidates)} candidates")
        
        for step in range(min(n_additional, len(candidates))):
            best_score = -float('inf')
            best_candidate = None
            
            # Evaluate each remaining candidate
            for candidate_idx in candidates:
                score = self._compute_hybrid_objective_score(
                    z_latent, current_selected, candidate_idx
                )
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_idx
            
            # Add best candidate
            if best_candidate is not None:
                additional_selected.append(best_candidate)
                current_selected.append(best_candidate)
                candidates.remove(best_candidate)
                
                logger.debug(f"Step {step + 1}: Selected point {best_candidate} (score: {best_score:.4f})")
        
        return additional_selected
    
    def _compute_hybrid_objective_score(
        self,
        z_latent: np.ndarray,
        current_selected: List[int],
        candidate_idx: int
    ) -> float:
        """
        Compute objective score for hybrid selection.
        
        This uses a simplified distance-based scoring that focuses on
        maximizing minimum distance to already selected points.
        
        Args:
            z_latent: Latent space coordinates
            current_selected: Currently selected point indices
            candidate_idx: Index of candidate point
            
        Returns:
            Objective score (higher is better)
        """
        if not current_selected:
            return 1.0
        
        # Simple distance-based score: maximize minimum distance to selected points
        from scipy.spatial.distance import cdist
        
        candidate_point = z_latent[candidate_idx]
        selected_points = z_latent[current_selected]
        
        distances = cdist([candidate_point], selected_points)[0]
        min_distance = np.min(distances)
        
        return min_distance
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """Create hybrid-specific visualizations."""
        # Call parent visualization first
        super().create_visualization(result, output_dir, title_suffix, **plot_kwargs)
        
        # Create hybrid-specific plots
        self._create_hybrid_breakdown_visualization(result, output_dir)
        self._create_method_comparison_plot(result, output_dir)
    
    def _create_hybrid_breakdown_visualization(self, result: SamplingResult, output_dir: str) -> None:
        """Create visualization showing breakdown by selection method."""
        try:
            import matplotlib.pyplot as plt
            
            z_latent = result.latent_coordinates
            method_info = result.method_info
            
            # Get indices for each method
            cluster_representatives = method_info.get('cluster_representatives', 0)
            cluster_indices = result.selected_indices[:cluster_representatives]
            distance_indices = result.selected_indices[cluster_representatives:]
            
            plt.figure(figsize=(14, 10))
            
            # Plot all points
            plt.scatter(
                z_latent[:, 0], z_latent[:, 1], 
                alpha=0.1, s=5, color='lightgray', label='All points'
            )
            
            # Plot cluster-based representatives
            if cluster_indices:
                cluster_points = z_latent[cluster_indices]
                plt.scatter(
                    cluster_points[:, 0], cluster_points[:, 1],
                    alpha=0.8, s=80, color='blue', marker='o',
                    edgecolors='darkblue', linewidths=1,
                    label=f'Cluster-based ({len(cluster_indices)})'
                )
            
            # Plot distance-based representatives
            if distance_indices:
                distance_points = z_latent[distance_indices]
                plt.scatter(
                    distance_points[:, 0], distance_points[:, 1],
                    alpha=0.8, s=80, color='red', marker='s',
                    edgecolors='darkred', linewidths=1,
                    label=f'Distance-based ({len(distance_indices)})'
                )
            
            cluster_frac = method_info.get('cluster_fraction', 0)
            distance_frac = method_info.get('distance_fraction', 0)
            
            plt.title(f'Hybrid Sampling Breakdown\n'
                     f'Cluster-based: {cluster_frac:.1%} ({len(cluster_indices)}), '
                     f'Distance-based: {distance_frac:.1%} ({len(distance_indices)})')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'hybrid_breakdown.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create hybrid breakdown visualization: {e}")
    
    def _create_method_comparison_plot(self, result: SamplingResult, output_dir: str) -> None:
        """Create comparison showing the effect of each method."""
        try:
            import matplotlib.pyplot as plt
            
            method_info = result.method_info
            cluster_info = method_info.get('cluster_info', {})
            distance_info = method_info.get('distance_info', {})
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Allocation breakdown
            methods = ['Cluster-based', 'Distance-based']
            sizes = [
                method_info.get('cluster_representatives', 0),
                method_info.get('distance_representatives', 0)
            ]
            colors = ['blue', 'red']
            
            ax1.pie(sizes, labels=methods, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Method Allocation')
            
            # Plot 2: Summary statistics
            ax2.axis('off')
            ax2.text(0.1, 0.9, 'Hybrid Sampling Summary', fontsize=14, weight='bold', 
                    transform=ax2.transAxes)
            
            y_pos = 0.8
            summary_items = [
                f"Total representatives: {result.n_selected}",
                f"Cluster fraction: {method_info.get('cluster_fraction', 0):.1%}",
                f"Distance fraction: {method_info.get('distance_fraction', 0):.1%}",
                f"Cluster representatives: {method_info.get('cluster_representatives', 0)}",
                f"Distance representatives: {method_info.get('distance_representatives', 0)}",
            ]
            
            for item in summary_items:
                ax2.text(0.1, y_pos, item, fontsize=11, transform=ax2.transAxes)
                y_pos -= 0.1
            
            # Add cluster info if available
            if 'n_clusters' in cluster_info:
                ax2.text(0.1, y_pos - 0.05, f"Clusters used: {cluster_info['n_clusters']}", 
                        fontsize=11, transform=ax2.transAxes)
            
            # Plot 3: Placeholder for additional metrics
            ax3.axis('off')
            ax3.text(0.5, 0.5, 'Additional metrics\nwould go here', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            
            # Plot 4: Method parameters
            ax4.axis('off')
            ax4.text(0.1, 0.9, 'Method Parameters', fontsize=12, weight='bold', 
                    transform=ax4.transAxes)
            
            param_y = 0.8
            param_items = [
                f"Cluster method: {cluster_info.get('cluster_method', 'N/A')}",
                f"Within-cluster method: {cluster_info.get('within_cluster_method', 'N/A')}",
                f"Coverage radius: {distance_info.get('coverage_radius', 'N/A')}",
            ]
            
            for item in param_items:
                ax4.text(0.1, param_y, item, fontsize=10, transform=ax4.transAxes)
                param_y -= 0.1
            
            plt.suptitle('Hybrid Sampling Method Comparison', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'hybrid_method_comparison.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create method comparison plot: {e}")