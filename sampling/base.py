"""
Base classes and utilities for latent space sampling methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SamplingResult:
    """Container for sampling results."""
    selected_indices: List[int]
    selected_points: pd.DataFrame
    method_info: Dict[str, Any]
    latent_coordinates: np.ndarray
    
    @property
    def n_selected(self) -> int:
        """Number of selected samples."""
        return len(self.selected_indices)
    
    def save_to_directory(self, output_dir: str) -> None:
        """Save sampling results to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save selected points and indices
        self.selected_points.to_csv(
            os.path.join(output_dir, 'selected_points.csv'), 
            index=False
        )
        np.save(
            os.path.join(output_dir, 'selected_indices.npy'), 
            np.array(self.selected_indices)
        )
        
        # Save method info
        with open(os.path.join(output_dir, 'method_info.pkl'), 'wb') as f:
            pickle.dump(self.method_info, f)
        
        # Save latent coordinates for selected points
        selected_latent = self.latent_coordinates[self.selected_indices]
        np.save(
            os.path.join(output_dir, 'selected_latent_coordinates.npy'),
            selected_latent
        )

class BaseSampler(ABC):
    """
    Abstract base class for latent space sampling methods.
    """
    
    def __init__(self, method_name: str, **kwargs):
        self.method_name = method_name
        self.params = kwargs
        logger.info(f"Initialized {method_name} sampler with params: {kwargs}")
    
    @abstractmethod
    def sample(
        self, 
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample representative points from latent space.
        
        Args:
            z_latent: Latent space coordinates (n_samples x latent_dim)
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            **kwargs: Additional method-specific parameters
            
        Returns:
            SamplingResult containing selected points and metadata
        """
        pass
    
    def create_standard_result(
        self,
        selected_indices: List[int],
        z_latent: np.ndarray,
        original_df: pd.DataFrame,
        sample_size: int,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> SamplingResult:
        """
        Create a standardized SamplingResult object.
        
        Args:
            selected_indices: List of selected indices
            z_latent: Latent space coordinates
            original_df: Original data DataFrame
            sample_size: Requested sample size
            additional_info: Additional method-specific information
            
        Returns:
            SamplingResult object
        """
        # Create selected points DataFrame
        selected_points = original_df.iloc[selected_indices].copy()
        selected_points['latent_1'] = z_latent[selected_indices, 0]
        selected_points['latent_2'] = z_latent[selected_indices, 1]
        selected_points['representative_id'] = range(len(selected_indices))
        selected_points['selection_method'] = self.method_name
        selected_points['sample_size'] = sample_size
        
        # Method info
        method_info = {
            'method': self.method_name,
            'n_representatives': len(selected_indices),
            'requested_sample_size': sample_size,
            'params': self.params.copy()
        }
        
        if additional_info:
            method_info.update(additional_info)
        
        return SamplingResult(
            selected_indices=selected_indices,
            selected_points=selected_points,
            method_info=method_info,
            latent_coordinates=z_latent
        )
    
    def create_visualization(
        self,
        result: SamplingResult,
        output_dir: str,
        title_suffix: str = "",
        **plot_kwargs
    ) -> None:
        """
        Create standard visualization for sampling results.
        
        Args:
            result: SamplingResult object
            output_dir: Directory to save plots
            title_suffix: Additional text for plot title
            **plot_kwargs: Additional plotting parameters
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Overview
        ax1.scatter(
            result.latent_coordinates[:, 0], 
            result.latent_coordinates[:, 1], 
            alpha=0.3, s=8, color='lightblue', label='All points'
        )
        
        selected_latent = result.latent_coordinates[result.selected_indices]
        ax1.scatter(
            selected_latent[:, 0], 
            selected_latent[:, 1],
            alpha=0.9, s=100, color='red', edgecolors='darkred', 
            linewidths=2, label=f'Selected ({result.n_selected})', zorder=5
        )
        
        ax1.set_title(f'{self.method_name.title()} Sampling {title_suffix}')
        ax1.set_xlabel('Latent Dimension 1')
        ax1.set_ylabel('Latent Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Distance distribution (if more than 1 point selected)
        if result.n_selected > 1:
            from scipy.spatial.distance import cdist
            
            distances_to_nearest = []
            for point in result.latent_coordinates:
                distances = cdist([point], selected_latent)[0]
                distances_to_nearest.append(np.min(distances))
            
            ax2.hist(
                distances_to_nearest, bins=50, alpha=0.7, 
                color='skyblue', edgecolor='black'
            )
            ax2.axvline(
                np.mean(distances_to_nearest), color='red', 
                linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances_to_nearest):.3f}'
            )
            ax2.set_title('Distance to Nearest Representative')
            ax2.set_xlabel('Distance')
            ax2.set_ylabel('Number of Points')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Not enough points\nfor distance analysis', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distance Analysis')
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'{self.method_name}_sampling.png'), 
            dpi=300, bbox_inches='tight'
        )
        plt.close()

class SamplingUtils:
    """Utility functions for sampling methods."""
    
    @staticmethod
    def calculate_coverage_statistics(
        z_latent: np.ndarray, 
        selected_indices: List[int],
        coverage_radius: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate coverage statistics for sampled points.
        
        Args:
            z_latent: All latent coordinates
            selected_indices: Indices of selected points
            coverage_radius: Radius for coverage calculation
            
        Returns:
            Dictionary with coverage statistics
        """
        from scipy.spatial.distance import cdist
        
        n_samples = len(z_latent)
        selected_points = z_latent[selected_indices]
        
        if len(selected_points) == 0:
            return {'coverage_fraction': 0.0}
        
        # Compute coverage
        covered_points = set()
        for i, point in enumerate(z_latent):
            distances = cdist([point], selected_points)[0]
            if np.min(distances) <= coverage_radius:
                covered_points.add(i)
        
        coverage_fraction = len(covered_points) / n_samples
        
        # Distances to nearest representative
        distances_to_nearest = []
        for point in z_latent:
            distances = cdist([point], selected_points)[0]
            distances_to_nearest.append(np.min(distances))
        
        # Representative balance
        assignment = []
        for point in z_latent:
            distances = cdist([point], selected_points)[0]
            nearest_rep = np.argmin(distances)
            assignment.append(nearest_rep)
        
        assignment_counts = np.bincount(assignment, minlength=len(selected_indices))
        balance = (np.std(assignment_counts) / np.mean(assignment_counts) 
                  if np.mean(assignment_counts) > 0 else 0)
        
        return {
            'coverage_fraction': coverage_fraction,
            'avg_distance_to_nearest_rep': np.mean(distances_to_nearest),
            'median_distance_to_nearest_rep': np.median(distances_to_nearest),
            'max_distance_to_nearest_rep': np.max(distances_to_nearest),
            'representative_balance': balance,
            'coverage_radius_used': coverage_radius,
            'points_covered': len(covered_points),
            'total_points': n_samples
        }
    
    @staticmethod
    def calculate_adaptive_clusters(
        n_samples: int, 
        sample_size: int, 
        method: str = 'adaptive',
        **params
    ) -> int:
        """
        Calculate optimal number of clusters based on data size and sample size.
        
        Args:
            n_samples: Total number of data points
            sample_size: Number of representatives to select
            method: Method for determining clusters
            **params: Additional parameters
            
        Returns:
            Optimal number of clusters
        """
        import math
        
        if method == 'adaptive':
            data_to_sample_ratio = n_samples / sample_size
            
            if data_to_sample_ratio < 10:
                n_clusters = max(2, min(sample_size // 3, int(math.sqrt(sample_size))))
            elif data_to_sample_ratio < 100:
                n_clusters = max(5, int(math.sqrt(sample_size * 2)))
            else:
                base_clusters = int(math.sqrt(sample_size))
                data_factor = min(3, math.log10(n_samples / 1000))
                n_clusters = int(base_clusters * (1 + data_factor))
            
            # Apply bounds
            min_clusters = max(2, sample_size // 50)
            max_clusters = min(sample_size, n_samples // 10, 500)
            n_clusters = max(min_clusters, min(max_clusters, n_clusters))
            
        elif method == 'sqrt_rule':
            n_clusters = int(math.sqrt(sample_size))
            n_clusters = max(2, min(n_clusters, sample_size // 2))
            
        elif method == 'proportional':
            base_factor = params.get('base_factor', 0.15)
            data_modifier = min(2.0, math.log10(n_samples / 1000))
            n_clusters = int(sample_size * base_factor * (1 + data_modifier * 0.3))
            n_clusters = max(5, min(n_clusters, sample_size // 3))
            
        else:  # fixed or unknown
            n_clusters_factor = params.get('n_clusters_factor', 0.1)
            min_clusters = params.get('min_clusters', 5)
            max_clusters = params.get('max_clusters', 50)
            
            n_clusters = max(
                min_clusters,
                min(max_clusters, int(n_samples * n_clusters_factor))
            )
        
        return int(n_clusters)

class MultiMethodSampler:
    """
    Manager class for running multiple sampling methods.
    """
    
    def __init__(self):
        self.samplers: Dict[str, BaseSampler] = {}
        self.results: Dict[str, Dict[int, SamplingResult]] = {}
    
    def register_sampler(self, name: str, sampler: BaseSampler) -> None:
        """Register a sampling method."""
        self.samplers[name] = sampler
        self.results[name] = {}
        logger.info(f"Registered sampler: {name}")
    
    def run_sampling(
        self,
        z_latent: np.ndarray,
        original_df: pd.DataFrame,
        sample_sizes: List[int],
        output_base_dir: str,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, SamplingResult]]:
        """
        Run multiple sampling methods on the same data.
        
        Args:
            z_latent: Latent space coordinates
            original_df: Original data DataFrame
            sample_sizes: List of sample sizes to test
            output_base_dir: Base output directory
            methods: List of method names to run (None for all)
            
        Returns:
            Dictionary of results organized by method and sample size
        """
        if methods is None:
            methods = list(self.samplers.keys())
        
        os.makedirs(output_base_dir, exist_ok=True)
        
        for method_name in methods:
            if method_name not in self.samplers:
                logger.warning(f"Sampler {method_name} not registered. Skipping.")
                continue
            
            logger.info(f"Running {method_name} sampling...")
            sampler = self.samplers[method_name]
            method_dir = os.path.join(output_base_dir, f'method_{method_name}')
            
            for sample_size in sample_sizes:
                logger.info(f"  Sample size: {sample_size}")
                
                try:
                    result = sampler.sample(z_latent, sample_size, original_df)
                    
                    # Save result
                    sample_dir = os.path.join(method_dir, f'samples_{sample_size}')
                    result.save_to_directory(sample_dir)
                    
                    # Create visualization
                    sampler.create_visualization(
                        result, sample_dir, 
                        title_suffix=f"(n={sample_size})"
                    )
                    
                    self.results[method_name][sample_size] = result
                    
                    logger.info(f"    Selected {result.n_selected} representatives")
                    
                except Exception as e:
                    logger.error(f"Error in {method_name} sampling (size {sample_size}): {e}")
                    continue
        
        return self.results
    
    def create_comparison_plots(
        self, 
        output_dir: str, 
        sample_size: int,
        z_latent: np.ndarray
    ) -> None:
        """Create comparison plots between methods for a specific sample size."""
        methods_with_results = [
            method for method in self.results 
            if sample_size in self.results[method]
        ]
        
        if len(methods_with_results) <= 1:
            return
        
        fig, axes = plt.subplots(
            1, len(methods_with_results), 
            figsize=(6 * len(methods_with_results), 6)
        )
        if len(methods_with_results) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods_with_results):
            ax = axes[i]
            result = self.results[method][sample_size]
            
            # Plot all points
            ax.scatter(
                z_latent[:, 0], z_latent[:, 1], 
                alpha=0.1, s=5, color='lightblue'
            )
            
            # Plot selected points
            selected_latent = z_latent[result.selected_indices]
            ax.scatter(
                selected_latent[:, 0], selected_latent[:, 1], 
                alpha=0.8, s=30, color='red'
            )
            
            ax.set_title(f'{method.title()}\n{result.n_selected} samples')
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sampling Method Comparison - {sample_size} Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f'method_comparison_{sample_size}.png'), 
            dpi=300, bbox_inches='tight'
        )
        plt.close()