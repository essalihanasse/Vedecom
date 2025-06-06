"""
Optimized base classes for latent space sampling methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cdist

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
        
        # Save core results
        self.selected_points.to_csv(os.path.join(output_dir, 'selected_points.csv'), index=False)
        np.save(os.path.join(output_dir, 'selected_indices.npy'), np.array(self.selected_indices))
        
        with open(os.path.join(output_dir, 'method_info.pkl'), 'wb') as f:
            pickle.dump(self.method_info, f)
        
        # Save latent coordinates for selected points
        selected_latent = self.latent_coordinates[self.selected_indices]
        np.save(os.path.join(output_dir, 'selected_latent_coordinates.npy'), selected_latent)

class BaseSampler(ABC):
    """Abstract base class for latent space sampling methods."""
    
    def __init__(self, method_name: str, **kwargs):
        self.method_name = method_name
        self.params = kwargs
        logger.info(f"Initialized {method_name} sampler")
    
    @abstractmethod
    def sample(self, z_latent: np.ndarray, sample_size: int, 
               original_df: pd.DataFrame, **kwargs) -> SamplingResult:
        """Sample representative points from latent space."""
        pass
    
    def create_standard_result(self, selected_indices: List[int], z_latent: np.ndarray,
                             original_df: pd.DataFrame, sample_size: int,
                             additional_info: Optional[Dict[str, Any]] = None) -> SamplingResult:
        """Create a standardized SamplingResult object."""
        # Create selected points DataFrame
        selected_points = original_df.iloc[selected_indices].copy()
        selected_points['latent_1'] = z_latent[selected_indices, 0]
        selected_points['latent_2'] = z_latent[selected_indices, 1]
        selected_points['representative_id'] = range(len(selected_indices))
        
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
    
    def create_visualization(self, result: SamplingResult, output_dir: str, 
                           title_suffix: str = "", **plot_kwargs) -> None:
        """Create standard visualization for sampling results."""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Overview
        self._plot_overview(axes[0], result, title_suffix)
        
        # Plot 2: Distance analysis
        self._plot_distance_analysis(axes[1], result)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{self.method_name}_sampling.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overview(self, ax, result: SamplingResult, title_suffix: str):
        """Plot overview of sampling results."""
        ax.scatter(result.latent_coordinates[:, 0], result.latent_coordinates[:, 1], 
                  alpha=0.3, s=8, color='lightblue', label='All points')
        
        selected_latent = result.latent_coordinates[result.selected_indices]
        ax.scatter(selected_latent[:, 0], selected_latent[:, 1],
                  alpha=0.9, s=100, color='red', edgecolors='darkred', 
                  linewidths=2, label=f'Selected ({result.n_selected})', zorder=5)
        
        ax.set_title(f'{self.method_name.title()} Sampling {title_suffix}')
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distance_analysis(self, ax, result: SamplingResult):
        """Plot distance distribution analysis."""
        if result.n_selected > 1:
            selected_latent = result.latent_coordinates[result.selected_indices]
            distances_to_nearest = []
            
            for point in result.latent_coordinates:
                distances = cdist([point], selected_latent)[0]
                distances_to_nearest.append(np.min(distances))
            
            ax.hist(distances_to_nearest, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(distances_to_nearest), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(distances_to_nearest):.3f}')
            ax.set_title('Distance to Nearest Representative')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Number of Points')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Not enough points\nfor distance analysis', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distance Analysis')

class SamplingUtils:
    """Consolidated utility functions for sampling methods."""
    
    @staticmethod
    def calculate_coverage_statistics(z_latent: np.ndarray, selected_indices: List[int],
                                    coverage_radius: float = 0.2) -> Dict[str, float]:
        """Calculate comprehensive coverage statistics."""
        if not selected_indices:
            return {'coverage_fraction': 0.0}
        
        n_samples = len(z_latent)
        selected_points = z_latent[selected_indices]
        
        # Calculate distances and coverage
        distances_to_nearest = []
        covered_count = 0
        
        for point in z_latent:
            distances = cdist([point], selected_points)[0]
            min_distance = np.min(distances)
            distances_to_nearest.append(min_distance)
            
            if min_distance <= coverage_radius:
                covered_count += 1
        
        # Representative balance
        assignment = []
        for point in z_latent:
            distances = cdist([point], selected_points)[0]
            assignment.append(np.argmin(distances))
        
        assignment_counts = np.bincount(assignment, minlength=len(selected_indices))
        balance = (np.std(assignment_counts) / np.mean(assignment_counts) 
                  if np.mean(assignment_counts) > 0 else 0)
        
        return {
            'coverage_fraction': covered_count / n_samples,
            'avg_distance_to_nearest_rep': np.mean(distances_to_nearest),
            'median_distance_to_nearest_rep': np.median(distances_to_nearest),
            'max_distance_to_nearest_rep': np.max(distances_to_nearest),
            'representative_balance': balance,
            'points_covered': covered_count,
            'total_points': n_samples
        }
    
    @staticmethod
    def calculate_adaptive_clusters(n_samples: int, sample_size: int, 
                                  method: str = 'adaptive', **params) -> int:
        """Calculate optimal number of clusters."""
        import math
        
        if method == 'adaptive':
            ratio = n_samples / sample_size
            if ratio < 10:
                n_clusters = max(2, min(sample_size // 3, int(math.sqrt(sample_size))))
            elif ratio < 100:
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
            
        else:  # fixed
            n_clusters_factor = params.get('n_clusters_factor', 0.1)
            min_clusters = params.get('min_clusters', 5)
            max_clusters = params.get('max_clusters', 50)
            n_clusters = max(min_clusters, min(max_clusters, int(n_samples * n_clusters_factor)))
        
        return int(n_clusters)

class MultiMethodSampler:
    """Manager class for running multiple sampling methods."""
    
    def __init__(self):
        self.samplers: Dict[str, BaseSampler] = {}
        self.results: Dict[str, Dict[int, SamplingResult]] = {}
    
    def register_sampler(self, name: str, sampler: BaseSampler) -> None:
        """Register a sampling method."""
        self.samplers[name] = sampler
        self.results[name] = {}
        logger.info(f"Registered sampler: {name}")
    
    def run_sampling(self, z_latent: np.ndarray, original_df: pd.DataFrame,
                    sample_sizes: List[int], output_base_dir: str,
                    methods: Optional[List[str]] = None) -> Dict[str, Dict[int, SamplingResult]]:
        """Run multiple sampling methods on the same data."""
        methods = methods or list(self.samplers.keys())
        os.makedirs(output_base_dir, exist_ok=True)
        
        for method_name in methods:
            if method_name not in self.samplers:
                logger.warning(f"Sampler {method_name} not registered. Skipping.")
                continue
            
            logger.info(f"Running {method_name} sampling...")
            sampler = self.samplers[method_name]
            method_dir = os.path.join(output_base_dir, f'method_{method_name}')
            
            for sample_size in sample_sizes:
                try:
                    result = sampler.sample(z_latent, sample_size, original_df)
                    
                    # Save result
                    sample_dir = os.path.join(method_dir, f'samples_{sample_size}')
                    result.save_to_directory(sample_dir)
                    
                    # Create visualization
                    sampler.create_visualization(result, sample_dir, title_suffix=f"(n={sample_size})")
                    
                    self.results[method_name][sample_size] = result
                    logger.info(f"  {method_name} (n={sample_size}): {result.n_selected} representatives")
                    
                except Exception as e:
                    logger.error(f"Error in {method_name} sampling (size {sample_size}): {e}")
        
        return self.results
    
    def create_comparison_plots(self, output_dir: str, sample_size: int, z_latent: np.ndarray) -> None:
        """Create comparison plots between methods for a specific sample size."""
        methods_with_results = [m for m in self.results if sample_size in self.results[m]]
        
        if len(methods_with_results) <= 1:
            return
        
        fig, axes = plt.subplots(1, len(methods_with_results), 
                               figsize=(6 * len(methods_with_results), 6))
        if len(methods_with_results) == 1:
            axes = [axes]
        
        for i, method in enumerate(methods_with_results):
            ax = axes[i]
            result = self.results[method][sample_size]
            
            # Plot all points
            ax.scatter(z_latent[:, 0], z_latent[:, 1], alpha=0.1, s=5, color='lightblue')
            
            # Plot selected points
            selected_latent = z_latent[result.selected_indices]
            ax.scatter(selected_latent[:, 0], selected_latent[:, 1], alpha=0.8, s=30, color='red')
            
            ax.set_title(f'{method.title()}\n{result.n_selected} samples')
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Sampling Method Comparison - {sample_size} Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'method_comparison_{sample_size}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()