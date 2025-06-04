# evaluation/metrics.py
"""
Evaluation metrics for assessing sampling quality and model performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

class SamplingQualityMetrics:
    """
    Metrics for evaluating the quality of representative sampling.
    """
    
    @staticmethod
    def coverage_score(
        z_latent: np.ndarray, 
        selected_indices: List[int], 
        radius: float = 0.2
    ) -> Dict[str, float]:
        """
        Calculate coverage metrics for selected representatives.
        
        Args:
            z_latent: All latent coordinates
            selected_indices: Indices of selected representatives
            radius: Coverage radius
            
        Returns:
            Dictionary with coverage metrics
        """
        if not selected_indices:
            return {'coverage_ratio': 0.0, 'avg_distance': float('inf')}
        
        selected_points = z_latent[selected_indices]
        
        # Calculate distances from each point to nearest representative
        distances_to_nearest = []
        covered_count = 0
        
        for point in z_latent:
            distances = cdist([point], selected_points)[0]
            min_distance = np.min(distances)
            distances_to_nearest.append(min_distance)
            
            if min_distance <= radius:
                covered_count += 1
        
        coverage_ratio = covered_count / len(z_latent)
        avg_distance = np.mean(distances_to_nearest)
        median_distance = np.median(distances_to_nearest)
        max_distance = np.max(distances_to_nearest)
        
        return {
            'coverage_ratio': coverage_ratio,
            'avg_distance_to_nearest': avg_distance,
            'median_distance_to_nearest': median_distance,
            'max_distance_to_nearest': max_distance,
            'covered_points': covered_count,
            'total_points': len(z_latent)
        }
    
    @staticmethod
    def diversity_score(selected_indices: List[int], z_latent: np.ndarray) -> Dict[str, float]:
        """
        Calculate diversity metrics for selected representatives.
        
        Args:
            selected_indices: Indices of selected representatives
            z_latent: Latent coordinates
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(selected_indices) < 2:
            return {'min_distance': 0.0, 'avg_distance': 0.0, 'diversity_index': 0.0}
        
        selected_points = z_latent[selected_indices]
        
        # Pairwise distances between representatives
        pairwise_distances = pdist(selected_points)
        
        min_distance = np.min(pairwise_distances)
        avg_distance = np.mean(pairwise_distances)
        std_distance = np.std(pairwise_distances)
        
        # Diversity index (coefficient of variation)
        diversity_index = std_distance / avg_distance if avg_distance > 0 else 0
        
        return {
            'min_pairwise_distance': min_distance,
            'avg_pairwise_distance': avg_distance,
            'std_pairwise_distance': std_distance,
            'diversity_index': diversity_index,
            'n_representatives': len(selected_indices)
        }
    
    @staticmethod
    def representativeness_score(
        z_latent: np.ndarray, 
        selected_indices: List[int]
    ) -> Dict[str, float]:
        """
        Calculate how well the selected points represent the full distribution.
        
        Args:
            z_latent: All latent coordinates
            selected_indices: Indices of selected representatives
            
        Returns:
            Dictionary with representativeness metrics
        """
        if not selected_indices:
            return {'centroid_distance': float('inf'), 'variance_ratio': 0.0}
        
        selected_points = z_latent[selected_indices]
        
        # Compare centroids
        full_centroid = np.mean(z_latent, axis=0)
        selected_centroid = np.mean(selected_points, axis=0)
        centroid_distance = np.linalg.norm(full_centroid - selected_centroid)
        
        # Compare variances
        full_variance = np.var(z_latent, axis=0)
        selected_variance = np.var(selected_points, axis=0)
        
        # Variance ratio (how much of the original variance is preserved)
        variance_ratio = np.mean(selected_variance / (full_variance + 1e-8))
        
        # Compare distributions using KS test
        ks_stats = []
        for dim in range(z_latent.shape[1]):
            ks_stat, _ = stats.ks_2samp(z_latent[:, dim], selected_points[:, dim])
            ks_stats.append(ks_stat)
        
        avg_ks_stat = np.mean(ks_stats)
        
        return {
            'centroid_distance': centroid_distance,
            'variance_ratio': variance_ratio,
            'avg_ks_statistic': avg_ks_stat,
            'sampling_efficiency': len(selected_indices) / len(z_latent)
        }
    
    @staticmethod
    def clustering_quality_score(
        z_latent: np.ndarray, 
        cluster_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality using standard metrics.
        
        Args:
            z_latent: Latent coordinates
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with clustering metrics
        """
        if len(np.unique(cluster_labels)) < 2:
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
        
        try:
            silhouette = silhouette_score(z_latent, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(z_latent, cluster_labels)
            
            # Inertia (within-cluster sum of squares)
            inertia = 0.0
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                cluster_points = z_latent[cluster_labels == cluster_id]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'inertia': inertia,
                'n_clusters': len(np.unique(cluster_labels[cluster_labels != -1])),
                'n_noise_points': np.sum(cluster_labels == -1)
            }
            
        except Exception as e:
            logger.warning(f"Could not calculate clustering metrics: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}

class DistributionMetrics:
    """
    Metrics for comparing distributions between original and sampled data.
    """
    
    @staticmethod
    def statistical_distance_metrics(
        original_data: np.ndarray, 
        sampled_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate various statistical distance metrics.
        
        Args:
            original_data: Original dataset
            sampled_data: Sampled dataset
            
        Returns:
            Dictionary with distance metrics
        """
        metrics = {}
        
        # For each dimension, calculate statistical tests
        for dim in range(min(original_data.shape[1], sampled_data.shape[1])):
            orig_dim = original_data[:, dim]
            samp_dim = sampled_data[:, dim]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(orig_dim, samp_dim)
            metrics[f'ks_statistic_dim_{dim}'] = ks_stat
            metrics[f'ks_p_value_dim_{dim}'] = ks_p
            
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(orig_dim, samp_dim, alternative='two-sided')
            metrics[f'mannwhitney_statistic_dim_{dim}'] = mw_stat
            metrics[f'mannwhitney_p_value_dim_{dim}'] = mw_p
            
            # Wasserstein distance
            wasserstein_dist = stats.wasserstein_distance(orig_dim, samp_dim)
            metrics[f'wasserstein_distance_dim_{dim}'] = wasserstein_dist
        
        # Overall metrics
        metrics['avg_ks_statistic'] = np.mean([metrics[k] for k in metrics.keys() if 'ks_statistic_dim' in k])
        metrics['avg_wasserstein_distance'] = np.mean([metrics[k] for k in metrics.keys() if 'wasserstein_distance_dim' in k])
        
        return metrics
    
    @staticmethod
    def moment_comparison(
        original_data: np.ndarray, 
        sampled_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare statistical moments between datasets.
        
        Args:
            original_data: Original dataset
            sampled_data: Sampled dataset
            
        Returns:
            Dictionary with moment comparisons
        """
        metrics = {}
        
        for dim in range(min(original_data.shape[1], sampled_data.shape[1])):
            orig_dim = original_data[:, dim]
            samp_dim = sampled_data[:, dim]
            
            # Mean difference
            mean_diff = abs(np.mean(orig_dim) - np.mean(samp_dim))
            metrics[f'mean_difference_dim_{dim}'] = mean_diff
            
            # Variance ratio
            orig_var = np.var(orig_dim)
            samp_var = np.var(samp_dim)
            var_ratio = samp_var / (orig_var + 1e-8)
            metrics[f'variance_ratio_dim_{dim}'] = var_ratio
            
            # Skewness difference
            orig_skew = stats.skew(orig_dim)
            samp_skew = stats.skew(samp_dim)
            skew_diff = abs(orig_skew - samp_skew)
            metrics[f'skewness_difference_dim_{dim}'] = skew_diff
            
            # Kurtosis difference
            orig_kurt = stats.kurtosis(orig_dim)
            samp_kurt = stats.kurtosis(samp_dim)
            kurt_diff = abs(orig_kurt - samp_kurt)
            metrics[f'kurtosis_difference_dim_{dim}'] = kurt_diff
        
        return metrics

class ModelPerformanceMetrics:
    """
    Metrics for evaluating VAE model performance.
    """
    
    @staticmethod
    def reconstruction_quality(
        original: np.ndarray, 
        reconstructed: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            
        Returns:
            Dictionary with reconstruction metrics
        """
        # Mean squared error
        mse = np.mean((original - reconstructed) ** 2)
        
        # Root mean squared error
        rmse = np.sqrt(mse)
        
        # Mean absolute error
        mae = np.mean(np.abs(original - reconstructed))
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((original - reconstructed) ** 2)
        ss_tot = np.sum((original - np.mean(original)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Cosine similarity
        cosine_sim = np.mean([
            np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-8)
            for orig, recon in zip(original, reconstructed)
        ])
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'cosine_similarity': cosine_sim
        }
    
    @staticmethod
    def latent_space_quality(
        z_mean: np.ndarray, 
        z_logvar: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate latent space quality.
        
        Args:
            z_mean: Latent means
            z_logvar: Latent log variances
            
        Returns:
            Dictionary with latent space metrics
        """
        # KL divergence components
        kl_div = -0.5 * np.sum(1 + z_logvar - z_mean**2 - np.exp(z_logvar))
        avg_kl_per_dim = kl_div / z_mean.shape[1]
        
        # Posterior collapse detection
        active_dims = np.sum(np.var(z_mean, axis=0) > 0.01)  # Dimensions with sufficient variance
        posterior_collapse_ratio = 1 - (active_dims / z_mean.shape[1])
        
        # Latent space utilization
        latent_utilization = np.mean(np.var(z_mean, axis=0))
        
        return {
            'kl_divergence': kl_div,
            'avg_kl_per_dimension': avg_kl_per_dim,
            'active_dimensions': active_dims,
            'posterior_collapse_ratio': posterior_collapse_ratio,
            'latent_utilization': latent_utilization
        }

def compute_comprehensive_metrics(
    z_latent: np.ndarray,
    selected_indices: List[int],
    original_data: Optional[np.ndarray] = None,
    cluster_labels: Optional[np.ndarray] = None,
    coverage_radius: float = 0.2
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        z_latent: Latent space coordinates
        selected_indices: Selected representative indices
        original_data: Original data (optional)
        cluster_labels: Cluster labels (optional)
        coverage_radius: Coverage radius for metrics
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Sampling quality metrics
    sampling_metrics = SamplingQualityMetrics()
    metrics['coverage'] = sampling_metrics.coverage_score(z_latent, selected_indices, coverage_radius)
    metrics['diversity'] = sampling_metrics.diversity_score(selected_indices, z_latent)
    metrics['representativeness'] = sampling_metrics.representativeness_score(z_latent, selected_indices)
    
    # Clustering quality (if cluster labels provided)
    if cluster_labels is not None:
        metrics['clustering'] = sampling_metrics.clustering_quality_score(z_latent, cluster_labels)
    
    # Distribution metrics (if original data provided)
    if original_data is not None and selected_indices:
        sampled_data = original_data[selected_indices]
        dist_metrics = DistributionMetrics()
        metrics['statistical_distances'] = dist_metrics.statistical_distance_metrics(original_data, sampled_data)
        metrics['moment_comparison'] = dist_metrics.moment_comparison(original_data, sampled_data)
    
    return metrics