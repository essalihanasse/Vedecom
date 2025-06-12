"""
Baseline sampling methods for comparison with advanced VAE sampling techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult, SamplingUtils

logger = logging.getLogger(__name__)

class RandomSampler(BaseSampler):
    """
    Simple uniform random sampling baseline.
    
    The fundamental baseline that all other methods should beat.
    Provides unbiased coverage with O(n) complexity.
    """
    
    def __init__(self, random_state: Optional[int] = 42, **kwargs):
        """
        Initialize random sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__('random', **kwargs)
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
        Sample points randomly from the latent space.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting random sampling for {sample_size} samples")
        
        n_samples = len(z_latent)
        
        if sample_size >= n_samples:
            logger.warning(f"Sample size ({sample_size}) >= dataset size ({n_samples}), returning all points")
            selected_indices = list(range(n_samples))
        else:
            # Simple random sampling without replacement
            selected_indices = np.random.choice(
                n_samples, 
                size=sample_size, 
                replace=False
            ).tolist()
        
        # Calculate coverage statistics
        coverage_stats = SamplingUtils.calculate_coverage_statistics(
            z_latent, selected_indices
        )
        
        # Additional method info
        additional_info = {
            'random_state': self.random_state,
            'coverage_statistics': coverage_stats,
            'theoretical_efficiency': 1.0,  # Random sampling baseline
            'selection_bias': 0.0  # Unbiased by design
        }
        
        logger.info(f"Random sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Coverage ratio: {coverage_stats.get('coverage_fraction', 0):.3f}")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )


class StratifiedSampler(BaseSampler):
    """
    Density-based stratified sampling baseline.
    
    Divides latent space into strata based on density estimation
    and samples proportionally from each stratum.
    """
    
    def __init__(
        self,
        n_strata: Optional[int] = None,
        stratification_method: str = 'density',
        bandwidth: str = 'scott',
        random_state: Optional[int] = 42,
        **kwargs
    ):
        """
        Initialize stratified sampler.
        
        Args:
            n_strata: Number of strata (auto if None)
            stratification_method: Method for creating strata ('density', 'grid', 'kmeans')
            bandwidth: KDE bandwidth for density estimation
            random_state: Random seed for reproducibility
        """
        super().__init__('stratified', **kwargs)
        
        self.n_strata = n_strata
        self.stratification_method = stratification_method
        self.bandwidth = bandwidth
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
        Sample using stratified approach.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting stratified sampling ({self.stratification_method}) for {sample_size} samples")
        
        n_samples = len(z_latent)
        
        # Determine number of strata
        if self.n_strata is None:
            self.n_strata = min(max(5, int(np.sqrt(sample_size))), sample_size // 2)
        
        logger.info(f"Using {self.n_strata} strata")
        
        # Create strata
        if self.stratification_method == 'density':
            strata_assignments = self._create_density_strata(z_latent)
        elif self.stratification_method == 'kmeans':
            strata_assignments = self._create_kmeans_strata(z_latent)
        elif self.stratification_method == 'grid':
            strata_assignments = self._create_grid_strata(z_latent)
        else:
            raise ValueError(f"Unknown stratification method: {self.stratification_method}")
        
        # Sample from each stratum
        selected_indices = self._sample_from_strata(strata_assignments, sample_size)
        
        # Calculate stratification quality
        strata_info = self._calculate_strata_info(strata_assignments, selected_indices)
        
        # Additional method info
        additional_info = {
            'n_strata': self.n_strata,
            'stratification_method': self.stratification_method,
            'bandwidth': self.bandwidth,
            'strata_info': strata_info,
            'strata_assignments': strata_assignments.tolist()
        }
        
        logger.info(f"Stratified sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _create_density_strata(self, z_latent: np.ndarray) -> np.ndarray:
        """Create strata based on density estimation."""
        # Estimate density using KDE
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(z_latent)
        
        log_densities = kde.score_samples(z_latent)
        densities = np.exp(log_densities)
        
        # Create strata based on density quantiles
        density_quantiles = np.linspace(0, 100, self.n_strata + 1)
        density_thresholds = np.percentile(densities, density_quantiles)
        
        strata_assignments = np.digitize(densities, density_thresholds) - 1
        strata_assignments = np.clip(strata_assignments, 0, self.n_strata - 1)
        
        return strata_assignments
    
    def _create_kmeans_strata(self, z_latent: np.ndarray) -> np.ndarray:
        """Create strata using k-means clustering."""
        kmeans = KMeans(n_clusters=self.n_strata, random_state=self.random_state, n_init=10)
        strata_assignments = kmeans.fit_predict(z_latent)
        return strata_assignments
    
    def _create_grid_strata(self, z_latent: np.ndarray) -> np.ndarray:
        """Create strata using regular grid (works best for 2D)."""
        n_dims = z_latent.shape[1]
        
        if n_dims == 2:
            # Calculate grid dimensions
            grid_size = int(np.ceil(np.sqrt(self.n_strata)))
            
            # Create grid boundaries
            x_edges = np.linspace(z_latent[:, 0].min(), z_latent[:, 0].max(), grid_size + 1)
            y_edges = np.linspace(z_latent[:, 1].min(), z_latent[:, 1].max(), grid_size + 1)
            
            # Assign points to grid cells
            x_indices = np.digitize(z_latent[:, 0], x_edges) - 1
            y_indices = np.digitize(z_latent[:, 1], y_edges) - 1
            
            x_indices = np.clip(x_indices, 0, grid_size - 1)
            y_indices = np.clip(y_indices, 0, grid_size - 1)
            
            strata_assignments = y_indices * grid_size + x_indices
        else:
            # Fall back to k-means for higher dimensions
            logger.warning("Grid stratification only supports 2D, falling back to k-means")
            strata_assignments = self._create_kmeans_strata(z_latent)
        
        return strata_assignments
    
    def _sample_from_strata(self, strata_assignments: np.ndarray, sample_size: int) -> List[int]:
        """Sample proportionally from each stratum."""
        selected_indices = []
        
        # Calculate stratum sizes
        unique_strata, stratum_counts = np.unique(strata_assignments, return_counts=True)
        total_points = len(strata_assignments)
        
        # Allocate samples to strata proportionally
        allocated_samples = 0
        for stratum_id, stratum_count in zip(unique_strata, stratum_counts):
            # Proportional allocation
            stratum_proportion = stratum_count / total_points
            stratum_samples = max(1, int(stratum_proportion * sample_size))
            
            # Get points in this stratum
            stratum_indices = np.where(strata_assignments == stratum_id)[0]
            
            # Sample from stratum
            if stratum_samples >= len(stratum_indices):
                # Take all points if stratum is too small
                selected_from_stratum = stratum_indices.tolist()
            else:
                selected_from_stratum = np.random.choice(
                    stratum_indices, 
                    size=stratum_samples, 
                    replace=False
                ).tolist()
            
            selected_indices.extend(selected_from_stratum)
            allocated_samples += len(selected_from_stratum)
            
            # Stop if we've allocated enough samples
            if allocated_samples >= sample_size:
                break
        
        # Trim to exact sample size if needed
        if len(selected_indices) > sample_size:
            selected_indices = selected_indices[:sample_size]
        
        return selected_indices
    
    def _calculate_strata_info(self, strata_assignments: np.ndarray, selected_indices: List[int]) -> Dict[str, Any]:
        """Calculate stratification quality metrics."""
        unique_strata, stratum_counts = np.unique(strata_assignments, return_counts=True)
        
        # Count selected points per stratum
        selected_strata = strata_assignments[selected_indices]
        selected_counts = np.bincount(selected_strata, minlength=len(unique_strata))
        
        strata_info = {
            'n_strata_actual': len(unique_strata),
            'stratum_sizes': stratum_counts.tolist(),
            'selected_per_stratum': selected_counts.tolist(),
            'sampling_rates': (selected_counts / stratum_counts).tolist(),
            'empty_strata': np.sum(stratum_counts == 0),
            'unsampled_strata': np.sum(selected_counts == 0)
        }
        
        return strata_info


class UncertaintySampler(BaseSampler):
    """
    Uncertainty-based sampling using VAE reconstruction uncertainty.
    
    Selects points where the VAE has high reconstruction uncertainty,
    indicating potentially informative or difficult-to-represent samples.
    """
    
    def __init__(
        self,
        uncertainty_method: str = 'reconstruction_error',
        batch_size: int = 512,
        **kwargs
    ):
        """
        Initialize uncertainty sampler.
        
        Args:
            uncertainty_method: Method for calculating uncertainty
            batch_size: Batch size for VAE inference
        """
        super().__init__('uncertainty', **kwargs)
        
        self.uncertainty_method = uncertainty_method
        self.batch_size = batch_size
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        model=None,  # VAE model for uncertainty calculation
        preprocessed_data=None,  # Preprocessed data tensor
        **kwargs
    ) -> SamplingResult:
        """
        Sample based on reconstruction uncertainty.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            model: Trained VAE model (optional)
            preprocessed_data: Preprocessed data tensor (optional)
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting uncertainty sampling ({self.uncertainty_method}) for {sample_size} samples")
        
        if model is None or preprocessed_data is None:
            logger.warning("No model or data provided, falling back to latent space variance")
            uncertainty_scores = self._calculate_latent_uncertainty(z_latent)
        else:
            uncertainty_scores = self._calculate_model_uncertainty(model, preprocessed_data)
        
        # Select samples with highest uncertainty
        uncertainty_order = np.argsort(uncertainty_scores)[::-1]  # Descending order
        selected_indices = uncertainty_order[:sample_size].tolist()
        
        # Calculate uncertainty statistics
        uncertainty_stats = {
            'mean_uncertainty': float(np.mean(uncertainty_scores)),
            'std_uncertainty': float(np.std(uncertainty_scores)),
            'selected_mean_uncertainty': float(np.mean(uncertainty_scores[selected_indices])),
            'uncertainty_percentile': float(np.percentile(uncertainty_scores, 90)),
            'method': self.uncertainty_method
        }
        
        # Additional method info
        additional_info = {
            'uncertainty_method': self.uncertainty_method,
            'uncertainty_statistics': uncertainty_stats,
            'uncertainty_scores': uncertainty_scores.tolist()
        }
        
        logger.info(f"Uncertainty sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Mean uncertainty (selected): {uncertainty_stats['selected_mean_uncertainty']:.4f}")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _calculate_latent_uncertainty(self, z_latent: np.ndarray) -> np.ndarray:
        """Calculate uncertainty based on latent space properties."""
        if self.uncertainty_method == 'distance_to_center':
            # Distance from latent space center
            center = np.mean(z_latent, axis=0)
            uncertainty_scores = np.sqrt(np.sum((z_latent - center) ** 2, axis=1))
        
        elif self.uncertainty_method == 'local_density':
            # Inverse of local density (low density = high uncertainty)
            kde = KernelDensity(bandwidth='scott')
            kde.fit(z_latent)
            log_densities = kde.score_samples(z_latent)
            uncertainty_scores = -log_densities  # Higher uncertainty for lower density
        
        elif self.uncertainty_method == 'nearest_neighbor_distance':
            # Distance to nearest neighbor
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=2)  # k=2 to exclude self
            nbrs.fit(z_latent)
            distances, _ = nbrs.kneighbors(z_latent)
            uncertainty_scores = distances[:, 1]  # Distance to nearest neighbor
        
        else:
            # Default: variance along each dimension
            uncertainty_scores = np.var(z_latent, axis=1)
        
        return uncertainty_scores
    
    def _calculate_model_uncertainty(self, model, preprocessed_data) -> np.ndarray:
        """Calculate uncertainty using the VAE model."""
        import torch
        
        model.eval()
        uncertainty_scores = []
        
        with torch.no_grad():
            for i in range(0, len(preprocessed_data), self.batch_size):
                batch = preprocessed_data[i:i + self.batch_size]
                
                if self.uncertainty_method == 'reconstruction_error':
                    # Reconstruction error
                    reconstructed, _, _ = model(batch)
                    batch_errors = torch.mean((batch - reconstructed) ** 2, dim=1)
                    uncertainty_scores.extend(batch_errors.cpu().numpy())
                
                elif self.uncertainty_method == 'kl_divergence':
                    # KL divergence of latent distribution
                    _, mu, logvar = model(batch)
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    uncertainty_scores.extend(kl_div.cpu().numpy())
                
                elif self.uncertainty_method == 'total_loss':
                    # Total VAE loss
                    reconstructed, mu, logvar = model(batch)
                    recon_loss = torch.mean((batch - reconstructed) ** 2, dim=1)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                    total_loss = recon_loss + kl_loss
                    uncertainty_scores.extend(total_loss.cpu().numpy())
                
                else:
                    raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
        
        return np.array(uncertainty_scores)


class KMeansPlusPlusSampler(BaseSampler):
    """
    K-means++ initialization-based sampling for diversity.
    
    Uses the k-means++ seeding algorithm to select diverse representatives
    that are well-spread throughout the latent space.
    """
    
    def __init__(self, random_state: Optional[int] = 42, **kwargs):
        """
        Initialize k-means++ sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__('kmeans_plus_plus', **kwargs)
        
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
        Sample using k-means++ initialization strategy.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting k-means++ sampling for {sample_size} samples")
        
        n_samples = len(z_latent)
        
        if sample_size >= n_samples:
            selected_indices = list(range(n_samples))
        else:
            selected_indices = self._kmeans_plus_plus_init(z_latent, sample_size)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(z_latent, selected_indices)
        
        # Additional method info
        additional_info = {
            'random_state': self.random_state,
            'diversity_metrics': diversity_metrics,
            'selection_method': 'kmeans_plus_plus_initialization'
        }
        
        logger.info(f"K-means++ sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Min pairwise distance: {diversity_metrics.get('min_pairwise_distance', 0):.4f}")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _kmeans_plus_plus_init(self, z_latent: np.ndarray, k: int) -> List[int]:
        """
        Implement k-means++ initialization algorithm.
        
        Args:
            z_latent: Data points
            k: Number of centers to select
            
        Returns:
            List of selected indices
        """
        n_samples, n_features = z_latent.shape
        selected_indices = []
        
        # Step 1: Choose first center randomly
        first_center = np.random.randint(n_samples)
        selected_indices.append(first_center)
        
        # Step 2: Choose remaining centers
        for _ in range(k - 1):
            # Calculate distances to nearest selected center
            distances = np.full(n_samples, np.inf)
            
            for i in range(n_samples):
                if i in selected_indices:
                    distances[i] = 0
                    continue
                
                # Distance to nearest selected center
                point_distances = [
                    np.sum((z_latent[i] - z_latent[j]) ** 2)
                    for j in selected_indices
                ]
                distances[i] = min(point_distances)
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / np.sum(distances)
            
            # Handle numerical issues
            probabilities = np.nan_to_num(probabilities)
            if np.sum(probabilities) == 0:
                probabilities = np.ones(n_samples) / n_samples
            else:
                probabilities = probabilities / np.sum(probabilities)
            
            next_center = np.random.choice(n_samples, p=probabilities)
            selected_indices.append(next_center)
        
        return selected_indices
    
    def _calculate_diversity_metrics(self, z_latent: np.ndarray, selected_indices: List[int]) -> Dict[str, float]:
        """Calculate diversity metrics for selected points."""
        if len(selected_indices) < 2:
            return {'min_pairwise_distance': 0.0, 'avg_pairwise_distance': 0.0}
        
        selected_points = z_latent[selected_indices]
        
        # Pairwise distances
        distances = cdist(selected_points, selected_points)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances
        
        min_distance = np.min(distances)
        avg_distance = np.mean(distances[distances != np.inf])
        
        # Coverage metrics
        coverage_stats = SamplingUtils.calculate_coverage_statistics(
            z_latent, selected_indices
        )
        
        diversity_metrics = {
            'min_pairwise_distance': float(min_distance),
            'avg_pairwise_distance': float(avg_distance),
            'coverage_fraction': coverage_stats.get('coverage_fraction', 0.0),
            'avg_distance_to_nearest_rep': coverage_stats.get('avg_distance_to_nearest_rep', 0.0)
        }
        
        return diversity_metrics


class GaussianPriorSampler(BaseSampler):
    """
    Gaussian prior sampling for VAE latent spaces.
    
    Samples from the standard normal distribution (VAE prior) and finds
    nearest neighbors in the actual latent space.
    """
    
    def __init__(self, random_state: Optional[int] = 42, **kwargs):
        """
        Initialize Gaussian prior sampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__('gaussian_prior', **kwargs)
        
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
        Sample by drawing from Gaussian prior and finding nearest neighbors.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting Gaussian prior sampling for {sample_size} samples")
        
        n_samples, n_dims = z_latent.shape
        
        if sample_size >= n_samples:
            selected_indices = list(range(n_samples))
        else:
            # Sample from standard normal (VAE prior)
            prior_samples = np.random.standard_normal((sample_size, n_dims))
            
            # Find nearest neighbors in actual latent space
            selected_indices = self._find_nearest_neighbors(z_latent, prior_samples)
        
        # Calculate prior alignment metrics
        prior_metrics = self._calculate_prior_alignment(z_latent, selected_indices)
        
        # Additional method info
        additional_info = {
            'random_state': self.random_state,
            'prior_alignment_metrics': prior_metrics,
            'latent_dimensions': n_dims
        }
        
        logger.info(f"Gaussian prior sampling completed: {len(selected_indices)} representatives")
        logger.info(f"Prior KL divergence: {prior_metrics.get('kl_divergence_estimate', 0):.4f}")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _find_nearest_neighbors(self, z_latent: np.ndarray, prior_samples: np.ndarray) -> List[int]:
        """Find nearest neighbors in latent space for each prior sample."""
        from sklearn.neighbors import NearestNeighbors
        
        # Fit nearest neighbors on latent space
        nbrs = NearestNeighbors(n_neighbors=1)
        nbrs.fit(z_latent)
        
        # Find nearest neighbors for prior samples
        _, indices = nbrs.kneighbors(prior_samples)
        
        # Remove duplicates while preserving order
        selected_indices = []
        seen = set()
        
        for idx_array in indices:
            idx = idx_array[0]
            if idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
        
        return selected_indices
    
    def _calculate_prior_alignment(self, z_latent: np.ndarray, selected_indices: List[int]) -> Dict[str, float]:
        """Calculate how well selected points align with Gaussian prior."""
        selected_points = z_latent[selected_indices]
        
        # Estimate KL divergence from standard normal
        # KL(q||p) where q is empirical distribution, p is N(0,I)
        mean_est = np.mean(selected_points, axis=0)
        cov_est = np.cov(selected_points.T)
        
        # Simple KL divergence estimate for multivariate normal
        n_dims = selected_points.shape[1]
        
        try:
            # KL(N(μ,Σ) || N(0,I)) = 0.5 * (tr(Σ) + μᵀμ - k - log|Σ|)
            trace_cov = np.trace(cov_est)
            mean_squared_norm = np.sum(mean_est ** 2)
            log_det_cov = np.linalg.slogdet(cov_est)[1]
            
            kl_divergence = 0.5 * (trace_cov + mean_squared_norm - n_dims - log_det_cov)
        except np.linalg.LinAlgError:
            kl_divergence = float('inf')
        
        # Distance from origin
        origin_distances = np.sqrt(np.sum(selected_points ** 2, axis=1))
        
        prior_metrics = {
            'kl_divergence_estimate': float(kl_divergence),
            'mean_distance_from_origin': float(np.mean(origin_distances)),
            'std_distance_from_origin': float(np.std(origin_distances)),
            'mean_norm': float(np.linalg.norm(mean_est)),
            'trace_covariance': float(trace_cov) if not np.isinf(kl_divergence) else float('inf')
        }
        
        return prior_metrics