"""
Optimized optimal transport-based sampling methods.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult

logger = logging.getLogger(__name__)

class OptimalTransportSampler(BaseSampler):
    """
    Optimized optimal transport subsampling to minimize transport cost.
    """
    
    def __init__(
        self,
        method: str = 'greedy',
        subsample_factor: float = 0.05,
        max_iter: int = 50,
        batch_size: int = 1000,
        use_knn_acceleration: bool = True,
        **kwargs
    ):
        """
        Initialize optimal transport sampler.
        
        Args:
            method: 'greedy', 'hungarian', or 'kmeans_init' algorithm
            subsample_factor: Factor for initial subsampling (reduced default for speed)
            max_iter: Maximum iterations for greedy method (reduced default for speed)
            batch_size: Batch size for distance computations (new parameter)
            use_knn_acceleration: Use k-NN for faster candidate selection (new parameter)
        """
        
        super().__init__('optimal_transport', **kwargs)
        
        self.method = method
        self.subsample_factor = subsample_factor
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.use_knn_acceleration = use_knn_acceleration
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """Sample using optimized optimal transport."""
        logger.info(f"Starting optimal transport sampling ({self.method}) for {sample_size} samples")
        
        if self.method == 'greedy':
            selected_indices = self._greedy_optimal_transport(z_latent, sample_size)
        elif self.method == 'kmeans_init':
            selected_indices = self._kmeans_initialized_selection(z_latent, sample_size)
        elif self.method == 'hungarian':
            selected_indices = self._hungarian_optimal_transport(z_latent, sample_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        additional_info = {
            'method': self.method,
            'subsample_factor': self.subsample_factor,
            'batch_size': self.batch_size,
            'use_knn_acceleration': self.use_knn_acceleration
        }
        
        logger.info(f"Optimal transport sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _greedy_optimal_transport(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """Fast greedy selection with caching and k-NN acceleration."""
        n_points = len(z_latent)
        
        # Pre-compute k-NN for acceleration
        if self.use_knn_acceleration and n_points > 1000:
            k = min(100, n_points // 10)
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(z_latent)
            logger.info(f"Using k-NN acceleration with k={k}")
        else:
            nbrs = None
        
        # Subsample candidates if dataset is too large
        if n_points > 10000:
            n_candidates = max(sample_size * 20, int(n_points * self.subsample_factor))
            candidate_indices = np.random.choice(n_points, n_candidates, replace=False)
            candidate_points = z_latent[candidate_indices]
            work_with_subset = True
        else:
            candidate_indices = np.arange(n_points)
            candidate_points = z_latent
            work_with_subset = False
        
        selected_indices = []
        
        # Initialize with point that's farthest from center (good heuristic)
        center = np.mean(candidate_points, axis=0)
        distances_to_center = np.sum((candidate_points - center) ** 2, axis=1)
        first_idx = candidate_indices[np.argmax(distances_to_center)]
        selected_indices.append(first_idx)
        
        # Pre-allocate distance matrix for efficiency
        if work_with_subset:
            distance_matrix = np.full((len(candidate_points), sample_size), np.inf)
            # Compute distances to first selected point
            distances = np.sum((candidate_points - z_latent[first_idx]) ** 2, axis=1)
            distance_matrix[:, 0] = distances
        
        # Greedy selection with optimizations
        for i in range(1, sample_size):
            best_candidate_local_idx = None
            best_cost = float('inf')
            
            # Use k-NN to limit candidates if enabled
            if nbrs is not None and len(selected_indices) > 0:
                # Find candidates near existing representatives
                last_selected = z_latent[selected_indices[-1]]
                _, neighbor_indices = nbrs.kneighbors([last_selected])
                candidate_pool = set(neighbor_indices[0])
                
                # Add some random candidates for diversity
                n_random = min(50, n_points // 20)
                random_candidates = np.random.choice(candidate_indices, n_random, replace=False)
                candidate_pool.update(random_candidates)
                
                # Convert to list and filter already selected
                evaluation_candidates = [
                    idx for idx in candidate_pool 
                    if idx not in selected_indices
                ][:200]  # Limit evaluation set
            else:
                evaluation_candidates = [
                    idx for j, idx in enumerate(candidate_indices) 
                    if idx not in selected_indices
                ]
            
            if not evaluation_candidates:
                break
            
            # Evaluate candidates in batches
            for batch_start in range(0, len(evaluation_candidates), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(evaluation_candidates))
                batch_candidates = evaluation_candidates[batch_start:batch_end]
                
                # Compute costs for batch
                batch_costs = self._compute_batch_transport_costs(
                    z_latent, selected_indices, batch_candidates
                )
                
                # Find best in batch
                best_in_batch_idx = np.argmin(batch_costs)
                if batch_costs[best_in_batch_idx] < best_cost:
                    best_cost = batch_costs[best_in_batch_idx]
                    best_candidate_local_idx = batch_candidates[best_in_batch_idx]
            
            if best_candidate_local_idx is not None:
                selected_indices.append(best_candidate_local_idx)
                
                if work_with_subset:
                    # Update distance matrix
                    new_point = z_latent[best_candidate_local_idx]
                    distances = np.sum((candidate_points - new_point) ** 2, axis=1)
                    distance_matrix[:, i] = distances
        
        return selected_indices
    
    def _compute_transport_cost(self, full_points: np.ndarray, sample_points: np.ndarray) -> float:
        """Compute transport cost from full dataset to sample (optimized version)."""
        # For each point in full dataset, find distance to nearest sample point
        # Use squared distances for speed (avoids sqrt computation)
        distances_squared = np.sum(
            (full_points[:, np.newaxis, :] - sample_points[np.newaxis, :, :]) ** 2, 
            axis=2
        )
        min_distances_squared = np.min(distances_squared, axis=1)
        
        # Transport cost is sum of minimum squared distances
        return np.sum(min_distances_squared)
    
    def _compute_batch_transport_costs(
        self, 
        z_latent: np.ndarray, 
        selected_indices: List[int], 
        candidate_indices: List[int]
    ) -> np.ndarray:
        """Compute transport costs for a batch of candidates efficiently."""
        if not selected_indices:
            # For first point, return negative variance as heuristic
            candidates_points = z_latent[candidate_indices]
            variances = np.var(candidates_points, axis=1)
            return -variances
        
        selected_points = z_latent[selected_indices]
        candidate_points = z_latent[candidate_indices]
        
        costs = np.zeros(len(candidate_indices))
        
        for i, candidate_point in enumerate(candidate_points):
            # Compute distance from candidate to all selected points
            candidate_to_selected = np.sum((selected_points - candidate_point) ** 2, axis=1)
            
            # Simple heuristic: minimize sum of squared distances to existing representatives
            # (approximates transport cost without full computation)
            costs[i] = np.sum(candidate_to_selected)
        
        return costs
    
    def _kmeans_initialized_selection(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """Use k-means centroids as initialization for faster convergence."""
        from sklearn.cluster import KMeans
        
        logger.info("Using k-means initialization for optimal transport sampling")
        
        # Use k-means to find initial representatives
        n_init_clusters = min(sample_size * 2, len(z_latent) // 10)
        kmeans = KMeans(n_clusters=n_init_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(z_latent)
        
        # Find closest real points to centroids
        selected_indices = []
        centroids = kmeans.cluster_centers_
        
        for centroid in centroids:
            distances = np.sum((z_latent - centroid) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
            
            if len(selected_indices) >= sample_size:
                break
        
        # If we need more points, use fast greedy to fill the rest
        if len(selected_indices) < sample_size:
            remaining_needed = sample_size - len(selected_indices)
            remaining_candidates = [i for i in range(len(z_latent)) if i not in selected_indices]
            
            # Simple farthest-first for remaining points
            for _ in range(remaining_needed):
                if not remaining_candidates:
                    break
                
                max_min_distance = -1
                best_candidate = None
                
                for candidate in remaining_candidates:
                    candidate_point = z_latent[candidate]
                    min_distance = float('inf')
                    
                    for selected_idx in selected_indices:
                        distance = np.sum((candidate_point - z_latent[selected_idx]) ** 2)
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected_indices.append(best_candidate)
                    remaining_candidates.remove(best_candidate)
        
        return selected_indices[:sample_size]
    
    def _hungarian_optimal_transport(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """Optimized Hungarian algorithm with smart subsampling."""
        n_points = len(z_latent)
        
        # More aggressive subsampling for Hungarian
        if n_points > 2000:
            subsample_size = min(2000, max(sample_size * 10, int(n_points * 0.3)))
            # Use stratified sampling based on k-means clusters for better coverage
            from sklearn.cluster import KMeans
            n_clusters = min(subsample_size // 10, 50)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(z_latent)
            
            # Sample proportionally from each cluster
            subsample_indices = []
            for cluster_id in range(n_clusters):
                cluster_points = np.where(cluster_labels == cluster_id)[0]
                if len(cluster_points) > 0:
                    n_from_cluster = max(1, len(cluster_points) * subsample_size // n_points)
                    selected = np.random.choice(cluster_points, 
                                              min(n_from_cluster, len(cluster_points)), 
                                              replace=False)
                    subsample_indices.extend(selected)
            
            subsample_indices = np.array(subsample_indices[:subsample_size])
            work_points = z_latent[subsample_indices]
        else:
            subsample_indices = np.arange(n_points)
            work_points = z_latent
        
        if sample_size >= len(work_points):
            return subsample_indices.tolist()
        
        # Use k-means++ initialization instead of random
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=sample_size, init='k-means++', n_init=1, random_state=42)
        kmeans.fit(work_points)
        
        # Find closest real points to final centroids
        selected_local = []
        for centroid in kmeans.cluster_centers_:
            distances = np.sum((work_points - centroid) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            if closest_idx not in selected_local:
                selected_local.append(closest_idx)
        
        # Convert back to original indices
        selected_indices = subsample_indices[selected_local].tolist()
        
        return selected_indices


class SlicedWassersteinSampler(BaseSampler):
    """
    Optimized sliced Wasserstein barycenter sampling.
    """
    
    def __init__(
        self,
        n_projections: int = 20,  # Reduced from 50
        max_iter: int = 5,        # Reduced from 10
        n_candidates: int = 50,   # Limit candidate evaluations
        convergence_thresh: float = 1e-3,
        **kwargs
    ):
        super().__init__('sliced_wasserstein', **kwargs)
        self.n_projections = n_projections
        self.max_iter = max_iter
        self.n_candidates = n_candidates
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """Optimized sliced Wasserstein sampling."""
        logger.info(f"Starting sliced Wasserstein sampling for {sample_size} samples")
        
        selected_indices = self._sliced_wasserstein_selection(z_latent, sample_size)
        
        additional_info = {
            'n_projections': self.n_projections,
            'max_iter': self.max_iter,
            'n_candidates': self.n_candidates
        }
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _sliced_wasserstein_selection(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """Fast sliced Wasserstein selection with approximations."""
        n_points, n_dims = z_latent.shape
        
        # Reduced number of projections for speed
        np.random.seed(42)
        projections = np.random.randn(self.n_projections, n_dims)
        projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)
        
        # Project data once
        projected_data = np.dot(z_latent, projections.T)
        
        # Initialize with k-means++ style selection for better starting point
        selected_indices = self._kmeans_plus_plus_init(z_latent, sample_size)
        
        # Fast approximation of sliced Wasserstein cost
        best_cost = self._compute_sliced_wasserstein_cost(projected_data, selected_indices)
        
        # Limited improvement iterations
        for iteration in range(self.max_iter):
            improved = False
            
            for i in range(len(selected_indices)):
                current_idx = selected_indices[i]
                
                # Limited candidate evaluation
                unselected = [j for j in range(n_points) if j not in selected_indices]
                candidates = np.random.choice(unselected, 
                                            min(self.n_candidates, len(unselected)), 
                                            replace=False)
                
                for candidate_idx in candidates:
                    trial_selection = selected_indices.copy()
                    trial_selection[i] = candidate_idx
                    
                    trial_cost = self._compute_sliced_wasserstein_cost(projected_data, trial_selection)
                    
                    if trial_cost < best_cost:
                        best_cost = trial_cost
                        selected_indices[i] = candidate_idx
                        improved = True
                        break  # Accept first improvement
                
                if improved:
                    break  # Move to next iteration after any improvement
            
            if not improved:
                break
        
        return selected_indices
    
    def _kmeans_plus_plus_init(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """K-means++ style initialization."""
        selected = []
        n_points = len(z_latent)
        
        # First point randomly
        selected.append(np.random.randint(n_points))
        
        for _ in range(1, sample_size):
            distances = np.full(n_points, np.inf)
            
            for i in range(n_points):
                if i not in selected:
                    min_dist = min([np.sum((z_latent[i] - z_latent[s]) ** 2) 
                                   for s in selected])
                    distances[i] = min_dist
            
            # Weighted random selection
            distances[selected] = 0  # Don't reselect
            if np.sum(distances) > 0:
                probabilities = distances / np.sum(distances)
                next_point = np.random.choice(n_points, p=probabilities)
                selected.append(next_point)
        
        return selected
    
    def _compute_sliced_wasserstein_cost(self, projected_data: np.ndarray, selected_indices: List[int]) -> float:
        """Fast approximation of sliced Wasserstein cost."""
        # Use quantile-based approximation instead of full Wasserstein distance
        total_cost = 0.0
        sample_data = projected_data[selected_indices]
        
        for proj_idx in range(projected_data.shape[1]):
            full_proj = np.sort(projected_data[:, proj_idx])
            sample_proj = np.sort(sample_data[:, proj_idx])
            
            # Quantile-based approximation (much faster than full Wasserstein)
            quantiles = np.linspace(0, 1, min(len(sample_proj), 20))
            full_quantiles = np.quantile(full_proj, quantiles)
            sample_quantiles = np.quantile(sample_proj, quantiles)
            
            cost = np.mean(np.abs(full_quantiles - sample_quantiles))
            total_cost += cost
        
        return total_cost / projected_data.shape[1]