"""
Cluster-based representative sampling implementation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult, SamplingUtils

logger = logging.getLogger(__name__)

class ClusterBasedSampler(BaseSampler):
    """
    Cluster-based representative sampling method.
    
    First clusters the latent space, then selects representatives from each cluster
    based on information-theoretic measures.
    """
    
    def __init__(
        self,
        cluster_method: str = 'kmeans',
        cluster_sizing_method: str = 'adaptive',
        within_cluster_method: str = 'centroid_distance',
        min_clusters: int = 2,
        max_clusters: int = 500,
        n_clusters_factor: float = 0.1,
        info_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        **kwargs
    ):
        """
        Initialize cluster-based sampler.
        
        Args:
            cluster_method: Clustering algorithm ('kmeans' or 'dbscan')
            cluster_sizing_method: Method for determining number of clusters
            within_cluster_method: Method for selecting within clusters
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
            n_clusters_factor: Factor for determining clusters (for 'fixed' sizing)
            info_weight: Weight for information gain
            redundancy_weight: Weight for redundancy penalty
        """
        super().__init__('cluster_based', **kwargs)
        
        self.cluster_method = cluster_method
        self.cluster_sizing_method = cluster_sizing_method
        self.within_cluster_method = within_cluster_method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_clusters_factor = n_clusters_factor
        self.info_weight = info_weight
        self.redundancy_weight = redundancy_weight
        
        self.scaler = StandardScaler()
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        one_per_cluster: bool = False,
        **kwargs
    ) -> SamplingResult:
        """
        Sample representatives using cluster-based approach.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            one_per_cluster: If True, sets n_clusters = sample_size (one sample per cluster)
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting cluster-based sampling for {sample_size} samples")
        
        n_samples = len(z_latent)
        
        # Standardize latent coordinates
        z_scaled = self.scaler.fit_transform(z_latent)
        
        # Determine number of clusters
        if one_per_cluster:
            n_clusters = min(sample_size, n_samples)
            logger.info(f"One-per-cluster mode: using {n_clusters} clusters for {sample_size} samples")
        else:
            n_clusters = SamplingUtils.calculate_adaptive_clusters(
                n_samples=n_samples,
                sample_size=sample_size,
                method=self.cluster_sizing_method,
                min_clusters=self.min_clusters,
                max_clusters=self.max_clusters,
                n_clusters_factor=self.n_clusters_factor
            )
            # Ensure reasonable bounds
            n_clusters = min(n_clusters, sample_size, n_samples)
        
        logger.info(f"Using {n_clusters} clusters for {n_samples} points → {sample_size} samples")
        logger.info(f"Cluster sizing method: {self.cluster_sizing_method}")
        
        # Perform clustering
        cluster_labels = self._perform_clustering(z_scaled, n_clusters)
        
        # Handle clustering results
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:  # Remove noise points for DBSCAN
            unique_labels.remove(-1)
        
        if len(unique_labels) == 0:
            logger.warning("No clusters found, using fallback selection")
            return self._fallback_selection(z_latent, sample_size, original_df)
        
        # Allocate representatives to clusters
        cluster_sizes = self._calculate_cluster_sizes(cluster_labels, unique_labels)
        
        if one_per_cluster:
            # One representative per cluster
            representatives_per_cluster = {label: 1 for label in unique_labels}
            logger.info("One-per-cluster mode: each cluster gets exactly 1 representative")
        else:
            representatives_per_cluster = self._allocate_representatives(
                cluster_sizes, sample_size
            )
        
        # Select representatives from each cluster
        selected_indices = self._select_from_clusters(
            z_latent, cluster_labels, representatives_per_cluster, original_df
        )
        
        # Fill remaining spots if needed
        if len(selected_indices) < sample_size:
            selected_indices = self._fill_remaining_spots(
                z_latent, selected_indices, sample_size
            )
        
        # Ensure we don't exceed requested size
        selected_indices = selected_indices[:sample_size]
        
        # Create cluster info
        cluster_info = self._create_cluster_info(
            cluster_labels, representatives_per_cluster, selected_indices
        )
        
        # Additional method info
        additional_info = {
            'n_clusters': len(unique_labels),
            'cluster_sizing_method': self.cluster_sizing_method,
            'data_to_sample_ratio': n_samples / sample_size,
            'cluster_info': cluster_info,
            'cluster_method': self.cluster_method,
            'within_cluster_method': self.within_cluster_method,
            'cluster_labels': cluster_labels.tolist(),
            'one_per_cluster': one_per_cluster
        }
        
        logger.info(f"Cluster-based sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _perform_clustering(self, z_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform clustering on scaled latent coordinates."""
        if self.cluster_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif self.cluster_method == 'dbscan':
            # For DBSCAN, we need to estimate appropriate eps
            from sklearn.neighbors import NearestNeighbors
            k = min(10, len(z_scaled) // 10)  # Adaptive k
            if k > 0:
                nbrs = NearestNeighbors(n_neighbors=k).fit(z_scaled)
                distances, _ = nbrs.kneighbors(z_scaled)
                eps = np.mean(distances[:, -1])  # Use k-th nearest neighbor distance
            else:
                eps = 0.5  # Default
            
            clusterer = DBSCAN(eps=eps, min_samples=max(3, n_clusters // 10))
        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")
        
        return clusterer.fit_predict(z_scaled)
    
    def _calculate_cluster_sizes(
        self, 
        cluster_labels: np.ndarray, 
        unique_labels: set
    ) -> Dict[int, int]:
        """Calculate size of each cluster."""
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[label] = np.sum(cluster_labels == label)
        return cluster_sizes
    
    def _allocate_representatives(
        self, 
        cluster_sizes: Dict[int, int], 
        total_representatives: int
    ) -> Dict[int, int]:
        """Allocate representatives to clusters proportionally."""
        total_points = sum(cluster_sizes.values())
        allocation = {}
        
        # First pass: proportional allocation
        allocated = 0
        for cluster_id, size in cluster_sizes.items():
            proportion = size / total_points
            n_reps = max(1, int(proportion * total_representatives))
            allocation[cluster_id] = n_reps
            allocated += n_reps
        
        # Second pass: adjust for over/under allocation
        remaining = total_representatives - allocated
        
        if remaining > 0:
            # Add to largest clusters
            sorted_clusters = sorted(
                cluster_sizes.items(), key=lambda x: x[1], reverse=True
            )
            for cluster_id, _ in sorted_clusters[:remaining]:
                allocation[cluster_id] += 1
        elif remaining < 0:
            # Remove from smallest clusters (but keep at least 1)
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1])
            to_remove = abs(remaining)
            for cluster_id, _ in sorted_clusters:
                if to_remove == 0:
                    break
                if allocation[cluster_id] > 1:
                    allocation[cluster_id] -= 1
                    to_remove -= 1
        
        return allocation
    
    def _select_from_clusters(
        self,
        z_latent: np.ndarray,
        cluster_labels: np.ndarray,
        representatives_per_cluster: Dict[int, int],
        original_df: pd.DataFrame
    ) -> List[int]:
        """Select representatives from each cluster."""
        selected_indices = []
        
        for cluster_id, n_reps in representatives_per_cluster.items():
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_points = z_latent[cluster_mask]
            
            if len(cluster_indices) < n_reps:
                # Take all points if cluster is too small
                cluster_selected = cluster_indices.tolist()
            else:
                # Select representatives within cluster
                cluster_selected = self._select_within_cluster(
                    cluster_points, cluster_indices, n_reps
                )
            
            selected_indices.extend(cluster_selected)
        
        return selected_indices
    
    def _select_within_cluster(
        self,
        cluster_points: np.ndarray,
        cluster_indices: np.ndarray,
        n_representatives: int
    ) -> List[int]:
        """Select representatives within a single cluster."""
        if self.within_cluster_method == 'centroid_distance':
            return self._select_by_centroid_distance(
                cluster_points, cluster_indices, n_representatives
            )
        elif self.within_cluster_method == 'maxmin_distance':
            return self._select_by_maxmin_distance(
                cluster_points, cluster_indices, n_representatives
            )
        else:
            # Default to centroid distance
            return self._select_by_centroid_distance(
                cluster_points, cluster_indices, n_representatives
            )
    
    def _select_by_centroid_distance(
        self,
        cluster_points: np.ndarray,
        cluster_indices: np.ndarray,
        n_representatives: int
    ) -> List[int]:
        """Select points closest to cluster centroid."""
        centroid = np.mean(cluster_points, axis=0)
        distances = np.sqrt(np.sum((cluster_points - centroid) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:n_representatives]
        return cluster_indices[closest_indices].tolist()
    
    def _select_by_maxmin_distance(
        self,
        cluster_points: np.ndarray,
        cluster_indices: np.ndarray,
        n_representatives: int
    ) -> List[int]:
        """Select points using max-min distance strategy."""
        selected = []
        remaining = list(range(len(cluster_points)))
        
        # Start with point closest to centroid
        centroid = np.mean(cluster_points, axis=0)
        distances_to_centroid = np.sqrt(
            np.sum((cluster_points - centroid) ** 2, axis=1)
        )
        first_idx = np.argmin(distances_to_centroid)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Iteratively add points that maximize minimum distance
        for _ in range(n_representatives - 1):
            if not remaining:
                break
            
            best_idx = None
            best_min_distance = -1
            
            for idx in remaining:
                min_distance = min([
                    np.sqrt(np.sum((cluster_points[idx] - cluster_points[sel_idx]) ** 2))
                    for sel_idx in selected
                ])
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return cluster_indices[selected].tolist()
    
    def _fill_remaining_spots(
        self,
        z_latent: np.ndarray,
        selected_indices: List[int],
        target_size: int
    ) -> List[int]:
        """Fill remaining spots using distance-based selection."""
        remaining_needed = target_size - len(selected_indices)
        remaining_candidates = [
            i for i in range(len(z_latent)) if i not in selected_indices
        ]
        
        if remaining_needed <= 0 or not remaining_candidates:
            return selected_indices
        
        # Use greedy distance-based selection for remaining spots
        current_selected = selected_indices.copy()
        
        for _ in range(remaining_needed):
            if not remaining_candidates:
                break
            
            best_candidate = None
            best_score = -float('inf')
            
            for candidate_idx in remaining_candidates:
                # Score based on minimum distance to already selected points
                if current_selected:
                    distances = cdist(
                        [z_latent[candidate_idx]], 
                        z_latent[current_selected]
                    )[0]
                    score = np.min(distances)
                else:
                    score = 1.0
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                current_selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
        
        return current_selected
    
    def _fallback_selection(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame
    ) -> SamplingResult:
        """Fallback to random selection if clustering fails."""
        logger.warning("Using random fallback selection")
        selected_indices = np.random.choice(
            len(z_latent), 
            min(sample_size, len(z_latent)), 
            replace=False
        ).tolist()
        
        additional_info = {
            'method': 'cluster_based_fallback',
            'fallback_reason': 'clustering_failed'
        }
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _create_cluster_info(
        self,
        cluster_labels: np.ndarray,
        representatives_per_cluster: Dict[int, int],
        selected_indices: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """Create detailed cluster information."""
        cluster_info = {}
        
        for cluster_id, n_reps in representatives_per_cluster.items():
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Find which selected indices belong to this cluster
            cluster_selected = [
                idx for idx in selected_indices 
                if idx in cluster_indices
            ]
            
            cluster_info[cluster_id] = {
                'size': len(cluster_indices),
                'representatives_allocated': n_reps,
                'representatives_selected': len(cluster_selected),
                'selected_indices': cluster_selected
            }
        
        return cluster_info