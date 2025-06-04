"""
Density-aware sampling methods for better distribution preservation.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Any
import logging

from .base import BaseSampler, SamplingResult

logger = logging.getLogger(__name__)

class DensityAwareSampler(BaseSampler):
    """
    Density-aware sampling using KDE + stratified sampling.
    
    Estimates density and samples proportionally to preserve distribution shape.
    """
    
    def __init__(
        self,
        method: str = 'kde_stratified',
        bandwidth: str = 'scott',
        n_strata: Optional[int] = None,
        min_samples_per_stratum: int = 1,
        **kwargs
    ):
        """
        Initialize density-aware sampler.
        
        Args:
            method: Sampling method ('kde_stratified', 'importance')
            bandwidth: KDE bandwidth ('scott', 'silverman', or float)
            n_strata: Number of density strata (auto if None)
            min_samples_per_stratum: Minimum samples per stratum
        """
        super().__init__('density_aware', **kwargs)
        
        self.method = method
        self.bandwidth = bandwidth
        self.n_strata = n_strata
        self.min_samples_per_stratum = min_samples_per_stratum
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample using density-aware methods.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting density-aware sampling ({self.method}) for {sample_size} samples")
        
        if self.method == 'kde_stratified':
            selected_indices = self._kde_stratified_sampling(z_latent, sample_size)
        elif self.method == 'importance':
            selected_indices = self._importance_sampling(z_latent, sample_size)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Additional method info
        additional_info = {
            'method': self.method,
            'bandwidth': self.bandwidth,
            'n_strata': self.n_strata,
            'min_samples_per_stratum': self.min_samples_per_stratum
        }
        
        logger.info(f"Density-aware sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _kde_stratified_sampling(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """KDE + stratified sampling."""
        # Estimate density using KDE
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(z_latent)
        
        # Compute log densities
        log_densities = kde.score_samples(z_latent)
        densities = np.exp(log_densities)
        
        # Determine number of strata
        if self.n_strata is None:
            self.n_strata = min(max(5, int(np.sqrt(sample_size))), sample_size // 2)
        
        # Create density-based strata
        density_percentiles = np.linspace(0, 100, self.n_strata + 1)
        density_thresholds = np.percentile(densities, density_percentiles)
        
        selected_indices = []
        
        # Sample from each stratum
        for i in range(self.n_strata):
            # Find points in this stratum
            if i == 0:
                stratum_mask = densities <= density_thresholds[i + 1]
            elif i == self.n_strata - 1:
                stratum_mask = densities >= density_thresholds[i]
            else:
                stratum_mask = (densities >= density_thresholds[i]) & (densities <= density_thresholds[i + 1])
            
            stratum_indices = np.where(stratum_mask)[0]
            
            if len(stratum_indices) == 0:
                continue
            
            # Calculate samples for this stratum (proportional to stratum size)
            stratum_proportion = len(stratum_indices) / len(z_latent)
            stratum_samples = max(
                self.min_samples_per_stratum,
                int(stratum_proportion * sample_size)
            )
            stratum_samples = min(stratum_samples, len(stratum_indices))
            
            # Sample from stratum
            if stratum_samples >= len(stratum_indices):
                selected_indices.extend(stratum_indices.tolist())
            else:
                # Weight by density within stratum for better representativeness
                stratum_densities = densities[stratum_indices]
                stratum_weights = stratum_densities / np.sum(stratum_densities)
                
                sampled_indices = np.random.choice(
                    stratum_indices,
                    size=stratum_samples,
                    replace=False,
                    p=stratum_weights
                )
                selected_indices.extend(sampled_indices.tolist())
        
        # If we need more samples, add highest density points
        while len(selected_indices) < sample_size:
            remaining_indices = [i for i in range(len(z_latent)) if i not in selected_indices]
            if not remaining_indices:
                break
            
            remaining_densities = densities[remaining_indices]
            best_idx = remaining_indices[np.argmax(remaining_densities)]
            selected_indices.append(best_idx)
        
        return selected_indices[:sample_size]
    
    def _importance_sampling(self, z_latent: np.ndarray, sample_size: int) -> List[int]:
        """Importance sampling based on estimated density."""
        # Estimate density
        kde = KernelDensity(bandwidth=self.bandwidth)
        kde.fit(z_latent)
        
        log_densities = kde.score_samples(z_latent)
        densities = np.exp(log_densities)
        
        # Normalize to get sampling probabilities
        probabilities = densities / np.sum(densities)
        
        # Sample according to density
        selected_indices = np.random.choice(
            len(z_latent),
            size=sample_size,
            replace=False,
            p=probabilities
        )
        
        return selected_indices.tolist()

class ProgressiveWassersteinSampler(BaseSampler):
    """
    Progressive sampling that minimizes Wasserstein distance iteratively.
    """
    
    def __init__(
        self,
        initial_size: int = 10,
        batch_size: int = 5,
        max_candidates: int = 1000,
        **kwargs
    ):
        """
        Initialize progressive Wasserstein sampler.
        
        Args:
            initial_size: Size of initial random sample
            batch_size: Number of points to add per iteration
            max_candidates: Maximum candidates to consider per iteration
        """
        super().__init__('progressive_wasserstein', **kwargs)
        
        self.initial_size = initial_size
        self.batch_size = batch_size
        self.max_candidates = max_candidates
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample using progressive Wasserstein minimization.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting progressive Wasserstein sampling for {sample_size} samples")
        
        # Start with initial random sample
        initial_indices = np.random.choice(
            len(z_latent),
            size=min(self.initial_size, sample_size),
            replace=False
        ).tolist()
        
        selected_indices = initial_indices.copy()
        
        # Progressively add points to minimize Wasserstein distance
        while len(selected_indices) < sample_size:
            remaining_needed = sample_size - len(selected_indices)
            batch_size = min(self.batch_size, remaining_needed)
            
            # Get candidates
            candidates = [i for i in range(len(z_latent)) if i not in selected_indices]
            
            if len(candidates) > self.max_candidates:
                candidates = np.random.choice(candidates, self.max_candidates, replace=False)
            
            # Find best candidates that minimize Wasserstein distance
            best_candidates = self._find_best_wasserstein_candidates(
                z_latent, selected_indices, candidates, batch_size
            )
            
            selected_indices.extend(best_candidates)
            
            if not candidates or len(selected_indices) >= len(z_latent):
                break
        
        # Additional method info
        additional_info = {
            'initial_size': self.initial_size,
            'batch_size': self.batch_size,
            'max_candidates': self.max_candidates,
            'iterations': len(selected_indices) // self.batch_size
        }
        
        logger.info(f"Progressive Wasserstein sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )
    
    def _find_best_wasserstein_candidates(
        self,
        z_latent: np.ndarray,
        selected_indices: List[int],
        candidates: List[int],
        batch_size: int
    ) -> List[int]:
        """Find candidates that minimize Wasserstein distance."""
        from scipy.stats import wasserstein_distance
        
        best_score = float('inf')
        best_candidates = []
        
        # Try different combinations of candidates
        n_trials = min(100, len(candidates) ** min(batch_size, 3))
        
        for _ in range(n_trials):
            if batch_size == 1:
                trial_candidates = [np.random.choice(candidates)]
            else:
                trial_candidates = np.random.choice(
                    candidates,
                    size=min(batch_size, len(candidates)),
                    replace=False
                ).tolist()
            
            # Evaluate Wasserstein distance
            trial_indices = selected_indices + trial_candidates
            trial_sample = z_latent[trial_indices]
            
            # Compute 1D Wasserstein distances for each dimension
            total_distance = 0
            for dim in range(z_latent.shape[1]):
                distance = wasserstein_distance(
                    z_latent[:, dim],
                    trial_sample[:, dim]
                )
                total_distance += distance
            
            if total_distance < best_score:
                best_score = total_distance
                best_candidates = trial_candidates
        
        return best_candidates

class BlueNoiseSampler(BaseSampler):
    """
    Blue noise/Poisson disk sampling with density weighting.
    """
    
    def __init__(
        self,
        min_distance_factor: float = 0.1,
        max_attempts: int = 30,
        density_weight: float = 0.5,
        **kwargs
    ):
        """
        Initialize blue noise sampler.
        
        Args:
            min_distance_factor: Factor for minimum distance (relative to space size)
            max_attempts: Maximum attempts per sample
            density_weight: Weight for density vs. spacing trade-off
        """
        super().__init__('blue_noise', **kwargs)
        
        self.min_distance_factor = min_distance_factor
        self.max_attempts = max_attempts
        self.density_weight = density_weight
    
    def sample(
        self,
        z_latent: np.ndarray,
        sample_size: int,
        original_df: pd.DataFrame,
        **kwargs
    ) -> SamplingResult:
        """
        Sample using blue noise with density weighting.
        
        Args:
            z_latent: Latent space coordinates
            sample_size: Number of representatives to select
            original_df: Original data DataFrame
            
        Returns:
            SamplingResult with selected representatives
        """
        logger.info(f"Starting blue noise sampling for {sample_size} samples")
        
        # Estimate density
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth='scott')
        kde.fit(z_latent)
        
        log_densities = kde.score_samples(z_latent)
        densities = np.exp(log_densities)
        
        # Normalize densities
        densities = densities / np.max(densities)
        
        # Calculate adaptive minimum distance based on local density
        space_size = np.max(z_latent, axis=0) - np.min(z_latent, axis=0)
        base_min_distance = self.min_distance_factor * np.min(space_size)
        
        selected_indices = []
        
        # Start with highest density point
        first_idx = np.argmax(densities)
        selected_indices.append(first_idx)
        
        # Iteratively add points
        for _ in range(sample_size - 1):
            best_candidate = None
            best_score = -float('inf')
            
            # Try multiple candidates
            candidates = [i for i in range(len(z_latent)) if i not in selected_indices]
            if not candidates:
                break
            
            # Sample candidates based on density
            n_candidates = min(self.max_attempts, len(candidates))
            candidate_probs = densities[candidates]
            candidate_probs = candidate_probs / np.sum(candidate_probs)
            
            sampled_candidates = np.random.choice(
                candidates,
                size=n_candidates,
                replace=False,
                p=candidate_probs
            )
            
            for candidate_idx in sampled_candidates:
                candidate_point = z_latent[candidate_idx]
                
                # Calculate minimum distance to existing points
                if selected_indices:
                    selected_points = z_latent[selected_indices]
                    distances = cdist([candidate_point], selected_points)[0]
                    min_distance = np.min(distances)
                else:
                    min_distance = float('inf')
                
                # Adaptive minimum distance based on local density
                local_density = densities[candidate_idx]
                adaptive_min_distance = base_min_distance / (local_density + 0.1)
                
                # Score combines distance and density
                distance_score = min_distance / adaptive_min_distance
                density_score = local_density
                
                combined_score = (
                    (1 - self.density_weight) * distance_score +
                    self.density_weight * density_score
                )
                
                if combined_score > best_score and min_distance >= adaptive_min_distance:
                    best_score = combined_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
            else:
                # Relax distance constraint if no valid candidates
                remaining_candidates = [i for i in range(len(z_latent)) if i not in selected_indices]
                if remaining_candidates:
                    # Choose highest density remaining point
                    remaining_densities = densities[remaining_candidates]
                    best_remaining = remaining_candidates[np.argmax(remaining_densities)]
                    selected_indices.append(best_remaining)
        
        # Additional method info
        additional_info = {
            'min_distance_factor': self.min_distance_factor,
            'max_attempts': self.max_attempts,
            'density_weight': self.density_weight,
            'base_min_distance': base_min_distance
        }
        
        logger.info(f"Blue noise sampling completed: {len(selected_indices)} representatives")
        
        return self.create_standard_result(
            selected_indices, z_latent, original_df, sample_size, additional_info
        )