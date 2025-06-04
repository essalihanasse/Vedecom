"""
Sampling methods manager for coordinating multiple sampling approaches.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from .base import MultiMethodSampler, SamplingResult
from .equiprobable import EquiprobableSampler
from .representative import DistanceBasedSampler
from .cluster_based import ClusterBasedSampler
from .hybrid import HybridSampler
from .density_aware import DensityAwareSampler, ProgressiveWassersteinSampler, BlueNoiseSampler
from .optimal_transport import OptimalTransportSampler, SlicedWassersteinSampler

logger = logging.getLogger(__name__)

class SamplingManager:
    """
    Manages multiple sampling methods and coordinates their execution.
    """
    
    def __init__(self, config):
        """
        Initialize sampling manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_sampler = MultiMethodSampler()
        
        # Available sampling methods
        self.available_methods = {
            'equiprobable': EquiprobableSampler,
            'distance_based': DistanceBasedSampler,
            'cluster_based': ClusterBasedSampler,
            'hybrid': HybridSampler,
            'density_aware_kde': lambda **kwargs: DensityAwareSampler(method='kde_stratified', **kwargs),
            'density_aware_importance': lambda **kwargs: DensityAwareSampler(method='importance', **kwargs),
            'progressive_wasserstein': ProgressiveWassersteinSampler,
            'blue_noise': BlueNoiseSampler,
            'optimal_transport_greedy': lambda **kwargs: OptimalTransportSampler(method='greedy', **kwargs),
            'optimal_transport_hungarian': lambda **kwargs: OptimalTransportSampler(method='hungarian', **kwargs),
            'sliced_wasserstein': SlicedWassersteinSampler,
            'cluster_one_per': lambda **kwargs: ClusterBasedSampler(one_per_cluster=True, **kwargs)
        }
        
        logger.info(f"Sampling manager initialized with {len(self.available_methods)} methods")
    
    def register_method(self, method_name: str, **kwargs) -> None:
        """
        Register a sampling method.
        
        Args:
            method_name: Name of the sampling method
            **kwargs: Parameters for the sampling method
        """
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown sampling method: {method_name}")
        
        # Create sampler instance with parameters
        sampler_class = self.available_methods[method_name]
        sampler = sampler_class(**kwargs)
        
        # Register with multi-sampler
        self.multi_sampler.register_sampler(method_name, sampler)
        
        logger.info(f"Registered {method_name} sampling method")
    
    def run_sampling_for_model(
        self,
        strategy: str,
        beta: float,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, SamplingResult]]:
        """
        Run sampling for a specific trained model.
        
        Args:
            strategy: Annealing strategy
            beta: Beta value
            methods: List of methods to run (None for all registered)
            
        Returns:
            Dictionary of sampling results
        """
        logger.info(f"Running sampling for {strategy} strategy, beta={beta}")
        
        # Load model and get latent encodings
        z_latent, original_df = self._load_model_and_data(strategy, beta)
        
        # Set up output directory
        output_dir = os.path.join(
            self.config.paths.SAMPLES_DIR,
            strategy,
            f'beta_{beta}'
        )
        
        # Run multi-method sampling
        results = self.multi_sampler.run_sampling(
            z_latent=z_latent,
            original_df=original_df,
            sample_sizes=self.config.training.SAMPLE_SIZES,
            output_base_dir=output_dir,
            methods=methods
        )
        
        # Create comparison plots for each sample size
        for sample_size in self.config.training.SAMPLE_SIZES:
            self.multi_sampler.create_comparison_plots(
                output_dir, sample_size, z_latent
            )
        
        # Save sampling summary
        self._save_sampling_summary(results, output_dir, strategy, beta)
        
        return results
    
    def run_all_sampling(self) -> Dict[str, Dict[str, Dict[int, SamplingResult]]]:
        """
        Run sampling for all trained models.
        
        Returns:
            Nested dictionary: {strategy: {beta: {sample_size: result}}}
        """
        all_results = {}
        
        total_configs = len(self.config.training.ANNEALING_STRATEGIES) * len(self.config.training.BETA_VALUES)
        current_config = 0
        
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            strategy_results = {}
            
            for beta in self.config.training.BETA_VALUES:
                current_config += 1
                logger.info(f"\n🎲 Sampling configuration {current_config}/{total_configs}")
                
                try:
                    results = self.run_sampling_for_model(strategy, beta)
                    strategy_results[beta] = results
                    
                except Exception as e:
                    logger.error(f"Sampling failed for {strategy}-{beta}: {e}")
                    strategy_results[beta] = {'error': str(e)}
            
            all_results[strategy] = strategy_results
        
        # Create overall summary
        self._create_overall_summary(all_results)
        
        return all_results
    
    def _load_model_and_data(self, strategy: str, beta: float) -> tuple:
        """
        Load trained model and generate latent encodings.
        
        Args:
            strategy: Annealing strategy
            beta: Beta value
            
        Returns:
            Tuple of (latent_coordinates, original_dataframe)
        """
        from models.vae import VAE, get_latent_encoding
        
        # Load model
        model_path = os.path.join(
            self.config.paths.MODELS_DIR,
            strategy,
            f'beta_{beta}',
            'vae_model_final.pth'
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model instance
        model = VAE(
            input_dim=checkpoint['input_dim'],
            num_numerical=checkpoint['num_numerical'],
            hidden_dim=checkpoint['hidden_dim'],
            latent_dim=checkpoint['latent_dim'],
            cat_dict=checkpoint['categorical_cardinality']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load data
        original_df = pd.read_csv(os.path.join(self.config.paths.DATA_DIR, 'filtered_data.csv'))
        preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
        
        # Get latent encodings
        data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
        z_latent = get_latent_encoding(model, data_tensor, self.device)
        
        logger.info(f"Loaded model and data: {z_latent.shape} latent points")
        
        return z_latent, original_df
    
    def _save_sampling_summary(
        self,
        results: Dict[str, Dict[int, SamplingResult]],
        output_dir: str,
        strategy: str,
        beta: float
    ) -> None:
        """Save sampling summary for a model configuration."""
        summary_data = []
        
        for method, method_results in results.items():
            for sample_size, result in method_results.items():
                if hasattr(result, 'method_info'):
                    summary_data.append({
                        'method': method,
                        'sample_size': sample_size,
                        'n_selected': result.n_selected,
                        'strategy': strategy,
                        'beta': beta,
                        **result.method_info
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(output_dir, 'sampling_summary.csv'), index=False)
            
            logger.info(f"Sampling summary saved for {strategy}-{beta}")
    
    def _create_overall_summary(self, all_results: Dict) -> None:
        """Create overall summary across all configurations."""
        summary_path = os.path.join(self.config.paths.SAMPLES_DIR, 'overall_sampling_summary.csv')
        
        summary_data = []
        for strategy, strategy_results in all_results.items():
            for beta, beta_results in strategy_results.items():
                if 'error' not in beta_results:
                    for method, method_results in beta_results.items():
                        for sample_size, result in method_results.items():
                            if hasattr(result, 'method_info'):
                                summary_data.append({
                                    'strategy': strategy,
                                    'beta': beta,
                                    'method': method,
                                    'sample_size': sample_size,
                                    'n_selected': result.n_selected,
                                    'success': True
                                })
                else:
                    summary_data.append({
                        'strategy': strategy,
                        'beta': beta,
                        'method': 'all',
                        'sample_size': 'all',
                        'n_selected': 0,
                        'success': False,
                        'error': beta_results.get('error', 'Unknown error')
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(summary_path, index=False)
            
            # Log summary statistics
            total_runs = len(df[df['success'] == True])
            failed_runs = len(df[df['success'] == False])
            
            logger.info(f"📊 Overall sampling summary:")
            logger.info(f"  Successful runs: {total_runs}")
            logger.info(f"  Failed runs: {failed_runs}")
            
            if total_runs > 0:
                methods_used = df[df['success'] == True]['method'].unique()
                logger.info(f"  Methods used: {list(methods_used)}")

def create_default_sampling_manager(config) -> SamplingManager:
    """
    Create a sampling manager with default methods registered.
    
    Args:
        config: Configuration object
        
    Returns:
        SamplingManager with default methods
    """
    manager = SamplingManager(config)
    
    # Register default methods with parameters from config
    sampling_params = config.get_sampling_params()
    
    for method in config.sampling.DEFAULT_METHODS:
        try:
            manager.register_method(method, **sampling_params)
        except Exception as e:
            logger.warning(f"Failed to register {method}: {e}")
    
    return manager