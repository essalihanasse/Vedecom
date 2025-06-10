"""
Enhanced sampling manager with multiple latent dimensions support.
"""

import os
import torch
import pandas as pd
import numpy as np
import glob
import re
import shutil
import pickle
from typing import Dict, List, Optional, Any
import logging

from sampling.manager import SamplingManager
from sampling.base import MultiMethodSampler, SamplingResult
from sampling.equiprobable import EquiprobableSampler
from sampling.cluster_based import ClusterBasedSampler  
from sampling.latin_hypercube import LatinHypercubeSampler, AdaptiveLatinHypercubeSampler

logger = logging.getLogger(__name__)

class EnhancedSamplingManager(SamplingManager):
    """
    Enhanced sampling manager with multiple latent dimensions support.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.latent_dims = getattr(config.model, 'LATENT_DIMS', [2])
        self.latent_dim_samplers = {}  # Store samplers per latent dimension
        
        logger.info(f"Enhanced sampling manager initialized for latent dimensions: {self.latent_dims}")
    
    def register_method_for_latent_dim(self, method_name: str, latent_dim: int, **kwargs) -> None:
        """Register a sampling method for a specific latent dimension."""
        if latent_dim not in self.latent_dim_samplers:
            self.latent_dim_samplers[latent_dim] = MultiMethodSampler()
        
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown sampling method: {method_name}")
        
        sampler_class = self.available_methods[method_name]
        
        # Apply latent dimension specific parameters
        latent_specific_kwargs = self._get_latent_specific_params(method_name, latent_dim, kwargs)
        
        sampler = sampler_class(**latent_specific_kwargs)
        self.latent_dim_samplers[latent_dim].register_sampler(method_name, sampler)
        
        logger.info(f"Registered {method_name} for latent dimension {latent_dim}")
    
    def _get_latent_specific_params(self, method_name: str, latent_dim: int, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get latent dimension specific parameters for sampling methods."""
        latent_kwargs = base_kwargs.copy()
        
        # Cluster-based method adjustments
        if method_name == 'cluster_based':
            # Adjust cluster parameters based on latent dimension
            if latent_dim <= 2:
                latent_kwargs['min_clusters'] = max(5, latent_kwargs.get('min_clusters', 5))
                latent_kwargs['max_clusters'] = min(100, latent_kwargs.get('max_clusters', 100))
            elif latent_dim <= 8:
                latent_kwargs['min_clusters'] = max(8, latent_kwargs.get('min_clusters', 8))
                latent_kwargs['max_clusters'] = min(200, latent_kwargs.get('max_clusters', 200))
            else:
                latent_kwargs['min_clusters'] = max(12, latent_kwargs.get('min_clusters', 12))
                latent_kwargs['max_clusters'] = min(500, latent_kwargs.get('max_clusters', 500))
        
        # Latin Hypercube adjustments
        elif method_name in ['latin_hypercube', 'adaptive_latin_hypercube']:
            # Increase optimization iterations for higher dimensions
            if latent_dim > 8:
                latent_kwargs['iterations'] = max(20, latent_kwargs.get('iterations', 10))
            
            # Adjust criterion for higher dimensions
            if latent_dim > 4 and latent_kwargs.get('criterion') == 'maximin':
                latent_kwargs['criterion'] = 'centermaximin'  # Better for high-dim spaces
        
        return latent_kwargs
    
    def run_all_sampling_with_latent_dims(self) -> Dict[str, Dict[str, Any]]:
        """Run sampling for all latent dimensions and model configurations."""
        logger.info("🎲 Starting enhanced sampling across all latent dimensions...")
        
        all_results = {}
        
        total_configs = (len(self.config.training.ANNEALING_STRATEGIES) * 
                        len(self.config.training.BETA_VALUES) * 
                        len(self.latent_dims))
        current_config = 0
        
        for latent_dim in self.latent_dims:
            logger.info(f"\n📏 Processing latent dimension: {latent_dim}")
            latent_results = {}
            
            for strategy in self.config.training.ANNEALING_STRATEGIES:
                strategy_results = {}
                
                for beta in self.config.training.BETA_VALUES:
                    current_config += 1
                    logger.info(f"\n🎲 Sampling configuration {current_config}/{total_configs}")
                    logger.info(f"   Latent Dim: {latent_dim}, Strategy: {strategy}, Beta: {beta}")
                    
                    results = self.run_sampling_for_model_with_latent_dim(
                        strategy, beta, latent_dim
                    )
                    strategy_results[beta] = results
                
                latent_results[strategy] = strategy_results
            
            all_results[latent_dim] = latent_results
        
        # Create overall summary with latent dimension analysis
        self._create_enhanced_overall_summary(all_results)
        
        return all_results
    
    def run_sampling_for_model_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[int, SamplingResult]]:
        """Run sampling for a specific trained model with latent dimension."""
        logger.info(f"Running sampling for {strategy}-{beta} with latent_dim={latent_dim}")
        
        # Use default methods if none specified
        if methods is None:
            methods = list(self.available_methods.keys())
        
        # Validate methods
        invalid_methods = [m for m in methods if m not in self.available_methods]
        if invalid_methods:
            logger.warning(f"Invalid methods will be skipped: {invalid_methods}")
            methods = [m for m in methods if m in self.available_methods]
        
        if not methods:
            logger.error("No valid methods specified")
            return {'error': 'No valid methods specified'}
        
        try:
            # Load model and get latent encodings
            z_latent, original_df = self._load_model_and_data_with_latent_dim(
                strategy, beta, latent_dim
            )
            
            # Set up output directory
            output_dir = os.path.join(
                self.config.paths.SAMPLES_DIR, 
                f'latent_{latent_dim}',
                strategy, 
                f'beta_{beta}'
            )
            
            # Get the appropriate multi-sampler for this latent dimension
            if latent_dim not in self.latent_dim_samplers:
                logger.warning(f"No samplers registered for latent_dim {latent_dim}, using defaults")
                self._register_default_methods_for_latent_dim(latent_dim)
            
            multi_sampler = self.latent_dim_samplers[latent_dim]
            
            # Run multi-method sampling
            results = multi_sampler.run_sampling(
                z_latent=z_latent,
                original_df=original_df,
                sample_sizes=self.config.training.SAMPLE_SIZES,
                output_base_dir=output_dir,
                methods=methods
            )
            
            # Create comparison plots
            for sample_size in self.config.training.SAMPLE_SIZES:
                multi_sampler.create_comparison_plots(output_dir, sample_size, z_latent)
            
            # Save sampling summary with latent dimension info
            self._save_sampling_summary_with_latent_dim(
                results, output_dir, strategy, beta, latent_dim
            )
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Sampling failed for {strategy}-{beta}-{latent_dim}: {e}")
            return {'error': str(e)}
    
    def _register_default_methods_for_latent_dim(self, latent_dim: int) -> None:
        """Register default methods for a specific latent dimension."""
        default_params = self._get_default_params_for_latent_dim(latent_dim)
        
        for method_name, params in default_params.items():
            try:
                self.register_method_for_latent_dim(method_name, latent_dim, **params)
                logger.info(f"✅ Registered default {method_name} for latent_dim {latent_dim}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to register {method_name} for latent_dim {latent_dim}: {e}")
    
    def _get_default_params_for_latent_dim(self, latent_dim: int) -> Dict[str, Dict[str, Any]]:
        """Get default parameters adapted for specific latent dimension."""
        base_params = {
            'cluster_based': {
                'cluster_method': 'kmeans',
                'cluster_sizing_method': 'adaptive',
                'within_cluster_method': 'centroid_distance',
                'info_weight': 1.0,
                'redundancy_weight': 1.0
            },
            'equiprobable': {},
            'latin_hypercube': {
                'criterion': 'maximin',
                'random_state': 42
            },
            'adaptive_latin_hypercube': {
                'criterion': 'maximin',
                'density_weight': 0.3,
                'adaptive_iterations': 20,
                'random_state': 42
            }
        }
        
        # Apply latent dimension specific adjustments
        adapted_params = {}
        for method_name, params in base_params.items():
            adapted_params[method_name] = self._get_latent_specific_params(
                method_name, latent_dim, params
            )
        
        return adapted_params
    
    def _load_model_and_data_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int
    ) -> tuple:
        """
        Load trained model and generate latent encodings for specific latent dimension.
        """
        from models.vae import AdaptiveVAE, get_latent_encoding
        
        model_dir = os.path.join(
            self.config.paths.MODELS_DIR, 
            f'latent_{latent_dim}',
            strategy, 
            f'beta_{beta}'
        )
        model_path = os.path.join(model_dir, 'vae_model_final.pth')
        
        logger.info(f"🔍 Looking for model: {model_path}")
        
        # Enhanced model existence and recovery logic
        if not os.path.exists(model_path):
            logger.warning(f"❌ Final model not found: {model_path}")
            logger.info("🔄 Attempting to recover from checkpoints...")
            
            try:
                self._recover_final_model_from_checkpoint(model_dir)
                if not os.path.exists(model_path):
                    raise FileNotFoundError("Recovery failed - no final model created")
                logger.info("✅ Successfully recovered final model from checkpoint")
            except Exception as e:
                logger.error(f"Model recovery failed for {strategy}-{beta}-{latent_dim}: {e}")
                raise Exception(f"Model loading failed for {strategy}-{beta}-{latent_dim}: {e}")
        
        # Load and validate the model
        try:
            logger.info(f"📥 Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Validate checkpoint structure
            required_keys = ['model_state_dict', 'input_dim', 'num_numerical']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            
            if missing_keys:
                logger.warning(f"⚠️ Checkpoint missing keys: {missing_keys}")
                # Try to reconstruct missing information
                checkpoint = self._reconstruct_checkpoint_info(checkpoint, strategy, beta)
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise Exception(f"Model loading failed for {strategy}-{beta}-{latent_dim}: {e}")
        
        # Create model instance with error handling
        try:
            # Handle different checkpoint formats
            cat_dict = checkpoint.get('categorical_cardinality', {})
            if isinstance(cat_dict, dict) and not cat_dict:
                # Load from preprocessing objects if empty
                logger.info("🔄 Loading categorical info from preprocessing objects")
                cat_dict = self._load_categorical_info()
            
            model = AdaptiveVAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=latent_dim,
                cat_dict=cat_dict
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ Model loaded successfully for latent_dim={latent_dim}")
            
        except Exception as e:
            logger.error(f"Failed to create model instance: {e}")
            raise Exception(f"Model instantiation failed: {e}")
        
        # Load data with error handling
        try:
            # Try multiple data file locations
            data_paths = [
                os.path.join(self.config.paths.DATA_DIR, 'filtered_data.csv'),
                os.path.join(self.config.paths.DATA_DIR, 'data.csv'),
                self.config.paths.DATA_FILE
            ]
            
            original_df = None
            for data_path in data_paths:
                if os.path.exists(data_path):
                    logger.info(f"📊 Loading data from: {data_path}")
                    original_df = pd.read_csv(data_path)
                    break
            
            if original_df is None:
                raise FileNotFoundError(f"No data file found in: {data_paths}")
            
            # Load preprocessed data
            if not os.path.exists(self.config.paths.PREPROCESSED_FILE):
                raise FileNotFoundError(f"Preprocessed data not found: {self.config.paths.PREPROCESSED_FILE}")
            
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            
            # Get latent encodings using the enhanced method
            data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
            z_latent = self._get_latent_encoding_enhanced(model, data_tensor)
            
            logger.info(f"✅ Loaded model and data: {z_latent.shape} latent points for dim={latent_dim}")
            
            return z_latent, original_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise Exception(f"Data loading failed: {e}")
    
    def _get_latent_encoding_enhanced(
        self, 
        model: AdaptiveVAE, 
        data_tensor: torch.Tensor, 
        batch_size: int = 512
    ) -> np.ndarray:
        """Enhanced latent encoding with better memory management."""
        model.eval()
        data_tensor = data_tensor.to(self.device)
        
        encodings = []
        with torch.no_grad():
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i + batch_size]
                latent_repr = model.get_latent_representation(batch, use_mean=True)
                encodings.append(latent_repr.cpu().numpy())
        
        return np.vstack(encodings)
    
    def _save_sampling_summary_with_latent_dim(
        self, 
        results: Dict[str, Dict[int, SamplingResult]], 
        output_dir: str, 
        strategy: str, 
        beta: float,
        latent_dim: int
    ) -> None:
        """Save sampling summary with latent dimension information."""
        if 'error' in results:
            return
            
        summary_data = []
        
        for method, method_results in results.items():
            for sample_size, result in method_results.items():
                if hasattr(result, 'method_info'):
                    summary_data.append({
                        'latent_dim': latent_dim,
                        'method': method,
                        'sample_size': sample_size,
                        'n_selected': result.n_selected,
                        'strategy': strategy,
                        'beta': beta,
                        'success': True
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, 'sampling_summary.csv'), index=False)
            logger.info(f"📊 Sampling summary saved for {strategy}-{beta}-{latent_dim}")
    
    def _create_enhanced_overall_summary(self, all_results: Dict) -> None:
        """Create enhanced overall summary across all latent dimensions."""
        summary_path = os.path.join(self.config.paths.SAMPLES_DIR, 'enhanced_sampling_summary.csv')
        
        summary_data = []
        for latent_dim, latent_results in all_results.items():
            for strategy, strategy_results in latent_results.items():
                for beta, beta_results in strategy_results.items():
                    if 'error' not in beta_results:
                        for method, method_results in beta_results.items():
                            for sample_size, result in method_results.items():
                                if hasattr(result, 'method_info'):
                                    summary_data.append({
                                        'latent_dim': latent_dim,
                                        'strategy': strategy,
                                        'beta': beta,
                                        'method': method,
                                        'sample_size': sample_size,
                                        'n_selected': result.n_selected,
                                        'success': True
                                    })
                    else:
                        summary_data.append({
                            'latent_dim': latent_dim,
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
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df.to_csv(summary_path, index=False)
            
            # Create latent dimension analysis
            self._create_latent_dimension_sampling_analysis(df)
            
            # Log enhanced summary statistics
            total_runs = len(df[df['success'] == True])
            failed_runs = len(df[df['success'] == False])
            
            logger.info(f"📊 Enhanced sampling summary:")
            logger.info(f"  ✅ Successful runs: {total_runs}")
            logger.info(f"  ❌ Failed runs: {failed_runs}")
            
            if total_runs > 0:
                methods_used = df[df['success'] == True]['method'].unique()
                latent_dims_processed = df[df['success'] == True]['latent_dim'].unique()
                logger.info(f"  📋 Methods used: {', '.join(methods_used)}")
                logger.info(f"  📏 Latent dimensions processed: {sorted(latent_dims_processed)}")
    
    def _create_latent_dimension_sampling_analysis(self, df: pd.DataFrame) -> None:
        """Create analysis of sampling performance across latent dimensions."""
        try:
            analysis_dir = os.path.join(self.config.paths.SAMPLES_DIR, 'latent_dimension_analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Filter successful runs
            success_df = df[df['success'] == True].copy()
            
            if len(success_df) == 0:
                logger.warning("No successful runs for latent dimension analysis")
                return
            
            # Analysis by latent dimension
            latent_analysis = success_df.groupby(['latent_dim', 'method']).agg({
                'n_selected': ['mean', 'std', 'count'],
                'sample_size': 'mean'
            }).round(3)
            
            latent_analysis.columns = ['_'.join(col).strip() for col in latent_analysis.columns]
            latent_analysis = latent_analysis.reset_index()
            latent_analysis.to_csv(os.path.join(analysis_dir, 'latent_dimension_method_analysis.csv'), index=False)
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            # Method performance across latent dimensions
            methods = success_df['method'].unique()
            latent_dims = sorted(success_df['latent_dim'].unique())
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Average samples selected by latent dimension
            ax1 = axes[0, 0]
            for method in methods:
                method_data = success_df[success_df['method'] == method]
                method_means = method_data.groupby('latent_dim')['n_selected'].mean()
                ax1.plot(method_means.index, method_means.values, 'o-', label=method, linewidth=2)
            
            ax1.set_title('Average Samples Selected by Latent Dimension')
            ax1.set_xlabel('Latent Dimension')
            ax1.set_ylabel('Average Samples Selected')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)
            
            # Plot 2: Success rate by latent dimension
            ax2 = axes[0, 1]
            success_rates = df.groupby('latent_dim')['success'].mean()
            ax2.bar(range(len(success_rates)), success_rates.values, alpha=0.7)
            ax2.set_title('Success Rate by Latent Dimension')
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Success Rate')
            ax2.set_xticks(range(len(success_rates)))
            ax2.set_xticklabels([str(dim) for dim in success_rates.index])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: Method distribution across latent dimensions
            ax3 = axes[1, 0]
            method_counts = success_df.groupby(['latent_dim', 'method']).size().unstack(fill_value=0)
            method_counts.plot(kind='bar', ax=ax3, alpha=0.7)
            ax3.set_title('Method Usage Across Latent Dimensions')
            ax3.set_xlabel('Latent Dimension')
            ax3.set_ylabel('Number of Runs')
            ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Sample size efficiency
            ax4 = axes[1, 1]
            efficiency = success_df.groupby('latent_dim').apply(
                lambda x: (x['n_selected'] / x['sample_size']).mean()
            )
            ax4.plot(efficiency.index, efficiency.values, 'o-', linewidth=2, markersize=8)
            ax4.set_title('Sampling Efficiency by Latent Dimension')
            ax4.set_xlabel('Latent Dimension')
            ax4.set_ylabel('Avg. (Selected / Requested)')
            ax4.grid(True, alpha=0.3)
            ax4.set_xscale('log', base=2)
            
            plt.suptitle('Latent Dimension Sampling Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'latent_dimension_sampling_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("📊 Latent dimension sampling analysis created")
            
        except Exception as e:
            logger.warning(f"Could not create latent dimension sampling analysis: {e}")
SamplingManager=EnhancedSamplingManager

class EnhancedSamplingManagerFactory:
    """Factory for creating enhanced sampling managers with different configurations."""
    
    @staticmethod
    def create_default_manager(config) -> EnhancedSamplingManager:
        """Create enhanced sampling manager with default methods for all latent dimensions."""
        manager = EnhancedSamplingManager(config)
        
        # Register methods for all latent dimensions
        for latent_dim in manager.latent_dims:
            manager._register_default_methods_for_latent_dim(latent_dim)
        
        return manager
    
    @staticmethod
    def create_manager_with_methods(
        config, 
        methods: List[str], 
        latent_dims: Optional[List[int]] = None
    ) -> EnhancedSamplingManager:
        """
        Create enhanced sampling manager with specific methods and latent dimensions.
        
        Args:
            config: Configuration object
            methods: List of method names to register
            latent_dims: List of latent dimensions (uses config default if None)
        """
        manager = EnhancedSamplingManager(config)
        
        if latent_dims:
            manager.latent_dims = latent_dims
        
        # Register specified methods for all latent dimensions
        for latent_dim in manager.latent_dims:
            default_params = manager._get_default_params_for_latent_dim(latent_dim)
            
            for method in methods:
                if method in manager.available_methods:
                    params = default_params.get(method, {})
                    try:
                        manager.register_method_for_latent_dim(method, latent_dim, **params)
                        logger.info(f"✅ Registered {method} for latent_dim {latent_dim}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to register {method} for latent_dim {latent_dim}: {e}")
                else:
                    logger.warning(f"⚠️ Unknown method: {method}")
        
        return manager


# Factory functions for backwards compatibility
def create_enhanced_sampling_manager(config) -> EnhancedSamplingManager:
    """Create enhanced sampling manager with default methods."""
    return EnhancedSamplingManagerFactory.create_default_manager(config)

def create_sampling_manager_with_latent_dims(
    config, 
    methods: List[str], 
    latent_dims: Optional[List[int]] = None
) -> EnhancedSamplingManager:
    """Create enhanced sampling manager with specific methods and latent dimensions."""
    return EnhancedSamplingManagerFactory.create_manager_with_methods(config, methods, latent_dims)