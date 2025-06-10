"""
Enhanced testing system with multiple latent dimensions support.
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle
import glob
import re
import shutil
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from evaluation.testing import SimplifiedTester

logger = logging.getLogger(__name__)

class EnhancedTester(SimplifiedTester):
    """
    Enhanced testing system with multiple latent dimensions support.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.latent_dims = getattr(config.model, 'LATENT_DIMS', [2])
        logger.info(f"Enhanced tester initialized for latent dimensions: {self.latent_dims}")
    
    def run_all_tests_with_latent_dims(self) -> Dict[str, Any]:
        """
        Run enhanced tests for all latent dimensions and sampling results.
        
        Returns:
            Dictionary with test results and rankings organized by latent dimension
        """
        logger.info("🧪 Starting enhanced testing with latent dimension analysis...")
        
        # Load original data
        try:
            original_df = self._load_original_data()
        except Exception as e:
            logger.error(f"Failed to load original data: {e}")
            return {'error': f'Failed to load original data: {e}'}
        
        all_results = {}
        
        total_configs = (len(self.config.training.ANNEALING_STRATEGIES) * 
                        len(self.config.training.BETA_VALUES) * 
                        len(self.latent_dims))
        current_config = 0
        
        # Process each latent dimension
        for latent_dim in self.latent_dims:
            logger.info(f"\n📏 Testing latent dimension: {latent_dim}")
            latent_results = []
            
            # Process each configuration for this latent dimension
            for strategy in self.config.training.ANNEALING_STRATEGIES:
                for beta in self.config.training.BETA_VALUES:
                    current_config += 1
                    logger.info(f"\n🧪 Testing configuration {current_config}/{total_configs}")
                    logger.info(f"   Latent Dim: {latent_dim}, Strategy: {strategy}, Beta: {beta}")
                    
                    config_results = self._test_single_configuration_with_latent_dim(
                        strategy, beta, latent_dim, original_df
                    )
                    
                    if config_results:
                        latent_results.extend(config_results)
            
            all_results[latent_dim] = latent_results
            
            # Create latent dimension specific analysis
            if latent_results:
                self._create_latent_dim_analysis(latent_dim, latent_results)
        
        # Create comprehensive cross-latent dimension analysis
        if all_results:
            self._create_cross_latent_dim_analysis(all_results)
            self._create_enhanced_rankings_and_summary(all_results)
        
        # Calculate overall statistics
        total_tests = sum(len(results) for results in all_results.values())
        
        return {
            'results': all_results, 
            'total_tests': total_tests,
            'latent_dims_tested': list(all_results.keys()),
            'summary_created': True
        }
    
    def _test_single_configuration_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        original_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Test a single strategy-beta-latent_dim configuration."""
        
        config_dir = os.path.join(
            self.config.paths.SAMPLES_DIR, 
            f'latent_{latent_dim}',
            strategy, 
            f'beta_{beta}'
        )
        
        if not os.path.exists(config_dir):
            logger.warning(f"No sampling results found for {strategy}-{beta}-{latent_dim}")
            return []
        
        # Find available methods
        available_methods = self._find_available_methods(config_dir)
        
        if not available_methods:
            logger.warning(f"No valid sampling methods found for {strategy}-{beta}-{latent_dim}")
            return []
        
        config_results = []
        
        # Test each sample size
        for sample_size in self.config.training.SAMPLE_SIZES:
            logger.info(f"  Sample size: {sample_size}")
            
            # Test each method
            for method in available_methods:
                logger.info(f"    Method: {method}")
                
                try:
                    method_result = self._test_single_method_with_latent_dim(
                        strategy, beta, latent_dim, sample_size, method, original_df
                    )
                    if method_result:
                        config_results.append(method_result)
                    
                except Exception as e:
                    logger.error(f"Testing failed for {method} with latent_dim {latent_dim}: {e}")
        
        return config_results
    
    def _test_single_method_with_latent_dim(
        self,
        strategy: str,
        beta: float,
        latent_dim: int,
        sample_size: int,
        method: str,
        original_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Test a single sampling method with specific latent dimension."""
        
        # Load sampled data
        sampled_df = self._load_sampled_data_with_latent_dim(
            strategy, beta, latent_dim, sample_size, method
        )
        
        if sampled_df is None:
            return None
        
        # Try to get the original indices of sampled points
        sampled_indices = self._get_sampled_indices_with_latent_dim(
            strategy, beta, latent_dim, sample_size, method
        )
        
        # Get latent encodings for Wasserstein distance
        z_original, z_sampled = self._get_latent_encodings_with_latent_dim(
            original_df, sampled_df, strategy, beta, latent_dim
        )
        
        if z_original is None or z_sampled is None:
            logger.warning(f"Could not get latent encodings for {strategy}-{beta}-{latent_dim}-{method}")
            return None
        
        # Calculate enhanced metrics for multiple dimensions
        metrics = self._calculate_enhanced_metrics(z_original, z_sampled, latent_dim)
        
        # Run classifier-based 2-sample test on original data
        classifier_results = self._run_classifier_test(original_df, sampled_df, sampled_indices)
        
        result = {
            'latent_dim': latent_dim,
            'strategy': strategy,
            'beta': beta,
            'method': method,
            'sample_size': sample_size,
            'n_original': len(z_original),
            'n_sampled': len(z_sampled),
            **metrics,
            **classifier_results
        }
        
        logger.debug(f"    Latent Dim {latent_dim}: Wasserstein: {metrics.get('wasserstein_distance', 0):.4f}, "
                    f"Balanced Acc: {classifier_results['balanced_accuracy']:.3f}")
        
        return result
    
    def _load_sampled_data_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        sample_size: int, 
        method: str
    ) -> Optional[pd.DataFrame]:
        """Load sampled data for a specific configuration with latent dimension."""
        sampled_file = os.path.join(
            self.config.paths.SAMPLES_DIR,
            f'latent_{latent_dim}',
            strategy,
            f'beta_{beta}',
            f'method_{method}',
            f'samples_{sample_size}',
            'selected_points.csv'
        )
        
        if not os.path.exists(sampled_file):
            logger.warning(f"Sampled data not found: {sampled_file}")
            return None
        
        return pd.read_csv(sampled_file)
    
    def _get_sampled_indices_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        sample_size: int, 
        method: str
    ) -> Optional[List[int]]:
        """Get the original indices of sampled points with latent dimension."""
        indices_file = os.path.join(
            self.config.paths.SAMPLES_DIR,
            f'latent_{latent_dim}',
            strategy,
            f'beta_{beta}',
            f'method_{method}',
            f'samples_{sample_size}',
            'selected_indices.npy'
        )
        
        try:
            if os.path.exists(indices_file):
                indices = np.load(indices_file)
                return indices.tolist()
        except Exception as e:
            logger.debug(f"Could not load indices file {indices_file}: {e}")
        
        return None
    
    def _get_latent_encodings_with_latent_dim(
        self,
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame,
        strategy: str,
        beta: float,
        latent_dim: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latent encodings for original and sampled data with specific latent dimension."""
        try:
            from models.vae import AdaptiveVAE
            
            # Load model with recovery
            model_dir = os.path.join(
                self.config.paths.MODELS_DIR, 
                f'latent_{latent_dim}',
                strategy, 
                f'beta_{beta}'
            )
            model_path = os.path.join(model_dir, 'vae_model_final.pth')
            
            # Check if model exists, try recovery if not
            if not os.path.exists(model_path):
                logger.warning(f"Final model not found: {model_path}")
                try:
                    self._recover_final_model_from_checkpoint(model_dir)
                    if not os.path.exists(model_path):
                        raise FileNotFoundError("Recovery failed - no final model created")
                    logger.info("Successfully recovered final model from checkpoint")
                except Exception as e:
                    logger.error(f"Model recovery failed for {strategy}-{beta}-{latent_dim}: {e}")
                    return None, None
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            checkpoint = self._validate_and_reconstruct_checkpoint(checkpoint, strategy, beta)
            
            # Create model instance
            cat_dict = checkpoint.get('categorical_cardinality', {})
            if isinstance(cat_dict, dict) and not cat_dict:
                cat_dict = self._load_categorical_info()
            
            model = AdaptiveVAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=latent_dim,
                cat_dict=cat_dict
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Load preprocessed data
            if not os.path.exists(self.config.paths.PREPROCESSED_FILE):
                logger.error(f"Preprocessed data not found: {self.config.paths.PREPROCESSED_FILE}")
                return None, None
            
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            
            # Subsample for efficiency
            if len(original_df) > self.max_samples:
                original_indices = np.random.choice(len(original_df), self.max_samples, replace=False)
                original_subset = original_df.iloc[original_indices]
            else:
                original_indices = np.arange(len(original_df))
                original_subset = original_df
            
            # Ensure indices are within bounds
            valid_indices = original_indices[original_indices < len(preprocessed_df)]
            if len(valid_indices) == 0:
                logger.error("No valid indices for original data")
                return None, None
            
            # Get encodings for original data
            original_preprocessed = torch.FloatTensor(
                preprocessed_df.iloc[valid_indices].values
            ).to(self.device)
            z_original = self._get_latent_encoding_enhanced(model, original_preprocessed)
            
            # Get encodings for sampled data
            try:
                if hasattr(sampled_df, 'index') and max(sampled_df.index) < len(preprocessed_df):
                    sampled_preprocessed = torch.FloatTensor(
                        preprocessed_df.iloc[sampled_df.index].values
                    ).to(self.device)
                else:
                    # Use random subset if indices don't match
                    sampled_size = min(len(sampled_df), len(preprocessed_df))
                    sampled_indices = np.random.choice(
                        len(preprocessed_df), sampled_size, replace=False
                    )
                    sampled_preprocessed = torch.FloatTensor(
                        preprocessed_df.iloc[sampled_indices].values
                    ).to(self.device)
                
                z_sampled = self._get_latent_encoding_enhanced(model, sampled_preprocessed)
                
            except Exception as e:
                logger.warning(f"Could not map sampled data indices: {e}")
                return None, None
            
            logger.debug(f"Latent encodings obtained for dim {latent_dim}: original {z_original.shape}, sampled {z_sampled.shape}")
            return z_original, z_sampled
            
        except Exception as e:
            logger.error(f"Error getting latent encodings for latent_dim {latent_dim}: {e}")
            return None, None
    
    def _get_latent_encoding_enhanced(self, model, data_tensor: torch.Tensor, batch_size: int = 512) -> np.ndarray:
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
    
    def _calculate_enhanced_metrics(
        self, 
        z_original: np.ndarray, 
        z_sampled: np.ndarray, 
        latent_dim: int
    ) -> Dict[str, float]:
        """
        Calculate enhanced metrics for multiple latent dimensions.
        """
        metrics = {}
        
        # Basic Wasserstein distance (average across dimensions)
        wasserstein_distances = []
        for dim in range(min(z_original.shape[1], z_sampled.shape[1])):
            dist = wasserstein_distance(z_original[:, dim], z_sampled[:, dim])
            wasserstein_distances.append(dist)
        
        metrics['wasserstein_distance'] = np.mean(wasserstein_distances)
        metrics['wasserstein_std'] = np.std(wasserstein_distances)
        
        # Dimension-specific metrics
        if latent_dim > 2:
            metrics['wasserstein_per_dim'] = wasserstein_distances[:latent_dim]
            
            # Variance preservation
            orig_variances = np.var(z_original, axis=0)
            samp_variances = np.var(z_sampled, axis=0)
            
            # Calculate variance ratio for each dimension
            variance_ratios = samp_variances / (orig_variances + 1e-8)
            metrics['variance_preservation'] = np.mean(variance_ratios)
            metrics['variance_preservation_std'] = np.std(variance_ratios)
            
            # Active dimensions (dimensions with meaningful variance)
            active_dims_orig = np.sum(orig_variances > 0.01)
            active_dims_samp = np.sum(samp_variances > 0.01)
            metrics['active_dims_preservation'] = active_dims_samp / max(active_dims_orig, 1)
            
            # Correlation structure preservation
            if z_original.shape[1] > 1 and z_sampled.shape[1] > 1:
                orig_corr = np.corrcoef(z_original.T)
                samp_corr = np.corrcoef(z_sampled.T)
                
                # Frobenius norm of correlation difference
                corr_diff = np.linalg.norm(orig_corr - samp_corr, 'fro')
                metrics['correlation_preservation'] = 1.0 / (1.0 + corr_diff)
        
        # Coverage metrics using first 2 dimensions for visualization
        z_orig_2d = z_original[:, :2]
        z_samp_2d = z_sampled[:, :2]
        
        # Calculate coverage using convex hull area ratio
        try:
            from scipy.spatial import ConvexHull
            
            if len(z_orig_2d) >= 3 and len(z_samp_2d) >= 3:
                hull_orig = ConvexHull(z_orig_2d)
                hull_samp = ConvexHull(z_samp_2d)
                
                coverage_ratio = hull_samp.volume / hull_orig.volume if hull_orig.volume > 0 else 0
                metrics['coverage_ratio'] = min(coverage_ratio, 1.0)  # Cap at 1.0
            else:
                metrics['coverage_ratio'] = 0.0
                
        except Exception:
            metrics['coverage_ratio'] = 0.0
        
        # Representativeness score based on multiple factors
        representativeness_components = []
        
        # Wasserstein-based score (lower is better, so invert)
        wasserstein_score = 1.0 / (1.0 + metrics['wasserstein_distance'])
        representativeness_components.append(wasserstein_score)
        
        # Coverage score
        representativeness_components.append(metrics.get('coverage_ratio', 0.0))
        
        # Variance preservation score (for higher dimensions)
        if 'variance_preservation' in metrics:
            # Closer to 1.0 is better
            var_score = 1.0 - abs(1.0 - metrics['variance_preservation'])
            representativeness_components.append(max(0.0, var_score))
        
        # Overall representativeness score
        metrics['representativeness_score'] = np.mean(representativeness_components)
        
        return metrics
    
    def _create_latent_dim_analysis(self, latent_dim: int, results: List[Dict[str, Any]]) -> None:
        """Create analysis for a specific latent dimension."""
        try:
            analysis_dir = os.path.join(self.config.paths.TESTS_DIR, f'latent_{latent_dim}_analysis')
            os.makedirs(analysis_dir, exist_ok=True)
            
            df = pd.DataFrame(results)
            
            # Save detailed results for this latent dimension
            df.to_csv(os.path.join(analysis_dir, f'latent_{latent_dim}_detailed_results.csv'), index=False)
            
            # Create latent dimension specific rankings
            self._create_latent_dim_rankings(df, analysis_dir, latent_dim)
            
            # Create latent dimension specific plots
            self._create_latent_dim_plots(df, analysis_dir, latent_dim)
            
            logger.info(f"📊 Analysis created for latent dimension {latent_dim}")
            
        except Exception as e:
            logger.warning(f"Could not create analysis for latent_dim {latent_dim}: {e}")
    
    def _create_latent_dim_rankings(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create rankings for specific latent dimension."""
        try:
            # Create rankings by different metrics
            rankings_wasserstein = df.groupby(['method', 'sample_size']).agg({
                'wasserstein_distance': 'mean',
                'strategy': 'count'
            }).rename(columns={'strategy': 'n_experiments'}).reset_index()
            
            rankings_wasserstein = rankings_wasserstein.sort_values('wasserstein_distance')
            rankings_wasserstein['wasserstein_rank'] = range(1, len(rankings_wasserstein) + 1)
            
            # Rankings by representativeness score
            rankings_repr = df.groupby(['method', 'sample_size']).agg({
                'representativeness_score': 'mean',
                'balanced_accuracy': 'mean',
                'strategy': 'count'
            }).rename(columns={'strategy': 'n_experiments'}).reset_index()
            
            rankings_repr = rankings_repr.sort_values('representativeness_score', ascending=False)
            rankings_repr['representativeness_rank'] = range(1, len(rankings_repr) + 1)
            
            # Combine rankings
            combined_rankings = pd.merge(
                rankings_wasserstein[['method', 'sample_size', 'wasserstein_distance', 'wasserstein_rank']],
                rankings_repr[['method', 'sample_size', 'representativeness_score', 'balanced_accuracy', 'representativeness_rank']],
                on=['method', 'sample_size']
            )
            
            # Calculate overall score
            combined_rankings['overall_score'] = (
                combined_rankings['wasserstein_rank'] + combined_rankings['representativeness_rank']
            ) / 2
            combined_rankings = combined_rankings.sort_values('overall_score')
            combined_rankings['overall_rank'] = range(1, len(combined_rankings) + 1)
            
            # Save rankings
            rankings_path = os.path.join(analysis_dir, f'latent_{latent_dim}_method_rankings.csv')
            combined_rankings.to_csv(rankings_path, index=False)
            
            # Log top rankings for this latent dimension
            logger.info(f"\n🏆 TOP 3 METHODS FOR LATENT DIMENSION {latent_dim}:")
            top_3 = combined_rankings.head(3)
            for _, row in top_3.iterrows():
                logger.info(f"{row['overall_rank']:2d}. {row['method']} (n={row['sample_size']}) - "
                          f"Wasserstein: {row['wasserstein_distance']:.4f}, "
                          f"Repr.Score: {row['representativeness_score']:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not create rankings for latent_dim {latent_dim}: {e}")
    
    def _create_latent_dim_plots(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create plots for specific latent dimension."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Method performance comparison
            method_performance = df.groupby('method').agg({
                'wasserstein_distance': ['mean', 'std'],
                'representativeness_score': ['mean', 'std']
            })
            
            methods = method_performance.index
            wasserstein_means = method_performance[('wasserstein_distance', 'mean')]
            wasserstein_stds = method_performance[('wasserstein_distance', 'std')]
            
            axes[0, 0].errorbar(range(len(methods)), wasserstein_means, yerr=wasserstein_stds,
                              marker='o', capsize=5, capthick=2)
            axes[0, 0].set_title(f'Wasserstein Distance by Method\n(Latent Dim {latent_dim})')
            axes[0, 0].set_xlabel('Method')
            axes[0, 0].set_ylabel('Wasserstein Distance')
            axes[0, 0].set_xticks(range(len(methods)))
            axes[0, 0].set_xticklabels(methods, rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Representativeness score by method
            repr_means = method_performance[('representativeness_score', 'mean')]
            repr_stds = method_performance[('representativeness_score', 'std')]
            
            axes[0, 1].errorbar(range(len(methods)), repr_means, yerr=repr_stds,
                              marker='s', capsize=5, capthick=2, color='orange')
            axes[0, 1].set_title(f'Representativeness Score by Method\n(Latent Dim {latent_dim})')
            axes[0, 1].set_xlabel('Method')
            axes[0, 1].set_ylabel('Representativeness Score')
            axes[0, 1].set_xticks(range(len(methods)))
            axes[0, 1].set_xticklabels(methods, rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Sample size effects
            sample_effects = df.groupby('sample_size')['wasserstein_distance'].mean()
            axes[1, 0].plot(sample_effects.index, sample_effects.values, 'o-', linewidth=2, markersize=8)
            axes[1, 0].set_title(f'Sample Size Effects\n(Latent Dim {latent_dim})')
            axes[1, 0].set_xlabel('Sample Size')
            axes[1, 0].set_ylabel('Mean Wasserstein Distance')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Strategy vs Beta heatmap
            if len(df['strategy'].unique()) > 1 and len(df['beta'].unique()) > 1:
                heatmap_data = df.groupby(['strategy', 'beta'])['representativeness_score'].mean().unstack()
                im = axes[1, 1].imshow(heatmap_data.values, cmap='viridis', aspect='auto')
                axes[1, 1].set_title(f'Strategy vs Beta Performance\n(Latent Dim {latent_dim})')
                axes[1, 1].set_xlabel('Beta')
                axes[1, 1].set_ylabel('Strategy')
                axes[1, 1].set_xticks(range(len(heatmap_data.columns)))
                axes[1, 1].set_xticklabels([f'{x:.1f}' for x in heatmap_data.columns])
                axes[1, 1].set_yticks(range(len(heatmap_data.index)))
                axes[1, 1].set_yticklabels(heatmap_data.index)
                plt.colorbar(im, ax=axes[1, 1], label='Representativeness Score')
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title(f'Strategy vs Beta Performance\n(Latent Dim {latent_dim})')
            
            plt.suptitle(f'Latent Dimension {latent_dim} Testing Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'latent_{latent_dim}_analysis_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create plots for latent_dim {latent_dim}: {e}")
    
    def _create_cross_latent_dim_analysis(self, all_results: Dict[int, List[Dict[str, Any]]]) -> None:
        """Create analysis across different latent dimensions."""
        try:
            # Combine all results
            combined_data = []
            for latent_dim, results in all_results.items():
                combined_data.extend(results)
            
            if not combined_data:
                logger.warning("No data for cross-latent dimension analysis")
                return
            
            df = pd.DataFrame(combined_data)
            
            # Create cross-latent dimension analysis
            cross_analysis_dir = os.path.join(self.config.paths.TESTS_DIR, 'cross_latent_analysis')
            os.makedirs(cross_analysis_dir, exist_ok=True)
            
            # Save combined results
            df.to_csv(os.path.join(cross_analysis_dir, 'cross_latent_detailed_results.csv'), index=False)
            
            # Create cross-latent dimension plots
            self._create_cross_latent_plots(df, cross_analysis_dir)
            
            # Create latent dimension scaling analysis
            self._create_latent_scaling_analysis(df, cross_analysis_dir)
            
            logger.info("📊 Cross-latent dimension analysis created")
            
        except Exception as e:
            logger.warning(f"Could not create cross-latent dimension analysis: {e}")
    
    def _create_cross_latent_plots(self, df: pd.DataFrame, analysis_dir: str) -> None:
        """Create plots comparing performance across latent dimensions."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Performance vs latent dimension
            latent_performance = df.groupby('latent_dim').agg({
                'wasserstein_distance': ['mean', 'std'],
                'representativeness_score': ['mean', 'std']
            })
            
            latent_dims = sorted(latent_performance.index)
            wasserstein_means = [latent_performance.loc[dim, ('wasserstein_distance', 'mean')] for dim in latent_dims]
            wasserstein_stds = [latent_performance.loc[dim, ('wasserstein_distance', 'std')] for dim in latent_dims]
            
            axes[0, 0].errorbar(latent_dims, wasserstein_means, yerr=wasserstein_stds,
                              marker='o', capsize=5, capthick=2, linewidth=2)
            axes[0, 0].set_title('Wasserstein Distance vs Latent Dimension')
            axes[0, 0].set_xlabel('Latent Dimension')
            axes[0, 0].set_ylabel('Mean Wasserstein Distance')
            axes[0, 0].set_xscale('log', base=2)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Representativeness vs latent dimension
            repr_means = [latent_performance.loc[dim, ('representativeness_score', 'mean')] for dim in latent_dims]
            repr_stds = [latent_performance.loc[dim, ('representativeness_score', 'std')] for dim in latent_dims]
            
            axes[0, 1].errorbar(latent_dims, repr_means, yerr=repr_stds,
                              marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
            axes[0, 1].set_title('Representativeness Score vs Latent Dimension')
            axes[0, 1].set_xlabel('Latent Dimension')
            axes[0, 1].set_ylabel('Mean Representativeness Score')
            axes[0, 1].set_xscale('log', base=2)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Method performance across latent dimensions
            methods = df['method'].unique()
            for method in methods:
                method_data = df[df['method'] == method]
                method_performance = method_data.groupby('latent_dim')['wasserstein_distance'].mean()
                axes[1, 0].plot(method_performance.index, method_performance.values, 
                              'o-', label=method, linewidth=2)
            
            axes[1, 0].set_title('Method Performance Across Latent Dimensions')
            axes[1, 0].set_xlabel('Latent Dimension')
            axes[1, 0].set_ylabel('Mean Wasserstein Distance')
            axes[1, 0].set_xscale('log', base=2)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Variance preservation (if available)
            if 'variance_preservation' in df.columns:
                var_preservation = df.groupby('latent_dim')['variance_preservation'].mean()
                axes[1, 1].plot(var_preservation.index, var_preservation.values, 
                              'o-', linewidth=2, markersize=8, color='green')
                axes[1, 1].set_title('Variance Preservation vs Latent Dimension')
                axes[1, 1].set_xlabel('Latent Dimension')
                axes[1, 1].set_ylabel('Mean Variance Preservation')
                axes[1, 1].set_xscale('log', base=2)
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Variance preservation\ndata not available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Variance Preservation vs Latent Dimension')
            
            plt.suptitle('Cross-Latent Dimension Performance Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'cross_latent_performance_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create cross-latent plots: {e}")
    
    def _create_latent_scaling_analysis(self, df: pd.DataFrame, analysis_dir: str) -> None:
        """Create analysis of how sampling methods scale with latent dimension."""
        try:
            # Scaling analysis for each method
            scaling_data = []
            
            for method in df['method'].unique():
                method_data = df[df['method'] == method]
                
                for latent_dim in sorted(method_data['latent_dim'].unique()):
                    dim_data = method_data[method_data['latent_dim'] == latent_dim]
                    
                    scaling_data.append({
                        'method': method,
                        'latent_dim': latent_dim,
                        'mean_wasserstein': dim_data['wasserstein_distance'].mean(),
                        'mean_representativeness': dim_data['representativeness_score'].mean(),
                        'mean_balanced_accuracy': dim_data['balanced_accuracy'].mean(),
                        'std_wasserstein': dim_data['wasserstein_distance'].std(),
                        'n_experiments': len(dim_data)
                    })
            
            scaling_df = pd.DataFrame(scaling_data)
            scaling_df.to_csv(os.path.join(analysis_dir, 'latent_scaling_analysis.csv'), index=False)
            
            # Create scaling plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Wasserstein distance scaling
            for method in scaling_df['method'].unique():
                method_data = scaling_df[scaling_df['method'] == method]
                ax1.plot(method_data['latent_dim'], method_data['mean_wasserstein'], 
                        'o-', label=method, linewidth=2)
            
            ax1.set_title('Wasserstein Distance Scaling')
            ax1.set_xlabel('Latent Dimension')
            ax1.set_ylabel('Mean Wasserstein Distance')
            ax1.set_xscale('log', base=2)
            ax1.set_yscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Representativeness scaling
            for method in scaling_df['method'].unique():
                method_data = scaling_df[scaling_df['method'] == method]
                ax2.plot(method_data['latent_dim'], method_data['mean_representativeness'], 
                        's-', label=method, linewidth=2)
            
            ax2.set_title('Representativeness Score Scaling')
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Mean Representativeness Score')
            ax2.set_xscale('log', base=2)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('Method Scaling Analysis Across Latent Dimensions', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'method_scaling_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create scaling analysis: {e}")
    
    def _create_enhanced_rankings_and_summary(self, all_results: Dict[int, List[Dict[str, Any]]]) -> None:
        """Create enhanced rankings considering all latent dimensions."""
        try:
            # Combine all results
            combined_data = []
            for latent_dim, results in all_results.items():
                combined_data.extend(results)
            
            if not combined_data:
                logger.warning("No data for enhanced rankings")
                return
            
            df = pd.DataFrame(combined_data)
            
            # Create overall rankings across all latent dimensions
            overall_rankings = df.groupby(['method', 'sample_size']).agg({
                'wasserstein_distance': 'mean',
                'representativeness_score': 'mean',
                'balanced_accuracy': 'mean',
                'latent_dim': 'count'
            }).rename(columns={'latent_dim': 'n_experiments'}).reset_index()
            
            # Calculate ranks
            overall_rankings['wasserstein_rank'] = overall_rankings['wasserstein_distance'].rank()
            overall_rankings['representativeness_rank'] = overall_rankings['representativeness_score'].rank(ascending=False)
            overall_rankings['overall_score'] = (
                overall_rankings['wasserstein_rank'] + overall_rankings['representativeness_rank']
            ) / 2
            overall_rankings = overall_rankings.sort_values('overall_score')
            overall_rankings['overall_rank'] = range(1, len(overall_rankings) + 1)
            
            # Save enhanced rankings
            rankings_path = os.path.join(self.config.paths.TESTS_DIR, 'enhanced_method_rankings.csv')
            overall_rankings.to_csv(rankings_path, index=False)
            
            # Create method summary across all latent dimensions
            method_summary = df.groupby('method').agg({
                'wasserstein_distance': ['mean', 'std', 'min', 'max'],
                'representativeness_score': ['mean', 'std', 'min', 'max'],
                'balanced_accuracy': ['mean', 'std'],
                'latent_dim': ['count', 'nunique']
            }).round(4)
            
            method_summary.columns = ['_'.join(col).strip() for col in method_summary.columns]
            method_summary = method_summary.reset_index()
            
            summary_path = os.path.join(self.config.paths.TESTS_DIR, 'enhanced_method_summary.csv')
            method_summary.to_csv(summary_path, index=False)
            
            # Log enhanced results
            logger.info("\n🏆 TOP 5 METHODS ACROSS ALL LATENT DIMENSIONS:")
            top_5 = overall_rankings.head(5)
            for _, row in top_5.iterrows():
                logger.info(f"{row['overall_rank']:2d}. {row['method']} (n={row['sample_size']}) - "
                          f"Wasserstein: {row['wasserstein_distance']:.4f}, "
                          f"Repr.Score: {row['representativeness_score']:.3f}, "
                          f"Experiments: {row['n_experiments']}")
            
            logger.info(f"\n📁 Enhanced results saved to: {self.config.paths.TESTS_DIR}")
            
        except Exception as e:
            logger.warning(f"Could not create enhanced rankings: {e}")


# Factory function
def create_enhanced_tester(config) -> EnhancedTester:
    """Create enhanced tester with multiple latent dimensions support."""
    return EnhancedTester(config)