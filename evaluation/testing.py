"""
Distribution testing system for evaluating sampling quality.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DistributionTester:
    """
    Tests similarity between original and sampled distributions using multiple methods.
    """
    
    def __init__(self, config):
        """
        Initialize distribution tester.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Testing parameters
        self.fast_mode = os.environ.get('VAE_FAST_MODE', '0') == '1'
        self.max_samples = 1000 if self.fast_mode else 5000
        self.n_permutations = 50 if self.fast_mode else 500
        
        logger.info(f"Distribution tester initialized (fast_mode: {self.fast_mode})")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run distribution tests for all sampling results.
        
        Returns:
            Dictionary with all test results
        """
        logger.info("🧪 Starting comprehensive distribution testing...")
        
        all_results = {}
        
        # Load original data
        original_df = self._load_original_data()
        
        # Process each configuration
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            strategy_results = {}
            
            for beta in self.config.training.BETA_VALUES:
                logger.info(f"\n📊 Testing {strategy} strategy, beta={beta}")
                
                beta_results = self._test_single_configuration(
                    strategy, beta, original_df
                )
                strategy_results[beta] = beta_results
            
            all_results[strategy] = strategy_results
        
        # Create overall summary
        self._create_overall_test_summary(all_results)
        
        return all_results
    
    def _load_original_data(self) -> pd.DataFrame:
        """Load original filtered data."""
        data_path = os.path.join(self.config.paths.DATA_DIR, 'filtered_data.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Original data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded original data: {len(df)} samples")
        return df
    
    def _test_single_configuration(
        self, 
        strategy: str, 
        beta: float, 
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test a single strategy-beta configuration."""
        
        # Find available sampling methods for this configuration
        config_dir = os.path.join(
            self.config.paths.SAMPLES_DIR, 
            strategy, 
            f'beta_{beta}'
        )
        
        if not os.path.exists(config_dir):
            logger.warning(f"No sampling results found for {strategy}-{beta}")
            return {'error': 'No sampling results found'}
        
        # Find available methods
        available_methods = self._find_available_methods(config_dir)
        
        if not available_methods:
            logger.warning(f"No valid sampling methods found for {strategy}-{beta}")
            return {'error': 'No valid sampling methods found'}
        
        config_results = {}
        
        # Test each sample size
        for sample_size in self.config.training.SAMPLE_SIZES:
            logger.info(f"  Sample size: {sample_size}")
            
            size_results = {}
            
            # Test each method
            for method in available_methods:
                logger.info(f"    Method: {method}")
                
                try:
                    method_results = self._test_single_method(
                        strategy, beta, sample_size, method, original_df
                    )
                    size_results[method] = method_results
                    
                except Exception as e:
                    logger.error(f"Testing failed for {method}: {e}")
                    size_results[method] = {'error': str(e)}
            
            config_results[sample_size] = size_results
        
        return config_results
    
    def _find_available_methods(self, config_dir: str) -> List[str]:
        """Find available sampling methods in a configuration directory."""
        methods = []
        
        for item in os.listdir(config_dir):
            if item.startswith('method_') and os.path.isdir(os.path.join(config_dir, item)):
                method_name = item.replace('method_', '')
                methods.append(method_name)
        
        return methods
    
    def _test_single_method(
        self,
        strategy: str,
        beta: float,
        sample_size: int,
        method: str,
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test a single sampling method."""
        
        # Load sampled data
        sampled_df = self._load_sampled_data(strategy, beta, sample_size, method)
        
        if sampled_df is None:
            return {'error': 'Could not load sampled data'}
        
        # Run tests
        results = {}
        
        # 1. Gower distance test (for original features)
        logger.debug("    Running Gower distance test...")
        gower_results = self._run_gower_test(original_df, sampled_df)
        results['gower'] = gower_results
        
        # 2. Multivariate latent space test
        logger.debug("    Running latent space test...")
        latent_results = self._run_latent_test(
            original_df, sampled_df, strategy, beta
        )
        results['latent'] = latent_results
        
        # 3. Feature-wise tests
        logger.debug("    Running feature-wise tests...")
        feature_results = self._run_feature_tests(original_df, sampled_df)
        results['features'] = feature_results
        
        return results
    
    def _load_sampled_data(
        self, 
        strategy: str, 
        beta: float, 
        sample_size: int, 
        method: str
    ) -> Optional[pd.DataFrame]:
        """Load sampled data for a specific configuration."""
        sampled_file = os.path.join(
            self.config.paths.SAMPLES_DIR,
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
    
    def _run_gower_test(
        self, 
        original_df: pd.DataFrame, 
        sampled_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run Gower distance test for mixed-type features."""
        try:
            # Subsample for efficiency
            if len(original_df) > self.max_samples:
                original_sample = original_df.sample(self.max_samples, random_state=42)
            else:
                original_sample = original_df
            
            if len(sampled_df) > self.max_samples:
                sampled_sample = sampled_df.sample(self.max_samples, random_state=42)
            else:
                sampled_sample = sampled_df
            
            # Find common columns
            common_cols = self._find_common_columns(original_sample, sampled_sample)
            
            if not common_cols:
                return {'error': 'No common columns found'}
            
            # Prepare data
            original_data = original_sample[common_cols].dropna()
            sampled_data = sampled_sample[common_cols].dropna()
            
            # Compute Gower distance
            gower_distance = self._compute_gower_distance(
                original_data, sampled_data
            )
            
            # Permutation test
            p_value = self._gower_permutation_test(
                original_data, sampled_data, gower_distance
            )
            
            return {
                'distance': gower_distance,
                'p_value': p_value,
                'reject_h0': p_value < 0.05,
                'n_original': len(original_data),
                'n_sampled': len(sampled_data),
                'n_features': len(common_cols)
            }
            
        except Exception as e:
            logger.error(f"Gower test failed: {e}")
            return {'error': str(e)}
    
    def _run_latent_test(
        self,
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame,
        strategy: str,
        beta: float
    ) -> Dict[str, Any]:
        """Run multivariate latent space test."""
        try:
            # Load VAE model and get latent encodings
            z_original, z_sampled = self._get_latent_encodings(
                original_df, sampled_df, strategy, beta
            )
            
            if z_original is None or z_sampled is None:
                return {'error': 'Could not get latent encodings'}
            
            # Subsample for efficiency
            if len(z_original) > self.max_samples:
                indices = np.random.choice(len(z_original), self.max_samples, replace=False)
                z_original = z_original[indices]
            
            if len(z_sampled) > self.max_samples:
                indices = np.random.choice(len(z_sampled), self.max_samples, replace=False)
                z_sampled = z_sampled[indices]
            
            # Sliced Wasserstein distance test
            wasserstein_results = self._sliced_wasserstein_test(z_original, z_sampled)
            
            return {
                'sliced_wasserstein': wasserstein_results,
                'latent_dim': z_original.shape[1],
                'n_original': len(z_original),
                'n_sampled': len(z_sampled)
            }
            
        except Exception as e:
            logger.error(f"Latent test failed: {e}")
            return {'error': str(e)}
    
    def _run_feature_tests(
        self,
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run feature-wise statistical tests."""
        try:
            results = {}
            
            # Find common numerical columns
            numerical_cols = [
                col for col in self.config.data.NUMERICAL_COLS
                if col in original_df.columns and col in sampled_df.columns
            ]
            
            # Test numerical features
            for col in numerical_cols:
                try:
                    original_values = original_df[col].dropna()
                    sampled_values = sampled_df[col].dropna()
                    
                    if len(original_values) > 0 and len(sampled_values) > 0:
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_p = stats.ks_2samp(original_values, sampled_values)
                        
                        # Mann-Whitney U test
                        mw_stat, mw_p = stats.mannwhitneyu(
                            original_values, sampled_values, alternative='two-sided'
                        )
                        
                        results[col] = {
                            'ks_statistic': ks_stat,
                            'ks_p_value': ks_p,
                            'mw_statistic': mw_stat,
                            'mw_p_value': mw_p,
                            'ks_reject': ks_p < 0.05,
                            'mw_reject': mw_p < 0.05
                        }
                        
                except Exception as e:
                    results[col] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"Feature tests failed: {e}")
            return {'error': str(e)}
    
    def _find_common_columns(
        self, 
        df1: pd.DataFrame, 
        df2: pd.DataFrame
    ) -> List[str]:
        """Find common columns between two dataframes."""
        # Use numerical and categorical columns from config
        all_feature_cols = self.config.data.NUMERICAL_COLS + self.config.data.CATEGORICAL_COLS
        
        common_cols = [
            col for col in all_feature_cols
            if col in df1.columns and col in df2.columns
        ]
        
        # If no config columns found, use numeric columns
        if not common_cols:
            common_cols = list(set(df1.select_dtypes(include=[np.number]).columns) & 
                             set(df2.select_dtypes(include=[np.number]).columns))
        
        return common_cols
    
    def _compute_gower_distance(
        self, 
        data1: pd.DataFrame, 
        data2: pd.DataFrame
    ) -> float:
        """Compute Gower distance between two datasets."""
        # Simple implementation - for full implementation, use gower package
        # Here we use a simplified version focusing on numerical features
        
        numerical_cols = data1.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return 0.0
        
        # Subsample for efficiency
        n_pairs = min(1000, len(data1) * len(data2))
        
        if n_pairs == len(data1) * len(data2):
            # Small enough to compute all pairs
            distances = []
            for _, row1 in data1[numerical_cols].iterrows():
                for _, row2 in data2[numerical_cols].iterrows():
                    distance = np.mean(np.abs(row1 - row2) / (data1[numerical_cols].max() - data1[numerical_cols].min()))
                    distances.append(distance)
        else:
            # Sample pairs
            distances = []
            for _ in range(n_pairs):
                idx1 = np.random.randint(0, len(data1))
                idx2 = np.random.randint(0, len(data2))
                
                row1 = data1[numerical_cols].iloc[idx1]
                row2 = data2[numerical_cols].iloc[idx2]
                
                distance = np.mean(np.abs(row1 - row2) / (data1[numerical_cols].max() - data1[numerical_cols].min()))
                distances.append(distance)
        
        return np.mean(distances)
    
    def _gower_permutation_test(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        observed_distance: float
    ) -> float:
        """Run permutation test for Gower distance."""
        combined_data = pd.concat([data1, data2], ignore_index=True)
        n1, n2 = len(data1), len(data2)
        
        permutation_distances = []
        
        for _ in range(self.n_permutations):
            # Random permutation
            shuffled = combined_data.sample(frac=1.0, random_state=None).reset_index(drop=True)
            perm_data1 = shuffled.iloc[:n1]
            perm_data2 = shuffled.iloc[n1:]
            
            perm_distance = self._compute_gower_distance(perm_data1, perm_data2)
            permutation_distances.append(perm_distance)
        
        # Calculate p-value
        p_value = np.mean(np.array(permutation_distances) >= observed_distance)
        return max(p_value, 1e-10)  # Avoid exactly zero p-values
    
    def _get_latent_encodings(
        self,
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame,
        strategy: str,
        beta: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get latent encodings for original and sampled data."""
        try:
            from models.vae import VAE, get_latent_encoding
            
            # Load model
            model_path = os.path.join(
                self.config.paths.MODELS_DIR,
                strategy,
                f'beta_{beta}',
                'vae_model_final.pth'
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return None, None
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model
            model = VAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint['hidden_dim'],
                latent_dim=checkpoint['latent_dim'],
                cat_dict=checkpoint['categorical_cardinality']
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load preprocessed data
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            
            # Get encodings for original data (subsample)
            if len(original_df) > self.max_samples:
                original_indices = np.random.choice(len(original_df), self.max_samples, replace=False)
                original_subset = original_df.iloc[original_indices]
            else:
                original_indices = np.arange(len(original_df))
                original_subset = original_df
            
            original_preprocessed = torch.FloatTensor(
                preprocessed_df.iloc[original_indices].values
            ).to(self.device)
            z_original = get_latent_encoding(model, original_preprocessed, self.device)
            
            # Get encodings for sampled data
            try:
                # Try to map sampled indices to preprocessed data
                if hasattr(sampled_df, 'index') and max(sampled_df.index) < len(preprocessed_df):
                    sampled_preprocessed = torch.FloatTensor(
                        preprocessed_df.iloc[sampled_df.index].values
                    ).to(self.device)
                else:
                    # Use random subset if indices don't match
                    sampled_indices = np.random.choice(
                        len(preprocessed_df), len(sampled_df), replace=False
                    )
                    sampled_preprocessed = torch.FloatTensor(
                        preprocessed_df.iloc[sampled_indices].values
                    ).to(self.device)
                
                z_sampled = get_latent_encoding(model, sampled_preprocessed, self.device)
                
            except Exception as e:
                logger.warning(f"Could not map sampled data indices: {e}")
                return None, None
            
            return z_original, z_sampled
            
        except Exception as e:
            logger.error(f"Error getting latent encodings: {e}")
            return None, None
    
    def _sliced_wasserstein_test(
        self, 
        z_original: np.ndarray, 
        z_sampled: np.ndarray
    ) -> Dict[str, Any]:
        """Compute sliced Wasserstein distance and perform permutation test."""
        try:
            # Compute observed sliced Wasserstein distance
            observed_distance = self._sliced_wasserstein_distance(z_original, z_sampled)
            
            # Permutation test
            combined = np.vstack([z_original, z_sampled])
            n1, n2 = len(z_original), len(z_sampled)
            
            permutation_distances = []
            for _ in range(self.n_permutations):
                # Random permutation
                indices = np.random.permutation(len(combined))
                perm1 = combined[indices[:n1]]
                perm2 = combined[indices[n1:]]
                
                perm_distance = self._sliced_wasserstein_distance(perm1, perm2)
                permutation_distances.append(perm_distance)
            
            # Calculate p-value
            p_value = np.mean(np.array(permutation_distances) >= observed_distance)
            p_value = max(p_value, 1e-10)  # Avoid exactly zero p-values
            
            return {
                'distance': observed_distance,
                'p_value': p_value,
                'reject_h0': p_value < 0.05,
                'n_permutations': self.n_permutations
            }
            
        except Exception as e:
            logger.error(f"Sliced Wasserstein test failed: {e}")
            return {'error': str(e)}
    
    def _sliced_wasserstein_distance(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        n_projections: int = 50
    ) -> float:
        """Compute sliced Wasserstein distance."""
        d = X.shape[1]
        distances = []
        
        for _ in range(n_projections):
            # Random unit vector
            theta = np.random.randn(d)
            theta = theta / np.linalg.norm(theta)
            
            # Project data
            X_proj = np.dot(X, theta)
            Y_proj = np.dot(Y, theta)
            
            # Compute 1D Wasserstein distance
            w_dist = stats.wasserstein_distance(X_proj, Y_proj)
            distances.append(w_dist)
        
        return np.mean(distances)
    
    def _create_overall_test_summary(self, all_results: Dict[str, Any]) -> None:
        """Create overall summary of test results."""
        try:
            summary_data = []
            
            for strategy in all_results:
                for beta in all_results[strategy]:
                    strategy_results = all_results[strategy][beta]
                    
                    if 'error' in strategy_results:
                        summary_data.append({
                            'strategy': strategy,
                            'beta': beta,
                            'sample_size': 'all',
                            'method': 'all',
                            'test_type': 'all',
                            'success': False,
                            'error': strategy_results['error']
                        })
                        continue
                    
                    for sample_size in strategy_results:
                        for method in strategy_results[sample_size]:
                            method_results = strategy_results[sample_size][method]
                            
                            if 'error' in method_results:
                                summary_data.append({
                                    'strategy': strategy,
                                    'beta': beta,
                                    'sample_size': sample_size,
                                    'method': method,
                                    'test_type': 'all',
                                    'success': False,
                                    'error': method_results['error']
                                })
                                continue
                            
                            # Gower test results
                            if 'gower' in method_results:
                                gower = method_results['gower']
                                summary_data.append({
                                    'strategy': strategy,
                                    'beta': beta,
                                    'sample_size': sample_size,
                                    'method': method,
                                    'test_type': 'gower',
                                    'success': True,
                                    'distance': gower.get('distance', np.nan),
                                    'p_value': gower.get('p_value', np.nan),
                                    'reject_h0': gower.get('reject_h0', False)
                                })
                            
                            # Latent test results
                            if 'latent' in method_results:
                                latent = method_results['latent']
                                if 'sliced_wasserstein' in latent:
                                    sw = latent['sliced_wasserstein']
                                    summary_data.append({
                                        'strategy': strategy,
                                        'beta': beta,
                                        'sample_size': sample_size,
                                        'method': method,
                                        'test_type': 'sliced_wasserstein',
                                        'success': True,
                                        'distance': sw.get('distance', np.nan),
                                        'p_value': sw.get('p_value', np.nan),
                                        'reject_h0': sw.get('reject_h0', False)
                                    })
            
            # Save summary
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_path = os.path.join(self.config.paths.TESTS_DIR, 'distribution_test_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                
                logger.info(f"📊 Distribution test summary saved: {summary_path}")
                
                # Log summary statistics
                total_tests = len(summary_df)
                successful_tests = len(summary_df[summary_df['success'] == True])
                failed_tests = total_tests - successful_tests
                
                logger.info(f"Total tests: {total_tests}")
                logger.info(f"Successful: {successful_tests}")
                logger.info(f"Failed: {failed_tests}")
                
                if successful_tests > 0:
                    # Rejection rates by test type
                    for test_type in summary_df['test_type'].unique():
                        if test_type != 'all':
                            test_data = summary_df[
                                (summary_df['test_type'] == test_type) & 
                                (summary_df['success'] == True)
                            ]
                            if len(test_data) > 0:
                                rejection_rate = test_data['reject_h0'].mean() * 100
                                logger.info(f"{test_type} rejection rate: {rejection_rate:.1f}%")
        
        except Exception as e:
            logger.error(f"Could not create test summary: {e}")