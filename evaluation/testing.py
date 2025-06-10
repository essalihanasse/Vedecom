"""
Simplified testing system focused on Wasserstein distance ranking and classifier-based validation.
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

logger = logging.getLogger(__name__)

class SimplifiedTester:
    """
    Simplified testing system using Wasserstein distance and classifier-based validation.
    """
    
    def __init__(self, config):
        """
        Initialize simplified tester.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Testing parameters
        self.fast_mode = os.environ.get('VAE_FAST_MODE', '0') == '1'
        self.max_samples = 2000 if not self.fast_mode else 1000
        
        logger.info(f"Simplified tester initialized (fast_mode: {self.fast_mode})")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run simplified tests for all sampling results.
        
        Returns:
            Dictionary with test results and rankings
        """
        logger.info("🧪 Starting simplified testing with Wasserstein distance ranking...")
        
        # Load original data
        try:
            original_df = self._load_original_data()
        except Exception as e:
            logger.error(f"Failed to load original data: {e}")
            return {'error': f'Failed to load original data: {e}'}
        
        all_results = []
        
        # Process each configuration
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            for beta in self.config.training.BETA_VALUES:
                logger.info(f"\n📊 Testing {strategy} strategy, beta={beta}")
                
                config_results = self._test_single_configuration(
                    strategy, beta, original_df
                )
                
                if config_results:
                    all_results.extend(config_results)
        
        # Create rankings and summary
        if all_results:
            self._create_rankings_and_summary(all_results)
            self._create_comparison_plots(all_results)
        
        return {'results': all_results, 'total_tests': len(all_results)}
    
    def _load_original_data(self) -> pd.DataFrame:
        """Load original filtered data."""
        data_paths = [
            os.path.join(self.config.paths.DATA_DIR, 'filtered_data.csv'),
            os.path.join(self.config.paths.DATA_DIR, 'data.csv'),
            self.config.paths.DATA_FILE
        ]
        
        for data_path in data_paths:
            if os.path.exists(data_path):
                logger.info(f"Loading original data from: {data_path}")
                df = pd.read_csv(data_path)
                logger.info(f"Loaded original data: {len(df)} samples")
                return df
        
        raise FileNotFoundError(f"Original data not found in any of: {data_paths}")
    
    def _test_single_configuration(
        self, 
        strategy: str, 
        beta: float, 
        original_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Test a single strategy-beta configuration."""
        
        config_dir = os.path.join(
            self.config.paths.SAMPLES_DIR, 
            strategy, 
            f'beta_{beta}'
        )
        
        if not os.path.exists(config_dir):
            logger.warning(f"No sampling results found for {strategy}-{beta}")
            return []
        
        # Find available methods
        available_methods = self._find_available_methods(config_dir)
        
        if not available_methods:
            logger.warning(f"No valid sampling methods found for {strategy}-{beta}")
            return []
        
        config_results = []
        
        # Test each sample size
        for sample_size in self.config.training.SAMPLE_SIZES:
            logger.info(f"  Sample size: {sample_size}")
            
            # Test each method
            for method in available_methods:
                logger.info(f"    Method: {method}")
                
                try:
                    method_result = self._test_single_method(
                        strategy, beta, sample_size, method, original_df
                    )
                    if method_result:
                        config_results.append(method_result)
                    
                except Exception as e:
                    logger.error(f"Testing failed for {method}: {e}")
        
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
    ) -> Optional[Dict[str, Any]]:
        """Test a single sampling method."""
        
        # Load sampled data
        sampled_df = self._load_sampled_data(strategy, beta, sample_size, method)
        
        if sampled_df is None:
            return None
        
        # Try to get the original indices of sampled points
        sampled_indices = self._get_sampled_indices(strategy, beta, sample_size, method)
        
        # Get latent encodings for Wasserstein distance
        z_original, z_sampled = self._get_latent_encodings(
            original_df, sampled_df, strategy, beta
        )
        
        if z_original is None or z_sampled is None:
            logger.warning(f"Could not get latent encodings for {strategy}-{beta}-{method}")
            return None
        
        # Calculate Wasserstein distance on latent space
        wasserstein_dist = self._calculate_wasserstein_distance(z_original, z_sampled)
        
        # Run classifier-based 2-sample test on original data
        classifier_results = self._run_classifier_test(original_df, sampled_df, sampled_indices)
        
        result = {
            'strategy': strategy,
            'beta': beta,
            'method': method,
            'sample_size': sample_size,
            'n_original': len(z_original),
            'n_sampled': len(z_sampled),
            'wasserstein_distance': wasserstein_dist,
            'balanced_accuracy': classifier_results['balanced_accuracy'],
            'classifier_auc': classifier_results['auc'],
            'classifier_cv_mean': classifier_results['cv_mean'],
            'classifier_cv_std': classifier_results['cv_std'],
            'representativeness_score': classifier_results['representativeness_score'],
            'n_remaining_original': classifier_results['n_remaining_original'],
            'n_representatives': classifier_results['n_representatives'],
            'n_features_used': classifier_results['n_features']
        }
        
        logger.debug(f"    Wasserstein: {wasserstein_dist:.4f}, "
                    f"Balanced Acc: {classifier_results['balanced_accuracy']:.3f}, "
                    f"Repr Score: {classifier_results['representativeness_score']:.3f}")
        
        return result
    
    def _get_sampled_indices(
        self, 
        strategy: str, 
        beta: float, 
        sample_size: int, 
        method: str
    ) -> Optional[List[int]]:
        """Get the original indices of sampled points if available."""
        indices_file = os.path.join(
            self.config.paths.SAMPLES_DIR,
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
            
            # Load model with recovery
            model_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
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
                    logger.error(f"Model recovery failed for {strategy}-{beta}: {e}")
                    return None, None
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            checkpoint = self._validate_and_reconstruct_checkpoint(checkpoint, strategy, beta)
            
            # Create model instance
            cat_dict = checkpoint.get('categorical_cardinality', {})
            if isinstance(cat_dict, dict) and not cat_dict:
                cat_dict = self._load_categorical_info()
            
            model = VAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=checkpoint.get('latent_dim', self.config.model.LATENT_DIM),
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
            z_original = get_latent_encoding(model, original_preprocessed, self.device)
            
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
                
                z_sampled = get_latent_encoding(model, sampled_preprocessed, self.device)
                
            except Exception as e:
                logger.warning(f"Could not map sampled data indices: {e}")
                return None, None
            
            logger.debug(f"Latent encodings obtained: original {z_original.shape}, sampled {z_sampled.shape}")
            return z_original, z_sampled
            
        except Exception as e:
            logger.error(f"Error getting latent encodings: {e}")
            return None, None
    
    def _calculate_wasserstein_distance(
        self, 
        z_original: np.ndarray, 
        z_sampled: np.ndarray
    ) -> float:
        """
        Calculate multidimensional Wasserstein distance.
        Uses the average of 1D Wasserstein distances across dimensions.
        """
        try:
            distances = []
            for dim in range(z_original.shape[1]):
                dist = wasserstein_distance(z_original[:, dim], z_sampled[:, dim])
                distances.append(dist)
            
            return np.mean(distances)
            
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return float('inf')
    
    def _run_classifier_test(
        self, 
        original_df: pd.DataFrame,
        sampled_df: pd.DataFrame,
        sampled_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Run classifier-based 2-sample test on original data.
        
        Compares representative set vs remaining original data (excluding representatives).
        Lower balanced accuracy = better representativeness (classifier can't distinguish)
        Higher balanced accuracy = poor representativeness (classifier can easily distinguish)
        """
        try:
            from sklearn.metrics import balanced_accuracy_score
            
            # Get common numerical columns for classification
            common_cols = self._find_common_columns(original_df, sampled_df)
            if not common_cols:
                logger.warning("No common columns found for classifier test")
                return self._default_classifier_results()
            
            # Prepare representative set data
            representative_data = sampled_df[common_cols].dropna()
            
            # Prepare remaining original data (excluding representatives)
            if sampled_indices is not None:
                # Remove representatives from original data
                remaining_indices = [i for i in range(len(original_df)) if i not in sampled_indices]
                remaining_original_data = original_df.iloc[remaining_indices][common_cols].dropna()
            else:
                # Fallback: use all original data if indices not available
                remaining_original_data = original_df[common_cols].dropna()
                logger.warning("Representative indices not available, using all original data")
            
            if len(representative_data) == 0 or len(remaining_original_data) == 0:
                logger.warning("Empty datasets for classifier test")
                return self._default_classifier_results()
            
            # Subsample if datasets are too large
            max_samples_per_class = self.max_samples // 2
            
            if len(representative_data) > max_samples_per_class:
                representative_data = representative_data.sample(max_samples_per_class, random_state=42)
            
            if len(remaining_original_data) > max_samples_per_class:
                remaining_original_data = remaining_original_data.sample(max_samples_per_class, random_state=42)
            
            # Create labels (0 = remaining original, 1 = representative)
            X = pd.concat([remaining_original_data, representative_data], ignore_index=True)
            y = np.hstack([
                np.zeros(len(remaining_original_data)), 
                np.ones(len(representative_data))
            ])
            
            # Convert to numpy and handle any remaining NaN values
            X_values = X.values
            if np.any(np.isnan(X_values)):
                # Simple imputation with median
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_values = imputer.fit_transform(X_values)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_values)
            
            # Split for testing
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train classifier
            clf = RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced'  # Handle class imbalance
            )
            clf.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Use balanced accuracy instead of regular accuracy
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5  # Default to random classifier performance
            
            # Cross-validation for more robust estimate using balanced accuracy
            cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='balanced_accuracy')
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Representativeness score: 1.0 - (balanced_accuracy - 0.5) * 2
            # Perfect representation = 0.5 balanced accuracy -> score = 1.0
            # Poor representation = 1.0 balanced accuracy -> score = 0.0
            representativeness_score = max(0.0, 1.0 - (cv_mean - 0.5) * 2)
            
            logger.debug(f"Classifier test: Remaining original: {len(remaining_original_data)}, "
                        f"Representatives: {len(representative_data)}, "
                        f"Balanced accuracy: {balanced_acc:.3f}")
            
            return {
                'balanced_accuracy': balanced_acc,
                'accuracy': balanced_acc,  # Keep for backward compatibility
                'auc': auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'representativeness_score': representativeness_score,
                'n_remaining_original': len(remaining_original_data),
                'n_representatives': len(representative_data),
                'n_features': len(common_cols)
            }
            
        except Exception as e:
            logger.error(f"Error in classifier test: {e}")
            return self._default_classifier_results()
    
    def _find_common_columns(self, df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
        """Find common numerical columns between two dataframes."""
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
    
    def _default_classifier_results(self) -> Dict[str, float]:
        """Return default classifier results for error cases."""
        return {
            'balanced_accuracy': 1.0,  # Worst case
            'accuracy': 1.0,
            'auc': 1.0,
            'cv_mean': 1.0,
            'cv_std': 0.0,
            'representativeness_score': 0.0,
            'n_remaining_original': 0,
            'n_representatives': 0,
            'n_features': 0
        }
    
    def _validate_and_reconstruct_checkpoint(self, checkpoint: dict, strategy: str, beta: float) -> dict:
        """Validate checkpoint and reconstruct missing information."""
        required_keys = ['model_state_dict', 'input_dim', 'num_numerical']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            logger.warning(f"Checkpoint missing keys: {missing_keys}")
            checkpoint = self._reconstruct_checkpoint_info(checkpoint, strategy, beta)
        
        return checkpoint
    
    def _reconstruct_checkpoint_info(self, checkpoint: dict, strategy: str, beta: float) -> dict:
        """Reconstruct missing checkpoint information."""
        logger.info("Reconstructing missing checkpoint information")
        
        # Try to load preprocessing objects for missing info
        try:
            preprocessing_path = os.path.join(self.config.paths.DATA_DIR, 'preprocessing_objects.pkl')
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    preprocessing_objects = pickle.load(f)
                
                if 'categorical_cardinality' not in checkpoint:
                    checkpoint['categorical_cardinality'] = preprocessing_objects.get('categorical_cardinality', {})
                
                if 'num_numerical' not in checkpoint:
                    checkpoint['num_numerical'] = len(preprocessing_objects.get('num_cols', []))
            
        except Exception as e:
            logger.warning(f"Could not load preprocessing objects: {e}")
        
        # Set defaults for missing values
        if 'input_dim' not in checkpoint:
            try:
                first_layer = None
                for key in checkpoint['model_state_dict'].keys():
                    if 'encoder' in key and 'weight' in key:
                        first_layer = checkpoint['model_state_dict'][key]
                        break
                
                if first_layer is not None:
                    checkpoint['input_dim'] = first_layer.shape[1]
                    logger.info(f"Inferred input_dim: {checkpoint['input_dim']}")
                else:
                    checkpoint['input_dim'] = self.config.model.HIDDEN_DIM * 2
                    
            except Exception:
                checkpoint['input_dim'] = self.config.model.HIDDEN_DIM * 2
        
        if 'num_numerical' not in checkpoint:
            checkpoint['num_numerical'] = len(self.config.data.NUMERICAL_COLS)
        
        if 'hidden_dim' not in checkpoint:
            checkpoint['hidden_dim'] = self.config.model.HIDDEN_DIM
        
        if 'latent_dim' not in checkpoint:
            checkpoint['latent_dim'] = self.config.model.LATENT_DIM
        
        return checkpoint
    
    def _load_categorical_info(self) -> dict:
        """Load categorical information from preprocessing objects."""
        try:
            preprocessing_path = os.path.join(self.config.paths.DATA_DIR, 'preprocessing_objects.pkl')
            with open(preprocessing_path, 'rb') as f:
                preprocessing_objects = pickle.load(f)
            return preprocessing_objects.get('categorical_cardinality', {})
        except Exception as e:
            logger.warning(f"Could not load categorical info: {e}")
            return {}
    
    def _recover_final_model_from_checkpoint(self, model_dir: str) -> None:
        """Recover final model by copying the best checkpoint."""
        
        checkpoint_pattern = os.path.join(model_dir, "checkpoint_epoch_*_val_loss_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise Exception(f"No checkpoint files found in {model_dir}")
        
        logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        def get_val_loss(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'val_loss_(\d+\.?\d*)', filename)
            return float(match.group(1)) if match else float('inf')
        
        checkpoint_losses = [(f, get_val_loss(f)) for f in checkpoint_files]
        valid_checkpoints = [(f, loss) for f, loss in checkpoint_losses if loss != float('inf')]
        
        if not valid_checkpoints:
            raise Exception("No valid checkpoints found")
        
        best_checkpoint, best_loss = min(valid_checkpoints, key=lambda x: x[1])
        
        final_model_path = os.path.join(model_dir, 'vae_model_final.pth')
        shutil.copy2(best_checkpoint, final_model_path)
        
        if not os.path.exists(final_model_path):
            raise Exception("Failed to copy checkpoint as final model")
        
        file_size = os.path.getsize(final_model_path)
        logger.info(f"Recovered: {os.path.basename(best_checkpoint)} → vae_model_final.pth")
        logger.info(f"   Validation loss: {best_loss:.4f}, Size: {file_size:,} bytes")
    
    def _create_rankings_and_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """Create rankings and summary CSV files."""
        try:
            df = pd.DataFrame(all_results)
            
            # Save detailed results
            results_path = os.path.join(self.config.paths.TESTS_DIR, 'detailed_test_results.csv')
            os.makedirs(self.config.paths.TESTS_DIR, exist_ok=True)
            df.to_csv(results_path, index=False)
            logger.info(f"📊 Detailed results saved: {results_path}")
            
            # Create rankings by Wasserstein distance (lower is better)
            rankings_wasserstein = df.groupby(['method', 'sample_size']).agg({
                'wasserstein_distance': 'mean',
                'strategy': 'count'  # Count as number of experiments
            }).rename(columns={'strategy': 'n_experiments'}).reset_index()
            
            rankings_wasserstein = rankings_wasserstein.sort_values('wasserstein_distance')
            rankings_wasserstein['wasserstein_rank'] = range(1, len(rankings_wasserstein) + 1)
            
            # Create rankings by representativeness score (higher is better)
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
            
            # Calculate overall score (lower is better)
            combined_rankings['overall_score'] = (
                combined_rankings['wasserstein_rank'] + combined_rankings['representativeness_rank']
            ) / 2
            combined_rankings = combined_rankings.sort_values('overall_score')
            combined_rankings['overall_rank'] = range(1, len(combined_rankings) + 1)
            
            # Save rankings
            rankings_path = os.path.join(self.config.paths.TESTS_DIR, 'method_rankings.csv')
            combined_rankings.to_csv(rankings_path, index=False)
            logger.info(f"📊 Method rankings saved: {rankings_path}")
            
            # Create summary statistics
            summary_stats = df.groupby('method').agg({
                'wasserstein_distance': ['mean', 'std', 'min', 'max'],
                'representativeness_score': ['mean', 'std', 'min', 'max'],
                'balanced_accuracy': ['mean', 'std'],
                'strategy': 'count'
            }).round(4)
            
            summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
            summary_stats = summary_stats.rename(columns={'strategy_count': 'total_experiments'})
            summary_stats = summary_stats.reset_index()
            
            summary_path = os.path.join(self.config.paths.TESTS_DIR, 'method_summary_statistics.csv')
            summary_stats.to_csv(summary_path, index=False)
            logger.info(f"📊 Summary statistics saved: {summary_path}")
            
            # Print top rankings
            logger.info("\n🏆 TOP 5 METHODS BY OVERALL RANKING:")
            top_5 = combined_rankings.head(5)
            for _, row in top_5.iterrows():
                logger.info(f"{row['overall_rank']:2d}. {row['method']} (n={row['sample_size']}) - "
                          f"Wasserstein: {row['wasserstein_distance']:.4f}, "
                          f"Repr.Score: {row['representativeness_score']:.3f}")
            
            logger.info(f"\n📁 All results saved to: {self.config.paths.TESTS_DIR}")
            
        except Exception as e:
            logger.error(f"Could not create rankings and summary: {e}")
    
    def _create_comparison_plots(self, all_results: List[Dict[str, Any]]) -> None:
        """Create comparison plots for each sample size."""
        try:
            df = pd.DataFrame(all_results)
            
            if len(df) == 0:
                logger.warning("No data available for plotting")
                return
            
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create plots for each sample size
            sample_sizes = sorted(df['sample_size'].unique())
            
            for sample_size in sample_sizes:
                logger.info(f"Creating comparison plots for sample size {sample_size}")
                self._create_sample_size_plots(df, sample_size)
            
            # Create overall summary plots
            self._create_overall_summary_plots(df)
            
            logger.info(f"📊 Comparison plots saved to: {self.config.paths.TESTS_DIR}")
            
        except Exception as e:
            logger.error(f"Could not create comparison plots: {e}")
    
    def _create_sample_size_plots(self, df: pd.DataFrame, sample_size: int) -> None:
        """Create detailed plots for a specific sample size."""
        try:
            # Filter data for this sample size
            df_size = df[df['sample_size'] == sample_size].copy()
            
            if len(df_size) == 0:
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Method Comparison - Sample Size {sample_size}', fontsize=16, fontweight='bold')
            
            # Plot 1: Wasserstein Distance by Strategy and Beta
            self._plot_wasserstein_by_strategy_beta(df_size, axes[0, 0])
            
            # Plot 2: Balanced Accuracy with Error Bars by Method
            self._plot_accuracy_with_errors(df_size, axes[0, 1])
            
            # Plot 3: Representativeness Score by Method
            self._plot_representativeness_by_method(df_size, axes[1, 0])
            
            # Plot 4: Strategy vs Beta Heatmap (Wasserstein)
            self._plot_strategy_beta_heatmap(df_size, axes[1, 1])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.config.paths.TESTS_DIR, f'comparison_sample_size_{sample_size}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Could not create plots for sample size {sample_size}: {e}")
    
    def _plot_wasserstein_by_strategy_beta(self, df: pd.DataFrame, ax) -> None:
        """Plot Wasserstein distance by strategy and beta values."""
        try:
            # Group by method, strategy, and beta
            plot_data = df.groupby(['method', 'strategy', 'beta']).agg({
                'wasserstein_distance': 'mean'
            }).reset_index()
            
            # Create a bar plot
            methods = plot_data['method'].unique()
            strategies = plot_data['strategy'].unique()
            betas = sorted(plot_data['beta'].unique())
            
            x_pos = np.arange(len(methods))
            width = 0.8 / (len(strategies) * len(betas))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(strategies) * len(betas)))
            
            for i, strategy in enumerate(strategies):
                for j, beta in enumerate(betas):
                    subset = plot_data[(plot_data['strategy'] == strategy) & (plot_data['beta'] == beta)]
                    if len(subset) > 0:
                        # Align data with methods
                        y_values = []
                        for method in methods:
                            method_data = subset[subset['method'] == method]
                            if len(method_data) > 0:
                                y_values.append(method_data['wasserstein_distance'].iloc[0])
                            else:
                                y_values.append(0)
                        
                        offset = (i * len(betas) + j) * width - 0.4 + width/2
                        color_idx = i * len(betas) + j
                        
                        ax.bar(x_pos + offset, y_values, width, 
                              label=f'{strategy}-β{beta}', 
                              color=colors[color_idx], alpha=0.7)
            
            ax.set_xlabel('Sampling Method')
            ax.set_ylabel('Wasserstein Distance')
            ax.set_title('Wasserstein Distance by Strategy & Beta')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Wasserstein Distance by Strategy & Beta (Error)')
    
    def _plot_accuracy_with_errors(self, df: pd.DataFrame, ax) -> None:
        """Plot balanced accuracy with error bars (boxplot style)."""
        try:
            # Group by method to get mean and std
            accuracy_stats = df.groupby('method').agg({
                'balanced_accuracy': ['mean', 'std', 'count'],
                'classifier_cv_std': 'mean'  # Use cross-validation std as error
            }).round(4)
            
            # Flatten column names
            accuracy_stats.columns = ['_'.join(col).strip() for col in accuracy_stats.columns]
            accuracy_stats = accuracy_stats.reset_index()
            
            methods = accuracy_stats['method']
            means = accuracy_stats['balanced_accuracy_mean']
            stds = accuracy_stats['classifier_cv_std_mean']  # Use CV std as error bars
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(methods)), means, yerr=stds, 
                         capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Add value labels on bars
            for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                       f'{mean_val:.3f}±{std_val:.3f}',
                       ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Sampling Method')
            ax.set_ylabel('Balanced Accuracy')
            ax.set_title('Balanced Accuracy ± Std (Lower is Better)')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add reference line at 0.5 (perfect performance)
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect (0.5)')
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Balanced Accuracy ± Std (Error)')
    
    def _plot_representativeness_by_method(self, df: pd.DataFrame, ax) -> None:
        """Plot representativeness scores by method with individual points."""
        try:
            # Create violin plot or box plot
            methods = df['method'].unique()
            data_for_plot = [df[df['method'] == method]['representativeness_score'].values 
                           for method in methods]
            
            # Create violin plot
            parts = ax.violinplot(data_for_plot, range(1, len(methods) + 1), 
                                 showmeans=True, showmedians=True)
            
            # Customize violin plot
            for pc in parts['bodies']:
                pc.set_facecolor('lightcoral')
                pc.set_alpha(0.7)
            
            # Add individual points
            for i, method in enumerate(methods):
                method_data = df[df['method'] == method]['representativeness_score']
                y_positions = np.random.normal(i + 1, 0.04, len(method_data))
                ax.scatter(y_positions, method_data, alpha=0.6, s=20, color='darkred')
            
            ax.set_xlabel('Sampling Method')
            ax.set_ylabel('Representativeness Score')
            ax.set_title('Representativeness Score Distribution (Higher is Better)')
            ax.set_xticks(range(1, len(methods) + 1))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add reference line at 1.0 (perfect performance)
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect (1.0)')
            ax.legend()
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating plot:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Representativeness Score Distribution (Error)')
    
    def _plot_strategy_beta_heatmap(self, df: pd.DataFrame, ax) -> None:
        """Plot heatmap of strategy vs beta values for Wasserstein distance."""
        try:
            # Create pivot table for heatmap
            heatmap_data = df.groupby(['strategy', 'beta']).agg({
                'wasserstein_distance': 'mean'
            }).reset_index()
            
            if len(heatmap_data) == 0:
                ax.text(0.5, 0.5, 'No data for heatmap', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Strategy vs Beta Heatmap (No Data)')
                return
            
            pivot_table = heatmap_data.pivot(index='strategy', columns='beta', values='wasserstein_distance')
            
            # Create heatmap
            sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Wasserstein Distance'})
            
            ax.set_title('Strategy vs Beta: Wasserstein Distance')
            ax.set_xlabel('Beta Value')
            ax.set_ylabel('Annealing Strategy')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Strategy vs Beta Heatmap (Error)')
    
    def _create_overall_summary_plots(self, df: pd.DataFrame) -> None:
        """Create overall summary plots across all sample sizes."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Overall Method Comparison Across All Sample Sizes', fontsize=16, fontweight='bold')
            
            # Plot 1: Wasserstein distance by sample size and method
            self._plot_wasserstein_by_sample_size(df, axes[0, 0])
            
            # Plot 2: Accuracy trends by sample size
            self._plot_accuracy_trends(df, axes[0, 1])
            
            # Plot 3: Method performance correlation
            self._plot_performance_correlation(df, axes[1, 0])
            
            # Plot 4: Overall rankings comparison
            self._plot_overall_rankings(df, axes[1, 1])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.config.paths.TESTS_DIR, 'overall_comparison_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Could not create overall summary plots: {e}")
    
    def _plot_wasserstein_by_sample_size(self, df: pd.DataFrame, ax) -> None:
        """Plot Wasserstein distance trends by sample size."""
        try:
            methods = df['method'].unique()
            sample_sizes = sorted(df['sample_size'].unique())
            
            for method in methods:
                method_data = df[df['method'] == method]
                size_means = []
                size_stds = []
                
                for size in sample_sizes:
                    size_data = method_data[method_data['sample_size'] == size]['wasserstein_distance']
                    if len(size_data) > 0:
                        size_means.append(size_data.mean())
                        size_stds.append(size_data.std() if len(size_data) > 1 else 0)
                    else:
                        size_means.append(np.nan)
                        size_stds.append(0)
                
                ax.errorbar(sample_sizes, size_means, yerr=size_stds, 
                           label=method, marker='o', capsize=3)
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Wasserstein Distance')
            ax.set_title('Wasserstein Distance by Sample Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_accuracy_trends(self, df: pd.DataFrame, ax) -> None:
        """Plot balanced accuracy trends by sample size."""
        try:
            methods = df['method'].unique()
            sample_sizes = sorted(df['sample_size'].unique())
            
            for method in methods:
                method_data = df[df['method'] == method]
                size_means = []
                size_stds = []
                
                for size in sample_sizes:
                    size_data = method_data[method_data['sample_size'] == size]['balanced_accuracy']
                    if len(size_data) > 0:
                        size_means.append(size_data.mean())
                        size_stds.append(size_data.std() if len(size_data) > 1 else 0)
                    else:
                        size_means.append(np.nan)
                        size_stds.append(0)
                
                ax.errorbar(sample_sizes, size_means, yerr=size_stds, 
                           label=method, marker='s', capsize=3)
            
            ax.set_xlabel('Sample Size')
            ax.set_ylabel('Balanced Accuracy')
            ax.set_title('Balanced Accuracy by Sample Size')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Perfect (0.5)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_performance_correlation(self, df: pd.DataFrame, ax) -> None:
        """Plot correlation between Wasserstein distance and representativeness score."""
        try:
            methods = df['method'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
            
            for i, method in enumerate(methods):
                method_data = df[df['method'] == method]
                ax.scatter(method_data['wasserstein_distance'], 
                          method_data['representativeness_score'],
                          label=method, alpha=0.7, color=colors[i], s=50)
            
            ax.set_xlabel('Wasserstein Distance')
            ax.set_ylabel('Representativeness Score')
            ax.set_title('Performance Correlation\n(Bottom-right is best)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add quadrant lines
            ax.axhline(y=df['representativeness_score'].median(), color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=df['wasserstein_distance'].median(), color='gray', linestyle=':', alpha=0.5)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_overall_rankings(self, df: pd.DataFrame, ax) -> None:
        """Plot overall method rankings."""
        try:
            # Calculate simple overall scores
            method_scores = df.groupby('method').agg({
                'wasserstein_distance': 'mean',
                'representativeness_score': 'mean'
            }).reset_index()
            
            # Normalize scores (lower Wasserstein is better, higher representativeness is better)
            method_scores['wasserstein_rank'] = method_scores['wasserstein_distance'].rank()
            method_scores['repr_rank'] = method_scores['representativeness_score'].rank(ascending=False)
            method_scores['overall_score'] = (method_scores['wasserstein_rank'] + method_scores['repr_rank']) / 2
            method_scores = method_scores.sort_values('overall_score')
            
            # Create horizontal bar plot
            y_pos = np.arange(len(method_scores))
            ax.barh(y_pos, method_scores['overall_score'], alpha=0.7, color='lightblue')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(method_scores['method'])
            ax.set_xlabel('Overall Score (Lower is Better)')
            ax.set_title('Overall Method Rankings')
            ax.grid(True, alpha=0.3)
            
            # Add score labels
            for i, score in enumerate(method_scores['overall_score']):
                ax.text(score + 0.05, i, f'{score:.2f}', va='center')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)