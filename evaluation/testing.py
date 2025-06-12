"""
Complete enhanced testing system with classifier test implementation.
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
    Base tester class with classifier test implementation.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = 50000  # Limit for efficiency
        
    def _load_original_data(self) -> pd.DataFrame:
        """Load original data for testing."""
        if os.path.exists(self.config.paths.DATA_FILE):
            return pd.read_csv(self.config.paths.DATA_FILE)
        elif os.path.exists(self.config.paths.PREPROCESSED_FILE):
            return pd.read_csv(self.config.paths.PREPROCESSED_FILE)
        else:
            raise FileNotFoundError("No original data file found")
    
    def _find_available_methods(self, config_dir: str) -> List[str]:
        """Find available sampling methods in the given directory."""
        if not os.path.exists(config_dir):
            return []
        
        methods = []
        for item in os.listdir(config_dir):
            if os.path.isdir(os.path.join(config_dir, item)) and item.startswith('method_'):
                method_name = item.replace('method_', '')
                methods.append(method_name)
        
        return methods
    
    def _run_classifier_test(
        self, 
        original_df: pd.DataFrame, 
        sampled_df: pd.DataFrame, 
        sampled_indices: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Run classifier-based 2-sample test to evaluate sampling quality.
        
        This test trains a classifier to distinguish between original and sampled data.
        If the sampling is good, the classifier should perform poorly (around 50% accuracy).
        
        Args:
            original_df: Original dataset
            sampled_df: Sampled dataset  
            sampled_indices: Optional indices of sampled points
            
        Returns:
            Dictionary with classifier test results
        """
        try:
            # Prepare data for classification
            n_original = min(len(original_df), self.max_samples)
            n_sampled = min(len(sampled_df), self.max_samples)
            
            # Sample from original data if too large
            if len(original_df) > n_original:
                original_subset = original_df.sample(n=n_original, random_state=42)
            else:
                original_subset = original_df.copy()
            
            # Sample from sampled data if too large
            if len(sampled_df) > n_sampled:
                sampled_subset = sampled_df.sample(n=n_sampled, random_state=42)
            else:
                sampled_subset = sampled_df.copy()
            
            # Create labels (0 = original, 1 = sampled)
            original_labels = np.zeros(len(original_subset))
            sampled_labels = np.ones(len(sampled_subset))
            
            # Combine data and labels
            X = pd.concat([original_subset, sampled_subset], ignore_index=True)
            y = np.concatenate([original_labels, sampled_labels])
            
            # Handle non-numeric columns
            X_numeric = X.select_dtypes(include=[np.number])
            
            if X_numeric.empty:
                logger.warning("No numeric columns found for classifier test")
                return {
                    'balanced_accuracy': 0.5,
                    'accuracy': 0.5,
                    'roc_auc': 0.5,
                    'cross_val_mean': 0.5,
                    'cross_val_std': 0.0,
                    'n_features': 0,
                    'error': 'No numeric features'
                }
            
            # Handle missing values
            X_numeric = X_numeric.fillna(X_numeric.mean())
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Train Random Forest classifier
            classifier = RandomForestClassifier(
                n_estimators=20,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            
            classifier.fit(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            y_pred_proba = classifier.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            
            # ROC AUC (handle edge case where all samples are from one class)
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                roc_auc = 0.5
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                classifier, X_scaled, y, cv=5, scoring='balanced_accuracy'
            )
            
            # Feature importance (top 5)
            feature_importance = classifier.feature_importances_
            top_features = np.argsort(feature_importance)[-5:][::-1]
            
            results = {
                'balanced_accuracy': float(balanced_acc),
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'cross_val_mean': float(cv_scores.mean()),
                'cross_val_std': float(cv_scores.std()),
                'n_features': int(X_numeric.shape[1]),
                'n_samples_original': int(len(original_subset)),
                'n_samples_sampled': int(len(sampled_subset)),
                'top_feature_indices': top_features.tolist(),
                'feature_importance_top5': feature_importance[top_features].tolist()
            }
            
            # Interpretation
            if balanced_acc < 0.6:
                interpretation = "Excellent sampling (classifier cannot distinguish)"
            elif balanced_acc < 0.7:
                interpretation = "Good sampling (some distinguishability)"
            elif balanced_acc < 0.8:
                interpretation = "Fair sampling (noticeable differences)"
            else:
                interpretation = "Poor sampling (easily distinguishable)"
            
            results['interpretation'] = interpretation
            
            logger.debug(f"Classifier test completed: balanced_acc={balanced_acc:.3f} ({interpretation})")
            
            return results
            
        except Exception as e:
            logger.error(f"Classifier test failed: {e}")
            return {
                'balanced_accuracy': 0.5,
                'accuracy': 0.5,
                'roc_auc': 0.5,
                'cross_val_mean': 0.5,
                'cross_val_std': 0.0,
                'error': str(e)
            }
    
    def _load_categorical_info(self) -> Dict[str, Any]:
        """Load categorical information from preprocessing objects."""
        try:
            preprocessing_path = os.path.join(self.config.paths.DATA_DIR, 'preprocessing_objects.pkl')
            if os.path.exists(preprocessing_path):
                with open(preprocessing_path, 'rb') as f:
                    preprocessing_objects = pickle.load(f)
                return preprocessing_objects.get('categorical_cardinality', {})
        except Exception as e:
            logger.warning(f"Could not load categorical info: {e}")
        
        return {}
    
    def _validate_and_reconstruct_checkpoint(
        self, 
        checkpoint: Dict[str, Any], 
        strategy: str, 
        beta: float
    ) -> Dict[str, Any]:
        """Validate and reconstruct checkpoint if needed."""
        required_keys = ['model_state_dict', 'input_dim', 'num_numerical']
        
        for key in required_keys:
            if key not in checkpoint:
                logger.warning(f"Missing key in checkpoint: {key}")
                
                # Try to reconstruct missing information
                if key == 'input_dim':
                    checkpoint['input_dim'] = 64  # Default fallback
                elif key == 'num_numerical':
                    checkpoint['num_numerical'] = 8  # Default fallback
        
        # Ensure categorical_cardinality exists
        if 'categorical_cardinality' not in checkpoint:
            checkpoint['categorical_cardinality'] = self._load_categorical_info()
        
        return checkpoint
    
    def _recover_final_model_from_checkpoint(self, model_dir: str) -> None:
        """Recover final model from best checkpoint."""
        checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {model_dir}")
        
        def get_val_loss(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'val_loss_(\d+\.?\d*)', filename)
            return float(match.group(1)) if match else float('inf')
        
        best_checkpoint = min(checkpoint_files, key=get_val_loss)
        final_path = os.path.join(model_dir, 'vae_model_final.pth')
        shutil.copy2(best_checkpoint, final_path)
        
        logger.info(f"Recovered model from: {os.path.basename(best_checkpoint)}")


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
        
        logger.info(f"    Latent Dim {latent_dim}: Wasserstein: {metrics.get('wasserstein_distance', 0):.4f}, "
                   f"Balanced Acc: {classifier_results['balanced_accuracy']:.3f} "
                   f"({classifier_results.get('interpretation', 'Unknown')})")
        
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
            
            # Create classifier test summary
            self._create_classifier_test_summary(df, analysis_dir, latent_dim)
            
            logger.info(f"📊 Analysis created for latent dimension {latent_dim}")
            
        except Exception as e:
            logger.warning(f"Could not create analysis for latent_dim {latent_dim}: {e}")
    
    def _create_classifier_test_summary(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create detailed classifier test summary and visualizations."""
        try:
            # Filter out rows with classifier test errors
            valid_df = df[~df['balanced_accuracy'].isna() & (df['balanced_accuracy'] != 0.5)]
            
            if len(valid_df) == 0:
                logger.warning(f"No valid classifier test results for latent dimension {latent_dim}")
                return
            
            # Create classifier test summary
            classifier_summary = valid_df.groupby(['method', 'sample_size']).agg({
                'balanced_accuracy': ['mean', 'std', 'min', 'max'],
                'accuracy': ['mean', 'std'],
                'roc_auc': ['mean', 'std'],
                'cross_val_mean': ['mean', 'std'],
                'interpretation': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(4)
            
            # Flatten column names
            classifier_summary.columns = ['_'.join(col).strip() for col in classifier_summary.columns]
            classifier_summary = classifier_summary.reset_index()
            
            # Save classifier summary
            classifier_summary.to_csv(
                os.path.join(analysis_dir, f'latent_{latent_dim}_classifier_test_summary.csv'), 
                index=False
            )
            
            # Create classifier test visualization
            self._create_classifier_test_plots(valid_df, analysis_dir, latent_dim)
            
            # Log classifier test insights
            best_method = classifier_summary.loc[
                classifier_summary['balanced_accuracy_mean'].idxmin()
            ]
            
            logger.info(f"\n🎯 CLASSIFIER TEST INSIGHTS - Latent Dimension {latent_dim}:")
            logger.info(f"   Best method: {best_method['method']} (sample_size: {best_method['sample_size']})")
            logger.info(f"   Balanced accuracy: {best_method['balanced_accuracy_mean']:.3f} ± {best_method['balanced_accuracy_std']:.3f}")
            logger.info(f"   Interpretation: {best_method['interpretation_<lambda>']}")
            
        except Exception as e:
            logger.warning(f"Could not create classifier test summary for latent_dim {latent_dim}: {e}")
    
    def _create_classifier_test_plots(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create classifier test specific plots."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Balanced accuracy by method
            method_means = df.groupby('method')['balanced_accuracy'].agg(['mean', 'std'])
            methods = method_means.index
            means = method_means['mean']
            stds = method_means['std']
            
            bars = ax1.bar(range(len(methods)), means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            ax1.set_title(f'Classifier Test: Balanced Accuracy by Method\n(Latent Dim {latent_dim})')
            ax1.set_xlabel('Sampling Method')
            ax1.set_ylabel('Balanced Accuracy')
            ax1.set_xticks(range(len(methods)))
            ax1.set_xticklabels(methods, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Sample size effects on classifier performance
            sample_effects = df.groupby('sample_size')['balanced_accuracy'].agg(['mean', 'std'])
            ax2.errorbar(sample_effects.index, sample_effects['mean'], 
                        yerr=sample_effects['std'], marker='o', capsize=5, linewidth=2)
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            ax2.set_title(f'Sample Size Effects on Classifier Performance\n(Latent Dim {latent_dim})')
            ax2.set_xlabel('Sample Size')
            ax2.set_ylabel('Balanced Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: ROC AUC vs Balanced Accuracy scatter
            scatter = ax3.scatter(df['balanced_accuracy'], df['roc_auc'], 
                                 c=df['sample_size'], cmap='viridis', alpha=0.6, s=50)
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            ax3.axvline(x=0.5, color='red', linestyle='--', alpha=0.7)
            ax3.set_title(f'ROC AUC vs Balanced Accuracy\n(Latent Dim {latent_dim})')
            ax3.set_xlabel('Balanced Accuracy')
            ax3.set_ylabel('ROC AUC')
            plt.colorbar(scatter, ax=ax3, label='Sample Size')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Interpretation distribution
            interpretation_counts = df['interpretation'].value_counts()
            colors = ['green', 'lightgreen', 'yellow', 'orange', 'red'][:len(interpretation_counts)]
            ax4.pie(interpretation_counts.values, labels=interpretation_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            ax4.set_title(f'Sampling Quality Distribution\n(Latent Dim {latent_dim})')
            
            plt.suptitle(f'Classifier Test Analysis - Latent Dimension {latent_dim}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'latent_{latent_dim}_classifier_test_plots.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create classifier test plots for latent_dim {latent_dim}: {e}")
    
    def _create_latent_dim_rankings(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create rankings for specific latent dimension including classifier metrics."""
        try:
            # Create comprehensive rankings including classifier metrics
            rankings = df.groupby(['method', 'sample_size']).agg({
                'wasserstein_distance': 'mean',
                'representativeness_score': 'mean',
                'balanced_accuracy': 'mean',
                'roc_auc': 'mean',
                'cross_val_mean': 'mean',
                'strategy': 'count'
            }).rename(columns={'strategy': 'n_experiments'}).reset_index()
            
            # Calculate composite ranks
            rankings['wasserstein_rank'] = rankings['wasserstein_distance'].rank()
            rankings['representativeness_rank'] = rankings['representativeness_score'].rank(ascending=False)
            rankings['classifier_rank'] = (1 - rankings['balanced_accuracy']).rank()  # Lower balanced_acc is better
            
            # Calculate overall score (equal weighting)
            rankings['overall_score'] = (
                rankings['wasserstein_rank'] + 
                rankings['representativeness_rank'] + 
                rankings['classifier_rank']
            ) / 3
            
            rankings = rankings.sort_values('overall_score')
            rankings['overall_rank'] = range(1, len(rankings) + 1)
            
            # Save comprehensive rankings
            rankings_path = os.path.join(analysis_dir, f'latent_{latent_dim}_comprehensive_rankings.csv')
            rankings.to_csv(rankings_path, index=False)
            
            # Log top rankings for this latent dimension
            logger.info(f"\n🏆 TOP 3 METHODS FOR LATENT DIMENSION {latent_dim} (including classifier test):")
            top_3 = rankings.head(3)
            for _, row in top_3.iterrows():
                logger.info(f"{row['overall_rank']:2d}. {row['method']} (n={row['sample_size']}) - "
                          f"Wasserstein: {row['wasserstein_distance']:.4f}, "
                          f"Repr.Score: {row['representativeness_score']:.3f}, "
                          f"Classifier Bal.Acc: {row['balanced_accuracy']:.3f}")
            
        except Exception as e:
            logger.warning(f"Could not create comprehensive rankings for latent_dim {latent_dim}: {e}")
    
    def _create_latent_dim_plots(self, df: pd.DataFrame, analysis_dir: str, latent_dim: int) -> None:
        """Create plots for specific latent dimension."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Method performance comparison (now includes classifier results)
            method_performance = df.groupby('method').agg({
                'wasserstein_distance': ['mean', 'std'],
                'representativeness_score': ['mean', 'std'],
                'balanced_accuracy': ['mean', 'std']
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
            
            # Plot 2: Classifier balanced accuracy by method
            balanced_acc_means = method_performance[('balanced_accuracy', 'mean')]
            balanced_acc_stds = method_performance[('balanced_accuracy', 'std')]
            
            axes[0, 1].errorbar(range(len(methods)), balanced_acc_means, yerr=balanced_acc_stds,
                              marker='s', capsize=5, capthick=2, color='orange')
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            axes[0, 1].set_title(f'Classifier Balanced Accuracy by Method\n(Latent Dim {latent_dim})')
            axes[0, 1].set_xlabel('Method')
            axes[0, 1].set_ylabel('Balanced Accuracy')
            axes[0, 1].set_xticks(range(len(methods)))
            axes[0, 1].set_xticklabels(methods, rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Sample size effects on both metrics
            sample_effects_wass = df.groupby('sample_size')['wasserstein_distance'].mean()
            sample_effects_class = df.groupby('sample_size')['balanced_accuracy'].mean()
            
            ax3_twin = axes[1, 0].twinx()
            
            line1 = axes[1, 0].plot(sample_effects_wass.index, sample_effects_wass.values, 
                                   'o-', linewidth=2, markersize=8, color='blue', label='Wasserstein')
            line2 = ax3_twin.plot(sample_effects_class.index, sample_effects_class.values, 
                                 's-', linewidth=2, markersize=8, color='orange', label='Balanced Acc')
            ax3_twin.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            axes[1, 0].set_title(f'Sample Size Effects\n(Latent Dim {latent_dim})')
            axes[1, 0].set_xlabel('Sample Size')
            axes[1, 0].set_ylabel('Wasserstein Distance', color='blue')
            ax3_twin.set_ylabel('Balanced Accuracy', color='orange')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 0].legend(lines, labels, loc='upper left')
            
            # Plot 4: Correlation between Wasserstein and Classifier performance
            axes[1, 1].scatter(df['wasserstein_distance'], df['balanced_accuracy'], 
                             alpha=0.6, s=60, c=df['sample_size'], cmap='viridis')
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            axes[1, 1].set_title(f'Wasserstein vs Classifier Performance\n(Latent Dim {latent_dim})')
            axes[1, 1].set_xlabel('Wasserstein Distance')
            axes[1, 1].set_ylabel('Balanced Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar for sample size
            scatter = axes[1, 1].collections[0]
            plt.colorbar(scatter, ax=axes[1, 1], label='Sample Size')
            
            plt.suptitle(f'Latent Dimension {latent_dim} Testing Analysis (with Classifier Test)', 
                        fontsize=16, fontweight='bold')
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
            
            # Create cross-latent dimension plots (now includes classifier metrics)
            self._create_cross_latent_plots(df, cross_analysis_dir)
            
            # Create latent dimension scaling analysis
            self._create_latent_scaling_analysis(df, cross_analysis_dir)
            
            logger.info("📊 Cross-latent dimension analysis created")
            
        except Exception as e:
            logger.warning(f"Could not create cross-latent dimension analysis: {e}")
    
    def _create_cross_latent_plots(self, df: pd.DataFrame, analysis_dir: str) -> None:
        """Create plots comparing performance across latent dimensions (now includes classifier metrics)."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Plot 1: Wasserstein performance vs latent dimension
            latent_performance = df.groupby('latent_dim').agg({
                'wasserstein_distance': ['mean', 'std'],
                'representativeness_score': ['mean', 'std'],
                'balanced_accuracy': ['mean', 'std']
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
            
            # Plot 2: Classifier performance vs latent dimension
            classifier_means = [latent_performance.loc[dim, ('balanced_accuracy', 'mean')] for dim in latent_dims]
            classifier_stds = [latent_performance.loc[dim, ('balanced_accuracy', 'std')] for dim in latent_dims]
            
            axes[0, 1].errorbar(latent_dims, classifier_means, yerr=classifier_stds,
                              marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
            axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            axes[0, 1].set_title('Classifier Performance vs Latent Dimension')
            axes[0, 1].set_xlabel('Latent Dimension')
            axes[0, 1].set_ylabel('Mean Balanced Accuracy')
            axes[0, 1].set_xscale('log', base=2)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Representativeness vs latent dimension
            repr_means = [latent_performance.loc[dim, ('representativeness_score', 'mean')] for dim in latent_dims]
            repr_stds = [latent_performance.loc[dim, ('representativeness_score', 'std')] for dim in latent_dims]
            
            axes[0, 2].errorbar(latent_dims, repr_means, yerr=repr_stds,
                              marker='^', capsize=5, capthick=2, linewidth=2, color='green')
            axes[0, 2].set_title('Representativeness Score vs Latent Dimension')
            axes[0, 2].set_xlabel('Latent Dimension')
            axes[0, 2].set_ylabel('Mean Representativeness Score')
            axes[0, 2].set_xscale('log', base=2)
            axes[0, 2].grid(True, alpha=0.3)
            
            # Plot 4: Method performance across latent dimensions (Wasserstein)
            methods = df['method'].unique()
            for method in methods:
                method_data = df[df['method'] == method]
                method_performance = method_data.groupby('latent_dim')['wasserstein_distance'].mean()
                axes[1, 0].plot(method_performance.index, method_performance.values, 
                              'o-', label=method, linewidth=2)
            
            axes[1, 0].set_title('Method Performance Across Latent Dimensions (Wasserstein)')
            axes[1, 0].set_xlabel('Latent Dimension')
            axes[1, 0].set_ylabel('Mean Wasserstein Distance')
            axes[1, 0].set_xscale('log', base=2)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Method performance across latent dimensions (Classifier)
            for method in methods:
                method_data = df[df['method'] == method]
                method_performance = method_data.groupby('latent_dim')['balanced_accuracy'].mean()
                axes[1, 1].plot(method_performance.index, method_performance.values, 
                              's-', label=method, linewidth=2)
            
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            axes[1, 1].set_title('Method Performance Across Latent Dimensions (Classifier)')
            axes[1, 1].set_xlabel('Latent Dimension')
            axes[1, 1].set_ylabel('Mean Balanced Accuracy')
            axes[1, 1].set_xscale('log', base=2)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Correlation between metrics
            valid_df = df.dropna(subset=['wasserstein_distance', 'balanced_accuracy'])
            if len(valid_df) > 0:
                scatter = axes[1, 2].scatter(valid_df['wasserstein_distance'], valid_df['balanced_accuracy'], 
                                           c=valid_df['latent_dim'], cmap='viridis', alpha=0.6, s=50)
                axes[1, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
                axes[1, 2].set_title('Wasserstein vs Classifier Performance')
                axes[1, 2].set_xlabel('Wasserstein Distance')
                axes[1, 2].set_ylabel('Balanced Accuracy')
                plt.colorbar(scatter, ax=axes[1, 2], label='Latent Dimension')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                axes[1, 2].text(0.5, 0.5, 'No valid data\nfor correlation plot', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Wasserstein vs Classifier Performance')
            
            plt.suptitle('Cross-Latent Dimension Performance Analysis (with Classifier Test)', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'cross_latent_performance_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create cross-latent plots: {e}")
    
    def _create_latent_scaling_analysis(self, df: pd.DataFrame, analysis_dir: str) -> None:
        """Create analysis of how sampling methods scale with latent dimension."""
        try:
            # Scaling analysis for each method (now includes classifier metrics)
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
                        'mean_roc_auc': dim_data['roc_auc'].mean(),
                        'std_wasserstein': dim_data['wasserstein_distance'].std(),
                        'std_balanced_accuracy': dim_data['balanced_accuracy'].std(),
                        'n_experiments': len(dim_data)
                    })
            
            scaling_df = pd.DataFrame(scaling_data)
            scaling_df.to_csv(os.path.join(analysis_dir, 'latent_scaling_analysis.csv'), index=False)
            
            # Create enhanced scaling plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
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
            
            # Classifier performance scaling
            for method in scaling_df['method'].unique():
                method_data = scaling_df[scaling_df['method'] == method]
                ax2.plot(method_data['latent_dim'], method_data['mean_balanced_accuracy'], 
                        's-', label=method, linewidth=2)
            
            ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Performance')
            ax2.set_title('Classifier Performance Scaling')
            ax2.set_xlabel('Latent Dimension')
            ax2.set_ylabel('Mean Balanced Accuracy')
            ax2.set_xscale('log', base=2)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Representativeness scaling
            for method in scaling_df['method'].unique():
                method_data = scaling_df[scaling_df['method'] == method]
                ax3.plot(method_data['latent_dim'], method_data['mean_representativeness'], 
                        '^-', label=method, linewidth=2)
            
            ax3.set_title('Representativeness Score Scaling')
            ax3.set_xlabel('Latent Dimension')
            ax3.set_ylabel('Mean Representativeness Score')
            ax3.set_xscale('log', base=2)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Performance variability across latent dimensions
            for method in scaling_df['method'].unique():
                method_data = scaling_df[scaling_df['method'] == method]
                ax4.plot(method_data['latent_dim'], method_data['std_balanced_accuracy'], 
                        'o-', label=f'{method} (Classifier)', linewidth=2)
                ax4.plot(method_data['latent_dim'], method_data['std_wasserstein'], 
                        '--', label=f'{method} (Wasserstein)', linewidth=2, alpha=0.7)
            
            ax4.set_title('Performance Variability Scaling')
            ax4.set_xlabel('Latent Dimension')
            ax4.set_ylabel('Standard Deviation')
            ax4.set_xscale('log', base=2)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Method Scaling Analysis Across Latent Dimensions (with Classifier Test)', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'method_scaling_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create scaling analysis: {e}")
    
    def _create_enhanced_rankings_and_summary(self, all_results: Dict[int, List[Dict[str, Any]]]) -> None:
        """Create enhanced rankings considering all latent dimensions and classifier test."""
        try:
            # Combine all results
            combined_data = []
            for latent_dim, results in all_results.items():
                combined_data.extend(results)
            
            if not combined_data:
                logger.warning("No data for enhanced rankings")
                return
            
            df = pd.DataFrame(combined_data)
            
            # Create overall rankings across all latent dimensions (now includes classifier metrics)
            overall_rankings = df.groupby(['method', 'sample_size']).agg({
                'wasserstein_distance': 'mean',
                'representativeness_score': 'mean',
                'balanced_accuracy': 'mean',
                'roc_auc': 'mean',
                'cross_val_mean': 'mean',
                'latent_dim': 'count'
            }).rename(columns={'latent_dim': 'n_experiments'}).reset_index()
            
            # Calculate ranks (lower balanced_accuracy is better for sampling quality)
            overall_rankings['wasserstein_rank'] = overall_rankings['wasserstein_distance'].rank()
            overall_rankings['representativeness_rank'] = overall_rankings['representativeness_score'].rank(ascending=False)
            overall_rankings['classifier_rank'] = (1 - overall_rankings['balanced_accuracy']).rank()
            
            # Calculate overall score (equal weighting of all three metrics)
            overall_rankings['overall_score'] = (
                overall_rankings['wasserstein_rank'] + 
                overall_rankings['representativeness_rank'] + 
                overall_rankings['classifier_rank']
            ) / 3
            
            overall_rankings = overall_rankings.sort_values('overall_score')
            overall_rankings['overall_rank'] = range(1, len(overall_rankings) + 1)
            
            # Save enhanced rankings
            rankings_path = os.path.join(self.config.paths.TESTS_DIR, 'enhanced_method_rankings_with_classifier.csv')
            overall_rankings.to_csv(rankings_path, index=False)
            
            # Create method summary across all latent dimensions (including classifier metrics)
            method_summary = df.groupby('method').agg({
                'wasserstein_distance': ['mean', 'std', 'min', 'max'],
                'representativeness_score': ['mean', 'std', 'min', 'max'],
                'balanced_accuracy': ['mean', 'std', 'min', 'max'],
                'roc_auc': ['mean', 'std'],
                'cross_val_mean': ['mean', 'std'],
                'latent_dim': ['count', 'nunique']
            }).round(4)
            
            method_summary.columns = ['_'.join(col).strip() for col in method_summary.columns]
            method_summary = method_summary.reset_index()
            
            summary_path = os.path.join(self.config.paths.TESTS_DIR, 'enhanced_method_summary_with_classifier.csv')
            method_summary.to_csv(summary_path, index=False)
            
            # Create comprehensive performance report
            self._create_comprehensive_performance_report(df, overall_rankings)
            
            # Log enhanced results
            logger.info("\n🏆 TOP 5 METHODS ACROSS ALL LATENT DIMENSIONS (with Classifier Test):")
            top_5 = overall_rankings.head(5)
            for _, row in top_5.iterrows():
                logger.info(f"{row['overall_rank']:2d}. {row['method']} (n={row['sample_size']}) - "
                          f"Wasserstein: {row['wasserstein_distance']:.4f}, "
                          f"Repr.Score: {row['representativeness_score']:.3f}, "
                          f"Classifier Bal.Acc: {row['balanced_accuracy']:.3f}, "
                          f"Experiments: {row['n_experiments']}")
            
            logger.info(f"\n📁 Enhanced results with classifier test saved to: {self.config.paths.TESTS_DIR}")
            
        except Exception as e:
            logger.warning(f"Could not create enhanced rankings: {e}")
    
    def _create_comprehensive_performance_report(self, df: pd.DataFrame, rankings: pd.DataFrame) -> None:
        """Create a comprehensive performance report with insights."""
        try:
            report_path = os.path.join(self.config.paths.TESTS_DIR, 'comprehensive_performance_report.txt')
            
            with open(report_path, 'w') as f:
                f.write("COMPREHENSIVE VAE SAMPLING EVALUATION REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                # Overall statistics
                f.write("OVERALL STATISTICS:\n")
                f.write(f"Total evaluations: {len(df)}\n")
                f.write(f"Latent dimensions tested: {sorted(df['latent_dim'].unique())}\n")
                f.write(f"Sampling methods: {list(df['method'].unique())}\n")
                f.write(f"Sample sizes: {sorted(df['sample_size'].unique())}\n\n")
                
                # Best performing configurations
                f.write("TOP PERFORMING CONFIGURATIONS:\n")
                top_configs = rankings.head(10)
                for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                    f.write(f"{i:2d}. {row['method']} (sample_size={row['sample_size']}) - "
                          f"Overall Score: {row['overall_score']:.3f}\n")
                    f.write(f"    Wasserstein: {row['wasserstein_distance']:.4f} "
                          f"(rank {row['wasserstein_rank']:.0f})\n")
                    f.write(f"    Representativeness: {row['representativeness_score']:.3f} "
                          f"(rank {row['representativeness_rank']:.0f})\n")
                    f.write(f"    Classifier Bal.Acc: {row['balanced_accuracy']:.3f} "
                          f"(rank {row['classifier_rank']:.0f})\n")
                    f.write(f"    Experiments: {row['n_experiments']}\n\n")
                
                # Method-wise analysis
                f.write("METHOD-WISE ANALYSIS:\n")
                method_analysis = df.groupby('method').agg({
                    'wasserstein_distance': ['mean', 'std'],
                    'balanced_accuracy': ['mean', 'std'],
                    'representativeness_score': ['mean', 'std']
                }).round(4)
                
                for method in df['method'].unique():
                    f.write(f"\n{method.upper()}:\n")
                    wass_mean = method_analysis.loc[method, ('wasserstein_distance', 'mean')]
                    wass_std = method_analysis.loc[method, ('wasserstein_distance', 'std')]
                    bal_acc_mean = method_analysis.loc[method, ('balanced_accuracy', 'mean')]
                    bal_acc_std = method_analysis.loc[method, ('balanced_accuracy', 'std')]
                    repr_mean = method_analysis.loc[method, ('representativeness_score', 'mean')]
                    repr_std = method_analysis.loc[method, ('representativeness_score', 'std')]
                    
                    f.write(f"  Wasserstein Distance: {wass_mean:.4f} ± {wass_std:.4f}\n")
                    f.write(f"  Classifier Bal.Acc: {bal_acc_mean:.3f} ± {bal_acc_std:.3f}\n")
                    f.write(f"  Representativeness: {repr_mean:.3f} ± {repr_std:.3f}\n")
                    
                    # Interpret classifier performance
                    if bal_acc_mean < 0.6:
                        interpretation = "Excellent (indistinguishable from original)"
                    elif bal_acc_mean < 0.7:
                        interpretation = "Good (slight distinguishability)"
                    elif bal_acc_mean < 0.8:
                        interpretation = "Fair (noticeable differences)"
                    else:
                        interpretation = "Poor (easily distinguishable)"
                    
                    f.write(f"  Sampling Quality: {interpretation}\n")
                
                # Latent dimension effects
                f.write("\nLATENT DIMENSION EFFECTS:\n")
                latent_analysis = df.groupby('latent_dim').agg({
                    'wasserstein_distance': 'mean',
                    'balanced_accuracy': 'mean',
                    'representativeness_score': 'mean'
                }).round(4)
                
                for latent_dim in sorted(df['latent_dim'].unique()):
                    f.write(f"\nLatent Dimension {latent_dim}:\n")
                    wass = latent_analysis.loc[latent_dim, 'wasserstein_distance']
                    bal_acc = latent_analysis.loc[latent_dim, 'balanced_accuracy']
                    repr_score = latent_analysis.loc[latent_dim, 'representativeness_score']
                    
                    f.write(f"  Mean Wasserstein: {wass:.4f}\n")
                    f.write(f"  Mean Classifier Bal.Acc: {bal_acc:.3f}\n")
                    f.write(f"  Mean Representativeness: {repr_score:.3f}\n")
                
                # Key insights
                f.write("\nKEY INSIGHTS:\n")
                
                # Best method overall
                best_method = rankings.iloc[0]['method']
                f.write(f"• Best performing method overall: {best_method}\n")
                
                # Classifier test insights
                excellent_samples = df[df['balanced_accuracy'] < 0.6]
                if len(excellent_samples) > 0:
                    excellent_rate = len(excellent_samples) / len(df) * 100
                    f.write(f"• {excellent_rate:.1f}% of samples achieved excellent quality "
                          f"(classifier cannot distinguish)\n")
                
                # Sample size effects
                size_effects = df.groupby('sample_size')['balanced_accuracy'].mean()
                best_size = size_effects.idxmin()  # Lower balanced accuracy is better
                f.write(f"• Optimal sample size for classifier performance: {best_size}\n")
                
                # Latent dimension insights
                dim_effects = df.groupby('latent_dim')['balanced_accuracy'].mean()
                best_dim = dim_effects.idxmin()
                f.write(f"• Best performing latent dimension: {best_dim}\n")
                
                f.write(f"\nReport generated successfully.\n")
            
            logger.info(f"📋 Comprehensive performance report saved to: {report_path}")
            
        except Exception as e:
            logger.warning(f"Could not create comprehensive performance report: {e}")


# Factory function
def create_enhanced_tester(config) -> EnhancedTester:
    """Create enhanced tester with multiple latent dimensions support."""
    return EnhancedTester(config)