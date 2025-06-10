"""
Fixed and enhanced latent space visualization system for VAE pipeline.
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import pickle

logger = logging.getLogger(__name__)

class LatentVisualizer:
    """
    Creates comprehensive visualizations of VAE latent spaces.
    """
    
    def __init__(self, config):
        """
        Initialize latent visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load preprocessing info
        self._load_preprocessing_info()
        
        logger.info("Latent visualizer initialized")
    
    def _load_preprocessing_info(self) -> None:
        """Load preprocessing information."""
        try:
            with open(os.path.join(self.config.paths.DATA_DIR, 'preprocessing_objects.pkl'), 'rb') as f:
                self.preprocessing_objects = pickle.load(f)
                
            self.cat_cols = self.preprocessing_objects['cat_cols']
            self.num_cols = self.preprocessing_objects['num_cols']
            
        except FileNotFoundError:
            logger.warning("Preprocessing objects not found, using config defaults")
            self.cat_cols = self.config.data.CATEGORICAL_COLS
            self.num_cols = self.config.data.NUMERICAL_COLS
    
    def create_all_visualizations(self) -> Dict[str, Any]:
        """
        Create visualizations for all trained models.
        
        Returns:
            Dictionary with visualization results
        """
        logger.info("🎨 Creating comprehensive latent space visualizations...")
        
        all_results = {}
        
        # Load original data for coloring
        original_df = self._load_original_data()
        
        # Store all latent encodings for cross-comparisons
        all_latent_encodings = {}
        
        # Process each model
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            strategy_results = {}
            all_latent_encodings[strategy] = {}
            
            for beta in self.config.training.BETA_VALUES:
                logger.info(f"📊 Visualizing {strategy} strategy, beta={beta}")
                
                try:
                    # Create visualizations for this model
                    model_results = self._visualize_single_model(
                        strategy, beta, original_df
                    )
                    strategy_results[beta] = model_results
                    
                    # Store latent encodings for comparison
                    if 'latent_encodings' in model_results:
                        all_latent_encodings[strategy][beta] = model_results['latent_encodings']
                    
                except Exception as e:
                    logger.error(f"Visualization failed for {strategy}-{beta}: {e}")
                    strategy_results[beta] = {'error': str(e)}
            
            all_results[strategy] = strategy_results
        
        # Create cross-model comparisons
        self._create_cross_model_comparisons(all_latent_encodings, original_df)
        
        logger.info("✅ Latent space visualizations completed")
        return all_results
    
    def _load_original_data(self) -> pd.DataFrame:
        """Load original filtered data for visualization coloring."""
        data_path = os.path.join(self.config.paths.DATA_DIR, 'filtered_data.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(self.config.paths.DATA_DIR, 'data.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Original data not found in: {self.config.paths.DATA_DIR}")
        
        return pd.read_csv(data_path)
    
    def _visualize_single_model(
        self, 
        strategy: str, 
        beta: float, 
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create visualizations for a single model."""
        
        # Load model and get latent encodings
        z_mean = self._load_model_and_encode(strategy, beta)
        
        if z_mean is None:
            return {'error': 'Could not load model or encode data'}
        
        # Create output directory
        vis_dir = os.path.join(
            self.config.paths.VISUALIZATIONS_DIR,
            strategy,
            f'beta_{beta}'
        )
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save latent coordinates
        latent_df = pd.DataFrame(z_mean, columns=['latent_1', 'latent_2'])
        latent_df.to_csv(os.path.join(vis_dir, 'latent_coordinates.csv'), index=False)
        
        # Create various visualizations
        viz_results = {}
        
        # 1. Overview plot
        self._create_overview_plot(z_mean, vis_dir, strategy, beta)
        viz_results['overview'] = True
        
        # 2. Feature-colored plots
        feature_results = self._create_feature_plots(
            z_mean, original_df, vis_dir, strategy, beta
        )
        viz_results['features'] = feature_results
        
        # 3. Density plots
        self._create_density_plots(z_mean, vis_dir, strategy, beta)
        viz_results['density'] = True
        
        # 4. Statistical summary plots
        self._create_statistical_plots(z_mean, vis_dir, strategy, beta)
        viz_results['statistics'] = True
        
        # Store latent encodings for cross-comparisons
        viz_results['latent_encodings'] = z_mean
        
        return viz_results
    
    def _load_model_and_encode(self, strategy: str, beta: float) -> Optional[np.ndarray]:
        """Load VAE model and encode data to latent space."""
        try:
            from models.vae import AdaptiveVAE, get_latent_encoding
            
            # Load model
            model_path = os.path.join(
                self.config.paths.MODELS_DIR,
                strategy,
                f'beta_{beta}',
                'vae_model_final.pth'
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model instance
            model = AdaptiveVAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=checkpoint.get('latent_dim', self.config.model.LATENT_DIM),
                cat_dict=checkpoint.get('categorical_cardinality', {})
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load preprocessed data
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
            
            # Get latent encodings using the model method
            z_mean = self._get_latent_encoding(model, data_tensor)
            
            logger.debug(f"Encoded {len(z_mean)} points to latent space")
            return z_mean
            
        except Exception as e:
            logger.error(f"Error loading model and encoding: {e}")
            return None
    
    def _get_latent_encoding(self, model, data_tensor: torch.Tensor, batch_size: int = 512) -> np.ndarray:
        """Get latent space encodings for input data in batches."""
        model.eval()
        data_tensor = data_tensor.to(self.device)
        
        encodings = []
        with torch.no_grad():
            for i in range(0, len(data_tensor), batch_size):
                batch = data_tensor[i:i + batch_size]
                latent_repr = model.get_latent_representation(batch, use_mean=True)
                encodings.append(latent_repr.cpu().numpy())
        
        return np.vstack(encodings)
    
    def _create_overview_plot(
        self, 
        z_mean: np.ndarray, 
        vis_dir: str, 
        strategy: str, 
        beta: float
    ) -> None:
        """Create overview plot of latent space."""
        try:
            plt.figure(figsize=(10, 8))
            
            plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=8, c='blue')
            plt.title(f'Latent Space Overview\nStrategy: {strategy}, Beta: {beta}')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            plt.text(0.02, 0.98, 
                    f'Points: {len(z_mean)}\n'
                    f'Dim 1: μ={np.mean(z_mean[:, 0]):.3f}, σ={np.std(z_mean[:, 0]):.3f}\n'
                    f'Dim 2: μ={np.mean(z_mean[:, 1]):.3f}, σ={np.std(z_mean[:, 1]):.3f}',
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(vis_dir, 'latent_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create overview plot: {e}")
    
    def _create_feature_plots(
        self, 
        z_mean: np.ndarray, 
        original_df: pd.DataFrame, 
        vis_dir: str, 
        strategy: str, 
        beta: float
    ) -> Dict[str, bool]:
        """Create feature-colored latent space plots."""
        feature_dir = os.path.join(vis_dir, 'features')
        os.makedirs(feature_dir, exist_ok=True)
        
        results = {}
        
        # Ensure data alignment
        min_len = min(len(z_mean), len(original_df))
        z_mean = z_mean[:min_len]
        original_df = original_df.iloc[:min_len]
        
        # Numerical features
        for feature in self.num_cols:
            if feature not in original_df.columns:
                continue
                
            try:
                plt.figure(figsize=(10, 8))
                
                scatter = plt.scatter(
                    z_mean[:, 0], z_mean[:, 1],
                    c=original_df[feature],
                    cmap='viridis',
                    alpha=0.7,
                    s=10
                )
                
                plt.colorbar(scatter, label=feature)
                plt.title(f'Latent Space - {feature}\nStrategy: {strategy}, Beta: {beta}')
                plt.xlabel('Latent Dimension 1')
                plt.ylabel('Latent Dimension 2')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(feature_dir, f'{feature}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                results[feature] = True
                
            except Exception as e:
                logger.warning(f"Could not create plot for {feature}: {e}")
                results[feature] = False
        
        # Categorical features
        for feature in self.cat_cols:
            if feature not in original_df.columns:
                continue
                
            try:
                plt.figure(figsize=(12, 8))
                
                # Create mapping for categorical values
                unique_values = original_df[feature].unique()
                n_unique = len(unique_values)
                
                if n_unique > 20:  # Too many categories
                    logger.warning(f"Skipping {feature}: too many categories ({n_unique})")
                    continue
                
                # Use different markers/colors for different categories
                colors = plt.cm.tab20(np.linspace(0, 1, n_unique))
                
                for i, value in enumerate(unique_values):
                    mask = original_df[feature] == value
                    if np.sum(mask) > 0:
                        plt.scatter(
                            z_mean[mask, 0], z_mean[mask, 1],
                            alpha=0.7, s=10, color=colors[i],
                            label=f'{value} ({np.sum(mask)})'
                        )
                
                plt.title(f'Latent Space - {feature}\nStrategy: {strategy}, Beta: {beta}')
                plt.xlabel('Latent Dimension 1')
                plt.ylabel('Latent Dimension 2')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(feature_dir, f'{feature}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                results[feature] = True
                
            except Exception as e:
                logger.warning(f"Could not create plot for {feature}: {e}")
                results[feature] = False
        
        return results
    
    def _create_density_plots(
        self, 
        z_mean: np.ndarray, 
        vis_dir: str, 
        strategy: str, 
        beta: float
    ) -> None:
        """Create density-based visualizations."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 2D density plot
            axes[0, 0].hexbin(z_mean[:, 0], z_mean[:, 1], gridsize=30, cmap='Blues')
            axes[0, 0].set_title('2D Density (Hexbin)')
            axes[0, 0].set_xlabel('Latent Dimension 1')
            axes[0, 0].set_ylabel('Latent Dimension 2')
            
            # Contour plot
            from scipy.stats import gaussian_kde
            
            # Create KDE
            kde = gaussian_kde(z_mean.T)
            
            # Create mesh
            x_min, x_max = z_mean[:, 0].min() - 1, z_mean[:, 0].max() + 1
            y_min, y_max = z_mean[:, 1].min() - 1, z_mean[:, 1].max() + 1
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate KDE
            f = kde(positions).reshape(xx.shape)
            
            axes[0, 1].contourf(xx, yy, f, cmap='viridis', alpha=0.8)
            axes[0, 1].scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.3, s=1, color='white')
            axes[0, 1].set_title('2D Density (KDE Contours)')
            axes[0, 1].set_xlabel('Latent Dimension 1')
            axes[0, 1].set_ylabel('Latent Dimension 2')
            
            # Marginal distributions
            axes[1, 0].hist(z_mean[:, 0], bins=50, alpha=0.7, density=True, color='blue')
            axes[1, 0].set_title('Latent Dimension 1 Distribution')
            axes[1, 0].set_xlabel('Value')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].hist(z_mean[:, 1], bins=50, alpha=0.7, density=True, color='red')
            axes[1, 1].set_title('Latent Dimension 2 Distribution')
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Latent Space Density Analysis\nStrategy: {strategy}, Beta: {beta}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'density_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create density plots: {e}")
    
    def _create_statistical_plots(
        self, 
        z_mean: np.ndarray, 
        vis_dir: str, 
        strategy: str, 
        beta: float
    ) -> None:
        """Create statistical analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Q-Q plots for normality testing
            from scipy import stats
            
            stats.probplot(z_mean[:, 0], dist="norm", plot=axes[0, 0])
            axes[0, 0].set_title('Q-Q Plot - Latent Dimension 1')
            axes[0, 0].grid(True, alpha=0.3)
            
            stats.probplot(z_mean[:, 1], dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot - Latent Dimension 2')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Correlation analysis
            correlation = np.corrcoef(z_mean[:, 0], z_mean[:, 1])[0, 1]
            
            axes[1, 0].scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=5)
            axes[1, 0].set_title(f'Correlation Analysis\nCorrelation: {correlation:.4f}')
            axes[1, 0].set_xlabel('Latent Dimension 1')
            axes[1, 0].set_ylabel('Latent Dimension 2')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Summary statistics
            axes[1, 1].axis('off')
            
            # Calculate statistics
            stats_text = f"""
Statistical Summary:

Latent Dimension 1:
  Mean: {np.mean(z_mean[:, 0]):.4f}
  Std:  {np.std(z_mean[:, 0]):.4f}
  Min:  {np.min(z_mean[:, 0]):.4f}
  Max:  {np.max(z_mean[:, 0]):.4f}

Latent Dimension 2:
  Mean: {np.mean(z_mean[:, 1]):.4f}
  Std:  {np.std(z_mean[:, 1]):.4f}
  Min:  {np.min(z_mean[:, 1]):.4f}
  Max:  {np.max(z_mean[:, 1]):.4f}

Correlation: {correlation:.4f}

Normality Tests:
  Shapiro-Wilk p-value (Dim 1): {stats.shapiro(z_mean[:1000, 0])[1]:.6f}
  Shapiro-Wilk p-value (Dim 2): {stats.shapiro(z_mean[:1000, 1])[1]:.6f}
"""
            
            axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            plt.suptitle(f'Statistical Analysis\nStrategy: {strategy}, Beta: {beta}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create statistical plots: {e}")
    
    def _create_cross_model_comparisons(
        self, 
        all_latent_encodings: Dict[str, Dict[float, np.ndarray]], 
        original_df: pd.DataFrame
    ) -> None:
        """Create comparison plots across different models."""
        try:
            comparison_dir = os.path.join(self.config.paths.VISUALIZATIONS_DIR, 'cross_model_comparisons')
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Beta comparison for each strategy
            for strategy in all_latent_encodings:
                self._create_beta_comparison(
                    all_latent_encodings[strategy], strategy, comparison_dir, original_df
                )
            
            # Strategy comparison for each beta
            for beta in self.config.training.BETA_VALUES:
                self._create_strategy_comparison(
                    all_latent_encodings, beta, comparison_dir, original_df
                )
            
            logger.info("Cross-model comparison plots created")
            
        except Exception as e:
            logger.warning(f"Could not create cross-model comparisons: {e}")
    
    def _create_beta_comparison(
        self, 
        strategy_encodings: Dict[float, np.ndarray], 
        strategy: str, 
        comparison_dir: str, 
        original_df: pd.DataFrame
    ) -> None:
        """Create comparison across beta values for a strategy."""
        try:
            betas = sorted(strategy_encodings.keys())
            n_betas = len(betas)
            
            if n_betas <= 1:
                return
            
            # Calculate grid size
            n_cols = min(3, n_betas)
            n_rows = (n_betas + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_betas == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, beta in enumerate(betas):
                if i < len(axes):
                    z_mean = strategy_encodings[beta]
                    
                    axes[i].scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=5)
                    axes[i].set_title(f'Beta = {beta}')
                    axes[i].set_xlabel('Latent Dimension 1')
                    axes[i].set_ylabel('Latent Dimension 2')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics
                    corr = np.corrcoef(z_mean[:, 0], z_mean[:, 1])[0, 1]
                    axes[i].text(0.02, 0.98, f'Corr: {corr:.3f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_betas, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Beta Comparison - {strategy.title()} Strategy', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'{strategy}_beta_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create beta comparison for {strategy}: {e}")
    
    def _create_strategy_comparison(
        self, 
        all_latent_encodings: Dict[str, Dict[float, np.ndarray]], 
        beta: float, 
        comparison_dir: str, 
        original_df: pd.DataFrame
    ) -> None:
        """Create comparison across strategies for a beta value."""
        try:
            strategies = []
            encodings = []
            
            for strategy in all_latent_encodings:
                if beta in all_latent_encodings[strategy]:
                    strategies.append(strategy)
                    encodings.append(all_latent_encodings[strategy][beta])
            
            if len(strategies) <= 1:
                return
            
            n_strategies = len(strategies)
            n_cols = min(3, n_strategies)
            n_rows = (n_strategies + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_strategies == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (strategy, z_mean) in enumerate(zip(strategies, encodings)):
                if i < len(axes):
                    axes[i].scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=5)
                    axes[i].set_title(f'{strategy.title()}')
                    axes[i].set_xlabel('Latent Dimension 1')
                    axes[i].set_ylabel('Latent Dimension 2')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics
                    corr = np.corrcoef(z_mean[:, 0], z_mean[:, 1])[0, 1]
                    axes[i].text(0.02, 0.98, f'Corr: {corr:.3f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_strategies, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Strategy Comparison - Beta = {beta}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'beta_{beta}_strategy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create strategy comparison for beta {beta}: {e}")


class EnhancedLatentVisualizer(LatentVisualizer):
    """Enhanced visualizer with additional features for multiple latent dimensions."""
    
    def __init__(self, config):
        super().__init__(config)
        self.latent_dims = getattr(config.model, 'LATENT_DIMS', [2])
    
    def create_all_visualizations_with_latent_dims(self) -> Dict[str, Any]:
        """
        Create visualizations for all latent dimensions and models.
        
        Returns:
            Dictionary with visualization results organized by latent dimension
        """
        logger.info("🎨 Creating enhanced visualizations across multiple latent dimensions...")
        
        all_results = {}
        
        # Load original data
        original_df = self._load_original_data()
        
        # Process each latent dimension
        for latent_dim in self.latent_dims:
            logger.info(f"📏 Processing latent dimension: {latent_dim}")
            
            latent_results = {}
            all_latent_encodings = {}
            
            # Process each model for this latent dimension
            for strategy in self.config.training.ANNEALING_STRATEGIES:
                strategy_results = {}
                all_latent_encodings[strategy] = {}
                
                for beta in self.config.training.BETA_VALUES:
                    logger.info(f"📊 Visualizing {strategy} strategy, beta={beta}, latent_dim={latent_dim}")
                    
                    try:
                        # Create visualizations for this model
                        model_results = self._visualize_single_model_with_latent_dim(
                            strategy, beta, latent_dim, original_df
                        )
                        strategy_results[beta] = model_results
                        
                        # Store latent encodings for comparison
                        if 'latent_encodings' in model_results:
                            all_latent_encodings[strategy][beta] = model_results['latent_encodings']
                        
                    except Exception as e:
                        logger.error(f"Visualization failed for {strategy}-{beta}-{latent_dim}: {e}")
                        strategy_results[beta] = {'error': str(e)}
                
                latent_results[strategy] = strategy_results
            
            # Create cross-model comparisons for this latent dimension
            self._create_cross_model_comparisons_with_latent_dim(
                all_latent_encodings, original_df, latent_dim
            )
            
            all_results[latent_dim] = latent_results
        
        # Create cross-latent-dimension comparisons
        self._create_latent_dimension_comparisons(all_results, original_df)
        
        logger.info("✅ Enhanced latent space visualizations completed")
        return all_results
    
    def _visualize_single_model_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        original_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create visualizations for a single model with specific latent dimension."""
        
        # Load model and get latent encodings
        z_mean = self._load_model_and_encode_with_latent_dim(strategy, beta, latent_dim)
        
        if z_mean is None:
            return {'error': 'Could not load model or encode data'}
        
        # Create output directory
        vis_dir = os.path.join(
            self.config.paths.VISUALIZATIONS_DIR,
            f'latent_{latent_dim}',
            strategy,
            f'beta_{beta}'
        )
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save latent coordinates
        if latent_dim == 2:
            latent_df = pd.DataFrame(z_mean, columns=['latent_1', 'latent_2'])
        else:
            # For higher dimensions, save all but only visualize first 2
            columns = [f'latent_{i+1}' for i in range(latent_dim)]
            latent_df = pd.DataFrame(z_mean, columns=columns)
        
        latent_df.to_csv(os.path.join(vis_dir, 'latent_coordinates.csv'), index=False)
        
        # Create various visualizations
        viz_results = {}
        
        # Always visualize first 2 dimensions
        z_vis = z_mean[:, :2] if latent_dim > 2 else z_mean
        
        # 1. Overview plot
        self._create_overview_plot_with_latent_dim(z_vis, vis_dir, strategy, beta, latent_dim)
        viz_results['overview'] = True
        
        # 2. Feature-colored plots (only for 2D visualization)
        feature_results = self._create_feature_plots(
            z_vis, original_df, vis_dir, strategy, beta
        )
        viz_results['features'] = feature_results
        
        # 3. High-dimensional analysis (if applicable)
        if latent_dim > 2:
            self._create_high_dimensional_analysis(z_mean, vis_dir, strategy, beta, latent_dim)
            viz_results['high_dimensional'] = True
        
        # 4. Density plots
        self._create_density_plots(z_vis, vis_dir, strategy, beta)
        viz_results['density'] = True
        
        # Store latent encodings for cross-comparisons
        viz_results['latent_encodings'] = z_mean
        viz_results['latent_dim'] = latent_dim
        
        return viz_results
    
    def _load_model_and_encode_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int
    ) -> Optional[np.ndarray]:
        """Load VAE model with specific latent dimension and encode data."""
        try:
            from models.vae import AdaptiveVAE
            
            # Load model
            model_path = os.path.join(
                self.config.paths.MODELS_DIR,
                f'latent_{latent_dim}',
                strategy,
                f'beta_{beta}',
                'vae_model_final.pth'
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Create model instance
            model = AdaptiveVAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=latent_dim,
                cat_dict=checkpoint.get('categorical_cardinality', {})
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load preprocessed data
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
            
            # Get latent encodings
            z_mean = self._get_latent_encoding(model, data_tensor)
            
            logger.debug(f"Encoded {len(z_mean)} points to {latent_dim}D latent space")
            return z_mean
            
        except Exception as e:
            logger.error(f"Error loading model and encoding for latent_dim {latent_dim}: {e}")
            return None
    
    def _create_overview_plot_with_latent_dim(
        self, 
        z_vis: np.ndarray, 
        vis_dir: str, 
        strategy: str, 
        beta: float,
        latent_dim: int
    ) -> None:
        """Create overview plot with latent dimension information."""
        try:
            plt.figure(figsize=(10, 8))
            
            plt.scatter(z_vis[:, 0], z_vis[:, 1], alpha=0.5, s=8, c='blue')
            title = f'Latent Space Overview (Dim {latent_dim})\nStrategy: {strategy}, Beta: {beta}'
            if latent_dim > 2:
                title += f'\n(Showing first 2 of {latent_dim} dimensions)'
            
            plt.title(title)
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f'Points: {len(z_vis)}\n'
            stats_text += f'Dim 1: μ={np.mean(z_vis[:, 0]):.3f}, σ={np.std(z_vis[:, 0]):.3f}\n'
            stats_text += f'Dim 2: μ={np.mean(z_vis[:, 1]):.3f}, σ={np.std(z_vis[:, 1]):.3f}'
            
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(os.path.join(vis_dir, 'latent_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create overview plot: {e}")
    
    def _create_high_dimensional_analysis(
        self, 
        z_mean: np.ndarray, 
        vis_dir: str, 
        strategy: str, 
        beta: float,
        latent_dim: int
    ) -> None:
        """Create analysis for high-dimensional latent spaces."""
        try:
            # Create directory for high-dimensional analysis
            hd_dir = os.path.join(vis_dir, 'high_dimensional')
            os.makedirs(hd_dir, exist_ok=True)
            
            # Variance explained by each dimension
            variances = np.var(z_mean, axis=0)
            
            plt.figure(figsize=(12, 8))
            
            # Plot variance per dimension
            plt.subplot(2, 2, 1)
            plt.bar(range(1, latent_dim + 1), variances)
            plt.title('Variance per Latent Dimension')
            plt.xlabel('Latent Dimension')
            plt.ylabel('Variance')
            plt.grid(True, alpha=0.3)
            
            # Cumulative variance
            plt.subplot(2, 2, 2)
            cumulative_var = np.cumsum(variances) / np.sum(variances)
            plt.plot(range(1, latent_dim + 1), cumulative_var, 'o-')
            plt.title('Cumulative Variance Explained')
            plt.xlabel('Latent Dimension')
            plt.ylabel('Cumulative Proportion')
            plt.grid(True, alpha=0.3)
            
            # Correlation matrix
            plt.subplot(2, 2, 3)
            corr_matrix = np.corrcoef(z_mean.T)
            im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar(im)
            plt.title('Latent Dimension Correlations')
            plt.xlabel('Latent Dimension')
            plt.ylabel('Latent Dimension')
            
            # Dimension activity (non-zero variance)
            plt.subplot(2, 2, 4)
            active_dims = np.sum(variances > 0.01)  # Dimensions with meaningful variance
            activity_ratio = active_dims / latent_dim
            
            labels = ['Active', 'Inactive']
            sizes = [active_dims, latent_dim - active_dims]
            colors = ['lightgreen', 'lightcoral']
            
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            plt.title(f'Dimension Activity\n{active_dims}/{latent_dim} active')
            
            plt.suptitle(f'High-Dimensional Analysis (Latent Dim {latent_dim})\n{strategy}-β{beta}', 
                        fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(hd_dir, 'high_dimensional_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create high-dimensional analysis: {e}")
    
    def _create_cross_model_comparisons_with_latent_dim(
        self, 
        all_latent_encodings: Dict[str, Dict[float, np.ndarray]], 
        original_df: pd.DataFrame,
        latent_dim: int
    ) -> None:
        """Create comparison plots for a specific latent dimension."""
        try:
            comparison_dir = os.path.join(
                self.config.paths.VISUALIZATIONS_DIR, 
                f'latent_{latent_dim}',
                'cross_model_comparisons'
            )
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Beta comparison for each strategy
            for strategy in all_latent_encodings:
                if all_latent_encodings[strategy]:  # Check if not empty
                    self._create_beta_comparison_with_latent_dim(
                        all_latent_encodings[strategy], strategy, comparison_dir, latent_dim
                    )
            
            # Strategy comparison for each beta
            for beta in self.config.training.BETA_VALUES:
                self._create_strategy_comparison_with_latent_dim(
                    all_latent_encodings, beta, comparison_dir, latent_dim
                )
            
        except Exception as e:
            logger.warning(f"Could not create cross-model comparisons for latent_dim {latent_dim}: {e}")
    
    def _create_beta_comparison_with_latent_dim(
        self, 
        strategy_encodings: Dict[float, np.ndarray], 
        strategy: str, 
        comparison_dir: str,
        latent_dim: int
    ) -> None:
        """Create beta comparison for specific latent dimension."""
        try:
            betas = sorted(strategy_encodings.keys())
            n_betas = len(betas)
            
            if n_betas <= 1:
                return
            
            # Calculate grid size
            n_cols = min(3, n_betas)
            n_rows = (n_betas + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_betas == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, beta in enumerate(betas):
                if i < len(axes):
                    z_mean = strategy_encodings[beta]
                    # Use first 2 dimensions for visualization
                    z_vis = z_mean[:, :2] if z_mean.shape[1] > 2 else z_mean
                    
                    axes[i].scatter(z_vis[:, 0], z_vis[:, 1], alpha=0.5, s=5)
                    axes[i].set_title(f'Beta = {beta}')
                    axes[i].set_xlabel('Latent Dimension 1')
                    axes[i].set_ylabel('Latent Dimension 2')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics
                    corr = np.corrcoef(z_vis[:, 0], z_vis[:, 1])[0, 1]
                    axes[i].text(0.02, 0.98, f'Corr: {corr:.3f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_betas, len(axes)):
                axes[i].set_visible(False)
            
            title = f'Beta Comparison - {strategy.title()} Strategy (Latent Dim {latent_dim})'
            if latent_dim > 2:
                title += f'\n(Showing first 2 of {latent_dim} dimensions)'
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'{strategy}_beta_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create beta comparison for {strategy} with latent_dim {latent_dim}: {e}")
    
    def _create_strategy_comparison_with_latent_dim(
        self, 
        all_latent_encodings: Dict[str, Dict[float, np.ndarray]], 
        beta: float, 
        comparison_dir: str,
        latent_dim: int
    ) -> None:
        """Create strategy comparison for specific latent dimension."""
        try:
            strategies = []
            encodings = []
            
            for strategy in all_latent_encodings:
                if beta in all_latent_encodings[strategy]:
                    strategies.append(strategy)
                    encodings.append(all_latent_encodings[strategy][beta])
            
            if len(strategies) <= 1:
                return
            
            n_strategies = len(strategies)
            n_cols = min(3, n_strategies)
            n_rows = (n_strategies + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_strategies == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (strategy, z_mean) in enumerate(zip(strategies, encodings)):
                if i < len(axes):
                    # Use first 2 dimensions for visualization
                    z_vis = z_mean[:, :2] if z_mean.shape[1] > 2 else z_mean
                    
                    axes[i].scatter(z_vis[:, 0], z_vis[:, 1], alpha=0.5, s=5)
                    axes[i].set_title(f'{strategy.title()}')
                    axes[i].set_xlabel('Latent Dimension 1')
                    axes[i].set_ylabel('Latent Dimension 2')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add statistics
                    corr = np.corrcoef(z_vis[:, 0], z_vis[:, 1])[0, 1]
                    axes[i].text(0.02, 0.98, f'Corr: {corr:.3f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_strategies, len(axes)):
                axes[i].set_visible(False)
            
            title = f'Strategy Comparison - Beta = {beta} (Latent Dim {latent_dim})'
            if latent_dim > 2:
                title += f'\n(Showing first 2 of {latent_dim} dimensions)'
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'beta_{beta}_strategy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create strategy comparison for beta {beta} with latent_dim {latent_dim}: {e}")
    
    def _create_latent_dimension_comparisons(
        self, 
        all_results: Dict[int, Dict[str, Any]], 
        original_df: pd.DataFrame
    ) -> None:
        """Create comparison plots across different latent dimensions."""
        try:
            comparison_dir = os.path.join(
                self.config.paths.VISUALIZATIONS_DIR, 
                'latent_dimension_comparisons'
            )
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Extract latent dimensions that have results
            available_latent_dims = [
                latent_dim for latent_dim in all_results.keys()
                if any(
                    'latent_encodings' in strategy_results.get(beta, {})
                    for strategy_results in all_results[latent_dim].values()
                    for beta in strategy_results.keys()
                )
            ]
            
            if len(available_latent_dims) <= 1:
                logger.info("Not enough latent dimensions for comparison")
                return
            
            # Create comparison for each strategy-beta combination
            for strategy in self.config.training.ANNEALING_STRATEGIES:
                for beta in self.config.training.BETA_VALUES:
                    self._create_single_latent_dim_comparison(
                        all_results, strategy, beta, available_latent_dims, comparison_dir
                    )
            
            # Create overall latent dimension analysis
            self._create_latent_dimension_analysis(all_results, available_latent_dims, comparison_dir)
            
        except Exception as e:
            logger.warning(f"Could not create latent dimension comparisons: {e}")
    
    def _create_single_latent_dim_comparison(
        self,
        all_results: Dict[int, Dict[str, Any]],
        strategy: str,
        beta: float,
        available_latent_dims: List[int],
        comparison_dir: str
    ) -> None:
        """Create comparison plot for a single strategy-beta across latent dimensions."""
        try:
            # Collect encodings for this strategy-beta across latent dimensions
            encodings_by_dim = {}
            
            for latent_dim in available_latent_dims:
                if (strategy in all_results[latent_dim] and 
                    beta in all_results[latent_dim][strategy] and
                    'latent_encodings' in all_results[latent_dim][strategy][beta]):
                    
                    encodings_by_dim[latent_dim] = all_results[latent_dim][strategy][beta]['latent_encodings']
            
            if len(encodings_by_dim) <= 1:
                return
            
            # Create subplot grid
            n_dims = len(encodings_by_dim)
            n_cols = min(3, n_dims)
            n_rows = (n_dims + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_dims == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, (latent_dim, z_mean) in enumerate(encodings_by_dim.items()):
                if i < len(axes):
                    # Use first 2 dimensions for visualization
                    z_vis = z_mean[:, :2] if z_mean.shape[1] > 2 else z_mean
                    
                    axes[i].scatter(z_vis[:, 0], z_vis[:, 1], alpha=0.5, s=8)
                    
                    title = f'Latent Dim {latent_dim}'
                    if latent_dim > 2:
                        title += f'\n(First 2 of {latent_dim})'
                    
                    axes[i].set_title(title)
                    axes[i].set_xlabel('Latent Dimension 1')
                    axes[i].set_ylabel('Latent Dimension 2')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add variance information
                    var_total = np.sum(np.var(z_mean, axis=0))
                    var_2d = np.sum(np.var(z_vis, axis=0))
                    var_ratio = var_2d / var_total if var_total > 0 else 0
                    
                    axes[i].text(0.02, 0.98, f'2D Var: {var_ratio:.2%}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(n_dims, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Latent Dimension Comparison\n{strategy.title()} Strategy, Beta = {beta}', 
                        fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, f'{strategy}_beta_{beta}_latent_dims.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create latent dim comparison for {strategy}-{beta}: {e}")
    
    def _create_latent_dimension_analysis(
        self,
        all_results: Dict[int, Dict[str, Any]],
        available_latent_dims: List[int],
        comparison_dir: str
    ) -> None:
        """Create overall analysis across latent dimensions."""
        try:
            # Collect variance and correlation statistics
            dim_stats = {}
            
            for latent_dim in available_latent_dims:
                dim_stats[latent_dim] = {
                    'variances': [],
                    'correlations': [],
                    'active_dims': [],
                    'total_variance': []
                }
                
                for strategy in all_results[latent_dim]:
                    for beta, results in all_results[latent_dim][strategy].items():
                        if 'latent_encodings' in results:
                            z_mean = results['latent_encodings']
                            
                            # Calculate statistics
                            variances = np.var(z_mean, axis=0)
                            dim_stats[latent_dim]['variances'].append(variances)
                            dim_stats[latent_dim]['total_variance'].append(np.sum(variances))
                            
                            # Count active dimensions (variance > threshold)
                            active_dims = np.sum(variances > 0.01)
                            dim_stats[latent_dim]['active_dims'].append(active_dims)
                            
                            # Calculate correlation between first 2 dims if available
                            if z_mean.shape[1] >= 2:
                                corr = np.corrcoef(z_mean[:, 0], z_mean[:, 1])[0, 1]
                                dim_stats[latent_dim]['correlations'].append(abs(corr))
            
            # Create analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Average total variance by latent dimension
            dims = sorted(dim_stats.keys())
            avg_variances = [np.mean(dim_stats[dim]['total_variance']) for dim in dims]
            std_variances = [np.std(dim_stats[dim]['total_variance']) for dim in dims]
            
            axes[0, 0].errorbar(dims, avg_variances, yerr=std_variances, 
                              marker='o', capsize=5, capthick=2, linewidth=2)
            axes[0, 0].set_title('Total Variance by Latent Dimension')
            axes[0, 0].set_xlabel('Latent Dimension')
            axes[0, 0].set_ylabel('Total Variance')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log', base=2)
            
            # Plot 2: Active dimensions ratio
            avg_active = [np.mean(dim_stats[dim]['active_dims']) for dim in dims]
            active_ratios = [active / dim for active, dim in zip(avg_active, dims)]
            
            axes[0, 1].bar(range(len(dims)), active_ratios, alpha=0.7)
            axes[0, 1].set_title('Active Dimensions Ratio')
            axes[0, 1].set_xlabel('Latent Dimension')
            axes[0, 1].set_ylabel('Active Ratio')
            axes[0, 1].set_xticks(range(len(dims)))
            axes[0, 1].set_xticklabels([str(d) for d in dims])
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Plot 3: First 2 dimensions correlation
            avg_correlations = [np.mean(dim_stats[dim]['correlations']) if dim_stats[dim]['correlations'] 
                              else 0 for dim in dims]
            
            axes[1, 0].plot(dims, avg_correlations, 'o-', linewidth=2, markersize=8)
            axes[1, 0].set_title('Average |Correlation| Between First 2 Dimensions')
            axes[1, 0].set_xlabel('Latent Dimension')
            axes[1, 0].set_ylabel('Average |Correlation|')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale('log', base=2)
            
            # Plot 4: Variance distribution analysis
            axes[1, 1].boxplot([dim_stats[dim]['total_variance'] for dim in dims], 
                              labels=[str(d) for d in dims])
            axes[1, 1].set_title('Total Variance Distribution')
            axes[1, 1].set_xlabel('Latent Dimension')
            axes[1, 1].set_ylabel('Total Variance')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.suptitle('Latent Dimension Analysis Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'latent_dimension_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save numerical summary
            self._save_latent_dimension_summary(dim_stats, comparison_dir)
            
        except Exception as e:
            logger.warning(f"Could not create latent dimension analysis: {e}")
    
    def _save_latent_dimension_summary(
        self, 
        dim_stats: Dict[int, Dict[str, List]], 
        comparison_dir: str
    ) -> None:
        """Save numerical summary of latent dimension analysis."""
        try:
            summary_data = []
            
            for latent_dim, stats in dim_stats.items():
                summary_data.append({
                    'latent_dim': latent_dim,
                    'avg_total_variance': np.mean(stats['total_variance']),
                    'std_total_variance': np.std(stats['total_variance']),
                    'avg_active_dims': np.mean(stats['active_dims']),
                    'active_ratio': np.mean(stats['active_dims']) / latent_dim,
                    'avg_correlation': np.mean(stats['correlations']) if stats['correlations'] else 0,
                    'n_models': len(stats['total_variance'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(comparison_dir, 'latent_dimension_summary.csv'), index=False)
            
            logger.info("Latent dimension summary saved")
            
        except Exception as e:
            logger.warning(f"Could not save latent dimension summary: {e}")


# Factory functions for backwards compatibility
def create_enhanced_visualizer(config) -> EnhancedLatentVisualizer:
    """Create enhanced visualizer with multiple latent dimensions support."""
    return EnhancedLatentVisualizer(config)

def create_visualizer(config) -> LatentVisualizer:
    """Create standard visualizer (backwards compatibility)."""
    return LatentVisualizer(config)