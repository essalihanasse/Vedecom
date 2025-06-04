"""
Latent space visualization system for VAE pipeline.
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
            raise FileNotFoundError(f"Original data not found: {data_path}")
        
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
                return None
            
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
            
            # Load preprocessed data
            preprocessed_df = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
            data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
            
            # Get latent encodings
            z_mean = get_latent_encoding(model, data_tensor, self.device)
            
            logger.debug(f"Encoded {len(z_mean)} points to latent space")
            return z_mean
            
        except Exception as e:
            logger.error(f"Error loading model and encoding: {e}")
            return None
    
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