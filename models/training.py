"""
Enhanced model training system for VAE pipeline with multiple latent dimensions support.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import time
from datetime import timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import glob
import re
import shutil

from .vae import AdaptiveVAE, EnhancedVAELoss, BetaScheduler, create_adaptive_model
from .callbacks import EarlyStopping, ModelCheckpoint, CallbackList

logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Enhanced trainer for VAE models with multiple latent dimensions support."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_random_seeds()
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Training tracking
        self.all_results = {}
        
    def _set_random_seeds(self, seed: int = 42) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders."""
        # Load data
        df_prepared = pd.read_csv(self.config.paths.PREPROCESSED_FILE)
        
        with open(os.path.join(self.config.paths.DATA_DIR, 'preprocessing_objects.pkl'), 'rb') as f:
            self.preprocessing_objects = pickle.load(f)
        
        # Train/validation split
        X_train, X_val = train_test_split(df_prepared.values, test_size=0.2, random_state=42)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(X_train).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(X_val).to(self.device)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.model.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.model.BATCH_SIZE, shuffle=False)
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        return train_loader, val_loader
    
    def train_all_models_with_latent_dims(self) -> Dict[str, Any]:
        """Train models for all combinations of strategies, betas, and latent dimensions."""
        logger.info("🧠 Starting comprehensive training across all latent dimensions...")
        
        total_combinations = (len(self.config.training.ANNEALING_STRATEGIES) * 
                             len(self.config.training.BETA_VALUES) * 
                             len(self.config.model.LATENT_DIMS))
        
        current_combination = 0
        
        for latent_dim in self.config.model.LATENT_DIMS:
            logger.info(f"\n🔢 Training models for latent dimension: {latent_dim}")
            latent_results = {}
            
            for strategy in self.config.training.ANNEALING_STRATEGIES:
                strategy_results = {}
                
                for beta in self.config.training.BETA_VALUES:
                    current_combination += 1
                    logger.info(f"\n🎯 Training combination {current_combination}/{total_combinations}")
                    logger.info(f"   Latent Dim: {latent_dim}, Strategy: {strategy}, Beta: {beta}")
                    
                    try:
                        results = self.train_single_model_with_latent_dim(
                            strategy=strategy, 
                            beta=beta, 
                            latent_dim=latent_dim
                        )
                        strategy_results[beta] = results
                    except Exception as e:
                        logger.error(f"Failed to train model {strategy}-{beta}-{latent_dim}: {e}")
                        strategy_results[beta] = {'error': str(e)}
                
                latent_results[strategy] = strategy_results
                self._create_strategy_summary(strategy, strategy_results, latent_dim)
            
            self.all_results[latent_dim] = latent_results
            self._create_latent_dim_summary(latent_dim, latent_results)
        
        # Create comprehensive comparison across all latent dimensions
        self._create_comprehensive_comparison()
        
        return self.all_results
    
    def train_single_model_with_latent_dim(
        self, 
        strategy: str, 
        beta: float, 
        latent_dim: int,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train a single VAE model with specific latent dimension."""
        logger.info(f"Training model: strategy={strategy}, beta={beta}, latent_dim={latent_dim}")
        
        if save_dir is None:
            save_dir = os.path.join(
                self.config.paths.MODELS_DIR, 
                f'latent_{latent_dim}',
                strategy, 
                f'beta_{beta}'
            )
        os.makedirs(save_dir, exist_ok=True)
        
        # Get configuration for this latent dimension
        model_config = self.config.get_model_params(latent_dim)
        training_config = self.config.get_training_params(latent_dim)
        
        # Create model with adaptive architecture
        model = self._create_adaptive_model(latent_dim, model_config)
        
        # Create optimizer with dimension-specific learning rate
        optimizer = optim.Adam(
            model.parameters(), 
            lr=model_config['learning_rate'],
            weight_decay=1e-5 if latent_dim > 16 else 0
        )
        
        # Create loss function
        loss_fn = EnhancedVAELoss(
            self.preprocessing_objects['categorical_cardinality'],
            len(self.preprocessing_objects['num_cols']),
            latent_dim=latent_dim
        )
        
        # Get beta schedule
        beta_schedule = BetaScheduler.get_schedule(
            beta_start=0.0, 
            beta_end=beta, 
            num_epochs=model_config['num_epochs'], 
            strategy=strategy
        )
        
        # Setup callbacks with dimension-specific parameters
        callbacks = self._setup_callbacks(save_dir, training_config)
        
        # Training loop
        training_results = self._training_loop(
            model, optimizer, loss_fn, beta_schedule, callbacks, model_config
        )
        
        # Save final model with enhanced metadata
        self._save_model_with_metadata(
            model, optimizer, training_results, save_dir, 
            strategy, beta, latent_dim, model_config
        )
        
        return training_results
    
    def _create_adaptive_model(self, latent_dim: int, model_config: Dict[str, Any]) -> AdaptiveVAE:
        """Create adaptive VAE model for specific latent dimension."""
        # Determine architecture type based on latent dimension
        if latent_dim <= 4:
            architecture_type = 'adaptive'
        elif latent_dim <= 16:
            architecture_type = 'deep'
        else:
            architecture_type = 'wide'
        
        model = create_adaptive_model(
            input_dim=len(self.train_loader.dataset[0][0]),
            num_numerical=len(self.preprocessing_objects['num_cols']),
            cat_dict=self.preprocessing_objects['categorical_cardinality'],
            latent_dim=latent_dim,
            hidden_dim=model_config['hidden_dim'],
            architecture_type=architecture_type
        ).to(self.device)
        
        logger.info(f"Created {architecture_type} model with {model._count_parameters():,} parameters")
        return model
    
    def _setup_callbacks(self, save_dir: str, training_config: Dict[str, Any]) -> CallbackList:
        """Setup training callbacks with dimension-specific parameters."""
        callbacks = []
        
        # Early stopping with dimension-specific parameters
        if training_config.get('early_stopping', True):
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                min_delta=training_config.get('min_delta', 0.001),
                patience=training_config.get('patience', 10),
                verbose=True,
                restore_best_weights=training_config.get('restore_best_weights', True)
            ))
        
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(save_dir, 'checkpoint_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ))
        
        return CallbackList(callbacks)
    
    def _training_loop(
        self, 
        model: AdaptiveVAE, 
        optimizer: torch.optim.Optimizer, 
        loss_fn: EnhancedVAELoss, 
        beta_schedule: List[float], 
        callbacks: CallbackList,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced training loop with adaptive features."""
        history = {
            'train_loss': [], 'val_loss': [], 'num_loss': [], 
            'cat_loss': [], 'kl_loss': [], 'beta_values': [],
            'learning_rates': [], 'grad_norms': []
        }
        
        callbacks.set_model(model)
        callbacks.on_train_begin()
        
        start_time = time.time()
        
        # Learning rate scheduler for high-dimensional latent spaces
        if loss_fn.latent_dim > 16:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=5, verbose=True
            )
        else:
            scheduler = None
        
        for epoch in range(model_config['num_epochs']):
            current_beta = beta_schedule[epoch]
            callbacks.on_epoch_begin(epoch)
            
            # Training phase
            train_metrics = self._run_epoch(
                model, self.train_loader, optimizer, loss_fn, current_beta, is_training=True
            )
            
            # Validation phase  
            val_metrics = self._run_epoch(
                model, self.val_loader, None, loss_fn, current_beta, is_training=False
            )
            
            # Update learning rate for high-dimensional spaces
            if scheduler:
                scheduler.step(val_metrics['total_loss'])
            
            # Calculate gradient norm for monitoring
            grad_norm = self._calculate_grad_norm(model)
            
            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['num_loss'].append(train_metrics['num_loss'])
            history['cat_loss'].append(train_metrics['cat_loss'])
            history['kl_loss'].append(train_metrics['kl_loss'])
            history['beta_values'].append(current_beta)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            history['grad_norms'].append(grad_norm)
            
            # Enhanced logging
            if (epoch + 1) % 10 == 0 or epoch == model_config['num_epochs'] - 1:
                logger.info(f'Epoch {epoch+1}/{model_config["num_epochs"]}: '
                          f'Train: {train_metrics["total_loss"]:.4f}, '
                          f'Val: {val_metrics["total_loss"]:.4f}, '
                          f'Beta: {current_beta:.3f}, '
                          f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                          f'GradNorm: {grad_norm:.3f}')
            
            # Callback logging with enhanced metrics
            logs = {
                'loss': train_metrics['total_loss'], 
                'val_loss': val_metrics['total_loss'],
                'num_loss': train_metrics['num_loss'], 
                'cat_loss': train_metrics['cat_loss'],
                'kl_loss': train_metrics['kl_loss'], 
                'beta': current_beta,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'grad_norm': grad_norm
            }
            
            callbacks.on_epoch_end(epoch, logs)
            
            if callbacks.get_stop_training():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        callbacks.on_train_end()
        
        # Enhanced training results
        training_time = time.time() - start_time
        self._save_training_artifacts(history, os.path.dirname(callbacks.callbacks[1].filepath))
        
        return {
            'history': history,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'final_metrics': {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1],
                'final_lr': history['learning_rates'][-1],
                'final_grad_norm': history['grad_norms'][-1]
            },
            'model_complexity': {
                'total_parameters': model._count_parameters(),
                'latent_dim': loss_fn.latent_dim,
                'architecture_type': model.architecture_type
            }
        }
    
    def _run_epoch(
        self, 
        model: AdaptiveVAE, 
        data_loader: DataLoader, 
        optimizer: Optional[torch.optim.Optimizer],
        loss_fn: EnhancedVAELoss, 
        beta: float, 
        is_training: bool
    ) -> Dict[str, float]:
        """Enhanced epoch runner with gradient monitoring."""
        model.train() if is_training else model.eval()
        
        total_loss = num_loss = cat_loss = kl_loss = 0
        
        context = torch.enable_grad() if is_training else torch.no_grad()
        
        with context:
            for data, _ in data_loader:
                if is_training:
                    optimizer.zero_grad()
                
                # Forward pass
                decoded_outputs, mu, logvar = model(data)
                loss, n_loss, c_loss, k_loss = loss_fn(decoded_outputs, data, mu, logvar, beta)
                
                if is_training:
                    loss.backward()
                    
                    # Gradient clipping for stability
                    if loss_fn.latent_dim > 16:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                num_loss += n_loss.item()
                cat_loss += c_loss.item()
                kl_loss += k_loss.item()
        
        # Average by number of samples
        n_samples = len(data_loader.dataset)
        return {
            'total_loss': total_loss / n_samples,
            'num_loss': num_loss / n_samples,
            'cat_loss': cat_loss / n_samples,
            'kl_loss': kl_loss / n_samples
        }
    
    def _calculate_grad_norm(self, model: AdaptiveVAE) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def _save_model_with_metadata(
        self, 
        model: AdaptiveVAE, 
        optimizer: torch.optim.Optimizer, 
        training_results: Dict[str, Any], 
        save_dir: str, 
        strategy: str, 
        beta: float,
        latent_dim: int,
        model_config: Dict[str, Any]
    ) -> None:
        """Save trained model with comprehensive metadata."""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Enhanced checkpoint with latent dimension info
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'categorical_cardinality': self.preprocessing_objects['categorical_cardinality'],
                'num_numerical': len(self.preprocessing_objects['num_cols']),
                'input_dim': model.input_dim,
                'hidden_dim': model_config['hidden_dim'],
                'latent_dim': latent_dim,
                'architecture_type': model.architecture_type,
                'total_parameters': model._count_parameters(),
                'beta': beta,
                'strategy': strategy,
                'training_results': training_results,
                'model_config': model_config,
                'config_version': '2.0_latent_dims'
            }
            
            model_path = os.path.join(save_dir, 'vae_model_final.pth')
            torch.save(checkpoint, model_path)
            
            # Verify save
            if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
                logger.info(f"✅ Model saved: {model_path} ({os.path.getsize(model_path):,} bytes)")
            else:
                raise Exception("Model save verification failed")
                
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            # Attempt recovery as in original code
            self._copy_best_checkpoint_as_final(save_dir)
    
    def _copy_best_checkpoint_as_final(self, save_dir: str) -> None:
        """Recovery method to copy best checkpoint as final model."""
        checkpoint_files = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            raise Exception(f"No checkpoint files found in {save_dir}")
        
        def get_val_loss(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'val_loss_(\d+\.?\d*)', filename)
            return float(match.group(1)) if match else float('inf')
        
        best_checkpoint = min(checkpoint_files, key=get_val_loss)
        final_path = os.path.join(save_dir, 'vae_model_final.pth')
        shutil.copy2(best_checkpoint, final_path)
        
        if os.path.exists(final_path):
            logger.info(f"✅ Recovered model from: {os.path.basename(best_checkpoint)}")
    
    def _save_training_artifacts(self, history: Dict[str, List], save_dir: str) -> None:
        """Save enhanced training artifacts with additional metrics."""
        # Save history
        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
        
        # Create enhanced plots
        self._create_enhanced_training_plots(history, save_dir)
    
    def _create_enhanced_training_plots(self, history: Dict[str, List], save_dir: str) -> None:
        """Create enhanced training visualization plots."""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main losses with beta schedule
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Beta schedule on secondary y-axis
        ax1_beta = ax1.twinx()
        ax1_beta.plot(epochs, history['beta_values'], 'g--', alpha=0.7, label='Beta')
        ax1_beta.set_ylabel('Beta Value', color='g')
        ax1_beta.tick_params(axis='y', labelcolor='g')
        
        # Component losses
        ax2.plot(epochs, history['num_loss'], 'g-', label='Numerical Loss')
        ax2.plot(epochs, history['cat_loss'], 'm-', label='Categorical Loss')
        ax2.plot(epochs, history['kl_loss'], 'c-', label='KL Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Component Loss')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate and gradient norms
        ax3.plot(epochs, history['learning_rates'], 'orange', label='Learning Rate')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(epochs, history['grad_norms'], 'purple', label='Gradient Norm')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norm Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'enhanced_training_curves.png'), dpi=300)
        plt.close()
    
    def _create_strategy_summary(self, strategy: str, strategy_results: Dict[str, Any], latent_dim: int) -> None:
        """Create summary for a single strategy with latent dimension info."""
        strategy_dir = os.path.join(self.config.paths.MODELS_DIR, f'latent_{latent_dim}', strategy)
        os.makedirs(strategy_dir, exist_ok=True)
        
        summary_data = []
        for beta, results in strategy_results.items():
            if 'error' not in results:
                final_metrics = results['final_metrics']
                model_complexity = results['model_complexity']
                summary_data.append({
                    'latent_dim': latent_dim,
                    'beta': beta,
                    'train_loss': final_metrics['train_loss'],
                    'val_loss': final_metrics['val_loss'],
                    'training_time': results['training_time'],
                    'epochs_trained': results['epochs_trained'],
                    'total_parameters': model_complexity['total_parameters'],
                    'architecture_type': model_complexity['architecture_type'],
                    'final_lr': final_metrics.get('final_lr', 0),
                    'final_grad_norm': final_metrics.get('final_grad_norm', 0)
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(strategy_dir, f'latent_{latent_dim}_beta_results_summary.csv'), index=False)
            
            # Create enhanced trade-off plot
            self._create_enhanced_tradeoff_plot(df, strategy_dir, strategy, latent_dim)
    
    def _create_enhanced_tradeoff_plot(self, df: pd.DataFrame, strategy_dir: str, strategy: str, latent_dim: int) -> None:
        """Create enhanced reconstruction vs KL trade-off plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Trade-off plot
        scatter = ax1.scatter(df['val_loss'], df['train_loss'], s=100, c=df['beta'], 
                             cmap='viridis', alpha=0.8, edgecolors='black')
        
        for _, row in df.iterrows():
            ax1.annotate(f'β={row["beta"]}', (row['val_loss'], row['train_loss']),
                        xytext=(10, 5), textcoords='offset points', fontsize=10)
        
        plt.colorbar(scatter, ax=ax1, label='Beta Value')
        ax1.set_title(f'VAE Trade-off: Latent Dim {latent_dim}\n({strategy} annealing)')
        ax1.set_xlabel('Validation Loss')
        ax1.set_ylabel('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Model complexity vs performance
        ax2.scatter(df['total_parameters'], df['val_loss'], s=100, c=df['beta'], 
                   cmap='plasma', alpha=0.8, edgecolors='black')
        ax2.set_title(f'Model Complexity vs Performance\nLatent Dim {latent_dim}')
        ax2.set_xlabel('Total Parameters')
        ax2.set_ylabel('Validation Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(strategy_dir, f'latent_{latent_dim}_enhanced_analysis.png'), dpi=300)
        plt.close()
    
    def _create_latent_dim_summary(self, latent_dim: int, latent_results: Dict[str, Any]) -> None:
        """Create comprehensive summary for a specific latent dimension."""
        latent_dir = os.path.join(self.config.paths.MODELS_DIR, f'latent_{latent_dim}')
        
        # Collect all results for this latent dimension
        all_results = []
        for strategy, strategy_results in latent_results.items():
            for beta, results in strategy_results.items():
                if 'error' not in results:
                    final_metrics = results['final_metrics']
                    model_complexity = results['model_complexity']
                    all_results.append({
                        'latent_dim': latent_dim,
                        'strategy': strategy,
                        'beta': beta,
                        'train_loss': final_metrics['train_loss'],
                        'val_loss': final_metrics['val_loss'],
                        'training_time': results['training_time'],
                        'epochs_trained': results['epochs_trained'],
                        'total_parameters': model_complexity['total_parameters'],
                        'architecture_type': model_complexity['architecture_type']
                    })
        
        if all_results:
            df = pd.DataFrame(all_results)
            df.to_csv(os.path.join(latent_dir, f'latent_{latent_dim}_complete_summary.csv'), index=False)
            
            # Create latent dimension specific analysis
            self._create_latent_dim_analysis(df, latent_dir, latent_dim)
    
    def _create_latent_dim_analysis(self, df: pd.DataFrame, latent_dir: str, latent_dim: int) -> None:
        """Create analysis plots for specific latent dimension."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Strategy comparison
        strategy_means = df.groupby('strategy')['val_loss'].mean()
        ax1.bar(strategy_means.index, strategy_means.values, alpha=0.7, color='skyblue')
        ax1.set_title(f'Strategy Comparison - Latent Dim {latent_dim}')
        ax1.set_ylabel('Mean Validation Loss')
        ax1.grid(True, alpha=0.3)
        
        # Beta value effects
        beta_means = df.groupby('beta')['val_loss'].mean()
        ax2.plot(beta_means.index, beta_means.values, 'o-', markersize=8, linewidth=2)
        ax2.set_title(f'Beta Value Effects - Latent Dim {latent_dim}')
        ax2.set_xlabel('Beta Value')
        ax2.set_ylabel('Mean Validation Loss')
        ax2.grid(True, alpha=0.3)
        
        # Training efficiency
        ax3.scatter(df['training_time'], df['val_loss'], alpha=0.7, s=60)
        ax3.set_title(f'Training Efficiency - Latent Dim {latent_dim}')
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_ylabel('Validation Loss')
        ax3.grid(True, alpha=0.3)
        
        # Model size vs performance
        ax4.scatter(df['total_parameters'], df['val_loss'], alpha=0.7, s=60, c=df['beta'], cmap='viridis')
        ax4.set_title(f'Model Size vs Performance - Latent Dim {latent_dim}')
        ax4.set_xlabel('Total Parameters')
        ax4.set_ylabel('Validation Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(latent_dir, f'latent_{latent_dim}_detailed_analysis.png'), dpi=300)
        plt.close()
    
    def _create_comprehensive_comparison(self) -> None:
        """Create comprehensive comparison across all latent dimensions."""
        # Collect all results
        all_results = []
        for latent_dim, latent_results in self.all_results.items():
            for strategy, strategy_results in latent_results.items():
                for beta, results in strategy_results.items():
                    if 'error' not in results:
                        final_metrics = results['final_metrics']
                        model_complexity = results['model_complexity']
                        all_results.append({
                            'latent_dim': latent_dim,
                            'strategy': strategy,
                            'beta': beta,
                            'train_loss': final_metrics['train_loss'],
                            'val_loss': final_metrics['val_loss'],
                            'training_time': results['training_time'],
                            'epochs_trained': results['epochs_trained'],
                            'total_parameters': model_complexity['total_parameters'],
                            'architecture_type': model_complexity['architecture_type']
                        })
        
        if all_results:
            df = pd.DataFrame(all_results)
            comparison_dir = os.path.join(self.config.paths.MODELS_DIR, 'latent_dimension_comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            
            df.to_csv(os.path.join(comparison_dir, 'comprehensive_results.csv'), index=False)
            
            # Create comprehensive analysis plots
            self._create_comprehensive_plots(df, comparison_dir)
    
    def _create_comprehensive_plots(self, df: pd.DataFrame, comparison_dir: str) -> None:
        """Create comprehensive comparison plots across all latent dimensions."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Latent dimension effects on performance
        latent_means = df.groupby('latent_dim')['val_loss'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(latent_means['latent_dim'], latent_means['mean'], 
                    yerr=latent_means['std'], marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_title('Performance vs Latent Dimension', fontsize=14)
        ax1.set_xlabel('Latent Dimension')
        ax1.set_ylabel('Validation Loss (mean ± std)')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Model complexity scaling
        complexity_means = df.groupby('latent_dim')['total_parameters'].mean()
        ax2.plot(complexity_means.index, complexity_means.values, 'o-', markersize=8, linewidth=2, color='orange')
        ax2.set_title('Model Complexity vs Latent Dimension', fontsize=14)
        ax2.set_xlabel('Latent Dimension')
        ax2.set_ylabel('Total Parameters')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        
        # Training time scaling
        time_means = df.groupby('latent_dim')['training_time'].mean()
        ax3.plot(time_means.index, time_means.values, 's-', markersize=8, linewidth=2, color='green')
        ax3.set_title('Training Time vs Latent Dimension', fontsize=14)
        ax3.set_xlabel('Latent Dimension')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log', base=2)
        
        # Performance heatmap by strategy and latent dimension
        pivot_table = df.groupby(['strategy', 'latent_dim'])['val_loss'].mean().unstack()
        im = ax4.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax4.set_title('Performance Heatmap: Strategy vs Latent Dimension', fontsize=14)
        ax4.set_xlabel('Latent Dimension')
        ax4.set_ylabel('Strategy')
        ax4.set_xticks(range(len(pivot_table.columns)))
        ax4.set_xticklabels(pivot_table.columns)
        ax4.set_yticks(range(len(pivot_table.index)))
        ax4.set_yticklabels(pivot_table.index)
        
        # Add colorbar
        plt.colorbar(im, ax=ax4, label='Validation Loss')
        
        # Add text annotations to heatmap
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                if not np.isnan(pivot_table.iloc[i, j]):
                    ax4.text(j, i, f'{pivot_table.iloc[i, j]:.3f}', 
                            ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'comprehensive_latent_dimension_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create detailed performance summary
        self._create_performance_summary_table(df, comparison_dir)
    
    def _create_performance_summary_table(self, df: pd.DataFrame, comparison_dir: str) -> None:
        """Create detailed performance summary table."""
        # Create summary statistics
        summary_stats = df.groupby(['latent_dim', 'strategy']).agg({
            'val_loss': ['mean', 'std', 'min'],
            'train_loss': ['mean', 'std', 'min'],
            'training_time': ['mean', 'std'],
            'total_parameters': 'mean',
            'epochs_trained': 'mean'
        }).round(4)
        
        # Flatten column names
        summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
        summary_stats = summary_stats.reset_index()
        
        # Save detailed summary
        summary_stats.to_csv(os.path.join(comparison_dir, 'detailed_performance_summary.csv'), index=False)
        
        # Find best performing configurations
        best_configs = df.loc[df.groupby('latent_dim')['val_loss'].idxmin()]
        best_configs_summary = best_configs[['latent_dim', 'strategy', 'beta', 'val_loss', 'total_parameters']].copy()
        best_configs_summary.to_csv(os.path.join(comparison_dir, 'best_configurations_per_latent_dim.csv'), index=False)
        
        logger.info("📊 Comprehensive analysis completed!")
        logger.info("📁 Results saved in latent_dimension_comparison directory")
        
        # Log best configurations
        logger.info("\n🏆 BEST CONFIGURATIONS PER LATENT DIMENSION:")
        for _, row in best_configs_summary.iterrows():
            logger.info(f"  Latent {int(row['latent_dim'])}: {row['strategy']}-β{row['beta']} "
                       f"(val_loss: {row['val_loss']:.4f}, params: {int(row['total_parameters']):,})")

# Factory function for the enhanced trainer
def create_enhanced_trainer(config):
    """Create enhanced trainer with latent dimension support."""
    return EnhancedModelTrainer(config)