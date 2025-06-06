"""
Optimized model training system for VAE pipeline.
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

from .vae import VAE, VAELoss, BetaScheduler, create_model
from .callbacks import EarlyStopping, ModelCheckpoint, CallbackList

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles training of VAE models with different configurations."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_random_seeds()
        
        # Setup data
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
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
    
    def train_single_model(self, strategy: str, beta: float, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """Train a single VAE model."""
        logger.info(f"Training model: strategy={strategy}, beta={beta}")
        
        if save_dir is None:
            save_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Create model and training components
        model = self._create_model()
        optimizer = optim.Adam(model.parameters(), lr=self.config.model.LEARNING_RATE)
        loss_fn = VAELoss(
            self.preprocessing_objects['categorical_cardinality'],
            len(self.preprocessing_objects['num_cols'])
        )
        
        # Get beta schedule
        beta_schedule = BetaScheduler.get_schedule(
            beta_start=0.0, beta_end=beta, num_epochs=self.config.model.NUM_EPOCHS, strategy=strategy
        )
        
        # Setup callbacks
        callbacks = self._setup_callbacks(save_dir)
        
        # Training loop
        training_results = self._training_loop(model, optimizer, loss_fn, beta_schedule, callbacks)
        
        # Save final model
        self._save_model(model, optimizer, training_results, save_dir, strategy, beta)
        
        return training_results
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Train models for all strategy-beta combinations."""
        all_results = {}
        
        total_models = len(self.config.training.ANNEALING_STRATEGIES) * len(self.config.training.BETA_VALUES)
        current_model = 0
        
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            strategy_results = {}
            
            for beta in self.config.training.BETA_VALUES:
                current_model += 1
                logger.info(f"\n🧠 Training model {current_model}/{total_models}")
                
                try:
                    results = self.train_single_model(strategy, beta)
                    strategy_results[beta] = results
                except Exception as e:
                    logger.error(f"Failed to train model {strategy}-{beta}: {e}")
                    strategy_results[beta] = {'error': str(e)}
            
            all_results[strategy] = strategy_results
            self._create_strategy_summary(strategy, strategy_results)
        
        return all_results
    
    def _create_model(self) -> VAE:
        """Create VAE model instance."""
        return create_model(
            input_dim=len(self.train_loader.dataset[0][0]),
            num_numerical=len(self.preprocessing_objects['num_cols']),
            cat_dict=self.preprocessing_objects['categorical_cardinality'],
            hidden_dim=self.config.model.HIDDEN_DIM,
            latent_dim=self.config.model.LATENT_DIM
        ).to(self.device)
    
    def _setup_callbacks(self, save_dir: str) -> CallbackList:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        if self.config.training.EARLY_STOPPING:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                min_delta=self.config.training.EARLY_STOPPING_MIN_DELTA,
                patience=self.config.training.EARLY_STOPPING_PATIENCE,
                verbose=True,
                restore_best_weights=self.config.training.RESTORE_BEST_WEIGHTS
            ))
        
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=os.path.join(save_dir, 'checkpoint_epoch_{epoch:03d}_val_loss_{val_loss:.4f}.pth'),
            monitor=self.config.training.CHECKPOINT_MONITOR,
            save_best_only=self.config.training.SAVE_BEST_ONLY,
            mode=self.config.training.CHECKPOINT_MODE
        ))
        
        return CallbackList(callbacks)
    
    def _training_loop(self, model: VAE, optimizer: torch.optim.Optimizer, 
                      loss_fn: VAELoss, beta_schedule: List[float], 
                      callbacks: CallbackList) -> Dict[str, Any]:
        """Consolidated training loop for both training and validation."""
        history = {
            'train_loss': [], 'val_loss': [], 'num_loss': [], 
            'cat_loss': [], 'kl_loss': [], 'beta_values': []
        }
        
        callbacks.set_model(model)
        callbacks.on_train_begin()
        
        start_time = time.time()
        
        for epoch in range(self.config.model.NUM_EPOCHS):
            current_beta = beta_schedule[epoch]
            callbacks.on_epoch_begin(epoch)
            
            # Training phase
            train_metrics = self._run_epoch(model, self.train_loader, optimizer, loss_fn, current_beta, is_training=True)
            
            # Validation phase  
            val_metrics = self._run_epoch(model, self.val_loader, None, loss_fn, current_beta, is_training=False)
            
            # Update history
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_metrics['total_loss'])
            history['num_loss'].append(train_metrics['num_loss'])
            history['cat_loss'].append(train_metrics['cat_loss'])
            history['kl_loss'].append(train_metrics['kl_loss'])
            history['beta_values'].append(current_beta)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == self.config.model.NUM_EPOCHS - 1:
                logger.info(f'Epoch {epoch+1}/{self.config.model.NUM_EPOCHS}: '
                          f'Train: {train_metrics["total_loss"]:.4f}, '
                          f'Val: {val_metrics["total_loss"]:.4f}, Beta: {current_beta:.2f}')
            
            # Callback logging
            logs = {
                'loss': train_metrics['total_loss'], 'val_loss': val_metrics['total_loss'],
                'num_loss': train_metrics['num_loss'], 'cat_loss': train_metrics['cat_loss'],
                'kl_loss': train_metrics['kl_loss'], 'beta': current_beta
            }
            
            callbacks.on_epoch_end(epoch, logs)
            
            if callbacks.get_stop_training():
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        callbacks.on_train_end()
        
        # Save results
        training_time = time.time() - start_time
        self._save_training_artifacts(history, os.path.dirname(callbacks.callbacks[1].filepath))
        
        return {
            'history': history,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'final_metrics': {
                'train_loss': history['train_loss'][-1],
                'val_loss': history['val_loss'][-1]
            }
        }
    
    def _run_epoch(self, model: VAE, data_loader: DataLoader, optimizer: Optional[torch.optim.Optimizer],
                  loss_fn: VAELoss, beta: float, is_training: bool) -> Dict[str, float]:
        """Unified epoch runner for training and validation."""
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
    
    def _save_model(self, model: VAE, optimizer: torch.optim.Optimizer, 
                   training_results: Dict[str, Any], save_dir: str, strategy: str, beta: float) -> None:
        """Save trained model and metadata with enhanced error handling."""
        try:
            # Ensure directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Create checkpoint with all necessary information
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'categorical_cardinality': self.preprocessing_objects['categorical_cardinality'],
                'num_numerical': len(self.preprocessing_objects['num_cols']),
                'input_dim': model.input_dim,
                'hidden_dim': self.config.model.HIDDEN_DIM,
                'latent_dim': self.config.model.LATENT_DIM,
                'beta': beta,
                'strategy': strategy,
                'training_results': training_results,
                'config': {
                    'batch_size': self.config.model.BATCH_SIZE,
                    'learning_rate': self.config.model.LEARNING_RATE,
                    'num_epochs': self.config.model.NUM_EPOCHS
                }
            }
            
            model_path = os.path.join(save_dir, 'vae_model_final.pth')
            
            # Save with enhanced verification
            torch.save(checkpoint, model_path)
            
            # Multiple verification checks
            if not os.path.exists(model_path):
                raise Exception("Model file was not created")
            
            file_size = os.path.getsize(model_path)
            if file_size < 1024:  # Less than 1KB indicates corruption
                raise Exception(f"Model file too small ({file_size} bytes), likely corrupted")
            
            # Test loading the saved model
            try:
                test_checkpoint = torch.load(model_path, map_location='cpu')
                required_keys = ['model_state_dict', 'input_dim', 'num_numerical', 'categorical_cardinality']
                missing_keys = [key for key in required_keys if key not in test_checkpoint]
                if missing_keys:
                    raise Exception(f"Missing keys in checkpoint: {missing_keys}")
                    
            except Exception as load_error:
                raise Exception(f"Saved model failed verification: {load_error}")
            
            logger.info(f"✅ Model saved and verified successfully: {model_path} ({file_size:,} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final model: {e}")
            
            # Enhanced fallback: copy the best checkpoint as final model
            try:
                logger.info("🔄 Attempting fallback: copying best checkpoint as final model")
                self._copy_best_checkpoint_as_final(save_dir)
                
                # Verify the fallback worked
                model_path = os.path.join(save_dir, 'vae_model_final.pth')
                if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:
                    logger.info("✅ Fallback model creation successful")
                else:
                    raise Exception("Fallback model creation failed verification")
                    
            except Exception as fallback_error:
                logger.error(f"❌ Fallback also failed: {fallback_error}")
                # Final fallback - save minimal working model
                try:
                    self._save_minimal_model(model, save_dir, strategy, beta)
                except Exception as final_error:
                    logger.error(f"❌ All save attempts failed: {final_error}")
                    raise Exception(f"Complete model saving failure: {e}")
    
    def _save_minimal_model(self, model: VAE, save_dir: str, strategy: str, beta: float) -> None:
        """Last resort: save minimal working model."""
        logger.info("🆘 Attempting minimal model save as last resort")
        
        minimal_checkpoint = {
            'model_state_dict': model.state_dict(),
            'input_dim': model.input_dim,
            'num_numerical': model.num_numerical,
            'categorical_cardinality': model.cat_dict,
            'hidden_dim': self.config.model.HIDDEN_DIM,
            'latent_dim': self.config.model.LATENT_DIM,
            'beta': beta,
            'strategy': strategy
        }
        
        model_path = os.path.join(save_dir, 'vae_model_final.pth')
        torch.save(minimal_checkpoint, model_path)
        
        if os.path.exists(model_path):
            logger.info(f"✅ Minimal model saved: {model_path}")
        else:
            raise Exception("Even minimal model save failed")
    
    def _copy_best_checkpoint_as_final(self, save_dir: str) -> None:
        """Copy the best checkpoint as the final model."""
        
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(save_dir, "checkpoint_epoch_*_val_loss_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise Exception(f"No checkpoint files found in {save_dir}")
        
        logger.info(f"🔍 Found {len(checkpoint_files)} checkpoint files")
        
        # Extract validation loss and find the best one
        def get_val_loss(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'val_loss_(\d+\.?\d*)', filename)
            return float(match.group(1)) if match else float('inf')
        
        # Find best checkpoint
        checkpoint_losses = [(f, get_val_loss(f)) for f in checkpoint_files]
        valid_checkpoints = [(f, loss) for f, loss in checkpoint_losses if loss != float('inf')]
        
        if not valid_checkpoints:
            raise Exception("No valid checkpoints found (could not parse validation losses)")
        
        best_checkpoint, best_loss = min(valid_checkpoints, key=lambda x: x[1])
        
        # Copy as final model
        final_model_path = os.path.join(save_dir, 'vae_model_final.pth')
        shutil.copy2(best_checkpoint, final_model_path)
        
        # Verify copy
        if not os.path.exists(final_model_path):
            raise Exception("Failed to copy checkpoint as final model")
        
        file_size = os.path.getsize(final_model_path)
        logger.info(f"✅ Recovered: {os.path.basename(best_checkpoint)} → vae_model_final.pth")
        logger.info(f"   Validation loss: {best_loss:.4f}, Size: {file_size:,} bytes")
    
    def _save_training_artifacts(self, history: Dict[str, List], save_dir: str) -> None:
        """Save training history and create plots."""
        # Save history
        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)
        
        # Create plots
        self._create_training_plots(history, save_dir)
    
    def _create_training_plots(self, history: Dict[str, List], save_dir: str) -> None:
        """Create consolidated training visualization plots."""
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main losses with beta schedule
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    def _create_strategy_summary(self, strategy: str, strategy_results: Dict[str, Any]) -> None:
        """Create summary for a single strategy."""
        strategy_dir = os.path.join(self.config.paths.MODELS_DIR, strategy)
        
        summary_data = []
        for beta, results in strategy_results.items():
            if 'error' not in results:
                final_metrics = results['final_metrics']
                summary_data.append({
                    'beta': beta,
                    'train_loss': final_metrics['train_loss'],
                    'val_loss': final_metrics['val_loss'],
                    'training_time': results['training_time'],
                    'epochs_trained': results['epochs_trained']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(strategy_dir, 'beta_results_summary.csv'), index=False)
            
            # Create trade-off plot
            self._create_tradeoff_plot(df, strategy_dir, strategy)
    
    def _create_tradeoff_plot(self, df: pd.DataFrame, strategy_dir: str, strategy: str) -> None:
        """Create reconstruction vs KL trade-off plot."""
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(df['val_loss'], df['train_loss'], s=100, c=df['beta'], cmap='viridis')
        
        for _, row in df.iterrows():
            plt.annotate(f'β={row["beta"]}', (row['val_loss'], row['train_loss']),
                        xytext=(10, 5), textcoords='offset points', fontsize=10)
        
        plt.colorbar(scatter, label='Beta Value')
        plt.title(f'VAE Trade-off: Training vs Validation Loss ({strategy} annealing)')
        plt.xlabel('Validation Loss')
        plt.ylabel('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(strategy_dir, 'vae_tradeoff.png'), dpi=300)
        plt.close()