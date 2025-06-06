"""
Fixed sampling methods manager with automatic model recovery.
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

from .base import MultiMethodSampler, SamplingResult
from .equiprobable import EquiprobableSampler
from .representative import DistanceBasedSampler
from .cluster_based import ClusterBasedSampler
from .hybrid import HybridSampler

logger = logging.getLogger(__name__)

class SamplingManager:
    """
    Manages multiple sampling methods with automatic model recovery.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_sampler = MultiMethodSampler()
        
        # Available sampling methods (simplified list for core methods)
        self.available_methods = {
            'equiprobable': EquiprobableSampler,
            'distance_based': DistanceBasedSampler,
            'cluster_based': ClusterBasedSampler,
            'hybrid': HybridSampler,
        }
        
        logger.info(f"Sampling manager initialized with {len(self.available_methods)} methods")
    
    def register_method(self, method_name: str, **kwargs) -> None:
        """Register a sampling method."""
        if method_name not in self.available_methods:
            raise ValueError(f"Unknown sampling method: {method_name}")
        
        sampler_class = self.available_methods[method_name]
        sampler = sampler_class(**kwargs)
        self.multi_sampler.register_sampler(method_name, sampler)
        
        logger.info(f"Registered {method_name} sampling method")
    
    def run_sampling_for_model(self, strategy: str, beta: float, 
                              methods: Optional[List[str]] = None) -> Dict[str, Dict[int, SamplingResult]]:
        """Run sampling for a specific trained model with automatic recovery."""
        logger.info(f"Running sampling for {strategy} strategy, beta={beta}")
        
        try:
            # Load model and get latent encodings
            z_latent, original_df = self._load_model_and_data(strategy, beta)
            
            # Set up output directory
            output_dir = os.path.join(self.config.paths.SAMPLES_DIR, strategy, f'beta_{beta}')
            
            # Run multi-method sampling
            results = self.multi_sampler.run_sampling(
                z_latent=z_latent,
                original_df=original_df,
                sample_sizes=self.config.training.SAMPLE_SIZES,
                output_base_dir=output_dir,
                methods=methods
            )
            
            # Create comparison plots
            for sample_size in self.config.training.SAMPLE_SIZES:
                self.multi_sampler.create_comparison_plots(output_dir, sample_size, z_latent)
            
            # Save sampling summary
            self._save_sampling_summary(results, output_dir, strategy, beta)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Sampling failed for {strategy}-{beta}: {e}")
            return {'error': str(e)}
    
    def run_all_sampling(self) -> Dict[str, Dict[str, Any]]:
        """Run sampling for all trained models."""
        all_results = {}
        
        total_configs = len(self.config.training.ANNEALING_STRATEGIES) * len(self.config.training.BETA_VALUES)
        current_config = 0
        
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            strategy_results = {}
            
            for beta in self.config.training.BETA_VALUES:
                current_config += 1
                logger.info(f"\n🎲 Sampling configuration {current_config}/{total_configs}")
                
                results = self.run_sampling_for_model(strategy, beta)
                strategy_results[beta] = results
            
            all_results[strategy] = strategy_results
        
        # Create overall summary
        self._create_overall_summary(all_results)
        
        return all_results
    
    def _load_model_and_data(self, strategy: str, beta: float) -> tuple:
        """
        Load trained model and generate latent encodings with enhanced recovery.
        """
        from models.vae import VAE, get_latent_encoding
        
        model_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
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
                logger.error(f"Model recovery failed for {strategy}-{beta}: {e}")
                
                # Last resort: look for any available model
                alternative_paths = [
                    os.path.join(model_dir, 'vae_model.pth'),
                    os.path.join(model_dir, 'model.pth'),
                    os.path.join(model_dir, 'checkpoint.pth')
                ]
                
                # Also check for checkpoint files
                checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pth"))
                if checkpoint_files:
                    # Use the most recent checkpoint
                    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                    alternative_paths.insert(0, latest_checkpoint)
                
                model_path = None
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        logger.info(f"🔄 Using alternative model: {alt_path}")
                        model_path = alt_path
                        break
                
                if model_path is None:
                    raise FileNotFoundError(f"No model files found for {strategy}-{beta} in {model_dir}")
        
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
            raise Exception(f"Model loading failed for {strategy}-{beta}: {e}")
        
        # Create model instance with error handling
        try:
            # Handle different checkpoint formats
            cat_dict = checkpoint.get('categorical_cardinality', {})
            if isinstance(cat_dict, dict) and not cat_dict:
                # Load from preprocessing objects if empty
                logger.info("🔄 Loading categorical info from preprocessing objects")
                cat_dict = self._load_categorical_info()
            
            model = VAE(
                input_dim=checkpoint['input_dim'],
                num_numerical=checkpoint['num_numerical'],
                hidden_dim=checkpoint.get('hidden_dim', self.config.model.HIDDEN_DIM),
                latent_dim=checkpoint.get('latent_dim', self.config.model.LATENT_DIM),
                cat_dict=cat_dict
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()  # Set to evaluation mode
            
            logger.info(f"✅ Model loaded successfully")
            
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
            
            # Get latent encodings
            data_tensor = torch.FloatTensor(preprocessed_df.values).to(self.device)
            z_latent = get_latent_encoding(model, data_tensor, self.device)
            
            logger.info(f"✅ Loaded model and data: {z_latent.shape} latent points")
            
            return z_latent, original_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise Exception(f"Data loading failed: {e}")
    
    def _reconstruct_checkpoint_info(self, checkpoint: dict, strategy: str, beta: float) -> dict:
        """Reconstruct missing checkpoint information."""
        logger.info("🔧 Reconstructing missing checkpoint information")
        
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
            # Try to infer from model state dict
            try:
                first_layer = None
                for key in checkpoint['model_state_dict'].keys():
                    if 'encoder' in key and 'weight' in key:
                        first_layer = checkpoint['model_state_dict'][key]
                        break
                
                if first_layer is not None:
                    checkpoint['input_dim'] = first_layer.shape[1]
                    logger.info(f"🔧 Inferred input_dim: {checkpoint['input_dim']}")
                else:
                    checkpoint['input_dim'] = self.config.model.HIDDEN_DIM * 2  # Fallback
                    
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
        
        # Find all checkpoint files
        checkpoint_pattern = os.path.join(model_dir, "checkpoint_epoch_*_val_loss_*.pth")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            raise Exception(f"No checkpoint files found in {model_dir}")
        
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
        final_model_path = os.path.join(model_dir, 'vae_model_final.pth')
        shutil.copy2(best_checkpoint, final_model_path)
        
        # Verify copy
        if not os.path.exists(final_model_path):
            raise Exception("Failed to copy checkpoint as final model")
        
        file_size = os.path.getsize(final_model_path)
        logger.info(f"✅ Recovered: {os.path.basename(best_checkpoint)} → vae_model_final.pth")
        logger.info(f"   Validation loss: {best_loss:.4f}, Size: {file_size:,} bytes")
    
    def _save_sampling_summary(self, results: Dict[str, Dict[int, SamplingResult]], 
                              output_dir: str, strategy: str, beta: float) -> None:
        """Save sampling summary for a model configuration."""
        if 'error' in results:
            return
            
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
                        'success': True
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(os.path.join(output_dir, 'sampling_summary.csv'), index=False)
            logger.info(f"📊 Sampling summary saved for {strategy}-{beta}")
    
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
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            df.to_csv(summary_path, index=False)
            
            # Log summary statistics
            total_runs = len(df[df['success'] == True])
            failed_runs = len(df[df['success'] == False])
            
            logger.info(f"📊 Overall sampling summary:")
            logger.info(f"  ✅ Successful runs: {total_runs}")
            logger.info(f"  ❌ Failed runs: {failed_runs}")
            
            if total_runs > 0:
                methods_used = df[df['success'] == True]['method'].unique()
                logger.info(f"  📋 Methods used: {', '.join(methods_used)}")

def create_default_sampling_manager(config) -> SamplingManager:
    """Create a sampling manager with default methods registered."""
    manager = SamplingManager(config)
    
    # Register default methods with parameters from config
    try:
        sampling_params = config.get_sampling_params()
    except AttributeError:
        # Fallback if method doesn't exist
        sampling_params = {
            'info_weight': 1.0,
            'redundancy_weight': 1.0,
            'coverage_radius': 0.2,
            'candidate_fraction': 1.0
        }
    
    # Get default methods from config or use fallback
    try:
        default_methods = config.sampling.DEFAULT_METHODS
    except AttributeError:
        default_methods = ['equiprobable', 'distance_based', 'cluster_based', 'hybrid']
    
    for method in default_methods:
        try:
            manager.register_method(method, **sampling_params)
        except Exception as e:
            logger.warning(f"Failed to register {method}: {e}")
    
    return manager