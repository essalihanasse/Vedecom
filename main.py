"""
Main orchestrator for the clean VAE pipeline - OPTIMIZED VERSION
"""

import os
import sys
import time
import argparse
import logging
import glob
import re
import shutil
from datetime import timedelta
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VAEPipeline:
    """Main VAE pipeline orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        try:
            from config.settings import config
            self.config = config
        except ImportError:
            logger.error("Could not import configuration.")
            raise
        
        self.results = {}
        self.execution_times = {}
    
    def run(self, stages: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Run the complete VAE pipeline."""
        start_time = time.time()
        logger.info("🚀 Starting clean VAE pipeline...")
        
        stages = stages or ['preprocess', 'train', 'visualize', 'sample', 'test']
        
        # Override configuration with provided kwargs
        self._update_config(kwargs)
        
        # Run stages
        for stage in stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔄 Running stage: {stage.upper()}")
            logger.info(f"{'='*60}")
            
            stage_start = time.time()
            
            try:
                method_name = f'_run_{stage}'
                if hasattr(self, method_name):
                    getattr(self, method_name)(**kwargs)
                else:
                    logger.warning(f"Unknown stage: {stage}")
                    continue
                    
                stage_time = time.time() - stage_start
                self.execution_times[stage] = stage_time
                logger.info(f"✅ Stage {stage} completed in {timedelta(seconds=int(stage_time))}")
                
            except Exception as e:
                logger.error(f"❌ Pipeline failed during {stage} stage: {e}")
                raise
        
        total_time = time.time() - start_time
        self.execution_times['total'] = total_time
        
        logger.info(f"\n{'='*80}")
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"⏱️ Total execution time: {timedelta(seconds=int(total_time))}")
        
        return {
            'stages_completed': stages,
            'execution_times': self.execution_times,
            'results': self.results
        }
    
    def _update_config(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration with provided parameters."""
        if kwargs.get('strategy'):
            self.config.training.ANNEALING_STRATEGIES = [kwargs['strategy']]
        if kwargs.get('beta'):
            self.config.training.BETA_VALUES = [kwargs['beta']]
    
    def _run_preprocess(self, **kwargs) -> None:
        """Run data preprocessing stage."""
        from data.preprocessing import preprocess_data
        
        original_df, preprocessed_df, metadata = preprocess_data(
            data_file=self.config.paths.DATA_FILE,
            output_dir=self.config.paths.DATA_DIR,
            categorical_cols=self.config.data.CATEGORICAL_COLS,
            numerical_cols=self.config.data.NUMERICAL_COLS
        )
        
        self.results['preprocessing'] = {
            'metadata': metadata,
            'original_shape': original_df.shape,
            'preprocessed_shape': preprocessed_df.shape
        }
        
        logger.info(f"📊 Preprocessed {metadata['n_samples_original']} → {metadata['n_samples_final']} samples")
    
    def _run_train(self, **kwargs) -> None:
        """Run model training stage."""
        from models.training import ModelTrainer
        
        trainer = ModelTrainer(self.config)
        training_results = trainer.train_all_models()
        self.results['training'] = training_results
        logger.info(f"🧠 Trained {len(training_results)} models successfully")
    
    def _run_visualize(self, **kwargs) -> None:
        """Run latent space visualization stage."""
        from visualization.latent_viz import LatentVisualizer
        
        try:
            visualizer = LatentVisualizer(self.config)
            viz_results = visualizer.create_all_visualizations()
            self.results['visualization'] = viz_results
            logger.info("🎨 Created visualizations for all trained models")
        except Exception as e:
            logger.warning(f"Visualization failed: {e}. Continuing...")
    
    def _run_sample(self, **kwargs) -> None:
        """Run sampling stage with enhanced error handling and model recovery."""
        logger.info("🎲 Starting sampling stage...")
        
        # Enhanced import handling
        try:
            # Try direct import first
            from sampling.manager import SamplingManager, create_default_sampling_manager
            logger.info("✅ Successfully imported SamplingManager")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import SamplingManager: {e}")
            
            # Try alternative import paths
            # Add multiple potential paths
            potential_paths = [
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                os.getcwd()
            ]
            
            for path in potential_paths:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            try:
                from sampling.manager import SamplingManager, create_default_sampling_manager
                logger.info("✅ Successfully imported SamplingManager with path adjustment")
            except ImportError as e2:
                logger.error(f"❌ All import attempts failed: {e2}")
                logger.warning("🔄 Using enhanced fallback sampling")
                return self._run_enhanced_sample_fallback(**kwargs)
        
        # Check if models exist before attempting sampling
        missing_models = self._check_model_availability()
        if missing_models:
            logger.warning(f"⚠️ Some models missing: {missing_models}")
            logger.info("🔄 Attempting to recover missing models...")
            
            # Try to recover missing models
            recovered = self._attempt_model_recovery(missing_models)
            if recovered:
                logger.info(f"✅ Recovered {recovered} models")
            
            # Re-check after recovery
            still_missing = self._check_model_availability()
            if still_missing:
                logger.error(f"❌ Could not recover models: {still_missing}")
                logger.info("💡 Try running training first: python main.py --stages train")
                # Continue with available models instead of failing completely
        
        # Create and configure sampling manager
        try:
            manager = SamplingManager(self.config)
            logger.info("✅ SamplingManager created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create SamplingManager: {e}")
            return self._run_enhanced_sample_fallback(**kwargs)
        
        # Register sampling methods with error handling
        methods = kwargs.get('sampling_methods', self.config.sampling.DEFAULT_METHODS)
        params = kwargs.get('sampling_params', self.config.get_sampling_params())
        
        # Handle 'all' methods selection
        if 'all' in methods:
            methods = ['equiprobable', 'distance_based', 'cluster_based', 'hybrid']
        
        successfully_registered = []
        for method in methods:
            try:
                manager.register_method(method, **params)
                successfully_registered.append(method)
                logger.info(f"✅ Registered method: {method}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to register {method}: {e}")
        
        if not successfully_registered:
            logger.error("❌ No sampling methods could be registered")
            return self._run_enhanced_sample_fallback(**kwargs)
        
        logger.info(f"📋 Successfully registered {len(successfully_registered)} methods: {successfully_registered}")
        
        # Run sampling with automatic recovery
        try:
            sampling_results = manager.run_all_sampling()
            self.results['sampling'] = sampling_results
            
            # Count successful runs
            successful_runs = 0
            total_runs = 0
            for strategy_results in sampling_results.values():
                for beta_results in strategy_results.values():
                    total_runs += 1
                    if 'error' not in beta_results:
                        successful_runs += len(beta_results)
            
            logger.info(f"🎲 Sampling completed: {successful_runs} successful runs out of {total_runs} total configurations")
            
            if successful_runs == 0:
                logger.warning("⚠️ No sampling runs succeeded - check model availability and configuration")
            
        except Exception as e:
            logger.error(f"❌ Sampling execution failed: {e}")
            self.results['sampling'] = {'error': str(e)}
    
    def _check_model_availability(self) -> List[str]:
        """Check which models are missing and return list of missing model identifiers."""
        missing_models = []
        
        for strategy in self.config.training.ANNEALING_STRATEGIES:
            for beta in self.config.training.BETA_VALUES:
                model_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
                model_path = os.path.join(model_dir, 'vae_model_final.pth')
                
                if not os.path.exists(model_path):
                    # Check for alternative model files
                    alternative_files = [
                        os.path.join(model_dir, 'vae_model.pth'),
                        os.path.join(model_dir, 'model.pth')
                    ]
                    
                    # Check for checkpoint files
                    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pth"))
                    
                    if not any(os.path.exists(f) for f in alternative_files) and not checkpoint_files:
                        missing_models.append(f"{strategy}-{beta}")
                    else:
                        logger.info(f"🔍 Found alternative model files for {strategy}-{beta}")
        
        return missing_models
    
    def _attempt_model_recovery(self, missing_models: List[str]) -> int:
        """Attempt to recover missing models from checkpoints."""
        recovered_count = 0
        
        for model_id in missing_models:
            try:
                strategy, beta_str = model_id.split('-')
                beta = float(beta_str)
                
                model_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
                
                # Try to recover from checkpoints
                if os.path.exists(model_dir):
                    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_epoch_*.pth"))
                    
                    if checkpoint_files:
                        # Find the best checkpoint
                        best_checkpoint = self._find_best_checkpoint(checkpoint_files)
                        if best_checkpoint:
                            final_path = os.path.join(model_dir, 'vae_model_final.pth')
                            shutil.copy2(best_checkpoint, final_path)
                            
                            if os.path.exists(final_path):
                                logger.info(f"✅ Recovered model for {model_id}")
                                recovered_count += 1
                            
            except Exception as e:
                logger.warning(f"⚠️ Could not recover {model_id}: {e}")
        
        return recovered_count
    
    def _find_best_checkpoint(self, checkpoint_files: List[str]) -> str:
        """Find the best checkpoint based on validation loss."""
        
        best_checkpoint = None
        best_loss = float('inf')
        
        for checkpoint_file in checkpoint_files:
            try:
                # Extract validation loss from filename
                filename = os.path.basename(checkpoint_file)
                match = re.search(r'val_loss_(\d+\.?\d*)', filename)
                
                if match:
                    val_loss = float(match.group(1))
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_checkpoint = checkpoint_file
                else:
                    # If no loss in filename, use the most recent file
                    if best_checkpoint is None:
                        best_checkpoint = checkpoint_file
                        
            except Exception as e:
                logger.warning(f"Could not parse checkpoint {checkpoint_file}: {e}")
        
        return best_checkpoint
    
    def _run_enhanced_sample_fallback(self, **kwargs) -> None:
        """Enhanced fallback sampling with better diagnostics and recovery."""
        logger.warning("🔄 Using enhanced fallback sampling mode")
        
        # Enhanced diagnostics
        missing_models = self._check_model_availability()
        
        if missing_models:
            logger.error(f"❌ Missing final models for: {', '.join(missing_models)}")
            
            # Check for partial models
            partial_models = []
            for model_id in missing_models:
                strategy, beta_str = model_id.split('-')
                beta = float(beta_str)
                model_dir = os.path.join(self.config.paths.MODELS_DIR, strategy, f'beta_{beta}')
                
                if os.path.exists(model_dir):
                    checkpoint_files = glob.glob(os.path.join(model_dir, "checkpoint_*.pth"))
                    if checkpoint_files:
                        partial_models.append(model_id)
            
            if partial_models:
                logger.info(f"🔍 Found checkpoints for: {', '.join(partial_models)}")
                logger.info("💡 These can potentially be recovered. Try:")
                logger.info("   python main.py --stages sample  # This will attempt automatic recovery")
            
            logger.info("💡 To generate missing models, run: python main.py --stages train")
            
            self.results['sampling'] = {
                'error': f'Missing models: {missing_models}',
                'recoverable': partial_models,
                'total_missing': len(missing_models)
            }
        else:
            logger.info("✅ All models found, but sampling functionality limited in fallback mode")
            logger.info("💡 Try restarting or check import paths for full functionality")
            
            self.results['sampling'] = {
                'status': 'Models found but sampling not implemented in fallback',
                'available_models': len(self.config.training.ANNEALING_STRATEGIES) * len(self.config.training.BETA_VALUES)
            }
    
    def _run_test(self, **kwargs) -> None:
        """Run distribution testing stage."""
        from evaluation.testing import DistributionTester
        
        try:
            tester = DistributionTester(self.config)
            test_results = tester.run_all_tests()
            self.results['testing'] = test_results
            logger.info("🧪 Completed distribution testing")
        except Exception as e:
            logger.warning(f"Testing failed: {e}. Continuing...")

def main():
    """Main entry point with simplified argument parsing."""
    parser = argparse.ArgumentParser(description='Clean VAE Pipeline')
    
    # Core options
    parser.add_argument('--stages', nargs='+', 
                       choices=['preprocess', 'train', 'visualize', 'sample', 'test'],
                       help='Stages to run (default: all)')
    parser.add_argument('--strategy', choices=['linear', 'exponential', 'constant', 'cyclical'])
    parser.add_argument('--beta', type=float)
    parser.add_argument('--methods', nargs='+', 
                       choices=['equiprobable', 'distance_based', 'cluster_based', 'hybrid',
                               'density_aware_kde', 'optimal_transport_greedy', 'all'],
                       default=['equiprobable'])
    
    # Sampling parameters
    parser.add_argument('--info-weight', type=float, default=1.0)
    parser.add_argument('--redundancy-weight', type=float, default=1.0)
    parser.add_argument('--coverage-radius', type=float, default=0.2)
    
    # General options
    parser.add_argument('--fast', action='store_true', help='Fast mode for testing')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.fast:
        os.environ['VAE_FAST_MODE'] = '1'
        logger.info("⚡ Fast mode enabled")
    
    if 'all' in args.methods:
        args.methods = ['equiprobable', 'distance_based', 'cluster_based', 'hybrid']
    
    # Prepare parameters
    sampling_params = {
        'info_weight': args.info_weight,
        'redundancy_weight': args.redundancy_weight,
        'coverage_radius': args.coverage_radius,
    }
    
    try:
        pipeline = VAEPipeline()
        results = pipeline.run(
            stages=args.stages,
            strategy=args.strategy,
            beta=args.beta,
            sampling_methods=args.methods,
            sampling_params=sampling_params
        )
        
        logger.info("🎯 Pipeline completed successfully!")
        logger.info(f"📁 Results available in: {pipeline.config.paths.OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()