"""
Updated main orchestrator for VAE pipeline with multiple latent dimensions support.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

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

class EnhancedVAEPipeline:
    """Enhanced VAE pipeline with multiple latent dimensions support."""
    
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
        """Run the enhanced VAE pipeline with latent dimension support."""
        start_time = time.time()
        logger.info("🚀 Starting enhanced VAE pipeline with multiple latent dimensions...")
        logger.info(f"📏 Latent dimensions to explore: {self.config.model.LATENT_DIMS}")
        
        stages = stages or ['preprocess', 'train', 'visualize', 'sample', 'test']
        
        # Override configuration with provided kwargs
        self._update_config(kwargs)
        
        # Run stages
        for stage in stages:
            logger.info(f"\n{'='*70}")
            logger.info(f"🔄 Running stage: {stage.upper()}")
            logger.info(f"{'='*70}")
            
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
        logger.info("🎉 Enhanced pipeline completed successfully!")
        logger.info(f"⏱️ Total execution time: {timedelta(seconds=int(total_time))}")
        logger.info(f"📏 Latent dimensions explored: {len(self.config.model.LATENT_DIMS)}")
        
        return {
            'stages_completed': stages,
            'execution_times': self.execution_times,
            'results': self.results,
            'latent_dims_explored': self.config.model.LATENT_DIMS
        }
    
    def _update_config(self, kwargs: Dict[str, Any]) -> None:
        """Update configuration with provided parameters."""
        if kwargs.get('strategy'):
            self.config.training.ANNEALING_STRATEGIES = [kwargs['strategy']]
        if kwargs.get('beta'):
            self.config.training.BETA_VALUES = [kwargs['beta']]
        if kwargs.get('latent_dims'):
            self.config.model.LATENT_DIMS = kwargs['latent_dims']
        
        # Log configuration
        logger.info(f"📋 Strategies: {self.config.training.ANNEALING_STRATEGIES}")
        logger.info(f"📊 Beta values: {self.config.training.BETA_VALUES}")
        logger.info(f"📏 Latent dimensions: {self.config.model.LATENT_DIMS}")
    
    def _run_preprocess(self, **kwargs) -> None:
        """Run data preprocessing stage."""
        from data.preprocessing import preprocess_data
        
        logger.info("📊 Preprocessing data for multiple latent dimension experiments...")
        
        original_df, preprocessed_df, metadata = preprocess_data(
            data_file=self.config.paths.DATA_FILE,
            output_dir=self.config.paths.DATA_DIR,
            categorical_cols=self.config.data.CATEGORICAL_COLS,
            numerical_cols=self.config.data.NUMERICAL_COLS
        )
        
        self.results['preprocessing'] = {
            'metadata': metadata,
            'original_shape': original_df.shape,
            'preprocessed_shape': preprocessed_df.shape,
            'input_dim': preprocessed_df.shape[1]
        }
        
        logger.info(f"📊 Preprocessed {metadata['n_samples_original']} → {metadata['n_samples_final']} samples")
        logger.info(f"📐 Input dimension: {preprocessed_df.shape[1]} features")
    
    def _run_train(self, **kwargs) -> None:
        """Run enhanced model training with multiple latent dimensions."""
        from models.training import create_enhanced_trainer
        
        logger.info("🧠 Starting enhanced training across multiple latent dimensions...")
        
        trainer = create_enhanced_trainer(self.config)
        training_results = trainer.train_all_models_with_latent_dims()
        
        self.results['training'] = training_results
        
        # Count successful trainings
        total_models = 0
        successful_models = 0
        
        for latent_dim, latent_results in training_results.items():
            for strategy, strategy_results in latent_results.items():
                for beta, results in strategy_results.items():
                    total_models += 1
                    if 'error' not in results:
                        successful_models += 1
        
        logger.info(f"🧠 Training completed: {successful_models}/{total_models} models successful")
        logger.info(f"📏 Latent dimensions trained: {list(training_results.keys())}")
    
    def _run_visualize(self, **kwargs) -> None:
        """Run enhanced latent space visualization for multiple dimensions."""
        from visualization.latent_viz import create_enhanced_visualizer
        
        try:
            logger.info("🎨 Creating visualizations for all latent dimensions...")
            
            visualizer = create_enhanced_visualizer(self.config)
            viz_results = visualizer.create_all_visualizations_with_latent_dims()
            self.results['visualization'] = viz_results
            
            # Count visualizations created
            total_viz = sum(len(latent_viz) for latent_viz in viz_results.values())
            logger.info(f"🎨 Created {total_viz} visualization sets across {len(viz_results)} latent dimensions")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}. Continuing...")
            # Create fallback visualizations for available models
            self._create_fallback_visualizations()
    
    def _run_sample(self, **kwargs) -> None:
        """Run sampling with latent dimension awareness."""
        logger.info("🎲 Starting sampling across multiple latent dimensions...")
        
        try:
            from sampling.manager import create_enhanced_sampling_manager
            
            # Get sampling methods
            sampling_methods = kwargs.get('sampling_methods', ['cluster_based', 'equiprobable', 'latin_hypercube'])
            
            if 'all' in sampling_methods:
                sampling_methods = ['cluster_based', 'equiprobable', 'latin_hypercube', 'adaptive_latin_hypercube']
            
            manager = create_enhanced_sampling_manager(self.config)
            
            # Register methods for each latent dimension
            for latent_dim in self.config.model.LATENT_DIMS:
                sampling_params = self.config.get_sampling_params(latent_dim)
                
                for method in sampling_methods:
                    try:
                        manager.register_method_for_latent_dim(method, latent_dim, **sampling_params)
                    except Exception as e:
                        logger.warning(f"Failed to register {method} for latent_dim {latent_dim}: {e}")
            
            # Run sampling for all latent dimensions
            sampling_results = manager.run_all_sampling_with_latent_dims()
            self.results['sampling'] = sampling_results
            
            # Count successful sampling runs
            total_runs = 0
            successful_runs = 0
            
            for latent_dim, latent_results in sampling_results.items():
                for strategy, strategy_results in latent_results.items():
                    for beta, beta_results in strategy_results.items():
                        total_runs += 1
                        if 'error' not in beta_results:
                            successful_runs += len(beta_results)
            
            logger.info(f"🎲 Sampling completed: {successful_runs} successful runs across {len(sampling_results)} latent dimensions")
            
        except Exception as e:
            logger.error(f"❌ Sampling failed: {e}")
            self.results['sampling'] = {'error': str(e)}
    
    def _run_test(self, **kwargs) -> None:
        """Run enhanced testing with latent dimension analysis."""
        from evaluation.testing import create_enhanced_tester
        
        try:
            logger.info("🧪 Starting enhanced testing across latent dimensions...")
            
            tester = create_enhanced_tester(self.config)
            test_results = tester.run_all_tests_with_latent_dims()
            self.results['testing'] = test_results
            
            # Summary of testing results
            if 'results' in test_results:
                total_tests = test_results.get('total_tests', 0)
                latent_dims_tested = test_results.get('latent_dims_tested', [])
                
                logger.info(f"🧪 Testing completed: {total_tests} method evaluations")
                logger.info(f"📏 Latent dimensions tested: {latent_dims_tested}")
                logger.info("📊 Enhanced results include:")
                logger.info("  - Wasserstein distance rankings per latent dimension")
                logger.info("  - Statistical significance testing")
                logger.info("  - Latent dimension performance comparison")
                logger.info("  - Bootstrap confidence intervals")
            else:
                logger.warning("No test results generated")
                
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            self.results['testing'] = {'error': str(e)}
    
    def _create_fallback_visualizations(self) -> None:
        """Create basic visualizations if enhanced version fails."""
        logger.info("🔄 Creating fallback visualizations...")
        
        try:
            from visualization.latent_viz import LatentVisualizer
            
            visualizer = LatentVisualizer(self.config)
            
            # Try to create visualizations for available models
            available_models = self._find_available_models()
            
            if available_models:
                basic_viz_results = {}
                for latent_dim, models in available_models.items():
                    if models:
                        basic_viz_results[latent_dim] = len(models)
                
                self.results['visualization'] = {
                    'status': 'fallback_mode',
                    'basic_visualizations': basic_viz_results
                }
                
                logger.info(f"🎨 Created fallback visualizations for {len(basic_viz_results)} latent dimensions")
            else:
                logger.warning("No trained models found for visualization")
                
        except Exception as e:
            logger.warning(f"Fallback visualization also failed: {e}")
    
    def _find_available_models(self) -> Dict[int, List[str]]:
        """Find available trained models organized by latent dimension."""
        available_models = {}
        
        for latent_dim in self.config.model.LATENT_DIMS:
            models = []
            latent_dir = os.path.join(self.config.paths.MODELS_DIR, f'latent_{latent_dim}')
            
            if os.path.exists(latent_dir):
                for strategy in self.config.training.ANNEALING_STRATEGIES:
                    strategy_dir = os.path.join(latent_dir, strategy)
                    if os.path.exists(strategy_dir):
                        for beta in self.config.training.BETA_VALUES:
                            model_file = os.path.join(strategy_dir, f'beta_{beta}', 'vae_model_final.pth')
                            if os.path.exists(model_file):
                                models.append(f"{strategy}-{beta}")
            
            available_models[latent_dim] = models
        
        return available_models

def main():
    """Enhanced main entry point with latent dimension support."""
    parser = argparse.ArgumentParser(description='Enhanced VAE Pipeline with Multiple Latent Dimensions')
    
    # Core options
    parser.add_argument('--stages', nargs='+', 
                       choices=['preprocess', 'train', 'visualize', 'sample', 'test'],
                       help='Stages to run (default: all)')
    parser.add_argument('--strategy', choices=['linear', 'exponential', 'constant', 'cyclical'])
    parser.add_argument('--beta', type=float)
    
    # Latent dimension options
    parser.add_argument('--latent-dims', nargs='+', type=int,
                       help='Specific latent dimensions to use (e.g., --latent-dims 2 4 8)')
    parser.add_argument('--latent-range', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       help='Range of latent dimensions (powers of 2, e.g., --latent-range 2 16)')
    parser.add_argument('--single-latent-dim', type=int,
                       help='Train only for a single latent dimension')
    
    # Sampling options
    parser.add_argument('--methods', nargs='+', 
                       choices=['cluster_based', 'equiprobable', 'latin_hypercube', 
                               'adaptive_latin_hypercube', 'all'],
                       default=['cluster_based', 'equiprobable'],
                       help='Sampling methods to use')
    
    # Sampling parameters
    parser.add_argument('--info-weight', type=float, default=1.0)
    parser.add_argument('--redundancy-weight', type=float, default=1.0)
    parser.add_argument('--coverage-radius', type=float, default=0.2)
    
    # Latin Hypercube specific parameters
    parser.add_argument('--lhs-criterion', choices=['maximin', 'correlation', 'centermaximin'], 
                       default='maximin', help='LHS optimization criterion')
    parser.add_argument('--lhs-iterations', type=int, default=10, 
                       help='Number of LHS optimization iterations')
    parser.add_argument('--density-weight', type=float, default=0.3,
                       help='Density weight for adaptive LHS')
    
    # Training options
    parser.add_argument('--architecture', choices=['adaptive', 'deep', 'wide'],
                       default='adaptive', help='Model architecture type')
    parser.add_argument('--early-stopping', action='store_true', default=True,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, help='Early stopping patience (auto-determined by latent dim if not set)')
    
    # General options
    parser.add_argument('--fast', action='store_true', help='Fast mode for testing')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu-memory-limit', type=float, help='GPU memory limit in GB')
    
    # Analysis options
    parser.add_argument('--create-comparison', action='store_true', default=True,
                       help='Create comprehensive comparison across latent dimensions')
    parser.add_argument('--statistical-tests', action='store_true', default=True,
                       help='Run statistical significance tests')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.fast:
        os.environ['VAE_FAST_MODE'] = '1'
        logger.info("⚡ Fast mode enabled")
    
    # Handle latent dimension specifications
    latent_dims = None
    
    if args.single_latent_dim:
        latent_dims = [args.single_latent_dim]
        logger.info(f"📏 Single latent dimension mode: {args.single_latent_dim}")
    elif args.latent_dims:
        latent_dims = sorted(args.latent_dims)
        logger.info(f"📏 Custom latent dimensions: {latent_dims}")
    elif args.latent_range:
        min_dim, max_dim = args.latent_range
        latent_dims = [2**i for i in range(int(np.log2(min_dim)), int(np.log2(max_dim)) + 1)]
        logger.info(f"📏 Latent dimension range: {latent_dims}")
    else:
        logger.info("📏 Using default latent dimensions from config")
    
    if 'all' in args.methods:
        args.methods = ['cluster_based', 'equiprobable', 'latin_hypercube', 'adaptive_latin_hypercube']
    
    # Log configuration
    logger.info(f"📋 Selected sampling methods: {', '.join(args.methods)}")
    if latent_dims:
        logger.info(f"📏 Latent dimensions to explore: {latent_dims}")
    
    # Prepare parameters
    sampling_params = {
        'info_weight': args.info_weight,
        'redundancy_weight': args.redundancy_weight,
        'coverage_radius': args.coverage_radius,
        'lhs_criterion': args.lhs_criterion,
        'lhs_iterations': args.lhs_iterations,
        'density_weight': args.density_weight,
        'random_seed': args.random_seed
    }
    
    try:
        pipeline = EnhancedVAEPipeline()
        
        # Apply GPU memory limit if specified
        if args.gpu_memory_limit:
            try:
                import torch
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_limit / torch.cuda.get_device_properties(0).total_memory * 1e9)
                logger.info(f"🔧 GPU memory limit set to {args.gpu_memory_limit}GB")
            except Exception as e:
                logger.warning(f"Could not set GPU memory limit: {e}")
        
        results = pipeline.run(
            stages=args.stages,
            strategy=args.strategy,
            beta=args.beta,
            latent_dims=latent_dims,
            sampling_methods=args.methods,
            sampling_params=sampling_params,
            architecture=args.architecture,
            patience=args.patience,
            create_comparison=args.create_comparison,
            statistical_tests=args.statistical_tests
        )
        
        logger.info("🎯 Enhanced pipeline completed successfully!")
        logger.info(f"📁 Results available in: {pipeline.config.paths.OUTPUT_DIR}")
        
        # Enhanced summary
        if 'training' in results['results']:
            training_results = results['results']['training']
            logger.info("📊 Training Summary:")
            for latent_dim in results['latent_dims_explored']:
                if latent_dim in training_results:
                    models_count = sum(1 for strategy_results in training_results[latent_dim].values() 
                                     for results in strategy_results.values() 
                                     if 'error' not in results)
                    logger.info(f"  Latent {latent_dim}: {models_count} successful models")
        
        if 'sampling' in results['results']:
            logger.info("📊 Sampling Summary:")
            logger.info(f"  Methods used: {', '.join(args.methods)}")
            logger.info(f"  Equiprobable: Uses Gaussian quantiles")
            logger.info(f"  Latin Hypercube: {args.lhs_criterion} criterion")
        
        if args.create_comparison:
            logger.info("📊 Comprehensive comparison created in latent_dimension_comparison/")
        
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