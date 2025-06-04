"""
Main orchestrator for the clean VAE pipeline.
"""

import os
import sys
import time
import argparse
import logging
from datetime import timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VAEPipeline:
    """
    Main VAE pipeline orchestrator.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Import configuration
        try:
            from config.settings import config
            self.config = config
        except ImportError:
            logger.error("Could not import configuration. Make sure config module is available.")
            raise
        
        self.results = {}
        self.execution_times = {}
    
    def run(
        self,
        stages: Optional[List[str]] = None,
        strategy: Optional[str] = None,
        beta: Optional[float] = None,
        sampling_methods: Optional[List[str]] = None,
        sampling_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete VAE pipeline.
        
        Args:
            stages: List of stages to run (None for all)
            strategy: Specific annealing strategy
            beta: Specific beta value
            sampling_methods: Sampling methods to use
            sampling_params: Parameters for sampling methods
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = time.time()
        logger.info("🚀 Starting clean VAE pipeline...")
        
        # Define all available stages
        all_stages = ['preprocess', 'train', 'visualize', 'sample', 'test']
        
        if stages is None:
            stages = all_stages
        
        # Set defaults
        if sampling_methods is None:
            sampling_methods = self.config.sampling.DEFAULT_METHODS
        
        if sampling_params is None:
            sampling_params = self.config.get_sampling_params()
        
        # Override configuration if specific values provided
        if strategy is not None:
            self.config.training.ANNEALING_STRATEGIES = [strategy]
        
        if beta is not None:
            self.config.training.BETA_VALUES = [beta]
        
        # Run stages
        try:
            for stage in stages:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Running stage: {stage.upper()}")
                logger.info(f"{'='*60}")
                
                stage_start = time.time()
                
                if stage == 'preprocess':
                    self._run_preprocessing()
                elif stage == 'train':
                    self._run_training()
                elif stage == 'visualize':
                    self._run_visualization()
                elif stage == 'sample':
                    self._run_sampling(sampling_methods, sampling_params)
                elif stage == 'test':
                    self._run_testing()
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
        logger.info(f"{'='*80}")
        logger.info(f"⏱️ Total execution time: {timedelta(seconds=int(total_time))}")
        
        # Return summary
        return {
            'stages_completed': stages,
            'execution_times': self.execution_times,
            'results': self.results,
            'config_used': {
                'sampling_methods': sampling_methods,
                'sampling_params': sampling_params,
                'beta_values': self.config.training.BETA_VALUES,
                'strategies': self.config.training.ANNEALING_STRATEGIES
            }
        }
    
    def _run_preprocessing(self) -> None:
        """Run data preprocessing stage."""
        from data.preprocessing import preprocess_data
        
        try:
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
            logger.info(f"🔢 Features {metadata['n_features_original']} → {metadata['n_features_final']}")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def _run_training(self) -> None:
        """Run model training stage."""
        from models.training import ModelTrainer
        
        try:
            trainer = ModelTrainer(self.config)
            training_results = trainer.train_all_models()
            
            self.results['training'] = training_results
            
            n_models = len(training_results)
            logger.info(f"🧠 Trained {n_models} models successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _run_visualization(self) -> None:
        """Run latent space visualization stage."""
        from visualization.latent_viz import LatentVisualizer
        
        try:
            visualizer = LatentVisualizer(self.config)
            viz_results = visualizer.create_all_visualizations()
            
            self.results['visualization'] = viz_results
            
            logger.info(f"🎨 Created visualizations for all trained models")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            # Don't raise - visualization is not critical
            logger.warning("Continuing without visualizations...")
    
    def _run_sampling(
        self, 
        sampling_methods: List[str], 
        sampling_params: Dict[str, Any]
    ) -> None:
        """Run sampling stage."""
        from sampling.manager import SamplingManager
        
        try:
            manager = SamplingManager(self.config)
            
            # Register desired sampling methods
            for method in sampling_methods:
                manager.register_method(method, **sampling_params)
            
            # Run sampling for all trained models
            sampling_results = manager.run_all_sampling()
            
            self.results['sampling'] = sampling_results
            
            n_methods = len(sampling_methods)
            n_configs = len(self.config.training.ANNEALING_STRATEGIES) * len(self.config.training.BETA_VALUES)
            n_sizes = len(self.config.training.SAMPLE_SIZES)
            total_runs = n_methods * n_configs * n_sizes
            
            logger.info(f"🎲 Completed {total_runs} sampling runs")
            logger.info(f"📊 Methods: {sampling_methods}")
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise
    
    def _run_testing(self) -> None:
        """Run distribution testing stage."""
        from evaluation.testing import DistributionTester
        
        try:
            tester = DistributionTester(self.config)
            test_results = tester.run_all_tests()
            
            self.results['testing'] = test_results
            
            logger.info(f"🧪 Completed distribution testing")
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            # Don't raise - testing is not critical for the main pipeline
            logger.warning("Continuing without distribution tests...")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Clean VAE Pipeline with Multi-Method Sampling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run full pipeline with defaults
  python main.py --stages preprocess train         # Run only preprocessing and training
  python main.py --methods cluster_based           # Use only cluster-based sampling
  python main.py --methods all                     # Use all sampling methods
  python main.py --strategy linear --beta 1.0      # Specific strategy and beta
  python main.py --fast                           # Fast mode for testing
        """
    )
    
    # Stage selection
    parser.add_argument(
        '--stages', nargs='+',
        choices=['preprocess', 'train', 'visualize', 'sample', 'test'],
        help='Stages to run (default: all)'
    )
    
    # Model configuration
    parser.add_argument(
        '--strategy', type=str,
        choices=['linear', 'exponential', 'constant', 'cyclical'],
        help='Specific annealing strategy'
    )
    
    parser.add_argument(
        '--beta', type=float,
        help='Specific beta value'
    )
    
    # Sampling configuration
    
    # REPLACE WITH THIS NEW CODE:
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=[
            # Basic methods
            'equiprobable', 'distance_based', 'cluster_based', 'hybrid',
            # Density-aware methods  
            'density_aware_kde', 'density_aware_importance', 
            'progressive_wasserstein', 'blue_noise',
            # Optimal transport methods
            'optimal_transport_greedy', 'optimal_transport_hungarian',
            'sliced_wasserstein',
            # Specialized methods
            'cluster_one_per',
            # Run all methods
            'all'
        ],
        default=['equiprobable'],
        help='Sampling methods to use'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--info-weight', type=float, default=1.0,
        help='Information gain weight'
    )
    
    parser.add_argument(
        '--redundancy-weight', type=float, default=1.0,
        help='Redundancy penalty weight'
    )
    
    parser.add_argument(
        '--coverage-radius', type=float, default=0.2,
        help='Coverage radius for distance-based methods'
    )
    
    parser.add_argument(
        '--cluster-method', type=str, default='kmeans',
        choices=['kmeans', 'dbscan'],
        help='Clustering method'
    )
    
    parser.add_argument(
        '--cluster-sizing', type=str, default='adaptive',
        choices=['adaptive', 'sqrt_rule', 'proportional', 'fixed'],
        help='Cluster sizing method'
    )
    
    # General options
    parser.add_argument(
        '--fast', action='store_true',
        help='Fast mode (reduced epochs and samples for testing)'
    )
    
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--config', type=str,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Fast mode adjustments
    if args.fast:
        os.environ['VAE_FAST_MODE'] = '1'
        logger.info("⚡ Fast mode enabled")
    
    # Handle 'all' methods selection
    if 'all' in args.methods:
        args.methods = ['equiprobable', 'distance_based', 'cluster_based', 'hybrid']
    
    # Prepare sampling parameters
    sampling_params = {
        'info_weight': args.info_weight,
        'redundancy_weight': args.redundancy_weight,
        'coverage_radius': args.coverage_radius,
        'cluster_method': args.cluster_method,
        'cluster_sizing_method': args.cluster_sizing,
    }
    
    # Log configuration
    logger.info("🎯 VAE Pipeline Configuration:")
    logger.info(f"  Stages: {args.stages or 'all'}")
    logger.info(f"  Sampling methods: {args.methods}")
    if args.strategy:
        logger.info(f"  Strategy: {args.strategy}")
    if args.beta:
        logger.info(f"  Beta: {args.beta}")
    logger.info(f"  Fast mode: {args.fast}")
    
    try:
        # Initialize and run pipeline
        pipeline = VAEPipeline(args.config)
        results = pipeline.run(
            stages=args.stages,
            strategy=args.strategy,
            beta=args.beta,
            sampling_methods=args.methods,
            sampling_params=sampling_params
        )
        
        # Print summary
        logger.info("\n📊 Execution Summary:")
        for stage, exec_time in results['execution_times'].items():
            if stage != 'total':
                logger.info(f"  {stage}: {timedelta(seconds=int(exec_time))}")
        
        logger.info(f"\n🎯 Pipeline completed successfully!")
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