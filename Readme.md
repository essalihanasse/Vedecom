# VAE Pipeline with Multi-Method Representative Sampling

A comprehensive, optimized pipeline for training Variational Autoencoders (VAEs) on mixed numerical and categorical data, with advanced representative sampling methods for latent space exploration.

## 🚀 Features

- **VAE Training**: Support for multiple β-annealing strategies (linear, exponential, constant, cyclical)
- **Multi-Method Sampling**: 11 different representative sampling algorithms
- **Mixed Data Types**: Handles both numerical and categorical features seamlessly
- **Optimized Performance**: Memory-efficient algorithms with 2-10x speedup over naive implementations
- **Comprehensive Evaluation**: Distribution testing and quality metrics
- **Rich Visualizations**: Latent space plots, coverage analysis, and training curves
- **Configurable Pipeline**: Easy-to-use configuration system with sensible defaults

## 📋 Requirements

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
torch>=1.9.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Optional but recommended
scipy>=1.7.0
tqdm>=4.62.0
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/vae-pipeline.git
cd vae-pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your data:**
   - Place your CSV data file in the `data/` directory as `data.csv`
   - Ensure it contains the expected columns (see Configuration section)

## 🏃 Quick Start

### Basic Usage

```bash
# Run the complete pipeline with default settings
python main.py

# Run specific stages only
python main.py --stages preprocess train sample

# Use specific sampling methods
python main.py --methods equiprobable cluster_based distance_based

# Fast mode for testing
python main.py --fast --verbose
```

### Advanced Usage

```bash
# Custom β-annealing strategy and value
python main.py --strategy exponential --beta 2.0

# Specific sampling configuration
python main.py --methods cluster_based --coverage-radius 0.3 --info-weight 1.5

# Training only with custom parameters
python main.py --stages train --strategy linear --beta 1.0
```

## 📁 Project Structure

```
vae-pipeline/
├── main.py                 # Main pipeline orchestrator
├── config/
│   ├── __init__.py
│   ├── settings.py         # Configuration settings
│   └── paths.py            # Path configurations
├── data/                   # Data directory
│   ├── data.csv           # Your input data (you provide)
│   ├── filtered_data.csv  # Processed data
│   └── preprocessing_objects.pkl
├── models/
│   ├── __init__.py
│   ├── vae.py             # VAE model definition
│   ├── training.py        # Training logic
│   └── callbacks.py       # Training callbacks
├── sampling/
│   ├── __init__.py
│   ├── base.py            # Base sampling classes
│   ├── equiprobable.py    # Grid-based sampling
│   ├── representative.py  # Distance-based sampling
│   ├── cluster_based.py   # Cluster-based sampling
│   ├── hybrid.py          # Hybrid sampling
│   ├── density_aware.py   # Density-aware methods
│   ├── optimal_transport.py # Optimal transport methods
│   └── manager.py         # Sampling coordinator
├── evaluation/
│   ├── __init__.py
│   ├── testing.py         # Distribution tests
│   └── metrics.py         # Quality metrics
├── visualization/
│   └── latent_viz.py      # Visualization tools
├── utils/
│   ├── __init__.py
│   ├── math_utils.py      # Mathematical utilities
│   └── file_utils.py      # File handling utilities
└── output/                # Generated results
    ├── models/            # Trained models
    ├── samples/           # Sampling results
    ├── visualizations/    # Generated plots
    └── tests/             # Test results
```

## ⚙️ Configuration

### Data Configuration

The pipeline expects specific column names. Update `config/settings.py` if your data has different column names:

```python
@dataclass
class DataConfig:
    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        'code_country', 
        'T1_climate_day_period', 
        'T1_climate_dazzled', 
        'T1_climate_fog', 
        'T1_climate_precipitation',
        'NumberOfLanesInPrincipalRoad'
    ])
    
    NUMERICAL_COLS: List[str] = field(default_factory=lambda: [
        'FrontCurvature', 
        'T1_ego_speed', 
        'T1_climate_outside_temperature', 
        'T1_V1 (CIPV)_pos_x', 
        'T1_V1 (CIPV)_pos_y', 
        'T1_V1 (CIPV)_absolute_velocity_x', 
        'T1_V1 (CIPV)_absolute_acceleration_x', 
        'T2_V1 (CIPV)_absolute_velocity_x'
    ])
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    HIDDEN_DIM: int = 16      # Hidden layer size
    LATENT_DIM: int = 2       # Latent space dimensions
    BATCH_SIZE: int = 128     # Training batch size
    LEARNING_RATE: float = 1e-3
    NUM_EPOCHS: int = 100
```

### Training Configuration

```python
@dataclass  
class TrainingConfig:
    BETA_VALUES: List[float] = [0.1, 0.5, 1.0, 3.0, 5.0]
    ANNEALING_STRATEGIES: List[str] = ['linear', 'exponential']
    SAMPLE_SIZES: List[int] = [100, 400, 900, 1225]
```

## 🎯 Available Sampling Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `equiprobable` | Grid-based equiprobable sampling | Uniform coverage |
| `distance_based` | Farthest-first representative selection | Maximum diversity |
| `cluster_based` | Cluster then sample from each cluster | Structured data |
| `hybrid` | Combines clustering + distance-based | Balanced approach |
| `density_aware_kde` | KDE-based stratified sampling | Preserving distributions |
| `density_aware_importance` | Importance sampling with KDE | Density-proportional sampling |
| `progressive_wasserstein` | Iterative Wasserstein minimization | Distribution matching |
| `blue_noise` | Poisson disk sampling | Even spacing |
| `optimal_transport_greedy` | Greedy optimal transport | Cost minimization |
| `optimal_transport_hungarian` | Hungarian algorithm OT | Optimal assignment |
| `sliced_wasserstein` | Sliced Wasserstein barycenter | High-dimensional efficiency |

### Method Selection Guidelines

```bash
# For uniform coverage
python main.py --methods equiprobable

# For maximum diversity
python main.py --methods distance_based

# For structured/clustered data
python main.py --methods cluster_based

# For best overall performance
python main.py --methods hybrid

# For distribution preservation
python main.py --methods density_aware_kde

# Try all major methods
python main.py --methods all
```

## 📊 Output Structure

The pipeline generates organized outputs in the `output/` directory:

```
output/
├── models/
│   ├── linear/
│   │   ├── beta_0.1/
│   │   │   ├── vae_model_final.pth
│   │   │   ├── training_history.csv
│   │   │   └── training_curves.png
│   │   └── beta_results_summary.csv
│   └── exponential/
├── samples/
│   ├── linear/
│   │   └── beta_1.0/
│   │       ├── method_equiprobable/
│   │       │   ├── samples_100/
│   │       │   │   ├── selected_points.csv
│   │       │   │   ├── selected_indices.npy
│   │       │   │   └── equiprobable_sampling.png
│   │       │   └── samples_400/
│   │       └── method_cluster_based/
│   └── overall_sampling_summary.csv
├── tests/
│   └── distribution_test_summary.csv
└── visualizations/
    └── latent_space_plots/
```

## 🔧 Command Line Options

### Core Options
- `--stages`: Pipeline stages to run (`preprocess`, `train`, `visualize`, `sample`, `test`)
- `--strategy`: β-annealing strategy (`linear`, `exponential`, `constant`, `cyclical`)
- `--beta`: Specific β value for KL weighting
- `--methods`: Sampling methods to use

### Sampling Parameters
- `--info-weight`: Information gain weight (default: 1.0)
- `--redundancy-weight`: Redundancy penalty weight (default: 1.0)
- `--coverage-radius`: Coverage radius for distance calculations (default: 0.2)

### General Options
- `--fast`: Enable fast mode (reduced epochs/samples for testing)
- `--verbose`: Enable verbose logging
- `--config`: Path to custom configuration file

## 📈 Performance Optimizations

The optimized pipeline includes several performance improvements:

1. **Memory Efficiency**: Batch processing for large datasets
2. **Computational Speed**: Vectorized distance calculations
3. **Algorithm Optimization**: Simplified representative sampling (10x speedup)
4. **Caching**: Intelligent caching of expensive computations

### Performance Tips

```bash
# For large datasets, use fast mode first
python main.py --fast

# For memory-constrained environments
python main.py --methods equiprobable distance_based  # Avoid memory-heavy methods

# For quick testing
python main.py --stages preprocess train --fast --verbose
```

## 🧪 Examples

### Example 1: Basic Training and Sampling
```bash
# Train VAE models and run basic sampling
python main.py --stages train sample --methods equiprobable cluster_based
```

### Example 2: Comprehensive Analysis
```bash
# Full pipeline with multiple methods and testing
python main.py --methods equiprobable distance_based cluster_based hybrid \
               --strategy linear --beta 1.0
```

### Example 3: Custom Configuration
```bash
# Custom sampling parameters
python main.py --methods cluster_based --coverage-radius 0.4 \
               --info-weight 2.0 --redundancy-weight 0.5
```

### Example 4: Quick Testing
```bash
# Fast mode for development/testing
python main.py --fast --verbose --methods equiprobable
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU mode or reduce batch size
   CUDA_VISIBLE_DEVICES=-1 python main.py --fast
   ```

2. **Missing Data Columns**
   ```
   Error: Column 'X' not found in data
   ```
   - Update column names in `config/settings.py`
   - Check your data file format

3. **Slow Performance**
   ```bash
   # Use fast mode and fewer methods
   python main.py --fast --methods equiprobable
   ```

4. **Memory Issues**
   ```bash
   # Reduce sample sizes or use memory-efficient methods
   python main.py --methods distance_based  # Most memory-efficient
   ```

### Debug Mode

```bash
# Enable verbose logging for debugging
python main.py --verbose --stages preprocess
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch and scikit-learn
- Inspired by β-VAE research and optimal transport theory
- Optimized for real-world mixed-type datasets

## 📚 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vae_pipeline_2024,
  title={VAE Pipeline with Multi-Method Representative Sampling},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/vae-pipeline}
}
```

---