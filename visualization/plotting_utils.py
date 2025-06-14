"""
Enhanced plotting utilities for the VAE pipeline visualization system with automotive data support.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy import stats
import warnings

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

class PlotStyle:
    """Consistent plotting style for the VAE pipeline with automotive data considerations."""
    
    # Color schemes optimized for automotive data visualization
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'success': '#2E8B57',  # Sea green for good coverage
        'warning': '#FF8C00',  # Dark orange for medium coverage
        'danger': '#DC143C',   # Crimson for poor coverage
        'background': '#F5F5F5',
        'text': '#2C2C2C',
        'automotive': {
            'speed': '#FF6B35',      # Orange-red for speed/velocity
            'position': '#4ECDC4',   # Teal for position data
            'acceleration': '#45B7D1', # Blue for acceleration
            'climate': '#96CEB4',    # Light green for climate
            'road': '#FECA57',       # Yellow for road features
            'country': '#FF9FF3'     # Pink for country codes
        }
    }
    
    # Common figure sizes
    FIGSIZE = {
        'small': (8, 6),
        'medium': (12, 8),
        'large': (16, 10),
        'wide': (20, 6),
        'square': (10, 10),
        'comparison': (16, 12),
        'grid': (20, 15)
    }
    
    # Style parameters
    SCATTER_PARAMS = {
        'alpha': 0.7,
        's': 10,
        'edgecolors': 'none'
    }
    
    LINE_PARAMS = {
        'linewidth': 2,
        'alpha': 0.8
    }
    
    HIST_PARAMS = {
        'alpha': 0.7,
        'edgecolor': 'black',
        'linewidth': 0.5
    }
    
    # Automotive-specific styling
    AUTOMOTIVE_FEATURES = {
        'speed': ['speed', 'velocity'],
        'position': ['pos_x', 'pos_y', 'position'],
        'acceleration': ['acceleration', 'accel'],
        'climate': ['climate', 'temperature', 'fog', 'precipitation'],
        'road': ['road', 'lane', 'curvature'],
        'country': ['country', 'code_country']
    }

def setup_plot_style():
    """Setup consistent plot styling optimized for automotive data."""
    plt.rcParams.update({
        'figure.figsize': PlotStyle.FIGSIZE['medium'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'grid.alpha': 0.3,
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.axisbelow': True
    })

def get_feature_color(feature_name: str) -> str:
    """Get appropriate color for automotive feature based on its name."""
    feature_lower = feature_name.lower()
    
    for category, keywords in PlotStyle.AUTOMOTIVE_FEATURES.items():
        if any(keyword in feature_lower for keyword in keywords):
            return PlotStyle.COLORS['automotive'][category]
    
    return PlotStyle.COLORS['primary']

def create_figure(figsize: Union[str, Tuple[int, int]] = 'medium', **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create figure with consistent styling.
    
    Args:
        figsize: Figure size (string key or tuple)
        **kwargs: Additional arguments for plt.subplots
        
    Returns:
        Figure and axes objects
    """
    if isinstance(figsize, str):
        figsize = PlotStyle.FIGSIZE.get(figsize, PlotStyle.FIGSIZE['medium'])
    
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax

def create_subplots(nrows: int, ncols: int, figsize: Union[str, Tuple[int, int]] = 'large', **kwargs) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create subplots with consistent styling.
    
    Args:
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size
        **kwargs: Additional arguments for plt.subplots
        
    Returns:
        Figure and axes array
    """
    if isinstance(figsize, str):
        figsize = PlotStyle.FIGSIZE.get(figsize, PlotStyle.FIGSIZE['large'])
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes

def scatter_plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    c: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    colorbar: bool = True,
    feature_name: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create standardized 2D scatter plot with automotive data considerations.
    
    Args:
        x: X coordinates
        y: Y coordinates
        c: Colors for points
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar: Whether to add colorbar
        feature_name: Name of feature for color selection
        **kwargs: Additional scatter plot arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    # Merge with default scatter parameters
    scatter_params = {**PlotStyle.SCATTER_PARAMS, **kwargs}
    
    if c is not None:
        # Use automotive-appropriate colormap if possible
        cmap = 'viridis'  # Default
        if feature_name:
            feature_lower = feature_name.lower()
            if any(keyword in feature_lower for keyword in ['speed', 'velocity']):
                cmap = 'Reds'
            elif any(keyword in feature_lower for keyword in ['temperature']):
                cmap = 'coolwarm'
            elif any(keyword in feature_lower for keyword in ['position']):
                cmap = 'viridis'
        
        scatter_params.setdefault('cmap', cmap)
        scatter = ax.scatter(x, y, c=c, **scatter_params)
        if colorbar:
            plt.colorbar(scatter, ax=ax, label=feature_name or 'Value')
    else:
        color = get_feature_color(feature_name) if feature_name else PlotStyle.COLORS['primary']
        ax.scatter(x, y, color=color, **scatter_params)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return ax

def histogram_comparison_plot(
    original_data: np.ndarray,
    sampled_data: np.ndarray,
    feature_name: str,
    ax: Optional[plt.Axes] = None,
    bins: Optional[Union[int, np.ndarray]] = None,
    density: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Create histogram comparison plot for original vs sampled data.
    
    Args:
        original_data: Original dataset values
        sampled_data: Sampled dataset values
        feature_name: Name of the feature
        ax: Existing axes (optional)
        bins: Number of bins or bin edges
        density: Whether to normalize histograms
        **kwargs: Additional histogram arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    # Clean data
    original_clean = original_data[~np.isnan(original_data)]
    sampled_clean = sampled_data[~np.isnan(sampled_data)]
    
    # Determine bins if not provided
    if bins is None:
        data_min = min(original_clean.min(), sampled_clean.min())
        data_max = max(original_clean.max(), sampled_clean.max())
        bins = np.linspace(data_min, data_max, 50)
    
    hist_params = {**PlotStyle.HIST_PARAMS, **kwargs}
    
    # Plot histograms
    ax.hist(original_clean, bins=bins, alpha=0.7, label=f'Original (n={len(original_clean):,})', 
           density=density, color='blue', **hist_params)
    ax.hist(sampled_clean, bins=bins, alpha=0.7, label=f'Sampled (n={len(sampled_clean):,})', 
           density=density, color='red', **hist_params)
    
    ax.set_title(f'Distribution Comparison: {feature_name}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density' if density else 'Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax

def categorical_comparison_plot(
    original_data: pd.Series,
    sampled_data: pd.Series,
    feature_name: str,
    ax: Optional[plt.Axes] = None,
    max_categories: int = 15,
    **kwargs
) -> plt.Axes:
    """
    Create categorical comparison plot for original vs sampled data.
    
    Args:
        original_data: Original dataset categorical values
        sampled_data: Sampled dataset categorical values
        feature_name: Name of the feature
        ax: Existing axes (optional)
        max_categories: Maximum number of categories to display
        **kwargs: Additional bar plot arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    # Get value counts
    original_counts = original_data.value_counts()
    sampled_counts = sampled_data.value_counts()
    
    # Get all unique values, limit if too many
    all_values = sorted(set(original_counts.index) | set(sampled_counts.index))
    
    if len(all_values) > max_categories:
        # Keep top categories by frequency in original data
        top_categories = original_counts.head(max_categories).index.tolist()
        all_values = top_categories
    
    # Create aligned counts
    original_aligned = [original_counts.get(val, 0) for val in all_values]
    sampled_aligned = [sampled_counts.get(val, 0) for val in all_values]
    
    # Create bar plot
    x = np.arange(len(all_values))
    width = 0.35
    
    ax.bar(x - width/2, original_aligned, width, label=f'Original (n={original_data.count():,})', 
           alpha=0.8, color='blue', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, sampled_aligned, width, label=f'Sampled (n={sampled_data.count():,})', 
           alpha=0.8, color='red', edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'Category Distribution: {feature_name}')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(all_values, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax

def create_quality_heatmap(
    quality_matrix: np.ndarray,
    feature_names: List[str],
    method_names: List[str],
    title: str = "Quality Matrix",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Create quality heatmap with appropriate styling.
    
    Args:
        quality_matrix: 2D array of quality scores
        feature_names: Names of features (rows)
        method_names: Names of methods (columns)
        title: Plot title
        ax: Existing axes (optional)
        **kwargs: Additional imshow arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure(figsize='grid')
    
    # Create custom colormap (red-yellow-green)
    colors = ['#DC143C', '#FF8C00', '#FFD700', '#ADFF2F', '#32CD32']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('quality', colors, N=n_bins)
    
    im = ax.imshow(quality_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1, **kwargs)
    
    # Set labels
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    
    # Add text annotations
    for i in range(len(feature_names)):
        for j in range(len(method_names)):
            score = quality_matrix[i, j]
            text_color = 'white' if score < 0.5 else 'black'
            ax.text(j, i, f'{score:.2f}', ha='center', va='center', 
                   color=text_color, fontsize=8, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return ax

def add_statistical_comparison_text(
    ax: plt.Axes,
    original_data: np.ndarray,
    sampled_data: np.ndarray,
    feature_name: str,
    position: str = 'upper right'
) -> None:
    """
    Add statistical comparison text to a plot.
    
    Args:
        ax: Axes object
        original_data: Original dataset values
        sampled_data: Sampled dataset values
        feature_name: Name of the feature
        position: Position for text box
    """
    # Clean data
    orig_clean = original_data[~np.isnan(original_data)]
    samp_clean = sampled_data[~np.isnan(sampled_data)]
    
    if len(orig_clean) == 0 or len(samp_clean) == 0:
        return
    
    # Calculate statistics
    orig_mean = np.mean(orig_clean)
    samp_mean = np.mean(samp_clean)
    orig_std = np.std(orig_clean)
    samp_std = np.std(samp_clean)
    
    # Statistical tests
    try:
        ks_stat, ks_pvalue = stats.ks_2samp(orig_clean, samp_clean)
        ttest_stat, ttest_pvalue = stats.ttest_ind(orig_clean, samp_clean)
    except:
        ks_stat = ks_pvalue = ttest_stat = ttest_pvalue = np.nan
    
    # Coverage ratio
    coverage = len(samp_clean) / len(orig_clean)
    
    # Create text
    stats_text = f"""Statistics for {feature_name}:
Original: μ={orig_mean:.3f}, σ={orig_std:.3f}
Sampled:  μ={samp_mean:.3f}, σ={samp_std:.3f}

KS Test: p={ks_pvalue:.4f}
T-Test:  p={ttest_pvalue:.4f}
Coverage: {coverage:.2%}"""
    
    add_text_box(ax, stats_text, position, fontsize=8)

def create_automotive_feature_summary(
    feature_stats: Dict[str, Dict[str, float]],
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Create summary plot for automotive features categorized by type.
    
    Args:
        feature_stats: Dictionary with feature statistics
        ax: Existing axes (optional)
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure(figsize='wide')
    
    # Categorize features
    categories = {
        'Speed/Velocity': [],
        'Position': [],
        'Acceleration': [],
        'Climate': [],
        'Road/Infrastructure': [],
        'Other': []
    }
    
    for feature in feature_stats.keys():
        feature_lower = feature.lower()
        categorized = False
        
        if any(keyword in feature_lower for keyword in ['speed', 'velocity']):
            categories['Speed/Velocity'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in ['pos_x', 'pos_y', 'position']):
            categories['Position'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in ['acceleration', 'accel']):
            categories['Acceleration'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in ['climate', 'temperature', 'fog', 'precipitation']):
            categories['Climate'].append(feature)
            categorized = True
        elif any(keyword in feature_lower for keyword in ['road', 'lane', 'curvature']):
            categories['Road/Infrastructure'].append(feature)
            categorized = True
        
        if not categorized:
            categories['Other'].append(feature)
    
    # Create summary bars
    category_names = []
    category_scores = []
    category_colors = []
    
    for cat_name, features in categories.items():
        if features:  # Only include categories with features
            # Calculate average quality score for this category
            avg_score = np.mean([feature_stats[f]['quality_score'] for f in features])
            category_names.append(f"{cat_name}\n({len(features)} features)")
            category_scores.append(avg_score)
            
            # Color based on score
            if avg_score > 0.8:
                category_colors.append(PlotStyle.COLORS['success'])
            elif avg_score > 0.5:
                category_colors.append(PlotStyle.COLORS['warning'])
            else:
                category_colors.append(PlotStyle.COLORS['danger'])
    
    # Create bar plot
    bars = ax.bar(category_names, category_scores, color=category_colors, alpha=0.7, 
                  edgecolor='black', linewidth=1)
    
    # Add score labels on bars
    for bar, score in zip(bars, category_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Automotive Feature Categories - Quality Summary', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Quality Score')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add quality thresholds
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (>0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair (>0.5)')
    ax.legend()
    
    return ax

def line_plot(
    x: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    label: str = "",
    **kwargs
) -> plt.Axes:
    """
    Create standardized line plot.
    
    Args:
        x: X values
        y: Y values
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        label: Line label
        **kwargs: Additional plot arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    # Merge with default line parameters
    line_params = {**PlotStyle.LINE_PARAMS, **kwargs}
    
    ax.plot(x, y, label=label, **line_params)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if label:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    return ax

def histogram_plot(
    data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 50,
    feature_name: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Create standardized histogram with automotive data considerations.
    
    Args:
        data: Data to plot
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        feature_name: Name of feature for color selection
        **kwargs: Additional hist arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    color = get_feature_color(feature_name) if feature_name else PlotStyle.COLORS['primary']
    
    hist_params = {
        'alpha': 0.7,
        'color': color,
        'edgecolor': 'black',
        'linewidth': 0.5,
        **kwargs
    }
    
    ax.hist(data, bins=bins, **hist_params)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    return ax

def heatmap_plot(
    data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
    xlabel: str = "X",
    ylabel: str = "Y",
    cmap: str = 'viridis',
    **kwargs
) -> plt.Axes:
    """
    Create standardized heatmap.
    
    Args:
        data: 2D data array
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        cmap: Colormap
        **kwargs: Additional imshow arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', **kwargs)
    plt.colorbar(im, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return ax

def add_text_box(ax: plt.Axes, text: str, position: str = 'upper right', **kwargs) -> None:
    """
    Add text box to plot.
    
    Args:
        ax: Axes object
        text: Text to display
        position: Position of text box
        **kwargs: Additional text arguments
    """
    # Position mapping
    positions = {
        'upper right': (0.98, 0.98),
        'upper left': (0.02, 0.98),
        'lower right': (0.98, 0.02),
        'lower left': (0.02, 0.02),
        'center': (0.5, 0.5)
    }
    
    x, y = positions.get(position, (0.98, 0.98))
    ha = 'right' if 'right' in position else 'center' if position == 'center' else 'left'
    va = 'top' if 'upper' in position else 'center' if position == 'center' else 'bottom'
    
    text_params = {
        'transform': ax.transAxes,
        'bbox': dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
        'verticalalignment': va,
        'horizontalalignment': ha,
        'fontsize': 10,
        **kwargs
    }
    
    ax.text(x, y, text, **text_params)

def add_legend_outside(ax: plt.Axes, **kwargs) -> None:
    """
    Add legend outside plot area.
    
    Args:
        ax: Axes object
        **kwargs: Additional legend arguments
    """
    legend_params = {
        'bbox_to_anchor': (1.05, 1),
        'loc': 'upper left',
        'fontsize': 'small',
        **kwargs
    }
    
    ax.legend(**legend_params)

def save_plot(fig: plt.Figure, filepath: str, dpi: int = 300, **kwargs) -> None:
    """
    Save plot with consistent parameters.
    
    Args:
        fig: Figure object
        filepath: Output file path
        dpi: Resolution
        **kwargs: Additional savefig arguments
    """
    save_params = {
        'dpi': dpi,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none',
        **save_params
    }
    
    fig.savefig(filepath, **save_params)

def create_comparison_plot(
    data_dict: Dict[str, np.ndarray],
    plot_type: str = 'scatter',
    figsize: Union[str, Tuple[int, int]] = 'large',
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comparison plot for multiple datasets.
    
    Args:
        data_dict: Dictionary with labels as keys and data as values
        plot_type: Type of plot ('scatter', 'line', 'hist')
        figsize: Figure size
        **kwargs: Additional plotting arguments
        
    Returns:
        Figure and axes array
    """
    n_plots = len(data_dict)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = create_subplots(n_rows, n_cols, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (label, data) in enumerate(data_dict.items()):
        if i < len(axes):
            ax = axes[i]
            
            if plot_type == 'scatter' and data.shape[1] >= 2:
                scatter_plot_2d(data[:, 0], data[:, 1], ax=ax, title=label, **kwargs)
            elif plot_type == 'hist':
                histogram_plot(data.flatten(), ax=ax, title=label, **kwargs)
            elif plot_type == 'line':
                line_plot(range(len(data)), data, ax=ax, title=label, **kwargs)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axes

def create_method_comparison_grid(
    results_dict: Dict[str, Dict[str, Any]],
    z_latent: np.ndarray,
    figsize: Union[str, Tuple[int, int]] = 'large'
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create grid comparison of different sampling methods.
    
    Args:
        results_dict: Dictionary with method results
        z_latent: Latent coordinates
        figsize: Figure size
        
    Returns:
        Figure and axes array
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    if n_methods <= 1:
        return create_figure(figsize)
    
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = create_subplots(n_rows, n_cols, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, method in enumerate(methods):
        if i < len(axes):
            ax = axes[i]
            result = results_dict[method]
            
            # Plot all points
            ax.scatter(z_latent[:, 0], z_latent[:, 1], 
                      alpha=0.1, s=5, color='lightblue')
            
            # Plot selected points if available
            if 'selected_indices' in result:
                selected_indices = result['selected_indices']
                selected_latent = z_latent[selected_indices]
                ax.scatter(selected_latent[:, 0], selected_latent[:, 1], 
                          alpha=0.8, s=30, color='red')
                
                ax.set_title(f'{method.title()}\n{len(selected_indices)} samples')
            else:
                ax.set_title(f'{method.title()}\nNo data')
            
            ax.set_xlabel('Latent Dimension 1')
            ax.set_ylabel('Latent Dimension 2')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    return fig, axes

def add_statistical_annotations(ax: plt.Axes, data: np.ndarray, position: str = 'upper right') -> None:
    """
    Add statistical annotations to plot.
    
    Args:
        ax: Axes object
        data: Data array
        position: Position for annotations
    """
    if data.ndim == 1:
        stats_text = f"""
Mean: {np.mean(data):.3f}
Std: {np.std(data):.3f}
Min: {np.min(data):.3f}
Max: {np.max(data):.3f}
N: {len(data)}
"""
    elif data.ndim == 2 and data.shape[1] >= 2:
        corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
        stats_text = f"""
Correlation: {corr:.3f}
N: {len(data)}
Dim 1: μ={np.mean(data[:, 0]):.3f}, σ={np.std(data[:, 0]):.3f}
Dim 2: μ={np.mean(data[:, 1]):.3f}, σ={np.std(data[:, 1]):.3f}
"""
    else:
        stats_text = f"N: {len(data)}"
    
    add_text_box(ax, stats_text.strip(), position)

def create_automotive_dashboard(
    feature_data: Dict[str, np.ndarray],
    sampled_data: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (20, 15)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a comprehensive dashboard for automotive data analysis.
    
    Args:
        feature_data: Original feature data
        sampled_data: Sampled feature data
        figsize: Figure size
        
    Returns:
        Figure and axes array
    """
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    # Define feature categories and their priorities
    feature_priorities = {
        'speed': ['T1_ego_speed', 'T1_V1 (CIPV)_absolute_velocity_x', 'T2_V1 (CIPV)_absolute_velocity_x'],
        'position': ['T1_V1 (CIPV)_pos_x', 'T1_V1 (CIPV)_pos_y'],
        'acceleration': ['T1_V1 (CIPV)_absolute_acceleration_x'],
        'climate': ['T1_climate_outside_temperature'],
        'road': ['FrontCurvature'],
        'categorical': ['code_country', 'T1_climate_day_period', 'NumberOfLanesInPrincipalRoad']
    }
    
    plot_idx = 0
    
    # Plot key numerical features
    for category, features in feature_priorities.items():
        if category == 'categorical':
            continue
            
        for feature in features:
            if feature in feature_data and feature in sampled_data and plot_idx < 8:
                ax = axes[plot_idx]
                
                histogram_comparison_plot(
                    feature_data[feature], 
                    sampled_data[feature],
                    feature,
                    ax=ax
                )
                
                plot_idx += 1
    
    # Plot key categorical features
    for feature in feature_priorities['categorical']:
        if feature in feature_data and feature in sampled_data and plot_idx < 11:
            ax = axes[plot_idx]
            
            # Convert to pandas Series for categorical plotting
            orig_series = pd.Series(feature_data[feature])
            samp_series = pd.Series(sampled_data[feature])
            
            categorical_comparison_plot(
                orig_series,
                samp_series,
                feature,
                ax=ax
            )
            
            plot_idx += 1
    
    # Create summary plot in the last subplot
    if plot_idx < 12:
        ax = axes[-1]
        
        # Calculate quality scores for each feature
        feature_stats = {}
        for feature in feature_data.keys():
            if feature in sampled_data:
                if feature in feature_priorities['categorical']:
                    # Use chi-square test for categorical
                    try:
                        orig_counts = pd.Series(feature_data[feature]).value_counts()
                        samp_counts = pd.Series(sampled_data[feature]).value_counts()
                        
                        all_values = sorted(set(orig_counts.index) | set(samp_counts.index))
                        orig_aligned = [orig_counts.get(v, 0) for v in all_values]
                        samp_aligned = [samp_counts.get(v, 0) for v in all_values]
                        
                        contingency = np.array([orig_aligned, samp_aligned])
                        _, p_value = stats.chi2_contingency(contingency)[:2]
                        quality_score = p_value
                    except:
                        quality_score = 0
                else:
                    # Use KS test for numerical
                    try:
                        orig_clean = feature_data[feature][~np.isnan(feature_data[feature])]
                        samp_clean = sampled_data[feature][~np.isnan(sampled_data[feature])]
                        _, p_value = stats.ks_2samp(orig_clean, samp_clean)
                        quality_score = p_value
                    except:
                        quality_score = 0
                
                feature_stats[feature] = {'quality_score': quality_score}
        
        create_automotive_feature_summary(feature_stats, ax=ax)
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        if i != len(axes) - 1:  # Don't hide the summary plot
            axes[i].set_visible(False)
    
    plt.suptitle('Automotive Data Sampling Quality Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, axes

# Initialize plotting style on import
setup_plot_style()