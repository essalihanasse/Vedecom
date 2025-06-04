"""
Shared plotting utilities for the VAE pipeline visualization system.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import warnings

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")

class PlotStyle:
    """Consistent plotting style for the VAE pipeline."""
    
    # Color schemes
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'accent': '#F18F01',
        'success': '#C73E1D',
        'background': '#F5F5F5',
        'text': '#2C2C2C'
    }
    
    # Common figure sizes
    FIGSIZE = {
        'small': (8, 6),
        'medium': (12, 8),
        'large': (16, 10),
        'wide': (16, 6),
        'square': (10, 10)
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

def setup_plot_style():
    """Setup consistent plot styling."""
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
        'figure.facecolor': 'white'
    })

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
    **kwargs
) -> plt.Axes:
    """
    Create standardized 2D scatter plot.
    
    Args:
        x: X coordinates
        y: Y coordinates
        c: Colors for points
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar: Whether to add colorbar
        **kwargs: Additional scatter plot arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    # Merge with default scatter parameters
    scatter_params = {**PlotStyle.SCATTER_PARAMS, **kwargs}
    
    if c is not None:
        scatter = ax.scatter(x, y, c=c, **scatter_params)
        if colorbar and hasattr(scatter, 'colorbar'):
            plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(x, y, color=PlotStyle.COLORS['primary'], **scatter_params)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
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
    **kwargs
) -> plt.Axes:
    """
    Create standardized histogram.
    
    Args:
        data: Data to plot
        ax: Existing axes (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        **kwargs: Additional hist arguments
        
    Returns:
        Axes object
    """
    if ax is None:
        fig, ax = create_figure()
    
    hist_params = {
        'alpha': 0.7,
        'color': PlotStyle.COLORS['primary'],
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
        'lower left': (0.02, 0.02)
    }
    
    x, y = positions.get(position, (0.98, 0.98))
    ha = 'right' if 'right' in position else 'left'
    va = 'top' if 'upper' in position else 'bottom'
    
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
        **kwargs
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

# Initialize plotting style on import
setup_plot_style()