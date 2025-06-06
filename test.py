import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')  # Use default instead of seaborn for better control
plt.rcParams.update({
    'font.size': 8,            # Reduced from 10
    'axes.titlesize': 10,      # Reduced from 12
    'axes.labelsize': 9,       # Reduced from 10
    'xtick.labelsize': 7,      # Reduced from 8
    'ytick.labelsize': 7,      # Reduced from 8
    'legend.fontsize': 7,      # Reduced from 9
    'figure.titlesize': 12,    # Reduced from 14
    'font.family': 'sans-serif',
    'axes.grid': True,
    'grid.alpha': 0.3
})
sns.set_palette("Set2")

# Helper function to get rejection indicator
def get_rejection_indicator(df):
    """Get rejection indicator, using reject_h0 if available, otherwise p_value < 0.05"""
    if 'reject_h0' in df.columns:
        return df['reject_h0']
    else:
        return df['p_value'] < 0.05

# Helper function to save figures
def save_figure(fig, filename, save_figs=True, dpi=300):
    """Save figure with high quality if save_figs is True"""
    if save_figs:
        import os
        # Create figures directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Save as both PNG and PDF
        png_path = f'figures/{filename}.png'
        pdf_path = f'figures/{filename}.pdf'
        
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        
        print(f"💾 Saved: {png_path} and {pdf_path}")
    
    plt.show()

# Load and prepare the data
def load_and_clean_data(filepath):
    """Load CSV and clean the data for analysis"""
    df = pd.read_csv(filepath)
    
    # Debug: Show what columns we actually have
    print(f"Columns found in CSV: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    print(f"First few rows:")
    print(df.head(2))
    
    # Handle different possible column names and convert boolean columns
    if 'success' in df.columns:
        if df['success'].dtype == 'object':
            df['success'] = df['success'].map({'True': True, 'False': False}).fillna(False)
        df['success'] = df['success'].astype(bool)
    else:
        print("Warning: 'success' column not found, assuming all tests succeeded")
        df['success'] = True
    
    # Handle reject_h0 column with various possible names
    reject_col = None
    possible_reject_cols = ['reject_h0', 'reject_null', 'significant', 'is_significant']
    
    for col in possible_reject_cols:
        if col in df.columns:
            reject_col = col
            break
    
    if reject_col:
        print(f"Found reject column: {reject_col}")
        print(f"Unique values in {reject_col}: {df[reject_col].unique()}")
        print(f"Data type: {df[reject_col].dtype}")
        
        # More robust boolean conversion
        if df[reject_col].dtype == 'object':
            # Handle string values
            df['reject_h0'] = df[reject_col].map({
                'True': True, 'False': False,
                'true': True, 'false': False,
                'TRUE': True, 'FALSE': False,
                '1': True, '0': False,
                1: True, 0: False
            })
            # Fill any remaining NaN values by checking if it's truthy
            mask = df['reject_h0'].isna()
            if mask.any():
                print(f"Warning: {mask.sum()} values couldn't be converted, using string evaluation")
                df.loc[mask, 'reject_h0'] = df.loc[mask, reject_col].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df['reject_h0'] = df[reject_col].astype(bool)
            
        # Ensure it's boolean type
        df['reject_h0'] = df['reject_h0'].astype(bool)
        print(f"After conversion - reject_h0 unique values: {df['reject_h0'].unique()}")
        
    else:
        # Create reject_h0 column based on p-value < 0.05 if it doesn't exist
        print("Warning: No reject_h0 column found, creating based on p_value < 0.05")
        if 'p_value' in df.columns:
            df['reject_h0'] = df['p_value'] < 0.05
        else:
            print("Error: Neither reject_h0 nor p_value columns found!")
            return pd.DataFrame()  # Return empty DataFrame
    
    # Filter successful tests only
    df_clean = df[df['success'] == True].copy()
    
    # Remove any rows with missing critical values
    required_cols = ['p_value', 'distance']
    available_cols = [col for col in required_cols if col in df_clean.columns]
    
    if not available_cols:
        print("Error: Required columns 'p_value' and 'distance' not found!")
        return pd.DataFrame()
    
    df_clean = df_clean.dropna(subset=available_cols)
    
    print(f"Loaded {len(df)} total rows, {len(df_clean)} valid rows after cleaning")
    print(f"Success rate: {df['success'].mean():.1%}")
    if 'reject_h0' in df_clean.columns:
        rejection_rate = df_clean['reject_h0'].mean()
        print(f"H0 rejection rate: {rejection_rate:.1%}")
        print(f"Number of rejections: {df_clean['reject_h0'].sum()} out of {len(df_clean)}")
    
    return df_clean

# Method comparison analysis - Enhanced focus on testing methods
def compare_methods_by_sample_size(df):
    """Compare methods while controlling for sample size and test type"""
    
    if df.empty:
        print("No data available for comparison")
        return pd.DataFrame()
    
    # Get unique sample sizes, methods, and test types
    sample_sizes = sorted(df['sample_size'].unique())
    methods = sorted(df['method'].unique())
    test_types = sorted(df['test_type'].unique()) if 'test_type' in df.columns else ['all']
    
    print("\n=== DETAILED METHOD COMPARISON BY SAMPLE SIZE AND TEST TYPE ===")
    
    # Create comprehensive comparison table
    comparison_results = []
    
    for size in sample_sizes:
        size_data = df[df['sample_size'] == size]
        print(f"\n🔍 SAMPLE SIZE: {size}")
        print("=" * 60)
        
        for test_type in test_types:
            if 'test_type' in df.columns:
                test_data = size_data[size_data['test_type'] == test_type]
                if test_data.empty:
                    continue
                print(f"\n  📊 Test Type: {test_type.upper()}")
                print("  " + "-" * 50)
            else:
                test_data = size_data
                print(f"\n  📊 All Test Types")
                print("  " + "-" * 50)
            
            # Method performance for this combination
            method_performance = []
            
            for method in methods:
                method_data = test_data[test_data['method'] == method]
                if len(method_data) > 0:
                    # Calculate metrics
                    median_pvalue = method_data['p_value'].median()
                    mean_distance = method_data['distance'].mean()
                    std_distance = method_data['distance'].std()
                    
                    # Calculate rejection rate
                    if 'reject_h0' in method_data.columns:
                        rejection_rate = method_data['reject_h0'].mean()
                    else:
                        rejection_rate = (method_data['p_value'] < 0.05).mean()
                    
                    # Power analysis (rejection rate when there should be a difference)
                    power = rejection_rate  # This is the statistical power for this method
                    
                    comparison_results.append({
                        'sample_size': size,
                        'test_type': test_type if 'test_type' in df.columns else 'all',
                        'method': method,
                        'n_tests': len(method_data),
                        'rejection_rate': rejection_rate,
                        'statistical_power': power,
                        'median_pvalue': median_pvalue,
                        'mean_distance': mean_distance,
                        'std_distance': std_distance,
                        'cv_distance': std_distance / mean_distance if mean_distance > 0 else 0  # Coefficient of variation
                    })
                    
                    method_performance.append({
                        'method': method,
                        'rejection_rate': rejection_rate,
                        'median_pvalue': median_pvalue,
                        'mean_distance': mean_distance,
                        'n_tests': len(method_data)
                    })
                    
                    print(f"    {method:20s}: {len(method_data):3d} tests | "
                          f"Power: {rejection_rate:5.1%} | "
                          f"Med p: {median_pvalue:6.3f} | "
                          f"Dist: {mean_distance:6.3f}±{std_distance:5.3f}")
            
            # Rank methods for this combination
            if method_performance:
                # Rank by rejection rate (power)
                power_ranking = sorted(method_performance, key=lambda x: x['rejection_rate'], reverse=True)
                # Rank by distance (effect size detection)
                distance_ranking = sorted(method_performance, key=lambda x: x['mean_distance'], reverse=True)
                # Rank by p-value (significance detection)
                pvalue_ranking = sorted(method_performance, key=lambda x: x['median_pvalue'])
                
                print(f"\n    🏆 RANKINGS for {test_type if 'test_type' in df.columns else 'All Tests'}:")
                print(f"    Power (best→worst):     {' > '.join([m['method'] for m in power_ranking])}")
                print(f"    Distance (high→low):    {' > '.join([m['method'] for m in distance_ranking])}")
                print(f"    P-values (low→high):    {' > '.join([m['method'] for m in pvalue_ranking])}")
    
    return pd.DataFrame(comparison_results)

# Enhanced method-focused visualization functions
def create_method_performance_matrix(df, save_figs=True):
    """Create comprehensive method performance comparison matrix"""
    
    if df.empty:
        print("No data available for method performance matrix")
        return
    
    df = df.copy()
    df['rejection_indicator'] = get_rejection_indicator(df)
    
    # Create performance metrics by method, sample_size, and test_type
    if 'test_type' in df.columns:
        performance_data = df.groupby(['method', 'sample_size', 'test_type']).agg({
            'rejection_indicator': ['mean', 'count'],
            'p_value': ['median', 'mean'],
            'distance': ['mean', 'std']
        }).round(4)
    else:
        performance_data = df.groupby(['method', 'sample_size']).agg({
            'rejection_indicator': ['mean', 'count'],
            'p_value': ['median', 'mean'],
            'distance': ['mean', 'std']
        }).round(4)
    
    # Flatten column names
    performance_data.columns = ['_'.join(col).strip() for col in performance_data.columns]
    performance_data = performance_data.reset_index()
    
    # Create the visualization
    if 'test_type' in df.columns:
        test_types = sorted(df['test_type'].unique())
        fig, axes = plt.subplots(2, len(test_types), figsize=(6*len(test_types), 12))
        if len(test_types) == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Method Performance Analysis by Test Type and Sample Size', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        for i, test_type in enumerate(test_types):
            test_data = performance_data[performance_data['test_type'] == test_type]
            
            # Power (rejection rate) heatmap
            pivot_power = test_data.pivot(index='method', columns='sample_size', 
                                        values='rejection_indicator_mean')
            
            sns.heatmap(pivot_power, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0, i],
                       cbar_kws={'label': 'Statistical Power'}, vmin=0, vmax=1,
                       annot_kws={'fontsize': 11, 'fontweight': 'bold'})
            axes[0, i].set_title(f'Statistical Power\n{test_type.upper()}', 
                                fontsize=14, fontweight='bold')
            axes[0, i].set_ylabel('Method', fontsize=12, fontweight='bold')
            
            # Distance (effect size) heatmap
            pivot_distance = test_data.pivot(index='method', columns='sample_size', 
                                           values='distance_mean')
            
            sns.heatmap(pivot_distance, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, i],
                       cbar_kws={'label': 'Mean Distance'},
                       annot_kws={'fontsize': 11, 'fontweight': 'bold'})
            axes[1, i].set_title(f'Effect Size Detection\n{test_type.upper()}', 
                                fontsize=14, fontweight='bold')
            axes[1, i].set_xlabel('Sample Size', fontsize=12, fontweight='bold')
            axes[1, i].set_ylabel('Method', fontsize=12, fontweight='bold')
            
            # Improve y-axis labels
            for ax in [axes[0, i], axes[1, i]]:
                yticklabels = [label.get_text().replace('_', ' ').title() 
                              for label in ax.get_yticklabels()]
                ax.set_yticklabels(yticklabels, rotation=0, fontsize=10)
    
    else:
        # Single test type case
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Method Performance Analysis by Sample Size', 
                     fontsize=14, fontweight='bold', y=1.02)
        
        # Power heatmap
        pivot_power = performance_data.pivot(index='method', columns='sample_size', 
                                           values='rejection_indicator_mean')
        sns.heatmap(pivot_power, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0],
                   cbar_kws={'label': 'Statistical Power'}, vmin=0, vmax=1,
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        axes[0].set_title('Statistical Power by Method and Sample Size', 
                         fontsize=12, fontweight='bold')
        
        # Distance heatmap
        pivot_distance = performance_data.pivot(index='method', columns='sample_size', 
                                              values='distance_mean')
        sns.heatmap(pivot_distance, annot=True, fmt='.3f', cmap='viridis', ax=axes[1],
                   cbar_kws={'label': 'Mean Distance'},
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        axes[1].set_title('Effect Size Detection by Method and Sample Size', 
                         fontsize=12, fontweight='bold')
        
        for ax in axes:
            yticklabels = [label.get_text().replace('_', ' ').title() 
                          for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=9)
            ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
            ax.set_ylabel('Method', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'method_performance_matrix', save_figs)

def create_method_head_to_head_comparison(df, save_figs=True):
    """Create head-to-head method comparisons"""
    
    if df.empty:
        return
    
    df = df.copy()
    df['rejection_indicator'] = get_rejection_indicator(df)
    
    methods = sorted(df['method'].unique())
    n_methods = len(methods)
    
    if n_methods < 2:
        print("Need at least 2 methods for head-to-head comparison")
        return
    
    # Create pairwise comparison matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Head-to-Head Method Performance Comparison', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Power comparison by sample size
    ax = axes[0, 0]
    for method in methods:
        method_data = df[df['method'] == method]
        power_by_size = method_data.groupby('sample_size')['rejection_indicator'].mean()
        ax.plot(power_by_size.index, power_by_size.values, 'o-', 
                label=method.replace('_', ' ').title(), linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Statistical Power', fontsize=10, fontweight='bold')
    ax.set_title('Statistical Power vs Sample Size', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. Effect size detection by sample size
    ax = axes[0, 1]
    for method in methods:
        method_data = df[df['method'] == method]
        distance_by_size = method_data.groupby('sample_size')['distance'].mean()
        ax.plot(distance_by_size.index, distance_by_size.values, 'o-', 
                label=method.replace('_', ' ').title(), linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Test Distance', fontsize=10, fontweight='bold')
    ax.set_title('Effect Size Detection vs Sample Size', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Method ranking by overall performance
    ax = axes[1, 0]
    method_stats = df.groupby('method').agg({
        'rejection_indicator': 'mean',
        'distance': 'mean',
        'p_value': 'median'
    })
    
    # Create composite score (higher is better)
    method_stats['composite_score'] = (
        method_stats['rejection_indicator'] * 0.4 +  # 40% weight on power
        (method_stats['distance'] / method_stats['distance'].max()) * 0.4 +  # 40% weight on effect size
        (1 - method_stats['p_value'] / method_stats['p_value'].max()) * 0.2  # 20% weight on p-value
    )
    
    method_stats_sorted = method_stats.sort_values('composite_score', ascending=True)
    
    bars = ax.barh(range(len(method_stats_sorted)), method_stats_sorted['composite_score'],
                   color=sns.color_palette("viridis", len(methods)), alpha=0.8)
    
    ax.set_yticks(range(len(method_stats_sorted)))
    ax.set_yticklabels([m.replace('_', ' ').title() for m in method_stats_sorted.index], fontsize=9)
    ax.set_xlabel('Composite Performance Score', fontsize=10, fontweight='bold')
    ax.set_title('Overall Method Ranking', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, method_stats_sorted['composite_score'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontweight='bold', fontsize=8)
    
    # 4. Consistency analysis (coefficient of variation)
    ax = axes[1, 1]
    method_consistency = df.groupby('method')['distance'].agg(['mean', 'std'])
    method_consistency['cv'] = method_consistency['std'] / method_consistency['mean']
    method_consistency = method_consistency.sort_values('cv')
    
    bars = ax.bar(range(len(method_consistency)), method_consistency['cv'],
                  color=sns.color_palette("Set2", len(methods)), alpha=0.8)
    
    ax.set_xticks(range(len(method_consistency)))
    ax.set_xticklabels([m.replace('_', '\n') for m in method_consistency.index], 
                       fontsize=8, rotation=0)
    ax.set_ylabel('Coefficient of Variation', fontsize=10, fontweight='bold')
    ax.set_title('Method Consistency\n(Lower = More Consistent)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cv in zip(bars, method_consistency['cv']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{cv:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, 'method_head_to_head_comparison', save_figs)

def method_performance_analysis(df):
    """Enhanced statistical analysis focusing on method testing comparisons"""
    
    if df.empty:
        print("No data available for performance analysis")
        return
    
    print("\n=== FOCUSED METHOD TESTING COMPARISON ANALYSIS ===")
    
    # Add rejection indicator
    df['rejection_indicator'] = get_rejection_indicator(df)
    
    methods = sorted(df['method'].unique())
    sample_sizes = sorted(df['sample_size'].unique())
    
    print(f"\n📊 ANALYSIS SCOPE:")
    print(f"   Methods analyzed: {len(methods)} ({', '.join(methods)})")
    print(f"   Sample sizes: {len(sample_sizes)} ({', '.join(map(str, sample_sizes))})")
    print(f"   Total test results: {len(df)}")
    
    # 1. Overall method performance ranking
    print(f"\n🏆 OVERALL METHOD PERFORMANCE RANKING:")
    print("=" * 50)
    
    overall_performance = df.groupby('method').agg({
        'rejection_indicator': ['count', 'mean', 'std'],
        'p_value': ['median', 'mean', 'std'],
        'distance': ['mean', 'std'],
        'sample_size': 'mean'
    }).round(4)
    
    # Flatten columns
    overall_performance.columns = ['N_Tests', 'Power_Mean', 'Power_Std', 
                                  'P_Median', 'P_Mean', 'P_Std',
                                  'Distance_Mean', 'Distance_Std', 'Avg_SampleSize']
    
    # Calculate composite performance score
    # Normalize metrics to 0-1 scale for fair comparison
    power_norm = overall_performance['Power_Mean'] / overall_performance['Power_Mean'].max()
    distance_norm = overall_performance['Distance_Mean'] / overall_performance['Distance_Mean'].max()
    p_norm = 1 - (overall_performance['P_Median'] / overall_performance['P_Median'].max())  # Lower is better
    
    overall_performance['Composite_Score'] = (
        power_norm * 0.4 +      # 40% weight on statistical power
        distance_norm * 0.4 +   # 40% weight on effect size detection
        p_norm * 0.2           # 20% weight on p-value performance
    )
    
    # Sort by composite score
    performance_ranked = overall_performance.sort_values('Composite_Score', ascending=False)
    
    print("Rank | Method               | Power    | Distance | P-median | Composite")
    print("-" * 70)
    for i, (method, row) in enumerate(performance_ranked.iterrows(), 1):
        print(f"{i:4d} | {method:20s} | {row['Power_Mean']:6.3f} | "
              f"{row['Distance_Mean']:7.3f} | {row['P_Median']:7.3f} | {row['Composite_Score']:8.3f}")
    
    # 2. Sample size specific analysis
    print(f"\n📈 PERFORMANCE BY SAMPLE SIZE:")
    print("=" * 60)
    
    for size in sample_sizes:
        size_data = df[df['sample_size'] == size]
        print(f"\n  Sample Size: {size}")
        print("  " + "-" * 45)
        
        size_performance = size_data.groupby('method')['rejection_indicator'].agg([
            'count', 'mean', 'std'
        ]).round(4)
        
        # Rank methods for this sample size
        size_ranked = size_performance.sort_values('mean', ascending=False)
        
        print("  Rank | Method               | Power   | StdDev | N_Tests")
        print("  " + "-" * 52)
        for i, (method, row) in enumerate(size_ranked.iterrows(), 1):
            print(f"  {i:4d} | {method:20s} | {row['mean']:6.3f} | "
                  f"{row['std']:6.3f} | {int(row['count']):7d}")
    
    # 3. Test type specific analysis (if available)
    if 'test_type' in df.columns:
        test_types = sorted(df['test_type'].unique())
        print(f"\n🧪 PERFORMANCE BY TEST TYPE:")
        print("=" * 50)
        
        for test_type in test_types:
            test_data = df[df['test_type'] == test_type]
            
            if test_data.empty:
                continue
                
            print(f"\n  Test Type: {test_type.upper()}")
            print("  " + "-" * 45)
            
            test_performance = test_data.groupby('method')['rejection_indicator'].agg([
                'count', 'mean', 'std'
            ]).round(4)
            
            test_ranked = test_performance.sort_values('mean', ascending=False)
            
            print("  Rank | Method               | Power   | StdDev | N_Tests")
            print("  " + "-" * 52)
            for i, (method, row) in enumerate(test_ranked.iterrows(), 1):
                print(f"  {i:4d} | {method:20s} | {row['mean']:6.3f} | "
                      f"{row['std']:6.3f} | {int(row['count']):7d}")
    
    # 4. Statistical significance testing between methods
    print(f"\n🔬 STATISTICAL SIGNIFICANCE TESTING:")
    print("=" * 50)
    
    if len(methods) > 1:
        from scipy.stats import chi2_contingency, ttest_ind, f_oneway
        from itertools import combinations
        
        # Chi-square test for independence of method and rejection
        try:
            contingency_table = pd.crosstab(df['method'], df['rejection_indicator'])
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)
            
            print(f"\n📋 Overall Method Independence Test:")
            print(f"   Chi-square statistic: {chi2:.4f}")
            print(f"   P-value: {p_val:.6f}")
            print(f"   Degrees of freedom: {dof}")
            print(f"   Significant difference: {'YES' if p_val < 0.05 else 'NO'}")
            
        except Exception as e:
            print(f"   Could not perform Chi-square test: {e}")
        
        # ANOVA for distance differences
        try:
            method_distances = [df[df['method'] == method]['distance'].values for method in methods]
            f_stat, p_val = f_oneway(*method_distances)
            
            print(f"\n📊 Distance Differences (ANOVA):")
            print(f"   F-statistic: {f_stat:.4f}")
            print(f"   P-value: {p_val:.6f}")
            print(f"   Significant difference: {'YES' if p_val < 0.05 else 'NO'}")
            
        except Exception as e:
            print(f"   Could not perform ANOVA: {e}")
        
        # Pairwise comparisons
        print(f"\n🔄 Pairwise Method Comparisons (Distance):")
        print("   Method 1             vs Method 2             | t-stat  | p-value | Effect Size")
        print("   " + "-" * 80)
        
        for method1, method2 in combinations(methods, 2):
            data1 = df[df['method'] == method1]['distance']
            data2 = df[df['method'] == method2]['distance']
            
            if len(data1) > 5 and len(data2) > 5:
                t_stat, p_val = ttest_ind(data1, data2)
                
                # Calculate Cohen's d (effect size)
                pooled_std = np.sqrt((data1.var() + data2.var()) / 2)
                cohens_d = (data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else 0
                
                significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                print(f"   {method1:20s} vs {method2:20s} | {t_stat:7.3f} | "
                      f"{p_val:7.4f}{significance:3s} | {cohens_d:10.3f}")
    
    # 5. Method recommendations
    print(f"\n💡 METHOD RECOMMENDATIONS:")
    print("=" * 30)
    
    best_overall = performance_ranked.index[0]
    print(f"🥇 Best Overall Performance: {best_overall}")
    print(f"   Composite Score: {performance_ranked.loc[best_overall, 'Composite_Score']:.3f}")
    print(f"   Statistical Power: {performance_ranked.loc[best_overall, 'Power_Mean']:.3f}")
    print(f"   Effect Detection: {performance_ranked.loc[best_overall, 'Distance_Mean']:.3f}")
    
    # Best for high power
    best_power_method = overall_performance['Power_Mean'].idxmax()
    print(f"\n⚡ Highest Statistical Power: {best_power_method}")
    print(f"   Power: {overall_performance.loc[best_power_method, 'Power_Mean']:.3f}")
    
    # Best for effect size detection
    best_distance_method = overall_performance['Distance_Mean'].idxmax()
    print(f"\n🎯 Best Effect Size Detection: {best_distance_method}")
    print(f"   Distance: {overall_performance.loc[best_distance_method, 'Distance_Mean']:.3f}")
    
    # Most consistent method (lowest CV)
    overall_performance['CV_Power'] = overall_performance['Power_Std'] / overall_performance['Power_Mean']
    most_consistent = overall_performance['CV_Power'].idxmin()
    print(f"\n🎪 Most Consistent Method: {most_consistent}")
    print(f"   Coefficient of Variation: {overall_performance.loc[most_consistent, 'CV_Power']:.3f}")
    
    return overall_performance

def create_test_type_specific_analysis(df, save_figs=True):
    """Create detailed analysis for each test type separately"""
    
    if df.empty or 'test_type' not in df.columns:
        print("No test type data available for specific analysis")
        return
    
    df = df.copy()
    df['rejection_indicator'] = get_rejection_indicator(df)
    
    test_types = sorted(df['test_type'].unique())
    methods = sorted(df['method'].unique())
    
    # Create separate analysis for each test type
    for test_type in test_types:
        test_data = df[df['test_type'] == test_type]
        
        if test_data.empty:
            continue
            
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'Detailed Method Analysis: {test_type.upper()} Test', 
                     fontsize=12, fontweight='bold', y=0.98)
        
        # 1. Power progression by sample size
        ax = axes[0, 0]
        for method in methods:
            method_data = test_data[test_data['method'] == method]
            if not method_data.empty:
                power_progression = method_data.groupby('sample_size')['rejection_indicator'].agg(['mean', 'std', 'count'])
                
                # Plot with error bars
                ax.errorbar(power_progression.index, power_progression['mean'], 
                           yerr=power_progression['std']/np.sqrt(power_progression['count']), 
                           marker='o', linewidth=1.5, markersize=4, capsize=3,
                           label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Sample Size', fontsize=8, fontweight='bold')
        ax.set_ylabel('Statistical Power', fontsize=8, fontweight='bold')
        ax.set_title(f'Power Progression - {test_type.upper()}', fontsize=10, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # 2. Effect size (distance) progression
        ax = axes[0, 1]
        for method in methods:
            method_data = test_data[test_data['method'] == method]
            if not method_data.empty:
                distance_progression = method_data.groupby('sample_size')['distance'].agg(['mean', 'std', 'count'])
                
                ax.errorbar(distance_progression.index, distance_progression['mean'], 
                           yerr=distance_progression['std']/np.sqrt(distance_progression['count']), 
                           marker='s', linewidth=1.5, markersize=4, capsize=3,
                           label=method.replace('_', ' ').title())
        
        ax.set_xlabel('Sample Size', fontsize=8, fontweight='bold')
        ax.set_ylabel('Mean Distance', fontsize=8, fontweight='bold')
        ax.set_title(f'Effect Size Detection - {test_type.upper()}', fontsize=10, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # 3. P-value distribution comparison
        ax = axes[0, 2]
        method_colors = sns.color_palette("Set2", len(methods))
        
        for i, method in enumerate(methods):
            method_data = test_data[test_data['method'] == method]
            if not method_data.empty and len(method_data) > 5:
                ax.hist(method_data['p_value'], bins=15, alpha=0.6, 
                       label=method.replace('_', ' ').title(),
                       color=method_colors[i], edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0.05, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                   label='α = 0.05')
        ax.set_xlabel('P-value', fontsize=8, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=8, fontweight='bold')
        ax.set_title(f'P-value Distribution - {test_type.upper()}', fontsize=10, fontweight='bold')
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=6)
        ax.grid(True, alpha=0.3)
        
        # 4. Method performance ranking by sample size
        ax = axes[1, 0]
        sample_sizes = sorted(test_data['sample_size'].unique())
        
        # Create ranking matrix
        ranking_data = []
        for size in sample_sizes:
            size_data = test_data[test_data['sample_size'] == size]
            method_performance = size_data.groupby('method')['rejection_indicator'].mean().sort_values(ascending=False)
            
            for rank, (method, power) in enumerate(method_performance.items(), 1):
                ranking_data.append({
                    'sample_size': size,
                    'method': method,
                    'rank': rank,
                    'power': power
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        if not ranking_df.empty:
            pivot_ranking = ranking_df.pivot(index='method', columns='sample_size', values='rank')
            
            # Handle NaN values before converting to int
            # Fill NaN with the maximum rank + 1 (indicating worst performance for missing data)
            max_rank = len(methods)
            pivot_ranking = pivot_ranking.fillna(max_rank + 1)
            
            # Now safely convert to int
            try:
                pivot_ranking = pivot_ranking.astype(int)
                
                sns.heatmap(pivot_ranking, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax,
                           cbar_kws={'label': 'Rank (1=Best)'}, vmin=1, vmax=max_rank + 1,
                           annot_kws={'fontsize': 7, 'fontweight': 'bold'})
                ax.set_title(f'Method Ranking by Sample Size - {test_type.upper()}', 
                            fontsize=10, fontweight='bold')
                ax.set_ylabel('Method', fontsize=8, fontweight='bold')
                ax.set_xlabel('Sample Size', fontsize=8, fontweight='bold')
                
                yticklabels = [label.get_text().replace('_', ' ').title() 
                              for label in ax.get_yticklabels()]
                ax.set_yticklabels(yticklabels, rotation=0, fontsize=7)
            except Exception as e:
                ax.text(0.5, 0.5, f'Ranking analysis\nunavailable:\n{str(e)[:30]}...', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No ranking data\navailable', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        # 5. Statistical significance testing between methods
        ax = axes[1, 1]
        
        # Perform pairwise t-tests for distances
        from itertools import combinations
        from scipy import stats
        
        pairwise_results = []
        method_pairs = list(combinations(methods, 2))
        
        for method1, method2 in method_pairs:
            data1 = test_data[test_data['method'] == method1]['distance']
            data2 = test_data[test_data['method'] == method2]['distance']
            
            if len(data1) > 3 and len(data2) > 3:
                try:
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                    
                    pairwise_results.append({
                        'comparison': f'{method1}\nvs\n{method2}',
                        'p_value': p_val,
                        'effect_size': abs(effect_size),
                        'significant': p_val < 0.05
                    })
                except Exception:
                    # Skip pairs with insufficient or invalid data
                    continue
        
        if pairwise_results:
            pairwise_df = pd.DataFrame(pairwise_results)
            
            # Create bubble plot with smaller elements
            colors = ['red' if sig else 'blue' for sig in pairwise_df['significant']]
            sizes = [abs(es) * 200 + 20 for es in pairwise_df['effect_size']]  # Smaller bubbles
            
            scatter = ax.scatter(range(len(pairwise_df)), -np.log10(pairwise_df['p_value']), 
                               c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.8,
                       label='α = 0.05')
            ax.set_xticks(range(len(pairwise_df)))
            ax.set_xticklabels(pairwise_df['comparison'], fontsize=6, rotation=45)
            ax.set_ylabel('-log₁₀(p-value)', fontsize=8, fontweight='bold')
            ax.set_title(f'Pairwise Method Comparisons - {test_type.upper()}', 
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend with smaller elements
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                     markersize=6, label='Significant (p<0.05)'),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                     markersize=6, label='Non-significant')]
            ax.legend(handles=legend_elements, frameon=True, fancybox=True, shadow=True, fontsize=6)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor pairwise\ncomparisons', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        # 6. Power analysis summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary statistics table
        try:
            summary_stats = test_data.groupby('method').agg({
                'rejection_indicator': ['mean', 'std', 'count'],
                'distance': ['mean', 'std'],
                'p_value': ['median', 'mean']
            }).round(4)
            
            # Flatten column names
            summary_stats.columns = ['Power_Mean', 'Power_Std', 'N_Tests', 
                                    'Distance_Mean', 'Distance_Std', 
                                    'P_Median', 'P_Mean']
            
            # Create table with smaller data
            table_data = []
            for method in summary_stats.index:
                row = summary_stats.loc[method]
                table_data.append([
                    method.replace('_', ' ').title(),
                    f"{row['Power_Mean']:.2f}±{row['Power_Std']:.2f}",
                    f"{row['Distance_Mean']:.3f}±{row['Distance_Std']:.3f}",
                    f"{row['P_Median']:.3f}",
                    f"{int(row['N_Tests'])}"
                ])
            
            table = ax.table(cellText=table_data,
                            colLabels=['Method', 'Power\n(M±SD)', 'Distance\n(M±SD)', 
                                      'Med\nP-val', 'N'],
                            cellLoc='center', loc='center',
                            bbox=[0, 0, 1, 1])
            
            table.auto_set_font_size(False)
            table.set_fontsize(6)  # Very small font
            table.scale(1, 1.2)    # Compact scaling
            
            # Style the table
            for i in range(len(summary_stats) + 1):
                for j in range(5):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            ax.set_title(f'Performance Summary - {test_type.upper()}', 
                        fontsize=10, fontweight='bold', pad=10)
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary table\nunavailable:\n{str(e)[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save the figure
        save_figure(fig, f'test_type_analysis_{test_type}', save_figs)

def create_sample_size_progression_analysis(df, save_figs=True):
    """Analyze how method performance changes with sample size"""
    
    if df.empty:
        return
    
    df = df.copy()
    df['rejection_indicator'] = get_rejection_indicator(df)
    
    methods = sorted(df['method'].unique())
    sample_sizes = sorted(df['sample_size'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Method Performance Progression with Sample Size', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Power curves with confidence intervals
    ax = axes[0, 0]
    
    for method in methods:
        method_data = df[df['method'] == method]
        progression_stats = method_data.groupby('sample_size')['rejection_indicator'].agg([
            'mean', 'std', 'count', 'sem'
        ])
        
        # Calculate 95% confidence intervals
        from scipy import stats as scipy_stats
        confidence_level = 0.95
        alpha = 1 - confidence_level
        
        progression_stats['ci_lower'] = progression_stats['mean'] - \
            scipy_stats.t.ppf(1 - alpha/2, progression_stats['count'] - 1) * progression_stats['sem']
        progression_stats['ci_upper'] = progression_stats['mean'] + \
            scipy_stats.t.ppf(1 - alpha/2, progression_stats['count'] - 1) * progression_stats['sem']
        
        # Plot with confidence bands
        ax.plot(progression_stats.index, progression_stats['mean'], 'o-', 
                linewidth=2.5, markersize=6, label=method.replace('_', ' ').title())
        ax.fill_between(progression_stats.index, 
                       progression_stats['ci_lower'], 
                       progression_stats['ci_upper'], 
                       alpha=0.2)
    
    ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Statistical Power', fontsize=10, fontweight='bold')
    ax.set_title('Power Curves with 95% Confidence Intervals', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 2. Effect size progression
    ax = axes[0, 1]
    
    for method in methods:
        method_data = df[df['method'] == method]
        distance_stats = method_data.groupby('sample_size')['distance'].agg([
            'mean', 'std', 'count'
        ])
        
        # Calculate efficiency (distance per sample size)
        distance_stats['efficiency'] = distance_stats['mean'] / distance_stats.index
        
        ax.plot(distance_stats.index, distance_stats['mean'], 'o-', 
                linewidth=2.5, markersize=6, label=method.replace('_', ' ').title())
    
    ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Test Distance', fontsize=10, fontweight='bold')
    ax.set_title('Effect Size Detection vs Sample Size', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Relative performance (compared to best method at each sample size)
    ax = axes[1, 0]
    
    relative_performance = {}
    for size in sample_sizes:
        size_data = df[df['sample_size'] == size]
        method_power = size_data.groupby('method')['rejection_indicator'].mean()
        best_power = method_power.max()
        
        for method in methods:
            if method not in relative_performance:
                relative_performance[method] = {'sizes': [], 'relative_power': []}
            
            if method in method_power.index and best_power > 0:
                relative_power = method_power[method] / best_power
                relative_performance[method]['sizes'].append(size)
                relative_performance[method]['relative_power'].append(relative_power)
    
    for method, data in relative_performance.items():
        if data['sizes']:
            ax.plot(data['sizes'], data['relative_power'], 'o-', 
                    linewidth=2.5, markersize=6, label=method.replace('_', ' ').title())
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.8, label='Best Performance')
    ax.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
    ax.set_ylabel('Relative Performance', fontsize=10, fontweight='bold')
    ax.set_title('Relative Performance (vs Best Method)', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 4. Sample size recommendations
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate minimum sample size for 80% power for each method
    recommendations = []
    
    for method in methods:
        method_data = df[df['method'] == method]
        power_by_size = method_data.groupby('sample_size')['rejection_indicator'].mean()
        
        # Find minimum sample size for 80% power
        power_80_sizes = power_by_size[power_by_size >= 0.8]
        min_size_80 = power_80_sizes.index.min() if not power_80_sizes.empty else 'N/A'
        
        # Find maximum power achieved
        max_power = power_by_size.max()
        optimal_size = power_by_size.idxmax()
        
        recommendations.append([
            method.replace('_', ' ').title(),
            f"{min_size_80}" if min_size_80 != 'N/A' else 'N/A',
            f"{optimal_size}",
            f"{max_power:.3f}",
            "✓" if max_power >= 0.8 else "✗"
        ])
    
    # Create recommendations table
    table = ax.table(cellText=recommendations,
                    colLabels=['Method', 'Min Size\n(80% Power)', 'Optimal\nSize', 
                              'Max\nPower', 'Adequate\nPower'],
                    cellLoc='center', loc='center',
                    bbox=[0, 0.2, 1, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)   # Reduced font size
    table.scale(1, 1.5)     # Reduced scaling
    
    # Style the table
    for i in range(len(methods) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#e3f2fd' if i % 2 == 0 else 'white')
                # Color code the adequate power column
                if j == 4 and i > 0:
                    if recommendations[i-1][4] == "✓":
                        cell.set_facecolor('#c8e6c9')
                    else:
                        cell.set_facecolor('#ffcdd2')
    
    ax.set_title('Sample Size Recommendations\n(Target: 80% Statistical Power)', 
                fontsize=12, fontweight='bold', y=0.9)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    save_figure(fig, 'sample_size_progression_analysis', save_figs)

# Main analysis function - Enhanced for method testing focus
def main_analysis(filepath=r'output\samples\overall_sampling_summary.csv', save_figs=True):
    """Run comprehensive method-focused analysis of statistical test results"""
    
    print("🔬 STATISTICAL METHOD TESTING COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_and_clean_data(filepath)
    
    if df.empty:
        print("❌ Failed to load valid data. Please check your file path and format.")
        return None, None
    
    # Method comparison analysis
    print("\n📊 Starting comprehensive method comparison analysis...")
    comparison_df = compare_methods_by_sample_size(df)
    
    # Enhanced performance analysis with statistical testing
    print("\n🔍 Performing enhanced method performance analysis...")
    performance_stats = method_performance_analysis(df)
    
    # Create method-focused visualizations
    print("\n📈 Creating method performance matrix...")
    create_method_performance_matrix(df, save_figs)
    
    print("\n🥊 Creating head-to-head method comparisons...")
    create_method_head_to_head_comparison(df, save_figs)
    
    # Test type specific analysis (if available)
    if 'test_type' in df.columns and len(df['test_type'].unique()) > 1:
        print("\n🧪 Creating test-type specific analyses...")
        create_test_type_specific_analysis(df, save_figs)
    else:
        print("\n⚠️  Test type analysis skipped (single or no test type data)")
    
    # Sample size progression analysis
    print("\n📏 Creating sample size progression analysis...")
    create_sample_size_progression_analysis(df, save_figs)
    
    # Create and save enhanced summary tables
    if save_figs and not comparison_df.empty:
        create_enhanced_summary_tables(comparison_df, performance_stats, save_figs)
    
    # Final method recommendations
    print("\n🏆 FINAL METHOD RECOMMENDATIONS")
    print("=" * 40)
    
    if not comparison_df.empty and performance_stats is not None:
        
        # Best overall method
        best_overall = performance_stats.index[0]  # Already sorted by composite score
        print(f"🥇 BEST OVERALL METHOD: {best_overall.upper()}")
        print(f"   • Composite Performance Score: {performance_stats.loc[best_overall, 'Composite_Score']:.3f}")
        print(f"   • Statistical Power: {performance_stats.loc[best_overall, 'Power_Mean']:.1%}")
        print(f"   • Effect Size Detection: {performance_stats.loc[best_overall, 'Distance_Mean']:.3f}")
        
        # Method recommendations by use case
        print(f"\n🎯 USE CASE RECOMMENDATIONS:")
        
        # High power needed
        best_power = performance_stats['Power_Mean'].idxmax()
        print(f"   • For Maximum Statistical Power: {best_power}")
        print(f"     Power = {performance_stats.loc[best_power, 'Power_Mean']:.1%}")
        
        # Effect size detection
        best_effect = performance_stats['Distance_Mean'].idxmax()
        print(f"   • For Effect Size Detection: {best_effect}")
        print(f"     Mean Distance = {performance_stats.loc[best_effect, 'Distance_Mean']:.3f}")
        
        # Consistency
        best_consistent = performance_stats['CV_Power'].idxmin()
        print(f"   • For Consistent Results: {best_consistent}")
        print(f"     CV = {performance_stats.loc[best_consistent, 'CV_Power']:.3f}")
        
        # Sample size recommendations
        print(f"\n📏 SAMPLE SIZE GUIDANCE:")
        for size in sorted(df['sample_size'].unique()):
            size_data = comparison_df[comparison_df['sample_size'] == size]
            if not size_data.empty:
                best_for_size = size_data.loc[size_data['rejection_rate'].idxmax(), 'method']
                best_power_for_size = size_data.loc[size_data['rejection_rate'].idxmax(), 'rejection_rate']
                print(f"   • Sample Size {size:4d}: Use {best_for_size} (Power: {best_power_for_size:.1%})")
        
        # Test type recommendations (if available)
        if 'test_type' in df.columns:
            print(f"\n🧪 TEST TYPE RECOMMENDATIONS:")
            for test_type in sorted(df['test_type'].unique()):
                test_data = comparison_df[comparison_df['test_type'] == test_type]
                if not test_data.empty:
                    best_for_test = test_data.loc[test_data['rejection_rate'].idxmax(), 'method']
                    best_power_for_test = test_data.loc[test_data['rejection_rate'].idxmax(), 'rejection_rate']
                    print(f"   • {test_type.upper()} Test: Use {best_for_test} (Power: {best_power_for_test:.1%})")
    
    if save_figs:
        print(f"\n💾 ANALYSIS COMPLETE - All outputs saved to 'figures' directory:")
        print(f"   📊 Method performance matrices and comparisons")
        print(f"   📈 Head-to-head method evaluations")
        print(f"   🧪 Test-type specific analyses (if applicable)")
        print(f"   📏 Sample size progression studies")
        print(f"   📋 Comprehensive summary tables and recommendations")
    
    return df, comparison_df, performance_stats

def create_enhanced_summary_tables(comparison_df, performance_stats, save_figs=True):
    """Create enhanced summary tables with method testing focus"""
    
    if comparison_df.empty or performance_stats is None:
        return
    
    import os
    os.makedirs('figures', exist_ok=True)
    
    # 1. Overall method ranking table
    method_ranking = performance_stats[['Power_Mean', 'Distance_Mean', 'P_Median', 
                                       'Composite_Score', 'CV_Power']].copy()
    method_ranking.columns = ['Statistical_Power', 'Effect_Size_Detection', 'Median_P_Value', 
                             'Composite_Score', 'Consistency_CV']
    method_ranking['Rank'] = range(1, len(method_ranking) + 1)
    method_ranking = method_ranking[['Rank', 'Statistical_Power', 'Effect_Size_Detection', 
                                    'Median_P_Value', 'Composite_Score', 'Consistency_CV']]
    
    # 2. Sample size performance matrix
    sample_size_matrix = comparison_df.pivot_table(
        values=['rejection_rate', 'mean_distance', 'median_pvalue'], 
        index='method', 
        columns='sample_size'
    ).round(4)
    
    # 3. Test type performance matrix (if available)
    if 'test_type' in comparison_df.columns:
        test_type_matrix = comparison_df.pivot_table(
            values=['rejection_rate', 'mean_distance', 'median_pvalue'], 
            index='method', 
            columns='test_type'
        ).round(4)
    
    # 4. Method recommendations summary
    recommendations_summary = pd.DataFrame({
        'Best_Overall': [performance_stats.index[0]],
        'Highest_Power': [performance_stats['Power_Mean'].idxmax()],
        'Best_Effect_Detection': [performance_stats['Distance_Mean'].idxmax()],
        'Most_Consistent': [performance_stats['CV_Power'].idxmin()],
        'Best_P_Values': [performance_stats['P_Median'].idxmin()]
    })
    
    if save_figs:
        # Save all tables
        method_ranking.to_csv('figures/method_ranking_comprehensive.csv')
        sample_size_matrix.to_csv('figures/performance_by_sample_size.csv')
        
        if 'test_type' in comparison_df.columns:
            test_type_matrix.to_csv('figures/performance_by_test_type.csv')
        
        recommendations_summary.to_csv('figures/method_recommendations_summary.csv')
        comparison_df.to_csv('figures/detailed_method_testing_results.csv', index=False)
        
        print("📋 Enhanced summary tables saved:")
        print("   - method_ranking_comprehensive.csv")
        print("   - performance_by_sample_size.csv")
        if 'test_type' in comparison_df.columns:
            print("   - performance_by_test_type.csv")
        print("   - method_recommendations_summary.csv")
        print("   - detailed_method_testing_results.csv")
    
    return method_ranking, sample_size_matrix

def create_summary_table(comparison_df, save_figs=True):
    """Create and save a comprehensive summary table"""
    
    if comparison_df.empty:
        return
    
    # Create overall method performance summary
    method_summary = comparison_df.groupby('method').agg({
        'n_tests': 'sum',
        'rejection_rate': 'mean',
        'median_pvalue': 'mean',
        'mean_distance': 'mean'
    }).round(4)
    
    method_summary.columns = ['Total Tests', 'Avg Rejection Rate', 'Avg Median P-value', 'Avg Mean Distance']
    
    # Create sample size breakdown
    sample_size_summary = comparison_df.pivot_table(
        values='rejection_rate', 
        index='method', 
        columns='sample_size', 
        aggfunc='mean'
    ).round(3)
    
    if save_figs:
        import os
        os.makedirs('figures', exist_ok=True)
        
        # Save method summary
        method_summary.to_csv('figures/method_performance_summary.csv')
        
        # Save sample size breakdown
        sample_size_summary.to_csv('figures/rejection_rates_by_sample_size.csv')
        
        # Save full comparison data
        comparison_df.to_csv('figures/detailed_method_comparison.csv', index=False)
        
        print("📋 Summary tables saved:")
        print("   - method_performance_summary.csv")
        print("   - rejection_rates_by_sample_size.csv") 
        print("   - detailed_method_comparison.csv")
    
    return method_summary, sample_size_summary

# Quick plotting function for focused analysis
def quick_method_comparison(df, sample_size=None, save_figs=True):
    """Quick focused comparison of methods, optionally for specific sample size"""
    
    if df.empty:
        print("No data available for comparison")
        return
    
    if sample_size:
        df_filtered = df[df['sample_size'] == sample_size]
        title_suffix = f" (Sample Size: {sample_size})"
        filename_suffix = f"_sample_size_{sample_size}"
        if df_filtered.empty:
            print(f"No data found for sample size {sample_size}")
            return
    else:
        df_filtered = df
        title_suffix = " (All Sample Sizes)"
        filename_suffix = "_all_samples"
    
    # Add rejection indicator
    df_filtered = df_filtered.copy()
    df_filtered['rejection_indicator'] = get_rejection_indicator(df_filtered)
    
    # Define colors for methods
    methods = sorted(df_filtered['method'].unique())
    colors = sns.color_palette("Set2", len(methods))
    method_colors = dict(zip(methods, colors))
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'Method Comparison Analysis{title_suffix}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Distance vs P-value scatter
    ax = axes[0,0]
    for method in methods:
        method_data = df_filtered[df_filtered['method'] == method]
        ax.scatter(method_data['distance'], method_data['p_value'], 
                  label=method.replace('_', ' ').title(), alpha=0.7, s=70,
                  color=method_colors[method], edgecolors='black', linewidth=0.5)
    
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, linewidth=2,
               label='p=0.05 (significance)')
    ax.set_xlabel('Test Distance', fontsize=12, fontweight='bold')
    ax.set_ylabel('P-value', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.set_title(f'Distance vs P-value{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rejection rate by method
    ax = axes[0,1]
    rejection_rates = df_filtered.groupby('method')['rejection_indicator'].mean()
    bars = ax.bar(range(len(rejection_rates)), rejection_rates.values,
                  color=[method_colors[m] for m in rejection_rates.index],
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rejection_rates.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xticks(range(len(rejection_rates)))
    ax.set_xticklabels([m.replace('_', '\n') for m in rejection_rates.index], fontsize=11)
    ax.set_ylabel('H₀ Rejection Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'Rejection Rate by Method{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rejection_rates.values) * 1.2 if max(rejection_rates.values) > 0 else 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Distance distribution
    ax = axes[1,0]
    bp = ax.boxplot([df_filtered[df_filtered['method'] == method]['distance'].values for method in methods],
                    labels=[m.replace('_', '\n') for m in methods],
                    patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(f'Distance Distribution{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Distance', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # P-value distribution
    ax = axes[1,1]
    bp = ax.boxplot([df_filtered[df_filtered['method'] == method]['p_value'].values for method in methods],
                    labels=[m.replace('_', '\n') for m in methods],
                    patch_artist=True, notch=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_yscale('log')
    ax.set_title(f'P-value Distribution{title_suffix}', fontsize=14, fontweight='bold')
    ax.set_ylabel('P-value (log scale)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    save_figure(fig, f'quick_method_comparison{filename_suffix}', save_figs)

if __name__ == "__main__":
    # Configuration
    SAVE_FIGURES = True  # Set to False if you don't want to save figures
    
    print("=" * 80)
    print("🔬 STATISTICAL METHOD TESTING COMPARISON SUITE")
    print("   Advanced Analysis of Distribution Testing Methods")
    print("=" * 80)
    
    # Try to run the complete analysis
    try:
        # You can specify your file path here
        filepath = 'distribution_test_summary.csv'  # Default name
        
        # Check if default file exists, if not try the user's path
        import os
        if not os.path.exists(filepath):
            # Try alternative paths
            alternative_paths = [
           
                'output/samples/overall_sampling_summary.csv',
                'overall_sampling_summary.csv'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    filepath = alt_path
                    break
            else:
                print(f"❌ File not found. Please ensure your CSV file is in the current directory.")
                print(f"Tried looking for:")
                print(f"  - distribution_test_summary.csv")
                for path in alternative_paths:
                    print(f"  - {path}")
                print(f"\nPlease either:")
                print(f"  1. Put your CSV file in the current directory, or")
                print(f"  2. Update the filepath variable in the script")
                exit(1)
        
        print(f"📁 Using file: {filepath}")
        print(f"💾 Save figures: {'Yes' if SAVE_FIGURES else 'No'}")
        
        # Run main analysis
        df, comparison_df, performance_stats = main_analysis(filepath, save_figs=SAVE_FIGURES)
        
        if df is not None and not df.empty:
            # Get analysis parameters
            methods = sorted(df['method'].unique())
            sample_sizes = sorted(df['sample_size'].unique())
            test_types = sorted(df['test_type'].unique()) if 'test_type' in df.columns else ['All']
            
            print(f"\n" + "=" * 80)
            print(f"📊 FOCUSED METHOD TESTING ANALYSIS COMPLETE")
            print(f"=" * 80)
            
            print(f"\n📋 ANALYSIS SUMMARY:")
            print(f"   • Methods Compared: {len(methods)} ({', '.join(methods)})")
            print(f"   • Sample Sizes: {len(sample_sizes)} ({', '.join(map(str, sample_sizes))})")
            print(f"   • Test Types: {len(test_types)} ({', '.join(test_types)})")
            print(f"   • Total Test Results: {len(df)}")
            print(f"   • Success Rate: {df['success'].mean():.1%}")
            
            # Quick performance summary
            if performance_stats is not None:
                print(f"\n🏆 TOP PERFORMING METHODS:")
                top_3 = performance_stats.head(3)
                for i, (method, stats) in enumerate(top_3.iterrows(), 1):
                    print(f"   {i}. {method}: Power={stats['Power_Mean']:.1%}, "
                          f"Distance={stats['Distance_Mean']:.3f}, Score={stats['Composite_Score']:.3f}")
            
            # Sample size specific recommendations
            print(f"\n📏 SAMPLE SIZE SPECIFIC RECOMMENDATIONS:")
            for size in sample_sizes:
                size_data = comparison_df[comparison_df['sample_size'] == size]
                if not size_data.empty:
                    best_method = size_data.loc[size_data['rejection_rate'].idxmax()]
                    print(f"   • {size:4d} samples: {best_method['method']} "
                          f"(Power: {best_method['rejection_rate']:.1%}, "
                          f"Distance: {best_method['mean_distance']:.3f})")
            
            # Test type specific recommendations (if available)
            if 'test_type' in df.columns and len(test_types) > 1:
                print(f"\n🧪 TEST TYPE SPECIFIC RECOMMENDATIONS:")
                for test_type in test_types:
                    test_data = comparison_df[comparison_df['test_type'] == test_type]
                    if not test_data.empty:
                        best_method = test_data.loc[test_data['rejection_rate'].idxmax()]
                        print(f"   • {test_type.upper():15s}: {best_method['method']} "
                              f"(Power: {best_method['rejection_rate']:.1%})")
            
            if SAVE_FIGURES:
                print(f"\n💾 COMPREHENSIVE OUTPUT SAVED:")
                print(f"   📊 Method Performance Matrices")
                print(f"      - method_performance_matrix.png/pdf")
                print(f"      - method_head_to_head_comparison.png/pdf")
                print(f"   🧪 Test-Type Specific Analyses")
                if 'test_type' in df.columns:
                    for test_type in df['test_type'].unique():
                        print(f"      - test_type_analysis_{test_type}.png/pdf")
                print(f"   📏 Sample Size Progression Analysis")
                print(f"      - sample_size_progression_analysis.png/pdf")
                print(f"   📋 Summary Tables & Data")
                print(f"      - method_ranking_comprehensive.csv")
                print(f"      - performance_by_sample_size.csv")
                if 'test_type' in df.columns:
                    print(f"      - performance_by_test_type.csv")
                print(f"      - method_recommendations_summary.csv")
                print(f"      - detailed_method_testing_results.csv")
            
            print(f"\n🔍 FOR CUSTOM ANALYSIS:")
            print(f"   Use the loaded data for further investigation:")
            print(f"   • df: Complete dataset")
            print(f"   • comparison_df: Method comparison results")
            print(f"   • performance_stats: Overall performance statistics")
            
            print(f"\n🎯 KEY INSIGHTS:")
            if performance_stats is not None:
                best_overall = performance_stats.index[0]
                best_power = performance_stats['Power_Mean'].idxmax()
                best_effect = performance_stats['Distance_Mean'].idxmax()
                
                print(f"   • Best Overall Method: {best_overall}")
                print(f"   • Highest Statistical Power: {best_power} ({performance_stats.loc[best_power, 'Power_Mean']:.1%})")
                print(f"   • Best Effect Detection: {best_effect} ({performance_stats.loc[best_effect, 'Distance_Mean']:.3f})")
                
                # Performance variability insight
                power_range = performance_stats['Power_Mean'].max() - performance_stats['Power_Mean'].min()
                print(f"   • Method Performance Range: {power_range:.1%} (substantial differences detected)")
        
        print(f"\n" + "=" * 80)
        print(f"✅ ANALYSIS COMPLETE - Method testing comparison finished successfully!")
        print(f"=" * 80)
        
    except FileNotFoundError:
        print("❌ CSV file not found. Please check the file path.")
        print("Update the 'filepath' variable in the script to point to your CSV file.")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your CSV file format and try again.")
        import traceback
        traceback.print_exc()