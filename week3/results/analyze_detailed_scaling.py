#!/usr/bin/env python3
"""
AHPC Assignment 3 - Detailed Weak Scaling Analysis
Generates normalized scaling plots with error bars and computes parallel fractions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import os

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_detailed_data():
    """Load and process the detailed weak scaling CSV data"""
    data_file = 'data/detailed_weak_scaling.csv'
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found. Run SLURM job first.")
    
    df = pd.read_csv(data_file)
    
    # Filter out failed runs (NA values)
    df = df.dropna(subset=['FFT_time', 'NonFFT_time', 'Total_time'])
    
    # Convert to numeric
    numeric_cols = ['FFT_time', 'NonFFT_time', 'Total_time', 'Checksum']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove obvious outliers (>3 std from mean per group)
    def remove_outliers(group):
        for col in ['FFT_time', 'NonFFT_time', 'Total_time']:
            mean = group[col].mean()
            std = group[col].std()
            group = group[abs(group[col] - mean) <= 3 * std]
        return group
    
    df = df.groupby(['Strategy', 'Threads']).apply(remove_outliers).reset_index(drop=True)
    
    return df

def compute_statistics(df):
    """Compute mean and standard error for each configuration"""
    stats_df = df.groupby(['Strategy', 'Strategy_Name', 'Threads', 'NFREQ']).agg({
        'FFT_time': ['mean', 'std', 'count'],
        'NonFFT_time': ['mean', 'std', 'count'], 
        'Total_time': ['mean', 'std', 'count'],
        'Checksum': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = ['_'.join(col).strip('_') for col in stats_df.columns]
    
    # Compute standard error
    stats_df['FFT_time_se'] = stats_df['FFT_time_std'] / np.sqrt(stats_df['FFT_time_count'])
    stats_df['NonFFT_time_se'] = stats_df['NonFFT_time_std'] / np.sqrt(stats_df['NonFFT_time_count'])
    stats_df['Total_time_se'] = stats_df['Total_time_std'] / np.sqrt(stats_df['Total_time_count'])
    
    return stats_df

def normalize_to_single_thread(stats_df):
    """Normalize all times to single thread baseline"""
    normalized_df = stats_df.copy()
    
    for strategy in stats_df['Strategy'].unique():
        strategy_data = stats_df[stats_df['Strategy'] == strategy]
        
        # Get single-thread baseline
        baseline = strategy_data[strategy_data['Threads'] == 1]
        if len(baseline) == 0:
            print(f"Warning: No single-thread data for strategy {strategy}")
            continue
            
        fft_baseline = baseline['FFT_time_mean'].iloc[0]
        nonfft_baseline = baseline['NonFFT_time_mean'].iloc[0]
        total_baseline = baseline['Total_time_mean'].iloc[0]
        
        # Normalize means
        mask = normalized_df['Strategy'] == strategy
        normalized_df.loc[mask, 'FFT_time_norm'] = stats_df.loc[mask, 'FFT_time_mean'] / fft_baseline
        normalized_df.loc[mask, 'NonFFT_time_norm'] = stats_df.loc[mask, 'NonFFT_time_mean'] / nonfft_baseline
        normalized_df.loc[mask, 'Total_time_norm'] = stats_df.loc[mask, 'Total_time_mean'] / total_baseline
        
        # Normalize standard errors (error propagation)
        normalized_df.loc[mask, 'FFT_time_norm_se'] = stats_df.loc[mask, 'FFT_time_se'] / fft_baseline
        normalized_df.loc[mask, 'NonFFT_time_norm_se'] = stats_df.loc[mask, 'NonFFT_time_se'] / nonfft_baseline
        normalized_df.loc[mask, 'Total_time_norm_se'] = stats_df.loc[mask, 'Total_time_se'] / total_baseline
    
    return normalized_df

def plot_normalized_scaling(normalized_df, component, title, filename):
    """Create a normalized scaling plot with error bars"""
    plt.figure(figsize=(12, 8))
    
    strategies = sorted(normalized_df['Strategy'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        data = normalized_df[normalized_df['Strategy'] == strategy].sort_values('Threads')
        strategy_name = data['Strategy_Name'].iloc[0]
        
        plt.errorbar(data['Threads'], data[f'{component}_norm'], 
                    yerr=data[f'{component}_norm_se'],
                    marker='o', linewidth=2, markersize=8, capsize=5,
                    label=f'Strategy {strategy}: {strategy_name}',
                    color=colors[i])
    
    # Add ideal weak scaling line (constant normalized time = 1.0)
    threads_range = sorted(normalized_df['Threads'].unique())
    ideal_line = [1.0 for t in threads_range]  # Ideal weak scaling: constant time
    plt.plot(threads_range, ideal_line, 'k--', linewidth=2, alpha=0.7, label='Ideal Weak Scaling')
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Normalized Time (T₁/Tₚ)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.xticks(threads_range, [str(t) for t in threads_range])
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

def amdahl_law(p, f):
    """Amdahl's law: speedup = 1 / (f + (1-f)/p) where f is serial fraction"""
    return 1.0 / (f + (1.0 - f) / p)

def gustafson_law(p, f):
    """Gustafson's law: speedup = f + p*(1-f) where f is serial fraction"""
    return f + p * (1.0 - f)

def fft_scaling_model(n, a, b):
    """FFT scaling model: time = a * n * log2(n) + b"""
    return a * n * np.log2(n) + b

def fit_parallel_fraction(normalized_df, component):
    """Fit parallel fraction using Amdahl's law"""
    results = {}
    
    for strategy in sorted(normalized_df['Strategy'].unique()):
        data = normalized_df[normalized_df['Strategy'] == strategy].sort_values('Threads')
        strategy_name = data['Strategy_Name'].iloc[0]
        
        threads = data['Threads'].values
        # Convert normalized time to speedup
        speedup = 1.0 / data[f'{component}_norm'].values
        speedup_err = data[f'{component}_norm_se'].values / (data[f'{component}_norm'].values ** 2)
        
        try:
            # Fit Amdahl's law: speedup = 1 / (f + (1-f)/p)
            def amdahl_fit(p, f):
                return 1.0 / (f + (1.0 - f) / p)
            
            popt, pcov = curve_fit(amdahl_fit, threads, speedup, 
                                 bounds=([0], [1]), sigma=speedup_err)
            serial_fraction = popt[0]
            serial_fraction_err = np.sqrt(pcov[0, 0])
            
            # Compute R-squared
            ss_res = np.sum((speedup - amdahl_fit(threads, *popt)) ** 2)
            ss_tot = np.sum((speedup - np.mean(speedup)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            results[strategy] = {
                'strategy_name': strategy_name,
                'serial_fraction': serial_fraction,
                'serial_fraction_err': serial_fraction_err,
                'parallel_fraction': 1 - serial_fraction,
                'parallel_fraction_err': serial_fraction_err,
                'r_squared': r_squared
            }
            
        except Exception as e:
            print(f"Fitting failed for strategy {strategy}: {e}")
            results[strategy] = {
                'strategy_name': strategy_name,
                'serial_fraction': np.nan,
                'serial_fraction_err': np.nan,
                'parallel_fraction': np.nan,
                'parallel_fraction_err': np.nan,
                'r_squared': np.nan
            }
    
    return results

def plot_parallel_fractions(nonfft_results, fft_results):
    """Plot parallel fractions comparison"""
    plt.figure(figsize=(12, 6))
    
    strategies = sorted(nonfft_results.keys())
    x_pos = np.arange(len(strategies))
    
    nonfft_fractions = [nonfft_results[s]['parallel_fraction'] for s in strategies]
    nonfft_errors = [nonfft_results[s]['parallel_fraction_err'] for s in strategies]
    fft_fractions = [fft_results[s]['parallel_fraction'] for s in strategies]
    fft_errors = [fft_results[s]['parallel_fraction_err'] for s in strategies]
    
    width = 0.35
    
    plt.bar(x_pos - width/2, nonfft_fractions, width, yerr=nonfft_errors,
            label='Non-FFT Component', alpha=0.8, capsize=5)
    plt.bar(x_pos + width/2, fft_fractions, width, yerr=fft_errors,
            label='FFT Component', alpha=0.8, capsize=5)
    
    plt.xlabel('OpenMP Strategy', fontsize=12)
    plt.ylabel('Parallel Fraction', fontsize=12)
    plt.title('Parallel Fractions by Component and Strategy', fontsize=14)
    plt.xticks(x_pos, [f"S{s}" for s in strategies])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('plots/parallel_fractions.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_tables(stats_df, nonfft_results, fft_results):
    """Generate summary tables for the report"""
    
    # Runtime summary
    runtime_summary = stats_df[stats_df['Threads'] == 1][['Strategy', 'Strategy_Name', 
                                                         'FFT_time_mean', 'NonFFT_time_mean', 
                                                         'Total_time_mean']].copy()
    runtime_summary.columns = ['Strategy', 'Strategy_Name', 'FFT_Time_1T', 'NonFFT_Time_1T', 'Total_Time_1T']
    
    # Add max thread performance
    max_thread_data = stats_df[stats_df['Threads'] == 64][['Strategy', 'FFT_time_mean', 
                                                          'NonFFT_time_mean', 'Total_time_mean']]
    max_thread_data.columns = ['Strategy', 'FFT_Time_64T', 'NonFFT_Time_64T', 'Total_Time_64T']
    
    runtime_summary = runtime_summary.merge(max_thread_data, on='Strategy')
    
    # Add parallel fractions
    pf_data = []
    for strategy in sorted(nonfft_results.keys()):
        pf_data.append({
            'Strategy': strategy,
            'NonFFT_Parallel_Fraction': nonfft_results[strategy]['parallel_fraction'],
            'FFT_Parallel_Fraction': fft_results[strategy]['parallel_fraction'],
            'NonFFT_R_Squared': nonfft_results[strategy]['r_squared'],
            'FFT_R_Squared': fft_results[strategy]['r_squared']
        })
    
    pf_df = pd.DataFrame(pf_data)
    runtime_summary = runtime_summary.merge(pf_df, on='Strategy')
    
    # Save tables
    runtime_summary.to_csv('data/runtime_summary.csv', index=False)
    
    print("\n=== Runtime Summary (Single Thread Baseline) ===")
    print(runtime_summary[['Strategy_Name', 'FFT_Time_1T', 'NonFFT_Time_1T', 'Total_Time_1T']].to_string(index=False))
    
    print("\n=== Parallel Fractions (Amdahl's Law Fit) ===")
    print(runtime_summary[['Strategy_Name', 'NonFFT_Parallel_Fraction', 'FFT_Parallel_Fraction']].to_string(index=False))
    
    return runtime_summary

def main():
    """Main analysis pipeline"""
    print("Loading detailed weak scaling data...")
    
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and process data
    df = load_detailed_data()
    stats_df = compute_statistics(df)
    normalized_df = normalize_to_single_thread(stats_df)
    
    print(f"Processed {len(df)} individual measurements")
    print(f"Strategies: {sorted(df['Strategy'].unique())}")
    print(f"Thread counts: {sorted(df['Threads'].unique())}")
    
    # Generate normalized plots
    print("\nGenerating normalized scaling plots...")
    
    plot_normalized_scaling(normalized_df, 'NonFFT_time', 
                          'Normalized Non-FFT Component Scaling\n(NFREQ = 65536 × Threads)', 
                          'normalized_nonfft_scaling.png')
    
    plot_normalized_scaling(normalized_df, 'FFT_time',
                          'Normalized FFT Component Scaling\n(NFREQ = 65536 × Threads)',
                          'normalized_fft_scaling.png')
    
    plot_normalized_scaling(normalized_df, 'Total_time',
                          'Normalized Total Runtime Scaling\n(NFREQ = 65536 × Threads)',
                          'normalized_total_scaling.png')
    
    # Compute parallel fractions
    print("\nFitting parallel fractions...")
    nonfft_results = fit_parallel_fraction(normalized_df, 'NonFFT_time')
    fft_results = fit_parallel_fraction(normalized_df, 'FFT_time')
    
    # Plot parallel fractions
    plot_parallel_fractions(nonfft_results, fft_results)
    
    # Generate summary tables
    summary = generate_summary_tables(stats_df, nonfft_results, fft_results)
    
    print(f"\nAnalysis complete! Results saved to plots/ and data/")
    
    return summary, nonfft_results, fft_results

if __name__ == "__main__":
    summary, nonfft_results, fft_results = main()