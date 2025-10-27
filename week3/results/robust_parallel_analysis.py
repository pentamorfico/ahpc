#!/usr/bin/env python3
"""
AHPC Assignment 3 - Robust Parallel Fraction Analysis
Fixes fitting issues and computes parallel fractions using multiple approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
import os

def load_and_process_data():
    """Load the detailed weak scaling data and compute statistics"""
    df = pd.read_csv('data/detailed_weak_scaling.csv')
    
    # Remove any failed runs
    df = df.dropna(subset=['FFT_time', 'NonFFT_time', 'Total_time'])
    
    # Convert to numeric
    for col in ['FFT_time', 'NonFFT_time', 'Total_time', 'Checksum']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute statistics per configuration
    stats = df.groupby(['Strategy', 'Strategy_Name', 'Threads', 'NFREQ']).agg({
        'FFT_time': ['mean', 'std', 'count'],
        'NonFFT_time': ['mean', 'std', 'count'],
        'Total_time': ['mean', 'std', 'count'],
        'Checksum': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns]
    
    # Add standard errors
    stats['FFT_time_se'] = stats['FFT_time_std'] / np.sqrt(stats['FFT_time_count'])
    stats['NonFFT_time_se'] = stats['NonFFT_time_std'] / np.sqrt(stats['NonFFT_time_count'])
    stats['Total_time_se'] = stats['Total_time_std'] / np.sqrt(stats['Total_time_count'])
    
    return stats

def compute_efficiency_and_speedup(stats_df):
    """Compute efficiency and speedup relative to single thread"""
    result_df = stats_df.copy()
    
    for strategy in stats_df['Strategy'].unique():
        strategy_data = stats_df[stats_df['Strategy'] == strategy]
        
        # Get single-thread baseline
        baseline = strategy_data[strategy_data['Threads'] == 1]
        if len(baseline) == 0:
            continue
            
        fft_baseline = baseline['FFT_time_mean'].iloc[0]
        nonfft_baseline = baseline['NonFFT_time_mean'].iloc[0]
        total_baseline = baseline['Total_time_mean'].iloc[0]
        
        # Compute speedup (baseline_time / current_time)
        mask = result_df['Strategy'] == strategy
        result_df.loc[mask, 'FFT_speedup'] = fft_baseline / result_df.loc[mask, 'FFT_time_mean']
        result_df.loc[mask, 'NonFFT_speedup'] = nonfft_baseline / result_df.loc[mask, 'NonFFT_time_mean']
        result_df.loc[mask, 'Total_speedup'] = total_baseline / result_df.loc[mask, 'Total_time_mean']
        
        # Compute efficiency (speedup / threads)
        result_df.loc[mask, 'FFT_efficiency'] = result_df.loc[mask, 'FFT_speedup'] / result_df.loc[mask, 'Threads']
        result_df.loc[mask, 'NonFFT_efficiency'] = result_df.loc[mask, 'NonFFT_speedup'] / result_df.loc[mask, 'Threads']
        result_df.loc[mask, 'Total_efficiency'] = result_df.loc[mask, 'Total_speedup'] / result_df.loc[mask, 'Threads']
    
    return result_df

def fit_parallel_fraction_robust(data, component):
    """Robust parallel fraction estimation using multiple approaches"""
    
    def amdahl_model(p, f):
        """Amdahl's law: S(p) = 1 / (f + (1-f)/p)"""
        return 1.0 / (f + (1.0 - f) / p)
    
    def efficiency_model(p, f):
        """Efficiency model: E(p) = 1 / (p*f + (1-f))"""
        return 1.0 / (p * f + (1.0 - f))
    
    threads = data['Threads'].values
    speedup = data[f'{component}_speedup'].values
    efficiency = data[f'{component}_efficiency'].values
    
    results = {}
    
    # Method 1: Amdahl's law fitting with robust bounds
    try:
        # Use efficiency data which is more stable
        popt, pcov = curve_fit(efficiency_model, threads, efficiency, 
                             bounds=([0.0], [0.99]), 
                             p0=[0.1],
                             maxfev=5000)
        
        serial_fraction = popt[0]
        serial_fraction_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else np.nan
        
        # Compute fit quality
        predicted = efficiency_model(threads, serial_fraction)
        r_squared = 1 - np.sum((efficiency - predicted)**2) / np.sum((efficiency - np.mean(efficiency))**2)
        
        results['amdahl_fit'] = {
            'serial_fraction': serial_fraction,
            'parallel_fraction': 1 - serial_fraction,
            'serial_fraction_err': serial_fraction_err,
            'r_squared': r_squared,
            'method': 'Amdahl efficiency fit'
        }
        
    except Exception as e:
        print(f"Amdahl fitting failed: {e}")
        results['amdahl_fit'] = None
    
    # Method 2: Linear regression on log-log plot for power law
    try:
        # Fit E(p) = A * p^(-alpha) to get scaling behavior
        log_p = np.log(threads[threads > 1])
        log_e = np.log(efficiency[threads > 1])
        
        # Remove any infinite or NaN values
        valid_mask = np.isfinite(log_p) & np.isfinite(log_e)
        if np.sum(valid_mask) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_p[valid_mask], log_e[valid_mask])
            
            # Convert slope to approximate serial fraction
            # For pure parallel code: E(p) ~ 1/p (slope = -1)
            # For serial code: E(p) ~ 1 (slope = 0)
            alpha = -slope  # Negative because efficiency decreases
            approx_parallel_fraction = max(0, min(1, alpha))
            
            results['power_law'] = {
                'alpha': alpha,
                'parallel_fraction': approx_parallel_fraction,
                'serial_fraction': 1 - approx_parallel_fraction,
                'r_squared': r_value**2,
                'method': 'Power law fit'
            }
        else:
            results['power_law'] = None
            
    except Exception as e:
        print(f"Power law fitting failed: {e}")
        results['power_law'] = None
    
    # Method 3: Simple efficiency-based estimate
    try:
        # Use efficiency at maximum threads as indicator
        max_threads = threads.max()
        max_efficiency = efficiency[threads == max_threads][0]
        
        # Simple approximation: f ≈ 1 - E(p_max)
        simple_serial_fraction = 1 - max_efficiency
        simple_parallel_fraction = max_efficiency
        
        results['simple'] = {
            'serial_fraction': simple_serial_fraction,
            'parallel_fraction': simple_parallel_fraction,
            'max_threads': max_threads,
            'max_efficiency': max_efficiency,
            'method': 'Simple efficiency estimate'
        }
        
    except Exception as e:
        print(f"Simple estimation failed: {e}")
        results['simple'] = None
    
    return results

def analyze_all_strategies():
    """Analyze parallel fractions for all strategies"""
    stats_df = load_and_process_data()
    enhanced_df = compute_efficiency_and_speedup(stats_df)
    
    all_results = {}
    
    for strategy in sorted(enhanced_df['Strategy'].unique()):
        strategy_data = enhanced_df[enhanced_df['Strategy'] == strategy]
        strategy_name = strategy_data['Strategy_Name'].iloc[0]
        
        print(f"\nAnalyzing Strategy {strategy}: {strategy_name}")
        
        # Analyze both components
        nonfft_results = fit_parallel_fraction_robust(strategy_data, 'NonFFT')
        fft_results = fit_parallel_fraction_robust(strategy_data, 'FFT')
        
        all_results[strategy] = {
            'strategy_name': strategy_name,
            'nonfft': nonfft_results,
            'fft': fft_results
        }
        
        # Print summary
        print(f"  Non-FFT component:")
        for method, result in nonfft_results.items():
            if result:
                print(f"    {method}: parallel fraction = {result['parallel_fraction']:.3f}")
        
        print(f"  FFT component:")
        for method, result in fft_results.items():
            if result:
                print(f"    {method}: parallel fraction = {result['parallel_fraction']:.3f}")
    
    return all_results, enhanced_df

def create_summary_table(all_results):
    """Create a comprehensive summary table"""
    summary_data = []
    
    for strategy, results in all_results.items():
        strategy_name = results['strategy_name']
        
        # Get best estimates for each component
        nonfft_pf = None
        nonfft_method = None
        if results['nonfft']['amdahl_fit']:
            nonfft_pf = results['nonfft']['amdahl_fit']['parallel_fraction']
            nonfft_method = "Amdahl"
        elif results['nonfft']['power_law']:
            nonfft_pf = results['nonfft']['power_law']['parallel_fraction']
            nonfft_method = "Power Law"
        elif results['nonfft']['simple']:
            nonfft_pf = results['nonfft']['simple']['parallel_fraction']
            nonfft_method = "Simple"
        
        fft_pf = None
        fft_method = None
        if results['fft']['amdahl_fit']:
            fft_pf = results['fft']['amdahl_fit']['parallel_fraction']
            fft_method = "Amdahl"
        elif results['fft']['power_law']:
            fft_pf = results['fft']['power_law']['parallel_fraction']
            fft_method = "Power Law"
        elif results['fft']['simple']:
            fft_pf = results['fft']['simple']['parallel_fraction']
            fft_method = "Simple"
        
        summary_data.append({
            'Strategy': strategy,
            'Strategy_Name': strategy_name,
            'NonFFT_Parallel_Fraction': nonfft_pf,
            'NonFFT_Method': nonfft_method,
            'FFT_Parallel_Fraction': fft_pf,
            'FFT_Method': fft_method
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n=== Parallel Fraction Summary ===")
    print(summary_df[['Strategy_Name', 'NonFFT_Parallel_Fraction', 'NonFFT_Method', 
                     'FFT_Parallel_Fraction', 'FFT_Method']].to_string(index=False))
    
    summary_df.to_csv('data/parallel_fractions_summary.csv', index=False)
    
    return summary_df

def plot_scaling_analysis(enhanced_df):
    """Create scaling analysis plots"""
    
    # Plot 1: Efficiency vs Threads
    plt.figure(figsize=(15, 5))
    
    components = ['NonFFT', 'FFT', 'Total']
    for i, component in enumerate(components):
        plt.subplot(1, 3, i+1)
        
        for strategy in sorted(enhanced_df['Strategy'].unique()):
            data = enhanced_df[enhanced_df['Strategy'] == strategy]
            strategy_name = data['Strategy_Name'].iloc[0]
            
            plt.plot(data['Threads'], data[f'{component}_efficiency'], 
                    'o-', label=f'S{strategy}: {strategy_name}', linewidth=2, markersize=6)
        
        plt.xlabel('Number of Threads')
        plt.ylabel('Efficiency')
        plt.title(f'{component} Component Efficiency')
        plt.xscale('log', base=2)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis pipeline"""
    print("Performing robust parallel fraction analysis...")
    
    # Ensure directories exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Run analysis
    all_results, enhanced_df = analyze_all_strategies()
    
    # Create summary
    summary_df = create_summary_table(all_results)
    
    # Create plots
    plot_scaling_analysis(enhanced_df)
    
    print(f"\nRobust analysis complete! Results saved to data/ and plots/")
    
    return all_results, summary_df, enhanced_df

if __name__ == "__main__":
    all_results, summary_df, enhanced_df = main()