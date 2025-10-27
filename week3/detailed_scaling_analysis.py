#!/usr/bin/env python3
"""
AHPC Assignment 3 - Detailed Scaling Analysis with FFT Separation
Creates comprehensive scaling plots with error bars and parallel fraction analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import subprocess
import os
import time

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ScalingAnalysis:
    def __init__(self):
        self.strategies = ["strategy1", "strategy2", "strategy3", "strategy4", "strategy5", "strategy6"]
        self.strategy_names = ["Multiple_Regions", "Fewer_Regions", "Task_Based", "SIMD_Optimized", "Advanced_Scheduling", "Sections_Based"]
        self.thread_counts = [1, 2, 4, 8, 16, 32, 64]
        self.n_runs = 5
        self.results = []
        
    def run_single_benchmark(self, strategy, threads, nfreq, run_id):
        """Run a single benchmark and extract timing data"""
        binary_name = f"strategies/{strategy}_nfreq{nfreq}"
        
        # Build with specific NFREQ
        build_cmd = f"g++ -O3 -fopenmp -DNFREQ={nfreq} strategies/{strategy}.cpp -o {binary_name}"
        build_result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
        
        if build_result.returncode != 0:
            print(f"Build failed for {strategy} with NFREQ={nfreq}")
            return None
        
        # Set environment and run
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(threads)
        
        start_time = time.time()
        run_result = subprocess.run(f"./{binary_name}", shell=True, capture_output=True, 
                                  text=True, env=env, timeout=300)
        end_time = time.time()
        
        if run_result.returncode != 0:
            print(f"Run failed for {strategy} with {threads} threads")
            return None
        
        # Parse output for timing information
        lines = run_result.stdout.strip().split('\n')
        total_time = None
        fft_time = None
        non_fft_time = None
        checksum = None
        
        for line in lines:
            if "Elapsed time:" in line and "FFTs" not in line and "without" not in line:
                total_time = float(line.split()[-1])
            elif "Elapsed time for FFTs" in line:
                fft_time = float(line.split()[-1])
            elif "Elapsed time without FFTs" in line:
                non_fft_time = float(line.split()[-1])
            elif "Checksum" in line:
                checksum = float(line.split()[-1])
        
        # Clean up binary
        subprocess.run(f"rm -f {binary_name}", shell=True)
        
        if total_time is None:
            print(f"Could not parse timing for {strategy} with {threads} threads")
            return None
        
        # Calculate non-FFT time (use parsed value if available, otherwise calculate)
        if non_fft_time is None and total_time is not None and fft_time is not None:
            non_fft_time = total_time - fft_time
        elif non_fft_time is None:
            non_fft_time = total_time if total_time else 0
        
        return {
            'strategy': strategy,
            'strategy_name': self.strategy_names[self.strategies.index(strategy)],
            'threads': threads,
            'nfreq': nfreq,
            'run_id': run_id,
            'total_time': total_time,
            'fft_time': fft_time if fft_time else 0,
            'non_fft_time': non_fft_time,
            'checksum': checksum,
            'wall_time': end_time - start_time
        }
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmarks with multiple runs for error analysis"""
        print("Starting comprehensive scaling analysis...")
        print(f"Testing {len(self.strategies)} strategies × {len(self.thread_counts)} thread counts × {self.n_runs} runs = {len(self.strategies) * len(self.thread_counts) * self.n_runs} total experiments")
        
        experiment_count = 0
        total_experiments = len(self.strategies) * len(self.thread_counts) * self.n_runs
        
        for strategy in self.strategies:
            print(f"\n=== Testing Strategy: {strategy} ===")
            
            for threads in self.thread_counts:
                nfreq = 65536 * threads  # Weak scaling: nfreq = 65536 * ncore
                print(f"  {threads} threads (NFREQ={nfreq}): ", end="", flush=True)
                
                for run_id in range(self.n_runs):
                    experiment_count += 1
                    print(f"{run_id+1}", end="", flush=True)
                    
                    result = self.run_single_benchmark(strategy, threads, nfreq, run_id)
                    if result:
                        self.results.append(result)
                    
                    if run_id < self.n_runs - 1:
                        print(".", end="", flush=True)
                    
                    # Progress indicator
                    if experiment_count % 10 == 0:
                        progress = (experiment_count / total_experiments) * 100
                        print(f" [{progress:.1f}%]", end="", flush=True)
                
                print(" ✓")
        
        print(f"\nCompleted {len(self.results)} successful experiments")
        
        # Save raw results
        df = pd.DataFrame(self.results)
        df.to_csv('results/data/detailed_scaling_raw.csv', index=False)
        print("Raw results saved to results/data/detailed_scaling_raw.csv")
        
        return df
    
    def calculate_statistics(self, df):
        """Calculate mean and standard error for each configuration"""
        stats = []
        
        for strategy in self.strategies:
            strategy_name = self.strategy_names[self.strategies.index(strategy)]
            
            for threads in self.thread_counts:
                data = df[(df['strategy'] == strategy) & (df['threads'] == threads)]
                
                if len(data) > 0:
                    stats.append({
                        'strategy': strategy,
                        'strategy_name': strategy_name,
                        'threads': threads,
                        'nfreq': data['nfreq'].iloc[0],
                        'total_time_mean': data['total_time'].mean(),
                        'total_time_std': data['total_time'].std(),
                        'total_time_sem': data['total_time'].sem(),
                        'fft_time_mean': data['fft_time'].mean(),
                        'fft_time_std': data['fft_time'].std(),
                        'fft_time_sem': data['fft_time'].sem(),
                        'non_fft_time_mean': data['non_fft_time'].mean(),
                        'non_fft_time_std': data['non_fft_time'].std(),
                        'non_fft_time_sem': data['non_fft_time'].sem(),
                        'n_samples': len(data),
                        'checksum_mean': data['checksum'].mean()
                    })
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv('results/data/detailed_scaling_stats.csv', index=False)
        print("Statistics saved to results/data/detailed_scaling_stats.csv")
        
        return stats_df
    
    def create_normalized_plots(self, stats_df):
        """Create the three required normalized plots with error bars"""
        
        # Calculate normalization factors (1-thread performance for each strategy)
        normalization = {}
        for strategy in self.strategies:
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            baseline = strategy_data[strategy_data['threads'] == 1].iloc[0]
            normalization[strategy] = {
                'total_time': baseline['total_time_mean'],
                'fft_time': baseline['fft_time_mean'],
                'non_fft_time': baseline['non_fft_time_mean']
            }
        
        # Create the three plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot (a): Non-FFT component normalized
        for i, strategy in enumerate(self.strategies):
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            strategy_name = self.strategy_names[i]
            
            # Normalize to 1-thread performance
            normalized_time = strategy_data['non_fft_time_mean'] / normalization[strategy]['non_fft_time']
            normalized_error = strategy_data['non_fft_time_sem'] / normalization[strategy]['non_fft_time']
            
            axes[0].errorbar(strategy_data['threads'], normalized_time, 
                           yerr=normalized_error, marker='o', linewidth=2, 
                           markersize=6, label=f'{i+1}: {strategy_name}', capsize=4)
        
        axes[0].set_xlabel('Number of Threads')
        axes[0].set_ylabel('Normalized Time (1-thread = 1.0)')
        axes[0].set_title('(a) Program Excluding FFT\n(Normalized to 1 thread)')
        axes[0].set_xscale('log', base=2)
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].set_xticks(self.thread_counts)
        axes[0].set_xticklabels([str(t) for t in self.thread_counts])
        
        # Plot (b): FFT component normalized
        for i, strategy in enumerate(self.strategies):
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            strategy_name = self.strategy_names[i]
            
            # Normalize to 1-thread performance
            normalized_time = strategy_data['fft_time_mean'] / normalization[strategy]['fft_time']
            normalized_error = strategy_data['fft_time_sem'] / normalization[strategy]['fft_time']
            
            axes[1].errorbar(strategy_data['threads'], normalized_time, 
                           yerr=normalized_error, marker='s', linewidth=2, 
                           markersize=6, label=f'{i+1}: {strategy_name}', capsize=4)
        
        axes[1].set_xlabel('Number of Threads')
        axes[1].set_ylabel('Normalized Time (1-thread = 1.0)')
        axes[1].set_title('(b) FFT Component Only\n(Normalized to 1 thread)')
        axes[1].set_xscale('log', base=2)
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].set_xticks(self.thread_counts)
        axes[1].set_xticklabels([str(t) for t in self.thread_counts])
        
        # Plot (c): Total program normalized
        for i, strategy in enumerate(self.strategies):
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            strategy_name = self.strategy_names[i]
            
            # Normalize to 1-thread performance
            normalized_time = strategy_data['total_time_mean'] / normalization[strategy]['total_time']
            normalized_error = strategy_data['total_time_sem'] / normalization[strategy]['total_time']
            
            axes[2].errorbar(strategy_data['threads'], normalized_time, 
                           yerr=normalized_error, marker='^', linewidth=2, 
                           markersize=6, label=f'{i+1}: {strategy_name}', capsize=4)
        
        axes[2].set_xlabel('Number of Threads')
        axes[2].set_ylabel('Normalized Time (1-thread = 1.0)')
        axes[2].set_title('(c) Whole Program\n(Normalized to 1 thread)')
        axes[2].set_xscale('log', base=2)
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].set_xticks(self.thread_counts)
        axes[2].set_xticklabels([str(t) for t in self.thread_counts])
        
        plt.tight_layout()
        plt.savefig('results/plots/detailed_normalized_scaling.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return normalization
    
    def fit_scaling_laws(self, stats_df):
        """Fit scaling laws and compute parallel fractions"""
        
        def amdahl_law(p, f_parallel):
            """Amdahl's law: speedup = 1 / (f_serial + f_parallel/p)"""
            f_serial = 1 - f_parallel
            return 1 / (f_serial + f_parallel / p)
        
        def fft_scaling_law(p, f_parallel, n_base):
            """FFT scaling law accounting for O(N log N) complexity"""
            # For FFT: work scales as N log N, where N = n_base * p
            work_scaling = (n_base * p) * np.log2(n_base * p) / (n_base * np.log2(n_base))
            f_serial = 1 - f_parallel
            return work_scaling / (f_serial * work_scaling + f_parallel * p)
        
        scaling_results = []
        
        for strategy in self.strategies:
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            strategy_name = self.strategy_names[self.strategies.index(strategy)]
            
            threads = strategy_data['threads'].values
            
            # Non-FFT component fitting (standard Amdahl's law)
            baseline_non_fft = strategy_data[strategy_data['threads'] == 1]['non_fft_time_mean'].iloc[0]
            speedup_non_fft = baseline_non_fft / strategy_data['non_fft_time_mean'].values
            
            try:
                popt_non_fft, _ = curve_fit(amdahl_law, threads, speedup_non_fft, 
                                          bounds=[0, 1], maxfev=10000)
                f_parallel_non_fft = popt_non_fft[0]
            except:
                f_parallel_non_fft = np.nan
            
            # FFT component fitting (O(N log N) scaling)
            baseline_fft = strategy_data[strategy_data['threads'] == 1]['fft_time_mean'].iloc[0]
            if baseline_fft > 0:
                speedup_fft = baseline_fft / strategy_data['fft_time_mean'].values
                
                try:
                    # Use 65536 as base problem size
                    popt_fft, _ = curve_fit(lambda p, f: fft_scaling_law(p, f, 65536), 
                                          threads, speedup_fft, bounds=[0, 1], maxfev=10000)
                    f_parallel_fft = popt_fft[0]
                except:
                    f_parallel_fft = np.nan
            else:
                f_parallel_fft = np.nan
            
            # Total program fitting
            baseline_total = strategy_data[strategy_data['threads'] == 1]['total_time_mean'].iloc[0]
            speedup_total = baseline_total / strategy_data['total_time_mean'].values
            
            try:
                popt_total, _ = curve_fit(amdahl_law, threads, speedup_total, 
                                        bounds=[0, 1], maxfev=10000)
                f_parallel_total = popt_total[0]
            except:
                f_parallel_total = np.nan
            
            scaling_results.append({
                'strategy': strategy,
                'strategy_name': strategy_name,
                'f_parallel_non_fft': f_parallel_non_fft,
                'f_parallel_fft': f_parallel_fft,
                'f_parallel_total': f_parallel_total,
                'max_speedup_non_fft': np.max(speedup_non_fft),
                'max_speedup_fft': np.max(speedup_fft) if baseline_fft > 0 else np.nan,
                'max_speedup_total': np.max(speedup_total)
            })
        
        scaling_df = pd.DataFrame(scaling_results)
        scaling_df.to_csv('results/data/parallel_fractions.csv', index=False)
        print("Parallel fractions saved to results/data/parallel_fractions.csv")
        
        return scaling_df
    
    def create_analysis_report(self, stats_df, scaling_df):
        """Generate comprehensive analysis report"""
        
        report = """
# Detailed Scaling Analysis Report

## Experimental Setup

- **Strategies Tested**: 6 OpenMP parallelization approaches
- **Thread Counts**: 1, 2, 4, 8, 16, 32, 64
- **Problem Scaling**: nfreq = 65536 × ncore (weak scaling)
- **Repetitions**: 5 runs per configuration for statistical accuracy
- **Error Analysis**: Standard error of the mean (SEM) calculated

## Parallel Fraction Analysis

### Methodology

1. **Non-FFT Component**: Fitted using standard Amdahl's law
   - Speedup = 1 / (f_serial + f_parallel/p)
   - Where f_serial = 1 - f_parallel

2. **FFT Component**: Fitted using modified scaling law accounting for O(N log N) complexity
   - Work scaling = (N_base × p) × log₂(N_base × p) / (N_base × log₂(N_base))
   - Speedup = work_scaling / (f_serial × work_scaling + f_parallel × p)

### Results

| Strategy | Non-FFT f_parallel | FFT f_parallel | Total f_parallel | Max Speedup (Total) |
|----------|--------------------|-----------------|--------------------|---------------------|
"""
        
        for _, row in scaling_df.iterrows():
            report += f"| {row['strategy_name']} | {row['f_parallel_non_fft']:.3f} | {row['f_parallel_fft']:.3f} | {row['f_parallel_total']:.3f} | {row['max_speedup_total']:.2f}x |\n"
        
        report += """

## Key Findings

### Non-FFT Component Scaling
- Shows good parallelization potential across all strategies
- Strategy differences become apparent in parallel fraction values

### FFT Component Scaling  
- Limited by O(N log N) algorithmic complexity
- Additional overhead from weak scaling (larger problem sizes)

### Overall Program Performance
- Dominated by the component with lower parallel fraction
- Reveals strategy-specific optimization effectiveness

## Interpretation

The analysis reveals that different OpenMP strategies have varying effectiveness
in parallelizing the computation-intensive portions of the seismogram simulation.
The separation of FFT and non-FFT components provides insights into which
algorithmic parts benefit most from each parallelization approach.

"""
        
        with open('results/reports/detailed_scaling_analysis.md', 'w') as f:
            f.write(report)
        
        print("Analysis report saved to results/reports/detailed_scaling_analysis.md")

def main():
    """Main function to run comprehensive scaling analysis"""
    
    # Create necessary directories
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    
    # Initialize analysis
    analyzer = ScalingAnalysis()
    
    print("AHPC Assignment 3 - Detailed Scaling Analysis")
    print("=" * 50)
    
    # Run comprehensive benchmarks
    raw_df = analyzer.run_comprehensive_benchmark()
    
    # Calculate statistics
    stats_df = analyzer.calculate_statistics(raw_df)
    
    # Create normalized plots
    normalization = analyzer.create_normalized_plots(stats_df)
    
    # Fit scaling laws and compute parallel fractions
    scaling_df = analyzer.fit_scaling_laws(stats_df)
    
    # Generate analysis report
    analyzer.create_analysis_report(stats_df, scaling_df)
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")
    print("Generated files:")
    print("- results/data/detailed_scaling_raw.csv")
    print("- results/data/detailed_scaling_stats.csv") 
    print("- results/data/parallel_fractions.csv")
    print("- results/plots/detailed_normalized_scaling.png")
    print("- results/reports/detailed_scaling_analysis.md")

if __name__ == "__main__":
    main()