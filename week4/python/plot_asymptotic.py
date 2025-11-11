#!/usr/bin/env python3
"""
Plot asymptotic performance results for Shallow Water GPU simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_asymptotic_performance(filename='asymptotic_performance_results.txt'):
    """Plot asymptotic performance analysis"""
    # Read data
    data = pd.read_csv(filename)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Asymptotic Performance Analysis - GPU Shallow Water', fontsize=16)
    
    # Plot 1: ns per cell vs total cells (main result)
    ax1 = axes[0, 0]
    ax1.plot(data['Total_Cells'], data['ns_per_cell'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Total Grid Cells', fontsize=12)
    ax1.set_ylabel('Nanoseconds per Cell per Iteration', fontsize=12)
    ax1.set_title('Performance vs Problem Size')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Add horizontal line at asymptotic value (last few points)
    asymptotic_value = data['ns_per_cell'].iloc[-3:].mean()
    ax1.axhline(y=asymptotic_value, color='r', linestyle='--', 
                label=f'Asymptotic: {asymptotic_value:.2f} ns/cell')
    ax1.legend()
    
    # Plot 2: Total time vs total cells
    ax2 = axes[0, 1]
    ax2.plot(data['Total_Cells'], data['Time_sec'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Total Grid Cells', fontsize=12)
    ax2.set_ylabel('Total Execution Time (seconds)', fontsize=12)
    ax2.set_title('Execution Time vs Problem Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Efficiency (relative to largest problem)
    ax3 = axes[1, 0]
    best_ns_per_cell = data['ns_per_cell'].min()
    efficiency = (best_ns_per_cell / data['ns_per_cell']) * 100
    ax3.plot(data['Total_Cells'], efficiency, 'o-', linewidth=2, markersize=8, color='purple')
    ax3.axhline(y=100, color='r', linestyle='--', label='Peak efficiency')
    ax3.set_xlabel('Total Grid Cells', fontsize=12)
    ax3.set_ylabel('Relative Efficiency (%)', fontsize=12)
    ax3.set_title('GPU Utilization Efficiency')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend()
    
    # Plot 4: Speedup relative to smallest problem
    ax4 = axes[1, 1]
    baseline_time = data['Time_sec'].iloc[0]
    baseline_cells = data['Total_Cells'].iloc[0]
    # Ideal speedup: linear with problem size
    ideal_speedup = data['Total_Cells'] / baseline_cells
    actual_speedup = (baseline_time * data['Total_Cells'] / baseline_cells) / data['Time_sec']
    
    ax4.plot(data['Total_Cells'], actual_speedup, 'o-', linewidth=2, markersize=8, 
             color='orange', label='Actual')
    ax4.plot(data['Total_Cells'], ideal_speedup, '--', linewidth=2, 
             color='red', label='Ideal (Linear)')
    ax4.set_xlabel('Total Grid Cells', fontsize=12)
    ax4.set_ylabel('Speedup Factor', fontsize=12)
    ax4.set_title('Scalability: Actual vs Ideal')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('asymptotic_performance_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: asymptotic_performance_plot.png")
    
    # Find the "knee" - where performance starts to stabilize
    # Use derivative to find where rate of change slows
    ns_diff = np.diff(data['ns_per_cell'].values)
    pct_change = np.abs(ns_diff / data['ns_per_cell'].values[:-1]) * 100
    
    # Find first point where change is < 5%
    stable_threshold = 5.0
    stable_indices = np.where(pct_change < stable_threshold)[0]
    
    if len(stable_indices) > 0:
        knee_idx = stable_indices[0] + 1  # +1 because diff shortens array
        knee_cells = data['Total_Cells'].iloc[knee_idx]
        knee_grid = data['Grid_NX'].iloc[knee_idx]
        knee_ns = data['ns_per_cell'].iloc[knee_idx]
    else:
        knee_idx = len(data) // 2
        knee_cells = data['Total_Cells'].iloc[knee_idx]
        knee_grid = data['Grid_NX'].iloc[knee_idx]
        knee_ns = data['ns_per_cell'].iloc[knee_idx]
    
    # Print summary
    print("\n" + "="*70)
    print("Asymptotic Performance Summary")
    print("="*70)
    print(f"Smallest problem: {data['Total_Cells'].iloc[0]:,} cells ({data['Grid_NX'].iloc[0]}x{data['Grid_NY'].iloc[0]})")
    print(f"  Performance: {data['ns_per_cell'].iloc[0]:.2f} ns/cell")
    print(f"\nLargest problem: {data['Total_Cells'].iloc[-1]:,} cells ({data['Grid_NX'].iloc[-1]}x{data['Grid_NY'].iloc[-1]})")
    print(f"  Performance: {data['ns_per_cell'].iloc[-1]:.2f} ns/cell")
    print(f"\nAsymptotic performance: {asymptotic_value:.2f} ns/cell")
    print(f"  (average of 3 largest problems)")
    print(f"\nRecommended minimum grid size: {knee_grid}x{knee_grid} ({knee_cells:,} cells)")
    print(f"  At this size, performance is within {stable_threshold}% of asymptotic value")
    print(f"  Performance: {knee_ns:.2f} ns/cell")
    print(f"\nPerformance improvement from smallest to largest: {data['ns_per_cell'].iloc[0] / data['ns_per_cell'].iloc[-1]:.2f}x")
    print("="*70)
    
    return fig

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else 'asymptotic_performance_results.txt'
    plot_asymptotic_performance(filename)
    plt.show()
