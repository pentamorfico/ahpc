#!/usr/bin/env python3
"""
Plot weak scaling results for Shallow Water GPU simulation
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot_weak_scaling(filename='weak_scaling_results.txt'):
    """Plot weak scaling efficiency"""
    # Read data
    data = pd.read_csv(filename)
    
    # Calculate efficiency (ideal time is constant for perfect weak scaling)
    ideal_time = data['Time_sec'].iloc[0]
    data['Efficiency'] = (ideal_time / data['Time_sec']) * 100
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Weak Scaling Analysis - Shallow Water Simulation', fontsize=16)
    
    # Plot 1: Runtime vs SM count
    ax1 = axes[0, 0]
    ax1.plot(data['SM_Count'], data['Time_sec'], 'o-', linewidth=2, markersize=8, label='Actual')
    ax1.axhline(y=ideal_time, color='r', linestyle='--', label=f'Ideal ({ideal_time:.3f}s)')
    ax1.set_xlabel('Number of Streaming Multiprocessors', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs SM Count')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Weak scaling efficiency
    ax2 = axes[0, 1]
    ax2.plot(data['SM_Count'], data['Efficiency'], 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(y=100, color='r', linestyle='--', label='Ideal (100%)')
    ax2.set_xlabel('Number of Streaming Multiprocessors', fontsize=12)
    ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=12)
    ax2.set_title('Weak Scaling Efficiency')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 110])
    ax2.legend()
    
    # Plot 3: Total cells vs SM count
    ax3 = axes[1, 0]
    ax3.plot(data['SM_Count'], data['Total_Cells'], 'o-', linewidth=2, markersize=8, color='purple')
    ax3.set_xlabel('Number of Streaming Multiprocessors', fontsize=12)
    ax3.set_ylabel('Total Grid Cells', fontsize=12)
    ax3.set_title('Problem Size vs SM Count')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cells per SM (should be constant)
    ax4 = axes[1, 1]
    cells_per_sm = data['Total_Cells'] / data['SM_Count']
    ax4.plot(data['SM_Count'], cells_per_sm, 'o-', linewidth=2, markersize=8, color='orange')
    avg_cells_per_sm = cells_per_sm.mean()
    ax4.axhline(y=avg_cells_per_sm, color='r', linestyle='--', 
                label=f'Average ({avg_cells_per_sm:.0f} cells/SM)')
    ax4.set_xlabel('Number of Streaming Multiprocessors', fontsize=12)
    ax4.set_ylabel('Cells per SM', fontsize=12)
    ax4.set_title('Work per SM (should be constant)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('weak_scaling_plot.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: weak_scaling_plot.png")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Weak Scaling Summary")
    print("="*60)
    print(f"Base time (2 SMs): {ideal_time:.4f}s")
    print(f"Time at 14 SMs: {data['Time_sec'].iloc[-1]:.4f}s")
    print(f"Average efficiency: {data['Efficiency'].mean():.2f}%")
    print(f"Min efficiency: {data['Efficiency'].min():.2f}% at {data.loc[data['Efficiency'].idxmin(), 'SM_Count']} SMs")
    print(f"Max efficiency: {data['Efficiency'].max():.2f}% at {data.loc[data['Efficiency'].idxmax(), 'SM_Count']} SMs")
    print("="*60)
    
    return fig

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else 'weak_scaling_results.txt'
    plot_weak_scaling(filename)
    plt.show()
