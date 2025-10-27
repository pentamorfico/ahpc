#!/usr/bin/env python3
"""
AHPC Assignment 3 - Task 2: Weak Scaling Analysis Plots
Creates comprehensive plots for weak scaling efficiency analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and parse the weak scaling data"""
    data = pd.read_csv('data/weak_scaling_data.csv', 
                      names=['Strategy_ID', 'Strategy_Name', 'Threads', 'NFREQ', 'Runtime', 'Checksum', 'Efficiency'])
    return data

def plot_weak_scaling_efficiency():
    """Plot weak scaling efficiency for all strategies"""
    data = load_data()
    
    plt.figure(figsize=(12, 8))
    
    # Plot efficiency for each strategy
    for strategy_id in sorted(data['Strategy_ID'].unique()):
        strategy_data = data[data['Strategy_ID'] == strategy_id]
        strategy_name = strategy_data['Strategy_Name'].iloc[0]
        
        plt.plot(strategy_data['Threads'], strategy_data['Efficiency'], 
                'o-', linewidth=2, markersize=8, label=f'Strategy {strategy_id}: {strategy_name}')
    
    # Add ideal weak scaling line
    max_threads = data['Threads'].max()
    threads_range = data['Threads'].unique()
    plt.plot(threads_range, [1.0] * len(threads_range), 'k--', 
             linewidth=2, alpha=0.7, label='Ideal Weak Scaling')
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Weak Scaling Efficiency', fontsize=12)
    plt.title('Weak Scaling Efficiency: OpenMP Strategies\n(Problem Size: NFREQ = 65536 × Threads)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(threads_range)
    plt.gca().set_xticklabels([str(t) for t in threads_range])
    plt.ylim(0.05, 1.2)
    
    plt.tight_layout()
    plt.savefig('plots/weak_scaling_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_runtime_vs_threads():
    """Plot runtime vs threads for weak scaling analysis"""
    data = load_data()
    
    plt.figure(figsize=(12, 8))
    
    # Plot runtime for each strategy
    for strategy_id in sorted(data['Strategy_ID'].unique()):
        strategy_data = data[data['Strategy_ID'] == strategy_id]
        strategy_name = strategy_data['Strategy_Name'].iloc[0]
        
        plt.plot(strategy_data['Threads'], strategy_data['Runtime'], 
                'o-', linewidth=2, markersize=8, label=f'Strategy {strategy_id}: {strategy_name}')
    
    # Add ideal weak scaling line (constant runtime)
    baseline_runtime = data[data['Threads'] == 1]['Runtime'].mean()
    threads_range = data['Threads'].unique()
    plt.plot(threads_range, [baseline_runtime] * len(threads_range), 'k--', 
             linewidth=2, alpha=0.7, label=f'Ideal Weak Scaling ({baseline_runtime:.2f}s)')
    
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Weak Scaling Runtime Analysis\n(Problem Size: NFREQ = 65536 × Threads)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.xticks(threads_range)
    plt.gca().set_xticklabels([str(t) for t in threads_range])
    
    plt.tight_layout()
    plt.savefig('plots/weak_scaling_runtime.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_heatmap():
    """Create a heatmap of efficiency values"""
    data = load_data()
    
    # Pivot data for heatmap
    pivot_data = data.pivot(index='Strategy_Name', columns='Threads', values='Efficiency')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Weak Scaling Efficiency'},
                vmin=0, vmax=1.0)
    
    plt.title('Weak Scaling Efficiency Heatmap\n(Problem Size: NFREQ = 65536 × Threads)', fontsize=14)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('OpenMP Strategy', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('plots/weak_scaling_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_table():
    """Generate summary statistics table"""
    data = load_data()
    
    summary = []
    for strategy_id in sorted(data['Strategy_ID'].unique()):
        strategy_data = data[data['Strategy_ID'] == strategy_id]
        strategy_name = strategy_data['Strategy_Name'].iloc[0]
        
        # Calculate metrics
        max_threads = strategy_data['Threads'].max()
        final_efficiency = strategy_data[strategy_data['Threads'] == max_threads]['Efficiency'].iloc[0]
        avg_efficiency = strategy_data['Efficiency'].mean()
        runtime_increase = strategy_data['Runtime'].iloc[-1] / strategy_data['Runtime'].iloc[0]
        
        summary.append({
            'Strategy': f"{strategy_id}: {strategy_name}",
            'Max_Threads_Tested': max_threads,
            'Final_Efficiency': final_efficiency,
            'Average_Efficiency': avg_efficiency,
            'Runtime_Increase_Factor': runtime_increase
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n=== Weak Scaling Summary Statistics ===")
    print(summary_df.to_string(index=False))
    
    # Save to CSV
    summary_df.to_csv('data/weak_scaling_summary.csv', index=False)
    
    return summary_df

def main():
    """Main function to generate all plots and analysis"""
    print("Generating weak scaling analysis plots...")
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Generate all plots
    plot_weak_scaling_efficiency()
    plot_runtime_vs_threads()
    plot_efficiency_heatmap()
    
    # Generate summary
    summary = generate_summary_table()
    
    print(f"\nPlots saved to: plots/")
    print(f"Data saved to: data/")
    print(f"\nAnalysis complete!")

if __name__ == "__main__":
    main()