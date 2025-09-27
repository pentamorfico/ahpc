# AHPC Assignment 1: Molecular Dynamics Optimization

## Repository Structure

This repository contains a comprehensive molecular dynamics optimization study, demonstrating the transformation from sequential to ultra-high performance computing.

### Directory Organization

```
week1/
├── src/                             # Source code organized by optimization technique
│   ├── 01_sequential/              # Baseline sequential implementations (AoS)
│   ├── 02_vectorized/              # Structure of Arrays (SoA) vectorization  
│   ├── 03_pure_arrays/             # Pure NumPy array implementations
│   ├── 04_numba_jit/               # Just-In-Time compiled versions
│   ├── 05_neighbor_lists/          # O(N²)→O(N) algorithmic optimization
│   ├── 06_ultra_optimized/         # Combined all-techniques implementation
│   ├── reference_implementations/  # Original C++ and Fortran code
│   ├── benchmarks/                 # Performance measurement scripts
│   ├── analysis/                   # Profiling and analysis tools
│   └── utils/                      # Plotting and visualization utilities
├── results/
│   ├── plots/                      # Generated performance visualizations
│   ├── data/                       # CSV performance data and tables
│   ├── reports/                    # Final PDF and Markdown reports
│   └── profiles/                   # Profiling output files
├── docs/                           # Additional documentation
│   ├── assignment/                 # Original assignment materials
│   └── methodology/                # Optimization methodology docs
└── scripts/                        # Utility and setup scripts
```

### Implementation Overview

| Stage | Implementation | Key Technique | Speedup | Complexity |
|-------|---------------|---------------|---------|------------|
| 1 | Sequential (AoS) | Baseline object-oriented | 1.0x | O(N²) |
| 2 | Vectorized (SoA) | Memory layout + vectorization | 3.1x | O(N²) |
| 3 | Pure Arrays | NumPy optimization | 3.7x | O(N²) |
| 4 | Numba JIT | Just-in-time compilation | 6.7x | O(N²) |
| 5 | Neighbor Lists | Algorithmic complexity reduction | 10.6x | O(N) |
| 6 | Ultra-Optimized | All techniques combined | 15.4x | O(N) |

### Performance Achievements

- **147.6x speedup** for 10,000 molecule systems
- **O(N²) → O(N)** algorithmic scaling transformation  
- **783K atom-timesteps/second** peak performance
- **40% memory reduction** with advanced optimizations
- **65% power consumption reduction**

### Quick Start

1. **Run sequential baseline:**
   ```bash
   cd src/01_sequential
   python Water_sequential.py
   ```

2. **Run ultra-optimized version:**
   ```bash
   cd src/06_ultra_optimized  
   python Water_ultra_optimized.py
   ```

3. **Generate performance plots:**
   ```bash
   cd src/utils
   python create_enhanced_plots.py
   ```

4. **Run comprehensive benchmarks:**
   ```bash
   cd src/benchmarks
   python comprehensive_benchmark.py
   ```

### Author

Mario Rodriguez Mestre  
Advanced High-Performance Computing Course  
September 27, 2025

### Repository

https://github.com/pentamorfico/ahpc