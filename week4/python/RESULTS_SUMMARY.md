# Assignment 4: Shallow Water GPU Parallelization - Results Summary

**Student:** [Your Name]  
**Date:** November 11, 2025  
**GPU:** NVIDIA RTX A6000 (49GB, Compute Capability 8.6, 84 SMs)  
**Implementation:** CuPy (Python)

---

## Executive Summary

Successfully parallelized the Shallow Water simulation using CuPy, achieving **21.4× speedup** for 1024×1024 grids compared to CPU implementation. Profiling reveals efficient GPU utilization with asymptotic performance of **0.36 ns/cell**. Recommended minimum grid size: **1024×1024** for optimal GPU efficiency.

---

## Task 1: GPU Parallelization Strategy & Profiling

### Implementation Overview

**Key Modifications:**
1. Replaced NumPy with CuPy for GPU array operations
2. Added NVTX profiling markers for performance analysis
3. Minimized CPU↔GPU memory transfers
4. Implemented proper GPU synchronization for timing

**Code Changes:**
```python
# Before (CPU - NumPy)
import numpy as np
self.u = np.zeros((NY, NX), dtype=real_t)

# After (GPU - CuPy)
import cupy as cp
from cupyx.profiler import time_range
self.u = cp.zeros((NY, NX), dtype=real_t)  # GPU array

# Profiling markers
with time_range("integrate"):
    # Integration code
    with time_range("velocity_update"):
        # Velocity update operations
```

### NVIDIA nsys Profiling Results

**Test Configuration:** 512×512 grid, 500 iterations

#### [3/8] NVTX Function Timing

| Function | Time % | Total Time (ns) | Instances | Avg (ns) |
|----------|--------|-----------------|-----------|----------|
| integrate | 50.7% | 163,105,973 | 500 | 326,212 |
| exchange_vertical_ghost_lines | 16.2% | 52,083,535 | 1,000 | 52,084 |
| velocity_update | 14.0% | 45,049,826 | 500 | 90,100 |
| elevation_update | 12.9% | 41,521,070 | 500 | 83,042 |
| exchange_horizontal_ghost_lines | 6.2% | 19,865,837 | 1,000 | 19,866 |

**Analysis:**
- Integration step dominates (50.7%) as expected
- Ghost cell exchanges: 22.4% combined (some overhead but acceptable)
- Velocity/elevation updates: 26.9% (main computational kernels)
- Average integration time: **326 μs per step**

#### [5/8] CUDA API Timing

| Operation | Time % | Total Time (ns) | Calls | Avg (ns) |
|-----------|--------|-----------------|-------|----------|
| cudaMalloc | 61.1% | 81,447,515 | 19 | 4,286,711 |
| cuLaunchKernel | 26.6% | 35,509,278 | 10,022 | 3,543 |
| cudaMemcpyAsync | 11.0% | 14,601,688 | 2,007 | 7,275 |
| cuModuleLoadData | 0.9% | 1,170,887 | 14 | 83,635 |

**Analysis:**
- Memory allocation (cudaMalloc) dominates one-time setup
- Kernel launches: 10,022 total, ~3.5 μs per launch
- Memory transfers: minimal (11%), mostly for periodic snapshots

#### [6/8] GPU Kernel Timing

| Kernel | Time % | Total Time (ns) | Instances | Avg (ns) |
|--------|--------|-----------------|-----------|----------|
| cupy_subtract (float64) | 59.5% | 23,971,538 | 3,500 | 6,849 |
| cupy_multiply (float×float64) | 17.5% | 7,045,681 | 2,003 | 3,518 |
| cupy_copy (float64) | 14.5% | 5,835,769 | 4,009 | 1,456 |
| cupy_add (float64) | 8.4% | 3,372,055 | 501 | 6,731 |

**Analysis:**
- CuPy generates efficient fused operations
- Subtraction kernels dominate (stencil operations)
- Average kernel execution: **1.4-6.8 μs**
- Total kernels: 10,013 (highly optimized)

#### [7/8] Memory Transfer Statistics

| Operation | Time % | Total (ns) | Count | Avg (ns) |
|-----------|--------|------------|-------|----------|
| Device-to-Device | 82.1% | 2,750,368 | 2,000 | 1,375 |
| Device-to-Host | 17.6% | 589,754 | 7 | 84,251 |
| memset | 0.3% | 9,920 | 3 | 3,307 |

**Data Transfer Size:**
- Device-to-Host: 12.58 MB (periodic snapshots)
- Device-to-Device: 8.19 MB (ghost cell copies)
- memset: 6.29 MB (initialization)

**Analysis:**
- Minimal CPU↔GPU transfers (only 7 transfers to host)
- Most data movement is within GPU (ghost cells)
- Transfer overhead: **<1% of total runtime**

### Bottleneck Analysis

**Primary Bottlenecks:**
1. **Ghost cell exchanges** (22.4%): Required for periodic boundaries, but creates many small operations
2. **Kernel launch overhead**: 10,022 launches, ~3.5 μs each = ~35 ms total
3. **Initial allocation** (one-time): 81 ms for cudaMalloc

**Optimization Opportunities:**
1. **Kernel fusion**: Combine ghost exchanges with compute kernels
2. **Reduce snapshots**: Current: every 100 iterations, could reduce to 500
3. **Async I/O**: Use CUDA streams to overlap I/O with computation
4. **Custom kernels**: Replace multiple CuPy ops with single fused kernel

---

## Task 2A: Asymptotic Performance Analysis

### Methodology
Tested grid sizes from 64×64 to 2048×2048 on full GPU (RTX A6000, 84 SMs), measuring nanoseconds per cell per iteration.

### Results

| Grid Size | Total Cells | Time (sec) | ns/cell | Efficiency |
|-----------|-------------|------------|---------|------------|
| 64×64 | 4,096 | 0.224 | 54.76 | 0.7% |
| 96×96 | 9,216 | 0.232 | 25.13 | 1.4% |
| 128×128 | 16,384 | 0.230 | 14.02 | 2.6% |
| 192×192 | 36,864 | 0.238 | 6.47 | 5.6% |
| 256×256 | 65,536 | 0.233 | 3.56 | 10.1% |
| 384×384 | 147,456 | 0.235 | 1.59 | 22.6% |
| 512×512 | 262,144 | 0.231 | 0.88 | 40.9% |
| 768×768 | 589,824 | 0.234 | 0.40 | 90.0% |
| **1024×1024** | **1,048,576** | **0.400** | **0.38** | **94.7%** ✓ |
| 1536×1536 | 2,359,296 | 0.831 | 0.35 | 102.9% |
| 2048×2048 | 4,194,304 | 1.434 | 0.34 | 105.9% |

**Efficiency** = (Best ns/cell) / (Current ns/cell) × 100%

### Key Findings

**Asymptotic Performance:** **0.36 ns/cell** (average of 3 largest problems)

**Performance Improvement:** **160× faster** from smallest to largest grid

**Recommended Minimum Grid Size:** **1024×1024 (1,048,576 cells)**
- Achieves 94.7% of peak efficiency
- Performance: 0.38 ns/cell (within 5% of asymptotic)
- Below this: overhead dominates
- Above this: compute dominates

### Analysis

**Small Grids (< 512×512):**
- High ns/cell (>1.0)
- GPU underutilized
- Overhead dominates (kernel launches, synchronization)
- Efficiency < 50%

**Medium Grids (512×512 - 768×768):**
- Transitional region
- GPU starts to saturate
- ns/cell drops rapidly
- Efficiency 40-90%

**Large Grids (≥ 1024×1024):**
- Low ns/cell (~0.36)
- GPU fully utilized
- Compute dominates
- Efficiency > 94%

**Physical Interpretation:**
- RTX A6000: 84 SMs × 2048 threads/SM = 172,032 threads
- 1024×1024 grid = 1,048,576 cells
- Ratio: ~6 cells per thread (good workload balance)

---

## Task 2B: CPU vs GPU Speedup Analysis

### Methodology
Compared sequential NumPy (CPU) vs parallel CuPy (GPU) for various grid sizes, 500 iterations each.

### Results

| Grid Size | Total Cells | CPU Time (s) | GPU Time (s) | Speedup | Checksums Match |
|-----------|-------------|--------------|--------------|---------|-----------------|
| 128×128 | 16,384 | 0.054 | 0.124 | **0.43×** ⚠️ | ✓ |
| 256×256 | 65,536 | 0.183 | 0.127 | **1.43×** | ✓ |
| 512×512 | 262,144 | 1.397 | 0.122 | **11.42×** | ✓ |
| 768×768 | 589,824 | 2.106 | 0.121 | **17.40×** | ✓ |
| 1024×1024 | 1,048,576 | 4.318 | 0.201 | **21.43×** ⚠️ | ✓ |

**Note:** GPU time increases at 1024×1024 due to larger memory requirements and more complex operations.

### Analysis

**Small Grids (128×128):**
- GPU **slower** than CPU (0.43×)
- Overhead exceeds benefit
- Not recommended for GPU

**Medium Grids (256×256):**
- Modest speedup (1.43×)
- Barely breaks even
- CPU still competitive

**Large Grids (512×512+):**
- Strong speedups (11-21×)
- GPU clearly superior
- Scales well with problem size

**Speedup Trend:**
- Near-linear growth from 256×256 to 768×768
- Plateaus around 20× for very large grids
- Limited by memory bandwidth and Amdahl's law

**Why Not 100× Speedup?**
1. **Ghost cell overhead**: Small operations, high launch cost
2. **Memory bandwidth**: Limited by 768 GB/s GPU memory
3. **Amdahl's law**: Sequential portions (I/O, setup) limit scaling
4. **Periodic snapshots**: CPU transfers every 100 iterations

---

## Task 2C: Strong Scaling Analysis

### Methodology
Fixed grid size (512×512), varying iteration counts to measure consistency.

### Results

| Iterations | Total Time (s) | Time/Iteration (ms) | Checksum |
|------------|----------------|---------------------|----------|
| 100 | 0.027 | 0.268 | 4117.748 |
| 500 | 0.121 | 0.242 | 4117.748 |
| 1000 | 0.235 | 0.235 | 4117.908 |
| 2000 | 0.446 | 0.223 | 4137.816 |
| 5000 | 1.085 | 0.217 | 4321.796 |

### Analysis

**Time per Iteration Decreases:**
- 100 iters: 0.268 ms/iter
- 5000 iters: 0.217 ms/iter
- **Improvement: 19%**

**Why Performance Improves:**
1. **Amortized overhead**: Setup cost spread over more iterations
2. **Better GPU utilization**: Longer runs keep GPU busy
3. **Cache warming**: Data stays in GPU caches
4. **Reduced I/O impact**: Snapshot frequency relative to work

**Asymptotic Behavior:**
- Converges to ~0.22 ms/iter for large iteration counts
- Consistent with 0.88 ns/cell from asymptotic analysis
- 262,144 cells × 0.88 ns = 0.23 ms ✓

---

## Summary of Findings

### Implementation Success
✅ **GPU parallelization working correctly**
- Checksums match CPU version (within floating-point precision)
- NVTX profiling integrated successfully
- Memory transfers optimized (<1% overhead)

### Performance Achievements
✅ **Strong speedups for realistic workloads**
- 21.4× speedup for 1024×1024 grids
- 160× improvement from overhead regime to asymptotic
- 0.36 ns/cell asymptotic performance

### Optimal Configuration
✅ **Recommended settings**
- **Minimum grid:** 1024×1024 (94% efficiency)
- **Sweet spot:** 1024×1024 to 1536×1536
- **Avoid:** Grids smaller than 512×512

### Bottlenecks Identified
1. Ghost cell exchanges (22% of time)
2. Kernel launch overhead (~35 ms total)
3. Initial memory allocation (81 ms one-time)

### Optimization Potential
Estimated **2-3× additional speedup** possible through:
- Kernel fusion (eliminate 50% of launches)
- Custom CUDA kernels (reduce ghost overhead)
- Async I/O with streams

---

## Visualization

### Asymptotic Performance Plot
![Asymptotic Performance](asymptotic_performance_plot.png)

**Key Observations:**
- Clear "knee" around 512×512
- Performance stabilizes at 1024×1024
- 160× improvement from smallest to largest

---

## Conclusions

1. **GPU parallelization highly effective** for Shallow Water simulations at realistic scales (≥512×512)

2. **Asymptotic performance of 0.36 ns/cell** achieved, indicating excellent GPU utilization

3. **Recommended minimum: 1024×1024 grid** for optimal efficiency on RTX A6000

4. **21× speedup** demonstrates significant practical benefit for production workloads

5. **Further optimization possible** through kernel fusion and custom implementations

6. **CuPy provides excellent performance** with minimal code changes from NumPy

---

## Hardware Specifications

**Test System:**
- GPU: NVIDIA RTX A6000
- Memory: 49,140 MiB
- Compute Capability: 8.6
- SMs: 84
- Max Threads/SM: 2,048
- Total Threads: 172,032
- CUDA Version: 12.4
- Driver: 550.144.03

**Software:**
- CuPy: 13.6.0
- Python: 3.10
- CUDA Runtime: 12.9
- OS: Linux

---

## Files Generated

- `profile_output.txt` - NVIDIA nsys profiling output
- `profile_report.nsys-rep` - Binary profile for Nsight Systems GUI
- `asymptotic_performance_results.txt` - Raw asymptotic data
- `asymptotic_performance_plot.png` - Visualization
- `strong_scaling_results.txt` - Iteration scaling data
- `cpu_vs_gpu_speedup.txt` - Speedup comparison data
- `benchmark_asymptotic_log.txt` - Benchmark log

---

**Report Status:** ✅ COMPLETE  
**Date Generated:** November 11, 2025  
**Total Analysis Time:** ~30 minutes
