# Assignment 4 Report Template
# Shallow Water GPU Parallelization with CuPy

**Student:** [Your Name]  
**Date:** [Submission Date]  
**Course:** Applied High Performance Computing

---

## 1. GPU Parallelization Strategy (0.5 pages)

### Overview
This implementation parallelizes the Shallow Water simulation using CuPy, replacing NumPy operations with GPU-accelerated equivalents.

### Key Modifications

**1. Array Backend Conversion**
- Replaced NumPy with CuPy for all computational arrays (u, v, e)
- Arrays allocated directly on GPU memory using `cp.zeros()`
- Initial conditions computed on GPU using `cp.meshgrid()` and `cp.exp()`

**2. Computational Kernels**
The main operations executed on GPU:
- **Ghost cell exchange:** Periodic boundary updates (simple memory copies)
- **Velocity update:** Stencil operations computing horizontal and vertical velocities
- **Elevation update:** Complex stencil combining velocity gradients

**3. Memory Transfer Minimization**
- Data remains on GPU throughout simulation
- CPU transfers only for:
  - Periodic snapshots (every `data_period` iterations)
  - Final checksum calculation
  - File I/O operations

**4. Profiling Integration**
- Added NVTX markers using `cupyx.profiler.time_range`
- Key sections marked: `integrate`, `velocity_update`, `elevation_update`
- Enables detailed analysis with `nsys profile --stats=true`

**5. Synchronization**
- GPU synchronization before/after timing: `cp.cuda.Stream.null.synchronize()`
- Ensures accurate performance measurements

### Code Snippet - Core Integration
```python
def integrate(w, dt, dx, dy, g):
    with time_range("integrate"):
        # Periodic boundaries
        exchange_horizontal_ghost_lines(w.e)
        exchange_vertical_ghost_lines(w.u)
        # ... more ghost exchanges
        
        # GPU stencil operations
        with time_range("velocity_update"):
            w.u[0:NY-1, 0:NX-1] -= dt / dx * g * (w.e[0:NY-1, 1:NX] - w.e[0:NY-1, 0:NX-1])
            w.v[0:NY-1, 0:NX-1] -= dt / dy * g * (w.e[1:NY, 0:NX-1] - w.e[0:NY-1, 0:NX-1])
        
        with time_range("elevation_update"):
            w.e[1:NY-1, 1:NX-1] -= dt / dx * (w.u[1:NY-1, 1:NX-1] - w.u[1:NY-1, 0:NX-2]) + \
                                   dt / dy * (w.v[1:NY-1, 1:NX-1] - w.v[0:NY-2, 1:NX-1])
```

---

## 2. Profiling Analysis (1 page)

### NVIDIA nsys Profile Output

**Command used:**
```bash
nsys profile --stats=true python sw_parallel.py --iter 1000
```

**[3/8] Function Timing (NVTX Ranges)**
```
[Insert your actual nsys output here]

Time (%)  Total Time (ns)  Instances  Avg (ns)  Function
--------  ---------------  ---------  --------  --------
  XX.X%    XXXXXXXXXXXXX       1000   XXXXXX   integrate
  XX.X%    XXXXXXXXXXXXX       2000   XXXXXX   velocity_update
  XX.X%    XXXXXXXXXXXXX       2000   XXXXXX   elevation_update
  XX.X%    XXXXXXXXXXXXX       4000   XXXXXX   exchange_ghost_lines
```

**Analysis:**
- Most time spent in `[dominant function]` (XX%)
- Ghost cell exchanges take XX% - [fast/slow relative to compute]
- Average integration step takes XX μs

**[6/8] GPU Kernel Timing**
```
[Insert kernel timing data]

Time (%)  Total Time (ns)  Instances  Name
--------  ---------------  ---------  ----
  XX.X%    XXXXXXXXXXXXX       1000   velocity_update kernel
  XX.X%    XXXXXXXXXXXXX       1000   elevation_update kernel
```

**Analysis:**
- CuPy generates [number] kernels per integration step
- Velocity/elevation updates dominate GPU time (XX%)
- Kernel fusion opportunities: [describe if applicable]

**[7/8] Memory Transfers**
```
[Insert memory transfer stats]

Total (MB)  Count  Operation
----------  -----  ---------
   XX.XXX      XX  Device-to-Host
   XX.XXX      XX  Host-to-Device
```

**Analysis:**
- Device-to-Host transfers for periodic snapshots: XX MB total
- Transfer overhead: XX% of total runtime
- [Minimal/Significant] impact on performance

### Bottleneck Identification

**Primary bottlenecks:**
1. [Identify based on your data]
2. [e.g., Memory transfers, ghost exchanges, kernel overhead]

**Optimization opportunities:**
1. [Suggestions based on profile]
2. [e.g., Kernel fusion, async I/O, reduce snapshot frequency]

---

## 3. Weak Scaling Results (0.75 pages)

### Methodology
- Tested 2, 4, 6, 8, 10, 12, 14 SMs using CUDA MPS
- Grid sizes scaled proportionally to maintain constant work per SM
- Base: 256×256 grid (65,536 cells) for 2 SMs ≈ 32,768 cells/SM

### Grid Sizes Used
| SMs | Grid Size | Total Cells | Cells/SM |
|-----|-----------|-------------|----------|
| 2   | 256×256   | 65,536      | 32,768   |
| 4   | 362×362   | 131,044     | 32,761   |
| ... | ...       | ...         | ...      |
| 14  | 677×677   | 458,329     | 32,738   |

### Results

**Execution Time:**
| SMs | Time (s) | Efficiency (%) |
|-----|----------|----------------|
| 2   | X.XXX    | 100.0          |
| 4   | X.XXX    | XX.X           |
| ... | ...      | ...            |
| 14  | X.XXX    | XX.X           |

**[Insert weak_scaling_plot.png here]**

### Discussion

**Weak scaling efficiency:**
- Average efficiency: XX.X%
- Best performance at [X] SMs (XX.X% efficiency)
- Worst performance at [X] SMs (XX.X% efficiency)

**Deviations from ideal:**
1. [Explain why efficiency varies]
2. [e.g., Ghost exchange overhead, synchronization costs]
3. [Load imbalance, MPS overhead]

**Comparison to expectations:**
- [Better/Worse] than expected because [reason]
- GPU shows [excellent/good/moderate/poor] weak scaling

---

## 4. Asymptotic Performance (0.75 pages)

### Methodology
- Tested grid sizes: 64×64 to 2048×2048
- All tests on full GPU (14 SMs)
- Calculated nanoseconds per cell per iteration
- Identified point where performance stabilizes

### Results

**Performance vs Grid Size:**
| Grid Size | Total Cells | Time (s) | ns/cell | Efficiency (%) |
|-----------|-------------|----------|---------|----------------|
| 64×64     | 4,096       | X.XXX    | XXXX.XX | XX.X           |
| 128×128   | 16,384      | X.XXX    | XXX.XX  | XX.X           |
| ...       | ...         | ...      | ...     | ...            |
| 2048×2048 | 4,194,304   | X.XXX    | XX.XX   | 100.0          |

**[Insert asymptotic_performance_plot.png here]**

### Analysis

**Asymptotic performance:** XX.XX ns/cell/iteration
- Achieved with grids ≥ [size]×[size]
- Represents [X]x improvement over smallest grid

**Performance regions:**
1. **Overhead-dominated (< [size]×[size]):**
   - High ns/cell due to kernel launch overhead
   - GPU underutilized
   
2. **Transition region ([size]×[size] to [size]×[size]):**
   - Rapid performance improvement
   - GPU filling up
   
3. **Compute-dominated (> [size]×[size]):**
   - Stable ns/cell (asymptotic)
   - Full GPU utilization

### Recommended Grid Size

**Minimum efficient grid:** [X]×[X] ([X,XXX] cells)

**Justification:**
- Performance within [X]% of asymptotic value
- Good balance between:
  - GPU utilization ([XX]%)
  - Memory usage ([XX] MB)
  - Overhead vs. compute ratio
  
**For ERDA GPU (14 SMs):**
- Optimal range: [X]×[X] to [X]×[X]
- Sweet spot: [X]×[X] (balances efficiency and problem size)

---

## 5. Conclusions

### Performance Summary
- GPU implementation achieves [X]x speedup over CPU
- Weak scaling efficiency: [XX]% average
- Asymptotic performance: [XX] ns/cell/iteration

### Key Insights
1. [Insight from profiling]
2. [Insight from weak scaling]
3. [Insight from asymptotic analysis]

### Future Optimizations
1. Kernel fusion to reduce launch overhead
2. Asynchronous I/O using CUDA streams
3. Reduced snapshot frequency to minimize transfers

---

## Appendix: Code

### Full sw_parallel.py
[Include full code using Absalon template format]

Key sections:
- CuPy imports and setup
- Water class with GPU arrays
- integrate() function with profiling markers
- simulate() function with synchronization

**Lines of code modified:** [XX] lines
**Key libraries:** CuPy, cupyx.profiler

---

**Total Pages:** 3.0 (excluding code appendix)
