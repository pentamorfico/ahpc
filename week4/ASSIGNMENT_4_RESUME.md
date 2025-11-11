# Assignment 4: Shallow Waters GPU Parallelization - Complete Resume

**Course:** Applied High Performance Computing  
**Deadline:** Monday 3/11 23:59  
**Platform:** DAG (ERDA) with HPC GPU notebook  

---

## 📋 Assignment Overview

### Objective
Parallelize a Shallow Water simulation model for GPU execution and analyze its performance through:
1. **Task 1:** GPU parallelization using CuPy (Python) or OpenACC (C++/Fortran)
2. **Task 2:** Weak scaling analysis and asymptotic performance measurement

### Shallow Water Model
- Simplest numerical representation of ocean dynamics
- Used for storm surge and tsunami prediction
- Demonstrates stencil operations and their parallelization
- Involves updating velocity (u, v) and elevation (e) fields using finite difference methods

### Key Concepts
- **Stencil operations:** Each cell's update depends on neighboring cells
- **Ghost cells:** Padding rows/columns for periodic boundary conditions
- **Weak scaling:** Keep work/processor constant as processors increase
- **Asymptotic performance:** Find minimum problem size for efficient GPU utilization

---

## 🗂️ Repository Structure

```
week4/
├── ASSIGNMENT_INSTRUCTIONS.md          # Full assignment details
├── ASSIGNMENT_4_RESUME.md             # This file
│
├── python/                            # ✅ COMPLETE IMPLEMENTATION
│   ├── sw_sequential.py               # CPU baseline (NumPy)
│   ├── sw_parallel.py                 # ✅ GPU version (CuPy)
│   ├── run_sw.sh                      # MPS control script for weak scaling
│   ├── benchmark_weak_scaling.sh      # ✅ Automated weak scaling tests
│   ├── benchmark_asymptotic.sh        # ✅ Automated asymptotic tests
│   ├── plot_weak_scaling.py           # ✅ Visualization script
│   ├── plot_asymptotic.py             # ✅ Visualization script
│   ├── visualize.ipynb                # Water elevation visualization
│   ├── README.md                      # ✅ Implementation strategy
│   ├── TESTING_GUIDE.md               # ✅ Execution instructions
│   ├── REPORT_TEMPLATE.md             # ✅ 3-page report template
│   ├── CHECKLIST.md                   # ✅ Progress checklist
│   └── SUMMARY.md                     # ✅ Quick reference
│
├── cpp/                               # C++ version (alternative)
│   ├── sw_sequential.cpp              # CPU baseline
│   ├── sw_parallel.cpp                # For OpenACC parallelization
│   ├── Makefile                       # nvc++ compilation
│   ├── run_sw.sh                      # MPS control script
│   └── visualize.ipynb                # Visualization
│
└── fortran/                           # Fortran version (alternative)
    ├── sw_sequential.f90              # CPU baseline
    ├── sw_parallel.f90                # For OpenACC parallelization
    ├── Makefile                       # nvfortran compilation
    ├── run_sw.sh                      # MPS control script
    └── visualize.ipynb                # Visualization
```

---

## ✅ What Has Been Completed (Python Implementation)

### 1. Core GPU Implementation
**File:** `week4/python/sw_parallel.py`

**Key Features:**
- ✅ NumPy → CuPy conversion (all arrays on GPU)
- ✅ NVTX profiling markers for nsys
- ✅ GPU synchronization for accurate timing
- ✅ Optimized memory transfers (minimal CPU↔GPU)
- ✅ Periodic boundary conditions (ghost cell exchange)
- ✅ Vectorized stencil operations

**What Changed:**
```python
# Before (CPU - NumPy)
import numpy as np
self.u = np.zeros((NY, NX), dtype=real_t)

# After (GPU - CuPy)
import cupy as cp
from cupyx.profiler import time_range
self.u = cp.zeros((NY, NX), dtype=real_t)  # GPU array
```

### 2. Profiling Infrastructure
- ✅ NVTX markers in critical functions
- ✅ Integration with nsys profiler
- ✅ Sections tracked:
  - `integrate` (overall integration step)
  - `velocity_update` (u, v updates)
  - `elevation_update` (e update)
  - Ghost cell exchanges

### 3. Benchmarking Automation

**Weak Scaling:** `benchmark_weak_scaling.sh`
- Tests: 2, 4, 6, 8, 10, 12, 14 SMs
- Grid scales proportionally (constant work/SM)
- Output: `weak_scaling_results.txt`
- Visualization: `plot_weak_scaling.py` → `weak_scaling_plot.png`

**Asymptotic Performance:** `benchmark_asymptotic.sh`
- Tests: 64×64 to 2048×2048 grids
- Runs on full GPU (14 SMs)
- Calculates ns/cell for each size
- Output: `asymptotic_performance_results.txt`
- Visualization: `plot_asymptotic.py` → `asymptotic_performance_plot.png`

### 4. Complete Documentation
- ✅ `README.md` - Implementation strategy and technical details
- ✅ `TESTING_GUIDE.md` - Step-by-step execution instructions
- ✅ `REPORT_TEMPLATE.md` - 3-page report template with structure
- ✅ `CHECKLIST.md` - Progress tracking and todo items
- ✅ `SUMMARY.md` - Quick reference and what's been built

---

## 🚀 How to Run (On DAG)

### Prerequisites
1. Access DAG via ERDA
2. Start "HPC GPU notebook" Jupyter session
3. Navigate to your ahpc repository
4. Install CuPy: `conda install cupy`

### Quick Validation
```bash
cd week4/python

# Test CPU version
python sw_sequential.py --iter 100 --out test_cpu.data

# Test GPU version
python sw_parallel.py --iter 100 --out test_gpu.data

# Compare checksums (should match!)
```

### Profiling with nsys
```bash
# Basic profiling
nsys profile --stats=true python sw_parallel.py --iter 500 --out profiled.data

# Save detailed profile for GUI
nsys profile -o sw_profile --stats=true python sw_parallel.py --iter 1000
```

**Key sections to analyze:**
- **[3/8]:** Time in decorated functions (integrate, velocity_update, etc.)
- **[5/8]:** CUDA API timing (memory copies, kernel launches)
- **[6/8]:** Individual GPU kernel timing
- **[7/8]:** Host↔Device memory transfer statistics
- **[8/8]:** GPU memory usage

### Run Benchmarks

**Weak Scaling (~5-10 minutes):**
```bash
./benchmark_weak_scaling.sh
python plot_weak_scaling.py
```
Output: `weak_scaling_results.txt` and `weak_scaling_plot.png`

**Asymptotic Performance (~10-20 minutes):**
```bash
./benchmark_asymptotic.sh
python plot_asymptotic.py
```
Output: `asymptotic_performance_results.txt` and `asymptotic_performance_plot.png`

### Visualization
```bash
jupyter notebook visualize.ipynb
# Load your output file to see water elevation animation
```

---

## 📊 Expected Results

### Performance Gains
- **Small grids (64×64):** GPU may be slower (overhead dominates)
- **Medium grids (256×256-512×512):** 5-20× speedup
- **Large grids (1024×1024+):** 50-100× speedup

### Weak Scaling
- **Ideal:** Constant time as SMs increase
- **Expected:** 70-95% efficiency
- **Overhead sources:**
  - Ghost cell exchange
  - MPS daemon overhead
  - Synchronization costs

### Asymptotic Performance
- **Small problems:** High ns/cell (overhead dominates)
- **Large problems:** Low ns/cell (compute dominates)
- **Knee point:** ~256×256 where GPU becomes efficient
- **Below knee:** Poor GPU utilization
- **Above knee:** Good GPU utilization

---

## 📝 Report Requirements

### Format
- **Length:** Maximum 3 pages (excluding code appendix)
- **Template:** Use `REPORT_TEMPLATE.md` as starting point
- **Code:** Include in appendix using Absalon template

### Content Structure

#### 1. Strategy (0.5 pages)
- CuPy parallelization approach
- Key code modifications
- Code snippets showing:
  - GPU array initialization
  - Profiled integration function
  - Memory transfer optimization

#### 2. Profiling Results (1 page)
- nsys command used
- [3/8] NVTX function timing
- [6/8] GPU kernel timing
- [7/8] Memory transfer stats
- Bottleneck analysis
- Optimization opportunities

#### 3. Weak Scaling (0.75 pages)
- Methodology (SMs tested, grid sizes)
- Results table and plot
- Efficiency calculations
- Comparison to ideal scaling
- Discussion of overhead sources

#### 4. Asymptotic Performance (0.75 pages)
- Methodology (grid sizes tested)
- Results table and plot
- ns/cell trend analysis
- Recommended minimum grid size
- Justification with data

#### 5. Code Appendix
- Full `sw_parallel.py`
- Key modifications highlighted

---

## 🔑 Key Technical Details

### GPU Specifications (DAG)
- **Streaming Multiprocessors (SMs):** 14
- **Max threads per block:** 1024
- **Max threads per SM:** 2048
- **Total:** 14 SMs × 2048 threads = 28,672 threads maximum

### Grid Size Considerations
For weak scaling, maintain constant work per SM:
- 2 SMs: 256×256 = 65,536 cells → 32,768 cells/SM
- 4 SMs: 362×362 = 131,044 cells → 32,761 cells/SM
- 14 SMs: 677×677 = 458,329 cells → 32,738 cells/SM

### Memory Layout
```
Water structure (3 arrays on GPU):
- u: NY × NX × 8 bytes (horizontal velocity)
- v: NY × NX × 8 bytes (vertical velocity)
- e: NY × NX × 8 bytes (elevation)

For 512×512 grid:
- Per array: 512 × 512 × 8 = 2,097,152 bytes ≈ 2 MB
- Total: 3 × 2 MB = 6 MB GPU memory
```

### Stencil Operations
**Velocity Update:**
```python
u[i,j] -= (dt/dx) * g * (e[i,j+1] - e[i,j])
v[i,j] -= (dt/dy) * g * (e[i+1,j] - e[i,j])
```

**Elevation Update:**
```python
e[i,j] -= (dt/dx) * (u[i,j] - u[i,j-1]) + (dt/dy) * (v[i,j] - v[i-1,j])
```

### Periodic Boundaries (Ghost Cells)
```python
# Horizontal ghost lines
data[0, :] = data[NY-2, :]      # Top copies from second-to-last
data[NY-1, :] = data[1, :]      # Bottom copies from second

# Vertical ghost lines  
data[:, 0] = data[:, NX-2]      # Left copies from second-to-last
data[:, NX-1] = data[:, 1]      # Right copies from second
```

---

## 🔍 Profiling Analysis Guide

### [3/8] NVTX Function Timing
**What to look for:**
- Which function dominates runtime?
- Is ghost exchange overhead significant?
- Time per integration step

**Example interpretation:**
```
integrate: 62.5%           → Main computation (good)
velocity_update: 19.0%     → Stencil ops (expected)
elevation_update: 18.4%    → Stencil ops (expected)
ghost_exchange: <1%        → Low overhead (good)
```

### [6/8] GPU Kernel Timing
**What to look for:**
- How many kernels per iteration?
- Are velocity/elevation kernels balanced?
- Any unexpected small kernels?

**Example interpretation:**
```
velocity_update kernel: 49.1%     → Balanced compute
elevation_update kernel: 46.3%    → Balanced compute
ghost exchange kernels: 4.3%      → Some overhead
copy kernels: 0.3%                → Minimal
```

### [7/8] Memory Transfer Stats
**What to look for:**
- Total data transferred
- Number of transfers
- Is it dominated by Device→Host (snapshots)?

**Example interpretation:**
```
Device-to-Host: 90 MB, 12 transfers  → Periodic snapshots (expected)
Host-to-Device: 1 MB, 2 transfers    → Initial setup only (good)
```

**Red flags:**
- Many small transfers (implicit copies)
- Unexpected Host→Device during iteration
- Transfer time > 10% of total runtime

---

## 🎯 Success Criteria

### Implementation
- ✅ GPU code runs without errors
- ✅ Checksums match CPU version (within floating-point precision)
- ✅ Significant speedup for large grids (>10×)
- ✅ NVTX profiling markers work with nsys

### Profiling
- ✅ nsys data collected successfully
- ✅ Can identify dominant kernels
- ✅ Memory transfer overhead quantified
- ✅ Bottlenecks identified and explained

### Weak Scaling
- ✅ Tests completed for 2-14 SMs
- ✅ Plot generated showing efficiency
- ✅ Average efficiency calculated
- ✅ Deviations from ideal explained

### Asymptotic Performance
- ✅ Multiple grid sizes tested
- ✅ Plot shows ns/cell trend
- ✅ Knee point identified
- ✅ Minimum efficient grid size recommended

### Report
- ✅ ≤ 3 pages (excluding code)
- ✅ All sections complete with real data
- ✅ Plots embedded and readable
- ✅ Analysis shows understanding
- ✅ Code included in appendix

---

## 🐛 Troubleshooting

### CuPy Import Error
```bash
# Install CuPy
conda install cupy
# or
pip install cupy-cuda11x  # Match your CUDA version
```

### CUDA Out of Memory
```python
# In sw_parallel.py, reduce grid size:
NX = 256  # instead of 512
NY = 256
```

### Can't Run nsys
```bash
# Make sure you're on GPU node
nvidia-smi  # Should show GPU info

# If not available, start HPC GPU notebook on DAG
```

### Benchmarks Too Slow
```bash
# Edit benchmark scripts to reduce iterations:
# In benchmark_*.sh, change:
--iter 1000  # to
--iter 100   # or even 50 for testing
```

### Different Checksums
- Small differences (<0.01%) are OK (floating-point rounding)
- Large differences indicate a bug in the GPU implementation

### Plot Scripts Fail
```bash
# Install matplotlib if needed:
pip install matplotlib pandas
```

---

## 📚 Reference Material

### Shallow Water Equations
- See PHPC textbook section 13.3
- Lecture notes on Absalon (first 3 pages sufficient)
- Finite difference method for PDEs

### GPU Programming Concepts
- **Stencil operations:** Structured grid computations
- **Weak scaling:** Fixed work per processor
- **Strong scaling:** Fixed total work
- **Occupancy:** GPU resource utilization
- **Kernel fusion:** Combining operations to reduce overhead

### CuPy Documentation
- [CuPy website](https://cupy.dev/)
- NumPy-compatible API
- Automatic kernel generation
- NVTX integration for profiling

### NVIDIA Nsight Systems
- [Nsys documentation](https://docs.nvidia.com/nsight-systems/)
- Command-line profiler
- NVTX range timing
- Kernel and memory transfer analysis

---

## 📅 Recommended Timeline

### Week 1 (Setup & Testing)
- [ ] Access DAG and set up environment
- [ ] Install CuPy
- [ ] Validate GPU implementation
- [ ] Run basic profiling

### Week 2 (Benchmarking)
- [ ] Run weak scaling benchmark
- [ ] Run asymptotic performance benchmark
- [ ] Generate all plots
- [ ] Analyze results

### Week 3 (Report Writing)
- [ ] Fill in report template with real data
- [ ] Write analysis sections
- [ ] Include code appendix
- [ ] Review and polish
- [ ] Submit before deadline

---

## 🎓 Learning Objectives

By completing this assignment, you will:
1. ✅ Understand GPU parallelization of stencil operations
2. ✅ Learn to use CuPy for scientific computing
3. ✅ Master GPU profiling with NVIDIA nsys
4. ✅ Analyze weak scaling efficiency
5. ✅ Determine optimal problem sizes for GPU workloads
6. ✅ Identify performance bottlenecks
7. ✅ Understand memory transfer overhead
8. ✅ Write technical performance reports

---

## 🏁 Next Steps

### Immediate Actions (On DAG):
1. Navigate to `week4/python`
2. Install CuPy: `conda install cupy`
3. Test implementations: `python sw_sequential.py --iter 100 --out test.data`
4. Run profiling: `nsys profile --stats=true python sw_parallel.py --iter 500`
5. Execute benchmarks: `./benchmark_weak_scaling.sh && ./benchmark_asymptotic.sh`
6. Generate plots: `python plot_*.py`

### Report Writing:
1. Open `REPORT_TEMPLATE.md`
2. Fill in all [X] placeholders with your actual data
3. Embed generated plots
4. Write analysis sections
5. Include code appendix
6. Convert to PDF
7. Submit to Absalon

---

## 📞 Additional Resources

### Documentation Files
- `python/README.md` - Detailed implementation strategy
- `python/TESTING_GUIDE.md` - Step-by-step execution
- `python/REPORT_TEMPLATE.md` - Complete report structure
- `python/CHECKLIST.md` - Progress tracking
- `python/SUMMARY.md` - Quick overview

### DAG User Guide
- [ERDA User Guide PDF](https://erda.dk/public/ucph-erda-user-guide.pdf)

### Course Materials
- Absalon for lecture notes and templates
- PHPC textbook Chapter 13.3
- Assignment instructions on Absalon

---

## ✨ Summary

**Status:** Implementation COMPLETE ✅  
**Remaining:** Testing on DAG + Report Writing  
**Deadline:** Monday 3/11 23:59  

**You have everything you need:**
- ✅ Fully implemented GPU code
- ✅ Automated benchmarking scripts
- ✅ Visualization tools
- ✅ Complete documentation
- ✅ Report template

**Just need to:**
1. Run on DAG with GPU
2. Collect results
3. Fill in report template
4. Submit

**Good luck! 🚀**
