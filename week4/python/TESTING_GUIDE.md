# Testing and Execution Guide

## Prerequisites
You must be running on DAG with HPC GPU notebook image.

### Install CuPy
```bash
conda install cupy
# or
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

## Step-by-Step Execution

### 1. Validate GPU Implementation

**Test sequential (CPU) version:**
```bash
cd /home/bxl776_ku_dk/erda_mount/ahpc/week4/python
python sw_sequential.py --iter 100 --out test_cpu.data
```

**Test parallel (GPU) version:**
```bash
python sw_parallel.py --iter 100 --out test_gpu.data
```

**Compare checksums** - they should match within floating-point precision!

### 2. Quick Performance Test

```bash
# CPU version
time python sw_sequential.py --iter 500 --out cpu_500.data

# GPU version  
time python sw_parallel.py --iter 500 --out gpu_500.data
```

Expected: GPU should be significantly faster (10-100x depending on grid size).

### 3. Profile with NVIDIA nsys

**Basic profiling:**
```bash
nsys profile --stats=true python sw_parallel.py --iter 500 --out profiled.data
```

**Save detailed profile:**
```bash
nsys profile -o sw_profile --stats=true python sw_parallel.py --iter 1000
```

This generates `sw_profile.nsys-rep` that can be viewed in NVIDIA Nsight Systems GUI.

**Key sections to analyze:**
- [3/8] Time in decorated functions (integrate, velocity_update, elevation_update)
- [6/8] Individual CuPy kernel timing
- [7/8] Memory transfer statistics

### 4. Weak Scaling Benchmark

```bash
./benchmark_weak_scaling.sh
```

This will:
- Test with 2, 4, 6, 8, 10, 12, 14 SMs
- Scale grid size proportionally
- Save results to `weak_scaling_results.txt`

**Plot results:**
```bash
python plot_weak_scaling.py
```
Generates `weak_scaling_plot.png`

### 5. Asymptotic Performance Benchmark

```bash
./benchmark_asymptotic.sh
```

This will:
- Test grid sizes from 64×64 to 2048×2048
- Run on full GPU (14 SMs)
- Calculate ns/cell for each size
- Save results to `asymptotic_performance_results.txt`

**Plot results:**
```bash
python plot_asymptotic.py
```
Generates `asymptotic_performance_plot.png`

### 6. Visualize Results

Open `visualize.ipynb` in Jupyter to visualize the water elevation over time:
```python
# In the notebook
filename = "sw_output_gpu.data"
# Follow notebook instructions
```

## Troubleshooting

### CuPy not found
```bash
pip install cupy-cuda11x  # or appropriate CUDA version
```

### CUDA out of memory
Reduce grid size in `sw_parallel.py`:
```python
NX = 256  # Instead of 512
NY = 256
```

### Profiler not working
Make sure you're on DAG with GPU access:
```bash
nvidia-smi  # Should show GPU info
```

### Benchmarks taking too long
Reduce iterations in benchmark scripts:
- Edit `benchmark_*.sh`
- Change `--iter 1000` to `--iter 100`

### Different checksums CPU vs GPU
Small differences (<1%) are expected due to floating-point precision. Large differences indicate a bug.

## Expected Results

### Performance
- **Small grids (64×64):** GPU may be slower due to overhead
- **Medium grids (256×256):** GPU starts to show speedup
- **Large grids (512×512+):** GPU should be 10-100x faster

### Weak Scaling
- **Ideal:** Constant execution time as SMs increase
- **Reality:** Some overhead, efficiency 70-95%

### Asymptotic Performance
- **Small problems:** High ns/cell (overhead dominated)
- **Large problems:** Low ns/cell (compute dominated)
- **Knee point:** Where GPU becomes efficient (~256×256 or larger)

## Files Generated

- `test_*.data` - Binary output files for validation
- `weak_scaling_results.txt` - Weak scaling data
- `weak_scaling_plot.png` - Weak scaling visualization
- `asymptotic_performance_results.txt` - Asymptotic performance data
- `asymptotic_performance_plot.png` - Asymptotic performance visualization
- `sw_profile.nsys-rep` - NVIDIA nsys profile (viewable in Nsight Systems)

## Report Checklist

For the 3-page report, include:

1. **Strategy (0.5 pages)**
   - CuPy parallelization approach
   - Key code modifications
   
2. **Profiling Results (1 page)**
   - nsys output analysis
   - Kernel timing breakdown
   - Memory transfer overhead
   - Bottleneck identification

3. **Weak Scaling (0.75 pages)**
   - Plot with discussion
   - Efficiency analysis
   - Comparison to ideal scaling

4. **Asymptotic Performance (0.75 pages)**
   - Plot showing ns/cell vs grid size
   - Recommended minimum grid size
   - Justification based on data

5. **Code Appendix**
   - Key sections of `sw_parallel.py`
   - Use Absalon template

## Quick Command Reference

```bash
# Test
python sw_parallel.py --iter 100 --out test.data

# Profile
nsys profile --stats=true python sw_parallel.py --iter 500

# Weak scaling
./benchmark_weak_scaling.sh && python plot_weak_scaling.py

# Asymptotic
./benchmark_asymptotic.sh && python plot_asymptotic.py

# Visualize
jupyter notebook visualize.ipynb
```
