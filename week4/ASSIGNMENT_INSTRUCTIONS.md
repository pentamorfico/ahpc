# Assignment 4: Shallow Waters - GPU Parallelization

**Deadline:** Monday 3/11 23:59

## Overview
Implement GPU parallelization of a Shallow Water simulation model using CuPy (Python).

## Environment Setup
- **Platform:** DAG Jupyter with HPC GPU notebook image
- **Required:** `conda install cupy`
- **Profiler:** NVIDIA nsys profiler

## File Structure
```
week4/python/
├── sw_sequential.py      # Sequential baseline (backup)
├── sw_parallel.py        # Your parallelized version
├── visualize.ipynb       # Visualization tool
├── run_sw.sh            # Weak scaling benchmark script
```

## Running the Code

### Sequential (CPU)
```bash
python sw_sequential.py --iter 500 --out sw_output.txt
```

### Parallel (GPU)
```bash
python sw_parallel.py --iter 500 --out sw_output_gpu.txt
```

### Profiling
```bash
nsys profile --stats=true python sw_parallel.py --iter 500
```

## Tasks

### Task 1: CuPy Parallelization (GPU)
**Objective:** Replace NumPy with CuPy for GPU execution

**Strategy:**
1. Identify compute-intensive parts of the code
2. Replace NumPy arrays with CuPy arrays
3. Use CuPy functions instead of NumPy
4. Consider kernel fusion to reduce GPU invocations
5. Use streams for I/O operations

**Deliverables:**
- Parallelized code
- Strategy explanation
- nsys profiler output showing:
  - Time spent in decorated functions [3/8]
  - Timing for CUDA functions [5/8]
  - Individual kernel timing [6/8]
  - Memory transfer statistics [7/8]
- Performance analysis and bottleneck discussion

### Task 2: Weak Scaling & Asymptotic Performance

**Part A: Weak Scaling**
- Use `run_sw.sh` to constrain streaming multiprocessors (2-14 SMs, even numbers)
- Adjust NX and NY grid sizes for weak scaling measurements
- **GPU specs:** 14 SMs, max 1024 threads/block, max 2048 threads/SM
- Create weak scaling plot
- Discuss results

**Part B: Asymptotic Performance**
- Run on full GPU with increasing workload sizes
- Plot: ns per grid cell vs. number of grid cells
- Identify when runtime scales linearly with workload
- Determine minimum efficient grid size for ERDA GPU

**Deliverables:**
- Weak scaling plot with discussion
- Asymptotic performance plot
- Recommendation for optimal grid size

## Profiler Output Sections
- **[3/8]:** Time in decorated functions (source code analysis)
- **[4/8]:** OS/system calls (usually ignore)
- **[5/8]:** CUDA API functions timing
- **[6/8]:** Individual kernel timing (CuPy/OpenACC generated)
- **[7/8]:** Host↔Device memory transfer stats
- **[8/8]:** GPU memory usage

## Report Requirements
- Maximum 3 pages (excluding code)
- Use Absalon template for code inclusion
- Include:
  - Parallelization strategy
  - Code snippets
  - Profiler output analysis
  - Weak scaling results
  - Asymptotic performance analysis
  - Optimal grid size recommendation

## Key Considerations
- Visualize output at least once for validation
- Look for implicit small data transfers in profiler
- Consider operation/kernel fusion opportunities
- Balance GPU utilization vs. overhead
