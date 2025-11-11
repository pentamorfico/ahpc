# Shallow Water GPU Parallelization with CuPy

## GPU Parallelization Strategy

### Overview
This implementation converts the sequential NumPy-based Shallow Water simulation to run on GPUs using CuPy, achieving significant speedup through parallel execution of stencil operations.

### Key Modifications

#### 1. **Array Backend Replacement**
- **From:** NumPy (CPU) → **To:** CuPy (GPU)
- All computational arrays (u, v, e) are allocated on GPU memory
- CuPy provides a NumPy-compatible API with GPU acceleration

#### 2. **Data Initialization on GPU**
```python
# GPU arrays using CuPy
self.u = cp.zeros((NY, NX), dtype=real_t)  # Horizontal velocity
self.v = cp.zeros((NY, NX), dtype=real_t)  # Vertical velocity  
self.e = cp.zeros((NY, NX), dtype=real_t)  # Water elevation

# Initial condition computed on GPU
II, JJ = cp.meshgrid(ii, jj, indexing="ij")
self.e[1:NY-1, 1:NX-1] = cp.exp(-0.02 * (II * II + JJ * JJ))
```

#### 3. **GPU Kernel Execution**
The main computational kernels run as GPU operations:

**Ghost Cell Exchange (Periodic Boundaries):**
- Horizontal and vertical ghost line updates
- Simple array copies executed as GPU kernels
- Enables periodic boundary conditions

**Velocity Update:**
```python
w.u[0:NY-1, 0:NX-1] -= dt / dx * g * (w.e[0:NY-1, 1:NX] - w.e[0:NY-1, 0:NX-1])
w.v[0:NY-1, 0:NX-1] -= dt / dy * g * (w.e[1:NY, 0:NX-1] - w.e[0:NY-1, 0:NX-1])
```
- Fused operations: subtraction, multiplication, scalar operations
- All executed as single GPU kernel launches

**Elevation Update:**
```python
w.e[1:NY-1, 1:NX-1] -= dt / dx * (w.u[1:NY-1, 1:NX-1] - w.u[1:NY-1, 0:NX-2]) + \
                       dt / dy * (w.v[1:NY-1, 1:NX-1] - w.v[0:NY-2, 1:NX-1])
```
- Complex fused operation executed as single kernel
- Minimizes intermediate memory allocations

#### 4. **Profiling Integration**
Added NVTX markers using `cupyx.profiler.time_range`:
```python
with time_range("integrate"):
    # Integration code
    with time_range("velocity_update"):
        # Velocity updates
    with time_range("elevation_update"):
        # Elevation updates
```
Enables detailed profiling with `nsys profile --stats=true`

#### 5. **Memory Transfer Optimization**
**Minimized Host↔Device Transfers:**
- Arrays stay on GPU throughout simulation
- Only transfer to CPU when needed:
  - File I/O (periodic snapshots)
  - Final checksum calculation

**Efficient I/O Strategy:**
```python
# Transfer only when saving snapshots
if t % config.data_period == 0:
    water_history.append(water_world.e.copy())  # GPU copy

# Batch transfer to CPU for file write
history_cpu = [cp.asnumpy(frame) for frame in water_history]
```

#### 6. **GPU Synchronization**
Proper timing with GPU synchronization:
```python
cp.cuda.Stream.null.synchronize()  # Before timing
# Computation
cp.cuda.Stream.null.synchronize()  # After timing
```

### Expected Performance Characteristics

#### Bottlenecks to Watch:
1. **Ghost cell exchanges:** Small data operations (may have overhead)
2. **Memory transfers:** Periodic snapshots to CPU
3. **Kernel launch overhead:** Many small kernel launches

#### Optimization Opportunities:
1. **Kernel fusion:** CuPy automatically fuses some operations
2. **Async I/O:** Could use CUDA streams for overlapping I/O
3. **Reduce snapshots:** Less frequent saves = fewer transfers

### GPU Utilization Analysis

**Expected nsys Profile Sections:**
- **[3/8]:** Time in integrate, velocity_update, elevation_update
- **[5/8]:** cudaMemcpyAsync for snapshot transfers
- **[6/8]:** CuPy-generated kernels for array operations
- **[7/8]:** Device-to-Host transfers during snapshot saves
- **[8/8]:** GPU memory usage (~3 × NX × NY × 8 bytes for arrays)

### Running the Code

**Sequential CPU version:**
```bash
python sw_sequential.py --iter 1000 --out sw_output_cpu.data
```

**Parallel GPU version:**
```bash
python sw_parallel.py --iter 1000 --out sw_output_gpu.data
```

**With profiling:**
```bash
nsys profile --stats=true python sw_parallel.py --iter 500
```

**Weak scaling test:**
```bash
./run_sw.sh 2    # Run on 2 SMs
./run_sw.sh 4    # Run on 4 SMs
./run_sw.sh 14   # Run on all 14 SMs
```

### Validation
Compare checksums between CPU and GPU versions to verify correctness:
```bash
python sw_sequential.py --iter 500 --out cpu.data
python sw_parallel.py --iter 500 --out gpu.data
# Checksums should match within floating-point precision
```

### Grid Size Considerations
- Default: 512×512 grid
- Modify NX, NY at top of file for scaling studies
- GPU has 14 SMs, max 1024 threads/block, 2048 threads/SM
- For weak scaling: scale grid proportionally with SM count

### Next Steps
1. ✅ Implement CuPy parallelization
2. ⏳ Run and validate GPU implementation
3. ⏳ Collect profiling data with nsys
4. ⏳ Measure weak scaling (2-14 SMs)
5. ⏳ Determine asymptotic performance
6. ⏳ Write 3-page report
