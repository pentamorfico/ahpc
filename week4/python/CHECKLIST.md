# Assignment 4 - Complete Checklist

## ✅ Implementation Phase (DONE!)

### Core Files Created/Modified
- [x] `sw_parallel.py` - Converted to CuPy for GPU execution
  - [x] Import CuPy and NVTX profiling
  - [x] Replace NumPy arrays with CuPy arrays
  - [x] Add profiling markers (time_range)
  - [x] Add GPU synchronization
  - [x] Optimize memory transfers
  
### Benchmarking Scripts Created
- [x] `benchmark_weak_scaling.sh` - Automated weak scaling tests
- [x] `benchmark_asymptotic.sh` - Automated asymptotic performance tests
- [x] `plot_weak_scaling.py` - Visualization script for weak scaling
- [x] `plot_asymptotic.py` - Visualization script for asymptotic performance

### Documentation Created
- [x] `README.md` - Implementation strategy and details
- [x] `TESTING_GUIDE.md` - Step-by-step execution instructions
- [x] `REPORT_TEMPLATE.md` - Template for the 3-page report
- [x] `ASSIGNMENT_INSTRUCTIONS.md` - Assignment requirements
- [x] `SUMMARY.md` - Overview of implementation
- [x] `CHECKLIST.md` - This file!

### Permissions Set
- [x] Made all scripts executable (`chmod +x`)

## ⏳ Testing Phase (TODO - Must run on DAG with GPU)

### Environment Setup
- [ ] Access DAG via ERDA
- [ ] Start HPC GPU notebook Jupyter session
- [ ] Navigate to week4/python directory
- [ ] Install CuPy: `conda install cupy`

### Validation Tests
- [ ] Run CPU version: `python sw_sequential.py --iter 100 --out test_cpu.data`
- [ ] Run GPU version: `python sw_parallel.py --iter 100 --out test_gpu.data`
- [ ] Compare checksums (should match within floating-point precision)
- [ ] Verify GPU version is faster

### Profiling
- [ ] Run nsys profiler: `nsys profile --stats=true python sw_parallel.py --iter 500`
- [ ] Save output to file for report
- [ ] Analyze sections [3/8], [6/8], [7/8]
- [ ] Identify bottlenecks
- [ ] Note kernel timing breakdown

### Weak Scaling Benchmark
- [ ] Run benchmark: `./benchmark_weak_scaling.sh`
- [ ] Wait for completion (~5-10 minutes)
- [ ] Verify `weak_scaling_results.txt` created
- [ ] Generate plot: `python plot_weak_scaling.py`
- [ ] Verify `weak_scaling_plot.png` created
- [ ] Review results and note efficiency

### Asymptotic Performance Benchmark
- [ ] Run benchmark: `./benchmark_asymptotic.sh`
- [ ] Wait for completion (~10-20 minutes)
- [ ] Verify `asymptotic_performance_results.txt` created
- [ ] Generate plot: `python plot_asymptotic.py`
- [ ] Verify `asymptotic_performance_plot.png` created
- [ ] Identify minimum efficient grid size

### Optional: Visualization
- [ ] Open `visualize.ipynb` in Jupyter
- [ ] Load one of your output files
- [ ] Visualize water elevation evolution
- [ ] Include screenshot in report if desired

## 📝 Report Writing Phase (TODO)

### Section 1: Strategy (0.5 pages)
- [ ] Describe CuPy parallelization approach
- [ ] Explain key code modifications
- [ ] Show code snippets for:
  - [ ] Array initialization on GPU
  - [ ] integrate() function with profiling
  - [ ] Memory transfer optimization
- [ ] Discuss synchronization strategy

### Section 2: Profiling Results (1 page)
- [ ] Include nsys command used
- [ ] Show [3/8] NVTX timing output
- [ ] Show [6/8] kernel timing output
- [ ] Show [7/8] memory transfer statistics
- [ ] Analyze where time is spent
- [ ] Identify primary bottlenecks
- [ ] Discuss optimization opportunities

### Section 3: Weak Scaling (0.75 pages)
- [ ] Describe methodology (SMs, grid sizes)
- [ ] Create table of results (SMs vs Time vs Efficiency)
- [ ] Embed `weak_scaling_plot.png`
- [ ] Calculate average efficiency
- [ ] Discuss deviations from ideal scaling
- [ ] Explain causes of overhead

### Section 4: Asymptotic Performance (0.75 pages)
- [ ] Describe methodology (grid sizes tested)
- [ ] Create table of results (Size vs Time vs ns/cell)
- [ ] Embed `asymptotic_performance_plot.png`
- [ ] Identify asymptotic performance value
- [ ] Determine minimum efficient grid size
- [ ] Justify recommendation with data
- [ ] Discuss GPU utilization

### Section 5: Conclusions
- [ ] Summarize performance gains
- [ ] State key insights
- [ ] Suggest future optimizations

### Appendix: Code
- [ ] Include full `sw_parallel.py`
- [ ] Use Absalon template format
- [ ] Highlight key modifications

### Final Checks
- [ ] Page count ≤ 3 pages (excluding code)
- [ ] All plots embedded and readable
- [ ] All tables complete with real data
- [ ] All [X] placeholders replaced
- [ ] Code properly formatted
- [ ] References to profiler sections correct
- [ ] Spelling and grammar checked

## 📤 Submission Phase (TODO)

### Deadline Check
- [ ] Verify deadline: Monday 3/11 23:59
- [ ] Plan submission time (allow buffer)

### File Preparation
- [ ] Report in PDF format
- [ ] All plots embedded
- [ ] Code appendix included
- [ ] File size reasonable (<10MB)

### Submit
- [ ] Upload to Absalon
- [ ] Verify upload successful
- [ ] Keep backup copy
- [ ] Celebrate! 🎉

## Quick Command Reference

```bash
# Navigate to directory
cd /home/bxl776_ku_dk/erda_mount/ahpc/week4/python

# Install CuPy
conda install cupy

# Quick test
python sw_sequential.py --iter 100 --out cpu.data
python sw_parallel.py --iter 100 --out gpu.data

# Profile
nsys profile --stats=true python sw_parallel.py --iter 500 > profile_output.txt

# Weak scaling
./benchmark_weak_scaling.sh
python plot_weak_scaling.py

# Asymptotic
./benchmark_asymptotic.sh
python plot_asymptotic.py

# View results
cat weak_scaling_results.txt
cat asymptotic_performance_results.txt
```

## Expected Timeline

- **Testing & Profiling:** 30 minutes
- **Weak Scaling Benchmark:** 10 minutes
- **Asymptotic Benchmark:** 20 minutes
- **Report Writing:** 3-4 hours
- **Total:** ~5 hours

## Tips for Success

1. **Start on DAG early** - Don't wait until deadline
2. **Save profiler output** - Copy/paste to text files immediately
3. **Take screenshots** - Especially of nsys GUI if you use it
4. **Analyze as you go** - Understand results before writing report
5. **Use real data** - Don't leave placeholder [X] values
6. **Explain, don't just report** - Show understanding of results
7. **Keep it concise** - 3 pages is tight, every word counts

## Success Indicators

✅ **Implementation working if:**
- Checksums match between CPU and GPU
- GPU version runs faster (10-100x for large grids)
- No CUDA errors

✅ **Profiling successful if:**
- nsys completes without errors
- Can see NVTX ranges in [3/8] section
- Can identify kernel timing in [6/8]

✅ **Weak scaling good if:**
- Efficiency > 70% on average
- Time roughly constant as SMs increase
- Can explain deviations

✅ **Asymptotic analysis good if:**
- Clear trend from high to low ns/cell
- Can identify "knee" point
- Performance stabilizes for large grids

✅ **Report ready if:**
- All sections complete
- All data is real (no [X] placeholders)
- ≤ 3 pages excluding code
- Plots embedded and readable

## Need Help?

All documentation files have detailed information:
- **Implementation details:** `README.md`
- **How to run:** `TESTING_GUIDE.md`
- **Report structure:** `REPORT_TEMPLATE.md`
- **Overview:** `SUMMARY.md`

## Current Status

✅ **Implementation:** COMPLETE  
⏳ **Testing:** Awaiting DAG access  
⏳ **Benchmarking:** Awaiting DAG access  
⏳ **Report:** Awaiting test results  
⏳ **Submission:** Deadline Monday 3/11 23:59

---

**You're fully prepared! Everything is implemented and documented. Just need to run on DAG and write the report with your actual results. Good luck! 🚀**
