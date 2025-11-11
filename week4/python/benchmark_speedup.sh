#!/bin/bash
# CPU vs GPU Speedup Comparison

echo "======================================"
echo "CPU vs GPU Speedup Benchmark"
echo "======================================"
echo ""

OUTPUT_FILE="cpu_vs_gpu_speedup.txt"
echo "Grid_NX,Grid_NY,Total_Cells,CPU_Time_sec,GPU_Time_sec,Speedup,CPU_Checksum,GPU_Checksum" > $OUTPUT_FILE

# Test different grid sizes
for SIZE in 128 256 512 768 1024; do
    NX=$SIZE
    NY=$SIZE
    TOTAL_CELLS=$((NX * NY))
    
    echo "Testing Grid: ${NX}x${NY} (${TOTAL_CELLS} cells)"
    
    # Backup originals
    cp sw_sequential.py sw_sequential_backup.py
    cp sw_parallel.py sw_parallel_backup.py
    
    # Modify grid sizes
    sed -i "s/^NX = .*/NX = $NX/" sw_sequential.py
    sed -i "s/^NY = .*/NY = $NY/" sw_sequential.py
    sed -i "s/^NX = .*/NX = $NX/" sw_parallel.py
    sed -i "s/^NY = .*/NY = $NY/" sw_parallel.py
    
    # Run CPU version
    echo "  Running CPU version..."
    CPU_OUTPUT=$(python3 sw_sequential.py --iter 500 --out /tmp/cpu_test.data 2>&1)
    CPU_TIME=$(echo "$CPU_OUTPUT" | grep "elapsed time:" | awk '{print $3}')
    CPU_CHECKSUM=$(echo "$CPU_OUTPUT" | grep "checksum:" | awk '{print $2}')
    
    # Run GPU version
    echo "  Running GPU version..."
    GPU_OUTPUT=$(python3 sw_parallel.py --iter 500 --out /tmp/gpu_test.data 2>&1)
    GPU_TIME=$(echo "$GPU_OUTPUT" | grep "elapsed time:" | awk '{print $3}')
    GPU_CHECKSUM=$(echo "$GPU_OUTPUT" | grep "checksum:" | awk '{print $2}')
    
    # Calculate speedup
    SPEEDUP=$(echo "scale=2; $CPU_TIME / $GPU_TIME" | bc)
    
    echo "  CPU: ${CPU_TIME}s, GPU: ${GPU_TIME}s, Speedup: ${SPEEDUP}x"
    echo "$NX,$NY,$TOTAL_CELLS,$CPU_TIME,$GPU_TIME,$SPEEDUP,$CPU_CHECKSUM,$GPU_CHECKSUM" >> $OUTPUT_FILE
    echo ""
    
    # Restore originals
    mv sw_sequential_backup.py sw_sequential.py
    mv sw_parallel_backup.py sw_parallel.py
done

echo "======================================"
echo "Speedup results saved to: $OUTPUT_FILE"
echo "======================================"
echo ""
cat $OUTPUT_FILE
