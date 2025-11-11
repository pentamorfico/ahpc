#!/bin/bash
# Strong Scaling Benchmark for Shallow Water Simulation
# Tests performance with fixed problem size but varying iterations

echo "======================================"
echo "Strong Scaling Benchmark"
echo "======================================"
echo ""

# Output file for results
OUTPUT_FILE="strong_scaling_results.txt"
echo "Grid_NX,Grid_NY,Total_Cells,Iterations,Time_sec,Time_per_iter_ms,Checksum" > $OUTPUT_FILE

# Backup original sw_parallel.py
cp sw_parallel.py sw_parallel_backup.py

# Fixed grid size
NX=512
NY=512
TOTAL_CELLS=$((NX * NY))

echo "Testing strong scaling with fixed grid: ${NX}x${NY} (${TOTAL_CELLS} cells)"
echo "Varying number of iterations"
echo ""

# Test different iteration counts
for ITERS in 100 500 1000 2000 5000; do
    echo "Testing with $ITERS iterations"
    
    # Modify sw_parallel.py to use the correct grid size
    sed -i "s/^NX = .*/NX = $NX/" sw_parallel.py
    sed -i "s/^NY = .*/NY = $NY/" sw_parallel.py
    
    # Run the benchmark
    OUTPUT=$(python3 sw_parallel.py --iter $ITERS --out /tmp/scaling_test.data 2>&1)
    
    # Extract timing and checksum
    TIME=$(echo "$OUTPUT" | grep "elapsed time:" | awk '{print $3}')
    CHECKSUM=$(echo "$OUTPUT" | grep "checksum:" | awk '{print $2}')
    
    # Calculate time per iteration in milliseconds
    TIME_PER_ITER=$(echo "scale=4; $TIME * 1000 / $ITERS" | bc)
    
    echo "  Time: ${TIME}s, Time/iter: ${TIME_PER_ITER}ms, Checksum: $CHECKSUM"
    echo "$NX,$NY,$TOTAL_CELLS,$ITERS,$TIME,$TIME_PER_ITER,$CHECKSUM" >> $OUTPUT_FILE
    echo ""
done

# Restore original
mv sw_parallel_backup.py sw_parallel.py

echo "======================================"
echo "Strong scaling results saved to: $OUTPUT_FILE"
echo "======================================"
echo ""
echo "Summary:"
cat $OUTPUT_FILE
