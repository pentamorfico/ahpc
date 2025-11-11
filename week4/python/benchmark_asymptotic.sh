#!/bin/bash
# Asymptotic Performance Benchmark
# Determines minimum workload size for efficient GPU utilization

echo "======================================"
echo "Asymptotic Performance Benchmark"
echo "======================================"
echo ""

# Output file
OUTPUT_FILE="asymptotic_performance_results.txt"
echo "Grid_NX,Grid_NY,Total_Cells,Time_sec,ns_per_cell,Checksum" > $OUTPUT_FILE

# Backup original
cp sw_parallel.py sw_parallel_backup.py

# Test range of grid sizes from small to large
# Start small to see overhead, go large to find asymptotic behavior
GRID_SIZES=(64 96 128 192 256 384 512 768 1024 1536 2048)

echo "Testing various grid sizes on full GPU (14 SMs)..."
echo "Looking for point where ns/cell becomes constant"
echo ""

for SIZE in "${GRID_SIZES[@]}"; do
    NX=$SIZE
    NY=$SIZE
    TOTAL_CELLS=$((NX * NY))
    
    echo "Testing Grid: ${NX}x${NY} (${TOTAL_CELLS} cells)"
    
    # Modify grid size
    sed -i "s/^NX = .*/NX = $NX/" sw_parallel.py
    sed -i "s/^NY = .*/NY = $NY/" sw_parallel.py
    
    # Run on full GPU (14 SMs, no MPS restriction)
    OUTPUT=$(./run_sw.sh 14 0 2>&1)
    
    # Extract results
    TIME=$(echo "$OUTPUT" | grep "elapsed time:" | awk '{print $3}')
    CHECKSUM=$(echo "$OUTPUT" | grep "checksum:" | awk '{print $2}')
    
    # Calculate ns per cell (time in seconds * 1e9 / cells / iterations)
    # Assuming 1000 iterations (default)
    NS_PER_CELL=$(echo "scale=4; $TIME * 1000000000 / $TOTAL_CELLS / 1000" | bc)
    
    echo "  Time: ${TIME}s, ns/cell: ${NS_PER_CELL}, Checksum: $CHECKSUM"
    echo "$NX,$NY,$TOTAL_CELLS,$TIME,$NS_PER_CELL,$CHECKSUM" >> $OUTPUT_FILE
    echo ""
done

# Restore original
mv sw_parallel_backup.py sw_parallel.py

echo "======================================"
echo "Asymptotic performance results saved to: $OUTPUT_FILE"
echo "======================================"
echo ""
echo "Summary:"
cat $OUTPUT_FILE
echo ""
echo "Look for the grid size where ns/cell stabilizes (becomes constant)"
echo "This indicates the minimum workload for efficient GPU utilization"
