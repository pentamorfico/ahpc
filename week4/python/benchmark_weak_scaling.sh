#!/bin/bash
# Weak Scaling Benchmark for Shallow Water Simulation
# Tests performance across different numbers of streaming multiprocessors (SMs)

echo "======================================"
echo "Weak Scaling Benchmark"
echo "======================================"
echo ""

# Output file for results
OUTPUT_FILE="weak_scaling_results.txt"
echo "SM_Count,Grid_NX,Grid_NY,Total_Cells,Time_sec,Checksum" > $OUTPUT_FILE

# Define grid sizes that scale with SM count
# Strategy: Keep work per SM constant
# Base: 256x256 = 65,536 cells per 2 SMs = 32,768 cells/SM
declare -A GRID_SIZES
GRID_SIZES[2]="256 256"
GRID_SIZES[4]="362 362"    # ~2x cells
GRID_SIZES[6]="443 443"    # ~3x cells
GRID_SIZES[8]="512 512"    # ~4x cells
GRID_SIZES[10]="574 574"   # ~5x cells
GRID_SIZES[12]="627 627"   # ~6x cells
GRID_SIZES[14]="677 677"   # ~7x cells

# Backup original sw_parallel.py
cp sw_parallel.py sw_parallel_backup.py

echo "Running weak scaling tests..."
echo ""

for SM_COUNT in 2 4 6 8 10 12 14; do
    GRID_PAIR=${GRID_SIZES[$SM_COUNT]}
    NX=$(echo $GRID_PAIR | cut -d' ' -f1)
    NY=$(echo $GRID_PAIR | cut -d' ' -f2)
    TOTAL_CELLS=$((NX * NY))
    
    echo "Testing with $SM_COUNT SMs, Grid: ${NX}x${NY} (${TOTAL_CELLS} cells)"
    
    # Modify sw_parallel.py to use the correct grid size
    sed -i "s/^NX = .*/NX = $NX/" sw_parallel.py
    sed -i "s/^NY = .*/NY = $NY/" sw_parallel.py
    
    # Run the benchmark using run_sw.sh
    OUTPUT=$(./run_sw.sh $SM_COUNT 0 2>&1)
    
    # Extract timing and checksum
    TIME=$(echo "$OUTPUT" | grep "elapsed time:" | awk '{print $3}')
    CHECKSUM=$(echo "$OUTPUT" | grep "checksum:" | awk '{print $2}')
    
    echo "  Time: ${TIME}s, Checksum: $CHECKSUM"
    echo "$SM_COUNT,$NX,$NY,$TOTAL_CELLS,$TIME,$CHECKSUM" >> $OUTPUT_FILE
    echo ""
done

# Restore original
mv sw_parallel_backup.py sw_parallel.py

echo "======================================"
echo "Weak scaling results saved to: $OUTPUT_FILE"
echo "======================================"
echo ""
echo "Summary:"
cat $OUTPUT_FILE
