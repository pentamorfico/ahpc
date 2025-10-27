#!/bin/bash

# Comprehensive OpenMP strategy testing script
# Tests all 6 strategies with thread counts: 1,2,4,8,16,32,64

echo "=== OpenMP Strategy Performance Testing ==="
echo "Date: $(date)"
echo "NFREQ: 4096"
echo ""

# Array of thread counts to test
threads=(1 2 4 8 16 32 64)

# Array of strategy names
strategies=("strategy1" "strategy2" "strategy3" "strategy4" "strategy5" "strategy6")
strategy_names=(
    "Multiple Regions"
    "Fewer Regions" 
    "Task-Based"
    "SIMD Optimized"
    "Advanced Scheduling"
    "Sections-Based"
)

# Test each strategy
for i in "${!strategies[@]}"; do
    strategy="${strategies[$i]}"
    name="${strategy_names[$i]}"
    
    echo "=== Strategy $((i+1)): $name ==="
    echo "Strategy,Threads,Time,Checksum"
    
    for t in "${threads[@]}"; do
        echo -n "Strategy$((i+1)),$t,"
        result=$(OMP_NUM_THREADS=$t ./strategies/$strategy | grep -E "(Elapsed time:|Checksum)")
        time=$(echo "$result" | grep "Elapsed time" | awk '{print $3}')
        checksum=$(echo "$result" | grep "Checksum" | awk '{print $3}')
        echo "$time,$checksum"
    done
    echo ""
done

echo "=== Testing Complete ==="