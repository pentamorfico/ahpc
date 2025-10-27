#!/bin/bash
#SBATCH --job-name=seismic_weak_scaling
#SBATCH --account=ahpc  
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/home/bxl776_ku_dk/modi_mount/ahpc/week3/results/weak_scaling_nt%j.out
#SBATCH --error=/home/bxl776_ku_dk/modi_mount/ahpc/week3/results/weak_scaling_nt%j.err

# Week 3 Task 2: Weak Scaling Analysis
# Problem size scales with thread count: nfreq = 65536 * ncore
# Tests thread counts: 1, 2, 4, 8, 16, 32, 64

echo "=== SLURM Weak Scaling Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Strategy: Best performing (Strategy 2 - Fewer Regions)"
echo ""

# Build the strategy with different NFREQ values
cd /home/bxl776_ku_dk/modi_mount/ahpc/week3

# Thread counts to test
threads=(1 2 4 8 16 32 64)

echo "Thread,NFREQ,Runtime,Checksum,Speedup"

# Base runtime for speedup calculation
base_runtime=""

for nt in "${threads[@]}"; do
    nfreq=$((65536 * nt))
    echo "Testing $nt threads with NFREQ=$nfreq"
    
    # Build with specific NFREQ
    g++ -O3 -fopenmp -DNFREQ=$nfreq strategies/strategy2.cpp -o strategies/strategy2_nfreq${nfreq}
    
    # Run the test
    export OMP_NUM_THREADS=$nt
    result=$(./strategies/strategy2_nfreq${nfreq} | grep -E "(Elapsed time:|Checksum)")
    
    # Extract runtime and checksum
    runtime=$(echo "$result" | grep "Elapsed time" | awk '{print $3}')
    checksum=$(echo "$result" | grep "Checksum" | awk '{print $3}')
    
    # Calculate speedup relative to 1 thread
    if [ "$nt" = "1" ]; then
        base_runtime=$runtime
        speedup="1.00"
    else
        speedup=$(echo "scale=2; $base_runtime / $runtime" | bc -l)
    fi
    
    echo "$nt,$nfreq,$runtime,$checksum,$speedup"
    
    # Clean up binary
    rm strategies/strategy2_nfreq${nfreq}
done

echo ""
echo "=== Weak Scaling Test Complete ==="
echo "Expected behavior: Constant runtime for ideal weak scaling"
echo "Job completed on $(date)"