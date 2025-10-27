#!/bin/bash
#SBATCH --job-name=seismic_weak_scaling_multi
#SBATCH --account=ahpc  
#SBATCH --partition=modi_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/home/bxl776_ku_dk/modi_mount/ahpc/week3/results/weak_scaling_multi_%j.out
#SBATCH --error=/home/bxl776_ku_dk/modi_mount/ahpc/week3/results/weak_scaling_multi_%j.err

# Week 3 Task 2: Comprehensive Weak Scaling Analysis
# Tests multiple OpenMP strategies with weak scaling
# Problem size scales: nfreq = 65536 * ncore

echo "=== Comprehensive SLURM Weak Scaling Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Testing strategies: 1(Multiple), 2(Fewer), 3(Tasks), 4(SIMD), 5(Scheduling), 6(Sections)"
echo ""

cd /home/bxl776_ku_dk/modi_mount/ahpc/week3

# Thread counts and strategies to test
threads=(1 2 4 8 16 32 64)
strategies=("strategy1" "strategy2" "strategy3" "strategy4" "strategy5" "strategy6")
strategy_names=("Multiple_Regions" "Fewer_Regions" "Task_Based" "SIMD_Optimized" "Advanced_Scheduling" "Sections_Based")

echo "Strategy,Strategy_Name,Threads,NFREQ,Runtime,Checksum,Efficiency"

for i in "${!strategies[@]}"; do
    strategy="${strategies[$i]}"
    name="${strategy_names[$i]}"
    
    echo "=== Testing Strategy $((i+1)): $name ==="
    
    # Base runtime for efficiency calculation
    base_runtime=""
    
    for nt in "${threads[@]}"; do
        nfreq=$((65536 * nt))
        echo "  Building and testing $nt threads with NFREQ=$nfreq"
        
        # Build with specific NFREQ
        g++ -O3 -fopenmp -DNFREQ=$nfreq strategies/${strategy}.cpp -o strategies/${strategy}_nfreq${nfreq} 2>/dev/null
        
        if [ $? -eq 0 ]; then
            # Run the test
            export OMP_NUM_THREADS=$nt
            result=$(timeout 300 ./strategies/${strategy}_nfreq${nfreq} 2>/dev/null | grep -E "(Elapsed time:|Checksum)")
            
            if [ $? -eq 0 ] && [ -n "$result" ]; then
                # Extract runtime and checksum
                runtime=$(echo "$result" | grep "Elapsed time" | awk '{print $3}')
                checksum=$(echo "$result" | grep "Checksum" | awk '{print $3}')
                
                # Calculate efficiency (ideal weak scaling = 1.0)
                if [ "$nt" = "1" ]; then
                    base_runtime=$runtime
                    efficiency="1.00"
                else
                    if [ -n "$base_runtime" ] && [ "$base_runtime" != "0" ]; then
                        efficiency=$(awk -v base="$base_runtime" -v curr="$runtime" 'BEGIN{printf "%.3f", base/curr}')
                    else
                        efficiency="N/A"
                    fi
                fi
                
                echo "$((i+1)),$name,$nt,$nfreq,$runtime,$checksum,$efficiency"
            else
                echo "$((i+1)),$name,$nt,$nfreq,TIMEOUT,N/A,N/A"
            fi
            
            # Clean up binary
            rm -f strategies/${strategy}_nfreq${nfreq}
        else
            echo "$((i+1)),$name,$nt,$nfreq,BUILD_FAILED,N/A,N/A"
        fi
    done
    echo ""
done

echo "=== Comprehensive Weak Scaling Test Complete ==="
echo "Ideal weak scaling efficiency = 1.0 (constant runtime)"
echo "Job completed on $(date)"