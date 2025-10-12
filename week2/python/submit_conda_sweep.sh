#!/usr/bin/env bash
# Submit a sweep of SLURM jobs using conda environment to run Python
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
TEMPLATE="$ROOT/job_conda.sh"
OUTDIR="$ROOT/.."/results
mkdir -p "$OUTDIR"

NTASKS=(2 4 8 16 32 64)
JOBMAP_FILE="$OUTDIR/slurm_conda_jobs_map.txt"
: > "$JOBMAP_FILE"

for nt in "${NTASKS[@]}"; do
    jobfile="$ROOT/job_conda_ntasks_${nt}.sh"
    awk -v nt="$nt" '{ if ($0 ~ /^#SBATCH --ntasks=/) { print "#SBATCH --ntasks=" nt } else print $0 }' "$TEMPLATE" > "$jobfile"
    chmod +x "$jobfile"
    echo "Submitting job for ntasks=$nt -> $jobfile"
    sbatch_out=$(sbatch "$jobfile") || { echo "sbatch failed for $jobfile"; cat "$jobfile"; exit 1; }
    echo "$sbatch_out"
    jid=$(echo "$sbatch_out" | awk '{print $NF}')
    echo "$nt,$jid" >> "$JOBMAP_FILE"
    sleep 1
done

# simple poll
sleep 5
while :; do
    running=0
    while IFS=, read -r nt jid; do
        if squeue -j "$jid" -h | grep -q .; then
            running=$((running+1))
        fi
    done < "$JOBMAP_FILE"
    if [ $running -eq 0 ]; then
        echo "All jobs finished"
        break
    fi
    echo "$running jobs still running; sleeping 15s..."
    sleep 15
done

# move slurm outputs to results
dstdir="$OUTDIR"
while IFS=, read -r nt jid; do
    possible="slurm-${jid}.out"
    outname="$dstdir/slurm_conda_${nt}_${jid}.out"
    if [ -f "$possible" ]; then
        mv -f "$possible" "$outname"
        echo "Saved $possible -> $outname"
    else
        echo "No $possible found; leaving note in $outname"
        echo "No $possible found; check scontrol for job $jid" > "$outname"
    fi
done < "$JOBMAP_FILE"

echo "Done. Collected outputs in $dstdir"
