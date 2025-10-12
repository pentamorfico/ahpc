#!/usr/bin/env bash
# Simple benchmark script for week2/python task farm
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

PY=task_farm_HEP.py
INPUT=../mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv
OUTDIR=../results
mkdir -p "$OUTDIR"

# worker counts to try (total ranks = workers + 1 for master)
WORKERS=(1 2 4 8)

for w in "${WORKERS[@]}"; do
    ranks=$((w + 1))
    outfile="$OUTDIR/bench_workers_${w}.txt"
    echo "Running with $w workers (ranks: $ranks) -> $outfile"
    # Use shell time to avoid dependency on /usr/bin/time. Capture both stdout and stderr to the outfile.
    { time mpirun -n "$ranks" python3 "$PY" ; } >"$outfile" 2>&1 || true
    echo "--- done: $outfile ---"
done

echo "Bench results in $OUTDIR"
