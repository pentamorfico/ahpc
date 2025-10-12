#!/usr/bin/env bash
# Smoke test for week2/python task farm implementations
# Runs sequential reference and a small parallel run and compares basic outputs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

SEQ_PY=task_farm_HEP_seq.py
PAR_PY=task_farm_HEP.py

INPUT=../mc_ggH_16_13TeV_Zee_EGAM1_calocells_16249871.csv
OUT_SEQ=../results/hep_seq.out
OUT_PAR=../results/hep_par.out

mkdir -p ../results

echo "Running sequential baseline..."
python3 "$SEQ_PY" > "$OUT_SEQ" 2>&1

echo "Running parallel (2 workers)..."
mpirun -n 3 python3 "$PAR_PY" > "$OUT_PAR" 2>&1

# Simple check: both outputs should contain "Best accuracy"
if grep -q "Best accuracy" "$OUT_SEQ" && grep -q "Best accuracy" "$OUT_PAR"; then
    echo "Smoke test passed: 'Best accuracy' found in both outputs"
    exit 0
else
    echo "Smoke test failed: expected output missing"
    exit 2
fi
