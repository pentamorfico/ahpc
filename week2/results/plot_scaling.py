#!/usr/bin/env python3
"""Generate runtime and speedup plots from week2/results/data/bench_summary.csv

Produces PNG files in week2/results/plots/
"""
import os
import csv
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_CSV = os.path.join(os.path.dirname(__file__), 'data', 'bench_summary.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(DATA_CSV):
    print(f'No data CSV found at {DATA_CSV}. Run parse_slurm_outputs.py first.')
    raise SystemExit(1)

# read CSV and aggregate by ntasks, pick best (min) runtime per nt
by_nt = defaultdict(list)
with open(DATA_CSV, 'r') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        try:
            nt = int(row['ntasks'])
        except Exception:
            continue
        rt = row.get('runtime_s', '')
        try:
            rt = float(rt) if rt != '' else None
        except Exception:
            rt = None
        status = row.get('status', '')
        if rt is not None and status == 'OK':
            by_nt[nt].append(rt)

nts = sorted(by_nt.keys())
if not nts:
    print('No successful runtimes found in CSV')
    raise SystemExit(1)

best_rt = {nt: min(by_nt[nt]) for nt in nts}

# runtime plot
plt.figure()
plt.plot(nts, [best_rt[nt] for nt in nts], marker='o')
plt.xlabel('Number of MPI ranks (ntasks)')
plt.ylabel('Best runtime (s)')
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.title('Runtime scaling')
out_runtime = os.path.join(PLOTS_DIR, 'runtime.png')
plt.savefig(out_runtime, bbox_inches='tight', dpi=150)
print(f'Wrote {out_runtime}')

# speedup plot: speedup relative to smallest nt
base_nt = min(nts)
base_rt = best_rt[base_nt]
speedups = [base_rt / best_rt[nt] for nt in nts]
plt.figure()
plt.plot(nts, speedups, marker='o')
plt.xlabel('Number of MPI ranks (ntasks)')
plt.ylabel('Speedup')
plt.xscale('log', base=2)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.title(f'Speedup relative to {base_nt} ranks')
out_speedup = os.path.join(PLOTS_DIR, 'speedup.png')
plt.savefig(out_speedup, bbox_inches='tight', dpi=150)
print(f'Wrote {out_speedup}')
