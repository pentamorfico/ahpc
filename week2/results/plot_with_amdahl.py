#!/usr/bin/env python3
import csv
import os
import math
from statistics import mean, median, pstdev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_CSV = os.path.join(os.path.dirname(__file__), 'data', 'bench_summary.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
OUT_TCSV = os.path.join(os.path.dirname(__file__), 'data', 't_task.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

rows = []
with open(DATA_CSV) as fh:
    r = csv.DictReader(fh)
    for row in r:
        try:
            nt = int(row['ntasks'])
            rt = float(row['runtime_s']) if row['runtime_s']!='' else None
            n_settings = int(row['n_settings']) if row.get('n_settings') else None
            status = row.get('status','')
        except Exception:
            continue
        rows.append({'nt':nt,'rt':rt,'n_settings':n_settings,'status':status})

rows = sorted(rows, key=lambda x: x['nt'])
if not rows:
    raise SystemExit('no data')

N_task = rows[0]['n_settings'] if rows[0]['n_settings'] else 6561
# compute T_task = T_wall * N_worker / N_task (workers = nt-1)
for r in rows:
    workers = r['nt'] - 1
    if r['rt'] is None or workers < 1:
        r['t_task'] = None
    else:
        r['t_task'] = r['rt'] * workers / N_task

# write T_task CSV
with open(OUT_TCSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['ntasks','workers','T_wall_s','T_task_s','n_settings','status'])
    for r in rows:
        w.writerow([r['nt'], r['nt']-1, r['rt'] if r['rt'] is not None else '',
                    f"{r['t_task']:.9f}" if r['t_task'] is not None else '', r['n_settings'] if r['n_settings'] else '', r['status']])
print('Wrote', OUT_TCSV)

# prepare speedup data and Amdahl estimate
base = None
for r in rows:
    if r['nt']==2:
        base = r['rt']
        break
if base is None:
    raise SystemExit('no baseline')

nts = []
workers = []
rtvals = []
speedups = []
amdahl_s = []
for r in rows:
    if r['rt'] is None: continue
    nt = r['nt']
    P = nt - 1
    if P < 1: continue
    s = 0.0
    sp = base / r['rt']
    if P > 1:
        s = (1.0/sp - 1.0/P) / (1.0 - 1.0/P)
    nts.append(nt)
    workers.append(P)
    rtvals.append(r['rt'])
    speedups.append(sp)
    amdahl_s.append(s)

mean_s = mean([x for x in amdahl_s if math.isfinite(x)])
med_s = median([x for x in amdahl_s if math.isfinite(x)])
print('Estimated s mean', mean_s, 'median', med_s)

# theoretical curve
P_range = list(range(1, max(workers)+1))
theo = [1.0/(mean_s + (1-mean_s)/p) for p in P_range]

# plot speedup with theoretical curve
plt.figure()
plt.plot(workers, speedups, 'o-', label='measured')
plt.plot(P_range, theo, '--', label=f'Amdahl (s={mean_s:.4g})')
plt.xlabel('Workers (MPI ranks - 1)')
plt.ylabel('Speedup (relative to 2 ranks)')
plt.xscale('log', base=2)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
outp = os.path.join(PLOTS_DIR, 'speedup_with_amdahl.png')
plt.savefig(outp, bbox_inches='tight', dpi=150)
print('Wrote', outp)

# copy to reports folder
import shutil
reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
if os.path.isdir(reports_dir):
    shutil.copy(outp, os.path.join(reports_dir, 'speedup_with_amdahl.png'))
    print('Copied plot to reports folder')
