#!/usr/bin/env python3
"""Parse SLURM .out/.err files from the modi NFS mount and produce a bench_summary.csv

This script expects the SLURM outputs to be located under the cluster-visible path:
  /home/bxl776_ku_dk/modi_mount/ahpc/week2/results

It writes the summary CSV into the workspace path:
  week2/results/data/bench_summary.csv

Columns: jobid, ntasks, runtime_s, n_settings, status, out_path, err_path
"""
import re
import csv
import os
from glob import glob

MOUNT_RESULTS = '/home/bxl776_ku_dk/modi_mount/ahpc/week2/results'
WORK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(WORK_DATA_DIR, exist_ok=True)
OUT_CSV = os.path.join(WORK_DATA_DIR, 'bench_summary.csv')

OUT_GLOB = os.path.join(MOUNT_RESULTS, 'slurm-apptainer-nt*-*.out')
out_files = sorted(glob(OUT_GLOB))

records = []
for out_f in out_files:
    m = re.search(r'slurm-apptainer-nt(\d+)-([0-9]+)\.out$', out_f)
    if not m:
        continue
    nt = int(m.group(1))
    jobid = m.group(2)
    err_f = out_f[:-4] + '.err'

    runtime = None
    n_settings = None
    status = 'UNKNOWN'

    # parse .out for elapsed time and number of settings and other diagnostics
    try:
        with open(out_f, 'r', errors='ignore') as fh:
            txt = fh.read()
            # look for lines like: Elapsed time      :   12.3456 s
            mtime = re.search(r'Elapsed time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s', txt)
            if mtime:
                runtime = float(mtime.group(1))
                status = 'OK'
            mnset = re.search(r'Number of settings\s*:\s*([0-9]+)', txt)
            if mnset:
                n_settings = int(mnset.group(1))
            # also try older prints
            if runtime is None:
                # some versions print 'Elapsed time      :123.456 s' without spacing
                mtime2 = re.search(r'Elapsed time\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s', txt)
                if mtime2:
                    runtime = float(mtime2.group(1))
                    status = 'OK'
    except FileNotFoundError:
        txt = ''

    # inspect .err for TIMEOUT or cancellations
    try:
        with open(err_f, 'r', errors='ignore') as fh:
            errtxt = fh.read()
            if re.search(r'DUE TO TIME LIMIT|CANCELLED AT|TIME LIMIT|TIMEOUT', errtxt, re.IGNORECASE):
                status = 'TIMEOUT'
            # slurm may write 'error' lines; capture them for diagnostics
    except FileNotFoundError:
        errtxt = ''

    records.append({
        'jobid': jobid,
        'ntasks': nt,
        'runtime_s': '' if runtime is None else f"{runtime:.6f}",
        'n_settings': '' if n_settings is None else str(n_settings),
        'status': status,
        'out_path': out_f,
        'err_path': err_f,
    })

# write CSV
fieldnames = ['jobid', 'ntasks', 'runtime_s', 'n_settings', 'status', 'out_path', 'err_path']
with open(OUT_CSV, 'w', newline='') as csvf:
    w = csv.DictWriter(csvf, fieldnames=fieldnames)
    w.writeheader()
    for r in records:
        w.writerow(r)

print(f'Wrote summary for {len(records)} jobs to {OUT_CSV}')
