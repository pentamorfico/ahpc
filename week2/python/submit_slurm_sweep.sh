#!/usr/bin/env bash
# Submit apptainer-based sweep jobs for NT up to 512 (keeps original style)
OUTDIR=/home/bxl776_ku_dk/modi_mount/ahpc/week2/results
mkdir -p "$OUTDIR"

# allow easy overrides from environment or CLI-exported vars
NTASKS_LIST=(2 4 8 16 32 64 128 256 512)
NCUTS=${NCUTS:-3}
REPEAT=${REPEAT:-5}
CONTAINER=${CONTAINER:-~/modi_images/slurm-notebook-23.11.10.sif}
PY_IN_CONTAINER=${PY_IN_CONTAINER:-/opt/conda/envs/python3/bin/python}
PY_SCRIPT=${PY_SCRIPT:-task_farm_HEP.py}
CORES_PER_NODE=${CORES_PER_NODE:-64}

# Optional SLURM account (set env ACCOUNT=your_account to include SBATCH --account)
ACCOUNT=${ACCOUNT:-}

echo "Submit sweep with NTs=${NTASKS_LIST[*]} (n_cuts=${NCUTS}, repeat=${REPEAT})"

JOBIDS_FILE=${OUTDIR}/sweep_jobids.txt
: > "$JOBIDS_FILE"

for NT in "${NTASKS_LIST[@]}"; do
  nodes_needed=$(( (NT + CORES_PER_NODE - 1) / CORES_PER_NODE ))
  JOBFILE=week2/tmp/job_apptainer_nt${NT}_multi.sh
  # choose partition/time policy: use modi_short for small NT that previously timed out
  if [[ ${NT} -eq 2 || ${NT} -eq 4 || ${NT} -eq 8 ]]; then
    PARTITION=${PARTITION:-modi_short}
    TIME_LIMIT=${TIME_LIMIT:-00:10:00}
  else
    PARTITION=${PARTITION:-modi_HPPC}
    # set larger limits for larger runs
    if [ ${NT} -le 64 ]; then
      TIME_LIMIT=${TIME_LIMIT:-00:30:00}
    elif [ ${NT} -le 128 ]; then
      TIME_LIMIT=${TIME_LIMIT:-00:45:00}
    elif [ ${NT} -le 256 ]; then
      TIME_LIMIT=${TIME_LIMIT:-01:00:00}
    else
      TIME_LIMIT=${TIME_LIMIT:-02:00:00}
    fi
  fi

  cat > "$JOBFILE" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=Apptainer_NT${NT}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${nodes_needed}
#SBATCH --ntasks=${NT}
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --output=${OUTDIR}/slurm-apptainer-nt${NT}-%j.out
#SBATCH --error=${OUTDIR}/slurm-apptainer-nt${NT}-%j.err
EOF

  # append optional account directive
  if [ -n "$ACCOUNT" ]; then
    echo "#SBATCH --account=${ACCOUNT}" >> "$JOBFILE"
  fi

  cat >> "$JOBFILE" <<EOF

cd /home/bxl776_ku_dk/modi_mount/ahpc/week2/python || exit 1
mpiexec apptainer exec "$CONTAINER" "$PY_IN_CONTAINER" "$PY_SCRIPT" --n_cuts ${NCUTS} --repeat ${REPEAT}
EOF
  chmod +x "$JOBFILE"
  echo "Submitting NT=${NT} nodes=${nodes_needed} -> $JOBFILE"
  SBATCH_OUT=$(sbatch "$JOBFILE" 2>&1) || { echo "sbatch failed: $SBATCH_OUT"; continue; }
  echo "$SBATCH_OUT"
  # extract job id if present
  JOBID=$(echo "$SBATCH_OUT" | awk '{print $4}')
  if [[ "$JOBID" =~ ^[0-9]+$ ]]; then
    echo "$JOBID,$NT" >> "$JOBIDS_FILE"
  fi
  sleep 0.5
done

echo "Submitted sweep done. Job IDs written to ${JOBIDS_FILE}. Check ${OUTDIR} for slurm outputs and results." 
