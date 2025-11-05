#!/usr/bin/env bash
#SBATCH --job-name=FWC
#SBATCH --partition=modi_HPPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
##SBATCH --exclusive

mpiexec -- apptainer exec \
   ~/modi_images/slurm-notebook-23.11.10.sif \
   ./fwc_parallel --iter 1000 --model ../models/small.hdf5
