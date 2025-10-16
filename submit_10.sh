#!/bin/bash

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=faculty            # basic (default), staff, phd, faculty
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Yuan.Zhong@chicagobooth.edu

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard       # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=64         # number of CPUs requested (for parallel tasks)
#SBATCH --mem=32G                  # requested memory
#SBATCH --time=5-10:00:00          # wall clock limit (d-hh:mm:ss)
#SBATCH --ntasks=1
##SBATCH --array=1-10

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=MLP_1D_Test_10   # user-defined job name

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load julia/1.10

#---------------------------------------------------------------------------------
# Commands to execute below...

pwd

julia -e 'using Pkg; Pkg.add("Statistics"); Pkg.add("Distributions"); Pkg.add("Random"); Pkg.add("NonlinearSolve")'

julia --threads=auto 1d_test_mlt2_v2_4.jl

