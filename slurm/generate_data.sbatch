#!/usr/bin/env bash
#SBATCH --job-name=jcp-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=rocky
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
source slurm/activate_conda.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python3 -B paper_jcp/generate_corrected_benchmarks.py \
  --test all \
  --output-root paper_jcp/generated/publication_data \
  --output-steps 100 \
  --final-time-test2 0.3 \
  --final-time-test3 0.05 \
  --final-time-test4 0.2 \
  --cfl 0.2 \
  --progress summary \
  --workers "${SLURM_CPUS_PER_TASK}"
