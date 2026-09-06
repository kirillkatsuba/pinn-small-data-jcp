#!/usr/bin/env bash
#SBATCH --job-name=jcp-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

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
  --workers "${SLURM_CPUS_PER_TASK}"
