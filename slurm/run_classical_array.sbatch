#!/usr/bin/env bash
#SBATCH --job-name=jcp-classical
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --time=1:00:00
#SBATCH --array=1-4
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
cd "${PROJECT_DIR}"
source "${VENV_PATH}/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TESTS=(test1 test2 test3 test4)
TEST_NAME="${TESTS[$((SLURM_ARRAY_TASK_ID - 1))]}"

python3 -B paper_jcp/run_publication_suite.py \
  --test "${TEST_NAME}" \
  --part classical \
  --device cpu \
  --workers 1
