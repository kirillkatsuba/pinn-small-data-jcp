#!/usr/bin/env bash
#SBATCH --job-name=jcp-classical
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --time=00:20:00
#SBATCH --array=0-39
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
source slurm/activate_conda.sh

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TESTS=(test1 test2 test3 test4)
SEEDS=(101 102 103 104 105 106 107 108 109 110)
TEST_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
TEST_NAME="${TESTS[${TEST_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"

python3 -B paper_jcp/run_publication_suite.py \
  --test "${TEST_NAME}" \
  --part classical \
  --device cpu \
  --seeds "${SEED}" \
  --result-suffix "seed${SEED}" \
  --workers 1
