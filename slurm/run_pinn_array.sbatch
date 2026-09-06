#!/usr/bin/env bash
#SBATCH --job-name=jcp-pinn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=0-39
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
source slurm/activate_conda.sh

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TESTS=(test1 test2 test3 test4)
SEEDS=(101 102 103 104 105 106 107 108 109 110)
TEST_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
TEST_NAME="${TESTS[${TEST_INDEX}]}"
SEED="${SEEDS[${SEED_INDEX}]}"

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated the job, but PyTorch cannot see CUDA")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}")
PY

python3 -B paper_jcp/run_publication_suite.py \
  --test "${TEST_NAME}" \
  --part pinn \
  --device cuda:0 \
  --seeds "${SEED}" \
  --epochs 50000 \
  --result-suffix "seed${SEED}" \
  --workers 1
