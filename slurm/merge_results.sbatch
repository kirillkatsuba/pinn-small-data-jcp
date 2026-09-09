#!/usr/bin/env bash
#SBATCH --job-name=jcp-merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rocky
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
source slurm/activate_conda.sh

export PYTHONUNBUFFERED=1

MERGE_TEST="${JCP_MERGE_TEST:-all}"
MERGE_PART="${JCP_MERGE_PART:-all}"
CLASSICAL_TUNING_PROFILE="${JCP_CLASSICAL_TUNING_PROFILE:-extensive}"
CLASSICAL_CV="${JCP_CLASSICAL_CV:-time}"

python3 -B paper_jcp/merge_publication_shards.py \
  --test "${MERGE_TEST}" \
  --part "${MERGE_PART}" \
  --classical-tuning-profile "${CLASSICAL_TUNING_PROFILE}" \
  --classical-cv "${CLASSICAL_CV}"
