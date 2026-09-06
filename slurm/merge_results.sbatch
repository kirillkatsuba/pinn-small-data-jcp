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

python3 -B paper_jcp/merge_publication_shards.py
