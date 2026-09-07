#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
JCP_CONDA_ENV="${JCP_CONDA_ENV:-pinn-small-data-jcp}"
JCP_MODELS_PER_GPU="${JCP_MODELS_PER_GPU:-4}"
JCP_EPOCHS="${JCP_EPOCHS:-50000}"
mkdir -p "${PROJECT_DIR}/slurm/logs"

common_args=()
gpu_args=()
cpu_args=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  common_args+=("--account=${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_GPU_PARTITION:-}" ]]; then
  gpu_args+=("--partition=${SLURM_GPU_PARTITION}")
fi
if [[ -n "${SLURM_CPU_PARTITION:-}" ]]; then
  cpu_args+=("--partition=${SLURM_CPU_PARTITION}")
fi
if [[ -n "${JCP_GPU_COUNT:-}" ]]; then
  gpu_args+=("--gpus=${JCP_GPU_COUNT}")
fi
if [[ -n "${JCP_PINN_TIME:-}" ]]; then
  gpu_args+=("--time=${JCP_PINN_TIME}")
fi

export_values="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV},JCP_MODELS_PER_GPU=${JCP_MODELS_PER_GPU},JCP_EPOCHS=${JCP_EPOCHS}"

pinn_submission=$(sbatch --parsable "${common_args[@]}" "${gpu_args[@]}" \
  --export="${export_values}" slurm/run_test4_pinn_array.sbatch)
pinn_job="${pinn_submission%%;*}"

merge_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --dependency="afterok:${pinn_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV}" \
  slurm/merge_test4_pinn.sbatch)
merge_job="${merge_submission%%;*}"

printf 'test4_pinn_array=%s\ntest4_merge=%s\n' "${pinn_job}" "${merge_job}"
