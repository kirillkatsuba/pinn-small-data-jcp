#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv}"
mkdir -p "${PROJECT_DIR}/slurm/logs"

common_args=()
cpu_args=()
gpu_args=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  common_args+=("--account=${SLURM_ACCOUNT}")
fi
if [[ -n "${SLURM_CPU_PARTITION:-}" ]]; then
  cpu_args+=("--partition=${SLURM_CPU_PARTITION}")
fi
if [[ -n "${SLURM_GPU_PARTITION:-}" ]]; then
  gpu_args+=("--partition=${SLURM_GPU_PARTITION}")
fi
if [[ -n "${JCP_GPU_COUNT:-}" ]]; then
  gpu_args+=("--gpus=${JCP_GPU_COUNT}")
fi

data_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},VENV_PATH=${VENV_PATH}" \
  slurm/generate_data.sbatch)
data_job="${data_submission%%;*}"

pinn_submission=$(sbatch --parsable "${common_args[@]}" "${gpu_args[@]}" \
  --dependency="afterok:${data_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},VENV_PATH=${VENV_PATH}" \
  slurm/run_pinn_array.sbatch)
pinn_job="${pinn_submission%%;*}"

classical_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --dependency="afterok:${pinn_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},VENV_PATH=${VENV_PATH}" \
  slurm/run_classical_array.sbatch)
classical_job="${classical_submission%%;*}"

printf 'data_job=%s\npinn_array=%s\nclassical_array=%s\n' \
  "${data_job}" "${pinn_job}" "${classical_job}"
