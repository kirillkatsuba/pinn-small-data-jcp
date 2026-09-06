#!/usr/bin/env bash
# Bash 3.2 treats an empty array expansion as unbound under `set -u`.
# Keep the launcher portable while the submitted jobs themselves remain strict.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
JCP_CONDA_ENV="${JCP_CONDA_ENV:-pinn-small-data-jcp}"
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
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV}" \
  slurm/generate_data.sbatch)
data_job="${data_submission%%;*}"

pinn_submission=$(sbatch --parsable "${common_args[@]}" "${gpu_args[@]}" \
  --dependency="afterok:${data_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV}" \
  slurm/run_pinn_array.sbatch)
pinn_job="${pinn_submission%%;*}"

classical_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --dependency="afterok:${pinn_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV}" \
  slurm/run_classical_array.sbatch)
classical_job="${classical_submission%%;*}"

merge_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --dependency="afterok:${classical_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV}" \
  slurm/merge_results.sbatch)
merge_job="${merge_submission%%;*}"

printf 'data_job=%s\npinn_array=%s\nclassical_array=%s\nmerge_job=%s\n' \
  "${data_job}" "${pinn_job}" "${classical_job}" "${merge_job}"
