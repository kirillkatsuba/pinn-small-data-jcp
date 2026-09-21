#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
JCP_CONDA_ENV="${JCP_CONDA_ENV:-pinn-small-data-jcp}"
JCP_MODELS_PER_GPU="${JCP_MODELS_PER_GPU:-4}"
JCP_EPOCHS="${JCP_EPOCHS:-50000}"
JCP_PINN_BALANCERS="${JCP_PINN_BALANCERS:-h,ho}"
JCP_CHECKPOINT_EPOCHS="${JCP_CHECKPOINT_EPOCHS:-}"
JCP_EXPERIMENT_TAG="${JCP_EXPERIMENT_TAG:-}"
JCP_SCHEDULER_EPOCHS="${JCP_SCHEDULER_EPOCHS:-}"
TEST_NAME="${1:-${JCP_TEST:-}}"

case "${TEST_NAME}" in
  test1|test2|test3|test4) ;;
  *)
    echo "Usage: bash slurm/submit_pinn_test.sh test1|test2|test3|test4" >&2
    exit 2
    ;;
esac
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

export_values="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV},JCP_TEST=${TEST_NAME},JCP_MODELS_PER_GPU=${JCP_MODELS_PER_GPU},JCP_EPOCHS=${JCP_EPOCHS},JCP_PINN_BALANCERS=${JCP_PINN_BALANCERS},JCP_CHECKPOINT_EPOCHS=${JCP_CHECKPOINT_EPOCHS},JCP_EXPERIMENT_TAG=${JCP_EXPERIMENT_TAG},JCP_SCHEDULER_EPOCHS=${JCP_SCHEDULER_EPOCHS}"

pinn_submission=$(sbatch --parsable "${common_args[@]}" "${gpu_args[@]}" \
  --job-name="jcp-${TEST_NAME}-pinn" \
  --export="${export_values}" slurm/run_pinn_test_array.sbatch)
pinn_job="${pinn_submission%%;*}"

merge_submission=$(sbatch --parsable "${common_args[@]}" "${cpu_args[@]}" \
  --job-name="jcp-${TEST_NAME}-merge" \
  --dependency="afterok:${pinn_job}" \
  --export="ALL,PROJECT_DIR=${PROJECT_DIR},JCP_CONDA_ENV=${JCP_CONDA_ENV},JCP_TEST=${TEST_NAME},JCP_EXPERIMENT_TAG=${JCP_EXPERIMENT_TAG}" \
  slurm/merge_pinn_test.sbatch)
merge_job="${merge_submission%%;*}"

printf '%s_pinn_array=%s\n%s_merge=%s\n' \
  "${TEST_NAME}" "${pinn_job}" "${TEST_NAME}" "${merge_job}"
