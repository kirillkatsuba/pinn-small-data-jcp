#!/usr/bin/env bash
#SBATCH --job-name=jcp-pinn-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --array=0-9
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"
source slurm/activate_conda.sh

export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Every model is small; keep CPU helper threads from competing with each other.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SEEDS=(101 102 103 104 105 106 107 108 109 110)
BUDGETS=(20 50 100 200 500 1000)
BALANCER_SPEC="${JCP_PINN_BALANCERS:-h,ho}"
IFS=',' read -r -a BALANCERS <<< "${BALANCER_SPEC}"
TEST_NAME="${JCP_TEST:?JCP_TEST must be test1, test2, test3, or test4}"

case "${TEST_NAME}" in
  test1|test2|test3|test4) ;;
  *)
    echo "JCP_TEST must be test1, test2, test3, or test4; got: ${TEST_NAME}" >&2
    exit 2
    ;;
esac

if (( ${#BALANCERS[@]} == 0 )); then
  echo "JCP_PINN_BALANCERS must select vanilla, h, and/or ho" >&2
  exit 2
fi
for balancer in "${BALANCERS[@]}"; do
  case "${balancer}" in
    vanilla|h|ho) ;;
    *)
      echo "Unsupported balancer '${balancer}'; use vanilla, h, and/or ho" >&2
      exit 2
      ;;
  esac
done

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "Invalid array index: ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
EPOCHS="${JCP_EPOCHS:-50000}"
MODELS_PER_GPU="${JCP_MODELS_PER_GPU:-4}"

if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JCP_EPOCHS must be a positive integer, got: ${EPOCHS}" >&2
  exit 2
fi
if ! [[ "${MODELS_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || (( MODELS_PER_GPU > 12 )); then
  echo "JCP_MODELS_PER_GPU must be an integer from 1 to 12, got: ${MODELS_PER_GPU}" >&2
  exit 2
fi

python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Slurm allocated the job, but PyTorch cannot see CUDA")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}")
PY

MODEL_LOG_DIR="slurm/logs/${TEST_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MODEL_LOG_DIR}"

echo "${TEST_NAME}; seed=${SEED}; epochs=${EPOCHS}; modes=${BALANCER_SPEC}; parallel models/GPU=${MODELS_PER_GPU}"

pids=()
labels=()
failed=0

wait_batch() {
  local index
  for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
      echo "[done] ${labels[${index}]}"
    else
      echo "[failed] ${labels[${index}]} (see ${MODEL_LOG_DIR}/${labels[${index}]}.log)" >&2
      failed=1
    fi
  done
  pids=()
  labels=()
}

for budget in "${BUDGETS[@]}"; do
  for balancer in "${BALANCERS[@]}"; do
    label="n${budget}_${balancer}"
    echo "[start] ${label}"
    python3 -B paper_jcp/run_publication_suite.py \
      --test "${TEST_NAME}" \
      --part pinn \
      --device cuda:0 \
      --seeds "${SEED}" \
      --budgets "${budget}" \
      --balancers "${balancer}" \
      --epochs "${EPOCHS}" \
      --result-suffix "seed${SEED}_${label}" \
      --workers 1 \
      >"${MODEL_LOG_DIR}/${label}.log" 2>&1 &
    pids+=("$!")
    labels+=("${label}")

    if (( ${#pids[@]} == MODELS_PER_GPU )); then
      wait_batch
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  wait_batch
fi

if (( failed != 0 )); then
  exit 1
fi

echo "All ${TEST_NAME} PINN runs finished for seed ${SEED}."
