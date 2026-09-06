#!/usr/bin/env bash

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available; load the server's Conda module before sbatch" >&2
  exit 1
fi

JCP_CONDA_ENV="${JCP_CONDA_ENV:-pinn-small-data-jcp}"
jcp_conda_base="$(conda info --base)"
# shellcheck disable=SC1091
source "${jcp_conda_base}/etc/profile.d/conda.sh"
conda activate "${JCP_CONDA_ENV}"

echo "Conda environment: ${CONDA_DEFAULT_ENV}"
echo "Python: $(command -v python3)"
