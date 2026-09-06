# Slurm execution

The jobs are split into three stages:

1. `generate_data.sbatch` generates the deterministic finite-volume datasets
   for Tests 2--4 on a CPU node.
2. `run_pinn_array.sbatch` launches four independent GPU tasks, one per test.
3. `run_classical_array.sbatch` launches four independent CPU tasks, one per
   test, after the PINN array succeeds. This ordering prevents PINN and
   classical processes from writing the same per-test CSV simultaneously.

No site-specific account, partition, or module names are hard-coded. Supply
them to `sbatch` or add the corresponding `#SBATCH` lines for your cluster.

## Environment

From the cloned repository on the cluster:

```bash
# Adapt these module names to the cluster.
module purge
module load python/3.13
module load cuda/12.4

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-cuda124.txt
```

Verify CUDA on a GPU allocation:

```bash
srun --gres=gpu:1 --time=00:10:00 \
  .venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Submit the complete pipeline

Run from the repository root:

```bash
bash slurm/submit_pipeline.sh
```

Or pass common site settings without editing the scripts:

```bash
SLURM_ACCOUNT=MY_ACCOUNT \
SLURM_CPU_PARTITION=cpu \
SLURM_GPU_PARTITION=gpu \
SLURM_GPU_GRES=gpu:a100:1 \
bash slurm/submit_pipeline.sh
```

If the cluster requires an account and partitions, submit the stages directly:

```bash
DATA_SUBMISSION=$(sbatch --parsable --account=MY_ACCOUNT --partition=cpu \
  slurm/generate_data.sbatch)
DATA_JOB=${DATA_SUBMISSION%%;*}

PINN_SUBMISSION=$(sbatch --parsable --account=MY_ACCOUNT --partition=gpu \
  --dependency=afterok:${DATA_JOB} slurm/run_pinn_array.sbatch
)
PINN_JOB=${PINN_SUBMISSION%%;*}

sbatch --account=MY_ACCOUNT --partition=cpu \
  --dependency=afterok:${PINN_JOB} slurm/run_classical_array.sbatch
```

On clusters that require a GPU type, override the generic request, for example
`sbatch --gres=gpu:a100:1 ...`. Slurm exposes the allocated device through
`CUDA_VISIBLE_DEVICES`; the script intentionally uses `cuda:0`, which means the
first GPU visible inside that allocation.

Monitor and cancel jobs with:

```bash
squeue -u "$USER"
tail -f slurm/logs/jcp-data-<job-id>.out
scancel <job-id>
```

The runners are resumable. Re-submitting a completed task skips rows whose
configuration IDs are already present. Do not launch two jobs for the same test
and stage simultaneously because both would write the same result CSV.
