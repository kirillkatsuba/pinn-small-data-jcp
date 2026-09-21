# Slurm execution

The jobs are split into four stages:

1. `generate_data.sbatch` generates the deterministic finite-volume datasets
   for Tests 2--4 on a CPU node.
2. `run_pinn_array.sbatch` launches 40 independent GPU tasks: four tests times
   ten seeds.
3. `run_classical_array.sbatch` launches the corresponding 40 CPU tasks after
   the PINN array succeeds.
4. `merge_results.sbatch` validates and merges all seed-specific CSV shards.

The classical array uses the `extensive` search profile with expanding-window
time cross-validation by default. Override `JCP_CLASSICAL_TUNING_PROFILE`,
`JCP_CLASSICAL_CV`, or `JCP_CLASSICAL_METHODS` at submission time when a
smaller diagnostic run is required.

The checked-in jobs use the target server settings: partition `rocky`, one GPU
requested with `--gpus=1` for PINN jobs, and the time limits recorded in the
four `.sbatch` files. An account can still be supplied at submission time.

## Environment

From the cloned repository on the cluster:

```bash
# Load the server's Conda and CUDA modules; adapt their names if necessary.
module purge
module load conda
module load cuda/12.4

conda env create -f environment.yml
conda activate pinn-small-data-jcp
```

Verify CUDA on a GPU allocation:

```bash
srun --partition=rocky --gpus=1 --time=00:10:00 \
  conda run -n pinn-small-data-jcp python \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Submit the complete pipeline

Run from the repository root:

```bash
bash slurm/submit_pipeline.sh
```

Or pass common site settings without editing the scripts:

```bash
SLURM_ACCOUNT=MY_ACCOUNT \
SLURM_CPU_PARTITION=rocky \
SLURM_GPU_PARTITION=rocky \
JCP_GPU_COUNT=1 \
JCP_CONDA_ENV=pinn-small-data-jcp \
bash slurm/submit_pipeline.sh
```

If the cluster requires an account and partitions, submit the stages directly:

```bash
DATA_SUBMISSION=$(sbatch --parsable --account=MY_ACCOUNT --partition=rocky \
  slurm/generate_data.sbatch)
DATA_JOB=${DATA_SUBMISSION%%;*}

PINN_SUBMISSION=$(sbatch --parsable --account=MY_ACCOUNT --partition=rocky \
  --gpus=1 \
  --dependency=afterok:${DATA_JOB} slurm/run_pinn_array.sbatch
)
PINN_JOB=${PINN_SUBMISSION%%;*}

CLASSICAL_SUBMISSION=$(sbatch --parsable --account=MY_ACCOUNT --partition=rocky \
  --dependency=afterok:${PINN_JOB} slurm/run_classical_array.sbatch)
CLASSICAL_JOB=${CLASSICAL_SUBMISSION%%;*}

sbatch --account=MY_ACCOUNT --partition=rocky \
  --dependency=afterok:${CLASSICAL_JOB} slurm/merge_results.sbatch
```

Slurm exposes the allocated device through `CUDA_VISIBLE_DEVICES`; the script
intentionally uses `cuda:0`, which means the first GPU visible inside that
allocation.

Monitor and cancel jobs with:

```bash
squeue -u "$USER"
tail -f slurm/logs/jcp-data-<job-id>.out
scancel <job-id>
```

Array task `0` is Test 1 / seed 101, tasks `0`--`9` cover Test 1, `10`--`19`
cover Test 2, `20`--`29` cover Test 3, and `30`--`39` cover Test 4. Every task
writes a separate `*_seed<seed>.csv`, so concurrent writes are safe. The final
merge job creates the canonical `unified_test*.csv` files.

If Test 1 is already complete, submit only Tests 2--4 with the enlarged search:

```bash
sbatch --array=10-39 --time=12:00:00 slurm/run_classical_array.sbatch
```

To stage the expensive Gaussian-process fits separately, first run XGB and SVM:

```bash
JCP_CLASSICAL_METHODS=XGB,SVM \
sbatch --array=10-39 --time=12:00:00 slurm/run_classical_array.sbatch
```

Then submit GP using the same seed shards; completed rows are preserved:

```bash
JCP_CLASSICAL_METHODS=GP \
sbatch --array=10-39 --time=12:00:00 slurm/run_classical_array.sbatch
```

After every array has completed, merge one test at a time (replace `test2` by
`test3` and `test4`):

```bash
JCP_MERGE_TEST=test2 JCP_MERGE_PART=classical \
sbatch slurm/merge_results.sbatch
```

The runners are resumable. Re-submitting a completed array task skips rows
whose configuration IDs are already present. The configuration ID includes the
epoch count, so a 50,000-epoch run cannot be mistaken for an older 10,000-epoch
run.

## One PINN test at a time: N=20--1000

The dedicated launcher runs one selected test for budgets
`20,50,100,200,500,1000`, seeds `101`--`110`, and both H-PINN and HO-PINN. One
array task owns one seed. To make better use of a powerful GPU, four independent
models share it by default; every model writes an isolated CSV shard and log
file.

```bash
JCP_MODELS_PER_GPU=4 bash slurm/submit_pinn_test.sh test1
JCP_MODELS_PER_GPU=4 bash slurm/submit_pinn_test.sh test2
JCP_MODELS_PER_GPU=4 bash slurm/submit_pinn_test.sh test3
JCP_MODELS_PER_GPU=4 bash slurm/submit_pinn_test.sh test4
```

The launcher submits the ten-task GPU array and a dependent merge job. Override
the 50,000 epochs, concurrency, or the checked-in 30-minute limit when needed:

```bash
JCP_EPOCHS=50000 \
JCP_MODELS_PER_GPU=8 \
JCP_PINN_TIME=12:00:00 \
bash slurm/submit_pinn_test.sh test3
```

Select only the unit-weight Vanilla PINN while retaining the same architecture,
optimizer, residual sampling, budgets, seeds, and 50,000-epoch protocol:

```bash
JCP_PINN_BALANCERS=vanilla \
JCP_MODELS_PER_GPU=4 \
JCP_PINN_TIME=12:00:00 \
bash slurm/submit_pinn_test.sh test3
```

The dependent merge preserves an existing canonical result table and appends
the new `Vanilla PINN` rows. The configuration identifier starts with
`vanilla-unit`, so earlier experimental variants cannot be treated as complete
unit-weight runs.

Start with four concurrent models. Increase to eight only if `nvidia-smi` shows
substantial free memory and the elapsed time per model does not increase too
much. Per-model logs are stored under
`slurm/logs/<test>-<array-job>_<task>/`. The merged metrics are written to that
test's canonical experiment directory under `paper_jcp/generated/`.
