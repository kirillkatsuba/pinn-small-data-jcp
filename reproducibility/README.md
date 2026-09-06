# Reproducibility package

This directory records the reference experiments currently reported in the JCP
manuscript. The new cluster candidate run is intentionally longer and will be
promoted to the manuscript only after its results are analysed.

## Contents

- `data_inputs/`: small immutable permeability fields and Test 4 parameter
  vectors. These files are tracked by Git; generated solution fields are not.
- `results/`: immutable final CSV snapshots used to build manuscript tables.
- `RUN_COMMANDS.md`: direct Python commands for tests, data generation,
  training, table construction, and visualisation.
- `ENVIRONMENT.md`: reference hardware, package versions, and device policy.
- `verify_results.py`: tolerance-based comparison of regenerated metrics with
  immutable publication snapshots.
- `../paper_jcp/run_publication_suite.py`: Python entry point for the current
  ten-seed, 50,000-epoch candidate protocol.

Run commands from the repository root. The publication runner is resumable:
the underlying experiment code skips already complete rows.

```bash
python3 -B paper_jcp/run_publication_suite.py \
  --test test4 --part all --device cpu --workers 1
```

The available test selector is `test1`, `test2`, `test3`, `test4`, or `all`;
the stage selector is `pinn`, `classical`, or `all`. Add `--dry-run` to print
the expanded commands without executing them.

## Current cluster candidate

The launcher now defaults to seeds `101`--`110` and 50,000 PINN epochs. Slurm
runs one test/seed pair per array task and merges the seed shards after all
training jobs finish.

## Reference snapshot contract

| Item | Fixed value |
|---|---|
| Label budgets | 20, 50, 100, 200, 500, 1000 |
| Seeds | 101, 102, 103 |
| Label-time fraction | 0.7 |
| Time bins | 10 |
| Backbone | width 128, depth 6, GELU |
| Optimizer | Adam, initial LR 0.003 |
| Scheduler | cosine, minimum LR 1e-5 |
| Epochs | 10,000 |
| Residual points | `N_r = 2N`, resampled each epoch |
| Residual domain | full physical time domain |
| PINN variants | H-PINN, HO-PINN |
| Classical methods | XGB, RBF-SVM, GP |
| Reference device | CPU |

Tests 2--4 use strong time-aware cross-validation for all classical methods.
Test 1 retains the standard/random tuning used in its frozen result table; a
future uniformity check may repeat Test 1 with strong/time-aware tuning without
changing the primary flow-study conclusions.

PINN training and inference also support `cuda`, `cuda:0`, `mps`, and `auto`.
GPU runs use the same scientific protocol but are not expected to produce
byte-identical floating-point trajectories to the CPU reference. Classical
models and the finite-volume generator remain on CPU.

## Expected outputs

```text
paper_jcp/generated/<experiment>/unified_test*.csv
paper_jcp/experiments/<experiment>/checkpoints/*.pth
paper_jcp/experiments/<experiment>/histories/*.csv
```

Generated data and models are excluded from Git because the Test 4 reference
dataset alone is approximately 500 MB. See `RUN_COMMANDS.md` for the complete
Python-only reproduction workflow.

After a completed run, verify all numerical metrics with:

```bash
python3 -B reproducibility/verify_results.py --suite all
```

This verifier targets the existing three-seed, 10,000-epoch snapshots. It is
not a pass/fail check for the new 50,000-epoch candidate results.
