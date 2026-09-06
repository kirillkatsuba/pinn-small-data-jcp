# JCP computational study

This directory contains the unified implementation, experiment outputs, plots,
and Elsevier/JCP manuscript for the small-data PINN study. The public entry
point is `run_publication_suite.py`; it freezes the configuration used in the
reported Tests 1--4.

## Core files

- `unified_protocol.py`: analytical Test 1, corrected flow loaders, common
  network, H/HO losses, nested labels, residual sampling, and metrics.
- `classical_baselines.py`: training-only tuning of XGB, RBF-SVM, and GP.
- `two_phase_fv.py`: conservative finite-volume reference solver.
- `generate_corrected_benchmarks.py`: deterministic Tests 2--4 generator.
- `run_unified_protocol.py`: configurable, resumable experiment runner.
- `run_publication_suite.py`: frozen publication launcher.
- `run_zero_label_experiment.py`: additional `N=0,20` experiment.
- `plot_model_comparison.py`: GT/PINN/XGB/SVM/GP comparison figures.
- `plot_training_history_comparison.py`: raw/weighted losses and adaptive
  weights across label budgets.
- `build_final_metrics_report.py`: complete `mean ± std` tables.
- `jcp_article/`: JCP manuscript and local Elsevier template.

## Devices

PINN training and inference accept:

```text
--device cpu | cuda | cuda:0 | mps | auto
```

Deterministic execution is enabled by default; use `--no-deterministic` only
for an explicitly separate speed experiment. A single GPU process must use
`--workers 1`. Classical baselines and finite-volume generation remain
CPU-bound in the frozen publication protocol.

## Minimal workflow

From the repository root:

```bash
python3 -B -m unittest \
  tests.test_heat_equation tests.test_two_phase_fv \
  tests.test_corrected_unified_protocol tests.test_classical_tuning \
  tests.test_generator_inputs \
  tests.test_device_execution -v

python3 -B paper_jcp/generate_corrected_benchmarks.py \
  --test all --output-root paper_jcp/generated/publication_data \
  --output-steps 100 --final-time-test2 0.3 \
  --final-time-test3 0.05 --final-time-test4 0.2 --cfl 0.2 \
  --permeability-test2 reproducibility/data_inputs/test2/permeability.npy \
  --permeability-test3 reproducibility/data_inputs/test3/permeability.npy \
  --permeability-test4 reproducibility/data_inputs/test4/permeability.npy \
  --test4-parameter-dir reproducibility/data_inputs/test4
```

Then run the full CPU suite or a PINN stage on CUDA:

```bash
python3 -B paper_jcp/run_publication_suite.py \
  --test all --part all --device cpu --workers 1

python3 -B paper_jcp/run_publication_suite.py \
  --test test2 --part pinn --device cuda:0 --workers 1
```

Exact commands and the experimental contract are in
`../reproducibility/RUN_COMMANDS.md` and `../reproducibility/README.md`.

## Manuscript sources

- `jcp_article/main.tex`: JCP manuscript.
- `jcp_article/references.bib`: bibliography.
- `FINAL_EXPERIMENT_METRICS.md`: complete numerical result report.
- `OVERLAP_BOUNDARY.md`: scope boundary with the separate ML/theory paper.

Generated datasets, checkpoints, histories, caches, and LaTeX build products
are excluded from Git and are reproducible from the tracked inputs. Frozen
result CSVs are versioned under `../reproducibility/results/`.
