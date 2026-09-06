# GitHub release manifest

## Required source files

- `paper_jcp/unified_protocol.py`: common PINN, samplers, losses, metrics.
- `paper_jcp/run_unified_protocol.py`: resumable final experiment runner.
- `paper_jcp/run_publication_suite.py`: frozen Python launcher for Tests 1--4.
- `paper_jcp/classical_baselines.py`: XGB/SVM/GP tuning utilities.
- `paper_jcp/two_phase_fv.py`: conservative finite-volume reference solver.
- `paper_jcp/generate_corrected_benchmarks.py`: Tests 2--4 data generation.
- `paper_jcp/plot_model_comparison.py`: GT/model qualitative panels.
- `paper_jcp/build_final_metrics_report.py`: complete result tables.
- `benchmarking/pinn_ablation_utils.py`: shared metric functions.
- `loss_balancong_algorithms/algos.py`: H/HO loss balancer.

## Required tests

- `tests/test_heat_equation.py`
- `tests/test_two_phase_fv.py`
- `tests/test_corrected_unified_protocol.py`
- `tests/test_classical_tuning.py`
- `tests/test_device_execution.py`

## Tracked experimental artifacts

- seven small arrays under `reproducibility/data_inputs/`;
- five final CSV snapshots under `reproducibility/results/`;
- Markdown documentation and LaTeX manuscript sources.
- `reproducibility/RUN_COMMANDS.md` with direct Python commands.
- `reproducibility/ENVIRONMENT.md` with the reference execution environment.
- `reproducibility/verify_results.py` for metric verification.

## Excluded artifacts

- generated `.npz` solution fields;
- `.pth` checkpoints and training histories;
- caches, temporary figures, LaTeX build products;
- local legacy notebooks and superseded ablation outputs;
- local colleague packages and ZIP archives.

The excluded items can be recreated from the tracked inputs and Python launch
commands.
