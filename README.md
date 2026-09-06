# Physics-Informed Neural Networks in Small-Data Regimes

Reproducible code, data-generation tools, results, and manuscript sources for
the Journal of Computational Physics study of PINNs as small-data PDE
surrogates.

The central question is practical: when only a small number of labelled
solution points is available, can physics-informed learning outperform
conventional data-only surrogates? The current candidate protocol compares H-PINN and
HO-PINN with tuned XGBoost, RBF-SVM, and Gaussian-process regression on one
analytical heat problem and three two-phase porous-flow problems.

## What is reproducible

- label budgets `N = 20, 50, 100, 200, 500, 1000`;
- ten independent training seeds `101, ..., 110`;
- `50,000` PINN epochs with best-validation checkpoint selection;
- interpolation and temporal extrapolation evaluation;
- identical nested labelled subsets for every method;
- `N_r = 2N` residual points, resampled at every PINN epoch;
- time-aware training-only hyperparameter selection for the flow baselines;
- pressure and water/oil saturation metrics, constraint diagnostics, runtime,
  checkpoints, and per-epoch loss/weight histories.

The immutable result tables used by the manuscript are in
[`reproducibility/results`](reproducibility/results). Locally generated fields,
checkpoints, and histories are preserved in this working copy but excluded
from Git because they are large and reproducible.

## Repository layout

```text
paper_jcp/
  unified_protocol.py              common datasets, PINN, losses, metrics
  classical_baselines.py           XGB, RBF-SVM, and GP tuning
  two_phase_fv.py                  conservative reference solver
  generate_corrected_benchmarks.py Tests 2--4 data generator
  run_unified_protocol.py          configurable/resumable experiment runner
  run_publication_suite.py         current candidate configuration
  run_zero_label_experiment.py     N=0 and N=20 extension
  plot_model_comparison.py         GT/PINN/XGB/SVM/GP visualisations
  jcp_article/                     Elsevier/JCP LaTeX manuscript
loss_balancong_algorithms/         H-PINN and HO-PINN weighting
benchmarking/                      shared evaluation helpers
tests/                             numerical, protocol, tuning, and device tests
reproducibility/
  data_inputs/                     seven small immutable generator inputs
  results/                         frozen final CSV snapshots
  RUN_COMMANDS.md                  complete command reference
  ENVIRONMENT.md                   hardware/software reproducibility record
```

The spelling `loss_balancong_algorithms` is retained for checkpoint and import
compatibility with the completed experiments.

## Installation

Python 3.13.1 and PyTorch 2.6.0 were used for the reported CPU runs. Environments
are selected through Conda. For the CUDA server, create the checked-in environment:

```bash
conda env create -f environment.yml
conda activate pinn-small-data-jcp
```

`environment.yml` is the CUDA 12.4 server environment and already installs the
core stack and PyTorch. For a non-CUDA machine, create a Conda environment and
then install exactly one appropriate PyTorch build:

```bash
conda create -n pinn-small-data-jcp python=3.13.1 pip -y
conda activate pinn-small-data-jcp
python3 -m pip install -r requirements.txt
```

CPU-only Linux/Windows:

```bash
python3 -m pip install -r requirements-cpu.txt
```

NVIDIA CUDA 12.4 Linux/Windows:

```bash
python3 -m pip install -r requirements-cuda124.txt
```

macOS (CPU or Apple Metal/MPS):

```bash
python3 -m pip install torch==2.6.0
```

Check the installation:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.mps.is_available())"
```

## Validate the implementation

```bash
PYTHONPYCACHEPREFIX=/tmp/pinn-jcp-pycache \
MPLCONFIGDIR=/tmp/pinn-jcp-mpl \
python3 -B -m unittest \
  tests.test_heat_equation \
  tests.test_two_phase_fv \
  tests.test_corrected_unified_protocol \
  tests.test_classical_tuning \
  tests.test_generator_inputs \
  tests.test_publication_defaults \
  tests.test_device_execution -v
```

The CUDA smoke test runs automatically when CUDA is available and is skipped
otherwise.

## Generate the reference datasets

Test 1 is analytical. Generate Tests 2--4 from the tracked physical inputs:

```bash
python3 -B paper_jcp/generate_corrected_benchmarks.py \
  --test all \
  --output-root paper_jcp/generated/publication_data \
  --output-steps 100 \
  --final-time-test2 0.3 \
  --final-time-test3 0.05 \
  --final-time-test4 0.2 \
  --cfl 0.2 \
  --workers 1 \
  --permeability-test2 reproducibility/data_inputs/test2/permeability.npy \
  --permeability-test3 reproducibility/data_inputs/test3/permeability.npy \
  --permeability-test4 reproducibility/data_inputs/test4/permeability.npy \
  --test4-parameter-dir reproducibility/data_inputs/test4
```

Generation displays a `tqdm` progress bar. Test 4 is the largest output and
contains 100 parameter realisations.

## Run the current extended protocol

The defaults are ten seeds (`101`--`110`) and `50,000` PINN epochs. The best
validation checkpoint is retained for every run.

All four tests on the reference CPU protocol:

```bash
python3 -B paper_jcp/run_publication_suite.py \
  --test all --part all --device cpu --workers 1
```

One PINN stage on an NVIDIA GPU:

```bash
python3 -B paper_jcp/run_publication_suite.py \
  --test test2 --part pinn --device cuda:0 --workers 1
```

Apple Silicon can use `--device mps`; `--device auto` chooses CUDA, then MPS,
then CPU. Classical XGB/SVM/GP models and the finite-volume generator remain
CPU algorithms in the frozen protocol. This is intentional: changing their
implementation would no longer reproduce the reported baselines.

The runner is resumable. Completed configuration IDs are skipped. Use
`--dry-run` to inspect every expanded command without starting training:

```bash
python3 -B paper_jcp/run_publication_suite.py \
  --test all --part all --device cpu --workers 1 --dry-run
```

Outputs are written to:

```text
paper_jcp/generated/<experiment>/unified_test*.csv
paper_jcp/experiments/<experiment>/checkpoints/*.pth
paper_jcp/experiments/<experiment>/histories/*.csv
```

## Zero-label extension

The additional Tests 1--2 experiment with `N=0` and `N=20` uses a fixed 128
residual points and 50,000 epochs:

```bash
python3 -B paper_jcp/run_zero_label_experiment.py \
  --experiment-name zero_label_test12_e50k_nr128 \
  --test all --budgets 0,20 --seeds 101,102,103,104,105,106,107,108,109,110 \
  --balancers h,ho --epochs 50000 --collocation-points 128 \
  --hidden-dim 128 --depth 6 --activation gelu \
  --learning-rate 0.003 --min-learning-rate 0.00001 \
  --l0-data 0.03 --l0-ibc 0.003 --orthogonal-floor 0.25 \
  --gradient-clip 1 --physics-validation-size 512 --log-every 100 \
  --corrected-data-root paper_jcp/generated/publication_data \
  --device cpu --workers 1
```

Replace `cpu` with `cuda:0` to use one NVIDIA GPU.

## Tables, figures, and manuscript

```bash
python3 -B paper_jcp/build_final_metrics_report.py

python3 -B paper_jcp/plot_model_comparison.py \
  --test test4 --budget 50 --seed 101 --pinn h \
  --test4-sample 0 --time-fractions 0.7,1 --device cpu

cd paper_jcp/jcp_article
python3 build_article_assets.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

For every direct command, including training-history comparisons, see
[`reproducibility/RUN_COMMANDS.md`](reproducibility/RUN_COMMANDS.md).

## Slurm cluster

Ready-to-submit CPU/GPU jobs are provided in [`slurm/`](slurm/README.md):

```bash
bash slurm/submit_pipeline.sh
```

The pipeline first generates Tests 2--4 on CPU, then runs one GPU task per PINN
test, and finally runs one CPU task per classical test. The checked-in jobs use
the target server's `rocky` partition and request the PINN GPU with `--gpus=1`;
the account and optional submission overrides are documented separately.

## Reproducibility across devices

The default is deterministic execution: Python, NumPy, Torch, and CUDA seeds
are set; deterministic Torch kernels are requested; cuDNN benchmarking and
TF32 are disabled; and checkpoints are saved as CPU tensors so they can be
loaded on any supported device.

Exact bitwise equality is expected only when the same software stack, device
type, and hardware are used. CPU and CUDA use different floating-point kernels,
so their loss trajectories may differ slightly even with identical seeds.
Scientific replication should therefore compare the frozen splits,
configuration IDs, and metrics within stated numerical tolerances rather than
requiring byte-identical weights. The original machine and dependency versions
are recorded in [`reproducibility/ENVIRONMENT.md`](reproducibility/ENVIRONMENT.md).

## License and citation

License and final citation metadata should be added before the public GitHub
release. Until then, the repository should be treated as research code
associated with the manuscript in `paper_jcp/jcp_article/`.
