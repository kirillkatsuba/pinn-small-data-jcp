# Reference execution environment

The final result snapshots in `reproducibility/results/` were produced on the
following CPU machine.

| Component | Reference value |
|---|---|
| Computer | MacBook Pro |
| Processor | Apple M1, 8 CPU cores |
| Memory | 8 GB |
| Operating system | macOS 14.7.6 |
| Python | 3.13.1 |
| PyTorch | 2.6.0 |
| NumPy | 2.2.1 |
| pandas | 2.2.3 |
| SciPy | 1.14.1 |
| scikit-learn | 1.6.0 |
| XGBoost | 3.0.2 |
| Matplotlib | 3.10.0 |
| Numba | 0.63.1 |
| joblib | 1.4.2 |
| tqdm | 4.67.1 |

No machine serial number, user identifier, or host UUID is needed for
reproduction and none is recorded here.

## Device policy

The published result snapshots are the deterministic CPU reference. The PINN
training and inference paths additionally support `cuda`, an indexed CUDA
device such as `cuda:0`, Apple `mps`, and `auto`. The code seeds Python, NumPy,
Torch, and CUDA; requests deterministic Torch algorithms; disables cuDNN
benchmarking and TF32; and sets the CUDA BLAS workspace configuration required
by deterministic matrix multiplication.

Classical XGB, RBF-SVM, and GP runs remain on CPU so the frozen implementation
and its search grids are unchanged. The conservative finite-volume data
generator is also NumPy/Numba CPU code.

## Interpretation of reproducibility

With the same CPU architecture and package versions, the fixed seeds and
splits are intended to reproduce the reference results as closely as the
underlying libraries permit. CPU and CUDA kernels do not generally execute the
same floating-point operation order, so byte-identical checkpoints across
devices are neither promised nor scientifically necessary. CUDA replication
uses the same model, data, subsets, residual samples, optimizer schedule, and
metrics, and should be compared numerically.

The CUDA execution path is covered by an automatic smoke test when a CUDA
device is present. It could not be exercised on the original Apple M1 machine,
which has no NVIDIA device.
