# Corrected two-phase finite-volume generator

The new generator is implemented in `paper_jcp/two_phase_fv.py`. It does not
modify the legacy generator or any legacy datasets.

## Governing equations

The generator solves

\[
\nabla\cdot\mathbf q_t=0,
\qquad
\mathbf q_t=-K(\lambda_w+\lambda_o)\nabla p,
\]

\[
\phi\,\partial_t S_w+\nabla\cdot(f_w\mathbf q_t)=0,
\qquad
S_o=1-S_w,
\]

with

\[
\lambda_w=\frac{k_w S_w^{p_w}}{\mu_w},
\qquad
\lambda_o=\frac{k_o S_o^{p_o}}{\mu_o},
\qquad
f_w=\frac{\lambda_w}{\lambda_w+\lambda_o}.
\]

The default boundary conditions are

\[
p|_{x=0}=1,
\qquad
p|_{x=1}=0,
\]

pure-water inflow at the left boundary, and zero total normal flux on the top
and bottom boundaries.

## Numerical scheme

- cell-centred finite volumes;
- harmonic face transmissibilities;
- first-order upwind phase fluxes;
- conservative update divided by the pore volume
  \(\phi\,\Delta x\,\Delta y\);
- adaptive CFL substeps;
- pressure recomputed after every adaptive transport substep;
- synchronous snapshots \((p^n,S_w^n,S_o^n)\), including the exact initial
  state at \(t=0\).

## Single-run example

From the repository root:

```bash
python3 paper_jcp/generate_corrected_flow_data.py \
  --permeability test2-2phase-disp-simple/perm.npy \
  --output paper_jcp/generated/corrected/test2_reference.npz \
  --final-time 1.5 \
  --output-steps 100 \
  --water-exponent 2 \
  --oil-exponent 4 \
  --water-endpoint 1 \
  --oil-endpoint 0.3 \
  --water-viscosity 1 \
  --oil-viscosity 3
```

The output is an `.npz` file containing `times`, cell-centred `x` and `y`,
`pressure`, `water_saturation`, `oil_saturation`, `permeability`, `porosity`,
model metadata, and conservation diagnostics. The field axis order is
`(time, x, y)`.

The script refuses to overwrite an existing dataset unless `--overwrite` is
passed explicitly. Generation displays a `tqdm` progress bar for output time
layers. Pass `--no-progress` only for non-interactive logging environments.

## Pilot benchmark generation

Generate one corrected Test 2 and Test 3 solution plus the first five Test 4
parameter combinations:

```bash
python3 paper_jcp/generate_corrected_benchmarks.py \
  --test test2 \
  --output-steps 100

python3 paper_jcp/generate_corrected_benchmarks.py \
  --test test3 \
  --output-steps 100

python3 paper_jcp/generate_corrected_benchmarks.py \
  --test test4 \
  --test4-samples 5 \
  --output-steps 100 \
  --workers 4
```

The pilot defaults use different final times because the three permeability
fields have very different scales. These values must be inspected and frozen
before the final publication run. Every invocation writes or refreshes
`paper_jcp/generated/corrected_datasets/manifest.csv`. Existing `.npz` files
are reused unless `--overwrite` is supplied.

With one worker, `tqdm` reports time-layer progress inside every simulation.
With multiple workers, it reports completed parameter combinations so parallel
output remains readable.

After the pilot has been reviewed, generate all parameter combinations:

```bash
python3 paper_jcp/generate_corrected_benchmarks.py \
  --test all \
  --test4-samples 100 \
  --output-steps 100 \
  --workers 4
```

## Visualisations

```bash
python3 paper_jcp/plot_corrected_flow_data.py \
  --input paper_jcp/generated/corrected_datasets/test2/solution.npz \
  --time-fractions 0,0.25,0.5,1
```

This creates `fields.png`, `profiles.png`, and `time_series.png` next to the
selected dataset in a dedicated figures directory. For Test 4, pass an input
such as `test4/sample_000.npz`.
