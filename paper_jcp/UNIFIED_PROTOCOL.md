# Unified small-data interpolation/extrapolation protocol

This protocol is separate from the legacy JCP experiments. It does not alter
their checkpoints or CSV files.

## Fixed design

- Label budgets: `20, 50, 100, 200, 500, 1000`.
- Within a seed the labelled sets are nested.
- Labels cover the first 70% of the physical time interval and are balanced
  across ten time bins.
- Interpolation is evaluated inside the labelled time interval.
- Extrapolation is evaluated on the final 30% of the time interval.
- The primary `full` residual domain is physics-assisted extrapolation: no
  future solution labels are used, but PDE residuals are sampled over the full
  physical time domain.
- With `--collocation-ratio 1`, one epoch uses all `N` labels and a freshly
  sampled residual batch of size `N_r=N`. Labels stay fixed; residual points are
  replaced every epoch. Current labelled indices and both test sets are excluded
  from the residual pool.
- Inputs are mapped to `[-1, 1]` using known physical domain bounds, not sparse
  train statistics. Permeability is not a network input and remains a Darcy
  coefficient.
- One gated residual backbone is shared by all tests. Only input/output sizes
  and the governing equations differ.
- `H-PINN` uses hierarchical exponential gates. `HO-PINN` adds the orthogonal
  gradient factor.
- XGB, RBF-SVR, and GP use the same labels and training-only tuning. The final
  baseline protocol uses expanding-window temporal cross-validation: every
  validation fold lies strictly later than its training fold.

## Outputs

Each experiment namespace is independent:

```text
paper_jcp/generated/<experiment-name>/unified_test*.csv
paper_jcp/experiments/<experiment-name>/checkpoints/*.pth
paper_jcp/experiments/<experiment-name>/histories/*.csv
paper_jcp/experiments/<experiment-name>/histories/*.json
paper_jcp/figures/<experiment-name>/*.png
```

History CSVs contain raw component losses, balanced loss, weights, weighted
contributions, learning rate, gradient norm, and IBC/PDE losses on a fixed
unlabelled physics-validation set.

## 1. Short technical check

This is optional and intentionally small. It verifies that data loading,
automatic differentiation, checkpointing, and CSV output work before starting
long experiments:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name unified_smoke \
  --test test4 \
  --budgets 20 \
  --seeds 11 \
  --balancers h,ho \
  --epochs 20 \
  --skip-classical \
  --workers 1
```

### Corrected Test 2 data

The corrected finite-volume dataset is selected explicitly. The runner never
switches from legacy labels implicitly:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test2_smoke \
  --test test2 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/corrected_t3_pilot \
  --budgets 20 \
  --seeds 101 \
  --hidden-dim 16 --depth 2 --activation gelu \
  --balancers ho \
  --epochs 2 --log-every 1 \
  --physics-validation-size 20 \
  --skip-classical --workers 1
```

The legacy-data tuning winner is only a starting point for corrected-data
revalidation because the labels, physical time scale, and IBC loss have
changed. Recheck it on development budgets and seeds before enabling the
conventional baselines or final seeds:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test2_recheck_previous_winner \
  --test test2 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/corrected_t3_pilot \
  --budgets 50,200 \
  --seeds 101,102,103 \
  --time-fraction 0.7 --time-bins 10 \
  --hidden-dim 128 --depth 6 --activation gelu \
  --balancers h \
  --l0-data 0.03 --l0-ibc 0.003 \
  --optimizer adam --learning-rate 0.003 \
  --scheduler cosine --min-learning-rate 0.00001 \
  --epochs 5000 \
  --collocation-ratio 2 \
  --residual-domain full \
  --physics-validation-size 512 \
  --skip-classical --workers 1
```

Only after corrected-data development runs identify and freeze the winner
should it be run with budgets `20,50,100,200,500,1000`, untouched final seeds,
and classical baselines.

### Strong classical-baseline tuning

The publication run should use `--classical-tuning-profile strong` and
`--classical-cv time`. The strong profile searches 32 sampled XGB
configurations, 40 sampled RBF-SVR configurations, and 12 GP
kernel/regularisation combinations independently for Pressure, Soil, and Swat.
Input scaling and strong-profile SVR target scaling remain inside the
cross-validation estimator and are therefore fitted separately inside each
fold. GP uses its own training-target normalisation. The final 30%
extrapolation set is never used for tuning.

Start with XGB and SVR; this is a resumable pilot and will also expose any data
or protocol problems before the more expensive GP search:

```bash
python3 -B paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test2_classical_strong \
  --test test2 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/corrected_final \
  --budgets 20,50 \
  --seeds 101,102,103 \
  --time-fraction 0.7 --time-bins 10 \
  --classical-tuning-profile strong \
  --classical-cv time \
  --classical-methods XGB,SVM \
  --skip-pinn --workers 1
```

Then add GP to the same experiment namespace:

```bash
python3 -B paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test2_classical_strong \
  --test test2 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/corrected_final \
  --budgets 20,50 \
  --seeds 101,102,103 \
  --time-fraction 0.7 --time-bins 10 \
  --classical-tuning-profile strong \
  --classical-cv time \
  --classical-methods GP \
  --skip-pinn --workers 1
```

After checking the pilot, repeat both commands with
`--budgets 20,50,100,200,500,1000`. Existing rows are skipped. Every result row
stores the winning parameters and its CV score in
`classical_tuning_details_json`, so the table is auditable and reproducible.

The corrected loader uses the stored cell-centred coordinates and physical
times, maps labels to `(Pressure, Soil, Swat)`, and attaches the local
permeability only to the PDE residual. It also reads porosity and phase
viscosities from dataset metadata. The corrected IBC loss adds zero incoming
oil flux at the water inlet and zero phase-normal flux on both lateral
boundaries.

Corrected Test 3 uses the same runner and reads its own permeability field and
time range from `test3/solution.npz`. For the current horizon pilot:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test3_recheck_previous_winner \
  --test test3 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/corrected_horizon_pilots \
  --budgets 50,200 \
  --seeds 101,102,103 \
  --time-fraction 0.7 --time-bins 10 \
  --hidden-dim 128 --depth 6 --activation gelu \
  --balancers h \
  --l0-data 0.03 --l0-ibc 0.003 \
  --optimizer adam --learning-rate 0.003 \
  --scheduler cosine --min-learning-rate 0.00001 \
  --epochs 5000 --collocation-ratio 2 \
  --residual-domain full --physics-validation-size 512 \
  --skip-classical --workers 1
```

### Corrected Test 4

Corrected Test 4 reads 100 parameterised finite-volume samples with seven
network inputs `(t,x,y,pwat,poil,kwat,koil)`. To avoid flattening roughly 41
million grid points, the loader deterministically materialises a 100,000-point
pool: 50,000 labelled-time candidates and 50,000 later-time residual
candidates. The 1,500-point interpolation and extrapolation tests are disjoint
from this pool. The pool includes all 100 physical-parameter combinations.

Use the same frozen PINN configuration as corrected Test 2 and Test 3:

```bash
caffeinate -i python3 -B paper_jcp/run_unified_protocol.py \
  --experiment-name corrected_test4_final \
  --test test4 \
  --data-source corrected \
  --corrected-data-root paper_jcp/generated/test4_horizon_02 \
  --corrected-test4-pool-size 100000 \
  --budgets 20,50,100,200,500,1000 \
  --seeds 101,102,103 \
  --time-fraction 0.7 --time-bins 10 \
  --hidden-dim 128 --depth 6 --activation gelu \
  --balancers h,ho \
  --l0-data 0.03 --l0-ibc 0.003 --orthogonal-floor 0.25 \
  --optimizer adam --learning-rate 0.003 \
  --scheduler cosine --min-learning-rate 0.00001 \
  --epochs 10000 --collocation-ratio 2 \
  --residual-domain full --gradient-clip 1.0 \
  --checkpoint-criterion balanced --physics-validation-size 512 \
  --skip-classical --workers 1
```

Then append strong time-CV classical baselines to the same experiment. Run
XGB/SVR and GP sequentially because both commands update one resumable CSV.

## 2. Architecture screening

Use development seeds and two representative budgets. Keep these outputs out
of the final namespace. Example commands:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name architecture_w64_d4_tanh \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 64 --depth 4 --activation tanh \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4

python3 paper_jcp/run_unified_protocol.py \
  --experiment-name architecture_w128_d6_tanh \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4

python3 paper_jcp/run_unified_protocol.py \
  --experiment-name architecture_w128_d6_silu \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation silu \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4

python3 paper_jcp/run_unified_protocol.py \
  --experiment-name architecture_w128_d8_tanh \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 8 --activation tanh \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4
```

Aggregate any namespace with:

```bash
python3 paper_jcp/analyze_unified_protocol.py \
  --experiment-name architecture_w128_d6_tanh
```

Compare `unified_summary.csv`, especially interpolation/extrapolation
`Swat_R2`, `Swat_RMSE`, `mean_R2`, physics-validation residual, variance, and
fit time. Choose architecture on development results only.

## 3. H/HO weighting tuning

After selecting a backbone, tune loss gates in a new development namespace.
The commands below illustrate a staged search; they are intentionally separate
so that every configuration remains auditable.

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name weighting_l0d001_l0b001 \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho --l0-data 0.01 --l0-ibc 0.001 \
  --epochs 5000 --skip-classical --workers 4

python3 paper_jcp/run_unified_protocol.py \
  --experiment-name weighting_l0d01_l0b001 \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho --l0-data 0.1 --l0-ibc 0.001 \
  --epochs 5000 --skip-classical --workers 4

python3 paper_jcp/run_unified_protocol.py \
  --experiment-name weighting_l0d01_l0b01 \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho --l0-data 0.1 --l0-ibc 0.01 \
  --epochs 5000 --skip-classical --workers 4
```

For HO-PINN, the orthogonal floor can additionally be screened with
`--orthogonal-floor 0.25`, `0.5`, and `0.75`.

## 4. Optimisation screening

Using the selected architecture and H/HO parameters, compare at least:

```bash
# Adam with cosine decay, initial LR 1e-3
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name optimizer_adam_1e3 \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --learning-rate 0.001 --min-learning-rate 0.00001 \
  --optimizer adam --scheduler cosine \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4

# Adam with a smaller initial LR
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name optimizer_adam_3e4 \
  --test all --budgets 50,200 --seeds 101,102,103 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --learning-rate 0.0003 --min-learning-rate 0.00001 \
  --optimizer adam --scheduler cosine \
  --balancers h,ho --epochs 5000 --skip-classical --workers 4
```

The current runner supports Adam and AdamW. L-BFGS is deliberately not part of
the first unified protocol because per-epoch residual resampling makes its
closure evaluations non-deterministic. It can be added later as a fixed-pool
refinement after the Adam study.

## 5. Freeze the selected PINN configuration

Write down the selected backbone, balancer, gate parameters, and optimiser
before final evaluation. Do not select them from final seeds or final test
metrics. The command below uses the proposed defaults; replace values only with
the frozen development winner.

## 6. Full final analysis

Run all label budgets, five new final seeds, H/HO if both remain under study,
and all classical baselines:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name unified_final \
  --test all \
  --budgets 20,50,100,200,500,1000 \
  --seeds 211,222,233,244,255 \
  --time-fraction 0.7 --time-bins 10 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho \
  --l0-data 0.01 --l0-ibc 0.001 --orthogonal-floor 0.5 \
  --optimizer adam --learning-rate 0.001 \
  --scheduler cosine --min-learning-rate 0.00001 \
  --epochs 20000 \
  --collocation-ratio 1 \
  --residual-domain full \
  --physics-validation-size 512 \
  --classical-tuning-profile strong \
  --classical-cv time \
  --workers 4
```

The command is resumable. Completed method/seed/budget pairs are skipped when
the same command is repeated.

Then aggregate tables and figures:

```bash
python3 paper_jcp/analyze_unified_protocol.py \
  --experiment-name unified_final
```

Inspect the complete optimisation dynamics for a selected run:

```bash
python3 paper_jcp/plot_unified_training_history.py \
  --experiment-name unified_final \
  --test test4 --budget 100 --seed 211
```

The resulting figure separates raw losses, changing weights, weighted
contributions, unseen physics-validation residual, gradient norm, and learning
rate. The balanced total alone is not sufficient because its weights change
during training.

If only one PINN configuration should appear in plots, pass its exact
`config_id` from the result CSV:

```bash
python3 paper_jcp/analyze_unified_protocol.py \
  --experiment-name unified_final \
  --config-id ho_w128_d6_tanh_lr0.001_l0d0.01_l0b0.001_adam_cosine_nr1_full
```

## 7. Strict extrapolation sensitivity

The final command above is variant B: residuals cover the complete time domain.
To measure strict function extrapolation, repeat the frozen PINN with residuals
restricted to the labelled interval:

```bash
python3 paper_jcp/run_unified_protocol.py \
  --experiment-name unified_strict_extrapolation \
  --test all \
  --budgets 20,50,100,200,500,1000 \
  --seeds 211,222,233,244,255 \
  --hidden-dim 128 --depth 6 --activation tanh \
  --balancers h,ho --epochs 20000 \
  --collocation-ratio 1 \
  --residual-domain label \
  --skip-classical \
  --workers 4
```

Classical predictions are unchanged between `full` and `label` residual-domain
experiments, so they need not be recomputed.

## Interpretation boundary

If PINN wins only with `--residual-domain full`, the supported claim is a
physics-assisted forecasting advantage: labels stop at 70% of the time domain,
while known governing equations are enforced throughout it. It is not a claim
of purely data-driven function extrapolation.
