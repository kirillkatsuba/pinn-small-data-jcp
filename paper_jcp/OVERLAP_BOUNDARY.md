# Internal paper-split boundary

This file is an editorial control document, not part of the submitted manuscript.

| Item | JCP computational paper | Core A/A* methodological paper |
|---|---|---|
| Central question | Can a physics-informed surrogate be useful when labelled PDE states are scarce? | Why does the hierarchical PINN training mechanism work? |
| Main audience | Computational physics and numerical modelling | Machine learning and AI |
| Primary claim | Conditional label-efficiency relative to conventional non-neural surrogates | New training principle, theory, and optimization benefit |
| Baselines | Training-only tuned XGB, RBF SVR, and GP | Vanilla PINN, LRA, SoftAdapt, GradNorm, ReLoBRaLo, other PINN methods |
| Main evidence | Independent sparse-data repetitions, fixed disjoint tests, label noise, residual-point sensitivity, data-efficiency curves, state fields, fitting cost, failure regimes | Controlled balancer ablations, theory-facing experiments, convergence and stability |
| Method description | Only enough to reproduce the fixed PINN configuration | Full derivation, theoretical results, detailed ablations |
| Explicit exclusions | No claim that PINNs replace numerical solvers; no balancer leaderboard; no convergence theorem | No broad PINN-vs-classical-surrogate computational assessment |

## Asset allocation

| Claim / experiment / figure | Owner |
|---|---|
| Independent and fixed-holdout comparisons against XGB/SVR/GP | JCP only |
| Label-noise and residual-point sensitivity | JCP only |
| Accuracy versus labelled-data budget | JCP only |
| Pressure/saturation field comparison with non-neural models | JCP only |
| Surrogate fitting-time comparison | JCP only |
| Comparison among PINN balancers | Core A/A* only |
| Theoretical Robin relaxation and convergence arguments | Core A/A* only |
| Gradient-orthogonality ablations | Core A/A* only |

If a benchmark PDE appears in both papers, it must support a different scientific question, use distinct comparison sets, and avoid reusing the same headline table or figure. Shared implementation details should be disclosed and cited transparently in the final submissions.
