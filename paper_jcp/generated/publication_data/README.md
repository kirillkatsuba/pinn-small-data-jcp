# Reference flow data for the JCP study

This directory contains the fixed numerical reference solutions used for
Tests 2--4 of the small-data PINN study. These are author-generated simulations,
not measurements from an external dataset. Test 1 has an analytical reference
and therefore has no solution file here.

## Contents

| Path | Contents |
| --- | --- |
| `test2/solution.npz` | One Test 2 reference trajectory, final time 0.30 |
| `test3/solution.npz` | One Test 3 reference trajectory, final time 0.05 |
| `test4/sample_000.npz` ... `sample_099.npz` | 100 Test 4 parameter combinations, final time 0.20 |
| `permeability/test2.npy`, `test3.npy`, `test4.npy` | Three fixed absolute-permeability fields, each `(64, 64)` |
| `test4_parameters/*_100.npy` | The four fixed parameter arrays for Test 4 |
| `manifest.csv` | Per-simulation parameters and solver diagnostics |
| `SHA256SUMS` | SHA-256 checksums of all other files in this directory |

All 100 Test 4 solutions share the same absolute-permeability field.
For sample index `i`, the corresponding flattened parameter-array element `i`
is used: `pwat_100.npy` = water exponent, `poil_100.npy` = oil exponent,
`kwat_100.npy` = water endpoint, and `koil_100.npy` = oil endpoint.
These arrays and the separate permeability files are byte-identical copies of
the fixed inputs in `../../../reproducibility/data_inputs/`.

## NumPy file schema

Load files with NumPy and `allow_pickle=False`; no Python pickle is required.
All numerical fields retain the original benchmark scale. No SI-unit conversion
is implied; use the manuscript's benchmark definitions when interpreting values.

| Key | Shape | Meaning |
| --- | --- | --- |
| `times` | `(101,)` | Saved times, including the initial state |
| `x`, `y` | `(64,)` each | Cell-centre coordinates on the unit-square domain |
| `pressure` | `(101, 64, 64)` | Pressure; axes `(time, x, y)` |
| `water_saturation` | `(101, 64, 64)` | Water volume fraction; axes `(time, x, y)` |
| `oil_saturation` | `(101, 64, 64)` | Oil volume fraction; axes `(time, x, y)` |
| `permeability` | `(64, 64)` | Fixed absolute permeability; axes `(x, y)` |
| `porosity` | `(64, 64)` | Porosity; axes `(x, y)` |
| `metadata_json` | scalar Unicode string | JSON solver, fluid, boundary and relative-permeability settings |
| `diagnostics_json` | scalar Unicode string | JSON conservation and solver diagnostics |

The numerical arrays in solution files use `float64`. Saturations are
dimensionless and their sum is one. The reference solver uses cell-centred
finite volumes, harmonic transmissibility, first-order upwind transport and
adaptive explicit stepping with CFL 0.2. Pressure is recomputed at each
adaptive substep; the 101 saved times do not represent all internal substeps.
Each solution contains its exact settings in `metadata_json`.

```python
from pathlib import Path
import json
import numpy as np

root = Path("paper_jcp/generated/publication_data")
with np.load(root / "test2/solution.npz", allow_pickle=False) as data:
    saturation = data["water_saturation"]  # (time, x, y)
    settings = json.loads(data["metadata_json"].item())
permeability = np.load(root / "permeability/test2.npy", allow_pickle=False)
```

## Integrity and scope

From this directory on macOS, run `shasum -a 256 -c SHA256SUMS`.
On Linux, use `sha256sum -c SHA256SUMS`.
Paths in `manifest.csv` are relative to the repository root.

These files are the reference fields from which observations are sampled;
they are not pre-split training tables. Experiment budgets, random seeds and
the sampling/evaluation protocol belong to the experiment configuration.
Model weights, training logs and model-comparison metric CSVs are outside
this dataset directory. The code and reproduction instructions are being
synchronized separately; this data-only addition does not certify that the
current default code commands reproduce every result in the latest manuscript.
