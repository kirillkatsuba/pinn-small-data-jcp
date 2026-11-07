# Fixed absolute-permeability fields

These are the three author-generated absolute-permeability fields used in the
published flow benchmarks. Each file is a NumPy array with shape `(64, 64)`.
The files are exact copies of `reproducibility/data_inputs/test*/permeability.npy`.

| File | Benchmark | Use |
| --- | --- | --- |
| `test2.npy` | Test 2 | One fixed field for `test2/solution.npz` |
| `test3.npy` | Test 3 | One fixed field for `test3/solution.npz` |
| `test4.npy` | Test 4 | One shared fixed field for all 100 parameter combinations |

The same arrays are also embedded under the `permeability` key in every
corresponding solution `.npz` file. Array indices match the solution files;
use their `x` and `y` arrays for the spatial coordinates. These fields represent
absolute permeability, rather than the saturation-dependent relative
permeabilities varied in Test 4. Values retain the benchmark's numerical scale;
no physical-unit conversion or normalization has been applied.

```python
import numpy as np

K = np.load("permeability/test2.npy", allow_pickle=False)
```
