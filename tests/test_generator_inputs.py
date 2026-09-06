from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paper_jcp.generate_corrected_benchmarks import build_specs


class GeneratorInputTests(unittest.TestCase):
    def test_publication_inputs_are_explicit_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            parameter_dir = inputs / "parameters"
            parameter_dir.mkdir(parents=True)
            for name in ("pwat", "poil", "kwat", "koil"):
                np.save(parameter_dir / f"{name}_100.npy", np.ones(3))

            args = argparse.Namespace(
                test="all",
                output_root=root / "outputs",
                output_steps=100,
                final_time_test2=0.3,
                final_time_test3=0.05,
                final_time_test4=0.2,
                test4_samples=3,
                porosity=0.1,
                water_viscosity=1.0,
                oil_viscosity=3.0,
                cfl=0.2,
                workers=1,
                overwrite=False,
                permeability_test2=inputs / "test2.npy",
                permeability_test3=inputs / "test3.npy",
                permeability_test4=inputs / "test4.npy",
                test4_parameter_dir=parameter_dir,
            )
            specs = build_specs(args)
            self.assertEqual(len(specs), 5)
            self.assertEqual(
                Path(specs[0]["permeability_path"]), args.permeability_test2
            )
            self.assertEqual(
                Path(specs[1]["permeability_path"]), args.permeability_test3
            )
            self.assertTrue(
                all(
                    Path(spec["permeability_path"]) == args.permeability_test4
                    for spec in specs[2:]
                )
            )
            self.assertEqual(
                [spec["final_time"] for spec in specs],
                [0.3, 0.05, 0.2, 0.2, 0.2],
            )


if __name__ == "__main__":
    unittest.main()
