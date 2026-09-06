"""Tests for corrected-data integration in the unified JCP protocol."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from paper_jcp.unified_protocol import (
    ModelConfig,
    ResidualSampler,
    build_model,
    build_unified_bundle,
    flow_loss_components,
    nested_label_indices,
)


class CorrectedUnifiedProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        times = np.linspace(0.0, 3.0, 101)
        x = (np.arange(5) + 0.5) / 5.0
        y = (np.arange(5) + 0.5) / 5.0
        time_grid, x_grid, y_grid = np.meshgrid(times, x, y, indexing="ij")
        pressure = np.broadcast_to(1.0 - x_grid, time_grid.shape).copy()
        swat = 0.4 * time_grid / times[-1] * (1.0 - 0.2 * y_grid)
        soil = 1.0 - swat
        permeability = 1.0 + np.add.outer(x, y)
        porosity = np.full((len(x), len(y)), 0.1)
        metadata = {
            "schema": "corrected-two-phase-fv-v1",
            "fluids": {"water_viscosity": 1.0, "oil_viscosity": 3.0},
        }
        for test_name in ("test2", "test3"):
            dataset_dir = root / test_name
            dataset_dir.mkdir(parents=True)
            np.savez_compressed(
                dataset_dir / "solution.npz",
                times=times,
                x=x,
                y=y,
                pressure=pressure,
                water_saturation=swat,
                oil_saturation=soil,
                permeability=permeability,
                porosity=porosity,
                metadata_json=np.asarray(json.dumps(metadata)),
            )
        test4_dir = root / "test4"
        test4_dir.mkdir()
        for sample in range(4):
            test4_metadata = {
                **metadata,
                "boundary_conditions": {
                    "left_pressure": 1.0,
                    "right_pressure": 0.0,
                    "left_inflow_water_fraction": 1.0,
                },
                "relative_permeability": {
                    "water_exponent": 1.7 + 0.2 * sample,
                    "oil_exponent": 3.2 + 0.4 * sample,
                    "water_endpoint": 0.94 + 0.04 * sample,
                    "oil_endpoint": 0.23 + 0.04 * sample,
                },
            }
            sample_swat = np.clip(swat * (0.8 + 0.1 * sample), 0.0, 1.0)
            np.savez_compressed(
                test4_dir / f"sample_{sample:03d}.npz",
                times=times,
                x=x,
                y=y,
                pressure=pressure,
                water_saturation=sample_swat,
                oil_saturation=1.0 - sample_swat,
                permeability=permeability,
                porosity=porosity,
                metadata_json=np.asarray(json.dumps(test4_metadata)),
            )
        self.root = root

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_corrected_loader_and_temporal_split(self) -> None:
        bundle = build_unified_bundle(
            "test2",
            data_source="corrected",
            corrected_data_root=self.root,
            interpolation_size=20,
            extrapolation_size=20,
        )
        self.assertEqual(bundle.data_source, "corrected")
        self.assertEqual(bundle.features.shape, (2525, 3))
        self.assertEqual(bundle.targets.shape, (2525, 3))
        self.assertEqual(bundle.permeability.shape, (2525,))
        self.assertAlmostEqual(bundle.full_time, 3.0)
        self.assertAlmostEqual(bundle.time_cutoff, 2.1)
        self.assertTrue(np.all(bundle.interpolation_x[:, 0] <= 2.1 + 1e-7))
        self.assertTrue(np.all(bundle.extrapolation_x[:, 0] > 2.1 + 1e-7))
        self.assertTrue(np.allclose(bundle.targets[:, 1] + bundle.targets[:, 2], 1.0))
        self.assertEqual(bundle.input_upper, (3.0, 1.0, 1.0))

    def test_corrected_loss_is_differentiable(self) -> None:
        bundle = build_unified_bundle(
            "test2",
            data_source="corrected",
            corrected_data_root=self.root,
            interpolation_size=20,
            extrapolation_size=20,
        )
        indices = nested_label_indices(bundle, 101, max_budget=20, time_bins=5)
        sampler = ResidualSampler(bundle, 501, "full", exclude_indices=indices)
        residual_x, residual_perm = sampler.sample(20)
        model = build_model(
            bundle, ModelConfig(hidden_dim=16, depth=2, activation="gelu")
        )
        losses = flow_loss_components(
            bundle,
            model,
            torch.tensor(bundle.features[indices], dtype=torch.float32),
            torch.tensor(bundle.targets[indices], dtype=torch.float32),
            torch.tensor(residual_x, dtype=torch.float32),
            torch.tensor(residual_perm, dtype=torch.float32),
        )
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_corrected_test3_uses_dataset_time_bounds(self) -> None:
        bundle = build_unified_bundle(
            "test3",
            data_source="corrected",
            corrected_data_root=self.root,
            interpolation_size=20,
            extrapolation_size=20,
        )
        self.assertEqual(bundle.data_source, "corrected")
        self.assertTrue(bundle.dataset_path.endswith("test3/solution.npz"))
        self.assertAlmostEqual(bundle.full_time, 3.0)
        self.assertEqual(bundle.input_upper, (3.0, 1.0, 1.0))
        self.assertTrue(np.all(bundle.interpolation_x[:, 0] <= 2.1 + 1e-7))
        self.assertTrue(np.all(bundle.extrapolation_x[:, 0] > 2.1 + 1e-7))

    def test_corrected_test4_loader_and_loss(self) -> None:
        bundle = build_unified_bundle(
            "test4",
            data_source="corrected",
            corrected_data_root=self.root,
            interpolation_size=20,
            extrapolation_size=20,
            corrected_test4_pool_size=2000,
        )
        self.assertEqual(bundle.features.shape, (2040, 7))
        self.assertEqual(bundle.targets.shape, (2040, 3))
        self.assertEqual(len(bundle.train_candidates), 1000)
        self.assertEqual(len(bundle.residual_candidates_full), 2000)
        self.assertEqual(len(bundle.residual_candidates_label), 1000)
        self.assertEqual(bundle.interpolation_x.shape, (20, 7))
        self.assertEqual(bundle.extrapolation_x.shape, (20, 7))
        self.assertEqual(len(np.unique(bundle.features[:, 3], axis=0)), 4)
        self.assertEqual(bundle.input_upper, (3.0, 1.0, 1.0, 2.5, 5.0, 1.1, 0.4))
        self.assertTrue(np.all(bundle.interpolation_x[:, 0] <= 2.1 + 1e-7))
        self.assertTrue(np.all(bundle.extrapolation_x[:, 0] > 2.1 + 1e-7))

        indices = nested_label_indices(bundle, 101, max_budget=20, time_bins=5)
        sampler = ResidualSampler(bundle, 501, "full", exclude_indices=indices)
        residual_x, residual_perm = sampler.sample(20)
        model = build_model(
            bundle, ModelConfig(hidden_dim=16, depth=2, activation="gelu")
        )
        losses = flow_loss_components(
            bundle,
            model,
            torch.tensor(bundle.features[indices], dtype=torch.float32),
            torch.tensor(bundle.targets[indices], dtype=torch.float32),
            torch.tensor(residual_x, dtype=torch.float32),
            torch.tensor(residual_perm, dtype=torch.float32),
        )
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
