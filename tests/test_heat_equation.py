"""Tests for the self-contained analytical Test 1 benchmark."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from paper_jcp.unified_protocol import (
    build_unified_bundle,
    heat_exact_solution,
)


class HeatEquationTest(unittest.TestCase):
    def test_bundle_is_analytical_and_temporally_disjoint(self) -> None:
        bundle = build_unified_bundle("test1", time_fraction=0.7)
        self.assertAlmostEqual(bundle.full_time, 0.05)
        self.assertAlmostEqual(bundle.time_cutoff, 0.035)
        self.assertLessEqual(float(bundle.interpolation_x[:, 0].max()), 0.035 + 1e-7)
        self.assertGreater(float(bundle.extrapolation_x[:, 0].min()), 0.035)

    def test_exact_solution_matches_initial_condition(self) -> None:
        x = torch.linspace(0.0, 1.0, 257)
        truth = 2.0 + torch.sin(2.0 * torch.pi * x) + torch.cos(4.0 * torch.pi * x)
        prediction = heat_exact_solution(torch.zeros_like(x), x)
        np.testing.assert_allclose(prediction.numpy(), truth.numpy(), atol=1e-6)

if __name__ == "__main__":
    unittest.main()
