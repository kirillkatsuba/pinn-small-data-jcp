"""Fast structural tests for the enlarged classical hyperparameter profile."""

from __future__ import annotations

import unittest

import numpy as np

from paper_jcp.classical_baselines import _gp, _svm, _xgb, classical_cv_splits


class ExtensiveClassicalProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        times = np.repeat(np.linspace(0.0, 0.7, 8), 4)
        self.features = np.column_stack((times, np.zeros((len(times), 2))))
        self.cv = classical_cv_splits(self.features, seed=101, strategy="time")

    def test_extensive_random_search_sizes(self) -> None:
        self.assertEqual(_xgb("extensive", self.cv, 101, 0).n_iter, 96)
        self.assertEqual(_svm("extensive", self.cv, 101, 0).n_iter, 96)

    def test_extensive_gp_grid_has_twenty_candidates(self) -> None:
        search = _gp("extensive", self.cv, 101, input_dim=3)
        self.assertEqual(
            len(search.param_grid["gp__kernel"]) * len(search.param_grid["gp__alpha"]),
            20,
        )
        self.assertEqual(search.estimator.named_steps["gp"].n_restarts_optimizer, 3)


if __name__ == "__main__":
    unittest.main()
