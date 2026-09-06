#!/usr/bin/env python3
"""Regression tests for classical-model tuning in the unified protocol."""

from __future__ import annotations

import unittest

import numpy as np

from paper_jcp.classical_baselines import (
    classical_cv_splits,
    tune_classical,
)


class ClassicalTuningTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(17)
        times = np.repeat(np.linspace(0.0, 0.7, 8), 4)
        space = rng.uniform(0.0, 1.0, size=(len(times), 2))
        self.features = np.column_stack((times, space))
        self.target = (
            np.sin(np.pi * space[:, 0]) * np.exp(-times)
            + 0.05 * space[:, 1]
        )

    def test_temporal_cv_is_strictly_forward_looking(self) -> None:
        splits = classical_cv_splits(self.features, seed=101, strategy="time")
        self.assertEqual(len(splits), 3)
        previous_training_size = 0
        for train_indices, validation_indices in splits:
            train_times = self.features[train_indices, 0]
            validation_times = self.features[validation_indices, 0]
            self.assertLess(float(train_times.max()), float(validation_times.min()))
            self.assertGreater(len(train_indices), previous_training_size)
            previous_training_size = len(train_indices)

    def test_strong_svr_returns_reproducible_tuning_metadata(self) -> None:
        estimators, elapsed, details = tune_classical(
            "SVM",
            self.features,
            self.target,
            seed=101,
            tuning_profile="strong",
            cv_strategy="time",
            return_metadata=True,
        )
        self.assertEqual(len(estimators), 1)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(details[0]["target_name"], "target")
        self.assertIn("regressor__svr__C", details[0]["best_params"])
        self.assertIsInstance(details[0]["best_cv_neg_mse"], float)

    def test_standard_profile_keeps_two_value_api(self) -> None:
        result = tune_classical(
            "SVM",
            self.features,
            self.target,
            seed=101,
        )
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
