from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from paper_jcp.merge_publication_shards import merge_experiment
from paper_jcp.run_publication_suite import (
    DEFAULT_BALANCERS,
    DEFAULT_BUDGETS,
    DEFAULT_EPOCHS,
    DEFAULT_SEEDS,
    command,
)
from paper_jcp.unified_protocol import ModelConfig, TrainConfig, configuration_id


class PublicationDefaultTests(unittest.TestCase):
    def test_candidate_defaults_are_ten_seeds_and_fifty_thousand_epochs(self) -> None:
        self.assertEqual(
            tuple(int(seed) for seed in DEFAULT_SEEDS.split(",")),
            tuple(range(101, 111)),
        )
        self.assertEqual(DEFAULT_EPOCHS, 50_000)
        self.assertEqual(DEFAULT_BUDGETS, "20,50,100,200,500,1000")
        self.assertEqual(DEFAULT_BALANCERS, "h,ho")
        self.assertEqual(TrainConfig().epochs, 50_000)

    def test_epoch_count_is_part_of_configuration_identity(self) -> None:
        short = configuration_id("h", ModelConfig(), TrainConfig(epochs=10_000))
        long = configuration_id("h", ModelConfig(), TrainConfig(epochs=50_000))
        self.assertNotEqual(short, long)
        self.assertNotIn("_e10000", short)
        self.assertIn("_e50000", long)

    def test_publication_command_propagates_seed_epoch_and_shard(self) -> None:
        invocation = command(
            "test3",
            "pinn",
            1,
            "cuda:0",
            True,
            Path("data"),
            "107",
            50_000,
            "seed107",
        )
        rendered = " ".join(str(item) for item in invocation)
        self.assertIn("--seeds 107", rendered)
        self.assertIn("--epochs 50000", rendered)
        self.assertIn("--result-suffix seed107", rendered)

    def test_publication_command_can_select_one_budget_and_balancer(self) -> None:
        invocation = command(
            "test4",
            "pinn",
            1,
            "cuda:0",
            True,
            Path("data"),
            "101",
            50_000,
            "seed101_n20_h",
            budgets="20",
            balancers="h",
        )
        rendered = " ".join(str(item) for item in invocation)
        self.assertIn("--budgets 20", rendered)
        self.assertIn("--balancers h", rendered)

    def test_seed_shards_merge_into_canonical_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory)
            experiment = generated / "example"
            experiment.mkdir()
            for seed in (101, 102):
                pd.DataFrame(
                    {
                        "test_name": ["test1", "test1"],
                        "seed": [seed, seed],
                        "n_train": [20, 20],
                        "method": ["H-PINN", "H-PINN"],
                        "config_id": ["candidate_e50000", "candidate_e50000"],
                        "evaluation_region": ["interpolation", "extrapolation"],
                    }
                ).to_csv(experiment / f"unified_test1_seed{seed}.csv", index=False)
            with patch("paper_jcp.merge_publication_shards.GENERATED", generated):
                output = merge_experiment("example", "test1", (101, 102))
            merged = pd.read_csv(output)
            self.assertEqual(len(merged), 4)
            self.assertEqual(set(merged["seed"]), {101, 102})

    def test_split_budget_shards_merge_into_canonical_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generated = Path(directory)
            experiment = generated / "example"
            experiment.mkdir()
            for budget in (20, 50):
                pd.DataFrame(
                    {
                        "test_name": ["test4"],
                        "seed": [101],
                        "n_train": [budget],
                        "method": ["H-PINN"],
                        "config_id": ["candidate_e50000"],
                        "evaluation_region": ["interpolation"],
                    }
                ).to_csv(
                    experiment / f"unified_test4_seed101_n{budget}_h.csv",
                    index=False,
                )
            with patch("paper_jcp.merge_publication_shards.GENERATED", generated):
                output = merge_experiment("example", "test4", (101,))
            merged = pd.read_csv(output)
            self.assertEqual(set(merged["n_train"]), {20, 50})


if __name__ == "__main__":
    unittest.main()
