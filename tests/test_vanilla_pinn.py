from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from paper_jcp.run_publication_suite import command
from paper_jcp.unified_protocol import (
    ModelConfig,
    TrainConfig,
    build_unified_bundle,
    configuration_id,
    make_balancer,
    nested_heat_labels,
    predict_pinn,
    train_pinn,
)


class VanillaPINNTests(unittest.TestCase):
    def test_vanilla_balancer_uses_unit_weights(self) -> None:
        model = torch.nn.Linear(1, 1)
        prediction = model(torch.ones(1, 1)).sum()
        losses = (
            prediction.square(),
            (prediction - 1).square(),
            (prediction + 2).square(),
        )
        balancer = make_balancer("vanilla", TrainConfig())

        weights = balancer.compute_weights(model, *losses)

        for weight in weights[:3]:
            self.assertEqual(float(weight), 1.0)
            self.assertFalse(weight.requires_grad)
        self.assertEqual(weights[3]["a_mse"], 1.0)
        self.assertEqual(weights[3]["a_ibc"], 1.0)
        self.assertEqual(weights[3]["a_pde"], 1.0)

    def test_vanilla_has_a_distinct_configuration_identity(self) -> None:
        vanilla = configuration_id("vanilla", ModelConfig(), TrainConfig())
        hierarchical = configuration_id("h", ModelConfig(), TrainConfig())
        self.assertTrue(vanilla.startswith("vanilla-unit_"))
        self.assertNotEqual(vanilla, hierarchical)

    def test_publication_runner_propagates_vanilla_mode(self) -> None:
        invocation = command(
            "test2",
            "pinn",
            1,
            "cuda:0",
            True,
            Path("data"),
            "101",
            50_000,
            "seed101_vanilla",
            balancers="vanilla",
        )
        rendered = " ".join(str(item) for item in invocation)
        self.assertIn("--balancers vanilla", rendered)

    def test_vanilla_training_writes_portable_checkpoint_and_unit_history(self) -> None:
        bundle = build_unified_bundle(
            "test1", interpolation_size=16, extrapolation_size=16
        )
        x_data, y_data = nested_heat_labels(
            bundle, seed=101, max_budget=20, time_bins=2
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = train_pinn(
                bundle=bundle,
                x_data_np=x_data,
                y_data_np=y_data,
                label_indices=None,
                label_budget=20,
                seed=101,
                balance_mode="vanilla",
                model_config=ModelConfig(hidden_dim=8, depth=1, activation="gelu"),
                train_config=TrainConfig(
                    epochs=2,
                    checkpoint_epochs=(1, 2),
                    learning_rate=1e-3,
                    collocation_ratio=0.1,
                    log_every=1,
                    physics_validation_size=4,
                    device="cpu",
                    deterministic=True,
                ),
                checkpoint_path=root / "checkpoint.pth",
                history_path=root / "history.csv",
            )
            history = pd.read_csv(root / "history.csv")
            self.assertTrue(
                np.allclose(
                    history[["weight_data", "weight_ibc", "weight_pde"]],
                    1.0,
                )
            )
            self.assertTrue(
                np.allclose(history["balanced_total"], history["raw_total"])
            )
            prediction = predict_pinn("test1", model, bundle.interpolation_x[:4])
            self.assertTrue(np.isfinite(prediction).all())
            payload = torch.load(
                root / "checkpoint.pth", map_location="cpu", weights_only=False
            )
            self.assertEqual(payload["balance_mode"], "vanilla")
            self.assertEqual(payload["checkpoint_kind"], "best")
            self.assertIn(payload["best_epoch"], {1, 2})
            self.assertTrue(
                all(
                    value.device.type == "cpu"
                    for value in payload["state_dict"].values()
                )
            )
            for epoch in (1, 2):
                snapshot = root / f"checkpoint.epoch{epoch:06d}.pth"
                self.assertTrue(snapshot.exists())
                snapshot_payload = torch.load(
                    snapshot, map_location="cpu", weights_only=False
                )
                self.assertEqual(snapshot_payload["checkpoint_kind"], "periodic")
                self.assertEqual(snapshot_payload["epoch"], epoch)
                self.assertTrue(
                    all(
                        value.device.type == "cpu"
                        for value in snapshot_payload["state_dict"].values()
                    )
                )


if __name__ == "__main__":
    unittest.main()
