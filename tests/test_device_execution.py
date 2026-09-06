from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from paper_jcp.unified_protocol import (
    ModelConfig,
    TrainConfig,
    build_unified_bundle,
    nested_heat_labels,
    predict_pinn,
    resolve_device,
    train_pinn,
)


class DeviceExecutionTests(unittest.TestCase):
    def test_cpu_and_auto_resolution(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertIn(resolve_device("auto").type, {"cpu", "cuda", "mps"})

    def test_unavailable_cuda_has_clear_error(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("CUDA is available on this host")
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            resolve_device("cuda")

    def _train_smoke(self, device: str) -> None:
        bundle = build_unified_bundle(
            "test1", interpolation_size=32, extrapolation_size=32
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
                balance_mode="h",
                model_config=ModelConfig(hidden_dim=8, depth=1, activation="gelu"),
                train_config=TrainConfig(
                    epochs=2,
                    learning_rate=1e-3,
                    collocation_ratio=0.1,
                    log_every=1,
                    physics_validation_size=4,
                    device=device,
                    deterministic=True,
                ),
                checkpoint_path=root / "checkpoint.pth",
                history_path=root / "history.csv",
            )
            prediction = predict_pinn("test1", model, bundle.interpolation_x[:4])
            self.assertEqual(prediction.shape, (4,))
            self.assertTrue(np.isfinite(prediction).all())
            payload = torch.load(
                root / "checkpoint.pth", map_location="cpu", weights_only=False
            )
            self.assertTrue(
                all(value.device.type == "cpu" for value in payload["state_dict"].values())
            )

    def test_cpu_training_and_portable_checkpoint(self) -> None:
        self._train_smoke("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_training_and_portable_checkpoint(self) -> None:
        self._train_smoke("cuda")


if __name__ == "__main__":
    unittest.main()
