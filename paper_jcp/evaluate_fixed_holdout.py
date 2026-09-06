#!/usr/bin/env python3
"""Evaluate HO-PINN and classical surrogates on fixed disjoint test sets.

The original experimental pipeline aggregates each trained model over the other
pre-generated selections.  This script adds a stricter JCP-oriented protocol:

* Test 1 is evaluated on a deterministic dense space-time grid.
* Tests 2--4 are evaluated on a deterministic subset of the simulator pool that
  is disjoint from the union of every stored training selection.
* XGB, SVR, and GP use exactly the same labelled points as the PINN for each N.

No PINN is retrained.  The script loads the saved best HO-PINN checkpoints and
fits the inexpensive classical baselines from scratch.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
GENERATED_DIR = PAPER_DIR / "generated"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarking.pinn_ablation_utils import (  # noqa: E402
    compute_multioutput_metrics,
    compute_scalar_metrics,
    load_module,
)
from test_ml_pinn.generate_ml_pinn_results import (  # noqa: E402
    fit_ml_model,
    predict_ml_model,
)


TEST_CONFIGS = {
    "test1": {
        "directory": "test1-therm-conduct",
        "model_class": "PINN",
        "input_dim": 2,
        "output_dim": 1,
        "target_names": None,
    },
    "test2": {
        "directory": "test2-2phase-disp-simple",
        "model_class": "PINN",
        "input_dim": 3,
        "output_dim": 7,
        "target_names": ["Pressure", "Soil", "Swat"],
    },
    "test3": {
        "directory": "test3-2phase-disp-perm-nonlinear",
        "model_class": "ModifiedPINN",
        "input_dim": 3,
        "output_dim": 7,
        "target_names": ["Pressure", "Soil", "Swat"],
    },
    "test4": {
        "directory": "test4-2phase-disp-7d",
        "model_class": "ModifiedPINN",
        "input_dim": 7,
        "output_dim": 7,
        "target_names": ["Pressure", "Soil", "Swat"],
    },
}

ML_METHODS = ("XGB", "SVM", "GP")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_training_module(test_name: str):
    config = TEST_CONFIGS[test_name]
    test_dir = ROOT_DIR / config["directory"]
    for name in ("diffusion_equation", "train_logic"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(test_dir))
    try:
        module = load_module(
            test_dir / "train_pinn.py",
            f"jcp_fixed_holdout_{test_name}_train",
        )
    finally:
        try:
            sys.path.remove(str(test_dir))
        except ValueError:
            pass
    return module


def current_artifact_path(test_dir: Path, old_path: str) -> Path:
    """Map absolute paths from the source clone to the current repository."""
    path = Path(str(old_path))
    if path.exists():
        return path
    candidate = test_dir / "ablation_results" / "training_runs" / path.name
    if not candidate.exists():
        raise FileNotFoundError(f"Could not resolve artifact {old_path}")
    return candidate


def load_ho_rows(test_name: str) -> pd.DataFrame:
    test_dir = ROOT_DIR / TEST_CONFIGS[test_name]["directory"]
    results = pd.read_csv(test_dir / "ablation_results" / "ablation_results.csv")
    rows = results[results["balancer"] == "grad-orth"].copy()
    return rows.sort_values("n_collocation").reset_index(drop=True)


def load_test1_pool(module):
    test_dir = ROOT_DIR / TEST_CONFIGS["test1"]["directory"]
    t_grid = torch.linspace(0.0, 0.05, 81, dtype=torch.float32)
    x_grid = torch.linspace(0.0, 1.0, 257, dtype=torch.float32)
    test_features_torch = torch.cartesian_prod(t_grid, x_grid)
    test_targets = (
        module.thermal_conductivity_equation(
            test_features_torch[:, 0], test_features_torch[:, 1]
        )
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    def training_set(n_points: int):
        x = torch.load(test_dir / "data" / f"x_{n_points}.pt", map_location="cpu")
        t = torch.load(test_dir / "data" / f"t_{n_points}.pt", map_location="cpu")
        features = torch.stack((t, x), dim=-1)
        targets = (
            module.thermal_conductivity_equation(t, x)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        return features.detach().numpy().astype(np.float32), targets

    return {
        "test_features_np": test_features_torch.numpy().astype(np.float32),
        "test_features_torch": test_features_torch,
        "test_targets": test_targets,
        "training_set": training_set,
        "test_size": len(test_targets),
    }


def load_test23_pool(test_name: str):
    config = TEST_CONFIGS[test_name]
    test_dir = ROOT_DIR / config["directory"]
    if test_name == "test2":
        data_dir = test_dir / "data_5k_200225"
        perm_path = test_dir / "perm.npy"
    else:
        data_dir = test_dir / "data_5k_3sigma"
        perm_path = test_dir / "perm_3sigma.npy"

    perm = np.load(perm_path)
    nx0, nx1 = perm.shape
    sim = np.loadtxt(data_dir / "sim_5000.txt", dtype=np.float32)
    x = np.loadtxt(data_dir / "x_5000.txt", dtype=np.float32) / float(nx0)
    y = np.loadtxt(data_dir / "y_5000.txt", dtype=np.float32) / float(nx1)
    t = np.loadtxt(data_dir / "t_5000.txt", dtype=np.float32) * (3.0 / 5000.0)
    features = np.column_stack((t, x, y)).astype(np.float32)
    targets = sim[:, [0, 2, 1]].astype(np.float32)

    selection_paths = sorted(test_dir.glob("sim_*.npy"))
    used = np.unique(np.concatenate([np.load(path).astype(int) for path in selection_paths]))
    available = np.setdiff1d(np.arange(len(features)), used, assume_unique=False)
    rng = np.random.default_rng(20260727)
    test_indices = rng.choice(available, size=min(1500, len(available)), replace=False)

    def training_set(n_points: int):
        indices = np.load(test_dir / f"sim_{n_points}.npy").astype(int)
        return features[indices], targets[indices]

    return {
        "test_features_np": features[test_indices],
        "test_features_torch": torch.tensor(features[test_indices], dtype=torch.float32),
        "test_targets": targets[test_indices],
        "training_set": training_set,
        "test_size": len(test_indices),
    }


def load_test4_pool(module):
    test_dir = ROOT_DIR / TEST_CONFIGS["test4"]["directory"]
    set_seed(42)
    (
        simulation_data,
        _x_idx,
        _y_idx,
        x,
        y,
        t,
        pwat,
        poil,
        kwat,
        koil,
    ) = module.prepare_candidate_points(
        test_dir, "data-100-new-two-sigma/sim_100.npy"
    )
    features = np.column_stack((t, x, y, pwat, poil, kwat, koil)).astype(np.float32)
    targets = simulation_data[:, [0, 2, 1]].astype(np.float32)

    selection_paths = sorted(test_dir.glob("train-indexes-2sigma-*.npy"))
    used = np.unique(np.concatenate([np.load(path).astype(int) for path in selection_paths]))
    available = np.setdiff1d(np.arange(len(features)), used, assume_unique=False)
    rng = np.random.default_rng(20260727)
    test_indices = rng.choice(available, size=min(2000, len(available)), replace=False)

    def training_set(n_points: int):
        indices = np.load(test_dir / f"train-indexes-2sigma-{n_points}.npy").astype(int)
        return features[indices], targets[indices]

    return {
        "test_features_np": features[test_indices],
        "test_features_torch": torch.tensor(features[test_indices], dtype=torch.float32),
        "test_targets": targets[test_indices],
        "training_set": training_set,
        "test_size": len(test_indices),
    }


def load_pool(test_name: str, module):
    if test_name == "test1":
        return load_test1_pool(module)
    if test_name in {"test2", "test3"}:
        return load_test23_pool(test_name)
    return load_test4_pool(module)


def compute_metrics(test_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    target_names = TEST_CONFIGS[test_name]["target_names"]
    if target_names is None:
        return compute_scalar_metrics(y_true.reshape(-1), y_pred.reshape(-1))
    return compute_multioutput_metrics(y_true, y_pred, target_names)


def load_pinn(test_name: str, module, row: pd.Series):
    config = TEST_CONFIGS[test_name]
    test_dir = ROOT_DIR / config["directory"]
    summary_path = current_artifact_path(test_dir, row["summary_path"])
    summary = json.loads(summary_path.read_text())
    args = summary["args"]
    model_class = getattr(module, config["model_class"])
    model = model_class(
        input_dim=config["input_dim"],
        hidden_dim=int(args["hidden_dim"]),
        num_layers=int(args["num_layers"]),
        output_dim=config["output_dim"],
    )
    checkpoint = current_artifact_path(test_dir, row["best_checkpoint"])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def predict_pinn(test_name: str, model, features: torch.Tensor) -> np.ndarray:
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), 4096):
            parts.append(model(features[start : start + 4096]).cpu().numpy())
    prediction = np.concatenate(parts, axis=0)
    if test_name == "test1":
        return prediction.reshape(-1)
    return prediction[:, [0, 1, 2]]


def evaluate_test(test_name: str) -> list[dict]:
    print(f"Evaluating {test_name}")
    module = load_training_module(test_name)
    pool = load_pool(test_name, module)
    rows = []
    for _, ho_row in load_ho_rows(test_name).iterrows():
        n_points = int(ho_row["n_collocation"])
        x_train, y_train = pool["training_set"](n_points)

        model = load_pinn(test_name, module, ho_row)
        start = time.perf_counter()
        y_pred = predict_pinn(test_name, model, pool["test_features_torch"])
        inference_time = time.perf_counter() - start
        rows.append(
            {
                "test_name": test_name,
                "method": "HO-PINN",
                "model_family": "PINN",
                "n_train": n_points,
                "n_test": pool["test_size"],
                "train_time_sec": float(ho_row["total_train_time_sec"]),
                "inference_time_sec": inference_time,
                **compute_metrics(test_name, pool["test_targets"], y_pred),
            }
        )

        for method in ML_METHODS:
            estimator, train_time = fit_ml_model(method, x_train, y_train)
            start = time.perf_counter()
            y_pred = predict_ml_model(estimator, pool["test_features_np"])
            inference_time = time.perf_counter() - start
            rows.append(
                {
                    "test_name": test_name,
                    "method": method,
                    "model_family": "classical surrogate",
                    "n_train": n_points,
                    "n_test": pool["test_size"],
                    "train_time_sec": train_time,
                    "inference_time_sec": inference_time,
                    **compute_metrics(test_name, pool["test_targets"], y_pred),
                }
            )
        print(f"  N={n_points} complete")
    return rows


def main() -> None:
    set_seed(42)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for test_name in TEST_CONFIGS:
        rows.extend(evaluate_test(test_name))
    results = pd.DataFrame(rows)
    output_path = GENERATED_DIR / "fixed_holdout_results.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
