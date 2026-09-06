#!/usr/bin/env python3
"""Quantify state-constraint violations for the repeated flow experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.run_robustness_experiments import (
    BASE_SEEDS,
    CHECKPOINTS,
    CONFIGS,
    build_bundle,
    load_training_module,
    predict_classical,
    predict_pinn,
    select_points,
    tune_classical,
)


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "physical_constraint_results.csv"
METHODS = ["HO-PINN", "XGB", "SVM", "GP"]


def load_repeated_pinn(test_name: str, module, seed: int):
    cfg = CONFIGS[test_name]
    model_class = module.PINN if test_name == "test2" else module.ModifiedPINN
    model = model_class(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden"],
        num_layers=cfg["layers"], output_dim=cfg["output_dim"],
    )
    path = CHECKPOINTS / f"{test_name}_repeated_s{seed}_noise0_r1.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def constraint_metrics(prediction: np.ndarray) -> dict:
    pressure = prediction[:, 0]
    soil = prediction[:, 1]
    swat = prediction[:, 2]
    tolerance = 1e-3
    invalid_saturation = (
        (soil < -tolerance) | (soil > 1.0 + tolerance)
        | (swat < -tolerance) | (swat > 1.0 + tolerance)
    )
    closure = np.abs(soil + swat - 1.0)
    return {
        "negative_pressure_rate": float(np.mean(pressure < -tolerance)),
        "saturation_violation_rate": float(np.mean(invalid_saturation)),
        "closure_mae": float(np.mean(closure)),
        "closure_max": float(np.max(closure)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests", nargs="+", choices=["test2", "test3", "test4"],
        default=["test2", "test3", "test4"],
    )
    args = parser.parse_args()
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []
    for test_name in args.tests:
        module = load_training_module(test_name)
        bundle = build_bundle(test_name, module)
        for seed in BASE_SEEDS:
            if len(existing):
                mask = (existing.test_name == test_name) & (existing.seed == seed)
                if int(mask.sum()) == 4:
                    print(f"[{test_name}] skip physical constraints seed={seed}", flush=True)
                    continue
            selected = select_points(test_name, module, bundle, seed, ratio=1)
            model = load_repeated_pinn(test_name, module, seed)
            prediction = predict_pinn(test_name, model, bundle["test_x"])
            rows.append({
                "test_name": test_name, "seed": seed, "method": "HO-PINN",
                "n_test": len(prediction), **constraint_metrics(prediction),
            })
            for method in METHODS[1:]:
                estimators, _ = tune_classical(
                    method, selected["x_data"], selected["y_data"], seed
                )
                prediction = predict_classical(
                    estimators, bundle["test_x"], scalar=False
                )
                rows.append({
                    "test_name": test_name, "seed": seed, "method": method,
                    "n_test": len(prediction), **constraint_metrics(prediction),
                })
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)
            existing = pd.DataFrame(rows)
            print(f"[{test_name}] physical constraints seed={seed} complete", flush=True)
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
