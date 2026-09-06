#!/usr/bin/env python3
"""Compare uniform and front-aware labelled sampling on Test 3."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.run_robustness_experiments import (
    CHECKPOINTS,
    CONFIGS,
    STRESS_SEEDS,
    build_bundle,
    load_training_module,
    metrics,
    predict_classical,
    predict_pinn,
    train_pinn,
    tune_classical,
)


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "sampling_sensitivity_test3.csv"
METHODS = ["HO-PINN", "XGB", "SVM", "GP"]


def front_aware_selection(bundle: dict, seed: int) -> dict:
    candidates = bundle["train_candidates"]
    active = candidates[bundle["targets"][candidates, 2] > 0.01]
    inactive = candidates[bundle["targets"][candidates, 2] <= 0.01]
    rng = np.random.default_rng(seed)
    data_idx = np.concatenate(
        (rng.choice(active, size=25, replace=False),
         rng.choice(inactive, size=25, replace=False))
    )
    rng.shuffle(data_idx)
    colloc_idx = np.random.default_rng(seed + 500).choice(
        candidates, size=CONFIGS["test3"]["n"], replace=False
    )
    return {
        "x_data": bundle["features"][data_idx],
        "y_data": bundle["targets"][data_idx],
        "x_colloc": bundle["features"][colloc_idx],
        "perm_colloc": bundle["perm"][colloc_idx],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=STRESS_SEEDS)
    args = parser.parse_args()
    module = load_training_module("test3")
    bundle = build_bundle("test3", module)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []
    for seed in args.seeds:
        if len(existing) and int((existing.seed == seed).sum()) == 4:
            print(f"[test3 front-aware] skip seed={seed}", flush=True)
            continue
        selected = front_aware_selection(bundle, seed)
        checkpoint = CHECKPOINTS / f"test3_front_aware_s{seed}.pth"
        model, train_time = train_pinn(
            "test3", module, selected, selected["y_data"], seed,
            args.epochs, checkpoint,
        )
        start = time.perf_counter()
        prediction = predict_pinn("test3", model, bundle["test_x"])
        condition_rows = [{
            "test_name": "test3", "experiment": "front_aware",
            "seed": seed, "noise_level": 0.0, "collocation_ratio": 1,
            "method": "HO-PINN", "n_train": 50, "n_collocation": 50,
            "n_test": len(bundle["test_y"]), "train_time_sec": train_time,
            "inference_time_sec": time.perf_counter() - start,
            **metrics("test3", bundle["test_y"], prediction),
        }]
        for method in METHODS[1:]:
            estimators, fit_time = tune_classical(
                method, selected["x_data"], selected["y_data"], seed
            )
            start = time.perf_counter()
            prediction = predict_classical(estimators, bundle["test_x"], scalar=False)
            condition_rows.append({
                "test_name": "test3", "experiment": "front_aware",
                "seed": seed, "noise_level": 0.0, "collocation_ratio": 1,
                "method": method, "n_train": 50, "n_collocation": 0,
                "n_test": len(bundle["test_y"]), "train_time_sec": fit_time,
                "inference_time_sec": time.perf_counter() - start,
                **metrics("test3", bundle["test_y"], prediction),
            })
        if len(existing):
            existing = existing[existing.seed != seed]
            rows = existing.to_dict("records")
        rows.extend(condition_rows)
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        existing = pd.DataFrame(rows)
        print(f"[test3 front-aware] seed={seed} complete ({train_time:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
