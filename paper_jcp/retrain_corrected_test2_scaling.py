#!/usr/bin/env python3
"""Retrain the Test 2 label-budget sweep with the corrected oil residual."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.evaluate_fixed_holdout import fit_ml_model, predict_ml_model
from paper_jcp.run_robustness_experiments import (
    build_bundle,
    load_training_module,
    metrics,
    predict_pinn,
    train_pinn,
)


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "fixed_holdout_corrected_test2.csv"
CHECKPOINTS = PAPER / "experiments" / "checkpoints"
BUDGETS = [100, 200, 500, 1000]
METHODS = ["HO-PINN", "XGB", "SVM", "GP"]


def run_budget(n: int, epochs: int) -> list[dict]:
    module = load_training_module("test2")
    bundle = build_bundle("test2", module)
    indices = np.load(ROOT / "test2-2phase-disp-simple" / f"sim_{n}.npy").astype(int)
    selected = {
        "x_data": bundle["features"][indices],
        "y_data": bundle["targets"][indices],
        "x_colloc": bundle["features"][indices],
        "perm_colloc": bundle["perm"][indices],
    }
    checkpoint = CHECKPOINTS / f"test2_corrected_scaling_n{n}.pth"
    model, train_time = train_pinn(
        "test2", module, selected, selected["y_data"], 42, epochs, checkpoint,
        initialization_seed=42,
    )
    test_x = bundle["test_x"]
    test_y = bundle["test_y"]
    start = time.perf_counter()
    prediction = predict_pinn("test2", model, test_x)
    rows = [{
        "test_name": "test2", "method": "HO-PINN", "model_family": "PINN",
        "n_train": n, "n_test": len(test_y), "train_time_sec": train_time,
        "inference_time_sec": time.perf_counter() - start,
        **metrics("test2", test_y, prediction),
    }]
    for method in METHODS[1:]:
        estimator, fit_time = fit_ml_model(method, selected["x_data"], selected["y_data"])
        start = time.perf_counter()
        prediction = predict_ml_model(estimator, test_x)
        rows.append({
            "test_name": "test2", "method": method,
            "model_family": "classical surrogate", "n_train": n,
            "n_test": len(test_y), "train_time_sec": fit_time,
            "inference_time_sec": time.perf_counter() - start,
            **metrics("test2", test_y, prediction),
        })
    print(f"[test2 corrected scaling] N={n} complete ({train_time:.1f}s)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", choices=BUDGETS, default=BUDGETS)
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    completed = set()
    if len(existing):
        counts = existing.groupby("n_train").size()
        completed = {int(n) for n, count in counts.items() if count == 4}
    budgets = [n for n in args.budgets if n not in completed]
    rows = existing.to_dict("records") if len(existing) else []
    if not budgets:
        print(f"All corrected Test 2 budgets already present in {OUTPUT}")
        return
    with ProcessPoolExecutor(max_workers=min(args.workers, len(budgets))) as executor:
        futures = {executor.submit(run_budget, n, args.epochs): n for n in budgets}
        for future in as_completed(futures):
            rows.extend(future.result())
            pd.DataFrame(rows).sort_values(["n_train", "method"]).to_csv(OUTPUT, index=False)
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
