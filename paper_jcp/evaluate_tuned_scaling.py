#!/usr/bin/env python3
"""Fit training-only tuned classical baselines for every label budget."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.evaluate_fixed_holdout import load_ho_rows, load_pool, load_training_module
from paper_jcp.run_robustness_experiments import metrics, predict_classical, tune_classical


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "tuned_scaling_baselines.csv"
TESTS = ["test1", "test2", "test3", "test4"]
METHODS = ["XGB", "SVM", "GP"]


def evaluate_test(test_name: str) -> list[dict]:
    module = load_training_module(test_name)
    pool = load_pool(test_name, module)
    rows = []
    for n in load_ho_rows(test_name)["n_collocation"].astype(int):
        x_train, y_train = pool["training_set"](n)
        for method in METHODS:
            estimators, train_time = tune_classical(method, x_train, y_train, seed=42)
            start = time.perf_counter()
            prediction = predict_classical(
                estimators, pool["test_features_np"], scalar=test_name == "test1"
            )
            rows.append({
                "test_name": test_name, "method": method,
                "model_family": "classical surrogate", "n_train": n,
                "n_test": pool["test_size"], "train_time_sec": train_time,
                "inference_time_sec": time.perf_counter() - start,
                **metrics(test_name, pool["test_targets"], prediction),
            })
        print(f"[{test_name} tuned scaling] N={n} complete", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=TESTS)
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    completed = set()
    if len(existing):
        counts = existing.groupby("test_name").size()
        completed = {test for test, count in counts.items() if count == 12}
    tests = [test for test in args.tests if test not in completed]
    rows = existing.to_dict("records") if len(existing) else []
    if not tests:
        print(f"All tuned scaling baselines already present in {OUTPUT}")
        return
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tests))) as executor:
        futures = {executor.submit(evaluate_test, test): test for test in tests}
        for future in as_completed(futures):
            rows.extend(future.result())
            pd.DataFrame(rows).sort_values(["test_name", "n_train", "method"]).to_csv(OUTPUT, index=False)
    print(f"Saved {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
