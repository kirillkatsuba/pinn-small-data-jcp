#!/usr/bin/env python3
"""Measure representative smallest-budget fit times without CPU contention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.run_robustness_experiments import (
    CHECKPOINTS,
    CONFIGS,
    build_bundle,
    load_training_module,
    select_points,
    train_pinn,
    tune_classical,
)


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "isolated_timing.csv"
TESTS = ["test1", "test2", "test3", "test4"]
METHODS = ["XGB", "SVM", "GP"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=TESTS)
    args = parser.parse_args()
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []
    for test_name in args.tests:
        if len(existing) and int((existing.test_name == test_name).sum()) == 4:
            print(f"[{test_name}] skip isolated timing", flush=True)
            continue
        module = load_training_module(test_name)
        bundle = build_bundle(test_name, module)
        selected = select_points(test_name, module, bundle, seed=11, ratio=1)
        checkpoint = CHECKPOINTS / f"isolated_timing_{test_name}.pth"
        _, train_time = train_pinn(
            test_name, module, selected, selected["y_data"], seed=11,
            epochs=args.epochs, checkpoint_path=checkpoint,
        )
        condition_rows = [{
            "test_name": test_name, "method": "HO-PINN",
            "n_train": CONFIGS[test_name]["n"],
            "train_time_sec": train_time,
        }]
        for method in METHODS:
            _, fit_time = tune_classical(
                method, selected["x_data"], selected["y_data"], seed=11
            )
            condition_rows.append({
                "test_name": test_name, "method": method,
                "n_train": CONFIGS[test_name]["n"],
                "train_time_sec": fit_time,
            })
        if len(existing):
            existing = existing[existing.test_name != test_name]
            rows = existing.to_dict("records")
        rows.extend(condition_rows)
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        existing = pd.DataFrame(rows)
        print(f"[{test_name}] isolated timing complete ({train_time:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
