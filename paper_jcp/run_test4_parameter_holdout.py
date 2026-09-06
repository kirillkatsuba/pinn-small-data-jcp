#!/usr/bin/env python3
"""Evaluate Test 4 on physical-parameter combinations absent from training."""

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
    build_test4_bundle,
    load_training_module,
    metrics,
    predict_classical,
    predict_pinn,
    select_points,
    train_pinn,
    tune_classical,
)


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "parameter_holdout_test4.csv"
METHODS = ["HO-PINN", "XGB", "SVM", "GP"]


def build_group_holdout(module) -> tuple[dict, np.ndarray]:
    bundle = build_test4_bundle(module)
    parameter_rows = bundle["features"][:, 3:]
    groups, inverse = np.unique(parameter_rows, axis=0, return_inverse=True)
    if len(groups) != 100:
        raise ValueError(f"Expected 100 parameter groups, found {len(groups)}")
    rng = np.random.default_rng(20260728)
    heldout_groups = np.sort(rng.choice(len(groups), size=20, replace=False))
    test_parts = []
    for group in heldout_groups:
        candidates = np.flatnonzero(inverse == group)
        test_parts.append(rng.choice(candidates, size=100, replace=False))
    test_idx = np.concatenate(test_parts)
    train_candidates = np.flatnonzero(~np.isin(inverse, heldout_groups))
    bundle["test_x"] = bundle["features"][test_idx]
    bundle["test_y"] = bundle["targets"][test_idx]
    bundle["train_candidates"] = train_candidates
    return bundle, groups[heldout_groups]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=STRESS_SEEDS)
    args = parser.parse_args()
    module = load_training_module("test4")
    bundle, heldout_groups = build_group_holdout(module)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []
    for seed in args.seeds:
        if len(existing) and int((existing.seed == seed).sum()) == 4:
            print(f"[test4 parameter holdout] skip seed={seed}", flush=True)
            continue
        selected = select_points("test4", module, bundle, seed, ratio=1)
        checkpoint = CHECKPOINTS / f"test4_parameter_holdout_s{seed}.pth"
        model, train_time = train_pinn(
            "test4", module, selected, selected["y_data"], seed,
            args.epochs, checkpoint,
        )
        start = time.perf_counter()
        prediction = predict_pinn("test4", model, bundle["test_x"])
        condition_rows = [{
            "test_name": "test4", "experiment": "parameter_holdout",
            "seed": seed, "noise_level": 0.0, "collocation_ratio": 1,
            "method": "HO-PINN", "n_train": CONFIGS["test4"]["n"],
            "n_collocation": CONFIGS["test4"]["n"],
            "n_test": len(bundle["test_y"]), "train_time_sec": train_time,
            "inference_time_sec": time.perf_counter() - start,
            **metrics("test4", bundle["test_y"], prediction),
        }]
        for method in METHODS[1:]:
            estimators, fit_time = tune_classical(
                method, selected["x_data"], selected["y_data"], seed
            )
            start = time.perf_counter()
            prediction = predict_classical(estimators, bundle["test_x"], scalar=False)
            condition_rows.append({
                "test_name": "test4", "experiment": "parameter_holdout",
                "seed": seed, "noise_level": 0.0, "collocation_ratio": 1,
                "method": method, "n_train": CONFIGS["test4"]["n"],
                "n_collocation": 0, "n_test": len(bundle["test_y"]),
                "train_time_sec": fit_time,
                "inference_time_sec": time.perf_counter() - start,
                **metrics("test4", bundle["test_y"], prediction),
            })
        if len(existing):
            existing = existing[existing.seed != seed]
            rows = existing.to_dict("records")
        rows.extend(condition_rows)
        pd.DataFrame(rows).to_csv(OUTPUT, index=False)
        existing = pd.DataFrame(rows)
        print(
            f"[test4 parameter holdout] seed={seed} complete "
            f"(PINN {train_time:.1f}s)", flush=True,
        )
    np.savetxt(
        PAPER / "generated" / "parameter_holdout_test4_groups.csv",
        heldout_groups,
        delimiter=",",
        header="pwat,poil,kwat,koil",
        comments="",
    )


if __name__ == "__main__":
    main()
