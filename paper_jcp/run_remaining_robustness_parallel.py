#!/usr/bin/env python3
"""Complete missing robustness conditions in parallel with serialized CSV writes."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.run_robustness_experiments import (
    CHECKPOINTS,
    CONFIGS,
    GENERATED,
    add_label_noise,
    build_bundle,
    conditions_for,
    load_training_module,
    metrics,
    predict_classical,
    predict_pinn,
    select_points,
    set_seed,
    train_pinn,
    tune_classical,
)


METHODS = ["HO-PINN", "XGB", "SVM", "GP"]


def condition_key(frame: pd.DataFrame, condition: tuple[str, int, float, int]) -> pd.Series:
    kind, seed, noise, ratio = condition
    return (
        (frame.experiment == kind)
        & (frame.seed == seed)
        & (frame.noise_level.round(8) == round(noise, 8))
        & (frame.collocation_ratio == ratio)
    )


def run_condition(test_name: str, condition: tuple[str, int, float, int], epochs: int):
    kind, seed, noise, ratio = condition
    set_seed(42)
    module = load_training_module(test_name)
    bundle = build_bundle(test_name, module)
    selected = select_points(test_name, module, bundle, seed, ratio)
    noisy_y = add_label_noise(test_name, selected["y_data"], noise, seed)
    tag = f"{test_name}_{kind}_s{seed}_noise{noise:g}_r{ratio}"
    checkpoint = CHECKPOINTS / f"{tag}.pth"
    model, train_time = train_pinn(
        test_name, module, selected, noisy_y, seed, epochs, checkpoint
    )
    prediction = predict_pinn(test_name, model, bundle["test_x"])
    rows = [{
        "test_name": test_name, "experiment": kind, "seed": seed,
        "noise_level": noise, "collocation_ratio": ratio,
        "method": "HO-PINN", "n_train": CONFIGS[test_name]["n"],
        "n_collocation": CONFIGS[test_name]["n"] * ratio,
        "n_test": len(bundle["test_y"]), "train_time_sec": train_time,
        **metrics(test_name, bundle["test_y"], prediction),
    }]
    for method in METHODS[1:]:
        estimators, fit_time = tune_classical(
            method, selected["x_data"], noisy_y, seed
        )
        prediction = predict_classical(
            estimators, bundle["test_x"], scalar=test_name == "test1"
        )
        rows.append({
            "test_name": test_name, "experiment": kind, "seed": seed,
            "noise_level": noise, "collocation_ratio": ratio,
            "method": method, "n_train": CONFIGS[test_name]["n"],
            "n_collocation": 0, "n_test": len(bundle["test_y"]),
            "train_time_sec": fit_time,
            **metrics(test_name, bundle["test_y"], prediction),
        })
    return test_name, condition, rows, train_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tests", nargs="+", choices=list(CONFIGS), default=["test2", "test4"])
    args = parser.parse_args()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for test_name in args.tests:
        path = GENERATED / f"robustness_{test_name}.csv"
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        for condition in conditions_for(test_name):
            if len(frame) and int(condition_key(frame, condition).sum()) == 4:
                continue
            jobs.append((test_name, condition))
    if not jobs:
        print("No missing robustness conditions")
        return
    print(f"Running {len(jobs)} missing conditions with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {
            executor.submit(run_condition, test_name, condition, args.epochs): (test_name, condition)
            for test_name, condition in jobs
        }
        for future in as_completed(futures):
            test_name, condition, rows, train_time = future.result()
            path = GENERATED / f"robustness_{test_name}.csv"
            frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
            if len(frame):
                frame = frame.loc[~condition_key(frame, condition)].copy()
            frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
            frame.sort_values(
                ["experiment", "noise_level", "collocation_ratio", "seed", "method"]
            ).to_csv(path, index=False)
            print(
                f"[{test_name}] saved {condition} (PINN {train_time:.1f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
