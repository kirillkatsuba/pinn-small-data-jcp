#!/usr/bin/env python3
"""Run the unified interpolation/extrapolation small-data protocol.

No work is performed on import.  The command-line entry point is resumable and
writes separate result files per benchmark so that tests may run in parallel.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.unified_protocol import (
    LABEL_BUDGETS,
    TEST_NAMES,
    ModelConfig,
    TrainConfig,
    build_unified_bundle,
    configuration_id,
    evaluation_metrics,
    nested_heat_labels,
    nested_label_indices,
    predict_pinn,
    resolve_device,
    train_pinn,
)
from paper_jcp.classical_baselines import (
    predict_classical,
    tune_classical,
)


PAPER = Path(__file__).resolve().parent
GENERATED_ROOT = PAPER / "generated"
EXPERIMENTS_ROOT = PAPER / "experiments"


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def result_exists(
    table: pd.DataFrame,
    *,
    test_name: str,
    seed: int,
    budget: int,
    method: str,
    config_id: str,
) -> bool:
    if table.empty:
        return False
    mask = (
        (table["test_name"] == test_name)
        & (table["seed"] == seed)
        & (table["n_train"] == budget)
        & (table["method"] == method)
        & (table["config_id"] == config_id)
    )
    return set(table.loc[mask, "evaluation_region"]) == {"interpolation", "extrapolation"}


def evaluation_rows(
    *,
    test_name: str,
    seed: int,
    budget: int,
    method: str,
    config_id: str,
    train_time: float,
    n_collocation: int,
    interpolation_truth: np.ndarray,
    interpolation_prediction: np.ndarray,
    extrapolation_truth: np.ndarray,
    extrapolation_prediction: np.ndarray,
    time_fraction: float,
    metadata: dict,
) -> list[dict]:
    rows = []
    for region, truth, prediction in (
        ("interpolation", interpolation_truth, interpolation_prediction),
        ("extrapolation", extrapolation_truth, extrapolation_prediction),
    ):
        rows.append(
            {
                "test_name": test_name,
                "seed": seed,
                "n_train": budget,
                "n_collocation_per_epoch": n_collocation,
                "method": method,
                "config_id": config_id,
                "evaluation_region": region,
                "label_time_fraction": time_fraction,
                "n_test": len(truth),
                "train_time_sec": train_time,
                **metadata,
                **evaluation_metrics(test_name, truth, prediction),
            }
        )
    return rows


def run_test(
    test_name: str,
    experiment_name: str,
    data_source: str,
    corrected_data_root: str | None,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    balancers: tuple[str, ...],
    model_config: ModelConfig,
    train_config: TrainConfig,
    time_fraction: float,
    time_bins: int,
    interpolation_size: int,
    extrapolation_size: int,
    corrected_test4_pool_size: int,
    run_pinn: bool,
    run_classical: bool,
    classical_methods: tuple[str, ...],
    classical_tuning_profile: str,
    classical_cv: str,
) -> str:
    generated = GENERATED_ROOT / experiment_name
    checkpoints = EXPERIMENTS_ROOT / experiment_name / "checkpoints"
    histories = EXPERIMENTS_ROOT / experiment_name / "histories"
    output_path = generated / f"unified_{test_name}.csv"
    generated.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    bundle = build_unified_bundle(
        test_name,
        time_fraction=time_fraction,
        interpolation_size=interpolation_size,
        extrapolation_size=extrapolation_size,
        data_source=data_source,
        corrected_data_root=corrected_data_root,
        corrected_test4_pool_size=corrected_test4_pool_size,
    )
    dataset_metadata = {
        "data_source": bundle.data_source,
        "dataset_path": bundle.dataset_path or "",
        "full_time": bundle.full_time,
        "time_cutoff": bundle.time_cutoff,
        "porosity": bundle.porosity_value,
        "water_viscosity": bundle.water_viscosity,
        "oil_viscosity": bundle.oil_viscosity,
        "corrected_test4_pool_size": (
            corrected_test4_pool_size if test_name == "test4" else np.nan
        ),
    }
    for seed in seeds:
        if test_name == "test1":
            all_x, all_y = nested_heat_labels(
                bundle, seed, max_budget=max(budgets), time_bins=time_bins
            )
            all_indices = None
        else:
            indices = nested_label_indices(
                bundle, seed, max_budget=max(budgets), time_bins=time_bins
            )
            all_indices = indices
            assert bundle.features is not None and bundle.targets is not None
            all_x = bundle.features[indices]
            all_y = bundle.targets[indices]

        for budget in budgets:
            x_data = all_x[:budget]
            y_data = all_y[:budget]
            label_indices = None if all_indices is None else all_indices[:budget]
            n_collocation = max(
                1, int(np.ceil(budget * train_config.collocation_ratio))
            )

            if run_pinn:
                for balance_mode in balancers:
                    method = "H-PINN" if balance_mode == "h" else "HO-PINN"
                    config_id = configuration_id(
                        balance_mode, model_config, train_config
                    )
                    if bundle.data_source == "corrected":
                        corrected_prefix = (
                            f"corrected_pool{corrected_test4_pool_size}"
                            if test_name == "test4"
                            else "corrected"
                        )
                        config_id = f"{corrected_prefix}_{config_id}"
                    if result_exists(
                        existing,
                        test_name=test_name,
                        seed=seed,
                        budget=budget,
                        method=method,
                        config_id=config_id,
                    ):
                        print(
                            f"[{test_name}] skip {method} N={budget} seed={seed}",
                            flush=True,
                        )
                        continue
                    run_tag = f"{test_name}_n{budget}_s{seed}_{config_id}"
                    model, fit_time = train_pinn(
                        bundle=bundle,
                        x_data_np=x_data,
                        y_data_np=y_data,
                        label_indices=label_indices,
                        label_budget=budget,
                        seed=seed,
                        balance_mode=balance_mode,
                        model_config=model_config,
                        train_config=train_config,
                        checkpoint_path=checkpoints / f"{run_tag}.pth",
                        history_path=histories / f"{run_tag}.csv",
                    )
                    interp_prediction = predict_pinn(
                        test_name, model, bundle.interpolation_x
                    )
                    extra_prediction = predict_pinn(
                        test_name, model, bundle.extrapolation_x
                    )
                    rows.extend(
                        evaluation_rows(
                            test_name=test_name,
                            seed=seed,
                            budget=budget,
                            method=method,
                            config_id=config_id,
                            train_time=fit_time,
                            n_collocation=n_collocation,
                            interpolation_truth=bundle.interpolation_y,
                            interpolation_prediction=interp_prediction,
                            extrapolation_truth=bundle.extrapolation_y,
                            extrapolation_prediction=extra_prediction,
                            time_fraction=time_fraction,
                            metadata={
                                **dataset_metadata,
                                "model_hidden_dim": model_config.hidden_dim,
                                "model_depth": model_config.depth,
                                "model_activation": model_config.activation,
                                "balancer": balance_mode,
                                "residual_domain": train_config.residual_domain,
                                "optimizer": train_config.optimizer,
                                "scheduler": train_config.scheduler,
                                "learning_rate": train_config.learning_rate,
                                "l0_data": train_config.l0_data,
                                "l0_ibc": train_config.l0_ibc,
                                "device": train_config.device,
                                "deterministic": train_config.deterministic,
                            },
                        )
                    )
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                    existing = pd.DataFrame(rows)
                    print(
                        f"[{test_name}] done {method} N={budget} seed={seed} "
                        f"time={fit_time:.1f}s",
                        flush=True,
                    )

            if run_classical:
                for method in classical_methods:
                    if (
                        classical_tuning_profile == "standard"
                        and classical_cv == "random"
                    ):
                        config_id = "training-only-tuned"
                    else:
                        config_id = (
                            f"training-only-tuned-{classical_tuning_profile}-"
                            f"{classical_cv}cv"
                        )
                    if bundle.data_source == "corrected":
                        corrected_prefix = (
                            f"corrected_pool{corrected_test4_pool_size}"
                            if test_name == "test4"
                            else "corrected"
                        )
                        config_id = f"{corrected_prefix}_{config_id}"
                    if result_exists(
                        existing,
                        test_name=test_name,
                        seed=seed,
                        budget=budget,
                        method=method,
                        config_id=config_id,
                    ):
                        print(
                            f"[{test_name}] skip {method} N={budget} seed={seed}",
                            flush=True,
                        )
                        continue
                    estimators, fit_time, tuning_details = tune_classical(
                        method,
                        x_data,
                        y_data,
                        seed,
                        tuning_profile=classical_tuning_profile,
                        cv_strategy=classical_cv,
                        return_metadata=True,
                    )
                    scalar = test_name == "test1"
                    interp_prediction = predict_classical(
                        estimators, bundle.interpolation_x, scalar
                    )
                    extra_prediction = predict_classical(
                        estimators, bundle.extrapolation_x, scalar
                    )
                    rows.extend(
                        evaluation_rows(
                            test_name=test_name,
                            seed=seed,
                            budget=budget,
                            method=method,
                            config_id=config_id,
                            train_time=fit_time,
                            n_collocation=0,
                            interpolation_truth=bundle.interpolation_y,
                            interpolation_prediction=interp_prediction,
                            extrapolation_truth=bundle.extrapolation_y,
                            extrapolation_prediction=extra_prediction,
                            time_fraction=time_fraction,
                            metadata={
                                **dataset_metadata,
                                "classical_tuning_profile": classical_tuning_profile,
                                "classical_cv": classical_cv,
                                "classical_tuning_details_json": json.dumps(
                                    tuning_details, sort_keys=True
                                ),
                                "model_hidden_dim": np.nan,
                                "model_depth": np.nan,
                                "model_activation": "",
                                "balancer": "",
                                "residual_domain": "",
                                "optimizer": "",
                                "scheduler": "",
                                "learning_rate": np.nan,
                                "l0_data": np.nan,
                                "l0_ibc": np.nan,
                                "device": "cpu",
                                "deterministic": True,
                            },
                        )
                    )
                    pd.DataFrame(rows).to_csv(output_path, index=False)
                    existing = pd.DataFrame(rows)
                    print(
                        f"[{test_name}] done {method} N={budget} seed={seed} "
                        f"time={fit_time:.1f}s",
                        flush=True,
                    )
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified JCP small-data interpolation/extrapolation protocol"
    )
    parser.add_argument("--test", choices=("all", *TEST_NAMES), default="all")
    parser.add_argument(
        "--experiment-name",
        default="unified_protocol",
        help="Separate output/checkpoint namespace, e.g. architecture_tuning or final",
    )
    parser.add_argument(
        "--budgets", default=",".join(str(value) for value in LABEL_BUDGETS)
    )
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--balancers", default="h,ho")
    parser.add_argument("--time-fraction", type=float, default=0.7)
    parser.add_argument("--time-bins", type=int, default=10)
    parser.add_argument("--interpolation-size", type=int, default=1500)
    parser.add_argument("--extrapolation-size", type=int, default=1500)
    parser.add_argument(
        "--data-source",
        choices=("legacy", "corrected"),
        default="legacy",
        help="Use legacy sampled labels or corrected finite-volume labels.",
    )
    parser.add_argument(
        "--corrected-data-root",
        type=Path,
        default=GENERATED_ROOT / "publication_data",
        help="Root produced by generate_corrected_benchmarks.py.",
    )
    parser.add_argument(
        "--corrected-test4-pool-size",
        type=int,
        default=100_000,
        help="Materialised Test 4 residual/label pool, split equally by time region.",
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument(
        "--activation", choices=("tanh", "silu", "gelu"), default="tanh"
    )
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--scheduler", choices=("cosine", "none"), default="cosine")
    parser.add_argument("--l0-data", type=float, default=1e-2)
    parser.add_argument("--l0-ibc", type=float, default=1e-3)
    parser.add_argument("--orthogonal-floor", type=float, default=0.5)
    parser.add_argument("--collocation-ratio", type=float, default=1.0)
    parser.add_argument(
        "--residual-domain", choices=("full", "label"), default="full"
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-criterion", choices=("balanced", "raw"), default="balanced"
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--physics-validation-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PINN device: cpu, cuda, cuda:0, mps, or auto.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch kernels when available.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--classical-tuning-profile",
        choices=("standard", "strong"),
        default="standard",
        help="Compact legacy search or expanded publication-strength search.",
    )
    parser.add_argument(
        "--classical-cv",
        choices=("random", "time"),
        default="random",
        help="Random 3-fold CV or three expanding-window temporal folds.",
    )
    parser.add_argument(
        "--classical-methods",
        default="XGB,SVM,GP",
        help="Comma-separated subset of XGB,SVM,GP; useful for staged runs.",
    )
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--skip-classical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = comma_ints(args.budgets)
    seeds = comma_ints(args.seeds)
    balancers = comma_strings(args.balancers)
    classical_methods = tuple(value.upper() for value in comma_strings(args.classical_methods))
    if any(value not in LABEL_BUDGETS for value in budgets):
        raise ValueError(f"Budgets must be selected from {LABEL_BUDGETS}")
    if any(value not in {"h", "ho"} for value in balancers):
        raise ValueError("Balancers must be h and/or ho")
    if not classical_methods or any(
        value not in {"XGB", "SVM", "GP"} for value in classical_methods
    ):
        raise ValueError("Classical methods must be selected from XGB,SVM,GP")
    if args.skip_pinn and args.skip_classical:
        raise ValueError("Both PINN and classical runs were disabled")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.data_source == "corrected" and args.test not in {"test2", "test3", "test4"}:
        raise ValueError(
            "Corrected unified loading is available for Tests 2--4"
        )

    tests = TEST_NAMES if args.test == "all" else (args.test,)
    resolved_device = resolve_device(args.device) if not args.skip_pinn else None
    if (
        resolved_device is not None
        and resolved_device.type != "cpu"
        and len(tests) > 1
        and args.workers != 1
    ):
        raise ValueError(
            "A single GPU must use --workers 1 when --test all. "
            "Launch one process per GPU explicitly for multi-GPU execution."
        )

    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        l0_data=args.l0_data,
        l0_ibc=args.l0_ibc,
        orthogonal_floor=args.orthogonal_floor,
        collocation_ratio=args.collocation_ratio,
        residual_domain=args.residual_domain,
        gradient_clip=args.gradient_clip,
        checkpoint_criterion=args.checkpoint_criterion,
        log_every=args.log_every,
        physics_validation_size=args.physics_validation_size,
        device="cpu" if resolved_device is None else str(resolved_device),
        deterministic=args.deterministic,
    )
    print("Model configuration:", asdict(model_config), flush=True)
    print("Training configuration:", asdict(train_config), flush=True)
    print(
        "Data configuration:",
        {
            "data_source": args.data_source,
            "corrected_data_root": str(args.corrected_data_root),
            "classical_tuning_profile": args.classical_tuning_profile,
            "classical_cv": args.classical_cv,
            "classical_methods": classical_methods,
            "corrected_test4_pool_size": args.corrected_test4_pool_size,
            "resolved_device": None if resolved_device is None else str(resolved_device),
            "deterministic": args.deterministic,
        },
        flush=True,
    )

    arguments = dict(
        experiment_name=args.experiment_name,
        data_source=args.data_source,
        corrected_data_root=(
            str(args.corrected_data_root) if args.data_source == "corrected" else None
        ),
        budgets=budgets,
        seeds=seeds,
        balancers=balancers,
        model_config=model_config,
        train_config=train_config,
        time_fraction=args.time_fraction,
        time_bins=args.time_bins,
        interpolation_size=args.interpolation_size,
        extrapolation_size=args.extrapolation_size,
        corrected_test4_pool_size=args.corrected_test4_pool_size,
        run_pinn=not args.skip_pinn,
        run_classical=not args.skip_classical,
        classical_methods=classical_methods,
        classical_tuning_profile=args.classical_tuning_profile,
        classical_cv=args.classical_cv,
    )
    if len(tests) == 1:
        print(run_test(tests[0], **arguments))
        return

    with ProcessPoolExecutor(max_workers=min(args.workers, len(tests))) as executor:
        futures = {
            executor.submit(run_test, test_name, **arguments): test_name
            for test_name in tests
        }
        for future in as_completed(futures):
            test_name = futures[future]
            print(f"Completed {test_name}: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
