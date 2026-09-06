#!/usr/bin/env python3
"""Additional zero/interior-label PINN experiment for JCP Tests 1 and 2.

This runner intentionally leaves the unified publication code unchanged.  It
reuses the frozen model, PDE/IBC definitions, corrected Test 2 data, and label
selections, but decouples the residual batch size from the number of labelled
solution states.  In the N=0 case, the loss hierarchy becomes IBC -> PDE and no
reference-solution value is used for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import trange


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loss_balancong_algorithms.algos import (  # noqa: E402
    flat_gradient_from_loss,
    orthogonal_fraction,
)
from paper_jcp.unified_protocol import (  # noqa: E402
    LABEL_BUDGETS,
    ModelConfig,
    ResidualSampler,
    TrainConfig,
    build_model,
    build_unified_bundle,
    configure_torch_runtime,
    evaluation_metrics,
    loss_components,
    make_balancer,
    nested_heat_labels,
    nested_label_indices,
    predict_pinn,
    resolve_device,
)


PAPER = Path(__file__).resolve().parent
GENERATED_ROOT = PAPER / "generated"
EXPERIMENTS_ROOT = PAPER / "experiments"
DEFAULT_DATA_ROOT = GENERATED_ROOT / "publication_data"


def comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def config_id_for_job(job: dict) -> str:
    model = job["model_config"]
    train = job["train_config"]
    suffix = ""
    if train["device"] != "cpu":
        suffix += f"_{train['device'].replace(':', '-')}"
    if not train["deterministic"]:
        suffix += "_nondeterministic"
    return (
        f"zero-label-{job['mode']}_w{model['hidden_dim']}_d{model['depth']}_"
        f"{model['activation']}_lr{train['learning_rate']:g}_"
        f"l0d{train['l0_data']:g}_l0b{train['l0_ibc']:g}_"
        f"floor{train['orthogonal_floor']:g}_"
        f"fixednr{job['collocation_points']}_e{train['epochs']}{suffix}"
    )


def zero_safe_losses(
    *,
    bundle,
    model: torch.nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    x_collocation: torch.Tensor,
    permeability: torch.Tensor | None,
    label_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the unchanged unified losses, with a differentiable zero at N=0."""

    if label_budget > 0:
        return loss_components(
            bundle, model, x_data, y_data, x_collocation, permeability
        )

    # The existing loss functions also construct IBC and PDE terms.  A detached
    # current prediction provides a harmless one-row placeholder for their data
    # argument; its mismatch and gradient are exactly zero and are never weighted.
    dummy_x = x_collocation[:1]
    dummy_prediction = model(dummy_x).detach()
    dummy_y = (
        dummy_prediction.reshape(-1)
        if bundle.test_name == "test1"
        else dummy_prediction[:, :3]
    )
    loss_data, loss_ibc, loss_pde = loss_components(
        bundle,
        model,
        dummy_x,
        dummy_y,
        x_collocation,
        permeability,
    )
    return loss_data * 0.0, loss_ibc, loss_pde


def physics_only_weights(
    *,
    model: torch.nn.Module,
    loss_ibc: torch.Tensor,
    loss_pde: torch.Tensor,
    mode: str,
    l0_ibc: float,
    orthogonal_floor: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Two-level IBC -> PDE form of H-PINN/HO-PINN for N=0."""

    a_ibc = torch.ones((), device=loss_ibc.device, dtype=loss_ibc.dtype)
    ratio = torch.clamp(loss_ibc.detach() / (l0_ibc + eps), 0.0, 60.0)
    a_pde = torch.exp(-ratio)
    diagnostics: dict[str, float] = {}

    if mode == "ho":
        g_ibc = flat_gradient_from_loss(model, loss_ibc, eps=eps)
        g_pde = flat_gradient_from_loss(model, loss_pde, eps=eps)
        fraction = orthogonal_fraction(g_pde, basis=[g_ibc], eps=eps)
        factor = orthogonal_floor + (1.0 - orthogonal_floor) * fraction
        a_pde = a_pde * factor.to(device=a_pde.device, dtype=a_pde.dtype)
        diagnostics.update(
            {
                "orth_frac_pde": float(fraction.detach()),
                "orth_factor_pde": float(factor.detach()),
            }
        )

    denominator = a_ibc + a_pde + eps
    w_data = torch.zeros((), device=loss_ibc.device, dtype=loss_ibc.dtype)
    w_ibc = (a_ibc / denominator).detach()
    w_pde = (a_pde / denominator).detach()
    diagnostics.update(
        {
            "a_mse": 0.0,
            "a_ibc": float(a_ibc.detach()),
            "a_pde": float(a_pde.detach()),
            "w_mse": 0.0,
            "w_ibc": float(w_ibc),
            "w_pde": float(w_pde),
        }
    )
    return w_data, w_ibc, w_pde, diagnostics


def weights_for_losses(
    *,
    model: torch.nn.Module,
    label_budget: int,
    mode: str,
    train_config: TrainConfig,
    standard_balancer,
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    loss_data, loss_ibc, loss_pde = losses
    if label_budget > 0:
        return standard_balancer.compute_weights(
            model, loss_data, loss_ibc, loss_pde
        )
    return physics_only_weights(
        model=model,
        loss_ibc=loss_ibc,
        loss_pde=loss_pde,
        mode=mode,
        l0_ibc=train_config.l0_ibc,
        orthogonal_floor=train_config.orthogonal_floor,
    )


def make_labels(bundle, seed: int, budget: int):
    """Use the exact nested selection from the six-budget publication design."""

    max_budget = max(LABEL_BUDGETS)
    if bundle.test_name == "test1":
        all_x, all_y = nested_heat_labels(
            bundle, seed, max_budget=max_budget, time_bins=10
        )
        label_indices = None
    else:
        all_indices = nested_label_indices(
            bundle, seed, max_budget=max_budget, time_bins=10
        )
        assert bundle.features is not None and bundle.targets is not None
        all_x = bundle.features[all_indices]
        all_y = bundle.targets[all_indices]
        label_indices = all_indices[:budget]

    if budget > 0:
        return all_x[:budget], all_y[:budget], label_indices

    input_dim = 2 if bundle.test_name == "test1" else 3
    target_shape = (0,) if bundle.test_name == "test1" else (0, 3)
    return (
        np.empty((0, input_dim), dtype=np.float32),
        np.empty(target_shape, dtype=np.float32),
        None,
    )


def run_job(job: dict) -> list[dict]:
    test_name = job["test_name"]
    seed = job["seed"]
    budget = job["budget"]
    mode = job["mode"]
    model_config = ModelConfig(**job["model_config"])
    train_config = TrainConfig(**job["train_config"])
    collocation_points = job["collocation_points"]
    experiment_name = job["experiment_name"]
    corrected_data_root = Path(job["corrected_data_root"])

    bundle = build_unified_bundle(
        test_name,
        time_fraction=0.7,
        interpolation_size=1500,
        extrapolation_size=1500,
        data_source="legacy" if test_name == "test1" else "corrected",
        corrected_data_root=None if test_name == "test1" else corrected_data_root,
    )
    x_data_np, y_data_np, label_indices = make_labels(bundle, seed, budget)

    device = resolve_device(train_config.device)
    configure_torch_runtime(seed + 1000, device, train_config.deterministic)
    model = build_model(bundle, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.epochs,
        eta_min=train_config.min_learning_rate,
    )
    standard_balancer = make_balancer(mode, train_config)
    sampler = ResidualSampler(
        bundle,
        seed + 500,
        domain="full",
        exclude_indices=label_indices,
    )
    validation_sampler = ResidualSampler(
        bundle,
        seed + 50_000,
        domain="full",
        exclude_indices=label_indices,
    )
    validation_np, validation_perm_np = validation_sampler.sample(
        train_config.physics_validation_size
    )

    x_data = torch.as_tensor(x_data_np, dtype=torch.float32, device=device)
    y_data = torch.as_tensor(y_data_np, dtype=torch.float32, device=device)
    validation_x = torch.as_tensor(
        validation_np, dtype=torch.float32, device=device
    )
    validation_perm = (
        None
        if validation_perm_np is None
        else torch.as_tensor(validation_perm_np, dtype=torch.float32, device=device)
    )

    tag = (
        f"{test_name}_n{budget}_s{seed}_{mode}_fixednr{collocation_points}_"
        f"e{train_config.epochs}"
    )
    checkpoint_path = (
        EXPERIMENTS_ROOT / experiment_name / "checkpoints" / f"{tag}.pth"
    )
    history_path = EXPERIMENTS_ROOT / experiment_name / "histories" / f"{tag}.csv"
    best_value = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    started = time.perf_counter()

    iterator = trange(train_config.epochs, desc=tag, leave=False)
    for epoch in iterator:
        residual_np, permeability_np = sampler.sample(collocation_points)
        x_collocation = torch.as_tensor(
            residual_np, dtype=torch.float32, device=device
        )
        permeability = (
            None
            if permeability_np is None
            else torch.as_tensor(permeability_np, dtype=torch.float32, device=device)
        )
        optimizer.zero_grad(set_to_none=True)
        losses = zero_safe_losses(
            bundle=bundle,
            model=model,
            x_data=x_data,
            y_data=y_data,
            x_collocation=x_collocation,
            permeability=permeability,
            label_budget=budget,
        )
        weights = weights_for_losses(
            model=model,
            label_budget=budget,
            mode=mode,
            train_config=train_config,
            standard_balancer=standard_balancer,
            losses=losses,
        )
        w_data, w_ibc, w_pde, diagnostics = weights
        loss_data, loss_ibc, loss_pde = losses
        balanced = w_data * loss_data + w_ibc * loss_ibc + w_pde * loss_pde
        raw = (
            (loss_data if budget > 0 else 0.0) + loss_ibc + loss_pde
        )
        balanced.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        optimizer.step()
        scheduler.step()

        should_validate = (
            epoch == 0
            or (epoch + 1) % train_config.log_every == 0
            or epoch + 1 == train_config.epochs
        )
        if not should_validate:
            continue

        val_losses = zero_safe_losses(
            bundle=bundle,
            model=model,
            x_data=x_data,
            y_data=y_data,
            x_collocation=validation_x,
            permeability=validation_perm,
            label_budget=budget,
        )
        val_data, val_ibc, val_pde = val_losses
        # A stationary raw criterion avoids selecting an early checkpoint merely
        # because the hierarchical PDE gate had not activated yet.
        validation_score = float(
            ((val_data if budget > 0 else 0.0) + val_ibc + val_pde).detach()
        )
        if validation_score < best_value:
            best_value = validation_score
            best_state = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch + 1,
                "loss_data": float(loss_data.detach()),
                "loss_ibc": float(loss_ibc.detach()),
                "loss_pde": float(loss_pde.detach()),
                "raw_total": float(raw.detach()),
                "balanced_total": float(balanced.detach()),
                "weight_data": float(w_data),
                "weight_ibc": float(w_ibc),
                "weight_pde": float(w_pde),
                "physics_val_data": float(val_data.detach()),
                "physics_val_ibc": float(val_ibc.detach()),
                "physics_val_pde": float(val_pde.detach()),
                "physics_val_raw": validation_score,
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **diagnostics,
            }
        )
        iterator.set_postfix(val=f"{validation_score:.2e}")

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {tag}")
    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    portable_state = {
        name: value.detach().cpu().clone() for name, value in best_state.items()
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": portable_state,
            "test_name": test_name,
            "data_source": bundle.data_source,
            "dataset_path": bundle.dataset_path,
            "input_lower": bundle.input_lower,
            "input_upper": bundle.input_upper,
            "balance_mode": mode,
            "label_budget": budget,
            "collocation_points": collocation_points,
            "seed": seed,
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "checkpoint_criterion": "fixed_physics_validation_raw",
            "resolved_device": str(device),
            "best_value": best_value,
        },
        checkpoint_path,
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    history_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "elapsed_seconds": elapsed,
                "best_physics_validation_raw": best_value,
                "label_budget": budget,
                "collocation_points": collocation_points,
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "resolved_device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    method = "H-PINN" if mode == "h" else "HO-PINN"
    config_id = config_id_for_job(job)
    rows: list[dict] = []
    for region, features, truth in (
        ("interpolation", bundle.interpolation_x, bundle.interpolation_y),
        ("extrapolation", bundle.extrapolation_x, bundle.extrapolation_y),
    ):
        prediction = predict_pinn(test_name, model, features)
        rows.append(
            {
                "test_name": test_name,
                "seed": seed,
                "n_train": budget,
                "n_collocation_per_epoch": collocation_points,
                "method": method,
                "config_id": config_id,
                "evaluation_region": region,
                "label_time_fraction": 0.7,
                "n_test": len(truth),
                "train_time_sec": elapsed,
                "epochs": train_config.epochs,
                "checkpoint_criterion": "fixed_physics_validation_raw",
                "best_physics_validation_raw": best_value,
                "data_source": bundle.data_source,
                "dataset_path": bundle.dataset_path or "",
                "full_time": bundle.full_time,
                "model_hidden_dim": model_config.hidden_dim,
                "model_depth": model_config.depth,
                "model_activation": model_config.activation,
                "balancer": mode,
                "learning_rate": train_config.learning_rate,
                "l0_data": train_config.l0_data,
                "l0_ibc": train_config.l0_ibc,
                "orthogonal_floor": train_config.orthogonal_floor,
                "device": train_config.device,
                "deterministic": train_config.deterministic,
                **evaluation_metrics(test_name, truth, prediction),
            }
        )
    print(
        f"[{test_name}] done {method} N={budget} seed={seed} "
        f"Nr={collocation_points} time={elapsed:.1f}s",
        flush=True,
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run additional zero-label PINN experiments for Tests 1 and 2."
    )
    parser.add_argument("--experiment-name", default="zero_label_test12")
    parser.add_argument("--test", choices=("test1", "test2", "all"), default="all")
    parser.add_argument("--budgets", default="0,20")
    parser.add_argument("--seeds", default="101,102,103,104,105,106,107,108,109,110")
    parser.add_argument("--balancers", default="h")
    parser.add_argument("--epochs", type=int, default=50_000)
    parser.add_argument("--collocation-points", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--activation", choices=("tanh", "silu", "gelu"), default="gelu")
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--l0-data", type=float, default=0.03)
    parser.add_argument("--l0-ibc", type=float, default=0.003)
    parser.add_argument("--orthogonal-floor", type=float, default=0.25)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--physics-validation-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--device", default="cpu", help="PINN device: cpu, cuda, cuda:0, mps, or auto."
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch kernels when available.",
    )
    parser.add_argument("--corrected-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def completed(existing: pd.DataFrame, job: dict) -> bool:
    if existing.empty:
        return False
    method = "H-PINN" if job["mode"] == "h" else "HO-PINN"
    mask = (
        (existing["test_name"] == job["test_name"])
        & (existing["seed"] == job["seed"])
        & (existing["n_train"] == job["budget"])
        & (existing["method"] == method)
        & (existing["config_id"] == config_id_for_job(job))
        & (existing["n_collocation_per_epoch"] == job["collocation_points"])
        & (existing["epochs"] == job["train_config"]["epochs"])
    )
    return set(existing.loc[mask, "evaluation_region"]) == {
        "interpolation",
        "extrapolation",
    }


def main() -> None:
    args = parse_args()
    budgets = comma_ints(args.budgets)
    seeds = comma_ints(args.seeds)
    balancers = comma_strings(args.balancers)
    if not budgets or any(value < 0 or value > max(LABEL_BUDGETS) for value in budgets):
        raise ValueError(f"Budgets must lie in [0, {max(LABEL_BUDGETS)}]")
    if not seeds:
        raise ValueError("At least one seed is required")
    if any(value not in {"h", "ho"} for value in balancers):
        raise ValueError("Balancers must be h and/or ho")
    if args.epochs < 1 or args.collocation_points < 1 or args.workers < 1:
        raise ValueError("epochs, collocation-points, and workers must be positive")
    resolved_device = resolve_device(args.device)
    if resolved_device.type != "cpu" and args.workers != 1:
        raise ValueError(
            "A single GPU must use --workers 1. Launch one process per GPU "
            "explicitly for multi-GPU execution."
        )
    if args.test in {"test2", "all"}:
        expected = args.corrected_data_root / "test2" / "solution.npz"
        if not expected.is_file():
            raise FileNotFoundError(f"Corrected Test 2 data not found: {expected}")

    tests = ("test1", "test2") if args.test == "all" else (args.test,)
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        activation=args.activation,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        optimizer="adam",
        scheduler="cosine",
        l0_data=args.l0_data,
        l0_ibc=args.l0_ibc,
        orthogonal_floor=args.orthogonal_floor,
        collocation_ratio=1.0,
        residual_domain="full",
        gradient_clip=args.gradient_clip,
        checkpoint_criterion="raw",
        log_every=args.log_every,
        physics_validation_size=args.physics_validation_size,
        device=str(resolved_device),
        deterministic=args.deterministic,
    )

    output_dir = GENERATED_ROOT / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "zero_label_results.csv"
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    jobs = []
    for test_name in tests:
        for budget in budgets:
            for seed in seeds:
                for mode in balancers:
                    job = {
                        "test_name": test_name,
                        "budget": budget,
                        "seed": seed,
                        "mode": mode,
                        "collocation_points": args.collocation_points,
                        "experiment_name": args.experiment_name,
                        "corrected_data_root": str(args.corrected_data_root.resolve()),
                        "model_config": asdict(model_config),
                        "train_config": asdict(train_config),
                    }
                    if not completed(existing, job):
                        jobs.append(job)
                    else:
                        print(
                            f"[{test_name}] skip {mode} N={budget} seed={seed}",
                            flush=True,
                        )

    print(
        json.dumps(
            {
                "tests": tests,
                "budgets": budgets,
                "seeds": seeds,
                "balancers": balancers,
                "epochs": args.epochs,
                "fixed_collocation_points": args.collocation_points,
                "resolved_device": str(resolved_device),
                "deterministic": args.deterministic,
                "jobs_to_run": len(jobs),
                "output": str(output_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if not jobs:
        return

    rows = existing.to_dict("records") if not existing.empty else []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:
                raise RuntimeError(f"Failed job: {job}") from error
            frame = pd.DataFrame(rows).sort_values(
                ["test_name", "n_train", "seed", "method", "evaluation_region"]
            )
            frame.to_csv(output_path, index=False)

    print(f"Results: {output_path}", flush=True)


if __name__ == "__main__":
    main()
