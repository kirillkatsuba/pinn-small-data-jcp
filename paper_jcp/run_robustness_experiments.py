#!/usr/bin/env python3
"""Run the confirmatory small-data experiments used by the JCP manuscript.

The experiment matrix is intentionally concentrated at the smallest labelled
budget, where the paper makes its central claim:

* five independent repetitions for every benchmark;
* 1% and 5% label-noise tests (three paired repetitions) for Tests 2 and 4;
* PDE-collocation ratios 2 and 5 (three paired repetitions) for Tests 2 and 4;
* training-only tuning of XGB, RBF-SVR, and Gaussian-process baselines.

Every repetition uses the same labelled points for the PINN and conventional
surrogates.  The fixed test set is disjoint from the training candidate pool.
The script writes one resumable CSV per benchmark and never overwrites the
original ablation checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
GENERATED = PAPER / "generated"
CHECKPOINTS = PAPER / "experiments" / "checkpoints"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.pinn_ablation_utils import compute_multioutput_metrics, compute_scalar_metrics
from loss_balancong_algorithms.algos import HierarchicalOrthogonalPINNLossBalancer


CONFIGS = {
    "test1": dict(n=50, input_dim=2, output_dim=1, hidden=32, layers=4,
                  lr=3e-3, decay_every=3000, decay=0.9, criterion="balanced"),
    "test2": dict(n=100, input_dim=3, output_dim=7, hidden=32, layers=8,
                  lr=3e-3, decay_every=5000, decay=0.9, criterion="balanced"),
    "test3": dict(n=50, input_dim=3, output_dim=7, hidden=128, layers=6,
                  lr=3e-3, decay_every=300, decay=0.8, criterion="raw"),
    "test4": dict(n=100, input_dim=7, output_dim=7, hidden=32, layers=4,
                  lr=3e-4, decay_every=300, decay=0.9, criterion="raw"),
}

TARGET_NAMES = ["Pressure", "Soil", "Swat"]
BASE_SEEDS = [11, 22, 33, 44, 55]
STRESS_SEEDS = [11, 22, 33]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build_test23_bundle(test_name: str) -> dict:
    if test_name == "test2":
        test_dir = ROOT / "test2-2phase-disp-simple"
        data_dir = test_dir / "data_5k_200225"
        perm_path = test_dir / "perm.npy"
    else:
        test_dir = ROOT / "test3-2phase-disp-perm-nonlinear"
        data_dir = test_dir / "data_5k_3sigma"
        perm_path = test_dir / "perm_3sigma.npy"

    perm = np.load(perm_path).astype(np.float32)
    nx, ny = perm.shape
    raw_x = np.loadtxt(data_dir / "x_5000.txt", dtype=np.float32)
    raw_y = np.loadtxt(data_dir / "y_5000.txt", dtype=np.float32)
    raw_t = np.loadtxt(data_dir / "t_5000.txt", dtype=np.float32)
    sim = np.loadtxt(data_dir / "sim_5000.txt", dtype=np.float32)
    features = np.column_stack((raw_t * (3.0 / 5000.0), raw_x / nx, raw_y / ny)).astype(np.float32)
    targets = sim[:, [0, 2, 1]].astype(np.float32)
    perm_values = perm[raw_x.astype(int), raw_y.astype(int)].astype(np.float32)

    selection_paths = sorted(test_dir.glob("sim_*.npy"))
    used = np.unique(np.concatenate([np.load(path).astype(int) for path in selection_paths]))
    available = np.setdiff1d(np.arange(len(features)), used, assume_unique=False)
    rng = np.random.default_rng(20260727)
    test_idx = rng.choice(available, size=min(1500, len(available)), replace=False)
    train_candidates = np.setdiff1d(np.arange(len(features)), test_idx, assume_unique=False)
    return dict(
        features=features,
        targets=targets,
        perm=perm_values,
        test_x=features[test_idx],
        test_y=targets[test_idx],
        train_candidates=train_candidates,
    )


def build_test4_bundle(module) -> dict:
    test_dir = ROOT / "test4-2phase-disp-7d"
    set_seed(42)
    data = module.prepare_candidate_points(test_dir, "data-100-new-two-sigma/sim_100.npy")
    sim, x_idx, y_idx, x, y, t, pwat, poil, kwat, koil = data
    features = np.column_stack((t, x, y, pwat, poil, kwat, koil)).astype(np.float32)
    targets = sim[:, [0, 2, 1]].astype(np.float32)
    perm2d = np.load(test_dir / "perm_2.npy").astype(np.float32)
    perm_values = perm2d[x_idx.astype(int), y_idx.astype(int)].astype(np.float32)
    selection_paths = sorted(test_dir.glob("train-indexes-2sigma-*.npy"))
    used = np.unique(np.concatenate([np.load(path).astype(int) for path in selection_paths]))
    available = np.setdiff1d(np.arange(len(features)), used, assume_unique=False)
    rng = np.random.default_rng(20260727)
    test_idx = rng.choice(available, size=min(2000, len(available)), replace=False)
    train_candidates = np.setdiff1d(np.arange(len(features)), test_idx, assume_unique=False)
    return dict(
        features=features,
        targets=targets,
        perm=perm_values,
        test_x=features[test_idx],
        test_y=targets[test_idx],
        train_candidates=train_candidates,
    )


def build_test1_bundle(module) -> dict:
    t_grid = torch.linspace(0.0, 0.05, 81)
    x_grid = torch.linspace(0.0, 1.0, 257)
    test_x_t = torch.cartesian_prod(t_grid, x_grid)
    test_y = module.thermal_conductivity_equation(test_x_t[:, 0], test_x_t[:, 1]).numpy()
    return dict(test_x=test_x_t.numpy().astype(np.float32), test_y=test_y.astype(np.float32))


def build_bundle(test_name: str, module) -> dict:
    if test_name == "test1":
        return build_test1_bundle(module)
    if test_name in {"test2", "test3"}:
        return build_test23_bundle(test_name)
    return build_test4_bundle(module)


def select_points(test_name: str, module, bundle: dict, seed: int, ratio: int) -> dict:
    cfg = CONFIGS[test_name]
    n = cfg["n"]
    if test_name == "test1":
        data_rng = np.random.default_rng(seed)
        colloc_rng = np.random.default_rng(seed + 500)
        x_data = np.column_stack((
            data_rng.uniform(0.0, 0.05, n),
            data_rng.uniform(0.0, 1.0, n),
        )).astype(np.float32)
        y_data = module.thermal_conductivity_equation(
            torch.tensor(x_data[:, 0]), torch.tensor(x_data[:, 1])
        ).numpy().astype(np.float32)
        n_colloc = n * ratio
        x_colloc = np.column_stack((
            colloc_rng.uniform(0.0, 0.05, n_colloc),
            colloc_rng.uniform(0.0, 1.0, n_colloc),
        )).astype(np.float32)
        return dict(x_data=x_data, y_data=y_data, x_colloc=x_colloc, perm_colloc=None)

    candidates = bundle["train_candidates"]
    data_perm = np.random.default_rng(seed).permutation(candidates)
    colloc_perm = np.random.default_rng(seed + 500).permutation(candidates)
    data_idx = data_perm[:n]
    colloc_idx = colloc_perm[: n * ratio]
    return dict(
        x_data=bundle["features"][data_idx],
        y_data=bundle["targets"][data_idx],
        x_colloc=bundle["features"][colloc_idx],
        perm_colloc=bundle["perm"][colloc_idx],
    )


def add_label_noise(test_name: str, targets: np.ndarray, level: float, seed: int) -> np.ndarray:
    if level == 0.0:
        return targets.copy()
    rng = np.random.default_rng(seed + 700)
    scale = np.std(targets, axis=0, ddof=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    noisy = targets + rng.normal(size=targets.shape) * (level * scale)
    if test_name != "test1":
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype(np.float32)


def fixed_flow_pde_loss(model, points: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    t = points[:, 0:1].clone().detach().requires_grad_(True)
    x = points[:, 1:2].clone().detach().requires_grad_(True)
    y = points[:, 2:3].clone().detach().requires_grad_(True)
    result = model(torch.cat((t, x, y), dim=1))
    ones = torch.ones_like(result[:, 0])
    grad = lambda out, inp: torch.autograd.grad(
        out, inp, ones, create_graph=True, retain_graph=True, allow_unused=False
    )[0].reshape(-1)
    dpdx = grad(result[:, 0], x)
    dpdy = grad(result[:, 0], y)
    r1 = 0.1 * grad(result[:, 2], t) + grad(result[:, 5], x) + grad(result[:, 6], y)
    r2 = 0.1 * grad(result[:, 1], t) + grad(result[:, 3], x) + grad(result[:, 4], y)
    r3x = result[:, 5] + perm * result[:, 2] ** 2 * dpdx
    r3y = result[:, 6] + perm * result[:, 2] ** 2 * dpdy
    # Oil mobility depends on oil saturation (output 1), not water
    # saturation (output 2).  The legacy Test 2 script used output 2 here.
    r4x = result[:, 3] + (1.0 / 3.0) * perm * 0.1 * result[:, 1] ** 4 * dpdx
    r4y = result[:, 4] + (1.0 / 3.0) * perm * 0.1 * result[:, 1] ** 4 * dpdy
    r5 = result[:, 2] + result[:, 1] - 1.0
    return torch.mean(r1**2 + r2**2 + r3x**2 + r3y**2 + r4x**2 + r4y**2 + r5**2)


def heat_losses(model, module, x_data, y_data, x_colloc):
    prediction = model(x_data)
    loss_data = torch.mean((prediction.reshape(-1) - y_data.reshape(-1)) ** 2)

    tc = x_colloc[:, 0].clone().detach().requires_grad_(True)
    xc = x_colloc[:, 1].clone().detach().requires_grad_(True)
    interior = torch.stack((tc, xc), dim=-1)
    initial = torch.stack((torch.zeros_like(tc), xc), dim=-1)
    left = torch.stack((tc, torch.zeros_like(xc)), dim=-1)
    right = torch.stack((tc, torch.ones_like(xc)), dim=-1)
    target_initial = module.thermal_conductivity_equation(torch.zeros_like(tc), xc)
    loss_ibc = torch.mean(
        (model(initial).reshape(-1) - target_initial) ** 2
        + (model(left).reshape(-1) - model(right).reshape(-1)) ** 2
    )
    pde = model(interior).reshape(-1)
    dt = torch.autograd.grad(pde, tc, torch.ones_like(pde), create_graph=True)[0]
    dx = torch.autograd.grad(pde, xc, torch.ones_like(pde), create_graph=True)[0]
    dxx = torch.autograd.grad(dx, xc, torch.ones_like(dx), create_graph=True)[0]
    loss_pde = torch.mean((dt - dxx) ** 2)
    return loss_data, loss_ibc, loss_pde


def flow_losses(test_name, model, module, x_data, y_data, x_colloc, perm_colloc):
    if test_name == "test2":
        loss_pde = fixed_flow_pde_loss(model, x_colloc, perm_colloc)
    else:
        residuals = module.compute_pde_residuals(model, x_colloc, perm_colloc)
        loss_pde = torch.mean(sum(residual**2 for residual in residuals[:7]))

    if test_name in {"test2", "test3"}:
        t, _x, y = x_colloc[:, 0], x_colloc[:, 1], x_colloc[:, 2]
        left = torch.stack((t, torch.zeros_like(_x), y), dim=-1)
        right = torch.stack((t, torch.ones_like(_x), y), dim=-1)
        initial = torch.stack((torch.zeros_like(t), _x, y), dim=-1)
    else:
        t, _x, y = x_colloc[:, 0], x_colloc[:, 1], x_colloc[:, 2]
        params = [x_colloc[:, idx] for idx in range(3, 7)]
        left = torch.stack((t, torch.zeros_like(_x), y, *params), dim=-1)
        right = torch.stack((t, torch.ones_like(_x), y, *params), dim=-1)
        initial = torch.stack((torch.zeros_like(t), _x, y, *params), dim=-1)
    loss_ibc = torch.mean(
        (model(left)[:, 0] - 1.0) ** 2
        + model(right)[:, 0] ** 2
        + (model(initial)[:, 1] - 1.0) ** 2
        + model(initial)[:, 2] ** 2
    )
    prediction = model(x_data)
    loss_data = torch.mean(
        (prediction[:, 0] - y_data[:, 0]) ** 2
        + (prediction[:, 1] - y_data[:, 1]) ** 2
        + (prediction[:, 2] - y_data[:, 2]) ** 2
    )
    return loss_data, loss_ibc, loss_pde


def train_pinn(test_name: str, module, selected: dict, noisy_y: np.ndarray,
               seed: int, epochs: int, checkpoint_path: Path,
               initialization_seed: int | None = None) -> tuple[torch.nn.Module, float]:
    cfg = CONFIGS[test_name]
    set_seed(seed + 1000 if initialization_seed is None else initialization_seed)
    model_class = module.PINN if test_name in {"test1", "test2"} else module.ModifiedPINN
    model = model_class(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden"],
        num_layers=cfg["layers"], output_dim=cfg["output_dim"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    balancer = HierarchicalOrthogonalPINNLossBalancer(
        L0_mse=1e-2, L0_ibc=1e-3, use_orthogonal_factor=True, orthogonal_floor=0.5
    )
    x_data = torch.tensor(selected["x_data"], dtype=torch.float32)
    y_data = torch.tensor(noisy_y, dtype=torch.float32)
    x_colloc = torch.tensor(selected["x_colloc"], dtype=torch.float32)
    perm = None if selected["perm_colloc"] is None else torch.tensor(
        selected["perm_colloc"], dtype=torch.float32
    )
    best_value = float("inf")
    best_state = None
    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        if test_name == "test1":
            losses = heat_losses(model, module, x_data, y_data, x_colloc)
        else:
            losses = flow_losses(test_name, model, module, x_data, y_data, x_colloc, perm)
        loss_data, loss_ibc, loss_pde = losses
        w_data, w_ibc, w_pde, _ = balancer.compute_weights(
            model, loss_data, loss_ibc, loss_pde
        )
        balanced = w_data * loss_data + w_ibc * loss_ibc + w_pde * loss_pde
        raw = loss_data + loss_ibc + loss_pde
        balanced.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        value = float((balanced if cfg["criterion"] == "balanced" else raw).detach())
        if value < best_value:
            best_value = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if cfg["decay_every"] and (epoch + 1) % cfg["decay_every"] == 0:
            optimizer.param_groups[0]["lr"] *= cfg["decay"]
    train_time = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("PINN training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    return model, train_time


def predict_pinn(test_name: str, model, features: np.ndarray) -> np.ndarray:
    tensor = torch.tensor(features, dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), 4096):
            chunks.append(model(tensor[start:start + 4096]).numpy())
    pred = np.concatenate(chunks, axis=0)
    return pred.reshape(-1) if test_name == "test1" else pred[:, :3]


def classical_cv_splits(
    x_train: np.ndarray,
    seed: int,
    strategy: str = "random",
):
    """Return training-only CV with optional expanding temporal windows."""

    if strategy == "random":
        return KFold(n_splits=3, shuffle=True, random_state=seed)
    if strategy != "time":
        raise ValueError("classical CV strategy must be 'random' or 'time'")
    if x_train.ndim != 2 or x_train.shape[1] < 1:
        raise ValueError("time-aware CV requires time in the first input column")
    times = np.asarray(x_train[:, 0])
    unique_times = np.unique(times)
    if len(unique_times) < 4:
        raise ValueError(
            "time-aware CV requires at least four distinct labelled times; "
            f"got {len(unique_times)}"
        )
    time_blocks = [block for block in np.array_split(unique_times, 4) if len(block)]
    if len(time_blocks) != 4:
        raise ValueError("Could not construct four non-empty temporal CV blocks")
    splits = []
    for validation_block in range(1, 4):
        train_times = np.concatenate(time_blocks[:validation_block])
        validation_times = time_blocks[validation_block]
        train_indices = np.flatnonzero(np.isin(times, train_times))
        validation_indices = np.flatnonzero(np.isin(times, validation_times))
        if len(train_indices) < 2 or len(validation_indices) < 1:
            raise ValueError("Temporal CV produced an empty or undersized fold")
        if not float(np.max(times[train_indices])) < float(
            np.min(times[validation_indices])
        ):
            raise ValueError("Temporal CV fold is not strictly forward-looking")
        splits.append((train_indices, validation_indices))
    return splits


def _json_safe_parameter(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe_parameter(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_parameter(item) for key, item in value.items()}
    return str(value)


def tune_classical(
    method: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    *,
    tuning_profile: str = "standard",
    cv_strategy: str = "random",
    return_metadata: bool = False,
):
    """Tune one regressor per target using labels only.

    ``standard`` reproduces the original compact search. ``strong`` expands
    XGB/SVR search spaces and compares several GP kernels.  ``time`` CV uses
    three expanding-window folds over the labelled time range, so every
    validation block is strictly later than its corresponding training block.
    """

    if tuning_profile not in {"standard", "strong"}:
        raise ValueError("tuning_profile must be 'standard' or 'strong'")
    multi = y_train.ndim == 2
    targets = y_train.shape[1] if multi else 1
    y_matrix = y_train if multi else y_train.reshape(-1, 1)
    cv = classical_cv_splits(x_train, seed, cv_strategy)
    estimators = []
    tuning_details = []
    start = time.perf_counter()
    for idx in range(targets):
        target = y_matrix[:, idx]
        if method == "XGB":
            base = xgb.XGBRegressor(
                objective="reg:squarederror", random_state=seed, n_jobs=1,
                tree_method="hist",
            )
            if tuning_profile == "standard":
                estimator = GridSearchCV(
                    base,
                    {"n_estimators": [100, 300], "max_depth": [2, 4],
                     "learning_rate": [0.03, 0.1], "subsample": [0.9],
                     "colsample_bytree": [1.0]},
                    scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
                )
            else:
                estimator = RandomizedSearchCV(
                    base,
                    {
                        "n_estimators": [100, 300, 600, 1000],
                        "max_depth": [2, 3, 4, 6],
                        "learning_rate": [0.01, 0.03, 0.05, 0.1],
                        "min_child_weight": [1, 3, 5],
                        "subsample": [0.7, 0.85, 1.0],
                        "colsample_bytree": [0.7, 0.9, 1.0],
                        "reg_alpha": [0.0, 0.01, 0.1],
                        "reg_lambda": [0.1, 1.0, 10.0],
                    },
                    n_iter=32,
                    scoring="neg_mean_squared_error",
                    cv=cv,
                    n_jobs=1,
                    random_state=seed + idx,
                )
        elif method == "SVM":
            input_pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("svr", SVR(kernel="rbf")),
            ])
            if tuning_profile == "standard":
                base = input_pipeline
                estimator = GridSearchCV(
                    base,
                    {"svr__C": [1.0, 10.0, 100.0],
                     "svr__gamma": ["scale", 0.1, 1.0],
                     "svr__epsilon": [1e-3]},
                    scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
                )
            else:
                base = TransformedTargetRegressor(
                    regressor=input_pipeline,
                    transformer=StandardScaler(),
                )
                estimator = RandomizedSearchCV(
                    base,
                    {
                        "regressor__svr__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
                        "regressor__svr__gamma": [
                            "scale", "auto", 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
                        ],
                        "regressor__svr__epsilon": [
                            1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1
                        ],
                    },
                    n_iter=40,
                    scoring="neg_mean_squared_error",
                    cv=cv,
                    n_jobs=1,
                    random_state=seed + idx,
                )
        elif method == "GP":
            def kernel_with(base_kernel):
                return (
                    ConstantKernel(1.0, (1e-2, 1e2)) * base_kernel
                    + WhiteKernel(1e-5, (1e-8, 1e-1))
                )

            rbf_kernel = kernel_with(
                RBF(np.ones(x_train.shape[1]), (1e-2, 1e2))
            )
            base = Pipeline([
                ("scale", StandardScaler()),
                ("gp", GaussianProcessRegressor(
                    kernel=rbf_kernel, alpha=1e-6, normalize_y=True,
                    n_restarts_optimizer=1 if tuning_profile == "standard" else 2,
                    random_state=seed,
                )),
            ])
            if tuning_profile == "standard":
                estimator = base
            else:
                kernels = [
                    rbf_kernel,
                    kernel_with(
                        Matern(
                            np.ones(x_train.shape[1]),
                            (1e-2, 1e2),
                            nu=1.5,
                        )
                    ),
                    kernel_with(
                        Matern(
                            np.ones(x_train.shape[1]),
                            (1e-2, 1e2),
                            nu=2.5,
                        )
                    ),
                    kernel_with(RationalQuadratic(alpha=1.0, length_scale=1.0)),
                ]
                estimator = GridSearchCV(
                    base,
                    {"gp__kernel": kernels, "gp__alpha": [1e-8, 1e-6, 1e-4]},
                    scoring="neg_mean_squared_error",
                    cv=cv,
                    n_jobs=1,
                )
        else:
            raise ValueError(method)
        with warnings.catch_warnings():
            if method == "GP":
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            estimator.fit(x_train, target)
        estimators.append(estimator)
        detail = {
            "target_index": idx,
            "target_name": (
                TARGET_NAMES[idx]
                if multi and idx < len(TARGET_NAMES)
                else "target"
            ),
        }
        if hasattr(estimator, "best_params_"):
            detail["best_params"] = _json_safe_parameter(estimator.best_params_)
            detail["best_cv_neg_mse"] = float(estimator.best_score_)
            detail["search_type"] = type(estimator).__name__
            detail["candidate_count"] = len(estimator.cv_results_["params"])
            detail["cv_fold_count"] = (
                cv.get_n_splits(x_train, target)
                if hasattr(cv, "get_n_splits")
                else len(cv)
            )
            fitted = estimator.best_estimator_
        else:
            detail["best_params"] = {}
            detail["best_cv_neg_mse"] = None
            detail["search_type"] = "marginal_likelihood_optimisation"
            detail["candidate_count"] = 1
            detail["cv_fold_count"] = 0
            fitted = estimator
        if method == "GP":
            detail["fitted_kernel"] = str(fitted.named_steps["gp"].kernel_)
            detail["log_marginal_likelihood"] = float(
                fitted.named_steps["gp"].log_marginal_likelihood_value_
            )
        tuning_details.append(detail)
    elapsed = time.perf_counter() - start
    if return_metadata:
        return estimators, elapsed, tuning_details
    return estimators, elapsed


def predict_classical(estimators, features: np.ndarray, scalar: bool) -> np.ndarray:
    pred = np.column_stack([estimator.predict(features) for estimator in estimators])
    return pred.reshape(-1) if scalar else pred


def metrics(test_name: str, truth: np.ndarray, prediction: np.ndarray) -> dict:
    if test_name == "test1":
        return compute_scalar_metrics(truth.reshape(-1), prediction.reshape(-1))
    return compute_multioutput_metrics(truth, prediction, TARGET_NAMES)


def conditions_for(test_name: str) -> list[tuple[str, int, float, int]]:
    conditions = [("repeated", seed, 0.0, 1) for seed in BASE_SEEDS]
    if test_name in {"test2", "test4"}:
        conditions.extend(
            ("noise", seed, noise, 1)
            for noise in (0.01, 0.05)
            for seed in STRESS_SEEDS
        )
        conditions.extend(
            ("collocation", seed, 0.0, ratio)
            for ratio in (2, 5)
            for seed in STRESS_SEEDS
        )
    return conditions


def run_test(test_name: str, epochs: int, smoke: bool = False) -> str:
    from paper_jcp.evaluate_fixed_holdout import load_training_module

    set_seed(42)
    module = load_training_module(test_name)
    bundle = build_bundle(test_name, module)
    prefix = "smoke_" if smoke else ""
    output_path = GENERATED / f"{prefix}robustness_{test_name}.csv"
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if len(existing) else []

    conditions = conditions_for(test_name)[:1] if smoke else conditions_for(test_name)
    for kind, seed, noise, ratio in conditions:
        key_mask = None
        if len(existing):
            key_mask = (
                (existing["experiment"] == kind)
                & (existing["seed"] == seed)
                & np.isclose(existing["noise_level"], noise)
                & (existing["collocation_ratio"] == ratio)
            )
        if key_mask is not None and int(key_mask.sum()) == 4:
            print(f"[{test_name}] skip {kind} seed={seed} noise={noise} ratio={ratio}", flush=True)
            continue

        selected = select_points(test_name, module, bundle, seed, ratio)
        noisy_y = add_label_noise(test_name, selected["y_data"], noise, seed)
        tag = f"{prefix}{test_name}_{kind}_s{seed}_noise{noise:g}_r{ratio}"
        checkpoint = CHECKPOINTS / f"{tag}.pth"
        model, train_time = train_pinn(
            test_name, module, selected, noisy_y, seed, epochs, checkpoint
        )
        pred = predict_pinn(test_name, model, bundle["test_x"])
        condition_rows = [{
            "test_name": test_name, "experiment": kind, "seed": seed,
            "noise_level": noise, "collocation_ratio": ratio,
            "method": "HO-PINN", "n_train": CONFIGS[test_name]["n"],
            "n_collocation": CONFIGS[test_name]["n"] * ratio,
            "n_test": len(bundle["test_y"]), "train_time_sec": train_time,
            **metrics(test_name, bundle["test_y"], pred),
        }]
        for method in ("XGB", "SVM", "GP"):
            estimators, fit_time = tune_classical(
                method, selected["x_data"], noisy_y, seed
            )
            pred = predict_classical(estimators, bundle["test_x"], test_name == "test1")
            condition_rows.append({
                "test_name": test_name, "experiment": kind, "seed": seed,
                "noise_level": noise, "collocation_ratio": ratio,
                "method": method, "n_train": CONFIGS[test_name]["n"],
                "n_collocation": 0, "n_test": len(bundle["test_y"]),
                "train_time_sec": fit_time,
                **metrics(test_name, bundle["test_y"], pred),
            })
        if key_mask is not None and key_mask.any():
            existing = existing.loc[~key_mask].copy()
            rows = existing.to_dict("records")
        rows.extend(condition_rows)
        GENERATED.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        existing = pd.DataFrame(rows)
        print(
            f"[{test_name}] done {kind} seed={seed} noise={noise} ratio={ratio} "
            f"pinn_time={train_time:.1f}s",
            flush=True,
        )
    return str(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    tests = list(CONFIGS) if args.test == "all" else [args.test]
    if len(tests) == 1:
        print(run_test(tests[0], args.epochs, args.smoke))
        return
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tests))) as executor:
        futures = {
            executor.submit(run_test, test, args.epochs, args.smoke): test
            for test in tests
        }
        for future in as_completed(futures):
            test = futures[future]
            print(f"Completed {test}: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
