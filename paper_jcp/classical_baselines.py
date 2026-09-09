"""Training-only tuning for XGB, RBF-SVM, and Gaussian-process baselines."""

from __future__ import annotations

import time
import warnings

import numpy as np
import xgboost as xgb
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, RBF, RationalQuadratic, WhiteKernel,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TARGET_NAMES = ("Pressure", "Soil", "Swat")


def classical_cv_splits(x_train: np.ndarray, seed: int, strategy: str = "random"):
    """Return random folds or strictly forward expanding temporal folds."""
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
    blocks = [block for block in np.array_split(unique_times, 4) if len(block)]
    if len(blocks) != 4:
        raise ValueError("Could not construct four non-empty temporal CV blocks")
    splits = []
    for validation_block in range(1, 4):
        train_indices = np.flatnonzero(
            np.isin(times, np.concatenate(blocks[:validation_block]))
        )
        validation_indices = np.flatnonzero(np.isin(times, blocks[validation_block]))
        if len(train_indices) < 2 or len(validation_indices) < 1:
            raise ValueError("Temporal CV produced an empty or undersized fold")
        if not float(times[train_indices].max()) < float(times[validation_indices].min()):
            raise ValueError("Temporal CV fold is not strictly forward-looking")
        splits.append((train_indices, validation_indices))
    return splits


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _xgb(profile: str, cv, seed: int, target_index: int):
    base = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=seed, n_jobs=1, tree_method="hist"
    )
    if profile == "standard":
        return GridSearchCV(
            base,
            {
                "n_estimators": [100, 300], "max_depth": [2, 4],
                "learning_rate": [0.03, 0.1], "subsample": [0.9],
                "colsample_bytree": [1.0],
            },
            scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
        )
    if profile == "strong":
        distributions = {
            "n_estimators": [100, 300, 600, 1000], "max_depth": [2, 3, 4, 6],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "min_child_weight": [1, 3, 5], "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.9, 1.0], "reg_alpha": [0.0, 0.01, 0.1],
            "reg_lambda": [0.1, 1.0, 10.0],
        }
        n_iter = 32
    else:
        distributions = {
            "n_estimators": [100, 200, 400, 700, 1000, 1400, 1800],
            "max_depth": [2, 3, 4, 5, 6, 8],
            "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
            "min_child_weight": [1, 2, 3, 5, 8, 12],
            "subsample": [0.6, 0.75, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
            "gamma": [0.0, 1e-3, 1e-2, 0.1],
            "reg_alpha": [0.0, 1e-3, 1e-2, 0.1, 1.0],
            "reg_lambda": [0.01, 0.1, 1.0, 10.0, 100.0],
        }
        n_iter = 96
    return RandomizedSearchCV(
        base,
        distributions,
        n_iter=n_iter, scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
        random_state=seed + target_index,
    )


def _svm(profile: str, cv, seed: int, target_index: int):
    pipeline = Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    if profile == "standard":
        return GridSearchCV(
            pipeline,
            {
                "svr__C": [1.0, 10.0, 100.0],
                "svr__gamma": ["scale", 0.1, 1.0], "svr__epsilon": [1e-3],
            },
            scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
        )
    transformed = TransformedTargetRegressor(
        regressor=pipeline, transformer=StandardScaler()
    )
    if profile == "strong":
        distributions = {
            "regressor__svr__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
            "regressor__svr__gamma": [
                "scale", "auto", 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0
            ],
            "regressor__svr__epsilon": [1e-4, 1e-3, 1e-2, 0.03, 0.05, 0.1],
        }
        n_iter = 40
    else:
        distributions = {
            "regressor__svr__C": [
                0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0,
                100.0, 300.0, 1000.0, 3000.0,
            ],
            "regressor__svr__gamma": [
                "scale", "auto", 0.003, 0.01, 0.03, 0.1,
                0.3, 1.0, 3.0, 10.0, 30.0,
            ],
            "regressor__svr__epsilon": [
                1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 0.03, 0.05, 0.1,
            ],
        }
        n_iter = 96
    return RandomizedSearchCV(
        transformed,
        distributions,
        n_iter=n_iter, scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
        random_state=seed + target_index,
    )


def _gp(profile: str, cv, seed: int, input_dim: int):
    def kernel(base):
        return ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(
            1e-5, (1e-8, 1e-1)
        )

    rbf = kernel(RBF(np.ones(input_dim), (1e-2, 1e2)))
    base = Pipeline([
        ("scale", StandardScaler()),
        ("gp", GaussianProcessRegressor(
            kernel=rbf, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer={"standard": 1, "strong": 2, "extensive": 3}[profile],
            random_state=seed,
        )),
    ])
    if profile == "standard":
        return base
    kernels = [
        rbf,
        kernel(Matern(np.ones(input_dim), (1e-2, 1e2), nu=1.5)),
        kernel(Matern(np.ones(input_dim), (1e-2, 1e2), nu=2.5)),
        kernel(RationalQuadratic(alpha=1.0, length_scale=1.0)),
    ]
    alphas = [1e-8, 1e-6, 1e-4]
    if profile == "extensive":
        kernels.insert(1, kernel(Matern(np.ones(input_dim), (1e-2, 1e2), nu=0.5)))
        alphas = [1e-10, 1e-8, 1e-6, 1e-4]
    return GridSearchCV(
        base, {"gp__kernel": kernels, "gp__alpha": alphas},
        scoring="neg_mean_squared_error", cv=cv, n_jobs=1,
    )


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
    """Tune one regressor per output without accessing final evaluation data."""
    if tuning_profile not in {"standard", "strong", "extensive"}:
        raise ValueError(
            "tuning_profile must be 'standard', 'strong', or 'extensive'"
        )
    y_matrix = y_train if y_train.ndim == 2 else y_train.reshape(-1, 1)
    multi = y_train.ndim == 2
    cv = classical_cv_splits(x_train, seed, cv_strategy)
    estimators, details = [], []
    start = time.perf_counter()
    for index in range(y_matrix.shape[1]):
        if method == "XGB":
            estimator = _xgb(tuning_profile, cv, seed, index)
        elif method == "SVM":
            estimator = _svm(tuning_profile, cv, seed, index)
        elif method == "GP":
            estimator = _gp(tuning_profile, cv, seed, x_train.shape[1])
        else:
            raise ValueError(f"Unsupported classical method: {method}")
        with warnings.catch_warnings():
            if method == "GP":
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            estimator.fit(x_train, y_matrix[:, index])
        estimators.append(estimator)
        detail = {
            "target_index": index,
            "target_name": TARGET_NAMES[index] if multi else "target",
        }
        if hasattr(estimator, "best_params_"):
            detail.update({
                "best_params": _json_safe(estimator.best_params_),
                "best_cv_neg_mse": float(estimator.best_score_),
                "search_type": type(estimator).__name__,
                "candidate_count": len(estimator.cv_results_["params"]),
                "cv_fold_count": (
                    cv.get_n_splits(x_train, y_matrix[:, index])
                    if hasattr(cv, "get_n_splits") else len(cv)
                ),
            })
            fitted = estimator.best_estimator_
        else:
            detail.update({
                "best_params": {}, "best_cv_neg_mse": None,
                "search_type": "marginal_likelihood_optimisation",
                "candidate_count": 1, "cv_fold_count": 0,
            })
            fitted = estimator
        if method == "GP":
            detail["fitted_kernel"] = str(fitted.named_steps["gp"].kernel_)
            detail["log_marginal_likelihood"] = float(
                fitted.named_steps["gp"].log_marginal_likelihood_value_
            )
        details.append(detail)
    elapsed = time.perf_counter() - start
    return (estimators, elapsed, details) if return_metadata else (estimators, elapsed)


def predict_classical(estimators, features: np.ndarray, scalar: bool) -> np.ndarray:
    prediction = np.column_stack([model.predict(features) for model in estimators])
    return prediction.reshape(-1) if scalar else prediction
