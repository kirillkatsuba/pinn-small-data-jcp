#!/usr/bin/env python3
"""Build figures and LaTeX tables for the JCP manuscript."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PAPER_DIR / "figures"
GENERATED_DIR = PAPER_DIR / "generated"
CACHE_DIR = PAPER_DIR / ".cache"
CHECKPOINTS_DIR = PAPER_DIR / "experiments" / "checkpoints"

for path in (FIGURES_DIR, GENERATED_DIR, CACHE_DIR / "matplotlib"):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmarking.pinn_ablation_utils import load_module
from paper_jcp.evaluate_fixed_holdout import (
    ML_METHODS,
    TEST_CONFIGS,
    fit_ml_model,
    load_pool,
    load_training_module,
    predict_ml_model,
)
from paper_jcp.run_robustness_experiments import predict_classical, tune_classical


METHOD_ORDER = ["HO-PINN", "XGB", "SVM", "GP"]
METHOD_COLORS = {
    "HO-PINN": "#c0392b",
    "XGB": "#4c78a8",
    "SVM": "#59a14f",
    "GP": "#b279a2",
}
METHOD_MARKERS = {"HO-PINN": "*", "XGB": "o", "SVM": "s", "GP": "D"}
TEST_TITLES = {
    "test1": "Test 1: heat equation",
    "test2": "Test 2: low-variance flow",
    "test3": "Test 3: high-variance flow",
    "test4": "Test 4: 7D parametric flow",
}


def metric_name(test_name: str, base: str) -> str:
    return base if test_name == "test1" else f"mean_{base}"


def load_results() -> pd.DataFrame:
    path = GENERATED_DIR / "fixed_holdout_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run paper_jcp/evaluate_fixed_holdout.py first"
        )
    results = pd.read_csv(path)
    corrected = GENERATED_DIR / "fixed_holdout_corrected_test2.csv"
    if corrected.exists():
        replacement = pd.read_csv(corrected)
        if len(replacement) != 16:
            raise ValueError(f"Incomplete corrected Test 2 sweep: {len(replacement)} rows")
        results = pd.concat(
            [results[results["test_name"] != "test2"], replacement],
            ignore_index=True,
        )
    tuned = GENERATED_DIR / "tuned_scaling_baselines.csv"
    if tuned.exists():
        replacement = pd.read_csv(tuned)
        if len(replacement) != 48:
            raise ValueError(f"Incomplete tuned scaling baselines: {len(replacement)} rows")
        results = pd.concat(
            [results[results["method"] == "HO-PINN"], replacement],
            ignore_index=True,
        )
    return results


def plot_data_efficiency(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    for ax, test_name in zip(axes.flat, TEST_CONFIGS):
        subset = results[results["test_name"] == test_name]
        r2_col = metric_name(test_name, "R2")
        for method in METHOD_ORDER:
            rows = subset[subset["method"] == method].sort_values("n_train")
            ax.plot(
                rows["n_train"],
                rows[r2_col],
                label=method,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=9 if method == "HO-PINN" else 5,
                linewidth=2.2 if method == "HO-PINN" else 1.7,
            )
        ax.set_xscale("log")
        ax.set_xticks(sorted(subset["n_train"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_title(TEST_TITLES[test_name])
        ax.set_xlabel("Number of labelled training points, $N$")
        ax.set_ylabel("$R^2$" if test_name == "test1" else "Mean $R^2$")
        ax.grid(True, alpha=0.28)
        if test_name == "test1":
            ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.savefig(FIGURES_DIR / "data_efficiency_r2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_smallest_sample(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    for ax, test_name in zip(axes.flat, TEST_CONFIGS):
        subset = results[results["test_name"] == test_name]
        n_min = int(subset["n_train"].min())
        subset = subset[subset["n_train"] == n_min].set_index("method").loc[METHOD_ORDER]
        r2_col = metric_name(test_name, "R2")
        values = subset[r2_col].to_numpy()
        bars = ax.bar(
            np.arange(len(METHOD_ORDER)),
            values,
            color=[METHOD_COLORS[m] for m in METHOD_ORDER],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=9)
        ax.set_xticks(np.arange(len(METHOD_ORDER)), METHOD_ORDER, rotation=18, ha="right")
        ax.set_title(f"{TEST_TITLES[test_name]} ($N={n_min}$)")
        ax.set_ylabel("$R^2$" if test_name == "test1" else "Mean $R^2$")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES_DIR / "smallest_sample_r2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_cost_quality(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    for ax, test_name in zip(axes.flat, TEST_CONFIGS):
        subset = results[results["test_name"] == test_name]
        n_min = int(subset["n_train"].min())
        subset = subset[subset["n_train"] == n_min].set_index("method").loc[METHOD_ORDER]
        r2_col = metric_name(test_name, "R2")
        for method, row in subset.iterrows():
            ax.scatter(
                row["train_time_sec"],
                row[r2_col],
                s=150 if method == "HO-PINN" else 70,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                label=method,
                zorder=3,
            )
            ax.annotate(method, (row["train_time_sec"], row[r2_col]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(f"{TEST_TITLES[test_name]} ($N={n_min}$)")
        ax.set_xlabel("Model fitting time, s (log scale)")
        ax.set_ylabel("$R^2$" if test_name == "test1" else "Mean $R^2$")
        ax.grid(True, alpha=0.28)
    fig.savefig(FIGURES_DIR / "cost_quality_tradeoff.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def resolve_metrics_csv(test_name: str, row: pd.Series) -> Path:
    test_dir = ROOT_DIR / TEST_CONFIGS[test_name]["directory"]
    path = Path(str(row["metrics_csv_path"]))
    if path.exists():
        return path
    candidate = test_dir / "ablation_results" / "training_runs" / path.name
    if not candidate.exists():
        raise FileNotFoundError(path)
    return candidate


def plot_training_schedule() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    labels = {
        "weight_data": "Data",
        "weight_bc_ic": "Initial/boundary",
        "weight_pde": "PDE residual",
    }
    colors = {
        "weight_data": "#c0392b",
        "weight_bc_ic": "#4c78a8",
        "weight_pde": "#59a14f",
    }
    for ax, test_name in zip(axes.flat, TEST_CONFIGS):
        test_dir = ROOT_DIR / TEST_CONFIGS[test_name]["directory"]
        table = pd.read_csv(test_dir / "ablation_results" / "ablation_results.csv")
        subset = table[table["balancer"] == "grad-orth"]
        n_min = int(subset["n_collocation"].min())
        row = subset[subset["n_collocation"] == n_min].iloc[0]
        history = pd.read_csv(resolve_metrics_csv(test_name, row))
        stride = max(1, len(history) // 1200)
        history = history.iloc[::stride]
        progress = history["epoch"] / float(history["epoch"].max())
        for column in labels:
            ax.plot(progress, history[column], label=labels[column], color=colors[column], linewidth=1.8)
        ax.set_title(f"{TEST_TITLES[test_name]} ($N={n_min}$)")
        ax.set_xlabel("Normalized training progress")
        ax.set_ylabel("Normalized loss weight")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.28)
        if test_name == "test1":
            ax.legend(frameon=False, fontsize=9)
    fig.savefig(FIGURES_DIR / "hierarchical_weight_schedule.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def latex_value(value: float, best: bool, digits: int = 4) -> str:
    text = f"{value:.{digits}f}"
    return f"\\textbf{{{text}}}" if best else text


def write_metric_table(results: pd.DataFrame, base_metric: str, filename: str) -> None:
    test_names = list(TEST_CONFIGS)
    selected = {}
    for test_name in test_names:
        subset = results[results["test_name"] == test_name]
        n_min = int(subset["n_train"].min())
        selected[test_name] = subset[subset["n_train"] == n_min].set_index("method")

    higher_is_better = base_metric == "R2"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Fixed held-out performance at the smallest labelled-data budget. "
        + ("Higher is better." if higher_is_better else "Lower is better.")
        + "}",
        f"\\label{{tab:fixed_{base_metric.lower()}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Test 1 ($N=50$) & Test 2 ($N=100$) & Test 3 ($N=50$) & Test 4 ($N=100$) \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        cells = []
        for test_name in test_names:
            column = metric_name(test_name, base_metric)
            values = selected[test_name][column]
            target = values.max() if higher_is_better else values.min()
            value = float(values.loc[method])
            cells.append(latex_value(value, np.isclose(value, target)))
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def write_runtime_table(results: pd.DataFrame) -> None:
    test_names = list(TEST_CONFIGS)
    selected = {}
    for test_name in test_names:
        subset = results[results["test_name"] == test_name]
        n_min = int(subset["n_train"].min())
        selected[test_name] = subset[subset["n_train"] == n_min].set_index("method")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Model fitting time in seconds at the smallest labelled-data budget. "
        "Reference-simulator data generation is excluded for every method.}",
        "\\label{tab:runtime}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Test 1 & Test 2 & Test 3 & Test 4 \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        cells = [f"{selected[t].loc[method, 'train_time_sec']:.4g}" for t in test_names]
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED_DIR / "runtime_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_per_target_table(results: pd.DataFrame) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-output $R^2$ at the smallest labelled-data budget for the flow benchmarks. Higher is better within each test and output.}",
        "\\label{tab:per_target_r2}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Benchmark & Method & Pressure & Oil saturation & Water saturation \\\\",
        "\\midrule",
    ]
    for test_name in ("test2", "test3", "test4"):
        subset = results[results["test_name"] == test_name]
        n_min = int(subset["n_train"].min())
        subset = subset[subset["n_train"] == n_min].set_index("method").loc[METHOD_ORDER]
        best = {c: subset[c].max() for c in ("Pressure_R2", "Soil_R2", "Swat_R2")}
        for idx, method in enumerate(METHOD_ORDER):
            values = [latex_value(float(subset.loc[method, c]), np.isclose(subset.loc[method, c], best[c]))
                      for c in ("Pressure_R2", "Soil_R2", "Swat_R2")]
            label = f"Test {test_name[-1]} ($N={n_min}$)" if idx == 0 else ""
            lines.append(f"{label} & {method} & " + " & ".join(values) + " \\\\")
        if test_name != "test4":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED_DIR / "per_target_r2_table.tex").write_text("\n".join(lines), encoding="utf-8")


def load_plot_utils():
    return load_module(
        ROOT_DIR / "scripts" / "plot_min_n_validation_fields.py",
        "jcp_min_n_plot_utils",
    )


def reference_features(test_name: str, reference: dict, params: dict) -> torch.Tensor:
    nx0, nx1 = reference["nx0"], reference["nx1"]
    if test_name in {"test2", "test3"}:
        return torch.stack(
            (
                reference["time_for_model"],
                reference["cartesian_points"][:, 0],
                reference["cartesian_points"][:, 1],
            ),
            dim=-1,
        )
    n = nx0 * nx1
    final_time = 0.15e-1 * 100
    x = torch.linspace(0.0, 1.0 - 1.0 / nx0, nx0)
    y = torch.linspace(0.0, 1.0 - 1.0 / nx1, nx1)
    xy = torch.cartesian_prod(x, y)
    return torch.stack(
        (
            torch.full((n,), final_time),
            xy[:, 0],
            xy[:, 1],
            torch.full((n,), params["pwat"]),
            torch.full((n,), params["poil"]),
            torch.full((n,), params["kwat"]),
            torch.full((n,), params["koil"]),
        ),
        dim=-1,
    )


def build_qualitative_figures() -> None:
    plot_utils = load_plot_utils()
    device = torch.device("cpu")
    for test_name in TEST_CONFIGS:
        config = plot_utils.TEST_CONFIGS[test_name]
        test_dir, n_min, runs = plot_utils.load_min_n_runs(test_name)
        ho_row = runs[runs["balancer"] == "grad-orth"].iloc[0]
        module = load_training_module(test_name)
        summary = plot_utils.load_summary(ho_row, test_dir)
        checkpoint = plot_utils.resolve_repo_path(ho_row["best_checkpoint"], test_dir)
        corrected_checkpoint = CHECKPOINTS_DIR / f"test2_corrected_scaling_n{n_min}.pth"
        if test_name == "test2" and corrected_checkpoint.exists():
            checkpoint = corrected_checkpoint
        pool = load_pool(test_name, module)
        x_train, y_train = pool["training_set"](n_min)

        if test_name == "test1":
            times = [float(v) for v in config["profile_times"]]
            x_grid = torch.linspace(0.0, 1.0, 400)
            truth = plot_utils.build_test1_profiles(module, times, x_grid)
            predictions = {
                "HO-PINN": plot_utils.predict_test1_profiles(
                    module, summary, checkpoint, times, x_grid, device
                )
            }
            for method in ML_METHODS:
                estimator, _ = tune_classical(method, x_train, y_train, seed=42)
                profiles = {}
                for t_value in times:
                    features = np.column_stack(
                        (np.full(len(x_grid), t_value), x_grid.numpy())
                    ).astype(np.float32)
                    profiles[t_value] = predict_classical(estimator, features, scalar=True).reshape(-1)
                predictions[method] = profiles
            plot_utils.plot_test1_profiles_grid(
                times, n_min, x_grid, truth, predictions,
                FIGURES_DIR / "qualitative_test1.png", 220
            )
            continue

        params = plot_utils.get_plot_params(summary, config["default_plot_params"])
        # The historical Test 4 plotting defaults include an extrapolative
        # parameter combination.  The JCP comparison is intended to visualize
        # interpolation within the stated training ranges, so use their central
        # values for the qualitative panel.
        if test_name == "test4":
            params = {"pwat": 2.0, "poil": 4.0, "kwat": 1.0, "koil": 0.3}
        reference = plot_utils.build_reference(test_name, module, test_dir, params)
        truth = plot_utils.extract_ground_truth(reference)
        predictions = {
            "HO-PINN": plot_utils.predict_fields(
                test_name, module, summary, checkpoint, reference, params, device
            )
        }
        features = reference_features(test_name, reference, params).numpy().astype(np.float32)
        nx0, nx1, nx2 = reference["nx0"], reference["nx1"], reference["nx2"]
        for method in ML_METHODS:
            estimator, _ = tune_classical(method, x_train, y_train, seed=42)
            pred = np.asarray(predict_classical(estimator, features, scalar=False))
            predictions[method] = {
                "pressure": np.squeeze(pred[:, 0].reshape(nx0, nx1, nx2)),
                "swat": np.squeeze(pred[:, 2].reshape(nx0, nx1, nx2)),
            }
        plot_utils.plot_fields_grid(
            test_name, n_min, params, truth, predictions,
            FIGURES_DIR / f"qualitative_{test_name}.png", 220
        )


def main() -> None:
    results = load_results()
    plot_data_efficiency(results)
    build_qualitative_figures()
    print(f"Assets written to {PAPER_DIR}")


if __name__ == "__main__":
    main()
