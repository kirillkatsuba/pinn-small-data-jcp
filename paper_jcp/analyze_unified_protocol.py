#!/usr/bin/env python3
"""Aggregate and visualise completed unified-protocol experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
RESULTS_DIR = PAPER / "generated" / "unified_protocol"
FIGURES_DIR = PAPER / "figures" / "unified_protocol"
METHOD_ORDER = ("H-PINN", "HO-PINN", "XGB", "SVM", "GP")
COLORS = {
    "H-PINN": "#e67e22",
    "HO-PINN": "#c0392b",
    "XGB": "#4c78a8",
    "SVM": "#59a14f",
    "GP": "#b279a2",
}
TEST_TITLES = {
    "test1": "Test 1: heat",
    "test2": "Test 2: low-variance flow",
    "test3": "Test 3: high-variance flow",
    "test4": "Test 4: parametric flow",
}


def load_results() -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("unified_test*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No unified result files in {RESULTS_DIR}; run run_unified_protocol.py first"
        )
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "test_name",
        "n_train",
        "method",
        "config_id",
        "evaluation_region",
    ]
    numeric = [
        column
        for column in results.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "n_train"}
    ]
    grouped = results.groupby(keys, dropna=False)[numeric].agg(["mean", "std", "count"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def metric_for(test_name: str, target: str) -> str:
    if target == "macro":
        return "R2" if test_name == "test1" else "mean_R2"
    if target == "water":
        return "R2" if test_name == "test1" else "Swat_R2"
    raise ValueError(target)


def plot_curves(results: pd.DataFrame, target: str) -> None:
    tests = ("test1", "test2", "test3", "test4")
    regions = ("interpolation", "extrapolation")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
    for row, region in enumerate(regions):
        for column, test_name in enumerate(tests):
            ax = axes[row, column]
            metric = metric_for(test_name, target)
            subset = results[
                (results["test_name"] == test_name)
                & (results["evaluation_region"] == region)
            ]
            for method in METHOD_ORDER:
                method_rows = subset[subset["method"] == method]
                if method_rows.empty:
                    continue
                grouped = method_rows.groupby("n_train")[metric].agg(["mean", "std"])
                ax.errorbar(
                    grouped.index,
                    grouped["mean"],
                    yerr=grouped["std"].fillna(0.0),
                    marker="o",
                    linewidth=1.8,
                    capsize=2,
                    label=method,
                    color=COLORS[method],
                )
            ax.set_xscale("log")
            ax.set_xticks(sorted(subset["n_train"].unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.grid(alpha=0.25)
            ax.set_title(TEST_TITLES[test_name])
            ax.set_xlabel("Labelled points, N")
            ax.set_ylabel(f"{region.capitalize()} {metric}")
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=8)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / f"{target}_r2_interpolation_extrapolation.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_extrapolation_gap(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    for ax, test_name in zip(axes, TEST_TITLES):
        metric = metric_for(test_name, "water")
        subset = results[results["test_name"] == test_name]
        pivot = subset.pivot_table(
            index=["seed", "n_train", "method"],
            columns="evaluation_region",
            values=metric,
            aggfunc="first",
        ).dropna(subset=["interpolation", "extrapolation"])
        pivot["gap"] = pivot["interpolation"] - pivot["extrapolation"]
        pivot = pivot.reset_index()
        for method in METHOD_ORDER:
            method_rows = pivot[pivot["method"] == method]
            if method_rows.empty:
                continue
            grouped = method_rows.groupby("n_train")["gap"].agg(["mean", "std"])
            ax.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"].fillna(0.0),
                marker="o",
                label=method,
                color=COLORS[method],
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_title(TEST_TITLES[test_name])
        ax.set_xlabel("Labelled points, N")
        ax.set_ylabel("Interpolation R² − extrapolation R²")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES_DIR / "extrapolation_gap.png", dpi=240, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    global RESULTS_DIR, FIGURES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="unified_protocol")
    parser.add_argument(
        "--config-id",
        help="Optionally keep only one PINN configuration id; classical rows are retained",
    )
    args = parser.parse_args()
    RESULTS_DIR = PAPER / "generated" / args.experiment_name
    FIGURES_DIR = PAPER / "figures" / args.experiment_name
    results = load_results()
    if args.config_id:
        results = results[
            (results["config_id"] == args.config_id)
            | (results["method"].isin(("XGB", "SVM", "GP")))
        ]
    summary = aggregate(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "unified_summary.csv", index=False)
    plot_curves(results, "macro")
    plot_curves(results, "water")
    plot_extrapolation_gap(results)
    print(RESULTS_DIR / "unified_summary.csv")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
