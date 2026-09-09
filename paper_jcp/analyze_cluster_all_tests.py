#!/usr/bin/env python3
"""Aggregate and visualise the final 10-seed cluster runs for Tests 1--4."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
OUTPUT = PAPER / "generated" / "cluster_all_tests_analysis"
INPUTS = {
    "test1": PAPER / "generated" / "test1_corrected_candidate" / "unified_test1.csv",
    "test2": PAPER / "generated" / "corrected_test2_final" / "unified_test2.csv",
    "test3": PAPER / "generated" / "corrected_test3_final" / "unified_test3.csv",
    "test4": PAPER / "generated" / "corrected_test4_final" / "unified_test4.csv",
}
TEST_LABELS = {
    "test1": "Test 1: heat equation",
    "test2": "Test 2: homogeneous flow",
    "test3": "Test 3: heterogeneous flow",
    "test4": "Test 4: parametric flow",
}
METHODS = ("H-PINN", "HO-PINN")
REGIONS = ("interpolation", "extrapolation")
BUDGETS = (20, 50, 100, 200, 500, 1000)
COLORS = {"H-PINN": "#D55E00", "HO-PINN": "#0072B2"}
MARKERS = {"H-PINN": "o", "HO-PINN": "s"}


def load_all() -> pd.DataFrame:
    frames = []
    for test, path in INPUTS.items():
        frame = pd.read_csv(path)
        frame["test_name"] = test
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True, sort=False)
    expected = {
        (test, region, budget, method, seed)
        for test in INPUTS
        for region in REGIONS
        for budget in BUDGETS
        for method in METHODS
        for seed in range(101, 111)
    }
    actual = set(
        data[["test_name", "evaluation_region", "n_train", "method", "seed"]]
        .itertuples(index=False, name=None)
    )
    missing = expected - actual
    extra = actual - expected
    duplicates = data.duplicated(
        ["test_name", "evaluation_region", "n_train", "method", "seed"]
    ).sum()
    if missing or extra or duplicates:
        raise RuntimeError(
            f"Coverage failure: missing={len(missing)}, extra={len(extra)}, "
            f"duplicates={duplicates}"
        )
    return data


def metric_columns(data: pd.DataFrame) -> list[str]:
    exact = {"MSE", "RMSE", "MAE", "R2", "relative_L2", "NRMSE_std"}
    suffixes = ("_MSE", "_RMSE", "_MAE", "_R2", "_relative_L2", "_NRMSE_std")
    special = {"saturation_bounds_violation", "closure_MAE"}
    return [
        column
        for column in data.columns
        if column in exact or column.endswith(suffixes) or column in special
    ]


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["test_name", "evaluation_region", "n_train", "method"]
    for key, subset in data.groupby(keys, sort=False):
        for metric in metric_columns(data):
            values = subset[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "test_name": key[0],
                    "evaluation_region": key[1],
                    "n_train": int(key[2]),
                    "method": key[3],
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_seeds": int(values.size),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["test_name", "evaluation_region", "n_train", "method", "metric"]
    )


def primary_metric(test: str, error: bool = False) -> str:
    if test == "test1":
        return "MSE" if error else "R2"
    return "Swat_MSE" if error else "Swat_R2"


def primary_axis_label(test: str, region: str, error: bool = False) -> str:
    target = "temperature" if test == "test1" else "water saturation"
    measure = "MSE" if error else "R²"
    return f"{region.capitalize()} {target} {measure}"


def format_value(mean: float, std: float, scientific: bool = False) -> str:
    if scientific:
        return f"{mean:.3e} ± {std:.3e}"
    return f"{mean:.5f} ± {std:.5f}"


def compact_table(summary: pd.DataFrame, test: str, metric: str) -> list[str]:
    subset = summary[(summary.test_name == test) & (summary.metric == metric)]
    lines = [
        f"### {TEST_LABELS[test]}",
        "",
        "| N | H-PINN interp. | HO-PINN interp. | H-PINN extrap. | HO-PINN extrap. |",
        "|---:|---:|---:|---:|---:|",
    ]
    scientific = "R2" not in metric
    higher = metric.endswith("R2") or metric == "R2"
    for budget in BUDGETS:
        cells: list[str] = []
        for region in REGIONS:
            pair = subset[
                (subset.n_train == budget) & (subset.evaluation_region == region)
            ].set_index("method")
            means = pair["mean"]
            optimum = means.max() if higher else means.min()
            for method in METHODS:
                row = pair.loc[method]
                value = format_value(row["mean"], row["std"], scientific)
                if np.isclose(row["mean"], optimum, rtol=1e-12, atol=1e-15):
                    value = f"**{value}**"
                cells.append(value)
        lines.append(f"| {budget} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def detailed_table(summary: pd.DataFrame, test: str, metric: str) -> list[str]:
    subset = summary[(summary.test_name == test) & (summary.metric == metric)]
    if subset.empty:
        return []
    lines = [f"#### `{metric}`", ""]
    lines.extend(compact_table(summary, test, metric)[2:])
    return lines


def write_markdown(summary: pd.DataFrame) -> Path:
    lines = [
        "# Cluster results for Tests 1–4",
        "",
        "Values are arithmetic mean ± sample standard deviation across ten independent "
        "training seeds (101–110). Bold marks the better mean between H-PINN and "
        "HO-PINN for the same test, region, and label budget.",
        "",
        "The primary response is temperature for Test 1 and water saturation for "
        "Tests 2–4. The source files contain only the two PINN variants; classical "
        "baselines are intentionally not mixed into this ten-seed cluster summary.",
        "",
        "## Primary R²",
        "",
    ]
    for test in INPUTS:
        lines.extend(compact_table(summary, test, primary_metric(test)))
    lines.extend(["## Primary MSE", ""])
    for test in INPUTS:
        lines.extend(compact_table(summary, test, primary_metric(test, error=True)))
    lines.extend(["## Pressure R²", ""])
    for test in ("test2", "test3", "test4"):
        lines.extend(compact_table(summary, test, "Pressure_R2"))
    lines.extend(["## Pressure MSE", ""])
    for test in ("test2", "test3", "test4"):
        lines.extend(compact_table(summary, test, "Pressure_MSE"))
    lines.extend(
        [
            "## Complete metric appendix",
            "",
            "The long-form CSV accompanying this report contains mean, sample standard "
            "deviation, minimum, maximum, and seed count for every recorded metric. "
            "The tables below retain all scalar Test 1 metrics and the principal "
            "pressure/saturation diagnostics for Tests 2–4.",
            "",
        ]
    )
    selected = {
        "test1": ["RMSE", "MAE", "relative_L2", "NRMSE_std"],
        "test2": ["Swat_RMSE", "Swat_MAE", "Pressure_RMSE", "Pressure_MAE", "mean_R2", "closure_MAE", "saturation_bounds_violation"],
        "test3": ["Swat_RMSE", "Swat_MAE", "Pressure_RMSE", "Pressure_MAE", "mean_R2", "closure_MAE", "saturation_bounds_violation"],
        "test4": ["Swat_RMSE", "Swat_MAE", "Pressure_RMSE", "Pressure_MAE", "mean_R2", "closure_MAE", "saturation_bounds_violation"],
    }
    for test, metrics in selected.items():
        lines.extend([f"### {TEST_LABELS[test]}", ""])
        for metric in metrics:
            lines.extend(detailed_table(summary, test, metric))
    path = OUTPUT / "cluster_all_tests_metrics.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def setup_axis(ax: plt.Axes, test: str, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_title(TEST_LABELS[test], fontsize=11)
    ax.set_xlabel("Labelled points, N")
    ax.set_ylabel(ylabel)


def plot_primary(summary: pd.DataFrame, error: bool = False) -> Path:
    figure, axes = plt.subplots(2, 4, figsize=(15.8, 7.2), constrained_layout=True)
    for column, test in enumerate(INPUTS):
        metric = primary_metric(test, error=error)
        for row_index, region in enumerate(REGIONS):
            ax = axes[row_index, column]
            subset = summary[
                (summary.test_name == test)
                & (summary.evaluation_region == region)
                & (summary.metric == metric)
            ]
            for method in METHODS:
                values = subset[subset.method == method].set_index("n_train").reindex(BUDGETS)
                ax.errorbar(
                    BUDGETS,
                    values["mean"],
                    yerr=values["std"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.8,
                    markersize=4.5,
                    capsize=2.5,
                    label=method,
                )
            if error:
                ax.set_yscale("log")
            setup_axis(ax, test, primary_axis_label(test, region, error=error))
            if row_index == 0 and column == 0:
                ax.legend(frameon=False, fontsize=9)
    filename = "primary_mse_all_tests" if error else "primary_r2_all_tests"
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"{filename}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT / f"{filename}.png"


def plot_pressure(summary: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(12.2, 7.0), constrained_layout=True)
    for column, test in enumerate(("test2", "test3", "test4")):
        for row_index, region in enumerate(REGIONS):
            ax = axes[row_index, column]
            subset = summary[
                (summary.test_name == test)
                & (summary.evaluation_region == region)
                & (summary.metric == "Pressure_R2")
            ]
            for method in METHODS:
                values = subset[subset.method == method].set_index("n_train").reindex(BUDGETS)
                ax.errorbar(
                    BUDGETS,
                    values["mean"],
                    yerr=values["std"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.8,
                    capsize=2.5,
                    label=method,
                )
            setup_axis(ax, test, f"{region.capitalize()} pressure R²")
            if row_index == 0 and column == 0:
                ax.legend(frameon=False, fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"pressure_r2_flow_tests.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT / "pressure_r2_flow_tests.png"


def plot_extrapolation_gap(data: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 4, figsize=(15.8, 3.7), constrained_layout=True)
    for ax, test in zip(axes, INPUTS):
        metric = primary_metric(test)
        subset = data[data.test_name == test]
        pivot = subset.pivot(
            index=["seed", "n_train", "method"],
            columns="evaluation_region",
            values=metric,
        ).reset_index()
        pivot["gap"] = pivot["interpolation"] - pivot["extrapolation"]
        for method in METHODS:
            values = (
                pivot[pivot.method == method]
                .groupby("n_train")["gap"]
                .agg(["mean", "std"])
                .reindex(BUDGETS)
            )
            ax.errorbar(
                BUDGETS,
                values["mean"],
                yerr=values["std"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                capsize=2.5,
                label=method,
            )
        ax.axhline(0, color="#333333", linewidth=0.9)
        setup_axis(ax, test, "R² interpolation − extrapolation")
        if test == "test1":
            ax.legend(frameon=False, fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"extrapolation_gap_all_tests.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT / "extrapolation_gap_all_tests.png"


def plot_seed_stability(data: pd.DataFrame) -> Path:
    selected = data[data.n_train.isin((20, 50))].copy()
    records = []
    for test in INPUTS:
        metric = primary_metric(test)
        block = selected[selected.test_name == test]
        for row in block.itertuples(index=False):
            records.append(
                {
                    "test_name": test,
                    "evaluation_region": row.evaluation_region,
                    "n_train": row.n_train,
                    "method": row.method,
                    "value": getattr(row, metric),
                }
            )
    tidy = pd.DataFrame(records)
    figure, axes = plt.subplots(2, 4, figsize=(15.8, 7.0), constrained_layout=True)
    rng = np.random.default_rng(2026)
    positions = {("H-PINN", 20): 0.85, ("HO-PINN", 20): 1.15, ("H-PINN", 50): 1.85, ("HO-PINN", 50): 2.15}
    for column, test in enumerate(INPUTS):
        for row_index, region in enumerate(REGIONS):
            ax = axes[row_index, column]
            block = tidy[(tidy.test_name == test) & (tidy.evaluation_region == region)]
            for (method, budget), position in positions.items():
                values = block[(block.method == method) & (block.n_train == budget)].value.to_numpy()
                box = ax.boxplot(
                    values,
                    positions=[position],
                    widths=0.23,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                    whiskerprops={"color": COLORS[method]},
                    capprops={"color": COLORS[method]},
                )
                box["boxes"][0].set_facecolor(COLORS[method])
                box["boxes"][0].set_alpha(0.34)
                jitter = rng.normal(0, 0.025, size=len(values))
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=18,
                    color=COLORS[method],
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.set_xticks((1, 2), ("N=20", "N=50"))
            ax.grid(True, axis="y", alpha=0.22)
            ax.set_title(TEST_LABELS[test], fontsize=11)
            target = "temperature" if test == "test1" else "water saturation"
            ax.set_ylabel(f"{region.capitalize()} {target} R²")
    handles = [
        plt.Line2D([0], [0], marker=MARKERS[m], color=COLORS[m], linewidth=0, label=m)
        for m in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT / f"seed_stability_n20_n50.{extension}", dpi=300, bbox_inches="tight")
    plt.close(figure)
    return OUTPUT / "seed_stability_n20_n50.png"


def write_latex(summary: pd.DataFrame) -> Path:
    lines = [
        "% Auto-generated by analyze_cluster_all_tests.py",
        "% Each value is mean $\\pm$ sample standard deviation over seeds 101--110.",
    ]
    for test in INPUTS:
        metric = primary_metric(test)
        subset = summary[(summary.test_name == test) & (summary.metric == metric)]
        lines.extend(
            [
                "\\begin{table*}[t]",
                "\\centering",
                "\\small",
                f"\\caption{{Cluster {TEST_LABELS[test]}: {metric} over ten seeds.}}",
                f"\\label{{tab:cluster-{test}-{metric.lower().replace('_', '-')}}}",
                "\\begin{tabular}{rcccc}",
                "\\toprule",
                "$N$ & H-PINN interp. & HO-PINN interp. & H-PINN extrap. & HO-PINN extrap. \\\\",
                "\\midrule",
            ]
        )
        for budget in BUDGETS:
            cells = []
            for region in REGIONS:
                pair = subset[
                    (subset.n_train == budget) & (subset.evaluation_region == region)
                ].set_index("method")
                optimum = pair["mean"].max()
                for method in METHODS:
                    row = pair.loc[method]
                    value = f"{row['mean']:.4f} $\\pm$ {row['std']:.4f}"
                    if np.isclose(row["mean"], optimum, rtol=1e-12, atol=1e-15):
                        value = f"\\textbf{{{value}}}"
                    cells.append(value)
            lines.append(f"{budget} & " + " & ".join(cells) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path = OUTPUT / "cluster_primary_r2_tables.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_focused_r2_mse(summary: pd.DataFrame) -> tuple[Path, Path]:
    """Write the compact pressure/water-saturation tables requested for review."""
    records: list[dict[str, object]] = []
    for test in INPUTS:
        targets = (("temperature", "R2", "MSE"),) if test == "test1" else (
            ("pressure", "Pressure_R2", "Pressure_MSE"),
            ("water saturation", "Swat_R2", "Swat_MSE"),
        )
        for region in REGIONS:
            for budget in BUDGETS:
                for method in METHODS:
                    for target, r2_metric, mse_metric in targets:
                        r2 = summary[
                            (summary.test_name == test)
                            & (summary.evaluation_region == region)
                            & (summary.n_train == budget)
                            & (summary.method == method)
                            & (summary.metric == r2_metric)
                        ].iloc[0]
                        mse = summary[
                            (summary.test_name == test)
                            & (summary.evaluation_region == region)
                            & (summary.n_train == budget)
                            & (summary.method == method)
                            & (summary.metric == mse_metric)
                        ].iloc[0]
                        records.append(
                            {
                                "test_name": test,
                                "evaluation_region": region,
                                "n_train": budget,
                                "method": method,
                                "target": target,
                                "R2_mean": r2["mean"],
                                "R2_std": r2["std"],
                                "MSE_mean": mse["mean"],
                                "MSE_std": mse["std"],
                            }
                        )
    focused = pd.DataFrame(records)
    csv_path = OUTPUT / "cluster_pressure_swat_r2_mse.csv"
    focused.to_csv(csv_path, index=False)

    lines = [
        "# Cluster R² and MSE: pressure and water saturation",
        "",
        "Every value is the arithmetic mean ± sample standard deviation over ten "
        "independent training seeds (101–110). For R², larger is better; for MSE, "
        "smaller is better. Bold identifies the better of H-PINN and HO-PINN at "
        "the same label budget.",
        "",
        "Test 1 contains temperature only. Tests 2–4 contain pressure and water "
        "saturation; water saturation is the primary difficult response.",
        "",
    ]

    for test in INPUTS:
        lines.extend([f"## {TEST_LABELS[test]}", ""])
        targets = ("temperature",) if test == "test1" else ("pressure", "water saturation")
        for region in REGIONS:
            lines.extend([f"### {region.capitalize()}", ""])
            if test == "test1":
                lines.extend(
                    [
                        "| N | Method | Temperature R² | Temperature MSE |",
                        "|---:|---|---:|---:|",
                    ]
                )
            else:
                lines.extend(
                    [
                        "| N | Method | Pressure R² | Pressure MSE | Water saturation R² | Water saturation MSE |",
                        "|---:|---|---:|---:|---:|---:|",
                    ]
                )
            for budget in BUDGETS:
                block = focused[
                    (focused.test_name == test)
                    & (focused.evaluation_region == region)
                    & (focused.n_train == budget)
                ]
                best: dict[tuple[str, str], float] = {}
                for target in targets:
                    target_rows = block[block.target == target]
                    best[(target, "r2")] = target_rows.R2_mean.max()
                    best[(target, "mse")] = target_rows.MSE_mean.min()
                for method in METHODS:
                    method_rows = block[block.method == method].set_index("target")
                    cells: list[str] = []
                    for target in targets:
                        row = method_rows.loc[target]
                        r2_text = format_value(row.R2_mean, row.R2_std)
                        mse_text = format_value(row.MSE_mean, row.MSE_std, scientific=True)
                        if np.isclose(row.R2_mean, best[(target, "r2")], rtol=1e-12, atol=1e-15):
                            r2_text = f"**{r2_text}**"
                        if np.isclose(row.MSE_mean, best[(target, "mse")], rtol=1e-12, atol=1e-15):
                            mse_text = f"**{mse_text}**"
                        cells.extend((r2_text, mse_text))
                    lines.append(f"| {budget} | {method} | " + " | ".join(cells) + " |")
            lines.append("")

    md_path = OUTPUT / "cluster_pressure_swat_r2_mse.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path, csv_path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    summary = aggregate(data)
    summary_path = OUTPUT / "cluster_all_tests_summary_long.csv"
    summary.to_csv(summary_path, index=False)
    report_path = write_markdown(summary)
    latex_path = write_latex(summary)
    focused_md, focused_csv = write_focused_r2_mse(summary)
    figures = [
        plot_primary(summary),
        plot_primary(summary, error=True),
        plot_pressure(summary),
        plot_extrapolation_gap(data),
        plot_seed_stability(data),
    ]
    print(f"Rows checked: {len(data)}")
    print(f"Summary rows: {len(summary)}")
    print(summary_path)
    print(report_path)
    print(latex_path)
    print(focused_md)
    print(focused_csv)
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
