#!/usr/bin/env python3
"""Build confirmatory figures and LaTeX tables from robustness experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
CACHE = PAPER / ".cache"
for path in (GENERATED, FIGURES, CACHE / "matplotlib"):
    path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TESTS = ["test1", "test2", "test3", "test4"]
TEST_LABELS = {
    "test1": "Test 1: heat equation",
    "test2": "Test 2: low-variance flow",
    "test3": "Test 3: high-variance flow",
    "test4": "Test 4: 7D parametric flow",
}
METHODS = ["HO-PINN", "XGB", "SVM", "GP"]
COLORS = {
    "HO-PINN": "#c0392b",
    "XGB": "#4c78a8",
    "SVM": "#59a14f",
    "GP": "#b279a2",
}
MARKERS = {"HO-PINN": "*", "XGB": "o", "SVM": "s", "GP": "D"}
STRESS_SEEDS = [11, 22, 33]


def load_results() -> pd.DataFrame:
    frames = []
    for test in TESTS:
        path = GENERATED / f"robustness_{test}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run run_robustness_experiments.py")
        frame = pd.read_csv(path)
        frame["score"] = frame["R2"] if test == "test1" else frame["mean_R2"]
        frame["rmse_score"] = frame["RMSE"] if test == "test1" else frame["mean_RMSE"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    expected = {"test1": 20, "test2": 68, "test3": 20, "test4": 68}
    found = data.groupby("test_name").size().to_dict()
    if found != expected:
        raise ValueError(f"Incomplete robustness results: expected {expected}, found {found}")
    return data


def repeated(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        (data["experiment"] == "repeated")
        & (data["noise_level"] == 0.0)
        & (data["collocation_ratio"] == 1)
    ].copy()


def mean_sd(values: pd.Series, digits: int = 3) -> str:
    return f"{values.mean():.{digits}f} $\\pm$ {values.std(ddof=1):.{digits}f}"


def maybe_bold(text: str, is_best: bool) -> str:
    return f"\\textbf{{{text}}}" if is_best else text


def write_repeated_table(data: pd.DataFrame) -> None:
    clean = repeated(data)
    means = clean.groupby(["test_name", "method"])["score"].mean()
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Confirmatory held-out $R^2$ at the smallest label budget, reported as mean $\\pm$ sample standard deviation across five independent labelled-point selections and PINN initializations. Flow scores are macro-averaged over the three state variables. Conventional models are tuned using training data only. Higher is better.}",
        "\\label{tab:repeated_r2}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Test 1 ($N=50$) & Test 2 ($N=100$) & Test 3 ($N=50$) & Test 4 ($N=100$) \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        for test in TESTS:
            values = clean[(clean.test_name == test) & (clean.method == method)]["score"]
            best = means[test].max()
            cells.append(maybe_bold(mean_sd(values), np.isclose(values.mean(), best)))
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "repeated_r2_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_repeated_rmse_table(data: pd.DataFrame) -> None:
    clean = repeated(data)
    means = clean.groupby(["test_name", "method"])["rmse_score"].mean()
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Confirmatory held-out RMSE at the smallest label budget, reported as mean $\\pm$ sample standard deviation over the same five repetitions as \\Cref{tab:repeated_r2}. Flow scores are macro-averaged over the three state variables. Lower is better.}",
        "\\label{tab:repeated_rmse}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Test 1 ($N=50$) & Test 2 ($N=100$) & Test 3 ($N=50$) & Test 4 ($N=100$) \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        for test in TESTS:
            values = clean[(clean.test_name == test) & (clean.method == method)]["rmse_score"]
            best = means[test].min()
            cells.append(maybe_bold(mean_sd(values), np.isclose(values.mean(), best)))
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "repeated_rmse_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_repeated_per_target_table(data: pd.DataFrame) -> None:
    clean = repeated(data)
    columns = ["Pressure_R2", "Soil_R2", "Swat_R2"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Confirmatory per-output $R^2$ for the flow benchmarks (mean $\\pm$ sample standard deviation over five repetitions). Higher is better within each benchmark and output.}",
        "\\label{tab:repeated_per_target}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Benchmark & Method & Pressure & Oil saturation & Water saturation \\\\",
        "\\midrule",
    ]
    for test in ("test2", "test3", "test4"):
        test_rows = clean[clean.test_name == test]
        best = test_rows.groupby("method")[columns].mean().max(axis=0)
        for index, method in enumerate(METHODS):
            row = test_rows[test_rows.method == method]
            cells = [
                maybe_bold(mean_sd(row[column]), np.isclose(row[column].mean(), best[column]))
                for column in columns
            ]
            label = f"Test {test[-1]}" if index == 0 else ""
            lines.append(f"{label} & {method} & " + " & ".join(cells) + " \\\\")
        if test != "test4":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "repeated_per_target_r2_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_repeated_runtime_table(data: pd.DataFrame) -> None:
    path = GENERATED / "isolated_timing.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run benchmark_isolated_timing.py")
    timing = pd.read_csv(path)
    if len(timing) != 16:
        raise ValueError(f"Incomplete isolated timings: {len(timing)} rows")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Representative isolated model fitting time in seconds at the smallest label budget. One method is timed at a time; training-only hyperparameter search is included for the conventional models and reference-data generation is excluded for every method.}",
        "\\label{tab:repeated_runtime}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Test 1 & Test 2 & Test 3 & Test 4 \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        cells = []
        for test in TESTS:
            value = timing[(timing.test_name == test) & (timing.method == method)]["train_time_sec"].iloc[0]
            cells.append(f"{value:.3g}")
        lines.append(f"{method} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "repeated_runtime_table.tex").write_text("\n".join(lines), encoding="utf-8")


def noise_subset(data: pd.DataFrame) -> pd.DataFrame:
    clean = data[
        (data.experiment == "repeated")
        & data.seed.isin(STRESS_SEEDS)
        & data.test_name.isin(["test2", "test4"])
    ].copy()
    noisy = data[data.experiment == "noise"].copy()
    return pd.concat([clean, noisy], ignore_index=True)


def write_noise_table(data: pd.DataFrame) -> None:
    subset = noise_subset(data)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Paired label-noise sensitivity on the two representative flow problems. Entries are mean $R^2$ $\\pm$ sample standard deviation over three shared selections and initializations. Noise standard deviation is the stated fraction of each target's training-set standard deviation.}",
        "\\label{tab:noise}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Benchmark & Noise & HO-PINN & XGB & SVR & GP \\\\",
        "\\midrule",
    ]
    for test in ("test2", "test4"):
        for index, level in enumerate((0.0, 0.01, 0.05)):
            cells = []
            row = subset[(subset.test_name == test) & np.isclose(subset.noise_level, level)]
            group_means = row.groupby("method")["score"].mean()
            for method in METHODS:
                values = row[row.method == method]["score"]
                cells.append(maybe_bold(mean_sd(values), np.isclose(values.mean(), group_means.max())))
            label = f"Test {test[-1]}" if index == 0 else ""
            lines.append(f"{label} & {100 * level:.0f}\\% & " + " & ".join(cells) + " \\\\")
        if test == "test2":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "noise_r2_table.tex").write_text("\n".join(lines), encoding="utf-8")


def collocation_subset(data: pd.DataFrame) -> pd.DataFrame:
    clean = data[
        (data.experiment == "repeated")
        & data.seed.isin(STRESS_SEEDS)
        & data.test_name.isin(["test2", "test4"])
        & (data.method == "HO-PINN")
    ].copy()
    extra = data[(data.experiment == "collocation") & (data.method == "HO-PINN")].copy()
    return pd.concat([clean, extra], ignore_index=True)


def write_collocation_table(data: pd.DataFrame) -> None:
    subset = collocation_subset(data)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{HO-PINN sensitivity to the number of unlabelled PDE-residual points. Label counts remain fixed. Values are mean $\\pm$ sample standard deviation over three paired repetitions.}",
        "\\label{tab:collocation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrcc}",
        "\\toprule",
        "Test & $N$ & $N_r$ & Mean $R^2$ & Fit time (s) \\\\",
        "\\midrule",
    ]
    for test in ("test2", "test4"):
        for ratio in (1, 2, 5):
            row = subset[(subset.test_name == test) & (subset.collocation_ratio == ratio)]
            n = int(row.n_train.iloc[0])
            lines.append(
                f"{test[-1]} & {n} & {n * ratio} & {mean_sd(row.score)} & "
                f"{mean_sd(row.train_time_sec, digits=1)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    (GENERATED / "collocation_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_parameter_holdout_table(data: pd.DataFrame) -> None:
    path = GENERATED / "parameter_holdout_test4.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_test4_parameter_holdout.py")
    heldout = pd.read_csv(path)
    if len(heldout) != 12:
        raise ValueError(f"Incomplete Test 4 parameter holdout: {len(heldout)} rows")
    heldout["score"] = heldout["mean_R2"]
    pointwise = data[
        (data.test_name == "test4")
        & (data.experiment == "repeated")
        & data.seed.isin(STRESS_SEEDS)
    ].copy()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test 4 sensitivity to the validation split. The point-wise test mixes coordinates from all parameter groups; the group holdout contains 20 in-range physical-parameter combinations absent entirely from training. Entries are mean $R^2$ $\\pm$ sample standard deviation over three paired seeds.}",
        "\\label{tab:parameter_holdout}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Test split & HO-PINN & XGB & SVR & GP \\\\",
        "\\midrule",
    ]
    for label, frame in (("Point-wise", pointwise), ("Unseen parameter groups", heldout)):
        group_means = frame.groupby("method")["score"].mean()
        cells = []
        for method in METHODS:
            values = frame[frame.method == method]["score"]
            cells.append(maybe_bold(mean_sd(values), np.isclose(values.mean(), group_means.max())))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    (GENERATED / "parameter_holdout_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_sampling_table(data: pd.DataFrame) -> None:
    path = GENERATED / "sampling_sensitivity_test3.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run run_test3_sampling_sensitivity.py")
    front = pd.read_csv(path)
    if len(front) != 12:
        raise ValueError(f"Incomplete Test 3 sampling sensitivity: {len(front)} rows")
    front["score"] = front["mean_R2"]
    uniform = data[
        (data.test_name == "test3")
        & (data.experiment == "repeated")
        & data.seed.isin(STRESS_SEEDS)
    ].copy()
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test 3 sensitivity to labelled-point placement at $N=50$. Uniform sampling is compared with a paired design containing 25 active-front points ($s_w>0.01$) and 25 background points. Entries are mean $R^2$ $\\pm$ sample standard deviation over three seeds.}",
        "\\label{tab:sampling}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Label sampling & HO-PINN & XGB & SVR & GP \\\\",
        "\\midrule",
    ]
    for label, frame in (("Uniform", uniform), ("Front-aware", front)):
        group_means = frame.groupby("method")["score"].mean()
        cells = []
        for method in METHODS:
            values = frame[frame.method == method]["score"]
            cells.append(maybe_bold(mean_sd(values), np.isclose(values.mean(), group_means.max())))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    (GENERATED / "sampling_sensitivity_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_constraint_table() -> None:
    path = GENERATED / "physical_constraint_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run evaluate_physical_constraints.py")
    data = pd.read_csv(path)
    if len(data) != 60:
        raise ValueError(f"Incomplete physical-constraint results: {len(data)} rows")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Physical admissibility on the clean flow test sets (mean $\\pm$ sample standard deviation over five repetitions). ``Bounds'' is the percentage of predictions with either saturation outside $[-10^{-3},1+10^{-3}]$; closure is $|s_o+s_w-1|$. Lower is better.}",
        "\\label{tab:constraints}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Benchmark & Method & Bounds violation (\\%) & Closure MAE \\\\",
        "\\midrule",
    ]
    for test in ("test2", "test3", "test4"):
        subset = data[data.test_name == test]
        for index, method in enumerate(METHODS):
            row = subset[subset.method == method]
            bounds = row.saturation_violation_rate * 100.0
            label = f"Test {test[-1]}" if index == 0 else ""
            lines.append(
                f"{label} & {method} & {mean_sd(bounds, digits=2)} & "
                f"{mean_sd(row.closure_mae, digits=4)} \\\\")
        if test != "test4":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    (GENERATED / "physical_constraints_table.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_repetitions(data: pd.DataFrame) -> None:
    clean = repeated(data)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    rng = np.random.default_rng(20260727)
    for ax, test in zip(axes.flat, TESTS):
        values = [clean[(clean.test_name == test) & (clean.method == m)].score.to_numpy() for m in METHODS]
        bp = ax.boxplot(values, positions=np.arange(4), widths=0.55, patch_artist=True, showfliers=False)
        for patch, method in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[method]); patch.set_alpha(0.35)
        for index, (method, vals) in enumerate(zip(METHODS, values)):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(index + jitter, vals, s=34, color=COLORS[method], edgecolor="black", linewidth=0.35, zorder=3)
        ax.set_xticks(np.arange(4), METHODS, rotation=18, ha="right")
        ax.set_title(TEST_LABELS[test])
        ax.set_ylabel("$R^2$" if test == "test1" else "Mean $R^2$")
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES / "repeated_small_data_r2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_noise(data: pd.DataFrame) -> None:
    subset = noise_subset(data)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for ax, test in zip(axes, ("test2", "test4")):
        for method in METHODS:
            means, stds = [], []
            for level in (0.0, 0.01, 0.05):
                vals = subset[(subset.test_name == test) & (subset.method == method) & np.isclose(subset.noise_level, level)].score
                means.append(vals.mean()); stds.append(vals.std(ddof=1))
            ax.errorbar([0, 1, 5], means, yerr=stds, label=method, color=COLORS[method], marker=MARKERS[method], capsize=3, linewidth=1.8)
        ax.set_title(TEST_LABELS[test])
        ax.set_xlabel("Label-noise level (% of target s.d.)")
        ax.set_ylabel("Mean $R^2$")
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    fig.savefig(FIGURES / "noise_sensitivity_r2.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_collocation(data: pd.DataFrame) -> None:
    subset = collocation_subset(data)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for ax, test in zip(axes, ("test2", "test4")):
        scores, score_sd, times, time_sd = [], [], [], []
        for ratio in (1, 2, 5):
            row = subset[(subset.test_name == test) & (subset.collocation_ratio == ratio)]
            scores.append(row.score.mean()); score_sd.append(row.score.std(ddof=1))
            times.append(row.train_time_sec.mean()); time_sd.append(row.train_time_sec.std(ddof=1))
        score_line = ax.errorbar([1, 2, 5], scores, yerr=score_sd, marker="o", capsize=3, color=COLORS["HO-PINN"], label="Mean $R^2$")
        ax.set_title(TEST_LABELS[test])
        ax.set_xlabel("Residual points / labelled points")
        ax.set_ylabel("Mean $R^2$", color=COLORS["HO-PINN"])
        ax.tick_params(axis="y", labelcolor=COLORS["HO-PINN"])
        ax.grid(True, alpha=0.25)
        twin = ax.twinx()
        time_line = twin.errorbar([1, 2, 5], times, yerr=time_sd, marker="s", linestyle="--", capsize=3, color="#555555", label="Fit time")
        twin.set_ylabel("Fit time (s)", color="#555555")
        twin.tick_params(axis="y", labelcolor="#555555")
        ax.legend([score_line, time_line], ["Mean $R^2$", "Fit time"], frameon=False, fontsize=9, loc="best")
    fig.savefig(FIGURES / "collocation_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_cost_quality(data: pd.DataFrame) -> None:
    clean = repeated(data)
    timing = pd.read_csv(GENERATED / "isolated_timing.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    for ax, test in zip(axes.flat, TESTS):
        for method in METHODS:
            row = clean[(clean.test_name == test) & (clean.method == method)]
            fit_time = timing[(timing.test_name == test) & (timing.method == method)].train_time_sec.iloc[0]
            ax.errorbar(fit_time, row.score.mean(), yerr=row.score.std(ddof=1), fmt=MARKERS[method], markersize=10 if method == "HO-PINN" else 6, capsize=3, color=COLORS[method], label=method)
            ax.annotate(method, (fit_time, row.score.mean()), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_title(TEST_LABELS[test])
        ax.set_xlabel("Fitting time including tuning, s (log scale)")
        ax.set_ylabel("$R^2$" if test == "test1" else "Mean $R^2$")
        ax.grid(True, alpha=0.25)
    fig.savefig(FIGURES / "repeated_cost_quality.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(data: pd.DataFrame) -> None:
    clean = repeated(data)
    summary = {}
    for test in TESTS:
        summary[test] = {}
        for method in METHODS:
            row = clean[(clean.test_name == test) & (clean.method == method)]
            summary[test][method] = {
                "r2_mean": float(row.score.mean()),
                "r2_sd": float(row.score.std(ddof=1)),
                "fit_time_mean_sec": float(row.train_time_sec.mean()),
            }
    (GENERATED / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    data = load_results()
    write_repeated_table(data)
    write_repeated_rmse_table(data)
    write_repeated_per_target_table(data)
    write_repeated_runtime_table(data)
    write_noise_table(data)
    write_collocation_table(data)
    write_parameter_holdout_table(data)
    write_sampling_table(data)
    write_constraint_table()
    plot_repetitions(data)
    plot_noise(data)
    plot_collocation(data)
    plot_cost_quality(data)
    write_summary(data)
    print(f"Robustness assets written to {PAPER}")


if __name__ == "__main__":
    main()
