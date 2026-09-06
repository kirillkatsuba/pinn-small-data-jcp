#!/usr/bin/env python3
"""Build static figures and a compact report for the zero-label experiment."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER = Path(__file__).resolve().parent

from paper_jcp.unified_protocol import (  # noqa: E402
    ModelConfig,
    build_model,
    build_unified_bundle,
)


COLORS = {"H-PINN": "#C43C39", "HO-PINN": "#2C6EAA"}
MARKERS = {"H-PINN": "o", "HO-PINN": "s"}
REGION_LABELS = {"interpolation": "Early-time region", "extrapolation": "Late-time region"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def metric_for(test_name: str) -> str:
    return "R2" if test_name == "test1" else "Swat_R2"


def plot_main_summary(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))
    budgets = [0, 20]
    offsets = {"H-PINN": -1.1, "HO-PINN": 1.1}
    rng = np.random.default_rng(42)

    for row, test_name in enumerate(("test1", "test2")):
        metric = metric_for(test_name)
        for col, region in enumerate(("interpolation", "extrapolation")):
            ax = axes[row, col]
            subset = data[
                (data.test_name == test_name)
                & (data.evaluation_region == region)
            ]
            for method in ("H-PINN", "HO-PINN"):
                means, stds = [], []
                for budget in budgets:
                    values = subset[
                        (subset.n_train == budget) & (subset.method == method)
                    ][metric].to_numpy()
                    means.append(float(np.mean(values)))
                    stds.append(float(np.std(values, ddof=1)))
                    jitter = rng.uniform(-0.35, 0.35, size=len(values))
                    ax.scatter(
                        budget + offsets[method] + jitter,
                        values,
                        s=18,
                        facecolor="white",
                        edgecolor=COLORS[method],
                        linewidth=0.8,
                        alpha=0.9,
                        zorder=3,
                    )
                x = np.asarray(budgets, dtype=float) + offsets[method]
                ax.errorbar(
                    x,
                    means,
                    yerr=stds,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=5,
                    linewidth=1.6,
                    capsize=3,
                    label=method,
                    zorder=4,
                )

            ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle=":")
            ax.set_xticks(budgets)
            ax.set_xlabel("Interior solution labels, $N$")
            ax.set_ylabel("Temperature $R^2$" if test_name == "test1" else "Water-saturation $R^2$")
            ax.set_title(f"Test {1 if test_name == 'test1' else 2}: {REGION_LABELS[region]}")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            if test_name == "test1" and region == "interpolation":
                ax.set_ylim(0.85, 1.01)
            else:
                observed = subset[metric].to_numpy()
                lower = min(-0.1, float(np.min(observed) - 0.12))
                ax.set_ylim(lower, 1.03)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        title="Mean ± sample SD; open points are individual seeds",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.86, wspace=0.28, hspace=0.44)
    save(fig, output_dir, "zero_label_r2_summary")


def plot_test2_seed_stability(data: pd.DataFrame, output_dir: Path) -> None:
    subset = data[data.test_name == "test2"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.55))
    for ax, region in zip(axes, ("interpolation", "extrapolation")):
        region_data = subset[subset.evaluation_region == region]
        for method in ("H-PINN", "HO-PINN"):
            method_data = region_data[region_data.method == method]
            for seed, seed_data in method_data.groupby("seed"):
                seed_data = seed_data.sort_values("n_train")
                ax.plot(
                    seed_data.n_train,
                    seed_data.Swat_R2,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    linewidth=1.0,
                    alpha=0.75,
                )
                if method == "H-PINN" and seed == 102:
                    first = seed_data.iloc[0]
                    ax.annotate(
                        "H-PINN seed 102",
                        (first.n_train, first.Swat_R2),
                        xytext=(8, 7),
                        textcoords="offset points",
                        fontsize=7,
                        color=COLORS[method],
                    )
        ax.axhline(0.0, color="#777777", linewidth=0.7, linestyle=":")
        ax.set_xticks([0, 20])
        ax.set_xlabel("Interior solution labels, $N$")
        ax.set_ylabel("Water-saturation $R^2$")
        ax.set_title(REGION_LABELS[region])
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)

    handles = [
        plt.Line2D([], [], color=COLORS[m], marker=MARKERS[m], label=m)
        for m in ("H-PINN", "HO-PINN")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.80, wspace=0.24)
    save(fig, output_dir, "test2_seed_stability")


def load_publication_curves(paper: Path) -> pd.DataFrame:
    frames = []
    for test_name, path in (
        ("test1", paper / "generated/test1_corrected_candidate/unified_test1.csv"),
        ("test2", paper / "generated/corrected_test2_final/unified_test2.csv"),
    ):
        frame = pd.read_csv(path)
        frame = frame[
            frame.method.isin(["H-PINN", "HO-PINN"])
            & frame.n_train.isin([20, 50, 100, 200])
        ].copy()
        frame["test_name"] = test_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def best_method_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        frame.groupby(["evaluation_region", "n_train", "method"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    indices = summary.groupby(["evaluation_region", "n_train"])["mean"].idxmax()
    return summary.loc[indices].sort_values(["evaluation_region", "n_train"])


def plot_publication_context(data: pd.DataFrame, paper: Path, output_dir: Path) -> None:
    old = load_publication_curves(paper)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))
    for row, test_name in enumerate(("test1", "test2")):
        metric = metric_for(test_name)
        old_summary = best_method_summary(old[old.test_name == test_name], metric)
        new_ho = data[(data.test_name == test_name) & (data.method == "HO-PINN")]
        new_summary = (
            new_ho.groupby(["evaluation_region", "n_train"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for col, region in enumerate(("interpolation", "extrapolation")):
            ax = axes[row, col]
            old_region = old_summary[old_summary.evaluation_region == region]
            new_region = new_summary[new_summary.evaluation_region == region]
            ax.errorbar(
                old_region.n_train,
                old_region["mean"],
                yerr=old_region["std"],
                color="#777777",
                marker="o",
                linewidth=1.2,
                linestyle="--",
                capsize=2,
                label="Publication protocol: best PINN",
            )
            ax.errorbar(
                new_region.n_train,
                new_region["mean"],
                yerr=new_region["std"],
                color=COLORS["HO-PINN"],
                marker="s",
                linewidth=1.8,
                capsize=3,
                label="Additional protocol: HO-PINN",
            )
            ax.axhline(0.0, color="#999999", linewidth=0.6, linestyle=":")
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 50, 100, 200])
            ax.set_xticklabels(["0", "20", "50", "100", "200"])
            ax.set_xlabel("Interior solution labels, $N$")
            ax.set_ylabel("Temperature $R^2$" if test_name == "test1" else "Water-saturation $R^2$")
            ax.set_title(f"Test {1 if test_name == 'test1' else 2}: {REGION_LABELS[region]}")
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            values = np.concatenate((old_region["mean"].to_numpy(), new_region["mean"].to_numpy()))
            if np.min(values) > 0.8:
                ax.set_ylim(max(0.78, float(np.min(values) - 0.06)), 1.01)
            else:
                ax.set_ylim(min(-0.5, float(np.min(values) - 0.15)), 1.03)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.text(
        0.5,
        0.018,
        "Descriptive context only: protocols differ in epochs, residual batch, and checkpoint rule.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.88, wspace=0.28, hspace=0.44)
    save(fig, output_dir, "zero_label_vs_publication_context")


def history_records(history_dir: Path) -> pd.DataFrame:
    frames = []
    pattern = re.compile(r"(test\d)_n(\d+)_s(\d+)_(h|ho)_fixednr(\d+)_e(\d+)")
    for path in sorted(history_dir.glob("*.csv")):
        match = pattern.fullmatch(path.stem)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["test_name"] = match.group(1)
        frame["n_train"] = int(match.group(2))
        frame["seed"] = int(match.group(3))
        frame["method"] = "H-PINN" if match.group(4) == "h" else "HO-PINN"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No history CSV files found in {history_dir}")
    return pd.concat(frames, ignore_index=True)


def plot_training_dynamics(histories: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5))
    for col, test_name in enumerate(("test1", "test2")):
        ax = axes[0, col]
        subset = histories[(histories.test_name == test_name) & (histories.n_train == 0)]
        for (method, seed), group in subset.groupby(["method", "seed"]):
            ax.plot(
                group.epoch,
                group.physics_val_raw,
                color=COLORS[method],
                linewidth=1.0,
                alpha=0.75,
                linestyle="-" if seed == 102 else "--",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fixed physics-validation loss")
        ax.set_title(f"Test {1 if test_name == 'test1' else 2}, $N=0$")
        ax.grid(color="#D9D9D9", linewidth=0.5, which="both")

    for col, method in enumerate(("H-PINN", "HO-PINN")):
        ax = axes[1, col]
        group = histories[
            (histories.test_name == "test2")
            & (histories.n_train == 0)
            & (histories.seed == 102)
            & (histories.method == method)
        ].sort_values("epoch")
        ax.plot(group.epoch, group.physics_val_ibc, label="IBC", color="#4C78A8", linewidth=1.4)
        ax.plot(group.epoch, group.physics_val_pde, label="PDE", color="#F58518", linewidth=1.4)
        ax.plot(group.epoch, group.physics_val_raw, label="Sum", color="#333333", linewidth=1.1, linestyle=":")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss component")
        ax.set_title(f"Test 2, seed 102, $N=0$: {method}")
        ax.grid(color="#D9D9D9", linewidth=0.5, which="both")
        ax.legend(frameon=False)

    handles = [
        plt.Line2D([], [], color=COLORS[m], label=m) for m in ("H-PINN", "HO-PINN")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        title="Physics-only training histories",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.86, wspace=0.28, hspace=0.42)
    save(fig, output_dir, "zero_label_training_dynamics")


def load_checkpoint_model(checkpoint_path: Path, bundle):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(bundle, ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def checkpoint_path(checkpoint_dir: Path, test: str, budget: int, seed: int, method: str) -> Path:
    mode = "h" if method == "H-PINN" else "ho"
    matches = list(checkpoint_dir.glob(f"{test}_n{budget}_s{seed}_{mode}_fixednr*_e*.pth"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one checkpoint for {test}, N={budget}, seed={seed}, {method}; found {matches}")
    return matches[0]


def plot_test1_profiles(checkpoint_dir: Path, output_dir: Path) -> None:
    bundle = build_unified_bundle("test1")
    x = np.linspace(0.0, 1.0, 400, dtype=np.float32)
    times = [0.0, 0.035, 0.05]
    configurations = [
        (0, "H-PINN", COLORS["H-PINN"], "--"),
        (0, "HO-PINN", COLORS["HO-PINN"], "--"),
        (20, "H-PINN", COLORS["H-PINN"], "-"),
        (20, "HO-PINN", COLORS["HO-PINN"], "-"),
    ]
    models = {
        (budget, method, seed): load_checkpoint_model(
            checkpoint_path(checkpoint_dir, "test1", budget, seed, method), bundle
        )
        for budget, method, _, _ in configurations
        for seed in (101, 102, 103)
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True, constrained_layout=True)
    for ax, t in zip(axes, times):
        features = np.column_stack((np.full_like(x, t), x)).astype(np.float32)
        exact = (
            2.0
            + np.exp(-4.0 * np.pi**2 * t) * np.sin(2.0 * np.pi * x)
            + np.exp(-16.0 * np.pi**2 * t) * np.cos(4.0 * np.pi * x)
        )
        ax.plot(x, exact, color="black", linewidth=2.0, label="Reference")
        for budget, method, color, linestyle in configurations:
            predictions = []
            for seed in (101, 102, 103):
                with torch.no_grad():
                    values = models[(budget, method, seed)](torch.tensor(features)).reshape(-1).numpy()
                predictions.append(values)
            predictions = np.asarray(predictions)
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0, ddof=1)
            label = f"{method}, N={budget}"
            ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=1.2, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)
        ax.set_xlabel("$x$")
        ax.set_title(f"$t={t:g}$")
        ax.grid(color="#E0E0E0", linewidth=0.5)
    axes[0].set_ylabel("Temperature")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13))
    save(fig, output_dir, "test1_zero_label_profiles")


def plot_test2_fields(checkpoint_dir: Path, corrected_root: Path, output_dir: Path) -> None:
    dataset_path = corrected_root / "test2" / "solution.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float32)
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        reference = np.asarray(data["water_saturation"][-1], dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    features = np.column_stack(
        (np.full(xx.size, times[-1], dtype=np.float32), xx.ravel(), yy.ravel())
    )
    bundle = build_unified_bundle(
        "test2",
        data_source="corrected",
        corrected_data_root=corrected_root,
    )
    configurations = [
        (0, "H-PINN"),
        (0, "HO-PINN"),
        (20, "H-PINN"),
        (20, "HO-PINN"),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(11.0, 6.2), constrained_layout=True)
    for row, seed in enumerate((101, 102, 103)):
        image = axes[row, 0].imshow(reference.T, origin="lower", extent=(0, 1, 0, 1), vmin=0, vmax=1, cmap="viridis", aspect="equal")
        axes[row, 0].set_title("Reference" if row == 0 else "")
        axes[row, 0].set_ylabel(f"seed {seed}\n$y$")
        for col, (budget, method) in enumerate(configurations, start=1):
            model = load_checkpoint_model(
                checkpoint_path(checkpoint_dir, "test2", budget, seed, method), bundle
            )
            with torch.no_grad():
                prediction = model(torch.tensor(features))[:, 2].numpy().reshape(xx.shape)
            axes[row, col].imshow(
                prediction.T,
                origin="lower",
                extent=(0, 1, 0, 1),
                vmin=0,
                vmax=1,
                cmap="viridis",
                aspect="equal",
            )
            if row == 0:
                axes[row, col].set_title(f"{method}\n$N={budget}$")
        for ax in axes[row]:
            ax.set_xlabel("$x$")
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
    colorbar = fig.colorbar(image, ax=axes, location="right", shrink=0.82, pad=0.015)
    colorbar.set_label("Water saturation $S_w$")
    fig.suptitle(f"Test 2 final-time fields ($t={times[-1]:g}$)", y=1.025)
    save(fig, output_dir, "test2_zero_label_fields")


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def write_report(data: pd.DataFrame, histories: pd.DataFrame, paper: Path, output_dir: Path) -> None:
    lines = [
        "# Zero-label experiment: Tests 1 and 2",
        "",
        "All entries are mean ± sample standard deviation over seeds 101–103. The additional protocol uses 50,000 epochs and a fixed residual batch of 128 points for both N=0 and N=20.",
        "",
        "## Primary R²",
        "",
        "| Test | Region | N | H-PINN | HO-PINN |",
        "|---|---|---:|---:|---:|",
    ]
    for test_name, label, metric in (("test1", "Test 1 temperature", "R2"), ("test2", "Test 2 water saturation", "Swat_R2")):
        for region in ("interpolation", "extrapolation"):
            for budget in (0, 20):
                row = data[(data.test_name == test_name) & (data.evaluation_region == region) & (data.n_train == budget)]
                cells = []
                for method in ("H-PINN", "HO-PINN"):
                    values = row[row.method == method][metric]
                    cells.append(fmt(values.mean(), values.std(ddof=1)))
                lines.append(f"| {label} | {REGION_LABELS[region]} | {budget} | {cells[0]} | {cells[1]} |")

    lines.extend(
        [
            "",
            "## Main findings",
            "",
            "- Test 1 does not support a zero-label high-accuracy claim. HO-PINN obtains early-time R² 0.932 ± 0.026, but late-time R² is −0.270 ± 0.528. Twenty labels restore near-perfect early-time reconstruction and late-time R² 0.929 ± 0.044.",
            "- Test 2 provides strong positive evidence for physics-only HO-PINN. Its zero-label water-saturation R² is 0.963 ± 0.002 early and 0.938 ± 0.003 late, compared with 0.971 ± 0.008 and 0.938 ± 0.015 at N=20.",
            "- Physics-only H-PINN is not robust. Seed 102 converges to a degenerate solution with nearly zero validation residual but negative water-saturation R². HO-PINN avoids this failure in all three seeds.",
            "- These results support a benchmark-specific zero-interior-label claim for Test 2, not a universal statement that PINNs never need data.",
            "",
            "## Protocol caveat",
            "",
            "The additional N=0/20 protocol differs from the publication N=20–200 runs: 50,000 rather than 10,000 epochs, fixed N_r=128 rather than N_r=2N, and a fixed raw physics-validation checkpoint criterion. Cross-protocol comparisons are descriptive; causal label-budget conclusions should rely on N=0 versus N=20 within the additional protocol or on a future fixed-protocol N=0–200 sweep.",
            "",
        ]
    )
    (output_dir / "zero_label_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the zero-label Test 1/2 experiment.")
    parser.add_argument(
        "--experiment-name", default="zero_label_test12_e50k_nr128"
    )
    parser.add_argument(
        "--corrected-data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "publication_data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = PAPER / "generated" / args.experiment_name
    results_path = result_dir / "zero_label_results.csv"
    checkpoint_dir = PAPER / "experiments" / args.experiment_name / "checkpoints"
    history_dir = PAPER / "experiments" / args.experiment_name / "histories"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    output_dir = result_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(results_path)
    histories = history_records(history_dir)
    configure_style()
    plot_main_summary(data, output_dir)
    plot_test2_seed_stability(data, output_dir)
    plot_publication_context(data, PAPER, output_dir)
    plot_training_dynamics(histories, output_dir)
    plot_test1_profiles(checkpoint_dir, output_dir)
    plot_test2_fields(checkpoint_dir, args.corrected_data_root.resolve(), output_dir)
    write_report(data, histories, PAPER, output_dir)
    print(f"Figures and report: {output_dir}")


if __name__ == "__main__":
    main()
