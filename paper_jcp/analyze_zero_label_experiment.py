#!/usr/bin/env python3
"""Summarize and visualize the additional zero-label Test 1/2 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.unified_protocol import (  # noqa: E402
    ModelConfig,
    build_model,
    build_unified_bundle,
    predict_pinn,
)


PAPER = Path(__file__).resolve().parent
COLORS = {"H-PINN": "#b73535", "HO-PINN": "#e28b23"}
SEED_COLORS = {101: "#3366a8", 102: "#8c4a9c", 103: "#238b57"}
METHOD_TO_MODE = {"H-PINN": "h", "HO-PINN": "ho"}
REGION_LABEL = {"interpolation": "Early-time region", "extrapolation": "Late-time region"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="zero_label_test12_e50k_nr128")
    parser.add_argument(
        "--corrected-data-root",
        type=Path,
        default=PAPER / "generated" / "publication_data",
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def primary_metric(test_name: str) -> str:
    return "R2" if test_name == "test1" else "Swat_R2"


def aggregate_results(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_names = {
        "test1": ["R2", "RMSE", "relative_L2"],
        "test2": ["Swat_R2", "Swat_RMSE", "Pressure_R2", "Pressure_RMSE", "mean_R2"],
    }
    for test_name, metrics in metric_names.items():
        subset = data[data.test_name == test_name]
        grouped = subset.groupby(["evaluation_region", "n_train", "method"])
        for key, group in grouped:
            region, budget, method = key
            row = {
                "test_name": test_name,
                "evaluation_region": region,
                "n_train": budget,
                "method": method,
                "seeds": len(group),
                "train_time_mean_sec": group.train_time_sec.mean(),
                "train_time_std_sec": group.train_time_sec.std(),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std()
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["test_name", "evaluation_region", "n_train", "method"]
    )


def plot_primary_summary(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), sharex=True)
    for row, test_name in enumerate(("test1", "test2")):
        metric = primary_metric(test_name)
        for col, region in enumerate(("interpolation", "extrapolation")):
            ax = axes[row, col]
            subset = data[
                (data.test_name == test_name)
                & (data.evaluation_region == region)
            ]
            for method, offset in (("H-PINN", -0.08), ("HO-PINN", 0.08)):
                method_rows = subset[subset.method == method]
                stats = method_rows.groupby("n_train")[metric].agg(["mean", "std"])
                positions = np.arange(2) + offset
                ax.errorbar(
                    positions,
                    stats.reindex([0, 20])["mean"],
                    yerr=stats.reindex([0, 20])["std"],
                    fmt="o",
                    ms=7,
                    capsize=4,
                    lw=1.8,
                    color=COLORS[method],
                    label=method,
                    zorder=4,
                )
                for index, budget in enumerate((0, 20)):
                    values = method_rows[method_rows.n_train == budget][metric].to_numpy()
                    jitter = np.linspace(-0.035, 0.035, len(values))
                    ax.scatter(
                        positions[index] + jitter,
                        values,
                        s=18,
                        color=COLORS[method],
                        alpha=0.45,
                        zorder=3,
                    )
            ax.axhline(0.0, color="#777777", lw=0.8, ls=":")
            ax.set_xticks([0, 1], ["0", "20"])
            ax.grid(axis="y", alpha=0.22)
            ax.set_title(f"{REGION_LABEL[region]}")
            if col == 0:
                label = "Temperature $R^2$" if test_name == "test1" else "Water saturation $R^2$"
                ax.set_ylabel(f"Test {1 if test_name == 'test1' else 2}\n{label}")
            if row == 1:
                ax.set_xlabel("Interior solution labels $N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Zero-label and 20-label PINN accuracy", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "primary_r2_zero_vs_20")


def plot_seed_level(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), sharex=True)
    categories = [(0, "H-PINN"), (0, "HO-PINN"), (20, "H-PINN"), (20, "HO-PINN")]
    labels = ["H\n$N=0$", "HO\n$N=0$", "H\n$N=20$", "HO\n$N=20$"]
    for row, test_name in enumerate(("test1", "test2")):
        metric = primary_metric(test_name)
        for col, region in enumerate(("interpolation", "extrapolation")):
            ax = axes[row, col]
            subset = data[
                (data.test_name == test_name)
                & (data.evaluation_region == region)
            ]
            for seed in (101, 102, 103):
                values = []
                for budget, method in categories:
                    value = subset[
                        (subset.seed == seed)
                        & (subset.n_train == budget)
                        & (subset.method == method)
                    ][metric].iloc[0]
                    values.append(value)
                ax.plot(
                    np.arange(4),
                    values,
                    marker="o",
                    ms=4,
                    lw=1.2,
                    color=SEED_COLORS[seed],
                    alpha=0.85,
                    label=f"seed {seed}",
                )
            means = []
            for budget, method in categories:
                means.append(
                    subset[(subset.n_train == budget) & (subset.method == method)][metric].mean()
                )
            ax.scatter(np.arange(4), means, marker="_", s=220, lw=3, color="black", zorder=5)
            ax.axhline(0.0, color="#777777", lw=0.8, ls=":")
            ax.set_xticks(np.arange(4), labels)
            ax.grid(axis="y", alpha=0.22)
            ax.set_title(REGION_LABEL[region])
            if col == 0:
        target = "Temperature $R^2$" if test_name == "test1" else r"$S_w\;R^2$"
                ax.set_ylabel(f"Test {1 if test_name == 'test1' else 2}\n{target}")
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Seed-level stability (black bars denote means)", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "seed_level_primary_r2")


def parse_history_name(path: Path) -> tuple[str, int, int, str]:
    parts = path.stem.split("_")
    test_name = parts[0]
    budget = int(parts[1][1:])
    seed = int(parts[2][1:])
    mode = parts[3]
    return test_name, budget, seed, mode


def plot_convergence(history_dir: Path, output_dir: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    summary_rows = []
    for path in sorted(history_dir.glob("*.csv")):
        test_name, budget, seed, mode = parse_history_name(path)
        data = pd.read_csv(path)
        best = data.loc[data.physics_val_raw.idxmin()]
        summary_rows.append(
            {
                "test_name": test_name,
                "n_train": budget,
                "seed": seed,
                "method": "H-PINN" if mode == "h" else "HO-PINN",
                "best_epoch": int(best.epoch),
                "best_physics_val_raw": best.physics_val_raw,
                "best_physics_val_ibc": best.physics_val_ibc,
                "best_physics_val_pde": best.physics_val_pde,
                "final_physics_val_raw": data.iloc[-1].physics_val_raw,
            }
        )
        row = 0 if test_name == "test1" else 1
        col = 0 if budget == 0 else 1
        method = "H-PINN" if mode == "h" else "HO-PINN"
        axes[row, col].plot(
            data.epoch,
            np.maximum(data.physics_val_raw, 1e-25),
            color=COLORS[method],
            ls="-" if seed == 101 else "--" if seed == 102 else ":",
            lw=1.3,
            alpha=0.9,
            label=f"{method}, seed {seed}",
        )
    for row, test_name in enumerate(("test1", "test2")):
        for col, budget in enumerate((0, 20)):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.grid(alpha=0.2, which="both")
            ax.set_title(f"Test {1 if test_name == 'test1' else 2}, $N={budget}$")
            if col == 0:
                ax.set_ylabel("Fixed validation objective")
            if row == 1:
                ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("IBC/PDE validation convergence", y=1.03, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "physics_validation_convergence")
    return pd.DataFrame(summary_rows).sort_values(
        ["test_name", "n_train", "method", "seed"]
    )


def load_checkpoint_model(test_name: str, checkpoint_path: Path, corrected_root: Path):
    bundle = build_unified_bundle(
        test_name,
        time_fraction=0.7,
        interpolation_size=1500,
        extrapolation_size=1500,
        data_source="legacy" if test_name == "test1" else "corrected",
        corrected_data_root=None if test_name == "test1" else corrected_root,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(bundle, ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return bundle, model


def checkpoint_path(checkpoint_dir: Path, test: str, budget: int, seed: int, method: str) -> Path:
    mode = METHOD_TO_MODE[method]
    return checkpoint_dir / f"{test}_n{budget}_s{seed}_{mode}_fixednr128_e50000.pth"


def plot_test1_profiles(checkpoint_dir: Path, corrected_root: Path, output_dir: Path) -> None:
    x = np.linspace(0.0, 1.0, 401, dtype=np.float32)
    times = (0.025, 0.05)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4), sharex=True)
    for row, current_time in enumerate(times):
        features = np.column_stack((np.full_like(x, current_time), x))
        truth = 2.0 + np.exp(-4.0 * np.pi**2 * current_time) * np.sin(2.0 * np.pi * x)
        truth += np.exp(-16.0 * np.pi**2 * current_time) * np.cos(4.0 * np.pi * x)
        for col, budget in enumerate((0, 20)):
            ax = axes[row, col]
            ax.plot(x, truth, color="black", lw=2.4, label="Reference", zorder=5)
            for method in ("H-PINN", "HO-PINN"):
                predictions = []
                for seed in (101, 102, 103):
                    _, model = load_checkpoint_model(
                        "test1",
                        checkpoint_path(checkpoint_dir, "test1", budget, seed, method),
                        corrected_root,
                    )
                    prediction = predict_pinn("test1", model, features)
                    predictions.append(prediction)
                    ax.plot(x, prediction, color=COLORS[method], alpha=0.22, lw=0.9)
                ax.plot(
                    x,
                    np.mean(predictions, axis=0),
                    color=COLORS[method],
                    lw=2.0,
                    label=f"{method} mean",
                )
            ax.grid(alpha=0.2)
            ax.set_title(f"$N={budget}$, $t={current_time:g}$")
            if col == 0:
                ax.set_ylabel("Temperature")
            if row == 1:
                ax.set_xlabel("$x$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Test 1 profiles: faint lines are individual seeds", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, "test1_temperature_profiles")


def r2_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((truth - truth.mean()) ** 2)
    return float(1.0 - np.sum((truth - prediction) ** 2) / denominator)


def plot_test2_fields(
    checkpoint_dir: Path,
    corrected_root: Path,
    output_dir: Path,
    field_name: str,
) -> None:
    dataset_path = corrected_root / "test2" / "solution.npz"
    with np.load(dataset_path, allow_pickle=False) as data:
        times = data["times"]
        x = data["x"]
        y = data["y"]
        reference = data[field_name][-1]
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    features = np.column_stack(
        (
            np.full(grid_x.size, times[-1], dtype=np.float32),
            grid_x.reshape(-1),
            grid_y.reshape(-1),
        )
    ).astype(np.float32)
    target_index = 0 if field_name == "pressure" else 2
    configurations = [
        None,
        (0, "H-PINN"),
        (0, "HO-PINN"),
        (20, "H-PINN"),
        (20, "HO-PINN"),
    ]
    titles = ["Reference", "H, $N=0$", "HO, $N=0$", "H, $N=20$", "HO, $N=20$"]
    vmin = 0.0
    vmax = 1.0 if field_name == "pressure" else float(reference.max()) * 1.08
    fig, axes = plt.subplots(3, 5, figsize=(13.0, 7.2), sharex=True, sharey=True)
    image = None
    for row, seed in enumerate((101, 102, 103)):
        for col, configuration in enumerate(configurations):
            ax = axes[row, col]
            if configuration is None:
                field = reference
                score_text = "$R^2=1$"
            else:
                budget, method = configuration
                _, model = load_checkpoint_model(
                    "test2",
                    checkpoint_path(checkpoint_dir, "test2", budget, seed, method),
                    corrected_root,
                )
                prediction = predict_pinn("test2", model, features)[:, target_index]
                field = prediction.reshape(reference.shape)
                score_text = f"$R^2={r2_score(reference, field):.3f}$"
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(float(x.min()), float(x.max()), float(y.min()), float(y.max())),
                vmin=vmin,
                vmax=vmax,
                cmap="viridis" if field_name == "pressure" else "Blues",
                aspect="equal",
            )
            ax.text(
                0.03,
                0.95,
                score_text,
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
            )
            if row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(f"seed {seed}\n$y$")
            if row == 2:
                ax.set_xlabel("$x$")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.045, pad=0.09)
    colorbar.set_label("Pressure" if field_name == "pressure" else "Water saturation $S_w$")
    fig.suptitle(
        f"Test 2 final-time {'pressure' if field_name == 'pressure' else 'water saturation'} ($t={times[-1]:g}$)",
        y=0.99,
        fontsize=13,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.15, wspace=0.08, hspace=0.16)
    save_figure(fig, output_dir, f"test2_final_{field_name}_all_seeds")


def plot_context(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    article_sources = {
        "test1": PAPER / "generated" / "test1_corrected_candidate" / "unified_test1.csv",
        "test2": PAPER / "generated" / "corrected_test2_final" / "unified_test2.csv",
    }
    context_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    for row, test_name in enumerate(("test1", "test2")):
        metric = primary_metric(test_name)
        article = pd.read_csv(article_sources[test_name])
        article = article[
            article.method.isin(["H-PINN", "HO-PINN"])
            & article.n_train.isin([50, 100, 200])
        ]
        new = data[(data.test_name == test_name) & data.n_train.isin([0, 20])]
        for col, region in enumerate(("interpolation", "extrapolation")):
            ax = axes[row, col]
            for method in ("H-PINN", "HO-PINN"):
                new_stats = (
                    new[(new.method == method) & (new.evaluation_region == region)]
                    .groupby("n_train")[metric]
                    .agg(["mean", "std"])
                    .reindex([0, 20])
                )
                old_stats = (
                    article[(article.method == method) & (article.evaluation_region == region)]
                    .groupby("n_train")[metric]
                    .agg(["mean", "std"])
                    .reindex([50, 100, 200])
                )
                ax.errorbar(
                    [0, 20], new_stats["mean"], yerr=new_stats["std"],
                    color=COLORS[method], marker="o", capsize=3, lw=1.8,
                    label=f"{method}: new 50k",
                )
                ax.errorbar(
                    [50, 100, 200], old_stats["mean"], yerr=old_stats["std"],
                    color=COLORS[method], marker="s", markerfacecolor="white",
                    capsize=3, lw=1.2, ls="--", label=f"{method}: article 10k",
                )
                for budget, values, source in (
                    *[(b, new_stats.loc[b], "new_50k_fixed_Nr") for b in (0, 20)],
                    *[(b, old_stats.loc[b], "article_10k_ratio_Nr") for b in (50, 100, 200)],
                ):
                    context_rows.append(
                        {
                            "test_name": test_name,
                            "evaluation_region": region,
                            "method": method,
                            "n_train": budget,
                            "primary_R2_mean": values["mean"],
                            "primary_R2_std": values["std"],
                            "source_protocol": source,
                        }
                    )
            ax.axvline(35, color="#777777", lw=0.8, ls=":")
            ax.axhline(0, color="#777777", lw=0.7, ls=":")
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 50, 100, 200], ["0", "20", "50", "100", "200"])
            ax.grid(axis="y", alpha=0.22)
            ax.set_title(REGION_LABEL[region])
            if col == 0:
        target = "Temperature $R^2$" if test_name == "test1" else r"$S_w\;R^2$"
                ax.set_ylabel(f"Test {1 if test_name == 'test1' else 2}\n{target}")
            if row == 1:
                ax.set_xlabel("Interior solution labels $N$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Context against stored article runs (protocol changes after dotted line)", y=1.02, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output_dir, "context_against_article_budgets")
    return pd.DataFrame(context_rows).sort_values(
        ["test_name", "evaluation_region", "method", "n_train"]
    )


def mean_std(data: pd.DataFrame, test: str, region: str, budget: int, method: str, metric: str):
    values = data[
        (data.test_name == test)
        & (data.evaluation_region == region)
        & (data.n_train == budget)
        & (data.method == method)
    ][metric]
    return values.mean(), values.std()


def write_report(data: pd.DataFrame, output_dir: Path) -> None:
    def value(test, region, budget, method, metric):
        mean, std = mean_std(data, test, region, budget, method, metric)
        return f"{mean:.3f} ± {std:.3f}"

    lines = [
        "# Zero-label experiment: Tests 1 and 2",
        "",
        "All 24 model fits completed: two tests, two label budgets, two balancing modes, and three seeds. Every run used 50,000 epochs and 128 freshly resampled residual points per epoch.",
        "",
        "## Primary results",
        "",
        "| Test | Region | Method | N=0 R² | N=20 R² |",
        "|---|---|---|---:|---:|",
    ]
    for test, metric in (("test1", "R2"), ("test2", "Swat_R2")):
        for region in ("interpolation", "extrapolation"):
            for method in ("H-PINN", "HO-PINN"):
                lines.append(
                    f"| {test.title()} | {region} | {method} | "
                    f"{value(test, region, 0, method, metric)} | "
                    f"{value(test, region, 20, method, metric)} |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Test 1 does not support a high-accuracy zero-label claim under the current loss. Early-time R² is about 0.93, but late-time mean R² is negative. Twenty labels recover near-perfect early-time accuracy and late-time R² about 0.93.",
            "- Test 2 gives a strong but method-dependent zero-label result. HO-PINN reaches water-saturation R² "
            f"{value('test2', 'interpolation', 0, 'HO-PINN', 'Swat_R2')} early and "
            f"{value('test2', 'extrapolation', 0, 'HO-PINN', 'Swat_R2')} late, essentially matching its N=20 result.",
            "- H-PINN is not robust at N=0 on Test 2. Seed 102 converges to a dry stationary field (Sw=0) with a boundary-layer pressure solution, although its recorded physics-validation objective is almost zero. The mean and standard deviation are therefore poor despite two successful seeds.",
            "- The failure demonstrates that residual minimization alone is not sufficient evidence of solution correctness. The current soft boundary formulation admits a degenerate optimization basin when no interior labels are available.",
            "- HO-PINN avoids this basin for all three tested seeds. This is promising evidence for the orthogonal factor, but three seeds are insufficient for a general robustness claim.",
            "",
            "## Publication conclusion",
            "",
            "The broad hypothesis that PINNs need no solution labels is not supported across both tests. A narrower statement is supported: on the corrected heterogeneous Test 2 problem, HO-PINN can reproduce pressure and water saturation without interior solution labels at approximately the same accuracy as its 20-label counterpart. This result should be reported together with the H-PINN failure and the fact that PDE/IBC information and 6.4 million residual evaluations were still used.",
            "",
            "The comparison against N=50–200 in `context_against_article_budgets` is contextual rather than controlled: N=0/20 use 50,000 epochs and fixed Nr=128, whereas the stored article runs use 10,000 epochs and Nr=2N.",
        ]
    )
    (output_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_root = PAPER / "generated" / args.experiment_name
    result_path = experiment_root / "zero_label_results.csv"
    history_dir = PAPER / "experiments" / args.experiment_name / "histories"
    checkpoint_dir = PAPER / "experiments" / args.experiment_name / "checkpoints"
    output_dir = experiment_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(result_path)
    expected = 2 * 2 * 2 * 3 * 2
    if len(data) != expected:
        raise RuntimeError(f"Expected {expected} result rows, found {len(data)}")

    summary = aggregate_results(data)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    plot_primary_summary(data, output_dir)
    plot_seed_level(data, output_dir)
    convergence = plot_convergence(history_dir, output_dir)
    convergence.to_csv(output_dir / "convergence_summary.csv", index=False)
    plot_test1_profiles(checkpoint_dir, args.corrected_data_root, output_dir)
    plot_test2_fields(checkpoint_dir, args.corrected_data_root, output_dir, "water_saturation")
    plot_test2_fields(checkpoint_dir, args.corrected_data_root, output_dir, "pressure")
    context = plot_context(data, output_dir)
    context.to_csv(output_dir / "context_against_article.csv", index=False)
    write_report(data, output_dir)
    print(f"Analysis written to: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(path.name)


if __name__ == "__main__":
    main()
