#!/usr/bin/env python3
"""Compare local N=0/20 runs with the completed 50k-epoch cluster sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
COLORS = {"H-PINN": "#C43C39", "HO-PINN": "#2C6EAA"}
MARKERS = {"H-PINN": "o", "HO-PINN": "s"}
REGIONS = ("interpolation", "extrapolation")
CLUSTER_BUDGETS = (20, 50, 100, 200, 500, 1000)


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


def aggregate(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.groupby(["n_train", "method", "evaluation_region"])[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )


def test_sources(args: argparse.Namespace):
    additional = pd.read_csv(args.additional_results)
    cluster = {
        "test1": pd.read_csv(args.test1_cluster_results),
        "test2": pd.read_csv(args.test2_cluster_results),
    }
    return additional, cluster


def metric_name(test_name: str, kind: str) -> str:
    if test_name == "test1":
        return "R2" if kind == "r2" else "MSE"
    return "Swat_R2" if kind == "r2" else "Swat_MSE"


def ylabel(test_name: str, kind: str) -> str:
    field = "Temperature" if test_name == "test1" else "Water saturation"
    return f"{field} $R^2$" if kind == "r2" else f"{field} MSE"


def plot_metric(
    additional: pd.DataFrame,
    cluster: dict[str, pd.DataFrame],
    output_dir: Path,
    kind: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.6))
    for row, test_name in enumerate(("test1", "test2")):
        metric = metric_name(test_name, kind)
        add_summary = aggregate(
            additional[additional.test_name == test_name], metric
        )
        cluster_summary = aggregate(cluster[test_name], metric)
        for col, region in enumerate(REGIONS):
            ax = axes[row, col]
            for method in ("H-PINN", "HO-PINN"):
                curve = cluster_summary[
                    (cluster_summary.method == method)
                    & (cluster_summary.evaluation_region == region)
                ].sort_values("n_train")
                ax.errorbar(
                    curve.n_train,
                    curve["mean"],
                    yerr=curve["std"],
                    color=COLORS[method],
                    marker=MARKERS[method],
                    markersize=4.5,
                    linewidth=1.5,
                    capsize=2.5,
                    label=f"Cluster {method} (10 seeds)",
                    zorder=4,
                )
                points = add_summary[
                    (add_summary.method == method)
                    & (add_summary.evaluation_region == region)
                ].sort_values("n_train")
                ax.errorbar(
                    points.n_train,
                    points["mean"],
                    yerr=points["std"],
                    color=COLORS[method],
                    marker="X" if method == "H-PINN" else "P",
                    markersize=7,
                    linewidth=1.0,
                    linestyle=":",
                    capsize=3,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    label=f"Additional {method} (3 seeds)",
                    zorder=7,
                )
            ax.set_xscale("symlog", linthresh=20)
            ax.set_xticks([0, 20, 50, 100, 200, 500, 1000])
            ax.set_xticklabels(["0", "20", "50", "100", "200", "500", "1000"])
            ax.set_xlabel("Interior solution labels, $N$")
            ax.set_ylabel(ylabel(test_name, kind))
            region_label = "Interpolation" if region == "interpolation" else "Extrapolation"
            ax.set_title(f"Test {1 if test_name == 'test1' else 2}: {region_label}")
            ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.8, which="both")
            if kind == "mse":
                ax.set_yscale("log")
            else:
                ax.axhline(0.0, color="#888888", linewidth=0.65, linestyle=":")
                if test_name == "test1" and region == "interpolation":
                    ax.set_ylim(0.88, 1.005)
                else:
                    values = np.concatenate(
                        [add_summary[add_summary.evaluation_region == region]["mean"].to_numpy(),
                         cluster_summary[cluster_summary.evaluation_region == region]["mean"].to_numpy()]
                    )
                    ax.set_ylim(min(-0.4, float(np.nanmin(values) - 0.15)), 1.035)

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
        0.012,
        "Both protocols use 50k epochs. Additional N=0/20 uses fixed N_r=128 and 3 seeds; cluster N=20–1000 uses N_r=2N and 10 seeds.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.98,
        bottom=0.12,
        top=0.86,
        wspace=0.29,
        hspace=0.43,
    )
    save(fig, output_dir, f"{kind}_budget_comparison")


def fmt(mean: float, std: float, kind: str) -> str:
    if kind == "r2":
        return f"{mean:.5f} ± {std:.5f}"
    return f"{mean:.3e} ± {std:.3e}"


def table_lines(
    additional: pd.DataFrame,
    cluster: pd.DataFrame,
    test_name: str,
    kind: str,
) -> list[str]:
    metric = metric_name(test_name, kind)
    frames = []
    for protocol, frame in (("Additional", additional), ("Cluster", cluster)):
        summary = aggregate(frame, metric)
        summary["protocol"] = protocol
        frames.append(summary)
    summary = pd.concat(frames, ignore_index=True)
    lines = [
        f"### {'R²' if kind == 'r2' else 'MSE'}",
        "",
        "| Protocol | N | Method | Interpolation | Extrapolation | Seeds |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for protocol, budgets in (("Additional", (0, 20)), ("Cluster", CLUSTER_BUDGETS)):
        for budget in budgets:
            for method in ("H-PINN", "HO-PINN"):
                selected = summary[
                    (summary.protocol == protocol)
                    & (summary.n_train == budget)
                    & (summary.method == method)
                ]
                values = {}
                count = 0
                for region in REGIONS:
                    row = selected[selected.evaluation_region == region]
                    if row.empty:
                        values[region] = "n.a."
                    else:
                        item = row.iloc[0]
                        values[region] = fmt(item["mean"], item["std"], kind)
                        count = int(item["count"])
                lines.append(
                    f"| {protocol} | {budget} | {method} | {values['interpolation']} | {values['extrapolation']} | {count} |"
                )
    return lines


def write_report(
    additional: pd.DataFrame,
    cluster: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    lines = [
        "# Test 1–2 metrics: additional and cluster protocols",
        "",
        "All values are arithmetic mean ± sample standard deviation across independent training seeds.",
        "",
        "The additional protocol uses 50,000 epochs, fixed N_r=128, and seeds 101–103. The cluster protocol uses 50,000 epochs, N_r=2N, CUDA, and seeds 101–110. The two N=20 rows therefore test different residual-point budgets and checkpoint protocols.",
        "",
        "## Test 1: temperature",
        "",
    ]
    add1 = additional[additional.test_name == "test1"]
    lines.extend(table_lines(add1, cluster["test1"], "test1", "r2"))
    lines.append("")
    lines.extend(table_lines(add1, cluster["test1"], "test1", "mse"))
    lines.extend(["", "## Test 2: water saturation", ""])
    add2 = additional[additional.test_name == "test2"]
    lines.extend(table_lines(add2, cluster["test2"], "test2", "r2"))
    lines.append("")
    lines.extend(table_lines(add2, cluster["test2"], "test2", "mse"))
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- N=0 is available only in the additional three-seed protocol.",
            "- The two N=20 entries must remain separate because N_r and checkpoint selection differ.",
            "- Test 2 reports the primary water-saturation metrics, not the mean over pressure and both saturations.",
            "- Cross-protocol comparisons are descriptive; within-protocol comparisons support label-budget conclusions.",
            "",
        ]
    )
    (output_dir / "test12_metrics_comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--additional-results",
        type=Path,
        default=PAPER
        / "generated/zero_label_test12_e50k_nr128/zero_label_results.csv",
    )
    parser.add_argument(
        "--test1-cluster-results",
        type=Path,
        default=PAPER / "generated/test1_corrected_candidate/unified_test1.csv",
    )
    parser.add_argument(
        "--test2-cluster-results",
        type=Path,
        default=PAPER / "generated/corrected_test2_final/unified_test2.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PAPER
        / "generated/zero_label_test12_e50k_nr128/figures/cluster_50k_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    additional, cluster = test_sources(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_metric(additional, cluster, args.output_dir, "r2")
    plot_metric(additional, cluster, args.output_dir, "mse")
    write_report(additional, cluster, args.output_dir)
    print(f"Metrics and figures: {args.output_dir}")


if __name__ == "__main__":
    main()
