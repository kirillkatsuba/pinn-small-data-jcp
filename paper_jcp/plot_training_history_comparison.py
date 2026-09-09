#!/usr/bin/env python3
"""Compare PINN loss components and dynamic weights across two protocols.

The additional protocol contains N=0 and N=20 runs trained for 50,000
epochs with a fixed residual batch.  The cluster protocol contains
N=20,...,1000 runs trained for 50,000 epochs with N_r=2N.  Curves use every
available seed and are plotted against normalized optimization progress.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent


@dataclass(frozen=True)
class SeriesStyle:
    label: str
    color: str
    linestyle: str
    linewidth: float
    zorder: int


SERIES_ORDER = [
    ("additional", 0),
    ("additional", 20),
    ("publication", 20),
    ("publication", 50),
    ("publication", 100),
    ("publication", 200),
    ("publication", 500),
    ("publication", 1000),
]

STYLES = {
    ("additional", 0): SeriesStyle(
        "Additional: N=0", "#222222", "-", 2.2, 9
    ),
    ("additional", 20): SeriesStyle(
        "Additional: N=20", "#D55E00", "-", 2.2, 10
    ),
    ("publication", 20): SeriesStyle(
        "Cluster: N=20", "#D55E00", "--", 1.35, 7
    ),
    ("publication", 50): SeriesStyle(
        "Cluster: N=50", "#E69F00", "--", 1.35, 6
    ),
    ("publication", 100): SeriesStyle(
        "Cluster: N=100", "#009E73", "--", 1.35, 6
    ),
    ("publication", 200): SeriesStyle(
        "Cluster: N=200", "#0072B2", "--", 1.35, 6
    ),
    ("publication", 500): SeriesStyle(
        "Cluster: N=500", "#56B4E9", "--", 1.35, 6
    ),
    ("publication", 1000): SeriesStyle(
        "Cluster: N=1000", "#CC79A7", "--", 1.35, 6
    ),
}

ROW_ORDER = [
    ("test1", "H-PINN"),
    ("test1", "HO-PINN"),
    ("test2", "H-PINN"),
    ("test2", "HO-PINN"),
]

ROW_LABELS = {
    ("test1", "H-PINN"): "Test 1, H-PINN",
    ("test1", "HO-PINN"): "Test 1, HO-PINN",
    ("test2", "H-PINN"): "Test 2, H-PINN",
    ("test2", "HO-PINN"): "Test 2, HO-PINN",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def infer_identity(path: Path) -> tuple[str, int, int, str]:
    match = re.match(r"(test[12])_n(\d+)_s(\d+)_(.+)", path.stem)
    if match is None:
        raise ValueError(f"Cannot parse history name: {path.name}")
    test_name, budget, seed, remainder = match.groups()
    mode = re.search(r"(?:^|_)(ho|h)(?:_|$)", remainder)
    if mode is None:
        raise ValueError(f"Cannot parse H/HO mode: {path.name}")
    method = "HO-PINN" if mode.group(1) == "ho" else "H-PINN"
    return test_name, int(budget), int(seed), method


def load_history_directory(path: Path, protocol: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(path.glob("*.csv")):
        test_name, budget, seed, method = infer_identity(csv_path)
        frame = pd.read_csv(csv_path)
        frame["test_name"] = test_name
        frame["n_train"] = budget
        frame["seed"] = seed
        frame["method"] = method
        frame["protocol"] = protocol
        frame["progress_percent"] = 100.0 * frame["epoch"] / frame["epoch"].max()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No history CSV files found in {path}")
    return pd.concat(frames, ignore_index=True, sort=False)


def load_all_histories(
    additional_dir: Path,
    test1_publication_dir: Path,
    test2_publication_dir: Path,
) -> pd.DataFrame:
    additional = load_history_directory(additional_dir, "additional")
    publication = pd.concat(
        [
            load_history_directory(test1_publication_dir, "publication"),
            load_history_directory(test2_publication_dir, "publication"),
        ],
        ignore_index=True,
        sort=False,
    )
    histories = pd.concat([additional, publication], ignore_index=True, sort=False)
    histories = histories[
        histories["method"].isin(["H-PINN", "HO-PINN"])
    ].copy()
    return histories


def summarized_curve(
    histories: pd.DataFrame,
    test_name: str,
    method: str,
    protocol: str,
    budget: int,
    column: str,
) -> pd.DataFrame:
    subset = histories[
        (histories["test_name"] == test_name)
        & (histories["method"] == method)
        & (histories["protocol"] == protocol)
        & (histories["n_train"] == budget)
    ]
    if column not in subset.columns:
        return pd.DataFrame(columns=["progress_percent", "mean", "std"])
    subset = subset[["progress_percent", "seed", column]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=["progress_percent", "mean", "std"])
    return (
        subset.groupby("progress_percent", as_index=False)[column]
        .agg(mean="mean", std="std")
        .sort_values("progress_percent")
    )


def draw_series(
    ax: plt.Axes,
    curve: pd.DataFrame,
    style: SeriesStyle,
    *,
    log_scale: bool,
) -> None:
    if curve.empty:
        return
    x = curve["progress_percent"].to_numpy(dtype=float)
    mean = curve["mean"].to_numpy(dtype=float)
    std = curve["std"].fillna(0.0).to_numpy(dtype=float)
    if log_scale:
        mean = np.where(mean > 0.0, mean, np.nan)
        lower = np.where(mean - std > 0.0, mean - std, np.nan)
        upper = np.where(mean + std > 0.0, mean + std, np.nan)
    else:
        lower = mean - std
        upper = mean + std
    ax.plot(
        x,
        mean,
        label=style.label,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        zorder=style.zorder,
    )
    ax.fill_between(
        x,
        lower,
        upper,
        color=style.color,
        alpha=0.055 if style.linestyle == "--" else 0.09,
        linewidth=0,
        zorder=max(1, style.zorder - 3),
    )


def common_figure(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(4, 3, figsize=(11.4, 11.7), sharex=True)
    fig.suptitle(title, fontsize=12, y=0.992)
    return fig, axes


def finish_figure(
    fig: plt.Figure,
    axes: np.ndarray,
    output_dir: Path,
    stem: str,
    note: str,
) -> None:
    handles = [
        plt.Line2D(
            [],
            [],
            color=STYLES[key].color,
            linestyle=STYLES[key].linestyle,
            linewidth=STYLES[key].linewidth,
            label=STYLES[key].label,
        )
        for key in SERIES_ORDER
    ]
    labels = [STYLES[key].label for key in SERIES_ORDER]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=4,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.8,
    )
    fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=8, color="#4A4A4A")
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.905,
        wspace=0.25,
        hspace=0.34,
    )
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_training_losses(histories: pd.DataFrame, output_dir: Path) -> None:
    columns = ["loss_data", "loss_ibc", "loss_pde"]
    titles = [r"Data loss $\mathcal{L}_{data}$", r"IBC loss $\mathcal{L}_{IBC}$", r"PDE loss $\mathcal{L}_{PDE}$"]
    fig, axes = common_figure("Training loss components")
    for row, (test_name, method) in enumerate(ROW_ORDER):
        for col, (column, title) in enumerate(zip(columns, titles)):
            ax = axes[row, col]
            for protocol, budget in SERIES_ORDER:
                if column == "loss_data" and budget == 0:
                    continue
                curve = summarized_curve(
                    histories, test_name, method, protocol, budget, column
                )
                draw_series(ax, curve, STYLES[(protocol, budget)], log_scale=True)
            ax.set_yscale("log")
            ax.set_xlim(0.0, 100.0)
            ax.grid(color="#D9D9D9", linewidth=0.45, which="both", alpha=0.8)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{ROW_LABELS[(test_name, method)]}\nLoss")
            if row == len(ROW_ORDER) - 1:
                ax.set_xlabel("Optimization progress (%)")
    finish_figure(
        fig,
        axes,
        output_dir,
        "training_loss_components_comparison",
        "Lines are seed means; bands are ±1 sample SD. Additional N=0/20: 3 seeds, 50k epochs, fixed N_r=128. Cluster N=20–1000: 10 seeds, 50k epochs, N_r=2N. N=0 has no data term.",
    )


def plot_validation_losses(histories: pd.DataFrame, output_dir: Path) -> None:
    columns = ["physics_val_data", "physics_val_ibc", "physics_val_pde"]
    titles = [
        r"Recorded data evaluation $\mathcal{L}_{data}$",
        r"Fixed-set IBC validation $\mathcal{L}_{IBC}$",
        r"Fixed-set PDE validation $\mathcal{L}_{PDE}$",
    ]
    fig, axes = common_figure("Recorded validation loss components")
    for row, (test_name, method) in enumerate(ROW_ORDER):
        for col, (column, title) in enumerate(zip(columns, titles)):
            ax = axes[row, col]
            for protocol, budget in SERIES_ORDER:
                if column == "physics_val_data" and (protocol != "additional" or budget == 0):
                    continue
                curve = summarized_curve(
                    histories, test_name, method, protocol, budget, column
                )
                draw_series(ax, curve, STYLES[(protocol, budget)], log_scale=True)
            ax.set_yscale("log")
            ax.set_xlim(0.0, 100.0)
            ax.grid(color="#D9D9D9", linewidth=0.45, which="both", alpha=0.8)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{ROW_LABELS[(test_name, method)]}\nLoss")
            if row == len(ROW_ORDER) - 1:
                ax.set_xlabel("Optimization progress (%)")
            if col == 0:
                ax.text(
                    0.04,
                    0.07,
                    "Cluster data-validation\nnot recorded",
                    transform=ax.transAxes,
                    fontsize=7.5,
                    color="#666666",
                    bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.85},
                )
    finish_figure(
        fig,
        axes,
        output_dir,
        "validation_loss_components_comparison",
        "IBC/PDE use fixed collocation sets. The additional data curve re-evaluates the same N=20 labels after each logged update; it is not an independent holdout. Cluster data-validation histories were not stored.",
    )


def plot_weights(histories: pd.DataFrame, output_dir: Path) -> None:
    columns = ["weight_data", "weight_ibc", "weight_pde"]
    titles = [r"Data weight $w_{data}$", r"IBC weight $w_{IBC}$", r"PDE weight $w_{PDE}$"]
    fig, axes = common_figure("Dynamic normalized loss weights")
    for row, (test_name, method) in enumerate(ROW_ORDER):
        for col, (column, title) in enumerate(zip(columns, titles)):
            ax = axes[row, col]
            for protocol, budget in SERIES_ORDER:
                curve = summarized_curve(
                    histories, test_name, method, protocol, budget, column
                )
                draw_series(ax, curve, STYLES[(protocol, budget)], log_scale=False)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylim(-0.035, 1.035)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(color="#D9D9D9", linewidth=0.45, alpha=0.8)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{ROW_LABELS[(test_name, method)]}\nNormalized weight")
            if row == len(ROW_ORDER) - 1:
                ax.set_xlabel("Optimization progress (%)")
    finish_figure(
        fig,
        axes,
        output_dir,
        "dynamic_loss_weights_comparison",
        "Actual coefficients used in the optimized loss; w_data+w_IBC+w_PDE=1. Lines are seed means and bands are ±1 sample SD. For N=0, w_data is structurally zero.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare loss and weight histories for Test 1 and Test 2."
    )
    parser.add_argument(
        "--additional-history-dir",
        type=Path,
        default=PAPER / "experiments/zero_label_test12_e50k_nr128/histories",
    )
    parser.add_argument(
        "--test1-publication-history-dir",
        type=Path,
        default=PAPER / "experiments/test1_corrected_candidate/histories",
    )
    parser.add_argument(
        "--test2-publication-history-dir",
        type=Path,
        default=PAPER / "experiments/corrected_test2_final/histories",
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
    histories = load_all_histories(
        args.additional_history_dir.resolve(),
        args.test1_publication_history_dir.resolve(),
        args.test2_publication_history_dir.resolve(),
    )
    expected = {
        ("additional", "test1"): 12,
        ("additional", "test2"): 12,
        ("publication", "test1"): 120,
        ("publication", "test2"): 120,
    }
    observed = (
        histories[["protocol", "test_name", "n_train", "seed", "method"]]
        .drop_duplicates()
        .groupby(["protocol", "test_name"])
        .size()
        .to_dict()
    )
    if observed != expected:
        raise RuntimeError(f"Unexpected history coverage: {observed}; expected {expected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_training_losses(histories, args.output_dir)
    plot_validation_losses(histories, args.output_dir)
    plot_weights(histories, args.output_dir)
    print(f"Figures: {args.output_dir}")


if __name__ == "__main__":
    main()
