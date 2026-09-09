#!/usr/bin/env python3
"""Plot all final cluster PINN training-loss histories for Tests 1--4."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
EXPERIMENTS = {
    "test1": "test1_corrected_candidate",
    "test2": "corrected_test2_final",
    "test3": "corrected_test3_final",
    "test4": "corrected_test4_final",
}
TEST_LABELS = {
    "test1": "Test 1: heat equation",
    "test2": "Test 2: homogeneous flow",
    "test3": "Test 3: heterogeneous flow",
    "test4": "Test 4: parametric flow",
}
BUDGETS = (20, 50, 100, 200, 500, 1000)
SEEDS = tuple(range(101, 111))
METHODS = ("H-PINN", "HO-PINN")
MODE_LABEL = {"h": "H-PINN", "ho": "HO-PINN"}
COLORS = {
    "loss_data": "#0072B2",
    "loss_ibc": "#E69F00",
    "loss_pde": "#009E73",
    "balanced_total": "#202020",
}
CURVE_LABELS = {
    "loss_data": r"$L_{data}$",
    "loss_ibc": r"$L_{IBC}$",
    "loss_pde": r"$L_{PDE}$",
    "balanced_total": r"$L_{balanced}$",
}
LINESTYLES = {
    "loss_data": "-",
    "loss_ibc": "-",
    "loss_pde": "-",
    "balanced_total": "--",
}
METHOD_COLORS = {"H-PINN": "#D55E00", "HO-PINN": "#0072B2"}
OUTPUT = PAPER / "generated" / "cluster_loss_curves"
FILENAME = re.compile(
    r"^(test[1-4])_n(?P<budget>\d+)_s(?P<seed>\d+)_"
    r"(?:corrected(?:_pool\d+)?_)?(?P<mode>h|ho)_w.*\.csv$"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 260,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def discover_histories() -> dict[tuple[str, int, int, str], Path]:
    found: dict[tuple[str, int, int, str], Path] = {}
    for test, experiment in EXPERIMENTS.items():
        directory = PAPER / "experiments" / experiment / "histories"
        for path in directory.glob("*.csv"):
            match = FILENAME.match(path.name)
            if not match:
                continue
            key = (
                test,
                int(match.group("budget")),
                int(match.group("seed")),
                MODE_LABEL[match.group("mode")],
            )
            if key in found:
                raise RuntimeError(f"Duplicate history for {key}: {path}")
            found[key] = path
    expected = {
        (test, budget, seed, method)
        for test in EXPERIMENTS
        for budget in BUDGETS
        for seed in SEEDS
        for method in METHODS
    }
    missing = expected.difference(found)
    extra = set(found).difference(expected)
    if missing or extra:
        raise RuntimeError(
            f"History coverage failure: missing={len(missing)}, extra={len(extra)}"
        )
    return found


def read_history(path: Path) -> pd.DataFrame:
    columns = ["epoch", "loss_data", "loss_ibc", "loss_pde", "balanced_total"]
    history = pd.read_csv(path, usecols=columns).sort_values("epoch")
    if history.empty or int(history.epoch.iloc[-1]) != 50_000:
        raise RuntimeError(f"Incomplete history: {path}")
    if history[columns].isna().any().any():
        raise RuntimeError(f"Non-finite loss history: {path}")
    return history


def load_histories(
    paths: dict[tuple[str, int, int, str], Path]
) -> dict[tuple[str, int, int, str], pd.DataFrame]:
    return {key: read_history(path) for key, path in paths.items()}


def loss_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    values = np.concatenate(
        [
            frame[["loss_data", "loss_ibc", "loss_pde", "balanced_total"]]
            .to_numpy()
            .reshape(-1)
            for frame in frames
        ]
    )
    values = values[np.isfinite(values) & (values > 0)]
    lower = 10.0 ** np.floor(np.log10(np.quantile(values, 0.001)))
    upper = 10.0 ** np.ceil(np.log10(np.quantile(values, 0.999)))
    if lower >= upper:
        lower, upper = float(values.min() * 0.8), float(values.max() * 1.2)
    return max(lower, 1e-14), upper


def plot_seed_grid(
    test: str,
    budget: int,
    method: str,
    histories: dict[tuple[str, int, int, str], pd.DataFrame],
) -> Path:
    frames = [histories[(test, budget, seed, method)] for seed in SEEDS]
    y_limits = loss_limits(frames)
    figure, axes = plt.subplots(
        2, 5, figsize=(15.5, 6.2), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, seed, history in zip(axes.flat, SEEDS, frames):
        for column in ("loss_data", "loss_ibc", "loss_pde", "balanced_total"):
            values = np.maximum(history[column].to_numpy(), 1e-14)
            ax.plot(
                history.epoch,
                values,
                color=COLORS[column],
                linestyle=LINESTYLES[column],
                linewidth=1.0 if column != "balanced_total" else 1.35,
                alpha=0.88,
                label=CURVE_LABELS[column],
            )
        ax.set_yscale("log")
        ax.set_ylim(y_limits)
        ax.set_xlim(0, 50_000)
        ax.grid(True, which="both", alpha=0.18, linewidth=0.55)
        ax.set_title(f"Seed {seed}")
        if ax in axes[-1, :]:
            ax.set_xlabel("Epoch")
        if ax in axes[:, 0]:
            ax.set_ylabel("Training loss")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    figure.suptitle(
        f"{TEST_LABELS[test]} — {method}, N={budget}",
        fontsize=13,
        y=1.065,
    )
    destination = OUTPUT / "detailed" / test / f"n{budget}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{method.lower().replace('-', '_')}_loss_curves.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def aligned_stack(frames: list[pd.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray]:
    epoch = frames[0].epoch.to_numpy()
    values = []
    for frame in frames:
        if not np.array_equal(frame.epoch.to_numpy(), epoch):
            raise RuntimeError("Histories use different logging epochs")
        values.append(np.maximum(frame[column].to_numpy(), 1e-14))
    return epoch, np.vstack(values)


def plot_test_summary(
    test: str,
    histories: dict[tuple[str, int, int, str], pd.DataFrame],
) -> Path:
    figure, axes = plt.subplots(
        2, 3, figsize=(12.6, 7.0), sharex=True, constrained_layout=True
    )
    for ax, budget in zip(axes.flat, BUDGETS):
        for method in METHODS:
            frames = [histories[(test, budget, seed, method)] for seed in SEEDS]
            epoch, values = aligned_stack(frames, "balanced_total")
            median = np.median(values, axis=0)
            lower = np.quantile(values, 0.10, axis=0)
            upper = np.quantile(values, 0.90, axis=0)
            ax.plot(
                epoch,
                median,
                color=METHOD_COLORS[method],
                linewidth=1.55,
                label=method,
            )
            ax.fill_between(
                epoch,
                lower,
                upper,
                color=METHOD_COLORS[method],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_yscale("log")
        ax.set_xlim(0, 50_000)
        ax.grid(True, which="both", alpha=0.18, linewidth=0.55)
        ax.set_title(f"N={budget}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$L_{balanced}$")
    axes[0, 0].legend(loc="upper right", frameon=False)
    figure.suptitle(
        f"{TEST_LABELS[test]} — balanced training loss",
        fontsize=13,
        y=1.02,
    )
    destination = OUTPUT / "summary"
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / f"{test}_balanced_loss_summary.png"
    pdf = destination / f"{test}_balanced_loss_summary.pdf"
    figure.savefig(png, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return png


def write_endpoints(
    histories: dict[tuple[str, int, int, str], pd.DataFrame]
) -> Path:
    rows = []
    for (test, budget, seed, method), history in sorted(histories.items()):
        row: dict[str, object] = {
            "test_name": test,
            "n_train": budget,
            "seed": seed,
            "method": method,
        }
        for column in ("loss_data", "loss_ibc", "loss_pde", "balanced_total"):
            row[f"{column}_initial"] = float(history[column].iloc[0])
            row[f"{column}_final"] = float(history[column].iloc[-1])
            row[f"{column}_min_logged"] = float(history[column].min())
        rows.append(row)
    path = OUTPUT / "loss_endpoints.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_index(detailed: list[Path], summaries: list[Path]) -> Path:
    relative = {path.relative_to(OUTPUT).as_posix(): path for path in detailed}
    lines = [
        "# Cluster PINN training-loss curves",
        "",
        "Each detailed figure contains ten panels for seeds 101–110. Curves show "
        "the unweighted data, initial/boundary-condition, and PDE losses together "
        "with the dynamically weighted total training loss. All y-axes are logarithmic.",
        "",
        "Summary figures show the median balanced loss across seeds; shaded regions "
        "span the 10th–90th percentiles.",
        "",
        "## Summary",
        "",
    ]
    for path in summaries:
        lines.append(f"- [{path.stem}](summary/{path.name})")
    lines.extend(["", "## Detailed seed-wise figures", ""])
    for test in EXPERIMENTS:
        lines.extend([f"### {TEST_LABELS[test]}", ""])
        lines.append("| N | H-PINN | HO-PINN |")
        lines.append("|---:|---|---|")
        for budget in BUDGETS:
            h = f"detailed/{test}/n{budget}/h_pinn_loss_curves.png"
            ho = f"detailed/{test}/n{budget}/ho_pinn_loss_curves.png"
            if h not in relative or ho not in relative:
                raise RuntimeError(f"Missing detailed figure for {test}, N={budget}")
            lines.append(f"| {budget} | [H-PINN]({h}) | [HO-PINN]({ho}) |")
        lines.append("")
    path = OUTPUT / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    configure_style()
    paths = discover_histories()
    histories = load_histories(paths)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detailed = []
    for test in EXPERIMENTS:
        for budget in BUDGETS:
            for method in METHODS:
                detailed.append(plot_seed_grid(test, budget, method, histories))
    summaries = [plot_test_summary(test, histories) for test in EXPERIMENTS]
    endpoints = write_endpoints(histories)
    index = write_index(detailed, summaries)
    print(f"Histories checked: {len(histories)}")
    print(f"Detailed figures: {len(detailed)}")
    print(f"Summary figures: {len(summaries)}")
    print(index)
    print(endpoints)


if __name__ == "__main__":
    main()
