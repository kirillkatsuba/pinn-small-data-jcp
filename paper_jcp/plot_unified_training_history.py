#!/usr/bin/env python3
"""Plot raw losses, weights, contributions, validation physics, and LR."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PAPER = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--test", required=True, choices=("test1", "test2", "test3", "test4"))
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    history_dir = PAPER / "experiments" / args.experiment_name / "histories"
    pattern = f"{args.test}_n{args.budget}_s{args.seed}_*.csv"
    paths = sorted(history_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No histories matching {history_dir / pattern}")
    output_dir = PAPER / "figures" / args.experiment_name / "training_histories"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        table = pd.read_csv(path)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

        for column in ("loss_data", "loss_ibc", "loss_pde", "raw_total", "balanced_total"):
            axes[0, 0].plot(table["epoch"], table[column], label=column)
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Raw and total training losses")
        axes[0, 0].legend(fontsize=7)

        for column in ("weight_data", "weight_ibc", "weight_pde"):
            axes[0, 1].plot(table["epoch"], table[column], label=column)
        axes[0, 1].set_title("Dynamic loss weights")
        axes[0, 1].legend(fontsize=8)

        for column in ("weighted_data", "weighted_ibc", "weighted_pde"):
            axes[0, 2].plot(table["epoch"], table[column], label=column)
        axes[0, 2].set_yscale("log")
        axes[0, 2].set_title("Weighted loss contributions")
        axes[0, 2].legend(fontsize=8)

        axes[1, 0].plot(table["epoch"], table["loss_pde"], label="resampled train PDE")
        axes[1, 0].plot(table["epoch"], table["physics_val_pde"], label="fixed validation PDE")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_title("Physics generalisation")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].plot(table["epoch"], table["gradient_norm_before_clip"])
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_title("Gradient norm before clipping")

        axes[1, 2].plot(table["epoch"], table["learning_rate"])
        axes[1, 2].set_yscale("log")
        axes[1, 2].set_title("Learning rate")

        for ax in axes.flat:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
        fig.suptitle(path.stem, fontsize=10)
        destination = output_dir / f"{path.stem}.png"
        fig.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(destination)


if __name__ == "__main__":
    main()
