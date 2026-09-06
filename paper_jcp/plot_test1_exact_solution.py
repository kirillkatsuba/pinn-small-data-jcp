"""Plot the analytical heat-equation solution used in unified Test 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def exact_solution(time: np.ndarray, space: np.ndarray) -> np.ndarray:
    """u(t, x) = 2 + exp(-4 pi^2 t) sin(2 pi x) + exp(-16 pi^2 t) cos(4 pi x)."""
    t, x = np.meshgrid(time, space, indexing="ij")
    return (
        2.0
        + np.exp(-4.0 * np.pi**2 * t) * np.sin(2.0 * np.pi * x)
        + np.exp(-16.0 * np.pi**2 * t) * np.cos(4.0 * np.pi * x)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the exact heat-equation solution from Test 1."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_jcp/figures/test1_exact_solution.png"),
    )
    parser.add_argument("--final-time", type=float, default=0.05)
    parser.add_argument(
        "--time-fractions",
        default="0,0.02,0.05,0.1,0.25,0.5,0.7,1",
        help="Comma-separated fractions of final time in [0, 1].",
    )
    parser.add_argument("--space-points", type=int, default=512)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_time <= 0.0:
        raise ValueError("final-time must be positive")
    fractions = np.asarray(
        [float(value.strip()) for value in args.time_fractions.split(",") if value.strip()]
    )
    if not len(fractions) or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("time-fractions must be non-empty and lie in [0, 1]")
    space = np.linspace(0.0, 1.0, args.space_points)
    times = fractions * args.final_time
    solution = exact_solution(times, space)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.04, 0.96, len(times)))
    for time, values, color in zip(times, solution, colors, strict=True):
        axes[0].plot(space, values, color=color, linewidth=2.0, label=fr"$t={time:.4g}$")
    axes[0].set(xlabel="$x$", ylabel="$u(t,x)$", title="Exact heat-equation profiles")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    image = axes[1].imshow(
        exact_solution(np.linspace(0.0, args.final_time, 300), space).T,
        origin="lower",
        aspect="auto",
        extent=[0.0, args.final_time, 0.0, 1.0],
        cmap="viridis",
    )
    axes[1].set(xlabel="$t$", ylabel="$x$", title="Exact solution over space and time")
    fig.colorbar(image, ax=axes[1], label="$u(t,x)$")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
