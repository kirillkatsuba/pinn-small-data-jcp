"""Create publication-oriented diagnostic plots for a corrected FV dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_jcp.two_phase_fv import load_result  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot corrected two-phase FV output.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--time-fractions",
        default="0,0.25,0.5,1",
        help="Comma-separated fractions of the simulated time interval.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def _selected_indices(times: np.ndarray, value: str) -> list[int]:
    fractions = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not fractions or any(item < 0.0 or item > 1.0 for item in fractions):
        raise ValueError("time-fractions must contain values in [0, 1]")
    indices = [int(round(item * (len(times) - 1))) for item in fractions]
    return list(dict.fromkeys(indices))


def plot_fields(result, indices: list[int], output: Path, dpi: int) -> None:
    fields = (
        (result.pressure, "Pressure", "full"),
        (result.water_saturation, r"Water saturation $S_w$", "water"),
        (result.oil_saturation, r"Oil saturation $S_o$", "oil"),
    )
    extent = [result.x[0], result.x[-1], result.y[0], result.y[-1]]
    figure, axes = plt.subplots(
        len(fields), len(indices), figsize=(3.7 * len(indices), 9.0), squeeze=False,
        constrained_layout=True,
    )
    for row, (history, label, scale_mode) in enumerate(fields):
        selected = history[indices]
        if scale_mode == "water":
            vmin, vmax = 0.0, float(np.max(selected))
        elif scale_mode == "oil":
            vmin, vmax = float(np.min(selected)), 1.0
        else:
            vmin, vmax = float(np.min(selected)), float(np.max(selected))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0e-12
        for column, time_index in enumerate(indices):
            axis = axes[row, column]
            image = axis.imshow(
                history[time_index].T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="viridis" if row == 0 else "cividis",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_title(f"$t={result.times[time_index]:.4g}$")
            axis.set_xlabel("x")
            if column == 0:
                axis.set_ylabel(f"{label}\ny")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_profiles(result, indices: list[int], output: Path, dpi: int) -> None:
    histories = (
        (result.pressure, "Pressure"),
        (result.water_saturation, r"$S_w$"),
        (result.oil_saturation, r"$S_o$"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(indices)))
    for axis, (history, label) in zip(axes, histories, strict=True):
        profiles = np.mean(history, axis=2)
        for color, time_index in zip(colors, indices, strict=True):
            axis.plot(
                result.x,
                profiles[time_index],
                color=color,
                linewidth=2.0,
                label=f"t={result.times[time_index]:.4g}",
            )
        axis.set_xlabel("x")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_time_series(result, output: Path, dpi: int) -> None:
    weights = result.porosity / np.sum(result.porosity)
    mean_water = np.sum(result.water_saturation * weights[None, :, :], axis=(1, 2))
    maximum_water = np.max(result.water_saturation, axis=(1, 2))
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.plot(result.times, mean_water, linewidth=2.2, label=r"Pore-volume mean $S_w$")
    axis.plot(
        result.times,
        maximum_water,
        linewidth=1.7,
        linestyle="--",
        label=r"Maximum $S_w$",
    )
    axis.set_xlabel("Time")
    axis.set_ylabel(r"Water saturation $S_w$")
    upper = max(float(np.max(maximum_water)) * 1.08, 1.0e-6)
    axis.set_ylim(-0.02 * upper, upper)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    result = load_result(args.input)
    indices = _selected_indices(result.times, args.time_fractions)
    output_dir = args.output_dir or args.input.parent / f"{args.input.stem}_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fields": output_dir / "fields.png",
        "profiles": output_dir / "profiles.png",
        "time_series": output_dir / "time_series.png",
    }
    plot_fields(result, indices, outputs["fields"], args.dpi)
    plot_profiles(result, indices, outputs["profiles"], args.dpi)
    plot_time_series(result, outputs["time_series"], args.dpi)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
