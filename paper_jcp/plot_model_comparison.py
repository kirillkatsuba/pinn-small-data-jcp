#!/usr/bin/env python3
"""Plot GT / PINN / XGB / SVM / GP for the final unified experiments.

The PINN is loaded from the saved final checkpoint. Classical models are
reconstructed from exactly the same nested labelled sample and cached after
the first fit, because their fitted estimators were not persisted originally.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_jcp.unified_protocol import (  # noqa: E402
    ModelConfig,
    build_model,
    build_unified_bundle,
    nested_heat_labels,
    nested_label_indices,
    predict_pinn,
    resolve_device,
)
from paper_jcp.classical_baselines import (  # noqa: E402
    predict_classical,
    tune_classical,
)


PAPER = ROOT / "paper_jcp"
FINAL_EXPERIMENTS = {
    "test1": "test1_corrected_candidate",
    "test2": "corrected_test2_final",
    "test3": "corrected_test3_final",
    "test4": "corrected_test4_final",
}
DATA_ROOTS = {
    "test2": PAPER / "generated" / "publication_data",
    "test3": PAPER / "generated" / "publication_data",
    "test4": PAPER / "generated" / "publication_data",
}
METHOD_LABELS = ("GT", "PINN", "XGB", "SVM", "GP")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GT/PINN/XGB/SVM/GP qualitative comparisons."
    )
    parser.add_argument("--test", required=True, choices=tuple(FINAL_EXPERIMENTS))
    parser.add_argument("--budget", type=int, default=50,
                        choices=(20, 50, 100, 200, 500, 1000))
    parser.add_argument("--seed", type=int, default=101, choices=tuple(range(101, 111)))
    parser.add_argument(
        "--epochs",
        type=int,
        default=50_000,
        help="Select the PINN checkpoint trained for this number of epochs.",
    )
    parser.add_argument("--pinn", choices=("h", "ho"), default="h")
    parser.add_argument(
        "--time-fractions", default="0,0.5,0.7,1",
        help="Comma-separated fractions of the full simulated time in [0, 1].",
    )
    parser.add_argument(
        "--test4-sample", type=int, default=0,
        help="Parameter-realisation index for Test 4 (0--99).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Default: paper_jcp/figures/final_model_comparison/<test>/.",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=PAPER / "experiments" / "qualitative_classical_cache",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--device", default="cpu", help="PINN inference device: cpu, cuda, cuda:0, mps, or auto."
    )
    parser.add_argument(
        "--force-refit-classical", action="store_true",
        help="Ignore cached XGB/SVM/GP estimators and tune them again.",
    )
    return parser.parse_args()


def fractions(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError("time-fractions must contain values in [0, 1]")
    return list(dict.fromkeys(result))


def resolved_data_root(test: str) -> Path:
    publication_root = PAPER / "generated" / "publication_data"
    target = publication_root / test
    if target.exists():
        return publication_root
    return DATA_ROOTS[test]


def build_bundle(test: str):
    if test == "test1":
        return build_unified_bundle(test, time_fraction=0.7, data_source="legacy")
    return build_unified_bundle(
        test,
        time_fraction=0.7,
        data_source="corrected",
        corrected_data_root=resolved_data_root(test),
        corrected_test4_pool_size=100_000,
    )


def training_labels(bundle, budget: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if bundle.test_name == "test1":
        x_all, y_all = nested_heat_labels(
            bundle, seed, max_budget=1000, time_bins=10
        )
        return x_all[:budget], y_all[:budget]
    indices = nested_label_indices(bundle, seed, max_budget=1000, time_bins=10)
    assert bundle.features is not None and bundle.targets is not None
    return bundle.features[indices[:budget]], bundle.targets[indices[:budget]]


def checkpoint_path(
    test: str, budget: int, seed: int, balance_mode: str, epochs: int
) -> Path:
    directory = PAPER / "experiments" / FINAL_EXPERIMENTS[test] / "checkpoints"
    candidates = sorted(directory.glob(f"{test}_n{budget}_s{seed}_*.pth"))
    matching = []
    for path in candidates:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        trained_epochs = payload.get("train_config", {}).get("epochs")
        if payload.get("balance_mode") == balance_mode and trained_epochs == epochs:
            matching.append(path)
    if len(matching) != 1:
        raise RuntimeError(
            f"Expected one {balance_mode} checkpoint for {test}, N={budget}, "
            f"seed={seed}, epochs={epochs}; found {matching}"
        )
    return matching[0]


def load_pinn(bundle, args: argparse.Namespace):
    path = checkpoint_path(
        args.test, args.budget, args.seed, args.pinn, args.epochs
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    device = resolve_device(args.device)
    model = build_model(bundle, ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, path


def classical_protocol(test: str) -> tuple[str, str]:
    # Matches the result tables: Test 1 predates the strong time-aware sweep.
    return ("standard", "random") if test == "test1" else ("strong", "time")


def fit_or_load_classical(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, object]:
    profile, cv = classical_protocol(args.test)
    cache = (
        args.cache_dir
        / f"{args.test}_n{args.budget}_s{args.seed}_{profile}_{cv}.joblib"
    )
    if cache.exists() and not args.force_refit_classical:
        print(f"[cache] {cache}", flush=True)
        return joblib.load(cache)
    estimators = {}
    for method in ("XGB", "SVM", "GP"):
        print(f"[fit] {method}: profile={profile}, cv={cv}", flush=True)
        fitted, _, _ = tune_classical(
            method,
            x_train,
            y_train,
            args.seed,
            tuning_profile=profile,
            cv_strategy=cv,
            return_metadata=True,
        )
        estimators[method] = fitted
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimators, cache)
    print(f"[saved] {cache}", flush=True)
    return estimators


def load_flow_slice(args: argparse.Namespace, fraction: float):
    data_root = resolved_data_root(args.test)
    if args.test == "test4":
        path = data_root / "test4" / f"sample_{args.test4_sample:03d}.npz"
    else:
        path = data_root / args.test / "solution.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"])
        x = np.asarray(data["x"])
        y = np.asarray(data["y"])
        index = int(round(fraction * (len(times) - 1)))
        index = max(0, min(index, len(times) - 1))
        truth = np.stack(
            (
                data["pressure"][index],
                data["oil_saturation"][index],
                data["water_saturation"][index],
            ),
            axis=-1,
        ).astype(np.float32)
        metadata = json.loads(str(data["metadata_json"].item()))
    tt, xx, yy = np.meshgrid([times[index]], x, y, indexing="ij")
    features = np.column_stack((tt.ravel(), xx.ravel(), yy.ravel()))
    if args.test == "test4":
        relative = metadata["relative_permeability"]
        parameters = np.asarray(
            [
                relative["water_exponent"],
                relative["oil_exponent"],
                relative["water_endpoint"],
                relative["oil_endpoint"],
            ],
            dtype=np.float32,
        )
        features = np.column_stack(
            (features, np.broadcast_to(parameters, (len(features), 4)))
        )
    return path, float(times[index]), x, y, truth, features.astype(np.float32)


def plot_flow(
    args: argparse.Namespace,
    bundle,
    pinn,
    classical: dict[str, object],
    output_dir: Path,
) -> list[Path]:
    outputs = []
    field_rows = ((0, "Pressure", "viridis"), (2, r"Water saturation $S_w$", "cividis"),
                  (1, r"Oil saturation $S_o$", "cividis"))
    for fraction in fractions(args.time_fractions):
        path, time, x, y, truth, features = load_flow_slice(args, fraction)
        nx, ny = len(x), len(y)
        predictions = {"GT": truth}
        predictions["PINN"] = predict_pinn(args.test, pinn, features).reshape(nx, ny, 3)
        for method in ("XGB", "SVM", "GP"):
            predictions[method] = predict_classical(
                classical[method], features, scalar=False
            ).reshape(nx, ny, 3)

        fig, axes = plt.subplots(
            3, 5, figsize=(17.0, 9.4), constrained_layout=True, squeeze=False
        )
        extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
        for row, (field, label, cmap) in enumerate(field_rows):
            if field == 0:
                vmin = float(np.min(truth[:, :, field]))
                vmax = float(np.max(truth[:, :, field]))
            else:
                vmin, vmax = 0.0, 1.0
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-12
            image = None
            for column, method in enumerate(METHOD_LABELS):
                axis = axes[row, column]
                image = axis.imshow(
                    predictions[method][:, :, field].T,
                    origin="lower",
                    extent=extent,
                    aspect="equal",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    axis.set_title(method, fontsize=12, fontweight="bold")
                axis.set_xlabel("x")
                if column == 0:
                    axis.set_ylabel(f"{label}\ny")
                else:
                    axis.set_ylabel("y")
            assert image is not None
            fig.colorbar(image, ax=axes[row, :], fraction=0.018, pad=0.01)
        suffix = f"_sample{args.test4_sample:03d}" if args.test == "test4" else ""
        fig.suptitle(
            f"{args.test.upper()}, N={args.budget}, seed={args.seed}, t={time:.5g}"
            f"{suffix} | source: {path.name}",
            fontsize=13,
        )
        destination = output_dir / f"{args.test}_n{args.budget}_s{args.seed}_t{time:.6g}{suffix}.png"
        fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(destination)
        outputs.append(destination)
    return outputs


def plot_heat(
    args: argparse.Namespace,
    bundle,
    pinn,
    classical: dict[str, object],
    output_dir: Path,
) -> list[Path]:
    selected = fractions(args.time_fractions)
    times = np.asarray(selected, dtype=np.float32) * bundle.full_time
    space = np.linspace(0.0, 1.0, 512, dtype=np.float32)
    columns = 2
    rows = math.ceil(len(times) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11.5, 3.7 * rows),
                             constrained_layout=True, squeeze=False)
    colors = {"GT": "black", "PINN": "#c0392b", "XGB": "#4c78a8",
              "SVM": "#59a14f", "GP": "#9467bd"}
    styles = {"GT": "-", "PINN": "--", "XGB": "-.", "SVM": ":", "GP": "--"}
    for axis, time in zip(axes.flat, times, strict=False):
        features = np.column_stack((np.full_like(space, time), space)).astype(np.float32)
        truth = bundle.module.thermal_conductivity_equation(
            torch.tensor(features[:, 0]), torch.tensor(features[:, 1])
        ).detach().numpy()
        curves = {"GT": truth, "PINN": predict_pinn("test1", pinn, features)}
        for method in ("XGB", "SVM", "GP"):
            curves[method] = predict_classical(classical[method], features, scalar=True)
        for method in METHOD_LABELS:
            axis.plot(space, curves[method], label=method, color=colors[method],
                      linestyle=styles[method], linewidth=2.3 if method in {"GT", "PINN"} else 1.7)
        axis.set_title(f"t={float(time):.5g}")
        axis.set_xlabel("x")
        axis.set_ylabel("u(t,x)")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(times):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=5, fontsize=9)
    fig.suptitle(f"TEST1, N={args.budget}, seed={args.seed}", fontsize=13)
    destination = output_dir / f"test1_n{args.budget}_s{args.seed}_profiles.png"
    fig.savefig(destination, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(destination)
    return [destination]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        PAPER / "figures" / "final_model_comparison" / args.test
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle(args.test)
    x_train, y_train = training_labels(bundle, args.budget, args.seed)
    pinn, checkpoint = load_pinn(bundle, args)
    print(f"[checkpoint] {checkpoint}", flush=True)
    classical = fit_or_load_classical(args, x_train, y_train)
    if args.test == "test1":
        plot_heat(args, bundle, pinn, classical, output_dir)
    else:
        plot_flow(args, bundle, pinn, classical, output_dir)


if __name__ == "__main__":
    main()
