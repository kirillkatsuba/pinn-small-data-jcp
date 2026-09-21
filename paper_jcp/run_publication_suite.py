#!/usr/bin/env python3
"""Run the current candidate protocol for Tests 1--4.

This is a Python-only orchestration layer over ``run_unified_protocol.py``.
Every command is resumable because the underlying runner skips complete rows.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "paper_jcp" / "run_unified_protocol.py"
DEFAULT_DATA_ROOT = ROOT / "paper_jcp" / "generated" / "publication_data"
TESTS = ("test1", "test2", "test3", "test4")
DEFAULT_SEEDS = "101,102,103,104,105,106,107,108,109,110"
DEFAULT_EPOCHS = 50_000
DEFAULT_BUDGETS = "20,50,100,200,500,1000"
DEFAULT_BALANCERS = "h,ho"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final JCP small-data experiment suite."
    )
    parser.add_argument("--test", choices=(*TESTS, "all"), required=True)
    parser.add_argument("--part", choices=("pinn", "classical", "all"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--scheduler-epochs",
        type=int,
        default=None,
        help="Optional cosine-scheduler horizon independent of the stop epoch.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        default="",
        help="Comma-separated periodic checkpoint epochs for PINN runs.",
    )
    parser.add_argument(
        "--experiment-tag",
        default="",
        help="Optional suffix that isolates outputs from the main publication runs.",
    )
    parser.add_argument(
        "--budgets",
        default=DEFAULT_BUDGETS,
        help="Comma-separated label budgets selected from 20,50,100,200,500,1000.",
    )
    parser.add_argument(
        "--balancers",
        default=DEFAULT_BALANCERS,
        help="Comma-separated PINN loss modes selected from vanilla,h,ho.",
    )
    parser.add_argument(
        "--classical-tuning-profile",
        choices=("standard", "strong", "extensive"),
        default="strong",
        help="Hyperparameter-search size for classical models.",
    )
    parser.add_argument(
        "--classical-cv",
        choices=("random", "time"),
        default="time",
        help="Use expanding-window time CV for the final interpolation/extrapolation study.",
    )
    parser.add_argument(
        "--classical-methods",
        default="XGB,SVM,GP",
        help="Comma-separated subset of XGB,SVM,GP for staged classical runs.",
    )
    parser.add_argument(
        "--result-suffix",
        default="",
        help="Optional suffix for a per-job result CSV shard, e.g. seed101.",
    )
    parser.add_argument(
        "--device", default="cpu", help="PINN device: cpu, cuda, cuda:0, mps, or auto."
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic PyTorch kernels when available.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running them."
    )
    return parser.parse_args()


def common_design(seeds: str, budgets: str) -> list[str]:
    return [
        "--budgets", budgets,
        "--seeds", seeds,
        "--time-fraction", "0.7",
        "--time-bins", "10",
    ]


def pinn_configuration(
    workers: int,
    device: str,
    deterministic: bool,
    epochs: int,
    balancers: str,
    checkpoint_epochs: str = "",
    scheduler_epochs: int | None = None,
) -> list[str]:
    result = [
        "--hidden-dim", "128",
        "--depth", "6",
        "--activation", "gelu",
        "--balancers", balancers,
        "--l0-data", "0.03",
        "--l0-ibc", "0.003",
        "--orthogonal-floor", "0.25",
        "--optimizer", "adam",
        "--learning-rate", "0.003",
        "--scheduler", "cosine",
        "--min-learning-rate", "0.00001",
        "--epochs", str(epochs),
        "--collocation-ratio", "2",
        "--residual-domain", "full",
        "--gradient-clip", "1.0",
        "--checkpoint-criterion", "balanced",
        "--physics-validation-size", "512",
        "--device", device,
        "--deterministic" if deterministic else "--no-deterministic",
        "--workers", str(workers),
    ]
    if checkpoint_epochs:
        result.extend(("--checkpoint-epochs", checkpoint_epochs))
    if scheduler_epochs is not None:
        result.extend(("--scheduler-epochs", str(scheduler_epochs)))
    return result


def corrected_data(test: str, data_root: Path) -> list[str]:
    if test == "test1":
        return []
    result = [
        "--data-source", "corrected",
        "--corrected-data-root", str(data_root),
    ]
    if test == "test4":
        result.extend(("--corrected-test4-pool-size", "100000"))
    return result


def experiment_name(
    test: str,
    part: str,
    classical_profile: str = "strong",
    classical_cv: str = "time",
    experiment_tag: str = "",
) -> str:
    if part == "classical":
        base = f"{test}_classical_{classical_profile}_{classical_cv}"
    elif test == "test1":
        base = "test1_corrected_candidate"
    else:
        base = f"corrected_{test}_final"
    return f"{base}_{experiment_tag}" if experiment_tag else base


def command(
    test: str,
    part: str,
    workers: int,
    device: str,
    deterministic: bool,
    data_root: Path,
    seeds: str,
    epochs: int,
    result_suffix: str,
    methods: str | None = None,
    budgets: str = DEFAULT_BUDGETS,
    balancers: str = DEFAULT_BALANCERS,
    classical_profile: str = "strong",
    classical_cv: str = "time",
    checkpoint_epochs: str = "",
    experiment_tag: str = "",
    scheduler_epochs: int | None = None,
) -> list[str]:
    result = [
        sys.executable,
        "-B",
        str(RUNNER),
        "--experiment-name", experiment_name(
            test, part, classical_profile, classical_cv, experiment_tag
        ),
        "--test", test,
        *corrected_data(test, data_root),
        *common_design(seeds, budgets),
    ]
    if result_suffix:
        result.extend(("--result-suffix", result_suffix))
    if part == "pinn":
        result.extend(
            (
                *pinn_configuration(
                    workers,
                    device,
                    deterministic,
                    epochs,
                    balancers,
                    checkpoint_epochs,
                    scheduler_epochs,
                ),
                "--skip-classical",
            )
        )
    elif part == "classical":
        result.extend(
            (
                "--classical-tuning-profile", classical_profile,
                "--classical-cv", classical_cv,
                "--classical-methods", methods or "XGB,SVM,GP",
                "--skip-pinn",
                "--workers", "1",
            )
        )
    else:
        raise ValueError(part)
    return result


def commands_for(
    test: str,
    part: str,
    workers: int,
    device: str,
    deterministic: bool,
    data_root: Path,
    seeds: str,
    epochs: int,
    result_suffix: str,
    budgets: str,
    balancers: str,
    classical_profile: str,
    classical_cv: str,
    classical_methods: str,
    checkpoint_epochs: str = "",
    experiment_tag: str = "",
    scheduler_epochs: int | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    if part in {"pinn", "all"}:
        commands.append(
            command(
                test, "pinn", workers, device, deterministic, data_root,
                seeds, epochs, result_suffix, budgets=budgets,
                balancers=balancers,
                checkpoint_epochs=checkpoint_epochs,
                experiment_tag=experiment_tag,
                scheduler_epochs=scheduler_epochs,
            )
        )
    if part in {"classical", "all"}:
        selected_methods = tuple(
            value.strip().upper()
            for value in classical_methods.split(",")
            if value.strip()
        )
        if test == "test1":
            commands.append(
                command(
                    test, "classical", workers, device, deterministic, data_root,
                    seeds, epochs, result_suffix, budgets=budgets,
                    balancers=balancers, classical_profile=classical_profile,
                    classical_cv=classical_cv,
                    experiment_tag=experiment_tag,
                    scheduler_epochs=scheduler_epochs,
                    methods=",".join(selected_methods),
                )
            )
        else:
            # Keep CSV writes sequential and make the expensive GP stage explicit.
            non_gp = tuple(method for method in selected_methods if method != "GP")
            if non_gp:
                commands.append(
                    command(
                        test, "classical", workers, device, deterministic, data_root,
                        seeds, epochs, result_suffix, ",".join(non_gp), budgets,
                        balancers, classical_profile, classical_cv,
                        checkpoint_epochs, experiment_tag,
                        scheduler_epochs,
                    )
                )
            if "GP" in selected_methods:
                commands.append(
                    command(
                        test, "classical", workers, device, deterministic, data_root,
                        seeds, epochs, result_suffix, "GP", budgets, balancers,
                        classical_profile, classical_cv,
                        checkpoint_epochs, experiment_tag,
                        scheduler_epochs,
                    )
                )
    return commands


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not args.budgets.strip():
        raise ValueError("budgets must not be empty")
    if not args.balancers.strip():
        raise ValueError("balancers must not be empty")
    if args.experiment_tag and not all(
        character.isalnum() or character in "_-"
        for character in args.experiment_tag
    ):
        raise ValueError("experiment tag may contain only letters, digits, '_' and '-'")
    classical_methods = tuple(
        value.strip().upper()
        for value in args.classical_methods.split(",")
        if value.strip()
    )
    if not classical_methods or any(
        value not in {"XGB", "SVM", "GP"} for value in classical_methods
    ):
        raise ValueError("classical methods must be selected from XGB,SVM,GP")
    selected = TESTS if args.test == "all" else (args.test,)
    for test in selected:
        for invocation in commands_for(
            test,
            args.part,
            args.workers,
            args.device,
            args.deterministic,
            args.data_root.resolve(),
            args.seeds,
            args.epochs,
            args.result_suffix,
            args.budgets,
            args.balancers,
            args.classical_tuning_profile,
            args.classical_cv,
            ",".join(classical_methods),
            args.checkpoint_epochs,
            args.experiment_tag,
            args.scheduler_epochs,
        ):
            print(f"\n$ {shlex.join(invocation)}", flush=True)
            if not args.dry_run:
                subprocess.run(invocation, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
