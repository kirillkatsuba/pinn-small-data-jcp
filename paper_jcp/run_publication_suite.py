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


def common_design(seeds: str) -> list[str]:
    return [
        "--budgets", "20,50,100,200,500,1000",
        "--seeds", seeds,
        "--time-fraction", "0.7",
        "--time-bins", "10",
    ]


def pinn_configuration(
    workers: int, device: str, deterministic: bool, epochs: int
) -> list[str]:
    return [
        "--hidden-dim", "128",
        "--depth", "6",
        "--activation", "gelu",
        "--balancers", "h,ho",
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


def experiment_name(test: str, part: str) -> str:
    if test == "test1":
        return "test1_corrected_candidate"
    if test == "test2" and part == "classical":
        return "corrected_test2_classical_strong"
    return f"corrected_{test}_final"


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
) -> list[str]:
    result = [
        sys.executable,
        "-B",
        str(RUNNER),
        "--experiment-name", experiment_name(test, part),
        "--test", test,
        *corrected_data(test, data_root),
        *common_design(seeds),
    ]
    if result_suffix:
        result.extend(("--result-suffix", result_suffix))
    if part == "pinn":
        result.extend(
            (
                *pinn_configuration(workers, device, deterministic, epochs),
                "--skip-classical",
            )
        )
    elif part == "classical":
        profile, cv = (("standard", "random") if test == "test1" else ("strong", "time"))
        result.extend(
            (
                "--classical-tuning-profile", profile,
                "--classical-cv", cv,
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
) -> list[list[str]]:
    commands: list[list[str]] = []
    if part in {"pinn", "all"}:
        commands.append(
            command(
                test, "pinn", workers, device, deterministic, data_root,
                seeds, epochs, result_suffix,
            )
        )
    if part in {"classical", "all"}:
        if test == "test1":
            commands.append(
                command(
                    test, "classical", workers, device, deterministic, data_root,
                    seeds, epochs, result_suffix,
                )
            )
        else:
            # Keep CSV writes sequential and make the expensive GP stage explicit.
            commands.append(
                command(
                    test, "classical", workers, device, deterministic, data_root,
                    seeds, epochs, result_suffix, "XGB,SVM"
                )
            )
            commands.append(
                command(
                    test, "classical", workers, device, deterministic, data_root,
                    seeds, epochs, result_suffix, "GP",
                )
            )
    return commands


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
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
        ):
            print(f"\n$ {shlex.join(invocation)}", flush=True)
            if not args.dry_run:
                subprocess.run(invocation, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
