#!/usr/bin/env python3
"""Merge independent Slurm seed shards into the canonical experiment CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper_jcp" / "generated"
DEFAULT_SEEDS = tuple(range(101, 111))
EXPERIMENTS = (
    ("test1_corrected_candidate", "test1"),
    ("corrected_test2_final", "test2"),
    ("corrected_test2_classical_strong", "test2"),
    ("corrected_test3_final", "test3"),
    ("corrected_test4_final", "test4"),
)
IDENTITY_COLUMNS = (
    "test_name",
    "seed",
    "n_train",
    "method",
    "config_id",
    "evaluation_region",
)


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def merge_experiment(directory: str, test: str, seeds: tuple[int, ...]) -> Path:
    experiment_dir = GENERATED / directory
    shards: list[Path] = []
    missing: list[str] = []
    for seed in seeds:
        stem = f"unified_{test}_seed{seed}"
        seed_shards = sorted(
            path
            for path in experiment_dir.glob(f"{stem}*.csv")
            if path.stem == stem or path.stem.startswith(f"{stem}_")
        )
        if seed_shards:
            shards.extend(seed_shards)
        else:
            missing.append(str(experiment_dir / f"{stem}.csv"))
    if missing:
        formatted = "\n".join(missing)
        raise FileNotFoundError(f"Missing result shards:\n{formatted}")

    tables = [pd.read_csv(path) for path in shards]
    merged = pd.concat(tables, ignore_index=True)
    identity = [column for column in IDENTITY_COLUMNS if column in merged.columns]
    merged = merged.drop_duplicates(subset=identity, keep="last")
    order = [
        column
        for column in ("test_name", "seed", "n_train", "method", "evaluation_region")
        if column in merged.columns
    ]
    merged = merged.sort_values(order, kind="stable").reset_index(drop=True)

    output = experiment_dir / f"unified_{test}.csv"
    temporary = output.with_suffix(".csv.tmp")
    merged.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"[merged] {len(shards)} shards, {len(merged)} rows -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--test",
        choices=("all", "test1", "test2", "test3", "test4"),
        default="all",
        help="Merge only one test or all publication experiments.",
    )
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise ValueError("At least one seed is required")
    experiments = (
        EXPERIMENTS
        if args.test == "all"
        else tuple(item for item in EXPERIMENTS if item[1] == args.test)
    )
    for directory, test in experiments:
        merge_experiment(directory, test, seeds)


if __name__ == "__main__":
    main()
