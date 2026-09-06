#!/usr/bin/env python3
"""Compare regenerated publication metrics with immutable CSV snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = ROOT / "reproducibility" / "results"
GENERATED_ROOT = ROOT / "paper_jcp" / "generated"

COMPARISONS = {
    "test1": (
        REFERENCE_ROOT / "test1.csv",
        GENERATED_ROOT / "test1_corrected_candidate" / "unified_test1.csv",
    ),
    "test2-pinn-standard": (
        REFERENCE_ROOT / "test2_pinn_and_standard_classical.csv",
        GENERATED_ROOT / "corrected_test2_final" / "unified_test2.csv",
    ),
    "test2-strong-classical": (
        REFERENCE_ROOT / "test2_strong_classical.csv",
        GENERATED_ROOT / "corrected_test2_classical_strong" / "unified_test2.csv",
    ),
    "test3": (
        REFERENCE_ROOT / "test3.csv",
        GENERATED_ROOT / "corrected_test3_final" / "unified_test3.csv",
    ),
    "test4": (
        REFERENCE_ROOT / "test4.csv",
        GENERATED_ROOT / "corrected_test4_final" / "unified_test4.csv",
    ),
}

KEY_COLUMNS = ["test_name", "seed", "n_train", "method", "evaluation_region"]
METRIC_TOKENS = (
    "MSE", "RMSE", "MAE", "R2", "relative_L2", "NRMSE_std",
    "saturation_bounds_violation", "closure_MAE",
)


def metric_columns(reference: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
    shared = set(reference.columns) & set(candidate.columns)
    return sorted(
        column
        for column in shared
        if column in METRIC_TOKENS
        or any(column.endswith(f"_{token}") for token in METRIC_TOKENS)
    )


def keyed(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing key columns: {missing}")
    if frame.duplicated(KEY_COLUMNS).any():
        duplicate = frame.loc[frame.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"{source} has duplicate result rows:\n{duplicate.head()}")
    return frame.set_index(KEY_COLUMNS).sort_index()


def compare_one(
    name: str,
    reference_path: Path,
    candidate_path: Path,
    *,
    rtol: float,
    atol: float,
) -> bool:
    if not candidate_path.is_file():
        print(f"[MISSING] {name}: {candidate_path.relative_to(ROOT)}")
        return False
    reference = keyed(pd.read_csv(reference_path), reference_path)
    candidate = keyed(pd.read_csv(candidate_path), candidate_path)
    missing_rows = reference.index.difference(candidate.index)
    extra_rows = candidate.index.difference(reference.index)
    if len(missing_rows) or len(extra_rows):
        print(f"[ROWS] {name}: missing={len(missing_rows)}, extra={len(extra_rows)}")
        return False
    columns = metric_columns(reference, candidate)
    if not columns:
        print(f"[ERROR] {name}: no shared metric columns")
        return False
    left = reference[columns].to_numpy(dtype=float)
    right = candidate.loc[reference.index, columns].to_numpy(dtype=float)
    close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
    finite = np.isfinite(left) & np.isfinite(right)
    abs_error = np.abs(left - right)
    rel_error = abs_error / np.maximum(np.abs(left), np.finfo(float).tiny)
    max_abs = float(np.max(abs_error[finite])) if finite.any() else 0.0
    max_rel = float(np.max(rel_error[finite])) if finite.any() else 0.0
    if close.all():
        print(
            f"[OK] {name}: rows={len(reference)}, metrics={len(columns)}, "
            f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
        )
        return True
    row_index, column_index = np.argwhere(~close)[0]
    key = reference.index[row_index]
    column = columns[column_index]
    print(
        f"[DIFF] {name}: {int((~close).sum())} values outside tolerance; "
        f"first key={key}, metric={column}, "
        f"reference={left[row_index, column_index]:.9g}, "
        f"candidate={right[row_index, column_index]:.9g}; "
        f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}"
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify regenerated result metrics against publication snapshots."
    )
    parser.add_argument("--suite", choices=(*COMPARISONS, "all"), default="all")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = (
        COMPARISONS
        if args.suite == "all"
        else {args.suite: COMPARISONS[args.suite]}
    )
    outcomes = [
        compare_one(name, *paths, rtol=args.rtol, atol=args.atol)
        for name, paths in selected.items()
    ]
    if not all(outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
