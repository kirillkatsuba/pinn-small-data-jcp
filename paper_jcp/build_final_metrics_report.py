#!/usr/bin/env python3
"""Build the complete mean ± std metric report for unified Tests 1--4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAPER = Path(__file__).resolve().parent
GENERATED = PAPER / "generated"
PUBLISHED = PAPER.parent / "reproducibility" / "results"
METHODS = ("H-PINN", "HO-PINN", "XGB", "SVM", "GP")
REGIONS = ("interpolation", "extrapolation")
TEST_PATHS = {
    "test1": PUBLISHED / "test1.csv",
    "test2": PUBLISHED / "test2_pinn_and_standard_classical.csv",
    "test3": PUBLISHED / "test3.csv",
    "test4": PUBLISHED / "test4.csv",
}


def load_test(test: str) -> pd.DataFrame:
    table = pd.read_csv(TEST_PATHS[test])
    if test == "test2":
        strong = pd.read_csv(PUBLISHED / "test2_strong_classical.csv")
        table = pd.concat(
            [table[table["method"].isin(("H-PINN", "HO-PINN"))], strong],
            ignore_index=True,
        )
    return table


def metric_columns(test: str, table: pd.DataFrame) -> list[str]:
    excluded = {
        "seed", "n_train", "n_collocation_per_epoch", "n_test", "train_time_sec",
        "label_time_fraction", "full_time", "time_cutoff", "porosity",
        "water_viscosity", "oil_viscosity", "corrected_test4_pool_size",
        "model_hidden_dim", "model_depth", "learning_rate", "l0_data", "l0_ibc",
    }
    suffixes = (
        "_MSE", "_RMSE", "_MAE", "_R2", "_relative_L2", "_NRMSE_std"
    )
    metrics = []
    for column in table.columns:
        if column in excluded:
            continue
        if test == "test1" and column in {
            "MSE", "RMSE", "MAE", "R2", "relative_L2", "NRMSE_std"
        }:
            metrics.append(column)
        elif test != "test1" and (
            column.endswith(suffixes)
            or column in {"saturation_bounds_violation", "closure_MAE"}
        ):
            metrics.append(column)
    return metrics


def larger_is_better(metric: str) -> bool:
    return metric == "R2" or metric.endswith("_R2")


def fmt(mean: float, std: float, best: bool) -> str:
    if not np.isfinite(mean):
        value = "—"
    else:
        value = f"{mean:.4f} ± {std:.4f}"
    return f"**{value}**" if best else value


def metric_table(table: pd.DataFrame, region: str, metric: str) -> list[str]:
    grouped = (
        table[table["evaluation_region"] == region]
        .groupby(["n_train", "method"])[metric]
        .agg(["mean", "std"])
    )
    lines = [
        f"#### `{metric}` ({'больше' if larger_is_better(metric) else 'меньше'} — лучше)",
        "",
        "| N | " + " | ".join(METHODS) + " |",
        "|---:|" + "---:|" * len(METHODS),
    ]
    for budget in sorted(table["n_train"].unique()):
        values = {method: grouped.loc[(budget, method)] for method in METHODS}
        means = np.asarray([values[method]["mean"] for method in METHODS])
        optimum = np.nanmax(means) if larger_is_better(metric) else np.nanmin(means)
        cells = []
        for method in METHODS:
            value = values[method]
            is_best = bool(np.isclose(value["mean"], optimum, rtol=1e-9, atol=1e-12))
            cells.append(fmt(float(value["mean"]), float(value["std"]), is_best))
        lines.append(f"| {int(budget)} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=PAPER / "FINAL_EXPERIMENT_METRICS.md"
    )
    args = parser.parse_args()

    tables = {test: load_test(test) for test in TEST_PATHS}
    lines = [
        "# Итоговые результаты экспериментов: mean ± std по трём seed",
        "",
        "В каждой строке жирным выделено лучшее среднее значение среди пяти методов. "
        "Для `R2` максимум является лучшим; для ошибок, нарушений физических "
        "ограничений и времени обучения лучшим является минимум.",
        "",
        "Все значения рассчитаны по seed `101`, `102`, `103`. Интерполяция "
        "соответствует времени внутри диапазона размеченных данных, экстраполяция — "
        "времени после границы обучения.",
        "",
        "Для Test 2 в отчёт подставлены classical baselines из усиленного "
        "`strong/time-CV` подбора. Tests 3--4 уже содержат этот подбор в итоговых "
        "CSV. В Test 1 classical baselines относятся к более раннему standard tuning.",
        "",
        "## Ключевые метрики",
        "",
        "Для Test 1 ключевой метрикой является scalar `R2`; для Tests 2--4 — "
        "`Swat_R2`, поскольку water saturation является наиболее сложным и "
        "физически значимым выходом.",
        "",
    ]
    for test, metric in (("test1", "R2"), ("test2", "Swat_R2"),
                         ("test3", "Swat_R2"), ("test4", "Swat_R2")):
        lines.extend([f"### {test.capitalize()}", ""])
        for region in REGIONS:
            lines.extend([f"#### {region.capitalize()}", ""])
            lines.extend(metric_table(tables[test], region, metric)[2:])

    lines.extend([
        "## Полные таблицы всех метрик",
        "",
        "Усреднённые `mean_*` показатели Tests 2--4 объединяют Pressure, Soil и "
        "Swat; их следует использовать только как вспомогательные, потому что "
        "лёгкое давление может скрывать ошибку насыщенности.",
        "",
    ])
    for test, table in tables.items():
        lines.extend([f"## {test.capitalize()}", ""])
        for region in REGIONS:
            lines.extend([f"### {region.capitalize()}", ""])
            for metric in metric_columns(test, table):
                lines.extend(metric_table(table, region, metric))
        lines.extend(["### Время обучения / подбора", ""])
        # Timing is identical for the two evaluation rows; keep one row per run.
        timing = table[table["evaluation_region"] == "interpolation"]
        lines.extend(metric_table(timing, "interpolation", "train_time_sec"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
