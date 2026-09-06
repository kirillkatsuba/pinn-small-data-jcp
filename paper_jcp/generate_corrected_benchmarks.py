"""Generate corrected Test 2--4 datasets with resumable per-run outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_jcp.two_phase_fv import (  # noqa: E402
    FluidProperties,
    RelativePermeability,
    SolverOptions,
    load_result,
    simulate,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate corrected finite-volume datasets for Tests 2--4."
    )
    parser.add_argument("--test", choices=("test2", "test3", "test4", "all"), default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "paper_jcp" / "generated" / "publication_data",
    )
    parser.add_argument("--output-steps", type=int, default=100)
    parser.add_argument(
        "--final-time-test2",
        type=float,
        default=0.05,
        help="Pilot value; inspect fields before freezing the publication protocol.",
    )
    parser.add_argument(
        "--final-time-test3",
        type=float,
        default=0.002,
        help="Pilot value adjusted for the much larger Test 3 permeability.",
    )
    parser.add_argument(
        "--final-time-test4",
        type=float,
        default=0.01,
        help="Pilot value shared by all Test 4 parameter combinations.",
    )
    parser.add_argument("--test4-samples", type=int, default=100)
    parser.add_argument("--porosity", type=float, default=0.1)
    parser.add_argument("--water-viscosity", type=float, default=1.0)
    parser.add_argument("--oil-viscosity", type=float, default=3.0)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _fixed_spec(
    test_name: str,
    permeability_path: Path,
    final_time: float,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "test": test_name,
        "sample": 0,
        "permeability_path": str(permeability_path),
        "output": str(output_root / test_name / "solution.npz"),
        "final_time": final_time,
        "output_steps": args.output_steps,
        "porosity": args.porosity,
        "water_viscosity": args.water_viscosity,
        "oil_viscosity": args.oil_viscosity,
        "cfl": args.cfl,
        "water_exponent": 2.0,
        "oil_exponent": 4.0,
        "water_endpoint": 1.0,
        "oil_endpoint": 0.3,
        "overwrite": args.overwrite,
    }


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = ("test2", "test3", "test4") if args.test == "all" else (args.test,)
    specs: list[dict[str, Any]] = []
    if "test2" in selected:
        specs.append(
            _fixed_spec(
                "test2",
                ROOT / "test2-2phase-disp-simple" / "perm.npy",
                args.final_time_test2,
                args.output_root,
                args,
            )
        )
    if "test3" in selected:
        specs.append(
            _fixed_spec(
                "test3",
                ROOT / "test3-2phase-disp-perm-nonlinear" / "perm_3sigma.npy",
                args.final_time_test3,
                args.output_root,
                args,
            )
        )
    if "test4" in selected:
        parameter_dir = ROOT / "test4-2phase-disp-7d" / "data-100-new-two-sigma"
        parameters = {
            "water_exponent": np.load(parameter_dir / "pwat_100.npy"),
            "oil_exponent": np.load(parameter_dir / "poil_100.npy"),
            "water_endpoint": np.load(parameter_dir / "kwat_100.npy"),
            "oil_endpoint": np.load(parameter_dir / "koil_100.npy"),
        }
        available = min(len(values) for values in parameters.values())
        count = min(args.test4_samples, available)
        if count < 1:
            raise ValueError("test4-samples must be positive")
        for sample in range(count):
            specs.append(
                {
                    "test": "test4",
                    "sample": sample,
                    "permeability_path": str(
                        ROOT / "test4-2phase-disp-7d" / "perm_2.npy"
                    ),
                    "output": str(
                        args.output_root / "test4" / f"sample_{sample:03d}.npz"
                    ),
                    "final_time": args.final_time_test4,
                    "output_steps": args.output_steps,
                    "porosity": args.porosity,
                    "water_viscosity": args.water_viscosity,
                    "oil_viscosity": args.oil_viscosity,
                    "cfl": args.cfl,
                    **{name: float(values[sample]) for name, values in parameters.items()},
                    "overwrite": args.overwrite,
                }
            )
    return specs


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["output"])
    if output.exists() and not spec["overwrite"]:
        result = load_result(output)
        expected_relperm = {
            "water_exponent": spec["water_exponent"],
            "oil_exponent": spec["oil_exponent"],
            "water_endpoint": spec["water_endpoint"],
            "oil_endpoint": spec["oil_endpoint"],
        }
        expected_fluids = {
            "water_viscosity": spec["water_viscosity"],
            "oil_viscosity": spec["oil_viscosity"],
        }
        compatible = (
            len(result.times) == spec["output_steps"] + 1
            and np.isclose(result.times[-1], spec["final_time"])
            and all(
                np.isclose(result.metadata["relative_permeability"][name], value)
                for name, value in expected_relperm.items()
            )
            and all(
                np.isclose(result.metadata["fluids"][name], value)
                for name, value in expected_fluids.items()
            )
        )
        if not compatible:
            raise FileExistsError(
                f"Existing dataset is incompatible with the requested protocol: {output}. "
                "Use --overwrite only if replacement is intentional."
            )
        status = "reused"
    else:
        result = simulate(
            np.load(spec["permeability_path"]),
            spec["porosity"],
            final_time=spec["final_time"],
            num_output_steps=spec["output_steps"],
            relative_permeability=RelativePermeability(
                water_exponent=spec["water_exponent"],
                oil_exponent=spec["oil_exponent"],
                water_endpoint=spec["water_endpoint"],
                oil_endpoint=spec["oil_endpoint"],
            ),
            fluids=FluidProperties(
                water_viscosity=spec["water_viscosity"],
                oil_viscosity=spec["oil_viscosity"],
            ),
            options=SolverOptions(cfl=spec["cfl"]),
            progress=bool(spec.get("show_progress", False)),
            progress_description=f"{spec['test']} sample {spec['sample']:03d}",
        )
        result.save(output, overwrite=spec["overwrite"])
        status = "generated"
    return {
        "test": spec["test"],
        "sample": spec["sample"],
        "status": status,
        "path": str(output),
        "final_time": spec["final_time"],
        "output_steps": spec["output_steps"],
        "water_exponent": spec["water_exponent"],
        "oil_exponent": spec["oil_exponent"],
        "water_endpoint": spec["water_endpoint"],
        "oil_endpoint": spec["oil_endpoint"],
        "water_viscosity": spec["water_viscosity"],
        "oil_viscosity": spec["oil_viscosity"],
        "substeps": result.diagnostics.total_substeps,
        "maximum_water_saturation": float(result.water_saturation.max()),
        "water_mass_balance_error": result.diagnostics.maximum_water_mass_balance_error,
        "oil_mass_balance_error": result.diagnostics.maximum_oil_mass_balance_error,
        "pressure_balance_error": result.diagnostics.maximum_pressure_balance_error,
    }


def write_manifest(rows: list[dict[str, Any]], output_root: Path) -> Path:
    manifest = output_root / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    if manifest.exists():
        with manifest.open("r", newline="", encoding="utf-8") as stream:
            for existing in csv.DictReader(stream):
                merged[(existing["test"], int(existing["sample"]))] = existing
    for row in rows:
        merged[(str(row["test"]), int(row["sample"]))] = row
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(merged.values(), key=lambda row: (row["test"], int(row["sample"])))
        )
    return manifest


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    specs = build_specs(args)
    for spec in specs:
        spec["show_progress"] = args.workers == 1
    rows: list[dict[str, Any]] = []
    if args.workers == 1:
        for spec in specs:
            row = _run(spec)
            rows.append(row)
            print(f"[{row['status']}] {row['test']} sample {row['sample']}: {row['path']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run, spec): spec for spec in specs}
            completed = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="corrected datasets",
                unit="run",
                dynamic_ncols=True,
            )
            for future in completed:
                row = future.result()
                rows.append(row)
                tqdm.write(
                    f"[{row['status']}] {row['test']} sample {row['sample']}: "
                    f"{row['path']}"
                )
    manifest = write_manifest(rows, args.output_root)
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
