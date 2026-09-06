"""Command-line entry point for the corrected two-phase FV generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_jcp.two_phase_fv import (
    BoundaryConditions,
    FluidProperties,
    RelativePermeability,
    SolverOptions,
    simulate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synchronous conservative two-phase-flow dataset."
    )
    parser.add_argument("--permeability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--porosity", type=float, default=0.1)
    parser.add_argument("--final-time", type=float, default=1.5)
    parser.add_argument("--output-steps", type=int, default=100)
    parser.add_argument("--water-exponent", type=float, default=2.0)
    parser.add_argument("--oil-exponent", type=float, default=4.0)
    parser.add_argument("--water-endpoint", type=float, default=1.0)
    parser.add_argument("--oil-endpoint", type=float, default=0.3)
    parser.add_argument("--water-viscosity", type=float, default=1.0)
    parser.add_argument("--oil-viscosity", type=float, default=3.0)
    parser.add_argument("--left-pressure", type=float, default=1.0)
    parser.add_argument("--right-pressure", type=float, default=0.0)
    parser.add_argument("--cfl", type=float, default=0.45)
    parser.add_argument("--initial-water-saturation", type=float, default=0.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permeability = np.load(args.permeability)
    result = simulate(
        permeability,
        args.porosity,
        final_time=args.final_time,
        num_output_steps=args.output_steps,
        relative_permeability=RelativePermeability(
            water_exponent=args.water_exponent,
            oil_exponent=args.oil_exponent,
            water_endpoint=args.water_endpoint,
            oil_endpoint=args.oil_endpoint,
        ),
        fluids=FluidProperties(
            water_viscosity=args.water_viscosity,
            oil_viscosity=args.oil_viscosity,
        ),
        boundary=BoundaryConditions(
            left_pressure=args.left_pressure,
            right_pressure=args.right_pressure,
        ),
        options=SolverOptions(cfl=args.cfl),
        initial_water_saturation=args.initial_water_saturation,
        progress=not args.no_progress,
        progress_description=f"FV {args.output.name}",
    )
    saved = result.save(args.output, overwrite=args.overwrite)
    summary = {
        "output": str(saved),
        "shape": list(result.pressure.shape),
        "time_range": [float(result.times[0]), float(result.times[-1])],
        "water_saturation_range": [
            float(result.water_saturation.min()),
            float(result.water_saturation.max()),
        ],
        "diagnostics": result.diagnostics.to_dict(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
