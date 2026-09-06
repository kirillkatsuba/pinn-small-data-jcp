"""Conservative cell-centred finite-volume solver for two-phase Darcy flow.

The model is incompressible, immiscible, gravity-free and capillary-free:

    div(q_t) = 0,
    q_t = -K (lambda_w + lambda_o) grad(p),
    phi dS_alpha/dt + div(f_alpha q_t) = 0,

where lambda_w = k_rw / mu_w, lambda_o = k_ro / mu_o and
f_alpha = lambda_alpha / (lambda_w + lambda_o).

Unlike the legacy generator, this implementation:

* uses total mobility, not its reciprocal;
* divides finite-volume fluxes by the pore volume phi * |cell|;
* stores pressure and saturations at the same physical time;
* includes the initial state at t=0;
* exports cell-centred coordinates;
* applies pure-water inflow on the left and no-flow boundaries in y.

The spatial discretisation is two-dimensional.  Permeability arrays with a
singleton third dimension are accepted for compatibility with the legacy data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve


Array = np.ndarray


@dataclass(frozen=True)
class RelativePermeability:
    """Corey-type relative-permeability parameters."""

    water_exponent: float = 2.0
    oil_exponent: float = 4.0
    water_endpoint: float = 1.0
    oil_endpoint: float = 0.3

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}")


@dataclass(frozen=True)
class FluidProperties:
    """Phase viscosities; explicit names avoid the legacy ``vr`` ambiguity."""

    water_viscosity: float = 1.0
    oil_viscosity: float = 3.0

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive; got {value}")


@dataclass(frozen=True)
class BoundaryConditions:
    """Pressure and inflow-composition boundary conditions."""

    left_pressure: float = 1.0
    right_pressure: float = 0.0
    left_inflow_water_fraction: float = 1.0
    right_inflow_water_fraction: float = 0.0

    def validate(self) -> None:
        if not self.left_pressure > self.right_pressure:
            raise ValueError("left_pressure must be greater than right_pressure")
        for name in ("left_inflow_water_fraction", "right_inflow_water_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class SolverOptions:
    """Time-integration controls for the explicit conservative transport step."""

    cfl: float = 0.45
    saturation_tolerance: float = 1.0e-10
    linear_residual_tolerance: float = 1.0e-9
    max_substeps_per_output: int = 100_000

    def validate(self) -> None:
        if not 0.0 < self.cfl <= 1.0:
            raise ValueError("cfl must lie in (0, 1]")
        if self.saturation_tolerance <= 0.0:
            raise ValueError("saturation_tolerance must be positive")
        if self.linear_residual_tolerance <= 0.0:
            raise ValueError("linear_residual_tolerance must be positive")
        if self.max_substeps_per_output < 1:
            raise ValueError("max_substeps_per_output must be positive")


@dataclass
class SimulationDiagnostics:
    total_substeps: int = 0
    minimum_substep: float = float("inf")
    maximum_substep: float = 0.0
    maximum_pressure_balance_error: float = 0.0
    maximum_water_mass_balance_error: float = 0.0
    maximum_oil_mass_balance_error: float = 0.0
    minimum_water_saturation: float = 1.0
    maximum_water_saturation: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        if not np.isfinite(result["minimum_substep"]):
            result["minimum_substep"] = 0.0
        return result


@dataclass
class SimulationResult:
    """Synchronous solution snapshots on cell-centred coordinates."""

    times: Array
    x: Array
    y: Array
    pressure: Array
    water_saturation: Array
    oil_saturation: Array
    permeability: Array
    porosity: Array
    metadata: dict[str, Any]
    diagnostics: SimulationDiagnostics

    def save(self, output: str | Path, overwrite: bool = False) -> Path:
        path = Path(output)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times,
            x=self.x,
            y=self.y,
            pressure=self.pressure,
            water_saturation=self.water_saturation,
            oil_saturation=self.oil_saturation,
            permeability=self.permeability,
            porosity=self.porosity,
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
            diagnostics_json=np.asarray(json.dumps(self.diagnostics.to_dict(), sort_keys=True)),
        )
        return path


def _as_2d(name: str, value: Array | float, shape: tuple[int, int] | None = None) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 0:
        if shape is None:
            raise ValueError(f"A shape is required to expand scalar {name}")
        array = np.full(shape, float(array), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D or have a singleton third dimension")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def relative_permeabilities(
    water_saturation: Array,
    parameters: RelativePermeability,
) -> tuple[Array, Array]:
    """Return Corey relative permeabilities for water and oil."""

    parameters.validate()
    sw = np.asarray(water_saturation, dtype=np.float64)
    if np.any(sw < -1.0e-12) or np.any(sw > 1.0 + 1.0e-12):
        raise ValueError("water saturation must lie in [0, 1]")
    sw = np.clip(sw, 0.0, 1.0)
    so = 1.0 - sw
    krw = parameters.water_endpoint * sw**parameters.water_exponent
    kro = parameters.oil_endpoint * so**parameters.oil_exponent
    return krw, kro


def phase_mobilities(
    water_saturation: Array,
    relative_permeability: RelativePermeability,
    fluids: FluidProperties,
) -> tuple[Array, Array]:
    """Return lambda_w=k_rw/mu_w and lambda_o=k_ro/mu_o."""

    fluids.validate()
    krw, kro = relative_permeabilities(water_saturation, relative_permeability)
    return krw / fluids.water_viscosity, kro / fluids.oil_viscosity


def _harmonic(left: Array, right: Array) -> Array:
    denominator = left + right
    result = np.zeros_like(denominator)
    valid = denominator > 0.0
    result[valid] = 2.0 * left[valid] * right[valid] / denominator[valid]
    return result


def pressure_conductivity(
    permeability: Array,
    water_saturation: Array,
    relative_permeability: RelativePermeability,
    fluids: FluidProperties,
) -> Array:
    """Return K(lambda_w+lambda_o), the correct total Darcy conductivity."""

    lambda_w, lambda_o = phase_mobilities(water_saturation, relative_permeability, fluids)
    return permeability * (lambda_w + lambda_o)


def _pressure_system(
    conductivity: Array,
    dx: float,
    dy: float,
    boundary: BoundaryConditions,
) -> tuple[csr_matrix, Array]:
    nx, ny = conductivity.shape
    size = nx * ny
    matrix = lil_matrix((size, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)

    def index(i: int, j: int) -> int:
        return i * ny + j

    tx = _harmonic(conductivity[:-1, :], conductivity[1:, :]) * dy / dx
    ty = _harmonic(conductivity[:, :-1], conductivity[:, 1:]) * dx / dy

    for i in range(nx):
        for j in range(ny):
            row = index(i, j)
            if i > 0:
                value = tx[i - 1, j]
                matrix[row, row] += value
                matrix[row, index(i - 1, j)] -= value
            else:
                value = 2.0 * conductivity[i, j] * dy / dx
                matrix[row, row] += value
                rhs[row] += value * boundary.left_pressure

            if i < nx - 1:
                value = tx[i, j]
                matrix[row, row] += value
                matrix[row, index(i + 1, j)] -= value
            else:
                value = 2.0 * conductivity[i, j] * dy / dx
                matrix[row, row] += value
                rhs[row] += value * boundary.right_pressure

            if j > 0:
                value = ty[i, j - 1]
                matrix[row, row] += value
                matrix[row, index(i, j - 1)] -= value
            if j < ny - 1:
                value = ty[i, j]
                matrix[row, row] += value
                matrix[row, index(i, j + 1)] -= value

    return matrix.tocsr(), rhs


def solve_pressure(
    conductivity: Array,
    dx: float,
    dy: float,
    boundary: BoundaryConditions,
    residual_tolerance: float = 1.0e-9,
) -> tuple[Array, float]:
    """Solve div(conductivity grad(p))=0 with x-Dirichlet/y-Neumann BCs."""

    matrix, rhs = _pressure_system(conductivity, dx, dy, boundary)
    pressure_vector = spsolve(matrix, rhs)
    if not np.all(np.isfinite(pressure_vector)):
        raise RuntimeError("Pressure solve produced non-finite values")
    residual = matrix @ pressure_vector - rhs
    scale = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
    relative_error = float(np.linalg.norm(residual, ord=np.inf) / scale)
    if relative_error > residual_tolerance:
        raise RuntimeError(
            f"Pressure linear residual {relative_error:.3e} exceeds "
            f"tolerance {residual_tolerance:.3e}"
        )
    return pressure_vector.reshape(conductivity.shape), relative_error


def total_fluxes(
    pressure: Array,
    conductivity: Array,
    dx: float,
    dy: float,
    boundary: BoundaryConditions,
) -> tuple[Array, Array]:
    """Return integrated face fluxes oriented in positive x and positive y."""

    nx, ny = pressure.shape
    qx = np.zeros((nx + 1, ny), dtype=np.float64)
    qy = np.zeros((nx, ny + 1), dtype=np.float64)

    tx = _harmonic(conductivity[:-1, :], conductivity[1:, :]) * dy / dx
    ty = _harmonic(conductivity[:, :-1], conductivity[:, 1:]) * dx / dy
    qx[1:nx, :] = tx * (pressure[:-1, :] - pressure[1:, :])
    qx[0, :] = 2.0 * conductivity[0, :] * dy / dx * (
        boundary.left_pressure - pressure[0, :]
    )
    qx[nx, :] = 2.0 * conductivity[-1, :] * dy / dx * (
        pressure[-1, :] - boundary.right_pressure
    )
    qy[:, 1:ny] = ty * (pressure[:, :-1] - pressure[:, 1:])
    # qy[:, 0] and qy[:, ny] remain zero: impermeable lateral boundaries.
    return qx, qy


def _phase_fluxes(
    total_qx: Array,
    total_qy: Array,
    water_fraction: Array,
    boundary: BoundaryConditions,
) -> tuple[Array, Array]:
    nx, ny = water_fraction.shape
    water_qx = np.zeros_like(total_qx)
    water_qy = np.zeros_like(total_qy)

    internal_x = total_qx[1:nx, :]
    water_qx[1:nx, :] = internal_x * np.where(
        internal_x >= 0.0,
        water_fraction[:-1, :],
        water_fraction[1:, :],
    )
    water_qx[0, :] = total_qx[0, :] * np.where(
        total_qx[0, :] >= 0.0,
        boundary.left_inflow_water_fraction,
        water_fraction[0, :],
    )
    water_qx[nx, :] = total_qx[nx, :] * np.where(
        total_qx[nx, :] >= 0.0,
        water_fraction[-1, :],
        boundary.right_inflow_water_fraction,
    )

    internal_y = total_qy[:, 1:ny]
    water_qy[:, 1:ny] = internal_y * np.where(
        internal_y >= 0.0,
        water_fraction[:, :-1],
        water_fraction[:, 1:],
    )
    return water_qx, water_qy


def _integrated_divergence(qx: Array, qy: Array) -> Array:
    return qx[1:, :] - qx[:-1, :] + qy[:, 1:] - qy[:, :-1]


def _stable_timestep(
    total_qx: Array,
    total_qy: Array,
    pore_volume: Array,
    cfl: float,
) -> float:
    outgoing = (
        np.maximum(total_qx[1:, :], 0.0)
        + np.maximum(-total_qx[:-1, :], 0.0)
        + np.maximum(total_qy[:, 1:], 0.0)
        + np.maximum(-total_qy[:, :-1], 0.0)
    )
    active = outgoing > 0.0
    if not np.any(active):
        return float("inf")
    return float(cfl * np.min(pore_volume[active] / outgoing[active]))


def _boundary_influx(qx: Array, qy: Array) -> float:
    return float(
        np.sum(qx[0, :])
        - np.sum(qx[-1, :])
        + np.sum(qy[:, 0])
        - np.sum(qy[:, -1])
    )


def _pressure_state(
    permeability: Array,
    water_saturation: Array,
    relative_permeability: RelativePermeability,
    fluids: FluidProperties,
    boundary: BoundaryConditions,
    options: SolverOptions,
    dx: float,
    dy: float,
) -> tuple[Array, Array, Array, float]:
    conductivity = pressure_conductivity(
        permeability, water_saturation, relative_permeability, fluids
    )
    if np.any(conductivity <= 0.0):
        raise ValueError("Total Darcy conductivity must remain positive")
    pressure, linear_error = solve_pressure(
        conductivity,
        dx,
        dy,
        boundary,
        residual_tolerance=options.linear_residual_tolerance,
    )
    total_qx, total_qy = total_fluxes(pressure, conductivity, dx, dy, boundary)
    pressure_balance_error = float(np.max(np.abs(_integrated_divergence(total_qx, total_qy))))
    return pressure, total_qx, total_qy, max(linear_error, pressure_balance_error)


def simulate(
    permeability: Array,
    porosity: Array | float = 0.1,
    *,
    final_time: float = 1.5,
    num_output_steps: int = 100,
    relative_permeability: RelativePermeability | None = None,
    fluids: FluidProperties | None = None,
    boundary: BoundaryConditions | None = None,
    options: SolverOptions | None = None,
    initial_water_saturation: Array | float = 0.0,
    domain_size: tuple[float, float] = (1.0, 1.0),
    progress: bool = False,
    progress_description: str = "two-phase FV",
    progress_diagnostics: bool = False,
) -> SimulationResult:
    """Simulate two-phase displacement and return synchronous snapshots.

    ``num_output_steps`` is the number of intervals, so every returned field has
    ``num_output_steps + 1`` time levels including the exact initial state.
    Adaptive CFL substeps are taken between output times.
    """

    relperm = relative_permeability or RelativePermeability()
    fluid = fluids or FluidProperties()
    bc = boundary or BoundaryConditions()
    solver = options or SolverOptions()
    relperm.validate()
    fluid.validate()
    bc.validate()
    solver.validate()

    perm = _as_2d("permeability", permeability)
    if np.any(perm <= 0.0):
        raise ValueError("permeability must be strictly positive")
    phi = _as_2d("porosity", porosity, perm.shape)
    if np.any(phi <= 0.0) or np.any(phi > 1.0):
        raise ValueError("porosity must lie in (0, 1]")
    sw = _as_2d("initial_water_saturation", initial_water_saturation, perm.shape)
    if np.any(sw < 0.0) or np.any(sw > 1.0):
        raise ValueError("initial_water_saturation must lie in [0, 1]")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be finite and positive")
    if num_output_steps < 1:
        raise ValueError("num_output_steps must be positive")
    lx, ly = map(float, domain_size)
    if not lx > 0.0 or not ly > 0.0:
        raise ValueError("domain dimensions must be positive")

    nx, ny = perm.shape
    dx = lx / nx
    dy = ly / ny
    cell_volume = dx * dy  # Unit thickness.
    pore_volume = phi * cell_volume
    x = (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * dy
    times = np.linspace(0.0, final_time, num_output_steps + 1, dtype=np.float64)

    pressure_history = np.empty((num_output_steps + 1, nx, ny), dtype=np.float64)
    sw_history = np.empty_like(pressure_history)
    so_history = np.empty_like(pressure_history)
    diagnostics = SimulationDiagnostics()

    pressure, _, _, pressure_error = _pressure_state(
        perm, sw, relperm, fluid, bc, solver, dx, dy
    )
    pressure_history[0] = pressure
    sw_history[0] = sw
    so_history[0] = 1.0 - sw
    diagnostics.maximum_pressure_balance_error = pressure_error
    diagnostics.minimum_water_saturation = float(np.min(sw))
    diagnostics.maximum_water_saturation = float(np.max(sw))

    output_indices = range(1, num_output_steps + 1)
    progress_bar = None
    if progress:
        from tqdm.auto import tqdm

        progress_bar = tqdm(
            output_indices,
            total=num_output_steps,
            desc=progress_description,
            unit="layer",
            dynamic_ncols=True,
        )
        output_indices = progress_bar

    for output_index in output_indices:
        remaining = float(times[output_index] - times[output_index - 1])
        local_substeps = 0
        while remaining > max(1.0e-14, 1.0e-13 * final_time):
            pressure, total_qx, total_qy, pressure_error = _pressure_state(
                perm, sw, relperm, fluid, bc, solver, dx, dy
            )
            diagnostics.maximum_pressure_balance_error = max(
                diagnostics.maximum_pressure_balance_error, pressure_error
            )
            lambda_w, lambda_o = phase_mobilities(sw, relperm, fluid)
            water_fraction = lambda_w / (lambda_w + lambda_o)
            water_qx, water_qy = _phase_fluxes(
                total_qx, total_qy, water_fraction, bc
            )
            oil_qx = total_qx - water_qx
            oil_qy = total_qy - water_qy

            stable_dt = _stable_timestep(total_qx, total_qy, pore_volume, solver.cfl)
            dt = min(remaining, stable_dt)
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"Invalid adaptive timestep: {dt}")

            water_mass_before = float(np.sum(pore_volume * sw))
            oil_mass_before = float(np.sum(pore_volume * (1.0 - sw)))
            updated = sw - dt * _integrated_divergence(water_qx, water_qy) / pore_volume
            minimum = float(np.min(updated))
            maximum = float(np.max(updated))
            tolerance = solver.saturation_tolerance
            if minimum < -tolerance or maximum > 1.0 + tolerance:
                raise RuntimeError(
                    "CFL-controlled transport left physical saturation bounds: "
                    f"min={minimum:.6e}, max={maximum:.6e}, dt={dt:.6e}"
                )
            sw = np.clip(updated, 0.0, 1.0)

            water_mass_after = float(np.sum(pore_volume * sw))
            oil_mass_after = float(np.sum(pore_volume * (1.0 - sw)))
            water_expected = dt * _boundary_influx(water_qx, water_qy)
            oil_expected = dt * _boundary_influx(oil_qx, oil_qy)
            diagnostics.maximum_water_mass_balance_error = max(
                diagnostics.maximum_water_mass_balance_error,
                abs((water_mass_after - water_mass_before) - water_expected),
            )
            diagnostics.maximum_oil_mass_balance_error = max(
                diagnostics.maximum_oil_mass_balance_error,
                abs((oil_mass_after - oil_mass_before) - oil_expected),
            )
            diagnostics.total_substeps += 1
            diagnostics.minimum_substep = min(diagnostics.minimum_substep, dt)
            diagnostics.maximum_substep = max(diagnostics.maximum_substep, dt)
            diagnostics.minimum_water_saturation = min(
                diagnostics.minimum_water_saturation, float(np.min(sw))
            )
            diagnostics.maximum_water_saturation = max(
                diagnostics.maximum_water_saturation, float(np.max(sw))
            )
            remaining -= dt
            local_substeps += 1
            if local_substeps > solver.max_substeps_per_output:
                raise RuntimeError(
                    "Exceeded max_substeps_per_output; reduce the output interval, "
                    "rescale the physical time, or inspect permeability units"
                )

        # Recompute p from the saturation stored at this same output time.
        pressure, _, _, pressure_error = _pressure_state(
            perm, sw, relperm, fluid, bc, solver, dx, dy
        )
        diagnostics.maximum_pressure_balance_error = max(
            diagnostics.maximum_pressure_balance_error, pressure_error
        )
        pressure_history[output_index] = pressure
        sw_history[output_index] = sw
        so_history[output_index] = 1.0 - sw
        if progress_bar is not None:
            if progress_diagnostics:
                mass_error = max(
                    diagnostics.maximum_water_mass_balance_error,
                    diagnostics.maximum_oil_mass_balance_error,
                )
                progress_bar.set_postfix(
                    t=f"{times[output_index]:.4g}/{final_time:.4g}",
                    layer_substeps=local_substeps,
                    total_substeps=diagnostics.total_substeps,
                    dt_min=f"{diagnostics.minimum_substep:.2e}",
                    dt_max=f"{diagnostics.maximum_substep:.2e}",
                    sw=f"[{np.min(sw):.3f},{np.max(sw):.3f}]",
                    mass_err=f"{mass_error:.2e}",
                    pressure_err=f"{diagnostics.maximum_pressure_balance_error:.2e}",
                    refresh=False,
                )
            else:
                progress_bar.set_postfix(
                    substeps=diagnostics.total_substeps,
                    sw_max=f"{diagnostics.maximum_water_saturation:.3f}",
                    refresh=False,
                )

    metadata: dict[str, Any] = {
        "schema": "corrected-two-phase-fv-v1",
        "model": "incompressible immiscible two-phase Darcy flow",
        "spatial_discretisation": "cell-centred finite volume, harmonic transmissibility",
        "transport_discretisation": "first-order upwind, adaptive explicit CFL",
        "pressure_transport_coupling": "pressure recomputed at every adaptive substep",
        "domain_size": [lx, ly],
        "grid_shape": [nx, ny],
        "cell_size": [dx, dy],
        "relative_permeability": asdict(relperm),
        "fluids": asdict(fluid),
        "boundary_conditions": asdict(bc),
        "solver_options": asdict(solver),
        "includes_initial_state": True,
        "coordinates": "cell centres",
        "field_axis_order": ["time", "x", "y"],
    }
    return SimulationResult(
        times=times,
        x=x,
        y=y,
        pressure=pressure_history,
        water_saturation=sw_history,
        oil_saturation=so_history,
        permeability=perm,
        porosity=phi,
        metadata=metadata,
        diagnostics=diagnostics,
    )


def load_result(path: str | Path) -> SimulationResult:
    """Load a result produced by :meth:`SimulationResult.save`."""

    with np.load(path) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        diagnostics_values = json.loads(str(data["diagnostics_json"].item()))
        diagnostics = SimulationDiagnostics(**diagnostics_values)
        return SimulationResult(
            times=data["times"].copy(),
            x=data["x"].copy(),
            y=data["y"].copy(),
            pressure=data["pressure"].copy(),
            water_saturation=data["water_saturation"].copy(),
            oil_saturation=data["oil_saturation"].copy(),
            permeability=data["permeability"].copy(),
            porosity=data["porosity"].copy(),
            metadata=metadata,
            diagnostics=diagnostics,
        )
