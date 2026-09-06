"""Fast invariant checks for the corrected two-phase finite-volume generator."""

from __future__ import annotations

import unittest

import numpy as np

from paper_jcp.two_phase_fv import (
    BoundaryConditions,
    FluidProperties,
    RelativePermeability,
    pressure_conductivity,
    simulate,
    solve_pressure,
)


class CorrectedGeneratorTest(unittest.TestCase):
    def test_uniform_pressure_is_linear_at_cell_centres(self) -> None:
        nx, ny = 12, 7
        conductivity = np.ones((nx, ny), dtype=np.float64)
        pressure, residual = solve_pressure(
            conductivity,
            dx=1.0 / nx,
            dy=1.0 / ny,
            boundary=BoundaryConditions(),
        )
        expected = np.repeat(
            (1.0 - (np.arange(nx) + 0.5) / nx)[:, None], ny, axis=1
        )
        np.testing.assert_allclose(pressure, expected, atol=1.0e-12)
        self.assertLess(residual, 1.0e-12)

    def test_total_conductivity_uses_mobility_not_inverse(self) -> None:
        sw = np.asarray([[0.25]])
        perm = np.asarray([[2.0]])
        relperm = RelativePermeability(
            water_exponent=2.0,
            oil_exponent=4.0,
            water_endpoint=1.0,
            oil_endpoint=0.3,
        )
        fluids = FluidProperties(water_viscosity=1.0, oil_viscosity=3.0)
        expected = 2.0 * (0.25**2 + 0.3 * 0.75**4 / 3.0)
        actual = pressure_conductivity(perm, sw, relperm, fluids)
        np.testing.assert_allclose(actual, expected)

    def test_simulation_is_synchronous_conservative_and_bounded(self) -> None:
        result = simulate(
            np.ones((10, 8), dtype=np.float64),
            0.1,
            final_time=0.02,
            num_output_steps=4,
        )
        self.assertEqual(result.pressure.shape, (5, 10, 8))
        np.testing.assert_allclose(result.times[0], 0.0)
        np.testing.assert_allclose(result.water_saturation[0], 0.0)
        np.testing.assert_allclose(result.oil_saturation[0], 1.0)
        np.testing.assert_allclose(
            result.water_saturation + result.oil_saturation,
            1.0,
            atol=1.0e-14,
        )
        self.assertGreaterEqual(float(result.water_saturation.min()), 0.0)
        self.assertLessEqual(float(result.water_saturation.max()), 1.0)
        self.assertLess(result.diagnostics.maximum_water_mass_balance_error, 1.0e-12)
        self.assertLess(result.diagnostics.maximum_oil_mass_balance_error, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
