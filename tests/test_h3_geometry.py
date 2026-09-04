import math
import unittest

from cauce.h3_geometry import (
    H3DomemasterCoordinatePatch,
    domemaster_coordinate_rows,
    equidistant_180_ray_xy,
)


class H3DomemasterGeometryTests(unittest.TestCase):
    def test_equidistant_center_and_rim_are_unit_rays(self):
        self.assertEqual(equidistant_180_ray_xy(0.0, 0.0), (0.0, 0.0, 1.0))
        for point in ((1.0, 0.0), (0.0, -1.0), (0.6, 0.8)):
            ray = equidistant_180_ray_xy(*point)
            self.assertAlmostEqual(sum(value * value for value in ray), 1.0, places=12)
            self.assertAlmostEqual(ray[2], 0.0, places=12)

    def test_equidistant_mid_radius_has_expected_angle(self):
        ray = equidistant_180_ray_xy(0.5, 0.0)
        self.assertAlmostEqual(ray[0], math.sqrt(0.5), places=12)
        self.assertAlmostEqual(ray[1], 0.0, places=12)
        self.assertAlmostEqual(ray[2], math.sqrt(0.5), places=12)

    def test_outside_disc_is_rejected(self):
        with self.assertRaises(ValueError):
            equidistant_180_ray_xy(1.0, 1.0)

    def test_zero_strength_matches_stock_h3_grid(self):
        rows = domemaster_coordinate_rows(48, 48, strength=0.0)
        axis = [index * (32.0 / 24.0) for index in range(24)]
        expected = [(h, w) for h in axis for w in axis]
        self.assertEqual(len(rows), 24 * 24)
        for actual, stock in zip(rows, expected, strict=True):
            self.assertAlmostEqual(actual[0], stock[0], places=12)
            self.assertAlmostEqual(actual[1], stock[1], places=12)

    def test_full_strength_preserves_outside_rows_and_expands_inside_radius(self):
        stock = domemaster_coordinate_rows(48, 48, strength=0.0)
        warped = domemaster_coordinate_rows(48, 48, strength=1.0)
        # Top-left lies outside the disc and remains stock.
        self.assertEqual(warped[0], stock[0])
        # A row halfway from center to the horizontal rim moves farther outward
        # under the ray-x parameterization (sin(theta) > normalized radius).
        row = 12 * 24 + 18
        center = (stock[12 * 24 + 0][0] + stock[12 * 24 + 23][0]) / 2.0
        self.assertGreater(abs(warped[row][1] - center), abs(stock[row][1] - center))

    def test_square_and_strength_contracts(self):
        with self.assertRaises(ValueError):
            domemaster_coordinate_rows(48, 84)
        with self.assertRaises(ValueError):
            domemaster_coordinate_rows(48, 48, strength=1.1)

    def test_zero_strength_wrapper_is_a_bit_exact_bypass(self):
        sentinel = object()
        calls = []

        def executor(*args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

        patch = H3DomemasterCoordinatePatch(0.0, True, "stock")
        result = patch(
            executor,
            ["video", "audio"],
            "timestep",
            "context",
            {"existing": True},
            minimax_payload={},
        )
        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        self.assertTrue(patch.last_report["bypassed_bit_exactly"])


if __name__ == "__main__":
    unittest.main()
