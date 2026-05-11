import math
import unittest

from app.idw import (
    IDWPoint,
    create_idw_grid,
    euclidean_distance_px,
    idw_interpolate,
    idw_weight,
    validate_points_inside_image,
)


class IDWTests(unittest.TestCase):
    def test_distance_between_two_points(self):
        point = IDWPoint("P1", 3, 4, -50)
        self.assertEqual(euclidean_distance_px(0, 0, point), 5)

    def test_idw_weight_follows_inverse_distance_power(self):
        self.assertTrue(math.isinf(idw_weight(0, power=2.2)))
        self.assertAlmostEqual(idw_weight(2, power=2), 0.25)

    def test_interpolation_on_measured_point_returns_measured_rssi(self):
        points = [IDWPoint("P1", 10, 10, -35), IDWPoint("P2", 30, 10, -80)]
        self.assertEqual(idw_interpolate(10, 10, points), -35)

    def test_interpolation_between_two_equal_distance_points(self):
        points = [IDWPoint("P1", 0, 0, -40), IDWPoint("P2", 10, 0, -80)]
        self.assertAlmostEqual(idw_interpolate(5, 0, points, power=2.2), -60)

    def test_interpolation_uses_sum_rssi_over_distance_power(self):
        points = [IDWPoint("P1", 0, 0, -40), IDWPoint("P2", 20, 0, -80)]
        self.assertAlmostEqual(idw_interpolate(5, 0, points, power=2), -44)

    def test_grid_limits_follow_image_dimensions(self):
        points = [IDWPoint("P1", 0, 0, -40), IDWPoint("P2", 9, 9, -80)]
        grid = create_idw_grid(10, 10, points, step=5)
        self.assertEqual(grid.shape, (2, 2))

    def test_validate_points_inside_image(self):
        points = [IDWPoint("P1", 9, 9, -40), IDWPoint("P2", 10, 5, -80), IDWPoint("P3", 5, -1, -70)]
        errors = validate_points_inside_image(points, width=10, height=10)
        self.assertEqual(len(errors), 2)
        self.assertIn("P2", errors[0])
        self.assertIn("P3", errors[1])


if __name__ == "__main__":
    unittest.main()
