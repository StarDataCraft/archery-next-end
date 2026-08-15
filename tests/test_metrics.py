import math
import unittest

from src.metrics import compute_metrics


class ComputeMetricsTests(unittest.TestCase):
    def test_principal_axis_is_real_on_numpy_complex_eig_case(self):
        points = [
            {"x": 449.976806640625, "y": 391.9542541503906},
            {"x": 449.9708251953125, "y": 508.33880615234375},
            {"x": 371.6979675292969, "y": 373.7781066894531},
            {"x": 529.174560546875, "y": 527.2056884765625},
            {"x": 540.4117431640625, "y": 537.19140625},
            {"x": 532.3660888671875, "y": 370.2581787109375},
        ]

        metrics = compute_metrics(points, center=(450.0, 450.0), outer_radius_px=405.0)

        self.assertTrue(math.isfinite(metrics["slope_deg"]))
        self.assertEqual(metrics["n"], 6)

    def test_single_point_has_zero_slope(self):
        metrics = compute_metrics([{"x": 10.0, "y": 20.0}])

        self.assertEqual(metrics["slope_deg"], 0.0)
        self.assertEqual(metrics["spread"], 0.0)


if __name__ == "__main__":
    unittest.main()
