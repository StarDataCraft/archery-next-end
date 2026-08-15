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

    def test_detects_one_escaped_arrow_without_calling_core_group_loose(self):
        points = [
            {"x": 446.0, "y": 448.0},
            {"x": 451.0, "y": 452.0},
            {"x": 454.0, "y": 447.0},
            {"x": 448.0, "y": 455.0},
            {"x": 452.0, "y": 450.0},
            {"x": 610.0, "y": 530.0},
        ]

        result = compute_metrics(points, center=(450.0, 450.0), outer_radius_px=405.0)

        self.assertTrue(result["outlier"]["present"])
        self.assertEqual(result["outlier"]["index"], 5)
        self.assertLess(result["outlier"]["core_spread"], result["spread"] * 0.25)

    def test_does_not_remove_an_arrow_from_a_genuinely_wide_group(self):
        points = [
            {"x": 330.0, "y": 450.0},
            {"x": 390.0, "y": 360.0},
            {"x": 500.0, "y": 365.0},
            {"x": 570.0, "y": 445.0},
            {"x": 500.0, "y": 540.0},
            {"x": 390.0, "y": 535.0},
        ]

        result = compute_metrics(points, center=(450.0, 450.0), outer_radius_px=405.0)

        self.assertFalse(result["outlier"]["present"])


if __name__ == "__main__":
    unittest.main()
