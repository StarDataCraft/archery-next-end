import unittest
from unittest.mock import patch

import numpy as np

from src.cv_utils import normalize_hough_circles, normalize_hough_lines
from src.cv_target import (
    _detect_arrow_present,
    _detect_x_center,
    _refine_circle,
    propose_hit_points,
    rectify_target,
)
from src.refine_points import _best_arrow_segment_in_roi


class NormalizeHoughLinesTests(unittest.TestCase):
    def test_accepts_common_opencv_shapes(self):
        expected = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)

        for value in (expected, expected.reshape(2, 1, 4), expected.reshape(-1)):
            with self.subTest(shape=value.shape):
                np.testing.assert_array_equal(normalize_hough_lines(value), expected)

    def test_malformed_or_non_finite_results_are_safe(self):
        self.assertEqual(normalize_hough_lines(None).shape, (0, 4))
        self.assertEqual(normalize_hough_lines(np.array([1, 2, 3])).shape, (0, 4))

        value = np.array([[1, 2, 3, 4], [5, np.nan, 7, 8]])
        np.testing.assert_array_equal(
            normalize_hough_lines(value),
            np.array([[1, 2, 3, 4]], dtype=np.float32),
        )

    def test_circle_normalization_accepts_flat_and_nested_shapes(self):
        expected = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)

        for value in (expected, expected.reshape(1, 2, 3), expected.reshape(-1)):
            with self.subTest(shape=value.shape):
                np.testing.assert_array_equal(normalize_hough_circles(value), expected)


class HoughFallbackTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((400, 400, 3), dtype=np.uint8)

    @patch("src.cv_target.cv2.HoughLinesP")
    def test_arrow_detection_accepts_flat_single_segment(self, hough):
        hough.return_value = np.array([0, 0, 120, 0], dtype=np.int32)

        present, count = _detect_arrow_present(self.image)

        self.assertFalse(present)
        self.assertEqual(count, 1)

    @patch("src.cv_target.cv2.HoughLinesP")
    def test_arrow_detection_ignores_malformed_result(self, hough):
        hough.return_value = np.array([1, 2, 3], dtype=np.int32)

        self.assertEqual(_detect_arrow_present(self.image), (False, 0))

    @patch("src.cv_target.cv2.HoughLinesP")
    def test_x_center_and_point_proposal_fall_back_without_crashing(self, hough):
        hough.return_value = np.array([1, 2, 3], dtype=np.int32)

        center, debug = _detect_x_center(self.image, (200.0, 200.0))
        points = propose_hit_points(self.image, (200.0, 200.0), arrow_present=True)

        self.assertIsNone(center)
        self.assertEqual(debug["fallback"], "no_lines")
        self.assertEqual(points, [])

    def test_full_pipeline_handles_featureless_image(self):
        result = rectify_target(np.zeros((360, 480, 3), dtype=np.uint8))

        self.assertEqual(result.rect_bgr.shape, (900, 900, 3))
        self.assertEqual(result.M_rect_to_canon.shape, (2, 3))
        self.assertFalse(result.arrow_present)

    @patch("src.cv_target.cv2.HoughCircles")
    def test_circle_detection_accepts_flat_single_circle(self, hough):
        hough.return_value = np.array([120, 130, 80], dtype=np.float32)

        center, radius, debug = _refine_circle(self.image)

        self.assertEqual(center, (120.0, 130.0))
        self.assertEqual(radius, 80.0)
        self.assertTrue(debug["circle_found"])

    @patch("src.refine_points.cv2.HoughLinesP")
    @patch("src.refine_points.cv2.createLineSegmentDetector")
    def test_refinement_hough_fallback_normalizes_shape(self, create_lsd, hough):
        create_lsd.return_value.detect.return_value = (None, None, None, None)
        hough.return_value = np.array([3, 4, 80, 90], dtype=np.int32)

        line = _best_arrow_segment_in_roi(
            np.zeros((120, 120), dtype=np.uint8), min_len=20
        )

        self.assertEqual(line, (3, 4, 80, 90))


if __name__ == "__main__":
    unittest.main()
