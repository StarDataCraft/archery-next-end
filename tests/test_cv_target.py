import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from src.cv_utils import normalize_hough_circles, normalize_hough_lines
from src.cv_target import (
    _detect_arrow_present,
    _detect_x_center,
    _fletched_shaft_candidates,
    _paired_edge_shaft_candidates,
    _refine_circle,
    propose_hit_points,
    rectify_target,
    transform_points,
)
from src.refine_points import _best_arrow_segment_in_roi
from src.target_face import render_target_face_bgr


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


class HitCandidateAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array(
            [
                (430.0, 440.0),
                (447.0, 425.0),
                (461.0, 448.0),
                (440.0, 468.0),
                (472.0, 465.0),
                (620.0, 540.0),
            ],
            dtype=np.float32,
        )
        self.clean_target = render_target_face_bgr(
            "80cm_10ring",
            size=900,
            center=(450.0, 450.0),
            outer_radius=405.0,
            draw_ring_lines=True,
            ring_line_thickness=2,
        )

    def _rectify_and_propose(self, bgr):
        result = rectify_target(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        proposed = propose_hit_points(
            result.rect_bgr,
            result.center_final,
            result.arrow_present,
            max_points=12,
            outer_radius=result.outer_radius,
        )
        return result, np.array(
            transform_points(proposed, result.M_rect_to_canon),
            dtype=np.float32,
        )

    def test_clean_target_does_not_invent_hit_points_from_ring_lines(self):
        _, proposed = self._rectify_and_propose(self.clean_target.copy())

        self.assertEqual(proposed.shape, (0,))

    def test_six_dark_holes_are_recovered_without_merging_a_tight_group(self):
        image = self.clean_target.copy()
        for x, y in self.truth.astype(int):
            cv2.circle(image, (int(x), int(y)), 7, (30, 30, 30), thickness=-1)

        result, proposed = self._rectify_and_propose(image)

        self.assertTrue(result.debug["ellipse_rotation_skipped"])
        self.assertEqual(len(proposed), len(self.truth))
        nearest_errors = [
            float(np.min(np.linalg.norm(proposed - expected, axis=1)))
            for expected in self.truth
        ]
        self.assertLess(max(nearest_errors), 4.0)

    def test_affine_camera_squash_is_corrected_before_mapping_points(self):
        image = self.clean_target.copy()
        for x, y in self.truth.astype(int):
            cv2.circle(image, (int(x), int(y)), 7, (30, 30, 30), thickness=-1)
        squash = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.75, 450.0 * 0.25]],
            dtype=np.float32,
        )
        distorted = cv2.warpAffine(
            image,
            squash,
            (900, 900),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        _, proposed = self._rectify_and_propose(distorted)

        self.assertEqual(len(proposed), len(self.truth))
        nearest_errors = [
            float(np.min(np.linalg.norm(proposed - expected, axis=1)))
            for expected in self.truth
        ]
        self.assertLess(max(nearest_errors), 5.0)

    def test_blurred_photo_does_not_emit_confident_but_wrong_points(self):
        image = self.clean_target.copy()
        for x, y in self.truth.astype(int):
            cv2.circle(image, (int(x), int(y)), 7, (30, 30, 30), thickness=-1)
        blurred = cv2.GaussianBlur(image, (5, 5), 1.1)

        result, proposed = self._rectify_and_propose(blurred)

        self.assertIn("image_blur", result.quality_flags)
        self.assertEqual(proposed.shape, (0,))

    def test_worn_real_target_traces_current_shafts_instead_of_old_holes(self):
        fixture = Path(__file__).parent / "fixtures" / "worn_target_five_arrows.jpg"
        image = cv2.imread(str(fixture))
        self.assertIsNotNone(image)

        result = rectify_target(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        diagnostics = {}
        proposed_rect = propose_hit_points(
            result.rect_bgr,
            result.center_final,
            result.arrow_present,
            max_points=12,
            outer_radius=result.outer_radius,
            diagnostics=diagnostics,
        )
        proposed = np.array(
            transform_points(proposed_rect, result.M_rect_to_canon),
            dtype=np.float32,
        )
        expected = np.array(
            [
                (366.8, 392.5),
                (427.9, 406.1),
                (351.0, 412.1),
                (448.1, 522.1),
                (625.3, 531.6),
            ],
            dtype=np.float32,
        )

        self.assertEqual(result.debug["rectify_mode"], "colored_target")
        self.assertEqual(diagnostics["mode"], "visible_shafts")
        self.assertEqual(len(proposed), len(expected))
        nearest_errors = [
            float(np.min(np.linalg.norm(proposed - point, axis=1)))
            for point in expected
        ]
        self.assertLess(max(nearest_errors), 12.0)

    def test_paired_edges_recover_the_unfletched_shaft_without_hough_votes(self):
        fixture = Path(__file__).parent / "fixtures" / "worn_target_five_arrows.jpg"
        image = cv2.imread(str(fixture))
        result = rectify_target(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        fletched, _ = _fletched_shaft_candidates(
            result.rect_bgr,
            result.center_final,
            result.outer_radius,
        )
        paired = _paired_edge_shaft_candidates(
            result.rect_bgr,
            result.center_final,
            result.outer_radius,
            fletched,
        )

        self.assertGreaterEqual(len(paired), 1)
        contacts = np.array([(x, y) for _, x, y in paired], dtype=np.float32)
        self.assertLess(
            float(np.min(np.linalg.norm(contacts - np.array((625.3, 531.6)), axis=1))),
            6.0,
        )


if __name__ == "__main__":
    unittest.main()
