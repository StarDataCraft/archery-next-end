import unittest

from src.ui_analyze import (
    CANON_SIZE,
    CANVAS_SIZE,
    _extract_points_from_canvas,
    _image_data_url,
    _points_to_initial_drawing,
    _sanitize_canonical_points,
    _scale_points,
    _update_points_from_canvas,
)


class CanvasPointTests(unittest.TestCase):
    def test_embedded_canvas_background_is_noninteractive(self):
        import numpy as np

        background = np.zeros((4, 5, 3), dtype=np.uint8)
        drawing = _points_to_initial_drawing([], background_rgb=background)
        image_object = drawing["objects"][0]

        self.assertEqual(image_object["type"], "image")
        self.assertFalse(image_object["selectable"])
        self.assertFalse(image_object["evented"])
        self.assertTrue(image_object["src"].startswith("data:image/png;base64,"))
        self.assertEqual(_image_data_url(background), image_object["src"])

    def test_canonical_canvas_round_trip(self):
        canonical = [{"x": 450.0, "y": 225.0}, {"x": 900.0, "y": 0.0}]

        canvas = _scale_points(canonical, CANVAS_SIZE / CANON_SIZE)
        restored = _scale_points(canvas, CANON_SIZE / CANVAS_SIZE)

        for actual, expected in zip(restored, canonical):
            self.assertAlmostEqual(actual["x"], expected["x"])
            self.assertAlmostEqual(actual["y"], expected["y"])

    def test_empty_component_payload_preserves_confirmed_points(self):
        current = [{"x": 300.0, "y": 400.0}]

        self.assertEqual(_update_points_from_canvas(current, None), current)
        self.assertEqual(
            _update_points_from_canvas(current, {"objects": []}),
            current,
        )

    def test_canvas_points_are_converted_to_canonical_coordinates(self):
        payload = {
            "objects": [
                {
                    "type": "circle",
                    "left": 342.0,
                    "top": 167.0,
                    "radius": 8.0,
                }
            ]
        }

        points = _update_points_from_canvas([], payload)

        self.assertAlmostEqual(points[0]["x"], 450.0)
        self.assertAlmostEqual(points[0]["y"], 225.0)

    def test_stale_smaller_canvas_payload_does_not_drop_a_confirmed_point(self):
        current = [
            {"x": 100.0 + index, "y": 200.0 + index}
            for index in range(6)
        ]
        stale_payload = {
            "objects": [
                {
                    "type": "circle",
                    "left": 70.0 + index,
                    "top": 150.0 + index,
                    "radius": 8.0,
                }
                for index in range(5)
            ]
        }

        self.assertEqual(_update_points_from_canvas(current, stale_payload), current)

    def test_fabric_circle_scale_is_included_in_center(self):
        payload = {
            "objects": [
                {
                    "type": "circle",
                    "left": 10,
                    "top": 20,
                    "radius": 5,
                    "scaleX": 2,
                    "scaleY": 3,
                }
            ]
        }

        self.assertEqual(
            _extract_points_from_canvas(payload),
            [{"x": 20.0, "y": 35.0}],
        )

    def test_invalid_or_off_canvas_points_are_removed(self):
        points = [
            {"x": 10, "y": 20},
            {"x": float("nan"), "y": 30},
            {"x": -1, "y": 30},
            {"x": CANON_SIZE + 1, "y": 30},
            "bad",
        ]

        self.assertEqual(
            _sanitize_canonical_points(points),
            [{"x": 10.0, "y": 20.0}],
        )


if __name__ == "__main__":
    unittest.main()
