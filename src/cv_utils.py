from __future__ import annotations

import numpy as np


def _normalize_detection_rows(values: object, columns: int) -> np.ndarray:
    if values is None:
        return np.empty((0, columns), dtype=np.float32)

    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return np.empty((0, columns), dtype=np.float32)

    if array.size == 0 or array.size % columns != 0:
        return np.empty((0, columns), dtype=np.float32)

    try:
        rows = array.reshape(-1, columns).astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return np.empty((0, columns), dtype=np.float32)

    return rows[np.isfinite(rows).all(axis=1)]


def normalize_hough_lines(lines: object) -> np.ndarray:
    """Return Hough line segments as a safe ``(N, 4)`` float array.

    OpenCV normally returns ``(N, 1, 4)`` from ``HoughLinesP``, but builds and
    wrappers can also expose ``(N, 4)`` or a single ``(4,)`` segment. Treat an
    empty or malformed result as no detection instead of crashing the app.
    """
    return _normalize_detection_rows(lines, columns=4)


def normalize_hough_circles(circles: object) -> np.ndarray:
    """Return Hough circles as a safe ``(N, 3)`` float array."""
    return _normalize_detection_rows(circles, columns=3)
