from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def _wa_colors_bgr():
    """
    World Archery standard face colors (approx, BGR):
    - White (1-2)
    - Black (3-4)
    - Blue  (5-6)
    - Red   (7-8)
    - Gold  (9-10)
    """
    return {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "blue":  (255, 0, 0),
        "red":   (0, 0, 255),
        "gold":  (0, 215, 255),
    }


def _color_for_score(score: int, colors: dict):
    if score in (1, 2):
        return colors["white"]
    if score in (3, 4):
        return colors["black"]
    if score in (5, 6):
        return colors["blue"]
    if score in (7, 8):
        return colors["red"]
    return colors["gold"]


def render_target_face_bgr(
    target_face: str,
    size: int,
    center: Tuple[float, float],
    outer_radius: float,
    draw_ring_lines: bool = True,
    ring_line_thickness: int = 2,
) -> np.ndarray:
    """
    Generate a clean, standard WA target face image (no photo background).
    Uses a canonical band-filling method:
      draw score=1 circle at radius=1.0R (white),
      draw score=2 circle at radius=0.9R (white),
      draw score=3 circle at radius=0.8R (black),
      ...
      draw score=10 circle at radius=0.1R (gold).
    Each inner circle overwrites the inside region, producing correct colored bands.
    """
    colors = _wa_colors_bgr()
    img = np.zeros((size, size, 3), dtype=np.uint8)

    cx, cy = int(center[0]), int(center[1])
    R = float(outer_radius)

    # Fill from outermost inward with correct radii mapping.
    # score 1 boundary radius = 1.0R
    # score 2 boundary radius = 0.9R
    # ...
    # score 10 boundary radius = 0.1R
    for score in range(1, 11):
        boundary = (11 - score) / 10.0  # 1.0, 0.9, ..., 0.1
        r = R * boundary
        col = _color_for_score(score, colors)
        cv2.circle(img, (cx, cy), int(round(r)), col, thickness=-1)

    # Ring boundary lines (black)
    if draw_ring_lines:
        for k in range(1, 11):
            rr = R * (k / 10.0)
            cv2.circle(img, (cx, cy), int(round(rr)), (0, 0, 0), thickness=ring_line_thickness)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), thickness=-1)

    return img
