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
    - target_face: "80cm_10ring" or "40cm_10ring" (same color layout)
    - size: output image size (size x size)
    - center/outer_radius: geometry in the same coordinate system as your hit points
    """
    colors = _wa_colors_bgr()
    img = np.zeros((size, size, 3), dtype=np.uint8)

    cx, cy = int(center[0]), int(center[1])
    R = float(outer_radius)

    # 10 ring boundaries radii (outer = 10/10)
    radii = [R * (k / 10.0) for k in range(1, 11)]

    # Draw filled color bands from outer -> inner.
    # Ring numbers: 1..10 (outer to inner)
    # Colors by score:
    # 1-2 white, 3-4 black, 5-6 blue, 7-8 red, 9-10 gold
    def color_for_score(score: int):
        if score in (1, 2):
            return colors["white"]
        if score in (3, 4):
            return colors["black"]
        if score in (5, 6):
            return colors["blue"]
        if score in (7, 8):
            return colors["red"]
        return colors["gold"]

    # Fill from outermost to innermost (score 1 -> 10)
    for score in range(1, 11):
        r_outer = radii[score - 1]
        col = color_for_score(score)
        cv2.circle(img, (cx, cy), int(r_outer), col, thickness=-1)

    # Optional: draw black ring boundary lines
    if draw_ring_lines:
        for r in radii:
            cv2.circle(img, (cx, cy), int(r), (0, 0, 0), thickness=ring_line_thickness)
        # small center dot
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), thickness=-1)

    return img
