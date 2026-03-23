# src/target_face.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import cv2

TARGET_FACES: Dict[str, Dict[str, Any]] = {
    "80cm_10ring": {"label_key": "target_80_10"},
    "40cm_10ring": {"label_key": "target_40_10"},
    "60cm_10ring": {"label_key": "target_60_10"},
    "122cm_10ring": {"label_key": "target_122_10"},
}


def _wa_colors_bgr():
    return {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "blue": (255, 0, 0),
        "red": (0, 0, 255),
        "gold": (0, 215, 255),
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
    Render a clean WA-style 10-ring face.
    target_face currently affects UI labeling / semantics (size), while rendering is canonical.
    """
    if target_face not in TARGET_FACES:
        target_face = "80cm_10ring"

    colors = _wa_colors_bgr()
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = int(center[0]), int(center[1])
    R = float(outer_radius)

    # Fill from outermost inward
    for score in range(1, 11):
        boundary = (11 - score) / 10.0  # 1.0, 0.9, ..., 0.1
        r = R * boundary
        col = _color_for_score(score, colors)
        cv2.circle(img, (cx, cy), int(round(r)), col, thickness=-1)

    if draw_ring_lines:
        for k in range(1, 11):
            rr = R * (k / 10.0)
            cv2.circle(img, (cx, cy), int(round(rr)), (0, 0, 0), thickness=ring_line_thickness)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), thickness=-1)

    return img
