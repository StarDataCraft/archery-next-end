from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def _rings_80cm_10ring() -> list:
    return [
        ("yellow", [10, 9]),
        ("red", [8, 7]),
        ("blue", [6, 5]),
        ("black", [4, 3]),
        ("white", [2, 1]),
    ]


def _rings_40cm_10ring() -> list:
    return _rings_80cm_10ring()


def _color_bgr(name: str) -> Tuple[int, int, int]:
    if name == "yellow":
        return (0, 215, 255)
    if name == "red":
        return (0, 0, 220)
    if name == "blue":
        return (220, 80, 0)
    if name == "black":
        return (0, 0, 0)
    if name == "white":
        return (245, 245, 245)
    return (200, 200, 200)


def render_target_face_bgr(
    face: str,
    size: int,
    center: Tuple[float, float],
    outer_radius: float,
    draw_ring_lines: bool = True,
    ring_line_thickness: int = 2,
) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    img[:] = (30, 30, 30)

    bands = _rings_40cm_10ring() if face == "40cm_10ring" else _rings_80cm_10ring()

    ring_step = outer_radius / 10.0
    ring_r = {score: ring_step * (11 - score) for score in range(1, 11)}

    for band_name, scores in bands[::-1]:
        r_out = ring_r[min(scores)]
        cv2.circle(img, (cx, cy), int(round(r_out)), _color_bgr(band_name), thickness=-1)

    if draw_ring_lines:
        for s in range(1, 11):
            r = ring_r[s]
            cv2.circle(img, (cx, cy), int(round(r)), (0, 0, 0), thickness=ring_line_thickness)

    cv2.circle(img, (cx, cy), int(round(ring_step * 0.35)), (0, 0, 0), thickness=2)
    return img
