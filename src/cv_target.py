from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


@dataclass
class TargetWarpResult:
    warped_bgr: np.ndarray
    center: Tuple[float, float]          # in warped coords
    radius: float                         # in warped coords
    debug: Dict[str, object]


def _to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    # image_rgb: HxWx3 RGB uint8
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def detect_target_and_warp(image_rgb: np.ndarray, out_size: int = 900) -> TargetWarpResult:
    """
    High-ROI MVP warp:
    - detect target circle (HoughCircles)
    - crop a square around it
    - resize to out_size x out_size

    This is not full projective rectification, but it stabilizes scale + framing
    and works well for typical phone photos with mild perspective.
    """
    bgr = _to_bgr(image_rgb)
    h, w = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Hough circle params are sensitive; these are reasonable defaults for archery targets.
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.2,
        param1=120,
        param2=35,
        minRadius=int(min(h, w) * 0.15),
        maxRadius=int(min(h, w) * 0.48),
    )

    debug: Dict[str, object] = {"circle_found": False}

    if circles is None:
        # fallback: no warp, just resize whole image (still lets pipeline run)
        warped = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return TargetWarpResult(
            warped_bgr=warped,
            center=(out_size / 2, out_size / 2),
            radius=min(out_size, out_size) * 0.45,
            debug={"circle_found": False, "fallback": "resize_full_image"},
        )

    circles = np.round(circles[0, :]).astype(int)

    # pick the circle with largest radius (usually outer ring)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    debug.update({"circle_found": True, "circle_xy_r": (int(x), int(y), int(r))})

    # crop square around circle with margin
    margin = int(r * 0.12)
    half = r + margin

    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(w, x + half)
    y2 = min(h, y + half)

    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        warped = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return TargetWarpResult(
            warped_bgr=warped,
            center=(out_size / 2, out_size / 2),
            radius=min(out_size, out_size) * 0.45,
            debug={"circle_found": False, "fallback": "empty_crop_resize_full_image"},
        )

    warped = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Map center/radius to warped coords
    crop_h, crop_w = crop.shape[:2]
    sx = out_size / crop_w
    sy = out_size / crop_h

    cx_w = (x - x1) * sx
    cy_w = (y - y1) * sy
    r_w = r * (sx + sy) / 2

    return TargetWarpResult(
        warped_bgr=warped,
        center=(float(cx_w), float(cy_w)),
        radius=float(r_w),
        debug=debug,
    )


def _dedupe_points(points: List[Tuple[float, float]], min_dist: float = 18.0) -> List[Tuple[float, float]]:
    """Simple greedy NMS in point space."""
    kept: List[Tuple[float, float]] = []
    for (x, y) in points:
        ok = True
        for (kx, ky) in kept:
            if (x - kx) ** 2 + (y - ky) ** 2 < min_dist ** 2:
                ok = False
                break
        if ok:
            kept.append((x, y))
    return kept


def _score_by_center_distance(
    pts: List[Tuple[float, float]], center: Tuple[float, float]
) -> List[Tuple[float, float]]:
    cx, cy = center
    return sorted(pts, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


def propose_hit_points(
    warped_bgr: np.ndarray,
    center: Tuple[float, float],
    max_points: int = 12,
) -> List[Tuple[float, float]]:
    """
    Traditional CV candidate hit-point proposal.
    Mix two sources:
    1) Dark small blobs (arrow holes / arrow tip region) via black-hat + blob detection
    2) Line endpoints (arrow shafts) via HoughLinesP
    Then dedupe + sort by closeness to center (usually shots cluster around center).
    """
    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

    candidates: List[Tuple[float, float]] = []

    # --- Source 1: black-hat to highlight dark small spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    # normalize + threshold
    bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove tiny noise
    th = cv2.medianBlur(th, 5)

    # connected components as blob candidates
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 25 or area > 2000:
            continue
        cx, cy = centroids[i]
        # filter out borders
        if cx < 10 or cy < 10 or cx > w - 10 or cy > h - 10:
            continue
        candidates.append((float(cx), float(cy)))

    # --- Source 2: arrow shaft line endpoints (helps when holes are unclear)
    edges = cv2.Canny(gray, 70, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(h, w) * 0.10),
        maxLineGap=12,
    )

    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, line)
            # take both endpoints; later dedupe will remove overlaps
            candidates.append((float(x1), float(y1)))
            candidates.append((float(x2), float(y2)))

    # Dedupe and keep the most plausible ones near center first
    candidates = _dedupe_points(candidates, min_dist=20.0)
    candidates = _score_by_center_distance(candidates, center)

    # Keep top-N
    return candidates[:max_points]
