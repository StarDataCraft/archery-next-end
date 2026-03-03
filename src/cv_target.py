from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


@dataclass
class TargetRectifyResult:
    rect_bgr: np.ndarray
    center: Tuple[float, float]   # in rect coords
    outer_radius: float           # in rect coords
    arrow_present: bool
    debug: Dict[str, object]


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _largest_contour(edge: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours[0]


def _affine_rectify_by_ellipse(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Fit ellipse to outer target boundary and apply an affine transform so that ellipse becomes closer to circle.
    This handles most typical phone-angle distortions (mild perspective -> ellipse).

    Returns rectified_bgr (same size as input) and debug info.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(gray, 60, 140)

    cnt = _largest_contour(edges)
    dbg: Dict[str, object] = {"ellipse_found": False}

    if cnt is None or len(cnt) < 50:
        return bgr.copy(), {"ellipse_found": False, "fallback": "no_contour"}

    if len(cnt) < 5:
        return bgr.copy(), {"ellipse_found": False, "fallback": "contour_too_small"}

    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (a, b), angle = ellipse  # axes lengths a,b; angle in degrees
    # ensure a is major axis
    major = max(a, b)
    minor = min(a, b)
    if minor < 1e-6:
        return bgr.copy(), {"ellipse_found": False, "fallback": "minor_zero"}

    dbg.update({
        "ellipse_found": True,
        "ellipse_center": (float(cx), float(cy)),
        "ellipse_axes": (float(a), float(b)),
        "ellipse_angle": float(angle),
    })

    # Rotate around center so major axis aligns with x-axis, then scale y to make minor -> major.
    rot = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)  # 2x3
    rotated = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    scale = major / minor
    # Scale in y-direction around center after rotation (approx):
    # y' = cy + scale*(y - cy)
    # x' = x
    S = np.array([[1.0, 0.0, 0.0],
                  [0.0, scale, cy * (1.0 - scale)]], dtype=np.float32)

    rect = cv2.warpAffine(rotated, S, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    dbg["affine_scale_y"] = float(scale)
    return rect, dbg


def _refine_circle(bgr: np.ndarray) -> Tuple[Tuple[float, float], float, Dict[str, object]]:
    """
    Find the outer circle on rectified image via HoughCircles.
    Fallback: use center of image.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.2,
        param1=120,
        param2=35,
        minRadius=int(min(h, w) * 0.15),
        maxRadius=int(min(h, w) * 0.49),
    )

    dbg: Dict[str, object] = {"circle_found": False}
    if circles is None:
        return (w / 2.0, h / 2.0), min(w, h) * 0.45, {"circle_found": False, "fallback": "no_hough"}

    circles = np.round(circles[0, :]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    dbg.update({"circle_found": True, "circle_xy_r": (int(x), int(y), int(r))})
    return (float(x), float(y)), float(r), dbg


def _crop_square_around_circle(bgr: np.ndarray, center: Tuple[float, float], radius: float, out_size: int) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """
    Crop a square around detected circle and resize to out_size.
    """
    h, w = bgr.shape[:2]
    cx, cy = center

    margin = int(radius * 0.10)
    half = int(radius + margin)

    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, int(cx) + half)
    y2 = min(h, int(cy) + half)

    crop = bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        resized = cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return resized, (out_size / 2.0, out_size / 2.0), min(out_size, out_size) * 0.45

    crop_h, crop_w = crop.shape[:2]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    sx = out_size / crop_w
    sy = out_size / crop_h
    new_cx = (cx - x1) * sx
    new_cy = (cy - y1) * sy
    new_r = radius * (sx + sy) / 2.0
    return resized, (float(new_cx), float(new_cy)), float(new_r)


def _detect_arrow_present(bgr: np.ndarray) -> Tuple[bool, int]:
    """
    Heuristic: if there are multiple long line segments, assume arrows are present.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=90,
        minLineLength=int(min(h, w) * 0.18),
        maxLineGap=12,
    )
    if lines is None:
        return False, 0

    # Count reasonably long lines
    cnt = 0
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length >= min(h, w) * 0.22:
            cnt += 1
    return cnt >= 2, cnt


def rectify_target(image_rgb: np.ndarray, out_size: int = 900) -> TargetRectifyResult:
    """
    Main entry:
    - affine rectify via ellipse
    - refine circle
    - crop+resize to canonical square
    - decide arrow-present
    """
    bgr = _rgb_to_bgr(image_rgb)

    rect1, dbg1 = _affine_rectify_by_ellipse(bgr)
    (cx, cy), r, dbg2 = _refine_circle(rect1)
    rect2, (rcx, rcy), rr = _crop_square_around_circle(rect1, (cx, cy), r, out_size)

    arrow_present, line_count = _detect_arrow_present(rect2)

    debug = {}
    debug.update(dbg1)
    debug.update(dbg2)
    debug.update({"arrow_present": arrow_present, "line_count": int(line_count)})
    return TargetRectifyResult(
        rect_bgr=rect2,
        center=(rcx, rcy),
        outer_radius=rr,
        arrow_present=arrow_present,
        debug=debug,
    )


def _dedupe_points(points: List[Tuple[float, float]], min_dist: float = 18.0) -> List[Tuple[float, float]]:
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


def _sort_by_center(points: List[Tuple[float, float]], center: Tuple[float, float]) -> List[Tuple[float, float]]:
    cx, cy = center
    return sorted(points, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)


def propose_hit_points(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    arrow_present: bool,
    max_points: int = 12,
) -> List[Tuple[float, float]]:
    """
    Two-branch candidate proposal:
    - arrow_present: line endpoints -> choose endpoint closer to center
    - no_arrow: black-hat blob detection for holes/dark marks
    """
    h, w = rect_bgr.shape[:2]
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)

    candidates: List[Tuple[float, float]] = []

    if arrow_present:
        edges = cv2.Canny(gray, 70, 160)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=90,
            minLineLength=int(min(h, w) * 0.18),
            maxLineGap=14,
        )
        if lines is not None:
            cx, cy = center
            for (x1, y1, x2, y2) in lines[:, 0, :]:
                # choose endpoint closer to center as "hit-side" endpoint
                d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
                d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
                if d1 < d2:
                    candidates.append((float(x1), float(y1)))
                else:
                    candidates.append((float(x2), float(y2)))

    else:
        # black-hat for dark small blobs (holes)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        bh = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
        _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 5)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if area < 25 or area > 1800:
                continue
            cx, cy = centroids[i]
            if cx < 10 or cy < 10 or cx > w - 10 or cy > h - 10:
                continue
            candidates.append((float(cx), float(cy)))

    candidates = _dedupe_points(candidates, min_dist=20.0)
    candidates = _sort_by_center(candidates, center)
    return candidates[:max_points]


def draw_overlay(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    ring_radii: List[float],
    points_xy: List[Tuple[float, float]],
    scores: List[int],
) -> np.ndarray:
    """
    Draw ring circles + hit points + per-hit score.
    """
    overlay = rect_bgr.copy()
    cx, cy = int(center[0]), int(center[1])

    # rings
    for r in ring_radii:
        cv2.circle(overlay, (cx, cy), int(r), (0, 255, 255), 2)  # yellow-ish

    # center mark
    cv2.circle(overlay, (cx, cy), 4, (0, 255, 255), -1)

    # hits
    for i, (x, y) in enumerate(points_xy):
        px, py = int(x), int(y)
        cv2.circle(overlay, (px, py), 9, (255, 0, 0), 2)  # blue circle
        s = scores[i] if i < len(scores) else None
        if s is not None:
            cv2.putText(
                overlay,
                str(s),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return overlay
