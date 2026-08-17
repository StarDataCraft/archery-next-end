# src/cv_target.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Sequence
import numpy as np
import cv2

from .cv_utils import normalize_hough_circles, normalize_hough_lines


@dataclass
class TargetRectifyResult:
    rect_bgr: np.ndarray  # rectified photo (square)
    circle_center: Tuple[float, float]  # center from circle (rect coords)
    outer_radius: float  # outer radius (rect coords)

    # refined pose
    midline_y: Optional[float]
    x_center: Optional[Tuple[float, float]]
    center_final: Tuple[float, float]

    # mapping from rect coords -> canonical coords (900x900)
    M_rect_to_canon: np.ndarray

    arrow_present: bool
    debug: Dict[str, object]

    # quality summary
    quality_score: float
    quality_flags: List[str]


CANON_SIZE = 900
CANON_CENTER = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)
CANON_OUTER = CANON_SIZE * 0.45  # 405
MIN_IMAGE_SHARPNESS = 90.0
MIN_IMAGE_CONTRAST = 170.0
MAX_DARK_CENTER_FRACTION = 0.035
MAX_GLARE_CENTER_FRACTION = 0.040


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _largest_contour(edge: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _odd_kernel(value: float, minimum: int = 3) -> int:
    size = max(int(minimum), int(round(float(value))))
    return size if size % 2 == 1 else size + 1


def _normalize_gray_contrast(gray: np.ndarray) -> np.ndarray:
    """Return a percentile-stretched grayscale image for edge detection.

    Fixed Canny/LSD thresholds otherwise lose thin carbon shafts in dark or
    low-contrast uploads even though the same geometry is plainly visible.
    Percentiles avoid letting a few clipped highlights or black holes set the
    whole range.
    """
    source = np.asarray(gray, dtype=np.uint8)
    low, high = np.percentile(source, (2.0, 98.0))
    if float(high - low) < 24.0:
        return source.copy()
    stretched = (source.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0.0, 255.0).astype(np.uint8)


def _gray_contrast_span(gray: np.ndarray) -> float:
    low, high = np.percentile(np.asarray(gray, dtype=np.uint8), (2.0, 98.0))
    return float(high - low)


def _exposure_diagnostics(
    bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
) -> Dict[str, float]:
    """Measure severe underexposure and washed-out glare near the scoring area."""
    h, w = bgr.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    central = (
        np.hypot(xx - float(center[0]), yy - float(center[1]))
        < float(outer_radius) * 0.65
    )
    if not np.any(central):
        return {"image_dark_fraction": 0.0, "image_glare_fraction": 0.0}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dark_fraction = float(np.mean(gray[central] < 20))
    # Real target colours are bright but strongly saturated. Glare instead
    # creates broad near-white patches, so require both high value and reduced
    # saturation to avoid flagging clean gold/red/blue printing.
    glare_fraction = float(
        np.mean((hsv[:, :, 2][central] >= 245) & (hsv[:, :, 1][central] < 180))
    )
    return {
        "image_dark_fraction": dark_fraction,
        "image_glare_fraction": glare_fraction,
    }


def _fit_colored_target_ellipse(
    bgr: np.ndarray,
) -> Tuple[Optional[Tuple[Tuple[float, float], Tuple[float, float], float]], Dict[str, object]]:
    """Fit the outer red scoring-zone boundary.

    The red zone is a much safer target reference than the largest edge
    contour: arrows, fletchings and the edge of the backing board can all be
    larger than a partially framed target face. On a WA-style face the outer
    red boundary is 40% of the full scoring radius, so one fitted ellipse gives
    both the target centre and the canonical scale.
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    red = (((hue < 16) | (hue > 170)) & (sat > 70) & (val > 50)).astype(np.uint8) * 255

    close_size = _odd_kernel(min(h, w) * 0.016, 9)
    open_size = _odd_kernel(min(h, w) * 0.005, 3)
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    red = cv2.morphologyEx(
        red,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size)),
    )

    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dbg: Dict[str, object] = {
        "color_ring_found": False,
        "color_ring_mask_ratio": float(red.mean() / 255.0),
    }
    if not contours:
        dbg["fallback"] = "no_red_contour"
        return None, dbg

    contour = max(contours, key=cv2.contourArea)
    area_ratio = float(cv2.contourArea(contour) / max(h * w, 1))
    dbg["color_ring_contour_area_ratio"] = area_ratio
    if len(contour) < 20 or area_ratio < 0.025:
        dbg["fallback"] = "red_contour_too_small"
        return None, dbg

    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (axis_a, axis_b), angle = ellipse
    minor = min(float(axis_a), float(axis_b))
    major = max(float(axis_a), float(axis_b))
    if minor < min(h, w) * 0.18 or major > max(h, w) * 1.65:
        dbg["fallback"] = "red_ellipse_implausible"
        return None, dbg
    if not (-0.15 * w <= cx <= 1.15 * w and -0.15 * h <= cy <= 1.15 * h):
        dbg["fallback"] = "red_ellipse_center_outside"
        return None, dbg

    points = contour[:, 0, :].astype(np.float64)
    theta = np.deg2rad(float(angle))
    axis_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    axis_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    centered = points - np.array([cx, cy], dtype=np.float64)
    normalized_radius = np.sqrt(
        (centered @ axis_u / max(float(axis_a) / 2.0, 1e-6)) ** 2
        + (centered @ axis_v / max(float(axis_b) / 2.0, 1e-6)) ** 2
    )
    fit_error_p90 = float(np.percentile(np.abs(normalized_radius - 1.0), 90))
    if fit_error_p90 > 0.38:
        dbg.update({"fallback": "red_ellipse_fit_error", "color_ring_fit_error_p90": fit_error_p90})
        return None, dbg

    dbg.update(
        {
            "color_ring_found": True,
            "color_ring_center": (float(cx), float(cy)),
            "color_ring_axes": (float(axis_a), float(axis_b)),
            "color_ring_angle": float(angle),
            "color_ring_fit_error_p90": fit_error_p90,
        }
    )
    return ellipse, dbg


def _rectify_by_colored_target(
    bgr: np.ndarray,
    out_size: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, object]]:
    ellipse, dbg = _fit_colored_target_ellipse(bgr)
    if ellipse is None:
        return None, None, dbg

    (cx, cy), (axis_a, axis_b), angle = ellipse
    target_center = np.array([out_size / 2.0, out_size / 2.0], dtype=np.float64)
    target_outer = float(out_size) * 0.45
    target_red_radius = target_outer * 0.40

    theta = np.deg2rad(float(angle))
    axis_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    axis_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    linear = (
        (target_red_radius / max(float(axis_a) / 2.0, 1e-6)) * np.outer(axis_u, axis_u)
        + (target_red_radius / max(float(axis_b) / 2.0, 1e-6)) * np.outer(axis_v, axis_v)
    )
    source_center = np.array([float(cx), float(cy)], dtype=np.float64)
    translation = target_center - linear @ source_center
    source_to_rect = np.hstack([linear, translation.reshape(2, 1)]).astype(np.float32)

    rect = cv2.warpAffine(
        bgr,
        source_to_rect,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30),
    )
    dbg.update(
        {
            "rectify_mode": "colored_target",
            "source_to_rect": source_to_rect.tolist(),
            "ellipse_found": True,
            "ellipse_center": (float(cx), float(cy)),
            "ellipse_axes": (float(axis_a), float(axis_b)),
            "ellipse_angle": float(angle),
            "ellipse_axis_ratio": float(max(axis_a, axis_b) / max(min(axis_a, axis_b), 1e-6)),
            "ellipse_rotation_skipped": abs(float(axis_a) - float(axis_b)) < 0.03 * max(axis_a, axis_b),
        }
    )
    return rect, source_to_rect, dbg


def _affine_rectify_by_ellipse(bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Coarse rectify: fit outer ellipse boundary then affine-correct.
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
    (cx, cy), (a, b), angle = ellipse
    major = max(a, b)
    minor = min(a, b)
    if minor < 1e-6:
        return bgr.copy(), {"ellipse_found": False, "fallback": "minor_zero"}

    dbg.update(
        {
            "ellipse_found": True,
            "ellipse_center": (float(cx), float(cy)),
            "ellipse_axes": (float(a), float(b)),
            "ellipse_angle": float(angle),
        }
    )

    axis_ratio = float(major / minor)
    dbg["ellipse_axis_ratio"] = axis_ratio
    if axis_ratio < 1.03:
        # fitEllipse's angle is numerically unstable for an almost circular
        # target. Rotating by that arbitrary angle moves otherwise correct hit
        # points tangentially (several pixels near the outer rings).
        dbg["ellipse_rotation_skipped"] = True
        dbg["affine_scale_y"] = 1.0
        return bgr.copy(), dbg

    # Scale along the fitted ellipse axes, then rotate that scaling basis back
    # into image coordinates. The previous rotate-then-scale implementation
    # left the target rotated and often stretched the *major* axis when
    # fitEllipse reported its first axis vertically.
    theta = np.deg2rad(float(angle))
    axis_a = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    axis_b = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    scale_a = float(major / max(float(a), 1e-6))
    scale_b = float(major / max(float(b), 1e-6))
    linear = (
        scale_a * np.outer(axis_a, axis_a)
        + scale_b * np.outer(axis_b, axis_b)
    )
    center_vec = np.array([float(cx), float(cy)], dtype=np.float64)
    translation = center_vec - linear @ center_vec
    affine = np.hstack([linear, translation.reshape(2, 1)]).astype(np.float32)
    rect = cv2.warpAffine(
        bgr,
        affine,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    dbg["affine_scale_y"] = float(max(scale_a, scale_b))
    dbg["affine_axis_scales"] = (scale_a, scale_b)
    dbg["ellipse_rotation_skipped"] = False
    return rect, dbg


def _refine_circle(bgr: np.ndarray) -> Tuple[Tuple[float, float], float, Dict[str, object]]:
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

    detected_circles = normalize_hough_circles(circles)
    dbg: Dict[str, object] = {"circle_found": False}
    if len(detected_circles) == 0:
        return (w / 2.0, h / 2.0), min(w, h) * 0.45, {"circle_found": False, "fallback": "no_hough"}

    detected_circles = np.round(detected_circles).astype(int)
    detected_circles = sorted(detected_circles, key=lambda c: c[2], reverse=True)
    x, y, r = detected_circles[0]
    dbg.update({"circle_found": True, "circle_xy_r": (int(x), int(y), int(r))})
    return (float(x), float(y)), float(r), dbg


def _crop_square_around_circle(
    bgr: np.ndarray, center: Tuple[float, float], radius: float, out_size: int
) -> Tuple[np.ndarray, Tuple[float, float], float]:
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
    segments = normalize_hough_lines(lines)
    if len(segments) == 0:
        return False, 0

    cnt = 0
    for x1, y1, x2, y2 in segments:
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length >= min(h, w) * 0.22:
            cnt += 1
    return cnt >= 2, cnt


def _black_ink_mask(rect_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    thr = int(np.percentile(L, 15))
    mask = (L < thr).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return mask


def _detect_midline_from_right_digits(
    rect_bgr: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    h, w = rect_bgr.shape[:2]
    dbg: Dict[str, object] = {"midline_found": False}
    mask = _black_ink_mask(rect_bgr)

    x0 = int(w * 0.60)
    roi = mask[:, x0:w]

    num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(roi, connectivity=8)
    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20 or area > 800:
            continue
        cx, cy = cents[i]
        pts.append((float(cx + x0), float(cy)))

    if len(pts) < 6:
        dbg["fallback"] = "not_enough_text_blobs"
        dbg["blobs"] = len(pts)
        return None, None, dbg

    P = np.array(pts, dtype=np.float32)
    mean = P.mean(axis=0)
    X = P - mean
    cov = (X.T @ X) / max(1, len(P) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]

    angle = float(np.degrees(np.arctan2(v[1], v[0])))
    mid_y = float(mean[1])
    dbg.update(
        {
            "midline_found": True,
            "midline_y": mid_y,
            "midline_angle_deg": angle,
            "midline_pts": len(pts),
        }
    )
    return mid_y, angle, dbg


def _rotate_about(
    rect_bgr: np.ndarray, center: Tuple[float, float], angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = rect_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    out = cv2.warpAffine(
        rect_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return out, M.astype(np.float32)


def _detect_x_center(
    rect_bgr: np.ndarray, approx_center: Tuple[float, float], search_r: int = 120
) -> Tuple[Optional[Tuple[float, float]], Dict[str, object]]:
    h, w = rect_bgr.shape[:2]
    cx, cy = approx_center
    dbg: Dict[str, object] = {"x_found": False}

    x1 = max(0, int(cx - search_r))
    y1 = max(0, int(cy - search_r))
    x2 = min(w, int(cx + search_r))
    y2 = min(h, int(cy + search_r))
    roi = rect_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        dbg["fallback"] = "empty_roi"
        return None, dbg

    mask = _black_ink_mask(roi)
    edges = cv2.Canny(mask, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(12, int(min(roi.shape[:2]) * 0.15)),
        maxLineGap=6,
    )
    segments = normalize_hough_lines(lines)
    if len(segments) < 2:
        dbg["fallback"] = "no_lines"
        return None, dbg

    segs = []
    for xA, yA, xB, yB in segments:
        dx, dy = float(xB - xA), float(yB - yA)
        L = (dx * dx + dy * dy) ** 0.5
        if L < 10:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        segs.append((xA, yA, xB, yB, ang, L))

    if len(segs) < 2:
        dbg["fallback"] = "short_lines_only"
        return None, dbg

    def _line_params(xA, yA, xB, yB):
        a = float(yA - yB)
        b = float(xB - xA)
        c = float(xA * yB - xB * yA)
        return a, b, c

    best = None
    best_score = 1e18
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a1 = segs[i][4]
            a2 = segs[j][4]
            d = abs(((a1 - a2 + 90) % 180) - 90)
            perp_err = abs(d - 90.0)
            if perp_err > 25.0:
                continue

            xA1, yA1, xB1, yB1 = segs[i][0], segs[i][1], segs[i][2], segs[i][3]
            xA2, yA2, xB2, yB2 = segs[j][0], segs[j][1], segs[j][2], segs[j][3]
            A1, B1, C1 = _line_params(xA1, yA1, xB1, yB1)
            A2, B2, C2 = _line_params(xA2, yA2, xB2, yB2)
            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-6:
                continue

            ix = (B1 * C2 - B2 * C1) / det
            iy = (C1 * A2 - C2 * A1) / det

            roi_cx, roi_cy = (x2 - x1) / 2.0, (y2 - y1) / 2.0
            dist = (ix - roi_cx) ** 2 + (iy - roi_cy) ** 2
            score = dist + perp_err * 50.0

            if score < best_score:
                best_score = score
                best = (ix, iy, perp_err)

    if best is None:
        dbg["fallback"] = "no_perp_pair"
        return None, dbg

    ix, iy, perp_err = best
    fx, fy = float(ix + x1), float(iy + y1)
    dbg.update({"x_found": True, "x_center": (fx, fy), "x_perp_err": float(perp_err)})
    return (fx, fy), dbg


# -----------------------------
# NEW: outer radius by WHITE ring color
# -----------------------------
def _refine_outer_radius_by_white(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
) -> Tuple[Optional[float], Dict[str, object]]:
    """
    Use HSV threshold to find the white outer ring/paper region and fit a circle.
    This is very effective for correcting 'outer_radius too small' problems.
    """
    h, w = rect_bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])

    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)

    # "white": low saturation + high value
    # (tolerant thresholds to handle lighting)
    lower = np.array([0, 0, 170], dtype=np.uint8)
    upper = np.array([180, 70, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # remove small holes / noise
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dbg: Dict[str, object] = {"white_outer_found": False}

    if not contours:
        dbg["fallback"] = "no_white_contours"
        return None, dbg

    # Prefer contour that contains the center (or the largest one)
    chosen = None
    chosen_area = -1.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < (h * w * 0.01):
            continue
        # test if center point is inside contour
        inside = cv2.pointPolygonTest(cnt, (cx, cy), False) >= 0
        # score: containing center preferred; otherwise use area
        score = area + (1e9 if inside else 0.0)
        if score > chosen_area:
            chosen_area = score
            chosen = cnt

    if chosen is None:
        dbg["fallback"] = "no_large_white_contour"
        return None, dbg

    (x, y), r = cv2.minEnclosingCircle(chosen)
    r = float(r)

    # sanity: radius should be plausible
    # since rect_bgr is a crop around the target, outer radius should be sizeable
    if r < 0.20 * min(h, w) or r > 0.55 * min(h, w):
        dbg.update(
            {
                "white_outer_found": False,
                "fallback": "radius_out_of_range",
                "r": r,
            }
        )
        return None, dbg

    dbg.update(
        {
            "white_outer_found": True,
            "white_circle_xy_r": (float(x), float(y), float(r)),
            "white_mask_ratio": float(mask.mean() / 255.0),
        }
    )
    return r, dbg


def _build_similarity_M(
    src_center: Tuple[float, float], src_outer: float, angle_deg: float = 0.0
) -> np.ndarray:
    sx, sy = src_center
    s = 1.0 if src_outer <= 1e-6 else float(CANON_OUTER) / float(src_outer)

    theta = np.deg2rad(angle_deg)
    c, sn = float(np.cos(theta)), float(np.sin(theta))
    A = np.array([[s * c, -s * sn], [s * sn, s * c]], dtype=np.float32)

    dst = np.array([[CANON_CENTER[0]], [CANON_CENTER[1]]], dtype=np.float32)
    src = np.array([[sx], [sy]], dtype=np.float32)
    t = dst - A @ src

    M = np.hstack([A, t]).astype(np.float32)
    return M


def _quality_from_debug(debug: Dict[str, object]) -> Tuple[float, List[str]]:
    """
    Cheap, robust quality score based on which pose steps succeeded.
    """
    score = 1.0
    flags: List[str] = []

    if not bool(debug.get("ellipse_found", False)):
        score -= 0.20
        flags.append("ellipse_not_found")

    # use 'circle_refine_after_midline' if present
    circle2 = debug.get("circle_refine_after_midline", {}) or {}
    circle_ok = bool(circle2.get("circle_found", debug.get("circle_found", False)))
    if not circle_ok:
        score -= 0.25
        flags.append("circle_not_found")

    mid_dbg = debug.get("midline_debug", {}) or {}
    if not bool(mid_dbg.get("midline_found", False)):
        score -= 0.10
        flags.append("midline_not_found")

    x_dbg = debug.get("x_debug", {}) or {}
    if not bool(x_dbg.get("x_found", False)):
        score -= 0.10
        flags.append("x_not_found")

    # NEW: outer radius calibration
    white_dbg = debug.get("white_outer_debug", {}) or {}
    if not bool(white_dbg.get("white_outer_found", False)):
        score -= 0.10
        flags.append("white_outer_not_found")

    sharpness = float(debug.get("image_sharpness", 999.0) or 0.0)
    if sharpness < MIN_IMAGE_SHARPNESS:
        score -= 0.25
        flags.append("image_blur")

    contrast = float(debug.get("image_contrast", 255.0) or 0.0)
    if contrast < MIN_IMAGE_CONTRAST:
        score -= 0.15
        flags.append("image_low_contrast")

    dark_fraction = float(debug.get("image_dark_fraction", 0.0) or 0.0)
    if dark_fraction > MAX_DARK_CENTER_FRACTION:
        score -= 0.25
        flags.append("image_low_light")

    glare_fraction = float(debug.get("image_glare_fraction", 0.0) or 0.0)
    if glare_fraction > MAX_GLARE_CENTER_FRACTION:
        score -= 0.25
        flags.append("image_glare")

    score = float(max(0.0, min(1.0, score)))
    if score < 0.55:
        flags.append("low_confidence")
    return score, flags


def rectify_target(image_rgb: np.ndarray, out_size: int = CANON_SIZE) -> TargetRectifyResult:
    bgr = _rgb_to_bgr(image_rgb)

    # Prefer the known color geometry when the red scoring zone is visible.
    # This path remains stable even when the outer face is cropped and arrows
    # or backing-board edges are the largest contours in the photograph.
    color_rect, _, color_debug = _rectify_by_colored_target(bgr, out_size)
    if color_rect is not None:
        center = (out_size / 2.0, out_size / 2.0)
        outer_radius = float(out_size) * 0.45
        arrow_present, line_count = _detect_arrow_present(color_rect)
        color_gray = cv2.cvtColor(color_rect, cv2.COLOR_BGR2GRAY)
        sharpness = float(
            cv2.Laplacian(
                color_gray,
                cv2.CV_64F,
            ).var()
        )
        contrast = _gray_contrast_span(color_gray)
        exposure_debug = _exposure_diagnostics(color_rect, center, outer_radius)
        color_debug.update(
            {
                "arrow_present": arrow_present,
                "line_count": int(line_count),
                "image_sharpness": sharpness,
                "image_contrast": contrast,
                "center_final_source": "color_ring",
                "outer_radius_source": "red_zone_ratio",
                **exposure_debug,
            }
        )

        fit_error = float(color_debug.get("color_ring_fit_error_p90", 0.0) or 0.0)
        quality_score = float(max(0.0, min(1.0, 0.97 - fit_error * 0.15)))
        quality_flags: List[str] = []
        if fit_error > 0.32:
            quality_score -= 0.12
            quality_flags.append("color_ring_fit_uncertain")
        if sharpness < MIN_IMAGE_SHARPNESS:
            quality_score -= 0.25
            quality_flags.append("image_blur")
        if contrast < MIN_IMAGE_CONTRAST:
            quality_score -= 0.15
            quality_flags.append("image_low_contrast")
        if exposure_debug["image_dark_fraction"] > MAX_DARK_CENTER_FRACTION:
            quality_score -= 0.25
            quality_flags.append("image_low_light")
        if exposure_debug["image_glare_fraction"] > MAX_GLARE_CENTER_FRACTION:
            quality_score -= 0.25
            quality_flags.append("image_glare")
        quality_score = float(max(0.0, min(1.0, quality_score)))
        if quality_score < 0.55:
            quality_flags.append("low_confidence")

        return TargetRectifyResult(
            rect_bgr=color_rect,
            circle_center=center,
            outer_radius=outer_radius,
            midline_y=None,
            x_center=center,
            center_final=center,
            M_rect_to_canon=_build_similarity_M(center, outer_radius),
            arrow_present=arrow_present,
            debug=color_debug,
            quality_score=quality_score,
            quality_flags=quality_flags,
        )

    rect1, dbg1 = _affine_rectify_by_ellipse(bgr)
    (cx, cy), r, dbg2 = _refine_circle(rect1)

    rect2, (rcx, rcy), rr = _crop_square_around_circle(rect1, (cx, cy), r, out_size)

    arrow_present, line_count = _detect_arrow_present(rect2)

    debug: Dict[str, object] = {}
    debug.update(dbg1)
    debug.update(dbg2)
    debug.update({"arrow_present": arrow_present, "line_count": int(line_count)})

    # midline -> rotate
    mid_y, mid_angle, dbg_mid = _detect_midline_from_right_digits(rect2)
    debug["midline_debug"] = dbg_mid

    rect3 = rect2
    if mid_angle is not None and abs(mid_angle) > 1.5:
        rect3, _ = _rotate_about(rect2, (rcx, rcy), -mid_angle)
        debug["midline_rotation_applied_deg"] = float(-mid_angle)
    else:
        debug["midline_rotation_applied_deg"] = 0.0

    gray_rect3 = cv2.cvtColor(rect3, cv2.COLOR_BGR2GRAY)
    debug["image_sharpness"] = float(cv2.Laplacian(gray_rect3, cv2.CV_64F).var())
    debug["image_contrast"] = _gray_contrast_span(gray_rect3)

    # after rotation, re-refine circle
    (ccx2, ccy2), rr2, dbg_circle2 = _refine_circle(rect3)
    debug["circle_refine_after_midline"] = dbg_circle2

    circle_center = (ccx2, ccy2)
    outer_radius = rr2

    # midline y after rotation (debug only)
    mid_y2, _, dbg_mid2 = _detect_midline_from_right_digits(rect3)
    debug["midline_debug_after"] = dbg_mid2
    midline_y_final = mid_y2

    # detect X near circle center
    x_center, dbg_x = _detect_x_center(rect3, circle_center, search_r=140)
    debug["x_debug"] = dbg_x

    center_final = x_center if x_center is not None else circle_center
    debug["center_final_source"] = "x" if x_center is not None else "circle"

    # -----------------------------
    # NEW: outer radius calibration using WHITE ring color
    # -----------------------------
    white_r, dbg_white = _refine_outer_radius_by_white(rect3, center_final)
    debug["white_outer_debug"] = dbg_white

    if white_r is not None:
        # If hough circle radius is too small (common), prefer white-based radius.
        # Also avoid crazy jump: only accept if within +/- 25% of current radius.
        ratio = float(white_r / max(1e-6, outer_radius))
        debug["white_vs_hough_radius_ratio"] = ratio

        if 0.75 <= ratio <= 1.35:
            outer_radius = float(white_r)
            debug["outer_radius_source"] = "white_outer"
        else:
            # still keep ratio in debug; don't apply if too inconsistent
            debug["outer_radius_source"] = "hough_circle"
    else:
        debug["outer_radius_source"] = "hough_circle"

    debug["outer_radius_final"] = float(outer_radius)
    debug.update(_exposure_diagnostics(rect3, center_final, outer_radius))

    # build mapping
    M_rect_to_canon = _build_similarity_M(center_final, outer_radius, angle_deg=0.0)
    debug["M_rect_to_canon"] = M_rect_to_canon.tolist()

    # quality
    quality_score, quality_flags = _quality_from_debug(debug)
    debug["quality_score"] = quality_score
    debug["quality_flags"] = quality_flags

    return TargetRectifyResult(
        rect_bgr=rect3,
        circle_center=circle_center,
        outer_radius=float(outer_radius),
        midline_y=midline_y_final,
        x_center=x_center,
        center_final=center_final,
        M_rect_to_canon=M_rect_to_canon,
        arrow_present=arrow_present,
        debug=debug,
        quality_score=quality_score,
        quality_flags=quality_flags,
    )


def transform_points(points_xy: List[Tuple[float, float]], M_2x3: np.ndarray) -> List[Tuple[float, float]]:
    if not points_xy:
        return []
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, M_2x3).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def _purple_fletch_groups(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Group the distinctive purple/magenta vanes visible around arrow shafts."""
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
    fletch_mask = cv2.inRange(
        hsv,
        np.array([135, 70, 35], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    fletch_mask = cv2.morphologyEx(
        fletch_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    join_size = _odd_kernel(float(outer_radius) * 0.075, 15)
    joined = cv2.dilate(
        fletch_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (join_size, join_size)),
    )
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)

    cx, cy = float(center[0]), float(center[1])
    min_area = max(180, int(float(outer_radius) ** 2 * 0.0010))
    groups: List[Dict[str, object]] = []
    for label in range(1, label_count):
        if int(stats[label, cv2.CC_STAT_AREA]) < min_area:
            continue
        component = (labels == label) & (fletch_mask > 0)
        ys, xs = np.where(component)
        if len(xs) < min_area:
            continue
        group_center = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)
        if np.linalg.norm(group_center - np.array([cx, cy])) > float(outer_radius) * 1.55:
            continue
        component_mask = np.zeros_like(fletch_mask)
        component_mask[ys, xs] = 255
        groups.append(
            {
                "center": group_center,
                "area": int(len(xs)),
                "mask": component_mask,
            }
        )

    groups.sort(key=lambda item: int(item["area"]), reverse=True)
    return fletch_mask, groups[:12]


def _fletch_angle_candidates(component_mask: np.ndarray) -> List[Tuple[float, float]]:
    ys, xs = np.where(component_mask > 0)
    if len(xs) < 20:
        return []
    margin = 15
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(component_mask.shape[1], int(xs.max()) + margin + 1)
    y2 = min(component_mask.shape[0], int(ys.max()) + margin + 1)
    roi = component_mask[y1:y2, x1:x2]
    edges = cv2.Canny(roi, 30, 100)
    min_length = max(12, int(min(roi.shape[:2]) * 0.18))
    lines = normalize_hough_lines(
        cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,
            threshold=max(10, int(min(roi.shape[:2]) * 0.10)),
            minLineLength=min_length,
            maxLineGap=max(8, int(min(roi.shape[:2]) * 0.12)),
        )
    )

    histogram = np.zeros(90, dtype=np.float64)
    for x_start, y_start, x_end, y_end in lines:
        dx = float(x_end - x_start)
        dy = float(y_end - y_start)
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        histogram[int(round(angle / 2.0)) % 90] += length

    peaks: List[Tuple[float, float]] = []
    for index, weight in enumerate(histogram):
        if weight < 10.0:
            continue
        if weight >= histogram[(index - 1) % 90] and weight >= histogram[(index + 1) % 90]:
            peaks.append((float(weight), float((index * 2) % 180)))

    if not peaks:
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        covariance = np.cov(points.T)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
        return [(1.0, angle)]
    return sorted(peaks, reverse=True)[:3]


def _image_shaft_angle_candidates(
    lines: np.ndarray,
    group_center: np.ndarray,
    outer_radius: float,
) -> List[Tuple[float, float]]:
    """Find line directions that visibly pass through a fletching group."""
    histogram = np.zeros(90, dtype=np.float64)
    maximum_distance = max(14.0, float(outer_radius) * 0.045)
    minimum_length = max(32.0, float(outer_radius) * 0.08)

    for x_start, y_start, x_end, y_end in lines:
        start = np.array((x_start, y_start), dtype=np.float64)
        vector = np.array((x_end - x_start, y_end - y_start), dtype=np.float64)
        length = float(np.linalg.norm(vector))
        if length < minimum_length:
            continue
        distance = abs(
            vector[0] * (float(group_center[1]) - start[1])
            - vector[1] * (float(group_center[0]) - start[0])
        ) / max(length, 1e-6)
        projection = float((group_center - start) @ vector / max(length * length, 1e-6))
        if distance > maximum_distance or not (-0.8 < projection < 1.8):
            continue
        angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
        histogram[int(round(angle / 2.0)) % 90] += length / (distance + 8.0)

    smoothed = histogram + 0.5 * np.roll(histogram, 1) + 0.5 * np.roll(histogram, -1)
    candidates: List[Tuple[float, float]] = []
    for index in np.argsort(smoothed)[::-1]:
        weight = float(smoothed[index])
        if weight < 1.0:
            break
        angle = float((int(index) * 2) % 180)
        if any(
            min(abs(angle - existing), 180.0 - abs(angle - existing)) < 8.0
            for _, existing in candidates
        ):
            continue
        candidates.append((weight, angle))
        if len(candidates) >= 5:
            break
    return candidates


def _fused_shaft_angle_candidates(
    mask_candidates: List[Tuple[float, float]],
    image_candidates: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Prefer directions supported by both vane shape and a visible shaft."""
    consensus: List[Tuple[float, float]] = []
    for image_weight, image_angle in image_candidates:
        for mask_weight, mask_angle in mask_candidates:
            delta = abs(float(image_angle) - float(mask_angle))
            delta = min(delta, 180.0 - delta)
            if delta > 14.0:
                continue
            weight = (
                float(image_weight)
                * float(np.sqrt(max(float(mask_weight), 1.0)))
                / (1.0 + delta * 0.20)
            )
            # The image line validates the shaft family, while the vane edge
            # usually gives the better centre-line angle (the visible line is
            # often only one side of a thick or blurred shaft).
            consensus.append((weight, float(mask_angle)))

    if consensus:
        # The image line is the shaft itself; the coloured vane only validates
        # that line. A single consensus direction avoids following a stronger
        # perpendicular vane edge into an old target hole.
        return [max(consensus, key=lambda item: item[0])]
    return mask_candidates


def _image_shaft_direction_hint(
    lines: np.ndarray,
    group_center: np.ndarray,
    angle_deg: float,
    outer_radius: float,
) -> Optional[np.ndarray]:
    """Orient an undirected shaft angle toward the side containing its line."""
    axis = np.array(
        [np.cos(np.deg2rad(float(angle_deg))), np.sin(np.deg2rad(float(angle_deg)))],
        dtype=np.float64,
    )
    negative_extent = 0.0
    positive_extent = 0.0
    maximum_distance = max(14.0, float(outer_radius) * 0.045)
    minimum_length = max(32.0, float(outer_radius) * 0.08)

    for x_start, y_start, x_end, y_end in lines:
        start = np.array((x_start, y_start), dtype=np.float64)
        end = np.array((x_end, y_end), dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < minimum_length:
            continue
        line_angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
        angle_delta = abs(line_angle - float(angle_deg))
        angle_delta = min(angle_delta, 180.0 - angle_delta)
        if angle_delta > 5.0:
            continue
        distance = abs(
            vector[0] * (float(group_center[1]) - start[1])
            - vector[1] * (float(group_center[0]) - start[0])
        ) / max(length, 1e-6)
        projection = float((group_center - start) @ vector / max(length * length, 1e-6))
        if distance > maximum_distance or not (-0.8 < projection < 1.8):
            continue
        positions = [float((point - group_center) @ axis) for point in (start, end)]
        positive_extent += max(0.0, max(positions))
        negative_extent += max(0.0, -min(positions))

    if max(positive_extent, negative_extent) < minimum_length * 0.65:
        return None
    return axis if positive_extent >= negative_extent else -axis


def _dark_shaft_chain(
    lightness: np.ndarray,
    group_center: np.ndarray,
    angle_deg: float,
    target_center: Tuple[float, float],
    outer_radius: float,
    *,
    reverse: bool = False,
) -> Dict[str, object]:
    """Trace a dark shaft from its fletching toward its target endpoint."""
    h, w = lightness.shape[:2]
    target = np.array(target_center, dtype=np.float64)
    theta = np.deg2rad(float(angle_deg))
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    if float(direction @ (target - group_center)) < 0.0:
        direction *= -1.0
    if reverse:
        direction *= -1.0
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    max_steps = max(120, int(float(outer_radius) * 1.42))
    flank_outer = max(16, int(round(float(outer_radius) * 0.062)))
    flank_inner = max(10, int(round(float(outer_radius) * 0.037)))
    center_half = max(3, int(round(float(outer_radius) * 0.012)))
    offsets = np.arange(-flank_outer, flank_outer + 1, dtype=np.float32)
    steps = np.arange(max_steps, dtype=np.float32)
    map_x = (
        float(group_center[0])
        + steps[:, None] * float(direction[0])
        + offsets[None, :] * float(normal[0])
    ).astype(np.float32)
    map_y = (
        float(group_center[1])
        + steps[:, None] * float(direction[1])
        + offsets[None, :] * float(normal[1])
    ).astype(np.float32)
    strip = cv2.remap(
        lightness.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    center_selector = np.abs(offsets) <= center_half
    left_selector = (offsets >= -flank_outer) & (offsets <= -flank_inner)
    right_selector = (offsets >= flank_inner) & (offsets <= flank_outer)
    center_light = strip[:, center_selector].mean(axis=1)
    flank_light = (strip[:, left_selector].mean(axis=1) + strip[:, right_selector].mean(axis=1)) / 2.0
    darkness = np.maximum(0.0, flank_light - center_light)

    median_radius = max(3, int(round(float(outer_radius) * 0.02)))
    smoothed = np.array(
        [
            float(np.median(darkness[max(0, i - median_radius) : min(max_steps, i + median_radius + 1)]))
            for i in range(max_steps)
        ],
        dtype=np.float32,
    )
    evidence = smoothed > 8.0
    minimum_run = max(8, int(round(float(outer_radius) * 0.02)))
    runs: List[Tuple[int, int, int]] = []
    start: Optional[int] = None
    for index, present in enumerate(np.append(evidence, False)):
        if bool(present) and start is None:
            start = index
        elif not bool(present) and start is not None:
            if index - start >= minimum_run:
                runs.append((start, index - 1, index - start))
            start = None

    maximum_gap = max(24, int(round(float(outer_radius) * 0.09)))
    chains: List[List[Tuple[int, int, int]]] = []
    for run in runs:
        gap = run[0] - chains[-1][-1][1] - 1 if chains else maximum_gap + 1
        if not chains or gap > maximum_gap:
            chains.append([run])
        else:
            chains[-1].append(run)

    if not chains:
        return {
            "span": 0,
            "end": 0,
            "mean_darkness": 0.0,
            "direction": direction,
            "angle": float(angle_deg % 180.0),
        }

    best_chain = max(
        chains,
        key=lambda chain: (chain[-1][1] - chain[0][0] + 1, sum(run[2] for run in chain)),
    )
    chain_start = int(best_chain[0][0])
    chain_end = int(best_chain[-1][1])
    span = int(chain_end - chain_start + 1)
    mean_darkness = float(smoothed[chain_start : chain_end + 1].mean())
    endpoint = group_center + direction * float(chain_end)
    if not (0 <= endpoint[0] < w and 0 <= endpoint[1] < h):
        span = 0

    return {
        "span": span,
        "start": chain_start,
        "end": chain_end,
        "mean_darkness": mean_darkness,
        "direction": direction,
        "angle": float(angle_deg % 180.0),
    }


def _fletched_shaft_candidates(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
) -> Tuple[List[Tuple[float, float, float]], List[Dict[str, object]]]:
    _, groups = _purple_fletch_groups(rect_bgr, center, outer_radius)
    if not groups:
        return [], []

    raw_lightness = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab)[:, :, 0]
    lightness = _normalize_gray_contrast(raw_lightness).astype(np.float32)
    line_gray = _normalize_gray_contrast(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
    line_gray = cv2.GaussianBlur(line_gray, (5, 5), 1.0)
    shaft_lines = normalize_hough_lines(
        cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(line_gray)[0]
    )
    candidates: List[Tuple[float, float, float]] = []
    accepted_groups: List[Dict[str, object]] = []
    target = np.array(center, dtype=np.float64)

    def trace(
        group_center: np.ndarray,
        angle: float,
        direction_hint: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        directions = [
            _dark_shaft_chain(
                lightness,
                group_center,
                angle,
                center,
                outer_radius,
                reverse=reverse,
            )
            for reverse in (False, True)
        ]
        maximum_span = max(int(result["span"]) for result in directions)
        eligible = [
            result
            for result in directions
            if int(result["span"]) >= max(1, int(maximum_span * 0.85))
        ]

        if direction_hint is not None:
            aligned = max(
                directions,
                key=lambda result: float(np.asarray(result["direction"]) @ direction_hint),
            )
            if int(aligned["span"]) >= max(1, int(maximum_span * 0.55)):
                return aligned

        def endpoint_texture(result: Dict[str, object]) -> float:
            endpoint = group_center + result["direction"] * float(result["end"])
            x = int(round(float(endpoint[0])))
            y = int(round(float(endpoint[1])))
            radius = max(12, int(round(float(outer_radius) * 0.05)))
            roi = lightness[
                max(0, y - radius) : min(lightness.shape[0], y + radius + 1),
                max(0, x - radius) : min(lightness.shape[1], x + radius + 1),
            ]
            return float(roi.std()) if roi.size else 999.0

        # When the fletching lies over the target centre, both directions can
        # have similar trace lengths. The nock is a large blurred dark object;
        # the embedded point is the narrower, less textured endpoint.
        return min(
            eligible,
            key=lambda result: (endpoint_texture(result), -int(result["span"])),
        )

    def trace_score(result: Dict[str, object], hough_weight: float = 1.0) -> float:
        endpoint = result["group_center"] + result["direction"] * float(result["end"])
        endpoint_radius = float(np.linalg.norm(endpoint - target))
        radial_factor = max(0.15, 1.0 - endpoint_radius / max(float(outer_radius) * 1.05, 1e-6))
        return (
            float(result["span"]) ** 1.5
            * float(np.sqrt(float(result["mean_darkness"]) + 1.0))
            * radial_factor
            * float(np.sqrt(max(hough_weight, 1.0)))
        )

    for group in groups:
        mask_peaks = _fletch_angle_candidates(group["mask"])
        image_peaks = _image_shaft_angle_candidates(
            shaft_lines,
            group["center"],
            outer_radius,
        )
        angle_peaks = _fused_shaft_angle_candidates(mask_peaks, image_peaks)
        if not angle_peaks:
            continue
        base_results = []
        for hough_weight, angle in angle_peaks:
            direction_hint = _image_shaft_direction_hint(
                shaft_lines,
                group["center"],
                angle,
                outer_radius,
            )
            result = trace(group["center"], angle, direction_hint)
            result["group_center"] = group["center"]
            base_results.append((result, hough_weight, direction_hint))
        base, _, direction_hint = max(
            base_results,
            key=lambda item: trace_score(item[0], item[1]),
        )
        if int(base["span"]) < max(45, int(float(outer_radius) * 0.11)):
            continue

        matching_image_angle: Optional[float] = None
        if image_peaks:
            _, nearest_image_angle = min(
                image_peaks,
                key=lambda item: min(
                    abs(float(item[1]) - float(base["angle"])),
                    180.0 - abs(float(item[1]) - float(base["angle"])),
                ),
            )
            nearest_delta = (
                (float(nearest_image_angle) - float(base["angle"]) + 90.0) % 180.0
            ) - 90.0
            if abs(nearest_delta) <= 14.0:
                matching_image_angle = float(nearest_image_angle)

        if matching_image_angle is None:
            refinement_offsets = np.arange(-4.0, 4.01, 0.5)
        else:
            signed_delta = (
                (matching_image_angle - float(base["angle"]) + 90.0) % 180.0
            ) - 90.0
            if abs(signed_delta) < 0.75:
                refinement_offsets = np.array([0.0])
            else:
                lower = max(-4.0, min(0.0, signed_delta) - 0.5)
                upper = min(4.0, max(0.0, signed_delta) + 0.5)
                refinement_offsets = np.arange(lower, upper + 0.01, 0.5)

        refined_results = []
        for offset in refinement_offsets:
            result = trace(
                group["center"],
                (float(base["angle"]) + offset) % 180.0,
                direction_hint,
            )
            result["group_center"] = group["center"]
            result["angle_offset"] = float(offset)
            refined_results.append(result)
        selection_pool = refined_results
        if float(np.linalg.norm(group["center"] - target)) < float(outer_radius) * 0.10:
            # When the vane itself overlaps the bull, a long dark chain is
            # commonly a printed ring or an old-hole trail. Keep the endpoint
            # within a plausible visible shaft length from that vane.
            nearby = [
                result
                for result in refined_results
                if float(result["end"]) <= float(outer_radius) * 0.40
            ]
            if nearby:
                selection_pool = nearby
        refined = max(
            selection_pool,
            key=lambda result: trace_score(result)
            * float(np.exp(-0.5 * (float(result["angle_offset"]) / 4.0) ** 2)),
        )
        endpoint = group["center"] + refined["direction"] * float(refined["end"])
        if np.linalg.norm(endpoint - target) > float(outer_radius) * 1.05:
            continue
        confidence = float(refined["span"]) * float(np.sqrt(float(refined["mean_darkness"]) + 1.0))
        candidates.append((confidence, float(endpoint[0]), float(endpoint[1])))
        accepted = dict(group)
        accepted.update({"shaft_angle": float(refined["angle"]), "shaft_endpoint": endpoint})
        accepted_groups.append(accepted)

    return sorted(candidates, reverse=True), accepted_groups


def _unfletched_shaft_candidates(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
    fletch_groups: List[Dict[str, object]],
) -> List[Tuple[float, float, float]]:
    """Find long parallel shaft edges when the fletching is outside the frame."""
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.0), 50, 140)
    lines = normalize_hough_lines(
        cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360.0,
            threshold=max(40, int(float(outer_radius) * 0.12)),
            minLineLength=max(80, int(float(outer_radius) * 0.32)),
            maxLineGap=max(20, int(float(outer_radius) * 0.085)),
        )
    )
    target = np.array(center, dtype=np.float64)
    raw: List[Dict[str, object]] = []

    for segment in lines:
        start = segment[:2].astype(np.float64)
        end = segment[2:].astype(np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < float(outer_radius) * 0.32:
            continue
        start_radius = float(np.linalg.norm(start - target))
        end_radius = float(np.linalg.norm(end - target))
        radial_span = abs(start_radius - end_radius)
        if min(start_radius, end_radius) > float(outer_radius) * 1.02:
            continue
        if max(start_radius, end_radius) > float(outer_radius) * 1.07:
            continue
        if radial_span < float(outer_radius) * 0.20:
            continue

        overlaps_fletch = False
        for group in fletch_groups:
            point = group["center"]
            distance = abs(
                vector[0] * (point[1] - start[1])
                - vector[1] * (point[0] - start[0])
            ) / max(length, 1e-6)
            projection = float((point - start) @ vector / max(length * length, 1e-6))
            if distance < float(outer_radius) * 0.16 and -0.4 < projection < 1.4:
                overlaps_fletch = True
                break
        if overlaps_fletch:
            continue

        tip = start if start_radius <= end_radius else end
        angle = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
        raw.append(
            {
                "angle": angle,
                "tip": tip,
                "length": length,
                "radial_span": radial_span,
            }
        )

    clusters: List[Dict[str, object]] = []
    for item in sorted(raw, key=lambda value: float(value["length"]), reverse=True):
        matched = None
        for cluster in clusters:
            angle_delta = abs(float(item["angle"]) - float(cluster["angle"]))
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            tip_delta = float(np.linalg.norm(item["tip"] - cluster["tip"]))
            if angle_delta <= 3.0 and tip_delta <= float(outer_radius) * 0.055:
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    "angle": float(item["angle"]),
                    "tip": item["tip"].copy(),
                    "items": [item],
                }
            )
            continue
        matched["items"].append(item)
        total_length = sum(float(value["length"]) for value in matched["items"])
        matched["angle"] = sum(
            float(value["angle"]) * float(value["length"])
            for value in matched["items"]
        ) / max(total_length, 1e-6)
        matched["tip"] = sum(
            (value["tip"] * float(value["length"]) for value in matched["items"]),
            np.zeros(2, dtype=np.float64),
        ) / max(total_length, 1e-6)

    ranked: List[Tuple[float, float, float]] = []
    for cluster in clusters:
        items = cluster["items"]
        if len(items) < 2:
            continue
        score = (
            len(items) * float(np.median([value["radial_span"] for value in items]))
            + 0.15 * sum(float(value["length"]) for value in items)
        )
        tip = cluster["tip"]
        ranked.append((float(score), float(tip[0]), float(tip[1])))

    ranked.sort(reverse=True)
    if not ranked:
        return []
    best_score = ranked[0][0]
    threshold = max(float(outer_radius) * 1.10, best_score * 0.72)
    return [candidate for candidate in ranked if candidate[0] >= threshold]


def _paired_edge_shaft_candidates(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
    fletched: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Trace an unfletched shaft from its two parallel visible edges.

    ``HoughLinesP`` can return a different number of fragments across OpenCV
    builds.  A line-segment detector gives us a complementary, deterministic
    cue: a real shaft has two long, near-parallel sides a few pixels apart.
    Once a pair is found, follow both edges inward until they disappear; that
    disappearance is the contact point rather than merely the end of the first
    detected segment.
    """
    gray = _normalize_gray_contrast(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edge_map = cv2.Canny(blurred, 25, 80)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(blurred)[0]
    lines = normalize_hough_lines(detected)
    target = np.array(center, dtype=np.float64)
    known_contacts = [np.array((x, y), dtype=np.float64) for _, x, y in fletched]

    segments: List[Dict[str, object]] = []
    for line in lines:
        first = line[:2].astype(np.float64)
        second = line[2:].astype(np.float64)
        first_radius = float(np.linalg.norm(first - target))
        second_radius = float(np.linalg.norm(second - target))
        inward, outward = (
            (first, second) if first_radius <= second_radius else (second, first)
        )
        vector = outward - inward
        length = float(np.linalg.norm(vector))
        if length < max(70.0, float(outer_radius) * 0.23):
            continue
        inward_radius = float(np.linalg.norm(inward - target))
        outward_radius = float(np.linalg.norm(outward - target))
        radial_span = outward_radius - inward_radius
        if inward_radius > float(outer_radius) * 1.02:
            continue
        if outward_radius > float(outer_radius) * 1.10:
            continue
        if radial_span < float(outer_radius) * 0.16:
            continue

        direction = vector / max(length, 1e-6)
        radial_direction = inward - target
        radial_length = float(np.linalg.norm(radial_direction))
        if radial_length < 1.0:
            continue
        alignment = abs(float(direction @ (radial_direction / radial_length)))
        if alignment < float(np.cos(np.deg2rad(25.0))):
            continue

        # Lines belonging to a shaft that was already recovered through its
        # coloured fletching must not be counted again as an unfletched arrow.
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        if any(
            (
                abs(float((contact - inward) @ normal))
                < float(outer_radius) * 0.035
                and -3.2
                < float((contact - inward) @ direction / max(length, 1e-6))
                < 1.5
            )
            or (
                abs(float((contact - inward) @ normal))
                < float(outer_radius) * 0.055
                and -1.5
                < float((contact - inward) @ direction / max(length, 1e-6))
                < 1.5
            )
            for contact in known_contacts
        ):
            continue

        segments.append(
            {
                "inward": inward,
                "outward": outward,
                "direction": direction,
                "length": length,
                "angle": float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0),
            }
        )

    candidates: List[Tuple[float, float, float]] = []
    for index, first in enumerate(segments):
        for second in segments[index + 1 :]:
            angle_delta = abs(float(first["angle"]) - float(second["angle"]))
            angle_delta = min(angle_delta, 180.0 - angle_delta)
            if angle_delta > 2.5:
                continue

            direction_a = np.asarray(first["direction"], dtype=np.float64)
            direction_b = np.asarray(second["direction"], dtype=np.float64)
            if float(direction_a @ direction_b) < 0.0:
                direction_b *= -1.0
            direction = direction_a + direction_b
            direction /= max(float(np.linalg.norm(direction)), 1e-6)
            normal = np.array([-direction[1], direction[0]], dtype=np.float64)

            midpoint_a = (
                np.asarray(first["inward"], dtype=np.float64)
                + np.asarray(first["outward"], dtype=np.float64)
            ) / 2.0
            midpoint_b = (
                np.asarray(second["inward"], dtype=np.float64)
                + np.asarray(second["outward"], dtype=np.float64)
            ) / 2.0
            edge_separation = abs(float((midpoint_b - midpoint_a) @ normal))
            if not (2.0 <= edge_separation <= max(16.0, float(outer_radius) * 0.045)):
                continue

            axis_values_a = [
                float(np.asarray(first[key], dtype=np.float64) @ direction)
                for key in ("inward", "outward")
            ]
            axis_values_b = [
                float(np.asarray(second[key], dtype=np.float64) @ direction)
                for key in ("inward", "outward")
            ]
            overlap = min(max(axis_values_a), max(axis_values_b)) - max(
                min(axis_values_a), min(axis_values_b)
            )
            if overlap < max(45.0, float(outer_radius) * 0.12):
                continue

            seed = (
                np.asarray(first["inward"], dtype=np.float64)
                + np.asarray(second["inward"], dtype=np.float64)
            ) / 2.0
            if float(direction @ (seed - target)) < 0.0:
                direction *= -1.0
                normal *= -1.0

            half_width = edge_separation / 2.0
            band = max(2, min(4, int(round(half_width * 0.75))))
            maximum_extension = max(35, int(round(float(outer_radius) * 0.30)))
            maximum_gap = max(8, int(round(float(outer_radius) * 0.025)))
            last_supported = -1
            gap = 0
            supported_steps = 0

            for step in range(maximum_extension + 1):
                point = seed - direction * float(step)
                side_support: List[bool] = []
                for side in (-1.0, 1.0):
                    found = False
                    edge_offset = side * half_width
                    for delta in range(-band, band + 1):
                        sample = point + normal * (edge_offset + float(delta))
                        x = int(round(float(sample[0])))
                        y = int(round(float(sample[1])))
                        if 0 <= x < edge_map.shape[1] and 0 <= y < edge_map.shape[0]:
                            if edge_map[y, x] > 0:
                                found = True
                                break
                    side_support.append(found)

                if all(side_support):
                    last_supported = step
                    supported_steps += 1
                    gap = 0
                else:
                    gap += 1
                    if gap > maximum_gap:
                        break

            extension = max(0, last_supported)
            if extension < max(24, int(round(float(outer_radius) * 0.06))):
                continue
            support_ratio = supported_steps / max(extension + 1, 1)
            if support_ratio < 0.55:
                continue

            contact = seed - direction * float(extension)
            contact_radius = float(np.linalg.norm(contact - target))
            if contact_radius > float(outer_radius) * 1.03:
                continue
            if any(
                float(np.linalg.norm(contact - known)) < float(outer_radius) * 0.07
                for known in known_contacts
            ):
                continue

            confidence = float(overlap + extension * 4.0 + support_ratio * 120.0)
            candidates.append((confidence, float(contact[0]), float(contact[1])))

    # Compression and dim lighting can erase one side of a thin shaft. A
    # single long edge is still useful when it is radial, does not coincide
    # with a fletched shaft, and can be followed continuously inward to a
    # clear disappearance point.
    for segment in segments:
        seed = np.asarray(segment["inward"], dtype=np.float64)
        direction = np.asarray(segment["direction"], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        maximum_extension = max(35, int(round(float(outer_radius) * 0.30)))
        maximum_gap = max(8, int(round(float(outer_radius) * 0.025)))
        last_supported = -1
        supported_steps = 0
        gap = 0

        for step in range(maximum_extension + 1):
            point = seed - direction * float(step)
            found = False
            for offset in range(-3, 4):
                sample = point + normal * float(offset)
                x = int(round(float(sample[0])))
                y = int(round(float(sample[1])))
                if 0 <= x < edge_map.shape[1] and 0 <= y < edge_map.shape[0]:
                    if edge_map[y, x] > 0:
                        found = True
                        break
            if found:
                last_supported = step
                supported_steps += 1
                gap = 0
            else:
                gap += 1
                if gap > maximum_gap:
                    break

        extension = max(0, last_supported)
        if extension < max(24, int(round(float(outer_radius) * 0.06))):
            continue
        support_ratio = supported_steps / max(extension + 1, 1)
        if support_ratio < 0.62:
            continue
        contact = seed - direction * float(extension)
        if float(np.linalg.norm(contact - target)) > float(outer_radius) * 1.03:
            continue
        if any(
            float(np.linalg.norm(contact - known)) < float(outer_radius) * 0.07
            for known in known_contacts
        ):
            continue
        confidence = float(
            float(segment["length"]) + extension * 3.0 + support_ratio * 80.0
        )
        candidates.append((confidence, float(contact[0]), float(contact[1])))

    return sorted(candidates, reverse=True)


def _shaft_hit_candidates(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
) -> Tuple[List[Tuple[float, float]], bool]:
    fletched, groups = _fletched_shaft_candidates(rect_bgr, center, outer_radius)
    hough_extra = _unfletched_shaft_candidates(rect_bgr, center, outer_radius, groups)
    paired_extra = _paired_edge_shaft_candidates(
        rect_bgr,
        center,
        outer_radius,
        fletched,
    )
    # A pair of visible shaft edges is stronger evidence than loose Hough
    # fragments. Combining both lists counted the same real shaft plus one
    # unrelated target line as two arrows after mild exposure changes. Keep
    # Hough as the fallback only when no paired-edge shaft survives.
    extra = paired_extra if paired_extra else hough_extra
    # A saturated colour blob alone is not proof of a current arrow. Only an
    # accepted traced shaft may suppress the historical-hole fallback.
    shaft_mode = bool(fletched or extra)
    ranked = list(fletched) + list(extra)
    ranked.sort(reverse=True)

    points: List[Tuple[float, float]] = []
    minimum_distance_sq = max(14.0, float(outer_radius) * 0.04) ** 2
    for _, x, y in ranked:
        if any((x - px) ** 2 + (y - py) ** 2 < minimum_distance_sq for px, py in points):
            continue
        points.append((float(x), float(y)))
    return points, shaft_mode


def _radial_anomaly_candidates(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    outer_radius: float,
    *,
    allow_elongated_on_black: bool,
) -> List[Tuple[float, float, float]]:
    """Return ``(confidence, x, y)`` candidates after removing target structure.

    A target face is radially repetitive: pixels at the same radius should have
    similar color and brightness. Building that expected radial profile lets us
    suppress rings and colored bands before looking for local arrow/hole
    anomalies. This is intentionally conservative; missing a point is safer
    than placing a confident-looking point on a printed ring.
    """
    h, w = rect_bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    radius = max(float(outer_radius), min(h, w) * 0.20)

    yy, xx = np.indices((h, w), dtype=np.float32)
    radial = np.hypot(xx - cx, yy - cy)
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)

    bin_width = max(3.0, radius / 150.0)
    radial_bin = np.floor(radial / bin_width).astype(np.int32)
    bin_count = int(radial_bin.max()) + 1
    flat_bins = radial_bin.ravel()
    counts = np.bincount(flat_bins, minlength=bin_count).astype(np.float32)

    light = lab[:, :, 0]
    light_sums = np.bincount(
        flat_bins,
        weights=light.ravel(),
        minlength=bin_count,
    )
    expected_light = (light_sums / np.maximum(counts, 1.0))[radial_bin]

    light_delta = expected_light - light
    absolute_light_delta = np.abs(light_delta)

    ring_unit = radius / 10.0
    ring_distance = np.abs((radial / ring_unit) - np.round(radial / ring_unit)) * ring_unit
    away_from_printed_rings = ring_distance > max(5.0, radius * 0.012)
    within_face = (radial < radius * 1.015) & (radial > max(9.0, radius * 0.02))

    # Dark holes and carbon shafts on colored/white rings.
    dark_anomaly = (expected_light > 58.0) & (light_delta > 27.0)
    # On black rings only a bright shaft edge can be distinguished. Accepting
    # compact bright marks there would mostly select printed digits.
    black_ring_contrast = (expected_light <= 58.0) & (absolute_light_delta > 48.0)
    anomaly_mask = (dark_anomaly | black_ring_contrast) & away_from_printed_rings & within_face
    mask = anomaly_mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Tuple[float, float, float]] = []
    min_area = max(14, int(radius * radius * 0.00006))
    max_area = max(800, int(radius * radius * 0.012))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        box_w = int(stats[label, cv2.CC_STAT_WIDTH])
        box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue

        ys, xs = np.where(labels == label)
        if len(xs) < min_area:
            continue
        points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
        centered = points - points.mean(axis=0)
        covariance = (centered.T @ centered) / max(len(points) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major_index = int(np.argmax(eigenvalues))
        major_value = float(max(eigenvalues[major_index], 1e-6))
        minor_value = float(max(eigenvalues[1 - major_index], 1e-6))
        elongation = float((major_value / minor_value) ** 0.5)

        component_expected_light = float(expected_light[ys, xs].mean())
        is_black_ring_component = component_expected_light <= 58.0
        if elongation >= 2.5 and not allow_elongated_on_black:
            continue
        if is_black_ring_component and (not allow_elongated_on_black or elongation < 3.0):
            continue

        component_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0.0
        if contours:
            perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True))
        circularity = float(4.0 * np.pi * area / max(perimeter * perimeter, 1.0))
        aspect = float(max(box_w, box_h) / max(min(box_w, box_h), 1))

        if elongation < 2.5 and (aspect > 2.8 or circularity < 0.22):
            continue

        if elongation >= 2.5 and max(box_w, box_h) >= 24:
            direction = eigenvectors[:, major_index]
            projection = points @ direction
            low_point = points[int(np.argmin(projection))]
            high_point = points[int(np.argmax(projection))]
            low_distance = (low_point[0] - cx) ** 2 + (low_point[1] - cy) ** 2
            high_distance = (high_point[0] - cx) ** 2 + (high_point[1] - cy) ** 2
            candidate_xy = low_point if low_distance <= high_distance else high_point
        else:
            weights = np.maximum(light_delta[ys, xs], absolute_light_delta[ys, xs])
            weights = np.maximum(weights, 1.0)
            candidate_xy = np.array(
                [np.average(xs, weights=weights), np.average(ys, weights=weights)],
                dtype=np.float32,
            )

        anomaly_strength = float(absolute_light_delta[ys, xs].mean())
        confidence = (
            anomaly_strength
            + min(area, 180) * 0.16
            + min(circularity, 1.0) * 22.0
            + min(elongation, 5.0) * (5.0 if elongation >= 2.5 else 0.0)
        )
        candidates.append((confidence, float(candidate_xy[0]), float(candidate_xy[1])))

    return sorted(candidates, key=lambda item: item[0], reverse=True)


def propose_hit_points(
    rect_bgr: np.ndarray,
    center_final: Tuple[float, float],
    arrow_present: bool,
    max_points: int = 12,
    outer_radius: Optional[float] = None,
    diagnostics: Optional[Dict[str, object]] = None,
    quality_flags: Optional[Sequence[str]] = None,
) -> List[Tuple[float, float]]:
    """
    Generate conservative hit candidates after subtracting the target's radial
    ring/color structure. ``arrow_present`` only permits elongated contrast on
    black rings; it no longer switches to an unvalidated global Hough search.
    """
    h, w = rect_bgr.shape[:2]
    cx, cy = center_final
    radius = float(outer_radius) if outer_radius is not None else min(h, w) * 0.45
    unstable_exposure = {"image_low_light", "image_glare"}.intersection(
        quality_flags or []
    )
    if unstable_exposure:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "mode": "exposure_rejected",
                    "count": 0,
                    "quality_flags": sorted(unstable_exposure),
                }
            )
        return []
    sharpness = float(cv2.Laplacian(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharpness < MIN_IMAGE_SHARPNESS:
        if diagnostics is not None:
            diagnostics.update({"mode": "blur_rejected", "count": 0})
        return []

    shaft_points, shaft_mode = _shaft_hit_candidates(rect_bgr, (cx, cy), radius)
    if shaft_mode:
        # A visible shaft is direct evidence of a current arrow. Do not mix it
        # with compact radial anomalies: on a well-used target those are mostly
        # historical holes and would silently fill the requested arrow count.
        selected = shaft_points[:max_points]
        if diagnostics is not None:
            diagnostics.update({"mode": "visible_shafts", "count": len(selected)})
        return selected

    ranked = _radial_anomaly_candidates(
        rect_bgr,
        (cx, cy),
        radius,
        allow_elongated_on_black=bool(arrow_present),
    )
    pts = [(x, y) for _, x, y in ranked]

    # de-dup quickly
    dedup: List[Tuple[float, float]] = []
    # Close arrows are valid; the previous 28 px threshold merged separate
    # holes in a tight gold group. Components have already removed duplicate
    # edge pixels, so only collapse almost-identical centres here.
    min_d2 = max(10.0, radius * 0.025) ** 2
    for p in pts:
        ok = True
        for q in dedup:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < min_d2:
                ok = False
                break
        if ok:
            dedup.append(p)
        if len(dedup) >= max_points:
            break

    if diagnostics is not None:
        diagnostics.update({"mode": "radial_anomalies", "count": len(dedup)})
    return dedup
