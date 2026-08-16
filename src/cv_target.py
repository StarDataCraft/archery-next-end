# src/cv_target.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
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


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _largest_contour(edge: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


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
    if sharpness < 85.0:
        score -= 0.25
        flags.append("image_blur")

    score = float(max(0.0, min(1.0, score)))
    if score < 0.55:
        flags.append("low_confidence")
    return score, flags


def rectify_target(image_rgb: np.ndarray, out_size: int = CANON_SIZE) -> TargetRectifyResult:
    bgr = _rgb_to_bgr(image_rgb)

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
) -> List[Tuple[float, float]]:
    """
    Generate conservative hit candidates after subtracting the target's radial
    ring/color structure. ``arrow_present`` only permits elongated contrast on
    black rings; it no longer switches to an unvalidated global Hough search.
    """
    h, w = rect_bgr.shape[:2]
    cx, cy = center_final
    radius = float(outer_radius) if outer_radius is not None else min(h, w) * 0.45
    sharpness = float(cv2.Laplacian(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if sharpness < 85.0:
        return []
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

    return dedup
