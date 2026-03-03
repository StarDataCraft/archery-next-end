from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2


@dataclass
class TargetRectifyResult:
    rect_bgr: np.ndarray                 # rectified photo (square)
    circle_center: Tuple[float, float]   # center from circle (rect coords)
    outer_radius: float                  # outer radius (rect coords)

    # refined pose
    midline_y: Optional[float]           # detected horizontal midline y in rect coords (after midline rotation)
    x_center: Optional[Tuple[float, float]]  # detected X center if found
    center_final: Tuple[float, float]    # final center (X if found else circle center)

    # mapping from rect coords -> canonical coords (900x900)
    M_rect_to_canon: np.ndarray          # 2x3 float32 similarity transform

    arrow_present: bool
    debug: Dict[str, object]


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
    This is just a first step; final pose refinement will be done by midline/X.
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

    dbg.update({
        "ellipse_found": True,
        "ellipse_center": (float(cx), float(cy)),
        "ellipse_axes": (float(a), float(b)),
        "ellipse_angle": float(angle),
    })

    rot = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    rotated = cv2.warpAffine(bgr, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    scale = major / minor
    S = np.array([[1.0, 0.0, 0.0],
                  [0.0, scale, cy * (1.0 - scale)]], dtype=np.float32)
    rect = cv2.warpAffine(rotated, S, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    dbg["affine_scale_y"] = float(scale)
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

    dbg: Dict[str, object] = {"circle_found": False}
    if circles is None:
        return (w / 2.0, h / 2.0), min(w, h) * 0.45, {"circle_found": False, "fallback": "no_hough"}

    circles = np.round(circles[0, :]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]
    dbg.update({"circle_found": True, "circle_xy_r": (int(x), int(y), int(r))})
    return (float(x), float(y)), float(r), dbg


def _crop_square_around_circle(bgr: np.ndarray, center: Tuple[float, float], radius: float, out_size: int) -> Tuple[np.ndarray, Tuple[float, float], float]:
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
        edges, rho=1, theta=np.pi / 180,
        threshold=90,
        minLineLength=int(min(h, w) * 0.18),
        maxLineGap=12,
    )
    if lines is None:
        return False, 0

    cnt = 0
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length >= min(h, w) * 0.22:
            cnt += 1
    return cnt >= 2, cnt


# ----------------------------
#  Pose refinement: midline + X
# ----------------------------

def _black_ink_mask(rect_bgr: np.ndarray) -> np.ndarray:
    """
    Find likely black ink (digits / ring lines / X) region.
    Avoid OCR: just isolate dark pixels in Lab-L.
    """
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0]
    # dark threshold (adaptive-ish)
    thr = int(np.percentile(L, 15))
    mask = (L < thr).astype(np.uint8) * 255
    # clean small noise
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask


def _detect_midline_from_right_digits(rect_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float], Dict[str, object]]:
    """
    Heuristic:
    - numbers are black ink clusters on the right side
    - their centroids align roughly horizontally -> midline y
    - PCA angle of those centroids -> rotation angle to make line horizontal
    """
    h, w = rect_bgr.shape[:2]
    dbg: Dict[str, object] = {"midline_found": False}

    mask = _black_ink_mask(rect_bgr)

    # focus on right third where digits are likely present
    x0 = int(w * 0.60)
    roi = mask[:, x0:w]
    num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(roi, connectivity=8)

    pts = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20 or area > 800:
            continue
        cx, cy = cents[i]
        # convert to full coords
        pts.append((float(cx + x0), float(cy)))

    if len(pts) < 6:
        dbg["fallback"] = "not_enough_text_blobs"
        dbg["blobs"] = len(pts)
        return None, None, dbg

    P = np.array(pts, dtype=np.float32)
    mean = P.mean(axis=0)
    X = P - mean
    # PCA
    cov = (X.T @ X) / max(1, len(P) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, np.argmax(eigvals)]  # dominant direction
    angle = float(np.degrees(np.arctan2(v[1], v[0])))  # degrees

    # We want this line to be horizontal => rotate by -angle
    # Midline y in current coords can be mean y
    mid_y = float(mean[1])

    dbg.update({
        "midline_found": True,
        "midline_y": mid_y,
        "midline_angle_deg": angle,
        "midline_pts": len(pts),
    })
    return mid_y, angle, dbg


def _rotate_about(rect_bgr: np.ndarray, center: Tuple[float, float], angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image about center by angle_deg (positive = CCW).
    Returns rotated image and the 2x3 affine matrix.
    """
    h, w = rect_bgr.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    out = cv2.warpAffine(rect_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, M.astype(np.float32)


def _detect_x_center(rect_bgr: np.ndarray, approx_center: Tuple[float, float], search_r: int = 120) -> Tuple[Optional[Tuple[float, float]], Dict[str, object]]:
    """
    Detect X mark near center using short-line intersection:
    - detect many short line segments in ROI
    - pick two near-perpendicular lines
    - compute intersection
    """
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
    if lines is None or len(lines) < 2:
        dbg["fallback"] = "no_lines"
        return None, dbg

    # Collect line segments and angles
    segs = []
    for (xA, yA, xB, yB) in lines[:, 0, :]:
        dx, dy = float(xB - xA), float(yB - yA)
        L = (dx * dx + dy * dy) ** 0.5
        if L < 10:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        segs.append((xA, yA, xB, yB, ang, L))

    if len(segs) < 2:
        dbg["fallback"] = "short_lines_only"
        return None, dbg

    # Find best perpendicular-ish pair
    best = None
    best_score = 1e18

    def _line_params(xA, yA, xB, yB):
        # ax + by + c = 0
        a = float(yA - yB)
        b = float(xB - xA)
        c = float(xA * yB - xB * yA)
        return a, b, c

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            a1 = segs[i][4]
            a2 = segs[j][4]
            # angle diff mod 180
            d = abs(((a1 - a2 + 90) % 180) - 90)
            # want near 90 => d near 0 (use |d-90| style)
            perp_err = abs(d - 90.0)
            if perp_err > 25.0:
                continue

            # intersection
            xA1, yA1, xB1, yB1 = segs[i][0], segs[i][1], segs[i][2], segs[i][3]
            xA2, yA2, xB2, yB2 = segs[j][0], segs[j][1], segs[j][2], segs[j][3]
            A1, B1, C1 = _line_params(xA1, yA1, xB1, yB1)
            A2, B2, C2 = _line_params(xA2, yA2, xB2, yB2)

            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-6:
                continue
            ix = (B1 * C2 - B2 * C1) / det
            iy = (C1 * A2 - C2 * A1) / det

            # score: close to ROI center and low perp_err
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
    # Convert to full coords
    fx, fy = float(ix + x1), float(iy + y1)
    dbg.update({"x_found": True, "x_center": (fx, fy), "x_perp_err": float(perp_err)})
    return (fx, fy), dbg


def _build_similarity_M(src_center: Tuple[float, float], src_outer: float, angle_deg: float = 0.0) -> np.ndarray:
    """
    Similarity transform rect -> canon:
      - rotate by angle_deg about src_center (typically 0 after we already rotated image)
      - scale so outer radius matches CANON_OUTER
      - translate so center maps to CANON_CENTER
    """
    sx, sy = src_center
    if src_outer <= 1e-6:
        s = 1.0
    else:
        s = float(CANON_OUTER) / float(src_outer)

    theta = np.deg2rad(angle_deg)
    c, sn = float(np.cos(theta)), float(np.sin(theta))

    # rotation+scale about origin
    A = np.array([[s * c, -s * sn],
                  [s * sn,  s * c]], dtype=np.float32)

    # want: A * [sx,sy] + t = CANON_CENTER
    dst = np.array([[CANON_CENTER[0]], [CANON_CENTER[1]]], dtype=np.float32)
    src = np.array([[sx], [sy]], dtype=np.float32)
    t = dst - A @ src

    M = np.hstack([A, t]).astype(np.float32)  # 2x3
    return M


def rectify_target(image_rgb: np.ndarray, out_size: int = CANON_SIZE) -> TargetRectifyResult:
    bgr = _rgb_to_bgr(image_rgb)

    # 1) coarse rectify by ellipse
    rect1, dbg1 = _affine_rectify_by_ellipse(bgr)

    # 2) circle refine in rect1
    (cx, cy), r, dbg2 = _refine_circle(rect1)

    # 3) crop square around circle
    rect2, (rcx, rcy), rr = _crop_square_around_circle(rect1, (cx, cy), r, out_size)

    # 4) detect arrow presence
    arrow_present, line_count = _detect_arrow_present(rect2)

    debug: Dict[str, object] = {}
    debug.update(dbg1)
    debug.update(dbg2)
    debug.update({"arrow_present": arrow_present, "line_count": int(line_count)})

    # 5) refine pose: detect midline from right-side digits -> rotate to horizontal
    mid_y, mid_angle, dbg_mid = _detect_midline_from_right_digits(rect2)
    debug["midline_debug"] = dbg_mid

    rect3 = rect2
    M_mid = None
    if mid_angle is not None and abs(mid_angle) > 1.5:
        rect3, M_mid = _rotate_about(rect2, (rcx, rcy), -mid_angle)
        debug["midline_rotation_applied_deg"] = float(-mid_angle)
    else:
        debug["midline_rotation_applied_deg"] = 0.0

    # If we rotated, circle center also transforms
    circle_center = (rcx, rcy)
    if M_mid is not None:
        p = np.array([[[rcx, rcy]]], dtype=np.float32)
        p2 = cv2.transform(p, M_mid)[0, 0]
        circle_center = (float(p2[0]), float(p2[1]))

    # re-detect midline y after rotation (for debug / UI)
    mid_y2, _, dbg_mid2 = _detect_midline_from_right_digits(rect3)
    debug["midline_debug_after"] = dbg_mid2
    midline_y_final = mid_y2

    # 6) detect X center near circle center
    x_center, dbg_x = _detect_x_center(rect3, circle_center, search_r=140)
    debug["x_debug"] = dbg_x

    center_final = x_center if x_center is not None else circle_center
    debug["center_final_source"] = "x" if x_center is not None else "circle"

    # 7) build rect->canon similarity transform
    M_rect_to_canon = _build_similarity_M(center_final, rr, angle_deg=0.0)  # rect3 already rotated
    debug["M_rect_to_canon"] = M_rect_to_canon.tolist()

    return TargetRectifyResult(
        rect_bgr=rect3,
        circle_center=circle_center,
        outer_radius=rr,
        midline_y=midline_y_final,
        x_center=x_center,
        center_final=center_final,
        M_rect_to_canon=M_rect_to_canon,
        arrow_present=arrow_present,
        debug=debug,
    )


def transform_points(points_xy: List[Tuple[float, float]], M_2x3: np.ndarray) -> List[Tuple[float, float]]:
    if not points_xy:
        return []
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.transform(pts, M_2x3).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]
