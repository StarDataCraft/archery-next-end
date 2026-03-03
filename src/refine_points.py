# src/refine_points.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import cv2


def _clip_roi(x, y, r, w, h):
    x1 = max(0, int(round(x - r)))
    y1 = max(0, int(round(y - r)))
    x2 = min(w, int(round(x + r)))
    y2 = min(h, int(round(y + r)))
    return x1, y1, x2, y2


def _subpix_refine(gray_roi: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    img = gray_roi.astype(np.float32)
    p = np.array([[pt]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(img, p, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria)
        return float(p[0, 0, 0]), float(p[0, 0, 1])
    except Exception:
        return float(pt[0]), float(pt[1])


# ----------------------------------------------------------------------
# Robust + backward-compatible color sampling (ignore dark arrow pixels)
# ----------------------------------------------------------------------
def _mask_dark_arrow_pixels(hsv_roi: np.ndarray) -> np.ndarray:
    """
    hsv_roi: OpenCV HSV (H:0-180, S:0-255, V:0-255)
    Returns boolean mask where True means "likely target face (not arrow)".
    """
    v = hsv_roi[:, :, 2]
    # carbon/arrow/holes tend to be low V
    return v > 55


def sample_contact_color_hsv(
    rect_bgr: np.ndarray,
    x_or_xy: Union[float, Tuple[float, float]],
    y: Optional[float] = None,
    *,
    roi_radius: Optional[int] = None,
    r: Optional[int] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Backward compatible sampler.

    Supported calls:
      1) sample_contact_color_hsv(rect_bgr, (x,y), roi_radius=18)
      2) sample_contact_color_hsv(rect_bgr, x, y, r=10)
      3) sample_contact_color_hsv(rect_bgr, (x,y), r=10)
      4) sample_contact_color_hsv(rect_bgr, x, y, roi_radius=18)

    Returns OpenCV HSV median as (H,S,V) in scales H:[0,180], S/V:[0,255]
    """
    # Resolve coordinates
    if y is None:
        if not (isinstance(x_or_xy, (tuple, list)) and len(x_or_xy) == 2):
            raise TypeError("sample_contact_color_hsv: expected (x,y) tuple/list when y is None")
        x = float(x_or_xy[0])
        yv = float(x_or_xy[1])
    else:
        x = float(x_or_xy)
        yv = float(y)

    # Resolve radius: prefer roi_radius; fallback to r; default 10
    rad = roi_radius if roi_radius is not None else r
    if rad is None:
        rad = 10
    rad = int(rad)

    h, w = rect_bgr.shape[:2]
    cx, cy = int(round(x)), int(round(yv))

    x1 = max(0, cx - rad)
    y1 = max(0, cy - rad)
    x2 = min(w, cx + rad + 1)
    y2 = min(h, cy + rad + 1)

    roi = rect_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    keep = _mask_dark_arrow_pixels(hsv)

    # If too many pixels are masked (arrow covers most), fall back to unmasked median
    if int(np.sum(keep)) < 10:
        pixels = hsv.reshape(-1, 3).astype(np.float32)
    else:
        pixels = hsv[keep].reshape(-1, 3).astype(np.float32)

    if pixels.size == 0:
        return None

    med = np.median(pixels, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))


def _sample_face_color_under_arrow(bgr: np.ndarray, x: float, y: float, r: int = 10) -> Dict[str, Any]:
    """
    (Backward-compat with pipeline)
    Implemented via sample_contact_color_hsv() for robustness.
    Returns:
      {"ok": bool, "hsv_median": [H,S,V], "note": "..."}
    """
    hsv = sample_contact_color_hsv(bgr, x, y, r=r)
    if hsv is None:
        return {"ok": False}

    return {
        "ok": True,
        "hsv_median": [int(round(hsv[0])), int(round(hsv[1])), int(round(hsv[2]))],
        "note": "masked_dark_or_fallback",
    }


# ----------------------------------------------------------------------
# Arrow segment + contact point refinement (kept)
# ----------------------------------------------------------------------
def _best_arrow_segment_in_roi(roi_g: np.ndarray, min_len: int) -> Optional[Tuple[int, int, int, int]]:
    """Prefer LSD (more stable than Hough) if available."""
    try:
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(roi_g)[0]
        if lines is not None and len(lines) > 0:
            best = None
            best_len = 0.0
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                L = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
                if L >= min_len and L > best_len:
                    best_len = L
                    best = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
            if best is not None:
                return best
    except Exception:
        pass

    edges = cv2.Canny(roi_g, 60, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_len,
        maxLineGap=10,
    )
    if lines is None:
        return None

    best = None
    best_len = 0.0
    for (x1, y1, x2, y2) in lines[:, 0, :]:
        L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if L > best_len:
            best_len = L
            best = (int(x1), int(y1), int(x2), int(y2))
    return best


def _find_contact_point(gray_roi: np.ndarray, line: Tuple[int, int, int, int], center_roi: Tuple[float, float]) -> Tuple[float, float]:
    """
    Contact point ≈ first strong edge peak near tip-side (endpoint closer to center).
    - use gradient magnitude
    - pick earliest strong peak within first N px from tip into shaft direction
    """
    x1, y1, x2, y2 = line
    cx, cy = center_roi

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    if d1 <= d2:
        tip = np.array([x1, y1], dtype=np.float32)
        tail = np.array([x2, y2], dtype=np.float32)
    else:
        tip = np.array([x2, y2], dtype=np.float32)
        tail = np.array([x1, y1], dtype=np.float32)

    v = tail - tip
    L = float(np.linalg.norm(v))
    if L < 8:
        return float(tip[0]), float(tip[1])
    v = v / (L + 1e-12)

    gX = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gY = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gX * gX + gY * gY)

    scan_n = int(min(55, max(18, L * 0.35)))
    vals, xs, ys = [], [], []
    for i in range(scan_n):
        p = tip + v * i
        px, py = int(round(p[0])), int(round(p[1]))
        if px < 1 or py < 1 or px >= gray_roi.shape[1] - 1 or py >= gray_roi.shape[0] - 1:
            continue
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        vals.append(float(mag[py, px]))

    if not vals:
        return float(tip[0]), float(tip[1])

    arr = np.array(vals, dtype=np.float32)
    thr = float(np.percentile(arr, 85))
    idx = np.where(arr >= thr)[0]
    if len(idx) == 0:
        k = int(np.argmax(arr))
        return xs[k], ys[k]
    k = int(idx[0])  # earliest strong edge
    return xs[k], ys[k]


def refine_points_and_colors(
    rect_bgr: np.ndarray,
    target_center: Tuple[float, float],
    coarse_points: List[Tuple[float, float]],
    arrow_present: bool,
    roi_radius: int = 70,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
    h, w = rect_bgr.shape[:2]
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)

    refined: List[Tuple[float, float]] = []
    colors: List[Dict[str, Any]] = []

    tcx, tcy = target_center

    for (x, y) in coarse_points:
        x1, y1, x2, y2 = _clip_roi(x, y, roi_radius, w, h)
        roi_g = gray[y1:y2, x1:x2]
        if roi_g.size == 0:
            refined.append((x, y))
            colors.append({"ok": False})
            continue

        center_roi = (tcx - x1, tcy - y1)
        lx, ly = float(x - x1), float(y - y1)

        if arrow_present:
            min_len = max(22, int(min(roi_g.shape) * 0.40))
            seg = _best_arrow_segment_in_roi(roi_g, min_len=min_len)
            if seg is not None:
                rx, ry = _find_contact_point(roi_g, seg, center_roi)
                rx2, ry2 = _subpix_refine(roi_g, (rx, ry))
                fx, fy = rx2 + x1, ry2 + y1
                refined.append((fx, fy))
                colors.append(_sample_face_color_under_arrow(rect_bgr, fx, fy, r=10))
                continue

            rx, ry = _subpix_refine(roi_g, (lx, ly))
            fx, fy = rx + x1, ry + y1
            refined.append((fx, fy))
            colors.append(_sample_face_color_under_arrow(rect_bgr, fx, fy, r=10))
            continue

        # no arrow: blob/hole path (stable)
        g = cv2.GaussianBlur(roi_g, (0, 0), 1.2)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        lap = np.abs(lap)
        lap_u8 = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, th = cv2.threshold(lap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.medianBlur(th, 5)

        num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
        best = None
        best_d = 1e18
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 30 or area > 2000:
                continue
            bx, by = cents[i]
            d = (bx - lx) ** 2 + (by - ly) ** 2
            if d < best_d:
                best_d = d
                best = (float(bx), float(by))

        if best is not None:
            rx, ry = _subpix_refine(roi_g, best)
            fx, fy = rx + x1, ry + y1
            refined.append((fx, fy))
            colors.append(_sample_face_color_under_arrow(rect_bgr, fx, fy, r=10))
            continue

        rx, ry = _subpix_refine(roi_g, (lx, ly))
        fx, fy = rx + x1, ry + y1
        refined.append((fx, fy))
        colors.append(_sample_face_color_under_arrow(rect_bgr, fx, fy, r=10))

    return refined, colors
