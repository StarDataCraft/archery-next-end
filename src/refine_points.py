from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

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


def _sample_color_features(bgr: np.ndarray, x: float, y: float, r: int = 8) -> Dict[str, Any]:
    h, w = bgr.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r + 1)
    y2 = min(h, cy + r + 1)
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"ok": False}

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)

    # robust: median
    feat = {
        "ok": True,
        "hsv_median": [int(np.median(hsv[:, :, 0])), int(np.median(hsv[:, :, 1])), int(np.median(hsv[:, :, 2]))],
        "lab_median": [int(np.median(lab[:, :, 0])), int(np.median(lab[:, :, 1])), int(np.median(lab[:, :, 2]))],
    }
    return feat


def _best_arrow_segment_in_roi(edges: np.ndarray, min_len: int) -> Optional[Tuple[int, int, int, int]]:
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=min_len,
        maxLineGap=8,
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


def _find_contact_point_along_line(
    gray_roi: np.ndarray,
    line: Tuple[int, int, int, int],
    center_roi: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Given a line segment in ROI coords, find "contact point" near the endpoint closer to target center:
    - choose endpoint closer to center as direction of tip
    - sample gradient magnitude along the line towards the center-side
    - contact point is first strong edge peak near that endpoint
    """
    x1, y1, x2, y2 = line
    cx, cy = center_roi

    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2

    # endpoint closer to center: tip direction
    if d1 <= d2:
        tip = np.array([x1, y1], dtype=np.float32)
        tail = np.array([x2, y2], dtype=np.float32)
    else:
        tip = np.array([x2, y2], dtype=np.float32)
        tail = np.array([x1, y1], dtype=np.float32)

    v = tail - tip
    L = float(np.linalg.norm(v))
    if L < 5:
        return float(tip[0]), float(tip[1])
    v = v / L

    # gradient magnitude in ROI
    gX = cv2.Sobel(gray_roi, cv2.CV_32F, 1, 0, ksize=3)
    gY = cv2.Sobel(gray_roi, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gX * gX + gY * gY)

    # scan a short distance from tip into the shaft direction
    # contact point typically sits just inside tip region; scan 0..35px
    scan_n = int(min(40, max(12, L * 0.25)))
    xs = []
    ys = []
    vals = []
    for i in range(scan_n):
        p = tip + v * i
        px, py = int(round(p[0])), int(round(p[1]))
        if px < 1 or py < 1 or px >= gray_roi.shape[1] - 1 or py >= gray_roi.shape[0] - 1:
            continue
        xs.append(p[0])
        ys.append(p[1])
        vals.append(float(mag[py, px]))

    if not vals:
        return float(tip[0]), float(tip[1])

    # find first strong peak: use percentile threshold
    arr = np.array(vals, dtype=np.float32)
    thr = float(np.percentile(arr, 80))
    idx_candidates = np.where(arr >= thr)[0]
    if len(idx_candidates) == 0:
        # fallback: max
        k = int(np.argmax(arr))
        return float(xs[k]), float(ys[k])

    # choose earliest index among candidates (closest to tip)
    k = int(idx_candidates[0])
    return float(xs[k]), float(ys[k])


def refine_points_and_colors(
    rect_bgr: np.ndarray,
    target_center: Tuple[float, float],
    coarse_points: List[Tuple[float, float]],
    arrow_present: bool,
    roi_radius: int = 55,
) -> Tuple[List[Tuple[float, float]], List[Dict[str, Any]]]:
    """
    Refine hit/contact points in rectified-photo coords (rect_bgr).
    Returns refined points + color features near contact point.
    """
    h, w = rect_bgr.shape[:2]
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)

    refined: List[Tuple[float, float]] = []
    colors: List[Dict[str, Any]] = []

    tcx, tcy = target_center

    for (x, y) in coarse_points:
        x1, y1, x2, y2 = _clip_roi(x, y, roi_radius, w, h)
        roi = rect_bgr[y1:y2, x1:x2]
        roi_g = gray[y1:y2, x1:x2]
        if roi.size == 0:
            refined.append((x, y))
            colors.append({"ok": False})
            continue

        # ROI center in ROI coords (target center projected into ROI)
        center_roi = (tcx - x1, tcy - y1)

        lx, ly = float(x - x1), float(y - y1)

        if arrow_present:
            edges = cv2.Canny(roi_g, 60, 150)
            min_len = max(18, int(min(roi_g.shape) * 0.35))
            seg = _best_arrow_segment_in_roi(edges, min_len=min_len)

            if seg is not None:
                # contact point along the line (ROI coords)
                rx, ry = _find_contact_point_along_line(roi_g, seg, center_roi)
                # subpixel refine around contact point
                rx2, ry2 = _subpix_refine(roi_g, (rx, ry))
                fx, fy = rx2 + x1, ry2 + y1
                refined.append((fx, fy))
                colors.append(_sample_color_features(rect_bgr, fx, fy, r=9))
                continue

            # fallback: refine around coarse
            rx, ry = _subpix_refine(roi_g, (lx, ly))
            fx, fy = rx + x1, ry + y1
            refined.append((fx, fy))
            colors.append(_sample_color_features(rect_bgr, fx, fy, r=9))
            continue

        # No arrow: treat as hole/blob
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
            if area < 20 or area > 1500:
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
            colors.append(_sample_color_features(rect_bgr, fx, fy, r=9))
            continue

        rx, ry = _subpix_refine(roi_g, (lx, ly))
        fx, fy = rx + x1, ry + y1
        refined.append((fx, fy))
        colors.append(_sample_color_features(rect_bgr, fx, fy, r=9))

    return refined, colors
