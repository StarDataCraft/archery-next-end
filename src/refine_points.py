from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2


def _clip_roi(x, y, r, w, h):
    x1 = max(0, int(round(x - r)))
    y1 = max(0, int(round(y - r)))
    x2 = min(w, int(round(x + r)))
    y2 = min(h, int(round(y + r)))
    return x1, y1, x2, y2


def _subpix_refine(gray_roi: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    """
    Refine point within ROI using cornerSubPix-like local optimization.
    Works best when there is an edge/corner-ish signal; we still try.
    """
    # cornerSubPix requires float32 image
    img = gray_roi.astype(np.float32)
    p = np.array([[pt]], dtype=np.float32)  # shape (1,1,2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    # winSize and zeroZone tuned for small ROI
    try:
        cv2.cornerSubPix(img, p, winSize=(7, 7), zeroZone=(-1, -1), criteria=criteria)
        return float(p[0, 0, 0]), float(p[0, 0, 1])
    except Exception:
        return float(pt[0]), float(pt[1])


def refine_points(
    rect_bgr: np.ndarray,
    center: Tuple[float, float],
    coarse_points: List[Tuple[float, float]],
    arrow_present: bool,
    roi_radius: int = 40,
) -> List[Tuple[float, float]]:
    """
    Automatically refine coarse hit points in rectified-photo coordinates.
    No user help.

    - arrow_present=True: prioritize line/tip structure
    - arrow_present=False: prioritize blob/hole structure
    """
    h, w = rect_bgr.shape[:2]
    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)

    refined: List[Tuple[float, float]] = []

    cx, cy = center

    for (x, y) in coarse_points:
        x1, y1, x2, y2 = _clip_roi(x, y, roi_radius, w, h)
        roi = rect_bgr[y1:y2, x1:x2]
        roi_g = gray[y1:y2, x1:x2]
        if roi.size == 0:
            refined.append((x, y))
            continue

        # local coordinates in ROI
        lx, ly = float(x - x1), float(y - y1)

        if arrow_present:
            # 1) detect line segments in ROI
            edges = cv2.Canny(roi_g, 60, 140)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=40,
                minLineLength=max(15, int(min(roi_g.shape) * 0.35)),
                maxLineGap=8,
            )

            if lines is not None and len(lines) > 0:
                # Choose the line whose one endpoint is closest to target center (tip side)
                best_pt = None
                best_dist = 1e18

                for (xA, yA, xB, yB) in lines[:, 0, :]:
                    # endpoints in full-image coords
                    AX, AY = xA + x1, yA + y1
                    BX, BY = xB + x1, yB + y1

                    dA = (AX - cx) ** 2 + (AY - cy) ** 2
                    dB = (BX - cx) ** 2 + (BY - cy) ** 2

                    # tip side: closer to center
                    if dA <= dB:
                        cand = (xA, yA)
                        d = dA
                    else:
                        cand = (xB, yB)
                        d = dB

                    if d < best_dist:
                        best_dist = d
                        best_pt = cand

                if best_pt is not None:
                    # 2) subpixel refine around that endpoint (in ROI coords)
                    rx, ry = _subpix_refine(roi_g, (float(best_pt[0]), float(best_pt[1])))
                    refined.append((rx + x1, ry + y1))
                    continue

            # fallback: just subpixel refine around coarse location
            rx, ry = _subpix_refine(roi_g, (lx, ly))
            refined.append((rx + x1, ry + y1))
            continue

        else:
            # No arrow: blob / hole refinement
            # Use Laplacian of Gaussian (LoG)-like via GaussianBlur + Laplacian
            g = cv2.GaussianBlur(roi_g, (0, 0), 1.2)
            lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
            lap = np.abs(lap)

            # Normalize and threshold to find candidate blob
            lap_u8 = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, th = cv2.threshold(lap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th = cv2.medianBlur(th, 5)

            # connected components centroids
            num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(th, connectivity=8)
            best = None
            best_d = 1e18

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 20 or area > 1500:
                    continue
                bx, by = cents[i]
                # prefer near coarse
                d = (bx - lx) ** 2 + (by - ly) ** 2
                if d < best_d:
                    best_d = d
                    best = (float(bx), float(by))

            if best is not None:
                rx, ry = _subpix_refine(roi_g, best)
                refined.append((rx + x1, ry + y1))
                continue

            # fallback: subpixel refine at coarse
            rx, ry = _subpix_refine(roi_g, (lx, ly))
            refined.append((rx + x1, ry + y1))

    return refined
