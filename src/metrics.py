# src/metrics.py
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any, List
import numpy as np


def compute_metrics(
    points: List[dict],
    center: Optional[Tuple[float, float]] = None,
    outer_radius_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute grouping metrics on canonical target coordinates.
    If `center` is provided, also compute offset dx/dy from target center.
    If `outer_radius_px` is provided, also provide normalized ratios.
    """
    if not points:
        return {
            "n": 0,
            "centroid": {"x": 0.0, "y": 0.0},
            "spread": 0.0,
            "sx": 0.0,
            "sy": 0.0,
            "slope_deg": 0.0,
            "offset": {"dx": 0.0, "dy": 0.0, "mag": 0.0},
            "spread_ratio": None,
            "offset_ratio": None,
        }

    xy = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    cx, cy = xy.mean(axis=0)

    # spread: avg distance to centroid
    d = np.sqrt(((xy - np.array([cx, cy])) ** 2).sum(axis=1))
    spread = float(d.mean())

    # axis-wise dispersion
    sx = float(xy[:, 0].std(ddof=0))
    sy = float(xy[:, 1].std(ddof=0))

    # principal axis direction
    cov = np.cov(xy.T, ddof=0)
    vals, vecs = np.linalg.eig(cov)
    main_vec = vecs[:, int(np.argmax(vals))]
    slope_rad = float(np.arctan2(main_vec[1], main_vec[0]))
    slope_deg = float(slope_rad * 180.0 / np.pi)

    # offset from target center (if provided)
    dx = dy = mag = 0.0
    if center is not None:
        tx, ty = float(center[0]), float(center[1])
        dx = float(cx - tx)
        dy = float(cy - ty)
        mag = float((dx * dx + dy * dy) ** 0.5)

    spread_ratio = None
    offset_ratio = None
    if outer_radius_px is not None and outer_radius_px > 1e-6:
        spread_ratio = float(spread / float(outer_radius_px))
        offset_ratio = float(mag / float(outer_radius_px))

    return {
        "n": int(len(points)),
        "centroid": {"x": float(cx), "y": float(cy)},
        "spread": spread,
        "sx": sx,
        "sy": sy,
        "slope_deg": slope_deg,
        "offset": {"dx": dx, "dy": dy, "mag": mag},
        "spread_ratio": spread_ratio,
        "offset_ratio": offset_ratio,
    }


def classify_shape(metrics: dict) -> str:
    sx, sy = float(metrics.get("sx", 0.0)), float(metrics.get("sy", 0.0))
    if sx == 0.0 and sy == 0.0:
        return "tight"
    ratio = (sx / sy) if sy > 1e-6 else 999.0
    if ratio > 1.4:
        return "horizontal"
    if ratio < (1 / 1.4):
        return "vertical"
    return "round"
