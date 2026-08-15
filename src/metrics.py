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
            "anisotropy": 1.0,
            "outlier": {"present": False, "index": None, "distance": 0.0, "core_spread": 0.0, "improvement_ratio": 0.0},
        }

    xy = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    cx, cy = xy.mean(axis=0)

    # spread: avg distance to centroid
    d = np.sqrt(((xy - np.array([cx, cy])) ** 2).sum(axis=1))
    spread = float(d.mean())

    # axis-wise dispersion
    sx = float(xy[:, 0].std(ddof=0))
    sy = float(xy[:, 1].std(ddof=0))
    anisotropy = float(max(sx, sy) / max(min(sx, sy), 1e-6))

    # A single escaped arrow should not be treated as a whole-group form
    # pattern. Detect it conservatively with both a robust distance test and a
    # meaningful improvement in the remaining core group's spread.
    outlier = {
        "present": False,
        "index": None,
        "distance": 0.0,
        "core_spread": spread,
        "improvement_ratio": 0.0,
    }
    if len(points) >= 5 and spread > 1e-6:
        robust_center = np.median(xy, axis=0)
        robust_distances = np.linalg.norm(xy - robust_center, axis=1)
        candidate_index = int(np.argmax(robust_distances))
        candidate_distance = float(robust_distances[candidate_index])
        median_distance = float(np.median(robust_distances))
        mad = float(np.median(np.abs(robust_distances - median_distance)))

        core_xy = np.delete(xy, candidate_index, axis=0)
        core_center = core_xy.mean(axis=0)
        core_distances = np.linalg.norm(core_xy - core_center, axis=1)
        core_spread = float(core_distances.mean())
        improvement_ratio = float(max(0.0, (spread - core_spread) / spread))

        robust_limit = median_distance + 3.0 * max(mad, 2.0)
        relative_limit = max(20.0, median_distance * 1.9)
        present = (
            candidate_distance > robust_limit
            and candidate_distance > relative_limit
            and improvement_ratio >= 0.25
        )
        outlier = {
            "present": bool(present),
            "index": candidate_index if present else None,
            "distance": candidate_distance,
            "core_spread": core_spread if present else spread,
            "improvement_ratio": improvement_ratio if present else 0.0,
        }

    # Principal-axis direction. ``eigh`` is the correct solver for this real,
    # symmetric covariance matrix and always returns real eigenvectors. Some
    # NumPy/LAPACK builds return complex values from the generic ``eig`` even
    # when the imaginary part is zero, which makes ``arctan2`` fail.
    slope_deg = 0.0
    if len(points) >= 2:
        cov = np.cov(xy.T, ddof=0)
        cov = (cov + cov.T) / 2.0
        if np.isfinite(cov).all():
            vals, vecs = np.linalg.eigh(cov)
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
        "anisotropy": anisotropy,
        "outlier": outlier,
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
