from __future__ import annotations
import numpy as np

def compute_metrics(points: list[dict]) -> dict:
    """
    points: [{"x": float, "y": float}, ...] in image coordinates
    returns: centroid, spread, sx, sy, slope_deg
    """
    xy = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    cx, cy = xy.mean(axis=0)

    d = np.sqrt(((xy - np.array([cx, cy])) ** 2).sum(axis=1))
    spread = float(d.mean())

    sx = float(xy[:, 0].std(ddof=0))
    sy = float(xy[:, 1].std(ddof=0))

    # principal direction (PCA 2D)
    cov = np.cov(xy.T, ddof=0)
    vals, vecs = np.linalg.eig(cov)
    main_vec = vecs[:, int(np.argmax(vals))]
    slope_rad = float(np.arctan2(main_vec[1], main_vec[0]))
    slope_deg = slope_rad * 180.0 / np.pi

    return {
        "n": int(len(points)),
        "centroid": {"x": float(cx), "y": float(cy)},
        "spread": spread,
        "sx": sx,
        "sy": sy,
        "slope_deg": slope_deg,
    }

def classify_shape(metrics: dict) -> str:
    sx, sy = metrics["sx"], metrics["sy"]
    if sx == 0 and sy == 0:
        return "tight"

    ratio = (sx / sy) if sy > 1e-6 else 999.0
    if ratio > 1.4:
        return "horizontal"
    if ratio < (1/1.4):
        return "vertical"
    return "round"
