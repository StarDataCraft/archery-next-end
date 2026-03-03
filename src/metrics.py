from __future__ import annotations
from typing import Dict, List
import math


def compute_metrics(points: List[dict]) -> Dict[str, float]:
    n = len(points)
    if n == 0:
        return {"n": 0, "cx": 0.0, "cy": 0.0, "sx": 0.0, "sy": 0.0, "spread": 0.0, "slope_deg": 0.0}

    xs = [float(p["x"]) for p in points]
    ys = [float(p["y"]) for p in points]
    cx = sum(xs) / n
    cy = sum(ys) / n

    if n == 1:
        return {"n": 1, "cx": cx, "cy": cy, "sx": 0.0, "sy": 0.0, "spread": 0.0, "slope_deg": 0.0}

    sx = math.sqrt(sum((x - cx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - cy) ** 2 for y in ys) / (n - 1))

    spread = sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / n

    denom = sum((x - cx) ** 2 for x in xs)
    if denom < 1e-9:
        slope_deg = 90.0
    else:
        a = sum((x - cx) * (y - cy) for x, y in zip(xs, ys)) / denom
        slope_deg = math.degrees(math.atan(a))

    return {"n": n, "cx": cx, "cy": cy, "sx": sx, "sy": sy, "spread": spread, "slope_deg": slope_deg}


def classify_shape(metrics: Dict[str, float]) -> str:
    n = int(metrics.get("n", 0))
    if n <= 1:
        return "single"

    sx = float(metrics.get("sx", 0.0))
    sy = float(metrics.get("sy", 0.0))
    if sx < 1e-6 and sy < 1e-6:
        return "tight"

    r = (sx + 1e-6) / (sy + 1e-6)
    if r > 1.6:
        return "horizontal"
    if r < 0.62:
        return "vertical"
    return "round"
