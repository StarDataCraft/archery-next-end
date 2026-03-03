from __future__ import annotations
from typing import Dict, List, Tuple
import math


def score_hits(center: Tuple[float, float], outer: float, pts_xy: List[Tuple[float, float]]) -> Dict:
    cx, cy = center
    ring_step = outer / 10.0

    scores = []
    for x, y in pts_xy:
        d = math.hypot(x - cx, y - cy)
        raw = 10 - int(d // ring_step)
        s = max(0, min(10, raw))
        scores.append(s)

    total = sum(scores)
    avg = total / len(scores) if scores else 0.0
    return {"scores": scores, "total": total, "avg": avg}
