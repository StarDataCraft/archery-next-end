from __future__ import annotations
from typing import List, Tuple

def ring_radii_px(outer_radius_px: float, target_face: str) -> List[float]:
    """
    WA 80cm/40cm 10-ring face:
    ring widths are uniform in radius: outer radius divided into 10 bands.
    - 80cm: diameters 8,16,...,80 => radii 4..40 => ratios 0.1..1.0
    - 40cm: diameters 4,8,...,40 => radii 2..20 => ratios 0.1..1.0
    So identical ratios work: r_k = outer * (k/10)
    """
    # Keep future-proof: if add other faces, branch here.
    ratios = [(k / 10.0) for k in range(1, 11)]  # 0.1..1.0
    return [outer_radius_px * r for r in ratios]

def score_hit(center: Tuple[float, float], outer_radius_px: float, x: float, y: float) -> int:
    """
    Score 10..1 based on which radial band the hit falls into.
    Miss => 0
    """
    cx, cy = center
    dx, dy = x - cx, y - cy
    d = (dx * dx + dy * dy) ** 0.5
    if d > outer_radius_px:
        return 0
    band = outer_radius_px / 10.0
    # d in (0..band] -> 10, (band..2band] -> 9, ...
    k = int(d // band)  # 0..9
    return max(10 - k, 1)

def score_hits(center: Tuple[float, float], outer_radius_px: float, points_xy: List[Tuple[float, float]]):
    scores = [score_hit(center, outer_radius_px, x, y) for (x, y) in points_xy]
    return {
        "scores": scores,
        "total": int(sum(scores)),
        "avg": float(sum(scores) / len(scores)) if scores else 0.0
    }
