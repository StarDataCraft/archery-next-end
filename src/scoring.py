# src/scoring.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import math


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _score_by_radius(center: Tuple[float, float], outer_radius: float, p: Tuple[float, float]) -> int:
    """
    WA 10-ring style: outer radius divided into 10 equal bands.
    10 at center, 1 near outer ring, 0 outside.
    """
    r = _dist(center, p)
    if outer_radius <= 1e-6:
        return 0
    if r > outer_radius:
        return 0
    band = outer_radius / 10.0
    # ring index: 0..9 (0 is inner)
    idx = int(r // band)
    # score: 10..1
    return max(1, 10 - idx)


def _classify_color_hsv(hsv: Tuple[float, float, float]) -> str:
    """
    Robust-ish HSV band classifier.
    hsv: (H in [0,180], S in [0,255], V in [0,255]) typical OpenCV.
    Returns: 'gold','red','blue','black','white','unknown'
    """
    h, s, v = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # black: very low value
    if v < 60:
        return "black"

    # white: low saturation, high value
    if s < 45 and v > 170:
        return "white"

    # very low saturation but not bright enough -> unknown (gray)
    if s < 35:
        return "unknown"

    # red wraps around hue
    if (h <= 10 or h >= 170) and s > 70 and v > 60:
        return "red"

    # gold/yellow: roughly 15..40
    if 15 <= h <= 40 and s > 60 and v > 60:
        return "gold"

    # blue: roughly 90..140
    if 90 <= h <= 140 and s > 70 and v > 50:
        return "blue"

    return "unknown"


def _allowed_scores_for_color(color: str) -> Optional[List[int]]:
    """
    WA 10-ring face colors:
      gold: 10,9
      red : 8,7
      blue: 6,5
      black:4,3
      white:2,1
    """
    mapping = {
        "gold": [10, 9],
        "red": [8, 7],
        "blue": [6, 5],
        "black": [4, 3],
        "white": [2, 1],
    }
    return mapping.get(color)


def _best_score_in_color_band(
    center: Tuple[float, float],
    outer_radius: float,
    p: Tuple[float, float],
    allowed_scores: List[int],
) -> int:
    """
    Choose the best score within allowed_scores using ring boundary distances.
    For each candidate score s, the corresponding band index is (10 - s).
    We pick the candidate whose band center radius is closest to r.
    """
    r = _dist(center, p)
    band = outer_radius / 10.0
    best_s = allowed_scores[0]
    best_err = 1e18
    for s in allowed_scores:
        idx = 10 - s  # 0..9
        # approximate band center radius
        r_center = (idx + 0.5) * band
        err = abs(r - r_center)
        if err < best_err:
            best_err = err
            best_s = s
    return best_s


def score_hits(
    center: Tuple[float, float],
    outer_radius: float,
    points_xy: List[Tuple[float, float]],
) -> Dict[str, Any]:
    scores = [_score_by_radius(center, outer_radius, p) for p in points_xy]
    total = int(sum(scores))
    avg = float(total / len(scores)) if scores else 0.0
    return {"scores": scores, "total": total, "avg": avg}


def score_hits_color_aware(
    center: Tuple[float, float],
    outer_radius: float,
    points_xy: List[Tuple[float, float]],
    contact_hsvs: Optional[List[Optional[Tuple[float, float, float]]]] = None,
) -> Dict[str, Any]:
    """
    Color-aware scoring:
      1) Compute radial score (baseline).
      2) Classify contact HSV into color band.
      3) If the band implies allowed scores and baseline score not in it,
         adjust score to best within that band based on r.
    """
    details: List[Dict[str, Any]] = []
    final_scores: List[int] = []

    if contact_hsvs is None:
        contact_hsvs = [None] * len(points_xy)

    for p, hsv in zip(points_xy, contact_hsvs):
        radial = _score_by_radius(center, outer_radius, p)

        if hsv is None:
            final = radial
            details.append(
                {
                    "radial_score": radial,
                    "color_class": None,
                    "final_score": final,
                    "note": "no_hsv",
                }
            )
            final_scores.append(final)
            continue

        color = _classify_color_hsv(hsv)
        allowed = _allowed_scores_for_color(color)

        if allowed is None:
            # unknown color: keep radial
            final = radial
            note = "unknown_color_keep_radial"
        else:
            if radial in allowed:
                final = radial
                note = "radial_matches_color"
            else:
                final = _best_score_in_color_band(center, outer_radius, p, allowed)
                note = "color_corrected"

        details.append(
            {
                "radial_score": radial,
                "color_class": color,
                "hsv": tuple(float(x) for x in hsv),
                "final_score": int(final),
                "note": note,
            }
        )
        final_scores.append(int(final))

    total = int(sum(final_scores))
    avg = float(total / len(final_scores)) if final_scores else 0.0
    return {"scores": final_scores, "total": total, "avg": avg, "details": details}
