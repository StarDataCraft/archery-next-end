# src/scoring.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math


# ------------------------------------------------------------
# Tunables (you can adjust these without touching core logic)
# ------------------------------------------------------------
# How close to a ring boundary to allow color to influence score (in normalized units of outer_radius)
BOUNDARY_BAND_NORM = 0.010  # 1.0% of outer radius; try 0.008 ~ 0.015

# Color can only adjust score if confidence is >= this threshold
MIN_COLOR_CONF = 0.60

# Color can only adjust score by at most +/-1
MAX_COLOR_DELTA = 1

# If color suggests something far from radial score, ignore it (hard gate)
MAX_ALLOWED_COLOR_SCORE_GAP = 1


@dataclass
class ColorHint:
    score: Optional[int] = None   # suggested score from color band
    conf: float = 0.0             # 0..1 confidence


# ------------------------------------------------------------
# Public API (keep signature stable)
# ------------------------------------------------------------
def score_hits_color_aware(
    center_xy: Tuple[float, float],
    outer_radius_px: float,
    points_xy: List[Tuple[float, float]],
    contact_hsvs: Optional[List[Optional[Tuple[float, float, float]]]] = None,
) -> Dict[str, Any]:
    """
    Priority order:
      1) radial distance -> base score (main truth after rectify)
      2) color hint -> ONLY as boundary-time correction / sanity-check

    Returns:
      {
        "scores": [int,...],
        "total": int,
        "avg": float,
        "details": [ {per-arrow debug dict}, ... ]
      }
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    R = float(outer_radius_px)

    scores: List[int] = []
    details: List[Dict[str, Any]] = []

    if contact_hsvs is None:
        contact_hsvs = [None] * len(points_xy)

    for i, ((x, y), hsv) in enumerate(zip(points_xy, contact_hsvs)):
        dx = float(x) - cx
        dy = float(y) - cy
        r_px = math.hypot(dx, dy)
        r_norm = r_px / (R + 1e-9)

        # 1) Base score from radial distance
        base_score, base_meta = _radial_score(r_norm)

        # 2) Optional color hint
        hint = _color_hint_from_hsv(hsv)

        # 3) Decide if color is allowed to influence (only near boundary)
        final_score, decision = _fuse_radial_and_color(
            base_score=base_score,
            base_meta=base_meta,
            r_norm=r_norm,
            hint=hint,
        )

        scores.append(final_score)
        details.append(
            {
                "i": i,
                "pt": {"x": float(x), "y": float(y)},
                "r_px": float(r_px),
                "r_norm": float(r_norm),
                "radial": {"score": int(base_score), **base_meta},
                "color": {
                    "hsv": None if hsv is None else [float(hsv[0]), float(hsv[1]), float(hsv[2])],
                    "hint_score": None if hint.score is None else int(hint.score),
                    "conf": float(hint.conf),
                },
                "decision": decision,
                "score": int(final_score),
            }
        )

    total = int(sum(scores))
    avg = float(total / max(1, len(scores)))

    return {"scores": scores, "total": total, "avg": avg, "details": details}


# ------------------------------------------------------------
# Radial scoring (main truth)
# ------------------------------------------------------------
def _radial_score(r_norm: float) -> Tuple[int, Dict[str, Any]]:
    """
    World Archery target logic (simplified):
      - after rectify, ring radii are concentric
      - score determined by distance to center

    We assume outer_radius corresponds to the 1-ring outer boundary.
    Then ring boundaries are equally spaced in normalized radius:
      boundary for score s is at r = (11 - s) / 10

    Example:
      score 10 boundary at 0.1
      score 9 boundary at 0.2
      ...
      score 1 boundary at 1.0
      beyond -> 0
    """
    # clamp
    if r_norm < 0:
        r_norm = 0.0

    if r_norm > 1.0:
        return 0, {"band": "miss", "nearest_boundary": 1.0, "boundary_dist": float(r_norm - 1.0)}

    # Determine score by which interval r falls into
    # score s means r <= (11-s)/10
    # Equivalent: k = ceil(r*10) gives ring index 1..10; score = 11-k
    k = int(math.ceil(r_norm * 10.0 - 1e-12))  # tiny epsilon to avoid boundary jitter
    k = max(1, min(10, k))
    score = 11 - k

    # Compute nearest boundary distance (for boundary band logic)
    # Boundaries list: 0.1, 0.2, ..., 1.0
    lower = (k - 1) / 10.0
    upper = k / 10.0
    # interval is (lower, upper], score = 11 - k
    # nearest boundary distance:
    dist_to_lower = abs(r_norm - lower)
    dist_to_upper = abs(upper - r_norm)
    nearest = lower if dist_to_lower <= dist_to_upper else upper
    boundary_dist = min(dist_to_lower, dist_to_upper)

    return int(score), {
        "band": f"{score}",
        "interval": [float(lower), float(upper)],
        "nearest_boundary": float(nearest),
        "boundary_dist": float(boundary_dist),
    }


# ------------------------------------------------------------
# Color hint (secondary, only for boundary correction)
# ------------------------------------------------------------
def _color_hint_from_hsv(hsv: Optional[Tuple[float, float, float]]) -> ColorHint:
    """
    Very lightweight HSV-based band classifier.
    It should NEVER override radial unless near boundary + high confidence.

    OpenCV HSV:
      H: [0,180], S: [0,255], V: [0,255]
    """
    if hsv is None:
        return ColorHint(score=None, conf=0.0)

    H, S, V = float(hsv[0]), float(hsv[1]), float(hsv[2])

    # low saturation/value -> unreliable (arrow shadow / black lines / glare)
    if V < 70 or S < 50:
        return ColorHint(score=None, conf=0.0)

    # Rough hue regions:
    # yellow ~ 20-35 (opencv scale)
    # red ~ near 0 or near 180 (wrap)
    # blue ~ 95-130
    # black/white are low S -> already filtered
    is_yellow = (18 <= H <= 40)
    is_blue = (85 <= H <= 135)
    is_red = (H <= 10) or (H >= 170)

    # Confidence heuristic: higher S and V => more confident
    conf = min(1.0, (S / 255.0) * 0.7 + (V / 255.0) * 0.3)

    # Map color region -> plausible score range (soft hint)
    # WA face: yellow(10-9), red(8-7), blue(6-5), black(4-3), white(2-1)
    # We keep it coarse: return center score of that region as hint.
    if is_yellow:
        return ColorHint(score=10, conf=conf)
    if is_red:
        return ColorHint(score=8, conf=conf)
    if is_blue:
        return ColorHint(score=6, conf=conf)

    # If it's something else (e.g., black ring line or background), ignore.
    return ColorHint(score=None, conf=0.0)


def _fuse_radial_and_color(
    *,
    base_score: int,
    base_meta: Dict[str, Any],
    r_norm: float,
    hint: ColorHint,
) -> Tuple[int, Dict[str, Any]]:
    """
    Fusion rule:
      - radial is default
      - color only allowed to correct by +/-1,
        only if (a) near boundary, (b) high confidence,
        (c) hint score within 1 of base score
    """
    decision: Dict[str, Any] = {"mode": "radial_only"}

    # If no hint, keep base
    if hint.score is None or hint.conf < MIN_COLOR_CONF:
        if hint.score is None:
            decision["reason"] = "no_color_hint"
        else:
            decision["reason"] = f"low_color_conf<{MIN_COLOR_CONF:.2f}"
            decision["color_conf"] = float(hint.conf)
        return base_score, decision

    # Only near boundary
    boundary_dist = float(base_meta.get("boundary_dist", 999.0))
    if boundary_dist > BOUNDARY_BAND_NORM:
        decision["reason"] = f"not_near_boundary>{BOUNDARY_BAND_NORM:.3f}"
        decision["boundary_dist"] = boundary_dist
        decision["color_hint"] = {"score": int(hint.score), "conf": float(hint.conf)}
        return base_score, decision

    # If hint differs too much, ignore (prevents crazy jumps)
    gap = abs(int(hint.score) - int(base_score))
    if gap > MAX_ALLOWED_COLOR_SCORE_GAP:
        decision["reason"] = f"color_gap>{MAX_ALLOWED_COLOR_SCORE_GAP}"
        decision["gap"] = int(gap)
        decision["color_hint"] = {"score": int(hint.score), "conf": float(hint.conf)}
        return base_score, decision

    # Allow at most +/-1 adjustment
    delta = int(hint.score) - int(base_score)
    if delta == 0:
        decision["mode"] = "radial_confirmed_by_color"
        decision["boundary_dist"] = boundary_dist
        decision["color_hint"] = {"score": int(hint.score), "conf": float(hint.conf)}
        return base_score, decision

    delta = max(-MAX_COLOR_DELTA, min(MAX_COLOR_DELTA, delta))
    new_score = int(base_score) + delta

    # Safety clamp
    new_score = max(0, min(10, new_score))

    decision["mode"] = "boundary_adjusted_by_color"
    decision["boundary_dist"] = boundary_dist
    decision["color_hint"] = {"score": int(hint.score), "conf": float(hint.conf)}
    decision["delta"] = int(delta)
    decision["base_score"] = int(base_score)
    decision["final_score"] = int(new_score)
    return new_score, decision
