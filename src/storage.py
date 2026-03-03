import json
from typing import Any, Dict, List


def make_log_entry(
    distance_m: int,
    arrows_per_end: int,
    handedness: str,
    target_face: str,
    metrics: Dict[str, Any],
    scoring: Dict[str, Any],
    advice: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "distance_m": distance_m,
        "arrows_per_end": arrows_per_end,
        "handedness": handedness,
        "target_face": target_face,
        "metrics": metrics,
        "scoring": scoring,
        "advice": advice,
    }


def export_log_json(log: List[Dict[str, Any]]) -> str:
    return json.dumps(log, ensure_ascii=False, indent=2)
