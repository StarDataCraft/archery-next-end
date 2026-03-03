from __future__ import annotations
import json
from datetime import datetime

def make_log_entry(distance_m: int, arrows_per_end: int, handedness: str, target_face: str, metrics: dict, scoring: dict, advice: dict) -> dict:
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "distance_m": distance_m,
        "arrows_per_end": arrows_per_end,
        "handedness": handedness,
        "target_face": target_face,
        "metrics": metrics,
        "scoring": scoring,
        "advice": advice,
    }

def export_log_json(log: list[dict]) -> str:
    return json.dumps(log, ensure_ascii=False, indent=2)
