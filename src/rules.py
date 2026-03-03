from __future__ import annotations
from typing import Dict, Any
from .kb_patterns import retrieve_patterns


def _dir_word(dx: float, dy: float, lang: str) -> str:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return {"en": "centered", "ja": "中央", "zh": "居中"}[lang]

    horiz = "right" if dx > 0 else "left"
    vert = "low" if dy > 0 else "high"  # y down

    maps = {
        "en": {"left": "left", "right": "right", "high": "high", "low": "low"},
        "ja": {"left": "左", "right": "右", "high": "上", "low": "下"},
        "zh": {"left": "左", "right": "右", "high": "上", "low": "下"},
    }
    return f"{maps[lang][vert]}-{maps[lang][horiz]}"


def next_end_advice(
    metrics: Dict[str, float],
    shape: str,
    handedness: str,
    distance_m: int,
    arrow_present: bool,
    lang: str = "en",
) -> Dict[str, Any]:
    lang = lang if lang in ("en", "ja", "zh") else "en"

    n = int(metrics.get("n", 0))
    cx = float(metrics.get("cx", 0.0))
    cy = float(metrics.get("cy", 0.0))

    dx = cx - 450.0
    dy = cy - 450.0
    dirw = _dir_word(dx, dy, lang)

    spread = float(metrics.get("spread", 0.0))
    slope = float(metrics.get("slope_deg", 0.0))

    if n <= 1:
        title = {"en": "Single-arrow cue", "ja": "1本だけのヒント", "zh": "单箭提示"}[lang]
        cue = {
            "en": f"Hit is {dirw}. Next end: keep the same rhythm; keep the front side quiet (no shoulder lift / no collapse).",
            "ja": f"着弾は「{dirw}」。次はリズム固定、フロント側を静かに（肩が上がらない・潰れない）。",
            "zh": f"落点在「{dirw}」。下一轮节奏固定，前侧保持安静（别耸肩、别塌）。",
        }[lang]
    else:
        title = {"en": "Next-end cue", "ja": "次エンドの合図", "zh": "下一轮提示"}[lang]
        cue = {
            "en": f"Centroid is {dirw}. Shape={shape}, spread≈{spread:.0f}px, slope≈{slope:.0f}°. Next end: keep rhythm, fix ONE thing only.",
            "ja": f"重心は「{dirw}」。形={shape} / ばらつき≈{spread:.0f}px / 角度≈{slope:.0f}°。次はリズム優先、直すのは1つだけ。",
            "zh": f"重心在「{dirw}」。形状={shape} / 离散≈{spread:.0f}px / 角度≈{slope:.0f}°。下一轮先稳节奏，只修一个点。",
        }[lang]

    # —— build retrieval context (flexible) ——
    ctx = (
        f"archery recurve. distance {distance_m}m. handedness {handedness}. "
        f"arrow_present {arrow_present}. group shape {shape}. spread {spread:.1f}. "
        f"centroid dx {dx:.1f} dy {dy:.1f}. "
    )
    if distance_m <= 18 and spread > 55:
        ctx += "group larger at short distance. clearance likely. "
    if distance_m >= 50 and spread > 75:
        ctx += "group larger at long distance drift. speed matters. "
    if shape in ("horizontal", "vertical"):
        ctx += f"group elongated {shape}. timing consistency. "

    top = retrieve_patterns(ctx, k=3)

    why_lines = []
    for item, score in top:
        if score <= 0.01:
            continue
        why_lines.append(item.texts[lang])

    if not why_lines:
        why_lines = [{
            "en": "Keep it boring: repeat the same shot. If it’s unstable, clean up clearance/alignment before micro-tuning.",
            "ja": "地味でOK：同じ動作を反復。不安定なら微調整より先にクリアランスとアライメント。",
            "zh": "无聊但有效：重复同一动作。不稳定先查清弦与对齐，再谈微调。",
        }[lang]]

    return {"title": title, "cue": cue, "why": "\n".join([f"- {x}" for x in why_lines])}
