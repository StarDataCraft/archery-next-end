# src/rules.py
from __future__ import annotations

from typing import Optional, Dict, Any


def _msg(lang: str, ja: str, en: str, zh: str) -> str:
    if lang == "ja":
        return ja
    if lang == "zh":
        return zh
    return en


def _dir_from_dxdy(dx: float, dy: float) -> str:
    """
    Screen coords: +x right, +y down.
    Return one of: "left", "right", "up", "down", "up_left", ...
    """
    ax, ay = abs(dx), abs(dy)
    if ax < 1e-6 and ay < 1e-6:
        return "center"
    if ax > 1.2 * ay:
        return "right" if dx > 0 else "left"
    if ay > 1.2 * ax:
        return "down" if dy > 0 else "up"
    # diagonal
    h = "right" if dx > 0 else "left"
    v = "down" if dy > 0 else "up"
    return f"{v}_{h}"


def next_end_advice(
    metrics: dict,
    shape: str,
    handedness: str,
    lang: str = "en",
    quality: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Priority:
      1) Low-confidence safety mode
      2) Systematic offset (dx/dy)
      3) Loose dispersion (spread)
      4) Directional dispersion (axis/shape)
      5) Otherwise: rhythm
    """
    sx = float(metrics.get("sx", 0.0))
    sy = float(metrics.get("sy", 0.0))
    spread = float(metrics.get("spread", 0.0))
    spread_ratio = metrics.get("spread_ratio", None)
    offset = metrics.get("offset", {}) or {}
    dx = float(offset.get("dx", 0.0))
    dy = float(offset.get("dy", 0.0))
    offset_ratio = metrics.get("offset_ratio", None)

    # --------------------------
    # 1) Confidence gate
    # --------------------------
    if quality is not None:
        q = float(quality.get("score", 1.0))
        if q < 0.55:
            flags = quality.get("flags", [])
            return {
                "title": _msg(
                    lang,
                    "解析の信頼度が低い（要確認）",
                    "Low confidence (please confirm)",
                    "识别可信度偏低（请先确认点位）",
                ),
                "cue": _msg(
                    lang,
                    "次のエンドの前に、紫の点を必ず手動で修正してから解析して。写真の傾き/反射/距離が原因のことが多い。",
                    "Before the next end: manually correct the purple points, then analyze again. Tilt/glare/distance often cause this.",
                    "下一靶前：请先手动修正紫色点位再分析。常见原因是拍摄倾斜/反光/距离不合适。",
                ),
                "why": _msg(
                    lang,
                    f"検出フラグ: {flags}",
                    f"Detection flags: {flags}",
                    f"检测标记: {flags}",
                ),
                "tag": "low_confidence",
            }

    # helper thresholds (ratios are relative to outer radius)
    # ~ one ring band is ~0.10 of outer radius (outer/10)
    offset_is_significant = (offset_ratio is not None and float(offset_ratio) >= 0.085) or (
        offset_ratio is None and (abs(dx) + abs(dy)) > 0.0 and ((dx * dx + dy * dy) ** 0.5) > 34.0
    )
    loose_is_significant = (spread_ratio is not None and float(spread_ratio) >= 0.075) or (
        spread_ratio is None and spread > max(12.0, 0.9 * (sx + sy))
    )

    # --------------------------
    # 2) Systematic offset
    # --------------------------
    if offset_is_significant:
        direction = _dir_from_dxdy(dx, dy)

        dir_text = {
            "left": _msg(lang, "左", "left", "左"),
            "right": _msg(lang, "右", "right", "右"),
            "up": _msg(lang, "上", "up", "上"),
            "down": _msg(lang, "下", "down", "下"),
            "up_left": _msg(lang, "左上", "up-left", "左上"),
            "up_right": _msg(lang, "右上", "up-right", "右上"),
            "down_left": _msg(lang, "左下", "down-left", "左下"),
            "down_right": _msg(lang, "右下", "down-right", "右下"),
            "center": _msg(lang, "中心", "center", "中心"),
        }.get(direction, _msg(lang, "特定不可", "unknown", "不确定"))

        return {
            "title": _msg(
                lang,
                "まとまっているが“中心からズレている”",
                "Grouped, but shifted from center",
                "组还可以，但“整体偏离靶心”",
            ),
            "cue": _msg(
                lang,
                f"次のエンドは『ズレの補正』を優先。全体が {dir_text} に寄っているなら、狙い/照準の基準を少しだけ調整（または同じアンカーを徹底）。",
                f"Next end: fix the shift. If the whole group sits {dir_text}, make a small sight/aim reference adjustment (or enforce identical anchor).",
                f"下一靶先修正“整体偏移”。如果整体偏向{dir_text}，请做小幅瞄准/照门基准调整（或更严格保持同一锚点）。",
            ),
            "why": _msg(
                lang,
                "形が良いのに中心からズレるのは“再現性はあるが基準がズレている”サイン。まず中心に戻すと伸びが早い。",
                "A tight group that’s off-center means repeatability is there but the reference is shifted. Re-center first for fast gains.",
                "组型还紧但整体偏心，通常代表“重复性有了，但基准偏了”。先拉回靶心，提升最快。",
            ),
            "tag": "offset",
            "debug": {"dx": dx, "dy": dy, "offset_ratio": offset_ratio},
        }

    # --------------------------
    # 3) Loose dispersion (random)
    # --------------------------
    if loose_is_significant:
        return {
            "title": _msg(lang, "全体的にまとまりが弱い", "Group is loose overall", "整体组偏散"),
            "cue": _msg(
                lang,
                "次のエンドは『力を引く』だけ。前肩・前腕・指先の緊張を落として、同じテンポで打つ。",
                "Next end: subtract tension. Relax front shoulder/forearm/fingers and shoot with the same tempo.",
                "下一靶只做一件事：减力。放松前肩/前臂/指尖，用同样节奏完成整套动作。",
            ),
            "why": _msg(
                lang,
                "力みは微修正を増やして散りやすい。構造に乗せるほど軽く安定する。",
                "Extra tension increases micro-corrections and wobble. The more you stack structure, the steadier it feels.",
                "用力会增加微调整与抖动。越“上结构”，就越轻松稳定。",
            ),
            "tag": "tension",
        }

    # --------------------------
    # 4) Directional dispersion
    # --------------------------
    if shape == "horizontal":
        return {
            "title": _msg(lang, "横に散っている", "Wide left–right spread", "左右散布偏大"),
            "cue": _msg(
                lang,
                "次のエンドは『弦の垂直感』だけ守る。引き上げ〜フルドローまで、弦が指の中でねじれないように。",
                "Next end: protect string verticality. From set-up to full draw, keep the string from twisting in the fingers.",
                "下一靶只守住“弦的垂直感”。从举弓到满弓，避免手指里把弦扭转。",
            ),
            "why": _msg(
                lang,
                "横散りはトルク（ねじれ）や前手側の一貫性不足で出やすい。まずねじれを減らすと締まりやすい。",
                "Horizontal spread often comes from torque/twist or inconsistent bow-hand pressure. Reduce twist first to tighten the group.",
                "左右散布常见原因是扭矩/扭弦或弓手压力不一致。先减小扭转，组会立刻更紧。",
            ),
            "tag": "torque",
        }

    if shape == "vertical":
        return {
            "title": _msg(lang, "縦に散っている", "Tall up–down spread", "上下散布偏大"),
            "cue": _msg(
                lang,
                "次のエンドは『掛け方を同じ』に固定。指の位置と圧の配分を毎回同じにして、抜ける方向を揃える。",
                "Next end: lock your hook. Same finger placement and pressure each time so the release direction stays consistent.",
                "下一靶固定“挂弦”。每次手指位置与受力比例一致，让放箭方向一致。",
            ),
            "why": _msg(
                lang,
                "縦散りはリリースの上下ブレ（掛け方・抜け方の違い）が原因になりやすい。まず再現性を上げる。",
                "Vertical spread often comes from release-height variation driven by inconsistent hook/exit. Raise repeatability first.",
                "上下散布多由挂弦/离弦不一致导致的放箭高度变化。先把重复性提上来。",
            ),
            "tag": "hook",
        }

    # --------------------------
    # 5) Otherwise: rhythm
    # --------------------------
    return {
        "title": _msg(lang, "まとまりはある（次は微調整）", "Group is decent (fine-tune)", "组还不错（做微调）"),
        "cue": _msg(
            lang,
            "次のエンドは『リズム優先』。狙いに居座らず、同じ準備・同じ呼吸・同じタイミングで抜く。",
            "Next end: rhythm first. Don’t camp on aim—same prep, same breath, same timing through the click/release.",
            "下一靶以节奏为王。不要在瞄准上停太久：同样准备、同样呼吸、同样时机完成放箭。",
        ),
        "why": _msg(
            lang,
            "良い日は“頑張らない”ほど当たることがある。再現性の鍵はテンポ。",
            "On good days, accuracy comes from not forcing it. Tempo is the repeatability lever.",
            "状态好时往往是“不用力反而更准”。重复性的杠杆在节奏。",
        ),
        "tag": "rhythm",
    }
