from __future__ import annotations


def _msg(lang: str, ja: str, en: str, zh: str) -> str:
    if lang == "ja":
        return ja
    if lang == "zh":
        return zh
    return en


def next_end_advice(metrics: dict, shape: str, handedness: str, lang: str = "en") -> dict:
    sx, sy, spread = metrics["sx"], metrics["sy"], metrics["spread"]
    loose = spread > max(12.0, 0.9 * (sx + sy))

    if shape == "horizontal":
        return {
            "title": _msg(
                lang,
                "横に散っている",
                "Wide left–right spread",
                "左右散布偏大",
            ),
            "cue": _msg(
                lang,
                "次のエンドは「弦の垂直感」だけ守る。引き上げ〜フルドローまで、弦が指の中でねじれないように。",
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
            "title": _msg(
                lang,
                "縦に散っている",
                "Tall up–down spread",
                "上下散布偏大",
            ),
            "cue": _msg(
                lang,
                "次のエンドは「掛け方を同じ」に固定。指の位置と圧の配分を毎回同じにして、抜ける方向を揃える。",
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

    if loose:
        return {
            "title": _msg(
                lang,
                "全体的にまとまりが弱い",
                "Group is loose overall",
                "整体组偏散",
            ),
            "cue": _msg(
                lang,
                "次のエンドは「余計な力を抜く」だけ。前肩・前腕・指先の緊張を落として、同じテンポで打つ。",
                "Next end: subtract tension. Relax front shoulder/forearm/fingers and shoot with the same tempo.",
                "下一靶只做一件事：减力。放松前肩/前臂/指尖，用同样节奏完成整套动作。",
            ),
            "why": _msg(
                lang,
                "力みは筋肉同士の“ぶつかり合い”を増やして微ブレが出る。構造に乗せるほど軽く安定する。",
                "Extra tension increases micro-corrections and wobble. The more you stack structure, the lighter and steadier it feels.",
                "用力会增加肌肉对抗与微调整，带来抖动。越“上结构”，就越轻松稳定。",
            ),
            "tag": "tension",
        }

    return {
        "title": _msg(
            lang,
            "まとまりはある（次は微調整）",
            "Group is decent (fine-tune)",
            "组还不错（做微调）",
        ),
        "cue": _msg(
            lang,
            "次のエンドは「リズム優先」。狙いに居座らず、同じ準備・同じ呼吸・同じタイミングで抜く。",
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
