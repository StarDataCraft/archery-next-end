from __future__ import annotations

def next_end_advice(metrics: dict, shape: str, handedness: str) -> dict:
    """
    handedness: right / left (影响左右方向的措辞你后面可加)
    输出尽量像教练口令：短、可执行、只一件事。
    """
    sx, sy, spread = metrics["sx"], metrics["sy"], metrics["spread"]

    # 经验阈值：你后续可以让用户自己校准（比如用“近期最佳spread”做相对阈值）
    loose = spread > max(12.0, 0.9 * (sx + sy))  # 粗略判断“松”

    if shape == "horizontal":
        return {
            "title": "横に散っている",
            "cue": "次のエンドは「弦の垂直感」だけ守る。引き上げ〜フルドローまで、弦が指の中でねじれないように。",
            "why": "横散りはトルク（ねじれ）や前手側の一貫性不足で出やすい。まずねじれを減らすと一気に締まることが多い。",
            "tag": "torque"
        }

    if shape == "vertical":
        return {
            "title": "縦に散っている",
            "cue": "次のエンドは「掛け方を同じ」に固定。指の位置と圧の配分を毎回同じにして、抜ける方向を揃える。",
            "why": "縦散りはリリースの上下ブレ（掛け方・抜け方の違い）が原因になりやすい。まず再現性を上げる。",
            "tag": "hook"
        }

    if loose:
        return {
            "title": "全体的にまとまりが弱い",
            "cue": "次のエンドは「余計な力を抜く」だけ。前肩・前腕・指先の緊張を落として、同じテンポで打つ。",
            "why": "力みは筋肉同士の“ぶつかり合い”を増やして微ブレが出る。構造に乗せるほど軽く安定する。",
            "tag": "tension"
        }

    return {
        "title": "まとまりはある（次は微調整）",
        "cue": "次のエンドは「リズム優先」。狙いに居座らず、同じ準備・同じ呼吸・同じタイミングで抜く。",
        "why": "良い日は“頑張らない”ほど当たることがある。再現性の鍵はテンポ。",
        "tag": "rhythm"
    }
