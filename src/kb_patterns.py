from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math
import re


@dataclass
class PatternItem:
    key: str
    query: str
    texts: Dict[str, str]


def _tok(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    return [w for w in s.split() if len(w) >= 2]


def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for k, va in a.items():
        na += va * va
        dot += va * b.get(k, 0.0)
    for vb in b.values():
        nb += vb * vb
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / math.sqrt(na * nb)


def _bow_vec(tokens: List[str]) -> Dict[str, float]:
    stop = {"the", "and", "to", "of", "in", "a", "for", "on", "with", "is", "are", "be", "your"}
    v: Dict[str, float] = {}
    for w in tokens:
        if w in stop:
            continue
        v[w] = v.get(w, 0.0) + 1.0
    for k in list(v.keys()):
        v[k] = 1.0 + math.log(1.0 + v[k])
    return v


# 说明：这里不是逐字引用书，而是把书里反复出现的“观察 → 归因 → 行动”训练思路做成可检索 pattern。
KB: List[PatternItem] = [
    PatternItem(
        key="repeat_same_shot",
        query="repeat same shot process same setup routine rhythm boring repeatability",
        texts={
            "en": "Make it boring: repeat the same setup and execution. Don’t ‘try harder’—repeat better.",
            "ja": "退屈でいい：同じセットアップと動作を繰り返す。「頑張る」より「再現する」。",
            "zh": "越无聊越有效：重复同一套动作。别“更用力”，要“更可重复”。",
        },
    ),
    PatternItem(
        key="one_change_rule",
        query="one change at a time isolate variable avoid changing many things",
        texts={
            "en": "Change one thing at a time. If you change two, you learn nothing.",
            "ja": "変えるのは1つだけ。2つ変えたら、何が効いたか分からない。",
            "zh": "一次只改一件事。两件一起改，你学不到原因。",
        },
    ),
    PatternItem(
        key="short_distance_big_group_clearance",
        query="group larger at short distance clearance string chest arm contact rest button",
        texts={
            "en": "If groups blow up at short distance, suspect clearance first: string on chest/arm or fletch catching rest/button.",
            "ja": "短距離で散るなら、まずクリアランス疑い。弦が胸/腕に当たる、羽がレスト/ボタンに触れる。",
            "zh": "短距离更散，先怀疑清弦：弦碰胸/护臂，或羽碰rest/button。",
        },
    ),
    PatternItem(
        key="long_distance_big_group_speed_drift",
        query="group larger at long distance arrow slowing drift point weight foc fletch helical",
        texts={
            "en": "If long distance opens up disproportionately, think speed/drift: check point weight (FOC) and fletch setup that bleeds speed.",
            "ja": "長距離で急に散るなら失速/ドリフト。ポイント重量(FOC)と効きすぎる羽角で速度を落としていないか。",
            "zh": "长距离散得多，考虑失速/飘：检查点重/FOC，以及羽角太大导致掉速。",
        },
    ),
    PatternItem(
        key="same_shot_all_distances_hip_tilt",
        query="same shot all distances hip tilt keep shoulder arm torso relationship",
        texts={
            "en": "Don’t learn a different shot per distance. Keep shoulder–arm–torso relationships; adjust elevation mainly from the hips.",
            "ja": "距離ごとに別の動作を作らない。肩・腕・体幹は一定で、主に腰（ヒップ）から傾ける。",
            "zh": "别把不同距离练成不同动作。保持肩-臂-躯干关系不变，主要用髋部倾斜调仰角。",
        },
    ),
    PatternItem(
        key="log_patterns",
        query="training log record changes conditions find patterns faster than memory",
        texts={
            "en": "Keep a simple log (distance, hits, what you changed, fatigue). Patterns show up faster than ‘memory’.",
            "ja": "簡単なログを残す（距離、着弾、変えたこと、疲労）。記憶より早くパターンが見える。",
            "zh": "保持简单日志（距离、落点、改了什么、疲劳）。比靠记忆更快看出规律。",
        },
    ),
    PatternItem(
        key="wind_reading",
        query="wind steady move sight chaotic wind observe flags short cycles",
        texts={
            "en": "In steady wind, moving sight can be cleaner. In chaotic wind, rely on flags and shorter observation cycles.",
            "ja": "一定の風ならサイト移動で保持を揃える。乱れる風は旗を見て、短い観察サイクルで読む。",
            "zh": "稳定风可直接移动瞄准。风乱就看旗，缩短观察周期再出箭。",
        },
    ),
    PatternItem(
        key="nock_fit_consistency",
        query="nock fit consistency affects grouping tune recheck after change",
        texts={
            "en": "Nock fit consistency matters. If you change nocks, recheck grouping/tune.",
            "ja": "ノックのフィットは重要。変えたらグループ/チューンを再確認。",
            "zh": "卡扣一致性很关键。换了卡扣要复查分组/调性。",
        },
    ),
    PatternItem(
        key="front_side_clean",
        query="front side clean bow shoulder stable no collapse alignment bone to bone",
        texts={
            "en": "Aim for a ‘quiet’ front side: stable bow shoulder, no collapse, bone alignment doing the holding.",
            "ja": "フロント側を静かに：弓肩は安定、潰れない。骨格アライメントで支える。",
            "zh": "让前侧“安静”：弓肩稳定、不塌。靠骨架对齐在撑，而不是靠硬顶。",
        },
    ),
]


def retrieve_patterns(context: str, k: int = 3) -> List[Tuple[PatternItem, float]]:
    qv = _bow_vec(_tok(context))
    scored: List[Tuple[PatternItem, float]] = []
    for item in KB:
        iv = _bow_vec(_tok(item.query))
        scored.append((item, _cos(qv, iv)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
