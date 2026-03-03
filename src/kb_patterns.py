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
        vb = b.get(k, 0.0)
        dot += va * vb
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


# 说明：
# 这些条目是把书中反复强调的“可复现的训练/诊断模式”抽象成可检索的 pattern。
KB: List[PatternItem] = [
    PatternItem(
        key="repeatable_shot_over_fancy",
        query="repeatable shot boring repeat process not chasing",
        texts={
            "en": "Make the shot repeatable before you make it fancy. If you can’t repeat it, you can’t trust the feedback.",
            "ja": "派手さより再現性。繰り返せない動作は、フィードバックの信用が落ちる。",
            "zh": "先要可重复，再谈高级。做不到重复，反馈就不可信。",
        },
    ),
    PatternItem(
        key="log_to_find_patterns",
        query="training log record patterns distance changes what changed",
        texts={
            "en": "Keep a simple log (distance, notes, what changed). Patterns show up faster than memory.",
            "ja": "距離・メモ・変えた点をログ化。記憶より早くパターンが見える。",
            "zh": "做简单日志（距离、备注、改了什么）。比靠记忆更快看规律。",
        },
    ),
    PatternItem(
        key="clearance_first_short_distance",
        query="short distance group big clearance string chest arm rest button contact",
        texts={
            "en": "If groups blow up at short distance, suspect clearance first (string/arm/chest, fletch hitting rest/button). Fix clearance before tune-chasing.",
            "ja": "短距離で急に散るなら、まずクリアランス（弦/腕/胸、羽がレスト/ボタン接触）を疑う。チューン前に解決。",
            "zh": "短距离反而更散，先查清弦（弦/护臂/胸）与羽碰rest/button。先修清弦再调弓。",
        },
    ),
    PatternItem(
        key="long_distance_drift",
        query="long distance group grows drift slow down arrow speed foc point weight fletch",
        texts={
            "en": "If the group grows disproportionately at long distance, think drift/slowdown: point weight (FOC) and ‘too much stabilizing’ fletch can cost speed.",
            "ja": "長距離でだけ散るなら失速/ドリフト。ポイント重量(FOC)や効きすぎる羽設定で速度を落としていないか。",
            "zh": "长距离才散，多半是失速/飘：点重与FOC、羽角/羽太“稳但掉速”要检查。",
        },
    ),
    PatternItem(
        key="same_shot_all_distances",
        query="same shot all distances hip tilt keep shoulder arm torso relationship",
        texts={
            "en": "Don’t learn a different shot per distance. Keep shoulder–arm–torso relationship; adjust elevation mainly by tilting from the hips.",
            "ja": "距離ごとに別フォームにしない。肩・腕・体幹の関係は固定し、角度は主に腰（ヒップ）から作る。",
            "zh": "不要不同距离练成不同动作。肩-臂-躯干关系固定，主要用髋部倾斜调整仰角。",
        },
    ),
    PatternItem(
        key="wind_strategy",
        query="wind steady move sight chaotic observe flags short cycles",
        texts={
            "en": "In steady wind, moving sight can be cleaner than forcing a hold. In chaotic wind, read flags and shoot in short observation cycles.",
            "ja": "一定風ならサイト移動で保持を揃える方が楽。乱風なら旗を見て短い観察サイクルで撃つ。",
            "zh": "稳定风可用移动瞄准替代硬扛。乱风看旗，缩短观察周期再出箭。",
        },
    ),
    PatternItem(
        key="nock_consistency",
        query="nock fit consistency changes grouping tune even color",
        texts={
            "en": "Nock fit consistency matters. Even changing nock type/color can shift grouping—keep it consistent and recheck when you change.",
            "ja": "ノックのフィットは重要。種類/色変更でもグルーピングが変わることがある。揃えて、変えたら再確認。",
            "zh": "卡扣一致性很重要。换型号/颜色都可能影响分组，尽量统一；换了就复查。",
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
