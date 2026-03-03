# src/rules.py
from __future__ import annotations

from typing import Dict, Any, Optional


def _msg(lang: str, ja: str, en: str, zh: str) -> str:
    if lang == "ja":
        return ja
    if lang == "zh":
        return zh
    return en


def _script_block(lang: str, lines_ja: list[str], lines_en: list[str], lines_zh: list[str]) -> str:
    if lang == "ja":
        return "\n".join(lines_ja)
    if lang == "zh":
        return "\n".join(lines_zh)
    return "\n".join(lines_en)


def _drill_library(tag: str, lang: str) -> Dict[str, Any]:
    """
    Training micro-drills: 30–120s, built for repeatability.
    Each drill must be executable immediately on the line.
    """
    if tag == "torque":
        return {
            "name": _msg(
                lang,
                "30秒：弦の垂直感ドリル",
                "30s: String verticality drill",
                "30秒：弦垂直感练习",
            ),
            "how": _script_block(
                lang,
                [
                    "1) セットアップで弦がサイトピン/リムに対して垂直に見える位置を作る。",
                    "2) 引き分け中、指の中で弦を回さない（“ねじらない”）。",
                    "3) フルドローで1秒止め、弦の線が同じ位置にあるかだけ確認。×6回（撃たなくてOK）",
                ],
                [
                    "1) At set-up, build a reference where the string looks vertical vs sight/window.",
                    "2) During the draw, do NOT rotate the string inside the fingers.",
                    "3) At full draw, hold 1s and check the string line is in the same place. x6 reps (no shot needed).",
                ],
                [
                    "1) 在举弓前建立“弦看起来是竖直的”参考线（对准瞄具/弓窗）。",
                    "2) 拉弓过程中不在手指里“拧弦”。",
                    "3) 到满弓停1秒，只检查弦线是否回到同一位置。做6次（不放箭也可以）。",
                ],
            ),
            "duration_s": 30,
        }

    if tag == "hook":
        return {
            "name": _msg(
                lang,
                "45秒：掛け方固定ドリル",
                "45s: Lock your hook drill",
                "45秒：挂弦固定练习",
            ),
            "how": _script_block(
                lang,
                [
                    "1) 指の第1関節ラインを基準に、毎回同じ深さで掛ける。",
                    "2) 圧の配分（人差し指:中指:薬指）を“同じ比率”にする。",
                    "3) フルドローで1秒、指先の圧が同じかだけ確認。×5回",
                ],
                [
                    "1) Use the first-joint crease as a depth reference: same hook depth every time.",
                    "2) Keep the pressure ratio (index/middle/ring) the same each rep.",
                    "3) At full draw, hold 1s and check finger pressure feels identical. x5 reps",
                ],
                [
                    "1) 用第一指节的折痕做基准：每次挂弦深度一致。",
                    "2) 三指受力比例保持一致（食/中/无名指）。",
                    "3) 满弓停1秒，只检查指尖受力是否“同一种感觉”。做5次。",
                ],
            ),
            "duration_s": 45,
        }

    if tag == "subtract_tension":
        return {
            "name": _msg(
                lang,
                "60秒：减力+节奏重置",
                "60s: Subtract tension + tempo reset",
                "60秒：减力+节奏重置",
            ),
            "how": _script_block(
                lang,
                [
                    "1) いまの引きで“2割軽く”感じるテンションまで落とす（前肩/前腕/指先）。",
                    "2) 3秒以内に決まらなければ一度下ろす（必ず）。",
                    "3) 同じテンポで×6射（結果よりテンポ一致）。",
                ],
                [
                    "1) Reduce perceived effort by ~20% (front shoulder/forearm/fingers).",
                    "2) If it’s not settling within 3 seconds, LET DOWN (always).",
                    "3) Shoot x6 with identical tempo; ignore score, protect rhythm.",
                ],
                [
                    "1) 主观用力降低约2成（前肩/前臂/指尖都要松下来）。",
                    "2) 瞄准3秒内没稳定就放下重来（必须）。",
                    "3) 连续6箭只追求“同样节奏”，先不追分。",
                ],
            ),
            "duration_s": 60,
        }

    if tag == "aim_reference":
        return {
            "name": _msg(
                lang,
                "45秒：瞄准参考点校准",
                "45s: Aim reference calibration",
                "45秒：瞄准参考点校准",
            ),
            "how": _script_block(
                lang,
                [
                    "1) セットアップで肩・頭・視線の位置を固定。",
                    "2) アンカーで“鼻/唇/顎”の接触点を同じにする。",
                    "3) 1秒止めてサイトリング内の見え方を確認。×5回（撃たなくてOK）",
                ],
                [
                    "1) Fix shoulder/head/eye line at set-up.",
                    "2) At anchor, make nose/lip/chin contacts identical.",
                    "3) Hold 1s and confirm sight picture is the same. x5 reps (no shot needed).",
                ],
                [
                    "1) 举弓时肩/头/视线位置固定。",
                    "2) 锚点处鼻/唇/下巴接触点保持一致。",
                    "3) 停1秒确认瞄准画面一致。做5次（不放箭也可以）。",
                ],
            ),
            "duration_s": 45,
        }

    # default
    return {
        "name": _msg(lang, "30秒：基本反復", "30s: Basic repetition", "30秒：基础重复"),
        "how": _script_block(
            lang,
            ["同じ合図で、同じテンポで×5回。"],
            ["Same cue, same tempo x5 reps."],
            ["同一句口令、同一节奏做5次。"],
        ),
        "duration_s": 30,
    }


def _build_advice(
    *,
    lang: str,
    tag: str,
    title_ja: str,
    title_en: str,
    title_zh: str,
    single_cue_ja: str,
    single_cue_en: str,
    single_cue_zh: str,
    pass_fail_ja: str,
    pass_fail_en: str,
    pass_fail_zh: str,
    fallback_ja: str,
    fallback_en: str,
    fallback_zh: str,
    why_ja: str,
    why_en: str,
    why_zh: str,
    stage: str,
) -> Dict[str, Any]:
    drill = _drill_library(tag, lang)
    mental_phrase = _msg(
        lang,
        "「同じことを、同じように」",
        '"Same thing. Same way."',
        "“同一件事，用同一种方式。”",
    )

    # “Shot Script”: fixed structure for repetition (book-aligned)
    script = _script_block(
        lang,
        [
            "【このエンドの一言】" + single_cue_ja,
            "【合格/失敗】" + pass_fail_ja,
            "【崩れたら】" + fallback_ja,
            "【すぐやる反復】" + drill["name"],
        ],
        [
            "[One cue this end] " + single_cue_en,
            "[PASS/FAIL] " + pass_fail_en,
            "[If it breaks] " + fallback_en,
            "[Immediate repetition] " + drill["name"],
        ],
        [
            "【本靶唯一口令】" + single_cue_zh,
            "【通过/失败】" + pass_fail_zh,
            "【崩了就回滚】" + fallback_zh,
            "【立刻做的重复练习】" + drill["name"],
        ],
    )

    return {
        # keep backwards compatibility
        "title": _msg(lang, title_ja, title_en, title_zh),
        "cue": _msg(lang, single_cue_ja, single_cue_en, single_cue_zh),
        "why": _msg(lang, why_ja, why_en, why_zh),
        "tag": tag,
        # new coaching fields
        "stage": stage,  # which part of shot process to “protect”
        "single_cue": _msg(lang, single_cue_ja, single_cue_en, single_cue_zh),
        "pass_fail": _msg(lang, pass_fail_ja, pass_fail_en, pass_fail_zh),
        "fallback": _msg(lang, fallback_ja, fallback_en, fallback_zh),
        "drill": drill,
        "mental_phrase": mental_phrase,
        "script": script,
        # optional: invariant reminder (stable across sessions)
        "principle": _msg(
            lang,
            "命中は才能より反復。",
            "Hits come from repetition, not inspiration.",
            "命中更多来自重复，而不是灵光一现。",
        ),
    }


def next_end_advice(
    metrics: dict,
    shape: str,
    handedness: str,
    lang: str = "en",
    quality: Optional[dict] = None,
) -> dict:
    sx, sy, spread = metrics["sx"], metrics["sy"], metrics["spread"]
    loose = spread > max(12.0, 0.9 * (sx + sy))

    # ---- 1) Horizontal spread: torque / string twist ----
    if shape == "horizontal":
        return _build_advice(
            lang=lang,
            tag="torque",
            title_ja="横に散っている",
            title_en="Wide left–right spread",
            title_zh="左右散布偏大",
            single_cue_ja="弦は“まっすぐ”。指の中で弦を回さない。",
            single_cue_en="String stays vertical. Don’t rotate it inside the fingers.",
            single_cue_zh="弦保持“垂直”。不要在手指里拧弦。",
            pass_fail_ja="フルドローで弦の線が毎回同じ位置に戻る＝PASS。戻らない/毎回違う＝FAIL。",
            pass_fail_en="PASS if the string line returns to the same place at full draw each rep. FAIL if it shifts.",
            pass_fail_zh="满弓时弦线每次回到同一位置＝通过；弦线位置每次不同＝失败。",
            fallback_ja="3秒で落ち着かなければ必ず下ろす。次は“弦だけ”を見る。",
            fallback_en="If it won’t settle in 3 seconds, always let down. Next rep: only watch the string line.",
            fallback_zh="3秒内不稳定就放下重来。下一次只盯“弦线”。",
            why_ja="横散りはトルク（ねじれ）や前手圧のブレで出やすい。まず“ねじれゼロ”を反復すると締まりやすい。",
            why_en="Horizontal spread is often torque/string twist or inconsistent bow-hand pressure. Repeating ‘zero twist’ tightens groups fast.",
            why_zh="左右散布常见原因是扭矩/拧弦或弓手压力不一致。把“零扭转”反复做到位，组会立刻收紧。",
            stage="setup→draw→full_draw",
        )

    # ---- 2) Vertical spread: hook/release height variability ----
    if shape == "vertical":
        return _build_advice(
            lang=lang,
            tag="hook",
            title_ja="縦に散っている",
            title_en="Tall up–down spread",
            title_zh="上下散布偏大",
            single_cue_ja="掛け方を固定。指の深さと圧の比率を同じに。",
            single_cue_en="Lock the hook. Same depth, same pressure ratio every time.",
            single_cue_zh="挂弦固定：深度一致、受力比例一致。",
            pass_fail_ja="アンカーで指先の圧が“同じ感覚”＝PASS。毎回違う/抜けが変わる＝FAIL。",
            pass_fail_en="PASS if finger pressure feels identical at anchor. FAIL if it changes or release direction varies.",
            pass_fail_zh="锚点处指尖受力“同一种感觉”＝通过；每次不一样/离弦方向变＝失败。",
            fallback_ja="撃つ前に1回だけ素引き（引いて戻す）して掛け方を作り直す。",
            fallback_en="Before shooting, do one blank-draw (draw and let down) to rebuild the hook.",
            fallback_zh="放箭前先做一次“空拉放下”，重新建立挂弦感觉。",
            why_ja="縦散りはリリースの上下ブレ（掛け方・抜け方の差）で起きやすい。まず再現性を反復で作る。",
            why_en="Vertical spread often comes from release-height variation driven by inconsistent hook/exit. Build repeatability through reps.",
            why_zh="上下散布多由挂弦/离弦不一致导致的放箭高度变化。先用重复把一致性练出来。",
            stage="hook→anchor→release",
        )

    # ---- 3) Loose group overall: too much tension, tempo drift ----
    if loose:
        return _build_advice(
            lang=lang,
            tag="subtract_tension",
            title_ja="全体的にまとまりが弱い",
            title_en="Group is loose overall",
            title_zh="整体组偏散",
            single_cue_ja="余計な力を引く。前肩・前腕・指先を軽く。",
            single_cue_en="Subtract tension. Lighten front shoulder/forearm/fingers.",
            single_cue_zh="减力：前肩/前臂/指尖都要松。",
            pass_fail_ja="出た後、弓が自然に前へ落ちる＝PASS。握る/止める＝FAIL。",
            pass_fail_en="PASS if the bow naturally moves forward after the shot. FAIL if you grab/stop it.",
            pass_fail_zh="出箭后弓自然前走/前落＝通过；手去抓/去停住＝失败。",
            fallback_ja="距離/狙いは捨ててテンポ一致だけ守る（6射）。",
            fallback_en="Drop score expectations. Protect identical tempo for 6 shots.",
            fallback_zh="先不追分，只守住“同样节奏”打6箭。",
            why_ja="力みは微調整を増やして散りやすい。構造に乗せるほど軽く、同じ動きが反復できる。",
            why_en="Extra tension creates micro-corrections and wobble. The more you rely on structure, the lighter—and more repeatable—you get.",
            why_zh="用力会带来大量微调与抖动，让组变散。越是依靠结构，越轻松，越能重复同一动作。",
            stage="whole_shot_tempo",
        )

    # ---- 4) Otherwise: give a stable “aim reference” anchor reminder ----
    return _build_advice(
        lang=lang,
        tag="aim_reference",
        title_ja="全体は悪くない。基準を揃える",
        title_en="Not bad. Normalize your reference",
        title_zh="整体不错：统一基准",
        single_cue_ja="基準（アンカー/見え方）を毎回同じに。",
        single_cue_en="Make the reference identical (anchor + sight picture).",
        single_cue_zh="把基准统一（锚点+瞄准画面每次一致）。",
        pass_fail_ja="鼻/唇/顎の接触とサイト内の見え方が同じ＝PASS。毎回違う＝FAIL。",
        pass_fail_en="PASS if nose/lip/chin contacts and sight picture match each rep. FAIL if they vary.",
        pass_fail_zh="鼻/唇/下巴接触点与瞄准画面每次一致＝通过；每次不同＝失败。",
        fallback_ja="決まらなければ下ろす。まず“同じ見え方”を作り直す。",
        fallback_en="If it doesn’t set, let down. Rebuild the same sight picture first.",
        fallback_zh="如果画面不对就放下重来：先把“同样画面”建立出来。",
        why_ja="散りが少ない日は“基準の反復”が効く。小さな差を潰すほど安定する。",
        why_en="On better days, repeating the same reference removes small variance and stabilizes the group.",
        why_zh="状态不错时，强化“基准的重复”最有效。越能消除微小差异，越稳定。",
        stage="anchor→aim",
    )
