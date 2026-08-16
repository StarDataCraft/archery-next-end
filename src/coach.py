from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CoachConfig:
    pdf_path: str = "docs/Archery The Art of Repetition (Simon Needham ).pdf"
    cache_dir: str = ".cache/coach"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_chars: int = 900
    chunk_overlap: int = 140
    top_k: int = 6

    # ``rag`` is retained as an alias for old saved sessions. The default
    # ``book`` mode needs no model download on Streamlit Cloud.
    mode: str = "book"  # "book" | "rules" | legacy "rag"
    gguf_path: str = "models/llm.gguf"
    llm_ctx: int = 2048
    llm_max_tokens: int = 420
    llm_temperature: float = 0.3
    router: str = "fine"


def _text(lang: str, values: Dict[str, str]) -> str:
    return values.get(lang, values.get("en", ""))


CHAPTERS = {
    "biomechanics": {
        "ja": "第5章 射法のバイオメカニクス",
        "en": "Chapter 5 · The Biomechanics of Shooting",
        "zh": "第5章 · 射箭动作的生物力学",
    },
    "training": {
        "ja": "第7章 試合・トレーニング・練習",
        "en": "Chapter 7 · Competitions, Training and Practice",
        "zh": "第7章 · 比赛、训练与练习",
    },
    "mind": {
        "ja": "第9章 心を最大限に活かす",
        "en": "Chapter 9 · Making the Most of Your Mind",
        "zh": "第9章 · 充分运用心理",
    },
    "better": {
        "ja": "第12章 より良い射ち方",
        "en": "Chapter 12 · Better Shooting",
        "zh": "第12章 · 更好的射箭",
    },
}


def _card(
    *,
    source_id: str,
    chapter: str,
    pages: str,
    title: Dict[str, str],
    cue: Dict[str, str],
    pass_fail: Dict[str, str],
    fallback: Dict[str, str],
    why: Dict[str, str],
    summary: Dict[str, str],
    drill_name: Dict[str, str],
    drill_how: Dict[str, str],
    duration_s: int,
    stage: str,
    mental: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "id": source_id,
        "chapter": chapter,
        "pages": pages,
        "title": title,
        "cue": cue,
        "pass_fail": pass_fail,
        "fallback": fallback,
        "why": why,
        "summary": summary,
        "drill_name": drill_name,
        "drill_how": drill_how,
        "duration_s": duration_s,
        "stage": stage,
        "mental": mental,
    }


# Short coaching cards distilled from the supplied book. They deliberately
# paraphrase the source rather than copying passages. Page numbers below are
# PDF page numbers so users can verify the grounding quickly.
BOOK_CARDS: Dict[str, Dict[str, Any]] = {
    "string_vertical": _card(
        source_id="string_vertical",
        chapter="biomechanics",
        pages="166",
        title={"ja": "横散り：まず弦のねじれを消す", "en": "Horizontal spread: remove string torque", "zh": "横向散布：先消除弦的扭转"},
        cue={"ja": "引き始めから離れまで、指の中の弦を縦のまま保つ。", "en": "Keep the string vertical in the fingers from draw to release.", "zh": "从起拉到离弦，让手指里的弦始终保持竖直。"},
        pass_fail={"ja": "PASS：フルドローの弦像が毎回同じ。FAIL：弦像やライザーの傾きが変わる。", "en": "PASS: the string picture is identical at full draw. FAIL: its angle or the riser tilt changes.", "zh": "通过：每次满弓的弦像一致；失败：弦线角度或弓把倾斜发生变化。"},
        fallback={"ja": "サイトで補正しない。下ろして、弦の向きだけを作り直す。", "en": "Do not compensate with the sight. Let down and rebuild only the string direction.", "zh": "不要用瞄具补偿；放下重来，只重建弦的方向。"},
        why={"ja": "横方向のばらつきが強く、弦のトルクを先に切り分ける価値があります。", "en": "The group varies mainly left-to-right, so string torque is the first variable worth isolating.", "zh": "箭群主要在左右方向变化，因此最值得先排查的是弦的扭转。"},
        summary={"ja": "弦の縦方向が変わるとライザーが回り、横方向の振動とグループ拡大につながる、という説明。", "en": "The book links changing string verticality with riser rotation, horizontal oscillation and larger groups.", "zh": "书中把弦线竖直方向的变化，与弓把旋转、横向摆动和箭群扩大联系起来。"},
        drill_name={"ja": "40秒・弦像6回", "en": "40s · Six string-picture reps", "zh": "40秒 · 6次弦像练习"},
        drill_how={"ja": "矢を射たずに6回引き、アンカーで弦像を1秒確認して下ろす。", "en": "Without shooting, draw six times; hold one second at anchor to check the string picture, then let down.", "zh": "不放箭空拉6次；到锚点停1秒检查弦像，然后放下。"},
        duration_s=40,
        stage="hook→draw→release",
        mental={"ja": "「縦のまま」", "en": "Stay vertical.", "zh": "“保持竖直。”"},
    ),
    "bow_hand_relaxed": _card(
        source_id="bow_hand_relaxed",
        chapter="biomechanics",
        pages="160–161",
        title={"ja": "グリップ圧を毎回同じにする", "en": "Make bow-hand pressure repeatable", "zh": "让弓手压力可以重复"},
        cue={"ja": "グリップのV字に預け、指で弓を握らない。", "en": "Settle into the grip V; do not grab the bow with the fingers.", "zh": "让弓把落在虎口的 V 区，不用手指抓弓。"},
        pass_fail={"ja": "PASS：発射後も指が柔らかく、スリングが弓を受ける。FAIL：指が閉じる、手首が跳ねる。", "en": "PASS: fingers stay soft and the sling receives the bow. FAIL: fingers close or the wrist flicks.", "zh": "通过：出箭后手指仍放松、由弓绳接住弓；失败：手指抓紧或手腕主动甩动。"},
        fallback={"ja": "1射やめて、プレドローで手をグリップに落ち着かせる動作を作り直す。", "en": "Skip one shot and rebuild how the hand settles into the grip during pre-draw.", "zh": "暂停一箭，在预拉阶段重新建立弓手落入握把的位置。"},
        why={"ja": "握り圧や手首位置の変化は、縦横どちらのグループも広げます。", "en": "Changing grip tension or wrist position can enlarge the group in both axes.", "zh": "握弓张力或手腕位置变化，会同时放大上下和左右散布。"},
        summary={"ja": "本書は、固定した力で握るより手をリラックスさせる方が反復しやすいと説明しています。", "en": "The book treats a relaxed bow hand as easier to repeat than a fixed muscular tension.", "zh": "书中指出，与其维持固定肌肉张力，放松弓手更容易重复。"},
        drill_name={"ja": "45秒・グリップ着地", "en": "45s · Grip-set drill", "zh": "45秒 · 弓手落位练习"},
        drill_how={"ja": "軽いプレドローで手をグリップ上部へ入れ、力を抜いて同じ場所へ6回落ち着かせる。", "en": "With light pre-draw tension, place the hand high in the grip and let it settle to the same spot six times.", "zh": "带一点预拉张力，把手放到握把上部，再放松落到同一位置，重复6次。"},
        duration_s=45,
        stage="setup→follow-through",
        mental={"ja": "「支える、握らない」", "en": "Support, don't grab.", "zh": "“支撑，不抓握。”"},
    ),
    "finish_to_target": _card(
        source_id="finish_to_target",
        chapter="biomechanics",
        pages="161, 180",
        title={"ja": "早く終わらせない", "en": "Do not finish the shot early", "zh": "不要提前结束动作"},
        cue={"ja": "矢が的に届くまで、弓肩・胴体・視線を残す。", "en": "Keep the bow side, torso and gaze in place until the arrow reaches the target.", "zh": "直到箭到靶前，弓肩、躯干和视线都保持不动。"},
        pass_fail={"ja": "PASS：着弾音まで形が残る。FAIL：離れと同時に肩が落ちる、覗きに行く。", "en": "PASS: the shape remains until impact. FAIL: the shoulder drops or you peek at release.", "zh": "通过：直到中靶声动作仍保持；失败：离弦同时肩膀下掉或急着看箭。"},
        fallback={"ja": "次の1射は得点を見ず、着弾音まで静止することだけを採点する。", "en": "On the next shot, ignore score and grade only whether you stayed through impact.", "zh": "下一箭不看分数，只判断自己是否保持到中靶。"},
        why={"ja": "発射直前の先回りした脱力は、グループを大きく不安定にします。", "en": "Anticipatory relaxation just before release makes groups larger and less stable.", "zh": "离弦前的提前松懈，会让箭群更大、更不稳定。"},
        summary={"ja": "反復動作では脳が次を先取りするため、本書は『的に届くまで』射を終えない考え方を示します。", "en": "Because the brain anticipates the next movement, the book extends the end of the shot through arrow impact.", "zh": "由于大脑会提前准备下一个动作，书中把动作结束点延长到箭真正中靶。"},
        drill_name={"ja": "3射・着弾まで残す", "en": "Three shots · Hold through impact", "zh": "3箭 · 保持到中靶"},
        drill_how={"ja": "3射だけ、離れの後に心の中で『届いた』と確認するまで形を変えない。", "en": "For three shots, change nothing after release until you silently register that the arrow has arrived.", "zh": "连续3箭，离弦后直到心里确认“箭到了”之前都不改变动作。"},
        duration_s=60,
        stage="release→follow-through",
        mental={"ja": "「的までが一射」", "en": "The shot ends at the target.", "zh": "“到靶才算一箭结束。”"},
    ),
    "same_draw_path": _card(
        source_id="same_draw_path",
        chapter="biomechanics",
        pages="167–170",
        title={"ja": "引き分けの経路を揃える", "en": "Repeat the same draw path", "zh": "统一开弓路径"},
        cue={"ja": "拉手を毎回同じ線で顔へ運び、アンカーで大きく修正しない。", "en": "Bring the draw hand to the face on the same line; avoid a large correction at anchor.", "zh": "拉弦手每次沿同一路径到脸，不要在锚点处做大幅修正。"},
        pass_fail={"ja": "PASS：アンカー到達時に肩・肘がほぼ完成。FAIL：満開で肘や肩を探し直す。", "en": "PASS: shoulder and elbow are nearly set on arrival at anchor. FAIL: you search for them at full draw.", "zh": "通过：到锚点时肩和肘基本就位；失败：满弓后还在重新找肩肘位置。"},
        fallback={"ja": "素引き1回。軽く滑らかに、同じ始点から同じ終点だけを確認する。", "en": "Do one blank draw: light and smooth, checking only the same start and finish points.", "zh": "空拉一次：轻、顺，只检查起点和终点是否一致。"},
        why={"ja": "満開での大きな修正は首・肩に余計な緊張を作り、離れを不安定にします。", "en": "Large full-draw corrections add neck and shoulder tension and make release less repeatable.", "zh": "满弓后的大幅修正会增加颈肩张力，使撒放更不稳定。"},
        summary={"ja": "本書は、同じ筋肉の読み込み方を作るため、滑らかで単純なTドローと一定の拉手経路を勧めています。", "en": "The book recommends a simple, smooth T-draw and a repeatable hand path so the muscles load the same way.", "zh": "书中建议使用简单顺畅的 T 型开弓和固定拉手路径，让肌肉每次以同样方式加载。"},
        drill_name={"ja": "45秒・始点と終点", "en": "45s · Same start, same finish", "zh": "45秒 · 同起点同终点"},
        drill_how={"ja": "弓腕内側への軽い接触を始点にし、同じ経路でアンカーへ5回運んで下ろす。", "en": "Use a light touch near the inside of the bow arm as the start reference; travel to anchor on the same path five times.", "zh": "用拉弦手轻触弓臂内侧作为起点，沿同一路径到锚点，重复5次后放下。"},
        duration_s=45,
        stage="pre-draw→anchor",
        mental={"ja": "「同じ線で顔へ」", "en": "Same line to the face.", "zh": "“沿同一条线到脸。”"},
    ),
    "release_from_back": _card(
        source_id="release_from_back",
        chapter="biomechanics",
        pages="171–174",
        title={"ja": "離れを作らず、背中から流す", "en": "Let the release flow from the back", "zh": "让撒放由背部自然带出"},
        cue={"ja": "肘を後方へ保ち、弦手は首の横へ自然に流れる。", "en": "Keep the elbow moving behind; let the draw hand flow past the neck naturally.", "zh": "让肘继续向后，拉弦手自然沿颈侧通过。"},
        pass_fail={"ja": "PASS：手が顎線に沿って後ろへ流れる。FAIL：手を意図的に跳ねる、前で止まる。", "en": "PASS: the hand travels back along the jaw line. FAIL: it is flicked deliberately or stops forward.", "zh": "通过：手沿下颌线自然后移；失败：主动甩手或停在脸前。"},
        fallback={"ja": "指を開こうとしない。両肘を反対方向へ伸ばす感覚を1回作り直す。", "en": "Do not try to open the fingers. Rebuild one rep of the elbows extending in opposite directions.", "zh": "不要主动张开手指；先重做一次双肘向相反方向延伸的感觉。"},
        why={"ja": "斜め・縦方向の散りでは、腕で作った離れやフォローの不足を切り分ける価値があります。", "en": "With diagonal or vertical variation, an arm-made release and missing follow-through are worth isolating.", "zh": "当箭群呈斜向或纵向变化时，值得排查用手臂制造撒放和随动不足。"},
        summary={"ja": "背中を使って力線を矢の後ろへ置くと、弦手のフォローは作る動作ではなく結果になる、という説明。", "en": "The book describes follow-through as the result of a back-led force line, not an artificial hand flick.", "zh": "书中把随动描述为背部主导的发力线所产生的结果，而不是手主动甩出的动作。"},
        drill_name={"ja": "30秒・両手組みリリース", "en": "30s · Linked-hands release", "zh": "30秒 · 双手相扣撒放练习"},
        drill_how={"ja": "胸の前で両手の指を組み、背中で左右へ広げて両腕が自然に離れる感覚を5回作る。", "en": "Interlace the fingers in front of the chest and expand from the back until both arms separate naturally; five reps.", "zh": "在胸前十指相扣，用背部向两侧展开，体会双臂自然分开的感觉，做5次。"},
        duration_s=30,
        stage="expansion→release",
        mental={"ja": "「肘は後ろ、手は結果」", "en": "Elbow back; hand follows.", "zh": "“肘向后，手只是结果。”"},
    ),
    "anchor_balance": _card(
        source_id="anchor_balance",
        chapter="biomechanics",
        pages="176–177",
        title={"ja": "アンカーで線を崩さない", "en": "Keep the line through anchor", "zh": "锚点处不要破坏发力线"},
        cue={"ja": "頭を迎えに行かず、弦を顔の同じ位置へ置いて前後圧を均等にする。", "en": "Do not reach with the head; place the string on the same facial reference and balance front/back pressure.", "zh": "头不要去迎弦；把弦放到脸上同一参考点，并平衡前后压力。"},
        pass_fail={"ja": "PASS：頭が静かで、弦が顔を滑らない。FAIL：顎へ押し上げる、頭が動く。", "en": "PASS: the head stays quiet and the string does not slide on the face. FAIL: the hand pushes upward or the head moves.", "zh": "通过：头保持安静、弦不在脸上滑动；失败：拉手向上顶或头部移动。"},
        fallback={"ja": "下ろして、アンカーの接触点を1つだけ選び直す。", "en": "Let down and reselect one anchor contact—only one.", "zh": "放下重来，只重新选择一个锚点接触参考。"},
        why={"ja": "高さや基準のばらつきは、頭・弦・肘の関係を毎回変えてしまいます。", "en": "Changing anchor height or reference changes the relationship between head, string and elbow.", "zh": "锚点高度或参考点变化，会改变头、弦和肘之间的关系。"},
        summary={"ja": "本書は、弦を顔へ確実に置き、頭を静かに保ち、前後圧を均等にして展開する考え方を示します。", "en": "The book places the string positively on a stable facial reference, with a quiet head and balanced expansion.", "zh": "书中强调把弦稳定放到固定面部参考点，头部安静，并以前后平衡的压力完成展开。"},
        drill_name={"ja": "40秒・接触点5回", "en": "40s · Five anchor contacts", "zh": "40秒 · 5次锚点接触"},
        drill_how={"ja": "射たずに5回、同じ顔の接触点・同じ頭位置へ入り、1秒確認して下ろす。", "en": "Without shooting, reach the same facial contact and head position five times; verify for one second, then let down.", "zh": "不放箭做5次：每次到同一面部接触点和头部位置，确认1秒后放下。"},
        duration_s=40,
        stage="anchor→expansion",
        mental={"ja": "「頭は静か、圧は均等」", "en": "Quiet head, balanced pressure.", "zh": "“头静，压力平衡。”"},
    ),
    "body_alignment": _card(
        source_id="body_alignment",
        chapter="biomechanics",
        pages="154–159",
        title={"ja": "まず土台を揃える", "en": "Normalize the foundation first", "zh": "先统一身体底座"},
        cue={"ja": "足圧を均等にし、背骨を保ったまま両肩を低く置く。", "en": "Balance the feet, keep the spine stable and set both shoulders low.", "zh": "双脚受力平衡，脊柱稳定，两侧肩膀保持下沉。"},
        pass_fail={"ja": "PASS：セットアップ後に重心と頭位置が動かない。FAIL：引きながら上体が傾く。", "en": "PASS: balance and head position stay put after set-up. FAIL: the torso leans during the draw.", "zh": "通过：举弓后重心和头位不变；失败：开弓过程中躯干倾斜。"},
        fallback={"ja": "一度弓を下ろし、足裏→背骨→肩の順にセットし直す。", "en": "Lower the bow and reset in order: feet, spine, shoulders.", "zh": "放下弓，按脚底—脊柱—肩膀的顺序重新设置。"},
        why={"ja": "全方向の散りでは、細部より先に動かない土台を確認する方が情報価値があります。", "en": "When variation is in every direction, a stable foundation is more informative than a small technical detail.", "zh": "箭群向各方向散开时，先检查稳定底座，比纠结小细节更有价值。"},
        summary={"ja": "身体を構造的に整え、重心と背骨を保つことを射の土台として説明しています。", "en": "The book treats structural body alignment, balance and a stable spine as the foundation of the shot.", "zh": "书中把身体结构对齐、重心和平稳脊柱视为整套动作的基础。"},
        drill_name={"ja": "30秒・足圧リセット", "en": "30s · Foot-pressure reset", "zh": "30秒 · 脚底压力重置"},
        drill_how={"ja": "弓なしで3回、足裏の左右差を消して肩を下げ、頭を的へ向ける。", "en": "Without the bow, set balanced foot pressure, lower the shoulders and turn the head to target three times.", "zh": "不拿弓做3次：平衡双脚压力、沉肩，再把头转向靶面。"},
        duration_s=30,
        stage="stance→setup",
        mental={"ja": "「足、背骨、肩」", "en": "Feet, spine, shoulders.", "zh": "“脚、脊柱、肩。”"},
    ),
    "big_things_first": _card(
        source_id="big_things_first",
        chapter="training",
        pages="211–214",
        title={"ja": "細部を増やさず、大きな一項目だけ", "en": "One large variable, not many small ones", "zh": "只练一个大变量，不堆小细节"},
        cue={"ja": "次の1エンドは、弓腕を着弾まで保つことだけを採点する。", "en": "For the next end, grade only whether the bow arm stays through impact.", "zh": "下一组只评一件事：弓臂是否保持到中靶。"},
        pass_fail={"ja": "PASS：6射中5射以上で同じ終了姿勢。FAIL：点数を追って別の修正を足す。", "en": "PASS: at least five of six shots finish in the same shape. FAIL: score chasing adds another correction.", "zh": "通过：6箭中至少5箭结束姿势一致；失败：因为追分又加入别的修正。"},
        fallback={"ja": "得点表示を閉じ、3mまたは近距離で同じ課題を3射。", "en": "Hide the score and repeat the same task for three shots at short range or 3 m.", "zh": "先不看分数，在近距离或3米处用同一课题再射3箭。"},
        why={"ja": "グループが全体に散る日は、複数の小修正より大きな反復課題を一つ選ぶ方が有効です。", "en": "On an all-direction loose group, one large repeatability task is more useful than several small corrections.", "zh": "箭群整体松散时，一个大的重复性课题，比同时做多个小修正更有效。"},
        summary={"ja": "本書は、まず大きな問題を扱い、一つの改善を十分な良い反復で定着させるよう勧めています。", "en": "The book recommends working on the big issue first and consolidating one change with many good repetitions.", "zh": "书中建议先处理大问题，并用大量高质量重复把一个改变真正固定下来。"},
        drill_name={"ja": "1エンド・一項目採点", "en": "One end · Single-variable grading", "zh": "1组 · 单变量评分"},
        drill_how={"ja": "各射を○/×だけで記録する。得点は見ず、エンド後に○が何個あったかだけ確認する。", "en": "Mark each shot only yes/no. Ignore score and count successful repetitions after the end.", "zh": "每箭只记“是/否”，不看分数；一组结束后只统计成功重复的次数。"},
        duration_s=90,
        stage="whole-shot repetition",
        mental={"ja": "「一射、一課題」", "en": "One shot, one task.", "zh": "“一箭，一个课题。”"},
    ),
    "diary_experiment": _card(
        source_id="diary_experiment",
        chapter="training",
        pages="211–213",
        title={"ja": "同じ問題は、記録できる実験にする", "en": "Turn a repeated issue into a recorded experiment", "zh": "把重复问题变成可记录的实验"},
        cue={"ja": "一項目だけ変え、6射のグループ幅と○/×を残す。", "en": "Change one variable only; record group width and a yes/no grade for six shots.", "zh": "一次只改变一个变量；记录6箭的箭群宽度和“是/否”评价。"},
        pass_fail={"ja": "PASS：変更前後を同じ距離・矢数で比較できる。FAIL：途中で別の修正も加える。", "en": "PASS: before/after can be compared at the same distance and arrow count. FAIL: another change is added mid-test.", "zh": "通过：前后都用相同距离和箭数，可直接比较；失败：测试中途又加入其他改动。"},
        fallback={"ja": "変数を元へ戻し、現在の設定をログへ残して終了する。", "en": "Return the variable to baseline, log the current setting and end the test.", "zh": "把变量恢复到基线，记录当前设置，然后结束测试。"},
        why={"ja": "同じ傾向が続いているため、別の一般論より履歴で原因を絞る段階です。", "en": "Because the same pattern is recurring, history is now more useful than another generic cue.", "zh": "同一种模式正在重复，此时用历史缩小原因范围，比再给一句泛泛口令更有价值。"},
        summary={"ja": "本書は、距離・得点・調整内容を短く記録し、改善につながるパターンを見つける方法を勧めています。", "en": "The book recommends short records of distance, score and adjustments so useful patterns become visible.", "zh": "书中建议简要记录距离、分数和调整内容，从而看出哪些模式真正带来改善。"},
        drill_name={"ja": "6射・A/Bログ", "en": "Six shots · A/B log", "zh": "6箭 · A/B记录"},
        drill_how={"ja": "現在をAとして保存。次の6射だけ一項目をBへ変え、広がり・平均点・○/×を比較する。", "en": "Save the current end as A. Change one item for the next six shots as B; compare spread, average and yes/no grade.", "zh": "把当前一组存为A；下一组6箭只改一项作为B，对比散布、平均分和“是/否”评价。"},
        duration_s=120,
        stage="training design",
        mental={"ja": "「変えるのは一つ」", "en": "Change one thing.", "zh": "“一次只改一件事。”"},
    ),
    "quiet_mind": _card(
        source_id="quiet_mind",
        chapter="mind",
        pages="234",
        title={"ja": "動きを操作せず、静かな感覚を一つ", "en": "Use one quiet sensation, not a movement command", "zh": "只保留一个安静感觉，不主动操控动作"},
        cue={"ja": "足裏か柔らかい弓手の感覚だけを見守る。離れは操作しない。", "en": "Observe only the feet or a soft bow hand; do not steer the release.", "zh": "只感受脚底或放松的弓手，不要主动操控撒放。"},
        pass_fail={"ja": "PASS：一つの静的感覚のまま射が流れる。FAIL：離れや弓腕を意識して動かす。", "en": "PASS: the shot flows while attention stays on one static sensation. FAIL: you consciously move the release or bow arm.", "zh": "通过：注意力停在一个静态感觉上，动作自然流动；失败：有意识地控制撒放或弓臂。"},
        fallback={"ja": "考える言葉を『足裏』の一語に戻し、1回下ろして再開する。", "en": "Reduce the thought to one word—feet—let down once, then restart.", "zh": "把想法缩成一个词“脚底”，放下重来一次。"},
        why={"ja": "まとまりが良い時は、意識で動作を助けようとすると逆に誇張が入りやすくなります。", "en": "When the group is already good, conscious help can exaggerate a movement and disturb it.", "zh": "箭群已经不错时，意识主动“帮忙”反而容易把动作夸大并打乱。"},
        summary={"ja": "本書は、動的な部位を意識で操作するより、弓手の脱力など静かな課題で意識を占める考え方を示します。", "en": "The book suggests occupying conscious attention with a quiet task, such as a relaxed bow hand, rather than controlling motion.", "zh": "书中建议用放松弓手等安静课题占住意识，而不是让意识直接控制动作。"},
        drill_name={"ja": "3射・静かな一語", "en": "Three shots · One quiet word", "zh": "3箭 · 一个安静词"},
        drill_how={"ja": "『足裏』または『柔らかい手』を選び、3射の間はその一語以外を足さない。", "en": "Choose feet or soft hand; add no other thought for three shots.", "zh": "选择“脚底”或“手放松”，连续3箭不再加入其他想法。"},
        duration_s=60,
        stage="attention→automatic shot",
        mental={"ja": "「静かな一語」", "en": "One quiet word.", "zh": "“一个安静词。”"},
    ),
    "focus_trigger": _card(
        source_id="focus_trigger",
        chapter="mind",
        pages="258–259",
        title={"ja": "集中の開始点を固定する", "en": "Give concentration a fixed start", "zh": "给专注一个固定起点"},
        cue={"ja": "指を弦に置いた瞬間に集中を始め、着弾で終える。", "en": "Start concentration when the fingers meet the string; end it at impact.", "zh": "手指触弦时开始专注，箭中靶时结束。"},
        pass_fail={"ja": "PASS：全射同じ合図で集中へ入る。FAIL：構えた後も考え事や点数が残る。", "en": "PASS: every shot enters focus from the same trigger. FAIL: score or stray thoughts remain after set-up.", "zh": "通过：每箭都由同一信号进入专注；失败：举弓后仍想着分数或其他事情。"},
        fallback={"ja": "下ろして、弦に触れる動作から集中サイクルを再開する。", "en": "Let down and restart the concentration cycle from the string-touch trigger.", "zh": "放下重来，从触弦这个信号重新启动专注循环。"},
        why={"ja": "良いグループを守るには、技術を足すより毎射同じ集中窓を再現する方が安全です。", "en": "To protect a good group, repeating the same focus window is safer than adding another technique.", "zh": "要守住好的箭群，重复同样的专注窗口，比再添加一个技术动作更稳妥。"},
        summary={"ja": "集中は一日中続けず、物理的な合図から一射分だけオンにする方法が説明されています。", "en": "The book uses a physical trigger to switch concentration on for one shot rather than trying to focus all day.", "zh": "书中建议用一个身体信号，只为这一箭开启专注，而不是整天持续用力集中。"},
        drill_name={"ja": "1エンド・集中スイッチ", "en": "One end · Focus switch", "zh": "1组 · 专注开关"},
        drill_how={"ja": "6射すべて、弦に触れた時だけ『オン』、着弾で『オフ』と心の中で区切る。", "en": "For all six shots, silently switch on at string contact and off at impact.", "zh": "连续6箭：触弦时心里“开”，中靶时“关”。"},
        duration_s=90,
        stage="pre-shot focus→impact",
        mental={"ja": "「触弦でオン」", "en": "String: on.", "zh": "“触弦，开启。”"},
    ),
    "feel_then_check": _card(
        source_id="feel_then_check",
        chapter="training",
        pages="213, 326",
        title={"ja": "結果を見る前に、良い射を識別する", "en": "Identify the shot before checking the result", "zh": "看结果前，先识别这一箭的感觉"},
        cue={"ja": "放った直後に○/×を決めてから、矢所を見る。", "en": "Grade the shot yes/no immediately after release, then look at the arrow.", "zh": "离弦后先判断这箭“是/否”，再看箭落点。"},
        pass_fail={"ja": "PASS：矢所を見る前に射感を言える。FAIL：点数だけで良い射・悪い射を決める。", "en": "PASS: you can name the shot feel before seeing impact. FAIL: score alone defines whether it was good.", "zh": "通过：看落点前就能说出射感；失败：只用分数判断动作好坏。"},
        fallback={"ja": "スコープを一射だけ見ず、終了姿勢と射感を先に記憶する。", "en": "Skip the scope for one shot and remember the finish shape and feel first.", "zh": "有一箭先不看镜，先记住结束姿势和射感。"},
        why={"ja": "中心にまとまり始めた段階では、『良い射の感覚』を結果と結び付けることが次の改善になります。", "en": "Once the group is clustering, linking the feel of a good shot to its result is the next useful step.", "zh": "当箭群开始集中后，把“好箭的感觉”和结果对应起来，才是下一步提升。"},
        summary={"ja": "本書は『射つ→感じる→矢所を見る』の順で、十点を生む感覚を学ぶよう勧めています。", "en": "The book puts feeling the shot before viewing the impact so the archer can learn what produces tens.", "zh": "书中建议按“射—感受—看落点”的顺序，学习什么样的感觉会产生好箭。"},
        drill_name={"ja": "6射・予測ログ", "en": "Six shots · Prediction log", "zh": "6箭 · 预测记录"},
        drill_how={"ja": "各射後、見る前に○/×を決める。エンド後に実際の矢所と照合する。", "en": "After each shot, mark yes/no before looking; compare predictions with impacts after the end.", "zh": "每箭后先不看落点，记“是/否”；一组结束后再与实际落点核对。"},
        duration_s=120,
        stage="release→feedback",
        mental={"ja": "「感じてから見る」", "en": "Feel, then look.", "zh": "“先感受，再看。”"},
    ),
    "group_then_adjust": _card(
        source_id="group_then_adjust",
        chapter="better",
        pages="326, 328",
        title={"ja": "まとまりは守り、狙点だけ補正する", "en": "Protect the group; correct only the reference", "zh": "保住箭群，只修正瞄准参考"},
        cue={"ja": "フォームは変えず、まとまりの方向へサイトを一目盛だけ動かす。", "en": "Keep the form unchanged; move the sight one controlled increment toward the group.", "zh": "动作不变，把瞄具朝箭群方向只移动一个可控刻度。"},
        pass_fail={"ja": "PASS：次の3射も同じ大きさで中心へ移る。FAIL：同時にフォームや複数設定も変える。", "en": "PASS: the next three arrows keep the same group size and move toward centre. FAIL: form or multiple settings also change.", "zh": "通过：接下来3箭箭群大小不变并向中心移动；失败：同时又改动作或多个设置。"},
        fallback={"ja": "元の目盛へ戻し、現在のサイト値と条件を記録する。", "en": "Return to the original mark and record the current sight value and conditions.", "zh": "恢复原刻度，并记录当前瞄具数值和环境条件。"},
        why={"ja": "グループは十分まとまっているのに中心から外れているため、再現性より照準の問題を先に確認します。", "en": "The group is already compact but displaced, so verify the aiming reference before changing repeatable form.", "zh": "箭群已经很紧但整体偏离中心，因此应先检查瞄准参考，而不是破坏可重复的动作。"},
        summary={"ja": "本書は、良いグループを識別した上でサイトを調整し、距離・設定・条件を記録する考え方を示します。", "en": "The book separates a good group from its position, then uses sight adjustment and recorded sight marks.", "zh": "书中把“箭群大小”和“箭群位置”分开处理，再通过瞄具调整和记录刻度来修正位置。"},
        drill_name={"ja": "3射・一目盛テスト", "en": "Three shots · One-increment test", "zh": "3箭 · 单刻度测试"},
        drill_how={"ja": "現在値を記録し、まとまり方向へ一目盛だけ動かす。3射後に中心距離と広がりを比較する。", "en": "Record the current mark, move one increment toward the group, then compare offset and spread after three shots.", "zh": "记录当前刻度，向箭群方向移动一个刻度；3箭后比较偏移和散布。"},
        duration_s=90,
        stage="feedback→sight setting",
        mental={"ja": "「形は守る、照準だけ」", "en": "Keep the shot; move the reference.", "zh": "“动作不变，只改参考。”"},
    ),
    "scoreless_groups": _card(
        source_id="scoreless_groups",
        chapter="training",
        pages="214–215",
        title={"ja": "得点から離れてグループを練習する", "en": "Train the group without score", "zh": "暂时离开分数，只练箭群"},
        cue={"ja": "次のエンドは点数を消し、全矢が同じ終了姿勢かだけを見る。", "en": "Hide score for the next end and judge only whether every shot shares the same finish.", "zh": "下一组隐藏分数，只看每箭结束姿势是否一致。"},
        pass_fail={"ja": "PASS：得点を見ずに決めた課題を全矢で守る。FAIL：悪い矢所を見て途中で技術を変える。", "en": "PASS: the chosen task survives every shot without score checking. FAIL: one impact makes you change technique mid-end.", "zh": "通过：不看分数，整组都守住选定课题；失败：看到一箭不好就中途改技术。"},
        fallback={"ja": "距離を短くし、3射だけ同じテンポで成功反復を作る。", "en": "Shorten the distance and make three successful repetitions at the same tempo.", "zh": "缩短距离，只用同一节奏完成3次成功重复。"},
        why={"ja": "平均点が低く広がりも大きい時は、結果の修正より成功動作の再構築が先です。", "en": "When both average and grouping are poor, rebuilding successful movement comes before correcting results.", "zh": "平均分低且箭群很散时，应先重建成功动作，而不是追着结果修正。"},
        summary={"ja": "本書は、練習目的に応じて得点ではなくグループを撃ち、パラメータを決めて評価する方法を示します。", "en": "The book separates group practice from scoring and recommends setting a clear success parameter.", "zh": "书中把箭群练习与计分练习分开，并建议预先设定清晰的成功标准。"},
        drill_name={"ja": "近距離3射・成功だけ", "en": "Three close shots · Success reps", "zh": "近距离3箭 · 只做成功重复"},
        drill_how={"ja": "近距離へ移り、同じテンポ・同じ終了姿勢の3射だけを行う。点数は記録しない。", "en": "Move close and make only three shots with identical tempo and finish; do not record score.", "zh": "移到近距离，只射3箭，保持相同节奏和结束姿势；不记录分数。"},
        duration_s=90,
        stage="rebuild→group practice",
        mental={"ja": "「点ではなく反復」", "en": "Repetition, not score.", "zh": "“先重复，不追分。”"},
    ),
}


DIAGNOSIS_CANDIDATES = {
    "single_outlier": ["feel_then_check", "finish_to_target", "focus_trigger"],
    "horizontal": ["string_vertical", "bow_hand_relaxed", "release_from_back"],
    "vertical": ["anchor_balance", "finish_to_target", "same_draw_path"],
    "loose": ["big_things_first", "body_alignment", "scoreless_groups"],
    "tight_offset": ["group_then_adjust", "feel_then_check", "focus_trigger"],
    "protect": ["feel_then_check", "quiet_mind", "focus_trigger", "finish_to_target"],
    "reference": ["anchor_balance", "same_draw_path", "bow_hand_relaxed", "body_alignment"],
}


SELF_REPORT_CARD = {
    "bow_hand_tense": "bow_hand_relaxed",
    "anchor_unclear": "anchor_balance",
    "release_forced": "release_from_back",
    "aim_too_long": "quiet_mind",
    "bow_arm_dropped": "finish_to_target",
    "fatigue_or_pain": "scoreless_groups",
    "wind_or_equipment": "diary_experiment",
}


SELF_REPORT_LABELS = {
    "none": {"ja": "特になし", "en": "Nothing specific", "zh": "没有明显感觉"},
    "bow_hand_tense": {"ja": "弓手が力んだ", "en": "Bow hand felt tense", "zh": "弓手偏紧"},
    "anchor_unclear": {"ja": "アンカーが曖昧", "en": "Anchor felt unclear", "zh": "锚点不清楚"},
    "release_forced": {"ja": "離れを作った", "en": "Release felt forced", "zh": "撒放有主动用力"},
    "aim_too_long": {"ja": "狙いが長すぎた", "en": "Aim lasted too long", "zh": "瞄准时间过长"},
    "bow_arm_dropped": {"ja": "弓腕が落ちた", "en": "Bow arm dropped", "zh": "弓臂提前下落"},
    "fatigue_or_pain": {"ja": "疲労または痛み", "en": "Fatigue or pain", "zh": "疲劳或疼痛"},
    "wind_or_equipment": {"ja": "風または用具が気になった", "en": "Wind or equipment concern", "zh": "风或器材有影响"},
}


def _ratio(metrics: Dict[str, Any], key: str, raw_key: str) -> float:
    value = metrics.get(key)
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    try:
        return float(metrics.get(raw_key, 0.0) or 0.0) / 405.0
    except (TypeError, ValueError):
        return 0.0


def _comparable_log(log: List[Dict[str, Any]], context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not context:
        return list(log or [])
    distance = context.get("distance_m")
    target_face = context.get("target_face")
    return [
        entry
        for entry in (log or [])
        if isinstance(entry, dict)
        and (distance is None or entry.get("distance_m") == distance)
        and (target_face is None or entry.get("target_face") == target_face)
    ]


def _last_metrics(log: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for entry in reversed(log or []):
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if isinstance(metrics, dict):
            return metrics
    return None


def _recent_source_ids(log: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    ids: List[str] = []
    for entry in reversed(log or []):
        advice = entry.get("advice", {}) if isinstance(entry, dict) else {}
        if not isinstance(advice, dict):
            continue
        source = advice.get("book_source", {}) or {}
        source_id = source.get("id") if isinstance(source, dict) else None
        if source_id:
            ids.append(str(source_id))
        if len(ids) >= limit:
            break
    return ids


def _last_advice(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    for entry in reversed(log or []):
        advice = entry.get("advice") if isinstance(entry, dict) else None
        if isinstance(advice, dict):
            return advice
    return {}


def _recent_diagnoses(log: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    keys: List[str] = []
    for entry in reversed(log or []):
        advice = entry.get("advice", {}) if isinstance(entry, dict) else {}
        diagnosis = advice.get("diagnosis", {}) if isinstance(advice, dict) else {}
        key = diagnosis.get("key") if isinstance(diagnosis, dict) else None
        if key:
            keys.append(str(key))
        if len(keys) >= limit:
            break
    return keys


def _direction(lang: str, dx: float, dy: float) -> str:
    horizontal = ""
    vertical = ""
    if abs(dx) >= 4:
        horizontal = _text(lang, {"ja": "右" if dx > 0 else "左", "en": "right" if dx > 0 else "left", "zh": "右" if dx > 0 else "左"})
    if abs(dy) >= 4:
        vertical = _text(lang, {"ja": "下" if dy > 0 else "上", "en": "low" if dy > 0 else "high", "zh": "下" if dy > 0 else "上"})
    if horizontal and vertical:
        if lang == "en":
            return f"{vertical}-{horizontal}"
        return f"{horizontal}{vertical}"
    return horizontal or vertical or _text(lang, {"ja": "中心付近", "en": "near centre", "zh": "中心附近"})


def _diagnose(metrics: Dict[str, Any], shape: str, scoring: Dict[str, Any]) -> Dict[str, Any]:
    spread_ratio = _ratio(metrics, "spread_ratio", "spread")
    offset = metrics.get("offset", {}) or {}
    dx = float(offset.get("dx", 0.0) or 0.0)
    dy = float(offset.get("dy", 0.0) or 0.0)
    value = metrics.get("offset_ratio")
    if value is None:
        offset_ratio = float(offset.get("mag", (dx * dx + dy * dy) ** 0.5) or 0.0) / 405.0
    else:
        offset_ratio = float(value or 0.0)
    avg = float(scoring.get("avg", 0.0) or 0.0)
    sx = float(metrics.get("sx", 0.0) or 0.0)
    sy = float(metrics.get("sy", 0.0) or 0.0)
    n = int(metrics.get("n", len(scoring.get("scores", []) or [])) or 6)
    outlier = metrics.get("outlier", {}) or {}
    spread = float(metrics.get("spread", 0.0) or 0.0)
    core_spread = float(outlier.get("core_spread", spread) or 0.0)
    # Preserve the target-size normalization already computed by metrics. The
    # canonical target is currently 405 px, but coaching should not depend on
    # that rendering detail.
    core_spread_ratio = spread_ratio * (core_spread / spread) if spread > 1e-6 else 0.0
    anisotropy = float(metrics.get("anisotropy", max(sx, sy) / max(min(sx, sy), 1e-6)) or 1.0)

    tight = spread_ratio <= 0.085
    loose = spread_ratio >= 0.135
    if bool(outlier.get("present")):
        key = "single_outlier"
    elif tight and offset_ratio >= 0.10:
        key = "tight_offset"
    elif shape == "horizontal" and sx > sy * 1.35:
        key = "horizontal"
    elif shape == "vertical" and sy > sx * 1.35:
        key = "vertical"
    elif loose:
        key = "loose"
    elif tight or avg >= 8.7:
        key = "protect"
    else:
        key = "reference"

    return {
        "key": key,
        "n": n,
        "spread_ratio": spread_ratio,
        "core_spread_ratio": core_spread_ratio,
        "offset_ratio": offset_ratio,
        "spread": spread,
        "dx": dx,
        "dy": dy,
        "sx": sx,
        "sy": sy,
        "anisotropy": anisotropy,
        "avg": avg,
        "outlier": outlier,
    }


def _trend(lang: str, current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not previous:
        return {"key": "first", "text": _text(lang, {"ja": "比較できる前回ログはまだありません。", "en": "No previous saved end is available for comparison.", "zh": "暂时没有可比较的上一组保存记录。"})}
    prev = _ratio(previous, "spread_ratio", "spread")
    now = float(current["spread_ratio"])
    if prev <= 1e-6:
        return {"key": "steady", "text": _text(lang, {"ja": "前回との差はまだ判定できません。", "en": "The change from the previous end is not yet measurable.", "zh": "暂时无法判断与上一组的变化。"})}
    change = (now - prev) / prev
    pct = abs(change) * 100
    if change <= -0.10:
        return {"key": "improving", "text": _text(lang, {"ja": f"前回より広がりが約{pct:.0f}%縮小。今は新しい修正を増やさない段階です。", "en": f"Spread is about {pct:.0f}% smaller than the last saved end; avoid adding another correction now.", "zh": f"散布比上一组缩小约{pct:.0f}%；此时不宜再增加新的修正。"})}
    if change >= 0.15:
        return {"key": "worse", "text": _text(lang, {"ja": f"前回より広がりが約{pct:.0f}%増加。課題を一つに戻します。", "en": f"Spread is about {pct:.0f}% larger than the last saved end; return to one task.", "zh": f"散布比上一组扩大约{pct:.0f}%；应回到单一课题。"})}
    return {"key": "steady", "text": _text(lang, {"ja": "前回と広がりはほぼ同じ。同じ原因候補の別チェックへ進みます。", "en": "Spread is similar to the last saved end, so use a different check within the same issue.", "zh": "散布与上一组接近，因此在同一问题范围内换一个检查点。"})}


def _previous_experiment(lang: str, diagnosis: Dict[str, Any], trend: Dict[str, str], log: List[Dict[str, Any]]) -> Dict[str, str]:
    previous = _last_advice(log)
    previous_diagnosis = previous.get("diagnosis", {}) if isinstance(previous, dict) else {}
    if not previous or previous_diagnosis.get("key") != diagnosis["key"]:
        return {}
    status = trend["key"]
    verdicts = {
        "improving": {
            "ja": "暫定的に有効",
            "en": "Provisionally effective",
            "zh": "暂定有效",
        },
        "steady": {
            "ja": "明確な効果なし",
            "en": "No clear effect yet",
            "zh": "暂未显示明显效果",
        },
        "worse": {
            "ja": "今回は支持されない",
            "en": "Not supported by this end",
            "zh": "本组结果暂不支持",
        },
    }
    if status not in verdicts:
        return {}
    return {
        "status": status,
        "verdict": _text(lang, verdicts[status]),
        "title": str(previous.get("title", "")),
        "cue": str(previous.get("single_cue", previous.get("cue", ""))),
        "detail": trend["text"],
    }


def _evidence(lang: str, diagnosis: Dict[str, Any]) -> str:
    key = diagnosis["key"]
    spread_pct = diagnosis["spread_ratio"] * 100
    offset_pct = diagnosis["offset_ratio"] * 100
    sx, sy = diagnosis["sx"], diagnosis["sy"]
    direction = _direction(lang, diagnosis["dx"], diagnosis["dy"])
    if key == "single_outlier":
        outlier = diagnosis["outlier"]
        arrow_number = int(outlier.get("index", 0)) + 1
        improvement = float(outlier.get("improvement_ratio", 0.0) or 0.0) * 100
        core_pct = diagnosis["core_spread_ratio"] * 100
        return _text(lang, {
            "ja": f"第{arrow_number}射を外すと、残りのコア散布は外径の{core_pct:.1f}%で、広がりが{improvement:.0f}%縮みます。全体崩れではなく単発の外れ候補です。",
            "en": f"Removing arrow {arrow_number} leaves a core spread of {core_pct:.1f}% of target radius and reduces spread by {improvement:.0f}%. This looks like one escaped shot, not a whole-group collapse.",
            "zh": f"去掉第 {arrow_number} 箭后，核心散布为靶半径的 {core_pct:.1f}%，整体散布缩小 {improvement:.0f}%。更像单箭失误，而不是整组动作崩溃。",
        })
    if key == "horizontal":
        multiple = sx / max(sy, 1e-6)
        return _text(lang, {"ja": f"横方向のばらつきは縦の{multiple:.1f}倍（sx {sx:.1f} / sy {sy:.1f}px）。", "en": f"Horizontal variation is {multiple:.1f}× vertical variation (sx {sx:.1f} / sy {sy:.1f}px).", "zh": f"横向波动是纵向的 {multiple:.1f} 倍（sx {sx:.1f} / sy {sy:.1f}px）。"})
    if key == "vertical":
        multiple = sy / max(sx, 1e-6)
        return _text(lang, {"ja": f"縦方向のばらつきは横の{multiple:.1f}倍（sy {sy:.1f} / sx {sx:.1f}px）。", "en": f"Vertical variation is {multiple:.1f}× horizontal variation (sy {sy:.1f} / sx {sx:.1f}px).", "zh": f"纵向波动是横向的 {multiple:.1f} 倍（sy {sy:.1f} / sx {sx:.1f}px）。"})
    if key == "tight_offset":
        return _text(lang, {"ja": f"広がりは外径の{spread_pct:.1f}%と小さい一方、中心から{direction}へ{offset_pct:.1f}%ずれています。", "en": f"The group is compact ({spread_pct:.1f}% of target radius) but sits {offset_pct:.1f}% off-centre toward {direction}.", "zh": f"箭群较紧（散布为靶半径的 {spread_pct:.1f}%），但整体向{direction}偏离 {offset_pct:.1f}%。"})
    if key == "loose":
        return _text(lang, {"ja": f"平均的な広がりは外径の{spread_pct:.1f}%、平均点は{diagnosis['avg']:.2f}。全体課題を優先します。", "en": f"Average spread is {spread_pct:.1f}% of target radius and average score is {diagnosis['avg']:.2f}; prioritize a whole-shot task.", "zh": f"平均散布为靶半径的 {spread_pct:.1f}%，平均分 {diagnosis['avg']:.2f}；应优先处理整体动作。"})
    if key == "protect":
        return _text(lang, {"ja": f"広がりは外径の{spread_pct:.1f}%、平均点は{diagnosis['avg']:.2f}。大きな変更より再現性を守ります。", "en": f"Spread is {spread_pct:.1f}% of target radius with a {diagnosis['avg']:.2f} average; protect repeatability rather than make a large change.", "zh": f"散布为靶半径的 {spread_pct:.1f}%，平均分 {diagnosis['avg']:.2f}；现在应守住重复性，而不是大改动作。"})
    return _text(lang, {"ja": f"広がりは外径の{spread_pct:.1f}%で、強い一方向性はありません。基準の再現性を確認します。", "en": f"Spread is {spread_pct:.1f}% of target radius without one dominant axis; check repeatable references.", "zh": f"散布为靶半径的 {spread_pct:.1f}%，没有明显单一方向；应检查参考点的重复性。"})


def _alternative_hypotheses(lang: str, diagnosis: Dict[str, Any]) -> List[str]:
    values = {
        "single_outlier": {
            "ja": ["その1射だけフォローが早く終わった", "着弾点のマーキング誤差", "突風や用具の一時的な影響"],
            "en": ["Follow-through ended early on that shot", "Hit-point marking error", "A brief wind or equipment disturbance"],
            "zh": ["那一箭随动提前结束", "命中点标记误差", "短时风力或器材扰动"],
        },
        "horizontal": {
            "ja": ["弦の縦方向または弦手のトルク", "弓手の圧が左右に変化", "横風またはサイト設定"],
            "en": ["String verticality or draw-hand torque", "Changing bow-hand pressure", "Crosswind or sight setting"],
            "zh": ["弦线竖直或拉弦手扭转", "弓手压力左右变化", "侧风或瞄具设置"],
        },
        "vertical": {
            "ja": ["アンカー高さまたは引き経路", "弓腕が早く落ちる", "照明・サイト高さ・ノッキング点"],
            "en": ["Anchor height or draw path", "Bow arm dropping early", "Light, sight height or nocking point"],
            "zh": ["锚点高度或开弓路径", "弓臂提前下落", "光线、瞄具高度或搭箭点"],
        },
        "loose": {
            "ja": ["疲労またはテンポの崩れ", "姿勢・肩・弓手など土台の変化", "画像補正または点の確認誤差"],
            "en": ["Fatigue or tempo drift", "Foundation changes in stance, shoulder or bow hand", "Image alignment or point-confirmation error"],
            "zh": ["疲劳或节奏漂移", "站姿、肩膀或弓手等基础变化", "图像校正或点位确认误差"],
        },
        "tight_offset": {
            "ja": ["サイト値または狙点", "一定方向の風", "的補正の中心ずれ"],
            "en": ["Sight mark or aiming reference", "Steady wind", "A small error in target-centre alignment"],
            "zh": ["瞄具刻度或瞄准参考", "稳定方向的风", "靶心校正存在小偏差"],
        },
        "protect": {
            "ja": ["技術問題ではなく通常の標本変動", "良い射を意識で操作し始めた", "点を早く見ようとした"],
            "en": ["Normal sample variation rather than a technical fault", "Conscious control entering an already good shot", "Looking for the result too early"],
            "zh": ["只是正常样本波动，并非技术故障", "意识开始干预原本良好的动作", "过早寻找落点"],
        },
        "reference": {
            "ja": ["アンカーまたは弦像の基準", "身体アライメント", "集中開始点の不一致"],
            "en": ["Anchor or string-picture reference", "Body alignment", "Inconsistent focus trigger"],
            "zh": ["锚点或弦像参考", "身体对齐", "专注启动点不一致"],
        },
    }
    return list(values[diagnosis["key"]].get(lang, values[diagnosis["key"]]["en"]))


def _adapt_arrow_count(text: str, lang: str, arrow_count: int) -> str:
    """Adapt reviewed card wording to the current end's actual arrow count."""
    arrow_count = max(1, int(arrow_count))
    pass_count = max(1, arrow_count - 1)
    if arrow_count == 6:
        return text
    replacements = {
        "ja": [
            ("6射中5射以上", f"{arrow_count}射中{pass_count}射以上"),
            ("次の6射", f"次の{arrow_count}射"),
            ("6射すべて", f"{arrow_count}射すべて"),
            ("6射", f"{arrow_count}射"),
        ],
        "en": [
            ("at least five of six shots", f"at least {pass_count} of {arrow_count} shots"),
            ("At least five of six shots", f"At least {pass_count} of {arrow_count} shots"),
            ("next six shots", f"next {arrow_count} shots"),
            ("next six arrows", f"next {arrow_count} arrows"),
            ("For all six shots", f"For all {arrow_count} shots"),
            ("Six shots", f"{arrow_count} shots"),
            ("six shots", f"{arrow_count} shots"),
        ],
        "zh": [
            ("6箭中至少5箭", f"{arrow_count}箭中至少{pass_count}箭"),
            ("至少 5/6 箭", f"至少 {pass_count}/{arrow_count} 箭"),
            ("至少5/6箭", f"至少{pass_count}/{arrow_count}箭"),
            ("下一组6箭", f"下一组{arrow_count}箭"),
            ("连续6箭", f"连续{arrow_count}箭"),
            ("6箭", f"{arrow_count}箭"),
        ],
    }
    for source, target in replacements.get(lang, replacements["en"]):
        text = text.replace(source, target)
    return text


def _success_criterion(lang: str, diagnosis: Dict[str, Any], arrow_count: int = 6) -> str:
    arrow_count = max(1, int(arrow_count))
    pass_count = max(1, arrow_count - 1)
    key = diagnosis["key"]
    if key == "single_outlier":
        return _text(lang, {
            "ja": f"次の{arrow_count}射で{pass_count}射以上を○にし、単独でコア半径の2倍以上離れる矢を0本にする。",
            "en": f"Grade at least {pass_count} of {arrow_count} shots as pass, with no arrow landing more than 2× the core radius alone.",
            "zh": f"下一组至少 {pass_count}/{arrow_count} 箭动作通过，并且没有单箭独自偏离核心半径 2 倍以上。",
        })
    if key in {"horizontal", "vertical"}:
        ratio = diagnosis["anisotropy"]
        axis = _text(lang, {"ja": "横/縦比" if key == "horizontal" else "縦/横比", "en": "horizontal/vertical ratio" if key == "horizontal" else "vertical/horizontal ratio", "zh": "横纵比" if key == "horizontal" else "纵横比"})
        return _text(lang, {
            "ja": f"次の{arrow_count}射で{axis}を現在の{ratio:.1f}から15%以上下げるか、1.5以下にする。平均点は0.5点以上落とさない。",
            "en": f"On the next {arrow_count} arrows, reduce the {axis} by at least 15% from {ratio:.1f}, or bring it to 1.5 or less, without losing more than 0.5 average points.",
            "zh": f"下一组把{axis}从当前 {ratio:.1f} 降低至少 15%，或降到 1.5 以下；平均分下降不超过 0.5。",
        })
    if key == "loose":
        current = diagnosis["core_spread_ratio"] * 100
        target = current * 0.85
        return _text(lang, {
            "ja": f"次の{arrow_count}射でコア散布を外径の{current:.1f}%から{target:.1f}%以下へ縮め、{pass_count}射以上を同じ終了姿勢にする。",
            "en": f"Reduce core spread from {current:.1f}% to {target:.1f}% of target radius or less, with at least {pass_count} matching finishes.",
            "zh": f"下一组把核心散布从靶半径的 {current:.1f}% 降到 {target:.1f}% 以下，并至少有 {pass_count} 箭结束姿势一致。",
        })
    if key == "tight_offset":
        offset = diagnosis["offset_ratio"] * 100
        target = offset * 0.8
        spread_limit = diagnosis["spread_ratio"] * 110
        return _text(lang, {
            "ja": f"中心ずれを{offset:.1f}%から{target:.1f}%以下へ減らし、広がりは{spread_limit:.1f}%以内に保つ。",
            "en": f"Reduce offset from {offset:.1f}% to {target:.1f}% of target radius or less while keeping spread at {spread_limit:.1f}% or less.",
            "zh": f"把中心偏移从靶半径的 {offset:.1f}% 降到 {target:.1f}% 以下，同时散布保持在 {spread_limit:.1f}% 以下。",
        })
    if key == "protect":
        limit = diagnosis["spread_ratio"] * 110
        return _text(lang, {
            "ja": f"{arrow_count}射中{pass_count}射以上で同じ合図を守り、広がりを外径の{limit:.1f}%以内に保つ。",
            "en": f"Keep the cue on at least {pass_count} of {arrow_count} shots and hold spread within {limit:.1f}% of target radius.",
            "zh": f"至少 {pass_count}/{arrow_count} 箭守住同一口令，并把散布维持在靶半径的 {limit:.1f}% 以内。",
        })
    target = diagnosis["spread_ratio"] * 90
    return _text(lang, {
        "ja": f"{arrow_count}射中{pass_count}射以上で基準を再現し、広がりを外径の{target:.1f}%以下へ縮める。",
        "en": f"Repeat the reference on at least {pass_count} of {arrow_count} shots and reduce spread to {target:.1f}% of target radius or less.",
        "zh": f"至少 {pass_count}/{arrow_count} 箭重复同一参考点，并把散布降到靶半径的 {target:.1f}% 以下。",
    })


def _do_not_change(lang: str, diagnosis: Dict[str, Any]) -> str:
    key = diagnosis["key"]
    values = {
        "single_outlier": {"ja": "1本の外れだけでサイトやフォーム全体を変えない。", "en": "Do not change sight or the whole form because of one escaped arrow.", "zh": "不要因为一支异常箭就调整瞄具或整套动作。"},
        "tight_offset": {"ja": "サイトとフォームを同時に変えない。", "en": "Do not change the sight and form at the same time.", "zh": "不要同时修改瞄具和动作。"},
        "protect": {"ja": "良いグループに新しい技術課題を足さない。", "en": "Do not add a new technical task to an already good group.", "zh": "不要给已经不错的箭群再增加新技术课题。"},
    }
    return _text(lang, values.get(key, {"ja": "このエンドでは他の技術項目を同時に変えない。", "en": "Do not change another technical variable during this end.", "zh": "这一组不要同时改动其他技术变量。"}))


def _confidence(lang: str, diagnosis: Dict[str, Any], quality: Optional[Dict[str, Any]], repeated: bool, self_report: str) -> Dict[str, str]:
    quality_score = float((quality or {}).get("score", 0.75) or 0.0)
    n = diagnosis["n"]
    strong_pattern = diagnosis["key"] == "single_outlier" or diagnosis["anisotropy"] >= 1.8 or diagnosis["offset_ratio"] >= 0.14
    if quality_score < 0.55 or n < 5:
        level = "low"
    elif repeated and (strong_pattern or self_report != "none"):
        level = "high"
    elif strong_pattern or self_report != "none":
        level = "medium"
    else:
        level = "low"
    labels = {
        "low": {"ja": "低", "en": "Low", "zh": "低"},
        "medium": {"ja": "中", "en": "Medium", "zh": "中"},
        "high": {"ja": "高", "en": "High", "zh": "高"},
    }
    reasons = {
        "low": {"ja": "少数矢または画像/形状の不確実性があるため、原因は仮説です。", "en": "The cause remains a hypothesis because of sample size or image/shape uncertainty.", "zh": "由于箭数较少或图像/形状存在不确定性，原因仍只是待验证假设。"},
        "medium": {"ja": "形状の傾向は明確ですが、1エンドだけでは原因を断定しません。", "en": "The pattern is clear, but one end is not enough to prove the cause.", "zh": "形状趋势较明确，但单独一组仍不足以证明具体原因。"},
        "high": {"ja": "同条件の履歴でも傾向が反復しています。それでも単一テストで確認します。", "en": "The pattern repeats in comparable saved ends; still verify it with one controlled test.", "zh": "相同条件的历史记录中也重复出现该趋势，但仍需用单变量测试确认。"},
    }
    return {"level": level, "label": _text(lang, labels[level]), "reason": _text(lang, reasons[level])}


def _profile_boosts(profile: Dict[str, Any]) -> Dict[str, float]:
    profile_text = " ".join(str(profile.get(k, "")) for k in ("goals", "recurring_issues", "constraints")).lower()
    groups = {
        "bow_hand_relaxed": ("grip", "bow hand", "握弓", "弓手", "グリップ"),
        "string_vertical": ("string", "弦", "弦像"),
        "release_from_back": ("release", "撒放", "离弦", "リリース", "離れ"),
        "anchor_balance": ("anchor", "锚点", "アンカー"),
        "body_alignment": ("shoulder", "肩", "alignment", "姿勢", "站姿"),
        "focus_trigger": ("focus", "mental", "集中", "心理", "专注"),
    }
    return {card_id: 3.0 for card_id, words in groups.items() if any(word in profile_text for word in words)}


def _pick_card(
    diagnosis: Dict[str, Any],
    profile: Dict[str, Any],
    log: List[Dict[str, Any]],
    trend: Dict[str, str],
    self_report: str,
) -> str:
    key = str(diagnosis["key"])
    candidates = list(DIAGNOSIS_CANDIDATES[key])
    recent_diagnoses = _recent_diagnoses(log)
    if len(recent_diagnoses) >= 2 and recent_diagnoses[:2] == [key, key] and key not in {"protect", "tight_offset"}:
        candidates.insert(0, "diary_experiment")
    if trend["key"] == "improving" and key not in {"tight_offset", "protect"}:
        candidates.insert(0, "finish_to_target")
    if trend["key"] == "worse" and key in {"loose", "reference"}:
        candidates.insert(0, "scoreless_groups")
    reported_card = SELF_REPORT_CARD.get(self_report)
    if reported_card:
        # A direct shot-feel report is stronger evidence about the movement
        # than geometry alone, so use it as the first controlled test.
        candidates.insert(0, reported_card)
    elif trend["key"] == "improving" and recent_diagnoses and recent_diagnoses[0] == key:
        # If the controlled check improved the same pattern, repeat it once
        # instead of changing the technical cue immediately.
        recent_sources = _recent_source_ids(log, limit=1)
        if recent_sources and recent_sources[0] in BOOK_CARDS:
            return recent_sources[0]

    boosts = _profile_boosts(profile)
    recent_sources = _recent_source_ids(log)
    scores: Dict[str, float] = {}
    for index, card_id in enumerate(candidates):
        score = 20.0 - index + boosts.get(card_id, 0.0)
        if card_id in recent_sources:
            score -= 8.0 - min(recent_sources.index(card_id), 3)
        scores[card_id] = max(scores.get(card_id, -999.0), score)
    return max(scores, key=lambda card_id: (scores[card_id], -candidates.index(card_id)))


def _localized_source(card: Dict[str, Any], lang: str, pdf_path: str) -> Dict[str, str]:
    return {"id": card["id"], "title": "Archery: The Art of Repetition — Simon Needham", "chapter": _text(lang, CHAPTERS[card["chapter"]]), "pdf_pages": card["pages"], "summary": _text(lang, card["summary"]), "file": os.path.basename(pdf_path)}


def _script(lang: str, cue: str, pass_fail: str, fallback: str, drill_name: str) -> str:
    labels = {"ja": ("一つの合図", "合格条件", "崩れたら", "すぐ行う練習"), "en": ("One cue", "Pass/fail", "If it breaks", "Immediate drill"), "zh": ("唯一口令", "通过/失败", "崩了就做", "立即练习")}
    a, b, c, d = labels.get(lang, labels["en"])
    return f"[{a}] {cue}\n[{b}] {pass_fail}\n[{c}] {fallback}\n[{d}] {drill_name}"


def _handedness_context(lang: str, handedness: str) -> str:
    if handedness == "left":
        return _text(lang, {
            "ja": "左利き：右手が弓手、左手が弦手です。",
            "en": "Left-handed: right hand holds the bow; left hand draws the string.",
            "zh": "左手弓：右手持弓，左手拉弦。",
        })
    return _text(lang, {
        "ja": "右利き：左手が弓手、右手が弦手です。",
        "en": "Right-handed: left hand holds the bow; right hand draws the string.",
        "zh": "右手弓：左手持弓，右手拉弦。",
    })


class CoachRAG:
    """Book-grounded deterministic coach; class name kept for compatibility."""

    def __init__(self, cfg: CoachConfig):
        self.cfg = cfg

    def enhance_advice(
        self,
        *,
        base_advice: Dict[str, Any],
        metrics: Dict[str, Any],
        shape: str,
        handedness: str,
        lang: str,
        scoring: Dict[str, Any],
        user_profile: Dict[str, Any],
        log: List[Dict[str, Any]],
        quality: Optional[Dict[str, Any]] = None,
        self_report: str = "none",
        session_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.cfg.mode == "rules":
            arrow_count = max(1, int(metrics.get("n", 6) or 6))
            out = dict(base_advice)
            for key in ("title", "single_cue", "pass_fail", "fallback", "why"):
                if isinstance(out.get(key), str):
                    out[key] = _adapt_arrow_count(out[key], lang, arrow_count)
            if isinstance(out.get("drill"), dict):
                out["drill"] = dict(out["drill"])
                for key in ("name", "how"):
                    if isinstance(out["drill"].get(key), str):
                        out["drill"][key] = _adapt_arrow_count(
                            out["drill"][key], lang, arrow_count
                        )
            return out

        comparable_log = _comparable_log(log, session_context)
        diagnosis = _diagnose(metrics, shape, scoring)
        arrow_count = max(1, int(diagnosis.get("n", 6) or 6))
        recent_diagnoses = _recent_diagnoses(comparable_log)
        repeated = bool(recent_diagnoses and recent_diagnoses[0] == diagnosis["key"])
        trend = _trend(lang, diagnosis, _last_metrics(comparable_log))
        previous_experiment = _previous_experiment(lang, diagnosis, trend, comparable_log)
        card_id = _pick_card(diagnosis, user_profile or {}, comparable_log, trend, self_report)

        bow = str((user_profile or {}).get("bow", "recurve")).lower()
        if card_id == "group_then_adjust" and bow == "barebow":
            card_id = "anchor_balance"

        card = BOOK_CARDS[card_id]
        cue = _adapt_arrow_count(_text(lang, card["cue"]), lang, arrow_count)
        pass_fail = _adapt_arrow_count(_text(lang, card["pass_fail"]), lang, arrow_count)
        fallback = _adapt_arrow_count(_text(lang, card["fallback"]), lang, arrow_count)
        drill_name = _adapt_arrow_count(_text(lang, card["drill_name"]), lang, arrow_count)
        drill_how = _adapt_arrow_count(_text(lang, card["drill_how"]), lang, arrow_count)
        source = _localized_source(card, lang, self.cfg.pdf_path)
        confidence = _confidence(lang, diagnosis, quality, repeated, self_report)
        self_report_label = _text(lang, SELF_REPORT_LABELS.get(self_report, SELF_REPORT_LABELS["none"]))

        out = dict(base_advice)
        out.update({
            "title": _text(lang, card["title"]),
            "cue": cue,
            "single_cue": cue,
            "pass_fail": pass_fail,
            "fallback": fallback,
            "why": _text(lang, card["why"]),
            "drill": {"name": drill_name, "how": drill_how, "duration_s": card["duration_s"]},
            "mental_phrase": _text(lang, card["mental"]),
            "stage": card["stage"],
            "tag": card_id,
            "script": _script(lang, cue, pass_fail, fallback, drill_name),
            "diagnosis": {
                "key": diagnosis["key"],
                "evidence": _evidence(lang, diagnosis),
                "trend": trend["text"],
                "trend_key": trend["key"],
                "handedness_context": _handedness_context(lang, handedness),
                "confidence": confidence["level"],
                "confidence_label": confidence["label"],
                "confidence_reason": confidence["reason"],
                "self_report": self_report,
                "self_report_label": self_report_label,
            },
            "feedback": {
                "success_criterion": _success_criterion(lang, diagnosis, arrow_count),
                "alternative_hypotheses": _alternative_hypotheses(lang, diagnosis),
                "do_not_change": _do_not_change(lang, diagnosis),
                "selected_by": "self_report" if self_report in SELF_REPORT_CARD else "group_pattern",
                "previous_experiment": previous_experiment,
            },
            "book_source": source,
            "book_sources": [source],
            "rag": {"engine": "reviewed_book_cards", "source_id": card_id, "diagnosis": diagnosis["key"], "history_used": bool(comparable_log)},
        })
        if quality is not None and float(quality.get("score", 1.0) or 0.0) < 0.55:
            out["diagnosis"]["evidence"] += _text(lang, {"ja": " 画像補正の信頼度が低いため、原因は仮説として扱ってください。", "en": " Image alignment confidence is low, so treat the cause as a hypothesis.", "zh": " 图像校正可信度较低，因此请把原因视为待验证假设。"})
        return out
