STRINGS = {
    "title": {"en": "Archery Next End", "ja": "次エンド提案", "zh": "下一靶建议"},
    "language": {"en": "Language", "ja": "言語", "zh": "语言"},
    "choose_language": {"en": "Choose language", "ja": "言語を選択", "zh": "选择语言"},
    "next": {"en": "Next", "ja": "次へ", "zh": "下一步"},
    "back": {"en": "Back", "ja": "戻る", "zh": "返回"},
    "handedness": {"en": "Handedness", "ja": "利き手", "zh": "左右手"},
    "choose_handedness": {"en": "Choose handedness", "ja": "利き手を選択", "zh": "选择左右手"},
    "right": {"en": "Right-handed", "ja": "右利き", "zh": "右手"},
    "left": {"en": "Left-handed", "ja": "左利き", "zh": "左手"},
    "distance": {"en": "Distance (m)", "ja": "距離 (m)", "zh": "距离 (m)"},
    "arrows": {"en": "Arrows / end", "ja": "1エンド本数", "zh": "每轮箭数"},
    "target_face": {"en": "Target face", "ja": "的紙", "zh": "靶面"},
    "target_80": {"en": "80cm outdoor (10-ring)", "ja": "80cm（屋外）", "zh": "80cm（户外）"},
    "target_40": {"en": "40cm indoor (10-ring)", "ja": "40cm（屋内）", "zh": "40cm（室内）"},
    "clear": {"en": "Clear", "ja": "リセット", "zh": "清空"},
    "upload": {"en": "Upload a target photo", "ja": "写真をアップロード", "zh": "上传靶面照片"},
    "tap_points": {"en": "Confirm hits (circles)", "ja": "着弾点の確認（丸）", "zh": "确认落点（画圈）"},
    "analyze": {"en": "Analyze", "ja": "解析", "zh": "分析"},
    "need_points": {
        "en": "Not enough points. Mark the required number of arrows.",
        "ja": "点が足りません。必要本数分をマークしてください。",
        "zh": "点数不足，请标记足够的箭数。",
    },
    "save_log": {"en": "Save to log", "ja": "ログに保存", "zh": "保存记录"},
}


def t(key: str, lang: str) -> str:
    lang = lang if lang in ("en", "ja", "zh") else "en"
    if key not in STRINGS:
        return key
    return STRINGS[key].get(lang, STRINGS[key].get("en", key))
