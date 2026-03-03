TEXT = {
    "title": {
        "ja": "次のエンド（次の的）アドバイス",
        "en": "Next End Advice",
        "zh": "下一靶建议",
    },
    "choose_language": {
        "ja": "言語を選択",
        "en": "Choose language",
        "zh": "选择语言",
    },
    "choose_handedness": {
        "ja": "利き手（弓手）を選択",
        "en": "Select handedness (bow hand)",
        "zh": "选择左右手（弓手）",
    },
    "right": {"ja": "右利き（右手弓）", "en": "Right-handed", "zh": "右手"},
    "left": {"ja": "左利き（左手弓）", "en": "Left-handed", "zh": "左手"},
    "next": {"ja": "次へ", "en": "Next", "zh": "下一步"},
    "back": {"ja": "戻る", "en": "Back", "zh": "返回"},
    "upload": {"ja": "的の写真をアップロード", "en": "Upload target photo", "zh": "上传靶面照片"},
    "tap_points": {
        "ja": "矢の位置をクリック（点で指定）",
        "en": "Confirm hits (edit points)",
        "zh": "确认命中点（编辑点位）",
    },
    "analyze": {"ja": "分析する", "en": "Analyze", "zh": "分析"},
    "clear": {"ja": "点をクリア", "en": "Clear points", "zh": "清空点位"},
    "save_log": {"ja": "ログに保存", "en": "Save to log", "zh": "保存到日志"},
    "distance": {"ja": "距離 (m)", "en": "Distance (m)", "zh": "距离 (m)"},
    "arrows": {"ja": "矢数/エンド", "en": "Arrows per end", "zh": "每靶箭数"},
    "need_points": {
        "ja": "先に矢の点を必要数だけ指定してください。",
        "en": "Please mark the required number of points first.",
        "zh": "请先标出足够数量的点。",
    },

    "target_face": {"ja": "的（ターゲットフェイス）", "en": "Target face", "zh": "靶面类型"},
    "target_80": {"ja": "80cm（屋外 / 10リング）", "en": "80cm (outdoor / 10-ring)", "zh": "80cm（户外 / 10环）"},
    "target_40": {"ja": "40cm（室内 / 10リング）", "en": "40cm (indoor / 10-ring)", "zh": "40cm（室内 / 10环）"},
}

def t(key: str, lang: str) -> str:
    d = TEXT.get(key, {})
    return d.get(lang, d.get("en", key))
