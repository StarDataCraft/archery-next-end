# src/i18n.py
TEXT = {
    "title": {"ja": "次のエンド（次の的）アドバイス", "en": "Next End Advice", "zh": "下一靶建议"},

    "choose_language": {"ja": "言語を選択", "en": "Choose language", "zh": "选择语言"},
    "choose_handedness": {"ja": "利き手（弓手）を選択", "en": "Select handedness (bow hand)", "zh": "选择左右手（弓手）"},

    "right": {"ja": "右利き（右手弓）", "en": "Right-handed", "zh": "右手"},
    "left": {"ja": "左利き（左手弓）", "en": "Left-handed", "zh": "左手"},

    "next": {"ja": "次へ", "en": "Next", "zh": "下一步"},
    "back": {"ja": "戻る", "en": "Back", "zh": "返回"},

    "input_mode": {"ja": "入力方法", "en": "Input mode", "zh": "输入方式"},
    "mode_upload": {"ja": "アップロード", "en": "Upload", "zh": "上传"},
    "mode_camera": {"ja": "撮影（カメラ）", "en": "Take photo (camera)", "zh": "拍照（相机）"},

    "upload": {"ja": "的の写真をアップロード", "en": "Upload target photo", "zh": "上传靶面照片"},
    "camera": {"ja": "カメラで撮影", "en": "Take a photo", "zh": "用相机拍照"},

    "tap_points": {"ja": "矢の位置をクリック（点で指定）", "en": "Confirm hits (edit points)", "zh": "确认命中点（编辑点位）"},
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
    "target_80_10": {"ja": "80cm（10リング）", "en": "80cm (10-ring)", "zh": "80cm（10环）"},
    "target_40_10": {"ja": "40cm（10リング）", "en": "40cm (10-ring)", "zh": "40cm（10环）"},
    "target_60_10": {"ja": "60cm（10リング）", "en": "60cm (10-ring)", "zh": "60cm（10环）"},
    "target_122_10": {"ja": "122cm（10リング）", "en": "122cm (10-ring)", "zh": "122cm（10环）"},

    "profile": {"ja": "コーチング用プロフィール", "en": "Coaching profile", "zh": "教练档案"},
    "profile_name": {"ja": "名前（任意）", "en": "Name (optional)", "zh": "名字（可选）"},
    "profile_bow": {"ja": "弓種", "en": "Bow type", "zh": "弓种"},
    "profile_exp": {"ja": "経験（月）", "en": "Experience (months)", "zh": "练习时长（月）"},
    "profile_eye": {"ja": "利き目（任意）", "en": "Dominant eye (optional)", "zh": "优势眼（可选）"},
    "profile_goals": {"ja": "目標", "en": "Goals", "zh": "目标"},
    "profile_issues": {"ja": "よくある課題", "en": "Recurring issues", "zh": "常见问题"},
    "profile_constraints": {"ja": "制約（痛み/時間など）", "en": "Constraints (pain/time/etc.)", "zh": "限制（疼痛/时间等）"},
    "profile_style": {"ja": "言い方", "en": "Style", "zh": "表达风格"},
    "style_tight": {"ja": "短く厳密", "en": "Tight & precise", "zh": "短而精准"},
    "style_gentle": {"ja": "やさしく", "en": "Gentle", "zh": "温和"},
    "style_technical": {"ja": "技術的", "en": "Technical", "zh": "技术向"},
}

def t(key: str, lang: str) -> str:
    d = TEXT.get(key, {})
    return d.get(lang, d.get("en", key))
