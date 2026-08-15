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

    "workflow_help": {
        "ja": "写真を追加すると、的を補正して着弾候補を表示します。最後に位置を確認して分析してください。",
        "en": "Add a photo to detect the target and suggest hit points. Confirm the points before analyzing.",
        "zh": "添加照片后，系统会校正靶面并建议命中点；请确认点位后再分析。",
    },
    "invalid_image": {
        "ja": "画像を読み込めませんでした。PNG または JPEG の別の写真をお試しください。",
        "en": "This image could not be read. Try another PNG or JPEG photo.",
        "zh": "无法读取这张图片，请换一张 PNG 或 JPEG 照片。",
    },
    "image_too_large": {
        "ja": "画像が大きすぎます。20 MB 以下の写真を選択してください。",
        "en": "The image is too large. Choose a photo under 20 MB.",
        "zh": "图片过大，请选择 20 MB 以下的照片。",
    },
    "processing_photo": {
        "ja": "写真を解析しています…",
        "en": "Analyzing the photo…",
        "zh": "正在分析照片…",
    },
    "cv_error": {
        "ja": "この写真を自動解析できませんでした。アプリは引き続き使用できます。",
        "en": "Automatic analysis could not process this photo. The app is still available.",
        "zh": "自动分析无法处理这张照片，但应用仍可继续使用。",
    },
    "cv_error_hint": {
        "ja": "的全体が明るく、正面から写っている別の写真をお試しください。",
        "en": "Try a well-lit, front-facing photo that includes the whole target.",
        "zh": "请尝试光线充足、正对靶面并包含完整靶纸的照片。",
    },
    "quality_warning": {
        "ja": "自動補正の信頼度が低めです。分析前に着弾点を確認・修正してください。",
        "en": "Automatic alignment confidence is low. Check and adjust the hit points before analyzing.",
        "zh": "自动校正可信度较低，请在分析前检查并调整命中点。",
    },
    "manual_points": {
        "ja": "着弾候補を検出できませんでした。的の上をクリックして位置を追加してください。",
        "en": "No hit candidates were found. Click the target to add the hit positions manually.",
        "zh": "未检测到命中点，请点击靶面手动添加点位。",
    },
    "marked": {
        "ja": "指定済み: **{count}** / {need}",
        "en": "Marked: **{count}** / {need}",
        "zh": "已标记：**{count}** / {need}",
    },

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
