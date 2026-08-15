# src/state.py
from copy import deepcopy

import streamlit as st

DEFAULTS = {
    "step": "language",
    "language": "ja",  # "ja" | "en" | "zh"
    "handedness": "right",  # right / left

    "distance_m": 30,
    "arrows_per_end": 6,

    # NEW: image input mode
    "image_mode": "upload",  # "upload" | "camera"

    # Target face (expandable)
    "target_face": "80cm_10ring",

    # user-confirmed hit points
    "points": [],  # list of dicts: {"x": float, "y": float}
    "canvas_revision": 0,

    # last result: metrics + scoring + advice
    "last_result": None,

    # Optional shot-feel observation for the current end. Geometry can show a
    # pattern, but the archer's own observation is needed to test a cause.
    "end_self_report": "none",

    # history
    "log": [],

    # cv cache
    "cv_cache_key": None,
    "warped_image_rgb": None,
    "overlay_image_rgb": None,
    "auto_points": None,
    "warp_debug": None,

    # NEW: coaching profile ("之前登录的内容")
    "user_profile": {
        "name": "",
        "bow": "recurve",
        "experience_months": 3,
        "dominant_eye": "",
        "goals": "",
        "recurring_issues": "",
        "constraints": "",
        "language_style": "tight",  # "tight" | "gentle" | "technical"
    },

    # NEW: coach settings
    "coach_version": 2,
    "coach_mode": "book",  # "book" | "rules"
    "coach_pdf_path": "docs/Archery The Art of Repetition (Simon Needham ).pdf",
    "coach_gguf_path": "models/llm.gguf",
    "coach_router": "fine",  # "coarse" | "fine"
}


def init_state():
    needs_coach_migration = "coach_version" not in st.session_state
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            # Lists and profile dictionaries must not be shared by concurrent
            # Streamlit sessions.
            st.session_state[k] = deepcopy(v)
    if needs_coach_migration:
        # Sessions created before v2 defaulted to the four-template rules
        # coach. Migrate once; later user choices between book/rules persist.
        st.session_state["coach_mode"] = "book"


def reset_shot():
    st.session_state.points = []
    st.session_state.last_result = None
    st.session_state.end_self_report = "none"


def reset_cv_cache():
    st.session_state.cv_cache_key = None
    st.session_state.warped_image_rgb = None
    st.session_state.overlay_image_rgb = None
    st.session_state.auto_points = None
    st.session_state.warp_debug = None
    st.session_state.cv_quality = None
    st.session_state._geom_center = None
    st.session_state._geom_outer = None
    st.session_state._rect_photo_bgr = None
    st.session_state._M_rect_to_canon = None


def goto_step(step: str):
    st.session_state.step = step
