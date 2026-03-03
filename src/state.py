# src/state.py
from __future__ import annotations

import streamlit as st


def init_state():
    if "step" not in st.session_state:
        st.session_state.step = "handedness"

    if "language" not in st.session_state:
        st.session_state.language = "en"

    if "handedness" not in st.session_state:
        st.session_state.handedness = "right"

    if "distance_m" not in st.session_state:
        st.session_state.distance_m = 18

    if "arrows_per_end" not in st.session_state:
        st.session_state.arrows_per_end = 6

    if "target_face" not in st.session_state:
        st.session_state.target_face = "80cm_10ring"

    if "log" not in st.session_state:
        st.session_state.log = []

    # -----------------------------
    # NEW: coaching flexibility settings
    # -----------------------------
    if "coach_mode" not in st.session_state:
        # "rules" | "rag" | "rag_llm"
        st.session_state.coach_mode = "rag"

    if "coach_pdf_path" not in st.session_state:
        st.session_state.coach_pdf_path = "docs/Archery_The_Art_of_Repetition.pdf"

    if "coach_gguf_path" not in st.session_state:
        st.session_state.coach_gguf_path = "models/llm.gguf"


def goto_step(step: str):
    st.session_state.step = step


def reset_shot():
    st.session_state.points = []
    st.session_state.last_result = None


def reset_cv_cache():
    # keep your existing keys but safe
    for k in [
        "cv_cache_key",
        "overlay_image_rgb",
        "auto_points",
        "_rect_photo_bgr",
        "_M_rect_to_canon",
        "warp_debug",
        "cv_quality",
    ]:
        if k in st.session_state:
            del st.session_state[k]
