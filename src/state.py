import streamlit as st


def init_state():
    if "step" not in st.session_state:
        st.session_state.step = "language"

    if "language" not in st.session_state:
        st.session_state.language = "en"

    if "handedness" not in st.session_state:
        st.session_state.handedness = "right"

    if "distance_m" not in st.session_state:
        st.session_state.distance_m = 18

    # ✅ 默认仍为 3，但下限允许 1
    if "arrows_per_end" not in st.session_state:
        st.session_state.arrows_per_end = 3

    if "target_face" not in st.session_state:
        st.session_state.target_face = "80cm_10ring"

    if "points" not in st.session_state:
        st.session_state.points = []

    if "auto_points" not in st.session_state:
        st.session_state.auto_points = []

    if "auto_contact_colors" not in st.session_state:
        st.session_state.auto_contact_colors = []

    if "overlay_image_rgb" not in st.session_state:
        st.session_state.overlay_image_rgb = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "log" not in st.session_state:
        st.session_state.log = []

    if "cv_cache_key" not in st.session_state:
        st.session_state.cv_cache_key = None


def goto_step(step: str):
    st.session_state.step = step


def reset_shot():
    st.session_state.points = []
    st.session_state.auto_points = []
    st.session_state.auto_contact_colors = []
    st.session_state.last_result = None


def reset_cv_cache():
    st.session_state.cv_cache_key = None
    st.session_state.overlay_image_rgb = None
    st.session_state._rect_photo_rgb = None
    st.session_state.warp_debug = None
