import streamlit as st

DEFAULTS = {
    "step": "language",
    "language": "ja",
    "handedness": "right",       # right / left
    "distance_m": 30,
    "arrows_per_end": 6,

    # 目标靶型：默认 80cm outdoor
    # values: "80cm_10ring" or "40cm_10ring"
    "target_face": "80cm_10ring",

    "points": [],                # list of dicts: {"x": float, "y": float}
    "last_result": None,         # dict metrics + scoring + advice
    "log": [],                   # history

    # cv cache
    "cv_cache_key": None,
    "warped_image_rgb": None,
    "overlay_image_rgb": None,
    "auto_points": None,
    "warp_debug": None,
}

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_shot():
    st.session_state.points = []
    st.session_state.last_result = None

def reset_cv_cache():
    st.session_state.cv_cache_key = None
    st.session_state.warped_image_rgb = None
    st.session_state.overlay_image_rgb = None
    st.session_state.auto_points = None
    st.session_state.warp_debug = None

def goto_step(step: str):
    st.session_state.step = step
