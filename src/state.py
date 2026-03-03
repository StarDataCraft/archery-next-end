import streamlit as st

DEFAULTS = {
    "step": "language",
    "language": "ja",
    "handedness": "right",   # right / left
    "distance_m": 18,
    "arrows_per_end": 6,
    "points": [],            # list of dicts: {"x": float, "y": float}
    "last_result": None,     # dict metrics + advice
    "log": [],               # list of dict (history)
}

def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_shot():
    st.session_state.points = []
    st.session_state.last_result = None

def goto_step(step: str):
    st.session_state.step = step
