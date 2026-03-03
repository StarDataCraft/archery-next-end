import streamlit as st

from src.state import init_state, goto_step
from src.ui_language import render_language_step
from src.ui_handedness import render_handedness_step
from src.ui_analyze import render_analyze_step

st.set_page_config(page_title="Archery Next End Coach", layout="wide")

def main():
    init_state()

    step = st.session_state.step

    if step == "language":
        render_language_step()
    elif step == "handedness":
        render_handedness_step()
    elif step == "analyze":
        render_analyze_step()
    else:
        # fallback
        goto_step("language")
        st.rerun()

if __name__ == "__main__":
    main()
