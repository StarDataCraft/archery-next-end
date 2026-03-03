import streamlit as st

from src.state import init_state
from src.ui_language import render_language_step
from src.ui_handedness import render_handedness_step
from src.ui_analyze import render_analyze_step


def main():
    st.set_page_config(page_title="Archery Next End", layout="wide")
    init_state()

    step = st.session_state.step
    if step == "language":
        render_language_step()
    elif step == "handedness":
        render_handedness_step()
    else:
        render_analyze_step()


if __name__ == "__main__":
    main()
