import streamlit as st
from .state import goto_step
from .i18n import t

def render_handedness_step():
    lang = st.session_state.language
    st.title(t("choose_handedness", lang))

    handed = st.radio(
        label="",
        options=["right", "left"],
        format_func=lambda x: t(x, lang),
        index=0 if st.session_state.handedness == "right" else 1,
    )
    st.session_state.handedness = handed

    col1, col2 = st.columns([1,1])
    with col1:
        if st.button(t("back", lang), use_container_width=True):
            goto_step("language")
            st.rerun()
    with col2:
        if st.button(t("next", lang), use_container_width=True):
            goto_step("analyze")
            st.rerun()
