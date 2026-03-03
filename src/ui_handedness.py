import streamlit as st
from .i18n import t
from .state import goto_step


def render_handedness_step():
    lang = st.session_state.language
    st.title(t("choose_handedness", lang))

    handed = st.radio(
        t("handedness", lang),
        options=["right", "left"],
        format_func=lambda x: t("right", lang) if x == "right" else t("left", lang),
        index=0 if st.session_state.handedness == "right" else 1,
    )
    st.session_state.handedness = handed

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(t("back", lang), use_container_width=True):
            goto_step("language")
            st.rerun()
    with c2:
        if st.button(t("next", lang), use_container_width=True):
            goto_step("analyze")
            st.rerun()
