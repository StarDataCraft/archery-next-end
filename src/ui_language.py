import streamlit as st
from .state import goto_step
from .i18n import t

def render_language_step():
    st.title(t("choose_language", st.session_state.language))

    lang = st.radio(
        label="",
        options=["ja", "en", "zh"],
        format_func=lambda x: {"ja": "日本語", "en": "English", "zh": "中文"}[x],
        index=["ja", "en", "zh"].index(st.session_state.language),
    )
    st.session_state.language = lang

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button(t("next", lang), use_container_width=True):
            goto_step("handedness")
            st.rerun()
