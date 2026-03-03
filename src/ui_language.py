import streamlit as st
from .i18n import t
from .state import goto_step


def render_language_step():
    st.title("Archery Next End")
    lang = st.selectbox(
        t("choose_language", "en"),
        options=["en", "ja", "zh"],
        format_func=lambda x: {"en": "English", "ja": "日本語", "zh": "中文"}[x],
        index=["en", "ja", "zh"].index(st.session_state.language),
    )
    st.session_state.language = lang

    if st.button(t("next", lang), use_container_width=True):
        goto_step("handedness")
        st.rerun()
