from __future__ import annotations

from pathlib import Path

import streamlit as st

# Navigation and theme are handled in streamlit_app.py


def render() -> None:
    """Render the math appendix page."""
    st.session_state["current_page"] = "_pages_disabled/07_Help_Math.py"

    st.header("Help & Math Appendix")
    md = Path(__file__).resolve().parents[2] / "README.md"
    try:
        txt = md.read_text(encoding="utf-8")
    except FileNotFoundError:
        st.error("README.md not found.")
        return

    # Show Glossary & Quick Reference section
    glossary_anchor = "## Glossary & Quick Reference"
    math_anchor = "## Math Appendix"
    
    if glossary_anchor in txt:
        glossary_section = txt.split(glossary_anchor, 1)[1]
        if math_anchor in glossary_section:
            glossary_section = glossary_section.split(math_anchor, 1)[0]
        st.markdown(glossary_anchor + glossary_section)
        st.markdown("---")
    
    # Show Math Appendix section
    if math_anchor in txt:
        math_section = txt.split(math_anchor, 1)[1]
        st.markdown(math_anchor + math_section)
    
    # Add disclaimer
    from scopehc.ui.common import render_disclaimer
    render_disclaimer()

