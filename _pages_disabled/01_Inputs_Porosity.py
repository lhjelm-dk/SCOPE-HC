"""
Sub-page: Porosity Inputs
This is a sub-page under "Inputs" in the sidebar.
"""
import streamlit as st

# st.set_page_config is called in streamlit_app.py

from scopehc.ui.inputs_reservoir import render_porosity


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/01_Inputs_Porosity.py"

    st.header("Porosity")

    defaults = {
        "p": {"dist": "PERT", "min": 0.08, "mode": 0.18, "max": 0.21},
    }

    num_sims = int(st.session_state.get("num_sims", 10_000))
    show_inline_tips = bool(st.session_state.get("show_inline_tips", False))

    render_porosity(num_sims, defaults, show_inline_tips)


if __name__ == "__main__":
    main()

