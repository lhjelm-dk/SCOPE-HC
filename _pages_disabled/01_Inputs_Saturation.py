"""
Sub-page: Saturation Inputs
This is a sub-page under "Inputs" in the sidebar.
"""
import streamlit as st

# st.set_page_config is called in streamlit_app.py

from scopehc.ui.inputs_reservoir import render_saturation_inputs


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/01_Inputs_Saturation.py"

    st.header("Saturation")

    render_saturation_inputs()


if __name__ == "__main__":
    main()

