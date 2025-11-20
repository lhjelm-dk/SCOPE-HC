"""
Sub-page: Gross Rock Volume (GRV) Inputs
This is a sub-page under "Inputs" in the sidebar.
"""
import streamlit as st

# st.set_page_config is called in streamlit_app.py

from scopehc.ui.inputs_grv import render_grv


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/01_Inputs_Gross_Rock_Volume_GRV.py"

    st.header("Gross Rock Volume (GRV)")

    defaults = {
        "A": {"dist": "PERT", "min": 105.0, "mode": 210.0, "max": 273.0},
        "GCF": {"dist": "PERT", "min": 0.6, "mode": 0.68, "max": 0.85},
        "h": {"dist": "PERT", "min": 100.0, "mode": 130.0, "max": 150.0},
        "d": {"dist": "PERT", "min": 584.25, "mode": 615.0, "max": 645.75},
    }

    num_sims = int(st.session_state.get("num_sims", 10_000))
    show_inline_tips = bool(st.session_state.get("show_inline_tips", False))

    render_grv(num_sims, defaults, show_inline_tips)


if __name__ == "__main__":
    main()

