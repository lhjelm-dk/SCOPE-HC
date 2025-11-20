"""
Sub-page: Fluids Inputs
This is a sub-page under "Inputs" in the sidebar.
"""
import streamlit as st

# st.set_page_config is called in streamlit_app.py

from scopehc.ui.inputs_fluids import render_fluids


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/01_Inputs_Fluids.py"

    st.header("Fluids")

    defaults = {
        "A": {"dist": "PERT", "min": 105.0, "mode": 210.0, "max": 273.0},
        "GCF": {"dist": "PERT", "min": 0.6, "mode": 0.68, "max": 0.85},
        "h": {"dist": "PERT", "min": 100.0, "mode": 130.0, "max": 150.0},
        "NtG": {"dist": "PERT", "min": 0.27, "mode": 0.40, "max": 0.50},
        "p": {"dist": "PERT", "min": 0.08, "mode": 0.18, "max": 0.21},
        "SE": {"dist": "PERT", "min": 0.03, "mode": 0.10, "max": 0.20},
        "d": {"dist": "PERT", "min": 584.25, "mode": 615.0, "max": 645.75},
    }

    num_sims = int(st.session_state.get("num_sims", 10_000))
    show_inline_tips = bool(st.session_state.get("show_inline_tips", False))

    render_fluids(num_sims, defaults, show_inline_tips)


if __name__ == "__main__":
    main()

