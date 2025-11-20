"""
Sub-page: Net-to-Gross (NtG) Inputs
This is a sub-page under "Inputs" in the sidebar.
"""
import streamlit as st

# st.set_page_config is called in streamlit_app.py

from scopehc.ui.inputs_reservoir import render_ntg


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/01_Inputs_Net_to_Gross_NtG.py"

    st.header("Net-to-Gross (NtG)")

    defaults = {
        "NtG": {"dist": "PERT", "min": 0.27, "mode": 0.40, "max": 0.50},
    }

    num_sims = int(st.session_state.get("num_sims", 10_000))
    show_inline_tips = bool(st.session_state.get("show_inline_tips", False))

    render_ntg(num_sims, defaults, show_inline_tips)


if __name__ == "__main__":
    main()

