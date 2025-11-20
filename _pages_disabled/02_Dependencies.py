import streamlit as st

from scopehc.ui.inputs_dependencies import render_dependencies, apply_dependencies


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/02_Dependencies.py"

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

    render_dependencies(num_sims, defaults, show_inline_tips)
    apply_dependencies(num_sims)


if __name__ == "__main__":
    main()

