from __future__ import annotations

import numpy as np
import streamlit as st

from .common import PALETTE, render_param, get_unit_system
from .helpers import render_fluid_estimator


def render_fluids(num_sims: int, defaults, show_inline_tips: bool) -> None:
    """Render the Fluids section and update session state."""
    unit_system = get_unit_system()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Fluids
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # Add Fluid Property Estimator helper
    render_fluid_estimator()

    st.info(
        "PVT correlations are simplified for teaching; not for design use. "
        "Consider classic Standing / Vasquez–Beggs equations before deployment."
    )

    # For now, only direct distribution input is supported
    st.session_state["fluid_method"] = "Direct distribution input"

    st.markdown("### Oil Properties")
    invbo_unit = "STB/rb" if unit_system == "oilfield" else "m³/m³"
    invbo_help = (
        "Oil shrinkage factor (stock tank barrels per reservoir barrel). "
        "Note: STB/rb is equivalent to STB/bbl in oilfield units."
        if unit_system == "oilfield"
        else "Oil shrinkage factor (cubic meters per cubic meter)"
    )
    sInvBo = render_param(
        "InvBo",
        "Oil Shrinkage Factor (1/Bo)",
        invbo_unit,
        "Triangular",
        {"min": 0.75, "mode": 0.80, "max": 0.85},
        num_sims,
        stats_decimals=4,
        help_text=invbo_help,
        param_name="InvBo",
    )

    st.markdown("### Gas Properties")
    bg_input_method = st.radio(
        "Gas expansion input method:",
        ["Enter 1/Bg (scf/cf) - Industry standard", "Enter Bg (rb/scf) - Traditional"],
        help=(
            "Industry standard uses 1/Bg (scf/cf) as primary input. "
            "Bg is reservoir barrel per standard cubic foot. 1/Bg is standard cubic feet per cubic foot."
        ),
    )

    if bg_input_method == "Enter 1/Bg (scf/cf) - Industry standard":
        invbg_unit = "scf/cf" if unit_system == "oilfield" else "m³/m³"
        invbg_help = (
            "Gas expansion factor (standard cubic feet per cubic foot) - industry standard primary input"
            if unit_system == "oilfield"
            else "Gas expansion factor (cubic meters per cubic meter)"
        )
        sInvBg = render_param(
            "InvBg",
            "Gas Expansion Factor (1/Bg)",
            invbg_unit,
            "Stretched Beta",
            {"min": 151, "mode": 162, "max": 214},
            num_sims,
            stats_decimals=4,
            help_text=invbg_help,
            param_name="InvBg",
        )

        if unit_system == "oilfield":
            sBg = 1.0 / (np.maximum(sInvBg, 1e-12) * 5.614583)
        else:
            sBg = 1.0 / np.maximum(sInvBg, 1e-12)

        mean_bg = np.mean(sBg)
        st.info(f"Internal Bg conversion: {mean_bg:.6f} rb/scf (calculated from 1/Bg input)")
    else:
        bg_unit = "rb/scf" if unit_system == "oilfield" else "m³/m³"
        bg_help = (
            "Gas expansion factor (reservoir barrel per standard cubic foot)"
            if unit_system == "oilfield"
            else "Gas expansion factor (cubic meters per cubic meter)"
        )
        sBg = render_param(
            "Bg",
            "Gas Expansion Factor (Bg)",
            bg_unit,
            "Stretched Beta",
            {"min": 0.003, "mode": 0.005, "max": 0.008},
            num_sims,
            stats_decimals=4,
            help_text=bg_help,
            param_name="Bg",
            context="invbg",
        )

    gor_unit = "scf/STB" if unit_system == "oilfield" else "m³/m³"
    gor_help = (
        "Gas-oil ratio for associated gas (standard cubic feet per stock tank barrel)"
        if unit_system == "oilfield"
        else "Gas-oil ratio (cubic meters per cubic meter)"
    )
    sGOR = render_param(
        "GOR",
        "GOR (Associated Gas)",
        gor_unit,
        "Triangular",
        {"min": 400, "mode": 500, "max": 600},
        num_sims,
        stats_decimals=4,
        help_text=gor_help,
        param_name="GOR",
    )

    st.markdown("### Condensate Properties")
    cy_unit = "STB/MMscf" if unit_system == "oilfield" else "m³/Mm³"
    cy_help = (
        "Condensate yield (stock tank barrels per million standard cubic feet of gas)"
        if unit_system == "oilfield"
        else "Condensate yield (cubic meters per million cubic meters of gas)"
    )
    sCY = render_param(
        "CY",
        "Condensate Yield (CY)",
        cy_unit,
        "Triangular",
        {"min": 30, "mode": 40, "max": 50},
        num_sims,
        stats_decimals=4,
        help_text=cy_help,
        param_name="CY",
    )
    st.caption("💡 Recovery Factors are configured in the Recovery Factor section below.")

    st.session_state["sBg"] = sBg
    st.session_state["sInvBo"] = sInvBo
    st.session_state["sGOR"] = sGOR
    st.session_state["sCY"] = sCY


def render_recovery(num_sims: int, defaults, show_inline_tips: bool) -> None:
    """Render the Recovery Factor section and update session state."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Recovery Factor (RF)
        </h2>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**Configure recovery factors for each hydrocarbon type:**")
    sRF_oil = render_param(
        "RF_oil",
        "Oil Recovery Factor (RF_oil)",
        "",
        "Triangular",
        {"min": 0.4, "mode": 0.55, "max": 0.7},
        num_sims,
        stats_decimals=4,
        help_text="Oil recovery factor (fraction of oil that can be recovered)",
        param_name="RF_oil",
    )

    sRF_gas = render_param(
        "RF_gas",
        "Free Gas Recovery Factor (RF_gas)",
        "",
        "Triangular",
        {"min": 0.6, "mode": 0.75, "max": 0.9},
        num_sims,
        stats_decimals=4,
        help_text="Free gas recovery factor (fraction of free gas that can be recovered)",
        param_name="RF_gas",
    )

    sRF_assoc_gas = render_param(
        "RF_assoc",
        "Associated Gas Recovery Factor (RF_assoc)",
        "",
        "Triangular",
        {"min": 0.4, "mode": 0.55, "max": 0.7},
        num_sims,
        stats_decimals=4,
        help_text="Associated gas recovery factor (fraction of associated gas that can be recovered). Default: same as oil recovery factor.",
        param_name="RF_assoc",
    )

    sRF_cond = render_param(
        "RF_cond",
        "Condensate Recovery Factor (RF_cond)",
        "",
        "Triangular",
        {"min": 0.6, "mode": 0.75, "max": 0.9},
        num_sims,
        stats_decimals=4,
        help_text="Condensate recovery factor (fraction of condensate that can be recovered)",
        param_name="RF_cond",
    )

    st.session_state["sRF_oil"] = sRF_oil
    st.session_state["sRF_gas"] = sRF_gas
    st.session_state["sRF_assoc_gas"] = sRF_assoc_gas
    st.session_state["sRF_cond"] = sRF_cond

