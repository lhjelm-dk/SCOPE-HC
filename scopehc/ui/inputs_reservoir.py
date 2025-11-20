from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import numpy as np

from .common import *  # noqa: F401,F403
from scopehc.config import PARAM_COLORS


def render_ntg(
    num_sims: int,
    defaults: Dict[str, Any],
    show_inline_tips: bool,
) -> None:
    """Render Net-to-Gross (NtG) input section only."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Net-to-Gross (NtG)
        </h2>
        <p style='color:{PALETTE["text_secondary"]};font-style:italic;margin-bottom:1em;'>
            The ratio of reservoir-quality rock (net) to the total rock interval (gross).
        </p>
        """,
        unsafe_allow_html=True,
    )

    sNtG = render_param(
        "NtG",
        "Net-to-Gross NtG",
        "fraction",
        defaults["NtG"]["dist"],
        defaults["NtG"],
        num_sims,
        stats_decimals=3,
    )
    sNtG = sanitize(sNtG, "NtG", min_allowed=0.0, max_allowed=1.0, fill=0.5, warn=st.caption)
    sNtG = clip01(sNtG)
    st.session_state["sNtG"] = sNtG

    if show_inline_tips:
        st.info("NtG is often estimated from net sand counts or pay flags in the reservoir interval.")


def render_porosity(
    num_sims: int,
    defaults: Dict[str, Any],
    show_inline_tips: bool,
) -> None:
    """Render Porosity input section only."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Porosity
        </h2>
        <p style='color:{PALETTE["text_secondary"]};font-style:italic;margin-bottom:1em;'>
            Effective porosity represents pore space available for hydrocarbons (≈ 1 − Sw<sub>ir</sub>).
        </p>
        """,
        unsafe_allow_html=True,
    )

    sp = render_param(
        "p",
        "Porosity p",
        "fraction",
        defaults["p"]["dist"],
        defaults["p"],
        num_sims,
        stats_decimals=3,
    )
    sp = sanitize(sp, "Porosity", min_allowed=0.0, max_allowed=1.0, fill=0.2, warn=st.caption)
    sp = clip01(sp)
    st.session_state["sp"] = sp


def render_reservoir(
    num_sims: int,
    defaults: Dict[str, Any],
    show_inline_tips: bool,
) -> None:
    """Render Net-to-Gross, Porosity, and Saturation input sections (all together)."""
    render_ntg(num_sims, defaults, show_inline_tips)
    render_porosity(num_sims, defaults, show_inline_tips)
    render_saturation_inputs()


def render_saturation_inputs() -> None:
    """Render saturation input section with three modes."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Saturation
        </h2>
        """,
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Choose saturation input mode",
        options=[
            "Global",
            "Water saturation Per zone",
            "Per phase",
        ],
        index=0,
        key="sat_mode",
        horizontal=False,
    )

    # ---- Mode A: Global ----
    if mode.startswith("Global"):
        st.markdown("**Global saturation** for all HC zones.")
        st.latex(r"S_{\mathrm{hc}} = 1 - S_{w,\mathrm{global}}")
        
        # Sub-mode: S_hc or S_w,global
        global_submode = st.radio(
            "Input type",
            options=["$S_{\\mathrm{hc}}$ (hydrocarbon saturation)", "$S_{w,\\mathrm{global}}$ (water saturation)"],
            index=0,
            key="global_sat_submode",
            horizontal=True,
        )
        
        use_sw_global = global_submode.startswith("$S_{w")
        
        if use_sw_global:
            st.markdown("Enter $S_{w,\\mathrm{global}}$ (fraction):")
            # Default: constant value 0.0 (which gives Shc = 1.0)
            default_val = 0.0
        else:
            st.markdown("Enter $S_{\\mathrm{hc}}$ (fraction):")
            # Default: constant value 1.0
            default_val = 1.0

        # Distribution type selector
        dist_type = st.selectbox(
            "Distribution type",
            DistributionChoiceWithConstant,
            index=0,  # Default to Constant
            key="Shc_global_dist_type",
        )

        if dist_type == "Constant":
            const_val = st.number_input(
                "Value",
                min_value=0.0,
                max_value=1.0,
                value=default_val,
                step=0.01,
                key="Shc_global_const",
            )
            st.session_state["Shc_global_dist"] = {
                "type": "Constant",
                "value": const_val,
            }
            # Show constant value plot
            from .common import make_hist_cdf_figure
            const_array = np.full(100, const_val)  # Sample array for display
            if use_sw_global:
                title = r"$S_{w,\mathrm{global}}$ distribution"
                xlabel = r"$S_{w,\mathrm{global}}$"
            else:
                title = r"$S_{\mathrm{hc}}$ distribution"
                xlabel = r"$S_{\mathrm{hc}}$"
            st.plotly_chart(
                make_hist_cdf_figure(const_array, title, xlabel, "input"),
                use_container_width=True,
            )
        else:
            # For PERT/Triangular/Uniform, use custom UI
            # For all other distributions, use render_param()
            if dist_type in ["PERT", "Triangular", "Uniform"]:
                # PERT/Triangular/Uniform
                if dist_type == "PERT":
                    min_v, mode_v, max_v = (0.0, 0.0, 0.0) if use_sw_global else (0.6, 0.8, 0.9)
                elif dist_type == "Triangular":
                    min_v, mode_v, max_v = (0.0, 0.1, 0.2) if use_sw_global else (0.6, 0.8, 0.9)
                else:  # Uniform
                    min_v, max_v = (0.0, 0.2) if use_sw_global else (0.6, 0.9)
                    mode_v = (min_v + max_v) / 2
                
                st.markdown(f"**Distribution for $S_{{{'w,global' if use_sw_global else 'hc'}}}$:**")
                
                if dist_type == "PERT":
                    st.session_state["Shc_global_dist"] = {
                        "type": "PERT",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_global_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_global_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_global_max",
                    ),
                }
            elif dist_type == "Triangular":
                st.session_state["Shc_global_dist"] = {
                    "type": "Triangular",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_global_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_global_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_global_max",
                    ),
                }
            else:  # Uniform
                st.session_state["Shc_global_dist"] = {
                    "type": "Uniform",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_global_min",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_global_max",
                    ),
                }
            
            # Always show distribution plot
            num_sims = int(st.session_state.get("num_sims", 10_000))
            param_name = "Sw_global" if use_sw_global else "Shc_global"
            if use_sw_global:
                title = r"$S_{w,\mathrm{global}}$ distribution"
                xlabel = r"$S_{w,\mathrm{global}}$"
            else:
                title = r"$S_{\mathrm{hc}}$ distribution"
                xlabel = r"$S_{\mathrm{hc}}$"
            dist_config = st.session_state["Shc_global_dist"]
            
            # Sample and display
            from scopehc.sampling import rng_from_seed, sample_pert, sample_triangular, sample_uniform
            seed = st.session_state.get("random_seed", 12345)
            rng = rng_from_seed(seed)
            
            if dist_type == "PERT":
                samples = sample_pert(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            elif dist_type == "Triangular":
                samples = sample_triangular(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            else:  # Uniform
                samples = sample_uniform(rng, dist_config["min"], dist_config["max"], num_sims)
            
            samples = np.clip(samples, 0.0, 1.0)
            st.session_state[f"s{param_name}"] = samples
            
            from .common import make_hist_cdf_figure
            st.plotly_chart(
                make_hist_cdf_figure(samples, title, xlabel, "input"),
                use_container_width=True,
            )
        
        # Store submode
        st.session_state["global_sat_use_sw"] = use_sw_global

    # ---- Mode B: Water saturation Per zone ----
    elif mode.startswith("Water saturation"):
        st.markdown("**Water saturation per zone** (converted to HC saturation internally):")
        st.latex(
            r"S_{\mathrm{oil}} = 1 - S_{w,\mathrm{oil\,zone}}\quad;\quad S_{\mathrm{gas}} = 1 - S_{w,\mathrm{gas\,zone}}"
        )

        st.markdown(r"**Distribution for $S_{w,\mathrm{oil\,zone}}$:**")
        dist_type_oil = st.selectbox(
            "Distribution type",
            DistributionChoiceWithConstant,
            index=1,  # Default to PERT
            key="Sw_oilzone_dist_type",
        )
        
        if dist_type_oil == "Constant":
            const_val = st.number_input(
                "Value",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                key="Sw_oilzone_const",
            )
            st.session_state["Sw_oilzone_dist"] = {
                "type": "Constant",
                "value": const_val,
            }
            # Show plot
            from .common import make_hist_cdf_figure
            const_array = np.full(100, const_val)
            st.plotly_chart(
                make_hist_cdf_figure(const_array, r"$S_{w,\mathrm{oil\,zone}}$ distribution", 
                                    r"$S_{w,\mathrm{oil\,zone}}$", "input"),
                use_container_width=True,
            )
        else:
            if dist_type_oil == "PERT":
                min_v, mode_v, max_v = 0.25, 0.35, 0.50
            elif dist_type_oil == "Triangular":
                min_v, mode_v, max_v = 0.25, 0.35, 0.50
            else:  # Uniform
                min_v, max_v = 0.25, 0.50
                mode_v = (min_v + max_v) / 2
            
            if dist_type_oil == "PERT":
                st.session_state["Sw_oilzone_dist"] = {
                    "type": "PERT",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_oilzone_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Sw_oilzone_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_oilzone_max",
                    ),
                }
            elif dist_type_oil == "Triangular":
                st.session_state["Sw_oilzone_dist"] = {
                    "type": "Triangular",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_oilzone_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Sw_oilzone_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_oilzone_max",
                    ),
                }
            else:  # Uniform
                st.session_state["Sw_oilzone_dist"] = {
                    "type": "Uniform",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_oilzone_min",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_oilzone_max",
                    ),
                }
            
            # Always show distribution plot
            from scopehc.sampling import rng_from_seed, sample_pert, sample_triangular, sample_uniform
            num_sims = int(st.session_state.get("num_sims", 10_000))
            seed = st.session_state.get("random_seed", 12345)
            rng = rng_from_seed(seed)
            dist_config = st.session_state["Sw_oilzone_dist"]
            
            if dist_type_oil == "PERT":
                samples = sample_pert(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            elif dist_type_oil == "Triangular":
                samples = sample_triangular(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            else:  # Uniform
                samples = sample_uniform(rng, dist_config["min"], dist_config["max"], num_sims)
            
            samples = np.clip(samples, 0.0, 1.0)
            st.session_state["sSw_oilzone"] = samples
            
            from .common import make_hist_cdf_figure
            st.plotly_chart(
                make_hist_cdf_figure(samples, r"$S_{w,\mathrm{oil\,zone}}$ distribution", 
                                    r"$S_{w,\mathrm{oil\,zone}}$", "input"),
                use_container_width=True,
            )

        st.markdown(r"**Distribution for $S_{w,\mathrm{gas\,zone}}$:**")
        dist_type_gas = st.selectbox(
            "Distribution type",
            DistributionChoiceWithConstant,
            index=1,  # Default to PERT
            key="Sw_gaszone_dist_type",
        )
        
        if dist_type_gas == "Constant":
            const_val = st.number_input(
                "Value",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="Sw_gaszone_const",
            )
            st.session_state["Sw_gaszone_dist"] = {
                "type": "Constant",
                "value": const_val,
            }
            # Show plot
            from .common import make_hist_cdf_figure
            const_array = np.full(100, const_val)
            st.plotly_chart(
                make_hist_cdf_figure(const_array, r"$S_{w,\mathrm{gas\,zone}}$ distribution", 
                                    r"$S_{w,\mathrm{gas\,zone}}$", "input"),
                use_container_width=True,
            )
        else:
            if dist_type_gas == "PERT":
                min_v, mode_v, max_v = 0.10, 0.15, 0.25
            elif dist_type_gas == "Triangular":
                min_v, mode_v, max_v = 0.10, 0.15, 0.25
            else:  # Uniform
                min_v, max_v = 0.10, 0.25
                mode_v = (min_v + max_v) / 2
            
            if dist_type_gas == "PERT":
                st.session_state["Sw_gaszone_dist"] = {
                    "type": "PERT",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_gaszone_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Sw_gaszone_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_gaszone_max",
                    ),
                }
            elif dist_type_gas == "Triangular":
                st.session_state["Sw_gaszone_dist"] = {
                    "type": "Triangular",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_gaszone_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Sw_gaszone_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_gaszone_max",
                    ),
                }
            else:  # Uniform
                st.session_state["Sw_gaszone_dist"] = {
                    "type": "Uniform",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Sw_gaszone_min",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Sw_gaszone_max",
                    ),
                }
            
            # Always show distribution plot
            from scopehc.sampling import rng_from_seed, sample_pert, sample_triangular, sample_uniform
            num_sims = int(st.session_state.get("num_sims", 10_000))
            seed = st.session_state.get("random_seed", 12345)
            rng = rng_from_seed(seed)
            dist_config = st.session_state["Sw_gaszone_dist"]
            
            if dist_type_gas == "PERT":
                samples = sample_pert(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            elif dist_type_gas == "Triangular":
                samples = sample_triangular(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            else:  # Uniform
                samples = sample_uniform(rng, dist_config["min"], dist_config["max"], num_sims)
            
            samples = np.clip(samples, 0.0, 1.0)
            st.session_state["sSw_gaszone"] = samples
            
            from .common import make_hist_cdf_figure
            st.plotly_chart(
                make_hist_cdf_figure(samples, r"$S_{w,\mathrm{gas\,zone}}$ distribution", 
                                    r"$S_{w,\mathrm{gas\,zone}}$", "input"),
                use_container_width=True,
            )

    # ---- Mode C: Per-phase HC saturations ----
    else:
        st.markdown("**Per-phase hydrocarbon saturation** (direct entry):")
        st.latex(r"S_{\mathrm{oil}},\; S_{\mathrm{gas}} \in [0,1]")

        st.markdown(r"**Distribution for $S_{\mathrm{oil}}$:**")
        dist_type_oil = st.selectbox(
            "Distribution type",
            DistributionChoiceWithConstant,
            index=1,  # Default to PERT
            key="Shc_oil_input_dist_type",
        )
        
        if dist_type_oil == "Constant":
            const_val = st.number_input(
                "Value",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.01,
                key="Shc_oil_input_const",
            )
            st.session_state["Shc_oil_input_dist"] = {
                "type": "Constant",
                "value": const_val,
            }
            # Show plot
            from .common import make_hist_cdf_figure
            const_array = np.full(100, const_val)
            st.plotly_chart(
                make_hist_cdf_figure(const_array, r"$S_{\mathrm{oil}}$ distribution", 
                                    r"$S_{\mathrm{oil}}$", "input"),
                use_container_width=True,
            )
        else:
            if dist_type_oil == "PERT":
                min_v, mode_v, max_v = 0.6, 0.8, 0.9
            elif dist_type_oil == "Triangular":
                min_v, mode_v, max_v = 0.6, 0.8, 0.9
            else:  # Uniform
                min_v, max_v = 0.6, 0.9
                mode_v = (min_v + max_v) / 2
            
            if dist_type_oil == "PERT":
                st.session_state["Shc_oil_input_dist"] = {
                    "type": "PERT",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_oil_input_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_oil_input_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_oil_input_max",
                    ),
                }
            elif dist_type_oil == "Triangular":
                st.session_state["Shc_oil_input_dist"] = {
                    "type": "Triangular",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_oil_input_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_oil_input_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_oil_input_max",
                    ),
                }
            else:  # Uniform
                st.session_state["Shc_oil_input_dist"] = {
                    "type": "Uniform",
                    "min": st.number_input(
                        "min",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_oil_input_min",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_oil_input_max",
                    ),
                }
            
            # Always show distribution plot
            from scopehc.sampling import rng_from_seed, sample_pert, sample_triangular, sample_uniform
            num_sims = int(st.session_state.get("num_sims", 10_000))
            seed = st.session_state.get("random_seed", 12345)
            rng = rng_from_seed(seed)
            dist_config = st.session_state["Shc_oil_input_dist"]
            
            if dist_type_oil == "PERT":
                samples = sample_pert(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            elif dist_type_oil == "Triangular":
                samples = sample_triangular(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            else:  # Uniform
                samples = sample_uniform(rng, dist_config["min"], dist_config["max"], num_sims)
            
            samples = np.clip(samples, 0.0, 1.0)
            st.session_state["sShc_oil_input"] = samples
            
            from .common import make_hist_cdf_figure
            st.plotly_chart(
                make_hist_cdf_figure(samples, r"$S_{\mathrm{oil}}$ distribution", 
                                    r"$S_{\mathrm{oil}}$", "input"),
                use_container_width=True,
            )

        st.markdown(r"**Distribution for $S_{\mathrm{gas}}$:**")
        dist_type_gas = st.selectbox(
            "Distribution type",
            DistributionChoiceWithConstant,
            index=1,  # Default to PERT
            key="Shc_gas_input_dist_type",
        )
        
        if dist_type_gas == "Constant":
            const_val = st.number_input(
                "Value",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.01,
                key="Shc_gas_input_const",
            )
            st.session_state["Shc_gas_input_dist"] = {
                "type": "Constant",
                "value": const_val,
            }
            # Show plot
            from .common import make_hist_cdf_figure
            const_array = np.full(100, const_val)
            st.plotly_chart(
                make_hist_cdf_figure(const_array, r"$S_{\mathrm{gas}}$ distribution", 
                                    r"$S_{\mathrm{gas}}$", "input"),
                use_container_width=True,
            )
        else:
            if dist_type_gas == "PERT":
                min_v, mode_v, max_v = 0.6, 0.8, 0.9
            elif dist_type_gas == "Triangular":
                min_v, mode_v, max_v = 0.6, 0.8, 0.9
            else:  # Uniform
                min_v, max_v = 0.6, 0.9
                mode_v = (min_v + max_v) / 2
            
            if dist_type_gas == "PERT":
                st.session_state["Shc_gas_input_dist"] = {
                    "type": "PERT",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_gas_input_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_gas_input_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_gas_input_max",
                    ),
                }
            elif dist_type_gas == "Triangular":
                st.session_state["Shc_gas_input_dist"] = {
                    "type": "Triangular",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_gas_input_min",
                    ),
                    "mode": st.number_input(
                        "mode",
                        min_value=0.0,
                        max_value=1.0,
                        value=mode_v,
                        step=0.01,
                        key="Shc_gas_input_mode",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_gas_input_max",
                    ),
                }
            else:  # Uniform
                st.session_state["Shc_gas_input_dist"] = {
                    "type": "Uniform",
                    "min": st.number_input(
                        "min ",
                        min_value=0.0,
                        max_value=1.0,
                        value=min_v,
                        step=0.01,
                        key="Shc_gas_input_min",
                    ),
                    "max": st.number_input(
                        "max",
                        min_value=0.0,
                        max_value=1.0,
                        value=max_v,
                        step=0.01,
                        key="Shc_gas_input_max",
                    ),
                }
            
            # Always show distribution plot
            from scopehc.sampling import rng_from_seed, sample_pert, sample_triangular, sample_uniform
            num_sims = int(st.session_state.get("num_sims", 10_000))
            seed = st.session_state.get("random_seed", 12345)
            rng = rng_from_seed(seed)
            dist_config = st.session_state["Shc_gas_input_dist"]
            
            if dist_type_gas == "PERT":
                samples = sample_pert(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            elif dist_type_gas == "Triangular":
                samples = sample_triangular(rng, dist_config["min"], dist_config["mode"], dist_config["max"], num_sims)
            else:  # Uniform
                samples = sample_uniform(rng, dist_config["min"], dist_config["max"], num_sims)
            
            samples = np.clip(samples, 0.0, 1.0)
            st.session_state["sShc_gas_input"] = samples
            
            from .common import make_hist_cdf_figure
            st.plotly_chart(
                make_hist_cdf_figure(samples, r"$S_{\mathrm{gas}}$ distribution", 
                                    r"$S_{\mathrm{gas}}$", "input"),
                use_container_width=True,
            )

    st.caption(
        "Saturation note: $S_w$ varies with height due to capillary pressure. "
        "For teaching, a single $S$ per zone is sufficient; advanced workflows model height functions."
    )

