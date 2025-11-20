from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import linregress

from .common import (
    PALETTE,
    HELP,
    render_param,
    summarize_array,
    make_hist_cdf_figure,
    create_dependency_matrix_ui_with_scatter_plots,
    validate_dependency_matrix,
    fix_correlation_matrix,
    sample_correlated,
    rng_from_seed,
)
from scopehc.sampling import sample_scalar_dist


def render_dependencies(num_sims: int, defaults, show_inline_tips: bool) -> None:
    st.markdown("---")
    st.markdown("## Parameter Dependencies/Correlations")
    st.markdown(
        "Configure correlations between specific parameter pairs to model realistic dependencies in your analysis."
    )
    st.markdown(
        "**Enhanced with Correlated Sampling:** This version now uses Cholesky decomposition to generate "
        "mathematically rigorous correlated parameter samples for Monte Carlo simulations."
    )

    with st.expander("ℹ️ About Correlation Methods", expanded=False):
        st.markdown(
            """
        **Enhanced Correlation Matrix Approach:**

        **Advanced Correlated Sampling (Current Implementation):**
        - Define correlations between multiple parameters simultaneously
        - Uses Cholesky decomposition to generate correlated normal random variables
        - Transforms correlated normals to uniform [0,1] using normal CDF
        - Applies inverse CDF (ppf) of each parameter's target distribution (PERT, Triangular, Lognormal, etc.)
        - Provides a heatmap visualization of the correlation matrix
        - Mathematically rigorous and preserves distribution properties
        - Automatically ensures correlation matrix is positive semi-definite

        **Key Benefits:**
        - **Mathematical Rigor:** Uses proper multivariate normal theory
        - **Distribution Preservation:** Maintains original parameter distributions
        - **Stability:** Automatically handles correlation matrix validation
        - **Performance:** Efficient vectorized operations for large sample sizes

        **How It Works:**
        1. Generate independent standard normal random variables
        2. Apply Cholesky decomposition: L * L^T = correlation_matrix
        3. Transform to correlated normals: X = L * Z
        4. Convert to uniform [0,1] using normal CDF
        5. Apply inverse CDF of target distributions
        """
        )

    dependencies_enabled = st.checkbox(
        "Enable parameter dependencies/correlations",
        value=st.session_state.get("dependencies_enabled", False),
        help="Toggle to enable or disable all parameter correlations",
    )
    st.session_state["dependencies_enabled"] = dependencies_enabled

    if dependencies_enabled:
        st.info(
            "Parameter dependencies will be applied during Monte Carlo sampling to create more realistic parameter relationships."
        )
        st.warning(
            "⚠️ **Note:** Configure and apply dependencies before running the simulation. "
            "The correlation effects will be visible in the cross plots below and will be used in the Monte Carlo simulation."
        )

        dependency_tabs = st.tabs(["Dependency Matrix"])
    else:
        dependency_tabs = []

    if "dependency_values" not in st.session_state:
        st.session_state.dependency_values = {}
    if "dependency_matrix_values" not in st.session_state:
        st.session_state.dependency_matrix_values = {}
    if "correlation_values" not in st.session_state:
        st.session_state.correlation_values = {}

    available_params: list[str] = []
    params_config: dict[str, dict[str, float]] = {}

    grv_option = st.session_state.get(
        "grv_option", "From Area, Geometry Factor and Thickness"
    )
    fluid_property_method = st.session_state.get(
        "fluid_property_method", "Direct distribution input"
    )

    def is_parameter_used(param_name: str, grv_opt: str, fluid_method: str) -> bool:
        if param_name in ["Area", "GCF", "Thickness"]:
            if grv_opt == "Direct input":
                return False
            if grv_opt == "From Area, Geometry Factor and Thickness":
                return True
            return False
        if param_name in ["Bg", "InvBo", "GOR"]:
            return fluid_method == "Direct distribution input"
        if param_name in ["NtG", "Porosity", "RF_oil", "RF_gas", "Oil_Fraction"]:
            return True
        return False

    if grv_option != "Direct input":
        if "sA" in st.session_state and is_parameter_used("Area", grv_option, fluid_property_method):
            available_params.append("Area")
            params_config["Area"] = {
                "dist": "PERT",
                "min": defaults["A"]["min"],
                "mode": defaults["A"]["mode"],
                "max": defaults["A"]["max"],
            }
        if "sGCF" in st.session_state and is_parameter_used("GCF", grv_option, fluid_property_method):
            available_params.append("GCF")
            params_config["GCF"] = {
                "dist": "PERT",
                "min": defaults["GCF"]["min"],
                "mode": defaults["GCF"]["mode"],
                "max": defaults["GCF"]["max"],
            }
        if "sh" in st.session_state and is_parameter_used("Thickness", grv_option, fluid_property_method):
            available_params.append("Thickness")
            params_config["Thickness"] = {
                "dist": "PERT",
                "min": defaults["h"]["min"],
                "mode": defaults["h"]["min"],
                "max": defaults["h"]["max"],
            }

    param_candidates = {
        "NtG": ("sNtG", defaults["NtG"]),
        "Porosity": ("sp", defaults["p"]),
        "RF_oil": ("sRF_oil", {"min": 0.15, "mode": 0.30, "max": 0.45}),
        "RF_gas": ("sRF_gas", {"min": 0.50, "mode": 0.70, "max": 0.85}),
        "Oil_Fraction": ("sf_oil", {"min": 0.0, "mode": 0.5, "max": 1.0}),
        "Bg": ("sBg", {"min": 0.0045, "mode": 0.0055, "max": 0.0065}),
        "InvBo": ("sInvBo", {"min": 0.6667, "mode": 0.7407, "max": 0.8333}),
        "GOR": ("sGOR", {"min": 400, "mode": 800, "max": 1200}),
        "CGR": ("sCY", {"min": 20, "mode": 45, "max": 80}),
    }

    for param, (state_key, defaults_map) in param_candidates.items():
        if state_key in st.session_state and is_parameter_used(param, grv_option, fluid_property_method):
            available_params.append(param)
            params_config[param] = {"dist": "PERT", **defaults_map}

    # Add saturation variables based on mode
    mode = st.session_state.get("sat_mode", "Global")
    use_sw_global = st.session_state.get("global_sat_use_sw", False)
    
    if mode.startswith("Global"):
        if "Shc_global_dist" in st.session_state:
            if use_sw_global:
                available_params.append("Sw_global")
                dist = st.session_state["Shc_global_dist"]
                if dist.get("type") == "Constant":
                    params_config["Sw_global"] = {
                        "dist": "PERT",
                        "min": dist.get("value", 0.0),
                        "mode": dist.get("value", 0.0),
                        "max": dist.get("value", 0.0),
                    }
                else:
                    params_config["Sw_global"] = {
                        "dist": "PERT",
                        "min": dist.get("min", 0.0),
                        "mode": dist.get("mode", 0.1),
                        "max": dist.get("max", 0.2),
                    }
            else:
                available_params.append("Shc_global")
                dist = st.session_state["Shc_global_dist"]
                if dist.get("type") == "Constant":
                    params_config["Shc_global"] = {
                        "dist": "PERT",
                        "min": dist.get("value", 1.0),
                        "mode": dist.get("value", 1.0),
                        "max": dist.get("value", 1.0),
                    }
                else:
                    params_config["Shc_global"] = {
                        "dist": "PERT",
                        "min": dist.get("min", 0.6),
                        "mode": dist.get("mode", 0.8),
                        "max": dist.get("max", 0.9),
                    }
    elif mode.startswith("Water saturation"):
        if "Sw_oilzone_dist" in st.session_state:
            available_params.append("Sw_oilzone")
            params_config["Sw_oilzone"] = {
                "dist": "PERT",
                "min": st.session_state["Sw_oilzone_dist"].get("min", 0.25),
                "mode": st.session_state["Sw_oilzone_dist"].get("mode", 0.35),
                "max": st.session_state["Sw_oilzone_dist"].get("max", 0.50),
            }
        if "Sw_gaszone_dist" in st.session_state:
            available_params.append("Sw_gaszone")
            params_config["Sw_gaszone"] = {
                "dist": "PERT",
                "min": st.session_state["Sw_gaszone_dist"].get("min", 0.10),
                "mode": st.session_state["Sw_gaszone_dist"].get("mode", 0.15),
                "max": st.session_state["Sw_gaszone_dist"].get("max", 0.25),
            }
    else:  # Per phase
        if "Shc_oil_input_dist" in st.session_state:
            available_params.append("Shc_oil_input")
            params_config["Shc_oil_input"] = {
                "dist": "PERT",
                "min": st.session_state["Shc_oil_input_dist"].get("min", 0.6),
                "mode": st.session_state["Shc_oil_input_dist"].get("mode", 0.8),
                "max": st.session_state["Shc_oil_input_dist"].get("max", 0.9),
            }
        if "Shc_gas_input_dist" in st.session_state:
            available_params.append("Shc_gas_input")
            params_config["Shc_gas_input"] = {
                "dist": "PERT",
                "min": st.session_state["Shc_gas_input_dist"].get("min", 0.6),
                "mode": st.session_state["Shc_gas_input_dist"].get("mode", 0.8),
                "max": st.session_state["Shc_gas_input_dist"].get("max", 0.9),
            }

    if dependencies_enabled:
        with dependency_tabs[0]:
            if len(available_params) < 2:
                st.warning("⚠️ At least 2 parameters must be defined to use dependency matrix")
                st.session_state.dependency_matrix = None
                st.session_state.use_dependency_matrix = False
            else:
                st.info(
                    f"{len(available_params)} parameters available for dependency matrix: "
                    f"{', '.join(available_params)}"
                )

                current_samples = {}
                mapping_keys = {
                    "Area": "sA",
                    "GCF": "sGCF",
                    "Thickness": "sh",
                    "NtG": "sNtG",
                    "Porosity": "sp",
                    "RF_oil": "sRF_oil",
                    "RF_gas": "sRF_gas",
                    "Bg": "sBg",
                    "InvBo": "sInvBo",
                    "GOR": "sGOR",
                    "CGR": "sCY",
                    "Oil_Fraction": "sf_oil",
                }

                for param in available_params:
                    key = mapping_keys.get(param)
                    if key and key in st.session_state:
                        current_samples[key] = st.session_state[key]

                dep_matrix, dep_dict = create_dependency_matrix_ui_with_scatter_plots(
                    available_params,
                    st.session_state.dependency_matrix_values,
                    current_samples,
                )

                is_valid, error_msg = validate_dependency_matrix(dep_matrix, available_params)
                if is_valid:
                    st.session_state.dependency_matrix = dep_matrix
                    st.session_state.dependency_matrix_params = available_params
                    st.session_state.dependency_matrix_config = params_config
                    st.session_state.dependency_matrix_values = dep_dict
                    st.session_state.use_dependency_matrix = True
                    st.info("Dependencies will be automatically applied during Monte Carlo sampling.")
                else:
                    st.error(f"❌ Dependency matrix is invalid: {error_msg}")
                    st.session_state.dependency_matrix = None
                    st.session_state.use_dependency_matrix = False

    if not st.session_state.get("use_dependency_matrix", False):
        st.info("🔒 Parameter dependencies are disabled. All parameters will be sampled independently.")


def apply_dependencies(num_sims: int) -> None:
    if not st.session_state.get("use_dependency_matrix", False):
        return

    if "dependency_matrix" not in st.session_state:
        return

    dep_matrix = st.session_state.dependency_matrix
    param_names = st.session_state.dependency_matrix_params
    params_config = st.session_state.dependency_matrix_config

    try:
        fixed_dep_matrix = fix_correlation_matrix(dep_matrix)
        correlated = sample_correlated(params_config, fixed_dep_matrix, param_names, num_sims)
    except Exception as exc:
        st.error(f"❌ Error in correlated sampling: {exc}")
        st.session_state.use_dependency_matrix = False
        return

    ui_to_state = {
        "Area": "sA",
        "GCF": "sGCF",
        "Thickness": "sh",
        "NtG": "sNtG",
        "Shc_global": "Shc_global",
        "Sw_global": "Sw_global",
        "Sw_oilzone": "Sw_oilzone",
        "Sw_gaszone": "Sw_gaszone",
        "Shc_oil_input": "Shc_oil_input",
        "Shc_gas_input": "Shc_gas_input",
        "Porosity": "sp",
        "RF_oil": "sRF_oil",
        "RF_gas": "sRF_gas",
        "Bg": "sBg",
        "InvBo": "sInvBo",
        "GOR": "sGOR",
        "CGR": "sCY",
        "Oil_Fraction": "sf_oil",
    }

    updated_count = 0
    for name, state_key in ui_to_state.items():
        if name in correlated:
            st.session_state[state_key] = correlated[name]
            updated_count += 1

    if "Area" in correlated and "Thickness" in correlated:
        gcf = correlated.get("GCF", st.session_state.get("sGCF", 1.0))
        st.session_state["sGRV_m3_final"] = (
            correlated["Area"] * 1_000_000.0 * correlated["Thickness"] * gcf
        )

    st.success(f"✅ Applied correlated sampling to {updated_count} parameter(s).")
    
    # Render cross plots for correlated parameters
    _render_correlation_cross_plots(dep_matrix, param_names, params_config, correlated, num_sims)


def _render_correlation_cross_plots(
    dep_matrix: np.ndarray,
    param_names: list[str],
    params_config: dict[str, dict[str, Any]],
    correlated: dict[str, np.ndarray],
    num_sims: int,
) -> None:
    """Render cross plots showing before/after correlation with regression lines."""
    # Find pairs with non-zero correlation
    correlated_pairs = []
    for i in range(len(param_names)):
        for j in range(i + 1, len(param_names)):
            corr_value = dep_matrix[i, j]
            if abs(corr_value) > 1e-6:  # Non-zero correlation
                correlated_pairs.append((i, j, corr_value, param_names[i], param_names[j]))
    
    if not correlated_pairs:
        return
    
    st.markdown("---")
    st.markdown("### Correlation Cross Plots")
    st.markdown(
        "Scatter plots showing parameter relationships before (gray) and after (red) applying correlations. "
        "Every 100th trial is shown. Linear regression lines, formulas, and R² values are displayed."
    )
    
    # Generate independent samples for comparison
    seed = st.session_state.get("random_seed", 12345)
    rng = rng_from_seed(seed)
    independent_samples = {}
    
    for param_name in param_names:
        if param_name in params_config:
            config = params_config[param_name]
            dist_type = config.get("dist", "PERT").lower()
            independent_samples[param_name] = sample_scalar_dist(rng, dist_type, config, num_sims)
    
    # Create plots for each correlated pair
    for i, j, corr_value, p1, p2 in correlated_pairs:
        if p1 not in correlated or p2 not in correlated:
            continue
        if p1 not in independent_samples or p2 not in independent_samples:
            continue
        
        # Get every 100th trial
        step = 100
        indices = np.arange(0, num_sims, step)
        
        # Independent samples (before correlation)
        x_indep = independent_samples[p1][indices]
        y_indep = independent_samples[p2][indices]
        
        # Correlated samples (after correlation)
        x_corr = correlated[p1][indices]
        y_corr = correlated[p2][indices]
        
        # Calculate linear regression for independent samples
        slope_indep, intercept_indep, r_value_indep, p_value_indep, std_err_indep = linregress(x_indep, y_indep)
        r2_indep = r_value_indep ** 2
        y_pred_indep = slope_indep * x_indep + intercept_indep
        
        # Calculate linear regression for correlated samples
        slope_corr, intercept_corr, r_value_corr, p_value_corr, std_err_corr = linregress(x_corr, y_corr)
        r2_corr = r_value_corr ** 2
        y_pred_corr = slope_corr * x_corr + intercept_corr
        
        # Create plot
        fig = go.Figure()
        
        # Independent samples (light gray)
        fig.add_trace(
            go.Scatter(
                x=x_indep,
                y=y_indep,
                mode="markers",
                name="Before correlation",
                marker=dict(color="#CCCCCC", size=8, opacity=0.6),
                hovertemplate=f"{p1}: %{{x:.4f}}<br>{p2}: %{{y:.4f}}<extra></extra>",
            )
        )
        
        # Regression line for independent samples
        x_line_indep = np.linspace(x_indep.min(), x_indep.max(), 100)
        y_line_indep = slope_indep * x_line_indep + intercept_indep
        fig.add_trace(
            go.Scatter(
                x=x_line_indep,
                y=y_line_indep,
                mode="lines",
                name=f"Before: y = {slope_indep:.4f}x + {intercept_indep:.4f} (R² = {r2_indep:.4f})",
                line=dict(color="#999999", width=2, dash="dash"),
                hovertemplate="Regression line (before)<extra></extra>",
            )
        )
        
        # Correlated samples (light red)
        fig.add_trace(
            go.Scatter(
                x=x_corr,
                y=y_corr,
                mode="markers",
                name="After correlation",
                marker=dict(color="#FF6B6B", size=8, opacity=0.7),
                hovertemplate=f"{p1}: %{{x:.4f}}<br>{p2}: %{{y:.4f}}<extra></extra>",
            )
        )
        
        # Regression line for correlated samples
        x_line_corr = np.linspace(x_corr.min(), x_corr.max(), 100)
        y_line_corr = slope_corr * x_line_corr + intercept_corr
        fig.add_trace(
            go.Scatter(
                x=x_line_corr,
                y=y_line_corr,
                mode="lines",
                name=f"After: y = {slope_corr:.4f}x + {intercept_corr:.4f} (R² = {r2_corr:.4f})",
                line=dict(color="#FF6B6B", width=2),
                hovertemplate="Regression line (after)<extra></extra>",
            )
        )
        
        # Update layout
        fig.update_layout(
            title=f"{p1} vs {p2} (Correlation: {corr_value:.3f})",
            xaxis_title=p1,
            yaxis_title=p2,
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10),
            ),
            template="plotly_white",
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display regression statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Before Correlation:**")
            st.markdown(f"- Formula: $y = {slope_indep:.4f}x + {intercept_indep:.4f}$")
            st.markdown(f"- R² = {r2_indep:.4f}")
        with col2:
            st.markdown(f"**After Correlation:**")
            st.markdown(f"- Formula: $y = {slope_corr:.4f}x + {intercept_corr:.4f}$")
            st.markdown(f"- R² = {r2_corr:.4f}")
        
        st.markdown("---")

