from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scopehc.compute import compute_results
from scopehc.utils import validate_rf_fractions, validate_fractions

from .common import (
    init_theme,
    make_hist_cdf_figure,
    color_for,
    summary_table,
    summarize_array,
    UNIT_CONVERSIONS,
    HELP,
    get_unit_system,
    render_custom_navigation,
    render_sidebar_settings,
    render_sidebar_help,
    PALETTE,
    collect_all_inputs_from_session,
    compute_input_hash,
)


def render() -> None:
    """Render the full results experience migrated from the legacy app."""
    st.session_state["current_page"] = "_pages_disabled/04_Results.py"
    st.header("Results")

    unit_system = get_unit_system()
    gas_scf_per_boe = float(st.session_state.get("gas_scf_per_boe", 6000.0))
    st.caption(
        f"Gas-to-BOE conversion factor (editable in sidebar): {gas_scf_per_boe:,.0f} scf/BOE"
    )
    
    # Display percentile convention
    use_exceedance = st.session_state.get("percentile_exceedance", True)
    mode_str = "probability of exceedance" if use_exceedance else "probability of non-exceedance"
    st.info(f"**Percentile convention:** P10/P50/P90 shown using {mode_str} convention. Change in sidebar if needed.")

    required = ["sGRV_m3_final", "sNtG", "sp", "sBg", "sInvBo", "sRF_oil", "sRF_gas"]
    missing = [key for key in required if key not in st.session_state]
    if missing:
        st.info(
            "Results require completed inputs and a simulation run. "
            "Ensure all inputs are configured and the simulation has been executed."
        )
        return

    sGRV_m3_final = np.asarray(st.session_state["sGRV_m3_final"], dtype=float)
    num_sims = int(len(sGRV_m3_final))
    if num_sims == 0:
        st.warning("No simulation samples available. Run the simulation to view results.")
        return

    sNtG = np.asarray(st.session_state["sNtG"], dtype=float)
    sp = np.asarray(st.session_state["sp"], dtype=float)
    sBg = np.asarray(st.session_state["sBg"], dtype=float)
    sInvBo = np.asarray(st.session_state["sInvBo"], dtype=float)
    sRF_oil = np.asarray(st.session_state["sRF_oil"], dtype=float)
    sRF_gas = np.asarray(st.session_state["sRF_gas"], dtype=float)
    sRF_assoc = np.asarray(
        st.session_state.get("sRF_assoc_gas", sRF_oil), dtype=float
    )
    # f_oil should be a scalar, but handle case where it might be an array
    f_oil_val = st.session_state.get("f_oil", 0.5)
    if isinstance(f_oil_val, np.ndarray):
        # If f_oil is an array, use its mean or first value
        f_oil_val = float(np.mean(f_oil_val)) if len(f_oil_val) > 0 else 0.5
    else:
        f_oil_val = float(f_oil_val)
    f_oil = np.full(num_sims, f_oil_val)
    sGOR = np.asarray(st.session_state["sGOR"], dtype=float)
    sCY = st.session_state.get("sCY")
    sRF_cond = st.session_state.get("sRF_cond")

    # CRITICAL: Get GRV split arrays, but handle case where they might not exist
    # If fluid_type changed, split arrays might be cleared, but we can recalculate them
    # from sGRV_m3_final and current fluid_type
    fluid_type = st.session_state.get("fluid_type", "Oil")
    grv_option = st.session_state.get("grv_option", "Direct GRV")
    
    # Check if this is a depth-based method
    depth_based_methods = {
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    }
    is_depth_based = grv_option in depth_based_methods
    
    GRV_oil_m3 = st.session_state.get("sGRV_oil_m3")
    GRV_gas_m3 = st.session_state.get("sGRV_gas_m3")
    
    # Convert to arrays and ensure they're at least 1D
    if GRV_oil_m3 is not None:
        GRV_oil_m3 = np.atleast_1d(np.asarray(GRV_oil_m3, dtype=float))
        if len(GRV_oil_m3) != num_sims:
            GRV_oil_m3 = None  # Size mismatch, will recalculate
    if GRV_gas_m3 is not None:
        GRV_gas_m3 = np.atleast_1d(np.asarray(GRV_gas_m3, dtype=float))
        if len(GRV_gas_m3) != num_sims:
            GRV_gas_m3 = None  # Size mismatch, will recalculate
    
    # If split arrays don't exist, recalculate based on current fluid_type
    if GRV_oil_m3 is None or GRV_gas_m3 is None:
        if fluid_type == "Oil":
            GRV_oil_m3 = sGRV_m3_final.copy()
            GRV_gas_m3 = np.zeros_like(sGRV_m3_final)
        elif fluid_type == "Gas":
            GRV_oil_m3 = np.zeros_like(sGRV_m3_final)
            GRV_gas_m3 = sGRV_m3_final.copy()
        else:  # Oil + Gas
            # For depth-based methods, we should NOT use f_oil split - the values should already be calculated
            # But if they're missing, fall back to f_oil split as a last resort
            if is_depth_based:
                # For depth-based, try to preserve existing values even if size doesn't match
                # This is a fallback - ideally the values should already be set correctly
                st.warning(
                    "⚠️ Depth-based GRV split arrays not found. "
                    "Please return to GRV page and ensure the calculation completes."
                )
            # Use f_oil to split (f_oil_val already calculated above)
            GRV_oil_m3 = sGRV_m3_final * f_oil_val
            GRV_gas_m3 = sGRV_m3_final * (1.0 - f_oil_val)

    sCY_arr = None if sCY is None else np.asarray(sCY, dtype=float)
    sRF_cond_arr = None if sRF_cond is None else np.asarray(sRF_cond, dtype=float)

    sRF_oil, sRF_gas, sRF_cond_arr, rf_warnings = validate_rf_fractions(
        sRF_oil, sRF_gas, sRF_cond_arr
    )
    f_oil, frac_warnings = validate_fractions(f_oil)

    warnings = rf_warnings + frac_warnings
    if warnings:
        st.warning(
            "⚠️ Parameter validation warnings:\n"
            + "\n".join(f"• {warning}" for warning in warnings)
        )

    # Get saturation arrays if available
    Shc_oil = st.session_state.get("Shc_oil", None)
    Shc_gas = st.session_state.get("Shc_gas", None)

    res = compute_results(
        GRV_m3=sGRV_m3_final,
        NtG=sNtG,
        Por=sp,
        f_oil=f_oil,
        RF_oil=sRF_oil,
        RF_gas=sRF_gas,
        Bg_rb_per_scf=sBg,
        InvBo_STB_per_rb=sInvBo,
        GOR_scf_per_STB=sGOR,
        CY_STB_per_MMscf=sCY_arr,
        RF_cond=sRF_cond_arr,
        gas_scf_per_boe=gas_scf_per_boe,
        GRV_oil_m3=GRV_oil_m3,
        GRV_gas_m3=GRV_gas_m3,
        Shc_oil=Shc_oil,
        Shc_gas=Shc_gas,
    )

    PV_total_m3 = res["PV_total_m3"]
    PV_oil_m3 = res["PV_oil_m3"]
    PV_gas_m3 = res["PV_gas_m3"]
    V_oil_insitu_m3 = res["V_oil_insitu_m3"]
    V_gas_insitu_m3 = res["V_gas_insitu_m3"]
    Oil_STB_rec = res["Oil_STB_rec"]
    Gas_scf_rec = res["Gas_free_scf_rec"]
    AssocGas_scf_rec = res["Gas_assoc_scf_rec"]
    Cond_rec_STB = res["Cond_STB_rec"]
    thr_boe = res["Total_surface_BOE"]

    Oil_BOE = res["Oil_BOE"]
    Cond_BOE = res["Cond_BOE"]
    Gas_free_BOE = res["Gas_free_BOE"]
    Gas_assoc_BOE = res["Gas_assoc_BOE"]
    
    # Diagnostic: Check if condensate should be calculated but isn't
    # Use Gas_free_scf_rec (free gas from gas zone) for condensate, not associated gas
    Gas_free_scf_rec = res.get("Gas_free_scf_rec", None)
    has_gas_volume = np.any(Gas_free_scf_rec > 0) if Gas_free_scf_rec is not None else False
    has_condensate = np.any(Cond_rec_STB > 0) if Cond_rec_STB is not None else False
    if has_gas_volume and not has_condensate:
        # Check if CY or RF_cond are missing
        if sCY is None:
            st.info(
                "💡 **Condensate not calculated:** Gas volumes are present, but Condensate Yield (CY) is not set. "
                "To calculate condensate, please go to the **Fluids** input section and set the Condensate Yield (CY) parameter. "
                "CY represents the amount of condensate (liquid) that can be recovered per million standard cubic feet of gas."
            )
        elif sRF_cond is None:
            st.info(
                "💡 **Condensate not calculated:** Gas volumes and CY are present, but Condensate Recovery Factor (RF_cond) is not set. "
                "Please set RF_cond in the **Recovery Factor** section."
            )
        else:
            # Both are set but condensate is still zero - check if CY values are all zero
            if sCY is not None:
                sCY_arr = np.asarray(sCY, dtype=float)
                if np.all(sCY_arr == 0):
                    st.info(
                        "💡 **Condensate not calculated:** Gas volumes are present, but Condensate Yield (CY) is set to zero. "
                        "Please set CY to a non-zero value in the **Fluids** input section."
                    )

    Oil_m3_rec = Oil_STB_rec * UNIT_CONVERSIONS["bbl_to_m3"]
    Gas_m3_rec = Gas_scf_rec * UNIT_CONVERSIONS["scf_to_m3"]
    AssocGas_m3_rec = AssocGas_scf_rec * UNIT_CONVERSIONS["scf_to_m3"]

    if unit_system == "oilfield":
        Oil_display = Oil_STB_rec / 1e6
        FreeGas_display = Gas_scf_rec / 1e9
        AssocGas_display = AssocGas_scf_rec / 1e9
        oil_unit = "MMSTB"
        gas_unit = "Bscf"
    else:
        Oil_display = Oil_m3_rec / 1e6
        FreeGas_display = Gas_m3_rec / 1e9
        AssocGas_display = AssocGas_m3_rec / 1e9
        oil_unit = "Mm³"
        gas_unit = "Bm³"

    V_oil_insitu_Mm3 = V_oil_insitu_m3 / 1e6
    V_gas_insitu_Mm3 = V_gas_insitu_m3 / 1e6

    if unit_system == "oilfield":
        Cond_display = Cond_rec_STB / 1e6
        cond_unit = "MMSTB"
    else:
        Cond_display = Cond_rec_STB * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        cond_unit = "Mm³"

    st.markdown("---")
    st.markdown("## In-situ (In-place) Volumes")

    insitu_results = [
        ("In-situ Oil Volume", V_oil_insitu_Mm3, "×10^6 m³", 2),
        ("In-situ Gas Volume", V_gas_insitu_Mm3, "×10^6 m³", 2),
    ]

    for title, arr, unit, decimals in insitu_results:
        st.plotly_chart(
            make_hist_cdf_figure(arr, title, f"{title} ({unit})", "result"),
            use_container_width=True,
        )
        # Get convention-aware interpretation
        use_exceedance = st.session_state.get("percentile_exceedance", True)
        if use_exceedance:
            interp_text = "_Interpretation:_ P10/P50/P90 use probability-of-exceedance convention (P10=optimistic, P90=conservative). "
        else:
            interp_text = "_Interpretation:_ P10/P50/P90 use non-exceedance convention (P10=conservative, P90=optimistic). "
        interp_text += "Wider histograms indicate larger uncertainty from GRV and petrophysical spread."
        st.caption(interp_text)
        st.dataframe(summary_table(arr, decimals=decimals), use_container_width=True)

    st.markdown("---")
    st.markdown("## Recoverable Volumes")

    recoverable_results = [
        ("Recoverable Oil", Oil_display, oil_unit, 2),
        ("Recoverable Free Gas", FreeGas_display, gas_unit, 2),
        ("Recoverable Associated Gas", AssocGas_display, gas_unit, 2),
        ("Recoverable Condensate", Cond_display, cond_unit, 1),
    ]

    for title, arr, unit, decimals in recoverable_results:
        st.plotly_chart(
            make_hist_cdf_figure(arr, title, f"{title} ({unit})", "result"),
            use_container_width=True,
        )
        # Get convention-aware interpretation
        use_exceedance = st.session_state.get("percentile_exceedance", True)
        if use_exceedance:
            interp_text = "_Interpretation:_ P10/P50/P90 use probability-of-exceedance convention (P10=optimistic, P90=conservative). "
        else:
            interp_text = "_Interpretation:_ P10/P50/P90 use non-exceedance convention (P10=conservative, P90=optimistic). "
        interp_text += "Wider histograms indicate larger uncertainty from GRV and petrophysical spread. CDF shows cumulative probability."
        st.caption(interp_text)
        st.dataframe(summary_table(arr, decimals=decimals), use_container_width=True)

    st.markdown("---")
    st.markdown("## Recoverable (Surface) Volumes")
    st.markdown("**Key percentiles for recoverable volumes:**")

    # Create a comprehensive table with all statistics
    def format_number(value: float, decimals: int = 1) -> str:
        """Format number with commas and consistent decimals."""
        return f"{value:,.{decimals}f}"

    # Collect all statistics
    oil_stats = summarize_array(Oil_display)
    free_gas_stats = summarize_array(FreeGas_display)
    assoc_gas_stats = summarize_array(AssocGas_display)
    condensate_stats = summarize_array(Cond_display)
    total_liquids_display = Oil_display + Cond_display
    total_liquids_stats = summarize_array(total_liquids_display)

    # Build table data with consistent columns
    table_data = []
    if oil_stats:
        table_data.append({
            "Parameter": "Oil",
            "P50": format_number(oil_stats['P50'], 1),
            "P90": format_number(oil_stats['P90'], 1),
            "P10": format_number(oil_stats['P10'], 1),
            "Mean": format_number(oil_stats['mean'], 1),
            "Std": format_number(oil_stats['std_dev'], 1),
            "Unit": oil_unit
        })
    if free_gas_stats:
        table_data.append({
            "Parameter": "Free Gas",
            "P50": format_number(free_gas_stats['P50'], 1),
            "P90": format_number(free_gas_stats['P90'], 1),
            "P10": format_number(free_gas_stats['P10'], 1),
            "Mean": format_number(free_gas_stats['mean'], 1),
            "Std": format_number(free_gas_stats['std_dev'], 1),
            "Unit": gas_unit
        })
    if assoc_gas_stats:
        table_data.append({
            "Parameter": "Assoc Gas",
            "P50": format_number(assoc_gas_stats['P50'], 1),
            "P90": format_number(assoc_gas_stats['P90'], 1),
            "P10": format_number(assoc_gas_stats['P10'], 1),
            "Mean": format_number(assoc_gas_stats['mean'], 1),
            "Std": format_number(assoc_gas_stats['std_dev'], 1),
            "Unit": gas_unit
        })
    if condensate_stats:
        table_data.append({
            "Parameter": "Condensate",
            "P50": format_number(condensate_stats['P50'], 1),
            "P90": format_number(condensate_stats['P90'], 1),
            "P10": format_number(condensate_stats['P10'], 1),
            "Mean": format_number(condensate_stats['mean'], 1),
            "Std": format_number(condensate_stats['std_dev'], 1),
            "Unit": cond_unit
        })
    if total_liquids_stats:
        table_data.append({
            "Parameter": "Total Liquids (Oil + Condensate)",
            "P50": format_number(total_liquids_stats['P50'], 1),
            "P90": format_number(total_liquids_stats['P90'], 1),
            "P10": format_number(total_liquids_stats['P10'], 1),
            "Mean": format_number(total_liquids_stats['mean'], 1),
            "Std": format_number(total_liquids_stats['std_dev'], 1),
            "Unit": oil_unit
        })

    if table_data:
        df_table = pd.DataFrame(table_data)
        
        # Create styled table with better formatting
        styled_df = df_table.style.set_properties(
            **{
                'text-align': 'right',
                'font-size': '0.95rem',
                'padding': '10px 15px',
                'white-space': 'nowrap',
            },
            subset=['P50', 'P90', 'P10', 'Mean', 'Std']
        ).set_properties(
            **{
                'text-align': 'left',
                'font-weight': '600',
                'padding': '10px 15px',
            },
            subset=['Parameter']
        ).set_properties(
            **{
                'text-align': 'center',
                'font-size': '0.85rem',
                'color': '#666',
                'padding': '10px 8px',
            },
            subset=['Unit']
        ).set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', PALETTE.get('bg_light', '#F5F1E6')),
                    ('color', PALETTE.get('text_primary', '#1E1E1E')),
                    ('font-weight', '700'),
                    ('text-align', 'center'),
                    ('padding', '12px 15px'),
                    ('border-bottom', '2px solid ' + PALETTE.get('primary', '#FF6B6B')),
                    ('font-size', '0.9rem'),
                ]
            },
            {
                'selector': 'tbody tr:nth-of-type(even)',
                'props': [
                    ('background-color', 'rgba(245, 241, 230, 0.5)'),
                ]
            },
            {
                'selector': 'tbody tr:hover',
                'props': [
                    ('background-color', '#E8E4D9'),
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('border', '1px solid #ddd'),
                    ('vertical-align', 'middle'),
                ]
            }
        ])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## Total Hydrocarbon Resource (THR)")

    if unit_system == "oilfield":
        thr_display = thr_boe / 1e6
        thr_unit = "MBOE"
    else:
        thr_display = thr_boe * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        thr_unit = "Mm³ BOE"

    thr_stats = summarize_array(thr_display)
    if thr_stats:
        # Create table format matching the Recoverable Volumes table
        thr_table_data = [{
            "Parameter": "THR",
            "P50": format_number(thr_stats['P50'], 1),
            "P90": format_number(thr_stats['P90'], 1),
            "P10": format_number(thr_stats['P10'], 1),
            "Mean": format_number(thr_stats['mean'], 1),
            "Std": format_number(thr_stats['std_dev'], 1),
            "Unit": thr_unit
        }]
        
        df_thr_table = pd.DataFrame(thr_table_data)
        
        # Create styled table with same formatting as Recoverable Volumes
        styled_thr_df = df_thr_table.style.set_properties(
            **{
                'text-align': 'right',
                'font-size': '0.95rem',
                'padding': '10px 15px',
                'white-space': 'nowrap',
            },
            subset=['P50', 'P90', 'P10', 'Mean', 'Std']
        ).set_properties(
            **{
                'text-align': 'left',
                'font-weight': '600',
                'padding': '10px 15px',
            },
            subset=['Parameter']
        ).set_properties(
            **{
                'text-align': 'center',
                'font-size': '0.85rem',
                'color': '#666',
                'padding': '10px 8px',
            },
            subset=['Unit']
        ).set_table_styles([
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', PALETTE.get('bg_light', '#F5F1E6')),
                    ('color', PALETTE.get('text_primary', '#1E1E1E')),
                    ('font-weight', '700'),
                    ('text-align', 'center'),
                    ('padding', '12px 15px'),
                    ('border-bottom', '2px solid ' + PALETTE.get('primary', '#FF6B6B')),
                    ('font-size', '0.9rem'),
                ]
            },
            {
                'selector': 'tbody tr:nth-of-type(even)',
                'props': [
                    ('background-color', 'rgba(245, 241, 230, 0.5)'),
                ]
            },
            {
                'selector': 'tbody tr:hover',
                'props': [
                    ('background-color', '#E8E4D9'),
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('border', '1px solid #ddd'),
                    ('vertical-align', 'middle'),
                ]
            }
        ])
        
        st.dataframe(styled_thr_df, use_container_width=True, hide_index=True)

    st.markdown("**THR Composition (mean):**")
    mean_oil_boe = np.mean(Oil_BOE) / 1e6
    mean_cond_boe = np.mean(Cond_BOE) / 1e6
    mean_gas_free_boe = np.mean(Gas_free_BOE) / 1e6
    mean_gas_assoc_boe = np.mean(Gas_assoc_BOE) / 1e6
    mean_thr_boe = thr_stats["mean"] if thr_stats else 0.0

    if mean_thr_boe > 0:
        pct_oil = (mean_oil_boe / mean_thr_boe) * 100
        pct_cond = (mean_cond_boe / mean_thr_boe) * 100
        pct_free_gas = (mean_gas_free_boe / mean_thr_boe) * 100
        pct_assoc_gas = (mean_gas_assoc_boe / mean_thr_boe) * 100
        composition = pd.DataFrame(
            {
                "Component": ["Oil", "Condensate", "Free Gas", "Associated Gas", "**Total**"],
                f"Mean ({thr_unit})": [
                    f"{mean_oil_boe:.1f}",
                    f"{mean_cond_boe:.1f}",
                    f"{mean_gas_free_boe:.1f}",
                    f"{mean_gas_assoc_boe:.1f}",
                    f"**{mean_thr_boe:.1f}**",
                ],
                "% of THR": [
                    f"{pct_oil:.1f}%",
                    f"{pct_cond:.1f}%",
                    f"{pct_free_gas:.1f}%",
                    f"{pct_assoc_gas:.1f}%",
                    "**100%**",
                ],
            }
        )
        st.dataframe(composition, use_container_width=True, hide_index=True)

    st.markdown("### Hydrocarbon Volume Distribution Analysis (BOE)")
    if unit_system == "oilfield":
        oil_boe_display = Oil_BOE / 1e6
        cond_boe_display = Cond_BOE / 1e6
        gas_free_boe_display = Gas_free_BOE / 1e6
        gas_assoc_boe_display = Gas_assoc_BOE / 1e6
        boe_unit = "MBOE"
    else:
        conversion = UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        oil_boe_display = Oil_BOE * conversion
        cond_boe_display = Cond_BOE * conversion
        gas_free_boe_display = Gas_free_BOE * conversion
        gas_assoc_boe_display = Gas_assoc_BOE * conversion
        boe_unit = "Mm³ BOE"

    fig_violin = go.Figure()
    fig_violin.add_trace(
        go.Violin(
            y=oil_boe_display,
            name=f"Oil ({boe_unit})",
            box_visible=True,
            line_color=color_for("Oil_STB_rec"),
            fillcolor=color_for("Oil_STB_rec"),
            opacity=0.6,
        )
    )
    fig_violin.add_trace(
        go.Violin(
            y=gas_free_boe_display,
            name=f"Free Gas ({boe_unit})",
            box_visible=True,
            line_color=color_for("Gas_free_scf_rec"),
            fillcolor=color_for("Gas_free_scf_rec"),
            opacity=0.6,
        )
    )
    fig_violin.add_trace(
        go.Violin(
            y=gas_assoc_boe_display,
            name=f"Associated Gas ({boe_unit})",
            box_visible=True,
            line_color=color_for("Gas_assoc_scf_rec"),
            fillcolor=color_for("Gas_assoc_scf_rec"),
            opacity=0.6,
        )
    )
    fig_violin.add_trace(
        go.Violin(
            y=cond_boe_display,
            name=f"Condensate ({boe_unit})",
            box_visible=True,
            line_color=color_for("Cond_STB_rec"),
            fillcolor=color_for("Cond_STB_rec"),
            opacity=0.6,
        )
    )
    fig_violin.update_layout(
        title="Hydrocarbon Volume Distribution Analysis (All Fluids in BOE)",
        yaxis_title=f"Volume ({boe_unit})",
        height=500,
        showlegend=True,
    )
    st.plotly_chart(fig_violin, use_container_width=True)
    st.caption(
        f"All volumes are shown in BOE units for direct comparison. "
        f"Gas conversion: {gas_scf_per_boe:,.0f} scf/BOE"
    )

    with st.expander("**Total Hydrocarbon Resource (THR)**", expanded=True):
        fig_thr = make_hist_cdf_figure(
            thr_display,
            "Total Hydrocarbon Resource (THR) Distribution",
            f"THR ({thr_unit})",
            "result",
        )
        if thr_stats:
            thr_color = color_for("Total_surface_BOE")
            for label in ("P10", "P50", "P90"):
                value = thr_stats[label]
                fig_thr.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color=thr_color,
                    annotation_text=f"{label}: {value:.1f} {thr_unit}",
                    annotation_position="top",
                )
        st.plotly_chart(fig_thr, use_container_width=True)
        # Get convention-aware interpretation
        use_exceedance = st.session_state.get("percentile_exceedance", True)
        if use_exceedance:
            interp_text = "_Interpretation:_ THR represents total recoverable hydrocarbon resources converted to oil equivalent. "
            interp_text += "P10/P50/P90 use probability-of-exceedance convention (P10=optimistic, P90=conservative). "
        else:
            interp_text = "_Interpretation:_ THR represents total recoverable hydrocarbon resources converted to oil equivalent. "
            interp_text += "P10/P50/P90 use non-exceedance convention (P10=conservative, P90=optimistic). "
        interp_text += "Wider distributions indicate greater uncertainty in volume estimates."
        st.caption(interp_text)
        st.dataframe(summary_table(thr_display, decimals=2), use_container_width=True)
        st.caption(
            f"THR includes Oil + Condensate + Gas/BOE. Default 6.0 Mscf/BOE; configurable per company standard. "
            f"Current factor: {gas_scf_per_boe:,.0f} scf/BOE"
        )

        if unit_system == "oilfield":
            conversion = 1e6
            boe_unit_detail = "MBOE"
        else:
            conversion = UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
            boe_unit_detail = "Mm³ BOE"

        oil_boe_display = Oil_BOE / conversion
        cond_boe_display = Cond_BOE / conversion
        gas_free_boe_display = Gas_free_BOE / conversion
        gas_assoc_boe_display = Gas_assoc_BOE / conversion
        thr_display_converted = thr_display.copy()

        mean_oil = np.mean(oil_boe_display)
        mean_cond = np.mean(cond_boe_display)
        mean_gas_free = np.mean(gas_free_boe_display)
        mean_gas_assoc = np.mean(gas_assoc_boe_display)
        mean_thr = np.mean(thr_display_converted)

        pct_oil = (mean_oil / mean_thr * 100) if mean_thr > 0 else 0
        pct_cond = (mean_cond / mean_thr * 100) if mean_thr > 0 else 0
        pct_gas_free = (mean_gas_free / mean_thr * 100) if mean_thr > 0 else 0
        pct_gas_assoc = (mean_gas_assoc / mean_thr * 100) if mean_thr > 0 else 0

        summary_data = {
            "Fluid Type": [
                "Oil",
                "Condensate",
                "Free Gas",
                "Associated Gas",
                "Total Hydrocarbon Resource (THR)",
            ],
            f"Mean Recoverable ({boe_unit_detail})": [
                f"{mean_oil:.1f}",
                f"{mean_cond:.1f}",
                f"{mean_gas_free:.1f}",
                f"{mean_gas_assoc:.1f}",
                f"{mean_thr:.1f}",
            ],
            "% of THR": [
                f"{pct_oil:.1f}%",
                f"{pct_cond:.1f}%",
                f"{pct_gas_free:.1f}%",
                f"{pct_gas_assoc:.1f}%",
                "100.0%",
            ],
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    _render_column_distributions_overlay()


def _render_column_distributions_overlay() -> None:
    """Render overlay histogram for Oil and Gas Column Height distributions."""
    # Expect sampled columns in session_state (compute step should set these)
    oil = st.session_state.get("sOil_Column_Height", None)
    gas = st.session_state.get("sGas_Column_Height", None)
    if oil is None and gas is None:
        return

    st.subheader("Oil and Gas Column Distributions")
    fig = go.Figure()
    bins = 100

    if gas is not None:
        fig.add_histogram(
            x=np.asarray(gas, dtype=float),
            nbinsx=bins,
            name="Gas Column",
            marker=dict(color=color_for("Gas_Column_Height")),
            opacity=0.55,
        )
    if oil is not None:
        fig.add_histogram(
            x=np.asarray(oil, dtype=float),
            nbinsx=bins,
            name="Oil Column",
            marker=dict(color=color_for("Oil_Column_Height")),
            opacity=0.55,
        )
    fig.update_layout(barmode="overlay", height=420, margin=dict(l=40, r=30, t=30, b=60))
    fig.update_xaxes(title_text="Column Height (m)")
    fig.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def _render_saturation_summary() -> None:
    """Render saturation summary panel."""
    mode = st.session_state.get("sat_mode", "Global")
    
    st.markdown("### Saturation Assumptions")
    st.write(f"**Mode:** {mode}")

    so = st.session_state.get("Shc_oil")
    sg = st.session_state.get("Shc_gas")
    if so is not None and sg is not None:
        st.write(f"Median $S_{{\\mathrm{{oil}}}}$: {float(np.median(so)):.2f}")
        st.write(f"Median $S_{{\\mathrm{{gas}}}}$: {float(np.median(sg)):.2f}")


def _hist(name: str, arr, title: str) -> None:
    """Helper to render a histogram."""
    if arr is None:
        return
    fig = go.Figure()
    fig.add_histogram(
        x=np.asarray(arr, dtype=float),
        nbinsx=60,
        marker=dict(color=color_for(name)),
        name=title,
        opacity=0.85,
    )
    fig.update_layout(height=300, margin=dict(l=30, r=20, t=30, b=40))
    fig.update_xaxes(title_text=title)
    fig.update_yaxes(title_text="Frequency")
    st.plotly_chart(fig, use_container_width=True)


def render_saturation_plots() -> None:
    """Render saturation distribution plots."""
    mode = st.session_state.get("sat_mode", "Global")
    use_sw_global = st.session_state.get("global_sat_use_sw", False)
    
    st.subheader("Saturation Distributions")
    
    if mode.startswith("Global"):
        if use_sw_global:
            _hist("Sw_global", st.session_state.get("Sw_global"), r"$S_{w,\mathrm{global}}$")
        else:
            _hist("Shc_global", st.session_state.get("Shc_global"), r"$S_{\mathrm{hc}}$")
    elif mode.startswith("Water saturation"):
        _hist("Sw_oilzone", st.session_state.get("Sw_oilzone"), r"$S_{w,\mathrm{oil\,zone}}$")
        _hist("Sw_gaszone", st.session_state.get("Sw_gaszone"), r"$S_{w,\mathrm{gas\,zone}}$")
    else:  # Per phase
        _hist("Shc_oil_input", st.session_state.get("Shc_oil_input"), r"$S_{\mathrm{oil}}$")
        _hist("Shc_gas_input", st.session_state.get("Shc_gas_input"), r"$S_{\mathrm{gas}}$")

    _hist("Shc_oil", st.session_state.get("Shc_oil"), r"$S_{\mathrm{oil}}$ (derived)")
    _hist("Shc_gas", st.session_state.get("Shc_gas"), r"$S_{\mathrm{gas}}$ (derived)")
