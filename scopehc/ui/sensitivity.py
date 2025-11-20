from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import (
    UNIT_CONVERSIONS,
    collect_all_trial_data,
    color_for,
    get_unit_system,
    init_theme,
    make_hist_cdf_figure,
    summarize_array,
    summary_table,
    render_custom_navigation,
    render_sidebar_settings,
    render_sidebar_help,
)


def _ensure_results():
    trial_data = st.session_state.get("trial_data")
    results = st.session_state.get("results_cache")

    if results is None or trial_data is None:
        trial_data = collect_all_trial_data()
        results = st.session_state.get("results_cache")

    return trial_data, results


def _render_violin_plots(
    oil_boe: np.ndarray,
    gas_free_boe: np.ndarray,
    gas_assoc_boe: np.ndarray,
    cond_boe: np.ndarray,
    unit_system: str,
    gas_scf_per_boe: float,
) -> None:
    if unit_system == "oilfield":
        oil_display = oil_boe / 1e6
        gas_free_display = gas_free_boe / 1e6
        gas_assoc_display = gas_assoc_boe / 1e6
        cond_display = cond_boe / 1e6
        boe_unit = "MBOE"
    else:
        conversion = UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        oil_display = oil_boe * conversion
        gas_free_display = gas_free_boe * conversion
        gas_assoc_display = gas_assoc_boe * conversion
        cond_display = cond_boe * conversion
        boe_unit = "Mm³ BOE"

    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            y=oil_display,
            name=f"Oil ({boe_unit})",
            box_visible=True,
            line_color=color_for("Oil_STB_rec"),
            fillcolor=color_for("Oil_STB_rec"),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Violin(
            y=gas_free_display,
            name=f"Free Gas ({boe_unit})",
            box_visible=True,
            line_color=color_for("Gas_free_scf_rec"),
            fillcolor=color_for("Gas_free_scf_rec"),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Violin(
            y=gas_assoc_display,
            name=f"Associated Gas ({boe_unit})",
            box_visible=True,
            line_color=color_for("Gas_assoc_scf_rec"),
            fillcolor=color_for("Gas_assoc_scf_rec"),
            opacity=0.6,
        )
    )
    fig.add_trace(
        go.Violin(
            y=cond_display,
            name=f"Condensate ({boe_unit})",
            box_visible=True,
            line_color=color_for("Cond_STB_rec"),
            fillcolor=color_for("Cond_STB_rec"),
            opacity=0.6,
        )
    )
    fig.update_layout(
        title="Hydrocarbon Volume Distribution Analysis (All Fluids in BOE)",
        yaxis_title=f"Volume ({boe_unit})",
        height=500,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"All volumes are shown in BOE units for direct comparison. "
        f"Gas conversion: {gas_scf_per_boe:,.0f} scf/BOE"
    )


def _render_thr_breakdown(
    thr_display: np.ndarray,
    thr_unit: str,
    oil_boe: np.ndarray,
    cond_boe: np.ndarray,
    gas_free_boe: np.ndarray,
    gas_assoc_boe: np.ndarray,
    unit_system: str,
) -> None:
    st.markdown("**THR Composition (mean):**")
    mean_oil = np.mean(oil_boe) / 1e6
    mean_cond = np.mean(cond_boe) / 1e6
    mean_free = np.mean(gas_free_boe) / 1e6
    mean_assoc = np.mean(gas_assoc_boe) / 1e6
    mean_thr = float(np.mean(thr_display))

    if mean_thr > 0:
        pct_oil = (mean_oil / mean_thr) * 100
        pct_cond = (mean_cond / mean_thr) * 100
        pct_free = (mean_free / mean_thr) * 100
        pct_assoc = (mean_assoc / mean_thr) * 100
        composition_data = pd.DataFrame(
            {
                "Component": ["Oil", "Condensate", "Free Gas", "Associated Gas", "**Total**"],
                f"Mean ({thr_unit})": [
                    f"{mean_oil:.1f}",
                    f"{mean_cond:.1f}",
                    f"{mean_free:.1f}",
                    f"{mean_assoc:.1f}",
                    f"**{mean_thr:.1f}**",
                ],
                "% of THR": [
                    f"{pct_oil:.1f}%",
                    f"{pct_cond:.1f}%",
                    f"{pct_free:.1f}%",
                    f"{pct_assoc:.1f}%",
                    "**100%**",
                ],
            }
        )
        st.dataframe(composition_data, use_container_width=True, hide_index=True)

    st.markdown("#### THR Breakdown by Fluid Type (BOE)")
    if unit_system == "oilfield":
        conversion = 1e6
        boe_unit = "MBOE"
        oil_display = oil_boe / conversion
        cond_display = cond_boe / conversion
        gas_free_display = gas_free_boe / conversion
        gas_assoc_display = gas_assoc_boe / conversion
        thr_display_units = thr_display
    else:
        conversion = UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        boe_unit = "Mm³ BOE"
        oil_display = oil_boe * conversion
        cond_display = cond_boe * conversion
        gas_free_display = gas_free_boe * conversion
        gas_assoc_display = gas_assoc_boe * conversion
        thr_display_units = thr_display * conversion

    mean_oil = np.mean(oil_display)
    mean_cond = np.mean(cond_display)
    mean_gas_free = np.mean(gas_free_display)
    mean_gas_assoc = np.mean(gas_assoc_display)
    mean_thr = np.mean(thr_display_units)

    pct_oil = (mean_oil / mean_thr * 100) if mean_thr > 0 else 0
    pct_cond = (mean_cond / mean_thr * 100) if mean_thr > 0 else 0
    pct_gas_free = (mean_gas_free / mean_thr * 100) if mean_thr > 0 else 0
    pct_gas_assoc = (mean_gas_assoc / mean_thr * 100) if mean_thr > 0 else 0
    summary_df = pd.DataFrame(
        {
            "Fluid Type": [
                "Oil",
                "Condensate",
                "Free Gas",
                "Associated Gas",
                "Total Hydrocarbon Resource (THR)",
            ],
            f"Mean Recoverable ({boe_unit})": [
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
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def _create_tornado_plot(unit_system: str, target_volume: str, gas_scf_per_boe: float, results: dict = None):
    """
    Create tornado plot for sensitivity analysis.
    
    CRITICAL: Use actual simulation results from results_cache for base case values,
    not recalculated mean values, to ensure consistency with Results page.
    
    Args:
        unit_system: Unit system ("oilfield" or "metric")
        target_volume: Target volume for sensitivity ("Oil", "Free Gas", etc.)
        gas_scf_per_boe: Gas to BOE conversion factor
        results: Results dictionary from results_cache (if None, will recalculate)
    """
    sGRV_m3_final = st.session_state.get("sGRV_m3_final", np.array([0]))
    sNtG = st.session_state.get("sNtG", np.array([0]))
    sp = st.session_state.get("sp", np.array([0]))
    f_oil = st.session_state.get("f_oil", np.array([0]))
    sRF_oil = st.session_state.get("sRF_oil", np.array([0]))
    sRF_gas = st.session_state.get("sRF_gas", np.array([0]))
    sInvBo = st.session_state.get("sInvBo", np.array([0]))
    sBg = st.session_state.get("sBg", np.array([0]))
    sGOR = st.session_state.get("sGOR", np.array([0]))
    sCY = st.session_state.get("sCY", None)
    sRF_cond = st.session_state.get("sRF_cond", None)
    
    # Get saturation arrays
    Shc_oil = st.session_state.get("Shc_oil", None)
    Shc_gas = st.session_state.get("Shc_gas", None)
    shc_oil_mean = float(np.mean(Shc_oil)) if Shc_oil is not None else 1.0
    shc_gas_mean = float(np.mean(Shc_gas)) if Shc_gas is not None else 1.0

    grv_mean = float(np.mean(sGRV_m3_final / 1e6))
    ntg_mean = float(np.mean(sNtG))
    porosity_mean = float(np.mean(sp))
    
    # CRITICAL: For Oil+Gas cases, calculate f_oil from actual GRV split if available
    # This handles depth-based methods where GOC determines the split, not a fraction
    fluid_type = st.session_state.get("fluid_type", "Oil")
    sGRV_oil_m3 = st.session_state.get("sGRV_oil_m3", None)
    sGRV_gas_m3 = st.session_state.get("sGRV_gas_m3", None)
    
    if fluid_type == "Oil + Gas" and sGRV_oil_m3 is not None and sGRV_gas_m3 is not None:
        # Use actual GRV split from simulation (from GOC or other method)
        grv_oil_mean = float(np.mean(sGRV_oil_m3))
        grv_gas_mean = float(np.mean(sGRV_gas_m3))
        grv_total_mean = grv_oil_mean + grv_gas_mean
        if grv_total_mean > 0:
            f_oil_mean = grv_oil_mean / grv_total_mean
        else:
            # Fallback to stored f_oil if total is zero
            f_oil_mean = float(np.mean(f_oil)) if len(f_oil) > 0 and np.any(f_oil > 0) else 0.5
    elif fluid_type == "Oil":
        # Oil-only: all GRV is oil
        f_oil_mean = 1.0
    elif fluid_type == "Gas":
        # Gas-only: all GRV is gas
        f_oil_mean = 0.0
    else:
        # Fallback to stored f_oil value
        f_oil_mean = float(np.mean(f_oil)) if len(f_oil) > 0 and np.any(f_oil > 0) else 0.5
    
    rf_oil_mean = float(np.mean(sRF_oil))
    rf_gas_mean = float(np.mean(sRF_gas))
    invbo_mean = float(np.mean(sInvBo))
    bg_mean = float(np.mean(sBg))
    gor_mean = float(np.mean(sGOR))

    PV_total_base = (grv_mean * 1e6) * ntg_mean * porosity_mean
    PV_oil_base = PV_total_base * f_oil_mean
    PV_gas_base = PV_total_base * (1 - f_oil_mean)
    # Apply saturation
    PV_oil_hc_base = PV_oil_base * shc_oil_mean
    PV_gas_hc_base = PV_gas_base * shc_gas_mean

    base_oil_m3 = PV_oil_hc_base * rf_oil_mean * invbo_mean
    base_gas_m3 = (PV_gas_hc_base * rf_gas_mean) / bg_mean
    base_assoc_m3 = base_oil_m3 * gor_mean

    cy_mean = float(np.mean(sCY)) if sCY is not None else None
    rf_cond_mean = float(np.mean(sRF_cond)) if sRF_cond is not None else None
    if cy_mean is not None and rf_cond_mean is not None:
        base_cond_m3 = base_gas_m3 * cy_mean * rf_cond_mean
    else:
        base_cond_m3 = 0.0

    # CRITICAL: Use actual simulation results for base case if available
    # This ensures consistency with Results page
    if results is not None:
        # Get actual mean values from simulation results
        Oil_STB_rec = np.asarray(results.get("Oil_STB_rec", np.array([0])), dtype=float)
        Gas_free_scf_rec = np.asarray(results.get("Gas_free_scf_rec", np.array([0])), dtype=float)
        Gas_assoc_scf_rec = np.asarray(results.get("Gas_assoc_scf_rec", np.array([0])), dtype=float)
        Cond_STB_rec = np.asarray(results.get("Cond_STB_rec", np.array([0])), dtype=float)
        Total_surface_BOE = np.asarray(results.get("Total_surface_BOE", np.array([0])), dtype=float)
        
        # Calculate means from actual simulation results
        base_oil_stb_mean = float(np.mean(Oil_STB_rec))
        base_gas_scf_mean = float(np.mean(Gas_free_scf_rec))
        base_assoc_scf_mean = float(np.mean(Gas_assoc_scf_rec))
        base_cond_stb_mean = float(np.mean(Cond_STB_rec))
        base_thr_boe_mean = float(np.mean(Total_surface_BOE))
        
        if unit_system == "oilfield":
            base_oil = base_oil_stb_mean / 1e6  # MMSTB
            base_gas = base_gas_scf_mean / 1e9  # Bscf
            base_assoc = base_assoc_scf_mean / 1e9  # Bscf
            base_cond = base_cond_stb_mean / 1e6  # MMSTB
            base_thr = base_thr_boe_mean / 1e6  # MBOE
        else:
            # Metric: convert STB to m³, scf to m³, BOE to m³
            base_oil = base_oil_stb_mean * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6  # Mm³
            base_gas = base_gas_scf_mean * UNIT_CONVERSIONS["scf_to_m3"] / 1e9  # Bm³
            base_assoc = base_assoc_scf_mean * UNIT_CONVERSIONS["scf_to_m3"] / 1e9  # Bm³
            base_cond = base_cond_stb_mean * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6  # Mm³
            base_thr = base_thr_boe_mean * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6  # Mm³ BOE
    else:
        # Fallback: recalculate from mean inputs (less accurate, but works if results not available)
        base_oil_stb = base_oil_m3 * UNIT_CONVERSIONS["m3_to_bbl"]
        base_gas_scf = base_gas_m3 * UNIT_CONVERSIONS["m3_to_scf"]
        base_assoc_scf = base_assoc_m3 * UNIT_CONVERSIONS["m3_to_scf"]
        base_cond_stb = base_cond_m3 * UNIT_CONVERSIONS["m3_to_bbl"]
        base_thr_boe = base_oil_stb + base_cond_stb + (base_gas_scf + base_assoc_scf) / gas_scf_per_boe

        if unit_system == "oilfield":
            base_oil = base_oil_m3 * UNIT_CONVERSIONS["m3_to_bbl"] / 1e6
            base_gas = base_gas_m3 * UNIT_CONVERSIONS["m3_to_scf"] / 1e9
            base_assoc = base_assoc_m3 * UNIT_CONVERSIONS["m3_to_scf"] / 1e9
            base_cond = base_cond_m3 * UNIT_CONVERSIONS["m3_to_bbl"] / 1e6
            base_thr = base_thr_boe / 1e6
        else:
            base_oil = base_oil_m3 / 1e6
            base_gas = base_gas_m3 / 1e9
            base_assoc = base_assoc_m3 / 1e9
            base_cond = base_cond_m3 / 1e6
            base_thr = base_thr_boe * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6

    if target_volume == "Oil":
        base_value = base_oil
    elif target_volume == "Free Gas":
        base_value = base_gas
    elif target_volume == "Associated Gas":
        base_value = base_assoc
    elif target_volume == "Condensate":
        base_value = base_cond
    elif target_volume == "Total Hydrocarbon Resource (THR)":
        base_value = base_thr
    else:
        base_value = base_oil

    def calculate_volume(grv_val, ntg_val, porosity_val, rf_oil_val, rf_gas_val, invbo_val, bg_val, gor_val, shc_oil_val=None, shc_gas_val=None):
        PV_total = (grv_val * 1e6) * ntg_val * porosity_val
        PV_oil = PV_total * f_oil_mean
        PV_gas = PV_total * (1 - f_oil_mean)
        
        # Apply saturation
        shc_oil_use = shc_oil_val if shc_oil_val is not None else shc_oil_mean
        shc_gas_use = shc_gas_val if shc_gas_val is not None else shc_gas_mean
        PV_oil_hc = PV_oil * shc_oil_use
        PV_gas_hc = PV_gas * shc_gas_use

        oil_m3 = PV_oil_hc * rf_oil_val * invbo_val
        gas_m3 = (PV_gas_hc * rf_gas_val) / bg_val
        assoc_m3 = oil_m3 * gor_val

        # CRITICAL: Condensate calculation must match compute.py formula exactly
        # From compute.py line 261: Cond_STB_rec = Gas_free_scf_rec * (CY_STB_per_MMscf / 1_000_000.0) * RF_cond
        # Condensate = (Free Gas in scf) × (CY in STB/MMscf / 1,000,000) × RF_cond
        # Then convert STB to m³ for consistency with other volumes
        if cy_mean is not None and rf_cond_mean is not None:
            # Convert gas_m3 to scf first (this is Gas_free_scf_rec equivalent)
            gas_scf_for_cond = gas_m3 * UNIT_CONVERSIONS["m3_to_scf"]
            # Calculate condensate in STB: gas_scf × (CY / 1e6) × RF_cond
            # This matches: Cond_STB_rec = Gas_free_scf_rec * (CY_STB_per_MMscf / 1_000_000.0) * RF_cond
            cond_stb = gas_scf_for_cond * (cy_mean / 1_000_000.0) * rf_cond_mean
            # Convert STB to m³ for consistency
            cond_m3 = cond_stb * UNIT_CONVERSIONS["bbl_to_m3"]
        else:
            cond_stb = 0.0
            cond_m3 = 0.0

        oil_stb = oil_m3 * UNIT_CONVERSIONS["m3_to_bbl"]
        gas_scf = gas_m3 * UNIT_CONVERSIONS["m3_to_scf"]
        assoc_scf = assoc_m3 * UNIT_CONVERSIONS["m3_to_scf"]
        thr_boe = oil_stb + cond_stb + (gas_scf + assoc_scf) / gas_scf_per_boe

        if unit_system == "oilfield":
            oil_vol = oil_m3 * UNIT_CONVERSIONS["m3_to_bbl"] / 1e6
            gas_vol = gas_m3 * UNIT_CONVERSIONS["m3_to_scf"] / 1e9
            assoc_vol = assoc_m3 * UNIT_CONVERSIONS["m3_to_scf"] / 1e9
            cond_vol = cond_m3 * UNIT_CONVERSIONS["m3_to_bbl"] / 1e6
            thr_vol = thr_boe / 1e6
        else:
            oil_vol = oil_m3 / 1e6
            gas_vol = gas_m3 / 1e9
            assoc_vol = assoc_m3 / 1e9
            cond_vol = cond_m3 / 1e6
            thr_vol = thr_boe * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6

        return oil_vol, gas_vol, assoc_vol, cond_vol, thr_vol

    sensitivities: list[dict[str, float]] = []
    debug_info: list[str] = []

    def append_sensitivity(name, p10_val, p90_val, base_tuple):
        nonlocal sensitivities
        oil_base, gas_base, assoc_base, cond_base, thr_base = base_tuple
        oil_p10, gas_p10, assoc_p10, cond_p10, thr_p10 = p10_val
        oil_p90, gas_p90, assoc_p90, cond_p90, thr_p90 = p90_val

        if target_volume == "Oil":
            impact_p10 = oil_p10 - oil_base
            impact_p90 = oil_p90 - oil_base
        elif target_volume == "Free Gas":
            impact_p10 = gas_p10 - gas_base
            impact_p90 = gas_p90 - gas_base
        elif target_volume == "Associated Gas":
            impact_p10 = assoc_p10 - assoc_base
            impact_p90 = assoc_p90 - assoc_base
        elif target_volume == "Condensate":
            impact_p10 = cond_p10 - cond_base
            impact_p90 = cond_p90 - cond_base
        elif target_volume == "Total Hydrocarbon Resource (THR)":
            impact_p10 = thr_p10 - thr_base
            impact_p90 = thr_p90 - thr_base
        else:
            impact_p10 = oil_p10 - oil_base
            impact_p90 = oil_p90 - oil_base

        impact_range = abs(impact_p90 - impact_p10)
        sensitivities.append(
            {
                "Parameter": name,
                "P10_Impact": impact_p10,
                "P90_Impact": impact_p90,
                "Range": impact_range,
            }
        )

    mean_tuple = calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean)

    grv_stats = summarize_array(sGRV_m3_final / 1e6)
    if grv_stats:
        grv_p10 = grv_stats.get("P10", grv_mean)
        grv_p90 = grv_stats.get("P90", grv_mean)
        append_sensitivity(
            "Gross Rock Volume (GRV)",
            calculate_volume(grv_p10, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            calculate_volume(grv_p90, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            mean_tuple,
        )

    ntg_stats = summarize_array(sNtG)
    if ntg_stats:
        ntg_p10 = ntg_stats.get("P10", ntg_mean)
        ntg_p90 = ntg_stats.get("P90", ntg_mean)
        append_sensitivity(
            "Net-to-Gross (NtG)",
            calculate_volume(grv_mean, ntg_p10, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            calculate_volume(grv_mean, ntg_p90, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            mean_tuple,
        )

    porosity_stats = summarize_array(sp)
    if porosity_stats:
        p10 = porosity_stats.get("P10", porosity_mean)
        p90 = porosity_stats.get("P90", porosity_mean)
        append_sensitivity(
            "Porosity",
            calculate_volume(grv_mean, ntg_mean, p10, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            calculate_volume(grv_mean, ntg_mean, p90, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            mean_tuple,
        )

    rf_oil_stats = summarize_array(sRF_oil)
    if rf_oil_stats:
        p10 = rf_oil_stats.get("P10", rf_oil_mean)
        p90 = rf_oil_stats.get("P90", rf_oil_mean)
        append_sensitivity(
            "Oil Recovery Factor (RF_oil)",
            calculate_volume(grv_mean, ntg_mean, porosity_mean, p10, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            calculate_volume(grv_mean, ntg_mean, porosity_mean, p90, rf_gas_mean, invbo_mean, bg_mean, gor_mean),
            mean_tuple,
        )

    rf_gas_stats = summarize_array(sRF_gas)
    if rf_gas_stats:
        p10 = rf_gas_stats.get("P10", rf_gas_mean)
        p90 = rf_gas_stats.get("P90", rf_gas_mean)
        append_sensitivity(
            "Free Gas Recovery Factor (RF_gas)",
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, p10, invbo_mean, bg_mean, gor_mean),
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, p90, invbo_mean, bg_mean, gor_mean),
            mean_tuple,
        )

    invbo_stats = summarize_array(sInvBo)
    if invbo_stats:
        p10 = invbo_stats.get("P10", invbo_mean)
        p90 = invbo_stats.get("P90", invbo_mean)
        append_sensitivity(
            "Inverse FVF (1/Bo)",
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, p10, bg_mean, gor_mean),
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, p90, bg_mean, gor_mean),
            mean_tuple,
        )

    bg_stats = summarize_array(sBg)
    if bg_stats:
        p10 = bg_stats.get("P10", bg_mean)
        p90 = bg_stats.get("P90", bg_mean)
        append_sensitivity(
            "Gas FVF (Bg)",
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, p10, gor_mean),
            calculate_volume(grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, p90, gor_mean),
            mean_tuple,
        )

    gor_stats = summarize_array(sGOR)
    if gor_stats:
        gor_p10 = gor_stats.get("P10", gor_mean)
        gor_p90 = gor_stats.get("P90", gor_mean)
        p10_tuple = calculate_volume(
            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_p10
        )
        p90_tuple = calculate_volume(
            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean, invbo_mean, bg_mean, gor_p90
        )

        if target_volume == "Associated Gas":
            gor_p10_impact = p10_tuple[2] - mean_tuple[2]
            gor_p90_impact = p90_tuple[2] - mean_tuple[2]
        elif target_volume == "Total Hydrocarbon Resource (THR)":
            gor_p10_impact = p10_tuple[4] - mean_tuple[4]
            gor_p90_impact = p90_tuple[4] - mean_tuple[4]
        else:
            gor_p10_impact = 0.0
            gor_p90_impact = 0.0

        gor_range = abs(gor_p90_impact - gor_p10_impact)
        if gor_range > 0:
            sensitivities.append(
                {
                    "Parameter": "Gas-Oil Ratio (GOR)",
                    "P10_Impact": gor_p10_impact,
                    "P90_Impact": gor_p90_impact,
                    "Range": gor_range,
                }
            )

    # Add saturation sensitivity if available
    mode = st.session_state.get("sat_mode", "Global")
    use_sw_global = st.session_state.get("global_sat_use_sw", False)
    
    if Shc_oil is not None and Shc_gas is not None:
        shc_oil_stats = summarize_array(Shc_oil)
        shc_gas_stats = summarize_array(Shc_gas)
        
        if mode.startswith("Global"):
            if use_sw_global:
                if "Sw_global" in st.session_state:
                    sw_global = st.session_state.get("Sw_global")
                    if sw_global is not None:
                        sw_global_stats = summarize_array(sw_global)
                        if sw_global_stats:
                            p10_sw = sw_global_stats.get("P90", 0.0)  # Higher Sw = lower Shc
                            p90_sw = sw_global_stats.get("P10", 0.0)
                            p10 = 1.0 - p10_sw
                            p90 = 1.0 - p90_sw
                            p10_tuple = calculate_volume(
                                grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                                invbo_mean, bg_mean, gor_mean, p10, p90
                            )
                            p90_tuple = calculate_volume(
                                grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                                invbo_mean, bg_mean, gor_mean, p90, p10
                            )
                            append_sensitivity(r"$S_{w,\mathrm{global}}$", p10_tuple, p90_tuple, mean_tuple)
            else:
                if "Shc_global" in st.session_state:
                    shc_global = st.session_state.get("Shc_global")
                    if shc_global is not None:
                        shc_global_stats = summarize_array(shc_global)
                        if shc_global_stats:
                            p10 = shc_global_stats.get("P10", shc_oil_mean)
                            p90 = shc_global_stats.get("P90", shc_oil_mean)
                            p10_tuple = calculate_volume(
                                grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                                invbo_mean, bg_mean, gor_mean, p10, p90
                            )
                            p90_tuple = calculate_volume(
                                grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                                invbo_mean, bg_mean, gor_mean, p90, p10
                            )
                            append_sensitivity(r"$S_{\mathrm{hc}}$ (Global)", p10_tuple, p90_tuple, mean_tuple)
        elif mode.startswith("Water saturation"):
            if "Sw_oilzone" in st.session_state and "Sw_gaszone" in st.session_state:
                sw_oil = st.session_state.get("Sw_oilzone")
                sw_gas = st.session_state.get("Sw_gaszone")
                if sw_oil is not None and sw_gas is not None:
                    sw_oil_stats = summarize_array(sw_oil)
                    sw_gas_stats = summarize_array(sw_gas)
                    if sw_oil_stats:
                        p10 = 1.0 - sw_oil_stats.get("P90", 1.0 - shc_oil_mean)
                        p90 = 1.0 - sw_oil_stats.get("P10", 1.0 - shc_oil_mean)
                        p10_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, p10, shc_gas_mean
                        )
                        p90_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, p90, shc_gas_mean
                        )
                        append_sensitivity(r"$S_{w,\mathrm{oil\,zone}}$", p10_tuple, p90_tuple, mean_tuple)
                    if sw_gas_stats:
                        p10 = 1.0 - sw_gas_stats.get("P90", 1.0 - shc_gas_mean)
                        p90 = 1.0 - sw_gas_stats.get("P10", 1.0 - shc_gas_mean)
                        p10_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, shc_oil_mean, p10
                        )
                        p90_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, shc_oil_mean, p90
                        )
                        append_sensitivity(r"$S_{w,\mathrm{gas\,zone}}$", p10_tuple, p90_tuple, mean_tuple)
        else:  # Per phase
            if "Shc_oil_input" in st.session_state and "Shc_gas_input" in st.session_state:
                shc_oil_in = st.session_state.get("Shc_oil_input")
                shc_gas_in = st.session_state.get("Shc_gas_input")
                if shc_oil_in is not None and shc_gas_in is not None:
                    shc_oil_in_stats = summarize_array(shc_oil_in)
                    shc_gas_in_stats = summarize_array(shc_gas_in)
                    if shc_oil_in_stats:
                        p10 = shc_oil_in_stats.get("P10", shc_oil_mean)
                        p90 = shc_oil_in_stats.get("P90", shc_oil_mean)
                        p10_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, p10, shc_gas_mean
                        )
                        p90_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, p90, shc_gas_mean
                        )
                        append_sensitivity(r"$S_{\mathrm{oil}}$", p10_tuple, p90_tuple, mean_tuple)
                    if shc_gas_in_stats:
                        p10 = shc_gas_in_stats.get("P10", shc_gas_mean)
                        p90 = shc_gas_in_stats.get("P90", shc_gas_mean)
                        p10_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, shc_oil_mean, p10
                        )
                        p90_tuple = calculate_volume(
                            grv_mean, ntg_mean, porosity_mean, rf_oil_mean, rf_gas_mean,
                            invbo_mean, bg_mean, gor_mean, shc_oil_mean, p90
                        )
                        append_sensitivity(r"$S_{\mathrm{gas}}$", p10_tuple, p90_tuple, mean_tuple)

    sensitivities = [s for s in sensitivities if s["Range"] > 0]
    # Sort by maximum absolute impact (largest at top)
    sensitivities.sort(
        key=lambda x: max(abs(x["P10_Impact"]), abs(x["P90_Impact"])),
        reverse=True,
    )

    # Collect labels in sorted order for y-axis categoryarray
    labels = [sens["Parameter"] for sens in sensitivities]

    fig = go.Figure()
    for sens in sensitivities:
        param_color = color_for(sens["Parameter"], color_for("Total_surface_BOE"))
        fig.add_trace(
            go.Bar(
                y=[sens["Parameter"]],
                x=[sens["P10_Impact"]],
                orientation="h",
                name="P10 Impact",
                marker_color=param_color,
                opacity=0.7,
                showlegend=False,
                hovertemplate=(
                    f"{sens['Parameter']}<br>"
                    f"P10 Impact: {sens['P10_Impact']:.2f}<br>"
                    f"Base Case: {base_value:.2f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                y=[sens["Parameter"]],
                x=[sens["P90_Impact"]],
                orientation="h",
                name="P90 Impact",
                marker_color=param_color,
                opacity=0.4,
                showlegend=False,
                hovertemplate=(
                    f"{sens['Parameter']}<br>"
                    f"P90 Impact: {sens['P90_Impact']:.2f}<br>"
                    f"Base Case: {base_value:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Base Case")

    if target_volume == "Oil":
        units = "MMSTB" if unit_system == "oilfield" else "Mm³"
        title = f"Recoverable Oil Sensitivity Analysis (Impact)<br><sub>Base Case: {base_value:.2f} {units}</sub>"
        xaxis_title = f"Impact on Recoverable Oil ({units})"
    elif target_volume == "Free Gas":
        units = "Bscf" if unit_system == "oilfield" else "Bm³"
        title = f"Recoverable Free Gas Sensitivity Analysis (Impact)<br><sub>Base Case: {base_value:.2f} {units}</sub>"
        xaxis_title = f"Impact on Recoverable Free Gas ({units})"
    elif target_volume == "Associated Gas":
        units = "Bscf" if unit_system == "oilfield" else "Bm³"
        title = f"Recoverable Associated Gas Sensitivity Analysis (Impact)<br><sub>Base Case: {base_value:.2f} {units}</sub>"
        xaxis_title = f"Impact on Recoverable Associated Gas ({units})"
    elif target_volume == "Condensate":
        units = "MMSTB" if unit_system == "oilfield" else "Mm³"
        title = f"Recoverable Condensate Sensitivity Analysis (Impact)<br><sub>Base Case: {base_value:.2f} {units}</sub>"
        xaxis_title = f"Impact on Recoverable Condensate ({units})"
    else:
        units = "MBOE" if unit_system == "oilfield" else "Mm³ BOE"
        title = f"Total Hydrocarbon Resource Sensitivity Analysis (Impact)<br><sub>Base Case: {base_value:.2f} {units}</sub>"
        xaxis_title = f"Impact on THR ({units})"

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis=dict(
            title="Input Parameters",
            categoryorder="array",
            categoryarray=list(reversed(labels)),  # Reverse so largest impact appears at top
        ),
        barmode="relative",
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    if sensitivities:
        max_impact = max(
            [abs(s["P10_Impact"]) for s in sensitivities]
            + [abs(s["P90_Impact"]) for s in sensitivities]
        )
        fig.update_xaxes(range=[-max_impact * 1.1, max_impact * 1.1])

    return fig, sensitivities, base_value, units


def _render_sensitivity_summary(
    sensitivities: list[dict[str, float]],
    base_value: float,
    target_volume: str,
    unit_system: str,
) -> None:
    st.markdown("### Sensitivity Summary (P10/P90 Method)")

    if target_volume == "Oil":
        units = "MMSTB" if unit_system == "oilfield" else "Mm³"
    elif target_volume in {"Free Gas", "Associated Gas"}:
        units = "Bscf" if unit_system == "oilfield" else "Bm³"
    elif target_volume == "Condensate":
        units = "MMSTB" if unit_system == "oilfield" else "Mm³"
    else:
        units = "MBOE" if unit_system == "oilfield" else "Mm³ BOE"

    rows = []
    for sens in sensitivities:
        if base_value != 0:
            relative = sens["Range"] / base_value * 100
            relative_str = f"{relative:.1f}%"
        else:
            relative_str = "N/A"
        rows.append(
            {
                "Parameter": sens["Parameter"],
                f"P10 Impact ({units})": f"{sens['P10_Impact']:.2f}",
                f"P90 Impact ({units})": f"{sens['P90_Impact']:.2f}",
                f"Range ({units})": f"{sens['Range']:.2f}",
                "Relative Impact (%)": relative_str,
            }
        )

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No significant sensitivities for the selected volume.")

    st.markdown(
        "**Tornado Plot Interpretation:**\n"
        "- **Left side (darker bars)**: Lower impact relative to base case\n"
        "- **Right side (lighter bars)**: Higher impact relative to base case\n"
        "- **Bar length**: Magnitude of impact\n"
        "- **Order**: Parameters sorted by total impact (P90 - P10)\n\n"
        "**Methodology:**\n"
        "1. Calculate volumes using P10 and P90 for each parameter\n"
        "2. Compare to base case (mean inputs)\n"
        "3. Range represents total sensitivity\n\n"
        "**Note:** Small impacts may signal narrow distributions or limited influence on the selected volume."
    )


def render() -> None:
    """Render the sensitivity analysis page."""
    st.session_state["current_page"] = "_pages_disabled/05_Sensitivity.py"
    st.header("Sensitivity Analysis")

    trial_data, results = _ensure_results()
    if results is None or trial_data is None:
        st.info("Configure inputs and run the simulation first to view sensitivity results.")
        return

    unit_system = get_unit_system()
    gas_scf_per_boe = st.session_state.get("gas_scf_per_boe", 6000.0)

    Oil_BOE = np.asarray(results["Oil_BOE"], dtype=float)
    Cond_BOE = np.asarray(results["Cond_BOE"], dtype=float)
    Gas_free_BOE = np.asarray(results["Gas_free_BOE"], dtype=float)
    Gas_assoc_BOE = np.asarray(results["Gas_assoc_BOE"], dtype=float)
    Total_gas_BOE = np.asarray(results["Total_gas_BOE"], dtype=float)
    thr_boe = np.asarray(results["Total_surface_BOE"], dtype=float)

    st.markdown("### Hydrocarbon Volume Distribution Analysis (BOE - comparable across all fluids)")
    _render_violin_plots(Oil_BOE, Gas_free_BOE, Gas_assoc_BOE, Cond_BOE, unit_system, gas_scf_per_boe)

    st.markdown("---")
    st.markdown("## Total Hydrocarbon Resource (THR)")
    if unit_system == "oilfield":
        thr_display = thr_boe / 1e6
        thr_unit = "MBOE"
    else:
        thr_display = thr_boe * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
        thr_unit = "Mm³ BOE"

    thr_stats = summarize_array(thr_display)

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
        st.caption(
            "_Interpretation:_ THR represents total recoverable volumes converted to oil equivalent. "
            "Wider distributions indicate greater uncertainty."
        )
        st.dataframe(summary_table(thr_display, decimals=2), use_container_width=True)
        st.caption(
            f"THR includes Oil + Condensate + Gas/BOE. Default 6.0 Mscf/BOE; company standards may vary. "
            f"Current factor: {gas_scf_per_boe:,.0f} scf/BOE"
        )
        _render_thr_breakdown(
            thr_display,
            thr_unit,
            Oil_BOE,
            Cond_BOE,
            Gas_free_BOE,
            Gas_assoc_BOE,
            unit_system,
        )

    st.markdown("---")
    st.markdown("## Tornado Sensitivities")
    target_volume = st.selectbox(
        "Select target volume for sensitivity analysis:",
        ["Oil", "Free Gas", "Associated Gas", "Condensate", "Total Hydrocarbon Resource (THR)"],
        index=0,
        help="Choose which recoverable volume to analyse for sensitivity",
    )

    tornado_fig, sensitivities, base_value, units = _create_tornado_plot(
        unit_system, target_volume, gas_scf_per_boe, results
    )

    if base_value == 0:
        st.warning(
            "⚠️ Base case recoverable volume is zero. Check input parameters for realism before relying on the sensitivity plot."
        )

    st.plotly_chart(tornado_fig, use_container_width=True)
    _render_sensitivity_summary(sensitivities, base_value, target_volume, unit_system)

