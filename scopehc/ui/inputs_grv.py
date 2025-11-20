from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .common import *  # noqa: F401,F403 - re-export shared helpers
from scopehc.geom import interpolate_gcf, get_gcf_lookup_table


def _get_scalar_from_state(ss, key: str, default: float = 0.0) -> float:
    """
    Safely extract a scalar value from session state, handling both arrays and scalars.
    
    Args:
        ss: Session state dict-like object
        key: Key to look up
        default: Default value if key not found
        
    Returns:
        Scalar float value (mean if array, direct value if scalar)
    """
    val = ss.get(key, default)
    if isinstance(val, np.ndarray):
        return float(np.mean(val))
    return float(val)


def _ensure_final_grv_arrays(grv_option: str, num_sims: int) -> None:
    """
    CRITICAL: Ensure sGRV_oil_m3 and sGRV_gas_m3 are ALWAYS set correctly based on CURRENT fluid_type.
    This function is called at the end of GRV calculation to ensure consistency.
    
    IMPORTANT: For depth-based methods, this function should NOT overwrite existing calculated arrays
    unless they are clearly invalid (empty or wrong size). The depth-based calculation is the source of truth.
    
    Args:
        grv_option: Current GRV method selected
        num_sims: Number of simulation trials
    """
    if "sGRV_m3_final" not in st.session_state:
        return
    
    sGRV_m3_final = np.asarray(st.session_state["sGRV_m3_final"], dtype=float)
    fluid_type = st.session_state.get("fluid_type", "Oil")
    
    # Check if this is a depth-based method
    depth_based_methods = {
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    }
    is_depth_based = grv_option in depth_based_methods
    
    # CRITICAL: For depth-based methods, if arrays already exist and have valid data,
    # DO NOT overwrite them - they were calculated from GOC and are the source of truth
    # The depth-based calculation is independent of sGRV_m3_final, so we must preserve those values
    # This check happens BEFORE any recalculation logic to prevent overwriting valid depth-based arrays
    if is_depth_based:
        # Check if arrays are marked as depth-based calculated (most reliable check)
        if st.session_state.get("_grv_arrays_depth_based", False):
            stored_method = st.session_state.get("_grv_arrays_method", "")
            # CRITICAL: Only preserve arrays if they were calculated for the CURRENT method
            # If the method changed, we need to recalculate, so don't preserve old method's arrays
            if stored_method == grv_option:
                # Arrays were calculated for this exact method - NEVER overwrite them
                if "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
                    arr_oil = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
                    arr_gas = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
                    if len(arr_oil) > 0 and len(arr_gas) > 0:
                        # Arrays are marked as depth-based for THIS method and have data - preserve them
                        return
            else:
                # Method changed! Clear the old method's flag so arrays can be recalculated
                # Don't return here - let the function continue to recalculate for new method
                st.session_state["_grv_arrays_depth_based"] = False
                st.session_state["_grv_arrays_method"] = ""
        
        # Fallback: if arrays exist and have valid length, preserve them for depth-based methods
        if "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
            arr_oil = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
            arr_gas = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
            # For depth-based methods, ALWAYS preserve existing arrays if they have valid length
            # This is critical because depth-based arrays are calculated independently from GOC
            # and should NEVER be recalculated using f_oil split or sGRV_m3_final
            if len(arr_oil) > 0 and len(arr_gas) > 0:
                # For depth-based methods, preserve arrays regardless of size match or fluid_type
                # The depth-based calculation (from GOC) is always more accurate than any split method
                # Only exception: if fluid_type changed to "Oil" or "Gas", we need to override
                if fluid_type == "Oil + Gas":
                    # For Oil+Gas, always preserve depth-based arrays - they're from GOC calculation
                    return
                elif fluid_type == "Oil" and np.any(arr_oil > 0):
                    # If fluid_type is Oil but we have valid oil array, preserve it
                    # (gas should be zero, but we'll let the override logic handle that)
                    pass  # Continue to override logic below
                elif fluid_type == "Gas" and np.any(arr_gas > 0):
                    # If fluid_type is Gas but we have valid gas array, preserve it
                    # (oil should be zero, but we'll let the override logic handle that)
                    pass  # Continue to override logic below
                else:
                    # Arrays exist and have data - preserve them for depth-based methods
                    return
    
    # Get split GRV from method-specific keys if available
    # CRITICAL: Only read from the CURRENT method's keys, not from old methods
    GRV_oil_m3 = None
    GRV_gas_m3 = None
    
    # Check if arrays are marked for the current method - if not, don't use them
    stored_method = st.session_state.get("_grv_arrays_method", "")
    method_matches = (stored_method == grv_option)
    
    if grv_option == "Direct GRV":
        if "direct_GRV_oil_m3" in st.session_state and method_matches:
            GRV_oil_m3 = np.asarray(st.session_state["direct_GRV_oil_m3"], dtype=float)
        if "direct_GRV_gas_m3" in st.session_state and method_matches:
            GRV_gas_m3 = np.asarray(st.session_state["direct_GRV_gas_m3"], dtype=float)
    elif grv_option == "Area × Thickness × GCF":
        if "atgcf_GRV_oil_m3" in st.session_state and method_matches:
            GRV_oil_m3 = np.asarray(st.session_state["atgcf_GRV_oil_m3"], dtype=float)
        if "atgcf_GRV_gas_m3" in st.session_state and method_matches:
            GRV_gas_m3 = np.asarray(st.session_state["atgcf_GRV_gas_m3"], dtype=float)
    else:
        # Depth-based methods: check if already set for THIS method
        # CRITICAL: For depth-based methods, the split GRV is calculated directly
        # and stored in sGRV_oil_m3 and sGRV_gas_m3. We should preserve these values
        # ONLY if they were calculated for the current method.
        if "sGRV_oil_m3" in st.session_state and method_matches:
            arr_oil = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
            if len(arr_oil) > 0 and len(arr_oil) == len(sGRV_m3_final):
                GRV_oil_m3 = arr_oil
        if "sGRV_gas_m3" in st.session_state and method_matches:
            arr_gas = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
            if len(arr_gas) > 0 and len(arr_gas) == len(sGRV_m3_final):
                GRV_gas_m3 = arr_gas
    
    # CRITICAL: Override based on CURRENT fluid_type (takes precedence)
    if fluid_type == "Oil":
        # All GRV is oil
        st.session_state["sGRV_oil_m3"] = sGRV_m3_final.copy()
        st.session_state["sGRV_gas_m3"] = np.zeros_like(sGRV_m3_final)
    elif fluid_type == "Gas":
        # All GRV is gas
        st.session_state["sGRV_oil_m3"] = np.zeros_like(sGRV_m3_final)
        st.session_state["sGRV_gas_m3"] = sGRV_m3_final.copy()
    else:  # Oil + Gas
        # CRITICAL: For depth-based methods, NEVER overwrite calculated values with f_oil split
        # Depth-based methods calculate GRV split directly from GOC, so we must preserve those values
        depth_based_methods = {
            "Depth-based: Top and Base res. + Contact(s)",
            "Depth-based: Top + Res. thickness + Contact(s)",
        }
        is_depth_based = grv_option in depth_based_methods
        
        # Use split GRV if available
        if GRV_oil_m3 is not None and GRV_gas_m3 is not None:
            # Ensure arrays match size
            if len(GRV_oil_m3) == len(sGRV_m3_final) and len(GRV_gas_m3) == len(sGRV_m3_final):
                st.session_state["sGRV_oil_m3"] = GRV_oil_m3.copy()
                st.session_state["sGRV_gas_m3"] = GRV_gas_m3.copy()
            elif not is_depth_based:
                # Size mismatch - recalculate using f_oil array (only for non-depth-based methods)
                f_oil_arr = np.full(len(sGRV_m3_final), 0.5)  # Default 0.5
                if grv_option == "Direct GRV" and "direct_f_oil" in st.session_state:
                    f_oil_arr_raw = st.session_state["direct_f_oil"]
                    f_oil_arr = np.asarray(f_oil_arr_raw, dtype=float)
                    if len(f_oil_arr) != len(sGRV_m3_final):
                        # Resize to match if needed
                        if len(f_oil_arr) > 0:
                            f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                        else:
                            f_oil_arr = np.full(len(sGRV_m3_final), 0.5)
                elif grv_option == "Area × Thickness × GCF" and "atgcf_f_oil" in st.session_state:
                    f_oil_arr_raw = st.session_state["atgcf_f_oil"]
                    f_oil_arr = np.asarray(f_oil_arr_raw, dtype=float)
                    if len(f_oil_arr) != len(sGRV_m3_final):
                        # Resize to match if needed
                        if len(f_oil_arr) > 0:
                            f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                        else:
                            f_oil_arr = np.full(len(sGRV_m3_final), 0.5)
                elif "f_oil" in st.session_state:
                    f_oil_val_raw = st.session_state["f_oil"]
                    if isinstance(f_oil_val_raw, np.ndarray):
                        f_oil_arr = np.asarray(f_oil_val_raw, dtype=float)
                        if len(f_oil_arr) != len(sGRV_m3_final):
                            f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                    else:
                        f_oil_arr = np.full(len(sGRV_m3_final), float(f_oil_val_raw))
                # Clip to [0, 1] and apply split
                f_oil_arr = np.clip(f_oil_arr, 0.0, 1.0)
                st.session_state["sGRV_oil_m3"] = sGRV_m3_final * f_oil_arr
                st.session_state["sGRV_gas_m3"] = sGRV_m3_final * (1.0 - f_oil_arr)
            # For depth-based methods with size mismatch, preserve existing values if they exist
            elif is_depth_based and "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
                # Keep existing depth-based calculated values even if size doesn't match
                # (this shouldn't happen, but preserve them if it does)
                # Ensure arrays are at least 1D before checking
                arr_oil_existing = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
                arr_gas_existing = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
                # Only preserve if they have valid length (even if size mismatch)
                if len(arr_oil_existing) > 0 and len(arr_gas_existing) > 0:
                    st.session_state["sGRV_oil_m3"] = arr_oil_existing
                    st.session_state["sGRV_gas_m3"] = arr_gas_existing
        elif not is_depth_based:
            # Split GRV not available - use f_oil array (only for non-depth-based methods)
            f_oil_arr = np.full(len(sGRV_m3_final), 0.5)  # Default 0.5
            if grv_option == "Direct GRV" and "direct_f_oil" in st.session_state:
                f_oil_arr_raw = st.session_state["direct_f_oil"]
                f_oil_arr = np.asarray(f_oil_arr_raw, dtype=float)
                if len(f_oil_arr) != len(sGRV_m3_final):
                    # Resize to match if needed
                    if len(f_oil_arr) > 0:
                        f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                    else:
                        f_oil_arr = np.full(len(sGRV_m3_final), 0.5)
            elif grv_option == "Area × Thickness × GCF" and "atgcf_f_oil" in st.session_state:
                f_oil_arr_raw = st.session_state["atgcf_f_oil"]
                f_oil_arr = np.asarray(f_oil_arr_raw, dtype=float)
                if len(f_oil_arr) != len(sGRV_m3_final):
                    # Resize to match if needed
                    if len(f_oil_arr) > 0:
                        f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                    else:
                        f_oil_arr = np.full(len(sGRV_m3_final), 0.5)
            elif "f_oil" in st.session_state:
                f_oil_val_raw = st.session_state["f_oil"]
                if isinstance(f_oil_val_raw, np.ndarray):
                    f_oil_arr = np.asarray(f_oil_val_raw, dtype=float)
                    if len(f_oil_arr) != len(sGRV_m3_final):
                        f_oil_arr = np.resize(f_oil_arr, len(sGRV_m3_final))
                else:
                    f_oil_arr = np.full(len(sGRV_m3_final), float(f_oil_val_raw))
            # Clip to [0, 1] and apply split
            f_oil_arr = np.clip(f_oil_arr, 0.0, 1.0)
            st.session_state["sGRV_oil_m3"] = sGRV_m3_final * f_oil_arr
            st.session_state["sGRV_gas_m3"] = sGRV_m3_final * (1.0 - f_oil_arr)
        # For depth-based methods, if split GRV not available, preserve existing values
        # (they should have been calculated earlier in render_grv)
        elif is_depth_based:
            # For depth-based methods, if split GRV arrays exist in session state, preserve them
            if "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
                arr_oil_existing = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
                arr_gas_existing = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
                # Preserve if they have valid length
                if len(arr_oil_existing) > 0 and len(arr_gas_existing) > 0:
                    # Ensure they match the size of sGRV_m3_final, or resize if needed
                    if len(arr_oil_existing) == len(sGRV_m3_final) and len(arr_gas_existing) == len(sGRV_m3_final):
                        st.session_state["sGRV_oil_m3"] = arr_oil_existing
                        st.session_state["sGRV_gas_m3"] = arr_gas_existing
                    else:
                        # Size mismatch - try to preserve by resizing or using as-is
                        # This preserves the calculated values even if there's a size issue
                        st.session_state["sGRV_oil_m3"] = arr_oil_existing
                        st.session_state["sGRV_gas_m3"] = arr_gas_existing


def _display_split_grv_results(prefix_key: str) -> None:
    """
    Display GRV results split by fluid type (oil, gas, total HC).
    
    Args:
        prefix_key: Prefix for session state keys (e.g., 'direct', 'atgcf')
    """
    fluid_type = st.session_state.get("fluid_type", "Oil")
    
    # Get split GRV arrays
    grv_oil_key = f"{prefix_key}_GRV_oil_m3"
    grv_gas_key = f"{prefix_key}_GRV_gas_m3"
    grv_total_key = f"{prefix_key}_GRV_total_m3"
    
    if grv_total_key not in st.session_state:
        return
    
    grv_total = st.session_state[grv_total_key]
    grv_oil = st.session_state.get(grv_oil_key, None)
    grv_gas = st.session_state.get(grv_gas_key, None)
    
    st.markdown("---")
    st.markdown("### GRV Results by Fluid Type")
    
    if fluid_type == "Oil":
        st.markdown("#### Oil GRV")
        st.plotly_chart(
            make_hist_cdf_figure(
                grv_total / 1e6,
                "Oil GRV Distribution (×10^6 m³)",
                "GRV (×10^6 m³)",
                "calculated"
            ),
            use_container_width=True,
        )
        st.dataframe(summary_table(grv_total / 1e6, decimals=2), use_container_width=True)
        st.caption("ℹ️ Oil only: entire GRV is oil.")
        
    elif fluid_type == "Gas":
        st.markdown("#### Gas GRV")
        st.plotly_chart(
            make_hist_cdf_figure(
                grv_total / 1e6,
                "Gas GRV Distribution (×10^6 m³)",
                "GRV (×10^6 m³)",
                "calculated"
            ),
            use_container_width=True,
        )
        st.dataframe(summary_table(grv_total / 1e6, decimals=2), use_container_width=True)
        st.caption("ℹ️ Gas only: entire GRV is gas.")
        
    else:  # Oil + Gas
        if grv_oil is not None and grv_gas is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Oil GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        grv_oil / 1e6,
                        "Oil GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(grv_oil / 1e6, decimals=2), use_container_width=True)
            
            with col2:
                st.markdown("#### Gas GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        grv_gas / 1e6,
                        "Gas GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(grv_gas / 1e6, decimals=2), use_container_width=True)
            
            with col3:
                st.markdown("#### Total HC GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        grv_total / 1e6,
                        "Total HC GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(grv_total / 1e6, decimals=2), use_container_width=True)
            
            # Combined summary table
            st.markdown("#### Combined GRV Summary")
            summary_data = {
                "Parameter": ["GRV Oil (×10⁶ m³)", "GRV Gas (×10⁶ m³)", "GRV Total HC (×10⁶ m³)"],
            }
            
            # Get statistics for each
            oil_stats = summarize_array(grv_oil / 1e6)
            gas_stats = summarize_array(grv_gas / 1e6)
            total_stats = summarize_array(grv_total / 1e6)
            
            summary_data["Mean"] = [
                f"{oil_stats.get('mean', 0.0):.2f}",
                f"{gas_stats.get('mean', 0.0):.2f}",
                f"{total_stats.get('mean', 0.0):.2f}",
            ]
            summary_data["P10"] = [
                f"{oil_stats.get('P10', 0.0):.2f}",
                f"{gas_stats.get('P10', 0.0):.2f}",
                f"{total_stats.get('P10', 0.0):.2f}",
            ]
            summary_data["P50"] = [
                f"{oil_stats.get('P50', 0.0):.2f}",
                f"{gas_stats.get('P50', 0.0):.2f}",
                f"{total_stats.get('P50', 0.0):.2f}",
            ]
            summary_data["P90"] = [
                f"{oil_stats.get('P90', 0.0):.2f}",
                f"{gas_stats.get('P90', 0.0):.2f}",
                f"{total_stats.get('P90', 0.0):.2f}",
            ]
            summary_data["Std"] = [
                f"{oil_stats.get('std_dev', 0.0):.2f}",
                f"{gas_stats.get('std_dev', 0.0):.2f}",
                f"{total_stats.get('std_dev', 0.0):.2f}",
            ]
            summary_data["Min"] = [
                f"{oil_stats.get('min', 0.0):.2f}",
                f"{gas_stats.get('min', 0.0):.2f}",
                f"{total_stats.get('min', 0.0):.2f}",
            ]
            summary_data["Max"] = [
                f"{oil_stats.get('max', 0.0):.2f}",
                f"{gas_stats.get('max', 0.0):.2f}",
                f"{total_stats.get('max', 0.0):.2f}",
            ]
            
            df_combined = pd.DataFrame(summary_data)
            st.dataframe(df_combined, use_container_width=True, hide_index=True)
            
            # Verify split
            f_oil_key = f"{prefix_key}_f_oil"
            if f_oil_key in st.session_state:
                f_oil_raw = st.session_state[f_oil_key]
                # Handle both array (from distribution) and scalar (backward compatibility)
                if isinstance(f_oil_raw, np.ndarray):
                    f_oil_mean = float(np.mean(f_oil_raw))
                    f_oil_min = float(np.min(f_oil_raw))
                    f_oil_max = float(np.max(f_oil_raw))
                    st.caption(
                        f"ℹ️ Split by oil fraction: fₒᵢₗ = {f_oil_mean:.2f} "
                        f"(mean: {f_oil_mean*100:.1f}% oil, {(1-f_oil_mean)*100:.1f}% gas; "
                        f"range: {f_oil_min*100:.1f}% - {f_oil_max*100:.1f}% oil)"
                    )
                else:
                    f_oil = float(f_oil_raw)
                    st.caption(f"ℹ️ Split by oil fraction: fₒᵢₗ = {f_oil:.2f} ({f_oil*100:.1f}% oil, {(1-f_oil)*100:.1f}% gas)")


def _render_column_heights() -> None:
    """
    Render Gas & Oil Column Heights section at the bottom of GRV page.
    Only shown when column heights are available (depth-based methods with Oil + Gas).
    """
    fluid_type = st.session_state.get("fluid_type", "Oil")
    
    # Only show for Oil + Gas and when column heights are available
    if fluid_type != "Oil + Gas":
        return
    
    if "sOil_Column_Height" not in st.session_state or "sGas_Column_Height" not in st.session_state:
        return
    
    gas_heights = st.session_state["sGas_Column_Height"]
    oil_heights = st.session_state["sOil_Column_Height"]
    
    # Only show if we have valid column heights
    if not (np.any(gas_heights > 0) or np.any(oil_heights > 0)):
        return
    
    st.markdown("---")
    st.markdown("### Gas & Oil Column Heights")
    st.caption("Calculated column heights based on GOC definition and depth-area integration.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Gas Column Height")
        if np.any(gas_heights > 0):
            st.plotly_chart(
                make_hist_cdf_figure(
                    gas_heights,
                    "Gas Column Height Distribution",
                    "Gas Column Height (m)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(gas_heights, decimals=1), use_container_width=True)
        else:
            st.info("No gas column (GOC at or below HCWC)")
    
    with col2:
        st.markdown("#### Oil Column Height")
        if np.any(oil_heights > 0):
            st.plotly_chart(
                make_hist_cdf_figure(
                    oil_heights,
                    "Oil Column Height Distribution",
                    "Oil Column Height (m)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(oil_heights, decimals=1), use_container_width=True)
        else:
            st.info("No oil column (GOC at or above top structure)")
    
    # Combined overlay plot
    st.markdown("#### Column Heights Overlay")
    fig = go.Figure()
    bins = 50
    if np.any(gas_heights > 0):
        fig.add_histogram(
            x=gas_heights[gas_heights > 0],
            nbinsx=bins,
            name="Gas Column",
            marker=dict(color=color_for("Gas_Column_Height")),
            opacity=0.6
        )
    if np.any(oil_heights > 0):
        fig.add_histogram(
            x=oil_heights[oil_heights > 0],
            nbinsx=bins,
            name="Oil Column",
            marker=dict(color=color_for("Oil_Column_Height")),
            opacity=0.6
        )
    fig.update_layout(
        barmode="overlay",
        height=400,
        margin=dict(l=40, r=30, t=30, b=60),
        xaxis_title="Column Height (m)",
        yaxis_title="Frequency",
        title="Gas & Oil Column Height Distributions"
    )
    st.plotly_chart(fig, use_container_width=True)


def _apply_fluid_type_split(prefix_key: str, total_grv: np.ndarray, num_sims: int) -> None:
    """
    Apply fluid type split to total GRV.
    
    Args:
        prefix_key: Prefix for session state keys (e.g., 'direct', 'atgcf')
        total_grv: Array of total GRV values
        num_sims: Number of simulations
    """
    fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
    total_arr = np.asarray(total_grv, dtype=float)
    
    if fluid_type == "Oil":
        st.session_state[f"{prefix_key}_GRV_oil_m3"] = total_arr
        st.session_state[f"{prefix_key}_GRV_gas_m3"] = np.zeros_like(total_arr)
    elif fluid_type == "Gas":
        st.session_state[f"{prefix_key}_GRV_oil_m3"] = np.zeros_like(total_arr)
        st.session_state[f"{prefix_key}_GRV_gas_m3"] = total_arr
    else:
        # Oil + Gas: split by fraction f_oil (now using distribution input)
        st.markdown("#### Oil + Gas Split (by Fraction)")
        f_oil_key = f"{prefix_key}_f_oil"
        
        # Use render_param to get distribution input (returns array of num_sims values)
        # Default: PERT distribution with 0.5, 0.55, 0.65
        f_oil_defaults = {
            "dist": "PERT",
            "min": 0.5,
            "mode": 0.55,
            "max": 0.65
        }
        
        # Get existing distribution if available, otherwise use defaults
        existing_dist = None
        if f"{f_oil_key}_dist" in st.session_state:
            dist_type = st.session_state[f"{f_oil_key}_dist"]
            existing_dist = {
                "type": dist_type,
                "min": st.session_state.get(f"{f_oil_key}_pert_min", f_oil_defaults["min"]),
                "mode": st.session_state.get(f"{f_oil_key}_pert_mode", f_oil_defaults["mode"]),
                "max": st.session_state.get(f"{f_oil_key}_pert_max", f_oil_defaults["max"]),
            }
            # Add other distribution parameters if needed
            if dist_type == "Triangular":
                existing_dist["min"] = st.session_state.get(f"{f_oil_key}_tri_min", f_oil_defaults["min"])
                existing_dist["mode"] = st.session_state.get(f"{f_oil_key}_tri_mode", f_oil_defaults["mode"])
                existing_dist["max"] = st.session_state.get(f"{f_oil_key}_tri_max", f_oil_defaults["max"])
            elif dist_type == "Uniform":
                existing_dist["min"] = st.session_state.get(f"{f_oil_key}_uni_min", f_oil_defaults["min"])
                existing_dist["max"] = st.session_state.get(f"{f_oil_key}_uni_max", f_oil_defaults["max"])
            elif dist_type == "Constant":
                existing_dist["value"] = st.session_state.get(f"{f_oil_key}_const", f_oil_defaults["mode"])
        
        # Use render_param to get f_oil as an array (one value per trial)
        f_oil_arr = render_param(
            f_oil_key,
            "Oil fraction of total hydrocarbon GRV (fₒᵢₗ)",
            "",
            f_oil_defaults["dist"] if existing_dist is None else existing_dist["type"],
            existing_dist if existing_dist is not None else f_oil_defaults,
            num_sims,
            stats_decimals=3,
            help_text="Fraction of total GRV that is oil (remainder is gas). Supports all distribution types. Values are automatically clipped to [0, 1].",
            param_name=f_oil_key,
        )
        
        # Clip f_oil to [0, 1] range
        f_oil_arr = np.clip(f_oil_arr, 0.0, 1.0)
        
        # Store f_oil array in session state for use in calculations
        st.session_state[f_oil_key] = f_oil_arr
        
        # Split GRV using array multiplication (element-wise)
        oil = total_arr * f_oil_arr
        gas = total_arr * (1.0 - f_oil_arr)
        st.session_state[f"{prefix_key}_GRV_oil_m3"] = oil
        st.session_state[f"{prefix_key}_GRV_gas_m3"] = gas
    
    # Always set total for downstream consistency
    st.session_state[f"{prefix_key}_GRV_total_m3"] = total_arr


def _initialize_grv_input_defaults() -> None:
    """
    Initialize ALL GRV input defaults in session_state at the start.
    CRITICAL: Only set defaults if key doesn't exist - NEVER overwrite existing values.
    This ensures all values persist across page navigation and GRV method changes.
    
    IMPORTANT: This function is called every time render_grv() runs.
    It will NOT overwrite existing values, so user changes are preserved.
    """
    # Don't use a flag - always check and initialize missing values
    # This ensures values exist even if session_state was partially cleared
    
    # Geometry factor inputs (GCF calculator)
    defaults = {
        "gcf_method": "Direct method",
        "gcf_reservoir_thickness": 100.0,
        "gcf_structural_relief": 150.0,
        "gcf_dip_angle": 0.0,
        "gcf_lw_ratio": 2,
        # Oil fraction inputs (for all GRV methods)
        # Note: These are now arrays from render_param (PERT distribution by default)
        # Defaults are handled by render_param with PERT(0.5, 0.55, 0.65)
        # Keeping scalar defaults for backward compatibility, but they won't be used
        "direct_f_oil": 0.55,  # Mean of PERT(0.5, 0.55, 0.65)
        "atgcf_f_oil": 0.55,   # Mean of PERT(0.5, 0.55, 0.65)
        # Note: da_oil_frac and da_oil_frac_D are now arrays from render_param (Stretched Beta distribution)
        # They are no longer scalars, so defaults are handled by render_param
        # Depth-based method inputs
        "da_step_size": 10.0,
        "da_extrap": True,
        "da_area_uncert_enabled": False,
        "da_area_dist": "PERT",
        # Depth-based method D inputs (Top + Res. thickness)
        "da_D_step_size": 10.0,
        "da_D_extrap": True,
        "da_D_area_uncert_enabled": False,
        "da_D_area_dist": "PERT",
        # NOTE: GOC mode inputs (da_goc_mode, da_goc_mode_D) are NOT initialized here
        # They are initialized in the GOC Definition sections to avoid overwriting user selection
    }
    
    # Only set defaults if they don't exist - preserve user changes
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Shape type needs special handling (tuple)
    shape_options_default = [
        (1, "Dome, Cone, or Pyramid"),
        (2, "Anticline, Prism, or Cylinder"),
        (3, "Flat-top Dome"),
        (4, "Flat-top Anticline"),
        (5, "Block or Vertical Cylinder"),
    ]
    if "gcf_shape_type" not in st.session_state:
        st.session_state["gcf_shape_type"] = shape_options_default[0]


def render_grv(num_sims: int, defaults: Dict[str, Any], show_inline_tips: bool) -> None:
    """
    Render the Gross Rock Volume (GRV) / geometry inputs section.
    
    CRITICAL: When GRV method changes, we need to clear old method's arrays and recalculate.

    Parameters
    ----------
    num_sims : int
        Number of Monte Carlo simulations.
    defaults : Dict[str, Any]
        Default distribution definitions (Area, GCF, thickness, etc.).
    show_inline_tips : bool
        Whether to display inline informational tips.
    """
    # CRITICAL: Initialize ALL input defaults FIRST, before any widgets are created
    # This ensures values persist across page navigation and method changes
    _initialize_grv_input_defaults()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <h2 style='color:{PALETTE["primary"]};border-left:4px solid {PALETTE["accent"]};
        padding-left:12px;background:linear-gradient(90deg, rgba(197, 78, 82, 0.1), transparent);
        padding:2px 0 2px 12px;border-radius:4px;margin:0.2em 0 0.1em 0;'>
            Gross Rock Volume (GRV)
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # Get default option from session state or use default
    grv_options = [
        "Direct GRV",
        "Area × Thickness × GCF",
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    ]
    default_index = 1  # Default to "Area × Thickness × GCF"
    if "grv_option" in st.session_state:
        try:
            default_index = grv_options.index(st.session_state["grv_option"])
        except ValueError:
            default_index = 1
    
    grv_option = st.radio(
        "Choose GRV input method",
        grv_options,
        index=default_index,
        key="grv_option_radio",
        horizontal=True,
        help=HELP["grv_method"],
    )
    
    # CRITICAL: Check if GRV method has changed AFTER the radio button sets it
    # If method changed, clear old method's arrays so new method can recalculate
    stored_method = st.session_state.get("_grv_arrays_method", "")
    if stored_method and stored_method != grv_option:
        # Method changed! Clear the old method's arrays and flags
        # This ensures we recalculate for the new method instead of using old values
        if "_grv_arrays_depth_based" in st.session_state:
            del st.session_state["_grv_arrays_depth_based"]
        if "_grv_arrays_method" in st.session_state:
            del st.session_state["_grv_arrays_method"]
        # Clear the split arrays so they get recalculated for the new method
        if "sGRV_oil_m3" in st.session_state:
            del st.session_state["sGRV_oil_m3"]
        if "sGRV_gas_m3" in st.session_state:
            del st.session_state["sGRV_gas_m3"]
        # CRITICAL: Also clear method-specific arrays to prevent _ensure_final_grv_arrays
        # from reading old method's arrays and overwriting the new method's calculation
        if "direct_GRV_oil_m3" in st.session_state:
            del st.session_state["direct_GRV_oil_m3"]
        if "direct_GRV_gas_m3" in st.session_state:
            del st.session_state["direct_GRV_gas_m3"]
        if "atgcf_GRV_oil_m3" in st.session_state:
            del st.session_state["atgcf_GRV_oil_m3"]
        if "atgcf_GRV_gas_m3" in st.session_state:
            del st.session_state["atgcf_GRV_gas_m3"]
    
    # Clear results cache if GRV option changed
    prev_grv_option = st.session_state.get("grv_option_prev")
    if prev_grv_option is not None and prev_grv_option != grv_option:
        if "results_cache" in st.session_state:
            del st.session_state["results_cache"]
        if "trial_data" in st.session_state:
            del st.session_state["trial_data"]
        if "df_results" in st.session_state:
            del st.session_state["df_results"]
    st.session_state["grv_option_prev"] = grv_option
    st.session_state["grv_option"] = grv_option

    with st.expander("ℹ️ How to configure GRV"):
        st.markdown(
            """
            - **Direct & Area×Thickness×GCF**: Choose oil/gas split explicitly.
            - **Depth-based**: Provide Spill Point and Effective HC depth distributions. Optionally enable **Gas–Oil Contact (GOC)** to split gas cap (top→GOC) and oil leg (GOC→HC).
            - The star (★) on depth plots shows the **mean Spill Point depth**.
            """
        )

    if show_inline_tips:
        st.info("💡 **Tip**: " + HELP["grv_method"])

    sGRV_m3 = None
    sGCF = None
    sA = None
    sh = None
    grv_mp_samples = None

    if grv_option == "Direct GRV":
        grv_min = (
            defaults["A"]["min"] * 1_000_000.0 * defaults["h"]["min"] * defaults["GCF"]["min"]
        )
        grv_mode = (
            defaults["A"]["mode"] * 1_000_000.0 * defaults["h"]["mode"] * defaults["GCF"]["mode"]
        )
        grv_max = (
            defaults["A"]["max"] * 1_000_000.0 * defaults["h"]["max"] * defaults["GCF"]["max"]
        )
        defaults_grv = {"dist": "PERT", "min": grv_min, "mode": grv_mode, "max": grv_max}

        sGRV_m3 = render_param(
            "GRV",
            "Gross Rock Volume (GRV)",
            "m³",
            defaults_grv["dist"],
            defaults_grv,
            num_sims,
            plot_unit_label="×10^6 m³",
            stats_decimals=1,
            display_scale=1e-6,
        )
        
        # Store total GRV for split logic
        st.session_state["direct_GRV_total_m3"] = sGRV_m3
        
        # Apply fluid type split
        _apply_fluid_type_split("direct", sGRV_m3, num_sims)
        
        # CRITICAL: Copy split GRV to sGRV arrays for use in compute/results
        # Note: sGRV_m3_final will be set later after multiplier is applied (if any)
        # CRITICAL: Mark arrays as calculated for Direct GRV method
        if "_grv_arrays_depth_based" in st.session_state:
            del st.session_state["_grv_arrays_depth_based"]
        st.session_state["_grv_arrays_method"] = "Direct GRV"
        
        if "direct_GRV_oil_m3" in st.session_state:
            st.session_state["sGRV_oil_m3"] = st.session_state["direct_GRV_oil_m3"].copy()
        if "direct_GRV_gas_m3" in st.session_state:
            st.session_state["sGRV_gas_m3"] = st.session_state["direct_GRV_gas_m3"].copy()
        
        # Display split GRV results
        _display_split_grv_results("direct")

    elif grv_option == "Area × Thickness × GCF":
        st.caption("Provide Area, GCF, and h; GRV is calculated and shown below.")
        sA = render_param(
            "A", "Area A", "km²", defaults["A"]["dist"], defaults["A"], num_sims, stats_decimals=3
        )

        # All defaults are already initialized by _initialize_grv_input_defaults()
        # Use a unique prefix to avoid conflicts with other parts of the app
        gcf_prefix = "gcf_"
        gcf_method_options = ["Direct method", "Geometric Correction Factor Calculator"]
        
        # Get index for initial display - use .get() to avoid KeyError
        stored_gcf_method = st.session_state.get(f"{gcf_prefix}method", "Direct method")
        try:
            gcf_method_index = gcf_method_options.index(stored_gcf_method)
        except ValueError:
            gcf_method_index = 0
            st.session_state[f"{gcf_prefix}method"] = "Direct method"
        
        # When using key=, Streamlit manages the value automatically
        gcf_method = st.radio(
            "GCF distribution method",
            gcf_method_options,
            index=gcf_method_index,
            horizontal=True,
            key=f"{gcf_prefix}method",
            help="Choose how to define the GCF distribution",
        )

        if gcf_method == "Direct method":
            sGCF = render_param(
                "GCF",
                "Geometry Correction Factor GCF",
                "fraction",
                defaults["GCF"]["dist"],
                defaults["GCF"],
                num_sims,
                stats_decimals=3,
            )
        else:
            st.markdown("### Geometric Correction Factor Calculator")
            st.markdown(
                "Calculate GCF based on reservoir geometry using the Gehman (1970) methodology."
            )
            with st.expander("Methodology and Assumptions", expanded=False):
                st.markdown(
                    """
                    ### Geometric Correction Factor Methodology

                    This calculator is based on the work of **Gehman, H.N. (1970)** and provides geometric correction factors for different trap geometries.

                    **Reference:** Gehman, H.N. (1970). Graphs to Derive Geometric Correction Factor: Exxon Training Materials (unpublished), Houston.

                    **Data Source:** The lookup table data has been digitized from the original graphs presented in the Gehman (1970) training materials.

                    #### Key Concepts:

                    1. **Geometric Shape Types:**
                       - **Type 1**: Dome, Cone, or Pyramid
                       - **Type 2**: Anticline, Prism, or Cylinder
                       - **Type 3**: Flat-top Dome
                       - **Type 4**: Flat-top Anticline
                       - **Type 5**: Block or Vertical Cylinder

                    2. **Length/Width Ratios:**
                       - For domes and flat-top domes: L/W = 1
                       - For anticlines: L/W = 2, 5, or 10
                       - For blocks: L/W = W (width)

                    3. **Reservoir Thickness/Closure Ratio:**
                       - Ratio of true reservoir thickness to structural relief
                       - True thickness accounts for dip angle correction
                       - As this ratio increases, GCF decreases (inverse relationship)

                    #### Step-by-Step Process:
                    1. Enter reservoir thickness and structural relief
                    2. Select geometric shape type and length/width ratio
                    3. Calculate Reservoir Thickness/Closure ratio
                    4. Apply dip angle correction if needed
                    5. Use lookup table to find corresponding GCF value
                    6. Apply uncertainty multiplier if desired
                    """
                )

            col1, col2 = st.columns(2)
            with col1:
                # CRITICAL: Widgets with key= automatically save to session_state
                # We provide value= to show the current persisted value
                # The key must match the session_state key for automatic persistence
                reservoir_thickness = st.number_input(
                    "Reservoir thickness (m)",
                    min_value=0.1,
                    value=st.session_state[f"{gcf_prefix}reservoir_thickness"],
                    step=0.1,
                    key=f"{gcf_prefix}reservoir_thickness",
                    help="True reservoir thickness in meters",
                )
                structural_relief = st.number_input(
                    "Structural relief (m)",
                    min_value=0.1,
                    value=st.session_state[f"{gcf_prefix}structural_relief"],
                    step=0.1,
                    key=f"{gcf_prefix}structural_relief",
                    help="Height of closure (spill point to apex) in meters",
                )
                dip_angle = st.number_input(
                    "Dip angle (degrees)",
                    min_value=0.0,
                    max_value=89.9,
                    value=st.session_state[f"{gcf_prefix}dip_angle"],
                    step=0.1,
                    key=f"{gcf_prefix}dip_angle",
                    help="Reservoir dip angle in degrees (0° = horizontal, 90° = vertical)",
                )

            with col2:
                if reservoir_thickness <= 0 or structural_relief <= 0:
                    st.error("❌ Reservoir thickness and structural relief must be positive values")
                    st.stop()

                if dip_angle >= 90:
                    st.error("❌ Dip angle must be less than 90 degrees")
                    st.stop()

                dip_radians = math.radians(dip_angle)
                true_thickness = reservoir_thickness / math.cos(dip_radians)
                res_tk_closure_ratio = true_thickness / structural_relief

                if res_tk_closure_ratio > 1.0:
                    st.warning(
                        f"⚠️ **Warning**: Reservoir thickness/closure ratio ({res_tk_closure_ratio:.3f}) exceeds 1.0. "
                        "Verify your input parameters."
                    )

                st.metric("True thickness (m)", f"{true_thickness:.1f}")
                st.metric("Thickness/Closure ratio", f"{res_tk_closure_ratio:.3f}")

            # Values are already initialized above, outside the conditional block
            shape_options = [
                (1, "Dome, Cone, or Pyramid"),
                (2, "Anticline, Prism, or Cylinder"),
                (3, "Flat-top Dome"),
                (4, "Flat-top Anticline"),
                (5, "Block or Vertical Cylinder"),
            ]
            
            # Get stored shape_type - Streamlit stores the tuple directly
            stored_shape_type = st.session_state[f"{gcf_prefix}shape_type"]
            # Find the index of the stored value
            try:
                shape_index = shape_options.index(stored_shape_type)
            except (ValueError, TypeError):
                # If stored value doesn't match (e.g., different tuple instance), try to match by first element
                if isinstance(stored_shape_type, tuple) and len(stored_shape_type) == 2:
                    # Try to find by first element (shape type number)
                    shape_index = next((i for i, opt in enumerate(shape_options) if opt[0] == stored_shape_type[0]), 0)
                else:
                    shape_index = 0
                st.session_state[f"{gcf_prefix}shape_type"] = shape_options[shape_index]
            
            shape_type = st.selectbox(
                "Geometric shape type",
                shape_options,
                index=shape_index,
                format_func=lambda x: x[1],
                key=f"{gcf_prefix}shape_type",
                help="Select the geometric shape of the reservoir",
            )

            if shape_type[0] in [1, 3, 5]:
                lw_ratio = 1
                # Store in session_state for consistency
                st.session_state[f"{gcf_prefix}lw_ratio"] = 1
                st.info(f"Length/Width ratio set to 1 for {shape_type[1]}")
            else:
                # Value is already initialized above, outside the conditional block
                lw_options = [2, 5, 10]
                stored_lw_ratio = st.session_state[f"{gcf_prefix}lw_ratio"]
                try:
                    lw_index = lw_options.index(stored_lw_ratio)
                except ValueError:
                    lw_index = 0
                    st.session_state[f"{gcf_prefix}lw_ratio"] = 2
                
                # When using key=, Streamlit manages the value automatically
                # We provide index= only for initial display if value doesn't match
                lw_ratio = st.selectbox(
                    "Length/Width ratio",
                    lw_options,
                    index=lw_index,
                    key=f"{gcf_prefix}lw_ratio",
                    help="Select the length to width ratio for the anticline",
                )

            gcf_calculated = interpolate_gcf(shape_type[0], lw_ratio, res_tk_closure_ratio)
            st.markdown(
                f"""
                <div class='card-container'>
                    <h3 style='color:{PALETTE["secondary"]};margin-top:0;'>GCF Calculation Results</h3>
                """,
                unsafe_allow_html=True,
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True thickness (m)", f"{true_thickness:.1f}")
            with col2:
                st.metric("Reservoir Thickness/Closure Ratio", f"{res_tk_closure_ratio:.3f}")
            with col3:
                st.metric("Calculated GCF", f"{gcf_calculated:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)

            res_tk_closure_ratios, lookup_table = get_gcf_lookup_table()
            fig = go.Figure()
            curve_configs = {
                (1, 1): {"color": "blue", "width": 3, "label": "Dome, Cone, Pyramid (L/W=1)"},
                (2, 2): {"color": "red", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=2)"},
                (2, 5): {"color": "orange", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=5)"},
                (2, 10): {"color": "purple", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=10)"},
                (3, 1): {"color": "green", "width": 2, "label": "Flat-top Dome (L/W=1)"},
                (4, 2): {"color": "brown", "width": 2, "label": "Flat-top Anticline (L/W=2)"},
                (4, 5): {"color": "pink", "width": 2, "label": "Flat-top Anticline (L/W=5)"},
                (4, 10): {"color": "gray", "width": 2, "label": "Flat-top Anticline (L/W=10)"},
                (5, 1): {"color": "black", "width": 2, "label": "Block, Vertical Cylinder (L/W=1)"},
            }

            for (shape_type_plot, lw_ratio_plot), gcf_curve in lookup_table.items():
                config = curve_configs.get(
                    (shape_type_plot, lw_ratio_plot),
                    {"color": "blue", "width": 1, "label": f"Shape {shape_type_plot} (L/W={lw_ratio_plot})"},
                )
                fig.add_trace(
                    go.Scatter(
                        x=res_tk_closure_ratios,
                        y=gcf_curve,
                        mode="lines",
                        name=config["label"],
                        line=dict(width=config["width"], color=config["color"]),
                        showlegend=True,
                        hovertemplate="<b>%{fullData.name}</b><br>"
                        "Reservoir Thickness/Closure Ratio: %{x:.3f}<br>"
                        "Geometric Correction Factor: %{y:.3f}<extra></extra>",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[res_tk_closure_ratio],
                    y=[gcf_calculated],
                    mode="markers",
                    name="Current Point (Red Star)",
                    marker=dict(size=24, color="red", symbol="star"),
                    showlegend=True,
                    hovertemplate="<b>Current Point</b><br>"
                    "Reservoir Thickness/Closure Ratio: %{x:.3f}<br>"
                    "Geometric Correction Factor: %{y:.3f}<extra></extra>",
                )
            )

            fig.update_layout(
                title=dict(
                    text="Geometric Correction Factor vs Reservoir Thickness/Closure Ratio",
                    y=0.98,
                ),
                xaxis_title="Reservoir Thickness/Closure Ratio",
                yaxis_title="Geometric Correction Factor",
                width=800,
                height=600,
                legend=dict(
                    yanchor="top",
                    y=-0.15,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                ),
                xaxis=dict(range=[0, 1], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=False)

            st.markdown(
                f"""
                <div class='card-container'>
                    <h3 style='color:{PALETTE["secondary"]};margin-top:0;'>GCF Uncertainty Multiplier (GCF_MP)</h3>
                """,
                unsafe_allow_html=True,
            )

            gcf_mp_enabled = st.checkbox(
                "Apply GCF uncertainty multiplier",
                value=False,
                help="Enable to apply uncertainty multiplier to calculated GCF value",
            )

            if gcf_mp_enabled:
                gcf_mp_type = st.radio(
                    "GCF_MP definition",
                    ["Constant value", "Probability distribution"],
                    horizontal=True,
                    help="Choose how to define the GCF uncertainty multiplier",
                )

                if gcf_mp_type == "Constant value":
                    gcf_mp_constant = st.number_input(
                        "GCF_MP constant value",
                        value=1.0,
                        min_value=0.1,
                        max_value=10.0,
                        step=0.01,
                        help="Constant multiplier for calculated GCF value",
                    )
                    gcf_mp_samples = np.full(num_sims, gcf_mp_constant)
                else:
                    gcf_mp_samples = render_param(
                        "GCF_MP",
                        "GCF Uncertainty Multiplier",
                        "multiplier",
                        "PERT",
                        {"min": 0.9, "mode": 1.0, "max": 1.1},
                        num_sims,
                        plot_unit_label="multiplier",
                        stats_decimals=3,
                    )
            else:
                gcf_mp_samples = np.full(num_sims, 1.0)
                st.info("GCF_MP disabled - using calculated GCF value as is (multiplier = 1.0)", icon="ℹ️")

            sGCF = np.full(num_sims, gcf_calculated) * gcf_mp_samples
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class='card-container'>
                    <h3 style='color:{PALETTE["secondary"]};margin-top:0;'>Final GCF for Calculations</h3>
                """,
                unsafe_allow_html=True,
            )
            gcf_calc_value = float(gcf_calculated)
            gcf_mp_mean = float(np.mean(gcf_mp_samples))
            gcf_final_mean = float(np.mean(sGCF))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calculated GCF", f"{gcf_calc_value:.3f}")
            with col2:
                st.metric("GCF_MP (mean)", f"{gcf_mp_mean:.3f}")
            with col3:
                st.metric("Final GCF (mean)", f"{gcf_final_mean:.3f}")

            st.plotly_chart(
                make_hist_cdf_figure(
                    sGCF, "Final GCF distribution (after GCF_MP)", "GCF", "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(sGCF, decimals=3), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class='card-container'>
                    <h3 style='color:{PALETTE["secondary"]};margin-top:0;'>GCF Sensitivity Analysis</h3>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                **How the Sensitivity Analysis Works:**

                The sensitivity analysis shows how the Geometric Correction Factor (GCF) changes when individual input
                parameters are varied while keeping all other parameters constant. This helps identify which parameters
                have the greatest impact on the final GCF value.
                """
            )

            thickness_range = np.linspace(reservoir_thickness * 0.5, reservoir_thickness * 1.5, 50)
            gcf_values_thickness = []
            for t in thickness_range:
                true_t = t / math.cos(dip_radians)
                ratio_t = true_t / structural_relief
                gcf_values_thickness.append(interpolate_gcf(shape_type[0], lw_ratio, ratio_t))

            relief_range = np.linspace(structural_relief * 0.5, structural_relief * 1.5, 50)
            gcf_values_relief = []
            for r in relief_range:
                ratio_r = true_thickness / r
                gcf_values_relief.append(interpolate_gcf(shape_type[0], lw_ratio, ratio_r))

            dip_range = np.linspace(0, 45, 50)
            gcf_values_dip = []
            for d in dip_range:
                dip_rad = math.radians(d)
                true_t_dip = reservoir_thickness / math.cos(dip_rad)
                ratio_dip = true_t_dip / structural_relief
                gcf_values_dip.append(interpolate_gcf(shape_type[0], lw_ratio, ratio_dip))

            fig_sens = make_subplots(
                rows=1, cols=3, subplot_titles=("Reservoir Thickness", "Structural Relief", "Dip Angle")
            )
            valid_thickness_mask = ~np.isnan(gcf_values_thickness)
            valid_relief_mask = ~np.isnan(gcf_values_relief)
            valid_dip_mask = ~np.isnan(gcf_values_dip)

            if np.any(valid_thickness_mask):
                fig_sens.add_trace(
                    go.Scatter(
                        x=thickness_range[valid_thickness_mask],
                        y=np.array(gcf_values_thickness)[valid_thickness_mask],
                        mode="lines",
                        name="Thickness",
                        line=dict(color=color_for("Effective_HC_depth"), width=2),
                    ),
                    row=1,
                    col=1,
                )
                fig_sens.add_trace(
                    go.Scatter(
                        x=[reservoir_thickness],
                        y=[gcf_calculated],
                        mode="markers",
                        name="Current",
                        marker=dict(symbol="star", size=12, color="red"),
                    ),
                    row=1,
                    col=1,
                )
            else:
                st.warning("⚠️ No valid thickness sensitivity data to plot")

            if np.any(valid_relief_mask):
                fig_sens.add_trace(
                    go.Scatter(
                        x=relief_range[valid_relief_mask],
                        y=np.array(gcf_values_relief)[valid_relief_mask],
                        mode="lines",
                        name="Relief",
                        line=dict(color="green", width=2),
                    ),
                    row=1,
                    col=2,
                )
                fig_sens.add_trace(
                    go.Scatter(
                        x=[structural_relief],
                        y=[gcf_calculated],
                        mode="markers",
                        name="Current",
                        marker=dict(symbol="star", size=12, color="red"),
                    ),
                    row=1,
                    col=2,
                )
            else:
                st.warning("⚠️ No valid relief sensitivity data to plot")

            if np.any(valid_dip_mask):
                fig_sens.add_trace(
                    go.Scatter(
                        x=dip_range[valid_dip_mask],
                        y=np.array(gcf_values_dip)[valid_dip_mask],
                        mode="lines",
                        name="Dip Angle",
                        line=dict(color=color_for("Effective_HC_depth"), width=2),
                    ),
                    row=1,
                    col=3,
                )
                fig_sens.add_trace(
                    go.Scatter(
                        x=[dip_angle],
                        y=[gcf_calculated],
                        mode="markers",
                        name="Current",
                        marker=dict(symbol="star", size=12, color="red"),
                    ),
                    row=1,
                    col=3,
                )
            else:
                st.warning("⚠️ No valid dip angle sensitivity data to plot")

            fig_sens.update_layout(
                height=450,
                showlegend=False,
                title=dict(text="GCF Sensitivity Analysis", x=0.5, xanchor="center", font=dict(size=16)),
                margin=dict(t=80, b=60, l=60, r=60),
            )
            fig_sens.update_xaxes(
                title_text="Reservoir Thickness (m)", row=1, col=1, gridcolor="lightgray", gridwidth=0.5
            )
            fig_sens.update_xaxes(
                title_text="Structural Relief (m)", row=1, col=2, gridcolor="lightgray", gridwidth=0.5
            )
            fig_sens.update_xaxes(
                title_text="Dip Angle (degrees)", row=1, col=3, gridcolor="lightgray", gridwidth=0.5
            )
            fig_sens.update_yaxes(
                title_text="Geometric Correction Factor", row=1, col=1, gridcolor="lightgray", gridwidth=0.5
            )
            fig_sens.update_yaxes(
                title_text="Geometric Correction Factor", row=1, col=2, gridcolor="lightgray", gridwidth=0.5
            )
            fig_sens.update_yaxes(
                title_text="Geometric Correction Factor", row=1, col=3, gridcolor="lightgray", gridwidth=0.5
            )

            st.markdown("#### Sensitivity Summary")
            col1, col2, col3 = st.columns(3)
            if np.any(valid_thickness_mask):
                thickness_gcf_range = np.nanmax(gcf_values_thickness) - np.nanmin(gcf_values_thickness)
                col1.metric("Thickness Impact", f"{thickness_gcf_range:.3f}", delta=f"±{thickness_gcf_range/2:.3f}")
            else:
                col1.metric("Thickness Impact", "N/A")
            if np.any(valid_relief_mask):
                relief_gcf_range = np.nanmax(gcf_values_relief) - np.nanmin(gcf_values_relief)
                col2.metric("Relief Impact", f"{relief_gcf_range:.3f}", delta=f"±{relief_gcf_range/2:.3f}")
            else:
                col2.metric("Relief Impact", "N/A")
            if np.any(valid_dip_mask):
                dip_gcf_range = np.nanmax(gcf_values_dip) - np.nanmin(gcf_values_dip)
                col3.metric("Dip Angle Impact", f"{dip_gcf_range:.3f}", delta=f"±{dip_gcf_range/2:.3f}")
            else:
                col3.metric("Dip Angle Impact", "N/A")
            st.plotly_chart(fig_sens, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        sh = render_param(
            "h",
            "Reservoir thickness h (average isochore thickness across the prospect)",
            "m",
            defaults["h"]["dist"],
            defaults["h"],
            num_sims,
            stats_decimals=2,
        )

        sGRV_m3 = (sA * 1_000_000.0) * sh * sGCF
        st.plotly_chart(
            make_hist_cdf_figure(
                sGRV_m3 / 1e6, "Calculated GRV distribution", "GRV (×10^6 m³)", "calculated"
            ),
            use_container_width=True,
        )
        st.dataframe(summary_table(sGRV_m3 / 1e6, decimals=2), use_container_width=True)
        
        # Store total GRV for split logic
        st.session_state["atgcf_GRV_total_m3"] = sGRV_m3
        
        # Apply fluid type split
        _apply_fluid_type_split("atgcf", sGRV_m3, num_sims)
        
        # CRITICAL: Copy split GRV to sGRV arrays for use in compute/results
        # Note: sGRV_m3_final will be set later after multiplier is applied (if any)
        # CRITICAL: Mark arrays as calculated for Area × Thickness × GCF method
        if "_grv_arrays_depth_based" in st.session_state:
            del st.session_state["_grv_arrays_depth_based"]
        st.session_state["_grv_arrays_method"] = "Area × Thickness × GCF"
        
        if "atgcf_GRV_oil_m3" in st.session_state:
            st.session_state["sGRV_oil_m3"] = st.session_state["atgcf_GRV_oil_m3"].copy()
        if "atgcf_GRV_gas_m3" in st.session_state:
            st.session_state["sGRV_gas_m3"] = st.session_state["atgcf_GRV_gas_m3"].copy()
        
        # Display split GRV results
        _display_split_grv_results("atgcf")

    elif grv_option == "Depth-based: Top and Base res. + Contact(s)":
        st.caption(
            "Enter depth slices and Top/Base areas. Choose depth step, extrapolation, and depth distributions. "
            "GRV at Effective HC depth will be used in PV."
        )
        st.info("Depth in meters; areas in km². 1 km²·m = 10⁶ m³.")
        
        # Warning about phase-specific volumes for Monte Carlo simulation
        fluid_type = st.session_state.get("fluid_type", "Oil")
        if fluid_type == "Oil + Gas":
            st.warning(
                "⚠️ **Calculation Required:** For accurate Monte Carlo simulation results, the phase-specific "
                "Gross Rock Volumes (GRV oil and GRV gas) must be calculated first. This calculation runs "
                "automatically when you configure the GRV inputs below. Please ensure the calculation completes "
                "and the results are displayed in the 'GRV Results by Fluid Type' section before navigating "
                "to the Run Simulation page."
            )
        default_table = pd.DataFrame(
            {
                "Depth": [
                    2040,
                    2050,
                    2060,
                    2070,
                    2080,
                    2090,
                    2100,
                    2110,
                    2120,
                    2130,
                    2140,
                    2150,
                    2160,
                    2170,
                    2180,
                    2190,
                    2200,
                    2210,
                    2220,
                    2230,
                    2240,
                    2250,
                    2260,
                    2270,
                    2280,
                    2290,
                    2300,
                    2310,
                    2320,
                    2330,
                    2340,
                    2350,
                    2360,
                    2370,
                    2380,
                    2390,
                    2400,
                ],
                "Top area (km2)": [
                    0.0,
                    0.06149313,
                    0.743428415,
                    1.31427065,
                    1.917695765,
                    2.581499555,
                    3.357041565,
                    4.24747926,
                    5.178613535,
                    6.122726925,
                    7.07979995,
                    8.04906025,
                    9.02872446,
                    10.02516695,
                    11.0326527,
                    12.09376061,
                    13.24614827,
                    14.32668123,
                    15.41667629,
                    16.61170115,
                    17.8503389,
                    19.01679186,
                    20.11461198,
                    21.1684545,
                    22.18402345,
                    23.13352475,
                    24.06078863,
                    24.95179712,
                    25.7778064,
                    26.55293472,
                    27.3037083,
                    28.04968615,
                    28.81130504,
                    29.61183057,
                    30.47923043,
                    31.46059753,
                    32.624375,
                ],
                "Base area (km2)": [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.06149313,
                    0.743428415,
                    1.31427065,
                    1.917695765,
                    2.581499555,
                    3.357041565,
                    4.24747926,
                    5.178613535,
                    6.122726925,
                    7.07979995,
                    8.04906025,
                    9.02872446,
                    10.02516695,
                    11.0326527,
                    12.09376061,
                    13.24614827,
                    14.32668123,
                    15.41667629,
                    16.61170115,
                    17.8503389,
                    19.01679186,
                    20.11461198,
                    21.1684545,
                    22.18402345,
                    23.13352475,
                    24.06078863,
                    24.95179712,
                    25.7778064,
                    26.55293472,
                    27.3037083,
                    28.04968615,
                ],
            }
        )

        if "grv_table_input" not in st.session_state:
            st.session_state["grv_table_input"] = default_table

        edited = st.data_editor(
            st.session_state["grv_table_input"],
            num_rows="dynamic",
            use_container_width=True,
            key="grv_editor",
            column_config={
                "Depth": st.column_config.NumberColumn("Depth (m)", format="%.10f"),
                "Top area (km2)": st.column_config.NumberColumn("Top area (km²)", format="%.10f"),
                "Base area (km2)": st.column_config.NumberColumn("Base area (km²)", format="%.10f"),
            },
        )
        st.session_state["grv_table_input"] = edited

        colc1, colc2 = st.columns(2)
        with colc1:
            # CRITICAL: Values are already initialized by _initialize_grv_input_defaults()
            # Use value= to ensure widget shows the persisted value
            step_size = st.number_input(
                "Depth step (m)", 
                min_value=1.0,
                value=st.session_state["da_step_size"],
                key="da_step_size",
                help="Step size for integration grid."
            )
            extrap = st.checkbox(
                "Allow linear extrapolation",
                value=st.session_state["da_extrap"],
                key="da_extrap",
                help="Allow linear extrapolation beyond final row if needed.",
            )
        with colc2:
            area_uncert_enabled = st.checkbox(
                "Enable area uncertainty multiplier",
                value=st.session_state["da_area_uncert_enabled"],
                key="da_area_uncert_enabled",
                help="Apply uncertainty multiplier to area curves.",
            )
            if area_uncert_enabled:
                # Values are already initialized by _initialize_grv_input_defaults()
                area_dist_options = ["PERT", "Triangular", "Uniform"]
                stored_area_dist = st.session_state["da_area_dist"]
                try:
                    area_dist_index = area_dist_options.index(stored_area_dist)
                except ValueError:
                    area_dist_index = 0
                area_dist = st.selectbox(
                    "Area multiplier distribution",
                    area_dist_options,
                    index=area_dist_index,
                    key="da_area_dist",
                    help="Distribution applied to area multiplier.",
                )
                if area_dist == "PERT":
                    sAreaMult = render_param(
                        "AreaMult", "Area multiplier", "", "PERT", {"min": 0.8, "mode": 1.0, "max": 1.2}, num_sims
                    )
                elif area_dist == "Triangular":
                    sAreaMult = render_param(
                        "AreaMult", "Area multiplier", "", "Triangular", {"min": 0.7, "mode": 1.0, "max": 1.3}, num_sims
                    )
                else:
                    sAreaMult = render_param(
                        "AreaMult", "Area multiplier", "", "Uniform", {"min": 0.85, "max": 1.15}, num_sims
                    )
            else:
                sAreaMult = None

        sGRV_m3 = np.zeros(num_sims)
        grv_values_km2m = np.zeros(num_sims)

        st.markdown("### Depth Distributions")
        col_depth1, col_depth2 = st.columns(2)
        with col_depth1:
            sSpillPoint = render_param(
                "SpillPointDepth",
                "Spill point depth",
                "m",
                defaults["d"]["dist"],
                {"min": 2200, "mode": 2350, "max": 2410},
                num_sims,
                stats_decimals=1,
            )
        with col_depth2:
            sHCDepth = render_param(
                "HCDepth",
                "Effective HC depth",
                "m",
                defaults["d"]["dist"],
                {"min": 2100, "mode": 2350, "max": 2400},
                num_sims,
                stats_decimals=1,
            )

        st.session_state["sD_spill"] = sSpillPoint
        st.session_state["sD_hc"] = sHCDepth

        # Get fluid type
        fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
        
        # For Oil + Gas, show GOC definition options
        sGOC_depth = None
        if fluid_type == "Oil + Gas":
            st.markdown("#### Gas-Oil Contact (GOC) Definition")
            goc_mode_key = "da_goc_mode"
            goc_mode_options = ["Direct depth", "Oil fraction of HC column", "Oil column height"]
            
            # CRITICAL: Use persistent storage similar to parameter values system
            # Store in a special location that won't be touched by initialization functions
            persistent_storage_key = "_ui_state"
            if persistent_storage_key not in st.session_state:
                st.session_state[persistent_storage_key] = {}
            
            # CRITICAL: Check if widget already exists in session state (user may have just clicked)
            # The widget's value (from session state) is the source of truth AFTER it's created
            widget_value = st.session_state.get(goc_mode_key, None)
            
            # If widget has a value, use it (user just clicked or it's already set)
            if widget_value is not None and widget_value in goc_mode_options:
                # Widget has a valid value - this is the source of truth
                desired_goc_mode = widget_value
                # Save to persistent storage
                st.session_state[persistent_storage_key][goc_mode_key] = desired_goc_mode
            else:
                # Widget doesn't have a value yet - read from persistent storage
                persistent_goc_mode = st.session_state[persistent_storage_key].get(goc_mode_key, None)
                
                # If not in persistent storage, use default
                if persistent_goc_mode is None:
                    persistent_goc_mode = "Oil fraction of HC column"
                
                # Validate the stored value
                if persistent_goc_mode not in goc_mode_options:
                    persistent_goc_mode = "Oil fraction of HC column"
                
                desired_goc_mode = persistent_goc_mode
                # Store in both locations
                st.session_state[persistent_storage_key][goc_mode_key] = desired_goc_mode
                st.session_state[goc_mode_key] = desired_goc_mode
            
            # Calculate index for display
            goc_mode_index = goc_mode_options.index(desired_goc_mode)
            
            # Create widget - Streamlit will use st.session_state[goc_mode_key] when key= is provided
            goc_mode = st.radio(
                "GOC definition mode",
                options=goc_mode_options,
                index=goc_mode_index,
                key=goc_mode_key,
                help="Choose how to define the Gas-Oil Contact depth"
            )
            
            # CRITICAL: After widget creation, ALWAYS use widget's value and save to persistent storage
            # The widget's value is the authoritative source after creation
            actual_goc_mode = st.session_state[goc_mode_key]
            if actual_goc_mode in goc_mode_options:
                # Valid value - save to persistent storage and use it
                st.session_state[persistent_storage_key][goc_mode_key] = actual_goc_mode
                goc_mode = actual_goc_mode
            else:
                # Invalid value - use what we had before
                goc_mode = desired_goc_mode
                st.session_state[goc_mode_key] = desired_goc_mode
            
            # CRITICAL: Use explicit if/elif/else to ensure only one section renders
            # Streamlit will rerun when radio button changes, so the correct section will render
            if goc_mode == "Direct depth":
                sGOC_depth = render_param(
                    "GOCDepth",
                    "GOC depth",
                    "m",
                    defaults["d"]["dist"],
                    {"min": 2100, "mode": 2250, "max": 2350},
                    num_sims,
                    stats_decimals=1,
                )
                st.session_state["sGOC_depth"] = sGOC_depth
            elif goc_mode == "Oil fraction of HC column":
                # Replace slider with distribution selector (Stretched Beta default)
                sF_oil = render_param(
                    "F_oil",
                    "Oil fraction of total HC column",
                    "",
                    "Stretched Beta",
                    {"min": 0.05, "mode": 0.2, "max": 0.3},
                    num_sims,
                    stats_decimals=4,
                    help_text="Fraction of hydrocarbon column that is oil (remainder is gas). Distribution of values between 0 and 1."
                )
                # Store the sampled array for use in calculations
                st.session_state["da_oil_frac"] = sF_oil
            elif goc_mode == "Oil column height":
                h_oil_key = "da_h_oil_m"
                st.markdown("**Oil column height (m above HCWC)**")
                sH_oil_m = render_param(
                    "H_oil_m",
                    "Oil column height",
                    "m",
                    "PERT",
                    {"min": 10.0, "mode": 20.0, "max": 50.0},
                    num_sims,
                    stats_decimals=1,
                )
                st.session_state[h_oil_key] = sH_oil_m
            
            st.caption("ℹ️ Gas zone: Top Structure → GOC | Oil zone: GOC → Effective HC depth")
        
        # Display depth-area plot with P10/P90 uncertainty bands (after depth distributions are defined)
        if len(edited) > 0 and sSpillPoint is not None and sHCDepth is not None:
            depths_plot = np.asarray(edited["Depth"].values, dtype=float)
            top_areas_plot = np.asarray(edited["Top area (km2)"].values, dtype=float)
            base_areas_plot = np.asarray(edited["Base area (km2)"].values, dtype=float)
            
            # Interpolate to a fine grid for smooth plotting
            from scipy.interpolate import interp1d
            depth_min = float(np.min(depths_plot))
            depth_max = float(np.max(depths_plot))
            depth_step_plot = 10.0  # 10m steps for smooth curves
            depths_interp = np.arange(depth_min, depth_max + depth_step_plot, depth_step_plot)
            
            # Interpolate areas
            f_top = interp1d(depths_plot, top_areas_plot, kind='linear', fill_value='extrapolate', bounds_error=False)
            f_base = interp1d(depths_plot, base_areas_plot, kind='linear', fill_value='extrapolate', bounds_error=False)
            top_interp = f_top(depths_interp)
            base_interp = f_base(depths_interp)
            
            # Calculate cumulative volumes using trapezoidal integration
            from scopehc.geom import _cumulative_trapz
            vol_top = _cumulative_trapz(top_interp, depth_step_plot)  # km²·m
            vol_base = _cumulative_trapz(base_interp, depth_step_plot)  # km²·m
            
            # Calculate GRV = Top Volume - Base Volume (same logic as Top + Contacts)
            dgrv = vol_top - vol_base  # km²·m (differential GRV = Top - Base)
            
            # Calculate P10/P90 uncertainty bands using depth distributions
            n_samples_uncert = min(1000, num_sims)  # Use up to 1000 samples for uncertainty
            top_areas_uncert = []
            base_areas_uncert = []
            vol_top_uncert = []
            vol_base_uncert = []
            dgrv_uncert = []
            
            for j in range(n_samples_uncert):
                spill_d = sSpillPoint[j % len(sSpillPoint)]
                hc_d = sHCDepth[j % len(sHCDepth)]
                
                # Interpolate areas at this depth
                top_area_at_d = np.interp(depths_interp, depths_plot, top_areas_plot)
                base_area_at_d = np.interp(depths_interp, depths_plot, base_areas_plot)
                
                top_areas_uncert.append(top_area_at_d)
                base_areas_uncert.append(base_area_at_d)
                
                # Calculate volumes for this realization
                vol_top_j = _cumulative_trapz(top_area_at_d, depth_step_plot)
                vol_base_j = _cumulative_trapz(base_area_at_d, depth_step_plot)
                dgrv_j = vol_top_j - vol_base_j  # GRV = Top - Base
                
                vol_top_uncert.append(vol_top_j)
                vol_base_uncert.append(vol_base_j)
                dgrv_uncert.append(dgrv_j)
            
            # Calculate percentiles
            top_areas_uncert = np.array(top_areas_uncert)
            base_areas_uncert = np.array(base_areas_uncert)
            vol_top_uncert = np.array(vol_top_uncert)
            vol_base_uncert = np.array(vol_base_uncert)
            dgrv_uncert = np.array(dgrv_uncert)
            
            top_p10 = np.percentile(top_areas_uncert, 10, axis=0)
            top_p90 = np.percentile(top_areas_uncert, 90, axis=0)
            base_p10 = np.percentile(base_areas_uncert, 10, axis=0)
            base_p90 = np.percentile(base_areas_uncert, 90, axis=0)
            
            # Get mean spill point and HC depth for display
            mean_spill = float(np.mean(sSpillPoint))
            mean_hc = float(np.mean(sHCDepth))
            
            # Get top structure depth (minimum depth from the depth array)
            mean_top = float(np.min(depths_plot)) if len(depths_plot) > 0 else mean_spill
            
            # Find GRV at spill point for marker
            spill_idx = np.argmin(np.abs(depths_interp - mean_spill))
            grv_sp_km2m = dgrv[spill_idx] if spill_idx < len(dgrv) else dgrv[-1]
            
            # Get mean GOC depth if available
            mean_goc = None
            fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
            sGOC_depth_check = st.session_state.get("sGOC_depth")
            if fluid_type == "Oil + Gas" and sGOC_depth_check is not None:
                mean_goc = float(np.mean(sGOC_depth_check))
            
            # Create plot similar to Top + Contacts: Top Volume, Base Volume, GRV = Top - Base
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            from scopehc.plots import color_for, rgba_from_hex
            
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Top & Base Area vs Depth", "Volume vs Depth"))
            
            # Left: Areas with P10/P90
            top_color = color_for("GRV_oil_m3")
            base_color = color_for("GRV_gas_m3")
            grayish_red = "#C8A8A8"  # Grayish-red for base P10/P90
            
            fig.add_trace(go.Scatter(x=top_interp, y=depths_interp, mode="lines",
                                     line=dict(color=top_color, width=2),
                                     name="Top Area"), row=1, col=1)
            fig.add_trace(go.Scatter(x=base_interp, y=depths_interp, mode="lines",
                                     line=dict(color=base_color, width=2, dash="dash"),
                                     name="Base Area"), row=1, col=1)
            
            if top_p10 is not None and top_p90 is not None:
                fig.add_trace(go.Scatter(x=top_p10, y=depths_interp, mode="lines",
                                         line=dict(color=rgba_from_hex(top_color, 0.75), width=2, dash="dash"),
                                         name="Top Area P10", showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=top_p90, y=depths_interp, mode="lines",
                                         line=dict(color=rgba_from_hex(top_color, 0.75), width=2, dash="dash"),
                                         name="Top Area P90", showlegend=False), row=1, col=1)
            
            if base_p10 is not None and base_p90 is not None:
                fig.add_trace(go.Scatter(x=base_p10, y=depths_interp, mode="lines",
                                         line=dict(color=rgba_from_hex(grayish_red, 0.75), width=2, dash="dash"),
                                         name="Base Area P10", showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=base_p90, y=depths_interp, mode="lines",
                                         line=dict(color=rgba_from_hex(grayish_red, 0.75), width=2, dash="dash"),
                                         name="Base Area P90", showlegend=False), row=1, col=1)
            
            # Right: Volumes (Top, Base, GRV = Top - Base)
            vol_color = color_for("GRV_total_m3")
            fig.add_trace(go.Scatter(x=vol_top, y=depths_interp, mode="lines",
                                     line=dict(color=vol_color, width=2),
                                     name="Top Volume"), row=1, col=2)
            fig.add_trace(go.Scatter(x=vol_base, y=depths_interp, mode="lines",
                                     line=dict(color=base_color, width=2, dash="dash"),
                                     name="Base Volume"), row=1, col=2)
            fig.add_trace(go.Scatter(x=dgrv, y=depths_interp, mode="lines",
                                     line=dict(color=top_color, width=2),
                                     name="GRV = Top − Base"), row=1, col=2)
            
            # Horizontal lines for contacts - add to BOTH plots
            def _hline(yval, name, color_key, x_max_area, x_max_vol):
                # Add to area plot (col=1)
                fig.add_shape(type="line", x0=0, x1=x_max_area,
                              y0=yval, y1=yval, xref="x1", yref="y1",
                              line=dict(color=color_for(color_key), width=1, dash="dot"))
                fig.add_annotation(x=x_max_area * 0.98, y=yval, xref="x1", yref="y1",
                                   text=name, showarrow=False, font=dict(size=10, color=color_for(color_key)),
                                   xanchor="right", yanchor="middle")
                # Add to volume plot (col=2)
                fig.add_shape(type="line", x0=0, x1=x_max_vol,
                              y0=yval, y1=yval, xref="x2", yref="y2",
                              line=dict(color=color_for(color_key), width=1, dash="dot"))
                fig.add_annotation(x=x_max_vol * 0.98, y=yval, xref="x2", yref="y2",
                                   text=name, showarrow=False, font=dict(size=10, color=color_for(color_key)),
                                   xanchor="right", yanchor="middle")
            
            # Calculate max values for both plots
            x_max_area = max(1e-9, float(np.nanmax([np.nanmax(top_interp), np.nanmax(base_interp)])) * 1.05) if len(top_interp) > 0 and len(base_interp) > 0 else 1.0
            x_max_vol = max(1e-9, float(np.nanmax(vol_top)) * 1.05) if len(vol_top) > 0 else 1.0
            
            # Add horizontal lines for Apex, Spill, HCWC, and GOC to BOTH plots
            _hline(mean_top, "Apex", "GRV_total_m3", x_max_area, x_max_vol)
            _hline(mean_spill, "Spill", "SpillPoint", x_max_area, x_max_vol)
            if mean_hc is not None:
                _hline(mean_hc, "HCWC", "Effective_HC_depth", x_max_area, x_max_vol)
            if mean_goc is not None:
                _hline(mean_goc, "GOC", "GOC", x_max_area, x_max_vol)
            
            # Spill point marker on volume plot
            if not np.isnan(grv_sp_km2m):
                fig.add_trace(go.Scatter(x=[grv_sp_km2m], y=[mean_spill], mode="markers",
                                         marker=dict(symbol="star", size=12, color=color_for("SpillPoint")),
                                         name="Spill Point", showlegend=False), row=1, col=2)
            
            fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
            fig.update_xaxes(title_text="Area (km²)", row=1, col=1)
            fig.update_xaxes(title_text="Volume (km²·m)", row=1, col=2)
            fig.update_layout(
                margin=dict(l=40, r=40, t=60, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                width=None,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Depth-area and volume relationships. GRV = Top Volume - Base Volume.")
        
        # Arrays to store split GRV and column heights
        sGRV_oil_m3 = np.zeros(num_sims)
        sGRV_gas_m3 = np.zeros(num_sims)
        sOil_Column_Height = np.zeros(num_sims)
        sGas_Column_Height = np.zeros(num_sims)
        
        # Pre-compute depth and area arrays (same for all trials)
        depths_m = np.asarray(edited["Depth"].values, dtype=float)
        top_areas_km2 = np.asarray(edited["Top area (km2)"].values, dtype=float)
        base_areas_km2 = np.asarray(edited["Base area (km2)"].values, dtype=float)
        areas_km2 = (top_areas_km2 + base_areas_km2) / 2.0
        top_structure_m = float(np.min(depths_m))
        
        # Track invalid trials for summary warning
        invalid_trials = []
        
        # Use v3-compatible function
        from scopehc.geom import grv_by_depth_v3_compatible, derive_goc_from_mode
        
        for i in range(num_sims):
            spill_depth = sSpillPoint[i]
            hc_depth = sHCDepth[i]
            
            # Derive GOC based on fluid type and mode
            if fluid_type == "Oil":
                # Oil only: integrate Top -> HCWC (OWC)
                goc_depth = None
                owc_depth = hc_depth
            elif fluid_type == "Gas":
                # Gas only: integrate Top -> HCWC (no oil)
                goc_depth = hc_depth  # GOC at HCWC means all gas
                owc_depth = None
            else:
                # Oil + Gas: derive GOC from mode
                goc_mode = st.session_state.get("da_goc_mode", "Oil fraction of HC column")
                if goc_mode == "Direct depth":
                    goc_depth = sGOC_depth[i] if sGOC_depth is not None else None
                else:
                    # Derive GOC from fraction or height
                    goc_depth = derive_goc_from_mode(top_structure_m, hc_depth, goc_mode, st.session_state, trial_idx=i)
                owc_depth = hc_depth
            
            # Validate: HCWC should be >= Top
            if hc_depth < top_structure_m:
                invalid_trials.append(i)
                sGRV_m3[i] = 0.0
                sGRV_oil_m3[i] = 0.0
                sGRV_gas_m3[i] = 0.0
                sOil_Column_Height[i] = 0.0
                sGas_Column_Height[i] = 0.0
                grv_values_km2m[i] = 0.0
                continue
            
            # Calculate GRV at spill point (for display)
            result_spill = grv_by_depth_v3_compatible(
                depth_m=depths_m,
                area_m2=areas_km2,
                top_structure_m=top_structure_m,
                goc_m=goc_depth if fluid_type == "Oil + Gas" else None,
                owc_m=spill_depth,
                spill_m=spill_depth
            )
            grv_values_km2m[i] = result_spill['GRV_total_m3'] / 1e6  # Convert to km²·m for display
            
            # Calculate GRV at HC depth (for use in calculations)
            result_hc = grv_by_depth_v3_compatible(
                depth_m=depths_m,
                area_m2=areas_km2,
                top_structure_m=top_structure_m,
                goc_m=goc_depth,
                owc_m=owc_depth,
                spill_m=spill_depth
            )
            sGRV_m3[i] = result_hc['GRV_total_m3']
            sGRV_oil_m3[i] = result_hc['GRV_oil_m3']
            sGRV_gas_m3[i] = result_hc['GRV_gas_m3']
            sOil_Column_Height[i] = result_hc['H_oil_m']
            sGas_Column_Height[i] = result_hc['H_gas_m']
        
        # Show summary warnings after loop completes
        if invalid_trials:
            st.warning(
                f"⚠️ **{len(invalid_trials)} trial(s) had invalid contacts** "
                f"(HCWC above Top structure). Volumes set to zero for these trials."
            )
        
        # Check for collapsed GOC (only check first few trials to avoid spam)
        collapsed_goc_count = 0
        if fluid_type == "Oil + Gas":
            for i in range(min(100, num_sims)):  # Sample first 100 trials
                if sGOC_depth is not None and i < len(sGOC_depth):
                    goc_depth_sample = sGOC_depth[i] if isinstance(sGOC_depth, np.ndarray) else sGOC_depth
                    hc_depth_sample = sHCDepth[i]
                    if goc_depth_sample is not None:
                        if abs(goc_depth_sample - top_structure_m) < 0.1 or abs(goc_depth_sample - hc_depth_sample) < 0.1:
                            collapsed_goc_count += 1
            if collapsed_goc_count > 0:
                st.caption("ℹ️ GOC collapsed to boundary in some trials; one phase may have zero height.")

        if sAreaMult is not None:
            grv_values_km2m = grv_values_km2m * sAreaMult
            sGRV_m3 = sGRV_m3 * sAreaMult
            sGRV_oil_m3 = sGRV_oil_m3 * sAreaMult
            sGRV_gas_m3 = sGRV_gas_m3 * sAreaMult
        
        # Store split GRV and column heights in session state
        st.session_state["sGRV_oil_m3"] = sGRV_oil_m3
        st.session_state["sGRV_gas_m3"] = sGRV_gas_m3
        st.session_state["sOil_Column_Height"] = sOil_Column_Height
        st.session_state["sGas_Column_Height"] = sGas_Column_Height
        
        # CRITICAL: Mark these arrays as depth-based calculated to prevent overwriting
        st.session_state["_grv_arrays_depth_based"] = True
        st.session_state["_grv_arrays_method"] = "Depth-based: Top and Base res. + Contact(s)"
        
        # Set final GRV for depth-based method
        st.session_state["sGRV_m3_final"] = sGRV_m3
        
        # Display split GRV results for depth-based method (spill + HC contact)
        st.markdown("---")
        st.markdown("### GRV Results by Fluid Type")
        fluid_type = st.session_state.get("fluid_type", "Oil")
        
        if fluid_type == "Oil":
            st.markdown("#### Oil GRV")
            st.plotly_chart(
                make_hist_cdf_figure(
                    sGRV_oil_m3 / 1e6,
                    "Oil GRV Distribution (×10^6 m³)",
                    "GRV (×10^6 m³)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(sGRV_oil_m3 / 1e6, decimals=2), use_container_width=True)
            st.caption("ℹ️ Oil only: entire hydrocarbon column is oil.")
            
        elif fluid_type == "Gas":
            st.markdown("#### Gas GRV")
            st.plotly_chart(
                make_hist_cdf_figure(
                    sGRV_gas_m3 / 1e6,
                    "Gas GRV Distribution (×10^6 m³)",
                    "GRV (×10^6 m³)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(sGRV_gas_m3 / 1e6, decimals=2), use_container_width=True)
            st.caption("ℹ️ Gas only: entire hydrocarbon column is gas.")
            
        else:  # Oil + Gas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Oil GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_oil_m3 / 1e6,
                        "Oil GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_oil_m3 / 1e6, decimals=2), use_container_width=True)
            
            with col2:
                st.markdown("#### Gas GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_gas_m3 / 1e6,
                        "Gas GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_gas_m3 / 1e6, decimals=2), use_container_width=True)
            
            with col3:
                st.markdown("#### Total HC GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_m3 / 1e6,
                        "Total HC GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_m3 / 1e6, decimals=2), use_container_width=True)
            
            # Combined summary table for depth-based method (spill + HC contact)
            st.markdown("#### Combined GRV Summary")
            summary_data = {
                "Parameter": ["GRV Oil (×10⁶ m³)", "GRV Gas (×10⁶ m³)", "GRV Total HC (×10⁶ m³)"],
            }
            
            # Get statistics for each
            oil_stats = summarize_array(sGRV_oil_m3 / 1e6)
            gas_stats = summarize_array(sGRV_gas_m3 / 1e6)
            total_stats = summarize_array(sGRV_m3 / 1e6)
            
            summary_data["Mean"] = [
                f"{oil_stats.get('mean', 0.0):.2f}",
                f"{gas_stats.get('mean', 0.0):.2f}",
                f"{total_stats.get('mean', 0.0):.2f}",
            ]
            summary_data["P10"] = [
                f"{oil_stats.get('P10', 0.0):.2f}",
                f"{gas_stats.get('P10', 0.0):.2f}",
                f"{total_stats.get('P10', 0.0):.2f}",
            ]
            summary_data["P50"] = [
                f"{oil_stats.get('P50', 0.0):.2f}",
                f"{gas_stats.get('P50', 0.0):.2f}",
                f"{total_stats.get('P50', 0.0):.2f}",
            ]
            summary_data["P90"] = [
                f"{oil_stats.get('P90', 0.0):.2f}",
                f"{gas_stats.get('P90', 0.0):.2f}",
                f"{total_stats.get('P90', 0.0):.2f}",
            ]
            summary_data["Std"] = [
                f"{oil_stats.get('std_dev', 0.0):.2f}",
                f"{gas_stats.get('std_dev', 0.0):.2f}",
                f"{total_stats.get('std_dev', 0.0):.2f}",
            ]
            summary_data["Min"] = [
                f"{oil_stats.get('min', 0.0):.2f}",
                f"{gas_stats.get('min', 0.0):.2f}",
                f"{total_stats.get('min', 0.0):.2f}",
            ]
            summary_data["Max"] = [
                f"{oil_stats.get('max', 0.0):.2f}",
                f"{gas_stats.get('max', 0.0):.2f}",
                f"{total_stats.get('max', 0.0):.2f}",
            ]
            
            df_combined = pd.DataFrame(summary_data)
            st.dataframe(df_combined, use_container_width=True, hide_index=True)
            
            # Show GOC mode info
            goc_mode = st.session_state.get("da_goc_mode", None)
            if goc_mode:
                mode_desc = {
                    "Direct depth": "GOC defined by direct depth distribution",
                    "Oil fraction of HC column": f"GOC derived from oil fraction: fₒᵢₗ = {_get_scalar_from_state(st.session_state, 'da_oil_frac', 0.2):.2f} (mean from distribution)",
                    "Oil column height": f"GOC derived from oil column height: {_get_scalar_from_state(st.session_state, 'da_h_oil_m', 0):.1f} m above HCWC (distribution)"
                }
                st.caption(f"ℹ️ {mode_desc.get(goc_mode, 'GOC split applied')}")
        
        # Edge case warnings
        if sGOC_depth is not None:
            # Check for reversed contacts
            reversed_count = np.sum((sGOC_depth > sHCDepth) & (sHCDepth > 0) & (sGOC_depth > 0))
            if reversed_count > 0:
                st.warning(f"⚠️ Contacts appear reversed in {reversed_count} trial(s); volumes truncated to zero where appropriate.")
        
        # Check if contacts are beyond depth range
        depth_min = float(np.min(edited["Depth"].values))
        depth_max = float(np.max(edited["Depth"].values))
        if np.any(sHCDepth < depth_min) or np.any(sHCDepth > depth_max):
            st.caption("ℹ️ Some contacts were clipped to the available depth range of the area-depth table.")
        
        if fluid_type != "Oil + Gas":
            if fluid_type == "Oil":
                st.caption("ℹ️ Oil only: entire hydrocarbon column treated as oil to OWC.")
            elif fluid_type == "Gas":
                st.caption("ℹ️ Gas only: entire hydrocarbon column treated as gas to HCWC.")

        st.markdown("#### GRV Distribution Above Spill Point")
        st.plotly_chart(
            make_hist_cdf_figure(
                grv_values_km2m * 1e6 / 1e6,
                "GRV distribution at Spill Point (×10^6 m³)",
                "GRV (×10^6 m³)",
                "calculated",
            ),
            use_container_width=True,
        )

    elif grv_option == "Depth-based: Top + Res. thickness + Contact(s)":
        st.caption("Provide top structure depth table and constant thickness. GRV is integrated accordingly.")
        st.info("Depth in meters; areas in km². 1 km²·m = 10⁶ m³.")
        
        # Warning about phase-specific volumes for Monte Carlo simulation
        fluid_type = st.session_state.get("fluid_type", "Oil")
        if fluid_type == "Oil + Gas":
            st.warning(
                "⚠️ **Calculation Required:** For accurate Monte Carlo simulation results, the phase-specific "
                "Gross Rock Volumes (GRV oil and GRV gas) must be calculated first. This calculation runs "
                "automatically when you configure the GRV inputs below. Please ensure the calculation completes "
                "and the results are displayed in the 'GRV Results by Fluid Type' section before navigating "
                "to the Run Simulation page."
            )

        default_top_table = pd.DataFrame(
            {
                "Depth": [
                    2040,
                    2050,
                    2060,
                    2070,
                    2080,
                    2090,
                    2100,
                    2110,
                    2120,
                    2130,
                    2140,
                    2150,
                    2160,
                    2170,
                    2180,
                    2190,
                    2200,
                    2210,
                    2220,
                    2230,
                    2240,
                    2250,
                    2260,
                    2270,
                    2280,
                    2290,
                    2300,
                    2310,
                    2320,
                    2330,
                    2340,
                    2350,
                    2360,
                    2370,
                    2380,
                    2390,
                    2400,
                ],
                "Top area (km2)": [
                    0.0,
                    0.06149313,
                    0.743428415,
                    1.31427065,
                    1.917695765,
                    2.581499555,
                    3.357041565,
                    4.24747926,
                    5.178613535,
                    6.122726925,
                    7.07979995,
                    8.04906025,
                    9.02872446,
                    10.02516695,
                    11.0326527,
                    12.09376061,
                    13.24614827,
                    14.32668123,
                    15.41667629,
                    16.61170115,
                    17.8503389,
                    19.01679186,
                    20.11461198,
                    21.1684545,
                    22.18402345,
                    23.13352475,
                    24.06078863,
                    24.95179712,
                    25.7778064,
                    26.55293472,
                    27.3037083,
                    28.04968615,
                    28.81130504,
                    29.61183057,
                    30.47923043,
                    31.46059753,
                    32.624375,
                ],
            }
        )

        if "top_only_table" not in st.session_state:
            st.session_state["top_only_table"] = default_top_table

        ed_top = st.data_editor(
            st.session_state["top_only_table"],
            num_rows="dynamic",
            use_container_width=True,
            key="top_table_editor",
            column_config={
                "Depth": st.column_config.NumberColumn("Depth (m)", format="%.3f"),
                "Top area (km2)": st.column_config.NumberColumn("Top area (km²)", format="%.3f"),
            },
        )
        st.session_state["top_only_table"] = ed_top

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            # Reservoir thickness as a distribution (PERT 100, 120, 200m)
            sThickness = render_param(
                "ReservoirThickness",
                "Reservoir thickness",
                "m",
                "PERT",
                {"min": 100.0, "mode": 120.0, "max": 200.0},
                num_sims,
                stats_decimals=1,
            )
            st.session_state["sReservoirThickness"] = sThickness
            # CRITICAL: Values are already initialized by _initialize_grv_input_defaults()
            # Use value= to ensure widget shows the persisted value
            step_size_d = st.number_input(
                "Depth step (m)", 
                min_value=1.0,
                value=st.session_state["da_D_step_size"],
                key="da_D_step_size",
                help="Step size for integration grid."
            )
            extrap_d = st.checkbox(
                "Allow linear extrapolation (top table)",
                value=st.session_state["da_D_extrap"],
                key="da_D_extrap",
                help="Allow extrapolation beyond last row."
            )
        with col_top2:
            area_uncert_enabled_D = st.checkbox(
                "Enable area uncertainty multiplier (top table)",
                value=st.session_state["da_D_area_uncert_enabled"],
                key="da_D_area_uncert_enabled",
                help="Apply uncertainty multiplier to top area curve.",
            )
            if area_uncert_enabled_D:
                # Values are already initialized by _initialize_grv_input_defaults()
                area_dist_D_options = ["PERT", "Triangular", "Uniform"]
                stored_area_dist_D = st.session_state["da_D_area_dist"]
                try:
                    area_dist_D_index = area_dist_D_options.index(stored_area_dist_D)
                except ValueError:
                    area_dist_D_index = 0
                area_dist_D = st.selectbox(
                    "Area multiplier distribution (top table)", 
                    area_dist_D_options,
                    index=area_dist_D_index,
                    key="da_D_area_dist"
                )
                if area_dist_D == "PERT":
                    sAreaMult_D = render_param(
                        "AreaMult_D",
                        "Top area multiplier",
                        "",
                        "PERT",
                        {"min": 0.8, "mode": 1.0, "max": 1.2},
                        num_sims,
                    )
                elif area_dist_D == "Triangular":
                    sAreaMult_D = render_param(
                        "AreaMult_D",
                        "Top area multiplier",
                        "",
                        "Triangular",
                        {"min": 0.7, "mode": 1.0, "max": 1.3},
                        num_sims,
                    )
                else:
                    sAreaMult_D = render_param(
                        "AreaMult_D", "Top area multiplier", "", "Uniform", {"min": 0.85, "max": 1.15}, num_sims
                    )
            else:
                sAreaMult_D = None

        sGRV_m3 = np.zeros(num_sims)
        grv_values_km2m_D = np.zeros(num_sims)

        st.markdown("### Depth Distributions (Top-only)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            sSpillPointD = render_param(
                "SpillPointDepthD",
                "Spill point depth",
                "m",
                defaults["d"]["dist"],
                {"min": 2200, "mode": 2350, "max": 2410},
                num_sims,
                stats_decimals=1,
            )
        with col_d2:
            sHCDepthD = render_param(
                "HCDepthD",
                "Effective HC depth",
                "m",
                defaults["d"]["dist"],
                {"min": 2100, "mode": 2350, "max": 2400},
                num_sims,
                stats_decimals=1,
            )

        st.session_state["sD_spill"] = sSpillPointD
        st.session_state["sD_hc"] = sHCDepthD

        # Get fluid type
        fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
        
        # For Oil + Gas, show GOC definition options
        sGOC_depth_D = None
        if fluid_type == "Oil + Gas":
            st.markdown("#### Gas-Oil Contact (GOC) Definition")
            goc_mode_key_D = "da_goc_mode_D"
            goc_mode_options_D = ["Direct depth", "Oil fraction of HC column", "Oil column height"]
            
            # CRITICAL: Use persistent storage similar to parameter values system
            # Store in a special location that won't be touched by initialization functions
            persistent_storage_key = "_ui_state"
            if persistent_storage_key not in st.session_state:
                st.session_state[persistent_storage_key] = {}
            
            # CRITICAL: Check if widget already exists in session state (user may have just clicked)
            # The widget's value (from session state) is the source of truth AFTER it's created
            widget_value_D = st.session_state.get(goc_mode_key_D, None)
            
            # If widget has a value, use it (user just clicked or it's already set)
            if widget_value_D is not None and widget_value_D in goc_mode_options_D:
                # Widget has a valid value - this is the source of truth
                desired_goc_mode_D = widget_value_D
                # Save to persistent storage
                st.session_state[persistent_storage_key][goc_mode_key_D] = desired_goc_mode_D
            else:
                # Widget doesn't have a value yet - read from persistent storage
                persistent_goc_mode_D = st.session_state[persistent_storage_key].get(goc_mode_key_D, None)
                
                # If not in persistent storage, use default
                if persistent_goc_mode_D is None:
                    persistent_goc_mode_D = "Oil fraction of HC column"
                
                # Validate the stored value
                if persistent_goc_mode_D not in goc_mode_options_D:
                    persistent_goc_mode_D = "Oil fraction of HC column"
                
                desired_goc_mode_D = persistent_goc_mode_D
                # Store in both locations
                st.session_state[persistent_storage_key][goc_mode_key_D] = desired_goc_mode_D
                st.session_state[goc_mode_key_D] = desired_goc_mode_D
            
            # Calculate index for display
            goc_mode_index_D = goc_mode_options_D.index(desired_goc_mode_D)
            
            # Create widget - Streamlit will use st.session_state[goc_mode_key_D] when key= is provided
            goc_mode = st.radio(
                "GOC definition mode",
                options=goc_mode_options_D,
                index=goc_mode_index_D,
                key=goc_mode_key_D,
                help="Choose how to define the Gas-Oil Contact depth"
            )
            
            # CRITICAL: After widget creation, ALWAYS use widget's value and save to persistent storage
            # The widget's value is the authoritative source after creation
            actual_goc_mode_D = st.session_state[goc_mode_key_D]
            if actual_goc_mode_D in goc_mode_options_D:
                # Valid value - save to persistent storage and use it
                st.session_state[persistent_storage_key][goc_mode_key_D] = actual_goc_mode_D
                goc_mode = actual_goc_mode_D
            else:
                # Invalid value - use what we had before
                goc_mode = desired_goc_mode_D
                st.session_state[goc_mode_key_D] = desired_goc_mode_D
            
            # CRITICAL: Use explicit if/elif/else to ensure only one section renders
            # Streamlit will rerun when radio button changes, so the correct section will render
            if goc_mode == "Direct depth":
                sGOC_depth_D = render_param(
                    "GOCDepthD",
                    "GOC depth",
                    "m",
                    defaults["d"]["dist"],
                    {"min": 2100, "mode": 2250, "max": 2350},
                    num_sims,
                    stats_decimals=1,
                )
                st.session_state["sGOC_depth"] = sGOC_depth_D
            elif goc_mode == "Oil fraction of HC column":
                # Replace slider with distribution selector (Stretched Beta default)
                sF_oil_D = render_param(
                    "F_oil_D",
                    "Oil fraction of total HC column",
                    "",
                    "Stretched Beta",
                    {"min": 0.05, "mode": 0.2, "max": 0.3},
                    num_sims,
                    stats_decimals=4,
                    help_text="Fraction of hydrocarbon column that is oil (remainder is gas). Distribution of values between 0 and 1."
                )
                # Store the sampled array for use in calculations
                st.session_state["da_oil_frac_D"] = sF_oil_D
            elif goc_mode == "Oil column height":
                h_oil_key_D = "da_h_oil_m_D"
                st.markdown("**Oil column height (m above HCWC)**")
                sH_oil_m_D = render_param(
                    "H_oil_m_D",
                    "Oil column height",
                    "m",
                    "PERT",
                    {"min": 10.0, "mode": 20.0, "max": 50.0},
                    num_sims,
                    stats_decimals=1,
                )
                st.session_state[h_oil_key_D] = sH_oil_m_D
            
            st.caption("ℹ️ Gas zone: Top Structure → GOC | Oil zone: GOC → Effective HC depth")
        
        # Display contacts-style plot using CO2 integrator pattern (Top + Contacts)
        if len(ed_top) > 0 and sHCDepthD is not None:
            depths_plot = np.asarray(ed_top["Depth"].values, dtype=float)
            top_areas_plot = np.asarray(ed_top["Top area (km2)"].values, dtype=float)

            from scopehc.geom_depth import compute_grv_top_plus_contacts, _cumulative_trapz as _cum_trapz

            top_df = pd.DataFrame({"Depth": depths_plot, "Top area (km2)": top_areas_plot})
            # CRITICAL: Use the correct step size and extrapolation settings for this method
            step_m = float(st.session_state.get("da_D_step_size", 10.0))
            extrapolate = bool(st.session_state.get("da_D_extrap", True))
            top_depth_m = float(np.min(depths_plot))
            hcwc_m = float(np.mean(sHCDepthD))

            case = "Oil"
            goc_depth = None
            if fluid_type == "Gas":
                case = "Gas"
                # For Gas-only: use HCWC (not GOC)
                # GOC is only needed for Oil+Gas to separate gas and oil zones
                goc_depth = None  # Not used for Gas-only case
            elif fluid_type == "Oil + Gas":
                case = "Oil+Gas"
                # CRITICAL: Derive GOC depth from the selected mode for plotting
                from scopehc.geom import derive_goc_from_mode
                goc_mode = st.session_state.get("da_goc_mode_D", "Oil fraction of HC column")
                
                # Create a temporary session state dict for derive_goc_from_mode
                temp_ss = {}
                if goc_mode == "Direct depth":
                    if sGOC_depth_D is not None:
                        goc_depth = float(np.mean(sGOC_depth_D))
                    else:
                        goc_depth = None
                elif goc_mode == "Oil fraction of HC column":
                    # Use mean oil fraction for plotting (now from distribution array)
                    f_oil_arr = st.session_state.get("da_oil_frac_D", None)
                    if f_oil_arr is not None and isinstance(f_oil_arr, np.ndarray) and len(f_oil_arr) > 0:
                        f_oil_mean = float(np.mean(f_oil_arr))
                    else:
                        f_oil_mean = 0.2  # Default from Stretched Beta mode
                    temp_ss["da_oil_frac"] = f_oil_mean
                    goc_depth = derive_goc_from_mode(top_depth_m, hcwc_m, goc_mode, temp_ss, trial_idx=None)
                elif goc_mode == "Oil column height":
                    # Use mean oil column height for plotting
                    if "da_h_oil_m_D" in st.session_state and st.session_state["da_h_oil_m_D"] is not None:
                        h_oil_mean = float(np.mean(st.session_state["da_h_oil_m_D"]))
                    else:
                        h_oil_mean = 20.0  # Default
                    temp_ss["da_h_oil_m"] = h_oil_mean
                    goc_depth = derive_goc_from_mode(top_depth_m, hcwc_m, goc_mode, temp_ss, trial_idx=None)
                else:
                    goc_depth = None

            res_contacts = compute_grv_top_plus_contacts(
                top_df=top_df,
                step_m=step_m,
                extrapolate=extrapolate,
                top_depth_m=top_depth_m,
                hcwc_m=hcwc_m,
                goc_m=goc_depth,
                case=case,
            )

            # CRITICAL: Check for errors first
            if "error" in res_contacts:
                st.warning(f"⚠️ Unable to generate depth-area plot: {res_contacts['error']}")
                # Skip plot generation but continue with rest of the function
            else:
                depths_interp = res_contacts.get("depths", np.array([]))
                top_interp = res_contacts.get("top_interp", np.array([]))
                
                # Guard: Check if arrays are empty before proceeding
                if len(depths_interp) == 0 or len(top_interp) == 0:
                    st.warning("⚠️ Unable to generate depth-area plot: empty depth or area arrays from integration.")
                    # Skip plot generation but continue with rest of the function
                else:
                    cum_vol = _cum_trapz(top_interp, step_m)

                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go
                    from scopehc.plots import color_for

                    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Top & Base Area vs Depth", "Volume vs Depth"))

                    # Top area
                    fig.add_trace(go.Scatter(x=top_interp, y=depths_interp, mode="lines",
                                             line=dict(color=color_for("GRV_oil_m3"), width=2),
                                             name="Top Area"),
                                  row=1, col=1)

                    # Calculate base area curves with uncertainty (P10, Mean, P90)
                    try:
                        mean_thickness = float(np.mean(sThickness)) if sThickness is not None and len(sThickness) > 0 else 0.0
                    except Exception:
                        mean_thickness = 0.0
                    
                    if mean_thickness > 0 and sThickness is not None and len(sThickness) > 0 and len(depths_interp) > 0:
                        # Sample thickness values for uncertainty bands
                        n_samples_uncert = min(1000, num_sims)
                        base_areas_uncert = []
                        base_depths_uncert = []
                        
                        for j in range(n_samples_uncert):
                            thickness_j = sThickness[j % len(sThickness)]
                            # Base curve: same area as top at depth z, but at depth z + h
                            base_depths_j = depths_interp + thickness_j
                            base_areas_j = top_interp.copy()  # Same area values
                            base_areas_uncert.append(base_areas_j)
                            base_depths_uncert.append(base_depths_j)
                        
                        # Calculate percentiles for base area at each depth point
                        # We need to interpolate to a common depth grid for percentile calculation
                        # Use mean depth grid for plotting
                        base_depths_mean = depths_interp + mean_thickness
                        base_mean = top_interp.copy()
                        
                        # For P10/P90, we need to interpolate each realization to the mean depth grid
                        base_areas_on_mean_grid = []
                        for j in range(n_samples_uncert):
                            # Guard: Check arrays are not empty before interpolation
                            if len(base_depths_uncert[j]) > 0 and len(base_areas_uncert[j]) > 0:
                                # Interpolate this realization's base areas to the mean depth grid
                                base_areas_interp = np.interp(base_depths_mean, base_depths_uncert[j], base_areas_uncert[j])
                                base_areas_on_mean_grid.append(base_areas_interp)
                        
                        # Only proceed if we have valid interpolated data
                        if len(base_areas_on_mean_grid) > 0:
                            base_areas_on_mean_grid = np.array(base_areas_on_mean_grid)
                            base_p10 = np.percentile(base_areas_on_mean_grid, 10, axis=0)
                            base_p90 = np.percentile(base_areas_on_mean_grid, 90, axis=0)
                            
                            # Plot base area curves (grayish-red for P10/P90)
                            from scopehc.plots import rgba_from_hex
                            base_color = color_for("GRV_gas_m3")
                            # Grayish-red color: mix of gray and red
                            grayish_red = "#C8A8A8"  # Light grayish-red
                            fig.add_trace(
                                go.Scatter(
                                    x=base_p10,
                                    y=base_depths_mean,
                                    mode="lines",
                                    line=dict(color=rgba_from_hex(grayish_red, 0.75), width=2, dash="dash"),
                                    name="Base Area P10",
                                ),
                                row=1,
                                col=1,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=base_mean,
                                    y=base_depths_mean,
                                    mode="lines",
                                    line=dict(color=base_color, width=2, dash="dash"),
                                    name="Base Area (Mean)",
                                ),
                                row=1,
                                col=1,
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=base_p90,
                                    y=base_depths_mean,
                                    mode="lines",
                                    line=dict(color=rgba_from_hex(grayish_red, 0.75), width=2, dash="dash"),
                                    name="Base Area P90",
                                ),
                                row=1,
                                col=1,
                            )

                    # Prepare volume curves (Top, Base mean, and GRV = Top - Base)
                    # Base area aligned to the same depth grid for integration (mean thickness)
                    base_on_grid_mean = np.interp(depths_interp - mean_thickness, depths_plot, top_areas_plot) if mean_thickness > 0 else np.zeros_like(top_interp)
                    vol_top = _cum_trapz(top_interp, step_m)
                    vol_base_mean = _cum_trapz(base_on_grid_mean, step_m)
                    vol_dgrv_mean = vol_top - vol_base_mean

                    # Right: volume curves
                    fig.add_trace(go.Scatter(x=vol_top, y=depths_interp, mode="lines",
                                             line=dict(color=color_for("GRV_total_m3"), width=2),
                                             name="Top Volume"),
                                  row=1, col=2)
                    fig.add_trace(go.Scatter(x=vol_base_mean, y=depths_interp, mode="lines",
                                             line=dict(color=color_for("GRV_gas_m3"), width=2, dash="dash"),
                                             name="Base Volume (Mean)"),
                                  row=1, col=2)
                    fig.add_trace(go.Scatter(x=vol_dgrv_mean, y=depths_interp, mode="lines",
                                             line=dict(color=color_for("GRV_oil_m3"), width=2),
                                             name="GRV = Top − Base"),
                                  row=1, col=2)

                    # Get mean spill point for display
                    mean_spill = None
                    if sSpillPointD is not None and len(sSpillPointD) > 0:
                        mean_spill = float(np.mean(sSpillPointD))
                    
                    # Horizontal lines for contacts - add to BOTH plots
                    def _hline(yval, name, color_key, x_max_area, x_max_vol):
                        # Add to area plot (col=1)
                        fig.add_shape(type="line", x0=0, x1=x_max_area,
                                      y0=yval, y1=yval, xref="x1", yref="y1",
                                      line=dict(color=color_for(color_key), width=1, dash="dot"))
                        fig.add_annotation(x=x_max_area * 0.98, y=yval, xref="x1", yref="y1",
                                           text=name, showarrow=False, font=dict(size=10, color=color_for(color_key)),
                                           xanchor="right", yanchor="middle")
                        # Add to volume plot (col=2)
                        fig.add_shape(type="line", x0=0, x1=x_max_vol,
                                      y0=yval, y1=yval, xref="x2", yref="y2",
                                      line=dict(color=color_for(color_key), width=1, dash="dot"))
                        fig.add_annotation(x=x_max_vol * 0.98, y=yval, xref="x2", yref="y2",
                                           text=name, showarrow=False, font=dict(size=10, color=color_for(color_key)),
                                           xanchor="right", yanchor="middle")

                    # Calculate max values for both plots
                    x_max_area = max(1e-9, float(np.nanmax(top_interp)) * 1.05) if len(top_interp) > 0 else 1.0
                    x_max_vol = max(1e-9, float(np.nanmax(vol_dgrv_mean)) * 1.05) if len(vol_dgrv_mean) > 0 else 1.0

                    # Add horizontal lines for Apex, Spill, HCWC, and GOC to BOTH plots
                    _hline(top_depth_m, "Apex", "GRV_total_m3", x_max_area, x_max_vol)
                    if mean_spill is not None:
                        _hline(mean_spill, "Spill", "SpillPoint", x_max_area, x_max_vol)
                    _hline(hcwc_m, "HCWC", "GRV_oil_m3", x_max_area, x_max_vol)
                    # GOC only for Oil+Gas case (not for Gas-only, which uses HCWC)
                    if goc_depth is not None and case == "Oil+Gas":
                        _hline(goc_depth, "GOC", "GRV_gas_m3", x_max_area, x_max_vol)
                    
                    # Spill point marker on volume plot
                    if mean_spill is not None:
                        # Find GRV volume at spill point
                        spill_idx = np.argmin(np.abs(depths_interp - mean_spill))
                        if spill_idx < len(vol_dgrv_mean):
                            grv_at_spill = vol_dgrv_mean[spill_idx]
                            if not np.isnan(grv_at_spill):
                                fig.add_trace(go.Scatter(x=[grv_at_spill], y=[mean_spill], mode="markers",
                                                         marker=dict(symbol="star", size=12, color=color_for("SpillPoint")),
                                                         name="Spill Point", showlegend=False), row=1, col=2)

                    # Ensure y-axis spans down to base depths (top + mean thickness), so curves are visually separated
                    y_min = float(np.min(depths_interp))
                    y_max = float(np.max(depths_interp) + max(mean_thickness, 0.0))
                    fig.update_yaxes(autorange=False, range=[y_max, y_min], title_text="Depth (m)")
                    fig.update_xaxes(title_text="Top Area (km²)", row=1, col=1)
                    fig.update_xaxes(title_text="Volume (km²·m)", row=1, col=2)
                    fig.update_layout(margin=dict(l=40, r=40, t=60, b=60),
                                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                                      width=None, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("Contacts integration using A_top(z): left shows top area; right shows cumulative ∫A(z)dz with Top/Spill/GOC/HCWC.")
        
        # Arrays to store split GRV and column heights
        sGRV_oil_m3 = np.zeros(num_sims)
        sGRV_gas_m3 = np.zeros(num_sims)
        sOil_Column_Height = np.zeros(num_sims)
        sGas_Column_Height = np.zeros(num_sims)
        
        # Pre-compute depth and area arrays (same for all trials)
        depths_m = np.asarray(ed_top["Depth"].values, dtype=float)
        top_areas_km2 = np.asarray(ed_top["Top area (km2)"].values, dtype=float)
        z_min = float(np.min(depths_m))
        step = float(step_size_d)
        
        # Pre-compute interpolator for top area (reused for all trials)
        from scipy.interpolate import interp1d
        f_top_interp_base = interp1d(
            depths_m, top_areas_km2, kind="linear", fill_value="extrapolate", bounds_error=False
        )
        
        # Track invalid trials for summary warning
        invalid_trials_D = []
        
        from scopehc.geom_depth import _cumulative_trapz as _cum_trapz
        from scopehc.geom import derive_goc_from_mode
        
        for i in range(num_sims):
            hc_depth = sHCDepthD[i]
            thickness_i = sThickness[i]
            
            # Build grid for this trial (depends on thickness)
            z_max = float(np.max(depths_m)) + float(thickness_i)
            grid = np.arange(z_min, z_max + step, step)

            # Top area on grid
            top_on_grid = f_top_interp_base(grid)
            # Base area on grid: A_base(z) = A_top(z - h)
            base_on_grid = f_top_interp_base(grid - float(thickness_i))

            # Cumulative volumes and dGRV per trial
            V_top = _cum_trapz(top_on_grid, step)
            V_base = _cum_trapz(base_on_grid, step)
            dgrv_grid = V_top - V_base  # km²·m
            
            # Top structure is minimum depth
            top_structure_m = z_min

            # Derive GOC based on fluid type and mode (same as before)
            from scopehc.geom import derive_goc_from_mode
            if fluid_type == "Oil":
                goc_depth = None
                owc_depth = hc_depth
            elif fluid_type == "Gas":
                goc_depth = hc_depth  # all gas
                owc_depth = None
            else:  # Oil + Gas
                goc_mode = st.session_state.get("da_goc_mode_D", "Oil fraction of HC column")
                if goc_mode == "Direct depth":
                    if sGOC_depth_D is not None and len(sGOC_depth_D) > i:
                        goc_depth = float(sGOC_depth_D[i])
                    else:
                        # Fallback: use mean of HC depth and top as default GOC
                        goc_depth = (top_structure_m + hc_depth) / 2.0
                        if i == 0:  # Warn once
                            st.warning("⚠️ GOC depth array not available for Direct depth mode. Using default (midpoint).")
                else:
                    # Map _D keys to expected keys for derive_goc_from_mode
                    temp_ss = dict(st.session_state)
                    if goc_mode == "Oil fraction of HC column":
                        # Map da_oil_frac_D to da_oil_frac
                        if "da_oil_frac_D" in temp_ss:
                            temp_ss["da_oil_frac"] = temp_ss["da_oil_frac_D"]
                    elif goc_mode == "Oil column height":
                        # Map da_h_oil_m_D to da_h_oil_m
                        if "da_h_oil_m_D" in temp_ss:
                            temp_ss["da_h_oil_m"] = temp_ss["da_h_oil_m_D"]
                    
                    try:
                        goc_depth = derive_goc_from_mode(top_structure_m, hc_depth, goc_mode, temp_ss, trial_idx=i)
                        if goc_depth is None or np.isnan(goc_depth):
                            # Fallback: use mean of HC depth and top as default GOC
                            goc_depth = (top_structure_m + hc_depth) / 2.0
                            if i == 0:  # Warn once
                                st.warning(f"⚠️ GOC derivation failed for mode '{goc_mode}'. Using default (midpoint).")
                    except Exception as e:
                        # Fallback: use mean of HC depth and top as default GOC
                        goc_depth = (top_structure_m + hc_depth) / 2.0
                        if i == 0:  # Warn once
                            st.warning(f"⚠️ Error deriving GOC depth: {e}. Using default (midpoint).")
                
                # Ensure goc_depth is valid
                if goc_depth is None or np.isnan(goc_depth):
                    goc_depth = (top_structure_m + hc_depth) / 2.0
                
                goc_depth = float(np.clip(goc_depth, top_structure_m, hc_depth))
                owc_depth = hc_depth
            
            # Validate: HCWC should be >= Top
            if hc_depth < top_structure_m:
                invalid_trials_D.append(i)
                sGRV_m3[i] = 0.0
                sGRV_oil_m3[i] = 0.0
                sGRV_gas_m3[i] = 0.0
                sOil_Column_Height[i] = 0.0
                sGas_Column_Height[i] = 0.0
                grv_values_km2m_D[i] = 0.0
                continue
            
            # Helper to sample cumulative dGRV at a given depth (clamped to grid)
            def _sample_dgrv_at(z: float) -> float:
                z = float(np.clip(z, grid[0], grid[-1]))
                return float(np.interp(z, grid, dgrv_grid))

            # Volumes by contacts (km²·m)
            if fluid_type == "Oil":
                grv_oil_km2m = _sample_dgrv_at(owc_depth) - _sample_dgrv_at(top_structure_m)
                grv_gas_km2m = 0.0
                H_oil = max(0.0, owc_depth - top_structure_m)
                H_gas = 0.0
            elif fluid_type == "Gas":
                if goc_depth is None:
                    grv_oil_km2m = 0.0
                    grv_gas_km2m = 0.0
                    H_oil = 0.0
                    H_gas = 0.0
                else:
                    grv_gas_km2m = _sample_dgrv_at(goc_depth) - _sample_dgrv_at(top_structure_m)
                    grv_oil_km2m = 0.0
                    H_gas = max(0.0, goc_depth - top_structure_m)
                    H_oil = 0.0
            else:  # Oil + Gas
                if goc_depth is None:
                    # CRITICAL: GOC is required for Oil+Gas - warn and set to default
                    if i == 0:  # Only warn once
                        st.warning(
                            "⚠️ **GOC depth is None for Oil+Gas case.** "
                            "This will result in zero gas GRV. "
                            "Please check GOC definition settings."
                        )
                    grv_oil_km2m = 0.0
                    grv_gas_km2m = 0.0
                    H_oil = 0.0
                    H_gas = 0.0
                else:
                    # Validate GOC is between top and HCWC
                    if goc_depth < top_structure_m:
                        goc_depth = top_structure_m
                    if goc_depth > hc_depth:
                        goc_depth = hc_depth
                    
                    grv_gas_km2m = _sample_dgrv_at(goc_depth) - _sample_dgrv_at(top_structure_m)
                    grv_oil_km2m = _sample_dgrv_at(owc_depth) - _sample_dgrv_at(goc_depth)
                    H_gas = max(0.0, goc_depth - top_structure_m)
                    H_oil = max(0.0, owc_depth - goc_depth)

            # Store (convert to m³)
            sGRV_gas_m3[i] = max(0.0, grv_gas_km2m) * 1e6
            sGRV_oil_m3[i] = max(0.0, grv_oil_km2m) * 1e6
            sGRV_m3[i] = sGRV_gas_m3[i] + sGRV_oil_m3[i]
            sOil_Column_Height[i] = H_oil
            sGas_Column_Height[i] = H_gas
        
        # Show summary warnings after loop completes
        if invalid_trials_D:
            st.warning(
                f"⚠️ **{len(invalid_trials_D)} trial(s) had invalid contacts** "
                f"(HCWC above Top structure). Volumes set to zero for these trials."
            )

        if sAreaMult_D is not None:
            grv_values_km2m_D = grv_values_km2m_D * sAreaMult_D
            sGRV_m3 = sGRV_m3 * sAreaMult_D
            sGRV_oil_m3 = sGRV_oil_m3 * sAreaMult_D
            sGRV_gas_m3 = sGRV_gas_m3 * sAreaMult_D
        
        # Store split GRV and column heights in session state
        st.session_state["sGRV_oil_m3"] = sGRV_oil_m3
        st.session_state["sGRV_gas_m3"] = sGRV_gas_m3
        st.session_state["sOil_Column_Height"] = sOil_Column_Height
        st.session_state["sGas_Column_Height"] = sGas_Column_Height
        
        # CRITICAL: Mark these arrays as depth-based calculated to prevent overwriting
        st.session_state["_grv_arrays_depth_based"] = True
        st.session_state["_grv_arrays_method"] = "Depth-based: Top + Res. thickness + Contact(s)"
        
        # Set final GRV for depth-based method
        st.session_state["sGRV_m3_final"] = sGRV_m3
        
        # Display split GRV results for depth-based method (top-only)
        st.markdown("---")
        st.markdown("### GRV Results by Fluid Type")
        fluid_type = st.session_state.get("fluid_type", "Oil")
        
        if fluid_type == "Oil":
            st.markdown("#### Oil GRV")
            st.plotly_chart(
                make_hist_cdf_figure(
                    sGRV_oil_m3 / 1e6,
                    "Oil GRV Distribution (×10^6 m³)",
                    "GRV (×10^6 m³)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(sGRV_oil_m3 / 1e6, decimals=2), use_container_width=True)
            st.caption("ℹ️ Oil only: entire hydrocarbon column is oil.")
            
        elif fluid_type == "Gas":
            st.markdown("#### Gas GRV")
            st.plotly_chart(
                make_hist_cdf_figure(
                    sGRV_gas_m3 / 1e6,
                    "Gas GRV Distribution (×10^6 m³)",
                    "GRV (×10^6 m³)",
                    "calculated"
                ),
                use_container_width=True,
            )
            st.dataframe(summary_table(sGRV_gas_m3 / 1e6, decimals=2), use_container_width=True)
            st.caption("ℹ️ Gas only: entire hydrocarbon column is gas.")
            
        else:  # Oil + Gas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### Oil GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_oil_m3 / 1e6,
                        "Oil GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_oil_m3 / 1e6, decimals=2), use_container_width=True)
            
            with col2:
                st.markdown("#### Gas GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_gas_m3 / 1e6,
                        "Gas GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_gas_m3 / 1e6, decimals=2), use_container_width=True)
            
            with col3:
                st.markdown("#### Total HC GRV")
                st.plotly_chart(
                    make_hist_cdf_figure(
                        sGRV_m3 / 1e6,
                        "Total HC GRV Distribution (×10^6 m³)",
                        "GRV (×10^6 m³)",
                        "calculated"
                    ),
                    use_container_width=True,
                )
                st.dataframe(summary_table(sGRV_m3 / 1e6, decimals=2), use_container_width=True)
            
            # Combined summary table for depth-based method
            st.markdown("#### Combined GRV Summary")
            summary_data = {
                "Parameter": ["GRV Oil (×10⁶ m³)", "GRV Gas (×10⁶ m³)", "GRV Total HC (×10⁶ m³)"],
            }
            
            # Get statistics for each
            oil_stats = summarize_array(sGRV_oil_m3 / 1e6)
            gas_stats = summarize_array(sGRV_gas_m3 / 1e6)
            total_stats = summarize_array(sGRV_m3 / 1e6)
            
            summary_data["Mean"] = [
                f"{oil_stats.get('mean', 0.0):.2f}",
                f"{gas_stats.get('mean', 0.0):.2f}",
                f"{total_stats.get('mean', 0.0):.2f}",
            ]
            summary_data["P10"] = [
                f"{oil_stats.get('P10', 0.0):.2f}",
                f"{gas_stats.get('P10', 0.0):.2f}",
                f"{total_stats.get('P10', 0.0):.2f}",
            ]
            summary_data["P50"] = [
                f"{oil_stats.get('P50', 0.0):.2f}",
                f"{gas_stats.get('P50', 0.0):.2f}",
                f"{total_stats.get('P50', 0.0):.2f}",
            ]
            summary_data["P90"] = [
                f"{oil_stats.get('P90', 0.0):.2f}",
                f"{gas_stats.get('P90', 0.0):.2f}",
                f"{total_stats.get('P90', 0.0):.2f}",
            ]
            summary_data["Std"] = [
                f"{oil_stats.get('std_dev', 0.0):.2f}",
                f"{gas_stats.get('std_dev', 0.0):.2f}",
                f"{total_stats.get('std_dev', 0.0):.2f}",
            ]
            summary_data["Min"] = [
                f"{oil_stats.get('min', 0.0):.2f}",
                f"{gas_stats.get('min', 0.0):.2f}",
                f"{total_stats.get('min', 0.0):.2f}",
            ]
            summary_data["Max"] = [
                f"{oil_stats.get('max', 0.0):.2f}",
                f"{gas_stats.get('max', 0.0):.2f}",
                f"{total_stats.get('max', 0.0):.2f}",
            ]
            
            df_combined = pd.DataFrame(summary_data)
            st.dataframe(df_combined, use_container_width=True, hide_index=True)
            
            # Show GOC mode info
            goc_mode = st.session_state.get("da_goc_mode_D", None)
            if goc_mode:
                mode_desc = {
                    "Direct depth": "GOC defined by direct depth distribution",
                    "Oil fraction of HC column": f"GOC derived from oil fraction: fₒᵢₗ = {_get_scalar_from_state(st.session_state, 'da_oil_frac_D', 0.2):.2f} (mean from distribution)",
                    "Oil column height": f"GOC derived from oil column height: {_get_scalar_from_state(st.session_state, 'da_h_oil_m_D', 0):.1f} m above HCWC (distribution)"
                }
                st.caption(f"ℹ️ {mode_desc.get(goc_mode, 'GOC split applied')}")
        
        # Edge case warnings
        if sGOC_depth_D is not None:
            # Check for reversed contacts
            reversed_count = np.sum((sGOC_depth_D > sHCDepthD) & (sHCDepthD > 0) & (sGOC_depth_D > 0))
            if reversed_count > 0:
                st.warning(f"⚠️ Contacts appear reversed in {reversed_count} trial(s); volumes truncated to zero where appropriate.")
        
        # Check if contacts are beyond depth range
        depth_min = float(np.min(ed_top["Depth"].values))
        depth_max = float(np.max(ed_top["Depth"].values))
        if np.any(sHCDepthD < depth_min) or np.any(sHCDepthD > depth_max):
            st.caption("ℹ️ Some contacts were clipped to the available depth range of the area-depth table.")
        
        if fluid_type != "Oil + Gas":
            if fluid_type == "Oil":
                st.caption("ℹ️ Oil only: entire hydrocarbon column treated as oil to OWC.")
            elif fluid_type == "Gas":
                st.caption("ℹ️ Gas only: entire hydrocarbon column treated as gas to HCWC.")

        st.markdown("#### GRV Distribution Above Spill Point")
        st.plotly_chart(
            make_hist_cdf_figure(
                sGRV_m3 / 1e6, "GRV distribution at Spill Point (×10^6 m³)", "GRV (×10^6 m³)", "calculated"
            ),
            use_container_width=True,
        )

    else:
        st.error("Unsupported GRV option selected.")
        return

    if grv_option in {"Direct GRV", "Area × Thickness × GCF"}:
        grv_mp_enabled = st.checkbox(
            "Apply GRV uncertainty multiplier", value=False, help="Enable to apply GRV multiplier."
        )
        if grv_mp_enabled:
            grv_mp_type = st.radio(
                "GRV_MP definition",
                ["Constant value", "Probability distribution"],
                horizontal=True,
                help="Choose how to define the GRV uncertainty multiplier",
            )
            if grv_mp_type == "Constant value":
                grv_mp_constant = st.number_input(
                    "GRV_MP constant value",
                    value=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    step=0.01,
                    help="Constant multiplier for GRV value",
                )
                grv_mp_samples = np.full(num_sims, grv_mp_constant)
            else:
                grv_mp_samples = render_param(
                    "GRV_MP",
                    "GRV Uncertainty Multiplier",
                    "multiplier",
                    "PERT",
                    {"min": 0.85, "mode": 1.00, "max": 1.20},
                    num_sims,
                    plot_unit_label="multiplier",
                    stats_decimals=3,
                )
        else:
            grv_mp_samples = np.full(num_sims, 1.0)
            st.info("GRV_MP disabled - using GRV value as calculated (multiplier = 1.0)", icon="ℹ️")
    else:
        grv_mp_samples = np.full(num_sims, 1.0)

    sGRV_m3_final = sGRV_m3 * grv_mp_samples
    st.session_state["sGRV_m3"] = sGRV_m3
    st.session_state["sGRV_m3_final"] = sGRV_m3_final
    st.session_state["grv_mp_samples"] = grv_mp_samples
    
    # Apply multiplier to split GRV arrays if they exist
    prefix = "direct" if grv_option == "Direct GRV" else "atgcf"
    if f"{prefix}_GRV_oil_m3" in st.session_state:
        st.session_state[f"{prefix}_GRV_oil_m3"] = st.session_state[f"{prefix}_GRV_oil_m3"] * grv_mp_samples
        st.session_state["sGRV_oil_m3"] = st.session_state[f"{prefix}_GRV_oil_m3"]
    if f"{prefix}_GRV_gas_m3" in st.session_state:
        st.session_state[f"{prefix}_GRV_gas_m3"] = st.session_state[f"{prefix}_GRV_gas_m3"] * grv_mp_samples
        st.session_state["sGRV_gas_m3"] = st.session_state[f"{prefix}_GRV_gas_m3"]
    if sA is not None:
        st.session_state["sA"] = sA
    if sGCF is not None:
        st.session_state["sGCF"] = sGCF
    if sh is not None:
        st.session_state["sh"] = sh

    depth_based_methods = {
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    }
    if grv_option not in depth_based_methods:
        st.markdown("---")
        st.markdown("### Final GRV for Calculations")
        grv_base_value = float(sGRV_m3[0] / 1_000_000.0)
        grv_mp_mean = float(np.mean(grv_mp_samples))
        grv_final_mean = float(np.mean(sGRV_m3_final) / 1_000_000.0)
        col1, col2, col3 = st.columns(3)
        col1.metric("Base GRV (×10^6 m³)", f"{grv_base_value:.3f}")
        col2.metric("GRV_MP (mean)", f"{grv_mp_mean:.3f}")
        col3.metric("Final GRV (×10^6 m³)", f"{grv_final_mean:.3f}")
        st.markdown(
            f"""
            <div class='card-container'>
                <h3 style='color:{PALETTE["secondary"]};margin-top:0;'>Final GRV Distribution Summary</h3>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_hist_cdf_figure(
                sGRV_m3_final / 1e6,
                "Final GRV distribution (after GRV_MP)",
                "GRV (×10^6 m³)",
                "calculated",
            ),
            use_container_width=True,
        )
        st.dataframe(summary_table(sGRV_m3_final / 1e6, decimals=2), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # CRITICAL: Always ensure sGRV_oil_m3 and sGRV_gas_m3 are set correctly based on CURRENT fluid_type
    # This ensures consistency regardless of which GRV method was used
    _ensure_final_grv_arrays(grv_option, num_sims)
    
    # Display column heights at the bottom (only for depth-based methods)
    depth_based_methods = {
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    }
    if grv_option in depth_based_methods:
        _render_column_heights()

    st.write("")

