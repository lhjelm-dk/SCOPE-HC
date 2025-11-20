"""
Input Quality Control (QC) Display Module

Shows all input parameters and their values BEFORE running simulation.
This helps users verify that their inputs are correct and will be used in calculations.
"""
import numpy as np
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple
from .common import summarize_array, get_unit_system


def render_input_qc_panel() -> Dict[str, Any]:
    """
    Render a comprehensive QC panel showing all input parameters that will be used in simulation.
    
    Returns:
        Dictionary with QC status and input summary
    """
    st.markdown("---")
    st.markdown("### 🔍 Input Quality Control (QC)")
    st.caption(
        "Review all input parameters and their values that will be used in the simulation. "
        "Verify that these match your intended inputs before running."
    )
    
    # Get current settings
    fluid_type = st.session_state.get("fluid_type", "Oil")
    grv_option = st.session_state.get("grv_option", "Direct GRV")
    num_sims = st.session_state.get("num_sims", 10_000)
    unit_system = get_unit_system()
    
    qc_status = {
        "all_required_present": True,
        "warnings": [],
        "errors": [],
        "input_summary": {}
    }
    
    # Check required arrays
    required_keys = ["sGRV_m3_final", "sNtG", "sp", "sRF_oil", "sRF_gas", "sBg", "sInvBo", "sGOR"]
    missing_keys = [key for key in required_keys if key not in st.session_state]
    
    if missing_keys:
        qc_status["all_required_present"] = False
        qc_status["errors"].append(f"Missing required inputs: {', '.join(missing_keys)}")
        st.error(f"❌ **Missing Required Inputs:** {', '.join(missing_keys)}")
        st.info("💡 Please complete the Inputs page first.")
        return qc_status
    
    # Get array lengths
    num_trials = len(st.session_state["sGRV_m3_final"])
    if num_trials != num_sims:
        qc_status["warnings"].append(
            f"Array size mismatch: Expected {num_sims:,} trials, but arrays have {num_trials:,} values. "
            "This may cause errors."
        )
    
    # Display input summary in expandable sections
    with st.expander("Core Parameters", expanded=True):
        _render_parameter_table("Core Parameters", [
            ("GRV Total", "sGRV_m3_final", "m³", 1e-6, "×10⁶ m³"),
            ("NtG", "sNtG", "fraction", 1.0, ""),
            ("Porosity", "sp", "fraction", 1.0, ""),
            ("RF Oil", "sRF_oil", "fraction", 1.0, ""),
            ("RF Gas", "sRF_gas", "fraction", 1.0, ""),
        ])
    
    # GRV Split by Fluid Type
    with st.expander("GRV by Fluid Type", expanded=True):
        st.markdown(f"**Selected Fluid Type:** `{fluid_type}`")
        st.markdown(f"**GRV Method:** `{grv_option}`")
        
        if fluid_type == "Oil":
            if "sGRV_oil_m3" in st.session_state:
                arr_oil = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
                if len(arr_oil) > 0:
                    _render_parameter_table("Oil GRV", [
                        ("GRV Oil", "sGRV_oil_m3", "m³", 1e-6, "×10⁶ m³"),
                    ])
                else:
                    st.warning("⚠️ Oil GRV array found but is empty. Will use total GRV as oil.")
            else:
                st.warning("⚠️ Oil GRV array not found. Will use total GRV as oil.")
        elif fluid_type == "Gas":
            if "sGRV_gas_m3" in st.session_state:
                arr_gas = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
                if len(arr_gas) > 0:
                    _render_parameter_table("Gas GRV", [
                        ("GRV Gas", "sGRV_gas_m3", "m³", 1e-6, "×10⁶ m³"),
                    ])
                else:
                    st.warning("⚠️ Gas GRV array found but is empty. Will use total GRV as gas.")
            else:
                st.warning("⚠️ Gas GRV array not found. Will use total GRV as gas.")
        else:  # Oil + Gas
            has_oil_grv = "sGRV_oil_m3" in st.session_state
            has_gas_grv = "sGRV_gas_m3" in st.session_state
            
            if has_oil_grv and has_gas_grv:
                # Check if arrays are valid (not empty, not all zeros)
                arr_oil = np.atleast_1d(np.asarray(st.session_state["sGRV_oil_m3"], dtype=float))
                arr_gas = np.atleast_1d(np.asarray(st.session_state["sGRV_gas_m3"], dtype=float))
                
                if len(arr_oil) > 0 and len(arr_gas) > 0:
                    _render_parameter_table("Oil + Gas GRV", [
                        ("GRV Oil", "sGRV_oil_m3", "m³", 1e-6, "×10⁶ m³"),
                        ("GRV Gas", "sGRV_gas_m3", "m³", 1e-6, "×10⁶ m³"),
                    ])
                else:
                    st.warning(f"⚠️ GRV arrays found but are empty. Oil: {len(arr_oil)}, Gas: {len(arr_gas)}")
            else:
                missing = []
                if not has_oil_grv:
                    missing.append("sGRV_oil_m3")
                if not has_gas_grv:
                    missing.append("sGRV_gas_m3")
                st.warning(f"⚠️ GRV arrays not found: {', '.join(missing)}. Will split total GRV using f_oil.")
                if "f_oil" in st.session_state:
                    f_oil_val = st.session_state["f_oil"]
                    if isinstance(f_oil_val, np.ndarray):
                        f_oil_val = float(np.mean(f_oil_val))
                    st.info(f"Using f_oil = {f_oil_val:.3f} to split GRV")
    
    # Fluid Properties
    with st.expander("Fluid Properties", expanded=False):
        _render_parameter_table("Fluid Properties", [
            ("Bg", "sBg", "rb/scf", 1.0, ""),
            ("1/Bo", "sInvBo", "STB/rb", 1.0, ""),
            ("GOR", "sGOR", "scf/STB", 1.0, ""),
        ])
        if "sCY" in st.session_state:
            _render_parameter_table("Condensate", [
                ("CY", "sCY", "STB/MMscf", 1.0, ""),
            ])
        if "sRF_cond" in st.session_state:
            _render_parameter_table("Condensate RF", [
                ("RF Condensate", "sRF_cond", "fraction", 1.0, ""),
            ])
    
    # Saturation
    with st.expander("Saturation", expanded=False):
        sat_mode = st.session_state.get("sat_mode", "Global")
        st.markdown(f"**Saturation Mode:** `{sat_mode}`")
        
        if sat_mode == "Global":
            if "Shc_global" in st.session_state:
                _render_parameter_table("Global Saturation", [
                    ("Shc Global", "Shc_global", "fraction", 1.0, ""),
                ])
            elif "Sw_global" in st.session_state:
                _render_parameter_table("Global Water Saturation", [
                    ("Sw Global", "Sw_global", "fraction", 1.0, ""),
                ])
        elif sat_mode == "Water saturation Per zone":
            sat_params = []
            if "Sw_oilzone" in st.session_state:
                sat_params.append(("Sw Oil Zone", "Sw_oilzone", "fraction", 1.0, ""))
            if "Sw_gaszone" in st.session_state:
                sat_params.append(("Sw Gas Zone", "Sw_gaszone", "fraction", 1.0, ""))
            if sat_params:
                _render_parameter_table("Per Zone Saturation", sat_params)
        elif sat_mode == "Per phase":
            sat_params = []
            if "Shc_oil_input" in st.session_state:
                sat_params.append(("Shc Oil", "Shc_oil_input", "fraction", 1.0, ""))
            if "Shc_gas_input" in st.session_state:
                sat_params.append(("Shc Gas", "Shc_gas_input", "fraction", 1.0, ""))
            if sat_params:
                _render_parameter_table("Per Phase Saturation", sat_params)
    
    # Simulation Settings
    with st.expander("Simulation Settings", expanded=False):
        settings_data = {
            "Number of Trials": f"{num_sims:,}",
            "Fluid Type": fluid_type,
            "GRV Method": grv_option,
            "Unit System": unit_system,
            "Random Seed": str(st.session_state.get("random_seed", 42)),
            "Percentile Convention": "P10 high (exceedance)" if st.session_state.get("percentile_exceedance", True) else "P10 low (non-exceedance)",
        }
        st.dataframe(pd.DataFrame(list(settings_data.items()), columns=["Setting", "Value"]), use_container_width=True)
    
    # Warnings and Errors
    if qc_status["warnings"]:
        st.warning("⚠️ **Warnings:**\n" + "\n".join(f"• {w}" for w in qc_status["warnings"]))
    if qc_status["errors"]:
        st.error("❌ **Errors:**\n" + "\n".join(f"• {e}" for e in qc_status["errors"]))
    
    if qc_status["all_required_present"] and not qc_status["errors"]:
        st.success("✅ All required inputs are present and ready for simulation.")
    
    return qc_status


def _render_parameter_table(title: str, params: List[Tuple[str, str, str, float, str]]):
    """Render a table showing parameter statistics.
    
    Args:
        title: Table title
        params: List of tuples (param_name, key, unit, scale, display_unit)
            - scale: Multiplier to convert from stored unit to display unit
            - For GRV: stored in m³, scale=1e-6 converts to ×10⁶ m³ (multiply by 1e-6 = divide by 1e6)
    """
    rows = []
    for param_name, key, unit, scale, display_unit in params:
        if key in st.session_state:
            arr = np.atleast_1d(np.asarray(st.session_state[key], dtype=float))
            if len(arr) > 0:
                # Apply scale: arr * scale converts from stored unit to display unit
                # For GRV: arr (m³) * 1e-6 = arr / 1e6 = value in ×10⁶ m³
                scaled_arr = arr * scale
                stats = summarize_array(scaled_arr)
                rows.append({
                    "Parameter": param_name,
                    "Unit": display_unit or unit,
                    "Mean": f"{stats.get('mean', 0):.4f}",
                    "P10": f"{stats.get('P10', 0):.4f}",
                    "P50": f"{stats.get('P50', 0):.4f}",
                    "P90": f"{stats.get('P90', 0):.4f}",
                    "Std": f"{stats.get('std_dev', 0):.4f}",
                    "Min": f"{stats.get('min', 0):.4f}",
                    "Max": f"{stats.get('max', 0):.4f}",
                    "Size": f"{len(arr):,}",
                })
    
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No data available for {title}")

