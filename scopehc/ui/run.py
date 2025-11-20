from __future__ import annotations

import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scopehc.sampling import rng_from_seed, sample_scalar_dist
from scopehc.compute import run_simulation
from scopehc.ui.common import collect_all_inputs_from_session, compute_input_hash, render_run_controls_and_diagnostics

from .common import (
    DEFAULTS,
    collect_all_trial_data,
    init_theme,
    update_progress,
    render_custom_navigation,
    render_sidebar_settings,
    render_sidebar_help,
)


REQUIRED_KEYS = [
    "sGRV_m3_final",
    "sNtG",
    "sp",
    "sRF_oil",
    "sRF_gas",
    "sBg",
    "sInvBo",
    "sGOR",
]

FALLBACK_PARAM_CONFIGS: Dict[str, Tuple[str, Dict[str, float]]] = {
    "sNtG": ("PERT", {"min": 0.27, "mode": 0.40, "max": 0.50}),
    "sp": ("PERT", {"min": 0.08, "mode": 0.18, "max": 0.21}),
    "sRF_oil": ("Triangular", {"min": 0.40, "mode": 0.55, "max": 0.70}),
    "sRF_gas": ("Triangular", {"min": 0.60, "mode": 0.75, "max": 0.90}),
    "sBg": ("Triangular", {"min": 0.003, "mode": 0.005, "max": 0.008}),
    "sInvBo": ("Triangular", {"min": 0.75, "mode": 0.80, "max": 0.85}),
    "sGOR": ("Triangular", {"min": 400.0, "mode": 500.0, "max": 600.0}),
}

GRV_FALLBACK_CONFIG = {
    "Area_km2": ("PERT", {"min": 105.0, "mode": 210.0, "max": 273.0}),
    "Thickness_m": ("PERT", {"min": 100.0, "mode": 130.0, "max": 150.0}),
    "GCF": ("PERT", {"min": 0.60, "mode": 0.68, "max": 0.85}),
}
FALLBACK_LABELS = {
    "sGRV_m3_final": "Gross Rock Volume",
    "sNtG": "Net-to-Gross",
    "sp": "Porosity",
    "sRF_oil": "Oil Recovery Factor",
    "sRF_gas": "Gas Recovery Factor",
    "sBg": "Bg",
    "sInvBo": "1/Bo",
    "sGOR": "GOR",
}


def _simulation_summary(trial_count: int) -> None:
    """Show a lightweight summary after the simulation finishes."""
    st.markdown("### Simulation Summary")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Trials", f"{trial_count:,}")
    with cols[1]:
        st.metric("Seed", str(st.session_state.get("random_seed", "Not set")))
    with cols[2]:
        st.metric(
            "Unit System",
            st.session_state.get("unit_system", DEFAULTS.get("unit_system", "oilfield")),
        )


def _ensure_required_inputs(num_sims: int) -> None:
    """Populate required session state keys with default distributions when missing."""
    missing_keys = [key for key in REQUIRED_KEYS if key not in st.session_state]
    if not missing_keys:
        return

    seed = st.session_state.get("random_seed", 42)
    rng = rng_from_seed(seed)
    generated_labels = []

    if "sGRV_m3_final" in missing_keys:
        area_km2 = sample_scalar_dist(rng, *GRV_FALLBACK_CONFIG["Area_km2"], num_sims)
        thickness_m = sample_scalar_dist(
            rng, *GRV_FALLBACK_CONFIG["Thickness_m"], num_sims
        )
        gcf = sample_scalar_dist(rng, *GRV_FALLBACK_CONFIG["GCF"], num_sims)

        area_m2 = np.asarray(area_km2, dtype=float) * 1_000_000.0
        thickness_m = np.asarray(thickness_m, dtype=float)
        gcf = np.asarray(gcf, dtype=float)

        grv = area_m2 * thickness_m * gcf
        st.session_state["sGRV_m3_final"] = grv
        st.session_state.setdefault("sA", area_km2)
        st.session_state.setdefault("sGCF", gcf)
        st.session_state.setdefault("sh", thickness_m)

        # Assume equal oil/gas split unless overridden later
        # f_oil should be a scalar (single float), not an array
        st.session_state.setdefault("f_oil", 0.5)
        st.session_state.setdefault("f_gas", 0.5)
        st.session_state.setdefault("sGRV_oil_m3", grv * 0.5)
        st.session_state.setdefault("sGRV_gas_m3", grv * 0.5)
        generated_labels.append(FALLBACK_LABELS["sGRV_m3_final"])

    for key, (dist_name, params) in FALLBACK_PARAM_CONFIGS.items():
        if key in missing_keys:
            samples = sample_scalar_dist(rng, dist_name, params, num_sims)
            st.session_state[key] = np.asarray(samples, dtype=float)
            generated_labels.append(FALLBACK_LABELS.get(key, key))

    # Derived state based on generated parameters
    if "sRF_oil" in st.session_state and "sRF_assoc_gas" not in st.session_state:
        st.session_state["sRF_assoc_gas"] = np.asarray(
            st.session_state["sRF_oil"], dtype=float
        )
    if "sRF_gas" in st.session_state and "sRF_cond" not in st.session_state:
        st.session_state["sRF_cond"] = np.full(num_sims, 0.6)

    if generated_labels:
        st.warning(
            "Using default distributions for missing inputs: "
            + ", ".join(sorted(set(generated_labels)))
            + ". Visit the Inputs page to customise these values."
        )


def render() -> None:
    """Render the simulation runner page using centralized run system."""
    st.session_state["current_page"] = "_pages_disabled/03_Run_Simulation.py"

    st.header("Run Simulation")
    st.markdown(
        "Execute the Monte Carlo workflow after configuring inputs. "
        "The results page relies on the samples generated here."
    )

    # Check if phase-specific GRV volumes are required and available
    fluid_type = st.session_state.get("fluid_type", "Oil")
    grv_option = st.session_state.get("grv_option", "Direct GRV")
    sGRV_oil_m3 = st.session_state.get("sGRV_oil_m3", None)
    sGRV_gas_m3 = st.session_state.get("sGRV_gas_m3", None)
    
    # Check if this is a depth-based method
    depth_based_methods = {
        "Depth-based: Top and Base res. + Contact(s)",
        "Depth-based: Top + Res. thickness + Contact(s)",
    }
    is_depth_based = grv_option in depth_based_methods
    
    # Show warning if phase-specific volumes are required but not available
    if fluid_type == "Oil + Gas" and is_depth_based:
        if sGRV_oil_m3 is None or sGRV_gas_m3 is None:
            st.warning(
                "⚠️ **Phase-Specific GRV Volumes Required:** The Monte Carlo simulation requires phase-specific "
                "Gross Rock Volumes (GRV oil and GRV gas) to be calculated first. These values are derived from "
                "your depth-based GRV inputs and must be computed before running the simulation. Please return to "
                "the GRV input page, ensure the calculation completes, and verify that the GRV results by fluid "
                "type are displayed before proceeding."
            )
        else:
            # Check if arrays are valid (not empty and have correct length)
            sGRV_oil_arr = np.atleast_1d(np.asarray(sGRV_oil_m3, dtype=float))
            sGRV_gas_arr = np.atleast_1d(np.asarray(sGRV_gas_m3, dtype=float))
            num_sims = st.session_state.get("num_sims", DEFAULTS.get("num_sims", 10_000))
            if len(sGRV_oil_arr) == 0 or len(sGRV_gas_arr) == 0 or len(sGRV_oil_arr) != num_sims or len(sGRV_gas_arr) != num_sims:
                st.warning(
                    "⚠️ **Phase-Specific GRV Volumes Required:** The Monte Carlo simulation requires phase-specific "
                    "Gross Rock Volumes (GRV oil and GRV gas) to be calculated first. These values are derived from "
                    "your depth-based GRV inputs and must be computed before running the simulation. Please return to "
                    "the GRV input page, ensure the calculation completes, and verify that the GRV results by fluid "
                    "type are displayed before proceeding."
                )

    num_sims = st.session_state.get("num_sims", DEFAULTS.get("num_sims", 10_000))
    st.info(
        f"Configured to run **{num_sims:,}** trials. You can change this under Monte Carlo "
        "Settings in the sidebar."
    )

    # Ensure required inputs exist (generate defaults if missing)
    _ensure_required_inputs(num_sims)
    
    # Collect all inputs into a dictionary
    inputs = collect_all_inputs_from_session()
    inputs["_input_hash"] = compute_input_hash(inputs)
    
    # Render Input QC Panel BEFORE run controls
    from .input_qc import render_input_qc_panel
    qc_status = render_input_qc_panel()
    
    # Render run controls and diagnostics
    render_run_controls_and_diagnostics(inputs)
    
    # Get run tracking info
    run_id = int(st.session_state.get("run_id", 0))
    seed = st.session_state.get("random_seed", st.session_state.get("rng_seed", None))
    nonce = int(st.session_state.get("no_cache_nonce", 0))
    
    # Check if we need to run simulation
    results_run_id = st.session_state.get("results_run_id", -1)
    results_input_hash = st.session_state.get("results_input_hash", "")
    current_hash = inputs["_input_hash"]
    
    # Run simulation if:
    # 1. Run button was clicked (run_id changed) OR
    # 2. Inputs changed (hash mismatch) OR
    # 3. No previous results exist
    should_run = (
        results_run_id != run_id or 
        results_input_hash != current_hash or
        "results_cache" not in st.session_state
    )
    
    if should_run and run_id > 0:  # Only run if button was actually clicked
        progress = st.progress(0.0)
        status = st.empty()
        
        try:
            update_progress(progress, status, 1, 3, "Running simulation with current inputs...")
            
            # Run centralized simulation
            results = run_simulation(inputs, run_id, seed, nonce)
            
            # Collect trial data for export/display
            trial_data = collect_all_trial_data()
            if trial_data:
                st.session_state["trial_data"] = trial_data
                st.session_state["df_results"] = pd.DataFrame(trial_data)
            
            update_progress(progress, status, 3, 3, "Simulation complete")
            st.success("✅ Simulation finished successfully with current inputs.")
            st.session_state["last_simulation_run"] = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            
            _simulation_summary(len(trial_data.get("Trial", [])) if trial_data else 0)
            
            # Display input parameters summary table
            st.markdown("---")
            st.markdown("### Input Parameters Summary")
            st.caption("Summary statistics for all input parameters used in this simulation run.")
            
            from scopehc.ui.common import summarize_array
            
            # Collect all input parameter arrays
            # Note: GRV arrays are in m³, will be converted to ×10⁶ m³ for display
            grv_total = st.session_state.get("sGRV_m3_final", np.array([]))
            grv_oil = st.session_state.get("sGRV_oil_m3", np.array([]))
            grv_gas = st.session_state.get("sGRV_gas_m3", np.array([]))
            
            input_params = {
                "GRV Total (×10⁶ m³)": grv_total / 1e6 if len(grv_total) > 0 else np.array([]),
                "GRV Oil (×10⁶ m³)": grv_oil / 1e6 if len(grv_oil) > 0 else np.array([]),
                "GRV Gas (×10⁶ m³)": grv_gas / 1e6 if len(grv_gas) > 0 else np.array([]),
                "NtG": st.session_state.get("sNtG", np.array([])),
                "Porosity": st.session_state.get("sp", np.array([])),
                "RF Oil": st.session_state.get("sRF_oil", np.array([])),
                "RF Gas": st.session_state.get("sRF_gas", np.array([])),
                "Bg (rb/scf)": st.session_state.get("sBg", np.array([])),
                "1/Bo (STB/rb)": st.session_state.get("sInvBo", np.array([])),
                "GOR (scf/STB)": st.session_state.get("sGOR", np.array([])),
            }
            
            # Add optional parameters if they exist
            if "sCY" in st.session_state and len(st.session_state["sCY"]) > 0:
                input_params["CY (STB/MMscf)"] = st.session_state["sCY"]
            if "sRF_cond" in st.session_state and len(st.session_state["sRF_cond"]) > 0:
                input_params["RF Condensate"] = st.session_state["sRF_cond"]
            if "sA" in st.session_state and len(st.session_state["sA"]) > 0:
                input_params["Area (km²)"] = st.session_state["sA"]
            if "sh" in st.session_state and len(st.session_state["sh"]) > 0:
                input_params["Thickness (m)"] = st.session_state["sh"]
            if "sGCF" in st.session_state and len(st.session_state["sGCF"]) > 0:
                input_params["GCF"] = st.session_state["sGCF"]
            
            # Add saturation parameters if they exist
            if "Shc_oil" in st.session_state and len(st.session_state["Shc_oil"]) > 0:
                input_params["S_hc Oil"] = st.session_state["Shc_oil"]
            if "Shc_gas" in st.session_state and len(st.session_state["Shc_gas"]) > 0:
                input_params["S_hc Gas"] = st.session_state["Shc_gas"]
            
            # Build summary table
            summary_rows = []
            for param_name, param_array in input_params.items():
                if len(param_array) > 0:
                    arr = np.asarray(param_array, dtype=float)
                    stats = summarize_array(arr)
                    # Handle empty stats (empty array case)
                    if not stats:
                        continue
                    summary_rows.append({
                        "Parameter": param_name,
                        "Mean": f"{stats.get('mean', 0.0):.4f}",
                        "P10": f"{stats.get('P10', 0.0):.4f}",
                        "P50": f"{stats.get('P50', 0.0):.4f}",
                        "P90": f"{stats.get('P90', 0.0):.4f}",
                        "Std": f"{stats.get('std_dev', 0.0):.4f}",
                        "Min": f"{stats.get('min', 0.0):.4f}",
                        "Max": f"{stats.get('max', 0.0):.4f}",
                    })
            
            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
            else:
                st.info("No input parameters available for summary.")
            
            st.caption(
                "You can now open the Results page to review distributions, percentiles, "
                "and export options."
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            if progress is not None:
                progress.progress(1.0)
    elif results_run_id == run_id and results_input_hash == current_hash:
        # Show status if results are current
        st.info("✅ Results are up-to-date with current inputs. Click 'Run Monte Carlo Simulation' to recompute.")
        if "last_simulation_run" in st.session_state:
            st.caption(f"Last run completed at {st.session_state['last_simulation_run']}.")
    else:
        # Initial state - no run yet
        st.info("Configure inputs and click 'Run Monte Carlo Simulation' to start.")

