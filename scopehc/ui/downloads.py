from __future__ import annotations

import pandas as pd
import streamlit as st

from .common import (
    collect_all_trial_data,
    get_unit_system,
    init_theme,
    summary_table,
    render_custom_navigation,
    render_sidebar_settings,
    render_sidebar_help,
)
from scopehc.export import export_to_csv, export_to_excel


def _get_results_frame(trial_data: dict[str, list]) -> pd.DataFrame:
    df_results = st.session_state.get("df_results")
    if df_results is not None:
        if isinstance(df_results, pd.DataFrame):
            return df_results
    df_results = pd.DataFrame(trial_data)
    st.session_state["df_results"] = df_results
    return df_results


def render() -> None:
    """Render the downloads/export page."""
    st.session_state["current_page"] = "_pages_disabled/06_Downloads.py"

    st.header("Downloads & Exports")
    st.caption("Generate CSV/Excel extracts containing every trial and summary statistics.")

    trial_data = st.session_state.get("trial_data")
    if trial_data is None:
        trial_data = collect_all_trial_data()

    if not trial_data:
        st.info("Run the simulation on the **Run Simulation** page to populate results before exporting.")
        return

    df_results = _get_results_frame(trial_data)
    num_trials = len(df_results)
    unit_system = get_unit_system()
    gas_scf_per_boe = st.session_state.get("gas_scf_per_boe", 6000.0)

    st.subheader("Results Preview")
    st.caption(f"Showing first 50 of {num_trials:,} trials.")
    st.dataframe(df_results.head(50), use_container_width=True, hide_index=True)

    thr_column = None
    for candidate in [
        "Total_surface_BOE",
        "THR_BOE",
        "THR_MBOE",
        "Total_Hydrocarbon_Resource_BOE",
    ]:
        if candidate in df_results.columns:
            thr_column = candidate
            break

    if thr_column:
        st.subheader(f"THR Summary ({thr_column})")
        st.dataframe(
            summary_table(df_results[thr_column].to_numpy(), decimals=2),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("THR column not found in results; preview limited to raw trials.")

    sim_data = {
        "df_results": df_results,
        "seed": st.session_state.get("random_seed", "N/A"),
        "n_trials": num_trials,
        "scf_per_BOE": gas_scf_per_boe,
        "unit_system": unit_system,
    }

    st.markdown("---")
    st.subheader("Export Options")
    st.caption(
        "Download raw trial data for further analysis or archiving. Excel exports include results, summary statistics, and metadata sheets."
    )

    csv_payload = export_to_csv(sim_data)
    excel_payload = export_to_excel(sim_data)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            data=csv_payload if csv_payload is not None else "",
            file_name=f"scopehc_results_{num_trials}_trials.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=csv_payload is None,
        )
    with col2:
        st.download_button(
            "📥 Download Excel",
            data=excel_payload if excel_payload is not None else b"",
            file_name=f"scopehc_results_{num_trials}_trials.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            disabled=excel_payload is None,
        )
        if excel_payload is None:
            st.info("Install `openpyxl` to enable Excel export (`pip install openpyxl`).")

