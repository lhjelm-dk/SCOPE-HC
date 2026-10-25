"""
Export: CSV/XLSX writers
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scopehc.utils import summarize_array


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = np.asarray(df[col]).ravel()
    return df


def render_export_buttons(sim_data: Dict[str, Any]):
    """
    Render export buttons for CSV and Excel.
    
    Args:
        sim_data: Simulation data dictionary
    """
    # Build DataFrame from sim_data
    df = sim_data.get("df_results")
    if df is None:
        st.warning("No results data available for export.")
        return
    
    st.subheader("Export Results")
    
    # CSV export
    df_flat = _flatten(df)
    csv_data = df_flat.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="scopehc_results.csv",
        mime="text/csv"
    )
    
    # Excel export (if openpyxl available)
    try:
        import openpyxl  # noqa
        
        # Create Excel file in memory
        output = create_excel_file(sim_data)
        
        st.download_button(
            "Download Excel",
            data=output,
            file_name="scopehc_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except ImportError:
        st.info("Install `openpyxl` to enable Excel export: `pip install openpyxl`")


def create_excel_file(sim_data: Dict[str, Any]) -> bytes:
    """
    Create Excel file with multiple sheets.
    
    Args:
        sim_data: Simulation data dictionary
        
    Returns:
        Excel file as bytes
    """
    import io
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Results sheet
        df_results = sim_data.get("df_results")
        if df_results is not None:
            _flatten(df_results).to_excel(writer, sheet_name='Results', index=False)
        
        # Summary sheet
        create_summary_sheet(writer, sim_data)

        # Cross-plot sheet (only when the Results-page explorer has been used)
        create_crossplot_sheet(writer)

        # Metadata sheet
        create_metadata_sheet(writer, sim_data)
    
    output.seek(0)
    return output.getvalue()


def create_summary_sheet(writer, sim_data: Dict[str, Any]):
    """Create summary statistics sheet."""
    df_results = sim_data.get("df_results")
    if df_results is None:
        return
    
    # Calculate summary statistics
    summary_data = []
    
    for col in df_results.columns:
        if df_results[col].dtype in ['float64', 'int64']:
            data = df_results[col]
            # Use convention-aware summarize_array for all statistics
            stats = summarize_array(data)
            summary_data.append({
                'Parameter': col,
                'Mean': stats.get('mean', np.mean(data)),
                # nan, not a raw percentile: stats is empty only for an all-NaN
                # column, and a raw percentile would ignore the active convention.
                'P10': stats.get('P10', float('nan')),
                'P50': stats.get('P50', float('nan')),
                'P90': stats.get('P90', float('nan')),
                'Std': stats.get('std_dev', np.std(data)),
                'Min': stats.get('min', np.min(data)),
                'Max': stats.get('max', np.max(data))
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)


def create_crossplot_sheet(writer):
    """Write the Results-page cross-plot selection to its own sheet.

    Reads what the explorer last computed (`crossplot_export` in session state) rather
    than recomputing it, so the workbook carries exactly the numbers that were on
    screen. Skipped entirely when the section has not been used in this session.
    """
    export = st.session_state.get("crossplot_export")
    if not isinstance(export, dict):
        return

    sheet = "Cross-plot stats"
    stats = export.get("stats") or {}
    header = pd.DataFrame(
        {
            "Item": [
                "X variable",
                "Y variable",
                "Bin width",
                "Trials used",
                "Trials in run",
                "Percentile convention",
                "Pearson r",
                "Pearson p",
                "Spearman rho",
                "Spearman p",
                "R squared",
                "Slope (Y per X)",
                "Intercept",
                "Log-log exponent",
            ],
            "Value": [
                export.get("x", ""),
                export.get("y", ""),
                export.get("bin_width"),
                export.get("n_used"),
                export.get("n_total"),
                "exceedance (P10 high)"
                if export.get("use_exceedance", True)
                else "non-exceedance (P10 low)",
                stats.get("pearson_r"),
                stats.get("pearson_p"),
                stats.get("spearman_rho"),
                stats.get("spearman_p"),
                stats.get("r_squared"),
                stats.get("slope"),
                stats.get("intercept"),
                stats.get("log_slope"),
            ],
        }
    )
    header.to_excel(writer, sheet_name=sheet, index=False, startrow=0)

    row = len(header) + 3
    bins = export.get("bins")
    if isinstance(bins, pd.DataFrame) and not bins.empty:
        pd.DataFrame({"Binned statistics": []}).to_excel(
            writer, sheet_name=sheet, index=False, startrow=row - 1
        )
        bins.to_excel(writer, sheet_name=sheet, index=False, startrow=row)
        row += len(bins) + 3

    matrix = export.get("matrix")
    if isinstance(matrix, pd.DataFrame) and not matrix.empty:
        pd.DataFrame({"Correlation matrix": []}).to_excel(
            writer, sheet_name=sheet, index=False, startrow=row - 1
        )
        matrix.to_excel(writer, sheet_name=sheet, startrow=row)


def create_metadata_sheet(writer, sim_data: Dict[str, Any]):
    """Create metadata sheet."""
    metadata_data = {
        'Parameter': [
            'Random Seed',
            'Number of Trials',
            'Gas to BOE Factor (scf/BOE)',
            'Unit System',
            'Export Date',
            'Application Version'
        ],
        'Value': [
            str(sim_data.get('seed', 'N/A')),
            str(sim_data.get('n_trials', 'N/A')),
            str(sim_data.get('scf_per_BOE', 6000.0)),
            sim_data.get('unit_system', 'oilfield'),
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'SCOPE-HC v0.1'
        ]
    }
    
    metadata_df = pd.DataFrame(metadata_data)
    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)


def export_to_csv(sim_data: Dict[str, Any]) -> Optional[str]:
    """
    Export simulation data to CSV format.
    
    Args:
        sim_data: Simulation data dictionary
        
    Returns:
        CSV data as string, or None if no data
    """
    df_results = sim_data.get("df_results")
    if df_results is None:
        return None
    
    return df_results.to_csv(index=False)


def export_to_excel(sim_data: Dict[str, Any]) -> Optional[bytes]:
    """
    Export simulation data to Excel format.
    
    Args:
        sim_data: Simulation data dictionary
        
    Returns:
        Excel data as bytes, or None if no data
    """
    try:
        import openpyxl  # noqa
        return create_excel_file(sim_data)
    except ImportError:
        st.error("openpyxl is not installed. Excel export is disabled.")
        return None
