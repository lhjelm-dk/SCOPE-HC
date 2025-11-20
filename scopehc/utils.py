"""
Utilities: validators, safe math, conversions
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from .config import RCF_PER_RB


def clip01(x):
    """Clip values to [0, 1]"""
    return np.minimum(np.maximum(x, 0.0), 1.0)


def safe_div(a, b):
    """Safe division, avoiding division by zero"""
    return a / np.maximum(b, 1e-12)


def sanitize(arr, name, min_allowed=-np.inf, max_allowed=np.inf, fill=0.0, warn=None):
    """
    Sanitize array: replace non-finite or out-of-range values with fill value.
    
    Args:
        arr: Input array
        name: Parameter name for warnings
        min_allowed: Minimum allowed value
        max_allowed: Maximum allowed value
        fill: Fill value for invalid entries
        warn: Optional warning function to call
        
    Returns:
        Sanitized array
    """
    arr = np.asarray(arr, float)
    bad = ~np.isfinite(arr) | (arr < min_allowed) | (arr > max_allowed)
    if np.any(bad) and warn:
        warn(f"{name}: {np.sum(bad)} invalid sample(s) replaced with {fill}.")
        arr = arr.copy()
        arr[bad] = fill
    return arr


def invBg_to_Bg_rb_per_scf(invBg_scf_per_rcf):
    """
    Convert 1/Bg (scf/rcf) to Bg (rb/scf).
    
    Formula: Bg = 1 / (InvBg * 5.614583)
    where 5.614583 is the conversion factor from cubic feet to barrels.
    """
    return 1.0 / (np.maximum(invBg_scf_per_rcf, 1e-12) * RCF_PER_RB)


def validate_rf_fractions(rf_oil, rf_gas, rf_cond=None):
    """Clamp recovery factors to [0,1] and return warnings if any were clamped."""
    warnings = []

    rf_oil = np.asarray(rf_oil, dtype=float)
    rf_gas = np.asarray(rf_gas, dtype=float)
    rf_oil_orig = rf_oil.copy()
    rf_gas_orig = rf_gas.copy()

    rf_oil = np.clip(rf_oil, 0.0, 1.0)
    rf_gas = np.clip(rf_gas, 0.0, 1.0)

    if np.any(rf_oil_orig != rf_oil):
        warnings.append(f"Oil recovery factor clamped to [0,1]: {np.sum(rf_oil_orig != rf_oil)} trials")
    if np.any(rf_gas_orig != rf_gas):
        warnings.append(f"Gas recovery factor clamped to [0,1]: {np.sum(rf_gas_orig != rf_gas)} trials")

    if rf_cond is not None:
        rf_cond = np.asarray(rf_cond, dtype=float)
        rf_cond_orig = rf_cond.copy()
        rf_cond = np.clip(rf_cond, 0.0, 1.0)
        if np.any(rf_cond_orig != rf_cond):
            warnings.append(f"Condensate recovery factor clamped to [0,1]: {np.sum(rf_cond_orig != rf_cond)} trials")
    else:
        rf_cond = None

    return rf_oil, rf_gas, rf_cond, warnings


def validate_fractions(f_oil):
    """Clamp oil fraction to [0,1] and return warnings if any were clamped."""
    warnings = []
    f_oil = np.asarray(f_oil, dtype=float)
    f_oil_orig = f_oil.copy()
    f_oil = np.clip(f_oil, 0.0, 1.0)
    
    if np.any(f_oil_orig != f_oil):
        warnings.append(f"Oil fraction clamped to [0,1]: {np.sum(f_oil_orig != f_oil)} trials")
    
    return f_oil, warnings


def validate_depths(D_top, D_hc):
    """Ensure D_hc >= D_top and return warnings if any were corrected."""
    warnings = []
    D_top = np.asarray(D_top, dtype=float)
    D_hc = np.asarray(D_hc, dtype=float)
    D_hc_orig = D_hc.copy()

    D_hc = np.maximum(D_hc, D_top)

    if np.any(D_hc_orig != D_hc):
        warnings.append(
            f"Hydrocarbon depth enforced to be >= top depth in {np.sum(D_hc_orig != D_hc)} trial(s)"
        )

    return D_top, D_hc, warnings


def summarize_array(x: np.ndarray) -> Dict[str, float]:
    """Summarize array with statistics including percentiles, mean, std dev, etc.
    
    Respects the percentile convention set in session state:
    - If percentile_exceedance=True: P10 high (90th percentile), P90 low (10th percentile)
    - If percentile_exceedance=False: P10 low (10th percentile), P90 high (90th percentile)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {}

    # Check percentile convention (default to exceedance if not set)
    try:
        import streamlit as st
        use_exceedance = st.session_state.get("percentile_exceedance", True)
    except (RuntimeError, AttributeError):
        # Not in Streamlit context, use default
        use_exceedance = True

    if use_exceedance:
        # Probability of exceedance convention: All percentiles are inverted
        # P1 = 99th percentile (high), P5 = 95th percentile (high), P10 = 90th percentile (high)
        # P25 = 75th percentile (high), P50 = 50th percentile (median, unchanged)
        # P75 = 25th percentile (low), P90 = 10th percentile (low), P95 = 5th percentile (low), P99 = 1st percentile (low)
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(x, percentiles)
        # Invert all percentiles: P1->P99, P5->P95, P10->P90, P25->P75, P50 stays, P75->P25, P90->P10, P95->P5, P99->P1
        p1_val = float(pvals[8])   # 99th percentile -> P1
        p5_val = float(pvals[7])   # 95th percentile -> P5
        p10_val = float(pvals[6])  # 90th percentile -> P10
        p25_val = float(pvals[5])  # 75th percentile -> P25
        p50_val = float(pvals[4])  # 50th percentile -> P50 (unchanged)
        p75_val = float(pvals[3])  # 25th percentile -> P75
        p90_val = float(pvals[2])  # 10th percentile -> P90
        p95_val = float(pvals[1])  # 5th percentile -> P95
        p99_val = float(pvals[0])  # 1st percentile -> P99
    else:
        # Traditional convention: P10 = 10th percentile (low), P90 = 90th percentile (high)
        # All percentiles use standard values
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pvals = np.percentile(x, percentiles)
        p1_val = float(pvals[0])   # 1st percentile
        p5_val = float(pvals[1])   # 5th percentile
        p10_val = float(pvals[2])  # 10th percentile
        p25_val = float(pvals[3])  # 25th percentile
        p50_val = float(pvals[4])  # 50th percentile
        p75_val = float(pvals[5])  # 75th percentile
        p90_val = float(pvals[6])  # 90th percentile
        p95_val = float(pvals[7])  # 95th percentile
        p99_val = float(pvals[8])  # 99th percentile

    mean_v = float(np.mean(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    std_v = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    var_v = float(np.var(x, ddof=1)) if x.size > 1 else 0.0

    # Calculate mode (most frequent value)
    if x.size > 1:
        n_bins = int(np.ceil(1 + 3.322 * np.log10(x.size)))
        n_bins = max(min(n_bins, 50), 10)
        hist, bin_edges = np.histogram(x, bins=n_bins)
        max_bin_idx = np.argmax(hist)
        mode_v = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2)
    else:
        mode_v = mean_v

    # Calculate skewness
    range_v = max_v - min_v
    if x.size > 2 and std_v > 0 and range_v > 1e-12 * max(1.0, abs(mean_v)):
        from scipy.stats import skew
        skew_v = float(skew(x, bias=False))
    else:
        skew_v = 0.0

    return {
        "mean": mean_v,
        "min": min_v,
        "max": max_v,
        "mode": mode_v,
        "std_dev": std_v,
        "variance": var_v,
        "skewness": skew_v,
        "P1": p1_val,   # Uses convention-aware value
        "P5": p5_val,   # Uses convention-aware value
        "P10": p10_val, # Uses convention-aware value
        "P25": p25_val, # Uses convention-aware value
        "P50": p50_val, # Uses convention-aware value
        "P75": p75_val, # Uses convention-aware value
        "P90": p90_val, # Uses convention-aware value
        "P95": p95_val, # Uses convention-aware value
        "P99": p99_val, # Uses convention-aware value
    }


def summary_table(x: np.ndarray, decimals: Optional[int] = None) -> pd.DataFrame:
    """Create a formatted summary table from array statistics.
    
    Percentile columns are ordered based on the active convention:
    - Exceedance: P99, P95, P90, P75, P50, P25, P10, P5, P1 (largest to smallest)
    - Non-exceedance: P1, P5, P10, P25, P50, P75, P90, P95, P99 (smallest to largest)
    """
    stats = summarize_array(x)
    if not stats:
        return pd.DataFrame()
    
    # Check percentile convention for column ordering
    try:
        import streamlit as st
        use_exceedance = st.session_state.get("percentile_exceedance", True)
    except (RuntimeError, AttributeError):
        use_exceedance = True
    
    if use_exceedance:
        # Exceedance: order from largest to smallest (P99 to P1)
        percentile_order = ["P99", "P95", "P90", "P75", "P50", "P25", "P10", "P5", "P1"]
    else:
        # Non-exceedance: order from smallest to largest (P1 to P99)
        percentile_order = ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]
    
    order = ["mean", "min", "max", "mode", "std_dev", "variance", "skewness"] + percentile_order
    df = pd.DataFrame([{k: stats[k] for k in order}])
    
    if decimals is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col] = df[col].apply(lambda x: f"{x:.{decimals}f}".rstrip('0').rstrip('.') if pd.notna(x) else x)
    
    column_mapping = {
        "mean": "Mean",
        "min": "Min.",
        "max": "Max.",
        "mode": "Mode",
        "std_dev": "Std Dev",
        "variance": "Variance",
        "skewness": "Skewness",
        "P1": "P1",
        "P5": "P5",
        "P10": "P10",
        "P25": "P25",
        "P50": "P50",
        "P75": "P75",
        "P90": "P90",
        "P95": "P95",
        "P99": "P99"
    }
    df = df.rename(columns=column_mapping)
    
    df_styled = df.style.set_properties(**{
        'text-align': 'center',
        'vertical-align': 'middle'
    })
    
    return df_styled


def compute_goc_depth(D_top, D_hc, mode, value):
    """Compute GOC depth based on different definition modes."""
    D_top = np.asarray(D_top, dtype=float)
    D_hc = np.asarray(D_hc, dtype=float)
    v = np.asarray(value, dtype=float)
    H_hc = np.maximum(D_hc - D_top, 0.0)
    
    if mode == "Direct depth":
        D_goc = v
    elif mode == "Fraction of column that is oil":
        f_oil = np.clip(v, 0.0, 1.0)
        D_goc = D_top + (1.0 - f_oil) * H_hc
    elif mode == "Oil column height":
        H_oil = np.maximum(v, 0.0)
        D_goc = D_hc - H_oil
    else:
        raise ValueError("Invalid GOC mode")
    
    return np.minimum(np.maximum(D_goc, D_top), D_hc)
