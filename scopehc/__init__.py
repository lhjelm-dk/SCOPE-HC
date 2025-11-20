"""
SCOPE-HC: Subsurface Capacity Overview and Probability Estimator for Hydrocarbons

A modular Monte Carlo application for hydrocarbon volume estimation.
"""

# Export key constants and functions for convenience
from .config import (
    RB_PER_M3, RCF_PER_RB, UNIT_DISPLAY, PARAM_COLORS, HELP, DEFAULTS, PALETTE
)
from .utils import (
    clip01, safe_div, sanitize, invBg_to_Bg_rb_per_scf,
    validate_rf_fractions, validate_fractions, validate_depths,
    summarize_array, summary_table, compute_goc_depth
)
from .compute import compute_results
from .plots import init_plotly_theme, color_for

__all__ = [
    # Config
    'RB_PER_M3', 'RCF_PER_RB', 'UNIT_DISPLAY', 'PARAM_COLORS', 'HELP', 'DEFAULTS', 'PALETTE',
    # Utils
    'clip01', 'safe_div', 'sanitize', 'invBg_to_Bg_rb_per_scf',
    'validate_rf_fractions', 'validate_fractions', 'validate_depths',
    'summarize_array', 'summary_table', 'compute_goc_depth',
    # Compute
    'compute_results',
    # Plots
    'init_plotly_theme', 'color_for',
]
