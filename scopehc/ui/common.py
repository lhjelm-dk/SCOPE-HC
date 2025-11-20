from __future__ import annotations

import math
import hashlib
import json
import time
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from scipy.stats import skew, norm, beta

from scopehc.config import (
    RB_PER_M3,
    RCF_PER_RB,
    UNIT_DISPLAY,
    PARAM_COLORS,
    HELP,
    DEFAULTS,
    PALETTE,
)
from scopehc.utils import (
    invBg_to_Bg_rb_per_scf,
    clip01,
    safe_div,
    sanitize,
    validate_rf_fractions,
    validate_fractions,
    validate_depths,
    summarize_array,
    summary_table,
    compute_goc_depth,
)
from scopehc.geom import (
    get_gcf_lookup_table,
    interpolate_gcf,
    calculate_grv_from_depth_table,
    _cumulative_trapz,
)
from scopehc.sampling import (
    rng_from_seed,
    sample_uniform,
    sample_triangular,
    sample_pert,
    sample_lognormal_mean_sd,
    sample_beta_subjective,
    sample_stretched_beta,
    sample_truncated_normal,
    sample_truncated_lognormal,
    sample_burr,
    sample_johnson_su,
    correlated_samples,
    validate_dependency_matrix,
    fix_correlation_matrix,
    apply_correlation,
    sample_scalar_dist,
)
from scopehc.compute import compute_results
from scopehc.plots import (
    color_for,
    hist_and_cdf,
    make_depth_area_plot,
    make_area_volume_plot,
    extract_param_name_from_title,
)
from scopehc.export import render_export_buttons, export_to_csv, export_to_excel


def render_disclaimer() -> None:
    """
    Render a standard software disclaimer stating no warranty and no liability for errors.
    """
    st.markdown("---")
    with st.expander("⚠️ Disclaimer", expanded=False):
        st.markdown(
            """
            **No Warranty or Liability**
            
            This software is provided "as is" without warranty of any kind, express or implied, 
            including but not limited to the warranties of merchantability, fitness for a particular 
            purpose, and non-infringement.
            
            The author and contributors shall not be liable for any errors, omissions, or inaccuracies 
            in the calculations, results, or any other output from this software, whether arising 
            from coding errors, algorithmic mistakes, or incorrect user inputs.
            
            Users are responsible for verifying all calculations and results independently before 
            making any decisions based on the output of this software. This tool is intended for 
            educational and research purposes and should not be used as the sole basis for critical 
            business or technical decisions without independent verification.
            
            **Use at your own risk.**
            """,
            unsafe_allow_html=False,
        )


__all__ = [
    "init_theme",
    "UNIT_CONVERSIONS",
    "DistributionChoice",
    "DistributionChoiceWithConstant",
    "color_swatch",
    "get_unit_help_text",
    "convert_fluid_property_value",
    "get_converted_default_params",
    "make_hist_cdf_figure",
    "calculate_grv",
    "calculate_grv_from_depth_table",
    "compute_dgrv_top_plus_thickness",
    "compute_dgrv_top_base_table",
    "get_unit_system",
    # Run tracker and diagnostics
    "compute_input_hash",
    "collect_all_inputs_from_session",
    "render_run_controls_and_diagnostics",
    "render_param",
    "apply_correlations_to_samples",
    "collect_all_trial_data",
    "update_progress",
    "sample_dependent_parameters",
    "sample_correlated",
    "create_dependency_matrix_ui",
    "create_dependency_matrix_ui_with_scatter_plots",
    "render_export_buttons",
    "export_to_csv",
    "export_to_excel",
    "render_disclaimer",
    "DEFAULTS",
    "HELP",
    "PARAM_COLORS",
    "PALETTE",
    "UNIT_DISPLAY",
    "rng_from_seed",
    "sample_uniform",
    "sample_triangular",
    "sample_pert",
    "sample_lognormal_mean_sd",
    "sample_beta_subjective",
    "sample_stretched_beta",
    "sample_truncated_normal",
    "sample_truncated_lognormal",
    "sample_burr",
    "sample_johnson_su",
    "correlated_samples",
    "validate_dependency_matrix",
    "fix_correlation_matrix",
    "apply_correlation",
    "sample_scalar_dist",
    "compute_results",
    "make_depth_area_plot",
    "make_area_volume_plot",
    "extract_param_name_from_title",
    "invBg_to_Bg_rb_per_scf",
    "summary_table",
    "summarize_array",
    "sanitize",
    "clip01",
    "render_color_legend",
    "render_sidebar_settings",
    "render_custom_navigation",
    "color_for",
]


_THEME_INITIALIZED = False


def init_theme() -> None:
    """Initialise shared theme, styles, and Plotly configuration once per session."""
    global _THEME_INITIALIZED
    if _THEME_INITIALIZED:
        return

    # Inject hiding code via components.html for later execution (runs after page render)
    try:
        import streamlit.components.v1 as components
        components.html("""
        <script>
        (function() {
            function forceHideNav() {
                // Find and remove/hide Streamlit's navigation - very aggressive
                const selectors = [
                    '[data-testid="stSidebarNav"]',
                    'nav[data-testid="stSidebarNav"]',
                    'div[data-testid="stSidebarNav"]',
                    'section[data-testid="stSidebarNav"]'
                ];
                
                selectors.forEach(sel => {
                    try {
                        const els = document.querySelectorAll(sel);
                        els.forEach(el => {
                            const text = (el.textContent || '').toLowerCase();
                            if (text.includes('streamlit app') || text.includes('overview') || text.includes('inputs') || el.querySelector('a[href*="pages/"]')) {
                                if (!el.closest('.custom-nav-item')) {
                                    try {
                                        el.remove();
                                    } catch(e) {
                                        el.style.cssText = 'display:none!important;visibility:hidden!important;height:0!important;width:0!important;opacity:0!important;position:absolute!important;left:-9999px!important;z-index:-9999!important;';
                                    }
                                }
                            }
                        });
                    } catch(e) {}
                });
                
                // Also check sidebar for any page links that aren't in buttons
                const sidebar = document.querySelector('[data-testid="stSidebar"]');
                if (sidebar) {
                    const pageLinks = sidebar.querySelectorAll('a[href*="pages/"]');
                    pageLinks.forEach(link => {
                        if (!link.closest('button') && !link.closest('.custom-nav-item')) {
                            const container = link.closest('nav, div, section, ul, li');
                            if (container && container !== sidebar) {
                                try {
                                    container.remove();
                                } catch(e) {
                                    container.style.cssText = 'display:none!important;visibility:hidden!important;height:0!important;width:0!important;';
                                }
                            }
                        }
                    });
                }
            }
            
            forceHideNav();
            setInterval(forceHideNav, 25);  // Check every 25ms
            const obs = new MutationObserver(forceHideNav);
            obs.observe(document.body, {childList: true, subtree: true, attributes: true});
            
            // Also on any click
            document.addEventListener('click', forceHideNav, true);
        })();
        </script>
        """, height=0)
    except Exception:
        pass

    plt.style.use("seaborn-v0_8-colorblind")

    pio.templates["scopehc_theme"] = pio.templates["plotly_white"]
    pio.templates["scopehc_theme"]["layout"]["colorway"] = [
        PALETTE["primary"],
        PALETTE["secondary"],
        PALETTE["accent"],
        PALETTE["neutral"],
        PALETTE["highlight"],
    ]
    pio.templates["scopehc_theme"]["layout"]["paper_bgcolor"] = "#F8F9FA"
    pio.templates["scopehc_theme"]["layout"]["plot_bgcolor"] = "#FFFFFF"
    pio.templates["scopehc_theme"]["layout"]["font"] = {
        "size": 13,
        "color": PALETTE["text_primary"],
    }
    pio.templates.default = "scopehc_theme"

    accent = PALETTE["accent"]
    accent_soft = "#FF9A9A"
    accent_dark = "#C84B4B"
    primary = PALETTE["primary"]
    secondary = PALETTE["secondary"]
    neutral = PALETTE["neutral"]
    highlight = PALETTE["highlight"]
    text_primary = PALETTE["text_primary"]
    text_secondary = PALETTE["text_secondary"]
    bg_light = PALETTE["bg_light"]
    border = PALETTE["border"]

    st.markdown(
        f"""
        <style>
        html, body, [class*="css"]  {{
            font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
            background-color: {bg_light} !important;
            color: {text_primary} !important;
        }}

        h1 {{
            color: {primary} !important;
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            margin-top: 0.5em;
            margin-bottom: 0.2em;
            text-align: center;
            letter-spacing: -0.02em;
        }}

        h2 {{
            color: {secondary} !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-top: 1.2em;
            margin-bottom: 0.3em;
            border-left: 4px solid {accent};
            padding-left: 12px;
            background: linear-gradient(90deg, rgba(161, 50, 50, 0.12), transparent);
            padding: 8px 0 8px 12px;
            border-radius: 4px;
        }}

        h3, h4 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
            margin-top: 1.0em;
            margin-bottom: 0.3em;
        }}

        p, label, span, div {{
            font-size: 0.95rem !important;
            color: {text_primary} !important;
            line-height: 1.5;
        }}

        .stButton>button {{
            background: linear-gradient(135deg, {accent_soft}, {accent}) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5em 1.2em !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            transition: all 0.2s ease;
            box-shadow: 0 2px 6px rgba(161, 50, 50, 0.25);
        }}

        .stButton>button:hover {{
            background: linear-gradient(135deg, {accent}, {accent_dark}) !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 12px rgba(161, 50, 50, 0.35);
        }}

        .stButton>button[data-testid*="recalc"] {{
            color: white !important;
        }}

        .stSlider>div>div>div>div {{
            background: linear-gradient(90deg, {accent}, {secondary}) !important;
            border-radius: 6px !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        }}

        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {{
            background-color: #FFFFFF !important;
            border: 1px solid {border} !important;
            border-radius: 6px !important;
            padding: 0.5em 0.8em !important;
            transition: border-color 0.2s ease;
        }}

        .stTextInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>select:focus {{
            border-color: {accent} !important;
            box-shadow: 0 0 0 3px rgba(161, 50, 50, 0.12) !important;
        }}

        .stDataFrame {{
            background-color: #FFFFFF !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            border: 1px solid {border};
            overflow: hidden;
        }}

        .card-container {{
            background: linear-gradient(135deg, #FFFFFF, {bg_light});
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            border: 1px solid {border};
        }}

        .section-divider {{
            background: linear-gradient(90deg, transparent, {accent}, transparent);
            height: 2px;
            margin: 2em 0;
            border-radius: 1px;
        }}

        .scopehc-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.35em 0.75em;
            border-radius: 999px;
            background: rgba(161, 50, 50, 0.1);
            color: {accent};
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border: 1px solid rgba(161, 50, 50, 0.22);
            margin-bottom: 0.5em;
        }}

        .scopehc-metric>div>div>div:first-child {{
            color: {primary} !important;
        }}

        .scopehc-metric>div>div>div:nth-child(2) {{
            color: {text_primary} !important;
        }}

        .scopehc-metric>div>div>div:last-child {{
            color: {text_secondary} !important;
        }}

        .scopehc-legend {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.5em 1em;
            margin-top: 0.5em;
        }}

        .scopehc-legend-item {{
            display: inline-flex;
            align-items: center;
            gap: 0.5em;
            font-size: 0.85rem;
            color: {text_secondary};
        }}

        .scopehc-legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 8px 8px 0 0 !important;
            border: 1px solid {border} !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            color: {text_secondary} !important;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {accent} !important;
            color: #FFFFFF !important;
            border-color: {accent} !important;
        }}

        .stAlert {{
            border-radius: 8px !important;
            border-left: 4px solid {accent} !important;
            background-color: rgba(161, 50, 50, 0.06) !important;
        }}

        .stProgress>div>div>div>div {{
            background: linear-gradient(90deg, {accent}, {secondary}) !important;
        }}

        /* Hide Streamlit's automatic navigation completely - target by structure and position */
        /* Target by data-testid */
        [data-testid="stSidebarNav"],
        nav[data-testid="stSidebarNav"],
        div[data-testid="stSidebarNav"],
        section[data-testid="stSidebarNav"],
        [data-testid="stSidebar"] [data-testid="stSidebarNav"],
        [data-testid="stSidebar"] nav[data-testid="stSidebarNav"],
        [data-testid="stSidebar"] div[data-testid="stSidebarNav"],
        [data-testid="stSidebarNav"] *,
        nav[data-testid="stSidebarNav"] *,
        div[data-testid="stSidebarNav"] *,
        section[data-testid="stSidebarNav"] *,
        [data-testid="stSidebar"] [data-testid="stSidebarNav"] * {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            max-height: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0 !important;
            pointer-events: none !important;
            position: absolute !important;
            left: -9999px !important;
            top: -9999px !important;
            z-index: -9999 !important;
            transform: scale(0) !important;
        }}
        
        /* Target by structure - any nav/ul/li in sidebar that contains page links */
        [data-testid="stSidebar"] > nav:first-child,
        [data-testid="stSidebar"] > div:first-child > nav,
        [data-testid="stSidebar"] nav:has(a[href*="pages/"]),
        [data-testid="stSidebar"] ul:has(a[href*="pages/"]),
        [data-testid="stSidebar"] > *:first-child:has(a[href*="pages/"]),
        [data-testid="stSidebar"] > *:first-child:has([href*="pages/"]) {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            opacity: 0 !important;
            position: absolute !important;
            left: -9999px !important;
            z-index: -9999 !important;
        }}
        
        /* Hide any ul/li lists that might be navigation */
        [data-testid="stSidebarNav"] ul,
        [data-testid="stSidebarNav"] li,
        [data-testid="stSidebarNav"] a,
        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNav"] div,
        [data-testid="stSidebar"] nav ul,
        [data-testid="stSidebar"] nav li,
        [data-testid="stSidebar"] nav a {{
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            width: 0 !important;
        }}
        
        /* Hide any links to pages/ that aren't in buttons */
        [data-testid="stSidebar"] a[href*="pages/"]:not(button a):not(.custom-nav-item a) {{
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            height: 0 !important;
            width: 0 !important;
        }}
        
        /* Prevent any content from showing through */
        [data-testid="stSidebarNav"]::before,
        [data-testid="stSidebarNav"]::after {{
            display: none !important;
            content: none !important;
        }}
        </style>
        <script>
        // Force hide Streamlit's automatic navigation - multiple strategies
        // Run this function as early as possible
        (function() {{
            'use strict';
            function hideAutoNav() {{
            // Try multiple selectors - be very aggressive
            const selectors = [
                '[data-testid="stSidebarNav"]',
                'nav[data-testid="stSidebarNav"]',
                'div[data-testid="stSidebarNav"]',
                'section[data-testid="stSidebarNav"]',
                '[class*="stSidebarNav"]',
                '[class*="sidebar-nav"]',
                'nav[class*="css"]',  // Streamlit often uses CSS classes
                '[role="navigation"]'
            ];
            
            selectors.forEach(selector => {{
                try {{
                    const elements = document.querySelectorAll(selector);
                    elements.forEach(el => {{
                        // First check if it's Streamlit's nav by looking for specific content
                        const text = (el.textContent || '').toLowerCase();
                        const isStreamlitNav = text.includes('streamlit app') || 
                                             text.includes('overview') || 
                                             text.includes('inputs') ||
                                             el.querySelector('a[href*="pages/"]');
                        
                        // Only hide if it's Streamlit's nav, not our custom nav
                        if (isStreamlitNav && !el.closest('.custom-nav-item')) {{
                            // Try to remove from DOM entirely first
                            try {{
                                if (el.parentNode) {{
                                    el.parentNode.removeChild(el);
                                    return;  // Successfully removed
                                }}
                            }} catch (e) {{
                                // If removal fails, hide it aggressively
                            }}
                            
                            // Aggressive hiding
                            el.style.setProperty('display', 'none', 'important');
                            el.style.setProperty('visibility', 'hidden', 'important');
                            el.style.setProperty('height', '0', 'important');
                            el.style.setProperty('width', '0', 'important');
                            el.style.setProperty('max-height', '0', 'important');
                            el.style.setProperty('max-width', '0', 'important');
                            el.style.setProperty('overflow', 'hidden', 'important');
                            el.style.setProperty('margin', '0', 'important');
                            el.style.setProperty('padding', '0', 'important');
                            el.style.setProperty('opacity', '0', 'important');
                            el.style.setProperty('pointer-events', 'none', 'important');
                            el.style.setProperty('position', 'absolute', 'important');
                            el.style.setProperty('left', '-9999px', 'important');
                            el.style.setProperty('top', '-9999px', 'important');
                            el.style.setProperty('z-index', '-9999', 'important');
                            el.style.setProperty('transform', 'scale(0)', 'important');
                            
                            // Also hide all children
                            const children = el.querySelectorAll('*');
                            children.forEach(child => {{
                                child.style.setProperty('display', 'none', 'important');
                                child.style.setProperty('visibility', 'hidden', 'important');
                                child.style.setProperty('opacity', '0', 'important');
                            }});
                        }}
                    }});
                }} catch (e) {{
                    // Ignore selector errors
                }}
            }});
            
            // Also check for any element in sidebar that looks like navigation
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar) {{
                const allLinks = sidebar.querySelectorAll('a[href*="pages/"]');
                allLinks.forEach(link => {{
                    // Check if this link is NOT in our custom nav
                    if (!link.closest('.custom-nav-item') && !link.closest('button')) {{
                        const parent = link.closest('nav, div, section, ul, li');
                        if (parent && parent !== sidebar) {{
                            // This looks like Streamlit's nav
                            try {{
                                if (parent.parentNode) {{
                                    parent.parentNode.removeChild(parent);
                                }}
                            }} catch (e) {{
                                parent.style.setProperty('display', 'none', 'important');
                                parent.style.setProperty('visibility', 'hidden', 'important');
                            }}
                        }}
                    }}
                }});
            }}
        }}
        
        // Run immediately and very frequently - especially after page loads/reruns
        hideAutoNav();
        setTimeout(hideAutoNav, 1);
        setTimeout(hideAutoNav, 5);
        setTimeout(hideAutoNav, 10);
        setTimeout(hideAutoNav, 20);
        setTimeout(hideAutoNav, 50);
        setTimeout(hideAutoNav, 100);
        setTimeout(hideAutoNav, 200);
        setTimeout(hideAutoNav, 500);
        setTimeout(hideAutoNav, 1000);
        setInterval(hideAutoNav, 50);  // Check every 50ms
        
        // Also use MutationObserver to catch dynamically added elements - very aggressive
        const navObserver = new MutationObserver((mutations) => {{
            hideAutoNav();
            // Also check again after a tiny delay
            setTimeout(hideAutoNav, 1);
            setTimeout(hideAutoNav, 10);
        }});
        navObserver.observe(document.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['style', 'class', 'data-testid'] }});
        
        // Also observe the sidebar specifically
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {{
            navObserver.observe(sidebar, {{ childList: true, subtree: true, attributes: true }});
        }}
        
        // Listen to all possible events that might cause re-rendering
        ['click', 'mousedown', 'mouseup', 'keydown', 'keyup', 'change', 'input', 'focus', 'blur', 'submit'].forEach(eventType => {{
            document.addEventListener(eventType, () => {{
                hideAutoNav();  // Immediate
                setTimeout(hideAutoNav, 1);
                setTimeout(hideAutoNav, 10);
                setTimeout(hideAutoNav, 50);
                setTimeout(hideAutoNav, 100);
                setTimeout(hideAutoNav, 200);
            }}, true);  // Use capture phase to catch early
        }});
        
        // Also listen to Streamlit-specific events if they exist
        if (window.parent) {{
            window.parent.addEventListener('message', () => {{
                hideAutoNav();
                setTimeout(hideAutoNav, 1);
                setTimeout(hideAutoNav, 10);
                setTimeout(hideAutoNav, 50);
            }});
        }}
        
        // Watch for any style changes that might unhide it
        const styleObserver = new MutationObserver((mutations) => {{
            mutations.forEach(mutation => {{
                if (mutation.type === 'attributes' && (mutation.attributeName === 'style' || mutation.attributeName === 'class')) {{
                    const target = mutation.target;
                    if (target.getAttribute('data-testid') === 'stSidebarNav' || target.closest('[data-testid="stSidebarNav"]')) {{
                        hideAutoNav();
                        setTimeout(hideAutoNav, 1);
                    }}
                }}
            }});
        }});
        
        // Observe all elements for style changes - set up immediately and on DOM ready
        function setupStyleObserver() {{
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {{
                if (el.getAttribute('data-testid') === 'stSidebarNav' || el.closest('[data-testid="stSidebarNav"]')) {{
                    styleObserver.observe(el, {{ attributes: true, attributeFilter: ['style', 'class'] }});
                }}
            }});
        }}
        
        setupStyleObserver();
        document.addEventListener('DOMContentLoaded', setupStyleObserver);
        
        // Also run on page visibility changes (when user switches tabs back)
        document.addEventListener('visibilitychange', () => {{
            if (!document.hidden) {{
                hideAutoNav();
                setTimeout(hideAutoNav, 10);
                setTimeout(hideAutoNav, 50);
            }}
        }});
        
        // Intercept button clicks on navigation buttons to hide immediately
        document.addEventListener('click', (e) => {{
            const target = e.target;
            // Check if it's one of our custom nav buttons
            if (target.closest && (target.closest('button[data-testid*="nav_"]') || target.closest('.custom-nav-button'))) {{
                // Hide immediately before navigation
                hideAutoNav();
                // And keep hiding after navigation
                setTimeout(hideAutoNav, 1);
                setTimeout(hideAutoNav, 10);
                setTimeout(hideAutoNav, 50);
                setTimeout(hideAutoNav, 100);
                setTimeout(hideAutoNav, 200);
                setTimeout(hideAutoNav, 500);
            }}
        }}, true);
        
        // Use requestAnimationFrame for continuous checking
        function continuousHide() {{
            hideAutoNav();
            requestAnimationFrame(continuousHide);
        }}
        continuousHide();
        }})();  // End IIFE
        </script>
        """,
        unsafe_allow_html=True,
    )

    _THEME_INITIALIZED = True


UNIT_CONVERSIONS: Dict[str, Any] = {
    "m3_to_bbl": 6.28981077,
    "m3_to_ft3": 35.3146667,
    "bbl_to_m3": 1 / 6.28981077,
    "ft3_to_m3": 1 / 35.3146667,
    "scf_to_m3": 0.0283168,
    "m3_to_scf": 1 / 0.0283168,
    "psia_to_pa": 6894.76,
    "pa_to_psia": 1 / 6894.76,
    "f_to_c": lambda f: (f - 32) * 5 / 9,
    "c_to_f": lambda c: c * 9 / 5 + 32,
    "rcf_per_rb": 5.614583,
    "f_to_k": lambda f: (f + 459.67) * 5 / 9,
    "k_to_f": lambda k: k * 9 / 5 - 459.67,
    "rb_per_scf_to_m3_per_m3": 5.614583,
    "m3_per_m3_to_rb_per_scf": 1 / 5.614583,
    "stb_per_rb_to_m3_per_m3": 1.0,
    "m3_per_m3_to_stb_per_rb": 1.0,
    "scf_per_stb_to_m3_per_m3": 0.1781076,
    "m3_per_m3_to_scf_per_stb": 1 / 0.1781076,
}


DistributionChoice = [
    "PERT",
    "Triangular",
    "Uniform",
    "Lognormal (mean, sd)",
    "Subjective Beta (Vose)",
    "Stretched Beta",
    "Truncated Normal",
    "Truncated Lognormal",
    "Burr (c, d, scale)",
    "Johnson SU",
]

# Full distribution list including Constant (for saturation inputs)
DistributionChoiceWithConstant = ["Constant"] + DistributionChoice


def color_swatch(hex_color: str, label: str) -> str:
    """Generate HTML for color swatch with label."""
    return (
        "<div style='display:flex;align-items:center;margin:2px 0;'>"
        f"<span style='display:inline-block;width:14px;height:14px;border-radius:3px;"
        f"background:{hex_color};margin-right:8px;border:1px solid rgba(0,0,0,.25);'></span>"
        f"<span style='font-size:.88rem;'>{label}</span></div>"
    )


def render_color_legend() -> None:
    """Render color legend in sidebar explaining the color scheme."""
    st.sidebar.markdown("### Color Legend")
    st.sidebar.caption("Light greens = oil-related, light reds = gas-related, light orange = condensate-related, earthy tones = rock/geometry, light blue = derived/systemic.")
    
    legend_items = [
        ("Oil_STB_rec", "Oil (surface)"),
        ("Gas_free_scf_rec", "Free Gas"),
        ("Gas_assoc_scf_rec", "Associated Gas"),
        ("Cond_STB_rec", "Condensate"),
        ("Total_surface_BOE", "Total Hydrocarbons"),
        ("GOC", "Gas-Oil Contact"),
        ("Effective_HC_Depth", "Effective HC Depth"),
        ("GRV_total_m3", "GRV"),
        ("Porosity", "Porosity"),
        ("NtG", "Net-to-Gross"),
    ]

    html = "".join(
        color_swatch(PARAM_COLORS.get(key, PALETTE["primary"]), label)
        for key, label in legend_items
        if key in PARAM_COLORS
    )

    if html:
        st.sidebar.markdown(
            f"<div class='scopehc-legend'>{html}</div>", unsafe_allow_html=True
        )


def render_custom_navigation() -> None:
    """Render custom navigation menu in the sidebar with proper indentation."""
    # Handle navigation target - set session state for router in streamlit_app.py
    nav_target = st.session_state.get("_nav_target")
    if nav_target:
        # Set the current page in session state (router will handle loading)
        st.session_state["current_page"] = nav_target
        # Clear the target to avoid infinite loops
        del st.session_state["_nav_target"]
        st.rerun()  # Rerun to load the new page
        return
    
    st.sidebar.markdown("### Navigation")
    
    # Get current page from session state (set by each page)
    try:
        current_page = st.session_state.get("current_page", "")
    except Exception:
        current_page = ""
    
    # Navigation items - using _pages_disabled to prevent Streamlit auto-detection
    nav_items = [
        {"title": "Overview", "page": "_pages_disabled/00_Overview.py", "indent": False},
        {"title": "Inputs (Main - all sections)", "page": "_pages_disabled/01_Inputs.py", "indent": False},
        {"title": "Gross Rock Volume (GRV)", "page": "_pages_disabled/01_Inputs_Gross_Rock_Volume_GRV.py", "indent": True},
        {"title": "Net to Gross (NtG)", "page": "_pages_disabled/01_Inputs_Net_to_Gross_NtG.py", "indent": True},
        {"title": "Porosity", "page": "_pages_disabled/01_Inputs_Porosity.py", "indent": True},
        {"title": "Saturation", "page": "_pages_disabled/01_Inputs_Saturation.py", "indent": True},
        {"title": "Fluids", "page": "_pages_disabled/01_Inputs_Fluids.py", "indent": True},
        {"title": "Recovery Factor", "page": "_pages_disabled/01_Inputs_Recovery.py", "indent": True},
        {"title": "Dependencies", "page": "_pages_disabled/02_Dependencies.py", "indent": False},
        {"title": "Run Simulation", "page": "_pages_disabled/03_Run_Simulation.py", "indent": False},
        {"title": "Results", "page": "_pages_disabled/04_Results.py", "indent": False},
        {"title": "Sensitivity", "page": "_pages_disabled/05_Sensitivity.py", "indent": False},
        {"title": "Downloads", "page": "_pages_disabled/06_Downloads.py", "indent": False},
        {"title": "Help & Math", "page": "_pages_disabled/07_Help_Math.py", "indent": False},
    ]
    
    # Create navigation using buttons with proper indentation
    for item in nav_items:
        display_title = f" - {item['title']}" if item['indent'] else item['title']
        
        # Check if this is the current page
        is_active = current_page == item['page']
        
        # Use button for navigation (works with Streamlit's navigation)
        if item['indent']:
            # Add indentation using empty column
            col1, col2 = st.sidebar.columns([0.15, 0.85])
            with col1:
                st.write("")  # Spacer for indentation
            with col2:
                if st.button(
                    display_title,
                    key=f"nav_{item['page']}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["_nav_target"] = item['page']
                    st.rerun()
        else:
            # No indentation for main items
            if st.sidebar.button(
                display_title,
                key=f"nav_{item['page']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["_nav_target"] = item['page']
                st.rerun()


def render_sidebar_settings() -> None:
    """Render global simulation settings in the sidebar."""
    st.sidebar.markdown(
        f"""
        <div style='background: linear-gradient(135deg, {PALETTE["bg_light"]}, rgba(245, 241, 230, 0.8));
            color: {PALETTE["text_primary"]}; padding: 9px; border-radius: 10px; margin-bottom: 12px;
            text-align: center; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); border: 1px solid {PALETTE["border"]};'>
            <h3 style='color: {PALETTE["primary"]}; margin: 0; font-weight: 600; font-size: 1.2rem;'>Simulation Settings</h3>
            <p style='color: {PALETTE["text_secondary"]}; margin: 5px 0 0 0; font-size: 0.85rem;'>
                Configure your analysis parameters
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Fluid type selector
    st.sidebar.markdown("### Fluid Type")
    
    fluid_options = ["Oil", "Gas", "Oil + Gas"]
    
    # CRITICAL: Use a separate persistent storage that we control completely
    # This avoids any conflicts with Streamlit's widget state management
    # CRITICAL: Use persistent storage similar to GOC mode
    # Store in a special location that won't be touched by initialization functions
    persistent_storage_key = "_ui_state"
    if persistent_storage_key not in st.session_state:
        st.session_state[persistent_storage_key] = {}
    
    fluid_type_key = "fluid_type"
    
    # CRITICAL: Check if widget already exists in session state (user may have just clicked)
    # The widget's value (from session state) is the source of truth AFTER it's created
    widget_fluid_type = st.session_state.get(fluid_type_key, None)
    
    # If widget has a value, use it (user just clicked or it's already set)
    if widget_fluid_type is not None and widget_fluid_type in fluid_options:
        # Widget has a valid value - this is the source of truth
        desired_fluid_type = widget_fluid_type
        # Save to persistent storage
        st.session_state[persistent_storage_key][fluid_type_key] = desired_fluid_type
    else:
        # Widget doesn't have a value yet - read from persistent storage
        persistent_fluid_type = st.session_state[persistent_storage_key].get(fluid_type_key, None)
        
        # If not in persistent storage, use default
        if persistent_fluid_type is None:
            persistent_fluid_type = "Oil"
        
        # Validate the stored value
        if persistent_fluid_type not in fluid_options:
            persistent_fluid_type = "Oil"
        
        desired_fluid_type = persistent_fluid_type
        # Store in both locations
        st.session_state[persistent_storage_key][fluid_type_key] = desired_fluid_type
        st.session_state[fluid_type_key] = desired_fluid_type
    
    # Calculate index for display
    try:
        final_index = fluid_options.index(desired_fluid_type)
    except ValueError:
        final_index = 0
        desired_fluid_type = "Oil"
        st.session_state[persistent_storage_key][fluid_type_key] = "Oil"
        st.session_state[fluid_type_key] = "Oil"
    
    fluid_type = st.sidebar.radio(
        "Select fluid case for GRV:",
        options=fluid_options,
        index=final_index,
        key=fluid_type_key,
        help="Determines how GRV is split between oil and gas zones. For 'Oil + Gas', you can define the split using GOC (depth-based) or oil fraction (other methods)."
    )
    
    # CRITICAL: After widget creation, ALWAYS use widget's value and save to persistent storage
    # The widget's value is the authoritative source after creation
    actual_fluid_type = st.session_state[fluid_type_key]
    if actual_fluid_type in fluid_options:
        # Valid value - save to persistent storage and use it
        st.session_state[persistent_storage_key][fluid_type_key] = actual_fluid_type
        fluid_type = actual_fluid_type
    else:
        # Invalid value - use what we had before
        fluid_type = desired_fluid_type
        st.session_state[fluid_type_key] = desired_fluid_type
    
    # Also maintain backward compatibility with _fluid_type_persistent for cache clearing logic
    if "_fluid_type_persistent" not in st.session_state or st.session_state.get("_fluid_type_persistent") != fluid_type:
        st.session_state["_fluid_type_persistent"] = fluid_type
    
    # CRITICAL: After widget creation, NEVER overwrite the value
    # The widget value (fluid_type) is now the source of truth
    # Streamlit has already updated session_state["fluid_type"] with the widget value
    # We should NOT modify it here
    
    # CRITICAL: Only clear cache if user ACTUALLY changed fluid_type (not on page load)
    # The key to detecting real changes is checking if the widget value differs from
    # what was stored in persistent storage before the widget was created
    # This prevents false positives from page navigation or reruns
    fluid_type_prev_stored = st.session_state.get("fluid_type_prev", None)
    
    # Get the persistent value from _ui_state for comparison
    persistent_fluid_type_for_comparison = st.session_state[persistent_storage_key].get(fluid_type_key, "Oil")
    
    # Detect if user changed it: widget value differs from persistent storage
    # AND it's different from the stored previous value (meaning it's a real change, not initialization)
    user_changed_fluid_type = (
        fluid_type_prev_stored is not None and  # Not first load (we have a previous value)
        fluid_type != persistent_fluid_type_for_comparison and  # Widget value differs from persistent storage
        persistent_fluid_type_for_comparison == fluid_type_prev_stored  # Persistent value matches stored (was stable)
    )
    
    if user_changed_fluid_type:
        # User actively changed fluid_type - clear split GRV arrays but keep sGRV_m3_final
        # sGRV_m3_final is needed by Results page to display results
        # We only clear the split arrays (oil/gas) which will be recalculated
        # Results cache is kept so user can still view existing results (though they may be outdated)
        grv_keys_to_clear = [
            "sGRV_oil_m3", "sGRV_gas_m3",
            "direct_GRV_total_m3", "direct_GRV_oil_m3", "direct_GRV_gas_m3",
            "atgcf_GRV_total_m3", "atgcf_GRV_oil_m3", "atgcf_GRV_gas_m3",
        ]
        # NOTE: We do NOT clear:
        # - sGRV_m3_final (needed by Results page)
        # - results_cache, trial_data, df_results (user can still view existing results)
        # The results will be invalidated when they run a new simulation anyway
        for key in grv_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        # Store change timestamp for diagnostics
        st.session_state["fluid_type_changed_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.warning("⚠️ Fluid type changed. GRV split will be recalculated on next simulation run or when visiting GRV page. Existing results may be outdated.")
    
    # Always update the stored previous value for next comparison
    st.session_state["fluid_type_prev"] = fluid_type
    
    options = ["oilfield", "si"]
    default_unit = st.session_state.get("unit_system", "oilfield")
    index = options.index(default_unit) if default_unit in options else 0
    unit_system = st.sidebar.selectbox(
        "Choose your preferred unit system",
        options,
        index=index,
        key="unit_system",
        format_func=lambda x: "Oilfield (STB, scf, psia, °F)"
        if x == "oilfield"
        else "SI (m³, Pa, °C)",
        help="Select the unit system used to display inputs and results. Calculations are carried out in SI internally.",
    )

    with st.sidebar.expander("Unit Help", expanded=False):
        st.markdown(get_unit_help_text(unit_system))

    st.sidebar.markdown(
        f"""
        <div style='background-color:{PALETTE["bg_light"]};padding:7px;border-radius:8px;
            margin:6px 0;border-left:3px solid {PALETTE["accent"]};'>
            <h4 style='color:{PALETTE["primary"]};margin:0 0 5px 0;font-size:1.0rem;'>Monte Carlo Settings</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    num_sims = st.sidebar.number_input(
        "Number of simulations",
        min_value=100,
        max_value=2_000_000,
        value=int(st.session_state.get("num_sims", 10_000)),
        step=1_000,
        key="num_sims",
        help=HELP["trials"],
    )
    st.sidebar.caption("💡 Recommended: 10,000-50,000 for balance of speed and accuracy.")

    # Random seed control (also store as rng_seed for compatibility)
    random_seed = st.sidebar.number_input(
        "Random seed (optional)",
        min_value=0,
        value=int(st.session_state.get("random_seed", st.session_state.get("rng_seed", 42))),
        step=1,
        key="random_seed",
        help=HELP.get("seed", "Base random seed. Effective seed = base + run_id, so each run produces different results even with same base seed."),
    )
    # Also store as rng_seed for compatibility
    st.session_state["rng_seed"] = int(random_seed)
    st.session_state["rng"] = np.random.default_rng(int(random_seed))

    st.sidebar.checkbox(
        "Show inline tips",
        value=st.session_state.get("show_inline_tips", False),
        key="show_inline_tips",
        help="Display contextual hints inside each input section.",
    )

    if num_sims > 100_000:
        st.sidebar.warning(
            f"⚠️ Large simulation size ({num_sims:,}) may slow down the application."
        )
        st.sidebar.info(
            "💡 Consider reducing to 10,000-50,000 trials for faster turnaround."
        )

    st.sidebar.markdown(
        f"""
        <div style='background-color:{PALETTE["bg_light"]};padding:7px;border-radius:8px;
            margin:6px 0;border-left:3px solid {PALETTE["neutral"]};'>
            <h4 style='color:{PALETTE["primary"]};margin:0 0 5px 0;font-size:1.0rem;'>THR Configuration</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.number_input(
        "Gas factor (scf per BOE)",
        min_value=5_000.0,
        max_value=7_000.0,
        value=float(st.session_state.get("gas_scf_per_boe", 6_000.0)),
        step=100.0,
        key="gas_scf_per_boe",
        help=HELP["boe_factor"],
    )

    st.sidebar.markdown(
        f"""
        <div style='background-color:{PALETTE["bg_light"]};padding:7px;border-radius:8px;
            margin:6px 0;border-left:3px solid {PALETTE["neutral"]};'>
            <h4 style='color:{PALETTE["primary"]};margin:0 0 5px 0;font-size:1.0rem;'>Percentile Convention</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get current convention value to set correct index
    old_value = st.session_state.get("percentile_exceedance", True)
    current_index = 0 if old_value else 1
    
    percentile_mode = st.sidebar.radio(
        "Select P10/P90 definition:",
        options=[
            "P10 high (probability of exceedance)",
            "P10 low (probability of non-exceedance)",
        ],
        index=current_index,
        key="percentile_mode",
        help="P10 high = optimistic case (10% chance of exceeding). P10 low = conservative case (10% chance of not exceeding).",
    )
    
    if percentile_mode.startswith("P10 high"):
        new_value = True
    else:
        new_value = False
    
    # Check if convention changed and clear cached results
    if old_value != new_value:
        # Clear cached results to force recalculation
        _clear_results_on_convention_change()
        st.sidebar.info("⚠️ Percentile convention changed. Please re-run the simulation to update results.")
    
    st.session_state["percentile_exceedance"] = new_value

    st.sidebar.markdown(
        f"""
        <div style='background-color:{PALETTE["bg_light"]};padding:7px;border-radius:8px;
            margin:6px 0;border-left:3px solid {PALETTE["neutral"]};'>
            <h4 style='color:{PALETTE["primary"]};margin:0 0 5px 0;font-size:1.0rem;'>Page Layout</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_margin = st.sidebar.number_input(
        "Main content margin (%)",
        min_value=0.1,
        max_value=20.0,
        value=float(st.session_state.get("page_margin_value", 2.0)),
        step=0.1,
        format="%.1f",
        key="page_margin",
        help="Control the horizontal margin/padding of the main content area. Enter a percentage value from 0.1% to 20%.",
    )
    # Store margin value in percentage
    margin_value = f"{page_margin}%"
    st.session_state["page_margin_value"] = page_margin
    
    # Apply margin CSS dynamically
    st.markdown(
        f"""
        <style>
        /* Apply margin to main content area */
        .main .block-container {{
            padding-left: {margin_value} !important;
            padding-right: {margin_value} !important;
            max-width: calc(100% - 2 * {margin_value}) !important;
        }}
        
        /* Ensure sidebar doesn't interfere */
        [data-testid="stSidebar"] {{
            min-width: 21rem !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_help() -> None:
    """Render quick reference help and legend content in the sidebar."""
    with st.sidebar.expander("ℹ️ Help & Glossary", expanded=False):
        st.markdown("**Quick start**")
        st.caption(HELP["quick_start"])

        st.markdown("---")
        st.markdown("**GRV methods**")
        st.caption(HELP["grv_method"])
        st.caption(f"• {HELP['spill_point']}")
        st.caption(f"• {HELP['eff_hc_depth']}")
        st.caption(f"• {HELP['goc']}")

        st.markdown("---")
        st.markdown("**Fluids & PVT**")
        st.caption(f"• {HELP['bg']}")
        st.caption(f"• {HELP['invbo']}")
        st.caption(f"• {HELP['gor']}")
        st.caption(f"• {HELP['cgr']}")

        st.markdown("---")
        st.markdown("**Recovery & Simulation**")
        st.caption(f"• {HELP['rf_oil']}")
        st.caption(f"• {HELP['rf_gas']}")
        st.caption(f"• {HELP['trials']}")
        st.caption(f"• {HELP['seed']}")

        st.markdown("---")
        st.markdown("**Charts**")
        st.caption(HELP["charts"])

        st.markdown("---")
        st.markdown("**Units quick reference**")
        st.code(
            "Bg: rb/scf\n1/Bg: scf/rcf\n1/Bo: STB/rb\nGOR: scf/STB\nCGR: STB/MMscf\nGRV/PV: m³\nBOE factor: scf/BOE",
            language="text",
        )

    render_color_legend()


def get_unit_help_text(unit_system: str) -> str:
    """Get help text for unit abbreviations."""
    if unit_system == "oilfield":
        return """
        **Oilfield Units Help:**
        - **rb**: Reservoir Barrel (volume at reservoir conditions)
        - **STB**: Stock Tank Barrel (volume at surface conditions)
        - **scf**: Standard Cubic Feet (gas volume at standard conditions)
        - **Bscf**: Billion Standard Cubic Feet
        - **MMSTB**: Million Stock Tank Barrels
        - **psia**: Pounds per Square Inch Absolute (pressure)
        - **°F**: Degrees Fahrenheit (temperature)

        **Conversion Examples:**
        - 500 scf/STB = 89.05 m³/m³
        - 0.005 rb/scf = 0.028 m³/m³
        - 1.3 STB/rb = 1.3 m³/m³
        """
    return """
        **SI Units Help:**
        - **m³**: Cubic Meters (volume)
        - **Mm³**: Million Cubic Meters
        - **Bm³**: Billion Cubic Meters
        - **Pa**: Pascal (pressure)
        - **°C**: Degrees Celsius (temperature)
        - **kg/m³**: Kilograms per Cubic Meter (density)

        **Conversion Examples:**
        - 89.05 m³/m³ = 500 scf/STB
        - 0.028 m³/m³ = 0.005 rb/scf
        - 1.3 m³/m³ = 1.3 STB/rb
        """


def convert_fluid_property_value(
    value: float,
    param_name: str,
    from_unit_system: str,
    to_unit_system: str,
) -> float:
    """Convert fluid property values between unit systems."""
    if from_unit_system == to_unit_system:
        return value

    if param_name == "Bg":
        if from_unit_system == "oilfield" and to_unit_system == "si":
            return value * UNIT_CONVERSIONS["rb_per_scf_to_m3_per_m3"]
        if from_unit_system == "si" and to_unit_system == "oilfield":
            return value * UNIT_CONVERSIONS["m3_per_m3_to_rb_per_scf"]

    if param_name == "InvBo":
        if from_unit_system == "oilfield" and to_unit_system == "si":
            return value * UNIT_CONVERSIONS["stb_per_rb_to_m3_per_m3"]
        if from_unit_system == "si" and to_unit_system == "oilfield":
            return value * UNIT_CONVERSIONS["m3_per_m3_to_stb_per_rb"]

    if param_name == "GOR":
        if from_unit_system == "oilfield" and to_unit_system == "si":
            return value * UNIT_CONVERSIONS["scf_per_stb_to_m3_per_m3"]
        if from_unit_system == "si" and to_unit_system == "oilfield":
            return value * UNIT_CONVERSIONS["m3_per_m3_to_scf_per_stb"]

    return value


def get_converted_default_params(
    param_name: str,
    default_params: Dict[str, Any],
    from_unit_system: str,
    to_unit_system: str,
) -> Dict[str, Any]:
    """Get default parameters converted to the target unit system."""
    if from_unit_system == to_unit_system:
        return default_params

    converted_params = {}
    for key, value in default_params.items():
        converted_params[key] = convert_fluid_property_value(
            value, param_name, from_unit_system, to_unit_system
        )
    return converted_params


def update_progress(
    progress_bar: st.progress,
    status_text: st.empty,
    current_step: int,
    total_steps: int,
    message: str = "",
) -> None:
    """Update the progress bar and status text."""
    progress = current_step / total_steps
    progress_bar.progress(progress)
    if message:
        status_text.text(f"{message} ({current_step}/{total_steps})")
    else:
        status_text.text(f"Progress: {current_step}/{total_steps}")


def sample_dependent_parameters(
    params_config: Dict[str, Dict[str, Any]],
    dependencies: Dict[str, float],
    param_names: List[str],
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Sample dependent parameters using a conditional sampling approach."""
    dependent_samples: Dict[str, np.ndarray] = {}
    for param_name in param_names:
        if param_name not in params_config:
            raise ValueError(f"Parameter {param_name} not found in configuration")

        config = params_config[param_name]
        dist_type = config.get("dist", "PERT")

        if dist_type == "Uniform":
            low, high = config["min"], config["max"]
            samples = np.random.uniform(low, high, n_samples)
        elif dist_type == "Triangular":
            low, mode, high = config["min"], config["mode"], config["max"]
            samples = np.random.triangular(low, mode, high, n_samples)
        elif dist_type == "PERT":
            min_v, mode_v, max_v = config["min"], config["mode"], config["max"]
            lam = config.get("lam", 4.0)
            alpha = 1.0 + lam * (mode_v - min_v) / (max_v - min_v)
            beta_param = 1.0 + lam * (max_v - mode_v) / (max_v - min_v)
            samples = np.random.beta(alpha, beta_param, n_samples)
            samples = min_v + samples * (max_v - min_v)
        elif dist_type == "Lognormal":
            mean, sd = config["mean"], config["sd"]
            sigma2 = np.log(1.0 + (sd * sd) / (mean * mean))
            sigma = np.sqrt(sigma2)
            mu = np.log(mean) - 0.5 * sigma2
            samples = np.random.lognormal(mean=mu, sigma=sigma, size=n_samples)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        dependent_samples[param_name] = samples

    return apply_correlations_to_samples(
        dependent_samples, dependencies, n_samples
    )


def sample_correlated(
    params_config: Dict[str, Dict[str, Any]],
    corr_matrix: np.ndarray,
    param_names: List[str],
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Sample parameters using a correlation matrix."""
    is_valid, error_message = validate_dependency_matrix(corr_matrix, param_names)
    if not is_valid:
        raise ValueError(error_message)

    fixed_matrix = fix_correlation_matrix(corr_matrix)
    correlated = correlated_samples(
        rng_from_seed(st.session_state.get("seed", 42)),
        params_config,
        fixed_matrix,
        param_names,
        n_samples,
    )
    return correlated


def create_dependency_matrix_ui(
    param_names: List[str],
    current_dependencies: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Render dependency matrix UI and return matrix/values."""
    if current_dependencies is None:
        current_dependencies = {}

    n_params = len(param_names)
    dep_matrix = np.eye(n_params)

    st.markdown("### Dependency Matrix")
    st.markdown(
        "Define dependencies between parameters. Values range from -0.99 "
        "(strong negative) to +0.99 (strong positive)."
    )

    for i in range(n_params):
        for j in range(i + 1, n_params):
            pair_key = f"{param_names[i]}_{param_names[j]}"
            default_value = current_dependencies.get(pair_key, 0.0)
            value = st.slider(
                f"{param_names[i]} ↔ {param_names[j]}",
                min_value=-0.99,
                max_value=0.99,
                value=float(default_value),
                step=0.01,
            )
            dep_matrix[i, j] = value
            dep_matrix[j, i] = value
            current_dependencies[pair_key] = value

    return dep_matrix, current_dependencies


def create_dependency_matrix_ui_with_scatter_plots(
    param_names: List[str],
    current_dependencies: Optional[Dict[str, float]] = None,
    all_samples: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extended dependency matrix UI with scatter plots."""
    dep_matrix, dependencies = create_dependency_matrix_ui(
        param_names, current_dependencies
    )

    if all_samples:
        st.markdown("### Sample Relationships")
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                p1, p2 = param_names[i], param_names[j]
                if p1 in all_samples and p2 in all_samples:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=all_samples[p1],
                            y=all_samples[p2],
                            mode="markers",
                            marker=dict(size=4, opacity=0.6),
                            name=f"{p1} vs {p2}",
                        )
                    )
                    fig.update_layout(
                        title=f"{p1} vs {p2} Samples",
                        xaxis_title=p1,
                        yaxis_title=p2,
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    return dep_matrix, dependencies


def _clear_results_on_convention_change() -> None:
    """Clear cached results when percentile convention changes."""
    if "results_cache" in st.session_state:
        del st.session_state["results_cache"]
    if "trial_data" in st.session_state:
        del st.session_state["trial_data"]
    if "df_results" in st.session_state:
        del st.session_state["df_results"]


def _extract_param_name_from_title(title: str) -> str:
    """Extract parameter name from title for color lookup.
    
    Maps display titles to PARAM_COLORS keys.
    """
    title_lower = title.lower()
    
    # Oil-related
    if "oil" in title_lower and ("recoverable" in title_lower or "surface" in title_lower):
        return "Oil_recoverable"
    elif "oil" in title_lower and ("in-place" in title_lower or "in-situ" in title_lower or "insitu" in title_lower):
        return "Oil_inplace"
    elif "oil" in title_lower and "column" in title_lower:
        return "Oil_Column_Height"
    elif "oil" in title_lower:
        return "Oil_STB_rec"
    
    # Gas-related
    elif "gas" in title_lower and "free" in title_lower and ("recoverable" in title_lower or "surface" in title_lower):
        return "Gas_recoverable"
    elif "gas" in title_lower and "assoc" in title_lower and ("recoverable" in title_lower or "surface" in title_lower):
        return "Gas_assoc_scf_rec"
    elif "gas" in title_lower and ("in-place" in title_lower or "in-situ" in title_lower or "insitu" in title_lower):
        return "Gas_inplace"
    elif "gas" in title_lower and "column" in title_lower:
        return "Gas_Column_Height"
    elif "gas" in title_lower and "free" in title_lower:
        return "Gas_free_scf_rec"
    elif "gas" in title_lower:
        return "Gas_free_scf_rec"
    
    # Condensate-related
    elif "condensate" in title_lower or "cond" in title_lower:
        if "recoverable" in title_lower or "surface" in title_lower:
            return "Condensate_recoverable"
        else:
            return "Cond_STB_rec"
    
    # Rock/geometry
    elif "grv" in title_lower:
        if "oil" in title_lower:
            return "GRV_oil_m3"
        elif "gas" in title_lower:
            return "GRV_gas_m3"
        else:
            return "GRV_total_m3"
    elif "porosity" in title_lower or "por" in title_lower:
        return "Porosity"
    elif "ntg" in title_lower or "net-to-gross" in title_lower or "net to gross" in title_lower:
        return "NtG"
    
    # Saturation
    elif "shc" in title_lower or "s_hc" in title_lower:
        if "oil" in title_lower:
            return "Shc_oil"
        elif "gas" in title_lower:
            return "Shc_gas"
        else:
            return "Shc_global"
    elif "sw" in title_lower or "s_w" in title_lower:
        if "oil" in title_lower:
            return "Sw_oilzone"
        elif "gas" in title_lower:
            return "Sw_gaszone"
        else:
            return "Sw_global"
    
    # THR/Total
    elif "thr" in title_lower or "total hydrocarbon" in title_lower:
        return "Total_surface_BOE"
    
    # Default fallback
    return "Total_surface_BOE"


def make_hist_cdf_figure(
    data: np.ndarray,
    title: str,
    xaxis_title: str,
    color_type: str = "calculated",
) -> go.Figure:
    """Create histogram + CDF figure for inputs/results.
    
    Args:
        data: Array of data to plot
        title: Plot title (used to extract parameter name for color)
        xaxis_title: X-axis label
        color_type: If a valid PARAM_COLORS key, use it directly. Otherwise, extract from title.
    """
    # Try to use color_type as a direct key first, otherwise extract from title
    if color_type in PARAM_COLORS:
        color = color_for(color_type)
    else:
        # Extract parameter name from title for proper color mapping
        param_name = _extract_param_name_from_title(title)
        color = color_for(param_name)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=40,
            name="Histogram",
            marker=dict(color=color, opacity=0.6),
            hovertemplate="Value: %{x:.4f}<br>Freq: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    sorted_data = np.sort(data)
    # Check percentile convention for CDF display
    use_exceedance = st.session_state.get("percentile_exceedance", True)
    if use_exceedance:
        # For exceedance convention: show probability of exceeding (1 - CDF)
        cdf = np.linspace(1, 0, len(sorted_data))  # Reversed: 1 to 0
        cdf_label = "Probability of Exceedance"
    else:
        # For non-exceedance convention: show standard CDF (probability of not exceeding)
        cdf = np.linspace(0, 1, len(sorted_data))  # Standard: 0 to 1
        cdf_label = "Cumulative Probability"
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=cdf,
            name=cdf_label,
            mode="lines",
            line=dict(color=color, width=3),
        ),
        secondary_y=True,
    )
    
    # Add P10/P50/P90 vertical lines using convention-aware values
    stats = summarize_array(data)
    if stats:
        for label in ("P10", "P50", "P90"):
            value = stats.get(label)
            if value is not None:
                fig.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.75,
                    annotation_text=f"{label}: {value:.2f}",
                    annotation_position="top",
                )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        bargap=0.02,
        bargroupgap=0.05,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        # Enable MathJax for LaTeX rendering in titles and labels
        # Plotly supports LaTeX with $...$ syntax when MathJax is available
        # Streamlit's plotly_chart should include MathJax by default
    )
    # Enable LaTeX for axes - Plotly supports $...$ syntax
    fig.update_xaxes(
        title_text=xaxis_title,
        title_font=dict(size=12),
    )
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    # Update secondary y-axis label based on convention
    use_exceedance = st.session_state.get("percentile_exceedance", True)
    if use_exceedance:
        fig.update_yaxes(title_text="Probability of Exceedance", secondary_y=True, range=[0, 1])
    else:
        fig.update_yaxes(title_text="Cumulative Probability", secondary_y=True, range=[0, 1])
    
    # Note: Plotly should render LaTeX automatically if MathJax is loaded
    # Streamlit's plotly_chart component includes MathJax, so $...$ syntax should work
    # If LaTeX is not rendering, it may be a browser/Streamlit version issue
    return fig


def calculate_grv_from_depth_table(df_or_depths, contact_depth, method="top_base", hc_depth=None, goc_depth=None):
    """
    Calculate GRV from depth table (DataFrame) or arrays using v3-compatible logic.
    
    Args:
        df_or_depths: DataFrame with depth/area columns OR array of depths
        contact_depth: Contact depth in meters (used as OWC or spill depending on context)
        method: "top_base" (use top and base areas) or "top_only" (use top area only)
        hc_depth: Optional HC depth (OWC) - if None, contact_depth is used as OWC
        goc_depth: Optional GOC depth for splitting
        
    Returns:
        If DataFrame input: float (GRV_total_m3 in m³)
        If array input: dict with GRV_total_m3, GRV_oil_m3, GRV_gas_m3
    """
    from scopehc.geom import grv_by_depth_v3_compatible
    
    # Handle DataFrame input
    if isinstance(df_or_depths, pd.DataFrame):
        df = df_or_depths.copy()
        
        # Extract depth column (try different possible names)
        depth_col = None
        # Try common depth column names in order of preference
        depth_candidates = [
            "Depth", "Depth (m)", "Depth_top (m)", "Depth_base (m)", 
            "depth", "depth (m)", "depth_top (m)", "depth_base (m)"
        ]
        for candidate in depth_candidates:
            if candidate in df.columns:
                depth_col = candidate
                break
        
        if depth_col is None:
            # Try to find any column with "depth" in name
            for col in df.columns:
                col_lower = col.lower()
                if "depth" in col_lower:
                    depth_col = col
                    break
        
        if depth_col is None:
            # Last resort: try to find first numeric column
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    depth_col = col
                    break
        
        if depth_col is None:
            raise ValueError(f"Could not find depth column in DataFrame. Available columns: {list(df.columns)}")
        
        depths_m = np.asarray(df[depth_col].values, dtype=float)
        
        # Extract area columns based on method
        if method == "top_base":
            # Use both top and base areas
            top_area_col = None
            base_area_col = None
            for col in df.columns:
                if "top" in col.lower() and "area" in col.lower():
                    top_area_col = col
                if "base" in col.lower() and "area" in col.lower():
                    base_area_col = col
            
            if top_area_col is None or base_area_col is None:
                raise ValueError(f"Could not find top/base area columns for method '{method}'")
            
            # Use average of top and base areas
            areas_km2 = (df[top_area_col].values + df[base_area_col].values) / 2.0
        elif method == "top_only":
            # Use only top area
            top_area_col = None
            for col in df.columns:
                if "top" in col.lower() and "area" in col.lower():
                    top_area_col = col
                    break
            
            if top_area_col is None:
                raise ValueError(f"Could not find top area column for method '{method}'")
            
            areas_km2 = df[top_area_col].values
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Determine top structure (minimum depth in table)
        top_structure_m = float(np.min(depths_m))
        
        # Determine OWC (hc_depth if provided, otherwise contact_depth)
        owc_m = float(hc_depth) if hc_depth is not None else float(contact_depth)
        
        # Use v3-compatible function
        result = grv_by_depth_v3_compatible(
            depth_m=depths_m,
            area_m2=areas_km2,  # Will be auto-converted if < 100
            top_structure_m=top_structure_m,
            goc_m=float(goc_depth) if goc_depth is not None else None,
            owc_m=owc_m,
            spill_m=float(contact_depth)  # Use contact_depth as spill limit
        )
        
        return result['GRV_total_m3']
    
    else:
        # Handle array input (original function signature)
        depths_m = df_or_depths
        areas_km2 = contact_depth  # In original signature, this is areas_km2
        spill_depth_m = method  # In original signature, this is spill_depth_m
        hc_depth_m = hc_depth
        goc_depth_m = goc_depth
        
        # Call v3-compatible function from geom.py
        from scopehc.geom import calculate_grv_from_depth_table as _calc_grv
        return _calc_grv(depths_m, areas_km2, spill_depth_m, hc_depth_m, goc_depth_m)


def calculate_grv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate GRV statistics from depth tables."""
    df = df.copy()
    df["Depth_mid"] = (df["Depth_top (m)"] + df["Depth_base (m)"]) / 2
    df["Thickness"] = df["Depth_base (m)"] - df["Depth_top (m)"]
    df["Volume_km2m"] = df["Area (km2)"] * df["Thickness"]
    df["Volume_m3"] = df["Volume_km2m"] * 1_000_000
    return df


def compute_dgrv_top_plus_thickness(
    depths: np.ndarray,
    top_areas: np.ndarray,
    thickness: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dGRV using top area plus constant thickness."""
    top_interp = np.interp(depths, depths, top_areas)
    base_interp = np.maximum(top_interp - thickness, 0.0)
    dgrv = top_interp - base_interp
    return top_interp, dgrv


def compute_dgrv_top_base_table(
    depths: np.ndarray,
    top_areas: np.ndarray,
    base_areas: np.ndarray,
) -> np.ndarray:
    """Compute dGRV using top/base area tables."""
    top_interp = np.interp(depths, depths, top_areas)
    base_interp = np.interp(depths, depths, base_areas)
    return top_interp - base_interp


def get_unit_system() -> str:
    """Get the current unit system from session state."""
    return st.session_state.get("unit_system", "oilfield")


def _get_dist_prefix(dist_name: str) -> str:
    """Get the prefix for a distribution type (e.g., 'PERT' -> 'pert', 'Triangular' -> 'tri')."""
    mapping = {
        "PERT": "pert",
        "Triangular": "tri",
        "Uniform": "uni",
        "Lognormal (mean, sd)": "ln",
        "Subjective Beta (Vose)": "sb",
        "Stretched Beta": "stb",
        "Truncated Normal": "tn",
        "Truncated Lognormal": "tln",
        "Burr (c, d, scale)": "burr",
        "Johnson SU": "jsu",
        "Constant": "const",
    }
    return mapping.get(dist_name, "pert")


# ============================================================================
# CONTROLLED PARAMETER STORAGE SYSTEM
# ============================================================================
# This system provides complete control over parameter persistence,
# independent of Streamlit's widget state management.

def _init_param_storage(name_key: str, default_dist: str, default_params: Dict[str, Any], context: str = "") -> None:
    """
    Initialize the controlled parameter storage structure.
    
    Creates: st.session_state["_param_values"][param_key][dist_type][param_name] = value
    """
    storage_key = "_param_values"
    if storage_key not in st.session_state:
        st.session_state[storage_key] = {}
    
    param_key = f"{name_key}{f'_{context}' if context else ''}"
    
    # Initialize parameter entry if it doesn't exist
    if param_key not in st.session_state[storage_key]:
        param_storage = {
            "dist": default_dist,
        }
        
        # Initialize all distribution types with defaults
        # This ensures values exist for all distributions, so switching doesn't lose data
        
        # Helper to get default value
        def get_def(key: str, fallback: float = 0.0) -> float:
            val = default_params.get(key)
            if val is None:
                return fallback
            try:
                return float(val)
            except (ValueError, TypeError):
                return fallback
        
        # PERT / Triangular / Stretched Beta (min, mode, max)
        for dist_name in ["PERT", "Triangular", "Stretched Beta"]:
            param_storage[dist_name] = {
                "min": get_def("min", 0.0),
                "mode": get_def("mode", get_def("min", 0.0)),
                "max": get_def("max", get_def("mode", get_def("min", 1.0))),
            }
        
        # Uniform (min, max)
        param_storage["Uniform"] = {
            "min": get_def("min", 0.0),
            "max": get_def("max", get_def("min", 1.0)),
        }
        
        # Lognormal (mean, sd)
        param_storage["Lognormal (mean, sd)"] = {
            "mean": get_def("mean", 1.0),
            "sd": get_def("sd", 0.1),
        }
        
        # Subjective Beta (min, max, p10, p50, p90)
        default_min = get_def("min", 0.0)
        default_max = get_def("max", max(default_min + 1.0, default_min))
        param_storage["Subjective Beta (Vose)"] = {
            "min": default_min,
            "max": default_max,
            "p10": get_def("p10", default_min),
            "p50": get_def("p50", (default_min + default_max) / 2),
            "p90": get_def("p90", default_max),
        }
        
        # Truncated Normal (mean, sd, min, max)
        default_mean = get_def("mean", 0.0)
        default_sd = get_def("sd", 1.0)
        param_storage["Truncated Normal"] = {
            "mean": default_mean,
            "sd": default_sd,
            "min": get_def("min", default_mean - default_sd),
            "max": get_def("max", default_mean + default_sd),
        }
        
        # Truncated Lognormal (mean, sd, min, max)
        default_mean_ln = get_def("mean", 1.0)
        default_sd_ln = get_def("sd", 0.2)
        param_storage["Truncated Lognormal"] = {
            "mean": default_mean_ln,
            "sd": default_sd_ln,
            "min": get_def("min", max(default_mean_ln * 0.5, 1e-6)),
            "max": get_def("max", default_mean_ln * 2.0),
        }
        
        # Burr (c, d, scale, loc)
        param_storage["Burr (c, d, scale)"] = {
            "c": get_def("c", 2.0),
            "d": get_def("d", 4.0),
            "scale": get_def("scale", 1.0),
            "loc": get_def("loc", 0.0),
        }
        
        # Johnson SU (gamma, delta, loc, scale)
        param_storage["Johnson SU"] = {
            "gamma": get_def("gamma", 0.0),
            "delta": get_def("delta", 1.0),
            "loc": get_def("loc", 0.0),
            "scale": get_def("scale", 1.0),
        }
        
        # Constant (value)
        param_storage["Constant"] = {
            "value": get_def("value", get_def("mode", get_def("mean", 0.0))),
        }
        
        st.session_state[storage_key][param_key] = param_storage


def _get_param_value(name_key: str, dist_type: str, param_name: str, default: float, context: str = "") -> float:
    """
    Get parameter value from controlled storage.
    
    This is the SINGLE SOURCE OF TRUTH for parameter values.
    """
    param_key = f"{name_key}{f'_{context}' if context else ''}"
    try:
        return float(st.session_state["_param_values"][param_key][dist_type][param_name])
    except (KeyError, TypeError, ValueError):
        return default


def _set_param_value(name_key: str, dist_type: str, param_name: str, value: float, context: str = ""):
    """
    Set parameter value in controlled storage.
    
    This is the ONLY place parameter values should be written.
    """
    param_key = f"{name_key}{f'_{context}' if context else ''}"
    
    # Ensure storage structure exists
    if "_param_values" not in st.session_state:
        st.session_state["_param_values"] = {}
    if param_key not in st.session_state["_param_values"]:
        st.session_state["_param_values"][param_key] = {}
    if dist_type not in st.session_state["_param_values"][param_key]:
        st.session_state["_param_values"][param_key][dist_type] = {}
    
    st.session_state["_param_values"][param_key][dist_type][param_name] = float(value)


def _get_param_dist(name_key: str, default_dist: str, context: str = "") -> str:
    """Get current distribution type for a parameter."""
    param_key = f"{name_key}{f'_{context}' if context else ''}"
    try:
        return st.session_state["_param_values"][param_key]["dist"]
    except (KeyError, TypeError):
        return default_dist


def _set_param_dist(name_key: str, dist_type: str, context: str = ""):
    """Set distribution type for a parameter."""
    param_key = f"{name_key}{f'_{context}' if context else ''}"
    if "_param_values" not in st.session_state:
        st.session_state["_param_values"] = {}
    if param_key not in st.session_state["_param_values"]:
        st.session_state["_param_values"][param_key] = {}
    st.session_state["_param_values"][param_key]["dist"] = dist_type


def initialize_parameter_defaults(
    name_key: str,
    default_dist: str,
    default_params: Dict[str, Any],
    context: Optional[str] = None,
) -> None:
    """
    Initialize ALL session_state keys for a parameter BEFORE widgets are created.
    
    This ensures values persist across page navigation and distribution type changes.
    Keys are initialized for ALL distributions, not just the current one, so switching
    distributions doesn't lose values.
    
    Parameters
    ----------
    name_key : str
        Parameter name (e.g., "A", "Por", "GCF")
    default_dist : str
        Default distribution type (e.g., "PERT", "Triangular")
    default_params : Dict[str, Any]
        Default parameter values (e.g., {"min": 10, "mode": 20, "max": 30})
    context : Optional[str]
        Optional context suffix (e.g., "_oilzone")
    """
    key_suffix = f"_{context}" if context else ""
    
    # Initialize distribution type key
    # CRITICAL: Only set if it doesn't exist - NEVER overwrite user's selection
    dist_key = f"dist_{name_key}{key_suffix}"
    if dist_key not in st.session_state:
        st.session_state[dist_key] = default_dist
    # If it exists, leave it alone - preserve user's choice
    
    # Helper to safely get float value from defaults
    def get_default(key: str, fallback: float = 0.0) -> float:
        val = default_params.get(key)
        if val is None:
            return fallback
        try:
            return float(val)
        except (ValueError, TypeError):
            return fallback
    
    # Initialize keys for ALL distributions (so switching doesn't lose values)
    
    # Constant
    const_key = f"{name_key}_const{key_suffix}"
    if const_key not in st.session_state:
        default_val = get_default("value", get_default("mode", get_default("mean", 0.0)))
        st.session_state[const_key] = default_val
    
    # PERT (min, mode, max)
    # CRITICAL: Only initialize if key doesn't exist - NEVER overwrite user changes
    for suffix in ["min", "mode", "max"]:
        key = f"{name_key}_pert_{suffix}{key_suffix}"
        if key not in st.session_state:
            # Key doesn't exist - safe to set default
            if suffix == "min":
                default_val = get_default("min", 0.0)
            elif suffix == "mode":
                default_val = get_default("mode", get_default("min", 0.0))
            else:  # max
                default_val = get_default("max", get_default("mode", get_default("min", 1.0)))
            st.session_state[key] = default_val
        # If key exists, DO NOTHING - preserve user's value
    
    # Triangular (min, mode, max) - same structure as PERT
    for suffix in ["min", "mode", "max"]:
        key = f"{name_key}_tri_{suffix}{key_suffix}"
        if key not in st.session_state:
            if suffix == "min":
                default_val = get_default("min", 0.0)
            elif suffix == "mode":
                default_val = get_default("mode", get_default("min", 0.0))
            else:  # max
                default_val = get_default("max", get_default("mode", get_default("min", 1.0)))
            st.session_state[key] = default_val
    
    # Uniform (min, max)
    for suffix in ["min", "max"]:
        key = f"{name_key}_uni_{suffix}{key_suffix}"
        if key not in st.session_state:
            if suffix == "min":
                default_val = get_default("min", 0.0)
            else:  # max
                default_val = get_default("max", get_default("min", 1.0))
            st.session_state[key] = default_val
    
    # Lognormal (mean, sd)
    mean_key = f"{name_key}_ln_mean{key_suffix}"
    if mean_key not in st.session_state:
        st.session_state[mean_key] = get_default("mean", 1.0)
    sd_key = f"{name_key}_ln_sd{key_suffix}"
    if sd_key not in st.session_state:
        st.session_state[sd_key] = get_default("sd", 0.1)
    
    # Subjective Beta (min, max, p10, p50, p90)
    sb_min_key = f"{name_key}_sb_min{key_suffix}"
    if sb_min_key not in st.session_state:
        st.session_state[sb_min_key] = get_default("min", 0.0)
    sb_max_key = f"{name_key}_sb_max{key_suffix}"
    if sb_max_key not in st.session_state:
        default_min = st.session_state.get(sb_min_key, 0.0)
        st.session_state[sb_max_key] = get_default("max", max(default_min + 1.0, default_min))
    for suffix in ["p10", "p50", "p90"]:
        key = f"{name_key}_sb_{suffix}{key_suffix}"
        if key not in st.session_state:
            default_min = st.session_state.get(sb_min_key, 0.0)
            default_max = st.session_state.get(sb_max_key, default_min + 1.0)
            if suffix == "p10":
                default_val = get_default("p10", default_min)
            elif suffix == "p50":
                default_val = get_default("p50", (default_min + default_max) / 2)
            else:  # p90
                default_val = get_default("p90", default_max)
            st.session_state[key] = default_val
    
    # Stretched Beta (min, mode, max) - same as PERT
    for suffix in ["min", "mode", "max"]:
        key = f"{name_key}_stb_{suffix}{key_suffix}"
        if key not in st.session_state:
            if suffix == "min":
                default_val = get_default("min", 0.0)
            elif suffix == "mode":
                default_val = get_default("mode", get_default("min", 0.0))
            else:  # max
                default_val = get_default("max", get_default("mode", get_default("min", 1.0)))
            st.session_state[key] = default_val
    
    # Truncated Normal (mean, sd, min, max)
    tn_mean_key = f"{name_key}_tn_mean{key_suffix}"
    if tn_mean_key not in st.session_state:
        st.session_state[tn_mean_key] = get_default("mean", 0.0)
    tn_sd_key = f"{name_key}_tn_sd{key_suffix}"
    if tn_sd_key not in st.session_state:
        st.session_state[tn_sd_key] = get_default("sd", 1.0)
    tn_min_key = f"{name_key}_tn_min{key_suffix}"
    if tn_min_key not in st.session_state:
        default_mean = st.session_state.get(tn_mean_key, 0.0)
        default_sd = st.session_state.get(tn_sd_key, 1.0)
        st.session_state[tn_min_key] = get_default("min", default_mean - default_sd)
    tn_max_key = f"{name_key}_tn_max{key_suffix}"
    if tn_max_key not in st.session_state:
        default_mean = st.session_state.get(tn_mean_key, 0.0)
        default_sd = st.session_state.get(tn_sd_key, 1.0)
        st.session_state[tn_max_key] = get_default("max", default_mean + default_sd)
    
    # Truncated Lognormal (mean, sd, min, max)
    tln_mean_key = f"{name_key}_tln_mean{key_suffix}"
    if tln_mean_key not in st.session_state:
        st.session_state[tln_mean_key] = get_default("mean", 1.0)
    tln_sd_key = f"{name_key}_tln_sd{key_suffix}"
    if tln_sd_key not in st.session_state:
        st.session_state[tln_sd_key] = get_default("sd", 0.2)
    tln_min_key = f"{name_key}_tln_min{key_suffix}"
    if tln_min_key not in st.session_state:
        default_mean = st.session_state.get(tln_mean_key, 1.0)
        st.session_state[tln_min_key] = get_default("min", max(default_mean * 0.5, 1e-6))
    tln_max_key = f"{name_key}_tln_max{key_suffix}"
    if tln_max_key not in st.session_state:
        default_mean = st.session_state.get(tln_mean_key, 1.0)
        st.session_state[tln_max_key] = get_default("max", default_mean * 2.0)
    
    # Burr (c, d, scale, loc)
    for suffix in ["c", "d", "scale", "loc"]:
        key = f"{name_key}_burr_{suffix}{key_suffix}"
        if key not in st.session_state:
            if suffix == "c":
                default_val = get_default("c", 2.0)
            elif suffix == "d":
                default_val = get_default("d", 4.0)
            elif suffix == "scale":
                default_val = get_default("scale", 1.0)
            else:  # loc
                default_val = get_default("loc", 0.0)
            st.session_state[key] = default_val
    
    # Johnson SU (gamma, delta, loc, scale)
    for suffix in ["gamma", "delta", "loc", "scale"]:
        key = f"{name_key}_jsu_{suffix}{key_suffix}"
        if key not in st.session_state:
            if suffix == "gamma":
                default_val = get_default("gamma", 0.0)
            elif suffix == "delta":
                default_val = get_default("delta", 1.0)
            elif suffix == "loc":
                default_val = get_default("loc", 0.0)
            else:  # scale
                default_val = get_default("scale", 1.0)
            st.session_state[key] = default_val


def render_param(
    name_key: str,
    label: str,
    unit_hint: str,
    default_dist: str,
    default_params: Dict[str, Any],
    n_samples: int,
    plot_unit_label: Optional[str] = None,
    stats_decimals: Optional[int] = None,
    display_scale: float = 1.0,
    help_text: Optional[str] = None,
    param_name: Optional[str] = None,
    context: Optional[str] = None,
) -> np.ndarray:
    """Render a parameter input widget with distribution controls."""
    rng = st.session_state.get("rng", np.random.default_rng(12345))

    current_unit_system = get_unit_system()
    converted_params = default_params

    if param_name and param_name in ["Bg", "InvBo", "GOR"]:
        prev_unit_system_key = f"prev_unit_system_{name_key}"
        if prev_unit_system_key in st.session_state:
            prev_unit_system = st.session_state[prev_unit_system_key]
            if prev_unit_system != current_unit_system:
                converted_params = get_converted_default_params(
                    param_name, default_params, prev_unit_system, current_unit_system
                )
        st.session_state[prev_unit_system_key] = current_unit_system

    # CRITICAL: Initialize controlled parameter storage BEFORE creating widgets
    # This is our single source of truth - completely independent of Streamlit's widget state
    _init_param_storage(name_key, default_dist, converted_params, context or "")
    
    # Get current distribution from controlled storage
    current_dist = _get_param_dist(name_key, default_dist, context or "")

    col1, col2 = st.columns([1.3, 2.2])
    key_suffix = f"_{context}" if context else ""

    with col1:
        # Get current distribution from controlled storage
        current_dist = _get_param_dist(name_key, default_dist, context or "")
        
        # Calculate index for selectbox
        try:
            default_index = DistributionChoiceWithConstant.index(current_dist)
        except ValueError:
            default_index = DistributionChoiceWithConstant.index(default_dist)
            _set_param_dist(name_key, default_dist, context or "")
            current_dist = default_dist
        
        # Use temporary widget key to avoid conflicts
        temp_dist_key = f"_widget_dist_{name_key}{key_suffix}"
        
        # Sync temp key from controlled storage
        st.session_state[temp_dist_key] = current_dist
        
        # Callback to save distribution change to controlled storage
        def sync_dist():
            new_dist = st.session_state[temp_dist_key]
            _set_param_dist(name_key, new_dist, context or "")
        
        # Create selectbox with temp key
        dist = st.selectbox(
            f"{label} distribution",
            DistributionChoiceWithConstant,
            index=default_index,
            key=temp_dist_key,
            on_change=sync_dist,
            help="Choose the distribution type for this parameter",
        )
        
        # Explicitly sync after widget (backup)
        _set_param_dist(name_key, st.session_state[temp_dist_key], context or "")
        
        # Get final distribution from controlled storage
        dist = _get_param_dist(name_key, default_dist, context or "")

    params: Dict[str, Any] = {}
    with col2:
        format_str = "%.4f" if param_name and param_name in ["Bg", "InvBo", "GOR"] else None

        if dist == "Constant":
            current_value = _get_param_value(name_key, "Constant", "value", float(converted_params.get("value", converted_params.get("mode", converted_params.get("mean", 0.0)))), context or "")
            temp_const_key = f"_widget_{name_key}_const{key_suffix}"
            st.session_state[temp_const_key] = current_value
            
            def sync_const():
                _set_param_value(name_key, "Constant", "value", st.session_state[temp_const_key], context or "")
            
            const_val = st.number_input(
                f"{label} value ({unit_hint})",
                value=current_value,
                key=temp_const_key,
                on_change=sync_const,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Constant", "value", st.session_state[temp_const_key], context or "")
            params = {"value": _get_param_value(name_key, "Constant", "value", 0.0, context or "")}
        elif dist == "PERT":
            # CRITICAL: Use controlled storage system - single source of truth
            # Read values from controlled storage
            current_min = _get_param_value(name_key, "PERT", "min", float(converted_params.get("min", 0.0)), context or "")
            current_mode = _get_param_value(name_key, "PERT", "mode", float(converted_params.get("mode", current_min)), context or "")
            current_max = _get_param_value(name_key, "PERT", "max", float(converted_params.get("max", current_mode)), context or "")
            
            # Use temporary widget keys to avoid conflicts
            temp_min_key = f"_widget_{name_key}_pert_min{key_suffix}"
            temp_mode_key = f"_widget_{name_key}_pert_mode{key_suffix}"
            temp_max_key = f"_widget_{name_key}_pert_max{key_suffix}"
            
            # Sync temp keys from controlled storage
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_mode_key] = current_mode
            st.session_state[temp_max_key] = current_max
            
            # Callbacks to save to controlled storage
            def sync_min():
                _set_param_value(name_key, "PERT", "min", st.session_state[temp_min_key], context or "")
            def sync_mode():
                _set_param_value(name_key, "PERT", "mode", st.session_state[temp_mode_key], context or "")
            def sync_max():
                _set_param_value(name_key, "PERT", "max", st.session_state[temp_max_key], context or "")
            
            # Create widgets with temp keys
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            # Explicit sync after widget (backup)
            _set_param_value(name_key, "PERT", "min", st.session_state[temp_min_key], context or "")
            
            mode_v = st.number_input(
                f"{label} mode ({unit_hint})",
                value=current_mode,
                key=temp_mode_key,
                on_change=sync_mode,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "PERT", "mode", st.session_state[temp_mode_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "PERT", "max", st.session_state[temp_max_key], context or "")
            
            # CRITICAL: Read params from controlled storage (single source of truth)
            params = {
                "min": _get_param_value(name_key, "PERT", "min", 0.0, context or ""),
                "mode": _get_param_value(name_key, "PERT", "mode", 0.0, context or ""),
                "max": _get_param_value(name_key, "PERT", "max", 0.0, context or ""),
            }
            
            # CRITICAL: Mark that parameters changed so samples will be recalculated
            # This ensures results update when values change
            ss_conf_key = f"conf_{name_key}{key_suffix}"
            if ss_conf_key in st.session_state:
                # Invalidate cached config to force recalculation
                old_conf = st.session_state[ss_conf_key]
                if old_conf.get("min") != params["min"] or old_conf.get("mode") != params["mode"] or old_conf.get("max") != params["max"]:
                    # Parameters changed - clear samples cache
                    ss_samples_key = f"samples_{name_key}{key_suffix}"
                    if ss_samples_key in st.session_state:
                        del st.session_state[ss_samples_key]
        elif dist == "Triangular":
            # CRITICAL: Use controlled storage system - single source of truth
            current_min = _get_param_value(name_key, "Triangular", "min", float(converted_params.get("min", 0.0)), context or "")
            current_mode = _get_param_value(name_key, "Triangular", "mode", float(converted_params.get("mode", current_min)), context or "")
            current_max = _get_param_value(name_key, "Triangular", "max", float(converted_params.get("max", current_mode)), context or "")
            
            temp_min_key = f"_widget_{name_key}_tri_min{key_suffix}"
            temp_mode_key = f"_widget_{name_key}_tri_mode{key_suffix}"
            temp_max_key = f"_widget_{name_key}_tri_max{key_suffix}"
            
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_mode_key] = current_mode
            st.session_state[temp_max_key] = current_max
            
            def sync_min():
                _set_param_value(name_key, "Triangular", "min", st.session_state[temp_min_key], context or "")
            def sync_mode():
                _set_param_value(name_key, "Triangular", "mode", st.session_state[temp_mode_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Triangular", "max", st.session_state[temp_max_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Triangular", "min", st.session_state[temp_min_key], context or "")
            
            mode_v = st.number_input(
                f"{label} mode ({unit_hint})",
                value=current_mode,
                key=temp_mode_key,
                on_change=sync_mode,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Triangular", "mode", st.session_state[temp_mode_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Triangular", "max", st.session_state[temp_max_key], context or "")
            
            params = {
                "min": _get_param_value(name_key, "Triangular", "min", 0.0, context or ""),
                "mode": _get_param_value(name_key, "Triangular", "mode", 0.0, context or ""),
                "max": _get_param_value(name_key, "Triangular", "max", 0.0, context or ""),
            }
        elif dist == "Uniform":
            current_min = _get_param_value(name_key, "Uniform", "min", float(converted_params.get("min", 0.0)), context or "")
            current_max = _get_param_value(name_key, "Uniform", "max", float(converted_params.get("max", current_min)), context or "")
            
            temp_min_key = f"_widget_{name_key}_uni_min{key_suffix}"
            temp_max_key = f"_widget_{name_key}_uni_max{key_suffix}"
            
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_max_key] = current_max
            
            def sync_min():
                _set_param_value(name_key, "Uniform", "min", st.session_state[temp_min_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Uniform", "max", st.session_state[temp_max_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Uniform", "min", st.session_state[temp_min_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Uniform", "max", st.session_state[temp_max_key], context or "")
            
            params = {
                "min": _get_param_value(name_key, "Uniform", "min", 0.0, context or ""),
                "max": _get_param_value(name_key, "Uniform", "max", 0.0, context or ""),
            }
        elif dist == "Lognormal (mean, sd)":
            current_mean = _get_param_value(name_key, "Lognormal (mean, sd)", "mean", float(converted_params.get("mean", 1.0)), context or "")
            current_sd = _get_param_value(name_key, "Lognormal (mean, sd)", "sd", float(converted_params.get("sd", 0.1)), context or "")
            
            temp_mean_key = f"_widget_{name_key}_ln_mean{key_suffix}"
            temp_sd_key = f"_widget_{name_key}_ln_sd{key_suffix}"
            
            st.session_state[temp_mean_key] = current_mean
            st.session_state[temp_sd_key] = current_sd
            
            def sync_mean():
                _set_param_value(name_key, "Lognormal (mean, sd)", "mean", st.session_state[temp_mean_key], context or "")
            def sync_sd():
                _set_param_value(name_key, "Lognormal (mean, sd)", "sd", st.session_state[temp_sd_key], context or "")
            
            mean_v = st.number_input(
                f"{label} arithmetic mean ({unit_hint})",
                value=current_mean,
                key=temp_mean_key,
                on_change=sync_mean,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Lognormal (mean, sd)", "mean", st.session_state[temp_mean_key], context or "")
            
            sd_v = st.number_input(
                f"{label} arithmetic sd ({unit_hint})",
                value=current_sd,
                key=temp_sd_key,
                on_change=sync_sd,
                min_value=0.0,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Lognormal (mean, sd)", "sd", st.session_state[temp_sd_key], context or "")
            
            params = {
                "mean": _get_param_value(name_key, "Lognormal (mean, sd)", "mean", 0.0, context or ""),
                "sd": _get_param_value(name_key, "Lognormal (mean, sd)", "sd", 0.0, context or ""),
            }
        elif dist == "Subjective Beta (Vose)":
            current_min = _get_param_value(name_key, "Subjective Beta (Vose)", "min", float(converted_params.get("min", 0.0)), context or "")
            default_max_val = max(current_min + 1.0, current_min)
            current_max = _get_param_value(name_key, "Subjective Beta (Vose)", "max", float(converted_params.get("max", default_max_val)), context or "")
            current_p10 = _get_param_value(name_key, "Subjective Beta (Vose)", "p10", float(converted_params.get("p10", current_min)), context or "")
            current_p50 = _get_param_value(name_key, "Subjective Beta (Vose)", "p50", float(converted_params.get("p50", (current_min + current_max) / 2)), context or "")
            current_p90 = _get_param_value(name_key, "Subjective Beta (Vose)", "p90", float(converted_params.get("p90", current_max)), context or "")
            
            temp_min_key = f"_widget_{name_key}_sb_min{key_suffix}"
            temp_max_key = f"_widget_{name_key}_sb_max{key_suffix}"
            temp_p10_key = f"_widget_{name_key}_sb_p10{key_suffix}"
            temp_p50_key = f"_widget_{name_key}_sb_p50{key_suffix}"
            temp_p90_key = f"_widget_{name_key}_sb_p90{key_suffix}"
            
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_max_key] = current_max
            st.session_state[temp_p10_key] = current_p10
            st.session_state[temp_p50_key] = current_p50
            st.session_state[temp_p90_key] = current_p90
            
            def sync_min():
                _set_param_value(name_key, "Subjective Beta (Vose)", "min", st.session_state[temp_min_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Subjective Beta (Vose)", "max", st.session_state[temp_max_key], context or "")
            def sync_p10():
                _set_param_value(name_key, "Subjective Beta (Vose)", "p10", st.session_state[temp_p10_key], context or "")
            def sync_p50():
                _set_param_value(name_key, "Subjective Beta (Vose)", "p50", st.session_state[temp_p50_key], context or "")
            def sync_p90():
                _set_param_value(name_key, "Subjective Beta (Vose)", "p90", st.session_state[temp_p90_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Subjective Beta (Vose)", "min", st.session_state[temp_min_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Subjective Beta (Vose)", "max", st.session_state[temp_max_key], context or "")
            
            p10_v = st.number_input(
                f"{label} P10 ({unit_hint})",
                value=current_p10,
                key=temp_p10_key,
                on_change=sync_p10,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Subjective Beta (Vose)", "p10", st.session_state[temp_p10_key], context or "")
            
            p50_v = st.number_input(
                f"{label} P50 ({unit_hint})",
                value=current_p50,
                key=temp_p50_key,
                on_change=sync_p50,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Subjective Beta (Vose)", "p50", st.session_state[temp_p50_key], context or "")
            
            p90_v = st.number_input(
                f"{label} P90 ({unit_hint})",
                value=current_p90,
                key=temp_p90_key,
                on_change=sync_p90,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Subjective Beta (Vose)", "p90", st.session_state[temp_p90_key], context or "")
            
            params = {
                "min": _get_param_value(name_key, "Subjective Beta (Vose)", "min", 0.0, context or ""),
                "max": _get_param_value(name_key, "Subjective Beta (Vose)", "max", 0.0, context or ""),
                "p10": _get_param_value(name_key, "Subjective Beta (Vose)", "p10", 0.0, context or ""),
                "p50": _get_param_value(name_key, "Subjective Beta (Vose)", "p50", 0.0, context or ""),
                "p90": _get_param_value(name_key, "Subjective Beta (Vose)", "p90", 0.0, context or ""),
            }
        elif dist == "Stretched Beta":
            current_min = _get_param_value(name_key, "Stretched Beta", "min", float(converted_params.get("min", 0.0)), context or "")
            current_mode = _get_param_value(name_key, "Stretched Beta", "mode", float(converted_params.get("mode", current_min)), context or "")
            current_max = _get_param_value(name_key, "Stretched Beta", "max", float(converted_params.get("max", current_mode)), context or "")
            
            temp_min_key = f"_widget_{name_key}_stb_min{key_suffix}"
            temp_mode_key = f"_widget_{name_key}_stb_mode{key_suffix}"
            temp_max_key = f"_widget_{name_key}_stb_max{key_suffix}"
            
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_mode_key] = current_mode
            st.session_state[temp_max_key] = current_max
            
            def sync_min():
                _set_param_value(name_key, "Stretched Beta", "min", st.session_state[temp_min_key], context or "")
            def sync_mode():
                _set_param_value(name_key, "Stretched Beta", "mode", st.session_state[temp_mode_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Stretched Beta", "max", st.session_state[temp_max_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Stretched Beta", "min", st.session_state[temp_min_key], context or "")
            
            mode_v = st.number_input(
                f"{label} mode ({unit_hint})",
                value=current_mode,
                key=temp_mode_key,
                on_change=sync_mode,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Stretched Beta", "mode", st.session_state[temp_mode_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Stretched Beta", "max", st.session_state[temp_max_key], context or "")
            
            params = {
                "min": _get_param_value(name_key, "Stretched Beta", "min", 0.0, context or ""),
                "mode": _get_param_value(name_key, "Stretched Beta", "mode", 0.0, context or ""),
                "max": _get_param_value(name_key, "Stretched Beta", "max", 0.0, context or ""),
            }
        elif dist == "Truncated Normal":
            default_mean = float(converted_params.get("mean", 0.0))
            default_sd = float(converted_params.get("sd", 1.0))
            default_min = float(converted_params.get("min", default_mean - default_sd))
            default_max = float(converted_params.get("max", default_mean + default_sd))
            
            current_mean = _get_param_value(name_key, "Truncated Normal", "mean", default_mean, context or "")
            current_sd = _get_param_value(name_key, "Truncated Normal", "sd", default_sd, context or "")
            current_min = _get_param_value(name_key, "Truncated Normal", "min", default_min, context or "")
            current_max = _get_param_value(name_key, "Truncated Normal", "max", default_max, context or "")
            
            temp_mean_key = f"_widget_{name_key}_tn_mean{key_suffix}"
            temp_sd_key = f"_widget_{name_key}_tn_sd{key_suffix}"
            temp_min_key = f"_widget_{name_key}_tn_min{key_suffix}"
            temp_max_key = f"_widget_{name_key}_tn_max{key_suffix}"
            
            st.session_state[temp_mean_key] = current_mean
            st.session_state[temp_sd_key] = current_sd
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_max_key] = current_max
            
            def sync_mean():
                _set_param_value(name_key, "Truncated Normal", "mean", st.session_state[temp_mean_key], context or "")
            def sync_sd():
                _set_param_value(name_key, "Truncated Normal", "sd", st.session_state[temp_sd_key], context or "")
            def sync_min():
                _set_param_value(name_key, "Truncated Normal", "min", st.session_state[temp_min_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Truncated Normal", "max", st.session_state[temp_max_key], context or "")
            
            mean_v = st.number_input(
                f"{label} mean ({unit_hint})",
                value=current_mean,
                key=temp_mean_key,
                on_change=sync_mean,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Normal", "mean", st.session_state[temp_mean_key], context or "")
            
            sd_v = st.number_input(
                f"{label} sd ({unit_hint})",
                value=current_sd,
                key=temp_sd_key,
                on_change=sync_sd,
                min_value=0.0,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Normal", "sd", st.session_state[temp_sd_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Normal", "min", st.session_state[temp_min_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Normal", "max", st.session_state[temp_max_key], context or "")
            
            params = {
                "mean": _get_param_value(name_key, "Truncated Normal", "mean", 0.0, context or ""),
                "sd": _get_param_value(name_key, "Truncated Normal", "sd", 0.0, context or ""),
                "min": _get_param_value(name_key, "Truncated Normal", "min", 0.0, context or ""),
                "max": _get_param_value(name_key, "Truncated Normal", "max", 0.0, context or ""),
            }
        elif dist == "Truncated Lognormal":
            default_mean = float(converted_params.get("mean", 1.0))
            default_sd = float(converted_params.get("sd", 0.2))
            default_min = float(converted_params.get("min", max(default_mean * 0.5, 1e-6)))
            default_max = float(converted_params.get("max", default_mean * 2.0))
            
            current_mean = _get_param_value(name_key, "Truncated Lognormal", "mean", default_mean, context or "")
            current_sd = _get_param_value(name_key, "Truncated Lognormal", "sd", default_sd, context or "")
            current_min = _get_param_value(name_key, "Truncated Lognormal", "min", default_min, context or "")
            current_max = _get_param_value(name_key, "Truncated Lognormal", "max", default_max, context or "")
            
            temp_mean_key = f"_widget_{name_key}_tln_mean{key_suffix}"
            temp_sd_key = f"_widget_{name_key}_tln_sd{key_suffix}"
            temp_min_key = f"_widget_{name_key}_tln_min{key_suffix}"
            temp_max_key = f"_widget_{name_key}_tln_max{key_suffix}"
            
            st.session_state[temp_mean_key] = current_mean
            st.session_state[temp_sd_key] = current_sd
            st.session_state[temp_min_key] = current_min
            st.session_state[temp_max_key] = current_max
            
            def sync_mean():
                _set_param_value(name_key, "Truncated Lognormal", "mean", st.session_state[temp_mean_key], context or "")
            def sync_sd():
                _set_param_value(name_key, "Truncated Lognormal", "sd", st.session_state[temp_sd_key], context or "")
            def sync_min():
                _set_param_value(name_key, "Truncated Lognormal", "min", st.session_state[temp_min_key], context or "")
            def sync_max():
                _set_param_value(name_key, "Truncated Lognormal", "max", st.session_state[temp_max_key], context or "")
            
            mean_v = st.number_input(
                f"{label} arithmetic mean ({unit_hint})",
                value=current_mean,
                key=temp_mean_key,
                on_change=sync_mean,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Lognormal", "mean", st.session_state[temp_mean_key], context or "")
            
            sd_v = st.number_input(
                f"{label} arithmetic sd ({unit_hint})",
                value=current_sd,
                key=temp_sd_key,
                on_change=sync_sd,
                min_value=0.0,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Lognormal", "sd", st.session_state[temp_sd_key], context or "")
            
            min_v = st.number_input(
                f"{label} min ({unit_hint})",
                value=current_min,
                key=temp_min_key,
                on_change=sync_min,
                min_value=0.0,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Lognormal", "min", st.session_state[temp_min_key], context or "")
            
            max_v = st.number_input(
                f"{label} max ({unit_hint})",
                value=current_max,
                key=temp_max_key,
                on_change=sync_max,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Truncated Lognormal", "max", st.session_state[temp_max_key], context or "")
            
            params = {
                "mean": _get_param_value(name_key, "Truncated Lognormal", "mean", 0.0, context or ""),
                "sd": _get_param_value(name_key, "Truncated Lognormal", "sd", 0.0, context or ""),
                "min": _get_param_value(name_key, "Truncated Lognormal", "min", 0.0, context or ""),
                "max": _get_param_value(name_key, "Truncated Lognormal", "max", 0.0, context or ""),
            }
        elif dist == "Burr (c, d, scale)":
            default_c = float(converted_params.get("c", 2.0))
            default_d = float(converted_params.get("d", 4.0))
            default_scale = float(converted_params.get("scale", 1.0))
            default_loc = float(converted_params.get("loc", 0.0))
            
            current_c = _get_param_value(name_key, "Burr (c, d, scale)", "c", default_c, context or "")
            current_d = _get_param_value(name_key, "Burr (c, d, scale)", "d", default_d, context or "")
            current_scale = _get_param_value(name_key, "Burr (c, d, scale)", "scale", default_scale, context or "")
            current_loc = _get_param_value(name_key, "Burr (c, d, scale)", "loc", default_loc, context or "")
            
            temp_c_key = f"_widget_{name_key}_burr_c{key_suffix}"
            temp_d_key = f"_widget_{name_key}_burr_d{key_suffix}"
            temp_scale_key = f"_widget_{name_key}_burr_scale{key_suffix}"
            temp_loc_key = f"_widget_{name_key}_burr_loc{key_suffix}"
            
            st.session_state[temp_c_key] = current_c
            st.session_state[temp_d_key] = current_d
            st.session_state[temp_scale_key] = current_scale
            st.session_state[temp_loc_key] = current_loc
            
            def sync_c():
                _set_param_value(name_key, "Burr (c, d, scale)", "c", st.session_state[temp_c_key], context or "")
            def sync_d():
                _set_param_value(name_key, "Burr (c, d, scale)", "d", st.session_state[temp_d_key], context or "")
            def sync_scale():
                _set_param_value(name_key, "Burr (c, d, scale)", "scale", st.session_state[temp_scale_key], context or "")
            def sync_loc():
                _set_param_value(name_key, "Burr (c, d, scale)", "loc", st.session_state[temp_loc_key], context or "")
            
            c_v = st.number_input(
                f"{label} shape c",
                value=current_c,
                key=temp_c_key,
                on_change=sync_c,
                min_value=1e-6,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Burr (c, d, scale)", "c", st.session_state[temp_c_key], context or "")
            
            d_v = st.number_input(
                f"{label} shape d",
                value=current_d,
                key=temp_d_key,
                on_change=sync_d,
                min_value=1e-6,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Burr (c, d, scale)", "d", st.session_state[temp_d_key], context or "")
            
            scale_v = st.number_input(
                f"{label} scale",
                value=current_scale,
                key=temp_scale_key,
                on_change=sync_scale,
                min_value=1e-6,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Burr (c, d, scale)", "scale", st.session_state[temp_scale_key], context or "")
            
            loc_v = st.number_input(
                f"{label} loc ({unit_hint})",
                value=current_loc,
                key=temp_loc_key,
                on_change=sync_loc,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Burr (c, d, scale)", "loc", st.session_state[temp_loc_key], context or "")
            
            params = {
                "c": _get_param_value(name_key, "Burr (c, d, scale)", "c", 0.0, context or ""),
                "d": _get_param_value(name_key, "Burr (c, d, scale)", "d", 0.0, context or ""),
                "scale": _get_param_value(name_key, "Burr (c, d, scale)", "scale", 0.0, context or ""),
                "loc": _get_param_value(name_key, "Burr (c, d, scale)", "loc", 0.0, context or ""),
            }
        elif dist == "Johnson SU":
            default_gamma = float(converted_params.get("gamma", 0.0))
            default_delta = float(converted_params.get("delta", 1.0))
            default_loc = float(converted_params.get("loc", 0.0))
            default_scale = float(converted_params.get("scale", 1.0))
            
            current_gamma = _get_param_value(name_key, "Johnson SU", "gamma", default_gamma, context or "")
            current_delta = _get_param_value(name_key, "Johnson SU", "delta", default_delta, context or "")
            current_loc = _get_param_value(name_key, "Johnson SU", "loc", default_loc, context or "")
            current_scale = _get_param_value(name_key, "Johnson SU", "scale", default_scale, context or "")
            
            temp_gamma_key = f"_widget_{name_key}_jsu_gamma{key_suffix}"
            temp_delta_key = f"_widget_{name_key}_jsu_delta{key_suffix}"
            temp_loc_key = f"_widget_{name_key}_jsu_loc{key_suffix}"
            temp_scale_key = f"_widget_{name_key}_jsu_scale{key_suffix}"
            
            st.session_state[temp_gamma_key] = current_gamma
            st.session_state[temp_delta_key] = current_delta
            st.session_state[temp_loc_key] = current_loc
            st.session_state[temp_scale_key] = current_scale
            
            def sync_gamma():
                _set_param_value(name_key, "Johnson SU", "gamma", st.session_state[temp_gamma_key], context or "")
            def sync_delta():
                _set_param_value(name_key, "Johnson SU", "delta", st.session_state[temp_delta_key], context or "")
            def sync_loc():
                _set_param_value(name_key, "Johnson SU", "loc", st.session_state[temp_loc_key], context or "")
            def sync_scale():
                _set_param_value(name_key, "Johnson SU", "scale", st.session_state[temp_scale_key], context or "")
            
            gamma_v = st.number_input(
                f"{label} gamma",
                value=current_gamma,
                key=temp_gamma_key,
                on_change=sync_gamma,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Johnson SU", "gamma", st.session_state[temp_gamma_key], context or "")
            
            delta_v = st.number_input(
                f"{label} delta",
                value=current_delta,
                key=temp_delta_key,
                on_change=sync_delta,
                min_value=1e-6,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Johnson SU", "delta", st.session_state[temp_delta_key], context or "")
            
            loc_v = st.number_input(
                f"{label} loc ({unit_hint})",
                value=current_loc,
                key=temp_loc_key,
                on_change=sync_loc,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Johnson SU", "loc", st.session_state[temp_loc_key], context or "")
            
            scale_v = st.number_input(
                f"{label} scale ({unit_hint})",
                value=current_scale,
                key=temp_scale_key,
                on_change=sync_scale,
                min_value=1e-6,
                help=help_text,
                format=format_str,
            )
            _set_param_value(name_key, "Johnson SU", "scale", st.session_state[temp_scale_key], context or "")
            
            params = {
                "gamma": _get_param_value(name_key, "Johnson SU", "gamma", 0.0, context or ""),
                "delta": _get_param_value(name_key, "Johnson SU", "delta", 0.0, context or ""),
                "loc": _get_param_value(name_key, "Johnson SU", "loc", 0.0, context or ""),
                "scale": _get_param_value(name_key, "Johnson SU", "scale", 0.0, context or ""),
            }
        else:
            params = {}

    recalc = st.button(
        f"Recalculate {label}",
        key=f"recalc_{name_key}{key_suffix}",
        type="primary",
    )

    ss_samples_key = f"samples_{name_key}{key_suffix}"
    ss_conf_key = f"conf_{name_key}{key_suffix}"
    need_init = ss_samples_key not in st.session_state
    current_conf = {"dist": dist, **params, "n": n_samples}
    
    # Check if configuration changed (distribution type or parameters)
    # Use a more robust comparison that handles float precision issues
    config_changed = False
    if ss_conf_key in st.session_state:
        old_conf = st.session_state[ss_conf_key]
        # Compare distribution type FIRST (most important)
        if old_conf.get("dist") != current_conf.get("dist"):
            config_changed = True
        # Compare number of samples
        elif old_conf.get("n") != current_conf.get("n"):
            config_changed = True
        else:
            # Compare all parameter values
            for key in set(list(old_conf.keys()) + list(current_conf.keys())):
                if key in ["dist", "n"]:
                    continue
                old_val = old_conf.get(key)
                new_val = current_conf.get(key)
                if old_val is None or new_val is None:
                    if old_val != new_val:
                        config_changed = True
                        break
                elif isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if abs(float(old_val) - float(new_val)) > 1e-10:
                        config_changed = True
                        break
                elif old_val != new_val:
                    config_changed = True
                    break
    
    # CRITICAL: If config changed, clear samples cache to force recalculation
    if config_changed and ss_samples_key in st.session_state:
        del st.session_state[ss_samples_key]
        need_init = True  # Force recalculation

    if recalc or need_init or config_changed or (
        ss_conf_key in st.session_state
        and st.session_state[ss_conf_key].get("n") != n_samples
    ):
        if dist == "Constant":
            const_val = params.get("value", 0.0)
            samples = np.full(n_samples, float(const_val))
        elif dist == "PERT":
            samples = sample_pert(rng, params["min"], params["mode"], params["max"], n_samples)
        elif dist == "Triangular":
            samples = sample_triangular(rng, params["min"], params["mode"], params["max"], n_samples)
        elif dist == "Uniform":
            samples = sample_uniform(rng, params["min"], params["max"], n_samples)
        elif dist == "Lognormal (mean, sd)":
            samples = sample_lognormal_mean_sd(rng, params["mean"], params["sd"], n_samples)
        elif dist == "Subjective Beta (Vose)":
            # Use mode if available, otherwise calculate mean from min/max, or use a default
            p50_est = params.get("mode", params.get("mean", (params["min"] + params["max"]) / 2.0))
            p10_est = params["min"] + 0.1 * (p50_est - params["min"])
            p90_est = p50_est + 0.1 * (params["max"] - p50_est)
            samples = sample_beta_subjective(
                rng, params["min"], params["max"], p10_est, p50_est, p90_est, n_samples
            )
        elif dist == "Stretched Beta":
            samples = sample_stretched_beta(rng, params["min"], params["max"], params["mode"], n_samples)
        elif dist == "Truncated Normal":
            samples = sample_truncated_normal(
                rng, params["mean"], params["sd"], params["min"], params["max"], n_samples
            )
        elif dist == "Truncated Lognormal":
            samples = sample_truncated_lognormal(
                rng, params["mean"], params["sd"], params["min"], params["max"], n_samples
            )
        elif dist == "Burr (c, d, scale)":
            samples = sample_burr(
                rng, params.get("c", 1.0), params.get("d", 1.0), params.get("scale", 1.0), n_samples
            )
        elif dist == "Johnson SU":
            samples = sample_johnson_su(
                rng, params.get("a", 0.0), params.get("b", 1.0), params.get("loc", 0.0), params.get("scale", 1.0), n_samples
            )
        else:
            samples = np.zeros(n_samples)

        st.session_state[ss_samples_key] = samples
        st.session_state[ss_conf_key] = current_conf
        
        # CRITICAL: Immediately store samples in the final session state key that simulation uses
        # This ensures that when inputs change, the simulation will use the updated values
        # Map name_key to final session state key (e.g., "NtG" -> "sNtG", "p" -> "sp", "RF_oil" -> "sRF_oil")
        final_key_map = {
            "NtG": "sNtG",
            "p": "sp",
            "RF_oil": "sRF_oil",
            "RF_gas": "sRF_gas",
            "RF_assoc": "sRF_assoc_gas",
            "RF_cond": "sRF_cond",
            "Bg": "sBg",
            "InvBo": "sInvBo",
            "InvBg": "sInvBg",
            "GOR": "sGOR",
            "CY": "sCY",
        }
        # Try to find the final key - either from map or by adding "s" prefix
        final_key = final_key_map.get(name_key)
        if final_key is None:
            # Try adding "s" prefix if name_key doesn't start with "s"
            if not name_key.startswith("s"):
                final_key = f"s{name_key}"
            else:
                final_key = name_key
        
        # Only store if it's a recognized parameter key (avoid storing internal keys)
        if final_key in final_key_map.values() or (final_key.startswith("s") and len(final_key) > 1):
            st.session_state[final_key] = samples.copy()  # Use copy to avoid reference issues
            # CRITICAL: Clear results cache when inputs change to force recalculation
            if "results_cache" in st.session_state:
                del st.session_state["results_cache"]
            if "trial_data" in st.session_state:
                del st.session_state["trial_data"]
            if "df_results" in st.session_state:
                del st.session_state["df_results"]
    else:
        samples = st.session_state.get(ss_samples_key, np.zeros(n_samples))
        # Also ensure final key is set if it exists (for consistency)
        final_key_map = {
            "NtG": "sNtG",
            "p": "sp",
            "RF_oil": "sRF_oil",
            "RF_gas": "sRF_gas",
            "RF_assoc": "sRF_assoc_gas",
            "RF_cond": "sRF_cond",
            "Bg": "sBg",
            "InvBo": "sInvBo",
            "InvBg": "sInvBg",
            "GOR": "sGOR",
            "CY": "sCY",
        }
        final_key = final_key_map.get(name_key)
        if final_key is None:
            if not name_key.startswith("s"):
                final_key = f"s{name_key}"
            else:
                final_key = name_key
        
        # Ensure final key is also set from cached samples if it doesn't exist
        if (final_key in final_key_map.values() or (final_key.startswith("s") and len(final_key) > 1)) and final_key not in st.session_state:
            st.session_state[final_key] = samples.copy()

    display_samples = samples * display_scale
    unit_lbl = plot_unit_label if plot_unit_label else unit_hint
    st.plotly_chart(
        make_hist_cdf_figure(display_samples, f"{label} distribution", f"{label} ({unit_lbl})", "input"),
        use_container_width=True,
    )
    st.dataframe(summary_table(display_samples, decimals=stats_decimals), use_container_width=True)

    return samples


def apply_correlations_to_samples(
    samples_dict: Dict[str, np.ndarray],
    correlation_values: Dict[str, float],
    n_samples: int,
) -> Dict[str, np.ndarray]:
    """Apply correlations to already sampled parameters."""
    updated_samples = samples_dict.copy()

    if "temp_pressure" in correlation_values and abs(correlation_values["temp_pressure"]) > 0.01:
        if "sT" in updated_samples and "sP" in updated_samples:
            updated_samples["sT"], updated_samples["sP"] = apply_correlation(
                updated_samples["sT"], updated_samples["sP"], correlation_values["temp_pressure"]
            )

    if "top_base_depth" in correlation_values and abs(correlation_values["top_base_depth"]) > 0.01:
        if "stopdepth_off" in updated_samples and "sbasedepth_off" in updated_samples:
            updated_samples["stopdepth_off"], updated_samples["sbasedepth_off"] = apply_correlation(
                updated_samples["stopdepth_off"],
                updated_samples["sbasedepth_off"],
                correlation_values["top_base_depth"],
            )

    if "porosity_se" in correlation_values and abs(correlation_values["porosity_se"]) > 0.01:
        if "sp" in updated_samples and "sSE" in updated_samples:
            updated_samples["sp"], updated_samples["sSE"] = apply_correlation(
                updated_samples["sp"], updated_samples["sSE"], correlation_values["porosity_se"]
            )

    if "porosity_ntg" in correlation_values and abs(correlation_values["porosity_ntg"]) > 0.01:
        if "sp" in updated_samples and "sNtG" in updated_samples:
            updated_samples["sp"], updated_samples["sNtG"] = apply_correlation(
                updated_samples["sp"], updated_samples["sNtG"], correlation_values["porosity_ntg"]
            )

    if "thickness_gcf" in correlation_values and abs(correlation_values["thickness_gcf"]) > 0.01:
        if "sh" in updated_samples and "sGCF" in updated_samples:
            updated_samples["sh"], updated_samples["sGCF"] = apply_correlation(
                updated_samples["sh"], updated_samples["sGCF"], correlation_values["thickness_gcf"]
            )

    return updated_samples


def collect_all_trial_data() -> Optional[Dict[str, List[float]]]:
    """Collect all input parameters and results for each trial."""
    required_keys = [
        "sGRV_m3_final",
        "sNtG",
        "sp",
        "sRF_oil",
        "sRF_gas",
        "sBg",
        "sInvBo",
        "sGOR",
    ]

    missing_keys = [key for key in required_keys if key not in st.session_state]
    if missing_keys:
        st.warning(
            "Please complete the Inputs page first. Missing: "
            + ", ".join(missing_keys)
        )
        return None

    # CRITICAL: Always read FRESH values from session state, never use cached arrays
    # This ensures the simulation uses the absolute latest input parameters
    num_trials = len(st.session_state["sGRV_m3_final"])
    
    # Read all input arrays directly from session state (fresh read, not cached)
    trial_data: Dict[str, List[float]] = {
        "Trial": list(range(1, num_trials + 1)),
        "GRV_m3": np.asarray(st.session_state["sGRV_m3_final"], dtype=float).tolist(),
        "NtG": np.asarray(st.session_state["sNtG"], dtype=float).tolist(),
        "Porosity": np.asarray(st.session_state["sp"], dtype=float).tolist(),
        "RF_oil": np.asarray(st.session_state["sRF_oil"], dtype=float).tolist(),
        "RF_gas": np.asarray(st.session_state["sRF_gas"], dtype=float).tolist(),
        "Bg_rb_per_scf": np.asarray(st.session_state["sBg"], dtype=float).tolist(),
        "InvBo_STB_per_rb": np.asarray(st.session_state["sInvBo"], dtype=float).tolist(),
        "GOR_scf_per_STB": np.asarray(st.session_state["sGOR"], dtype=float).tolist(),
    }

    additional_params = {
        "sA": "Area_km2",
        "sGCF": "GCF",
        "sh": "Thickness_m",
        "grv_mp_samples": "GRV_Multiplier",
        "gcf_mp_samples": "GCF_Multiplier",
    }
    for session_key, display_name in additional_params.items():
        if session_key in st.session_state:
            trial_data[display_name] = st.session_state[session_key].tolist()

    grv = np.array(trial_data["GRV_m3"])
    ntg = np.array(trial_data["NtG"])
    porosity = np.array(trial_data["Porosity"])
    rf_oil = np.array(trial_data["RF_oil"])
    rf_gas = np.array(trial_data["RF_gas"])
    bg = np.array(trial_data["Bg_rb_per_scf"])
    invbo = np.array(trial_data["InvBo_STB_per_rb"])
    gor = np.array(trial_data["GOR_scf_per_STB"])

    # CRITICAL: Always read the CURRENT fluid_type and grv_option from session state
    # This ensures the simulation uses the latest selections, not cached values
    fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
    grv_option = st.session_state.get("grv_option", "Direct GRV")
    
    # CRITICAL: Handle fluid_type FIRST to override any stale GRV arrays
    # This ensures that if fluid_type changed, we use the correct split regardless of cached arrays
    if fluid_type == "Oil":
        # All GRV is oil - override any cached split
        GRV_oil_m3 = grv.copy()
        GRV_gas_m3 = np.zeros_like(grv)
    elif fluid_type == "Gas":
        # All GRV is gas - override any cached split
        GRV_oil_m3 = np.zeros_like(grv)
        GRV_gas_m3 = grv.copy()
    else:  # Oil + Gas - try to get split from cached arrays
        GRV_oil_m3 = None
        GRV_gas_m3 = None
        
        # Get split GRV arrays based on CURRENT selected method
        # Check for split GRV from depth-based methods first (they're already stored)
        if "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
            GRV_oil_m3 = np.asarray(st.session_state["sGRV_oil_m3"], dtype=float)
            GRV_gas_m3 = np.asarray(st.session_state["sGRV_gas_m3"], dtype=float)
        elif grv_option == "Direct GRV":
            # Get from direct method split
            if "direct_GRV_oil_m3" in st.session_state:
                GRV_oil_m3 = np.asarray(st.session_state["direct_GRV_oil_m3"], dtype=float)
            if "direct_GRV_gas_m3" in st.session_state:
                GRV_gas_m3 = np.asarray(st.session_state["direct_GRV_gas_m3"], dtype=float)
        elif grv_option == "Area × Thickness × GCF":
            # Get from atgcf method split
            if "atgcf_GRV_oil_m3" in st.session_state:
                GRV_oil_m3 = np.asarray(st.session_state["atgcf_GRV_oil_m3"], dtype=float)
            if "atgcf_GRV_gas_m3" in st.session_state:
                GRV_gas_m3 = np.asarray(st.session_state["atgcf_GRV_gas_m3"], dtype=float)
    
    # Fallback: use f_oil if split GRV not available for Oil + Gas
    if fluid_type == "Oil + Gas" and (GRV_oil_m3 is None or GRV_gas_m3 is None):
        # Try to get f_oil from the appropriate method
        f_oil_val = 0.5  # default
        if grv_option == "Direct GRV" and "direct_f_oil" in st.session_state:
            f_oil_val = float(st.session_state["direct_f_oil"])
        elif grv_option == "Area × Thickness × GCF" and "atgcf_f_oil" in st.session_state:
            f_oil_val = float(st.session_state["atgcf_f_oil"])
        elif "f_oil" in st.session_state:
            f_oil_val = float(st.session_state["f_oil"])
        
        if GRV_oil_m3 is None:
            GRV_oil_m3 = grv * f_oil_val
        if GRV_gas_m3 is None:
            GRV_gas_m3 = grv * (1.0 - f_oil_val)
    
    # Get f_oil array - ensure we extract scalar value if f_oil is an array
    f_oil_val = st.session_state.get("f_oil", 0.5)
    if isinstance(f_oil_val, np.ndarray):
        # If f_oil is an array, use its mean or first value
        f_oil_val = float(np.mean(f_oil_val)) if len(f_oil_val) > 0 else 0.5
    else:
        f_oil_val = float(f_oil_val)
    f_oil = np.full(num_trials, f_oil_val)
    sCY = st.session_state.get("sCY", None)
    sRF_cond = st.session_state.get("sRF_cond", None)
    gas_scf_per_boe = st.session_state.get("gas_scf_per_boe", 6000.0)

    rf_oil, rf_gas, sRF_cond, rf_warnings = validate_rf_fractions(rf_oil, rf_gas, sRF_cond)
    f_oil, frac_warnings = validate_fractions(f_oil)
    all_warnings = rf_warnings + frac_warnings
    if all_warnings:
        warning_text = "⚠️ Parameter validation warnings:\n" + "\n".join(f"• {w}" for w in all_warnings)
        st.warning(warning_text)

    sRF_assoc_gas = trial_data.get("RF_assoc", trial_data.get("RF_oil", rf_oil))

    # Sample/derive saturations
    from scopehc.compute import derive_saturation_samples
    from scopehc.sampling import rng_from_seed
    
    mode = st.session_state.get("sat_mode", "Global")
    seed = st.session_state.get("random_seed", 42)
    rng = rng_from_seed(seed)
    
    # Saturation is ALWAYS used - derive_saturation_samples provides defaults if not configured
    try:
        sat = derive_saturation_samples(rng, num_trials, mode, st.session_state)
        Shc_oil = sat["Shc_oil"]
        Shc_gas = sat["Shc_gas"]
        # Store all saturation arrays in session state
        for k, v in sat.items():
            st.session_state[k] = v
        st.session_state["Shc_oil"] = Shc_oil
        st.session_state["Shc_gas"] = Shc_gas
    except Exception as e:
        # Fallback: use 100% saturation if derivation fails
        import warnings
        warnings.warn(f"Saturation derivation failed: {e}. Using 100% saturation.")
        Shc_oil = np.ones(num_trials)
        Shc_gas = np.ones(num_trials)
        st.session_state["Shc_oil"] = Shc_oil
        st.session_state["Shc_gas"] = Shc_gas

    res = compute_results(
        GRV_m3=grv,
        NtG=ntg,
        Por=porosity,
        f_oil=f_oil,
        RF_oil=rf_oil,
        RF_gas=rf_gas,
        Bg_rb_per_scf=bg,
        InvBo_STB_per_rb=invbo,
        GOR_scf_per_STB=gor,
        CY_STB_per_MMscf=sCY,
        RF_cond=sRF_cond,
        RF_assoc=sRF_assoc_gas,
        gas_scf_per_boe=gas_scf_per_boe,
        GRV_oil_m3=GRV_oil_m3,
        GRV_gas_m3=GRV_gas_m3,
        Shc_oil=Shc_oil,
        Shc_gas=Shc_gas,
    )

    trial_data["Pore_Volume_Total_m3"] = res["PV_total_m3"].tolist()
    trial_data["Pore_Volume_Oil_m3"] = res["PV_oil_m3"].tolist()
    trial_data["Pore_Volume_Gas_m3"] = res["PV_gas_m3"].tolist()
    trial_data["In_situ_Oil_Volume_m3"] = res["V_oil_insitu_m3"].tolist()
    trial_data["In_situ_Gas_Volume_m3"] = res["V_gas_insitu_m3"].tolist()
    trial_data["Recoverable_Oil_STB"] = res["Oil_STB_rec"].tolist()
    trial_data["Recoverable_Oil_MMSTB"] = (res["Oil_STB_rec"] / 1e6).tolist()
    trial_data["Recoverable_Oil_Mm3"] = (res["Oil_STB_rec"] * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6).tolist()
    trial_data["Recoverable_Free_Gas_scf"] = res["Gas_free_scf_rec"].tolist()
    trial_data["Recoverable_Free_Gas_Bscf"] = (res["Gas_free_scf_rec"] / 1e9).tolist()
    trial_data["Recoverable_Free_Gas_Bm3"] = (
        res["Gas_free_scf_rec"] * UNIT_CONVERSIONS["scf_to_m3"] / 1e9
    ).tolist()
    trial_data["Recoverable_Associated_Gas_scf"] = res["Gas_assoc_scf_rec"].tolist()
    trial_data["Recoverable_Associated_Gas_Bscf"] = (res["Gas_assoc_scf_rec"] / 1e9).tolist()
    trial_data["Recoverable_Associated_Gas_Bm3"] = (
        res["Gas_assoc_scf_rec"] * UNIT_CONVERSIONS["scf_to_m3"] / 1e9
    ).tolist()

    cond_rec_stb = res["Cond_STB_rec"]

    gas_total_scf = res["Total_gas_scf_rec"]
    total_liquids_stb = res["Total_liquids_STB"]
    thr_boe = res["Total_surface_BOE"]

    oip_stb = (res["V_oil_insitu_m3"] * RB_PER_M3) * invbo
    gip_scf = (res["V_gas_insitu_m3"] * RB_PER_M3) / bg

    trial_data["Recoverable_Condensate_STB"] = cond_rec_stb.tolist()
    trial_data["Recoverable_Condensate_MMSTB"] = (cond_rec_stb / 1e6).tolist()
    trial_data["Recoverable_Condensate_Mm3"] = (
        cond_rec_stb * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
    ).tolist()
    trial_data["Recoverable_Total_Gas_scf"] = gas_total_scf.tolist()
    trial_data["Recoverable_Total_Gas_Bscf"] = (gas_total_scf / 1e9).tolist()
    trial_data["Recoverable_Total_Gas_Bm3"] = (
        gas_total_scf * UNIT_CONVERSIONS["scf_to_m3"] / 1e9
    ).tolist()
    trial_data["Recoverable_Total_Liquids_STB"] = total_liquids_stb.tolist()
    trial_data["Recoverable_Total_Liquids_MMSTB"] = (total_liquids_stb / 1e6).tolist()
    trial_data["Recoverable_Total_Liquids_Mm3"] = (
        total_liquids_stb * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6
    ).tolist()
    trial_data["Total_Hydrocarbon_Resource_BOE"] = thr_boe.tolist()
    trial_data["Oil_in_Place_STB"] = oip_stb.tolist()
    trial_data["Gas_in_Place_scf"] = gip_scf.tolist()
    trial_data["Oil_in_Place_MMSTB"] = (oip_stb / 1e6).tolist()
    trial_data["Gas_in_Place_Bscf"] = (gip_scf / 1e9).tolist()
    trial_data["Oil_in_Place_Mm3"] = (oip_stb * UNIT_CONVERSIONS["bbl_to_m3"] / 1e6).tolist()
    trial_data["Gas_in_Place_Bm3"] = (gip_scf * UNIT_CONVERSIONS["scf_to_m3"] / 1e9).tolist()

    # Add saturation data to trial_data for export
    mode = st.session_state.get("sat_mode", "Global")
    use_sw_global = st.session_state.get("global_sat_use_sw", False)
    
    if mode.startswith("Global"):
        if use_sw_global:
            if "Sw_global" in st.session_state:
                trial_data["Sw_global"] = st.session_state["Sw_global"].tolist()
        if "Shc_global" in st.session_state:
            trial_data["Shc_global"] = st.session_state["Shc_global"].tolist()
    elif mode.startswith("Water saturation"):
        if "Sw_oilzone" in st.session_state:
            trial_data["Sw_oilzone"] = st.session_state["Sw_oilzone"].tolist()
        if "Sw_gaszone" in st.session_state:
            trial_data["Sw_gaszone"] = st.session_state["Sw_gaszone"].tolist()
    else:  # Per phase
        if "Shc_oil_input" in st.session_state:
            trial_data["Shc_oil_input"] = st.session_state["Shc_oil_input"].tolist()
        if "Shc_gas_input" in st.session_state:
            trial_data["Shc_gas_input"] = st.session_state["Shc_gas_input"].tolist()

    # Always add derived saturations
    if "Shc_oil" in st.session_state:
        trial_data["Shc_oil"] = st.session_state["Shc_oil"].tolist()
    if "Shc_gas" in st.session_state:
        trial_data["Shc_gas"] = st.session_state["Shc_gas"].tolist()

    # Add column heights if available (from depth-based methods or computed)
    if "sOil_Column_Height" in st.session_state:
        trial_data["Oil_Column_Height_m"] = st.session_state["sOil_Column_Height"].tolist()
    if "sGas_Column_Height" in st.session_state:
        trial_data["Gas_Column_Height_m"] = st.session_state["sGas_Column_Height"].tolist()
    
    # Add split GRV if available
    if "sGRV_oil_m3" in st.session_state:
        trial_data["GRV_oil_m3"] = st.session_state["sGRV_oil_m3"].tolist()
    if "sGRV_gas_m3" in st.session_state:
        trial_data["GRV_gas_m3"] = st.session_state["sGRV_gas_m3"].tolist()
    
    # Add GOC mode and derived GOC depth if available
    fluid_type = st.session_state.get("fluid_type", "Oil + Gas")
    if fluid_type == "Oil + Gas":
        goc_mode = st.session_state.get("da_goc_mode") or st.session_state.get("da_goc_mode_D")
        if goc_mode:
            trial_data["GOC_Mode"] = [goc_mode] * num_trials
            # Try to get GOC depth if it was computed
            if "sGOC_depth" in st.session_state:
                trial_data["GOC_Depth_m"] = st.session_state["sGOC_depth"].tolist()
        
        # Add oil fraction if used
        if "direct_f_oil" in st.session_state:
            arr = np.atleast_1d(np.asarray(st.session_state["direct_f_oil"], dtype=float))
            if len(arr) >= num_trials:
                trial_data["f_oil_direct"] = arr[:num_trials].tolist()
            else:
                # Pad with last value if needed
                trial_data["f_oil_direct"] = (list(arr) + [arr[-1]] * (num_trials - len(arr)))[:num_trials]
        if "atgcf_f_oil" in st.session_state:
            arr = np.atleast_1d(np.asarray(st.session_state["atgcf_f_oil"], dtype=float))
            if len(arr) >= num_trials:
                trial_data["f_oil_atgcf"] = arr[:num_trials].tolist()
            else:
                # Pad with last value if needed
                trial_data["f_oil_atgcf"] = (list(arr) + [arr[-1]] * (num_trials - len(arr)))[:num_trials]
        if "da_oil_frac" in st.session_state:
            # da_oil_frac is now an array from distribution, use it directly
            arr = np.atleast_1d(np.asarray(st.session_state["da_oil_frac"], dtype=float))
            if len(arr) >= num_trials:
                trial_data["f_oil_depth_area"] = arr[:num_trials].tolist()
            else:
                # If array is shorter, pad with last value or repeat
                trial_data["f_oil_depth_area"] = (list(arr) + [arr[-1]] * (num_trials - len(arr)))[:num_trials]

    df_results = pd.DataFrame(trial_data)
    st.session_state["trial_data"] = trial_data
    st.session_state["results_cache"] = res
    st.session_state["df_results"] = df_results

    return trial_data


# ============================================================================
# Run Tracker and Diagnostics System
# ============================================================================

def _stable_json(obj):
    """Convert object to stable JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, default=str)


def compute_input_hash(input_dict: dict) -> str:
    """Compute MD5 hash of input dictionary for change detection."""
    raw = _stable_json(input_dict).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]  # Use first 16 chars for readability


def collect_all_inputs_from_session() -> dict:
    """
    Collect all inputs from session state into a single dictionary.
    This includes: GRV method, fluid type, all distribution parameters, 
    correlations, saturation mode, percentile mode, seed, etc.
    """
    inputs = {}
    
    # Core settings
    inputs["fluid_type"] = st.session_state.get("fluid_type", "Oil + Gas")
    inputs["grv_option"] = st.session_state.get("grv_option", "Direct GRV")
    inputs["active_grv_method"] = inputs["grv_option"]  # Alias for clarity
    inputs["percentile_exceedance"] = st.session_state.get("percentile_exceedance", True)
    inputs["num_sims"] = st.session_state.get("num_sims", 10_000)
    inputs["rng_seed"] = st.session_state.get("random_seed", st.session_state.get("rng_seed", None))
    inputs["unit_system"] = st.session_state.get("unit_system", "oilfield")
    inputs["gas_scf_per_boe"] = st.session_state.get("gas_scf_per_boe", 6000.0)
    
    # Saturation mode
    inputs["sat_mode"] = st.session_state.get("sat_mode", "Global")
    inputs["global_sat_use_sw"] = st.session_state.get("global_sat_use_sw", False)
    
    # Distribution parameters (store as arrays converted to lists for hashing)
    param_keys = ["sGRV_m3_final", "sNtG", "sp", "sRF_oil", "sRF_gas", "sBg", "sInvBo", "sGOR", 
                  "sCY", "sRF_cond", "sRF_assoc_gas", "sA", "sGCF", "sh"]
    for key in param_keys:
        if key in st.session_state:
            arr = np.atleast_1d(np.asarray(st.session_state[key], dtype=float))
            # Store summary stats (mean, std, min, max) for hash, not full array
            if len(arr) > 0:
                inputs[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "len": len(arr)
                }
    
    # GRV split parameters
    if "direct_f_oil" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["direct_f_oil"], dtype=float))
        if len(arr) > 0:
            inputs["direct_f_oil"] = arr.tolist()  # Store as list for JSON serialization
        else:
            inputs["direct_f_oil"] = [0.55]  # Default mean from PERT
    if "atgcf_f_oil" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["atgcf_f_oil"], dtype=float))
        if len(arr) > 0:
            inputs["atgcf_f_oil"] = arr.tolist()  # Store as list for JSON serialization
        else:
            inputs["atgcf_f_oil"] = [0.55]  # Default mean from PERT
    # da_oil_frac and da_oil_frac_D are now arrays (from Stretched Beta distribution)
    # Store them as arrays, not scalars
    if "da_oil_frac" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["da_oil_frac"], dtype=float))
        if len(arr) > 0:
            # Store as array for use in calculations
            inputs["da_oil_frac"] = arr
        else:
            # Fallback to mean if array is empty
            inputs["da_oil_frac"] = 0.2  # Default mean from Stretched Beta
    if "da_oil_frac_D" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["da_oil_frac_D"], dtype=float))
        if len(arr) > 0:
            # Store as array for use in calculations
            inputs["da_oil_frac_D"] = arr
        else:
            # Fallback to mean if array is empty
            inputs["da_oil_frac_D"] = 0.2  # Default mean from Stretched Beta
    if "da_goc_mode" in st.session_state:
        inputs["da_goc_mode"] = st.session_state["da_goc_mode"]
    if "da_goc_mode_D" in st.session_state:
        inputs["da_goc_mode_D"] = st.session_state["da_goc_mode_D"]
    
    # Saturation inputs
    sat_keys = ["Shc_global", "Sw_global", "Sw_oilzone", "Sw_gaszone", "Shc_oil_input", "Shc_gas_input"]
    for key in sat_keys:
        if key in st.session_state:
            arr = np.atleast_1d(np.asarray(st.session_state[key], dtype=float))
            if len(arr) > 0:
                inputs[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr))
                }
    
    # Dependency/correlation matrix
    if st.session_state.get("dependencies_enabled", False):
        if "dependency_matrix" in st.session_state and st.session_state["dependency_matrix"] is not None:
            dep_matrix = st.session_state["dependency_matrix"]
            # Store matrix as nested list for hashing
            inputs["dependency_matrix"] = dep_matrix.tolist() if hasattr(dep_matrix, "tolist") else dep_matrix
        if "dependency_matrix_params" in st.session_state:
            inputs["dependency_matrix_params"] = st.session_state["dependency_matrix_params"]
    
    # Depth-based GRV parameters
    if "sD_spill" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["sD_spill"], dtype=float))
        if len(arr) > 0:
            inputs["sD_spill"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    if "sD_hc" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["sD_hc"], dtype=float))
        if len(arr) > 0:
            inputs["sD_hc"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    if "sGOC_depth" in st.session_state:
        arr = np.atleast_1d(np.asarray(st.session_state["sGOC_depth"], dtype=float))
        if len(arr) > 0:
            inputs["sGOC_depth"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    
    return inputs


def render_run_controls_and_diagnostics(input_dict: dict):
    """
    Shows Run button, increments run_id, stores input hash, and displays diagnostics.
    
    Args:
        input_dict: Dictionary containing all inputs that influence the simulation
    """
    # Ensure counters exist
    if "run_id" not in st.session_state:
        st.session_state["run_id"] = 0
    if "last_input_hash" not in st.session_state:
        st.session_state["last_input_hash"] = ""
    
    current_hash = compute_input_hash(input_dict)
    run_id = st.session_state.get("run_id", 0)
    last_hash = st.session_state.get("last_input_hash", "")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Run Monte Carlo Simulation", type="primary", key="btn_run_sim", use_container_width=True):
            st.session_state["run_id"] = run_id + 1
            st.session_state["last_input_hash"] = current_hash
            st.session_state["last_run_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            # Clear caches to force fresh computation
            if "results_cache" in st.session_state:
                del st.session_state["results_cache"]
            if "trial_data" in st.session_state:
                del st.session_state["trial_data"]
            if "df_results" in st.session_state:
                del st.session_state["df_results"]
            st.rerun()
    
    # Removed "Force Re-compute" button - not needed as regular "Run simulation" button
    # already handles cache invalidation properly. If users need to force recompute,
    # they can change any input parameter or use the run_id increment.
    

