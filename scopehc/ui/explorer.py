"""
Cross-plot & statistics explorer: the section at the foot of the Results page.

Pick any two simulated parameters, see the joint distribution, and get the statistics
that go with it - correlation, a fit, and a table of Y statistics per fixed-width
interval of X ("mean STOIIP per 10 m of depth").

Nothing here recomputes a volume. The section is handed the `res` dict the Results
page already computed and reads the sampled input arrays from session state, so its
numbers cannot drift from the ones plotted above it. The arithmetic lives in
scopehc.crossplot, which has no Streamlit dependency and is unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from scopehc.config import EXPLORER_GROUPS, EXPLORER_MATRIX_DEFAULTS, PALETTE
from scopehc.crossplot import (
    bin_statistics,
    correlation_matrix,
    fit_line,
    pair_statistics,
    rank_drivers,
    suggest_bin_width,
)

from .common import color_for
from .render_cache import fingerprint, memo

# Above this many trials a scatter is a solid blob, so the density view leads instead.
DENSITY_AUTO_THRESHOLD = 20_000
# Value labels inside a correlation heatmap stop being readable past this size.
HEATMAP_LABEL_LIMIT = 12

RAW_GROUP = "All trial columns"


# ---------------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------------


def _for_system(value: Any, unit_system: str) -> Any:
    """Resolve a registry field that may be per-unit-system."""
    if isinstance(value, dict):
        key = "oilfield" if unit_system == "oilfield" else "metric"
        return value.get(key, value.get("oilfield"))
    return value


def _as_array(raw: Any, n: int) -> Optional[np.ndarray]:
    """A per-trial float array of length n, or None if this source cannot provide one."""
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != n or not np.any(np.isfinite(arr)):
        return None
    return arr


def build_explorer_frame(
    res: Dict[str, Any], unit_system: str, include_raw: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Assemble the per-trial frame the explorer plots, in display units.

    Returns the frame plus metadata per column (unit, group, depth-like flag, colour).
    Entries whose source array is missing from this run - no condensate, a non
    depth-based GRV method - are left out rather than filled with zeros.
    """
    reference = res.get("STOIIP_STB")
    n = int(np.asarray(reference).size) if reference is not None else 0
    if n == 0:
        return pd.DataFrame(), {}

    data: Dict[str, np.ndarray] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for group, entries in EXPLORER_GROUPS:
        for entry in entries:
            kind, key = entry["source"]
            raw = st.session_state.get(key) if kind == "session" else res.get(key)
            arr = _as_array(raw, n)
            if arr is None:
                continue
            label = entry["label"]
            factor = float(_for_system(entry.get("factor", 1.0), unit_system))
            data[label] = arr * factor
            meta[label] = {
                "unit": str(_for_system(entry.get("unit", ""), unit_system)),
                "group": group,
                "invert": bool(entry.get("invert", False)),
                "color": entry.get("color"),
            }

    if include_raw:
        df_results = st.session_state.get("df_results")
        if isinstance(df_results, pd.DataFrame) and len(df_results) == n:
            for col in df_results.columns:
                if col in data or col == "Trial":
                    continue
                arr = _as_array(df_results[col].to_numpy(), n)
                if arr is None:
                    continue
                data[col] = arr
                meta[col] = {
                    "unit": _unit_from_column_name(col),
                    "group": RAW_GROUP,
                    "invert": "Depth" in col,
                    "color": None,
                }

    return pd.DataFrame(data), meta


def _unit_from_column_name(col: str) -> str:
    """Best-effort unit for a raw trial column, whose name already carries it."""
    for suffix, unit in (
        ("_m3", "m³"),
        ("_MMSTB", "MMSTB"),
        ("_STB", "STB"),
        ("_Bscf", "Bscf"),
        ("_scf", "scf"),
        ("_Mm3", "Mm³"),
        ("_Bm3", "Bm³"),
        ("_BOE", "BOE"),
        ("_km2", "km²"),
        ("_m", "m"),
    ):
        if col.endswith(suffix):
            return unit
    return ""


# ---------------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------------


def _unit_of(label: str, meta: Dict[str, Dict[str, Any]]) -> str:
    """The display unit, or "" for a dimensionless quantity.

    The registry writes "-" for fractions (NtG, porosity, saturations, RFs). That reads
    correctly in a table header but not inside a sentence, so it is normalised away
    here and every label is built conditionally.
    """
    unit = str(meta.get(label, {}).get("unit", "") or "")
    return "" if unit == "-" else unit


def _axis_title(label: str, meta: Dict[str, Any]) -> str:
    unit = meta.get("unit", "")
    return f"{label} ({unit})" if unit and unit != "-" else label


def _option_label(label: str, meta: Dict[str, Dict[str, Any]]) -> str:
    info = meta.get(label, {})
    unit = info.get("unit", "")
    group = info.get("group", "")
    tail = f" ({unit})" if unit and unit != "-" else ""
    return f"{group} · {label}{tail}" if group else f"{label}{tail}"


def _fmt(value: Any, digits: int = 3) -> str:
    """Compact number formatting that survives NaN and very small/large magnitudes."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(v):
        return "-"
    if v == 0:
        return "0"
    magnitude = abs(v)
    if magnitude >= 1e6 or magnitude < 1e-4:
        return f"{v:.{digits}g}"
    if magnitude >= 100:
        return f"{v:,.1f}"
    return f"{v:.{digits}g}"


def _p_text(p: float) -> str:
    """A p-value for a caption, without pretending to resolution it does not have."""
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 1e-3 else f"= {p:.3f}"


def _default_index(options: Sequence[str], preferred: Sequence[str]) -> int:
    for name in preferred:
        if name in options:
            return list(options).index(name)
    return 0


# ---------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------


def _build_figure(
    x: np.ndarray,
    y: np.ndarray,
    color_values: Optional[np.ndarray],
    x_label: str,
    y_label: str,
    color_label: Optional[str],
    meta: Dict[str, Dict[str, Any]],
    *,
    density: bool,
    log_x: bool,
    log_y: bool,
    marginals: bool,
    show_fit: bool,
    show_bin_overlay: bool,
    bins: Optional[pd.DataFrame],
    invert_x: bool,
    invert_y: bool,
) -> go.Figure:
    """One cross-plot, optionally with marginal histograms down the sides."""
    if marginals:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.86, 0.14],
            row_heights=[0.16, 0.84],
            horizontal_spacing=0.02,
            vertical_spacing=0.03,
            shared_xaxes=True,
            shared_yaxes=True,
            specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
        )
        main = {"row": 2, "col": 1}
    else:
        fig = make_subplots(rows=1, cols=1)
        main = {"row": 1, "col": 1}

    point_color = color_for(meta.get(y_label, {}).get("color") or "", PALETTE["primary"])

    if density:
        fig.add_trace(
            go.Histogram2d(
                x=x,
                y=y,
                nbinsx=60,
                nbinsy=60,
                colorscale="Blues",
                colorbar=dict(title="Trials", thickness=12, len=0.6, y=0.4),
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>trials: %{{z}}<extra></extra>",
            ),
            **main,
        )
    else:
        marker: Dict[str, Any] = dict(size=4, opacity=0.45, color=point_color)
        hover = f"{x_label}: %{{x:.4g}}<br>{y_label}: %{{y:.4g}}"
        if color_values is not None and color_label:
            marker = dict(
                size=4,
                opacity=0.65,
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=_axis_title(color_label, meta.get(color_label, {})),
                    thickness=12,
                    len=0.6,
                    y=0.4,
                ),
            )
            hover += f"<br>{color_label}: %{{marker.color:.4g}}"
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                marker=marker,
                name="Trials",
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            ),
            **main,
        )

    if show_bin_overlay and bins is not None and not bins.empty:
        drawn = bins[bins["count"] > 0]
        if not drawn.empty:
            fig.add_trace(
                go.Scatter(
                    x=drawn["bin_mid"],
                    y=drawn["mean"],
                    mode="lines+markers",
                    name="Binned mean",
                    line=dict(color=PALETTE["accent"], width=3),
                    marker=dict(size=6),
                    hovertemplate="bin mid %{x:.4g}<br>mean %{y:.4g}<extra>Binned mean</extra>",
                ),
                **main,
            )
            fig.add_trace(
                go.Scatter(
                    x=drawn["bin_mid"],
                    y=drawn["P50"],
                    mode="lines",
                    name="Binned P50",
                    line=dict(color=PALETTE["highlight"], width=2, dash="dash"),
                    hovertemplate="bin mid %{x:.4g}<br>P50 %{y:.4g}<extra>Binned P50</extra>",
                ),
                **main,
            )

    if show_fit:
        fitted = fit_line(x, y, log_x=log_x, log_y=log_y)
        if fitted is not None:
            fx, fy = fitted
            fig.add_trace(
                go.Scatter(
                    x=fx,
                    y=fy,
                    mode="lines",
                    name="Least-squares fit",
                    line=dict(color=PALETTE["neutral"], width=2, dash="dot"),
                    hoverinfo="skip",
                ),
                **main,
            )

    if marginals:
        fig.add_trace(
            go.Histogram(
                x=x,
                nbinsx=80,
                marker=dict(color=point_color),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"{x_label}: %{{x}}<br>trials: %{{y}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                y=y,
                nbinsy=80,
                marker=dict(color=point_color),
                opacity=0.7,
                showlegend=False,
                hovertemplate=f"{y_label}: %{{y}}<br>trials: %{{x}}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="n", row=1, col=1, showgrid=False)
        fig.update_xaxes(title_text="n", row=2, col=2, showgrid=False)

    fig.update_xaxes(title_text=_axis_title(x_label, meta.get(x_label, {})), **main)
    fig.update_yaxes(title_text=_axis_title(y_label, meta.get(y_label, {})), **main)
    if log_x:
        fig.update_xaxes(type="log", **main)
    if log_y:
        fig.update_yaxes(type="log", **main)
    # Depth increases downwards, as on the depth-area plots.
    if invert_x:
        fig.update_xaxes(autorange="reversed", **main)
        if marginals:
            fig.update_xaxes(autorange="reversed", row=1, col=1)
    if invert_y:
        fig.update_yaxes(autorange="reversed", **main)
        if marginals:
            fig.update_yaxes(autorange="reversed", row=2, col=2)

    # No figure title: the pair is named in a heading above the chart instead, so the
    # horizontal legend has the top of the plot to itself.
    fig.update_layout(
        height=640,
        # The top margin is deliberately generous: it is the band the horizontal
        # legend sits in. With less, the legend spills out of the figure box and
        # lands on top of the heading above the chart.
        margin=dict(l=70, r=30, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        bargap=0.02,
    )
    return fig


def _correlation_heatmap(matrix: pd.DataFrame, method_label: str) -> go.Figure:
    labels = list(matrix.columns)
    values = matrix.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(values, 2) if len(labels) <= HEATMAP_LABEL_LIMIT else None,
            texttemplate="%{text}" if len(labels) <= HEATMAP_LABEL_LIMIT else None,
            colorbar=dict(title=method_label, thickness=14),
            hovertemplate="%{y} vs %{x}<br>" + method_label + ": %{z:.3f}<extra></extra>",
        )
    )
    size = max(420, 34 * len(labels) + 220)
    fig.update_layout(
        height=size,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _drivers_bar(ranked: pd.DataFrame, target: str, method_label: str) -> go.Figure:
    top = ranked.head(15).iloc[::-1]
    colors = [
        PALETTE["primary"] if v >= 0 else PALETTE["accent"] for v in top["correlation"]
    ]
    fig = go.Figure(
        go.Bar(
            x=top["correlation"],
            y=top["variable"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="%{y}<br>" + method_label + ": %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 26 * len(top) + 120),
        margin=dict(l=10, r=20, t=40, b=40),
        title=f"Correlation with {target}",
        xaxis=dict(title=method_label, range=[-1, 1], zeroline=True),
    )
    return fig


# ---------------------------------------------------------------------------------
# Memoised statistics
#
# The fragment below re-runs in full whenever the script does - a sidebar widget
# re-runs the whole page, fragment or not - and the rank correlations are the
# most expensive thing on the Results page at 50,000 trials: rank_drivers alone
# measured 153 ms, with correlation_matrix at 49 ms, both of them inside an
# expander that is collapsed by default. Each wrapper keys on the exact values
# and options it passes through, so a different selection, method or data set
# lands on a different key rather than invalidating an old one. The returned
# frames are read-only to every caller here (`.rename`, `.to_csv`, `.head` and
# construction of new frames from their columns all copy).
# ---------------------------------------------------------------------------------


def _cached_bin_statistics(x, y, bin_width: float, use_exceedance: bool) -> pd.DataFrame:
    key = fingerprint("bin_stats", x, y, float(bin_width), use_exceedance)
    return memo(
        key, lambda: bin_statistics(x, y, bin_width, use_exceedance=use_exceedance)
    )


def _cached_pair_statistics(x, y) -> Dict[str, float]:
    key = fingerprint("pair_stats", x, y)
    return memo(key, lambda: pair_statistics(x, y))


def _cached_correlation_matrix(
    df: pd.DataFrame, columns: Sequence[str], method: str
) -> pd.DataFrame:
    # `columns` goes into the key in order, because it sets the row and column
    # order of the result; only the columns actually present carry data.
    present = [c for c in columns if c in df.columns]
    key = fingerprint(
        "corr_matrix", method, list(columns), present, df[present] if present else None
    )
    return memo(key, lambda: correlation_matrix(df, columns, method=method))


def _cached_rank_drivers(
    df: pd.DataFrame, target: str, columns: Sequence[str], method: str
) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    needed = present + ([target] if target in df.columns else [])
    key = fingerprint(
        "rank_drivers",
        method,
        target,
        list(columns),
        needed,
        df[needed] if needed else None,
    )
    return memo(key, lambda: rank_drivers(df, target, columns, method=method))


# ---------------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------------


def render_crossplot_section(res: Dict[str, Any], unit_system: str) -> None:
    """Entry point called from the Results page."""
    st.markdown("---")
    st.markdown("## Cross-plot & Statistics")
    st.caption(
        "Plot any two simulated parameters against each other, and read off the "
        "statistics of one per interval of the other - for example the mean STOIIP "
        "in each 10 m of contact depth."
    )
    _section(res, unit_system)


@st.fragment
def _section(res: Dict[str, Any], unit_system: str) -> None:
    """The interactive part, in a fragment so a widget change does not re-run the page."""
    include_raw = st.checkbox(
        "Show every trial column",
        value=False,
        key="xplot_include_raw",
        help=(
            "Adds the raw export columns, including the duplicate unit variants of each "
            "volume. Off by default: the curated list carries each quantity once, in "
            "your chosen unit system."
        ),
    )

    frame, meta = build_explorer_frame(res, unit_system, include_raw=include_raw)
    if frame.empty or len(frame.columns) < 2:
        st.info(
            "The cross-plot needs at least two simulated parameters. Configure the "
            "inputs and run the simulation, then come back to this section."
        )
        return

    options = list(frame.columns)
    n_total = len(frame)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_label = st.selectbox(
            "X variable",
            options,
            index=_default_index(
                options,
                ["HC contact depth (OWC)", "GOC depth", "Porosity", "GRV total"],
            ),
            format_func=lambda o: _option_label(o, meta),
            key="xplot_x",
        )
    with c2:
        y_label = st.selectbox(
            "Y variable",
            options,
            index=_default_index(
                options,
                ["Oil in place (STOIIP)", "Recoverable oil", "Total HC resource (THR)"],
            ),
            format_func=lambda o: _option_label(o, meta),
            key="xplot_y",
        )
    with c3:
        color_label = st.selectbox(
            "Colour by",
            ["(none)"] + options,
            index=0,
            format_func=lambda o: o if o == "(none)" else _option_label(o, meta),
            key="xplot_color",
        )
    color_label = None if color_label == "(none)" else color_label

    o1, o2, o3, o4, o5 = st.columns([1.4, 1, 1, 1, 1.2])
    with o1:
        view = st.radio(
            "View",
            ["Scatter", "Density"],
            index=1 if n_total > DENSITY_AUTO_THRESHOLD else 0,
            horizontal=True,
            key="xplot_view",
        )
    with o2:
        log_x = st.checkbox("Log X", value=False, key="xplot_log_x")
    with o3:
        log_y = st.checkbox("Log Y", value=False, key="xplot_log_y")
    with o4:
        marginals = st.checkbox("Marginals", value=True, key="xplot_marginals")
    with o5:
        show_fit = st.checkbox("Fit line", value=True, key="xplot_fit")

    meta_x = meta.get(x_label, {})
    meta_y = meta.get(y_label, {})
    invert_default = bool(meta_x.get("invert") or meta_y.get("invert"))
    invert = st.checkbox(
        "Depth axis increases downwards",
        value=invert_default,
        key="xplot_invert",
        disabled=not invert_default,
        help=(
            "Enabled when a depth variable is on an axis. Matches the convention on the "
            "depth-area plots."
        ),
    )
    invert_x = invert and bool(meta_x.get("invert"))
    invert_y = invert and bool(meta_y.get("invert"))

    # --- filters ---
    mask = np.ones(n_total, dtype=bool)
    with st.expander("Filter trials (optional)", expanded=False):
        f1, f2 = st.columns(2)
        for slot, column in (("a", f1), ("b", f2)):
            with column:
                choice = st.selectbox(
                    f"Filter {slot.upper()}",
                    ["(none)"] + options,
                    index=0,
                    format_func=lambda o: o if o == "(none)" else _option_label(o, meta),
                    key=f"xplot_filter_{slot}",
                )
                if choice == "(none)":
                    continue
                values = frame[choice].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
                    st.caption("This parameter does not vary - nothing to filter.")
                    continue
                lo, hi = float(np.min(finite)), float(np.max(finite))
                low, high = st.slider(
                    _axis_title(choice, meta.get(choice, {})),
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                    step=(hi - lo) / 100.0,
                    key=f"xplot_range_{slot}::{choice}",
                )
                mask &= (values >= low) & (values <= high)

    dff = frame[mask]
    x = dff[x_label].to_numpy(dtype=float)
    y = dff[y_label].to_numpy(dtype=float)
    color_values = (
        dff[color_label].to_numpy(dtype=float)
        if color_label and color_label in dff.columns
        else None
    )

    if mask.sum() < n_total:
        st.caption(
            f"Filtered: **{int(mask.sum()):,}** of {n_total:,} trials "
            f"({mask.sum() / n_total:.1%})."
        )
    if x.size == 0:
        st.warning("No trials left after filtering. Widen the ranges.")
        return

    if log_x and np.any(x <= 0):
        st.caption(
            f"Log X hides {int(np.sum(x <= 0)):,} trials at or below zero; the "
            "statistics below still use every trial shown."
        )
    if log_y and np.any(y <= 0):
        st.caption(f"Log Y hides {int(np.sum(y <= 0)):,} trials at or below zero.")

    # --- binned statistics ---
    x_unit = _unit_of(x_label, meta)
    y_unit = _unit_of(y_label, meta)
    # " m" / "" - a suffix that can be dropped straight into a sentence.
    x_suffix = f" {x_unit}" if x_unit else ""
    per_x = f"per {x_unit}" if x_unit else f"per unit {x_label.lower()}"
    b1, b1b, b2 = st.columns([1, 1, 2.4])
    with b1b:
        hide_empty = st.checkbox(
            "Hide empty intervals",
            value=False,
            key="xplot_hide_empty",
            help=(
                "Intervals with no trials are listed by default so the range reads "
                "continuously. Turn this on when the tails are sparse."
            ),
        )
    with b1:
        default_width = suggest_bin_width(x, depth_like=bool(meta_x.get("invert")))
        bin_width = st.number_input(
            f"Bin width ({x_unit})" if x_unit else "Bin width",
            min_value=0.0,
            value=float(default_width),
            step=float(default_width) / 2.0,
            format="%g",
            key=f"xplot_bin_width::{x_label}",
            help=(
                "Width of the X interval the Y statistics are computed over. "
                "The table below reports one row per interval."
            ),
        )

    bins: Optional[pd.DataFrame] = None
    bin_error: Optional[str] = None
    use_exceedance = bool(st.session_state.get("percentile_exceedance", True))
    if bin_width and bin_width > 0:
        try:
            bins = _cached_bin_statistics(x, y, bin_width, use_exceedance)
        except ValueError as exc:
            bin_error = str(exc)
    with b2:
        if bin_error:
            st.warning(f"Binning: {bin_error}")
        elif bins is not None and not bins.empty:
            filled = int((bins["count"] > 0).sum())
            st.caption(
                f"{len(bins)} intervals of {bin_width:g}{x_suffix} "
                f"({filled} populated). The red line on the plot is this table's mean."
            )

    fig = _build_figure(
        x,
        y,
        color_values,
        x_label,
        y_label,
        color_label,
        meta,
        density=(view == "Density"),
        log_x=log_x,
        log_y=log_y,
        marginals=marginals,
        show_fit=show_fit,
        show_bin_overlay=True,
        bins=bins,
        invert_x=invert_x,
        invert_y=invert_y,
    )
    st.markdown(f"#### {y_label} vs {x_label}")
    st.plotly_chart(fig, width="stretch")
    if view == "Density" and color_label:
        st.caption(
            "Colour-by applies to the scatter view; the density view already maps "
            "trial count to colour."
        )

    # --- pair statistics ---
    stats = _cached_pair_statistics(x, y)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Pearson r", _fmt(stats["pearson_r"]), help="Linear association, -1 to +1.")
    s2.metric(
        "Spearman ρ",
        _fmt(stats["spearman_rho"]),
        help="Rank association: catches monotonic but curved relationships.",
    )
    s3.metric("R²", _fmt(stats["r_squared"]), help="Variance in Y explained by a straight line in X.")
    s4.metric("Trials", f"{stats['n']:,}")
    slope_label = "Slope"
    if y_unit and x_unit:
        slope_label = f"Slope ({y_unit}/{x_unit})"
    elif y_unit:
        slope_label = f"Slope ({y_unit} per unit X)"
    s5.metric(
        slope_label,
        _fmt(stats["slope"]),
        help=f"Least-squares gradient of {y_label} on {x_label}, in display units.",
    )

    notes: List[str] = []
    if np.isfinite(stats["pearson_p"]):
        notes.append("Pearson p " + _p_text(stats["pearson_p"]))
    if np.isfinite(stats["spearman_p"]):
        notes.append("Spearman p " + _p_text(stats["spearman_p"]))
    if np.isfinite(stats["log_slope"]):
        notes.append(
            f"log-log fit: Y ∝ X^{_fmt(stats['log_slope'], 3)} "
            f"(R² {_fmt(stats['log_r_squared'], 2)})"
        )
    if notes:
        st.caption(" · ".join(notes))

    # --- binned table ---
    export_bins = pd.DataFrame()
    if bins is not None and not bins.empty:
        interval_header = (
            f"{x_label} interval ({x_unit})" if x_unit else f"{x_label} interval"
        )
        display = pd.DataFrame(
            {
                interval_header: [
                    f"{lo:g} – {hi:g}" for lo, hi in zip(bins["bin_min"], bins["bin_max"])
                ],
                "Trials": bins["count"].astype(int),
                f"Mean {y_label}": bins["mean"],
                "P10": bins["P10"],
                "P50": bins["P50"],
                "P90": bins["P90"],
                "Std dev": bins["std"],
                f"Mean {per_x}": bins["mean_per_unit_x"],
            }
        )
        # An interval with no trials has no statistics; Streamlit renders those cells
        # as a greyed "None", with Trials = 0 beside it. Empty intervals are kept by
        # default so the depth range reads continuously, and can be hidden when the
        # tails are sparse.
        if hide_empty:
            display = display[bins["count"].to_numpy() > 0]
        convention = (
            "exceedance (P10 high)" if use_exceedance else "non-exceedance (P10 low)"
        )
        st.markdown(
            f"**{y_label} per interval of {x_label}** "
            f"— percentiles on the {convention} convention"
            + (f", values in {y_unit}" if y_unit and y_unit != "-" else "")
        )
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.4g")
                for col in display.columns
                if col not in (interval_header, "Trials")
            },
        )

        # The download matches what is on screen, empty intervals included or not.
        unit_tag = f"_{x_unit}" if x_unit else ""
        export_source = bins[bins["count"] > 0] if hide_empty else bins
        export_bins = export_source.rename(
            columns={
                "bin_min": f"{x_label}_min{unit_tag}",
                "bin_max": f"{x_label}_max{unit_tag}",
                "bin_mid": f"{x_label}_mid{unit_tag}",
                "count": "trials",
                "mean": f"mean_{y_label}",
                "std": "std_dev",
                "mean_per_unit_x": f"mean_{per_x}".replace(" ", "_"),
            }
        )
        st.download_button(
            "Download binned statistics (CSV)",
            data=export_bins.to_csv(index=False).encode("utf-8"),
            file_name="scopehc_crossplot_bins.csv",
            mime="text/csv",
            key="xplot_dl_bins",
        )

    # --- correlation matrix ---
    matrix = pd.DataFrame()
    with st.expander("Correlation matrix & drivers", expanded=False):
        m1, m2 = st.columns([1, 3])
        with m1:
            method_name = st.radio(
                "Method",
                ["Spearman", "Pearson"],
                index=0,
                key="xplot_matrix_method",
                help="Spearman ranks the data first, so it reads curved relationships too.",
            )
        method = method_name.lower()
        defaults = [c for c in EXPLORER_MATRIX_DEFAULTS if c in options]
        if not defaults:
            defaults = options[: min(8, len(options))]
        with m2:
            selected = st.multiselect(
                "Variables",
                options,
                default=defaults,
                format_func=lambda o: _option_label(o, meta),
                key="xplot_matrix_vars",
            )

        if len(selected) < 2:
            st.info("Pick at least two variables.")
        else:
            matrix = _cached_correlation_matrix(dff, selected, method)
            if matrix.empty:
                st.info("Not enough variation in the selected variables to correlate.")
            else:
                st.plotly_chart(
                    _correlation_heatmap(matrix, method_name), width="stretch"
                )
                st.download_button(
                    f"Download {method_name} matrix (CSV)",
                    data=matrix.to_csv().encode("utf-8"),
                    file_name=f"scopehc_correlation_{method}.csv",
                    mime="text/csv",
                    key="xplot_dl_matrix",
                )

                ranked = _cached_rank_drivers(dff, y_label, selected, method)
                if not ranked.empty:
                    st.plotly_chart(
                        _drivers_bar(ranked, y_label, method_name), width="stretch"
                    )
                    st.caption(
                        "Ranked by absolute correlation with the current Y variable. "
                        "A strong input here is where tightening the uncertainty pays off."
                    )

    # Mirrored for the Excel export on the Downloads page.
    st.session_state["crossplot_export"] = {
        "x": x_label,
        "y": y_label,
        "x_unit": x_unit,
        "y_unit": y_unit,
        "bin_width": float(bin_width) if bin_width else None,
        "n_used": int(x.size),
        "n_total": int(n_total),
        "use_exceedance": use_exceedance,
        "stats": stats,
        "bins": export_bins,
        "matrix": matrix,
    }
