"""
Plots: plotly theme + hist/CDF helper + depth plots
"""
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from .config import PARAM_COLORS, PALETTE


def init_plotly_theme():
    """Initialize the SCOPE-HC plotly theme."""
    if "scopehc" in pio.templates:
        pio.templates.default = "scopehc"
        return
    
    tpl = pio.templates["plotly_white"]
    tpl.layout.colorway = [
        PARAM_COLORS.get("Oil_STB_rec"),
        PARAM_COLORS.get("Gas_free_scf_rec"),
        PARAM_COLORS.get("Gas_assoc_scf_rec"),
        PARAM_COLORS.get("Cond_STB_rec"),
        PARAM_COLORS.get("GRV_total_m3"),
    ]
    tpl.layout.paper_bgcolor = PALETTE["bg_light"]
    tpl.layout.plot_bgcolor = "#FFFFFF"
    tpl.layout.font = {"size": 13, "color": PALETTE["text_primary"]}
    
    pio.templates["scopehc"] = tpl
    pio.templates.default = "scopehc"


def color_for(key, fallback=None):
    """Get color for parameter key.
    
    Returns a color from PARAM_COLORS based on the parameter name.
    Falls back to a neutral gray if the key is not found.
    """
    if fallback is None:
        fallback = "#BDBDBD"  # Neutral gray fallback
    return PARAM_COLORS.get(key, fallback)


def rgba_from_hex(hex_color: str, alpha: float) -> str:
    """Convert a hex color to an rgba() string with the given alpha."""
    hex_color = (hex_color or "").lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0, 0, 0, {alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def hist_and_cdf(x, key_name: str, unit_label: str, nbins=100):
    """
    Create histogram and CDF plot with consistent colors and P10/P50/P90 annotations.
    
    Args:
        x: Data array
        key_name: Parameter name for color lookup from PARAM_COLORS
        unit_label: Unit label for display
        nbins: Number of histogram bins (default 100)
        
    Returns:
        tuple: (fig, stats_dict)
    """
    # Get color from PARAM_COLORS map
    c = color_for(key_name)
    x = np.asarray(x, float)
    
    # Calculate percentiles using convention-aware summarize_array
    from scopehc.utils import summarize_array
    stats = summarize_array(x)
    p10 = stats.get("P10", np.percentile(x, 10))
    p50 = stats.get("P50", np.percentile(x, 50))
    p90 = stats.get("P90", np.percentile(x, 90))
    mean_val = stats.get("mean", np.mean(x))
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram with nbinsx=100
    fig.add_histogram(
        x=x, 
        nbinsx=100,  # Always use 100 bins
        name=f"{key_name} ({unit_label})",
        marker=dict(color=c), 
        opacity=0.85, 
        hovertemplate="%{x:.4g}"
    )
    
    # Add P10/P50/P90 vertical lines with annotations
    for v, lbl in [(p10, "P10"), (p50, "P50"), (p90, "P90")]:
        fig.add_vline(
            x=v, 
            line_width=2, 
            line_dash="dash", 
            line_color=c, 
            opacity=0.75
        )
        fig.add_annotation(
            x=v, 
            y=1.02, 
            yref="paper", 
            xanchor="center", 
            showarrow=False,
            text=f"{lbl}={v:,.3g} {unit_label}", 
            font=dict(color=c, size=10)
        )
    
    # Add CDF with same color as histogram
    xs = np.sort(x)
    # Check percentile convention for CDF display
    try:
        import streamlit as st
        use_exceedance = st.session_state.get("percentile_exceedance", True)
    except (RuntimeError, AttributeError):
        use_exceedance = True
    
    if use_exceedance:
        # For exceedance convention: show probability of exceeding (1 - CDF)
        ys = np.linspace(1, 0, len(xs))  # Reversed: 1 to 0
        cdf_name = f"Probability of Exceedance {key_name}"
    else:
        # For non-exceedance convention: show standard CDF
        ys = np.linspace(0, 1, len(xs))  # Standard: 0 to 1
        cdf_name = f"CDF {key_name}"
    
    fig.add_trace(go.Scatter(
        x=xs, 
        y=ys, 
        mode="lines", 
        name=cdf_name,
        line=dict(color=c, width=2),  # Same color as histogram
        opacity=0.95, 
        yaxis="y2",
        hovertemplate="%{x:.4g} → %{y:.1%}"
    ))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=40, r=30, t=30, b=60),
        yaxis2=dict(
            overlaying="y", 
            side="right", 
            range=[0, 1], 
            showgrid=False, 
            title="Probability of Exceedance" if use_exceedance else "CDF"
        ),
        xaxis_title=f"{key_name} ({unit_label})",
        yaxis_title="Probability Density",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        width=None,
        height=400
    )
    
    stats = {
        'P10': p10,
        'P50': p50,
        'P90': p90,
        'Mean': mean_val,
        'Std': np.std(x),
        'Min': np.min(x),
        'Max': np.max(x)
    }
    
    return fig, stats


def make_depth_area_plot(depths_m, areas_km2, spill_depth_m, hc_depth_m, goc_depth_m=None):
    """
    Create depth-area plot with contacts.
    
    Args:
        depths_m: Array of depths in meters
        areas_km2: Array of areas in km²
        spill_depth_m: Spill point depth
        hc_depth_m: Hydrocarbon contact depth
        goc_depth_m: Optional GOC depth
        
    Returns:
        plotly figure
    """
    fig = go.Figure()
    
    # Add area-depth curve
    fig.add_trace(go.Scatter(
        x=areas_km2,
        y=depths_m,
        mode='lines+markers',
        name='Area-Depth',
        line=dict(color=color_for("GRV_total_m3"), width=2),
        marker=dict(size=4, color=color_for("GRV_total_m3"))
    ))
    
    # Add spill point
    fig.add_hline(
        y=spill_depth_m,
        line_dash="dash",
        line_color=color_for("SpillPoint"),
        annotation_text=f"Spill Point: {spill_depth_m:.0f} m"
    )
    
    # Add HC contact
    fig.add_hline(
        y=hc_depth_m,
        line_dash="dash",
        line_color=color_for("HCDepth"),
        annotation_text=f"HC Contact: {hc_depth_m:.0f} m"
    )
    
    # Add GOC if provided
    if goc_depth_m is not None:
        fig.add_hline(
            y=goc_depth_m,
            line_dash="dot",
            line_color=color_for("GOC"),
            annotation_text=f"GOC: {goc_depth_m:.0f} m"
        )
    
    fig.update_layout(
        xaxis_title="Area (km²)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed"),  # Depth increases downward
        title="Area-Depth Relationship"
    )
    
    return fig


def make_area_volume_plot(
	depths: np.ndarray,
	top_interp: np.ndarray,
	base_interp: np.ndarray,
	vol_top: np.ndarray,
	vol_base: np.ndarray,
	dgrv: np.ndarray,
	spill_point_m: float,
	grv_sp_km2m: float,
	mean_effective_hc_depth: float = None,
	mean_goc_depth: float = None,
	top_p10: np.ndarray = None,
	top_p90: np.ndarray = None,
	base_p10: np.ndarray = None,
	base_p90: np.ndarray = None,
	dgrv_p10: np.ndarray = None,
	dgrv_p90: np.ndarray = None,
) -> go.Figure:
	"""Create a two-panel plot: area vs depth and volume/dGRV vs depth."""
	fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Area vs Depth", "Volume vs Depth"))

	# Left: Areas with P10/P90 lines if provided
	if top_p10 is not None and top_p90 is not None:
		# Add P10/P90 lines for top area
		top_color = color_for("GRV_oil_m3")
		fig.add_trace(go.Scatter(
			x=top_p10,
			y=depths,
			mode='lines',
			line=dict(color=rgba_from_hex(top_color, 0.75), width=2, dash='dash'),
			showlegend=True,
			name='Top Area P10',
			legendgroup='uncertainty_top'
		), row=1, col=1)
		
		fig.add_trace(go.Scatter(
			x=top_p90,
			y=depths,
			mode='lines',
			line=dict(color=rgba_from_hex(top_color, 0.75), width=2, dash='dash'),
			showlegend=True,
			name='Top Area P90',
			legendgroup='uncertainty_top'
		), row=1, col=1)
		
		# Add P10/P90 lines for base area
		if base_p10 is not None and base_p90 is not None:
			base_color = color_for("GRV_gas_m3")
			fig.add_trace(go.Scatter(
				x=base_p10,
				y=depths,
				mode='lines',
				line=dict(color=rgba_from_hex(base_color, 0.75), width=2, dash='dash'),
				showlegend=True,
				name='Base Area P10',
				legendgroup='uncertainty_base'
			), row=1, col=1)
			
			fig.add_trace(go.Scatter(
				x=base_p90,
				y=depths,
				mode='lines',
				line=dict(color=rgba_from_hex(base_color, 0.75), width=2, dash='dash'),
				showlegend=True,
				name='Base Area P90',
				legendgroup='uncertainty_base'
			), row=1, col=1)

	# Left: Mean (P50) Areas
	fig.add_trace(go.Scatter(x=top_interp, y=depths, name="Top Area (Mean)", mode="lines", line=dict(color=color_for("GRV_oil_m3"), width=2)), row=1, col=1)
	fig.add_trace(go.Scatter(x=base_interp, y=depths, name="Base Area (Mean)", mode="lines", line=dict(color=color_for("GRV_gas_m3"), width=2)), row=1, col=1)

	# Right: Volumes and dGRV
	fig.add_trace(go.Scatter(x=vol_top, y=depths, name="Top Volume", mode="lines", line=dict(color=color_for("GRV_oil_m3"))), row=1, col=2)
	fig.add_trace(go.Scatter(x=vol_base, y=depths, name="Base Volume", mode="lines", line=dict(color=color_for("GRV_gas_m3"))), row=1, col=2)
	fig.add_trace(go.Scatter(x=dgrv, y=depths, name="dGRV (km²·m)", mode="lines", line=dict(color=color_for("GRV_total_m3"))), row=1, col=2)
	
	# Add dGRV P10/P90 lines if provided
	if dgrv_p10 is not None and dgrv_p90 is not None:
		dgrv_color = color_for("GRV_total_m3")
		fig.add_trace(go.Scatter(
			x=dgrv_p10,
			y=depths,
			mode='lines',
			line=dict(color=rgba_from_hex(dgrv_color, 0.65), width=2, dash='dash'),
			showlegend=True,
			name='dGRV P10',
			legendgroup='uncertainty_dgrv'
		), row=1, col=2)
		
		fig.add_trace(go.Scatter(
			x=dgrv_p90,
			y=depths,
			mode='lines',
			line=dict(color=rgba_from_hex(dgrv_color, 0.65), width=2, dash='dash'),
			showlegend=True,
			name='dGRV P90',
			legendgroup='uncertainty_dgrv'
		), row=1, col=2)

	# Spill point marker
	if not np.isnan(grv_sp_km2m):
		fig.add_trace(
			go.Scatter(
				x=[grv_sp_km2m], y=[spill_point_m], mode="markers",
				marker=dict(symbol="star", size=12, color=color_for("SpillPoint")),
				name="Mean Spill Point"
			),
			row=1, col=2
		)
	
	# Mean GOC depth marker
	if mean_goc_depth is not None:
		# Find the volume at mean GOC depth
		vol_at_goc = np.interp(mean_goc_depth, depths, dgrv)
		fig.add_trace(
			go.Scatter(
				x=[vol_at_goc], y=[mean_goc_depth], mode="markers",
				marker=dict(symbol="square", size=12, color=color_for("GOC")),
				name="Mean GOC Depth"
			),
			row=1, col=2
		)

	fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
	fig.update_xaxes(title_text="Area (km²)", row=1, col=1)
	fig.update_xaxes(title_text="Volume (km²·m)", row=1, col=2)
	fig.update_layout(
		margin=dict(l=40, r=40, t=60, b=60), 
		legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
		width=None,
		height=400
	)
	return fig


def extract_param_name_from_title(title: str) -> str:
    """Extract parameter name from title for consistent color mapping."""
    # Map common titles to parameter names
    title_lower = title.lower()
    
    if "gas" in title_lower and "column" in title_lower and "height" in title_lower:
        return "Gas_Column_Height"
    elif "oil" in title_lower and "column" in title_lower and "height" in title_lower:
        return "Oil_Column_Height"
    elif "effective" in title_lower and "hc" in title_lower:
        return "Effective_HC_Depth"
    elif "goc" in title_lower:
        return "GOC"
    elif "hc" in title_lower and "depth" in title_lower or "hydrocarbon" in title_lower and "contact" in title_lower:
        return "HCDepth"
    elif "spill" in title_lower:
        return "SpillPoint"
    elif "oil" in title_lower and "boe" in title_lower:
        return "Oil_BOE"
    elif "gas" in title_lower and "free" in title_lower and "boe" in title_lower:
        return "Gas_free_BOE"
    elif "gas" in title_lower and "assoc" in title_lower and "boe" in title_lower:
        return "Gas_assoc_BOE"
    elif "condensate" in title_lower or "cond" in title_lower:
        if "boe" in title_lower:
            return "Cond_BOE"
        else:
            return "Cond_STB_rec"
    elif "oil" in title_lower and "stb" in title_lower:
        return "Oil_STB_rec"
    elif "gas" in title_lower and "scf" in title_lower and "free" in title_lower:
        return "Gas_free_scf_rec"
    elif "gas" in title_lower and "scf" in title_lower and "assoc" in title_lower:
        return "Gas_assoc_scf_rec"
    elif "thr" in title_lower or "total hydrocarbon" in title_lower:
        return "Total_surface_BOE"
    elif "grv" in title_lower and "oil" in title_lower:
        return "GRV_oil_m3"
    elif "grv" in title_lower and "gas" in title_lower:
        return "GRV_gas_m3"
    elif "grv" in title_lower:
        return "GRV_total_m3"
    elif "pv" in title_lower and "oil" in title_lower:
        return "PV_oil_m3"
    elif "pv" in title_lower and "gas" in title_lower:
        return "PV_gas_m3"
    elif "bg" in title_lower:
        return "Bg_rb_per_scf"
    elif "bo" in title_lower or "invbo" in title_lower:
        return "InvBo_STB_per_rb"
    elif "gor" in title_lower:
        return "GOR_scf_per_STB"
    elif "cgr" in title_lower:
        return "CGR_STB_per_MMscf"
    else:
        # Default color for unrecognized parameters
        return "GRV_total_m3"
