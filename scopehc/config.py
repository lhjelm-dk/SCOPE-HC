"""
Configuration: constants, units, colors, help text, and schemas
"""

from dataclasses import dataclass

# Units & constants
RB_PER_M3 = 6.28981077
RCF_PER_RB = 5.614583

UNIT_DISPLAY = {
    "oilfield": {"oil_surface": "STB", "gas_surface": "scf", "boe": "BOE", "vol_m3": "m³"},
    "si": {"oil_surface": "m³", "gas_surface": "Sm³", "boe": "BOE", "vol_m3": "m³"},
}

# Global color palette
PALETTE = {
    "primary": "#1F4B99",      # strong blue for primary accents
    "secondary": "#587246",    # muted green for secondary elements
    "accent": "#FF6B6B",       # bold red accent for highlights
    "neutral": "#5B4B3A",      # earthy neutral brown
    "highlight": "#D98A3A",    # warm amber highlight
    "bg_light": "#F5F1E6",     # warm light background
    "bg_dark": "#2C2A25",
    "text_primary": "#1E1B16",
    "text_secondary": "#6D665C",
    "border": "#D6CEC0",
}

# Color map (consistent across plots)
# Structured by category: Oil (light green), Gas (light red), Condensate (light orange),
# Rock/Geometry (earthy neutrals), Derived/Systemic (light blue-gray)
PARAM_COLORS = {
    # --- Oil-related (light green) ---
    "Oil_STB_rec": "#A8E6A2",
    "Oil_BOE": "#A8E6A2",
    "Oil_inplace": "#A8E6A2",
    "Oil_recoverable": "#81C784",
    "Bo": "#B2DFB2",
    "InvBo_STB_per_rb": "#B2DFB2",
    "Shc_oil": "#C8E6C9",
    "Soil": "#AED581",
    "Oil_Column_Height": "#C5E1A5",
    "GRV_oil_m3": "#C5E1A5",
    "PV_oil_m3": "#C5E1A5",
    "PV_oil_hc_m3": "#A8E6A2",

    # --- Gas-related (light red) ---
    "Gas_free_scf_rec": "#F8BBD0",
    "Gas_free_BOE": "#F8BBD0",
    "Gas_inplace": "#F8BBD0",
    "Gas_recoverable": "#F48FB1",
    "Gas_assoc_scf_rec": "#F8BBD0",
    "Gas_assoc_BOE": "#F8BBD0",
    "Bg": "#FCE4EC",
    "Bg_rb_per_scf": "#FCE4EC",
    "Shc_gas": "#F8BBD0",
    "Gas_Column_Height": "#F48FB1",
    "GRV_gas_m3": "#F48FB1",
    "PV_gas_m3": "#F48FB1",
    "PV_gas_hc_m3": "#F8BBD0",
    "GOR_scf_per_STB": "#F06292",
    "Total_gas_scf_rec": "#F48FB1",

    # --- Condensate-related (light orange) ---
    "Cond_STB_rec": "#FFD8A6",
    "Cond_BOE": "#FFD8A6",
    "Condensate_recoverable": "#FFD8A6",
    "Condensate_inplace": "#FFE0B2",
    "CGR": "#FFCC80",
    "CGR_STB_per_MMscf": "#FFCC80",

    # --- Common rock properties (earthy neutrals) ---
    "Por": "#D7CCC8",
    "Porosity": "#D7CCC8",
    "NtG": "#BCAAA4",
    "GRV_total_m3": "#A1887F",
    "Sw": "#E0E0E0",
    "Sw_global": "#E0E0E0",
    "Sw_HCzone_input": "#E0E0E0",
    "Sw_oilzone": "#E0E0E0",
    "Sw_gaszone": "#E0E0E0",
    "Shc_global": "#BDBDBD",
    "Shc_oil_input": "#BDBDBD",
    "Shc_gas_input": "#BDBDBD",

    # --- Geometry / structure (earthy neutrals) ---
    "HCDepth": "#A1887F",
    "Effective_HC_Depth": "#A1887F",
    "GOC": "#8C6E4A",
    "SpillPoint": "#D98A3A",

    # --- Derived/systemic ---
    "RF": "#90CAF9",
    "RF_oil": "#90CAF9",
    "RF_gas": "#90CAF9",
    "RF_cond": "#90CAF9",
    "Probability": "#BBDEFB",
    "Uncertainty": "#B3E5FC",
    "Correlation": "#B0BEC5",
    "Total_surface_BOE": "#90CAF9",
}

# Help text (short + clear)
HELP = {
    "quick_start": "1) Choose GRV method. 2) Set Spill & HC depths (and GOC for depth-based). 3) Enter Fluids (Bg, 1/Bo, GOR, CGR). 4) Set RFs. 5) Run and review P10/P50/P90.",
    "grv_method": "Pick how GRV is computed. Depth-based methods support Gas–Oil Contact (GOC). Direct & Area×Thickness×GCF use an explicit oil/gas split.",
    "spill_point": "Spill point depth (m). The star (★) on depth plots marks the mean spill depth.",
    "eff_hc_depth": "Effective hydrocarbon contact (m): base of the hydrocarbon column.",
    "goc": "Gas–Oil Contact depth (m), or derived via oil fraction or oil column height. Gas above, oil below.",
    "ntg": "Net-to-Gross (0–1): fraction of the gross interval that is net reservoir.",
    "por": "Porosity φ (0–1): effective porosity used to compute pore volume.",
    "rf_oil": "Oil RF (0–1): fraction of oil in-place recovered to surface.",
    "rf_gas": "Gas RF (0–1): fraction of free gas in-place recovered to surface.",
    "rf_cond": "Condensate RF (0–1): fraction of condensate recovered from produced free gas.",
    "bg": "Gas FVF Bg (rb/scf). If using 1/Bg, units must be scf/rcf. Conversion: Bg = 1 / ((1/Bg)×5.614583).",
    "invbo": "Inverse Bo (STB/rb): converts reservoir barrels to stock-tank barrels.",
    "gor": "Solution GOR (scf/STB): associated gas per STB of oil.",
    "cgr": "CGR (STB/MMscf): condensate per million scf of produced free gas.",
    "boe_factor": "Gas-to-BOE conversion (scf/BOE). Default 6,000 (USGS); heat-content bases often use 5,620–5,800 scf/BOE—adjust to match company guidance.",
    "trials": "Monte Carlo trials. Higher → smoother distributions, longer runtime.",
    "seed": "Random seed for reproducibility.",
    "charts": "Histograms show frequency; dashed lines mark P10/P50/P90. CDF line uses the same color.",
    "area_uncert": "When enabled (depth methods only), a single trial-wise multiplier is sampled and applied to every area value in the depth vs. area table. Default: PERT(0.8, 1.0, 1.2).",
    "area_uncert_dist": "Choose the distribution for the area multiplier. PERT is typical for expert judgment; Triangular and Uniform are also supported.",
}

WORKFLOW_RIBBON = """
<div style="margin:.5rem 0 1rem 0;padding:.6rem 1rem;border-radius:8px;background:#f7f7fb;border:1px solid #ececf3;">
</div>
"""

DEFAULTS = {
    "Bg_rb_per_scf": (0.0045, 0.0055, 0.0065),
    "InvBo_STB_per_rb": (1 / 1.50, 1 / 1.35, 1 / 1.20),  # ≈ (0.6667, 0.7407, 0.8333)
    "GOR_scf_per_STB": (400, 800, 1200),
    "CGR_STB_per_MMscf": (20, 45, 80),
    "Porosity": (0.15, 0.20, 0.25),
    "NtG": (0.6, 0.75, 0.9),
    "RF_oil": (0.25, 0.35, 0.45),
    "RF_gas": (0.6, 0.7, 0.85),
    "RF_cond": (0.5, 0.6, 0.8),
    "scf_per_BOE": 6000.0,
    "AreaMultiplier_PERT": (0.8, 1.0, 1.2),
}

# In-app one-line help (expanded from original HELP)
HELP_EXTENDED = {
    **HELP,  # Include original help entries
    "cy": "Condensate yield (stock tank barrels per million standard cubic feet of gas)",
    "unit_system": "Choose between oilfield units (STB, scf) or SI units (m³, Sm³)",
    "area_thick": "Area (km²) and thickness (m) for geometric GRV calculation",
    "gcf": "Geometric correction factor (0–1): accounts for reservoir geometry and closure",
    "correlation": "Statistical correlation between parameters (-1 to +1): positive means higher values tend to occur together",
}

# Optional: About SCOPE-HC section text
ABOUT_SCOPE_HC = """
## About SCOPE-HC

SCOPE-HC is a Monte Carlo volumetric estimator for hydrocarbon resources. It uses industry-standard volumetric equations combined with uncertainty quantification to provide probabilistic estimates of in-place and recoverable hydrocarbon volumes.

### Method Overview

The application follows a structured workflow:
1. **GRV Calculation**: Choose from multiple methods (Direct, Geometric, or Depth-based integration)
2. **Hydrocarbon Fill**: Define spill points, HC contacts, and optional Gas-Oil Contact (GOC)
3. **Rock Properties**: Specify Net-to-Gross (NtG) and Porosity
4. **Fluids & Recovery**: Input PVT properties (Bg, Bo) and set recovery factors
5. **Results**: View distributions, P10/P50/P90 statistics, and sensitivity analysis

### Key Equations

- **Pore Volume**: \\( PV = GRV \\times NtG \\times \\phi \\)
- **Oil in Place**: \\( N = \\frac{PV_{oil}}{B_o} \\)
- **Gas in Place**: \\( G = \\frac{PV_{gas}}{B_g} \\)
- **Recoverable**: \\( R = IP \\times RF \\)
- **BOE Conversion**: \\( BOE = Oil + \\frac{Gas}{scf/BOE} \\)

### Uncertainty Quantification

All key inputs are treated as probability distributions (PERT, Triangular, Lognormal, etc.), allowing the application to quantify:
- **P10**: Optimistic (90% chance of exceeding)
- **P50**: Median (most likely)
- **P90**: Conservative (only 10% chance of exceeding)

### Assumptions

- Reservoir is at hydrostatic pressure
- Fluids behave according to standard PVT correlations
- Recovery factors are deterministic or specified as distributions
- No aquifer support or complex drive mechanisms modeled
"""
