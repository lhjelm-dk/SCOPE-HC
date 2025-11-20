# SCOPE-HC

## Overview

SCOPE-HC (SCalable Oilfield Probabilistic Estimation - Hydrocarbons) is a Monte Carlo-based probabilistic hydrocarbon resource estimation tool. It estimates Gross Rock Volume (GRV), pore volume, in-situ fluids, and recoverable resources for oil and gas prospects.

**Current Version:** v0.7 beta

## Current Structure

### Main Application
- **`streamlit_app.py`** - Entry point for the multi-page Streamlit application with custom page routing
- **`_pages_disabled/`** - Page modules loaded dynamically by the custom router:
  - `00_Overview.py` - Application overview, workflow, and assumptions
  - `01_Inputs.py` - Combined GRV, Reservoir, Fluids, and Recovery inputs
  - `01_Inputs_Gross_Rock_Volume_GRV.py` - GRV calculation inputs
  - `01_Inputs_Net_to_Gross_NtG.py` - Net-to-Gross ratio inputs
  - `01_Inputs_Porosity.py` - Porosity inputs
  - `01_Inputs_Saturation.py` - Hydrocarbon saturation inputs
  - `01_Inputs_Fluids.py` - Fluid properties and PVT inputs
  - `01_Inputs_Recovery.py` - Recovery factor inputs
  - `02_Dependencies.py` - Parameter Dependencies/Correlations configuration
  - `03_Run_Simulation.py` - Simulation execution and input QC
  - `04_Results.py` - Results visualization and analysis
  - `05_Sensitivity.py` - Sensitivity analysis and tornado plots
  - `06_Downloads.py` - Data export (CSV/Excel)
  - `07_Help_Math.py` - Mathematical appendix and documentation

### Modular Package (`scopehc/`)
The application has been fully modularized into a clean package structure:

- **`scopehc/config.py`** - Constants, units, colors, help text, defaults
- **`scopehc/utils.py`** - Utility functions (clip01, safe_div, sanitize, validation)
- **`scopehc/pvt.py`** - PVT calculations (Bg, Bo, Rs, Z-factor)
- **`scopehc/geom.py`** - GCF interpolation and GRV calculations
- **`scopehc/geom_depth.py`** - Depth-based GRV calculations from area-depth tables
- **`scopehc/sampling.py`** - Monte Carlo samplers and correlation handling
- **`scopehc/compute.py`** - Core volume calculations (single source of truth)
- **`scopehc/plots.py`** - Plotting utilities and theme
- **`scopehc/export.py`** - CSV/Excel export functionality

### UI Components (`scopehc/ui/`)
User interface modules organized by functionality:

- **`scopehc/ui/common.py`** - Shared UI components, navigation, theme, parameter rendering
- **`scopehc/ui/inputs_grv.py`** - GRV input interface
- **`scopehc/ui/inputs_reservoir.py`** - Reservoir property inputs (NtG, porosity, saturation)
- **`scopehc/ui/inputs_fluids.py`** - Fluid property inputs and PVT estimator
- **`scopehc/ui/inputs_dependencies.py`** - Parameter correlation/dependency configuration
- **`scopehc/ui/input_qc.py`** - Input quality control and validation panel
- **`scopehc/ui/run.py`** - Simulation execution interface
- **`scopehc/ui/results.py`** - Results visualization
- **`scopehc/ui/sensitivity.py`** - Sensitivity analysis interface
- **`scopehc/ui/downloads.py`** - Data export interface
- **`scopehc/ui/help_math.py`** - Mathematical documentation renderer
- **`scopehc/ui/helpers.py`** - Helper functions for UI components

## Key Features

### GRV Calculation Methods
1. **Direct GRV** - User provides GRV distribution directly
2. **Area × Thickness × GCF** - Geometric calculation with Geometric Correction Factor
3. **Depth-based: Top and Base res. + Contact(s)** - Area-depth table integration
4. **Depth-based: Top + Res. thickness + Contact(s)** - Top depth + thickness calculation

#### Fluid Handling

- **Oil**: GRV integrated from Top → HCWC (oil only).
- **Gas**: GRV integrated from Top → HCWC (gas only).
- **Oil + Gas**:
  - For Direct and Area–Thickness–GCF methods, total GRV is split by a user-defined oil fraction $f_\mathrm{oil}$ (default 0.60).
  - For Depth–Area, the oil/gas split is defined by a **Gas–Oil Contact (GOC)** which can be set by:
    1) direct depth, 2) oil fraction of the total HC column, or 3) oil column height above HCWC.
  - Gas column: Top → GOC; Oil column: GOC → HCWC.

### Fluids & PVT
- Gas FVF (Bg) or 1/Bg with automatic conversion
- Oil FVF (Bo) via inverse Bo
- Gas-Oil Ratio (GOR)
- Condensate Gas Ratio (CGR)
- P–T state helper for pressure/temperature calculations

#### Fluid Property Estimator

Under the *Fluid inputs* section, a collapsible helper box provides
approximate estimates of oil and gas PVT properties using the
**Standing (1947)** and **Vasquez–Beggs (1980)** correlations.

It calculates:
- Solution GOR (Rs) in scf/STB
- Oil formation volume factor (Bo) in reservoir bbl/STB
- Gas formation volume factor (Bg) in ft³/scf
- Oil density (ρo) in kg/m³

Default parameters correspond to a typical North Sea oil at ~2200 m depth
(38 °API, 90 °C, 220 bar). The tool is intended for educational use and
quick sensitivity checks only. All calculations assume ideal gas behavior
(Z=1.0) and may not be accurate for all reservoir conditions.

### Recovery Factors
- Oil recovery factor (RF_oil)
- Gas recovery factor (RF_gas)
- Associated gas recovery factor (RF_assoc_gas)
- Condensate recovery factor (RF_condensate)

### Visualization
- Area-depth and volume-depth plots
- Histograms and CDF plots (consistent colors per parameter)
- Tornado plot for sensitivity analysis
- THR breakdown visualization

### Export
- CSV export
- Excel export (requires `openpyxl`)

## Installation

```bash
cd SCOPE-HC
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run streamlit_app.py
```

The application uses a custom page routing system that dynamically loads pages from the `_pages_disabled/` folder. Navigation is handled through a custom sidebar interface rather than Streamlit's default multi-page navigation.

## Modular Architecture ✅

The application has been successfully modularized into a clean package structure. The `scopehc/` package contains:

1. **Config module** - Constants, theme, help text, colors ✅
2. **Core modules** - Validation, GOC, GRV, results, PVT, GCF ✅
3. **Sampling module** - Monte Carlo samplers and dependencies ✅
4. **Plotting module** - Visualization functions ✅
5. **Export module** - Data export functionality ✅

This modular structure provides:
- **Single source of truth** for all calculations
- **Clear separation of concerns**
- **Easier testing** and maintenance
- **Better code organization**
- **Reusable components**

## Dependencies

- **streamlit** (>=1.38) - Web application framework
- **numpy** (>=1.20.0) - Numerical computations
- **pandas** (>=1.3.0) - Data manipulation and analysis
- **plotly** (>=5.0.0) - Interactive visualizations
- **scipy** (>=1.7.0) - Scientific computing (statistical distributions, optimization)
- **openpyxl** (>=3.0.0) - Excel file export support
- **matplotlib** (>=3.5.0) - Plotting library (used for style configuration)

## Units

The application supports two unit systems:
- **Oilfield** - bbl, scf, STB, psia, °F
- **SI** - m³, m³, m³, bar, °C

## References

- Otis, R. M., & Schneidermann, N. (1997). “A comprehensive approach to prospect evaluation…” *AAPG Bulletin*.
- Rose, P. R. (2001). *Risk Analysis and Management of Petroleum Exploration Ventures*. AAPG Methods in Exploration No. 12.
- Haldorsen, J. B. U., & Damsleth, E. (1990). “Stochastic modeling.” *Journal of Petroleum Technology*.
- Vose, D. (2008). *Risk Analysis: A Quantitative Guide*. Wiley.
- Standing, M. B. (1947). Pressure–volume–temperature correlations for oil and gas; Beggs, H. D., & Robinson, J. R. (1975/1977) viscosity correlations.
- U.S. Geological Survey. “Conversion factors for oil and gas resources” (6,000 scf/BOE convention).
- Vendor positioning: Ariane LogiX, SLB GeoX, GoExplore Prospector (company literature).

## Glossary & Quick Reference

### Quick Start

1) Choose GRV method. 2) Set Spill & HC depths (and GOC for depth-based). 3) Enter Fluids (Bg, 1/Bo, GOR, CGR). 4) Set RFs. 5) Run and review P10/P50/P90.

### GRV Methods

Pick how GRV is computed. Depth-based methods support Gas–Oil Contact (GOC). Direct & Area×Thickness×GCF use an explicit oil/gas split.

- **Spill point depth (m)**: The star (★) on depth plots marks the mean spill depth.
- **Effective hydrocarbon contact (m)**: Base of the hydrocarbon column.
- **Gas–Oil Contact (GOC) depth (m)**: Or derived via oil fraction or oil column height. Gas above, oil below.

### Fluids & PVT

- **Gas FVF Bg (rb/scf)**: If using 1/Bg, units must be scf/rcf. Conversion: Bg = 1 / ((1/Bg)×5.614583).
- **Inverse Bo (STB/rb)**: Converts reservoir barrels to stock-tank barrels.
- **Solution GOR (scf/STB)**: Associated gas per STB of oil.
- **CGR (STB/MMscf)**: Condensate per million scf of produced free gas.

### Recovery & Simulation

- **Oil RF (0–1)**: Fraction of oil in-place recovered to surface.
- **Gas RF (0–1)**: Fraction of free gas in-place recovered to surface.
- **Monte Carlo trials**: Higher → smoother distributions, longer runtime.
- **Random seed**: For reproducibility.

### Charts

Histograms show frequency; dashed lines mark P10/P50/P90. CDF line uses the same color.

### Units Quick Reference

- **Bg**: rb/scf
- **1/Bg**: scf/rcf
- **1/Bo**: STB/rb
- **GOR**: scf/STB
- **CGR**: STB/MMscf
- **GRV/PV**: m³
- **BOE factor**: scf/BOE

## Math Appendix

### Volumetric Relationships

- **Gross Rock Volume (GRV)**  
  $$
  \text{GRV} = \int_{\text{closure}} A_\text{structure}(z)\, \mathrm{d}z
  $$
  For depth tables the application computes trapezoidal integrals of the top/base areas; for direct GRV inputs the integral collapses to the provided distribution.

#### GRV from depth–area curves (v3-compatible)

When a depth–area table is provided, GRV is computed by integrating the area curve between the fluid contacts using a trapezoidal rule:

- **Gas zone**: Top Structure → GOC  
- **Oil zone**: GOC → OWC  
- **Below OWC** is water only and excluded.  
- If the GOC is missing, the hydrocarbon column is taken from Top → OWC (oil only).  
- If both contacts are missing, the entire Top → deepest available depth is used.

Contacts are clipped to the available depth range; reversed inputs are handled gracefully. The integration uses piecewise linear interpolation of the area curve with interpolated endpoints at the contact depths.

- **Pore Volume (PV)**  
  $$
  PV = \text{GRV} \times \text{NtG} \times \phi
  $$
  with Effective Porosity $\phi$ and Net-to-Gross NtG sampled independently (or jointly if correlations are enabled).

- **Oil / Gas In-Place**  
  $$
  N = \frac{PV_\text{oil}}{B_o}, \qquad G = \frac{PV_\text{gas}}{B_g}
  $$
  The tool stores inverse $1/B_o$ and converts $B_g$ or $1/B_g$ to consistent units before computation.

- **Recoverable Volumes**  
  $$
  N_\text{rec} = N \times \text{RF}_\text{oil}, \qquad
  G_\text{rec} = G \times \text{RF}_\text{gas}
  $$
  Associated gas is computed as $N_\text{rec} \times \text{GOR}$ with optional associated-gas recovery factor.

- **Total Hydrocarbon Resource (THR)**  
  $$
  \text{THR}_\text{BOE} = N_\text{rec} + \frac{G_\text{rec} + G_\text{assoc}}{\text{scf/BOE}} + N_\text{cond}
  $$
  where condensate $N_\text{cond}$ is derived from condensate yield and condensate RF.

### Stochastic Workflow

1. **Parameter Sampling** – each input distribution (PERT, Triangular, Uniform, Lognormal, Subjective Beta, etc.) is sampled $n$ times using NumPy random generators seeded via the sidebar control.
2. **Dependency Handling** – optional rank correlations are applied through Higham's nearest correlation matrix projection followed by Cholesky decomposition and inverse-CDF mapping.
3. **Deterministic Guardrails** – values are clipped (e.g., RFs to $[0,1]$, $B_g > 0$) ensuring numerical stability.
4. **Compute Engine** – `scopehc.compute.compute_results()` evaluates all volumetric equations vectorised for performance.
5. **Post Processing** – summary statistics (mean, P10/P50/P90, etc.) are computed and cached, along with the full per-trial DataFrame for export.

### Correlation & Dependence

- **Higham Projection** – stabilises user-specified correlation matrices by projecting to the nearest positive semi-definite matrix with unit diagonal.
- **Cholesky Generation** – correlated standard normals are generated, converted to uniform quantiles via the Gaussian CDF, and then mapped to parameter-specific inverse CDFs.

### Units & Conversions

- Oilfield base units: m, km², m³, STB, scf. SI conversions use the factors declared in `scopehc/ui/common.py::UNIT_CONVERSIONS`.
- Gas-to-BOE factor defaults to 6,000 scf/BOE but is fully user-configurable; this factor is propagated into all exports and metadata.
- Bg conversions: when $1/B_g$ is entered, the system applies $B_g = \frac{1}{(1/B_g) \times 5.614583}$ to maintain reservoir-barrel vs. scf consistency.

### Simulation Metadata

- **Trials** – default 10,000 draws; configurable up to 2,000,000.
- **Seed** – stored in `st.session_state["random_seed"]` for reproducibility and passed to exports.
- **Cached Artefacts** – `trial_data`, `results_cache`, and `df_results` are persisted to share between pages (Run → Results → Downloads).

### Notes & Assumptions

- Structural closure is assumed to be fully trapped; aquifer support, drive mechanisms, and time-dependent recovery are not modelled.
- Recovery factors are user supplied; no automatic range validation beyond clipping to $[0,1]$.
- Condensate recovery relies on gas RF; if missing, defaults are imposed during fallback sampling.
- All plots and exports use nan-safe arrays (`np.nan_to_num`) to avoid downstream issues when distributions produce edge cases.

### Saturation

**Definitions.**  
Water saturation: $S_w$. Hydrocarbon saturation: $S_{\mathrm{hc}} = 1 - S_w$.

We model three input modes:

- **Global:** one $S_{\mathrm{hc}}$ applied to all hydrocarbon zones.
- **Per-zone water:** $S_{w,\mathrm{oil\,zone}}$ and $S_{w,\mathrm{gas\,zone}}$ with
  $S_{\mathrm{oil}}=1-S_{w,\mathrm{oil\,zone}}$, $S_{\mathrm{gas}}=1-S_{w,\mathrm{gas\,zone}}$.
- **Per-phase:** direct $S_{\mathrm{oil}}$ and $S_{\mathrm{gas}}$.

**Volumetrics with saturation.**  
For split GRV:

$$
PV_{\mathrm{oil}} = GRV_{\mathrm{oil}}\cdot N_tG \cdot \phi,\qquad
PV_{\mathrm{gas}} = GRV_{\mathrm{gas}}\cdot N_tG \cdot \phi
$$

$$
PV_{\mathrm{oil}}^{hc} = PV_{\mathrm{oil}} \cdot S_{\mathrm{oil}},\qquad
PV_{\mathrm{gas}}^{hc} = PV_{\mathrm{gas}} \cdot S_{\mathrm{gas}}
$$

In-place and recoverable volumes follow as in the base model.  
**Condensate** is not given an in-situ saturation here; it is treated as a surface product of produced gas using CGR.

**Typical ranges.**  
Clean oil zones: $S_{\mathrm{oil}}\sim 0.6{-}0.8$. Gas zones often higher hydrocarbon saturation (lower $S_w$). Near contacts, $S_w$ rises.

### PVT Correlations (Standing / Vasquez–Beggs)

The Fluid Property Estimator uses industry-standard correlations to estimate PVT properties:

**Standing (1947) Correlation:**

Solution Gas-Oil Ratio:
$$
R_s = \gamma_g \left[ \frac{p}{18.2 + 1.4 \times 10^{0.0125 \times \text{API}}} \right]^{1.2048}
$$

Oil Formation Volume Factor:
$$
B_o = 0.9759 + 0.00012 \left( R_s \times \left( \frac{\gamma_g}{\gamma_o} \right)^{0.5} + 1.25 T \right)^{1.2}
$$

**Vasquez–Beggs (1980) Correlation:**

Solution Gas-Oil Ratio:
$$
R_s = \gamma_g \left[ \frac{p}{a + b \times 10^{c \times \text{API}}} \right]^{1.2048}
$$

Where for API ≤ 30: $a = 18.2$, $b = 1.4$, $c = 0.0125$  
For API > 30: $a = 13.8$, $b = 1.4$, $c = 0.00091$

Oil Formation Volume Factor:
$$
B_o = 1.0 + C_1 R_s + C_2 (T-60) \frac{\text{API}}{\gamma_g} + C_3 R_s (T-60) \frac{\text{API}}{\gamma_g}
$$

Where for API ≤ 30: $C_1 = 4.677 \times 10^{-4}$, $C_2 = 1.751 \times 10^{-5}$, $C_3 = -1.811 \times 10^{-8}$  
For API > 30: $C_1 = 4.670 \times 10^{-4}$, $C_2 = 1.100 \times 10^{-5}$, $C_3 = 1.337 \times 10^{-9}$

**Common Formulas:**

Oil Density (Reservoir Conditions):
$$
\rho_o = \frac{350 \times \gamma_o}{5.615 \times B_o} \quad \text{(lb/ft³)} = \frac{350 \times \gamma_o}{5.615 \times B_o} \times 16.0185 \quad \text{(kg/m³)}
$$

Gas Formation Volume Factor (Ideal Gas):
$$
B_g = 0.02827 \times \frac{(T + 459.67) \times Z}{p} \quad \text{(ft³/scf)}
$$

Where $Z = 1.0$ (ideal gas assumption), $T$ is in °F, and $p$ is in psia.

Oil Specific Gravity:
$$
\gamma_o = \frac{141.5}{131.5 + \text{API}}
$$

**Notation:**
- $R_s$ = Solution gas-oil ratio (scf/STB)
- $B_o$ = Oil formation volume factor (rbbl/STB)
- $B_g$ = Gas formation volume factor (ft³/scf)
- $\rho_o$ = Oil density (lb/ft³ or kg/m³)
- $\gamma_g$ = Gas specific gravity (air = 1.0)
- $\gamma_o$ = Oil specific gravity (water = 1.0)
- $p$ = Pressure (psia)
- $T$ = Temperature (°F)
- API = Oil gravity (°API)

### Percentile Convention

By default, SCOPE-HC reports percentiles using the **probability-of-exceedance** convention common in exploration resource assessments:

- **P10** = optimistic case (10% chance of exceeding this value)
- **P50** = median case
- **P90** = conservative case (90% chance of exceeding)

You can switch to the classical **non-exceedance** definition in the sidebar (Percentile Convention section). All plots and summary tables update automatically when you change the setting.

### Probability Distributions

SCOPE-HC supports multiple probability distributions for modeling parameter uncertainty. Each distribution has specific characteristics that make it suitable for different types of geological and engineering parameters.

#### Constant Distribution

**Description:** A degenerate distribution where the parameter takes a single fixed value with probability 1.

**Formula:**
$$
P(X = c) = 1
$$

**Application:** Used when a parameter is known with certainty or when performing deterministic sensitivity analysis. Common for well-constrained parameters like unit conversion factors or fixed design parameters.

#### PERT Distribution (Program Evaluation and Review Technique)

**Description:** A continuous distribution defined on a bounded interval, similar to a Beta distribution but parameterized by minimum, mode, and maximum values. It is unimodal and can be symmetric or skewed.

**Formula:**
The PERT distribution is a special case of the Beta distribution with shape parameters:
$$
\alpha = 1 + \lambda \frac{m - a}{c - a}, \quad \beta = 1 + \lambda \frac{c - m}{c - a}
$$
where $a$ is the minimum, $m$ is the mode, $c$ is the maximum, and $\lambda$ is a shape parameter (typically 4).

**Probability Density Function:**
$$
f(x) = \frac{(x-a)^{\alpha-1}(c-x)^{\beta-1}}{B(\alpha,\beta)(c-a)^{\alpha+\beta-1}}, \quad a \leq x \leq c
$$
where $B(\alpha,\beta)$ is the Beta function.

**Application in Petroleum Geoscience:** Widely used in project risk analysis and resource estimation. Ideal for expert-elicited parameters where three-point estimates (low, most likely, high) are available. Commonly applied to:
- Reservoir thickness and area estimates
- Recovery factors based on analog fields
- Net-to-gross ratios from well logs
- Porosity estimates from core analysis

The PERT distribution provides a smooth, bounded distribution that emphasizes the mode while allowing for uncertainty in the tails.

#### Triangular Distribution

**Description:** A continuous distribution defined by three parameters: minimum ($a$), mode ($m$), and maximum ($b$). It has a linear probability density function that increases from the minimum to the mode, then decreases to the maximum.

**Formula:**
**Probability Density Function:**
$$
f(x) = \begin{cases}
\frac{2(x-a)}{(b-a)(m-a)} & \text{if } a \leq x \leq m \\
\frac{2(b-x)}{(b-a)(b-m)} & \text{if } m \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

**Mean and Variance:**
$$
\mu = \frac{a + m + b}{3}, \quad \sigma^2 = \frac{a^2 + m^2 + b^2 - ab - am - bm}{18}
$$

**Application in Petroleum Geoscience:** Simple and intuitive for three-point estimates when data is limited. Used for:
- Initial parameter estimates in early exploration stages
- Quick sensitivity analyses
- Parameters where the linear shape is considered appropriate

Less smooth than PERT but computationally simpler and easier to explain to stakeholders.

#### Uniform Distribution

**Description:** A distribution where all values within a specified range are equally likely. It represents maximum uncertainty when only bounds are known.

**Formula:**
**Probability Density Function:**
$$
f(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$

**Mean and Variance:**
$$
\mu = \frac{a + b}{2}, \quad \sigma^2 = \frac{(b-a)^2}{12}
$$

**Application in Petroleum Geoscience:** Used when there is minimal information about a parameter beyond its bounds. Applied to:
- Initial exploration assessments with limited data
- Parameters where no central tendency is known
- Conservative uncertainty modeling (maximum entropy principle)

Represents a state of maximum uncertainty and is often replaced with more informative distributions as data becomes available.

#### Lognormal Distribution

**Description:** A continuous distribution where the natural logarithm of the variable is normally distributed. It is positively skewed and cannot take negative values.

**Formula:**
If $Y = \ln(X)$ is normally distributed with mean $\mu$ and variance $\sigma^2$, then $X$ is lognormally distributed.

**Probability Density Function:**
$$
f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0
$$

**Mean and Variance:**
$$
E[X] = \exp\left(\mu + \frac{\sigma^2}{2}\right), \quad \text{Var}(X) = \exp(2\mu + \sigma^2)(\exp(\sigma^2) - 1)
$$

**Application in Petroleum Geoscience:** One of the most important distributions in resource assessment. Commonly used for:
- **Field and accumulation sizes:** Empirical studies show that discovered field sizes often follow lognormal distributions, with many small fields and few large ones
- **Permeability:** Results from multiplicative processes (porosity, pore connectivity, grain size)
- **Production rates:** Influenced by multiple multiplicative factors
- **Reserve estimates:** Often lognormally distributed due to the multiplicative nature of volumetric calculations

The lognormal distribution captures the heavy-tailed nature of resource distributions, where a small number of large discoveries contain most of the resources.

#### Beta Distribution (Subjective Beta / Vose Method)

**Description:** A flexible continuous distribution defined on the interval [0, 1], characterized by two shape parameters $\alpha$ and $\beta$. The Subjective Beta (Vose method) allows fitting from quantile estimates (P10, P50, P90).

**Formula:**
**Probability Density Function:**
$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad 0 \leq x \leq 1
$$
where $B(\alpha,\beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

**Mean and Variance:**
$$
\mu = \frac{\alpha}{\alpha + \beta}, \quad \sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

**Application in Petroleum Geoscience:** Ideal for parameters naturally bounded between 0 and 1:
- **Net-to-Gross (NtG) ratio:** Fraction of gross interval that is net reservoir
- **Porosity:** Pore volume fraction (typically 0.05–0.35 for sandstones)
- **Hydrocarbon saturation:** Fraction of pore space occupied by hydrocarbons
- **Recovery factor:** Fraction of in-place resources that can be recovered

The Beta distribution's flexibility allows modeling of symmetric, skewed, U-shaped, or J-shaped distributions, making it versatile for various reservoir properties.

#### Stretched Beta Distribution

**Description:** A Beta distribution scaled to an arbitrary interval $[a, b]$ rather than [0, 1]. It maintains the Beta distribution's flexibility while allowing bounds other than 0 and 1.

**Formula:**
If $Y \sim \text{Beta}(\alpha, \beta)$ on [0, 1], then $X = a + (b-a)Y$ is stretched Beta on $[a, b]$.

**Application in Petroleum Geoscience:** Used for parameters with known bounds that are not [0, 1]:
- Reservoir depths with known structural limits
- Temperature ranges in specific geological settings
- Pressure ranges constrained by overburden and hydrostatic gradients

#### Truncated Normal Distribution

**Description:** A normal distribution constrained to a specific interval $[a, b]$. Values outside this range are excluded, and the distribution is renormalized.

**Formula:**
**Probability Density Function:**
$$
f(x) = \begin{cases}
\frac{\phi\left(\frac{x-\mu}{\sigma}\right)}{\sigma\left[\Phi\left(\frac{b-\mu}{\sigma}\right) - \Phi\left(\frac{a-\mu}{\sigma}\right)\right]} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}
$$
where $\phi$ is the standard normal PDF and $\Phi$ is the standard normal CDF.

**Application in Petroleum Geoscience:** Useful when a parameter is approximately normally distributed but physically constrained:
- **Porosity:** Cannot exceed ~0.4 for sandstones or be negative
- **Saturation:** Must lie between 0 and 1
- **Permeability:** Often lognormally distributed, but truncation can model known bounds
- **Reservoir thickness:** Constrained by structural and stratigraphic limits

Provides a realistic model that respects physical constraints while maintaining the normal distribution's mathematical properties.

#### Truncated Lognormal Distribution

**Description:** A lognormal distribution constrained to a specific interval. Useful when a parameter is lognormally distributed but has known physical bounds.

**Formula:**
Similar to truncated normal, but applied to $\ln(X)$ rather than $X$ directly.

**Application in Petroleum Geoscience:** Applied to lognormally distributed parameters with constraints:
- **Permeability:** Lognormally distributed but bounded by rock physics limits
- **Field sizes:** Lognormal with minimum economic threshold and maximum geological constraint
- **Production rates:** Lognormal distribution truncated by well capacity and reservoir deliverability

#### Burr Distribution (Type XII)

**Description:** A flexible three-parameter distribution capable of modeling a wide variety of shapes, including heavy tails. It is a generalization of the Pareto distribution.

**Formula:**
**Probability Density Function:**
$$
f(x) = \frac{ck}{d}\left(\frac{x}{d}\right)^{c-1}\left[1 + \left(\frac{x}{d}\right)^c\right]^{-k-1}, \quad x > 0
$$
where $c > 0$ is the first shape parameter, $k > 0$ is the second shape parameter, and $d > 0$ is the scale parameter.

**Application in Petroleum Geoscience:** Less commonly used but valuable for:
- **Heavy-tailed phenomena:** Field size distributions with very large outliers
- **Extreme value modeling:** Rare but high-impact events in exploration
- **Alternative to lognormal:** When lognormal doesn't fit well but heavy tails are present

The Burr distribution's flexibility makes it useful for empirical fitting when standard distributions are inadequate.

#### Johnson SU Distribution

**Description:** A four-parameter distribution that can approximate many other distributions through transformation. The "SU" stands for "unbounded" (System of distributions).

**Formula:**
If $Z$ is standard normal, then:
$$
X = \gamma + \delta \sinh^{-1}\left(\frac{Z - \xi}{\lambda}\right)
$$
where $\gamma$ (shape), $\delta$ (shape), $\lambda$ (scale), and $\xi$ (location) are parameters.

**Application in Petroleum Geoscience:** Used for:
- **Flexible empirical fitting:** When data doesn't fit standard distributions well
- **Complex parameter relationships:** Parameters influenced by multiple processes
- **Advanced uncertainty modeling:** When distribution shape is critical for risk assessment

The Johnson SU distribution's ability to model various shapes (symmetric, skewed, heavy-tailed) makes it powerful for sophisticated uncertainty quantification.

#### Distribution Selection Guidelines

**For bounded parameters (0–1):**
- **Beta or Subjective Beta:** When expert judgment provides quantiles
- **Truncated Normal:** When parameter is approximately normal but bounded
- **Uniform:** Maximum uncertainty with only bounds known

**For positive, unbounded parameters:**
- **Lognormal:** Most common for multiplicative processes (field sizes, permeability, production rates)
- **Truncated Lognormal:** When lognormal but with known constraints
- **Burr or Johnson SU:** When lognormal doesn't fit well

**For three-point estimates:**
- **PERT:** Preferred for smooth, expert-elicited estimates
- **Triangular:** Simpler alternative when linear shape is acceptable

**For maximum uncertainty:**
- **Uniform:** When only bounds are known

The choice of distribution significantly impacts resource estimates and risk assessments. Empirical validation against historical data and expert judgment should guide distribution selection.

## License

See LICENSE file for details.

## Author

Lars Hjelm
