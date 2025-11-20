import streamlit as st

from scopehc.config import PALETTE, WORKFLOW_RIBBON


def main() -> None:
    st.session_state["current_page"] = "_pages_disabled/00_Overview.py"

    st.markdown(
        f"""
        <div style='text-align:center;margin-bottom:2em;'>
            <h1 style='color:{PALETTE["primary"]};font-size:3rem;font-weight:700;margin:0.5em 0 0.2em 0;letter-spacing:-0.02em;'>
                SCOPE-HC <span style='font-size:1.2rem;color:#999999;font-weight:400;'>(v0.7 beta)</span>
            </h1>
            <h5 style='color:{PALETTE["text_secondary"]};font-style:italic;margin:0.5em 0 1em 0;font-weight:400;'>
                Subsurface Capacity Overview and Probability Estimator for Hydrocarbons
            </h5>
            <p style='color:{PALETTE["text_secondary"]};font-size:0.9rem;margin:0;'>
                <em>by Lars Hjelm</em>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(WORKFLOW_RIBBON, unsafe_allow_html=True)

    with st.expander("How to Use SCOPE-HC (Quick Guide)", expanded=False):
        st.markdown(
            """
            SCOPE-HC estimates **in-place** and **recoverable (surface)** hydrocarbon volumes
            via **Monte Carlo** simulation of standard volumetric equations. Work left-to-right
            through the major inputs — each step builds on the previous one.

            **Steps**

            1. **Select Fluid Case** – Choose the hydrocarbon type for your analysis: **Oil**, **Gas**, or **Oil+Gas**. This selection determines how the Gross Rock Volume (GRV) is split between oil and gas zones, and which fluid properties are required in subsequent steps. For Oil+Gas cases, you can define a Gas-Oil Contact (GOC) to separate the gas cap from the oil leg.

            2. **GRV (Gross Rock Volume)** – Choose one of four calculation methods:
               - **Direct**: Direct input of total GRV volume
               - **Area × Thickness × GCF**: Geometric calculation using area, thickness, and Geometric Correction Factor
               - **Depth-based: Top and Base res. + Contact(s)**: Integration from depth-area tables with top and base reservoir surfaces
               - **Depth-based: Top + Res. thickness + Contact(s)**: Integration using structural top and constant reservoir thickness
               The method you choose depends on the available data and the complexity of your reservoir geometry.

            3. **HC Fill & Contacts** – Define the hydrocarbon fill and contact depths:
               - **Spill Point**: The shallowest depth where hydrocarbons can accumulate (top of trap)
               - **Effective HC Depth**: The base of the hydrocarbon column (hydrocarbon-water contact)
               - **Gas-Oil Contact (GOC)** (optional, for Oil+Gas cases): Depth separating the gas cap from the oil leg
               For depth-based GRV methods, these contacts are used to split the total GRV into gas and oil zones.

            4. **NtG, Porosity & Saturation** – Enter reservoir rock properties:
               - **Net-to-Gross (NtG)**: Fraction of the gross interval that is productive reservoir (0–1)
               - **Porosity (φ)**: Effective porosity used for pore volume calculation (0–1)
               - **Saturation**: Choose from three input modes:
                 - **Global**: Single hydrocarbon saturation (S_hc) or water saturation (S_w,global) applied to the entire reservoir
                 - **Water saturation Per zone**: Separate water saturations for oil zone (S_w,oilzone) and gas zone (S_w,gaszone)
                 - **Per phase, HC**: Direct hydrocarbon saturations for oil (S_oil) and gas (S_gas)
               These properties determine the effective pore volume available for hydrocarbons.

            5. **Fluids & Recovery** – Enter PVT (Pressure-Volume-Temperature) properties and recovery factors:
               - **Bg** (rb/scf): Gas formation volume factor (gas expansion reservoir → surface)
               - **1/Bo** (STB/rb): Inverse oil formation volume factor (oil shrinkage reservoir → surface)
               - **GOR** (scf/STB): Gas-Oil Ratio (associated gas per barrel of oil)
               - **CGR** (STB/MMscf): Condensate-Gas Ratio (condensate per million standard cubic feet of gas)
               - **Recovery Factors (RFs)**: Fractions applied to in-situ volumes for oil, gas, and condensate (0–1)
               Use the Fluid Property Estimator tool (Standing/Vasquez–Beggs) for guidance on typical values.

            6. **Dependencies (Optional)** – Define correlations between input parameters using the dependency matrix. This allows you to model relationships such as porosity increasing with depth, or recovery factor correlating with net-to-gross. The correlation matrix uses Higham's nearest correlation matrix projection to ensure valid correlations, and applies rank correlation through Cholesky decomposition and inverse-CDF mapping.

            7. **Results & THR** – Review the simulation outputs:
               - **In-place volumes**: Hydrocarbon volumes at reservoir conditions (oil, gas, condensate)
               - **Recoverable volumes**: Surface volumes after applying recovery factors (MMSTB, Bscf, etc.)
               - **Total Hydrocarbon Resource (THR)**: Combined resource in BOE (Barrel of Oil Equivalent)
               - **Distributions**: Histograms and cumulative distribution functions (CDFs) showing uncertainty
               - **Statistics**: P10, P50, P90 percentiles (based on your selected percentile convention)

            8. **Check Sensitivity** – Analyze parameter sensitivity using tornado plots. These plots show how variations in each input parameter affect the total recoverable volume, helping you identify which parameters have the greatest impact on your resource estimates. Parameters are ranked by their impact on the base case recoverable volume.

            **Key equations**

            - **GRV**: $\\text{GRV} = A \\times h \\times \\text{GCF}$ (or $\\int A(z)\\,\\mathrm{d}z$ for depth-based methods)  
            - **Bulk Volume**: $BV = \\text{GRV} \\times \\text{NtG}$ 
            - **Pore Volume**: $PV = BV \\times \\phi$ 
            - **Oil in place**: $N = PV_{oil}/B_o$  **Gas in place**: $G = PV_{gas}/B_g$ 
            - **Recoverable**: Oil = $N \\times RF_{oil}$; Free gas = $G \\times RF_{gas}$ 
            - **Associated gas**: $\\text{Gas}_{assoc} = \\text{Oil}_{STB} \\times GOR$ 
            - **Condensate**: $\\text{Cond}_{STB} = \\text{Gas}_{free} \\times CGR/10^6 \\times RF_{cond}$ 
            - **THR (BOE)**: $THR = Oil + Cond + \\frac{\\text{Gas}_{total}}{\\text{scf/BOE}}$
            """
        )

    # Workflow Diagram (moved out of expander)
    st.markdown("### Estimation Workflow")
    try:
        import graphviz as gv


        colors = [
            "#CAEDFE",  # Box 1 - lightest gray
            "#EDEDE8",  # Box 2
            "#EAEADF",  # Box 3
            "#E7E7D7",  # Box 4
            "#E4E4CF",  # Box 5
            "#F1F197",  # Box 6
            "#DEFEBF",  # Box 7
            "#C6E0B4",  # Box 8 - light green
        ]

        dot = gv.Digraph(graph_attr={"rankdir": "LR", "splines": "spline"})
        dot.node("FLUID_TYPE", "Select Fluid Case\n(Oil, Gas, or Oil+Gas)", 
                 shape="box", style="rounded,filled", fillcolor=colors[0])
        dot.node("GRV", "Gross Rock Volume (GRV)\n(choose calculation method)", 
                 shape="box", style="rounded,filled", fillcolor=colors[1])
        dot.node("FILL", "HC Fill & Contacts\n(Spill, HC, optional GOC)",
                 shape="box", style="rounded,filled", fillcolor=colors[2])
        dot.node("ROCK", "NtG, Porosity & Saturation\n(reservoir properties)",
                 shape="box", style="rounded,filled", fillcolor=colors[3])
        dot.node("FLUID", "Fluids & Recovery\n(Bg, 1/Bo, GOR, CGR, RFs)",
                 shape="box", style="rounded,filled", fillcolor=colors[4])
        dot.node("DEP", "Dependencies (opt.)\n(correlations)",
                 shape="box", style="rounded,filled", fillcolor=colors[5])
        dot.node("RES", "Results & THR\n(in-place, recoverable, BOE)",
                 shape="box", style="rounded,filled", fillcolor=colors[6])
        dot.node("SENS", "Check Sensitivity\n(tornado plots)",
                 shape="box", style="rounded,filled", fillcolor=colors[7])
        dot.edges(
            [
                ("FLUID_TYPE", "GRV"),
                ("GRV", "FILL"),
                ("FILL", "ROCK"),
                ("ROCK", "FLUID"),
                ("FLUID", "DEP"),
                ("DEP", "RES"),
                ("RES", "SENS"),
            ]
        )
        st.graphviz_chart(dot, use_container_width=True)
    except ImportError:
        st.info("💡 Install graphviz to view the workflow diagram: `pip install graphviz`")

    with st.expander("Assumptions and formulas", expanded=False):
        st.markdown(
            """
            ### **GRV (Gross Rock Volume) Methods**

            **1. Direct GRV**: Direct input of total GRV volume  
            • GRV = User input (m³)

            **2. Area × Thickness × GCF**: Geometric calculation  
            • GRV = A × GCF × h (with A converted from km² to m²)  
            • A = Area in km²; h = Thickness in meters; GCF = Geometric Correction Factor [0,1]

            **3. Depth-based: Top and Base res. + Contact(s)**: Integration from depth tables  
            • GRV = ∫ A(d) dd from spill point to effective HC contact depth  
            • Uses trapezoidal integration over area-depth relationship  
            • Supports Gas-Oil Contact (GOC) for gas cap and oil leg separation

            **4. Depth-based: Top + Res. thickness + Contact(s)**: Integration with constant thickness  
            • GRV = ∫ A(d) dd from structural top to effective HC contact depth  
            • Uses trapezoidal integration over area-depth relationship  
            • Supports Gas-Oil Contact (GOC) for gas cap and oil leg separation

            ### **Key Parameters**

            **GCF (Geometric Correction Factor)**: Accounts for reservoir geometry/closure (0–1).  
            **NtG (Net-to-Gross)**: Fraction of gross interval that is productive (0–1).  
            **Porosity (φ)**: Effective porosity used for pore volume calculation (0–1).  
            **Recovery Factors**: Fractions applied to in-situ volumes (oil, gas, condensate).

            ### **Volume Calculations**

            **Pore Volume (PV)**  
            • PV = GRV × NtG × φ

            **In-situ Volumes (reservoir conditions)**  
            • PVₒᵢₗ = PV × fₒᵢₗ  
            • PV_gas = PV × (1 − fₒᵢₗ)  
            • Depth-based splits use separate GRVₒᵢₗ and GRV_gas arrays.

            **Recoverable Volumes (surface conditions)**  
            • Oil_STB = PVₒᵢₗ × 6.2898 × RFₒᵢₗ × (1/Bo)  
            • Gas_scf = PV_gas × 6.2898 × RF_gas / Bg  
            • Associated gas = Oil_STB × GOR  
            • Condensate = Gas_free × CGR / 10⁶ × RF_cond

            ### **Formation Volume Factors (FVF)**

            • **Bo** (rb/STB) – oil volume change reservoir → surface  
            • **Bg** (rb/scf) – gas expansion reservoir → surface

            ### **Total Hydrocarbon Resource (THR)**

            **BOE Calculation**  
            • THR = Oil + Condensate + Gas_total / (scf/BOE)  
            • Default factor = 6,000 scf/BOE (adjust in sidebar)

            ### **Reporting Units**

            Inputs: Area (km²), Thickness (m), Depths (m), Bg (rb/scf), 1/Bo (STB/rb), GOR (scf/STB), CGR (STB/MMscf)  
            Outputs: Oil (MMSTB or Mm³), Gas (Bscf or Bm³), Condensate (MMSTB or Mm³), THR (MBOE or Mm³ BOE)
            """
        )

    # Add disclaimer
    from scopehc.ui.common import render_disclaimer
    render_disclaimer()


if __name__ == "__main__":
    main()


