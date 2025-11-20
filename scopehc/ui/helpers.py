"""
Helper UI components for SCOPE-HC.
"""
import streamlit as st
import numpy as np
import pandas as pd


def render_fluid_estimator() -> None:
    """
    Collapsible helper box for Standing / Vasquez–Beggs PVT property estimates.
    
    Provides approximate estimates of Rs, Bo, Bg, and ρo based on user input
    or default North Sea conditions (~2200 m depth).
    """
    with st.expander("Fluid Property Estimator (Standing / Vasquez–Beggs)"):
        st.caption(
            "Quick helper to estimate approximate PVT properties "
            "(Rs, Bo, Bg, ρo) for crude oils. "
            "Use for guidance only – values are approximate."
        )

        corr = st.selectbox(
            "Select correlation:",
            options=["Vasquez–Beggs (1980)", "Standing (1947)"],
            index=0,
            key="fluid_corr_choice"
        )

        st.markdown("#### Input parameters")

        api = st.number_input(
            "Oil gravity (°API)",
            value=38.0,
            min_value=10.0,
            max_value=60.0,
            step=0.5,
            key="est_api"
        )
        gas_grav = st.number_input(
            "Gas specific gravity (air=1.0)",
            value=0.75,
            min_value=0.5,
            max_value=1.5,
            step=0.01,
            key="est_gasgrav"
        )
        temp_C = st.number_input(
            "Reservoir temperature (°C)",
            value=90.0,
            min_value=0.0,
            max_value=200.0,
            step=1.0,
            key="est_tempC"
        )
        pres_bar = st.number_input(
            "Reservoir pressure (bar)",
            value=220.0,
            min_value=1.0,
            max_value=1000.0,
            step=5.0,
            key="est_presbar"
        )

        # --- Unit conversions
        temp_F = 1.8 * temp_C + 32.0
        pres_psia = pres_bar * 14.5038

        # --- Base property conversions
        gamma_o = 141.5 / (131.5 + api)
        gamma_g = gas_grav

        # --- Compute Rs and Bo using selected correlation
        if "Vasquez–Beggs" in corr:
            # Vasquez–Beggs (1980) correlation
            # Solution GOR (Rs)
            if api <= 30:
                a, b, c = 18.2, 1.4, 0.0125
                # Bo coefficients for API <= 30
                C1 = 4.677e-4
                C2 = 1.751e-5
                C3 = -1.811e-8
            else:
                a, b, c = 13.8, 1.4, 0.00091
                # Bo coefficients for API > 30
                C1 = 4.670e-4
                C2 = 1.100e-5
                C3 = 1.337e-9
            
            Rs = gamma_g * (pres_psia / (a + b * 10 ** (c * api))) ** 1.2048
            
            # Oil FVF (Bo) - Vasquez-Beggs (1980)
            # Bo = 1.0 + C1*Rs + C2*(T-60)*(API/gamma_g) + C3*Rs*(T-60)*(API/gamma_g)
            Bo = 1.0 + C1 * Rs + C2 * (temp_F - 60.0) * (api / gamma_g) + C3 * Rs * (temp_F - 60.0) * (api / gamma_g)
        else:
            # Standing (1947) correlation
            # Solution GOR (Rs)
            # Rs = γg [p / (18.2 + 1.4 × 10^(0.0125×API))]^1.2048
            Rs = gamma_g * (pres_psia / (18.2 + 1.4 * 10 ** (0.0125 * api))) ** 1.2048
            
            # Oil FVF (Bo) - Standing (1947)
            # Bo = 0.9759 + 0.00012 (Rs × (γg/γo)^0.5 + 1.25T)^1.2
            Bo = 0.9759 + 0.00012 * (Rs * (gamma_g / gamma_o) ** 0.5 + 1.25 * temp_F) ** 1.2

        # --- Gas formation volume factor (simplified)
        # Bg in reservoir ft³/scf
        # Formula Bg(ft³/scf) = 0.02827 * (T°F + 459.67) * Z / Ppsia
        Z = 1.0  # assume ideal gas
        Bg_ft3_scf = 0.02827 * (temp_F + 459.67) * Z / pres_psia
        
        # Convert to reservoir m³/scm (standard cubic meter)
        # 1 scf = 0.0283168 scm, 1 ft³ = 0.0283168 m³
        Bg_m3_scm = Bg_ft3_scf * (0.0283168 / 0.0283168)  # Same conversion factor
        # More accurately: Bg(m³/scm) = Bg(ft³/scf) * (ft³/scf to m³/scm)
        # Standard: 1 scf = 0.0283168 scm, so Bg_m3_scm = Bg_ft3_scf * 1.0 (same units)
        # Actually, we need: Bg(m³/scm) = Bg(ft³/scf) * (ft³/m³) / (scf/scm)
        # 1 ft³ = 0.0283168 m³, 1 scf = 0.0283168 scm
        # So Bg_m3_scm = Bg_ft3_scf (dimensionless ratio is the same)
        # But for display, we'll show ft³/scf as that's the standard unit

        # --- Oil density at reservoir conditions
        rho_o_lb_ft3 = (350.0 * gamma_o) / (5.615 * Bo)  # lb/ft³
        rho_o_kgm3 = rho_o_lb_ft3 * 16.0185  # convert to kg/m³

        st.markdown("#### Results (approximate)")
        
        # Display equations used (using HTML details/summary since we're already in an expander)
        st.markdown(
            """
            <details>
                        <summary style="cursor: pointer; font-weight: bold; margin: 10px 0;">Equations Used</summary>
            """,
            unsafe_allow_html=True
        )
        
        if "Vasquez–Beggs" in corr:
            st.markdown(
                r"""
                **Vasquez–Beggs (1980) Correlation:**
                
                **Solution Gas-Oil Ratio:**
                $$
                R_s = \gamma_g \left[ \frac{p}{a + b \times 10^{c \times \text{API}}} \right]^{1.2048}
                $$
                
                Where for API ≤ 30: $a = 18.2$, $b = 1.4$, $c = 0.0125$  
                For API > 30: $a = 13.8$, $b = 1.4$, $c = 0.00091$
                
                **Oil Formation Volume Factor:**
                $$
                B_o = 1.0 + C_1 R_s + C_2 (T-60) \frac{\text{API}}{\gamma_g} + C_3 R_s (T-60) \frac{\text{API}}{\gamma_g}
                $$
                
                Where for API ≤ 30: $C_1 = 4.677 \times 10^{-4}$, $C_2 = 1.751 \times 10^{-5}$, $C_3 = -1.811 \times 10^{-8}$  
                For API > 30: $C_1 = 4.670 \times 10^{-4}$, $C_2 = 1.100 \times 10^{-5}$, $C_3 = 1.337 \times 10^{-9}$
                """
            )
        else:
            st.markdown(
                r"""
                **Standing (1947) Correlation:**
                
                **Solution Gas-Oil Ratio:**
                $$
                R_s = \gamma_g \left[ \frac{p}{18.2 + 1.4 \times 10^{0.0125 \times \text{API}}} \right]^{1.2048}
                $$
                
                **Oil Formation Volume Factor:**
                $$
                B_o = 0.9759 + 0.00012 \left( R_s \times \left( \frac{\gamma_g}{\gamma_o} \right)^{0.5} + 1.25 T \right)^{1.2}
                $$
                """
            )
        
        # Common formulas for both correlations
        st.markdown(
            r"""
            **Common Formulas:**
            
            **Oil Density (Reservoir Conditions):**
            $$
            \rho_o = \frac{350 \times \gamma_o}{5.615 \times B_o} \quad \text{(lb/ft³)}
            $$
            
            $$
            \rho_o = \frac{350 \times \gamma_o}{5.615 \times B_o} \times 16.0185 \quad \text{(kg/m³)}
            $$
            
            **Gas Formation Volume Factor (Ideal Gas):**
            $$
            B_g = 0.02827 \times \frac{(T + 459.67) \times Z}{p} \quad \text{(ft³/scf)}
            $$
            
            Where $Z = 1.0$ (ideal gas assumption), $T$ is in °F, and $p$ is in psia.
            
            **Oil Specific Gravity:**
            $$
            \gamma_o = \frac{141.5}{131.5 + \text{API}}
            $$
            """
        )
        
        st.markdown("</details>")
        
        # Display results in a formatted table
        results_data = {
            "Property": [
                "Solution GOR (Rs)",
                "Oil FVF (Bo)",
                "Gas FVF (Bg)",
                "Oil Density (ρo)"
            ],
            "Value": [
                f"{Rs:,.1f}",
                f"{Bo:.4f}",
                f"{Bg_ft3_scf:.6f}",
                f"{rho_o_kgm3:,.1f}"
            ],
            "Units": [
                "scf/STB",
                "rbbl/STB",
                "ft³/scf",
                "kg/m³"
            ]
        }
        df_results = pd.DataFrame(results_data)
        
        # Style the table to match app format
        styled_df = df_results.style.set_properties(
            **{
                "background-color": "#F6F6F6",
                "color": "#1E1E1E",
                "border": "1px solid #E0E0E0",
                "text-align": "left",
                "padding": "8px",
            }
        ).set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#FF6B6B"),
                    ("color", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "center"),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#FAFAFA")],
            },
            {
                "selector": "tbody tr:hover",
                "props": [("background-color", "#F0F0F0")],
            },
        ])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.info(
            f"**Typical North Sea oil** (~38°API, 90 °C, 220 bar): "
            f"Rs ≈ {Rs:,.0f} scf/STB, Bo ≈ {Bo:.3f}, ρo ≈ {rho_o_kgm3:,.0f} kg/m³"
        )

        st.caption(
            "⚠️ These values are indicative only. Use lab PVT data if available. "
            "Correlations assume ideal gas behavior (Z=1.0) and may not be accurate "
            "for all reservoir conditions."
        )

