"""
Compute: single source of truth for volumes (in-place + recoverables)
"""
import numpy as np
import streamlit as st
from .config import RB_PER_M3
from .utils import safe_div, clip01
from .sampling import sample_pert, rng_from_seed


def sample_fraction_from_dist(rng, dct, n):
    """
    Sample a fraction from distribution (Constant, PERT, Triangular, or Uniform).
    
    Args:
        rng: Random number generator
        dct: Dictionary with "type" and distribution parameters
        n: Number of samples
        
    Returns:
        Array of samples clipped to [0, 1]
    """
    from .sampling import sample_pert, sample_triangular, sample_uniform
    
    dist_type = dct.get("type", "Constant")
    
    if dist_type == "Constant":
        val = dct.get("value", 1.0)
        return np.full(n, val)
    elif dist_type == "PERT":
        return np.clip(sample_pert(rng, dct["min"], dct["mode"], dct["max"], n), 0.0, 1.0)
    elif dist_type == "Triangular":
        return np.clip(sample_triangular(rng, dct["min"], dct["mode"], dct["max"], n), 0.0, 1.0)
    elif dist_type == "Uniform":
        return np.clip(sample_uniform(rng, dct["min"], dct["max"], n), 0.0, 1.0)
    else:
        # Fallback to constant 1.0
        return np.full(n, 1.0)


def derive_saturation_samples(rng, n, mode, ss):
    """
    Derive saturation samples based on input mode.
    Uses existing correlated arrays if available, otherwise samples from distributions.
    SATURATION IS ALWAYS USED - defaults to 1.0 if not configured.
    
    Args:
        rng: np.random.Generator
        n: simulation size
        mode: "Global" | "Water saturation Per zone" | "Per phase"
        ss: streamlit session_state dict-like
        
    Returns:
        dict with arrays: Shc_oil, Shc_gas
        also includes sampled input arrays for export consistency:
          Shc_global or Sw_global  or  Sw_oilzone, Sw_gaszone  or  Shc_oil_input, Shc_gas_input
    """
    out = {}

    if mode.startswith("Global"):
        use_sw_global = ss.get("global_sat_use_sw", False)
        
        # Check if already correlated
        if use_sw_global:
            if "Sw_global" in ss and len(ss["Sw_global"]) == n:
                Sw_global = np.asarray(ss["Sw_global"], dtype=float)
            else:
                d = ss.get("Shc_global_dist", None)
                if d is None:
                    # Default: constant Sw_global = 0.0 (gives Shc = 1.0)
                    Sw_global = np.zeros(n)
                else:
                    Sw_global = sample_fraction_from_dist(rng, d, n)
            Shc_global = np.clip(1.0 - Sw_global, 0.0, 1.0)
            out["Sw_global"] = Sw_global
            out["Shc_global"] = Shc_global
        else:
            if "Shc_global" in ss and len(ss["Shc_global"]) == n:
                Shc_global = np.asarray(ss["Shc_global"], dtype=float)
            else:
                d = ss.get("Shc_global_dist", None)
                if d is None:
                    # Default: constant Shc_global = 1.0
                    Shc_global = np.ones(n)
                else:
                    Shc_global = sample_fraction_from_dist(rng, d, n)
            out["Shc_global"] = Shc_global
        
        out["Shc_oil"] = out.get("Shc_global", Shc_global)
        out["Shc_gas"] = out.get("Shc_global", Shc_global)

    elif mode.startswith("Water saturation"):
        # Check if already correlated
        if "Sw_oilzone" in ss and len(ss["Sw_oilzone"]) == n:
            Sw_oil = np.asarray(ss["Sw_oilzone"], dtype=float)
        else:
            d_o = ss.get("Sw_oilzone_dist", None)
            if d_o is None:
                # Default: constant Sw_oilzone = 0.2 (gives Shc_oil = 0.8)
                Sw_oil = np.full(n, 0.2)
            else:
                Sw_oil = sample_fraction_from_dist(rng, d_o, n)
        
        if "Sw_gaszone" in ss and len(ss["Sw_gaszone"]) == n:
            Sw_gas = np.asarray(ss["Sw_gaszone"], dtype=float)
        else:
            d_g = ss.get("Sw_gaszone_dist", None)
            if d_g is None:
                # Default: constant Sw_gaszone = 0.1 (gives Shc_gas = 0.9)
                Sw_gas = np.full(n, 0.1)
            else:
                Sw_gas = sample_fraction_from_dist(rng, d_g, n)
        
        Shc_oil = np.clip(1.0 - Sw_oil, 0.0, 1.0)
        Shc_gas = np.clip(1.0 - Sw_gas, 0.0, 1.0)
        out["Sw_oilzone"] = Sw_oil
        out["Sw_gaszone"] = Sw_gas
        out["Shc_oil"] = Shc_oil
        out["Shc_gas"] = Shc_gas

    else:  # Per phase
        # Check if already correlated
        if "Shc_oil_input" in ss and len(ss["Shc_oil_input"]) == n:
            Shc_oil_in = np.asarray(ss["Shc_oil_input"], dtype=float)
        else:
            d_o = ss.get("Shc_oil_input_dist", None)
            if d_o is None:
                # Default: constant Shc_oil = 0.8
                Shc_oil_in = np.full(n, 0.8)
            else:
                Shc_oil_in = sample_fraction_from_dist(rng, d_o, n)
        
        if "Shc_gas_input" in ss and len(ss["Shc_gas_input"]) == n:
            Shc_gas_in = np.asarray(ss["Shc_gas_input"], dtype=float)
        else:
            d_g = ss.get("Shc_gas_input_dist", None)
            if d_g is None:
                # Default: constant Shc_gas = 0.85
                Shc_gas_in = np.full(n, 0.85)
            else:
                Shc_gas_in = sample_fraction_from_dist(rng, d_g, n)
        
        out["Shc_oil_input"] = Shc_oil_in
        out["Shc_gas_input"] = Shc_gas_in
        out["Shc_oil"] = Shc_oil_in
        out["Shc_gas"] = Shc_gas_in

    # final hardening - ensure saturation is always used (defaults to 1.0 if missing)
    out["Shc_oil"] = np.clip(out.get("Shc_oil", np.ones(n)), 0.0, 1.0)
    out["Shc_gas"] = np.clip(out.get("Shc_gas", np.ones(n)), 0.0, 1.0)
    return out


def compute_results(
    GRV_m3,
    NtG,
    Por,  # arrays
    RF_oil,
    RF_gas,
    Bg_rb_per_scf,
    InvBo_STB_per_rb,
    GOR_scf_per_STB,
    CY_STB_per_MMscf=None,
    RF_cond=None,
    RF_assoc=None,
    gas_scf_per_boe=6000.0,
    f_oil=None,  # optional array if no GOC split
    GRV_oil_m3=None,
    GRV_gas_m3=None,  # optional arrays for depth-based GOC split
    Shc_oil=None,  # optional hydrocarbon saturation for oil zone
    Shc_gas=None,  # optional hydrocarbon saturation for gas zone
):
    """
    Compute all in-place and recoverable volumes.
    
    Args:
        GRV_m3: Gross rock volume in m³
        NtG: Net-to-gross ratio
        Por: Porosity
        RF_oil: Oil recovery factor
        RF_gas: Gas recovery factor
        Bg_rb_per_scf: Gas formation volume factor in rb/scf
        InvBo_STB_per_rb: Inverse oil formation volume factor in STB/rb
        GOR_scf_per_STB: Gas-oil ratio in scf/STB
        CY_STB_per_MMscf: Condensate yield in STB/MMscf
        RF_cond: Condensate recovery factor
        gas_scf_per_boe: Gas to BOE conversion factor
        f_oil: Oil fraction (for non-GOC methods)
        GRV_oil_m3: Oil GRV in m³ (for GOC methods)
        GRV_gas_m3: Gas GRV in m³ (for GOC methods)
        
    Returns:
        Dictionary of computed volumes
    """
    # Convert to numpy arrays
    GRV_m3 = np.asarray(GRV_m3, dtype=float)
    NtG = np.asarray(NtG, dtype=float)
    Por = np.asarray(Por, dtype=float)
    RF_oil = np.asarray(RF_oil, dtype=float)
    RF_gas = np.asarray(RF_gas, dtype=float)
    # Input guards (robustness)
    GOR_scf_per_STB = np.maximum(np.asarray(GOR_scf_per_STB, dtype=float), 0.0)
    Bg_rb_per_scf = np.maximum(np.asarray(Bg_rb_per_scf, dtype=float), 1e-12)
    InvBo_STB_per_rb = np.maximum(np.asarray(InvBo_STB_per_rb, dtype=float), 0.0)
    
    # Clip recovery factors to [0, 1]
    RF_oil = clip01(RF_oil)
    RF_gas = clip01(RF_gas)
    
    # Calculate pore volume
    PV_total_m3 = GRV_m3 * NtG * Por
    
    # Calculate oil and gas pore volumes
    if GRV_oil_m3 is not None and GRV_gas_m3 is not None:
        # GOC split method
        PV_oil_m3 = GRV_oil_m3 * NtG * Por
        PV_gas_m3 = GRV_gas_m3 * NtG * Por
    else:
        # Explicit oil fraction method
        f = clip01(0.0 if f_oil is None else f_oil)
        PV_oil_m3 = PV_total_m3 * f
        PV_gas_m3 = PV_total_m3 * (1.0 - f)
    
    # Apply saturation (ALWAYS USED - defaults to 1.0 if not provided)
    if Shc_oil is not None and Shc_gas is not None:
        Shc_oil = np.clip(np.asarray(Shc_oil, dtype=float), 0.0, 1.0)
        Shc_gas = np.clip(np.asarray(Shc_gas, dtype=float), 0.0, 1.0)
    else:
        # Default: assume 100% hydrocarbon saturation if not provided
        Shc_oil = np.ones_like(PV_oil_m3)
        Shc_gas = np.ones_like(PV_gas_m3)
    
    PV_oil_hc_m3 = PV_oil_m3 * Shc_oil
    PV_gas_hc_m3 = PV_gas_m3 * Shc_gas
    
    # Calculate recoverable volumes (using hydrocarbon-saturated pore volumes)
    Oil_STB_rec = (PV_oil_hc_m3 * RB_PER_M3) * InvBo_STB_per_rb * RF_oil
    Gas_free_scf_rec = safe_div((PV_gas_hc_m3 * RB_PER_M3) * RF_gas, Bg_rb_per_scf)
    assoc_factor = np.ones_like(Oil_STB_rec)
    if RF_assoc is not None:
        assoc_factor = clip01(np.asarray(RF_assoc, dtype=float))
    Gas_assoc_scf_rec = Oil_STB_rec * GOR_scf_per_STB * assoc_factor
    
    # Condensate recovery (if applicable)
    # CRITICAL: Condensate is calculated ONLY from FREE GAS in the gas zone (Gas_free_scf_rec)
    # NOT from associated gas (Gas_assoc_scf_rec) - condensate comes from the free gas cap/zone
    # For Oil+Gas scenarios, this uses: GRV_gas_m3 -> PV_gas_hc_m3 -> Gas_free_scf_rec
    # Condensate is only calculated if BOTH CY (Condensate Yield) and RF_cond are provided
    Cond_STB_rec = np.zeros_like(Oil_STB_rec)
    has_gas = np.any(Gas_free_scf_rec > 0)
    
    if CY_STB_per_MMscf is not None and RF_cond is not None:
        CY_STB_per_MMscf = np.asarray(CY_STB_per_MMscf, dtype=float)
        RF_cond = np.asarray(RF_cond, dtype=float)
        RF_cond = clip01(RF_cond)
        # Condensate = (Free Gas in scf) × (Condensate Yield in STB/MMscf) × (Condensate RF)
        # IMPORTANT: Uses Gas_free_scf_rec (free gas from gas zone), NOT Gas_assoc_scf_rec (associated gas from oil)
        # Condensate is a liquid that drops out when free gas is produced, not from associated gas
        # Calculate condensate for all trials where free gas is present
        if has_gas:
            Cond_STB_rec = Gas_free_scf_rec * (np.maximum(CY_STB_per_MMscf, 0.0) / 1_000_000.0) * RF_cond
    elif has_gas:
        # Gas is present but condensate won't be calculated - this is expected if CY or RF_cond not set
        # (User must explicitly set Condensate Yield in Fluids inputs)
        pass
    
    # Total volumes
    Total_gas_scf_rec = Gas_free_scf_rec + Gas_assoc_scf_rec
    Total_liquids_STB = Oil_STB_rec + Cond_STB_rec
    THR_BOE = Oil_STB_rec + Cond_STB_rec + safe_div(Total_gas_scf_rec, gas_scf_per_boe)
    
    # BOE breakdown
    Oil_BOE = Oil_STB_rec
    Cond_BOE = Cond_STB_rec
    Gas_free_BOE = safe_div(Gas_free_scf_rec, gas_scf_per_boe)
    Gas_assoc_BOE = safe_div(Gas_assoc_scf_rec, gas_scf_per_boe)
    Total_gas_BOE = Gas_free_BOE + Gas_assoc_BOE
    
    # In-situ volumes (before applying recovery factors, using hydrocarbon-saturated pore volumes)
    V_oil_insitu_m3 = (PV_oil_hc_m3 * RB_PER_M3) * InvBo_STB_per_rb
    V_gas_insitu_m3 = safe_div((PV_gas_hc_m3 * RB_PER_M3), Bg_rb_per_scf)
    
    results = {
        # In-place volumes
        'PV_total_m3': PV_total_m3,
        'PV_oil_m3': PV_oil_m3,
        'PV_gas_m3': PV_gas_m3,
        'PV_oil_hc_m3': PV_oil_hc_m3,
        'PV_gas_hc_m3': PV_gas_hc_m3,
        'V_oil_insitu_m3': V_oil_insitu_m3,
        'V_gas_insitu_m3': V_gas_insitu_m3,
        
        # Recoverable volumes
        'Oil_STB_rec': Oil_STB_rec,
        'Gas_free_scf_rec': Gas_free_scf_rec,
        'Gas_assoc_scf_rec': Gas_assoc_scf_rec,
        'Cond_STB_rec': Cond_STB_rec,
        'Total_gas_scf_rec': Total_gas_scf_rec,
        'Total_liquids_STB': Total_liquids_STB,
        'Total_surface_BOE': THR_BOE,
        
        # BOE breakdown
        'Oil_BOE': Oil_BOE,
        'Cond_BOE': Cond_BOE,
        'Gas_free_BOE': Gas_free_BOE,
        'Gas_assoc_BOE': Gas_assoc_BOE,
        'Total_gas_BOE': Total_gas_BOE,
    }
    return {
        key: np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        for key, val in results.items()
    }


def run_simulation(inputs: dict, run_id: int, seed: int | None, no_cache_nonce: int = 0) -> dict:
    """
    Centralized simulation entry point that ensures fresh computation with current inputs.
    
    This function:
    - Does NOT use @st.cache_data (no caching)
    - Reads inputs from dict only (not scattered session reads)
    - Uses seed_effective = (seed or 0) + run_id for RNG
    - Returns results dict and stores into st.session_state
    
    Args:
        inputs: Dictionary containing all simulation inputs (from collect_all_inputs_from_session)
        run_id: Current run ID (increments on each Run button click)
        seed: Base random seed (optional)
        no_cache_nonce: Nonce to invalidate any accidental caches
        
    Returns:
        Dictionary of results arrays
    """
    # Hard invalidate any accidental caches by consuming nonce and run_id
    _ = (run_id, no_cache_nonce)
    
    # Effective seed: base seed + run_id ensures different stream each run
    seed_eff = (seed or 0) + int(run_id)
    rng = rng_from_seed(seed_eff)
    
    # Get simulation size
    num_sims = inputs.get("num_sims", 10_000)
    
    # CRITICAL: Read all arrays from session state (they should already be updated by render_param)
    # We use inputs dict for configuration, but actual sample arrays come from session state
    # This ensures we use the latest generated samples
    
    # Required arrays
    required_keys = ["sGRV_m3_final", "sNtG", "sp", "sRF_oil", "sRF_gas", "sBg", "sInvBo", "sGOR"]
    missing = [k for k in required_keys if k not in st.session_state]
    if missing:
        raise ValueError(f"Missing required arrays in session state: {missing}")
    
    # Read arrays from session state (fresh read)
    GRV_m3 = np.asarray(st.session_state["sGRV_m3_final"], dtype=float)
    NtG = np.asarray(st.session_state["sNtG"], dtype=float)
    Por = np.asarray(st.session_state["sp"], dtype=float)
    RF_oil = np.asarray(st.session_state["sRF_oil"], dtype=float)
    RF_gas = np.asarray(st.session_state["sRF_gas"], dtype=float)
    Bg = np.asarray(st.session_state["sBg"], dtype=float)
    InvBo = np.asarray(st.session_state["sInvBo"], dtype=float)
    GOR = np.asarray(st.session_state["sGOR"], dtype=float)
    
    # Optional arrays
    sCY = st.session_state.get("sCY", None)
    sRF_cond = st.session_state.get("sRF_cond", None)
    sRF_assoc = st.session_state.get("sRF_assoc_gas", RF_oil)
    gas_scf_per_boe = inputs.get("gas_scf_per_boe", 6000.0)
    
    # CRITICAL: If gas is present (Oil+Gas or Gas-only) and CY is not set, 
    # condensate won't be calculated. We should at least ensure RF_cond has a default.
    # Note: CY (Condensate Yield) must be explicitly set by user in Fluids inputs.
    # If not set, condensate will be zero even if gas is present.
    
    # Get GRV split based on current fluid_type and grv_option from inputs
    fluid_type = inputs.get("fluid_type", "Oil + Gas")
    grv_option = inputs.get("grv_option", "Direct GRV")
    
    # CRITICAL: Handle fluid_type FIRST to override any stale GRV arrays
    # This ensures that if fluid_type changed, we use the correct split regardless of cached arrays
    if fluid_type == "Oil":
        # All GRV is oil - override any cached split
        GRV_oil_m3 = GRV_m3.copy()
        GRV_gas_m3 = np.zeros_like(GRV_m3)
    elif fluid_type == "Gas":
        # All GRV is gas - override any cached split
        GRV_oil_m3 = np.zeros_like(GRV_m3)
        GRV_gas_m3 = GRV_m3.copy()
    else:  # Oil + Gas - try to get split from cached arrays
        GRV_oil_m3 = None
        GRV_gas_m3 = None
        
        # Get split GRV arrays based on CURRENT method
        if "sGRV_oil_m3" in st.session_state and "sGRV_gas_m3" in st.session_state:
            GRV_oil_m3 = np.asarray(st.session_state["sGRV_oil_m3"], dtype=float)
            GRV_gas_m3 = np.asarray(st.session_state["sGRV_gas_m3"], dtype=float)
        elif grv_option == "Direct GRV":
            if "direct_GRV_oil_m3" in st.session_state:
                GRV_oil_m3 = np.asarray(st.session_state["direct_GRV_oil_m3"], dtype=float)
            if "direct_GRV_gas_m3" in st.session_state:
                GRV_gas_m3 = np.asarray(st.session_state["direct_GRV_gas_m3"], dtype=float)
        elif grv_option == "Area × Thickness × GCF":
            if "atgcf_GRV_oil_m3" in st.session_state:
                GRV_oil_m3 = np.asarray(st.session_state["atgcf_GRV_oil_m3"], dtype=float)
            if "atgcf_GRV_gas_m3" in st.session_state:
                GRV_gas_m3 = np.asarray(st.session_state["atgcf_GRV_gas_m3"], dtype=float)
        
        # Fallback: use f_oil if split GRV not available
        if GRV_oil_m3 is None or GRV_gas_m3 is None:
            f_oil_val = 0.5
            if grv_option == "Direct GRV" and "direct_f_oil" in st.session_state:
                f_oil_val = float(st.session_state["direct_f_oil"])
            elif grv_option == "Area × Thickness × GCF" and "atgcf_f_oil" in st.session_state:
                f_oil_val = float(st.session_state["atgcf_f_oil"])
            elif "f_oil" in st.session_state:
                f_oil_val = float(st.session_state["f_oil"])
            
            if GRV_oil_m3 is None:
                GRV_oil_m3 = GRV_m3 * f_oil_val
            if GRV_gas_m3 is None:
                GRV_gas_m3 = GRV_m3 * (1.0 - f_oil_val)
    
    # Get f_oil array - ensure we extract scalar value if f_oil is an array
    f_oil_val = st.session_state.get("f_oil", 0.5)
    if isinstance(f_oil_val, np.ndarray):
        # If f_oil is an array, use its mean or first value
        f_oil_val = float(np.mean(f_oil_val)) if len(f_oil_val) > 0 else 0.5
    else:
        f_oil_val = float(f_oil_val)
    f_oil = np.full(num_sims, f_oil_val)
    
    # Derive saturation samples
    mode = inputs.get("sat_mode", "Global")
    sat = derive_saturation_samples(rng, num_sims, mode, st.session_state)
    Shc_oil = sat.get("Shc_oil", np.ones(num_sims))
    Shc_gas = sat.get("Shc_gas", np.ones(num_sims))
    
    # Store saturation arrays in session state
    for k, v in sat.items():
        st.session_state[k] = v
    st.session_state["Shc_oil"] = Shc_oil
    st.session_state["Shc_gas"] = Shc_gas
    
    # Compute results using centralized function
    results = compute_results(
        GRV_m3=GRV_m3,
        NtG=NtG,
        Por=Por,
        f_oil=f_oil,
        RF_oil=RF_oil,
        RF_gas=RF_gas,
        Bg_rb_per_scf=Bg,
        InvBo_STB_per_rb=InvBo,
        GOR_scf_per_STB=GOR,
        CY_STB_per_MMscf=sCY,
        RF_cond=sRF_cond,
        RF_assoc=sRF_assoc,
        gas_scf_per_boe=gas_scf_per_boe,
        GRV_oil_m3=GRV_oil_m3,
        GRV_gas_m3=GRV_gas_m3,
        Shc_oil=Shc_oil,
        Shc_gas=Shc_gas,
    )
    
    # Store results with run tracking metadata
    st.session_state["results_cache"] = results
    st.session_state["results_run_id"] = run_id
    st.session_state["results_input_hash"] = inputs.get("_input_hash", "")
    st.session_state["results_seed_used"] = seed_eff
    
    return results
