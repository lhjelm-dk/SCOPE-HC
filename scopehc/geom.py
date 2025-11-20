"""
Geometry: GCF table + interpolation, depth-table → GRV integrators
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

try:  # pragma: no cover - optional UI dependency
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _NoOp:
        def write(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def info(self, *a, **k): pass

    st = _NoOp()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_GCF_TABLE = DATA_DIR / "gcf_table.csv"


def _cumulative_trapz(y: np.ndarray, step: float) -> np.ndarray:
	"""Return cumulative integral using trapezoid rule with constant depth spacing."""
	y = np.asarray(y, dtype=float)
	if y.size == 0:
		return y
	# average adjacent points then multiply by step
	avg_pairs = (y[:-1] + y[1:]) * 0.5
	volumes = np.cumsum(avg_pairs) * float(step)
	# prepend 0 to align with depths array
	return np.insert(volumes, 0, 0.0)


def load_gcf_table(path: str | Path):
    """
    CSV with columns: ratio (0..1 ascending), gcf_dome, gcf_anticline_2, gcf_anticline_5, gcf_anticline_10,
    gcf_flat_dome, gcf_flat_anticline_2, gcf_flat_anticline_5, gcf_flat_anticline_10, gcf_block.
    """
    path = Path(path)
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"GCF table at {path} is empty")

    ratio = np.asarray(data["ratio"], dtype=float)
    order = np.argsort(ratio)
    ratio = ratio[order]

    table = {
        key: np.asarray(data[key], dtype=float)[order]
        for key in data.dtype.names
        if key != "ratio"
    }

    for name, values in table.items():
        diffs = np.diff(values)
        if not (np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)):
            raise AssertionError(f"GCF curve {name} must be monotonic")

    return ratio, table


@lru_cache(maxsize=1)
def _cached_gcf_table(path: str | None = None):
    target = Path(path) if path else DEFAULT_GCF_TABLE
    return load_gcf_table(target)


def get_gcf_lookup_table(path: str | None = None):
    """
    Return ratio array and dictionary of curves keyed by (shape_type, lw_ratio).
    """
    ratio, table = _cached_gcf_table(path)
    lookup_table = {
        (1, 1): table["gcf_dome"],
        (2, 2): table["gcf_anticline_2"],
        (2, 5): table["gcf_anticline_5"],
        (2, 10): table["gcf_anticline_10"],
        (3, 1): table["gcf_flat_dome"],
        (4, 2): table["gcf_flat_anticline_2"],
        (4, 5): table["gcf_flat_anticline_5"],
        (4, 10): table["gcf_flat_anticline_10"],
        (5, 1): table["gcf_block"],
    }
    return ratio, lookup_table


def interpolate_gcf(shape_type: int, lw_ratio: int, res_tk_closure_ratio: float, path: str | None = None) -> float:
    """
    Interpolate GCF value based on shape type, L/W ratio, and Res Tk/Closure ratio.
    """
    ratio, lookup_table = get_gcf_lookup_table(path)

    shape = int(np.asarray(shape_type).item()) if hasattr(shape_type, "__getitem__") else int(shape_type)
    lw = int(np.asarray(lw_ratio).item()) if hasattr(lw_ratio, "__getitem__") else int(lw_ratio)
    x = float(res_tk_closure_ratio)

    curve = lookup_table.get((shape, lw))
    if curve is None:
        if shape == 5:
            curve = lookup_table[(5, 1)]
        else:
            curve = lookup_table[(1, 1)]

    x = np.clip(x, ratio.min(), ratio.max())
    return float(np.interp(x, ratio, curve))


def integrate_grv_from_depth_table(depths_m, areas_km2, contact_depth_m) -> float:
    """
    Integrate GRV from depth-area table using trapezoidal integration.
    
    Args:
        depths_m: Array of depths in meters
        areas_km2: Array of areas in km²
        contact_depth_m: Contact depth in meters
        
    Returns:
        GRV in m³
    """
    # Trapezoidal integration with clipped arrays + area at exact contact
    dmin, dmax = np.min(depths_m), min(contact_depth_m, np.max(depths_m))
    if dmax <= dmin:
        return 0.0
    
    area_at_contact = np.interp(dmax, depths_m, areas_km2)
    mask = (depths_m >= dmin) & (depths_m <= dmax)
    d_clip = np.concatenate([depths_m[mask], [dmax]])
    a_clip = np.concatenate([areas_km2[mask], [area_at_contact]])
    
    # Sort by depth
    order = np.argsort(d_clip)
    d_clip = d_clip[order]
    a_clip = a_clip[order]
    
    # Remove duplicates
    d_clip, idx = np.unique(d_clip, return_index=True)
    a_clip = a_clip[idx]
    
    # Trapezoidal integration
    grv_km2m = np.trapz(a_clip, d_clip)
    return float(grv_km2m * 1_000_000.0)  # km²·m → m³


def _sort_unique_depth_area(depth_m, area_m2):
    """Sort depth and area arrays by depth, remove duplicate depths."""
    d = np.asarray(depth_m, dtype=float)
    a = np.asarray(area_m2, dtype=float)
    # Sort by depth increasing (positive downward)
    idx = np.argsort(d)
    d = d[idx]
    a = a[idx]
    # Drop duplicate depths by keeping first occurrence
    uniq, uidx = np.unique(d, return_index=True)
    return d[uidx], a[uidx]


def _clip_interval_and_interpolate(d, a, z_top, z_base):
    """
    Return depth & area arrays clipped to [z_top, z_base] with interpolated endpoints.
    If z_base <= z_top, return empty arrays.
    """
    if z_base <= z_top:
        return np.array([]), np.array([])
    
    # Ensure arrays are sorted and unique
    d, a = _sort_unique_depth_area(d, a)
    
    # Build mask for interior points
    m = (d > z_top) & (d < z_base)
    d_clip = d[m]
    a_clip = a[m]
    
    # Interpolate areas at z_top and z_base
    # If outside data range, clamp to nearest available value
    zmin, zmax = d[0], d[-1]
    zt = np.clip(z_top, zmin, zmax)
    zb = np.clip(z_base, zmin, zmax)
    
    a_top = np.interp(zt, d, a)
    a_base = np.interp(zb, d, a)
    
    d_final = np.concatenate(([zt], d_clip, [zb]))
    a_final = np.concatenate(([a_top], a_clip, [a_base]))
    return d_final, a_final


def _integrate_area_depth_trapz(depth_m, area_m2):
    """Compute volume (m^3) as ∫ A(z) dz using trapezoids over piecewise linear A(z)."""
    if depth_m.size < 2:
        return 0.0
    # Trapezoidal rule on the clipped segment
    dz = np.diff(depth_m)
    # Areas nonnegative
    a = np.maximum(area_m2, 0.0)
    vol = np.sum(0.5 * (a[:-1] + a[1:]) * dz)
    # Clip negatives (shouldn't occur)
    return max(float(vol), 0.0)


def grv_by_depth_v3_compatible(
    depth_m, area_m2,
    top_structure_m,
    goc_m=None,
    owc_m=None,
    spill_m=None
):
    """
    Reproduce v3 behavior for GRV split by fluid contacts using a depth-area table.
    
    Zones:
      Gas: Top -> GOC
      Oil: GOC -> OWC
    If GOC is None: single oil zone Top -> OWC
    If both GOC and OWC are None: entire Top -> spill (if given) else Top -> deepest sample
    Spill (if provided) caps the base of HC zones.
    
    Args:
        depth_m: Array of depths in meters (positive downward)
        area_m2: Array of areas in m² (or km², will be converted)
        top_structure_m: Top structure depth in meters
        goc_m: Optional GOC depth in meters
        owc_m: Optional OWC depth in meters (HC contact)
        spill_m: Optional spill point depth in meters
    
    Returns:
        dict with:
          GRV_gas_m3, GRV_oil_m3, GRV_total_m3,
          H_gas_m, H_oil_m (effective column heights; 0 if absent)
    """
    d, a = _sort_unique_depth_area(depth_m, area_m2)
    z_top = float(top_structure_m)
    
    # Convert area from km² to m² if needed (check typical range)
    # If areas are < 100, assume they're in km² and convert
    if np.max(a) < 100:
        a = a * 1_000_000.0  # km² -> m²
    
    # Determine base limit = spill or max depth
    z_max = d[-1]
    z_base_limit = float(spill_m) if spill_m is not None else z_max
    
    # Sanitize contacts and clip to valid range
    def _clip(z):
        if z is None:
            return None
        return float(np.clip(z, d[0], z_base_limit))
    
    z_goc = _clip(goc_m)
    z_owc = _clip(owc_m)
    
    # Enforce ordering: top <= GOC <= OWC <= base_limit
    # If contacts reversed, fix gracefully by sorting
    contacts = [z for z in [z_goc, z_owc] if z is not None]
    contacts_sorted = sorted(contacts)
    if len(contacts) == 2:
        z_goc, z_owc = contacts_sorted[0], contacts_sorted[1]
    elif len(contacts) == 1:
        # Single contact stays as is
        z_goc = contacts[0] if goc_m is not None else None
        z_owc = contacts[0] if owc_m is not None else None
    
    # Compute volumes by zones using v3 rules
    GRV_gas_m3 = 0.0
    GRV_oil_m3 = 0.0
    H_gas_m = 0.0
    H_oil_m = 0.0
    
    # Helper to integrate interval [z1, z2]
    def _vol(z1, z2):
        dd, aa = _clip_interval_and_interpolate(d, a, z1, z2)
        return _integrate_area_depth_trapz(dd, aa)
    
    # Cases
    if (z_goc is not None) and (z_owc is not None):
        # Gas: top -> GOC ; Oil: GOC -> OWC ; cap both by base limit
        z1g, z2g = z_top, min(z_goc, z_base_limit)
        z1o, z2o = max(z_goc, z_top), min(z_owc, z_base_limit)
        if z2g > z1g:
            GRV_gas_m3 = _vol(z1g, z2g)
            H_gas_m = z2g - z1g
        if z2o > z1o:
            GRV_oil_m3 = _vol(z1o, z2o)
            H_oil_m = z2o - z1o
    
    elif (z_goc is None) and (z_owc is not None):
        # No gas: Oil only from top -> OWC
        z1o, z2o = z_top, min(z_owc, z_base_limit)
        if z2o > z1o:
            GRV_oil_m3 = _vol(z1o, z2o)
            H_oil_m = z2o - z1o
    
    elif (z_goc is not None) and (z_owc is None):
        # Gas only from top -> GOC
        z1g, z2g = z_top, min(z_goc, z_base_limit)
        if z2g > z1g:
            GRV_gas_m3 = _vol(z1g, z2g)
            H_gas_m = z2g - z1g
    
    else:
        # No contacts: HC from top -> base_limit
        z1, z2 = z_top, z_base_limit
        if z2 > z1:
            v = _vol(z1, z2)
            # In v3, when no contacts, treat as oil zone unless a UI split f_oil exists upstream
            GRV_oil_m3 = v
            H_oil_m = z2 - z1
    
    GRV_total_m3 = GRV_gas_m3 + GRV_oil_m3
    return {
        'GRV_gas_m3': GRV_gas_m3,
        'GRV_oil_m3': GRV_oil_m3,
        'GRV_total_m3': GRV_total_m3,
        'H_gas_m': H_gas_m,
        'H_oil_m': H_oil_m,
    }


def calculate_grv_from_depth_table(depths_m, areas_km2, spill_depth_m, hc_depth_m, goc_depth_m=None):
    """
    Calculate GRV from depth table with optional GOC split (v3-compatible).
    
    This function now uses grv_by_depth_v3_compatible internally for v3 parity.
    
    Args:
        depths_m: Array of depths in meters
        areas_km2: Array of areas in km²
        spill_depth_m: Spill point depth in meters (used as top structure if no better value)
        hc_depth_m: Hydrocarbon contact depth in meters (OWC)
        goc_depth_m: Optional GOC depth in meters
        
    Returns:
        dict: GRV_total_m3, GRV_oil_m3, GRV_gas_m3
    """
    # Determine top structure (minimum depth in table, or spill if provided)
    depths_arr = np.asarray(depths_m, dtype=float)
    top_structure_m = float(np.min(depths_arr))
    
    # Use v3-compatible function
    # Note: areas_km2 will be auto-converted to m² in grv_by_depth_v3_compatible
    result = grv_by_depth_v3_compatible(
        depth_m=depths_m,
        area_m2=areas_km2,  # Will be converted if < 100
        top_structure_m=top_structure_m,
        goc_m=goc_depth_m,
        owc_m=hc_depth_m,
        spill_m=spill_depth_m
    )
    
    return {
        'GRV_total_m3': result['GRV_total_m3'],
        'GRV_oil_m3': result['GRV_oil_m3'],
        'GRV_gas_m3': result['GRV_gas_m3']
    }


def derive_goc_from_mode(top_m: float, hcwc_m: float, mode: str, ss, trial_idx: int = None) -> float:
    """
    Derive GOC depth from the chosen definition mode.
    
    Args:
        top_m: Top structure depth in meters
        hcwc_m: Hydrocarbon-water contact depth in meters
        mode: "Direct depth" | "Oil fraction of HC column" | "Oil column height"
        ss: Session state dict-like object
        trial_idx: Optional trial index for array-based values (default: None)
        
    Returns:
        z_GOC (depth in meters, positive downward)
    """
    top_m = float(top_m)
    hcwc_m = float(hcwc_m)
    
    if hcwc_m <= top_m:
        return top_m  # Degenerate column
    
    if mode == "Direct depth":
        z_goc = ss.get("da_goc_m", (top_m + hcwc_m) / 2)
        # Handle array or scalar
        if isinstance(z_goc, np.ndarray) and trial_idx is not None:
            z_goc = float(z_goc[trial_idx])
        else:
            z_goc = float(z_goc)
        return float(np.clip(z_goc, top_m, hcwc_m))
    
    H_hc = hcwc_m - top_m
    
    if mode == "Oil fraction of HC column":
        f_oil = ss.get("da_oil_frac", 0.60)
        # Handle array or scalar (for sliders, it's usually a scalar)
        if isinstance(f_oil, np.ndarray) and trial_idx is not None:
            f_oil = float(f_oil[trial_idx])
        else:
            f_oil = float(f_oil)
        H_oil = np.clip(f_oil, 0.0, 1.0) * H_hc
        return hcwc_m - H_oil  # Shallower depth (GOC above OWC)
    
    # Oil column height in meters above HCWC
    H_oil_val = ss.get("da_h_oil_m", 0.60 * H_hc)
    # Handle array or scalar
    if isinstance(H_oil_val, np.ndarray) and trial_idx is not None:
        H_oil = float(H_oil_val[trial_idx])
    else:
        H_oil = float(H_oil_val)
    H_oil = float(np.clip(H_oil, 0.0, H_hc))
    return hcwc_m - H_oil
