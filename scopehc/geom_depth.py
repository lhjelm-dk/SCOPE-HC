import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _cumulative_trapz(y: np.ndarray, step: float) -> np.ndarray:
	"""Cumulative integral via trapezoid rule with constant depth spacing."""
	y = np.asarray(y, dtype=float)
	if y.size == 0:
		return y
	avg_pairs = 0.5 * (y[:-1] + y[1:])
	volumes = np.cumsum(avg_pairs) * float(step)
	return np.insert(volumes, 0, 0.0)


def compute_dgrv_top_plus_thickness(
	top_df: pd.DataFrame,
	thickness_m: float,
	step_m: float,
	extrapolate: bool,
	target_depth_m: float,
) -> Dict[str, Any]:
	"""
	CO2-style: derive base by depth-shifting top by constant thickness.
	Returns depth grid, interpolated top/base areas (km2), cumulative volumes (km2·m), dGRV, and GRV at target depth.
	"""
	df = top_df.copy()
	df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)
	df["Top area (km2)"] = pd.to_numeric(df["Top area (km2)"], errors="coerce").fillna(0.0)

	depths_top = df["Depth"].astype(float).to_numpy()
	areas_top = df["Top area (km2)"].astype(float).to_numpy()
	if depths_top.size < 2:
		return {"error": "Need at least two rows in the top table"}

	max_top_depth = float(depths_top[-1])
	step_m = float(max(step_m, 1e-6))
	thickness_m = float(max(thickness_m, 0.0))

	target_max_depth = max(max_top_depth + thickness_m, float(target_depth_m))
	if extrapolate and target_max_depth > max_top_depth and depths_top.size >= 2:
		slope = (areas_top[-1] - areas_top[-2]) / (depths_top[-1] - depths_top[-2])
		extra_depths = np.arange(max_top_depth + step_m, target_max_depth + step_m, step_m)
		extra_areas = areas_top[-1] + slope * (extra_depths - max_top_depth)
		depths_top = np.concatenate([depths_top, extra_depths])
		areas_top = np.concatenate([areas_top, extra_areas])

	depths_base = depths_top + thickness_m
	areas_base = areas_top.copy()

	z_min = float(depths_top[0])
	z_max = max(float(depths_base[-1]), float(target_depth_m))
	depth_grid = np.arange(z_min, z_max + step_m, step_m)
	top_interp = np.interp(depth_grid, depths_top, areas_top)
	base_interp = np.interp(depth_grid, depths_base, areas_base)

	vol_top = _cumulative_trapz(top_interp, step_m)
	vol_base = _cumulative_trapz(base_interp, step_m)
	dgrv = vol_top - vol_base

	if target_depth_m < depth_grid[0] or target_depth_m > depth_grid[-1]:
		grv_at_target = np.nan
	else:
		grv_at_target = float(np.interp(target_depth_m, depth_grid, dgrv))

	return {
		"depths": depth_grid,
		"top_interp": top_interp,
		"base_interp": base_interp,
		"vol_top": vol_top,
		"vol_base": vol_base,
		"dgrv": dgrv,
		"grv_at_target": grv_at_target,
	}


def compute_dgrv_top_base_table(
	df_in: pd.DataFrame,
	step_m: float,
	extrapolate: bool,
	target_depth_m: float,
) -> Dict[str, Any]:
	"""
	CO2-style: use provided Depth/Top/Base area table directly.
	"""
	df = df_in.copy()
	for col in ["Depth", "Top area (km2)", "Base area (km2)"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)

	depths = df["Depth"].astype(float).to_numpy()
	top_areas = df["Top area (km2)"].fillna(0.0).astype(float).to_numpy()
	base_areas = df["Base area (km2)"].fillna(0.0).astype(float).to_numpy()
	if depths.size < 2:
		return {"error": "Need at least two rows in the depth/area table"}

	step_m = float(max(step_m, 1e-6))
	z_min = float(depths[0])
	z_max = max(float(depths[-1]), float(target_depth_m))
	depths_out = np.arange(z_min, z_max + step_m, step_m)

	# Optional linear extrapolation (top & base independently)
	if extrapolate and target_depth_m > depths[-1] and depths.size >= 2:
		slope_top = (top_areas[-1] - top_areas[-2]) / (depths[-1] - depths[-2])
		slope_base = (base_areas[-1] - base_areas[-2]) / (depths[-1] - depths[-2])
		extra_depths = np.arange(depths[-1] + step_m, z_max + step_m, step_m)
		top_areas = np.concatenate([top_areas, top_areas[-1] + slope_top * (extra_depths - depths[-1])])
		base_areas = np.concatenate([base_areas, base_areas[-1] + slope_base * (extra_depths - depths[-1])])
		depths = np.concatenate([depths, extra_depths])

	top_interp = np.interp(depths_out, depths, top_areas)
	base_interp = np.interp(depths_out, depths, base_areas)
	vol_top = _cumulative_trapz(top_interp, step_m)
	vol_base = _cumulative_trapz(base_interp, step_m)
	dgrv = vol_top - vol_base

	if target_depth_m < depths_out[0] or target_depth_m > depths_out[-1]:
		grv_at_target = np.nan
	else:
		grv_at_target = float(np.interp(target_depth_m, depths_out, dgrv))

	return {
		"depths": depths_out,
		"top_interp": top_interp,
		"base_interp": base_interp,
		"vol_top": vol_top,
		"vol_base": vol_base,
		"dgrv": dgrv,
		"grv_at_target": grv_at_target,
	}


def compute_grv_top_plus_contacts(
	top_df: pd.DataFrame,
	step_m: float,
	extrapolate: bool,
	top_depth_m: float,
	hcwc_m: float,
	goc_m: Optional[float],
	case: str,
) -> Dict[str, Any]:
	"""
	Integrate A_top(z) between Top and contacts (no synthetic base surface).
	- Oil-only: integrate Top -> HCWC
	- Gas-only: integrate Top -> GOC
	- Oil+Gas:  Gas: Top -> GOC;  Oil: GOC -> HCWC
	Returns km2·m volumes and column heights.
	"""
	df = top_df.copy()
	df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)
	df["Top area (km2)"] = pd.to_numeric(df["Top area (km2)"], errors="coerce").fillna(0.0)
	depths_top = df["Depth"].astype(float).to_numpy()
	areas_top = df["Top area (km2)"].astype(float).to_numpy()
	if depths_top.size < 2:
		return {"error": "Need at least two rows in the top table"}

	z_need = [top_depth_m, hcwc_m]
	if case == "Gas" and goc_m is not None:
		z_need.append(goc_m)
	if case == "Oil+Gas" and goc_m is not None:
		z_need.append(goc_m)

	z_min = float(min(min(deep for deep in depths_top), min(z_need)))
	z_max = float(max(max(deep for deep in depths_top), max(z_need)))
	step_m = float(max(step_m, 1e-6))
	depth_grid = np.arange(z_min, z_max + step_m, step_m)

	# Optional extrapolation of top area beyond table to cover contacts
	if extrapolate and (z_min < depths_top[0] or z_max > depths_top[-1]) and depths_top.size >= 2:
		# front
		if z_min < depths_top[0]:
			slope_front = (areas_top[1] - areas_top[0]) / (depths_top[1] - depths_top[0])
			extra_front = np.arange(z_min, depths_top[0], step_m)
			extra_front_areas = areas_top[0] + slope_front * (extra_front - depths_top[0])
			depths_top = np.concatenate([extra_front, depths_top])
			areas_top = np.concatenate([extra_front_areas, areas_top])
		# back
		if z_max > depths_top[-1]:
			slope_back = (areas_top[-1] - areas_top[-2]) / (depths_top[-1] - depths_top[-2])
			extra_back = np.arange(depths_top[-1] + step_m, z_max + step_m, step_m)
			extra_back_areas = areas_top[-1] + slope_back * (extra_back - depths_top[-1])
			depths_top = np.concatenate([depths_top, extra_back])
			areas_top = np.concatenate([areas_top, extra_back_areas])

	top_interp = np.interp(depth_grid, depths_top, areas_top)
	vol_top = _cumulative_trapz(top_interp, step_m)

	def vol_between(z1: float, z2: float) -> float:
		a, b = float(min(z1, z2)), float(max(z1, z2))
		if b < depth_grid[0] or a > depth_grid[-1]:
			return np.nan
		v_a = float(np.interp(a, depth_grid, vol_top))
		v_b = float(np.interp(b, depth_grid, vol_top))
		return max(0.0, v_b - v_a)

	H_hc = max(0.0, hcwc_m - top_depth_m)
	if case == "Oil":
		H_oil = H_hc
		GRV_oil = vol_between(top_depth_m, hcwc_m)
		return {
			"depths": depth_grid,
			"top_interp": top_interp,
			"H_oil_m": H_oil,
			"H_gas_m": 0.0,
			"GRV_oil_km2m": GRV_oil,
			"GRV_gas_km2m": 0.0,
			"GRV_total_km2m": GRV_oil,
		}
	if case == "Gas":
		# For Gas-only case: use HCWC (not GOC)
		# HCWC is the hydrocarbon-water contact, which works for both Oil and Gas
		# GOC is only needed for Oil+Gas cases to separate gas and oil zones
		if hcwc_m is None:
			return {"error": "HCWC is required for Gas-only case"}
		H_gas = max(0.0, hcwc_m - top_depth_m)
		GRV_gas = vol_between(top_depth_m, hcwc_m)
		return {
			"depths": depth_grid,
			"top_interp": top_interp,
			"H_oil_m": 0.0,
			"H_gas_m": H_gas,
			"GRV_oil_km2m": 0.0,
			"GRV_gas_km2m": GRV_gas,
			"GRV_total_km2m": GRV_gas,
		}

	# Oil + Gas
	if goc_m is None:
		return {"error": "GOC is required for Oil+Gas case"}
	H_gas = max(0.0, goc_m - top_depth_m)
	H_oil = max(0.0, hcwc_m - goc_m)
	GRV_gas = vol_between(top_depth_m, goc_m)
	GRV_oil = vol_between(goc_m, hcwc_m)
	return {
		"depths": depth_grid,
		"top_interp": top_interp,
		"H_oil_m": H_oil,
		"H_gas_m": H_gas,
		"GRV_oil_km2m": GRV_oil,
		"GRV_gas_km2m": GRV_gas,
		"GRV_total_km2m": GRV_oil + GRV_gas,
	}


