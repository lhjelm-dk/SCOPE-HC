"""
PVT: Bg/Bo/Rs correlations, P-T helpers
"""
import numpy as np
from typing import Optional, Tuple
from .utils import invBg_to_Bg_rb_per_scf


def bg_from_PT(psia, F, z=0.9):
    """
    Calculate gas formation volume factor from pressure and temperature.
    
    Bg [rb/scf] = 0.02827 * Z * T(°R) / P(psia)
    
    Args:
        psia: Pressure in psia
        F: Temperature in Fahrenheit
        z: Gas compressibility factor (default 0.9)
        
    Returns:
        Bg in rb/scf
    """
    return 0.02827 * z * (F + 459.67) / np.maximum(psia, 1e-6)


def standing_Rs(P_psia, T_F, API, gas_sg):
    """
    Standing correlation for solution gas-oil ratio.
    
    Args:
        P_psia: Pressure in psia
        T_F: Temperature in Fahrenheit
        API: Oil API gravity
        gas_sg: Gas specific gravity
        
    Returns:
        Rs in scf/STB
    """
    A = 0.0125 * API - 0.00091 * T_F
    denom = 18.2 + 1.4e-3 * T_F
    term = (P_psia / denom) ** 1.2048
    return gas_sg * term * (10 ** A)


def standing_Bo(P_psia, T_F, API, gas_sg, Rs_scf_STB):
    """
    Standing correlation for oil formation volume factor.
    
    Args:
        P_psia: Pressure in psia
        T_F: Temperature in Fahrenheit
        API: Oil API gravity
        gas_sg: Gas specific gravity
        Rs_scf_STB: Solution gas-oil ratio in scf/STB
        
    Returns:
        Bo in rb/STB
    """
    gamma_o = 141.5 / (API + 131.5)
    term = (Rs_scf_STB * (gas_sg / gamma_o) ** 0.5 + 1.25 * T_F)
    return 0.9759 + 12e-5 * (term ** 1.2)


def z_factor_standing_katz(P_psia, T_F, gas_sg):
    """
    Standing-Katz correlation for gas compressibility factor.
    
    Args:
        P_psia: Pressure in psia
        T_F: Temperature in Fahrenheit
        gas_sg: Gas specific gravity
        
    Returns:
        Z factor
    """
    # Simplified implementation - in practice would use full Standing-Katz chart
    # This is a placeholder that returns a reasonable range
    P_pr = P_psia / (677 + 15 * gas_sg)  # Pseudo-critical pressure
    T_pr = (T_F + 459.67) / (168 + 325 * gas_sg)  # Pseudo-critical temperature
    
    # Simple correlation for Z factor
    z = 1.0 - 0.27 * P_pr / T_pr
    return np.clip(z, 0.5, 1.2)  # Reasonable bounds


def compute_onshore_state(
    GL: Optional[float],
    topdepth: Optional[float],
    basedepth: Optional[float],
    avgmudline: Optional[float],
    GT_grad: float,
    a_surftemp: float,
) -> Tuple[float, float]:
    """Compute (T[K], P[MPa]) for onshore scenario.
    
    Option a (avgmudline provided):
      P[MPa] = 9.81 * avgmudline * 1000 / 1e6
      T[K]   = (avgmudline/1000) * GT_grad + a_surftemp
    Option b (from depths):
      mean_depth_msl = GL + topdepth + (basedepth - topdepth)/2
      P[MPa] = 9.81 * mean_depth_msl * 1000 / 1e6
      T[K]   = (mean_depth_msl/1000) * GT_grad + a_surftemp
    """
    if avgmudline is not None:
        pressure_mpa = 9.81 * avgmudline * 1000.0 / 1_000_000.0
        temperature_k = (avgmudline / 1000.0) * GT_grad + a_surftemp
        return float(temperature_k), float(pressure_mpa)
    
    # Option b
    if GL is None or topdepth is None or basedepth is None:
        raise ValueError("Onshore: need either avgmudline or GL+topdepth+basedepth")
    mean_depth_msl = float(GL + topdepth + (basedepth - topdepth) / 2.0)
    pressure_mpa = 9.81 * mean_depth_msl * 1000.0 / 1_000_000.0
    temperature_k = (mean_depth_msl / 1000.0) * GT_grad + a_surftemp
    return float(temperature_k), float(pressure_mpa)


def compute_offshore_state(
    waterdepth: Optional[float],
    topdepth: Optional[float],
    basedepth: Optional[float],
    avgmudline: Optional[float],
    GT_grad: float,
    a_seabtemp: float,
) -> Tuple[float, float]:
    """Compute (T[K], P[MPa]) for offshore scenario.
    
    Option a (avgmudline provided):
      P[MPa] = 9.81 * (avgmudline + waterdepth) * 1000 / 1e6
      T[K]   = (avgmudline/1000) * GT_grad + a_seabtemp
    Option b (from depths):
      mean_depth_msl = topdepth + (basedepth - topdepth)/2
      P[MPa] = 9.81 * mean_depth_msl * 1000 / 1e6
      T[K]   = (mean_depth_msl/1000) * GT_grad + a_seabtemp
    """
    # Option a
    if avgmudline is not None:
        if waterdepth is None:
            raise ValueError("Offshore: waterdepth is required when avgmudline is provided")
        pressure_mpa = 9.81 * (avgmudline + waterdepth) * 1000.0 / 1_000_000.0
        temperature_k = (avgmudline / 1000.0) * GT_grad + a_seabtemp
        return float(temperature_k), float(pressure_mpa)
    
    # Option b
    if topdepth is None or basedepth is None:
        raise ValueError("Offshore: need either avgmudline+waterdepth or topdepth+basedepth")
    mean_depth_msl = float(topdepth + (basedepth - topdepth) / 2.0)
    pressure_mpa = 9.81 * mean_depth_msl * 1000.0 / 1_000_000.0
    temperature_k = (mean_depth_msl / 1000.0) * GT_grad + a_seabtemp
    return float(temperature_k), float(pressure_mpa)
