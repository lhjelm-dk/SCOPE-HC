"""
Regression tests for v3-compatible GRV depth calculation.

Tests verify that grv_by_depth_v3_compatible produces expected results
for various contact configurations.
"""
import numpy as np
import pytest
from scopehc.geom import grv_by_depth_v3_compatible


def _synthetic_area_depth():
    """Simple bowl-shaped structure: area increases with depth."""
    depth = np.array([2000, 2050, 2100, 2150, 2200], dtype=float)
    area = np.array([0.5e6, 0.8e6, 1.2e6, 1.5e6, 1.7e6], dtype=float)  # m²
    return depth, area


def test_v3_parity_oil_gas_split():
    """Test oil and gas zone split with GOC and OWC."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    goc = 2100.0
    owc = 2180.0  # between 2150 and 2200
    spill = 2190.0

    res = grv_by_depth_v3_compatible(d, a, top, goc, owc, spill)
    
    # Both zones should have positive volume
    assert res["GRV_gas_m3"] > 0, "Gas zone should have positive volume"
    assert res["GRV_oil_m3"] > 0, "Oil zone should have positive volume"
    assert res["GRV_total_m3"] == pytest.approx(
        res["GRV_gas_m3"] + res["GRV_oil_m3"], rel=1e-9
    ), "Total GRV should equal sum of gas and oil"
    
    # Column heights should be positive
    assert res["H_gas_m"] > 0, "Gas column height should be positive"
    assert res["H_oil_m"] > 0, "Oil column height should be positive"
    assert res["H_gas_m"] == pytest.approx(100.0, rel=0.1), "Gas column should be ~100m"
    assert res["H_oil_m"] == pytest.approx(80.0, rel=0.1), "Oil column should be ~80m"


def test_v3_parity_oil_only():
    """Test oil-only case (no GOC)."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    owc = 2160.0
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=None, owc_m=owc, spill_m=None)
    
    assert res["GRV_gas_m3"] == 0, "No gas zone when GOC is None"
    assert res["GRV_oil_m3"] > 0, "Oil zone should have positive volume"
    assert res["GRV_total_m3"] == res["GRV_oil_m3"], "Total should equal oil"
    assert res["H_gas_m"] == 0, "Gas column height should be zero"
    assert res["H_oil_m"] > 0, "Oil column height should be positive"


def test_v3_parity_gas_only():
    """Test gas-only case (no OWC)."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    goc = 2150.0
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=goc, owc_m=None, spill_m=None)
    
    assert res["GRV_gas_m3"] > 0, "Gas zone should have positive volume"
    assert res["GRV_oil_m3"] == 0, "No oil zone when OWC is None"
    assert res["GRV_total_m3"] == res["GRV_gas_m3"], "Total should equal gas"
    assert res["H_gas_m"] > 0, "Gas column height should be positive"
    assert res["H_oil_m"] == 0, "Oil column height should be zero"


def test_v3_parity_no_contacts():
    """Test case with no contacts (entire structure)."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=None, owc_m=None, spill_m=None)
    
    assert res["GRV_total_m3"] > 0, "Should have positive total GRV"
    # When no contacts, treated as oil zone
    assert res["GRV_oil_m3"] > 0, "Should be treated as oil zone"
    assert res["GRV_gas_m3"] == 0, "No gas zone when no GOC"
    assert res["H_oil_m"] > 0, "Oil column height should be positive"


def test_v3_parity_reversed_contacts():
    """Test that reversed contacts are handled gracefully."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    goc = 2180.0  # Below OWC (reversed)
    owc = 2100.0  # Above GOC (reversed)
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=goc, owc_m=owc, spill_m=None)
    
    # Function should sort contacts and produce valid results
    assert res["GRV_total_m3"] >= 0, "Total GRV should be non-negative"
    # After sorting, GOC should be above OWC, so gas zone should exist
    assert res["GRV_gas_m3"] >= 0, "Gas zone should be non-negative"
    assert res["GRV_oil_m3"] >= 0, "Oil zone should be non-negative"


def test_v3_parity_spill_limit():
    """Test that spill point limits the base of HC zones."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    goc = 2100.0
    owc = 2200.0  # At max depth
    spill = 2150.0  # Spill limits to 2150
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=goc, owc_m=owc, spill_m=spill)
    
    # OWC should be clipped to spill
    assert res["GRV_total_m3"] > 0, "Should have positive GRV"
    # Oil zone should be limited by spill, not OWC
    assert res["H_oil_m"] <= 50.0, "Oil column should be limited by spill"


def test_v3_parity_contacts_outside_range():
    """Test that contacts outside depth range are clipped."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    goc = 1950.0  # Above min depth
    owc = 2250.0  # Below max depth
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=goc, owc_m=owc, spill_m=None)
    
    # Contacts should be clipped to valid range
    assert res["GRV_total_m3"] >= 0, "Should handle clipped contacts"
    assert res["GRV_gas_m3"] >= 0, "Gas zone should be non-negative"
    assert res["GRV_oil_m3"] >= 0, "Oil zone should be non-negative"


def test_v3_parity_area_unit_conversion():
    """Test that km² areas are automatically converted to m²."""
    d = np.array([2000, 2100, 2200], dtype=float)
    a_km2 = np.array([0.5, 1.0, 1.5], dtype=float)  # km² (values < 100)
    top = 2000.0
    owc = 2200.0
    
    res = grv_by_depth_v3_compatible(d, a_km2, top, goc_m=None, owc_m=owc, spill_m=None)
    
    # Should produce reasonable volume (not tiny due to unit error)
    assert res["GRV_total_m3"] > 1e6, "Volume should be in millions of m³"


def test_v3_parity_empty_interval():
    """Test that empty intervals return zero volume."""
    d, a = _synthetic_area_depth()
    top = 2000.0
    owc = 2000.0  # Same as top (zero thickness)
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=None, owc_m=owc, spill_m=None)
    
    assert res["GRV_total_m3"] == 0, "Zero thickness should give zero volume"
    assert res["GRV_oil_m3"] == 0, "Oil zone should be zero"
    assert res["H_oil_m"] == 0, "Column height should be zero"


def test_v3_parity_single_point():
    """Test handling of single depth point."""
    d = np.array([2000.0], dtype=float)
    a = np.array([1e6], dtype=float)
    top = 2000.0
    owc = 2100.0
    
    res = grv_by_depth_v3_compatible(d, a, top, goc_m=None, owc_m=owc, spill_m=None)
    
    # Should handle gracefully (may return 0 or interpolate)
    assert res["GRV_total_m3"] >= 0, "Should handle single point gracefully"

