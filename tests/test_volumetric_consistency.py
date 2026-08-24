"""
Regression tests for the volumetric unit and GRV-split bugs.

Each test here corresponds to a defect that shipped to production; the comment
above each one states what the wrong behaviour was, so a future change that
reintroduces it fails loudly.
"""
import numpy as np
import pytest

from scopehc.compute import compute_results, resolve_f_oil, resolve_grv_split
from scopehc.config import RB_PER_M3


# --------------------------------------------------------------------------
# Unit semantics of the in-place outputs
# --------------------------------------------------------------------------

def _base_case(n=1, **overrides):
    """A single-trial case with hand-checkable numbers."""
    kwargs = dict(
        GRV_m3=np.full(n, 500e6),
        NtG=np.full(n, 0.75),
        Por=np.full(n, 0.20),
        RF_oil=np.full(n, 0.35),
        RF_gas=np.full(n, 0.70),
        Bg_rb_per_scf=np.full(n, 0.0055),
        InvBo_STB_per_rb=np.full(n, 1 / 1.35),
        GOR_scf_per_STB=np.full(n, 800.0),
        GRV_oil_m3=np.full(n, 500e6 * 0.6),
        GRV_gas_m3=np.full(n, 500e6 * 0.4),
        Shc_oil=np.full(n, 0.80),
        Shc_gas=np.full(n, 0.80),
    )
    kwargs.update(overrides)
    return compute_results(**kwargs)


def test_stoiip_is_stock_tank_barrels_not_cubic_metres():
    """
    The key used to be called V_oil_insitu_m3 while holding STB. Callers then
    applied RB_PER_M3 and 1/Bo a second time, inflating exports ~4.7x.
    """
    res = _base_case()
    pv_oil_hc = 500e6 * 0.6 * 0.75 * 0.20 * 0.80
    expected_stb = pv_oil_hc * RB_PER_M3 * (1 / 1.35)

    assert res["STOIIP_STB"][0] == pytest.approx(expected_stb, rel=1e-9)
    # Guard against the old misleading names coming back.
    assert "V_oil_insitu_m3" not in res
    assert "V_gas_insitu_m3" not in res


def test_giip_is_standard_cubic_feet():
    res = _base_case()
    pv_gas_hc = 500e6 * 0.4 * 0.75 * 0.20 * 0.80
    expected_scf = pv_gas_hc * RB_PER_M3 / 0.0055

    assert res["GIIP_scf"][0] == pytest.approx(expected_scf, rel=1e-9)


def test_recoverable_is_in_place_times_recovery_factor():
    """Recoverable and in-place must be mutually consistent, not independently derived."""
    res = _base_case()
    assert res["Oil_STB_rec"][0] == pytest.approx(res["STOIIP_STB"][0] * 0.35, rel=1e-9)
    assert res["Gas_free_scf_rec"][0] == pytest.approx(res["GIIP_scf"][0] * 0.70, rel=1e-9)


# --------------------------------------------------------------------------
# Associated-gas recovery factor
# --------------------------------------------------------------------------

def test_rf_assoc_is_honoured_when_supplied():
    """
    results.py omitted RF_assoc entirely (implicit 1.0) while other paths passed
    RF_oil or the real value - three different answers for the same number.
    """
    n = 1
    res_with = _base_case(n, RF_assoc=np.full(n, 0.55))
    res_without = _base_case(n)

    assert res_with["Gas_assoc_scf_rec"][0] == pytest.approx(
        res_without["Gas_assoc_scf_rec"][0] * 0.55, rel=1e-9
    )


def test_assoc_gas_defaults_to_no_extra_reduction():
    res = _base_case()
    expected = res["Oil_STB_rec"][0] * 800.0
    assert res["Gas_assoc_scf_rec"][0] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# Condensate comes from free gas only
# --------------------------------------------------------------------------

def test_condensate_derives_from_free_gas_not_associated_gas():
    n = 1
    res = _base_case(n, CY_STB_per_MMscf=np.full(n, 45.0), RF_cond=np.full(n, 0.6))
    expected = res["Gas_free_scf_rec"][0] * (45.0 / 1e6) * 0.6
    assert res["Cond_STB_rec"][0] == pytest.approx(expected, rel=1e-9)


def test_no_condensate_without_gas():
    n = 1
    res = _base_case(
        n,
        GRV_oil_m3=np.full(n, 500e6),
        GRV_gas_m3=np.zeros(n),
        CY_STB_per_MMscf=np.full(n, 45.0),
        RF_cond=np.full(n, 0.6),
    )
    assert res["Gas_free_scf_rec"][0] == 0.0
    assert res["Cond_STB_rec"][0] == 0.0


# --------------------------------------------------------------------------
# GRV split - fluid_type must win over stale cached arrays
# --------------------------------------------------------------------------

def test_oil_only_ignores_stale_oil_gas_split():
    """
    The Results page only consulted fluid_type when the split arrays were
    missing, so a stale 50/50 split survived a switch to Oil and the page
    reported gas volumes for an oil-only prospect.
    """
    grv = np.full(4, 100.0)
    stale = {"sGRV_oil_m3": np.full(4, 50.0), "sGRV_gas_m3": np.full(4, 50.0)}

    oil, gas = resolve_grv_split("Oil", grv, stale)

    assert np.allclose(oil, 100.0)
    assert np.allclose(gas, 0.0)


def test_gas_only_ignores_stale_oil_gas_split():
    grv = np.full(4, 100.0)
    stale = {"sGRV_oil_m3": np.full(4, 50.0), "sGRV_gas_m3": np.full(4, 50.0)}

    oil, gas = resolve_grv_split("Gas", grv, stale)

    assert np.allclose(oil, 0.0)
    assert np.allclose(gas, 100.0)


def test_oil_gas_uses_explicit_split_when_present():
    """A GOC-derived split must not be overwritten by an f_oil fallback."""
    grv = np.full(3, 100.0)
    ss = {"sGRV_oil_m3": np.full(3, 30.0), "sGRV_gas_m3": np.full(3, 70.0), "f_oil": 0.9}

    oil, gas = resolve_grv_split("Oil + Gas", grv, ss)

    assert np.allclose(oil, 30.0)
    assert np.allclose(gas, 70.0)


def test_oil_gas_falls_back_to_f_oil():
    grv = np.full(3, 100.0)
    oil, gas = resolve_grv_split("Oil + Gas", grv, {"f_oil": 0.6})

    assert np.allclose(oil, 60.0)
    assert np.allclose(gas, 40.0)


def test_split_arrays_of_wrong_length_are_rejected():
    """A stale array from a previous trial count must not silently propagate."""
    grv = np.full(5, 100.0)
    ss = {"sGRV_oil_m3": np.full(2, 30.0), "sGRV_gas_m3": np.full(2, 70.0), "f_oil": 0.5}

    oil, gas = resolve_grv_split("Oil + Gas", grv, ss)

    assert oil.shape == grv.shape
    assert np.allclose(oil, 50.0)


def test_split_always_sums_to_total():
    grv = np.array([10.0, 20.0, 30.0])
    for fluid_type in ("Oil", "Gas", "Oil + Gas"):
        oil, gas = resolve_grv_split(fluid_type, grv, {"f_oil": 0.4})
        assert np.allclose(oil + gas, grv), fluid_type


def test_method_specific_f_oil_takes_precedence():
    assert resolve_f_oil({"direct_f_oil": 0.8, "f_oil": 0.2}, "Direct GRV") == pytest.approx(0.8)
    assert resolve_f_oil({"atgcf_f_oil": 0.3, "f_oil": 0.2}, "Area × Thickness × GCF") == pytest.approx(0.3)
    assert resolve_f_oil({"f_oil": 0.25}, "Direct GRV") == pytest.approx(0.25)
    assert resolve_f_oil({}, "Direct GRV") == pytest.approx(0.5)


def test_f_oil_is_clipped_to_unit_interval():
    assert resolve_f_oil({"f_oil": 1.7}, "Direct GRV") == pytest.approx(1.0)
    assert resolve_f_oil({"f_oil": -0.4}, "Direct GRV") == pytest.approx(0.0)
