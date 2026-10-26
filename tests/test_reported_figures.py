"""Characterization tests for the figures SCOPE-HC actually reports.

PURPOSE
=======
Every defect found in this codebase over the last two months had the same shape:
the displayed figure was wrong while the app looked perfectly healthy. Nothing
crashed, and the existing suite -- which is good where it reaches -- did not
notice, because it tests the arithmetic that produces an array, not the layer
between a correct array and what the screen says about it.

    - In-place volumes unit-converted twice: oil 4.7x high, gas 1,144x high (PR #2)
    - An oil-only case reporting 706,034 Bscf of gas instead of 0 (PR #2)
    - RF_assoc discarded on one of three code paths: three answers (PR #2)
    - Tornado sensitivities for gas 5.61x out of proportion (PR #2)
    - BOE figures labelled MBOE (thousands) on values in millions (PR #5)
    - GRV depth-area bands labelled P10/P90 the wrong way round (PR #8)

This module fixes two input scenarios and a seed, drives the real Results and
Downloads pages through ``streamlit.testing.v1.AppTest``, and pins the numbers
and the unit strings that reach the screen and the exports. A label is as much a
reported figure as a number: two of the six defects above were labels.

SCENARIOS
=========
1. ``oil_only`` -- fluid type "Oil", Direct GRV. The simplest path, and the one
   where every gas figure must be exactly zero.
2. ``oil_gas`` -- fluid type "Oil + Gas", "Depth-based: Top and Base res. +
   Contact(s)". Exercises the GRV split, the sampled HC contact, a derived GOC,
   associated gas and condensate.

All inputs are PERT and mutually independent -- no correlation matrix -- which is
what makes the hand derivations below exact in expectation, since E[XY] = E[X]E[Y]
for independent X and Y. A PERT(min, mode, max) with the app's lambda = 4 has
mean (min + 4*mode + max)/6: in ``scopehc.sampling.sample_pert``, alpha + beta is
always 6, so the beta mean is alpha/6.

HAND DERIVATION 1 -- oil_only mean STOIIP
=========================================
    STOIIP [STB] = GRV [m3] x NtG x Por x Shc x RB_PER_M3 [rb/m3] x (1/Bo) [STB/rb]

Input means, each (min + 4*mode + max)/6:

    GRV    PERT(300e6, 450e6, 700e6)  ->  466,666,666.7 m3
    NtG    PERT(0.60, 0.75, 0.85)     ->  0.741666667
    Por    PERT(0.15, 0.20, 0.26)     ->  0.201666667
    Shc    PERT(0.70, 0.80, 0.88)     ->  0.796666667
    1/Bo   PERT(0.69, 0.74, 0.80)     ->  0.741666667
    RB_PER_M3 (scopehc.config)        ->  6.28981077

    466,666,666.7 x 0.741666667  =  346,111,111 m3   bulk rock
                  x 0.201666667  =   69,799,074 m3   net pore
                  x 0.796666667  =   55,606,596 m3   hydrocarbon pore
                  x 6.28981077   =  349,754,964      reservoir barrels
                  x 0.741666667  =  259,401,599      STB
                                 =  259.40 MMSTB

The Results page reports 261.44 MMSTB -- 0.786% above the analytic value -- and
that gap reconciles as Monte Carlo noise in the input means at 4,000 trials: the
drawn GRV mean is 468,327,429 (+0.356% on 466,666,667) and the drawn porosity
mean is 0.202506 (+0.416% on 0.201667), while NtG, Shc and 1/Bo are each within
0.04%. 1.00356 x 1.00416 = 1.00774, against the 1.00786 observed. Agreed.
See ``test_oil_only_mean_stoiip_matches_hand_volumetric_equation``.

HAND DERIVATION 2 -- oil_gas GRV split
======================================
The depth-area table is a straight ramp: 0 km2 at 2000 m growing linearly to
8 km2 at 2400 m, so A(z) = 20,000 (z - 2000) m2 and the trapezoid integrator is
exact on it. With h = HCWC - 2000 (the full hydrocarbon column) and f the oil
fraction of that column, so that GOC = HCWC - f*h:

    GRV_total = integral(2000 -> HCWC) = 10,000 h^2
    GRV_gas   = integral(2000 -> GOC)  = 10,000 h^2 (1 - f)^2
    GRV_oil   = integral(GOC  -> HCWC) = 10,000 h^2 (1 - (1 - f)^2)

``test_oil_gas_grv_split_matches_hand_integrated_ramp`` checks all three per
trial against ``scopehc.geom.grv_by_depth_v3_compatible``, which is the function
the GRV page itself calls. They agree to 1e-9 relative (measured worst case
3.5e-15), so the split the rest of this scenario rests on is hand-verified
rather than merely recorded. Mean GRV over the 4,000 trials, mean h = 298.324 m
and mean f = 0.399556:

    total 892,795,974 m3  |  oil 567,881,443 m3  |  gas 324,914,531 m3

HAND DERIVATION 3 -- oil_gas mean STOIIP and mean GIIP
======================================================
The same volumetric equation, with the hand-derived oil GRV above in place of a
PERT mean. GRV_oil is a nonlinear function of the sampled h and f, so its mean
is taken from derivation 2 rather than from the means of h and f; NtG, Por, Shc
and 1/Bo are independent of it, so the product of means is still right.

    567,881,443 x 0.741666667  =  421,178,737 m3   bulk rock
                x 0.201666667  =   84,937,711 m3   net pore
                x 0.796666667  =   67,666,386 m3   hydrocarbon pore
                x 6.28981077   =  425,609,120      reservoir barrels
                x 0.741666667  =  315,660,097      STB
                               =  315.66 MMSTB

The page reports 314.81 MMSTB, 0.27% below that -- again Monte Carlo noise in
the drawn means (this scenario consumes the RNG stream in a different order, so
its NtG/Por/Shc/InvBo draws are not scenario 1's). Substituting this scenario's
own drawn means -- NtG 0.740854, Por 0.201515, Shc 0.796353, 1/Bo 0.741602 --
gives 314.93 MMSTB, 0.04% from the reported figure. Agreed.

GIIP is the same chain with 1/Bg in place of RB_PER_M3 x 1/Bo:

    GIIP [scf] = GRV_gas x NtG x Por x Shc x RB_PER_M3 / Bg

Note that E[1/Bg] = 180.325 is not 1/E[Bg] = 179.104, so the derivation uses the
mean of the drawn reciprocal:

    324,914,531 x 0.740854 x 0.201515 x 0.796353 x 6.28981077 x 180.325
                               =  43.813 Bscf

against the 43.81 Bscf reported. Agreed to 0.008%.

TOLERANCES
==========
``MC_TOL`` = 6% relative, with an absolute floor of half the display's last
printed digit (a value shown to one decimal cannot be pinned tighter than
+/- 0.05).

Why 6% and not exact equality. The seed is fixed, so under an unchanged
``scopehc.sampling`` these figures reproduce bit for bit and exact equality would
"work". It would also make the suite hostage to the order in which upstream code
draws random numbers: adding one sampled input anywhere ahead of these would
shift every stream and redden every assertion here without a single reported
figure being wrong. Measured over 12 seeds at 4,000 trials, the P10/P50/P90 of
these distributions move by at most 3.7% between seeds, so 6% is about 1.6x that
noise floor -- wide enough to survive a stream shift, and still 5x to 200x
tighter than the smallest defect in the list at the top of this file. An
exact-equality assertion on a Monte Carlo percentile is the wrong tool.

Exactness is reserved for the things that are exact, and those are asserted at
1e-9 or as string equality:

    - every gas figure in the oil-only case is identically zero;
    - the unit strings and chart titles;
    - the column set of the exported trial table;
    - the P10 <-> P90 swap between the two percentile conventions;
    - the per-trial GRV split of derivation 2.

WHAT THIS SUITE DOES AND DOES NOT CATCH
=======================================
Covered, as displayed: the per-figure summary table under each histogram on the
Results page (STOIIP, GIIP, recoverable oil, free gas, associated gas,
condensate, THR); the unit string in each of those histograms' axis labels; the
"Recoverable (Surface) Volumes" and THR tables including their Unit column; the
THR composition table and its header; the exported trial table's column set and
the THR summary on the Downloads page; both percentile conventions; both unit
systems.

NOT covered. The histogram and CDF traces themselves are not read, only the axis
titles -- a figure drawn from the wrong array under a correct label would pass.
The Sensitivity page is not driven at all. Formatting detail is not pinned:
cells are parsed to float, so a change from one decimal to two, or the loss of
thousands separators, goes unnoticed. That is deliberate -- ``summary_table``
formats with ``f"{x:.1f}".rstrip('0').rstrip('.')``, which prints 2.04 as "2",
and pinning that is pinning a quirk. Scenario state is injected into session
state rather than typed into the input pages, so the input widgets, the GRV page
and the sampling-to-session-state wiring are not exercised; what is exercised is
everything downstream of ``sGRV_m3_final`` and its siblings, which is where all
six defects lived.

KNOWN DEFECTS NOT PINNED
========================
The two THR composition tables on the Results page are wrong under the metric
unit system. They are recorded here as ``xfail(strict=True)`` rather than frozen
into expectations: see ``test_metric_thr_composition_is_in_metric_boe`` and
``test_metric_thr_breakdown_is_in_metric_boe`` for the numbers and the cause.
Oilfield is correct in both, which is why they have survived. Fixing them turns
those two xfails into passes, which under strict xfail fails the suite --
remove the markers at that point.

RUNTIME
=======
Eight page renders -- six Results (two scenarios x two unit systems, plus both
scenarios again under the non-exceedance convention) and two Downloads -- shared
across every test via ``_RENDER_CACHE``. That is about 14 s, taking ``pytest
tests/ -q`` from 2.3 s to 17 s on this machine.

A Streamlit render costs roughly the same at 500 trials as at 8,000, so the
trial count is not what the runtime buys; the render count is. 4,000 is simply
the smallest count at which the pinned percentiles sit inside ``MC_TOL`` with
margin (8,000 halves the noise to 2.5%, for no assertion this module makes).
Dropping a render would mean dropping either a unit system or a percentile
convention, both of which are coverage this suite exists for -- and the metric
renders are what turned up the two defects recorded at the bottom of this file.

DOES IT ACTUALLY BITE
=====================
Checked by reintroducing four of the historical defects one at a time, each
reverted afterwards. Failures, out of 39 tests:

    relabel MMBOE as MBOE throughout results.py          7 failures
    drop RF_assoc from the results page's compute call   2 failures
    convert STOIIP twice (RB_PER_M3 and 1/Bo again)      2 failures
    let a stale 50/50 GRV split survive fluid_type       13 failures
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from streamlit.testing.v1 import AppTest

from scopehc.config import M3_PER_BBL, M3_PER_SCF, RB_PER_M3
from scopehc.geom import derive_goc_from_mode, grv_by_depth_v3_compatible
from scopehc.sampling import rng_from_seed, sample_pert

# --------------------------------------------------------------------------
# Scenario definition
# --------------------------------------------------------------------------

SEED = 20261026
N_TRIALS = 4000

#: 6% relative; see TOLERANCES in the module docstring.
MC_TOL = 0.06

#: Tolerance for the hand derivations, which compare a displayed mean against
#: the volumetric equation evaluated on means rather than per trial. Observed
#: agreement is 0.04% to 0.79%; 2% leaves room without admitting a real error.
HAND_TOL = 0.02

#: PERT (min, mode, max) for every sampled input. Mean is (min + 4*mode + max)/6.
PERT = {
    "GRV": (300e6, 450e6, 700e6),
    "NtG": (0.60, 0.75, 0.85),
    "Por": (0.15, 0.20, 0.26),
    "Shc": (0.70, 0.80, 0.88),
    "InvBo": (0.69, 0.74, 0.80),
    "RF_oil": (0.25, 0.35, 0.45),
    "RF_gas": (0.60, 0.70, 0.80),
    "RF_assoc": (0.80, 0.90, 0.95),
    "GOR": (600.0, 800.0, 1000.0),
    "Bg": (0.0045, 0.0055, 0.0070),
    "CY": (80.0, 120.0, 160.0),
    "RF_cond": (0.40, 0.55, 0.65),
    "HCWC": (2250.0, 2300.0, 2340.0),
    "f_oil": (0.25, 0.40, 0.55),
}

# A straight area ramp, so the trapezoid integrator is exact on it and the GRV
# split is hand-checkable. See HAND DERIVATION 2.
AREA_TABLE_DEPTH_M = np.array([2000.0, 2100.0, 2200.0, 2300.0, 2400.0])
AREA_TABLE_KM2 = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
TOP_STRUCTURE_M = 2000.0
SPILL_DEPTH_M = 2400.0
#: d(area)/d(depth) below the top, in m2 per m: 8 km2 over 400 m.
AREA_SLOPE_M2_PER_M = 8.0e6 / 400.0

GAS_SCF_PER_BOE = 6000.0


def pert_mean(key: str) -> float:
    """Analytic mean of a PERT input, (min + 4*mode + max)/6."""
    low, mode, high = PERT[key]
    return (low + 4.0 * mode + high) / 6.0


def _draw(rng, key: str, n: int) -> np.ndarray:
    low, mode, high = PERT[key]
    return sample_pert(rng, low, mode, high, n)


def _oil_only_state(n: int, unit_system: str, exceedance: bool) -> dict:
    """Session state for scenario 1: oil only, Direct GRV."""
    rng = rng_from_seed(SEED)
    shc = _draw(rng, "Shc", n)
    return {
        "fluid_type": "Oil",
        "grv_option": "Direct GRV",
        "unit_system": unit_system,
        "percentile_exceedance": exceedance,
        "gas_scf_per_boe": GAS_SCF_PER_BOE,
        "num_sims": n,
        "random_seed": SEED,
        "sat_mode": "Global",
        "sGRV_m3_final": _draw(rng, "GRV", n),
        "sNtG": _draw(rng, "NtG", n),
        "sp": _draw(rng, "Por", n),
        "sInvBo": _draw(rng, "InvBo", n),
        "sRF_oil": _draw(rng, "RF_oil", n),
        "sRF_gas": _draw(rng, "RF_gas", n),
        "sRF_assoc_gas": _draw(rng, "RF_assoc", n),
        "sGOR": _draw(rng, "GOR", n),
        # Present but unused: there is no gas zone. Deliberately a plausible
        # value rather than zero, so a gas figure leaking into this scenario
        # shows up as a number rather than as another zero.
        "sBg": np.full(n, 0.0055),
        # Condensate yield is left unset, as it is for a user who never visits
        # the Fluids page: condensate must then be exactly zero.
        "Shc_global": shc,
        "Shc_oil": shc,
        "Shc_gas": shc,
    }


def _oil_gas_state(n: int, unit_system: str, exceedance: bool) -> dict:
    """Session state for scenario 2: oil + gas, depth-based GRV with a GOC.

    Reproduces what the GRV page's depth-based branch leaves in session state
    (``inputs_grv.py``, the per-trial loop around line 1924) by calling the same
    two functions it calls, rather than driving that 3,200-line page.
    """
    rng = rng_from_seed(SEED)
    shc = _draw(rng, "Shc", n)
    hcwc = _draw(rng, "HCWC", n)
    f_oil = _draw(rng, "f_oil", n)

    grv_oil = np.zeros(n)
    grv_gas = np.zeros(n)
    h_oil = np.zeros(n)
    h_gas = np.zeros(n)
    goc = np.zeros(n)
    goc_inputs = {"da_oil_frac": f_oil}
    for i in range(n):
        goc[i] = derive_goc_from_mode(
            TOP_STRUCTURE_M,
            float(hcwc[i]),
            "Oil fraction of HC column",
            goc_inputs,
            trial_idx=i,
        )
        split = grv_by_depth_v3_compatible(
            depth_m=AREA_TABLE_DEPTH_M,
            area_m2=AREA_TABLE_KM2,
            top_structure_m=TOP_STRUCTURE_M,
            goc_m=goc[i],
            owc_m=float(hcwc[i]),
            spill_m=SPILL_DEPTH_M,
        )
        grv_oil[i] = split["GRV_oil_m3"]
        grv_gas[i] = split["GRV_gas_m3"]
        h_oil[i] = split["H_oil_m"]
        h_gas[i] = split["H_gas_m"]

    return {
        "fluid_type": "Oil + Gas",
        "grv_option": "Depth-based: Top and Base res. + Contact(s)",
        "unit_system": unit_system,
        "percentile_exceedance": exceedance,
        "gas_scf_per_boe": GAS_SCF_PER_BOE,
        "num_sims": n,
        "random_seed": SEED,
        "sat_mode": "Global",
        "da_goc_mode": "Oil fraction of HC column",
        "da_oil_frac": f_oil,
        "sGRV_m3_final": grv_oil + grv_gas,
        "sGRV_oil_m3": grv_oil,
        "sGRV_gas_m3": grv_gas,
        "sOil_Column_Height": h_oil,
        "sGas_Column_Height": h_gas,
        "sGOC_depth": goc,
        "sD_hc": hcwc,
        "sD_spill": np.full(n, SPILL_DEPTH_M),
        "sNtG": _draw(rng, "NtG", n),
        "sp": _draw(rng, "Por", n),
        "sInvBo": _draw(rng, "InvBo", n),
        "sRF_oil": _draw(rng, "RF_oil", n),
        "sRF_gas": _draw(rng, "RF_gas", n),
        "sRF_assoc_gas": _draw(rng, "RF_assoc", n),
        "sGOR": _draw(rng, "GOR", n),
        "sBg": _draw(rng, "Bg", n),
        "sCY": _draw(rng, "CY", n),
        "sRF_cond": _draw(rng, "RF_cond", n),
        "Shc_global": shc,
        "Shc_oil": shc,
        "Shc_gas": shc,
        "_grv_arrays_depth_based": True,
        "_grv_arrays_method": "Depth-based: Top and Base res. + Contact(s)",
    }


SCENARIOS = {"oil_only": _oil_only_state, "oil_gas": _oil_gas_state}


# --------------------------------------------------------------------------
# Driving the real pages
# --------------------------------------------------------------------------
#
# AppTest.from_function sends the function's SOURCE to the script runner, so
# these two cannot close over anything in this module. Scenario state goes in
# through at.session_state before the run instead; AppTest runs in-process, so
# numpy arrays pass through unchanged.


def _results_page() -> None:
    from scopehc.ui import results

    results.render()


def _downloads_page() -> None:
    from scopehc.ui import downloads

    downloads.render()


_RENDER_CACHE: dict = {}


def _render(page: str, scenario: str, unit_system: str, exceedance: bool) -> AppTest:
    """Run a page once per (page, scenario, units, convention) and reuse it."""
    key = (page, scenario, unit_system, exceedance)
    if key not in _RENDER_CACHE:
        page_fn = {"results": _results_page, "downloads": _downloads_page}[page]
        state = SCENARIOS[scenario](N_TRIALS, unit_system, exceedance)
        at = AppTest.from_function(page_fn, default_timeout=300)
        for name, value in state.items():
            at.session_state[name] = value
        at.run()
        assert not at.exception, [exc.value for exc in at.exception]
        _RENDER_CACHE[key] = at
    return _RENDER_CACHE[key]


def results_page(
    scenario: str, unit_system: str = "oilfield", exceedance: bool = True
) -> AppTest:
    return _render("results", scenario, unit_system, exceedance)


def downloads_page(
    scenario: str, unit_system: str = "oilfield", exceedance: bool = True
) -> AppTest:
    return _render("downloads", scenario, unit_system, exceedance)


# --------------------------------------------------------------------------
# Reading what was displayed
# --------------------------------------------------------------------------


def _ordered_elements(block, out=None):
    """Flatten the element tree in document order.

    ``.children`` is a dict and ``element.type`` is None for leaves, so the kind
    of an element has to come from its class name.
    """
    if out is None:
        out = []
    children = block.children
    for child in children.values() if hasattr(children, "values") else children:
        if hasattr(child, "children"):
            _ordered_elements(child, out)
        else:
            out.append(child)
    return out


def _x_axis_title(chart) -> str | None:
    """The chart's x-axis label, which is where the unit string is displayed.

    ``element.value`` raises on a plotly chart in Streamlit 1.62, so this reads
    the serialised figure spec instead.
    """
    layout = json.loads(chart.proto.spec).get("layout", {})
    return ((layout.get("xaxis") or {}).get("title") or {}).get("text")


def _as_float(cell) -> float:
    """Parse a displayed cell. See the formatting note in the module docstring."""
    if isinstance(cell, str):
        return float(cell.replace(",", "").replace("%", "").strip("*"))
    return float(cell)


def figure_panels(at: AppTest) -> dict:
    """Each histogram's axis label mapped to the summary table shown beneath it.

    The key is the full displayed axis title, e.g. "Recoverable Oil (MMSTB)", so
    asserting the key set pins each figure's name and its unit string together
    and in the same breath as the numbers. The value is
    ``{"P10": ..., "P50": ..., "P90": ..., "Mean": ...}`` parsed from the table
    immediately following that chart in document order.
    """
    panels: dict = {}
    pending = None
    for element in _ordered_elements(at.main):
        kind = type(element).__name__
        if kind == "UnknownElement" and getattr(element, "type", None) == "plotly_chart":
            pending = _x_axis_title(element)
        elif kind == "Dataframe":
            frame = element.value
            if pending and list(frame.columns)[:2] == ["Mean", "Min."]:
                row = frame.iloc[0]
                panels[pending] = {
                    stat: _as_float(row[stat]) for stat in ("P10", "P50", "P90", "Mean")
                }
            pending = None
    return panels


def _table_with_column(at: AppTest, column: str, first_value: str):
    """The displayed dataframe that has ``column`` and starts with ``first_value``."""
    for element in at.dataframe:
        frame = element.value
        if column in frame.columns and str(frame[column].iloc[0]) == first_value:
            return frame
    raise AssertionError(
        f"no displayed table with {column!r} starting at {first_value!r}; saw "
        f"{[list(e.value.columns) for e in at.dataframe]}"
    )


def recoverable_table(at: AppTest):
    """The "Recoverable (Surface) Volumes" table, which carries a Unit column."""
    return _table_with_column(at, "Parameter", "Oil")


def thr_table(at: AppTest):
    """The single-row THR table, which carries a Unit column."""
    return _table_with_column(at, "Parameter", "THR")


def composition_table(at: AppTest):
    """The "THR Composition (mean)" table."""
    return _table_with_column(at, "Component", "Oil")


def breakdown_table(at: AppTest):
    """The "THR Breakdown by Fluid Type" table inside the THR expander."""
    return _table_with_column(at, "Fluid Type", "Oil")


def _decimals_for(title: str) -> int:
    """How many decimals the page prints this figure to (results.py)."""
    return 1 if "Condensate" in title else 2


def approx_displayed(expected: float, decimals: int = 2):
    """``MC_TOL`` relative, or half the last displayed digit, whichever is kinder."""
    return pytest.approx(expected, rel=MC_TOL, abs=0.5 * 10.0**-decimals)


# --------------------------------------------------------------------------
# The pinned figures
# --------------------------------------------------------------------------
# Recorded from a run and then reconciled against the hand derivations in the
# module docstring. The key is the full displayed histogram axis title, so these
# tables pin the figure's name, its unit string and its numbers together. Each
# value is (P10, P50, P90, Mean) under the EXCEEDANCE convention, which is the
# app default; under non-exceedance P10 and P90 swap exactly, which
# test_percentile_convention_swaps_p10_and_p90 asserts rather than tabulating
# every number twice.

EXPECTED_PANELS = {
    ("oil_only", "oilfield"): {
        "Oil in Place (STOIIP) (MMSTB)": (333.32, 256.97, 195.57, 261.44),
        "Gas in Place (GIIP) (Bscf)": (0.0, 0.0, 0.0, 0.0),
        "Recoverable Oil (MMSTB)": (120.82, 89.14, 65.63, 91.71),
        "Recoverable Free Gas (Bscf)": (0.0, 0.0, 0.0, 0.0),
        "Recoverable Associated Gas (Bscf)": (88.79, 63.64, 45.39, 65.51),
        "Recoverable Condensate (MMSTB)": (0.0, 0.0, 0.0, 0.0),
        "THR (MMBOE)": (135.21, 99.71, 73.32, 102.63),
    },
    ("oil_only", "si"): {
        "Oil in Place (STOIIP) (Mm³)": (52.99, 40.85, 31.09, 41.57),
        "Gas in Place (GIIP) (Bm³)": (0.0, 0.0, 0.0, 0.0),
        "Recoverable Oil (Mm³)": (19.21, 14.17, 10.43, 14.58),
        "Recoverable Free Gas (Bm³)": (0.0, 0.0, 0.0, 0.0),
        "Recoverable Associated Gas (Bm³)": (2.51, 1.80, 1.29, 1.85),
        "Recoverable Condensate (Mm³)": (0.0, 0.0, 0.0, 0.0),
        "THR (Mm³ BOE)": (21.50, 15.85, 11.66, 16.32),
    },
    ("oil_gas", "oilfield"): {
        "Oil in Place (STOIIP) (MMSTB)": (400.09, 310.19, 238.27, 314.81),
        "Gas in Place (GIIP) (Bscf)": (60.40, 42.57, 29.39, 43.81),
        "Recoverable Oil (MMSTB)": (143.93, 107.99, 79.50, 110.35),
        "Recoverable Free Gas (Bscf)": (42.20, 29.68, 20.54, 30.62),
        "Recoverable Associated Gas (Bscf)": (104.65, 76.65, 54.93, 78.80),
        "Recoverable Condensate (MMSTB)": (2.9, 1.9, 1.2, 2.0),
        "THR (MMBOE)": (169.05, 127.91, 95.99, 130.57),
    },
    ("oil_gas", "si"): {
        "Oil in Place (STOIIP) (Mm³)": (63.61, 49.32, 37.88, 50.05),
        "Gas in Place (GIIP) (Bm³)": (1.71, 1.21, 0.83, 1.24),
        "Recoverable Oil (Mm³)": (22.88, 17.17, 12.64, 17.54),
        "Recoverable Free Gas (Bm³)": (1.19, 0.84, 0.58, 0.87),
        "Recoverable Associated Gas (Bm³)": (2.96, 2.17, 1.56, 2.23),
        "Recoverable Condensate (Mm³)": (0.5, 0.3, 0.2, 0.3),
        "THR (Mm³ BOE)": (26.88, 20.34, 15.26, 20.76),
    },
}

#: Unit column of the "Recoverable (Surface) Volumes" table, by unit system.
EXPECTED_RECOVERABLE_UNITS = {
    "oilfield": {
        "Oil": "MMSTB",
        "Free Gas": "Bscf",
        "Assoc Gas": "Bscf",
        "Condensate": "MMSTB",
        "Total Liquids (Oil + Condensate)": "MMSTB",
    },
    "si": {
        "Oil": "Mm³",
        "Free Gas": "Bm³",
        "Assoc Gas": "Bm³",
        "Condensate": "Mm³",
        "Total Liquids (Oil + Condensate)": "Mm³",
    },
}

#: Unit column of the THR table, and the header of the THR composition table.
# Every row of these tables is printed to one decimal, so a sum of four of them
# can sit up to 4 x 0.05 = 0.2 away from the separately-rounded total. That is
# the display rounding, not a discrepancy between the figures. A relative
# tolerance alone hides this under oilfield, where the numbers are ~6x larger,
# and fails under metric - which is how it was found.
ROW_SUM_ABS = 0.25

EXPECTED_THR_UNIT = {"oilfield": "MMBOE", "si": "Mm³ BOE"}

#: Columns of the exported trial table. Asserted as a set, not a sequence, so
#: reordering is free but a column silently appearing or vanishing is not.
#: Note that In_place_Oil_STB / In_place_Gas_scf duplicate Oil_in_Place_STB /
#: Gas_in_Place_scf (common.py 3079-3080 and 3123-3124) -- redundant, recorded
#: as-is so that removing them reads as the deliberate export change it is.
_EXPORT_COLUMNS_COMMON = {
    "Trial",
    "GRV_m3",
    "NtG",
    "Porosity",
    "RF_oil",
    "RF_gas",
    "Bg_rb_per_scf",
    "InvBo_STB_per_rb",
    "GOR_scf_per_STB",
    "Pore_Volume_Total_m3",
    "Pore_Volume_Oil_m3",
    "Pore_Volume_Gas_m3",
    "In_place_Oil_STB",
    "In_place_Gas_scf",
    "Oil_in_Place_STB",
    "Oil_in_Place_MMSTB",
    "Oil_in_Place_Mm3",
    "Gas_in_Place_scf",
    "Gas_in_Place_Bscf",
    "Gas_in_Place_Bm3",
    "Recoverable_Oil_STB",
    "Recoverable_Oil_MMSTB",
    "Recoverable_Oil_Mm3",
    "Recoverable_Free_Gas_scf",
    "Recoverable_Free_Gas_Bscf",
    "Recoverable_Free_Gas_Bm3",
    "Recoverable_Associated_Gas_scf",
    "Recoverable_Associated_Gas_Bscf",
    "Recoverable_Associated_Gas_Bm3",
    "Recoverable_Condensate_STB",
    "Recoverable_Condensate_MMSTB",
    "Recoverable_Condensate_Mm3",
    "Recoverable_Total_Gas_scf",
    "Recoverable_Total_Gas_Bscf",
    "Recoverable_Total_Gas_Bm3",
    "Recoverable_Total_Liquids_STB",
    "Recoverable_Total_Liquids_MMSTB",
    "Recoverable_Total_Liquids_Mm3",
    "Total_Hydrocarbon_Resource_BOE",
    "Shc_global",
    "Shc_oil",
    "Shc_gas",
}

EXPECTED_EXPORT_COLUMNS = {
    "oil_only": _EXPORT_COLUMNS_COMMON,
    # The depth-based path additionally tabulates the geometry it sampled.
    "oil_gas": _EXPORT_COLUMNS_COMMON
    | {
        "GRV_oil_m3",
        "GRV_gas_m3",
        "Oil_Column_Height_m",
        "Gas_Column_Height_m",
        "HC_Contact_Depth_m",
        "Spill_Depth_m",
        "GOC_Mode",
        "GOC_Depth_m",
        "f_oil_depth_area",
    },
}

ALL_CASES = [
    pytest.param(scenario, units, id=f"{scenario}-{units}")
    for scenario, units in EXPECTED_PANELS
]


# ==========================================================================
# The reported figures, as displayed
# ==========================================================================


@pytest.mark.parametrize("scenario,units", ALL_CASES)
def test_displayed_figures_and_their_units_are_pinned(scenario, units):
    """Every headline figure on the Results page, with the unit shown beside it.

    The key set is asserted first and exactly: it carries each figure's title
    and its unit string, so a relabelled axis (PR #5's MBOE/MMBOE defect) or a
    section quietly appearing or disappearing fails here before any number is
    compared.
    """
    expected = EXPECTED_PANELS[(scenario, units)]
    panels = figure_panels(results_page(scenario, units))

    assert panels.keys() == expected.keys()

    for title, (p10, p50, p90, mean) in expected.items():
        decimals = _decimals_for(title)
        got = panels[title]
        assert got["P10"] == approx_displayed(p10, decimals), f"{title} P10"
        assert got["P50"] == approx_displayed(p50, decimals), f"{title} P50"
        assert got["P90"] == approx_displayed(p90, decimals), f"{title} P90"
        assert got["Mean"] == approx_displayed(mean, decimals), f"{title} mean"


@pytest.mark.parametrize("scenario,units", ALL_CASES)
def test_recoverable_and_thr_tables_carry_the_right_units(scenario, units):
    """The Unit column of the two summary tables, and the THR composition header.

    Separate from the axis labels above because these are a second, independent
    rendering of the same units: results.py builds them from its own
    ``oil_unit`` / ``gas_unit`` / ``cond_unit`` / ``thr_unit`` locals.
    """
    at = results_page(scenario, units)

    recoverable = recoverable_table(at)
    assert dict(zip(recoverable["Parameter"], recoverable["Unit"])) == (
        EXPECTED_RECOVERABLE_UNITS[units]
    )

    assert thr_table(at)["Unit"].iloc[0] == EXPECTED_THR_UNIT[units]

    # The THR composition table states its unit in the column header rather than
    # in a cell. This is the exact site of the MBOE/MMBOE defect (PR #5).
    assert list(composition_table(at).columns) == [
        "Component",
        f"Mean ({EXPECTED_THR_UNIT[units]})",
        "% of THR",
    ]


@pytest.mark.parametrize("scenario,units", ALL_CASES)
def test_recoverable_table_agrees_with_the_figure_above_it(scenario, units):
    """The two tables that state the same number must state the same number.

    Three of the six defects were one code path disagreeing with another about a
    figure the user sees in both places.
    """
    at = results_page(scenario, units)
    panels = figure_panels(at)
    rows = recoverable_table(at).set_index("Parameter")

    pairs = [
        ("Oil", "Recoverable Oil"),
        ("Free Gas", "Recoverable Free Gas"),
        ("Assoc Gas", "Recoverable Associated Gas"),
        ("Condensate", "Recoverable Condensate"),
    ]
    for row_name, figure_prefix in pairs:
        title = next(t for t in panels if t.startswith(figure_prefix + " ("))
        for stat in ("P10", "P50", "P90", "Mean"):
            # The table prints one decimal, the figure's summary two.
            assert _as_float(rows.loc[row_name, stat]) == pytest.approx(
                panels[title][stat], rel=1e-3, abs=0.05
            ), f"{row_name} {stat}"


# ==========================================================================
# Oil only: gas must be zero, not small
# ==========================================================================


@pytest.mark.parametrize("units", ["oilfield", "si"])
def test_oil_only_displays_zero_for_every_gas_figure(units):
    """A past defect reported 706,034 Bscf of gas for an oil-only prospect.

    This covers what the screen says. It cannot by itself distinguish 0.0 from
    1e-15, because both print as "0" -- see
    test_oil_only_export_is_identically_zero_for_gas for the exact claim.
    """
    at = results_page("oil_only", units)
    panels = figure_panels(at)

    zero_figures = [
        t
        for t in panels
        if t.startswith(("Gas in Place (GIIP)", "Recoverable Free Gas", "Recoverable Condensate"))
    ]
    assert len(zero_figures) == 3
    for title in zero_figures:
        for stat, value in panels[title].items():
            assert value == 0.0, f"{title} {stat} displayed as {value}"

    rows = recoverable_table(at).set_index("Parameter")
    for row_name in ("Free Gas", "Condensate"):
        for stat in ("P10", "P50", "P90", "Mean", "Std"):
            assert _as_float(rows.loc[row_name, stat]) == 0.0, f"{row_name} {stat}"

    # Associated gas, by contrast, must NOT be zero: dissolved gas comes out
    # with the oil. An oil-only case that zeroed this would also be wrong.
    assert panels["Recoverable Associated Gas " + ("(Bscf)" if units == "oilfield" else "(Bm³)")][
        "P50"
    ] > 0.0


def test_oil_only_export_is_identically_zero_for_gas():
    """The hard claim: exactly zero in every trial, in every gas column.

    Read from the frame behind the download buttons (``df_results``, which is
    what ``export_to_csv`` writes) rather than from the 50-row on-screen
    preview, so this covers all 4,000 trials.
    """
    at = downloads_page("oil_only")
    frame = at.session_state["df_results"]
    assert len(frame) == N_TRIALS

    for column in (
        "Gas_in_Place_scf",
        "Gas_in_Place_Bscf",
        "Gas_in_Place_Bm3",
        "In_place_Gas_scf",
        "Recoverable_Free_Gas_scf",
        "Recoverable_Free_Gas_Bscf",
        "Recoverable_Free_Gas_Bm3",
        "Recoverable_Condensate_STB",
        "Pore_Volume_Gas_m3",
    ):
        nonzero = int(np.count_nonzero(frame[column].to_numpy()))
        assert nonzero == 0, f"{column}: {nonzero} of {N_TRIALS} trials non-zero"

    # With no free gas, total gas is associated gas and nothing else.
    assert np.array_equal(
        frame["Recoverable_Total_Gas_scf"].to_numpy(),
        frame["Recoverable_Associated_Gas_scf"].to_numpy(),
    )
    # ... and total liquids is oil and nothing else.
    assert np.array_equal(
        frame["Recoverable_Total_Liquids_STB"].to_numpy(),
        frame["Recoverable_Oil_STB"].to_numpy(),
    )


# ==========================================================================
# Both percentile conventions
# ==========================================================================


@pytest.mark.parametrize("scenario", ["oil_only", "oil_gas"])
def test_percentile_convention_swaps_p10_and_p90(scenario):
    """The sidebar toggle inverts P10 and P90, and the tool is used both ways.

    Asserted exactly, not within ``MC_TOL``: the same array is being summarised
    twice, so this is a swap and not a recomputation. PR #8's defect was a
    "P10" that was really a P90 under the app's own convention.
    """
    exceedance = figure_panels(results_page(scenario, "oilfield", True))
    non_exceedance = figure_panels(results_page(scenario, "oilfield", False))

    assert exceedance.keys() == non_exceedance.keys()
    for title in exceedance:
        high, low = exceedance[title], non_exceedance[title]
        assert low["P10"] == high["P90"], f"{title}: P10 under non-exceedance"
        assert low["P90"] == high["P10"], f"{title}: P90 under non-exceedance"
        assert low["P50"] == high["P50"], f"{title}: median must not move"
        assert low["Mean"] == high["Mean"], f"{title}: mean must not move"

    # Under exceedance, P10 is the optimistic case, so P10 >= P50 >= P90.
    for title, stats in exceedance.items():
        if stats["P50"] == 0.0:
            continue
        assert stats["P10"] >= stats["P50"] >= stats["P90"], title


@pytest.mark.parametrize("exceedance", [True, False])
def test_percentile_convention_is_stated_on_the_page(exceedance):
    """The page says which convention is in force; a reader needs that to act."""
    at = results_page("oil_gas", "oilfield", exceedance)
    banners = " ".join(element.value for element in at.info)
    if exceedance:
        assert "probability of exceedance" in banners
        assert "non-exceedance" not in banners
    else:
        assert "probability of non-exceedance" in banners


# ==========================================================================
# Both unit systems
# ==========================================================================


@pytest.mark.parametrize("scenario", ["oil_only", "oil_gas"])
def test_metric_figures_are_the_oilfield_figures_converted(scenario):
    """Metric is the same volume in other units, not a separate calculation.

    This is the assertion the double-conversion defect (oil 4.7x, gas 1,144x)
    would have failed: the conversion factor has to be applied exactly once.
    """
    oilfield = figure_panels(results_page(scenario, "oilfield"))
    metric = figure_panels(results_page(scenario, "si"))

    # Pair the two by the figure name, which is the axis title minus its unit.
    def by_name(panels):
        return {title.rsplit(" (", 1)[0]: (title, stats) for title, stats in panels.items()}

    oilfield_by_name, metric_by_name = by_name(oilfield), by_name(metric)
    assert oilfield_by_name.keys() == metric_by_name.keys()

    for name, (oilfield_title, oilfield_stats) in oilfield_by_name.items():
        metric_title, metric_stats = metric_by_name[name]
        # Liquids and BOE go barrels -> m3; gas goes scf -> m3.
        factor = M3_PER_SCF if oilfield_title.endswith("(Bscf)") else M3_PER_BBL
        # Both sides are read off the screen already rounded, so the floor is
        # half the metric display's last digit plus the oilfield display's own
        # half-digit carried through the conversion. For associated gas that is
        # 0.005 + 0.005 x 0.0283 = 0.00514, and the observed gap is 0.00503:
        # without the second term this assertion sits exactly on the boundary.
        tolerance = 0.5 * 10.0 ** -_decimals_for(metric_title) + (
            0.5 * 10.0 ** -_decimals_for(oilfield_title) * factor
        )
        for stat in ("P10", "P50", "P90", "Mean"):
            assert metric_stats[stat] == pytest.approx(
                oilfield_stats[stat] * factor, rel=1e-3, abs=tolerance
            ), f"{name} {stat}"


# ==========================================================================
# Internal consistency of what is displayed
# ==========================================================================


@pytest.mark.parametrize("scenario,units", ALL_CASES)
def test_thr_composition_sums_to_the_reported_thr(scenario, units):
    """THR = oil + condensate + free gas + associated gas, as printed.

    Skipped for metric, where the component rows are a known defect; see
    test_metric_thr_composition_is_in_metric_boe.
    """
    table = composition_table(results_page(scenario, units))
    values = {row["Component"]: _as_float(row[f"Mean ({EXPECTED_THR_UNIT[units]})"])
              for _, row in table.iterrows()}
    percentages = {row["Component"]: _as_float(row["% of THR"])
                   for _, row in table.iterrows()}

    components = ["Oil", "Condensate", "Free Gas", "Associated Gas"]
    total = values["**Total**"]
    assert sum(values[c] for c in components) == pytest.approx(total, rel=1e-3, abs=ROW_SUM_ABS)
    assert sum(percentages[c] for c in components) == pytest.approx(100.0, abs=0.3)

    # ... and the total is the THR figure reported everywhere else on the page.
    assert total == approx_displayed(
        EXPECTED_PANELS[(scenario, units)][f"THR ({EXPECTED_THR_UNIT[units]})"][3], 1
    )


def test_downloads_thr_summary_agrees_with_the_results_page():
    """The export's THR column and the Results page must not disagree.

    ``Total_Hydrocarbon_Resource_BOE`` carries raw BOE -- no M or MM prefix --
    so the Downloads page's summary is 1e6 times the Results page's MMBOE.
    """
    for scenario in ("oil_only", "oil_gas"):
        at = downloads_page(scenario)
        summaries = [
            element.value
            for element in at.dataframe
            if list(element.value.columns)[:2] == ["Mean", "Min."]
        ]
        assert len(summaries) == 1, "expected exactly one THR summary table"
        exported_mean = _as_float(summaries[0].iloc[0]["Mean"])

        results_mean = EXPECTED_PANELS[(scenario, "oilfield")]["THR (MMBOE)"][3]
        assert exported_mean / 1e6 == approx_displayed(results_mean)

        # The column the page found really is the one holding raw BOE. The
        # heading is a numbered markdown heading rather than a subheader since
        # the section numbering landed, so the column name is read from there.
        headings = [element.value for element in at.markdown]
        assert any(
            "THR summary (Total_Hydrocarbon_Resource_BOE)" in heading
            for heading in headings
        ), f"column name not in any heading: {[h for h in headings if 'THR' in h]}"


# ==========================================================================
# The exported trial table
# ==========================================================================


@pytest.mark.parametrize("scenario", ["oil_only", "oil_gas"])
def test_exported_trial_table_column_set(scenario):
    """A column silently appearing or vanishing changes what downstream reads.

    Asserted as a set so reordering is free, and against the on-screen preview
    so this is the frame the user actually sees and downloads.
    """
    at = downloads_page(scenario)
    previews = [e.value for e in at.dataframe if "Trial" in e.value.columns]
    assert len(previews) == 1
    preview = previews[0]

    assert set(preview.columns) == EXPECTED_EXPORT_COLUMNS[scenario]
    assert len(preview) == 50, "the page advertises a 50-row preview"

    # The download payload carries the same columns for every trial.
    frame = at.session_state["df_results"]
    assert set(frame.columns) == set(preview.columns)
    assert len(frame) == N_TRIALS
    assert any(f"Showing first 50 of {N_TRIALS:,} trials." == c.value for c in at.caption)


def test_depth_based_export_carries_the_contacts_it_sampled():
    """The GOC path's geometry has to reach the export, with the right values.

    The oil-only scenario must not carry these columns at all; a depth-based
    column surviving a switch of method is how the stale-split defect looked.
    """
    at = downloads_page("oil_gas")
    frame = at.session_state["df_results"]
    state = _oil_gas_state(N_TRIALS, "oilfield", True)

    for column, key in (
        ("HC_Contact_Depth_m", "sD_hc"),
        ("GOC_Depth_m", "sGOC_depth"),
        ("Oil_Column_Height_m", "sOil_Column_Height"),
        ("Gas_Column_Height_m", "sGas_Column_Height"),
        ("GRV_oil_m3", "sGRV_oil_m3"),
        ("GRV_gas_m3", "sGRV_gas_m3"),
    ):
        assert np.allclose(frame[column].to_numpy(), state[key], rtol=1e-12), column

    assert set(frame["GOC_Mode"]) == {"Oil fraction of HC column"}
    # The GOC sits between the top and the contact in every trial.
    assert np.all(frame["GOC_Depth_m"] >= TOP_STRUCTURE_M)
    assert np.all(frame["GOC_Depth_m"] <= frame["HC_Contact_Depth_m"])

    oil_only_columns = set(
        downloads_page("oil_only").session_state["df_results"].columns
    )
    assert not oil_only_columns & {"GOC_Depth_m", "GOC_Mode", "f_oil_depth_area"}


# ==========================================================================
# Hand derivations (see the module docstring)
# ==========================================================================


def test_oil_only_mean_stoiip_matches_hand_volumetric_equation():
    """HAND DERIVATION 1: 259.40 MMSTB analytic vs 261.44 MMSTB reported.

    The 0.786% gap is Monte Carlo noise in the drawn input means, and this test
    checks both halves of that claim: the analytic value is within 2% of what is
    reported, and the product of the *drawn* means lands within 0.1%.
    """
    analytic = (
        pert_mean("GRV")
        * pert_mean("NtG")
        * pert_mean("Por")
        * pert_mean("Shc")
        * RB_PER_M3
        * pert_mean("InvBo")
    )
    assert analytic / 1e6 == pytest.approx(259.40, abs=0.01)

    reported = figure_panels(results_page("oil_only"))["Oil in Place (STOIIP) (MMSTB)"]["Mean"]
    assert reported == pytest.approx(analytic / 1e6, rel=HAND_TOL)

    state = _oil_only_state(N_TRIALS, "oilfield", True)
    from_draws = (
        float(np.mean(state["sGRV_m3_final"]))
        * float(np.mean(state["sNtG"]))
        * float(np.mean(state["sp"]))
        * float(np.mean(state["Shc_oil"]))
        * RB_PER_M3
        * float(np.mean(state["sInvBo"]))
    )
    assert reported == pytest.approx(from_draws / 1e6, rel=1e-3)


def test_oil_gas_grv_split_matches_hand_integrated_ramp():
    """HAND DERIVATION 2: the area ramp integrates to 10,000 h^2 in closed form.

    Per trial, exactly -- this is the one figure in the scenario that is
    arithmetic rather than a Monte Carlo statistic, so it is pinned at 1e-9.
    """
    state = _oil_gas_state(N_TRIALS, "oilfield", True)
    h = state["sD_hc"] - TOP_STRUCTURE_M
    f = state["da_oil_frac"]
    k = AREA_SLOPE_M2_PER_M / 2.0

    assert np.allclose(state["sGRV_m3_final"], k * h**2, rtol=1e-9)
    assert np.allclose(state["sGRV_oil_m3"], k * h**2 * (1.0 - (1.0 - f) ** 2), rtol=1e-9)
    assert np.allclose(state["sGRV_gas_m3"], k * h**2 * (1.0 - f) ** 2, rtol=1e-9)

    # The recorded means quoted in the docstring.
    assert float(np.mean(state["sGRV_m3_final"])) == pytest.approx(892_795_974, rel=1e-6)
    assert float(np.mean(state["sGRV_oil_m3"])) == pytest.approx(567_881_443, rel=1e-6)
    assert float(np.mean(state["sGRV_gas_m3"])) == pytest.approx(324_914_531, rel=1e-6)
    # The split is exhaustive: no rock is counted twice or lost.
    assert np.allclose(
        state["sGRV_oil_m3"] + state["sGRV_gas_m3"], state["sGRV_m3_final"], rtol=1e-12
    )


def test_oil_gas_mean_stoiip_matches_hand_volumetric_equation():
    """HAND DERIVATION 3: 315.66 MMSTB analytic vs 314.81 MMSTB reported.

    Pins the whole conversion chain for the depth-based path:
    GRV -> pore volume -> hydrocarbon pore volume -> reservoir barrels -> STB.
    """
    state = _oil_gas_state(N_TRIALS, "oilfield", True)
    grv_oil_mean = float(np.mean(state["sGRV_oil_m3"]))

    analytic = (
        grv_oil_mean
        * pert_mean("NtG")
        * pert_mean("Por")
        * pert_mean("Shc")
        * RB_PER_M3
        * pert_mean("InvBo")
    )
    assert analytic / 1e6 == pytest.approx(315.66, abs=0.01)

    reported = figure_panels(results_page("oil_gas"))["Oil in Place (STOIIP) (MMSTB)"]["Mean"]
    assert reported == pytest.approx(analytic / 1e6, rel=HAND_TOL)

    from_draws = (
        grv_oil_mean
        * float(np.mean(state["sNtG"]))
        * float(np.mean(state["sp"]))
        * float(np.mean(state["Shc_oil"]))
        * RB_PER_M3
        * float(np.mean(state["sInvBo"]))
    )
    assert reported == pytest.approx(from_draws / 1e6, rel=1e-3)


def test_oil_gas_mean_giip_matches_hand_volumetric_equation():
    """HAND DERIVATION 3, gas half: 43.813 Bscf vs 43.81 Bscf reported.

    Uses the mean of the drawn 1/Bg, because E[1/Bg] is not 1/E[Bg] -- 180.325
    against 179.104, a 0.7% difference that would otherwise be mistaken for
    noise.
    """
    state = _oil_gas_state(N_TRIALS, "oilfield", True)
    inv_bg_mean = float(np.mean(1.0 / state["sBg"]))
    assert inv_bg_mean == pytest.approx(180.325, rel=1e-4)
    assert 1.0 / pert_mean("Bg") == pytest.approx(179.104, rel=1e-4)

    from_draws = (
        float(np.mean(state["sGRV_gas_m3"]))
        * float(np.mean(state["sNtG"]))
        * float(np.mean(state["sp"]))
        * float(np.mean(state["Shc_gas"]))
        * RB_PER_M3
        * inv_bg_mean
    )
    assert from_draws / 1e9 == pytest.approx(43.813, rel=1e-3)

    reported = figure_panels(results_page("oil_gas"))["Gas in Place (GIIP) (Bscf)"]["Mean"]
    assert reported == pytest.approx(from_draws / 1e9, rel=HAND_TOL)


def test_recoverable_is_in_place_times_the_recovery_factor_as_displayed():
    """The two in-place figures and the two recoverable figures must agree.

    End to end through the display layer: this is PR #2's double-conversion
    defect expressed as a relationship between numbers on the screen rather
    than between arrays.
    """
    panels = figure_panels(results_page("oil_gas"))
    state = _oil_gas_state(N_TRIALS, "oilfield", True)

    oil_ratio = (
        panels["Recoverable Oil (MMSTB)"]["Mean"]
        / panels["Oil in Place (STOIIP) (MMSTB)"]["Mean"]
    )
    gas_ratio = (
        panels["Recoverable Free Gas (Bscf)"]["Mean"]
        / panels["Gas in Place (GIIP) (Bscf)"]["Mean"]
    )
    # The displayed ratio is a ratio of means, so it is close to but not equal
    # to the mean recovery factor; 2% is comfortably inside that.
    assert oil_ratio == pytest.approx(float(np.mean(state["sRF_oil"])), rel=HAND_TOL)
    assert gas_ratio == pytest.approx(float(np.mean(state["sRF_gas"])), rel=HAND_TOL)


def test_displayed_thr_equals_liquids_plus_gas_over_the_boe_factor():
    """THR = oil + condensate + total gas / 6000, from the displayed numbers.

    Also pins that the BOE factor the page prints is the one it divides by.
    """
    panels = figure_panels(results_page("oil_gas"))
    oil = panels["Recoverable Oil (MMSTB)"]["Mean"]
    condensate = panels["Recoverable Condensate (MMSTB)"]["Mean"]
    free_gas = panels["Recoverable Free Gas (Bscf)"]["Mean"]
    assoc_gas = panels["Recoverable Associated Gas (Bscf)"]["Mean"]

    # MMSTB + Bscf * 1e9 / (scf/BOE) / 1e6 -> MMBOE
    expected = oil + condensate + (free_gas + assoc_gas) * 1e3 / GAS_SCF_PER_BOE
    assert panels["THR (MMBOE)"]["Mean"] == pytest.approx(expected, rel=1e-3, abs=0.05)

    at = results_page("oil_gas")
    assert any("6,000 scf/BOE" in element.value for element in at.caption)


# ==========================================================================
# Known defects, recorded but deliberately NOT pinned
# ==========================================================================
# These two assert the CORRECT behaviour and are expected to fail today. Under
# These two were strict xfails when the tests were written: both defects were
# live, and freezing the wrong numbers into an assertion would have made them
# permanent. They are ordinary assertions now that the defects are fixed.


@pytest.mark.parametrize("scenario", ["oil_only", "oil_gas"])
def test_metric_thr_composition_is_in_metric_boe(scenario):
    table = composition_table(results_page(scenario, "si"))
    values = {row["Component"]: _as_float(row["Mean (Mm³ BOE)"])
              for _, row in table.iterrows()}
    percentages = {row["Component"]: _as_float(row["% of THR"])
                   for _, row in table.iterrows()}

    components = ["Oil", "Condensate", "Free Gas", "Associated Gas"]
    assert sum(values[c] for c in components) == pytest.approx(
        values["**Total**"], rel=1e-3, abs=ROW_SUM_ABS
    )
    assert sum(percentages[c] for c in components) == pytest.approx(100.0, abs=0.3)

    # Oil in Mm3 BOE is the oilfield MMBOE figure times M3_PER_BBL.
    oilfield = composition_table(results_page(scenario, "oilfield"))
    oilfield_oil = _as_float(oilfield.set_index("Component").loc["Oil", "Mean (MMBOE)"])
    assert values["Oil"] == pytest.approx(oilfield_oil * M3_PER_BBL, rel=1e-2, abs=0.05)


@pytest.mark.parametrize("scenario", ["oil_only", "oil_gas"])
def test_metric_thr_breakdown_is_in_metric_boe(scenario):
    table = breakdown_table(results_page(scenario, "si")).set_index("Fluid Type")
    column = "Mean Recoverable (Mm³ BOE)"

    components = ["Oil", "Condensate", "Free Gas", "Associated Gas"]
    total = _as_float(table.loc["Total Hydrocarbon Resource (THR)", column])
    assert sum(_as_float(table.loc[c, column]) for c in components) == pytest.approx(
        total, rel=1e-3, abs=ROW_SUM_ABS
    )
    assert sum(
        _as_float(table.loc[c, "% of THR"]) for c in components
    ) == pytest.approx(100.0, abs=0.3)
