"""Tests for the section/figure/table registry.

The snapshot below is the point of this file. Numbers are derived from the order
declared in SECTIONS, so inserting or moving a section renumbers what follows it -
which is correct for a document that has to read in order, but must never happen
by accident, because these numbers are how a figure is cited outside the app.
A reshuffle therefore fails here and lands in review as an explicit diff.
"""

import pytest

from scopehc import numbering as nb


# --- the snapshot ------------------------------------------------------------------

EXPECTED_SECTIONS = [
    ("1.0", "overview", "Overview"),
    ("1.1", "overview.workflow", "Estimation workflow"),
    ("1.2", "overview.assumptions", "Assumptions and formulas"),
    ("2.0", "inputs", "Inputs"),
    ("2.1", "inputs.grv", "Gross Rock Volume (GRV)"),
    ("2.1.1", "inputs.grv.method", "GRV input method"),
    ("2.1.2", "inputs.grv.direct", "Direct GRV"),
    ("2.1.3", "inputs.grv.atgcf", "Area × thickness × GCF"),
    ("2.1.3.1", "inputs.grv.atgcf.area", "Area"),
    ("2.1.3.2", "inputs.grv.atgcf.thickness", "Reservoir thickness"),
    ("2.1.3.3", "inputs.grv.atgcf.gcf", "Geometric correction factor"),
    ("2.1.3.4", "inputs.grv.atgcf.gcf_mp", "GCF uncertainty multiplier"),
    ("2.1.3.5", "inputs.grv.atgcf.calculator", "Geometric correction factor calculator"),
    ("2.1.3.6", "inputs.grv.atgcf.sensitivity", "Sensitivity summary"),
    ("2.1.3.7", "inputs.grv.atgcf.result", "Calculated GRV"),
    ("2.1.4", "inputs.grv.depth_topbase", "Depth-based: top and base reservoir + contacts"),
    ("2.1.5", "inputs.grv.depth_toponly", "Depth-based: top + reservoir thickness + contacts"),
    ("2.1.5.1", "inputs.grv.depth_toponly.resthickness", "Reservoir thickness"),
    ("2.1.6", "inputs.grv.depths", "Depth distributions"),
    ("2.1.6.1", "inputs.grv.depths.areamult", "Area uncertainty multiplier"),
    ("2.1.6.2", "inputs.grv.depths.spill", "Spill point depth"),
    ("2.1.6.3", "inputs.grv.depths.hc", "Effective hydrocarbon contact depth"),
    ("2.1.6.4", "inputs.grv.depths.goc", "Gas-oil contact (GOC) definition"),
    ("2.1.6.5", "inputs.grv.depths.foil", "Oil fraction of the hydrocarbon column"),
    ("2.1.6.6", "inputs.grv.depths.hoil", "Oil column height"),
    ("2.1.7", "inputs.grv.results", "GRV results by fluid type"),
    ("2.1.7.1", "inputs.grv.results.oil", "Oil GRV"),
    ("2.1.7.2", "inputs.grv.results.gas", "Gas GRV"),
    ("2.1.7.3", "inputs.grv.results.total", "Total hydrocarbon GRV"),
    ("2.1.7.4", "inputs.grv.results.summary", "Combined GRV summary"),
    ("2.1.8", "inputs.grv.abovespill", "GRV distribution above spill point"),
    ("2.1.9", "inputs.grv.columns", "Gas and oil column heights"),
    ("2.1.9.1", "inputs.grv.columns.gas", "Gas column height"),
    ("2.1.9.2", "inputs.grv.columns.oil", "Oil column height"),
    ("2.1.9.3", "inputs.grv.columns.overlay", "Column heights overlay"),
    ("2.1.10", "inputs.grv.split", "Oil and gas split by fraction"),
    ("2.1.10.1", "inputs.grv.split.foil", "Oil fraction of total hydrocarbon GRV"),
    ("2.1.11", "inputs.grv.final", "Final GRV for calculations"),
    ("2.1.11.1", "inputs.grv.final.mp", "GRV uncertainty multiplier"),
    ("2.2", "inputs.ntg", "Net-to-Gross (NtG)"),
    ("2.2.1", "inputs.ntg.dist", "Net-to-gross distribution"),
    ("2.3", "inputs.por", "Porosity"),
    ("2.3.1", "inputs.por.dist", "Porosity distribution"),
    ("2.4", "inputs.sat", "Saturation"),
    ("2.4.1", "inputs.sat.mode", "Saturation mode"),
    ("2.4.2", "inputs.sat.global", "Global hydrocarbon saturation"),
    ("2.4.3", "inputs.sat.sw_oil", "Water saturation, oil zone"),
    ("2.4.4", "inputs.sat.sw_gas", "Water saturation, gas zone"),
    ("2.4.5", "inputs.sat.shc_oil", "Hydrocarbon saturation, oil phase"),
    ("2.4.6", "inputs.sat.shc_gas", "Hydrocarbon saturation, gas phase"),
    ("2.5", "inputs.fluids", "Fluids & PVT"),
    ("2.5.1", "inputs.fluids.oil", "Oil properties"),
    ("2.5.1.1", "inputs.fluids.oil.invbo", "Oil shrinkage factor (1/Bo)"),
    ("2.5.2", "inputs.fluids.gas", "Gas properties"),
    ("2.5.2.1", "inputs.fluids.gas.invbg", "Gas expansion factor (1/Bg)"),
    ("2.5.2.2", "inputs.fluids.gas.bg", "Gas expansion factor (Bg)"),
    ("2.5.2.3", "inputs.fluids.gas.gor", "Gas-oil ratio (GOR)"),
    ("2.5.3", "inputs.fluids.cond", "Condensate properties"),
    ("2.5.3.1", "inputs.fluids.cond.cy", "Condensate yield (CY)"),
    ("2.6", "inputs.rf", "Recovery factors"),
    ("2.6.1", "inputs.rf.oil", "Oil recovery factor"),
    ("2.6.2", "inputs.rf.gas", "Free gas recovery factor"),
    ("2.6.3", "inputs.rf.assoc", "Associated gas recovery factor"),
    ("2.6.4", "inputs.rf.cond", "Condensate recovery factor"),
    ("3.0", "dependencies", "Dependencies"),
    ("3.1", "dependencies.matrix", "Parameter dependencies and correlations"),
    ("3.2", "dependencies.crossplots", "Correlation cross-plots"),
    ("4.0", "run", "Run Simulation"),
    ("4.1", "run.qc", "Input quality control"),
    ("4.2", "run.summary", "Simulation summary"),
    ("4.3", "run.inputs", "Input parameter summary"),
    ("5.0", "results", "Results"),
    ("5.1", "results.inplace", "In-place volumes (STOIIP / GIIP)"),
    ("5.1.1", "results.inplace.stoiip", "Oil in place (STOIIP)"),
    ("5.1.2", "results.inplace.giip", "Gas in place (GIIP)"),
    ("5.2", "results.recoverable", "Recoverable volumes"),
    ("5.2.1", "results.recoverable.oil", "Recoverable oil"),
    ("5.2.2", "results.recoverable.freegas", "Recoverable free gas"),
    ("5.2.3", "results.recoverable.assocgas", "Recoverable associated gas"),
    ("5.2.4", "results.recoverable.condensate", "Recoverable condensate"),
    ("5.3", "results.surface", "Recoverable (surface) volumes"),
    ("5.4", "results.thr", "Total hydrocarbon resource (THR)"),
    ("5.5", "results.boe", "Hydrocarbon volume distribution (BOE)"),
    ("5.6", "results.columns", "Oil and gas column distributions"),
    ("5.7", "results.crossplot", "Cross-plot and statistics"),
    ("6.0", "sensitivity", "Sensitivity"),
    ("6.1", "sensitivity.boe", "Hydrocarbon volume distribution (BOE)"),
    ("6.2", "sensitivity.thr", "Total hydrocarbon resource (THR)"),
    ("6.2.1", "sensitivity.thr.breakdown", "THR breakdown by fluid type"),
    ("6.3", "sensitivity.tornado", "Tornado sensitivities"),
    ("6.4", "sensitivity.summary", "Sensitivity summary (P10/P90 method)"),
    ("7.0", "downloads", "Downloads & Exports"),
    ("7.1", "downloads.preview", "Results preview"),
    ("7.2", "downloads.thr", "THR summary"),
    ("7.3", "downloads.options", "Export options"),
    ("8.0", "help", "Help & Math Appendix"),
]


def test_the_numbering_is_exactly_this():
    actual = [(e.number, e.key, e.title) for e in nb.walk()]
    assert actual == EXPECTED_SECTIONS


def test_every_page_is_a_whole_number():
    tops = [e for e in nb.walk() if e.level == 1]
    assert [e.number for e in tops] == [f"{i}.0" for i in range(1, len(tops) + 1)]


# --- the shape of a number ----------------------------------------------------------


def test_depth_is_visible_in_the_number():
    assert nb.number("overview") == "1.0"
    assert nb.number("inputs.grv") == "2.1"
    assert nb.number("inputs.sat.sw_oil") == "2.4.3"
    assert nb.level("overview") == 1
    assert nb.level("inputs.grv") == 2
    assert nb.level("inputs.sat.sw_oil") == 3


def test_heading_reads_as_number_then_title():
    assert nb.heading("inputs.sat.sw_oil") == "2.4.3 Water saturation, oil zone"


def test_a_heading_title_can_be_overridden_without_moving_the_number():
    # For headings built at runtime from a value the registry cannot know.
    assert nb.heading("inputs.grv", "GRV — Area × Thickness × GCF") == (
        "2.1 GRV — Area × Thickness × GCF"
    )


# --- figures and tables --------------------------------------------------------------


def test_items_hang_off_their_section_number():
    assert nb.label("inputs.ntg.dist.fig") == "Figure 2.2.1-1"
    assert nb.label("inputs.sat.sw_gas.fig") == "Figure 2.4.4-1"


def test_figures_and_tables_count_separately():
    # Built here rather than taken from the live registry, so the rule is pinned
    # even before a page with both is declared.
    section = nb.sec(
        "x",
        "X",
        items=(
            nb.fig("x.f1", "first figure"),
            nb.tab("x.t1", "first table"),
            nb.fig("x.f2", "second figure"),
        ),
    )
    saved = nb.SECTIONS
    try:
        nb.SECTIONS = (section,)
        sections, items = nb._build()
    finally:
        nb.SECTIONS = saved
    assert items["x.f1"][0] == "Figure 1.0-1"
    assert items["x.t1"][0] == "Table 1.0-1"
    assert items["x.f2"][0] == "Figure 1.0-2"


def test_caption_joins_label_and_title():
    assert nb.caption("inputs.por.dist.fig") == "Figure 2.3.1-1 — Porosity distribution"


def test_caption_title_can_be_overridden():
    assert nb.caption("inputs.por.dist.fig", "Porosity, P10–P90") == (
        "Figure 2.3.1-1 — Porosity, P10–P90"
    )


# --- the registry refuses to be ambiguous ---------------------------------------------


def test_an_unknown_key_names_itself_and_where_to_fix_it():
    with pytest.raises(KeyError, match="unknown section key"):
        nb.number("inputs.nope")
    with pytest.raises(KeyError, match="unknown figure/table key"):
        nb.label("inputs.nope.fig")


def test_duplicate_section_keys_are_refused():
    saved = nb.SECTIONS
    try:
        nb.SECTIONS = (nb.sec("dup", "One"), nb.sec("dup", "Two"))
        with pytest.raises(ValueError, match="duplicate section key"):
            nb._build()
    finally:
        nb.SECTIONS = saved


def test_duplicate_item_keys_are_refused():
    saved = nb.SECTIONS
    try:
        nb.SECTIONS = (
            nb.sec("a", "A", items=(nb.fig("same", "one"),)),
            nb.sec("b", "B", items=(nb.fig("same", "two"),)),
        )
        with pytest.raises(ValueError, match="duplicate item key"):
            nb._build()
    finally:
        nb.SECTIONS = saved


def test_mutually_exclusive_sections_keep_their_numbers():
    # The five saturation modes never appear together. Whichever renders, its
    # number is the same - the gap on screen is the feature.
    assert nb.number("inputs.sat.global") == "2.4.2"
    assert nb.number("inputs.sat.shc_gas") == "2.4.6"


def test_every_declared_item_belongs_to_a_known_section():
    section_keys = {e.key for e in nb.walk()}
    for key, _, _, _ in nb.items():
        owner = key.rsplit(".", 1)[0]
        assert owner in section_keys, f"{key} has no section {owner}"
