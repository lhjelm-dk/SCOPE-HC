"""
Section, figure and table numbering: one registry for the whole app.

Why a registry rather than counters
-----------------------------------
Numbers have to be stable, because they are how a figure gets cited in a report
written months later. Counting figures as they render cannot give that: an
oil-only run does not draw the gas plots, so a render-time counter would quietly
promote every figure after the skipped one and the same plot would carry a
different number in two runs of the same tool.

So every section, figure and table is *declared* here, whether or not a given run
shows it. A run that hides the gas section leaves a gap in the numbers on screen.
That gap is the point: 5.2.1 is the same figure in every run.

The same section is often written more than once in the source. inputs_grv.py
renders "GRV results by fluid type" once per GRV method, and inside each method
once per fluid case - eight source headings for one logical section. A number
belongs to the section, not to the line of code that happens to draw it, so all
eight sites look up the same key.

Numbers are derived from the order declared in SECTIONS, not written out by hand.
A hand-assigned number stays put when a section moves, which sounds like what
"stable" means until the page reads 2.1, 2.4, 2.2. Deriving them keeps the
document in order; the cost is that inserting a section renumbers what follows.
That cost is paid deliberately and visibly: tests/test_numbering.py pins the
whole numbering as a snapshot, so any movement shows up in review as an explicit
diff rather than as a surprise in a report six months later.

Levels render as:  top "1.0"  ·  second "2.1"  ·  third "2.1.3"
Items render as:   "Figure 2.1.3-1"  ·  "Table 2.1.3-1"  (separate counters)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Sequence, Tuple

FIGURE = "figure"
TABLE = "table"


@dataclass(frozen=True)
class Item:
    """A figure or a table belonging to one section."""

    key: str
    title: str
    kind: str


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    items: Tuple[Item, ...] = ()
    children: Tuple["Section", ...] = ()


def fig(key: str, title: str) -> Item:
    return Item(key=key, title=title, kind=FIGURE)


def tab(key: str, title: str) -> Item:
    return Item(key=key, title=title, kind=TABLE)


def sec(
    key: str,
    title: str,
    *,
    items: Sequence[Item] = (),
    children: Sequence[Section] = (),
) -> Section:
    return Section(key=key, title=title, items=tuple(items), children=tuple(children))


# ---------------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------------
# Order here is the order on screen. Keys are permanent handles used from the UI;
# rename a title freely, but changing a key breaks every call site that cites it.
#
# Sections are declared down to subsection level for every page. Figures and
# tables are declared per page as each page is converted; a page with none
# declared yet simply has no items, and its figures stay uncaptioned until then.

SECTIONS: Tuple[Section, ...] = (
    sec("overview", "Overview", children=(
        sec(
            "overview.workflow",
            "Estimation workflow",
            items=(fig("overview.workflow.fig", "Estimation workflow"),),
        ),
        sec("overview.assumptions", "Assumptions and formulas"),
    )),
    sec("inputs", "Inputs", children=(
        # The GRV page writes several of these sections more than once: the Direct
        # and Area x Thickness x GCF methods share _display_split_grv_results(),
        # while each depth-based method inlines its own copy of the same block.
        # One key each, so a number means the same section whichever method is on.
        sec("inputs.grv", "Gross Rock Volume (GRV)", children=(
            sec("inputs.grv.method", "GRV input method"),
            sec(
                "inputs.grv.direct",
                "Direct GRV",
                items=(
                    fig("inputs.grv.direct.fig", "Direct GRV distribution"),
                    tab("inputs.grv.direct.tab", "Direct GRV statistics"),
            ),
            ),
            sec("inputs.grv.atgcf", "Area, thickness and geometric correction", children=(
                sec(
                    "inputs.grv.atgcf.area",
                    "Area",
                    items=(
                        fig("inputs.grv.atgcf.area.fig", "Area distribution"),
                        tab("inputs.grv.atgcf.area.tab", "Area statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.atgcf.thickness",
                    "Reservoir thickness",
                    items=(
                        fig("inputs.grv.atgcf.thickness.fig", "Reservoir thickness distribution"),
                        tab("inputs.grv.atgcf.thickness.tab", "Reservoir thickness statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.atgcf.gcf",
                    "Geometric correction factor",
                    items=(
                        fig("inputs.grv.atgcf.gcf.fig", "Geometric correction factor distribution"),
                        tab("inputs.grv.atgcf.gcf.tab", "Geometric correction factor statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.atgcf.gcf_mp",
                    "GCF uncertainty multiplier",
                    items=(
                        fig("inputs.grv.atgcf.gcf_mp.fig", "GCF uncertainty multiplier distribution"),
                        tab("inputs.grv.atgcf.gcf_mp.tab", "GCF uncertainty multiplier statistics"),
                    ),
                ),

                sec(
                    "inputs.grv.atgcf.calculator",
                    "Geometric correction factor calculator",
                    items=(
                        fig("inputs.grv.atgcf.calculator.fig", "Final GCF distribution"),
                        tab("inputs.grv.atgcf.calculator.tab", "GCF statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.atgcf.sensitivity",
                    "Sensitivity summary",
                    items=(fig("inputs.grv.atgcf.sensitivity.fig", "GCF sensitivity to area, thickness and dip"),),
                ),
                # The calculated GRV had no heading of its own: its plot followed the
                # sensitivity summary and read as part of it.
                sec(
                    "inputs.grv.atgcf.result",
                    "Calculated GRV",
                    items=(
                        fig("inputs.grv.atgcf.result.fig", "Calculated GRV distribution"),
                        tab("inputs.grv.atgcf.result.tab", "Calculated GRV statistics"),
                    ),
                ),
            )),
            sec(
                "inputs.grv.deptharea",
                "Depth-area relationship",
                items=(
                    fig("inputs.grv.deptharea.fig", "Depth versus area, with GRV and spill point"),
                    tab("inputs.grv.deptharea.tab", "Depth-area table"),
                ),
            ),
            sec("inputs.grv.depths", "Depth distributions", children=(
                sec(
                    "inputs.grv.depths.resthickness",
                    "Reservoir thickness (top-only method)",
                    items=(
                        fig("inputs.grv.depths.resthickness.fig", "Reservoir thickness (top-only method) distribution"),
                        tab("inputs.grv.depths.resthickness.tab", "Reservoir thickness (top-only method) statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.depths.areamult",
                    "Area uncertainty multiplier",
                    items=(
                        fig("inputs.grv.depths.areamult.fig", "Area uncertainty multiplier distribution"),
                        tab("inputs.grv.depths.areamult.tab", "Area uncertainty multiplier statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.depths.spill",
                    "Spill point depth",
                    items=(fig("inputs.grv.depths.spill.fig", "Spill point depth distribution"), tab("inputs.grv.depths.spill.tab", "Spill point depth statistics"),),
                ),
                sec(
                    "inputs.grv.depths.hc",
                    "Effective hydrocarbon contact depth",
                    items=(fig("inputs.grv.depths.hc.fig", "Effective HC depth distribution"), tab("inputs.grv.depths.hc.tab", "Effective HC depth statistics"),),
                ),
                sec(
                    "inputs.grv.depths.goc",
                    "Gas-oil contact (GOC) definition",
                    items=(fig("inputs.grv.depths.goc.fig", "GOC depth distribution"), tab("inputs.grv.depths.goc.tab", "GOC depth statistics"),),
                ),
                sec(
                    "inputs.grv.depths.foil",
                    "Oil fraction of the hydrocarbon column",
                    items=(
                        fig("inputs.grv.depths.foil.fig", "Oil fraction of the hydrocarbon column distribution"),
                        tab("inputs.grv.depths.foil.tab", "Oil fraction of the hydrocarbon column statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.depths.hoil",
                    "Oil column height",
                    items=(
                        fig("inputs.grv.depths.hoil.fig", "Oil column height distribution"),
                        tab("inputs.grv.depths.hoil.tab", "Oil column height statistics"),
                    ),
                ),
            )
        ),
            sec("inputs.grv.results", "GRV results by fluid type", children=(
                sec(
                    "inputs.grv.results.oil",
                    "Oil GRV",
                    items=(
                        fig("inputs.grv.results.oil.fig", "Oil GRV distribution"),
                        tab("inputs.grv.results.oil.tab", "Oil GRV statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.results.gas",
                    "Gas GRV",
                    items=(
                        fig("inputs.grv.results.gas.fig", "Gas GRV distribution"),
                        tab("inputs.grv.results.gas.tab", "Gas GRV statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.results.total",
                    "Total hydrocarbon GRV",
                    items=(
                        fig("inputs.grv.results.total.fig", "Total HC GRV distribution"),
                        tab("inputs.grv.results.total.tab", "Total HC GRV statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.results.summary",
                    "Combined GRV summary",
                    items=(tab("inputs.grv.results.summary.tab", "GRV summary by fluid type"),),
                ),
            )),
            sec(
                "inputs.grv.abovespill",
                "GRV distribution above spill point",
                items=(fig("inputs.grv.abovespill.fig", "GRV above spill point"),),
            ),
            sec("inputs.grv.columns", "Gas and oil column heights", children=(
                sec(
                    "inputs.grv.columns.gas",
                    "Gas column height",
                    items=(
                        fig("inputs.grv.columns.gas.fig", "Gas column height distribution"),
                        tab("inputs.grv.columns.gas.tab", "Gas column height statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.columns.oil",
                    "Oil column height",
                    items=(
                        fig("inputs.grv.columns.oil.fig", "Oil column height distribution"),
                        tab("inputs.grv.columns.oil.tab", "Oil column height statistics"),
                    ),
                ),
                sec(
                    "inputs.grv.columns.overlay",
                    "Column heights overlay",
                    items=(fig("inputs.grv.columns.overlay.fig", "Oil and gas column heights"),),
                ),
            )),
            sec("inputs.grv.split", "Oil and gas split by fraction",
                children=(
                sec(
                    "inputs.grv.split.foil",
                    "Oil fraction of total hydrocarbon GRV",
                    items=(
                        fig("inputs.grv.split.foil.fig", "Oil fraction of total hydrocarbon GRV distribution"),
                        tab("inputs.grv.split.foil.tab", "Oil fraction of total hydrocarbon GRV statistics"),
                    ),
                ),
            ),
            ),
            sec(
                "inputs.grv.final",
                "Final GRV for calculations",
                items=(
                    fig("inputs.grv.final.fig", "Final GRV distribution"),
                    tab("inputs.grv.final.tab", "Final GRV statistics"),
                ),
                children=(
                sec(
                    "inputs.grv.final.mp",
                    "GRV uncertainty multiplier",
                    items=(
                        fig("inputs.grv.final.mp.fig", "GRV uncertainty multiplier distribution"),
                        tab("inputs.grv.final.mp.tab", "GRV uncertainty multiplier statistics"),
                    ),
                ),
            ),
            ),
        )),
        sec("inputs.ntg", "Net-to-Gross (NtG)", children=(
            sec(
                "inputs.ntg.dist",
                "Net-to-gross distribution",
                items=(fig("inputs.ntg.dist.fig", "Net-to-gross distribution"), tab("inputs.ntg.dist.tab", "Net-to-gross statistics"),),
            ),
        )),
        sec("inputs.por", "Porosity", children=(
            sec(
                "inputs.por.dist",
                "Porosity distribution",
                items=(fig("inputs.por.dist.fig", "Porosity distribution"), tab("inputs.por.dist.tab", "Porosity statistics"),),
            ),
        )),
        # The five saturation subsections below are mutually exclusive: the mode
        # chosen in 2.4.1 decides which one renders. Each keeps its number in
        # every run, which is exactly what the registry is for.
        sec("inputs.sat", "Saturation", children=(
            sec("inputs.sat.mode", "Saturation mode"),
            sec(
                "inputs.sat.global",
                "Global hydrocarbon saturation",
                items=(fig("inputs.sat.global.fig", "Global hydrocarbon saturation"), tab("inputs.sat.global.tab", "Global hydrocarbon saturation statistics"),),
            ),
            sec(
                "inputs.sat.sw_oil",
                "Water saturation, oil zone",
                items=(fig("inputs.sat.sw_oil.fig", "Water saturation, oil zone"),),
            ),
            sec(
                "inputs.sat.sw_gas",
                "Water saturation, gas zone",
                items=(fig("inputs.sat.sw_gas.fig", "Water saturation, gas zone"),),
            ),
            sec(
                "inputs.sat.shc_oil",
                "Hydrocarbon saturation, oil phase",
                items=(fig("inputs.sat.shc_oil.fig", "Hydrocarbon saturation, oil phase"),),
            ),
            sec(
                "inputs.sat.shc_gas",
                "Hydrocarbon saturation, gas phase",
                items=(fig("inputs.sat.shc_gas.fig", "Hydrocarbon saturation, gas phase"),),
            ),
        )),
        sec("inputs.fluids", "Fluids & PVT", children=(
            sec("inputs.fluids.oil", "Oil properties",
                children=(
                sec(
                    "inputs.fluids.oil.invbo",
                    "Oil shrinkage factor (1/Bo)",
                    items=(
                        fig("inputs.fluids.oil.invbo.fig", "Oil shrinkage factor (1/Bo) distribution"),
                        tab("inputs.fluids.oil.invbo.tab", "Oil shrinkage factor (1/Bo) statistics"),
                    ),
                ),
            ),
            ),
            sec("inputs.fluids.gas", "Gas properties",
                children=(
                sec(
                    "inputs.fluids.gas.invbg",
                    "Gas expansion factor (1/Bg)",
                    items=(
                        fig("inputs.fluids.gas.invbg.fig", "Gas expansion factor (1/Bg) distribution"),
                        tab("inputs.fluids.gas.invbg.tab", "Gas expansion factor (1/Bg) statistics"),
                    ),
                ),
                sec(
                    "inputs.fluids.gas.bg",
                    "Gas expansion factor (Bg)",
                    items=(
                        fig("inputs.fluids.gas.bg.fig", "Gas expansion factor (Bg) distribution"),
                        tab("inputs.fluids.gas.bg.tab", "Gas expansion factor (Bg) statistics"),
                    ),
                ),
                sec(
                    "inputs.fluids.gas.gor",
                    "Gas-oil ratio (GOR)",
                    items=(
                        fig("inputs.fluids.gas.gor.fig", "Gas-oil ratio (GOR) distribution"),
                        tab("inputs.fluids.gas.gor.tab", "Gas-oil ratio (GOR) statistics"),
                    ),
                ),
            ),
            ),
            sec("inputs.fluids.cond", "Condensate properties",
                children=(
                sec(
                    "inputs.fluids.cond.cy",
                    "Condensate yield (CY)",
                    items=(
                        fig("inputs.fluids.cond.cy.fig", "Condensate yield (CY) distribution"),
                        tab("inputs.fluids.cond.cy.tab", "Condensate yield (CY) statistics"),
                    ),
                ),
            ),
            ),
        )),
        sec("inputs.rf", "Recovery factors",
                children=(
                sec(
                    "inputs.rf.oil",
                    "Oil recovery factor",
                    items=(
                        fig("inputs.rf.oil.fig", "Oil recovery factor distribution"),
                        tab("inputs.rf.oil.tab", "Oil recovery factor statistics"),
                    ),
                ),
                sec(
                    "inputs.rf.gas",
                    "Free gas recovery factor",
                    items=(
                        fig("inputs.rf.gas.fig", "Free gas recovery factor distribution"),
                        tab("inputs.rf.gas.tab", "Free gas recovery factor statistics"),
                    ),
                ),
                sec(
                    "inputs.rf.assoc",
                    "Associated gas recovery factor",
                    items=(
                        fig("inputs.rf.assoc.fig", "Associated gas recovery factor distribution"),
                        tab("inputs.rf.assoc.tab", "Associated gas recovery factor statistics"),
                    ),
                ),
                sec(
                    "inputs.rf.cond",
                    "Condensate recovery factor",
                    items=(
                        fig("inputs.rf.cond.fig", "Condensate recovery factor distribution"),
                        tab("inputs.rf.cond.tab", "Condensate recovery factor statistics"),
                    ),
                ),
            ),
            ),
    )),
    sec("dependencies", "Dependencies", children=(
        sec("dependencies.matrix", "Parameter dependencies and correlations"),
        sec(
            "dependencies.crossplots",
            "Correlation cross-plots",
            items=(fig("dependencies.crossplots.fig", "Correlated parameter cross-plots"),),
        ),
    )),
    sec("run", "Run Simulation", children=(
        sec(
            "run.qc",
            "Input quality control",
            items=(
                tab("run.qc.core", "Core input parameters"),
                tab("run.qc.settings", "Simulation settings"),
            ),
        ),
        sec("run.summary", "Simulation summary"),
        sec("run.inputs", "Input parameter summary"),
    )),
    sec("results", "Results", children=(
        # The in-place and recoverable blocks each draw their plots from one loop,
        # so the subsection key travels with the loop item rather than being read
        # from the call site.
        sec("results.inplace", "In-place volumes (STOIIP / GIIP)", children=(
                sec(
                    "results.inplace.stoiip",
                    "Oil in place (STOIIP)",
                    items=(
                        fig("results.inplace.stoiip.fig", "STOIIP distribution"),
                        tab("results.inplace.stoiip.tab", "STOIIP statistics"),
                    ),
                ),
                sec(
                    "results.inplace.giip",
                    "Gas in place (GIIP)",
                    items=(
                        fig("results.inplace.giip.fig", "GIIP distribution"),
                        tab("results.inplace.giip.tab", "GIIP statistics"),
                    ),
                ),
        )),
        sec("results.recoverable", "Recoverable volumes", children=(
                sec(
                    "results.recoverable.oil",
                    "Recoverable oil",
                    items=(
                        fig("results.recoverable.oil.fig", "Recoverable oil distribution"),
                        tab("results.recoverable.oil.tab", "Recoverable oil statistics"),
                    ),
                ),
                sec(
                    "results.recoverable.freegas",
                    "Recoverable free gas",
                    items=(
                        fig("results.recoverable.freegas.fig", "Recoverable free gas distribution"),
                        tab("results.recoverable.freegas.tab", "Recoverable free gas statistics"),
                    ),
                ),
                sec(
                    "results.recoverable.assocgas",
                    "Recoverable associated gas",
                    items=(
                        fig("results.recoverable.assocgas.fig", "Recoverable associated gas distribution"),
                        tab("results.recoverable.assocgas.tab", "Recoverable associated gas statistics"),
                    ),
                ),
                sec(
                    "results.recoverable.condensate",
                    "Recoverable condensate",
                    items=(
                        fig("results.recoverable.condensate.fig", "Recoverable condensate distribution"),
                        tab("results.recoverable.condensate.tab", "Recoverable condensate statistics"),
                    ),
                ),
        )),
        sec(
            "results.surface",
            "Recoverable (surface) volumes",
            items=(tab("results.surface.tab", "Key percentiles for recoverable volumes"),),
        ),
        sec(
            "results.thr",
            "Total hydrocarbon resource (THR)",
            items=(
                tab("results.thr.tab", "THR percentiles"),
                tab("results.thr.composition", "THR composition (mean)"),
            ),
        ),
        sec(
            "results.boe",
            "Hydrocarbon volume distribution (BOE)",
            items=(
                fig("results.boe.fig", "Recoverable volumes by component (BOE)"),
                # The expander below the violin plot holds a second THR view.
                fig("results.boe.thr_fig", "THR distribution"),
                tab("results.boe.thr_tab", "THR statistics"),
                tab("results.boe.thr_breakdown", "THR breakdown by fluid type"),
            ),
        ),
        sec(
            "results.columns",
            "Oil and gas column distributions",
            items=(fig("results.columns.fig", "Oil and gas column heights"),),
        ),
        # Declared, but not rendered today: results.py defines
        # _render_saturation_summary() and render_saturation_plots() and calls
        # neither. See the PR that added this.
        sec("results.saturation", "Saturation distributions", children=(
            sec(
                "results.saturation.assumptions",
                "Saturation assumptions",
                items=(fig("results.saturation.assumptions.fig", "Saturation assumptions"),),
            ),
            sec(
                "results.saturation.plots",
                "Saturation distributions by zone",
                items=(fig("results.saturation.plots.fig", "Saturation distribution"),),
            ),
        )),
        sec("results.crossplot", "Cross-plot and statistics"),
    )),
    sec("sensitivity", "Sensitivity", children=(
        sec(
            "sensitivity.boe",
            "Hydrocarbon volume distribution (BOE)",
            items=(fig("sensitivity.boe.fig", "Hydrocarbon volumes, all fluids in BOE"),),
        ),
        # The breakdown is rendered inside the THR block, by _render_thr_breakdown.
        sec(
            "sensitivity.thr",
            "Total hydrocarbon resource (THR)",
            items=(
                fig("sensitivity.thr.fig", "THR distribution"),
                tab("sensitivity.thr.tab", "THR statistics"),
                tab("sensitivity.thr.composition", "THR composition (mean)"),
            ),
            children=(
                sec(
                    "sensitivity.thr.breakdown",
                    "THR breakdown by fluid type",
                    items=(tab("sensitivity.thr.breakdown.tab", "THR breakdown by fluid type"),),
                ),
            ),
        ),
        sec(
            "sensitivity.tornado",
            "Tornado sensitivities",
            items=(fig("sensitivity.tornado.fig", "Tornado sensitivity chart"),),
        ),
        sec(
            "sensitivity.summary",
            "Sensitivity summary (P10/P90 method)",
            items=(tab("sensitivity.summary.tab", "Parameter impact on P10/P90"),),
        ),
    )),
    sec("downloads", "Downloads & Exports", children=(
        sec(
            "downloads.preview",
            "Results preview",
            items=(tab("downloads.preview.tab", "First rows of the trial table"),),
        ),
        sec(
            "downloads.thr",
            "THR summary",
            items=(tab("downloads.thr.tab", "THR summary statistics"),),
        ),
        sec("downloads.options", "Export options"),
    )),
    sec("help", "Help & Math Appendix"),
)


# ---------------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------------


def _format(path: Sequence[int]) -> str:
    """(1,) -> '1.0'   (2, 1) -> '2.1'   (2, 1, 3) -> '2.1.3'."""
    if len(path) == 1:
        return f"{path[0]}.0"
    return ".".join(str(p) for p in path)


@dataclass(frozen=True)
class Entry:
    key: str
    title: str
    number: str
    level: int


def _build() -> Tuple[Dict[str, Entry], Dict[str, Tuple[str, str, str]]]:
    sections: Dict[str, Entry] = {}
    items: Dict[str, Tuple[str, str, str]] = {}  # key -> (label, title, kind)

    def visit(nodes: Sequence[Section], prefix: Tuple[int, ...]) -> None:
        for index, node in enumerate(nodes, start=1):
            path = prefix + (index,)
            number = _format(path)
            if node.key in sections:
                raise ValueError(f"duplicate section key: {node.key}")
            sections[node.key] = Entry(
                key=node.key, title=node.title, number=number, level=len(path)
            )
            counters = {FIGURE: 0, TABLE: 0}
            for item in node.items:
                if item.key in items:
                    raise ValueError(f"duplicate item key: {item.key}")
                counters[item.kind] += 1
                noun = "Figure" if item.kind == FIGURE else "Table"
                items[item.key] = (
                    f"{noun} {number}-{counters[item.kind]}",
                    item.title,
                    item.kind,
                )
            visit(node.children, path)

    visit(SECTIONS, ())
    return sections, items


_SECTIONS, _ITEMS = _build()


# ---------------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------------


def number(key: str) -> str:
    """'2.1.3' for a section key."""
    return _entry(key).number


def title(key: str) -> str:
    return _entry(key).title


def level(key: str) -> int:
    """1 for a page, 2 for a section within it, 3 for a subsection."""
    return _entry(key).level


def heading(key: str, override: str | None = None) -> str:
    """'2.1.3 GRV results by fluid type', for rendering as a heading.

    `override` replaces the registered title where a heading is built at runtime
    from a value the registry cannot know, without moving the number.
    """
    entry = _entry(key)
    return f"{entry.number} {override if override is not None else entry.title}"


def label(item_key: str) -> str:
    """'Figure 2.1.3-1'."""
    return _item(item_key)[0]


def caption(item_key: str, override: str | None = None) -> str:
    """'Figure 2.1.3-1 — Oil GRV distribution'."""
    lab, registered, _ = _item(item_key)
    text = override if override is not None else registered
    return f"{lab} — {text}" if text else lab


def kind(item_key: str) -> str:
    return _item(item_key)[2]


def walk() -> Iterator[Entry]:
    """Every section in document order - for tests and for building a contents list."""
    for entry in _SECTIONS.values():
        yield entry


def items() -> Iterator[Tuple[str, str, str, str]]:
    """(key, label, title, kind) for every declared figure and table, in order."""
    for key, (lab, text, item_kind) in _ITEMS.items():
        yield key, lab, text, item_kind


def _entry(key: str) -> Entry:
    try:
        return _SECTIONS[key]
    except KeyError:
        raise KeyError(
            f"unknown section key {key!r} - add it to SECTIONS in scopehc/numbering.py"
        ) from None


def _item(key: str) -> Tuple[str, str, str]:
    try:
        return _ITEMS[key]
    except KeyError:
        raise KeyError(
            f"unknown figure/table key {key!r} - add it to SECTIONS in scopehc/numbering.py"
        ) from None
