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
        sec("overview.workflow", "Estimation workflow"),
        sec("overview.assumptions", "Assumptions and formulas"),
    )),
    sec("inputs", "Inputs", children=(
        # The GRV page writes several of these sections more than once: the Direct
        # and Area x Thickness x GCF methods share _display_split_grv_results(),
        # while each depth-based method inlines its own copy of the same block.
        # One key each, so a number means the same section whichever method is on.
        sec("inputs.grv", "Gross Rock Volume (GRV)", children=(
            sec("inputs.grv.method", "GRV input method"),
            sec("inputs.grv.atgcf", "Area, thickness and geometric correction", children=(
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
                    "inputs.grv.depths.spill",
                    "Spill point depth",
                    items=(fig("inputs.grv.depths.spill.fig", "Spill point depth distribution"),),
                ),
                sec(
                    "inputs.grv.depths.hc",
                    "Effective hydrocarbon contact depth",
                    items=(fig("inputs.grv.depths.hc.fig", "Effective HC depth distribution"),),
                ),
                sec(
                    "inputs.grv.depths.goc",
                    "Gas-oil contact (GOC) definition",
                    items=(fig("inputs.grv.depths.goc.fig", "GOC depth distribution"),),
                ),
            )),
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
            sec("inputs.grv.split", "Oil and gas split by fraction"),
            sec(
                "inputs.grv.final",
                "Final GRV for calculations",
                items=(
                    fig("inputs.grv.final.fig", "Final GRV distribution"),
                    tab("inputs.grv.final.tab", "Final GRV statistics"),
                ),
            ),
        )),
        sec("inputs.ntg", "Net-to-Gross (NtG)", children=(
            sec(
                "inputs.ntg.dist",
                "Net-to-gross distribution",
                items=(fig("inputs.ntg.dist.hist", "Net-to-gross distribution"),),
            ),
        )),
        sec("inputs.por", "Porosity", children=(
            sec(
                "inputs.por.dist",
                "Porosity distribution",
                items=(fig("inputs.por.dist.hist", "Porosity distribution"),),
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
                items=(fig("inputs.sat.global.hist", "Global hydrocarbon saturation"),),
            ),
            sec(
                "inputs.sat.sw_oil",
                "Water saturation, oil zone",
                items=(fig("inputs.sat.sw_oil.hist", "Water saturation, oil zone"),),
            ),
            sec(
                "inputs.sat.sw_gas",
                "Water saturation, gas zone",
                items=(fig("inputs.sat.sw_gas.hist", "Water saturation, gas zone"),),
            ),
            sec(
                "inputs.sat.shc_oil",
                "Hydrocarbon saturation, oil phase",
                items=(fig("inputs.sat.shc_oil.hist", "Hydrocarbon saturation, oil phase"),),
            ),
            sec(
                "inputs.sat.shc_gas",
                "Hydrocarbon saturation, gas phase",
                items=(fig("inputs.sat.shc_gas.hist", "Hydrocarbon saturation, gas phase"),),
            ),
        )),
        sec("inputs.fluids", "Fluids & PVT", children=(
            sec("inputs.fluids.oil", "Oil properties"),
            sec("inputs.fluids.gas", "Gas properties"),
            sec("inputs.fluids.cond", "Condensate properties"),
        )),
        sec("inputs.rf", "Recovery factors"),
    )),
    sec("dependencies", "Dependencies", children=(
        sec("dependencies.matrix", "Parameter dependencies and correlations"),
        sec("dependencies.crossplots", "Correlation cross-plots"),
    )),
    sec("run", "Run Simulation", children=(
        sec("run.qc", "Input quality control"),
        sec("run.summary", "Simulation summary"),
        sec("run.inputs", "Input parameter summary"),
    )),
    sec("results", "Results", children=(
        sec("results.inplace", "In-place volumes (STOIIP / GIIP)"),
        sec("results.recoverable", "Recoverable volumes"),
        sec("results.surface", "Recoverable (surface) volumes"),
        sec("results.thr", "Total hydrocarbon resource (THR)"),
        sec("results.boe", "Hydrocarbon volume distribution (BOE)"),
        sec("results.columns", "Oil and gas column distributions"),
        sec("results.saturation", "Saturation distributions"),
        sec("results.crossplot", "Cross-plot and statistics"),
    )),
    sec("sensitivity", "Sensitivity", children=(
        sec("sensitivity.thr", "Total hydrocarbon resource (THR)"),
        sec("sensitivity.breakdown", "THR breakdown by fluid type"),
        sec("sensitivity.tornado", "Tornado sensitivities"),
        sec("sensitivity.summary", "Sensitivity summary (P10/P90 method)"),
    )),
    sec("downloads", "Downloads & Exports", children=(
        sec("downloads.preview", "Results preview"),
        sec("downloads.thr", "THR summary"),
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
